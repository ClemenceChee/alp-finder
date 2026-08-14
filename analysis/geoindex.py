#!/usr/bin/env python3
"""Геоиндекс: пиксель -> координата, координата -> кадры, карта покрытия.

Слой 1 из `docs/video-analysis.md`, три задачи одним инструментом:

  tochka     пиксель кадра -> широта, долгота, высота, погрешность
  kadry      координата -> все (видео, таймкод), которые её видели, и насколько хорошо
  pokrytie   по набору пролётов: что рельефа осмотрено, что нет; линия падения

Использование (из корня репозитория):

  python3 analysis/geoindex.py tochka data/.../ВИДЕО.MP4 --time 3:32 --pixel 41,282
  python3 analysis/geoindex.py kadry --coord 39.483144,73.585443 data/.../*.MP4
  python3 analysis/geoindex.py pokrytie data/.../*.MP4 --fall-line 39.482656,73.586792

Нужно: телеметрия рядом с видео (`python3 scripts/dji_meta_gps.py ВИДЕО.MP4`,
версия с углами подвеса) и модель высот (`bash scripts/download_dem.sh`).
Зависимости: numpy, opencv-python, rasterio.

Стоимость: самокалибровка фокусного декодирует видео и стоит порядка минуты на
минуту съёмки. Замеры складываются рядом с видео в `<видео>.MP4.focal.tsv` и
переиспользуются; ключ `--recalibrate` пересчитывает.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from georef import (
    DEFAULT_DEM, DEM_POSTING_M, EARTH_R, GRAZING_MIN_DEG, UNCERTAINTY_MAX_M,
    UNCERTAINTY_RANGE_RATIO, Azimuth, GeorefError, Terrain, cast_ray, datum_check,
    fix_at, format_timecode, locate, parse_timecode, pixel_ray, project, read_sidecar,
    uncertainty_m,
)
import selfcal

# Наземное разрешение, при котором клетка считается осмотренной, а не пролётной:
# предмет размером с рюкзак (60 см) даёт при 30 см/пикс два пикселя и практически
# невидим. Порог тот же, каким пилот конвейера мерил покрытие.
SEARCHED_GSD_CM_PX = 30.0

# Сетка пикселей, через которые пускаются лучи при построении покрытия. 7x7 на
# кадр: на дальности 300 м это узлы через ~50 м, мельче шага DEM смысла нет.
COVERAGE_GRID = 7

# Дальше этой дальности покрытие не считается. При фокусном 8000 пикселей 3 км
# дают 37 см/пикс - хуже порога «осмотрено» в любом случае, а марш луча до
# горизонта на каждом кадре стоит дороже всего остального вместе взятого.
COVERAGE_MAX_RANGE_M = 3000.0

# Луч до цели считается перекрытым, если рельеф остановил его дальше чем на один
# шаг сетки не доходя. Ближе шага DEM не отличает гребень от собственной земли
# цели, и решать в любую сторону значило бы выдумывать разрешение модели высот.
OCCLUSION_MARGIN_M = DEM_POSTING_M

_D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# --- общее ------------------------------------------------------------------


def video_geometry(video: Path) -> tuple[float, int, int, int]:
    """(кадров в секунду, ширина, высота, число кадров) из контейнера."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise GeorefError(f"не открывается видео: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return fps, w, h, n


class Flight:
    """Один файл: телеметрия, геометрия кадра и фокусное по кадрам."""

    def __init__(self, video: Path, *, fixed_focal: float | None = None,
                 recalibrate: bool = False, frame_from: int = 0, frame_to: int | None = None,
                 quiet: bool = False):
        self.video = video
        self.fps, self.width, self.height, self.n_frames = video_geometry(video)
        # n_frames, а не fps: строки телеметрии идут один к одному с кадрами,
        # и порядковый номер строки надёжнее любой оценки частоты кадров.
        self.fixes = read_sidecar(video, n_frames=self.n_frames)
        self.fixed_focal = fixed_focal
        self.series = None
        self.note = ""
        if fixed_focal:
            self.note = f"фокусное задано вручную: {fixed_focal:.0f} пикс"
            return
        cached = None if recalibrate else selfcal.load(video)
        # Кэш может быть собран по окну вокруг другого таймкода. Молча принять
        # его значило бы ответить «фокусного нет» там, где его просто не искали,
        # то есть выдать азимут вместо готовой координаты. Поэтому запрошенный
        # диапазон домеряется и дописывается в тот же кэш.
        need = cached is None or not self._covers(cached, frame_from, frame_to)
        if need:
            if not quiet:
                print(f"  самокалибровка {video.name} "
                      f"(кадры {frame_from}..{frame_to if frame_to is not None else 'конец'})...",
                      file=sys.stderr, flush=True)
            fresh = selfcal.measure(video, self.fixes, self.fps, width=self.width,
                                    frame_from=frame_from, frame_to=frame_to)
            merged = sorted(set(cached or []) | set(fresh))
            selfcal.save(video, merged)
            samples = merged
        else:
            samples = cached
        self.series = selfcal.FocalSeries(samples)
        self.note = f"замеров фокусного {len(self.series)}"

    def _covers(self, samples, frame_from: int, frame_to: int | None) -> bool:
        """Покрывает ли кэш запрошенный диапазон кадров.

        Полный проход (0..конец) всегда покрывает. Иначе достаточно, чтобы в
        диапазоне был хотя бы один замер: пустой диапазон - это либо неподвижный
        подвес, либо просто неизмеренный участок, и различить их можно только
        измерением.
        """
        if frame_from == 0 and frame_to is None:
            return bool(samples)
        hi = frame_to if frame_to is not None else self.n_frames
        return any(frame_from <= s[0] <= hi for s in samples)

    def focal(self, frame: int) -> tuple[float | None, float, str]:
        """(фокусное, относительная ошибка, причина отказа)."""
        if self.fixed_focal:
            return self.fixed_focal, 0.0, ""
        f = self.series.at(frame)
        if f is None:
            return None, 0.0, "нет замеров фокусного рядом с этим кадром"
        dis = self.series.axis_disagreement(frame)
        if dis is not None and dis > selfcal.MAX_AXIS_DISAGREEMENT:
            return None, 0.0, (f"оценки фокусного по рысканию и по тангажу расходятся на "
                               f"{dis:.0%}: сопоставление полей углов под вопросом")
        return f, self.series.rel_error(frame), ""

    def sampled(self, step: int):
        n = max(1, step)
        return self.fixes[::n]


def load_flights(videos, **kw) -> list[Flight]:
    out = []
    for v in videos:
        try:
            out.append(Flight(Path(v), **kw))
        except (GeorefError, RuntimeError) as exc:
            print(f"ОТКАЗ {Path(v).name}: {exc}", file=sys.stderr)
    return out


# --- tochka: пиксель -> координата -----------------------------------------


def cmd_tochka(args) -> int:
    terrain = Terrain(args.dem)
    video = Path(args.video)
    t_s = parse_timecode(args.time)
    fps, _, _, _ = video_geometry(video)
    frame = int(round(t_s * fps))
    flight = Flight(video, fixed_focal=args.focal, recalibrate=args.recalibrate,
                    frame_from=max(0, frame - selfcal.WINDOW - selfcal.BASELINE),
                    frame_to=frame + selfcal.WINDOW)
    fix = fix_at(flight.fixes, t_s)
    if fix is None:
        print(f"нет телеметрии на {args.time} (ближайшая дальше 0,5 с)")
        return 1

    x, y = (float(v) for v in args.pixel.split(","))
    # Фокусное ищется по номеру кадра ИЗ ТЕЛЕМЕТРИИ, а не по пересчитанному из
    # таймкода: метки времени в сайдкаре слегка расходятся с номером кадра, и
    # брать разные кадры для угла и для фокусного значило бы смешивать два
    # момента съёмки.
    f_px, f_err, why = flight.focal(fix.frame)
    result = locate(terrain, fix, x, y, f_px, width=flight.width, height=flight.height,
                    f_rel_err=f_err)
    if isinstance(result, Azimuth) and why and "фокусное" in result.reason:
        result = Azimuth(result.lat, result.lon, result.alt_m, result.azimuth_deg,
                         result.tilt_deg, why)

    print(f"{video.name}  {format_timecode(t_s)}  кадр {fix.frame}  пиксель {x:.0f},{y:.0f}")
    print(f"борт: {fix.lat:.6f} N {fix.lon:.6f} E, {fix.alt_ortho:.0f} м (EGM2008), "
          f"подвес азимут {fix.gb_yaw} наклон {fix.gb_pitch}")
    print(f"фокусное: {f_px:.0f} пикс (разброс {f_err:.0%})" if f_px else f"фокусное: нет ({why})")
    if isinstance(result, Azimuth):
        print(result.line())
        return 0
    print(f"КООРДИНАТА: {result.lat:.6f} N {result.lon:.6f} E, {result.elevation_m:.0f} м")
    print(f"погрешность +/-{result.uncertainty_m:.0f} м ({result.grade}); "
          f"дальность {result.range_m:.0f} м, угол падения луча {result.grazing_deg:.0f} град, "
          f"крутизна склона {result.slope_deg:.0f} град")
    print(f"наземное разрешение {100 * result.range_m / f_px:.1f} см/пикс")
    return 0


# --- kadry: координата -> кадры --------------------------------------------


def sightings(flight: Flight, terrain: Terrain, lat: float, lon: float, elev: float,
              step: int):
    """[(кадр, t_s, пиксель, дальность, угол, см/пикс, перекрытие)] для одной точки."""
    out = []
    for fix in flight.sampled(step):
        f_px, _, _ = flight.focal(fix.frame)
        if not f_px:
            continue
        xy = project(fix, f_px, lat, lon, elev, width=flight.width, height=flight.height)
        if xy is None or not all(map(math.isfinite, xy)):
            continue
        if not (0 <= xy[0] < flight.width and 0 <= xy[1] < flight.height):
            continue
        north = (lat - fix.lat) * (EARTH_R * math.pi / 180.0)
        east = (lon - fix.lon) * (EARTH_R * math.cos(math.radians(fix.lat)) * math.pi / 180.0)
        d = np.array([east, north, elev - fix.alt_ortho])
        slant = float(np.linalg.norm(d))
        if slant <= 0:
            continue
        d = d / slant
        # Точка может быть в поле зрения и при этом за гребнем: проекция про
        # рельеф не знает. Пускаем тот же луч тем же трассировщиком.
        hit = cast_ray(terrain, fix.lat, fix.lon, fix.alt_ortho, d, max_range=slant * 2.0)
        if hit is None:
            occl = "не определено"
        elif slant - hit.range_m > OCCLUSION_MARGIN_M:
            occl = "перекрыто"
        else:
            occl = "видно"
        nrm = terrain.normal(lat, lon)
        graz = 0.0 if nrm is None else math.degrees(math.asin(min(1.0, abs(float(np.dot(-d, nrm))))))
        out.append((fix.frame, fix.t_s, xy, slant, graz, 100.0 * slant / f_px, occl))
    return out


def cmd_kadry(args) -> int:
    terrain = Terrain(args.dem)
    lat, lon = (float(v) for v in args.coord.split(","))
    elev = args.elev if args.elev is not None else terrain.elevation(lat, lon)
    if elev is None:
        print("точка вне тайла модели высот")
        return 1
    print(f"цель {lat:.6f} N {lon:.6f} E, высота {elev:.0f} м "
          f"({'задана' if args.elev is not None else 'из модели высот'})")

    flights = load_flights(args.videos, recalibrate=args.recalibrate, fixed_focal=args.focal)
    rows, seen_any, no_focal = [], 0, 0
    for fl in flights:
        got = sightings(fl, terrain, lat, lon, elev, args.step)
        seen_any += len(got)
        no_focal += sum(1 for fx in fl.sampled(args.step) if not fl.focal(fx.frame)[0])
        for frame, t_s, xy, slant, graz, gsd, occl in got:
            rows.append((gsd, fl.video.name, t_s, frame, xy, slant, graz, occl))
    rows.sort()

    visible = [r for r in rows if r[7] == "видно"]
    if not visible:
        print("НИ ОДИН КАДР НЕ СМОТРЕЛ СЮДА. Это не пустой ответ, а вывод: "
              "участок не осмотрен, нужен облёт.")
    else:
        best = min(r[0] for r in visible)
        verdict = ("осмотрено" if best <= SEARCHED_GSD_CM_PX
                   else f"снято, но слишком грубо (лучшее {best:.0f} см/пикс), это пролёт, не осмотр")
        print(f"вывод: {verdict}")
    print(f"\n{'видео':<34} {'таймкод':>9} {'кадр':>7} {'пиксель':>12} {'дальн':>7} "
          f"{'угол':>5} {'см/пикс':>8}  перекрытие")
    for gsd, name, t_s, frame, xy, slant, graz, occl in rows[:args.limit]:
        print(f"{name:<34} {format_timecode(t_s):>9} {frame:>7} "
              f"{int(xy[0]):>5},{int(xy[1]):<6} {slant:>7.0f} {graz:>5.0f} {gsd:>8.1f}  {occl}")
    if len(rows) > args.limit:
        print(f"... ещё {len(rows) - args.limit} кадров (--limit)")
    print(f"\nкадров просмотрено: {sum(len(fl.sampled(args.step)) for fl in flights)}, "
          f"в поле зрения {seen_any}, из них видно {len(visible)}; "
          f"пропущено без фокусного {no_focal}")
    return 0


# --- pokrytie: карта покрытия и линия падения -------------------------------


def coverage(flights, terrain: Terrain, step: int) -> tuple[dict, dict]:
    """{(строка, столбец) DEM: лучшее см/пикс} и счётчики.

    Клетка засчитывается, только если луч через пиксель кадра реально упёрся в
    неё: трассировка сама отсекает то, что за гребнем.
    """
    cells: dict[tuple[int, int], float] = {}
    stats = {"кадров": 0, "без углов подвеса": 0, "без фокусного": 0, "лучей": 0,
             "попаданий": 0, "точек с координатой": 0}
    for fl in flights:
        for fix in fl.sampled(step):
            stats["кадров"] += 1
            # Строки без углов подвеса встречаются в реальных сайдкарах (потеря
            # пакета). Такой кадр не покрывает ничего: куда смотрела камера,
            # неизвестно.
            if not fix.can_point:
                stats["без углов подвеса"] += 1
                continue
            f_px, f_err, _ = fl.focal(fix.frame)
            if not f_px:
                stats["без фокусного"] += 1
                continue
            for gy in range(COVERAGE_GRID):
                for gx in range(COVERAGE_GRID):
                    x = (gx + 0.5) * fl.width / COVERAGE_GRID
                    y = (gy + 0.5) * fl.height / COVERAGE_GRID
                    d = pixel_ray(x, y, fix.gb_yaw, fix.gb_pitch, f_px,
                                  width=fl.width, height=fl.height)
                    stats["лучей"] += 1
                    hit = cast_ray(terrain, fix.lat, fix.lon, fix.alt_ortho, d,
                                   max_range=COVERAGE_MAX_RANGE_M)
                    if hit is None or hit.grazing_deg < 5.0:
                        continue
                    stats["попаданий"] += 1
                    # Снято - ещё не значит «можно назвать координату». Считаем
                    # отдельно, сколько лучей прошли бы пороги `locate`: это и
                    # есть доля, по которой инструмент выдаёт точку, а не азимут.
                    unc = uncertainty_m(hit.range_m, hit.grazing_deg, hit.slope_deg,
                                        f_px=f_px, f_rel_err=f_err,
                                        offaxis_px=math.hypot(x - fl.width / 2,
                                                              y - fl.height / 2))
                    if (hit.grazing_deg >= GRAZING_MIN_DEG and unc <= UNCERTAINTY_MAX_M
                            and unc <= UNCERTAINTY_RANGE_RATIO * hit.range_m):
                        stats["точек с координатой"] += 1
                    r, c = terrain.rowcol(hit.lat, hit.lon)
                    key = (int(round(r)), int(round(c)))
                    gsd = 100.0 * hit.range_m / f_px
                    if gsd < cells.get(key, float("inf")):
                        cells[key] = gsd
    return cells, stats


def fall_line(terrain: Terrain, lat: float, lon: float, max_steps: int = 400):
    """Линия наибыстрейшего спуска по DEM: [(широта, долгота, высота)].

    Падающий или скользящий предмет идёт по линии наибольшего уклона, а не по
    прямой. Это не предсказание, где что лежит, а перечень мест, достижимых
    сверху под действием тяжести.
    """
    band = terrain.band
    t = terrain.transform
    # Клетка по широте и по долготе разная: на 39,5 N градус долготы короче
    # градуса широты в 1,3 раза. Одна величина на обе оси перекосила бы выбор
    # соседа в пользу движения по долготе, и линия падения поехала бы вбок.
    m_lat = abs(t.e) * math.pi / 180.0 * EARTH_R
    m_lon = abs(t.a) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat))
    r, c = terrain.rowcol(lat, lon)
    r, c = int(round(r)), int(round(c))
    path, seen = [], set()
    for _ in range(max_steps):
        if (r, c) in seen or not (0 <= r < band.shape[0] and 0 <= c < band.shape[1]):
            break
        seen.add((r, c))
        path.append((t.f + r * t.e, t.c + c * t.a, float(band[r, c])))
        best, best_k = 0.0, -1
        for k, (dr, dc) in enumerate(_D8):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < band.shape[0] and 0 <= nc < band.shape[1]):
                continue
            # Уклон, а не перепад: иначе диагональные соседи выигрывали бы просто
            # потому, что они дальше.
            dist = math.hypot(dr * m_lat, dc * m_lon)
            drop = (band[r, c] - band[nr, nc]) / dist
            if drop > best:
                best, best_k = drop, k
        if best_k < 0:
            break
        r, c = r + _D8[best_k][0], c + _D8[best_k][1]
    return path


def cmd_pokrytie(args) -> int:
    terrain = Terrain(args.dem)
    flights = load_flights(args.videos, recalibrate=args.recalibrate, fixed_focal=args.focal)
    if not flights:
        return 1
    for fl in flights:
        d = datum_check(terrain, fl.fixes)
        if d.get("n"):
            print(f"{fl.video.name}: {fl.note}; превышение над рельефом "
                  f"{d['clearance_median']:+.0f} м (мин {d['clearance_min']:+.0f}), "
                  f"под рельефом без поправки геоида {d['below_raw']:.0%}, с поправкой "
                  f"{d['below_ortho']:.0%}")

    cells, stats = coverage(flights, terrain, args.step)
    area = DEM_POSTING_M ** 2
    searched = {k: v for k, v in cells.items() if v <= SEARCHED_GSD_CM_PX}
    print(f"\nпокрытие: клеток DEM задето {len(cells)} ({len(cells) * area / 1e6:.2f} кв.км), "
          f"из них осмотрено не грубее {SEARCHED_GSD_CM_PX:.0f} см/пикс - "
          f"{len(searched)} ({len(searched) * area / 1e6:.2f} кв.км)")
    coord = stats["точек с координатой"]
    print(f"кадров {stats['кадров']}, без углов подвеса {stats['без углов подвеса']}, "
          f"без фокусного {stats['без фокусного']} "
          f"({stats['без фокусного'] / max(1, stats['кадров']):.0%}), "
          f"лучей {stats['лучей']}, попаданий в рельеф {stats['попаданий']}, "
          f"из них годны для координаты {coord} "
          f"({coord / max(1, stats['лучей']):.0%} от всех лучей); "
          f"остальное - только азимут")
    if cells:
        v = np.array(sorted(cells.values()))
        print(f"наземное разрешение по клеткам: медиана {np.median(v):.0f} см/пикс, "
              f"лучшее {v[0]:.1f}, худшее {v[-1]:.0f}")

    if args.fall_line:
        lat, lon = (float(x) for x in args.fall_line.split(","))
        path = fall_line(terrain, lat, lon)
        print(f"\nлиния падения от {lat:.6f} N {lon:.6f} E: {len(path)} шагов, "
              f"с {path[0][2]:.0f} м до {path[-1][2]:.0f} м, "
              f"конец {path[-1][0]:.6f} N {path[-1][1]:.6f} E")
        print(f"{'шаг':>4} {'широта':>11} {'долгота':>11} {'выс':>6} {'см/пикс':>8}  состояние")
        gaps = 0
        for i, (la, lo, z) in enumerate(path[:args.fall_line_steps]):
            r, c = terrain.rowcol(la, lo)
            gsd = cells.get((int(round(r)), int(round(c))))
            if gsd is None:
                state, shown = "НЕ СНЯТО", "-"
                gaps += 1
            elif gsd <= SEARCHED_GSD_CM_PX:
                state, shown = "осмотрено", f"{gsd:.0f}"
            else:
                state, shown = "снято грубо", f"{gsd:.0f}"
            print(f"{i:>4} {la:>11.6f} {lo:>11.6f} {z:>6.0f} {shown:>8}  {state}")
        total_gaps = sum(1 for la, lo, _ in path
                         if (int(round(terrain.rowcol(la, lo)[0])),
                             int(round(terrain.rowcol(la, lo)[1]))) not in cells)
        print(f"не снято шагов: {total_gaps} из {len(path)} - это заявка штабу на доснятие")
    return 0


# --- разбор аргументов ------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dem", default=DEFAULT_DEM, help="тайл Copernicus GLO-30")
    ap.add_argument("--focal", type=float, help="задать фокусное в пикселях вместо самокалибровки")
    ap.add_argument("--recalibrate", action="store_true", help="пересчитать фокусное, игнорируя кэш")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tochka", help="пиксель кадра -> координата")
    p.add_argument("video", type=Path)
    p.add_argument("--time", required=True, help="таймкод, например 3:32")
    p.add_argument("--pixel", required=True, help="x,y в кадре")
    p.set_defaults(func=cmd_tochka)

    p = sub.add_parser("kadry", help="координата -> кадры, которые её видели")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--coord", required=True, help="широта,долгота")
    p.add_argument("--elev", type=float, help="высота цели, м (иначе из модели высот)")
    p.add_argument("--step", type=int, default=15, help="брать каждый N-й кадр телеметрии")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_kadry)

    p = sub.add_parser("pokrytie", help="карта покрытия и линия падения")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--step", type=int, default=30)
    p.add_argument("--fall-line", help="широта,долгота начала линии наибыстрейшего спуска")
    p.add_argument("--fall-line-steps", type=int, default=40)
    p.set_defaults(func=cmd_pokrytie)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except GeorefError as exc:
        print(f"ОТКАЗ: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
