#!/usr/bin/env python3
"""Геопроекция взгляда камеры: фокусное фотограмметрией, дистанция до склона, масштаб.

Три инструмента (слой «геоиндекс» конвейера, docs/video-analysis.md п. 1):

  focal ВИДЕО T0 T1   — фокусное в пикселях на интервале панорамирования [T0, T1]:
                        оптический сдвиг между соседними сэмплами (phaseCorrelate),
                        делённый на поворот подвеса из телеметрии за тот же интервал.
                        Дрон должен висеть (GPS не меняется), иначе сдвиг не чисто
                        вращательный. Печатает медиану и разброс по парам кадров.
  cast ВИДЕО T PX PY F — трассировка луча: из позиции дрона в момент T через пиксель
                        (PX, PY) при фокусном F (пикс.) до пересечения с рельефом
                        Copernicus GLO-30 (data/dem/N39E073.tif). Печатает координаты
                        точки, дистанцию и метры-на-пиксель (GSD).
  elev LAT LON        — высота рельефа в точке (проверка DEM).

Семантика углов телеметрии (gb_yaw: 0=север, по часовой; gb_pitch: минус=вниз)
подтверждена валидацией на рюкзаке: cast по видео DJI_20260813184253 в момент,
когда рюкзак виден в кадре, попадает в его триангулированные штабом координаты
(см. docs/nakhodki/README.md). Высоты DEM — эллипсоидальные поправки не вносим:
против MSL телеметрии расхождение единицы метров, для масштаба несущественно.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "data/dem/N39E073.tif"
FLOW_W = 960          # ширина центрального окна для phaseCorrelate
RAY_STEP_M = 5.0      # шаг марша луча
RAY_MAX_M = 6000.0


# --- DEM ---------------------------------------------------------------------


class Dem:
    def __init__(self, path=DEM_PATH):
        tif = tifffile.TiffFile(path)
        page = tif.pages[0]
        self.z = page.asarray().astype(np.float32)
        scale = page.tags["ModelPixelScaleTag"].value      # (sx, sy, sz)
        tie = page.tags["ModelTiepointTag"].value          # (i, j, k, x, y, z)
        self.lon0, self.lat0 = tie[3], tie[4]              # верхний левый угол
        self.dlon, self.dlat = scale[0], scale[1]
        self.h, self.w = self.z.shape

    def elev(self, lat, lon):
        """Билинейная высота рельефа, м."""
        x = (lon - self.lon0) / self.dlon
        y = (self.lat0 - lat) / self.dlat
        if not (0 <= x < self.w - 1 and 0 <= y < self.h - 1):
            raise ValueError(f"точка вне тайла DEM: {lat}, {lon}")
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        z = self.z
        return float(z[y0, x0] * (1 - fx) * (1 - fy) + z[y0, x0 + 1] * fx * (1 - fy)
                     + z[y0 + 1, x0] * (1 - fx) * fy + z[y0 + 1, x0 + 1] * fx * fy)


# --- телеметрия ---------------------------------------------------------------


def load_rows(video: Path):
    side = video.with_suffix(video.suffix + ".gps.tsv")
    with side.open(encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append({k: float(v) for k, v in r.items() if v != ""})
            except ValueError:
                continue
    return rows


def at(rows, t, key):
    """Линейная интерполяция поля key в момент t (yaw разворачивается от скачков ±360).

    Строки без этого поля (пакеты заголовка, потеря части полей) пропускаются.
    """
    rows = [r for r in rows if key in r and "time_s" in r]
    ts = [r["time_s"] for r in rows]
    vs = [r[key] for r in rows]
    if key.endswith("yaw"):
        vs = np.degrees(np.unwrap(np.radians(vs))).tolist()
    return float(np.interp(t, ts, vs))


# --- фокусное фотограмметрией --------------------------------------------------


def grab(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        raise ValueError(f"кадр t={t:.2f} не читается")
    return img


def gray_center(img):
    h, w = img.shape[:2]
    scale = w / FLOW_W
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (FLOW_W, int(h / scale)))
    return g.astype(np.float32), scale


def focal_px(video: Path, t0: float, t1: float, dt: float = 0.25):
    """Оценки фокусного (пикс. полного кадра) по парам кадров на [t0, t1]."""
    rows = load_rows(video)
    cap = cv2.VideoCapture(str(video))
    ests = []
    t = t0
    while t + dt <= t1:
        a, b = t, t + dt
        dyaw = math.radians(at(rows, b, "gb_yaw") - at(rows, a, "gb_yaw"))
        dpitch = math.radians(at(rows, b, "gb_pitch") - at(rows, a, "gb_pitch"))
        pitch = math.radians(at(rows, (a + b) / 2, "gb_pitch"))
        try:
            ga, sa = gray_center(grab(cap, a))
            gb, _ = gray_center(grab(cap, b))
        except ValueError:
            break
        (dx, dy), resp = cv2.phaseCorrelate(ga, gb)
        dx, dy = dx * sa, dy * sa
        # горизонтальный сдвиг задаётся рысканием (проекция на горизонт кадра —
        # cos(pitch)), вертикальный — тангажом; берём ту ось, где поворот заметнее
        if abs(dyaw) > math.radians(0.8) and abs(dyaw) > 2 * abs(dpitch):
            ests.append((-dx) / (dyaw * math.cos(pitch)))
        elif abs(dpitch) > math.radians(0.8):
            ests.append(dy / dpitch)
        t += dt
    cap.release()
    return [e for e in ests if e > 0]


# --- трассировка луча ----------------------------------------------------------


def ray_dir(yaw_deg, pitch_deg, px, py, w, h, f):
    """Единичный вектор ENU луча через пиксель (px, py); yaw 0=север по часовой,
    pitch минус=вниз; (0,0) — левый верхний угол кадра."""
    # углы отклонения от оптической оси
    ax = math.atan2(px - w / 2, f)          # вправо +
    ay = math.atan2(py - h / 2, f)          # вниз +
    yaw = math.radians(yaw_deg) + ax
    pitch = math.radians(pitch_deg) - ay
    ce = math.cos(pitch)
    return (math.sin(yaw) * ce, math.cos(yaw) * ce, math.sin(pitch))  # (E, N, Up)


def cast(dem: Dem, lat, lon, alt, direction):
    """Марш луча до рельефа → (lat, lon, alt_рельефа, дистанция) или None.

    Если стартовая точка «под» рельефом (дрон вплотную к крутому склону: 30-метровая
    сетка DEM сглаживает склон выше позиции дрона, плюс ошибка высоты GPS), начальный
    подземный участок пропускается, и ищется первое пересечение после выхода луча
    над поверхность.
    """
    de, dn, du = direction
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    prev = None
    above = False        # луч уже был над поверхностью
    d = RAY_STEP_M
    while d <= RAY_MAX_M:
        la = lat + dn * d / m_per_deg_lat
        lo = lon + de * d / m_per_deg_lon
        al = alt + du * d
        try:
            ground = dem.elev(la, lo)
        except ValueError:
            return None
        if al > ground:
            above = True
        elif above:
            lo_d, hi_d = prev, d     # уточнение бисекцией между prev и d
            for _ in range(20):
                mid = (lo_d + hi_d) / 2
                la = lat + dn * mid / m_per_deg_lat
                lo = lon + de * mid / m_per_deg_lon
                al = alt + du * mid
                if al <= dem.elev(la, lo):
                    hi_d = mid
                else:
                    lo_d = mid
            la = lat + dn * hi_d / m_per_deg_lat
            lo = lon + de * hi_d / m_per_deg_lon
            return la, lo, dem.elev(la, lo), hi_d
        prev = d
        d += RAY_STEP_M
    return None


def cmd_cast(args):
    dem = Dem()
    video = Path(args.video)
    rows = load_rows(video)
    lat, lon = at(rows, args.t, "lat"), at(rows, args.t, "lon")
    alt = at(rows, args.t, "alt_m")
    yaw, pitch = at(rows, args.t, "gb_yaw"), at(rows, args.t, "gb_pitch")
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    print(f"дрон: {lat:.6f}, {lon:.6f}, {alt:.0f} м; подвес yaw {yaw:.1f} pitch {pitch:.1f}")
    hit = cast(dem, lat, lon, alt, ray_dir(yaw, pitch, args.px, args.py, w, h, args.f))
    if hit is None:
        print("луч не пересёк рельеф (смотрит выше горизонта или вне тайла)")
        return
    la, lo, ground, dist = hit
    print(f"точка: {la:.6f}, {lo:.6f}, рельеф {ground:.0f} м, дистанция {dist:.0f} м")
    print(f"GSD: {dist / args.f * 100:.1f} см/пикс (объект 30 пикс ≈ {dist / args.f * 30:.2f} м)")


def cmd_focal(args):
    ests = focal_px(Path(args.video), args.t0, args.t1)
    if not ests:
        print("нет пар кадров с заметным поворотом подвеса на интервале — "
              "выберите участок панорамирования")
        return
    med = float(np.median(ests))
    print(f"оценок: {len(ests)}, медиана f = {med:.0f} пикс, "
          f"квартили {np.percentile(ests, 25):.0f}..{np.percentile(ests, 75):.0f}")


def cmd_elev(args):
    print(f"{Dem().elev(args.lat, args.lon):.1f} м")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("focal")
    p.add_argument("video")
    p.add_argument("t0", type=float)
    p.add_argument("t1", type=float)
    p.set_defaults(fn=cmd_focal)
    p = sub.add_parser("cast")
    p.add_argument("video")
    p.add_argument("t", type=float)
    p.add_argument("px", type=float)
    p.add_argument("py", type=float)
    p.add_argument("f", type=float)
    p.set_defaults(fn=cmd_cast)
    p = sub.add_parser("elev")
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.set_defaults(fn=cmd_elev)
    args = ap.parse_args()
    args.fn(args)
