#!/usr/bin/env python3
"""Геопривязка: пиксель кадра -> координата на рельефе, и обратно.

Слой 1 конвейера (`docs/video-analysis.md`). Кандидаты сейчас несут GPS дрона -
это где был борт, а не где лежит предмет. Здесь луч через пиксель пересекается
с моделью высот Copernicus GLO-30 и даёт координату самого предмета.

Геометрия - именно трассировка луча, не проекция на плоскость. Съёмка косая, с
зависающего борта поперёк склона, тангаж подвеса на местных пролётах от -70 до
+26 градусов. Плоская подстилающая поверхность (то, что делает ортотрансформация
надирных снимков) увела бы кандидата на сотни метров.

Три вещи, из-за которых модуль отказывается считать, вместо того чтобы выдать
правдоподобное число:

1. **Фокусное расстояние.** В метаданных M30T его нет: два поля меняются
   правдоподобно, но ни одно не привязывается к фокусному из первых принципов.
   Восстанавливается фотограмметрически (`analysis/selfcal.py`). Где не
   восстановилось - выдаётся только азимут, как в отчётах штаба.
2. **Датум высоты.** DJI пишет высоту над эллипсоидом WGS84, GLO-30 - над
   геоидом EGM2008. В районе поисков разница +33,15 м (см. GEOID_N). Без
   поправки борт оказывается ниже собственного рельефа: измерено на пяти
   локальных видео - 100% трека под землёй на трёх из них, после поправки 0%.
   Каждый луч наследует этот перекос целиком.
3. **Скользящий угол.** Луч, падающий на склон полого, съезжает вдоль линии
   визирования на десятки и сотни метров при ошибке высоты в единицы метров.
   Ниже GRAZING_MIN_DEG точка не выдаётся вообще.

Точность заявляется, а не подразумевается. GLO-30 - шаг 30 м, на 40-градусном
склоне это уже десятки метров по высоте; погрешность считается в
`uncertainty_m()` и растёт с дальностью и с пологостью падения луча.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_R = 6_371_000.0

# Модель высот. Тайл не хранится в репозитории, качается scripts/download_dem.sh.
DEFAULT_DEM = "data/dem/Copernicus_DSM_COG_10_N39_00_E073_00_DEM.tif"

# Шаг сетки GLO-30 в метрах и её горизонтальная погрешность привязки (LE90).
DEM_POSTING_M = 30.0
DEM_SIGMA_XY_M = 10.0
# Нижняя граница вертикальной погрешности DEM на пологом месте (паспорт GLO-30).
DEM_SIGMA_Z_MIN_M = 4.0
# Множитель к геометрической оценке вертикальной ошибки DEM на склоне. Не из
# паспорта: подобран так, чтобы заявленная погрешность накрывала измеренную
# невязку на эталоне. ИЗМЕРЕНО: на бирюзовой крышке (0015_Z, 4 кадра) луч
# проходит от реестровой координаты в 0,0-0,1 м вбок, но DEM останавливает его
# на 20 м вместо 43 м - это 23 м вдоль луча. DEM в точке находки выше
# фотограмметрически восстановленной поверхности на ~26 м и выше реестровых
# отметок четырёх находок на 14-61 м. Простая оценка `шаг * tg(склона)` такую
# невязку недооценивает вдвое.
DEM_RESIDUAL_K = 2.0

# Разделение эллипсоид/геоид (EGM2008) в районе Курумды: ортометрическая высота
# = высота DJI + GEOID_N. Значение получено pyproj (EPSG:4979 -> EPSG:9518) и по
# углам рамки 39.47-39.50 N, 73.57-73.60 E меняется на 0,25 м, поэтому взято
# константой: на площади операции это точнее полуметра, а pyproj тянет сетку
# геоида из сети и в поле недоступен.
GEOID_N = 33.15

# Угловая погрешность наведения: квантование углов подвеса 0,1 градуса плюс
# неучтённый крен подвеса и неподтверждённое сопоставление полей тангаж/крен
# (см. шапку scripts/dji_meta_gps.py). Оценка сверху, не измерение.
ATTITUDE_SIGMA_DEG = 0.5

# Ниже этого угла падения луча на склон точка не выдаётся: пересечение съезжает
# вдоль луча так далеко, что дальность (а значит и масштаб) не определена.
GRAZING_MIN_DEG = 10.0
# Погрешность больше этой - точку не выдаём ни при какой дальности.
UNCERTAINTY_MAX_M = 250.0
# ... и не выдаём, если погрешность сравнима с самой дальностью.
UNCERTAINTY_RANGE_RATIO = 1.5

# Столбцы сайдкара, без которых геопривязка невозможна.
REQUIRED_COLUMNS = ("time_s", "lat", "lon", "alt_m", "gb_yaw", "gb_pitch")


class GeorefError(RuntimeError):
    """Данных для геопривязки нет или они непригодны."""


# --- телеметрия -------------------------------------------------------------


@dataclass(frozen=True)
class Fix:
    """Одна строка сайдкара. Углы в градусах, высоты в метрах."""

    t_s: float
    lat: float
    lon: float
    alt_m: float           # высота DJI, над эллипсоидом WGS84
    gb_yaw: float | None
    gb_pitch: float | None
    ac_roll: float | None = None
    frame: int = 0

    @property
    def alt_ortho(self) -> float:
        """Высота в датуме DEM (EGM2008), метры."""
        return self.alt_m + GEOID_N

    @property
    def can_point(self) -> bool:
        return self.gb_yaw is not None and self.gb_pitch is not None


def sidecar_path(video: str | Path) -> Path:
    p = Path(video)
    return p.with_suffix(p.suffix + ".gps.tsv")


def read_sidecar(video: str | Path, fps: float | None = None,
                 n_frames: int | None = None) -> list[Fix]:
    """Телеметрия из `<видео>.MP4.gps.tsv`. Читается ПО ЗАГОЛОВКУ, не по позиции.

    Номер кадра для каждой строки берётся, по убыванию надёжности:

    1. Порядковый номер строки, если строк ровно столько же, сколько кадров в
       контейнере. Пакеты телеметрии идут один к одному с кадрами, так что это
       не оценка, а тождество. На пяти локальных видео совпадает точно.
    2. `round(time_s * fps)` с явно переданным fps.
    3. `round((time_s - t0) * fps)`, где fps выведен из самого сайдкара.

    Частота из контейнера через OpenCV сюда сознательно не попадает по
    умолчанию. ИЗМЕРЕНО: `CAP_PROP_FPS` даёт 29,9474..29,9700 на файлах, где
    метки времени пакетов дают 29,9692..29,9738 при номинальных 30000/1001 =
    29,9700. На 212-й секунде расхождение уже 4 кадра, и самокалибровка брала бы
    углы подвеса от чужого кадра.

    Столбцы `gb_yaw`/`gb_pitch` добавляет расширенный `scripts/dji_meta_gps.py`
    (ветка `pr/telemetry-field-paths`). Старый сайдкар с четырьмя столбцами
    (time_s, lat, lon, alt_m) читается без ошибки, но геопривязать по нему
    нельзя: без углов подвеса нет даже направления взгляда. В этом случае
    поднимается GeorefError с указанием, чем пересобрать сайдкар - молча
    подставлять курс борта вместо курса подвеса нельзя, подвес поворачивается
    относительно борта на любой угол.
    """
    path = sidecar_path(video)
    if not path.exists():
        raise GeorefError(
            f"нет сайдкара {path.name}. Собрать: python3 scripts/dji_meta_gps.py {Path(video).name}"
        )
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise GeorefError(
            f"{path.name}: нет столбцов {', '.join(missing)}. Сайдкар собран старой версией "
            "scripts/dji_meta_gps.py (только GPS). Пересоберите версией с углами борта и "
            "подвеса (ветка pr/telemetry-field-paths), иначе направление взгляда неизвестно."
        )

    def num(row: dict, key: str) -> float | None:
        v = row.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    one_to_one = n_frames is not None and len(rows) == n_frames
    times = [num(r, "time_s") for r in rows]
    clean = [t for t in times if t is not None]
    span = (clean[-1] - clean[0]) if len(clean) > 1 else 0.0
    fps_tsv = (len(rows) - 1) / span if span > 0 else None

    fixes: list[Fix] = []
    for i, row in enumerate(rows):
        t, lat, lon, alt = (num(row, k) for k in ("time_s", "lat", "lon", "alt_m"))
        if None in (t, lat, lon, alt):
            continue
        if one_to_one:
            frame = i
        elif fps:
            frame = int(round(t * fps))
        elif fps_tsv:
            frame = int(round((t - clean[0]) * fps_tsv))
        else:
            frame = i
        fixes.append(
            Fix(
                t_s=t, lat=lat, lon=lon, alt_m=alt,
                gb_yaw=num(row, "gb_yaw"), gb_pitch=num(row, "gb_pitch"),
                ac_roll=num(row, "ac_roll"), frame=frame,
            )
        )
    if not fixes:
        raise GeorefError(f"{path.name}: ни одной строки с координатами")
    return fixes


def fix_at(fixes: list[Fix], t_s: float) -> Fix | None:
    """Ближайшая по времени строка телеметрии, если она ближе полусекунды.

    Дальше полусекунды - это уже другой ракурс: при панораме 10 град/с подвес
    успевает уйти на 5 градусов, а на дальности 500 м это 44 м по земле.
    """
    if not fixes:
        return None
    best = min(fixes, key=lambda f: abs(f.t_s - t_s))
    return best if abs(best.t_s - t_s) <= 0.5 else None


def parse_timecode(text: str) -> float:
    """`3:32`, `3:32.5`, `212`, `00:03:32` -> секунды."""
    parts = str(text).split(":")
    if len(parts) > 3:
        raise ValueError(f"непонятный таймкод: {text}")
    total = 0.0
    for p in parts:
        total = total * 60.0 + float(p)
    return total


def format_timecode(t_s: float) -> str:
    return f"{int(t_s // 60)}:{t_s % 60:05.2f}"


# --- рельеф -----------------------------------------------------------------


class Terrain:
    """Высоты по тайлу Copernicus GLO-30 (ортометрические, EGM2008)."""

    def __init__(self, path: str | Path = DEFAULT_DEM):
        try:
            import rasterio
        except ImportError as exc:
            raise GeorefError("нужен rasterio: pip install rasterio") from exc
        p = Path(path)
        if not p.exists():
            raise GeorefError(f"нет модели высот {p}. Скачать: bash scripts/download_dem.sh")
        self._ds = rasterio.open(p)
        self.band = self._ds.read(1)
        self.transform = self._ds.transform
        self.nodata = self._ds.nodata

    def rowcol(self, lat: float, lon: float) -> tuple[float, float]:
        t = self.transform
        return (lat - t.f) / t.e, (lon - t.c) / t.a

    def elevation(self, lat: float, lon: float) -> float | None:
        """Билинейная высота, None вне тайла.

        Билинейно, а не по ближайшей ячейке: луч с шагом в единицы метров по
        сетке 30 м иначе поднимался бы ступеньками и останавливался не на той.
        """
        row, col = self.rowcol(lat, lon)
        r0, c0 = int(math.floor(row)), int(math.floor(col))
        h, w = self.band.shape
        if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
            return None
        fr, fc = row - r0, col - c0
        patch = self.band[r0:r0 + 2, c0:c0 + 2].astype(float)
        if self.nodata is not None and np.any(patch == self.nodata):
            return None
        top = patch[0, 0] * (1 - fc) + patch[0, 1] * fc
        bot = patch[1, 0] * (1 - fc) + patch[1, 1] * fc
        return float(top * (1 - fr) + bot * fr)

    def normal(self, lat: float, lon: float) -> np.ndarray | None:
        """Единичная нормаль к склону в ENU, по разностям на шаге DEM."""
        d = DEM_POSTING_M
        zn = self.elevation(*offset(lat, lon, 0, d))
        zs = self.elevation(*offset(lat, lon, 0, -d))
        ze = self.elevation(*offset(lat, lon, d, 0))
        zw = self.elevation(*offset(lat, lon, -d, 0))
        if None in (zn, zs, ze, zw):
            return None
        n = np.array([-(ze - zw) / (2 * d), -(zn - zs) / (2 * d), 1.0])
        return n / np.linalg.norm(n)

    def slope_deg(self, lat: float, lon: float) -> float:
        n = self.normal(lat, lon)
        return 0.0 if n is None else math.degrees(math.acos(min(1.0, abs(float(n[2])))))


def offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    dlat = north_m / (EARTH_R * math.pi / 180.0)
    dlon = east_m / (EARTH_R * math.cos(math.radians(lat)) * math.pi / 180.0)
    return lat + dlat, lon + dlon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


# --- геометрия камеры -------------------------------------------------------


def camera_basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оси камеры в местной системе Восток-Север-Верх: (вперёд, вправо, вниз).

    `yaw` - азимут по часовой от севера, `pitch` - положительный вверх. Крен
    подвеса не моделируется: в телеметрии его нет отдельным полем, а крен борта
    подвес компенсирует. Остаточный крен уходит в ATTITUDE_SIGMA_DEG.
    """
    a, p = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(a), -math.sin(a), 0.0])
    return forward, right, np.cross(forward, right)


def pixel_ray(x: float, y: float, yaw_deg: float, pitch_deg: float, f_px: float,
              *, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Единичное направление в ENU для луча через пиксель (x, y)."""
    forward, right, down = camera_basis(yaw_deg, pitch_deg)
    d = (x - width / 2.0) * right + (y - height / 2.0) * down + f_px * forward
    return d / np.linalg.norm(d)


@dataclass(frozen=True)
class GroundPoint:
    """Точка на рельефе с честной оценкой погрешности."""

    lat: float
    lon: float
    elevation_m: float
    range_m: float
    grazing_deg: float      # угол между лучом и плоскостью склона
    slope_deg: float
    uncertainty_m: float    # 1 сигма по горизонтали
    reason: str = ""        # почему точка непригодна, если непригодна

    @property
    def usable(self) -> bool:
        return not self.reason

    @property
    def grade(self) -> str:
        if self.reason:
            return "только азимут"
        if self.uncertainty_m > 60.0 or self.uncertainty_m > 0.5 * self.range_m:
            return "ориентировочно"
        return "надёжно"


@dataclass(frozen=True)
class Azimuth:
    """Отказ выдать точку: только точка съёмки и направление взгляда.

    Формат тот же, каким штаб уже описывает такие кандидаты в реестре
    ("только азимут"): откуда смотрели и куда, без дальности.
    """

    lat: float
    lon: float
    alt_m: float
    azimuth_deg: float
    tilt_deg: float
    reason: str

    def line(self) -> str:
        return (f"только азимут: съёмка с {self.lat:.6f} N {self.lon:.6f} E "
                f"{self.alt_m:.0f} м, азимут {self.azimuth_deg:.1f} град, "
                f"наклон {self.tilt_deg:+.1f} град. Причина: {self.reason}")


def uncertainty_m(range_m: float, grazing_deg: float, slope_deg: float,
                  *, f_px: float | None = None, f_rel_err: float = 0.0,
                  offaxis_px: float = 0.0) -> float:
    """Горизонтальная погрешность точки, метры (1 сигма).

    Три вклада, складываются квадратично:

    * **Модель высот.** На склоне крутизной s ошибка привязки в один шаг сетки
      даёт ошибку высоты порядка `шаг * tg(s)`; она сдвигает пересечение вдоль
      луча на `эта высота / tg(скользящий угол)`. Это главный член на всех
      дальностях: на местной съёмке DEM выше реестровых отметок находок на
      14-61 м. Множитель DEM_RESIDUAL_K подобран по измеренной невязке.
    * **Наведение.** Угловая ошибка ATTITUDE_SIGMA_DEG плюс вклад ошибки
      фокусного для пикселей вдали от центра кадра; переводится в метры как
      `дальность * угол / sin(скользящий угол)`.
    * **Плановая привязка DEM.** DEM_SIGMA_XY_M, постоянная.

    Погрешность растёт с дальностью (второй член) и с пологостью падения луча
    (оба первых), как и должна.
    """
    graz = math.radians(max(grazing_deg, 0.5))
    sigma_z = DEM_RESIDUAL_K * math.hypot(
        DEM_POSTING_M * math.tan(math.radians(min(slope_deg, 80.0))), DEM_SIGMA_Z_MIN_M)
    along = sigma_z / math.tan(graz)

    ang = math.radians(ATTITUDE_SIGMA_DEG)
    if f_px and f_rel_err:
        ang = math.hypot(ang, (offaxis_px / f_px) * f_rel_err)
    across = range_m * ang / math.sin(graz)

    return float(math.sqrt(along ** 2 + across ** 2 + DEM_SIGMA_XY_M ** 2))


def cast_ray(terrain: Terrain, lat: float, lon: float, alt: float, direction: np.ndarray,
             *, max_range: float = 20_000.0, step: float = 10.0, refine: int = 30):
    """Марш луча до ухода под рельеф, затем деление отрезка пополам.

    Возвращает GroundPoint или None, если луч не встретил землю (смотрит в небо
    или ушёл за край тайла). Этот случай обязан отличаться от попадания:
    иначе на дальнем конце луча появилась бы выдуманная координата на горизонте.
    """
    ground0 = terrain.elevation(lat, lon)
    if ground0 is None:
        return None
    if alt - ground0 <= 0:
        # Борт под поверхностью: либо датум высоты неверен, либо DEM здесь
        # врёт. Любая координата отсюда - вымысел.
        return None
    prev_gap, prev_t, t = alt - ground0, 0.0, step
    while t <= max_range:
        e, n, u = direction * t
        plat, plon = offset(lat, lon, e, n)
        ground = terrain.elevation(plat, plon)
        if ground is None:
            return None
        gap = (alt + u) - ground
        if prev_gap > 0 >= gap:
            lo, hi = prev_t, t
            for _ in range(refine):
                mid = (lo + hi) / 2
                e, n, u = direction * mid
                mlat, mlon = offset(lat, lon, e, n)
                g = terrain.elevation(mlat, mlon)
                if g is None:
                    return None
                lo, hi = (mid, hi) if (alt + u) - g > 0 else (lo, mid)
            rng = (lo + hi) / 2
            e, n, u = direction * rng
            hlat, hlon = offset(lat, lon, e, n)
            elev = terrain.elevation(hlat, hlon)
            nrm = terrain.normal(hlat, hlon)
            if elev is None or nrm is None:
                return None
            grazing = math.degrees(math.asin(min(1.0, abs(float(np.dot(-direction, nrm))))))
            slope = math.degrees(math.acos(min(1.0, abs(float(nrm[2])))))
            # Голая геометрия, без оценки погрешности: её ставит `locate`, у
            # которой есть фокусное и положение пикселя в кадре. Пока её нет,
            # точка помечена непригодной, чтобы промежуточный результат нельзя
            # было принять за готовую координату.
            return GroundPoint(hlat, hlon, elev, rng, grazing, slope, float("inf"),
                               reason="погрешность ещё не вычислена")
        prev_gap, prev_t = gap, t
        # Дальше от камеры шаг крупнее: 10 м на дальности 2 км не покупают ничего.
        t += step * (1.0 + t / 2000.0)
    return None


def locate(terrain: Terrain, fix: Fix, x: float, y: float, f_px: float | None,
           *, width: int = 1920, height: int = 1080,
           f_rel_err: float = 0.0) -> GroundPoint | Azimuth:
    """Координата для пикселя, либо отказ с одним азимутом.

    Отказ - правильный ответ. Уверенно неверная точка отправляет спасателей на
    чужой склон, а азимут просто сужает сектор.
    """
    if not fix.can_point:
        return Azimuth(fix.lat, fix.lon, fix.alt_ortho, float("nan"), float("nan"),
                       "в телеметрии нет углов подвеса")
    az_hint = Azimuth(fix.lat, fix.lon, fix.alt_ortho, fix.gb_yaw, fix.gb_pitch, "")

    if not f_px:
        return Azimuth(fix.lat, fix.lon, fix.alt_ortho, fix.gb_yaw, fix.gb_pitch,
                       "фокусное расстояние не восстановлено на этом участке съёмки")

    d = pixel_ray(x, y, fix.gb_yaw, fix.gb_pitch, f_px, width=width, height=height)
    hit = cast_ray(terrain, fix.lat, fix.lon, fix.alt_ortho, d)
    if hit is None:
        return Azimuth(az_hint.lat, az_hint.lon, az_hint.alt_m, fix.gb_yaw, fix.gb_pitch,
                       "луч не пересекает рельеф (небо, край тайла или борт ниже модели высот)")

    offaxis = math.hypot(x - width / 2.0, y - height / 2.0)
    unc = uncertainty_m(hit.range_m, hit.grazing_deg, hit.slope_deg,
                        f_px=f_px, f_rel_err=f_rel_err, offaxis_px=offaxis)

    reason = ""
    if hit.grazing_deg < GRAZING_MIN_DEG:
        reason = (f"луч падает на склон под {hit.grazing_deg:.1f} град, дальность не определена")
    elif unc > UNCERTAINTY_MAX_M:
        reason = f"погрешность +/-{unc:.0f} м"
    elif unc > UNCERTAINTY_RANGE_RATIO * hit.range_m:
        reason = (f"погрешность +/-{unc:.0f} м при дальности {hit.range_m:.0f} м: "
                  "масштаб сцены не определён")
    if reason:
        return Azimuth(az_hint.lat, az_hint.lon, az_hint.alt_m, fix.gb_yaw, fix.gb_pitch, reason)

    return GroundPoint(hit.lat, hit.lon, hit.elevation_m, hit.range_m,
                       hit.grazing_deg, hit.slope_deg, unc)


def project(fix: Fix, f_px: float, lat: float, lon: float, elevation: float,
            *, width: int = 1920, height: int = 1080) -> tuple[float, float] | None:
    """Пиксель, в который попадает точка земли. Обратная задача к `locate`."""
    if not fix.can_point or not f_px:
        return None
    north = (lat - fix.lat) * (EARTH_R * math.pi / 180.0)
    east = (lon - fix.lon) * (EARTH_R * math.cos(math.radians(fix.lat)) * math.pi / 180.0)
    d = np.array([east, north, elevation - fix.alt_ortho])
    forward, right, down = camera_basis(fix.gb_yaw, fix.gb_pitch)
    z = float(np.dot(d, forward))
    if z <= 1e-6:
        return None  # за спиной у камеры
    return (f_px * float(np.dot(d, right)) / z + width / 2.0,
            f_px * float(np.dot(d, down)) / z + height / 2.0)


def datum_check(terrain: Terrain, fixes: list[Fix], *, stride: int = 30) -> dict:
    """Насколько высота телеметрии согласуется с DEM. Диагностика, не поправка.

    Признак перепутанного датума однозначен: борт летит ниже собственного
    рельефа. Функция считает эту долю с поправкой GEOID_N и без неё.
    """
    raw, ortho = [], []
    for f in fixes[::stride]:
        g = terrain.elevation(f.lat, f.lon)
        if g is None:
            continue
        raw.append(f.alt_m - g)
        ortho.append(f.alt_ortho - g)
    if not raw:
        return {"n": 0}
    raw_a, ort_a = np.array(raw), np.array(ortho)
    return {
        "n": len(raw),
        "below_raw": float((raw_a < 0).mean()),
        "below_ortho": float((ort_a < 0).mean()),
        "clearance_min": float(ort_a.min()),
        "clearance_median": float(np.median(ort_a)),
        "clearance_max": float(ort_a.max()),
    }
