#!/usr/bin/env python3
"""Тесты геоиндекса на синтетической геометрии: ни видео, ни тайла DEM не нужно.

Запуск: `python3 analysis/test_geoindex.py` (или `pytest analysis/test_geoindex.py`).

Проверяется то, что ломается молча и дорого: знаки осей камеры, согласованность
прямой и обратной задачи, отказ вместо выдуманной точки, чтение сайдкара по
заголовку, направление линии наибыстрейшего спуска.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import georef  # noqa: E402
from georef import (  # noqa: E402
    Azimuth, Fix, GeorefError, GroundPoint, camera_basis, haversine_m, locate, offset,
    parse_timecode, pixel_ray, project, read_sidecar, uncertainty_m,
)

LAT0, LON0 = 39.4830, 73.5855


class PlaneTerrain:
    """Аналитический рельеф: плоскость с заданным уклоном к югу.

    Интерфейс тот же, что у `georef.Terrain`, чтобы трассировщик не отличал.
    """

    def __init__(self, z0: float = 4000.0, slope_deg: float = 0.0):
        self.z0 = z0
        self.k = math.tan(math.radians(slope_deg))   # рост высоты на метр к северу

    def elevation(self, lat, lon):
        north_m = (lat - LAT0) * (georef.EARTH_R * math.pi / 180.0)
        return self.z0 + self.k * north_m

    def normal(self, lat, lon):
        n = np.array([0.0, -self.k, 1.0])
        return n / np.linalg.norm(n)

    def slope_deg(self, lat, lon):
        return math.degrees(math.atan(abs(self.k)))


def fix(alt=4100.0, yaw=0.0, pitch=-90.0):
    return Fix(t_s=0.0, lat=LAT0, lon=LON0, alt_m=alt - georef.GEOID_N,
               gb_yaw=yaw, gb_pitch=pitch)


# --- геометрия камеры -------------------------------------------------------


def test_basis_orthonormal():
    for yaw, pitch in ((0, 0), (37, -25), (200, 61), (-140, -89)):
        f, r, d = camera_basis(yaw, pitch)
        for v in (f, r, d):
            assert abs(np.linalg.norm(v) - 1.0) < 1e-12
        assert abs(float(np.dot(f, r))) < 1e-12
        assert abs(float(np.dot(f, d))) < 1e-12


def test_forward_is_compass_bearing():
    """Азимут 90 градусов - взгляд строго на восток, 0 - на север."""
    f, _, _ = camera_basis(90.0, 0.0)
    assert f[0] > 0.999 and abs(f[1]) < 1e-9
    f, _, _ = camera_basis(0.0, 0.0)
    assert f[1] > 0.999 and abs(f[0]) < 1e-9
    f, _, _ = camera_basis(0.0, -90.0)
    assert f[2] < -0.999


def test_centre_pixel_is_optical_axis():
    d = pixel_ray(960, 540, 33.0, -20.0, 5000.0)
    f, _, _ = camera_basis(33.0, -20.0)
    assert np.allclose(d, f, atol=1e-12)


def test_pixel_below_centre_looks_lower():
    """Пиксель ниже центра кадра смотрит ниже: иначе перепутан знак оси `вниз`."""
    up = pixel_ray(960, 40, 0.0, -30.0, 5000.0)
    down = pixel_ray(960, 1040, 0.0, -30.0, 5000.0)
    assert down[2] < up[2]


# --- трассировка ------------------------------------------------------------


def test_nadir_hits_directly_below():
    t = PlaneTerrain(z0=4000.0)
    got = locate(t, fix(alt=4100.0, pitch=-90.0), 960, 540, 5000.0)
    assert isinstance(got, GroundPoint)
    assert abs(got.range_m - 100.0) < 0.5
    assert abs(got.grazing_deg - 90.0) < 0.5
    assert haversine_m(got.lat, got.lon, LAT0, LON0) < 1.0
    assert abs(got.elevation_m - 4000.0) < 0.5


def test_forty_five_degrees_hits_at_one_height_north():
    """Плоскость, наклон -45, азимут 0: точка ровно в `высота` метрах к северу."""
    t = PlaneTerrain(z0=4000.0)
    got = locate(t, fix(alt=4100.0, yaw=0.0, pitch=-45.0), 960, 540, 5000.0)
    assert isinstance(got, GroundPoint)
    assert abs(got.range_m - 100.0 * math.sqrt(2)) < 1.0
    north = (got.lat - LAT0) * georef.EARTH_R * math.pi / 180.0
    assert abs(north - 100.0) < 1.5
    assert abs((got.lon - LON0)) < 1e-6
    assert abs(got.grazing_deg - 45.0) < 1.0


def test_sky_ray_returns_azimuth_not_a_point():
    """Взгляд вверх над плоскостью не должен порождать координату на горизонте."""
    t = PlaneTerrain(z0=4000.0)
    got = locate(t, fix(alt=4100.0, pitch=+10.0), 960, 540, 5000.0)
    assert isinstance(got, Azimuth)
    assert "рельеф" in got.reason


def test_no_focal_returns_azimuth():
    t = PlaneTerrain(z0=4000.0)
    got = locate(t, fix(), 960, 540, None)
    assert isinstance(got, Azimuth)
    assert "окусное" in got.reason
    assert got.azimuth_deg == 0.0


def test_missing_gimbal_angles_return_azimuth():
    t = PlaneTerrain(z0=4000.0)
    f = Fix(t_s=0.0, lat=LAT0, lon=LON0, alt_m=4000.0, gb_yaw=None, gb_pitch=None)
    got = locate(t, f, 960, 540, 5000.0)
    assert isinstance(got, Azimuth)


def test_shallow_grazing_is_refused():
    """Луч, скользящий вдоль склона, даёт азимут, а не уверенную точку."""
    # Склон падает к югу на 40 градусов, камера смотрит вниз на 45: угол
    # падения луча на поверхность получается 5 градусов.
    t = PlaneTerrain(z0=4000.0, slope_deg=40.0)
    got = locate(t, fix(alt=4300.0, yaw=180.0, pitch=-45.0), 960, 540, 8000.0)
    assert isinstance(got, Azimuth), got
    assert "град" in got.reason or "погрешность" in got.reason


def test_camera_under_terrain_refused():
    """Борт ниже собственного рельефа - датум или DEM врут, координат не выдаём."""
    t = PlaneTerrain(z0=4000.0)
    got = locate(t, fix(alt=3900.0, pitch=-90.0), 960, 540, 5000.0)
    assert isinstance(got, Azimuth)


# --- прямая и обратная задача ----------------------------------------------


def test_round_trip_pixel_ground_pixel():
    """locate -> project возвращает исходный пиксель. Это же и связь двух команд."""
    t = PlaneTerrain(z0=4000.0, slope_deg=15.0)
    f = fix(alt=4400.0, yaw=190.0, pitch=-35.0)
    for px, py in ((960, 540), (300, 200), (1700, 900), (41, 282)):
        got = locate(t, f, px, py, 6000.0)
        assert isinstance(got, GroundPoint), (px, py, got)
        back = project(f, 6000.0, got.lat, got.lon, got.elevation_m)
        assert back is not None
        assert abs(back[0] - px) < 1.0 and abs(back[1] - py) < 1.0, (px, py, back)


def test_project_refuses_points_behind_camera():
    f = fix(yaw=0.0, pitch=0.0)
    behind = offset(LAT0, LON0, 0.0, -500.0)
    assert project(f, 5000.0, behind[0], behind[1], 4100.0) is None


# --- погрешность ------------------------------------------------------------


def test_uncertainty_grows_with_range_and_falls_with_grazing():
    near = uncertainty_m(100.0, 60.0, 30.0)
    far = uncertainty_m(2000.0, 60.0, 30.0)
    assert far > near
    steep = uncertainty_m(500.0, 70.0, 30.0)
    shallow = uncertainty_m(500.0, 12.0, 30.0)
    assert shallow > 3 * steep
    # Крутой склон хуже пологого при том же угле падения луча.
    assert uncertainty_m(500.0, 40.0, 50.0) > uncertainty_m(500.0, 40.0, 5.0)


def test_focal_error_only_matters_off_axis():
    centre = uncertainty_m(1000.0, 45.0, 30.0, f_px=8000.0, f_rel_err=0.2, offaxis_px=0.0)
    edge = uncertainty_m(1000.0, 45.0, 30.0, f_px=8000.0, f_rel_err=0.2, offaxis_px=1000.0)
    assert edge > centre


def test_raw_cast_is_not_mistaken_for_a_finished_point():
    """cast_ray отдаёт геометрию без погрешности - и помечает её непригодной."""
    from georef import cast_ray
    t = PlaneTerrain(z0=4000.0)
    raw = cast_ray(t, LAT0, LON0, 4100.0, np.array([0.0, 0.0, -1.0]))
    assert raw is not None
    assert not raw.usable and raw.grade == "только азимут"


def test_grade_labels():
    good = GroundPoint(0, 0, 0, 1000.0, 60.0, 20.0, 20.0)
    rough = GroundPoint(0, 0, 0, 100.0, 60.0, 20.0, 70.0)
    assert good.grade == "надёжно"
    assert rough.grade == "ориентировочно"
    assert GroundPoint(0, 0, 0, 100.0, 5.0, 20.0, 900.0, reason="слишком полого").grade == \
        "только азимут"


# --- телеметрия -------------------------------------------------------------


def test_sidecar_is_read_by_header_not_by_position():
    """Столбцы переставлены и добавлен лишний - чтение обязано это пережить."""
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / "X.MP4"
        video.write_bytes(b"")
        (Path(d) / "X.MP4.gps.tsv").write_text(
            "gb_pitch\tlat\tvoltage\tlon\ttime_s\tgb_yaw\talt_m\n"
            "-30.5\t39.4830\t22.1\t73.5855\t1.00\t126.1\t4550.0\n"
            "-30.6\t39.4831\t22.0\t73.5856\t1.50\t126.4\t4551.0\n",
            encoding="utf-8")
        fixes = read_sidecar(video, fps=30.0)
    assert len(fixes) == 2
    assert fixes[0].gb_yaw == 126.1 and fixes[0].gb_pitch == -30.5
    assert fixes[0].alt_m == 4550.0
    assert fixes[1].frame == 45


def test_sidecar_frame_index_follows_row_order_when_counts_match():
    """Строк столько же, сколько кадров - номер кадра равен номеру строки.

    Это защита от подсчёта по частоте кадров: на реальных файлах OpenCV даёт
    29,95 там, где метки времени пакетов дают 29,97, и к 212-й секунде номер
    кадра уезжает на четыре.
    """
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / "X.MP4"
        video.write_bytes(b"")
        rows = "".join(f"{0.033 * i:.3f}\t39.4830\t73.5855\t4550.0\t126.1\t-30.5\n"
                       for i in range(200))
        (Path(d) / "X.MP4.gps.tsv").write_text(
            "time_s\tlat\tlon\talt_m\tgb_yaw\tgb_pitch\n" + rows, encoding="utf-8")
        fixes = read_sidecar(video, n_frames=200)
    assert [f.frame for f in fixes[:4]] == [0, 1, 2, 3]
    assert fixes[-1].frame == 199


def test_old_four_column_sidecar_is_refused_with_instructions():
    """Сайдкар без углов подвеса - отказ, а не подстановка курса борта."""
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / "X.MP4"
        video.write_bytes(b"")
        (Path(d) / "X.MP4.gps.tsv").write_text(
            "time_s\tlat\tlon\talt_m\n1.00\t39.4830\t73.5855\t4550.0\n", encoding="utf-8")
        try:
            read_sidecar(video, fps=30.0)
        except GeorefError as exc:
            assert "gb_yaw" in str(exc) and "dji_meta_gps" in str(exc)
        else:
            raise AssertionError("старый сайдкар принят молча")


def test_geoid_offset_applied_once():
    f = Fix(t_s=0.0, lat=LAT0, lon=LON0, alt_m=4550.0, gb_yaw=0.0, gb_pitch=-90.0)
    assert abs(f.alt_ortho - (4550.0 + georef.GEOID_N)) < 1e-9


def test_parse_timecode():
    assert parse_timecode("3:32") == 212.0
    assert parse_timecode("212") == 212.0
    assert abs(parse_timecode("0:03:32.5") - 212.5) < 1e-9


# --- линия наибыстрейшего спуска -------------------------------------------


def _synthetic_dem(path: Path):
    """Тайл 60x60 с плоскостью, падающей на восток, и лишним уклоном к югу."""
    import rasterio
    from rasterio.transform import from_origin
    rows = np.arange(60)[:, None]
    cols = np.arange(60)[None, :]
    z = 5000.0 - 3.0 * cols - 1.0 * rows
    tr = from_origin(73.5000, 39.5000, 0.0002777777, 0.0002777777)
    with rasterio.open(path, "w", driver="GTiff", height=60, width=60, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr) as ds:
        ds.write(z.astype("float32"), 1)


def test_fall_line_descends_and_goes_downhill():
    from geoindex import fall_line
    from georef import Terrain
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dem.tif"
        _synthetic_dem(p)
        t = Terrain(p)
        path = fall_line(t, 39.4970, 73.5030)
    assert len(path) > 5
    heights = [z for _, _, z in path]
    assert all(b <= a for a, b in zip(heights, heights[1:])), heights[:10]
    # Уклон на восток круче, чем на юг: линия обязана уходить на восток.
    assert path[-1][1] > path[0][1] + 0.002


def test_coverage_skips_poses_without_gimbal_angles():
    """Строка телеметрии без углов подвеса не должна ронять расчёт покрытия.

    В реальных сайдкарах такие строки есть (потеря пакета), и раньше они
    доходили до `pixel_ray` и роняли команду `pokrytie` на пятом видео.
    """
    from geoindex import coverage

    class _Flight:
        width = height = 100
        n_frames = 2

        def __init__(self, fixes):
            self.fixes = fixes

        def focal(self, frame):
            return 5000.0, 0.0, ""

        def sampled(self, step):
            return self.fixes

    blind = Fix(t_s=0.0, lat=LAT0, lon=LON0, alt_m=4100.0, gb_yaw=None, gb_pitch=None)
    cells, stats = coverage([_Flight([blind])], PlaneTerrain(z0=4000.0), 1)
    assert cells == {} and stats["без углов подвеса"] == 1


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001 - сводка по всем тестам сразу
            failed += 1
            print(f"СБОЙ {name}: {type(exc).__name__}: {exc}")
    print(f"\nитог: {len(tests) - failed} из {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
