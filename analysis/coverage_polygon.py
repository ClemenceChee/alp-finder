#!/usr/bin/env python3
"""Покрытие произвольного полигона склона центрами кадров всех роликов с телеметрией.

Метод тот же, что в docs/coverage-gsd.md и в расчёте внешней группы
(docs/nezavisimyy-analiz/01-probel-pokrytiya.md): для каждого сэмпла телеметрии
центральный луч камеры трассируется в DEM; ячейка полигона считается
«смотрели», если хотя бы один центр кадра лёг ближе допуска. Масштаб
(различим ли предмет: порог 8 px из стенда врезок) подтягивается по таймкоду
из готовых analysis/coverage/*.coverage.tsv — он есть только для роликов
11–12.08; попадания роликов без измеренного фокусного считаются
«смотрели, масштаб неизвестен».

Оценка по центрам кадра занижает покрытие (кадр — пятно, а не точка),
поэтому вердикт «не смотрели» — безопасная сторона.

Использование:
  analysis/.venv/bin/python analysis/coverage_polygon.py ПОЛИГОН.json
  где ПОЛИГОН.json = [[lat, lon], ...]
Выход: таблица по ячейкам (--out-cells), сводка в stdout.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geoproject import Dem, cast, ray_dir, load_rows  # noqa: E402

DATA = HERE.parent / "data" / "drive"
COV = HERE / "coverage"

M_PER_DEG_LAT = 111132.0
THRESH_OBJ_CM = 100    # предмет 1 м (рюкзак/человек) при пороге 8 px


def m_per_deg_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


def point_in_poly(lat, lon, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        la1, lo1 = poly[i]
        la2, lo2 = poly[(i + 1) % n]
        if (lo1 > lon) != (lo2 > lon):
            t = (lon - lo1) / (lo2 - lo1)
            if lat < la1 + t * (la2 - la1):
                inside = not inside
    return inside


def gsd_table(video_name):
    """t -> obj8px_cm по готовому расчёту покрытия; None если расчёта нет."""
    path = COV / (video_name + ".coverage.tsv")
    if not path.exists():
        return None
    table = {}
    for line in path.read_text().splitlines()[1:]:
        t, _f, _d, _g, o8, status = line.split("\t")
        table[round(float(t), 1)] = float(o8) if status == "ok" and o8 else None
    return table


def video_hits(video, dem, step, bbox):
    """[(lat, lon, obj8px_cm | nan)] — попадания центра кадра в рельеф внутри bbox."""
    rows = load_rows(video)
    if not rows or "gb_yaw" not in rows[0]:
        return []
    t_arr = np.array([r["time_s"] for r in rows])
    fields = {k: np.array([r.get(k, math.nan) for r in rows])
              for k in ("lat", "lon", "alt_m", "gb_pitch")}
    yaw = np.degrees(np.unwrap(np.radians(
        np.array([r.get("gb_yaw", math.nan) for r in rows]))))
    gsd = gsd_table(video.name)

    hits = []
    t = float(t_arr[0])
    while t <= float(t_arr[-1]):
        la = float(np.interp(t, t_arr, fields["lat"]))
        lo = float(np.interp(t, t_arr, fields["lon"]))
        al = float(np.interp(t, t_arr, fields["alt_m"]))
        yw = float(np.interp(t, t_arr, yaw))
        pt = float(np.interp(t, t_arr, fields["gb_pitch"]))
        # центр кадра: фокусное на направление луча не влияет
        hit = cast(dem, la, lo, al, ray_dir(yw % 360, pt, 960, 540, 1920, 1080, 1000))
        if hit is not None:
            hla, hlo = hit[0], hit[1]
            if bbox[0] <= hla <= bbox[1] and bbox[2] <= hlo <= bbox[3]:
                o8 = math.nan
                if gsd is not None:
                    v = gsd.get(round(t, 1))
                    if v is not None:
                        o8 = v
                hits.append((hla, hlo, o8))
        t += step
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poly", help="JSON-файл [[lat, lon], ...]")
    ap.add_argument("--cell", type=float, default=30.0, help="шаг сетки, м")
    ap.add_argument("--radius", type=float, default=75.0,
                    help="допуск «центр кадра лёг рядом», м")
    ap.add_argument("--step", type=float, default=2.0, help="шаг сэмплов телеметрии, с")
    ap.add_argument("--out-cells", default=None, help="TSV по ячейкам")
    args = ap.parse_args()

    poly = json.load(open(args.poly))
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    lat_c = sum(lats) / len(lats)
    # запас bbox для попаданий: радиус допуска
    pad_la = (args.radius + args.cell) / M_PER_DEG_LAT
    pad_lo = (args.radius + args.cell) / m_per_deg_lon(lat_c)
    bbox = (min(lats) - pad_la, max(lats) + pad_la,
            min(lons) - pad_lo, max(lons) + pad_lo)

    dem = Dem()
    videos = sorted(p.with_suffix("").with_suffix("")  # <имя>.MP4.gps.tsv -> <имя>.MP4
                    for p in DATA.rglob("*.MP4.gps.tsv"))
    all_hits = []
    per_video = []
    for v in videos:
        hits = video_hits(v, dem, args.step, bbox)
        if hits:
            per_video.append((v.name, len(hits),
                              sum(1 for _, _, o8 in hits if not math.isnan(o8))))
        all_hits.extend(hits)

    # сетка ячеек внутри полигона
    d_la = args.cell / M_PER_DEG_LAT
    d_lo = args.cell / m_per_deg_lon(lat_c)
    cells = []
    la = min(lats) + d_la / 2
    while la < max(lats):
        lo = min(lons) + d_lo / 2
        while lo < max(lons):
            if point_in_poly(la, lo, poly):
                cells.append((la, lo))
            lo += d_lo
        la += d_la

    h = np.array([(la, lo) for la, lo, _ in all_hits]) if all_hits else np.zeros((0, 2))
    o8 = np.array([v for _, _, v in all_hits]) if all_hits else np.zeros(0)
    results = []
    for la, lo in cells:
        if len(h):
            dist = np.hypot((h[:, 0] - la) * M_PER_DEG_LAT,
                            (h[:, 1] - lo) * m_per_deg_lon(la))
            near = dist <= args.radius
            looked = bool(near.any())
            seen = bool((near & (o8 <= THRESH_OBJ_CM)).any())
            best_o8 = float(np.nanmin(o8[near])) if looked and not np.all(
                np.isnan(o8[near])) else math.nan
            min_d = float(dist.min())
        else:
            looked = seen = False
            best_o8 = math.nan
            min_d = math.inf
        alt = dem.elev(la, lo)
        results.append(dict(lat=la, lon=lo, alt=alt, looked=looked, seen=seen,
                            best_o8=best_o8, min_d=min_d))

    n = len(results)
    n_seen = sum(r["seen"] for r in results)
    n_lookonly = sum(r["looked"] and not r["seen"] for r in results)
    n_blind = n - n_seen - n_lookonly
    alt_blind = [r["alt"] for r in results if not r["looked"]]
    print(f"ячеек {args.cell:.0f} м внутри полигона: {n}")
    print(f"  осмотрено с масштабом (предмет ≤{THRESH_OBJ_CM} см различим): "
          f"{n_seen} ({n_seen / n:.0%})")
    print(f"  центр кадра ложился, но масштаб неизвестен/хуже: "
          f"{n_lookonly} ({n_lookonly / n:.0%})")
    print(f"  ни один центр кадра не лёг ближе {args.radius:.0f} м: "
          f"{n_blind} ({n_blind / n:.0%})")
    if alt_blind:
        print(f"  неосмотренные ячейки по высоте: {min(alt_blind):.0f}–{max(alt_blind):.0f} м")
    print("\nролики с попаданиями в полигон (всего / с измеренным масштабом):")
    for name, k, k_gsd in sorted(per_video, key=lambda x: -x[1]):
        print(f"  {name}: {k} / {k_gsd}")

    if args.out_cells:
        with open(args.out_cells, "w", encoding="utf-8") as f:
            f.write("lat\tlon\talt_m\tlooked\tseen\tbest_obj8_cm\tmin_dist_m\n")
            for r in results:
                o8_txt = "" if math.isnan(r["best_o8"]) else f"{r['best_o8']:.0f}"
                f.write(f"{r['lat']:.6f}\t{r['lon']:.6f}\t{r['alt']:.0f}\t"
                        f"{int(r['looked'])}\t{int(r['seen'])}\t{o8_txt}\t"
                        f"{r['min_d']:.0f}\n")
        print(f"\nячейки: {args.out_cells}")


if __name__ == "__main__":
    main()
