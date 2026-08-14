#!/usr/bin/env python3
"""Реальное покрытие дальних облётов 11–12.08: GSD по центру кадра против порога 8 px.

Стенд врезок (docs/video-analysis.md, «Порог различимости») дал правило: цветной
предмет уверенно доходит до отсмотра от 8 px кадра 1080p. Этот скрипт переводит
порог в сантиметры для каждого момента каждого ролика 11–12.08: фокусное —
самокалибровкой по панорамированию (docs/focal-length-calibration.md), дальность —
трассировкой центрального луча в рельеф (analysis/geoproject.py). Итог по ролику —
какую долю времени детектор реально «видел» предмет размера рюкзака / куртки /
крышки, и какую долю времени фокусное неизвестно (зависаний без панорам нет).

Считается только центр кадра — как в расчёте покрытия внешней группы
(docs/nezavisimyy-analiz/01-probel-pokrytiya.md): оценка занижает покрытие краёв
кадра, но для вердикта «осмотрено/не осмотрено» это безопасная сторона.

Использование:  cd analysis && .venv/bin/python coverage_gsd.py [--step 2]
Выход: coverage/<видео>.tsv (по строке на сэмпл), coverage/summary.tsv, таблица в stdout.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from geoproject import Dem, cast, ray_dir, load_rows, grab, gray_center

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "drive"
OUT = HERE / "coverage"

PAIR_DT = 0.5        # база пары кадров для фокусного, с
PAIR_EVERY = 2.0     # минимальный шаг между парами, с
ROT_MIN, ROT_MAX = 0.4, 8.0   # суммарный поворот подвеса в паре, градусы
F_BAND = (1000.0, 20000.0)    # оценки фокусного вне этой полосы - брак
F_WINDOW = 20.0      # окно поиска оценок фокусного вокруг сэмпла, с
F_MIN_ESTS = 3       # минимум оценок в окне, иначе фокусное «неизвестно»
THRESH_PX = 8        # порог различимости из стенда врезок
OBJECTS = (("ryukzak", 1.00), ("kurtka", 0.60), ("kryshka", 0.17))


def tracks(rows):
    """Массивы времени и полей телеметрии с развёрнутым yaw."""
    t = np.array([r["time_s"] for r in rows])
    out = {"t": t}
    for k in ("lat", "lon", "alt_m", "gb_pitch"):
        out[k] = np.array([r.get(k, math.nan) for r in rows])
    yaw = np.array([r.get("gb_yaw", math.nan) for r in rows])
    out["gb_yaw"] = np.degrees(np.unwrap(np.radians(yaw)))
    return out


def interp(tr, t, key):
    return float(np.interp(t, tr["t"], tr[key]))


def pan_pairs(tr):
    """[(t0, t1)] — пары кадров, где подвес повернулся на ROT_MIN..ROT_MAX градусов."""
    pairs = []
    t = float(tr["t"][0])
    end = float(tr["t"][-1])
    while t + PAIR_DT <= end:
        dyaw = interp(tr, t + PAIR_DT, "gb_yaw") - interp(tr, t, "gb_yaw")
        dpitch = interp(tr, t + PAIR_DT, "gb_pitch") - interp(tr, t, "gb_pitch")
        rot = math.hypot(dyaw, dpitch)
        dominant = abs(dyaw) > 2 * abs(dpitch) or abs(dpitch) > 2 * abs(dyaw)
        if ROT_MIN <= rot <= ROT_MAX and dominant:
            pairs.append((t, t + PAIR_DT))
            t += PAIR_EVERY
        else:
            t += PAIR_DT
    return pairs


def focal_series(video: Path, tr):
    """[(t_середины_пары, f_px)] по всем участкам панорамирования ролика."""
    cap = cv2.VideoCapture(str(video))
    ests = []
    for a, b in pan_pairs(tr):
        dyaw = math.radians(interp(tr, b, "gb_yaw") - interp(tr, a, "gb_yaw"))
        dpitch = math.radians(interp(tr, b, "gb_pitch") - interp(tr, a, "gb_pitch"))
        pitch = math.radians(interp(tr, (a + b) / 2, "gb_pitch"))
        try:
            ga, sa = gray_center(grab(cap, a))
            gb, _ = gray_center(grab(cap, b))
        except ValueError:
            continue
        (dx, dy), _resp = cv2.phaseCorrelate(ga, gb)
        dx, dy = dx * sa, dy * sa
        if abs(dyaw) > 2 * abs(dpitch):
            f = (-dx) / (dyaw * math.cos(pitch))
        elif abs(dpitch) > 2 * abs(dyaw):
            f = dy / dpitch
        else:
            continue
        if F_BAND[0] <= f <= F_BAND[1]:
            ests.append(((a + b) / 2, f))
    cap.release()
    return ests


def focal_at(ests, t):
    """Медиана оценок фокусного в окне ±F_WINDOW вокруг t или None."""
    near = [f for tf, f in ests if abs(tf - t) <= F_WINDOW]
    if len(near) < F_MIN_ESTS:
        return None
    return float(np.median(near))


def scan_video(video: Path, dem: Dem, step: float):
    rows = load_rows(video)
    if not rows or "gb_yaw" not in rows[0]:
        return None
    tr = tracks(rows)
    ests = focal_series(video, tr)
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    samples = []
    t = float(tr["t"][0])
    while t <= float(tr["t"][-1]):
        f = focal_at(ests, t)
        row = dict(t=t, f=f, dist=None, gsd=None, sky=False)
        if f is not None:
            yaw, pitch = interp(tr, t, "gb_yaw"), interp(tr, t, "gb_pitch")
            hit = cast(dem, interp(tr, t, "lat"), interp(tr, t, "lon"),
                       interp(tr, t, "alt_m"),
                       ray_dir(yaw % 360, pitch, w / 2, h / 2, w, h, f))
            if hit is None:
                row["sky"] = True
            else:
                row["dist"] = hit[3]
                row["gsd"] = hit[3] / f          # м/пикс в центре кадра
        samples.append(row)
        t += step
    return dict(video=video, n_ests=len(ests), samples=samples, width=w)


def write_video_tsv(res):
    out = OUT / (res["video"].name + ".coverage.tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("t\tf_px\tdist_m\tgsd_cm\tobj8px_cm\tstatus\n")
        for s in res["samples"]:
            if s["f"] is None:
                st, fpx, d, g = "no_focal", "", "", ""
            elif s["sky"]:
                st, fpx, d, g = "above_horizon", f"{s['f']:.0f}", "", ""
            else:
                st = "ok"
                fpx, d = f"{s['f']:.0f}", f"{s['dist']:.0f}"
                g = f"{s['gsd'] * 100:.1f}"
            o8 = f"{s['gsd'] * 100 * THRESH_PX:.0f}" if s["gsd"] else ""
            f.write(f"{s['t']:.1f}\t{fpx}\t{d}\t{g}\t{o8}\t{st}\n")


def summarize(res):
    ss = res["samples"]
    n = len(ss)
    known = [s for s in ss if s["gsd"] is not None]
    nofocal = sum(1 for s in ss if s["f"] is None)
    sky = sum(1 for s in ss if s["sky"])
    row = dict(video=res["video"].name, n=n, n_ests=res["n_ests"],
               no_focal=nofocal / n, sky=sky / n, width=res["width"])
    if known:
        gsds = np.array([s["gsd"] for s in known])
        row["gsd_med_cm"] = float(np.median(gsds)) * 100
        row["gsd_p90_cm"] = float(np.percentile(gsds, 90)) * 100
        row["obj8_med_cm"] = row["gsd_med_cm"] * THRESH_PX
        for name, size_m in OBJECTS:
            # покрыто = предмет size_m занимает >= THRESH_PX пикселей
            row[name] = float((gsds <= size_m / THRESH_PX).mean()) * len(known) / n
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=2.0, help="шаг сэмплов, с")
    ap.add_argument("--dates", default="20260811,20260812")
    args = ap.parse_args()
    dates = tuple(args.dates.split(","))
    videos = sorted({p.name: p for p in DATA.rglob("DJI_*.MP4")
                     if p.name[4:12] in dates}.values(), key=lambda p: p.name)
    OUT.mkdir(exist_ok=True)
    dem = Dem()
    summary = []
    for v in videos:
        res = scan_video(v, dem, args.step)
        if res is None:
            print(f"{v.name}: нет углов в сайдкаре, пропуск")
            continue
        write_video_tsv(res)
        row = summarize(res)
        summary.append(row)
        cov = " ".join(f"{name} {row.get(name, 0):.0%}" for name, _ in OBJECTS)
        med = f"{row['obj8_med_cm']:.0f} см" if "obj8_med_cm" in row else "—"
        print(f"{row['video']}: оценок f {row['n_ests']}, без фокусного "
              f"{row['no_focal']:.0%}, выше горизонта {row['sky']:.0%}, "
              f"мин. предмет (8 px, медиана) {med}; покрытие: {cov}")

    keys = ["video", "width", "n", "n_ests", "no_focal", "sky",
            "gsd_med_cm", "gsd_p90_cm", "obj8_med_cm"] + [n for n, _ in OBJECTS]
    with (OUT / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in summary:
            f.write("\t".join("" if r.get(k) is None else
                              (f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r[k]))
                              for k in keys) + "\n")
    print(f"\nсводка: {OUT / 'summary.tsv'}")


if __name__ == "__main__":
    main()
