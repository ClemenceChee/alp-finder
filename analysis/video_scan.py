#!/usr/bin/env python3
"""Сканер видео: цветовые аномалии → треки повторов → кропы и монтажи для отсмотра.

Кадры сэмплируются (по умолчанию 1 к/с), в каждом берутся топ-блобы детектора
(analysis/detect.py). Блоб, повторяющийся на близкой позиции в соседних сэмплах,
склеивается в один трек — дрон подолгу смотрит на одно место, без склейки каждый
камень размножился бы в десятки кропов. На трек сохраняется кроп с лучшим score.

Использование:
  .venv/bin/python video_scan.py ВИДЕО.MP4 --out results/ [--fps 1] [--top 25]

Выход в --out: crops/tNNNN_MMmSSs.jpg, tracks.tsv (трек, таймкоды, bbox, score,
lat/lon из сайдкара <видео>.gps.tsv, если он есть), montage_NN.jpg (сетки кропов).
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from detect import anomaly_blobs, CROP

MATCH_DIST = 60      # пикс.: ближе — считаем тем же треком
TRACK_TTL = 3        # сэмплов без совпадения — трек закрыт
TILE = 176           # сторона ячейки монтажа
GRID = 6             # монтаж GRID×GRID кропов
FLOW_W = 480         # ширина уменьшенного кадра для оценки сдвига сцены


def frame_shift(prev_small, cur_small, scale):
    """Глобальный сдвиг сцены между сэмплами (dx, dy) в пикселях полного кадра."""
    (dx, dy), _ = cv2.phaseCorrelate(prev_small, cur_small)
    return dx * scale, dy * scale


def load_gps(video: Path):
    """[(t, lat, lon, alt)] из сайдкара .gps.tsv, если есть.

    Читаем по заголовку, а не по позиции: dji_meta_gps.py пишет ещё и углы борта
    и подвеса, и число столбцов будет расти.
    """
    side = video.with_suffix(video.suffix + ".gps.tsv")
    if not side.exists():
        return []
    with side.open(encoding="utf-8") as f:
        return [(float(r["time_s"]), r["lat"], r["lon"], r["alt_m"])
                for r in csv.DictReader(f, delimiter="\t")]


def gps_at(rows, t):
    if not rows:
        return "", "", ""
    i = min(range(len(rows)), key=lambda k: abs(rows[k][0] - t))
    return rows[i][1], rows[i][2], rows[i][3]


def tc(t):
    return f"{int(t) // 60:02d}m{int(t) % 60:02d}s"


def scan(video: Path, out: Path, fps: float, top: int):
    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    gps = load_gps(video)

    tracks = []          # активные: dict(cx, cy, score, crop, t0, t1, bbox, misses)
    done = []
    prev_small = None
    n_frame = n_sample = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if n_frame % step:
            n_frame += 1
            continue
        ok, img = cap.retrieve()
        n_frame += 1
        if not ok:
            break
        t = (n_frame - 1) / native
        n_sample += 1
        scale = img.shape[1] / FLOW_W
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                           (FLOW_W, int(img.shape[0] / scale))).astype(np.float32)
        if prev_small is not None and small.shape == prev_small.shape:
            dx, dy = frame_shift(prev_small, small, scale)
            for tr in tracks:
                tr["cx"] += dx
                tr["cy"] += dy
        prev_small = small
        blobs = anomaly_blobs(img)[:top]

        for tr in tracks:
            tr["misses"] += 1
        for x, y, w, h, area, score in blobs:
            cx, cy = x + w // 2, y + h // 2
            best = None
            for tr in tracks:
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if d < MATCH_DIST and (best is None or d < best[0]):
                    best = (d, tr)
            if best:
                tr = best[1]
                tr.update(cx=cx, cy=cy, misses=0, t1=t)
                if score > tr["score"]:
                    x0, y0 = max(0, cx - CROP), max(0, cy - CROP)
                    tr.update(score=score, bbox=(x, y, w, h), t_best=t,
                              crop=img[y0:cy + CROP, x0:cx + CROP].copy())
            else:
                x0, y0 = max(0, cx - CROP), max(0, cy - CROP)
                tracks.append(dict(cx=cx, cy=cy, score=score, bbox=(x, y, w, h),
                                   t0=t, t1=t, t_best=t, misses=0,
                                   crop=img[y0:cy + CROP, x0:cx + CROP].copy()))
        still = []
        for tr in tracks:
            (done if tr["misses"] > TRACK_TTL else still).append(tr)
        tracks = still
    done += tracks
    cap.release()
    done.sort(key=lambda tr: -tr["score"])

    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    with (out / "tracks.tsv").open("w", encoding="utf-8") as f:
        f.write("track\tt_best\tt0\tt1\tbbox\tscore\tlat\tlon\talt\n")
        for i, tr in enumerate(done):
            lat, lon, alt = gps_at(gps, tr["t_best"])
            x, y, w, h = tr["bbox"]
            f.write(f"{i}\t{tr['t_best']:.1f}\t{tr['t0']:.1f}\t{tr['t1']:.1f}\t"
                    f"{x},{y},{w},{h}\t{tr['score']}\t{lat}\t{lon}\t{alt}\n")
            cv2.imwrite(str(crops_dir / f"t{i:04d}_{tc(tr['t_best'])}.jpg"),
                        tr["crop"], [cv2.IMWRITE_JPEG_QUALITY, 92])

    per = GRID * GRID
    for m in range(0, len(done), per):
        sheet = np.zeros((TILE * GRID, TILE * GRID, 3), np.uint8)
        for j, tr in enumerate(done[m:m + per]):
            cell = cv2.resize(tr["crop"], (TILE, TILE))
            cv2.putText(cell, f"t{m + j} {tc(tr['t_best'])}", (4, TILE - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            r, c = divmod(j, GRID)
            sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = cell
        cv2.imwrite(str(out / f"montage_{m // per:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"{video.name}: сэмплов {n_sample}, треков {len(done)}, "
          f"монтажей {math.ceil(len(done) / per)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    scan(args.video, args.out, args.fps, args.top)
