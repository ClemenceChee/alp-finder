#!/usr/bin/env python3
"""Детектор цветовых аномалий: пиксели, не вписывающиеся в палитру «снег+камень» кадра.

Палитра кадра строится как 3D-гистограмма Lab; аномалия — пиксель из редкой
ячейки (доля пикселей ячейки ниже порога). Редкость покрывает и цвет (синее,
красное, бирюзовое), и яркостные выбросы (белый стакан на тёмной осыпи).
Тени/синева снега не срабатывают: в горной сцене их много, ячейки плотные.

Использование:
  .venv/bin/python detect.py КАДР.jpg [ещё кадры...] --out results/
Выход в --out: <кадр>.det.jpg (рамки на кадре), crops/<кадр>_NN.jpg (кропы),
detections.tsv (кадр, bbox, площадь, средний цвет, score).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BINS = (16, 32, 32)          # квантование L, a, b
DENSITY_THR = 1.5e-4         # ячейка реже этой доли пикселей = аномалия
MIN_AREA = 6                 # блобы меньше (пикс. после морфологии) — шум
MAX_AREA = 20000             # больше — кусок сцены (облако, скала), не предмет
CROP = 128                   # полукроп вокруг центра блоба


def anomaly_blobs(img: np.ndarray):
    """[(x, y, w, h, area, score)] аномальных блобов кадра.

    score — суммарная редкость (−log10 плотности ячейки) по пикселям блоба:
    связное пятно редкого цвета набирает много, одиночная крапинка — мало.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    q = lab.astype(np.uint16)
    q[..., 0] = q[..., 0] * BINS[0] // 256
    q[..., 1] = q[..., 1] * BINS[1] // 256
    q[..., 2] = q[..., 2] * BINS[2] // 256
    flat = (q[..., 0] * BINS[1] + q[..., 1]) * BINS[2] + q[..., 2]
    hist = np.bincount(flat.ravel(), minlength=BINS[0] * BINS[1] * BINS[2])
    density = hist.astype(np.float64) / flat.size
    pix_density = density[flat]
    rarity = -np.log10(pix_density + 1e-9)

    mask = (pix_density < DENSITY_THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comp_score = np.bincount(labels.ravel(), weights=(rarity * mask).ravel(),
                             minlength=n)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        out.append((int(x), int(y), int(w), int(h), int(area),
                    round(float(comp_score[i]), 1)))
    out.sort(key=lambda b: -b[5])
    return out


def process(path: Path, out_dir: Path, tsv):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{path}: не читается", file=sys.stderr)
        return 0
    blobs = anomaly_blobs(img)
    vis = img.copy()
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h, area, score) in enumerate(blobs):
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 2)
        cv2.putText(vis, str(i), (x, max(12, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = max(0, cx - CROP), max(0, cy - CROP)
        crop = img[y0:cy + CROP, x0:cx + CROP]
        cv2.imwrite(str(crops_dir / f"{path.stem}_{i:02d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        bgr = img[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0).astype(int)
        tsv.write(f"{path.name}\t{i}\t{x},{y},{w},{h}\t{area}\t"
                  f"{bgr[2]},{bgr[1]},{bgr[0]}\t{score}\n")
    cv2.imwrite(str(out_dir / f"{path.stem}.det.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{path.name}: {len(blobs)} блобов")
    return len(blobs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    with (args.out / "detections.tsv").open("a", encoding="utf-8") as tsv:
        for p in args.images:
            total += process(p, args.out, tsv)
    print(f"итого блобов: {total}")
