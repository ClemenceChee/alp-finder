#!/usr/bin/env python3
"""Склейка пролёта видео в мозаику склона (слой 5 конвейера).

Кадры сэмплируются, соседние совмещаются по общим точкам (SIFT + гомография
RANSAC), цепочка гомографий укладывает все кадры на общий холст. Когда
совмещение рвётся (туман, небо, резкий рывок) — начинается новый сегмент,
на выходе несколько мозаик. Швы сглаживаются растушёвкой от краёв кадра.

Пределы метода: гомография точна для съёмки издали с зумом (рельеф почти
плоский в кадре); при близких пролётах параллакс даёт заметные швы. Длинная
цепочка накапливает дрейф — мозаика годится для отсмотра «борозды в контексте
склона», не для измерения расстояний.

Использование:
  .venv/bin/python stitch.py ВИДЕО.MP4 --out results/ [--fps 2] [--width 1280]
      [--t0 SEC] [--t1 SEC]

Выход в --out: mosaic_NN.jpg, mosaic_NN_preview.jpg (если мозаика крупная),
frames_NN.tsv (таймкод кадра, позиция центра на холсте, GPS дрона из сайдкара).
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from video_scan import load_gps, gps_at, tc

MIN_INLIERS = 40       # меньше инлайеров RANSAC — совмещение не доверяем
MAX_FAIL = 3           # подряд неудачных сэмплов — сегмент закрыт
SHIFT_FRAC = 0.12      # доля ширины кадра: меньший сдвиг — кадр в мозаику не нужен
SCALE_LIM = (0.25, 4)  # допустимое изменение масштаба всей цепочки
PREVIEW_MAX = 4000     # сторона превью


def sample_frames(video: Path, fps: float, t0: float, t1: float, width: int):
    """(t, кадр уменьшенный до width) по видео с шагом 1/fps."""
    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    n = 0
    while True:
        if not cap.grab():
            break
        t = n / native
        n += 1
        if (n - 1) % step or t < t0:
            continue
        if t1 and t > t1:
            break
        ok, img = cap.retrieve()
        if not ok:
            break
        h = round(img.shape[0] * width / img.shape[1])
        yield t, cv2.resize(img, (width, h))
    cap.release()


def match_homography(sift, matcher, key_gray, key_feat, cur_gray):
    """Гомография текущий→опорный кадр или None. Возвращает (H, feat, inliers)."""
    kp2, des2 = sift.detectAndCompute(cur_gray, None)
    kp1, des1 = key_feat
    if des1 is None or des2 is None or len(kp2) < 8:
        return None, (kp2, des2), 0
    pairs = matcher.knnMatch(des2, des1, k=2)
    good = [m for m, nn in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * nn.distance]
    if len(good) < MIN_INLIERS:
        return None, (kp2, des2), 0
    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inl = int(mask.sum()) if mask is not None else 0
    if H is None or inl < MIN_INLIERS:
        return None, (kp2, des2), inl
    return H, (kp2, des2), inl


def sane(G, w, h):
    """Цепочка не выродилась: масштаб в пределах, углы кадра — выпуклый квад."""
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    if det <= 0:
        return False
    s = math.sqrt(det)
    if not (SCALE_LIM[0] < s < SCALE_LIM[1]):
        return False
    corners = cv2.perspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), G)
    return cv2.isContourConvex(corners.astype(np.float32))


def feather(w, h):
    """Вес пикселя = расстояние до края кадра, нормированное в [0..1]."""
    m = np.zeros((h, w), np.uint8)
    m[1:-1, 1:-1] = 1
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (d / d.max()).astype(np.float32)


def render(segment, out: Path, idx: int, gps, max_canvas: int):
    """Уложить кадры сегмента на холст и записать mosaic/preview/tsv."""
    h, w = segment[0][2].shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = np.concatenate(
        [cv2.perspectiveTransform(corners, G) for _, G, _ in segment])
    x0, y0 = pts.reshape(-1, 2).min(0)
    x1, y1 = pts.reshape(-1, 2).max(0)
    scale = min(1.0, max_canvas / max(x1 - x0, y1 - y0))
    T = np.diag([scale, scale, 1.0]) @ np.array(
        [[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    cw, ch = math.ceil((x1 - x0) * scale), math.ceil((y1 - y0) * scale)

    canvas = np.zeros((ch, cw, 3), np.uint8)
    filled = np.zeros((ch, cw), bool)
    weight = feather(w, h)
    rows = []
    for t, G, img in segment:
        M = (T @ G).astype(np.float64)
        warped = cv2.warpPerspective(img, M, (cw, ch))
        a = cv2.warpPerspective(weight, M, (cw, ch))
        hit = a > 1e-3
        a = np.where(hit & ~filled, 1.0, a)[..., None].astype(np.float32)
        region = hit
        canvas[region] = (canvas[region] * (1 - a[region])
                          + warped[region] * a[region]).astype(np.uint8)
        filled |= hit
        cx, cy = cv2.perspectiveTransform(
            np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2), M).ravel()
        rows.append((t, cx, cy))

    cv2.imwrite(str(out / f"mosaic_{idx:02d}.jpg"), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    if max(cw, ch) > PREVIEW_MAX:
        k = PREVIEW_MAX / max(cw, ch)
        cv2.imwrite(str(out / f"mosaic_{idx:02d}_preview.jpg"),
                    cv2.resize(canvas, (round(cw * k), round(ch * k))),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    with (out / f"frames_{idx:02d}.tsv").open("w", encoding="utf-8") as f:
        f.write("t\tcanvas_x\tcanvas_y\tlat\tlon\talt\n")
        for t, cx, cy in rows:
            lat, lon, alt = gps_at(gps, t)
            f.write(f"{t:.1f}\t{cx:.0f}\t{cy:.0f}\t{lat}\t{lon}\t{alt}\n")
    print(f"  mosaic_{idx:02d}: кадров {len(segment)}, холст {cw}x{ch}, "
          f"{tc(segment[0][0])}–{tc(segment[-1][0])}")


def stitch(video: Path, out: Path, fps: float, width: int,
           t0: float, t1: float, max_canvas: int):
    out.mkdir(parents=True, exist_ok=True)
    gps = load_gps(video)
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    segments = []
    segment = []          # [(t, G_глобальная, кадр)]
    key = None            # (gray, feat, G) опорного кадра
    fails = 0
    n_samples = 0
    for t, img in sample_frames(video, fps, t0, t1, width):
        n_samples += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if key is None:
            feat = sift.detectAndCompute(gray, None)
            G = np.eye(3)
            segment.append((t, G, img))
            key = (gray, feat, G)
            continue
        H, feat, _ = match_homography(sift, matcher, key[0], key[1], gray)
        G = key[2] @ H if H is not None else None
        if G is None or not sane(G, img.shape[1], img.shape[0]):
            fails += 1
            if fails >= MAX_FAIL:
                if len(segment) > 1:
                    segments.append(segment)
                segment = [(t, np.eye(3), img)]
                key = (gray, feat, np.eye(3))
                fails = 0
            continue
        fails = 0
        shift = math.hypot(H[0, 2], H[1, 2])
        if shift < SHIFT_FRAC * width:
            continue
        segment.append((t, G, img))
        key = (gray, feat, G)
    if len(segment) > 1:
        segments.append(segment)

    print(f"{video.name}: сэмплов {n_samples}, сегментов {len(segments)}")
    for i, seg in enumerate(segments):
        render(seg, out, i, gps, max_canvas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--max-canvas", type=int, default=12000)
    args = ap.parse_args()
    stitch(args.video, args.out, args.fps, args.width,
           args.t0, args.t1, args.max_canvas)
