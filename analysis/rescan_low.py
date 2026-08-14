#!/usr/bin/env python3
"""Перескан плохо отснятых видео зоны интереса 5385–5485 м с ослабленным порогом.

Два видео отсмотрены ненадёжно (см. analysis/review/2026-08-11.md): 202621 —
туман/дымка (низкий контраст), 153236 — снегопад (полосы, мелкие объекты).
Порог детектора поднят с 1.5e-4 до 4e-4 (ячейка палитры «редкая» уже при
большей доле пикселей), сэмплирование 2 к/с, топ-40 блобов на кадр.

detect.py и video_scan.py не редактируются: порог и CLAHE подменяются только
в этом процессе. Для туманного видео перед детекцией применяется CLAHE по
L-каналу Lab in-place — кропы и монтажи тоже получаются контрастированными,
что упрощает отсмотр.

Использование:  cd analysis && .venv/bin/python rescan_low.py
Выход: scans-low/<видео>/ (crops/, tracks.tsv, montage_NN.jpg).
"""

from pathlib import Path

import cv2

import detect
import video_scan

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "drive"
OUT = HERE / "scans-low"
FPS = 2.0
TOP = 40

detect.DENSITY_THR = 4e-4

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_plain_blobs = detect.anomaly_blobs


def _blobs_clahe(img):
    """Контрастирование кадра (CLAHE по L) in-place, затем обычная детекция."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    img[:] = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return _plain_blobs(img)


def run(rel: str, clahe: bool):
    video = DATA / rel
    video_scan.anomaly_blobs = _blobs_clahe if clahe else _plain_blobs
    video_scan.scan(video, OUT / video.stem, FPS, TOP)


if __name__ == "__main__":
    run("2026-08-12/oblet-drona/DJI_20260811202621_0002_Z.MP4", clahe=True)
    run("2026-08-11/DJI_20260811153236_0001_Z.MP4", clahe=False)
