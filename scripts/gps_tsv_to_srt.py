#!/usr/bin/env python3
"""Конвертация извлечённой телеметрии (<видео>.MP4.gps.tsv из dji_meta_gps.py)
в стандартный DJI-SRT — формат, который волонтёры открывают в VLC вместе с видео
и который понимают их инструменты геопривязки.

Использование:
  python3 scripts/gps_tsv_to_srt.py <видео.MP4> [ещё видео...]

Рядом с видео появится <имя>.SRT. Стартовое время берётся из имени файла DJI
(DJI_YYYYMMDDHHMMSS_...) — это локальное время дрона.
"""

import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def convert(video: Path) -> Path:
    tsv = video.with_suffix(video.suffix + ".gps.tsv")
    rows = [(float(r["time_s"]), r["lat"], r["lon"], r["alt_m"])
            for r in csv.DictReader(tsv.open(), delimiter="\t")]

    m = re.search(r"DJI_(\d{14})", video.name)
    start = datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else datetime(2026, 1, 1)

    out = video.with_suffix(".SRT")
    with out.open("w", encoding="utf-8") as f:
        for i, (t, lat, lon, alt) in enumerate(rows):
            t_next = rows[i + 1][0] if i + 1 < len(rows) else t + 1 / 30
            diff_ms = max(1, int(round((t_next - t) * 1000)))
            wall = start + timedelta(seconds=t)
            f.write(f"{i + 1}\n{srt_time(t)} --> {srt_time(t_next)}\n")
            f.write(f'<font size="28">SrtCnt : {i + 1}, DiffTime: {diff_ms}ms\n')
            f.write(wall.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "\n")
            f.write(f"[latitude: {lat}] [longitude: {lon}] [altitude: {alt}] </font>\n\n")
    print(f"{out.name}: {len(rows)} записей")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        convert(Path(arg))
