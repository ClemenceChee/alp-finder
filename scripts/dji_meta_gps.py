#!/usr/bin/env python3
"""Извлечение GPS-телеметрии из DJI-видео без SRT (поток djmd, protobuf dvtm_pm320, M30T).

В каждом пакете data-потока лежит protobuf-сообщение; GPS — вложенное сообщение вида
{1: {2: широта(double, радианы), 3: долгота(double, радианы)}, 2: высота(varint, мм)}.
Схемы нет — сообщение ищется рекурсивным обходом по этой сигнатуре.

Использование:
  python3 scripts/dji_meta_gps.py <видео.MP4> [ещё видео...]

Рядом с каждым видео пишет сайдкар <видео>.gps.tsv: время_сек, lat, lon, alt_m.
"""

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

LAT_RAD = (0.5, 0.8)     # ~28.6°–45.8°
LON_RAD = (1.2, 1.35)    # ~68.8°–77.3°
ALT_MM = (3_000_000, 7_000_000)


def packets(video: Path):
    """(pts_time, размер) каждого пакета data-потока 0:1."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "1", "-show_packets",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    for p in json.loads(out)["packets"]:
        yield float(p["pts_time"]), int(p["size"])


def dump_stream(video: Path) -> bytes:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:1", "-c", "copy",
         "-f", "data", "-"],
        capture_output=True, check=True,
    )
    return out.stdout


def parse_fields(buf: bytes):
    """Плоский разбор одного protobuf-сообщения: [(номер поля, тип, значение)]."""
    i, out = 0, []
    n = len(buf)
    while i < n:
        tag = shift = 0
        while i < n:
            b = buf[i]; i += 1
            tag |= (b & 0x7F) << shift; shift += 7
            if not b & 0x80:
                break
        field, wt = tag >> 3, tag & 7
        if wt == 0:
            v = shift = 0
            while i < n:
                b = buf[i]; i += 1
                v |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80:
                    break
            out.append((field, "varint", v))
        elif wt == 1:
            if i + 8 > n:
                break
            out.append((field, "double", struct.unpack_from("<d", buf, i)[0])); i += 8
        elif wt == 2:
            ln = shift = 0
            while i < n:
                b = buf[i]; i += 1
                ln |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80:
                    break
            if i + ln > n:
                break
            out.append((field, "bytes", buf[i:i + ln])); i += ln
        elif wt == 5:
            if i + 4 > n:
                break
            out.append((field, "float", struct.unpack_from("<f", buf, i)[0])); i += 4
        else:
            break
    return out


def find_gps(chunk: bytes, depth: int = 0):
    """Рекурсивный поиск GPS-сообщения по сигнатуре. Возвращает (lat°, lon°, alt_m)."""
    if depth > 8:
        return None
    fields = parse_fields(chunk)
    sub = {f: v for f, t, v in fields if t == "bytes"}
    if 1 in sub:
        inner = {f: v for f, t, v in parse_fields(sub[1]) if t == "double"}
        lat_r, lon_r = inner.get(2), inner.get(3)
        if lat_r and lon_r and LAT_RAD[0] <= lat_r <= LAT_RAD[1] and LON_RAD[0] <= lon_r <= LON_RAD[1]:
            alt = next((v / 1000.0 for f, t, v in fields
                        if f == 2 and t == "varint" and ALT_MM[0] <= v <= ALT_MM[1]), None)
            return math.degrees(lat_r), math.degrees(lon_r), alt
    for _, _, v in fields:
        if isinstance(v, bytes) and len(v) > 10:
            got = find_gps(v, depth + 1)
            if got:
                return got
    return None


def process(video: Path) -> Path:
    data = dump_stream(video)
    rows = []
    pos = 0
    for pts, size in packets(video):
        gps = find_gps(data[pos:pos + size])
        pos += size
        if gps:
            lat, lon, alt = gps
            rows.append((pts, lat, lon, alt))
    out = video.with_suffix(video.suffix + ".gps.tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("time_s\tlat\tlon\talt_m\n")
        for pts, lat, lon, alt in rows:
            f.write(f"{pts:.2f}\t{lat:.7f}\t{lon:.7f}\t{alt if alt is None else round(alt, 1)}\n")
    print(f"{video.name}: {len(rows)} точек -> {out.name}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        process(Path(arg))
