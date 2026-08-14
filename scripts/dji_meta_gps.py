#!/usr/bin/env python3
"""Извлечение телеметрии из DJI-видео без SRT (поток djmd, protobuf dvtm_pm320, M30T).

В каждом пакете data-потока лежит protobuf-сообщение на кадр. Схема не опубликована,
поэтому сообщение разбирается структурно (по формату wire), а нужные значения берутся
по номерам полей - путь вида ``3.3.4.1.2`` читается как «поле 2 внутри 1 внутри 4
внутри 3 внутри 3».

Пути полей ниже сверены с 12.08 и 13.08 съёмками Курумды (независимый разбор того же
потока, см. tests/test_dji_meta_gps.py).

Достоверность:

* **Точно.** Широта, долгота, высота (MSL и относительная), метка времени. Координаты
  ложатся на массив, относительная высота отличается от MSL на постоянную величину,
  равную правдоподобной высоте точки взлёта, длительность метки совпадает с
  длительностью контейнера.
* **Предположительно.** Углы борта и подвеса. Диапазоны и поведение сходятся (крен и
  тангаж малы, рыскание - полный круг, тангаж подвеса ограничен), но назначение
  «тангаж/крен» независимо не подтверждено. Скрипт печатает диапазоны - сверяйте.
* **Не определено.** Фокусное расстояние/зум. Два поля меняются правдоподобно, но ни
  одно не привязывается к фокусному из первых принципов. Не брать из метаданных;
  восстанавливать фотограмметрически (при зависании движение картинки почти целиком
  задаётся вращением подвеса: оптический поток, делённый на угловую скорость подвеса
  за тот же интервал, даёт фокусное в пикселях без паспортных данных).

Использование:
  python3 scripts/dji_meta_gps.py <видео.MP4> [ещё видео...]

Рядом с каждым видео пишет сайдкар <видео>.gps.tsv со столбцами:
time_s, lat, lon, alt_m (MSL), alt_rel_m, ac_yaw, ac_pitch, ac_roll, gb_yaw, gb_pitch.
Первые четыре столбца и их имена - как раньше, gps_tsv_to_srt.py и video_scan.py
читают файл по заголовку, новые столбцы им не мешают.
"""

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

# Пути полей в dvtm-сообщении, номера через точку.
F_TIMESTAMP_US = "3.1.2"
F_LAT_RAD = "3.3.4.1.2"
F_LON_RAD = "3.3.4.1.3"
F_ALT_MSL_MM = "3.3.4.2"
F_ALT_REL_MM = "3.3.5.1"
F_AC_ROLL_DDEG = "3.3.3.1"
F_AC_PITCH_DDEG = "3.3.3.2"
F_AC_YAW_DDEG = "3.3.3.3"
F_GB_PITCH_DDEG = "3.4.3.1"
F_GB_YAW_DDEG = "3.4.3.3"

# Поля заголовка потока, встречаются один раз в первом пакете.
F_MODEL = "1.1.10"
F_SERIAL = "1.1.5"
F_FIRMWARE = "1.1.6"
F_WIDTH = "2.2.1"
F_HEIGHT = "2.2.2"
F_FPS = "2.2.3"

COLUMNS = ("time_s", "lat", "lon", "alt_m", "alt_rel_m",
           "ac_yaw", "ac_pitch", "ac_roll", "gb_yaw", "gb_pitch")

MAX_VARINT_SHIFT = 63
MAX_DEPTH = 8


# --- поиск потока -----------------------------------------------------------


def meta_stream_index(video: Path) -> int:
    """Индекс data-потока с телеметрией DJI.

    Раньше поток был жёстко прописан как 0:1. Это верно для съёмки М30Т, но ломается
    молча на любом другом раскладе дорожек - поэтому ищем по handler_name.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out)["streams"]
    data = [s for s in streams if s.get("codec_type") == "data"]
    for s in data:
        handler = (s.get("tags") or {}).get("handler_name", "").lower()
        if "dji" in handler or "meta" in handler:
            return int(s["index"])
    if data:
        return int(data[0]["index"])
    raise SystemExit(f"{video.name}: в файле нет data-потока с телеметрией")


def packets(video: Path, idx: int):
    """(pts_time, размер) каждого пакета телеметрийного потока."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", str(idx), "-show_packets",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    for p in json.loads(out)["packets"]:
        yield float(p["pts_time"]), int(p["size"])


def dump_stream(video: Path, idx: int) -> bytes:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{idx}", "-c", "copy",
         "-f", "data", "-"],
        capture_output=True, check=True,
    )
    return out.stdout


# --- разбор protobuf --------------------------------------------------------


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    n = len(buf)
    while i < n:
        b = buf[i]
        val |= (b & 0x7F) << shift
        i += 1
        if not b & 0x80:
            return val, i
        shift += 7
        if shift > MAX_VARINT_SHIFT:
            raise ValueError("varint длиннее 64 бит")
    raise ValueError("varint обрывается")


def parses_as_proto(buf: bytes) -> bool:
    """Является ли вложенный блок корректным сообщением.

    Без этой проверки любой бинарный кусок (строка, картинка) разбирается как
    сообщение и подсовывает случайные «поля» на нужных номерах. Проверка дешёвая:
    обойти все поля и потребовать, чтобы обход закончился ровно на конце буфера.
    """
    i = seen = 0
    n = len(buf)
    while i < n:
        try:
            key, i = read_varint(buf, i)
        except ValueError:
            return False
        fn, wt = key >> 3, key & 7
        if fn == 0 or wt in (3, 4, 6):
            return False
        if wt == 0:
            try:
                _, i = read_varint(buf, i)
            except ValueError:
                return False
        elif wt == 1:
            i += 8
        elif wt == 2:
            try:
                ln, i = read_varint(buf, i)
            except ValueError:
                return False
            i += ln
        elif wt == 5:
            i += 4
        if i > n:
            return False
        seen += 1
    return seen > 0


def decode_message(buf: bytes, path: str = "", depth: int = 0) -> dict:
    """Сообщение → {путь по номерам полей: значение}.

    Повторяющиеся поля сохраняют первое вхождение; ни одно из читаемых здесь полей
    в потоке не повторяется.
    """
    out: dict = {}
    i = 0
    n = len(buf)
    while i < n:
        try:
            key, i = read_varint(buf, i)
        except ValueError:
            break
        fn, wt = key >> 3, key & 7
        p = f"{path}.{fn}" if path else str(fn)
        if wt == 0:
            # Обрезанный пакет должен стоить нам остатка пакета, а не всей телеметрии
            # кадра: молча потерянный кадр - это кадр, про который никто не знает,
            # что его не осмотрели.
            try:
                v, i = read_varint(buf, i)
            except ValueError:
                break
            if v >= 1 << 63:      # отрицательные varint - дополнительный код
                v -= 1 << 64
            out.setdefault(p, v)
        elif wt == 1:
            raw = buf[i:i + 8]
            i += 8
            if len(raw) == 8:
                out.setdefault(p, struct.unpack("<d", raw)[0])
        elif wt == 2:
            try:
                ln, i = read_varint(buf, i)
            except ValueError:
                break
            sub = buf[i:i + ln]
            i += ln
            if depth < MAX_DEPTH and len(sub) > 1 and parses_as_proto(sub):
                for k, v in decode_message(sub, p, depth + 1).items():
                    out.setdefault(k, v)
            else:
                out.setdefault(p, sub)
        elif wt == 5:
            raw = buf[i:i + 4]
            i += 4
            if len(raw) == 4:
                out.setdefault(p, struct.unpack("<f", raw)[0])
        else:
            break
    return out


# --- значения ---------------------------------------------------------------


def num(v):
    return float(v) if isinstance(v, (int, float)) else None


def ddeg(v):
    """Децидеградусы → градусы."""
    f = num(v)
    return None if f is None else f / 10.0


def mm(v):
    f = num(v)
    return None if f is None else f / 1000.0


def deg(v):
    """Радианы → градусы."""
    f = num(v)
    return None if f is None else math.degrees(f)


def text(v):
    if isinstance(v, bytes):
        return v.decode("ascii", "replace").strip("\x00").strip() or None
    return None if v is None else str(v)


def fix_from(d: dict) -> dict:
    return {
        "lat": deg(d.get(F_LAT_RAD)),
        "lon": deg(d.get(F_LON_RAD)),
        "alt_m": mm(d.get(F_ALT_MSL_MM)),
        "alt_rel_m": mm(d.get(F_ALT_REL_MM)),
        "ac_yaw": ddeg(d.get(F_AC_YAW_DDEG)),
        "ac_pitch": ddeg(d.get(F_AC_PITCH_DDEG)),
        "ac_roll": ddeg(d.get(F_AC_ROLL_DDEG)),
        "gb_yaw": ddeg(d.get(F_GB_YAW_DDEG)),
        "gb_pitch": ddeg(d.get(F_GB_PITCH_DDEG)),
    }


def fmt(name: str, v) -> str:
    if v is None:
        return ""
    if name in ("lat", "lon"):
        return f"{v:.7f}"
    if name == "time_s":
        return f"{v:.2f}"      # кадр 30 к/с - 0.03 с, округление до 0.1 склеило бы кадры
    return f"{v:.1f}"


def span(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (min(vals), max(vals)) if vals else None


# --- обработка --------------------------------------------------------------


def process(video: Path) -> Path:
    idx = meta_stream_index(video)
    data = dump_stream(video, idx)
    rows = []
    header: dict = {}
    pos = n_packets = 0
    for pts, size in packets(video, idx):
        d = decode_message(data[pos:pos + size])
        pos += size
        n_packets += 1
        if not header:
            header = {k: v for k, v in d.items() if k.startswith(("1.", "2."))}
        row = fix_from(d)
        if row["lat"] is None or row["lon"] is None:
            continue
        row["time_s"] = pts
        rows.append(row)

    out = video.with_suffix(video.suffix + ".gps.tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(fmt(c, r.get(c)) for c in COLUMNS) + "\n")

    model = text(header.get(F_MODEL)) or "?"
    fw = text(header.get(F_FIRMWARE)) or "?"
    w, h = header.get(F_WIDTH), header.get(F_HEIGHT)
    print(f"{video.name}: поток 0:{idx}, {model} fw {fw}, {w}x{h}")
    print(f"  пакетов {n_packets}, с координатами {len(rows)} -> {out.name}")
    if not rows:
        print("  ВНИМАНИЕ: координат нет ни в одном пакете - разбор не сошёлся, "
              "телеметрию использовать нельзя", file=sys.stderr)
        return out
    if len(rows) < n_packets:
        print(f"  без координат: {n_packets - len(rows)} пакетов "
              "(заголовок и потеря спутников - норма; проверьте, если их много)")
    for key, label in (("alt_m", "высота MSL"), ("gb_pitch", "тангаж подвеса"),
                       ("ac_yaw", "рыскание борта")):
        s = span(rows, key)
        if s:
            print(f"  {label}: {s[0]:.1f} .. {s[1]:.1f}")
    print("  углы - предположительное сопоставление полей, сверьте диапазоны выше")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        process(Path(arg))
