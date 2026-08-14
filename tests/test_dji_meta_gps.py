"""Тесты разбора dvtm-телеметрии: scripts/dji_meta_gps.py.

Видео не нужны - пакет собирается синтетически по формату protobuf wire, поэтому
тест воспроизводим у любого волонтёра.

Запуск:  python3 -m pytest tests/ -q     (нужен pytest)
"""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dji_meta_gps as dg  # noqa: E402


# --- сборка protobuf --------------------------------------------------------


def varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64                      # отрицательные - дополнительный код
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tag(field: int, wt: int) -> bytes:
    return varint((field << 3) | wt)


def f_varint(field: int, value: int) -> bytes:
    return tag(field, 0) + varint(value)


def f_double(field: int, value: float) -> bytes:
    return tag(field, 1) + struct.pack("<d", value)


def f_bytes(field: int, payload: bytes) -> bytes:
    return tag(field, 2) + varint(len(payload)) + payload


def packet(lat_rad: float, lon_rad: float, alt_mm: int, alt_rel_mm: int,
           roll_dd: int, pitch_dd: int, yaw_dd: int,
           gb_pitch_dd: int, gb_yaw_dd: int, ts_us: int = 1_000_000) -> bytes:
    """Пакет той же формы, что кладёт M30T: см. пути полей в dji_meta_gps."""
    gps = f_bytes(1, f_double(2, lat_rad) + f_double(3, lon_rad)) + f_varint(2, alt_mm)
    attitude = f_varint(1, roll_dd) + f_varint(2, pitch_dd) + f_varint(3, yaw_dd)
    body = (
        f_bytes(3, attitude)
        + f_bytes(4, gps)
        + f_bytes(5, f_varint(1, alt_rel_mm))
    )
    gimbal = f_bytes(3, f_varint(1, gb_pitch_dd) + f_varint(3, gb_yaw_dd))
    return f_bytes(3, f_bytes(1, f_varint(2, ts_us)) + f_bytes(3, body) + f_bytes(4, gimbal))


KURUMDY = dict(lat_rad=math.radians(39.4833965), lon_rad=math.radians(73.5851387),
               alt_mm=4_552_100,
               alt_rel_mm=554_900, roll_dd=-15, pitch_dd=2, yaw_dd=1530,
               gb_pitch_dd=-279, gb_yaw_dd=1439)


# --- тесты ------------------------------------------------------------------


def test_reads_position_altitude_and_angles():
    fix = dg.fix_from(dg.decode_message(packet(**KURUMDY)))
    assert abs(fix["lat"] - 39.4833965) < 1e-9
    assert abs(fix["lon"] - 73.5851387) < 1e-9
    assert fix["alt_m"] == 4552.1
    assert fix["alt_rel_m"] == 554.9
    assert fix["ac_yaw"] == 153.0
    assert fix["gb_pitch"] == -27.9
    assert fix["gb_yaw"] == 143.9


def test_negative_angles_stay_negative():
    """Крен и тангаж подвеса уходят в минус; varint отрицательных - дополнительный код."""
    fix = dg.fix_from(dg.decode_message(packet(**KURUMDY)))
    assert fix["ac_roll"] == -1.5
    assert fix["gb_pitch"] < 0


def test_position_outside_kurumdy_is_not_discarded():
    """Регрессия: разбор по путям полей, а не по диапазону координат.

    Прошлая версия искала GPS перебором с жёсткими рамками
    (широта 0.5-0.8 рад, долгота 1.2-1.35 рад) - за пределами массива она молча
    не находила ничего. Скрипт должен работать на любой съёмке, включая соседние
    районы и учебные полёты.
    """
    alps = dict(KURUMDY, lat_rad=math.radians(46.5), lon_rad=math.radians(8.0))
    fix = dg.fix_from(dg.decode_message(packet(**alps)))
    assert abs(fix["lat"] - 46.5) < 1e-9
    assert abs(fix["lon"] - 8.0) < 1e-9


def test_low_altitude_is_not_dropped():
    """Регрессия: высота ниже 3000 м раньше отбрасывалась и строка уходила пустой."""
    low = dict(KURUMDY, alt_mm=812_000, alt_rel_mm=120_000)
    fix = dg.fix_from(dg.decode_message(packet(**low)))
    assert fix["alt_m"] == 812.0
    assert fix["alt_rel_m"] == 120.0


def test_binary_blob_is_not_parsed_as_a_message():
    """Строки и бинарные блоки не должны разбираться как вложенные сообщения.

    Иначе случайные байты дают «поля» на нужных номерах и подсовывают ложные
    координаты.
    """
    blob = b"\xff\xff\xff\xff\xff\xff"
    assert dg.parses_as_proto(blob) is False
    got = dg.decode_message(f_bytes(7, blob))
    assert got["7"] == blob


def test_truncated_tail_costs_only_the_tail():
    """Обрыв в конце пакета не должен стоить уже разобранных полей."""
    raw = packet(**KURUMDY) + tag(9, 0) + b"\xff"     # varint без завершающего байта
    fix = dg.fix_from(dg.decode_message(raw))
    assert abs(fix["lat"] - 39.4833965) < 1e-9


def test_time_column_keeps_frame_resolution():
    """0.03 с между кадрами: округление до 0.1 склеило бы кадры в сайдкаре."""
    assert dg.fmt("time_s", 0.033333) == "0.03"
    assert dg.fmt("lat", 39.4833965) == "39.4833965"
    assert dg.fmt("alt_m", 4552.14) == "4552.1"
    assert dg.fmt("gb_pitch", None) == ""
