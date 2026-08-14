#!/usr/bin/env python3
"""Фокусное расстояние - из самой съёмки, а не из метаданных.

В потоке dvtm нет поля, которое привязывается к фокусному из первых принципов
(см. шапку `scripts/dji_meta_gps.py`). Оно и не нужно: борт зависает, поэтому
движение картинки между кадрами почти целиком задаётся поворотом подвеса. Для
камеры-обскуры при малом повороте

    dx_пикс = f_пикс * d_рыскания      dy_пикс = f_пикс * d_тангажа

Сдвиг измеряется фазовой корреляцией, поворот берётся из телеметрии, фокусное
падает в руки. Паспортных данных не требуется.

Побочно это проверка сопоставления полей углов: если бы `gb_yaw`/`gb_pitch`
были не тем, чем считаются, отношение измеренного сдвига к заявленному повороту
скакало бы от пары к паре, а оценки по горизонтали и по вертикали не сходились
бы между собой.

Что здесь неверно и о чём обязан знать читатель отчёта:

* Камера зумит по ходу полёта. Одно фокусное на весь файл - неправда, поэтому
  оценка локальная: медиана по парам в окне вокруг нужного кадра.
* Пары берутся с базой в десятки кадров, а не соседние. Углы подвеса квантованы
  до 0,1 градуса, и при медленной панораме соседние кадры отличаются ровно на
  одну ступень квантования: ошибка равна сигналу.
* На однородном снегу фазовая корреляция цепляется за позёмку и меряет ветер.
  Отсюда порог по текстуре кадра и отбраковка по оптическому конверту камеры.
* Где пар в окне не набралось - возвращается None, и вызывающий обязан выдать
  азимут вместо координаты.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

# Границы правдоподобия для фокусного в пикселях при ширине кадра 1920.
# Нижняя - широкий конец оптики M30T (поле зрения ~30,6 градуса по диагонали).
# Верхняя НЕ равна узкому концу оптики (14315 пикс): у камеры есть цифровой зум,
# он вырезает кусок матрицы и растягивает до 1920, то есть множит эффективное
# фокусное. ИЗМЕРЕНО на DJI_20260813183727_0015_Z: между кадрами 6104 и 6301
# фокусное выросло с ~3100 до ~15000 пикс, и обрезание по оптическому конверту
# выбрасывало именно те замеры, которые нужны. Верхняя граница поставлена как
# заведомая нелепость (поле зрения 1,8 градуса), а не как паспортный предел.
F_PX_MIN_1920 = 3000.0
F_PX_MAX_1920 = 60000.0

BASELINE = 30          # кадров между парой: выше квантования углов
SCALE = 0.5            # корреляция на половинном размере: вчетверо быстрее
MIN_ROT_DEG = 0.6      # меньше по модулю суммарного поворота - шум квантования
MIN_AXIS_DEG = 0.4     # ось учитывается, только если сама повернулась на столько
MAX_ROT_DEG = 12.0     # больше - приближение малых углов уже врёт
MIN_RESPONSE = 0.06    # отклик фазовой корреляции
MIN_TEXTURE = 6.0      # СКО яркости кадра: ниже - смотреть не на что
# Окно локальной медианы. Короткое сознательно: зум на этой съёмке меняется в
# пять раз за 6,5 секунды, и окно в 450 кадров (как в исходной реализации)
# усредняет по разным зумам, выдавая фокусное, которого не было ни в один момент.
WINDOW = 90
MIN_SAMPLES = 4        # меньше замеров в окне - фокусного нет

# Сходимость осей считается только по парам, где ОБЕ оси повернулись хотя бы на
# столько. Углы квантованы до 0,1 градуса, поэтому замер при повороте 0,4
# градуса несёт до 12,5% собственной ошибки, и медианное расхождение осей около
# 20-25% получается на одном квантовании, ничего не говоря о сопоставлении
# полей. При пороге 1,5 градуса квантование даёт не более 3,3%, и расхождение
# уже означает то, ради чего его считают.
AGREEMENT_MIN_AXIS_DEG = 1.5
MAX_AXIS_DISAGREEMENT = 0.25   # расхождение осей выше - файл не геопривязываем


def _prep(bgr: np.ndarray) -> np.ndarray:
    """Серый float32 с вычтенным сильным размытием.

    На почти однородном снегу сырой сигнал - постоянная составляющая с
    шёпотом текстуры сверху, и корреляции не за что зацепиться. Вычитание
    размытия оставляет только то, что может сдвинуться.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 12 * SCALE)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def measure(video: str | Path, fixes, fps: float, *, width: int = 1920,
            frame_from: int = 0, frame_to: int | None = None,
            baseline: int = BASELINE, scale: float = SCALE):
    """[(кадр, фокусное_пикс, ось, поворот_град)] по парам кадров в диапазоне.

    Поворот сохраняется вместе с замером: по нему потом отбираются пары, на
    которых квантование углов не съедает сходимость осей.

    Диапазон ограничивается сознательно: полный проход по 4-минутному файлу
    стоит минуты, а для одного кандидата нужно окно вокруг его кадра.
    """
    by_frame = {f.frame: f for f in fixes}
    lo = max(0, frame_from - baseline)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается видео: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    envelope = (F_PX_MIN_1920 * width / 1920.0, F_PX_MAX_1920 * width / 1920.0)

    out: list[tuple[int, float, str, float]] = []
    buf: dict[int, np.ndarray] = {}
    tex: dict[int, float] = {}
    while True:
        # Номер кадра спрашивается у декодера, а не считается от точки перемотки:
        # перемотка по H.264 может встать на ближайший опорный кадр, и тогда все
        # пары получили бы чужие углы подвеса. Ошибка была бы тихой.
        i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, img = cap.read()
        if not ok:
            break
        if frame_to is not None and i > frame_to:
            break
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        buf[i] = _prep(small)
        tex[i] = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std())
        j = i - baseline
        buf.pop(j - 1, None)
        tex.pop(j - 1, None)
        if j < frame_from:
            continue
        a, b = by_frame.get(j), by_frame.get(i)
        if not (a and b and a.can_point and b.can_point):
            continue
        if min(tex.get(j, 0.0), tex[i]) < MIN_TEXTURE:
            continue
        d_yaw = _wrap(b.gb_yaw - a.gb_yaw)
        d_pitch = _wrap(b.gb_pitch - a.gb_pitch)
        if not MIN_ROT_DEG <= math.hypot(d_yaw, d_pitch) <= MAX_ROT_DEG:
            continue
        (dx, dy), response = cv2.phaseCorrelate(buf[j], buf[i])
        if response < MIN_RESPONSE:
            continue
        # Рыскание идёт вокруг мировой вертикали, а оптическая ось наклонена на
        # тангаж подвеса, поэтому по кадру заметается угол меньше в cos(тангажа).
        pitch_mid = math.radians((a.gb_pitch + b.gb_pitch) / 2)
        swept = math.radians(abs(d_yaw)) * math.cos(pitch_mid)
        # Каждая ось учитывается, когда повернулась сама, а не когда она
        # преобладает над второй. Горизонтальный сдвиг задаётся рысканием,
        # вертикальный тангажом; взаимное загрязнение - эффект второго порядка
        # (крен, поворот кадра). Пары, где обе оси прошли порог, дают два замера
        # с ОДНИМ номером кадра, и по ним же считается сходимость осей: это
        # единственное честное сравнение, потому что зум для них общий.
        if abs(d_yaw) >= MIN_AXIS_DEG and swept > 1e-3:
            f = abs(dx) / swept / scale
            if envelope[0] <= f <= envelope[1]:
                out.append((i, f, "yaw", abs(d_yaw)))
        if abs(d_pitch) >= MIN_AXIS_DEG:
            f = abs(dy) / math.radians(abs(d_pitch)) / scale
            if envelope[0] <= f <= envelope[1]:
                out.append((i, f, "pitch", abs(d_pitch)))
    cap.release()
    out.sort()
    return out


class FocalSeries:
    """Фокусное по кадрам, для камеры, которая зумит по ходу полёта.

    Одно число на файл здесь неверно и портит каждую производную координату
    молча. Поэтому оценка локальная, а кадры без набора пар рядом возвращают
    None, а не догадку.
    """

    def __init__(self, samples, *, window: int = WINDOW, min_samples: int = MIN_SAMPLES):
        self.window = window
        self.min_samples = min_samples
        self.frames = np.array([s[0] for s in samples], dtype=np.int64)
        self.values = np.array([s[1] for s in samples], dtype=float)
        self.axes = [s[2] for s in samples]
        self.rot = np.array([s[3] for s in samples], dtype=float)
        # Величина глобальная по файлу, а считается перебором всех замеров:
        # без запоминания она пересчитывалась бы на каждый кадр покрытия.
        self._disagreement: float | None = ...

    def __len__(self) -> int:
        return int(self.frames.size)

    def _slice(self, frame: int):
        lo = int(np.searchsorted(self.frames, frame - self.window, side="left"))
        hi = int(np.searchsorted(self.frames, frame + self.window, side="right"))
        return lo, hi

    def at(self, frame: int) -> float | None:
        """Медиана фокусного в окне вокруг кадра, либо None."""
        lo, hi = self._slice(frame)
        if hi - lo < self.min_samples:
            return None
        return float(np.median(self.values[lo:hi]))

    def rel_error(self, frame: int) -> float:
        """Относительный разброс (MAD/медиана) в окне. Идёт в оценку погрешности."""
        lo, hi = self._slice(frame)
        if hi - lo < self.min_samples:
            return 0.0
        v = self.values[lo:hi]
        med = float(np.median(v))
        return float(np.median(np.abs(v - med)) / med) if med else 0.0

    def axis_disagreement(self, frame: int | None = None) -> float | None:
        """Расхождение оценок по рысканию и по тангажу, по ОДНИМ И ТЕМ ЖЕ парам.

        Честная проверка сопоставления полей углов: если бы `gb_yaw`/`gb_pitch`
        были не тем, чем считаются, две независимые оси дали бы разные фокусные.

        Сравнивать медианы «все замеры по рысканию» против «все замеры по
        тангажу» нельзя, и это не мелочь: на DJI_20260813183727_0015_Z такое
        сравнение давало 36% расхождения и файл отвергался целиком, тогда как
        по парам с общим номером кадра расхождение 9%. Разницу давал зум между
        моментами, а не ошибка сопоставления полей.

        `frame` не используется: величина глобальная для файла. Аргумент оставлен
        ради совместимости вызова с `at()`.
        """
        if self._disagreement is not ...:
            return self._disagreement
        by_frame: dict[int, dict[str, list[float]]] = {}
        for k in range(len(self.frames)):
            if self.rot[k] < AGREEMENT_MIN_AXIS_DEG:
                continue
            by_frame.setdefault(int(self.frames[k]), {}).setdefault(self.axes[k], []).append(
                float(self.values[k]))
        ratios = [abs(np.median(d["yaw"]) - np.median(d["pitch"]))
                  / max(np.median(d["yaw"]), np.median(d["pitch"]))
                  for d in by_frame.values() if "yaw" in d and "pitch" in d]
        self._disagreement = float(np.median(ratios)) if len(ratios) >= 5 else None
        return self._disagreement


def cache_path(video: str | Path) -> Path:
    p = Path(video)
    return p.with_suffix(p.suffix + ".focal.tsv")


def save(video: str | Path, samples) -> Path:
    """Замеры рядом с видео: самокалибровка стоит минуты, повторять незачем."""
    path = cache_path(video)
    with path.open("w", encoding="utf-8") as f:
        f.write("frame\tf_px\taxis\trot_deg\n")
        for fr, val, axis, rot in samples:
            f.write(f"{fr}\t{val:.1f}\t{axis}\t{rot:.2f}\n")
    return path


def load(video: str | Path):
    """Замеры из кэша, либо None, если файла нет.

    Пустой файл возвращается пустым списком, а не None: «замеров нет» - это
    результат самокалибровки, а не её отсутствие. Иначе видео с неподвижным
    подвесом пересчитывалось бы при каждом запуске и каждый раз впустую.
    """
    path = cache_path(video)
    if not path.exists():
        return None
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4:
            out.append((int(parts[0]), float(parts[1]), parts[2], float(parts[3])))
    return out
