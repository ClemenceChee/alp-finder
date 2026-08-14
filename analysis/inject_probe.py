#!/usr/bin/env python3
"""Порог различимости детектора: синтетические врезки известного размера и цвета.

Пилот (docs/video-analysis.md) дал recall 100% на эталонах, снятых с близкого
расстояния. На дальних планах предмет занимает единицы пикселей, и recall там не
подтверждён ничем. Здесь он меряется: в реальные кадры операции вклеиваются
предметы известного размера, известного цветового контраста и известного
положения, кадр без изменений прогоняется через analysis/detect.py, и считается
доля врезок, которые детектор вернул. Пороги detect.py не трогаются: скрипт
измеряет детектор, а не подгоняет его.

Что делает результат честным:

* Врезка не резче съёмки. Локальная размытость (ширина края) и локальный шум
  (оценка Иммеркера) меряются в окне вокруг точки врезки, объект сворачивается с
  этой же размытостью и получает своё зерно. Без этих двух шагов измеряется не
  детектор, а скорость, с которой он находит нарисованный прямоугольник.
* Контраст задан по тому, что оказалось в кадре, а не по тому, что рисовали.
  Амплитуда нормируется ПОСЛЕ свёртки, поэтому dE - это отклонение от фона,
  реально присутствующее в пикселях (у объекта в 2-3 px свёртка съедает
  две трети амплитуды, и нормировка до неё завысила бы контраст втрое).
* Рабочая точка - бюджет ложных срабатываний, а не порог. В отсмотр уходит
  топ-25 блобов с кадра (video_scan.py --top), поэтому врезка считается
  найденной, только если она обошла по score шум кадра и попала в этот топ.
  Ранг найденной врезки пишется в TSV: он и есть цена в ложных срабатываниях.
* Одна врезка на прогон детектора. Сетка врезок в один кадр (так дешевле) для
  этого детектора недопустима: он ищет редкие ячейки Lab-гистограммы, и три
  десятка одинаковых предметов сделали бы свою ячейку частой, то есть подавили
  бы сами себя. Артефакт стенда выглядел бы как слепота детектора.

Известный оптимизм, прямо:

* Врезка идёт в уже раскодированный кадр, то есть не платит за сжатие, за
  которое заплатил настоящий предмет. Флаг --jpeg прогоняет кадр (и врезанный,
  и чистый) через JPEG-цикл и меряет эту разницу. Замеренная разница знак имеет
  разный: на мелких ярких предметах сжатие recall повышает (размазывает пятно
  на соседние блоки и тем добавляет ему площади, то есть score), на слабом
  контрасте понижает. H.264 на битрейтах дрона жёстче и здесь не проверялся.
* Оценщик размытости при sigma > 1.5 занижает её примерно на 10-18% (проверено
  на синтетических ступеньках), значит врезка чуть резче реальности.
* Форма - неровное пятно сплошного цвета. Настоящий предмет бывает пятнистым и
  частично закрыт снегом или камнем, такой найти труднее.
* Кадры взяты те, что есть в репозитории (съёмка 13.08), доли снега и осыпи в
  выборке - доли в этих кадрах, а не на маршруте.

Использование:
  .venv/bin/python inject_probe.py --out floor-01/ [--repeats 30] [--jpeg 92]
Выход в --out: trials.tsv (по строке на врезку), report.txt (таблицы recall),
preview/*.jpg (примеры врезок для глазной проверки, если --preview).
"""

import argparse
import hashlib
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import anomaly_blobs  # noqa: E402  (detect.py лежит рядом)

HERE = Path(__file__).resolve().parent
FRAME_DIRS = (HERE / "pilot/gt", HERE.parent / "docs/nakhodki/frames")

SIZES = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48)  # видимый размер предмета, пикс.
CONTRASTS = (10.0, 20.0, 40.0, 80.0)  # пиковое отличие от фона, dE76 (CIE Lab)

# Цвета реальных находок (docs/nakhodki/README.md), RGB. Направление в Lab
# берётся от локального фона к этому цвету, длина задаётся контрастом: так
# оттенок остаётся правдоподобным, а величина - управляемой.
PALETTE = {
    "sinij": (30, 60, 140),      # синий рюкзак
    "biruza": (30, 170, 170),    # бирюзовая крышка
    "oranzh": (230, 110, 30),    # оранжевый предмет
    "belyj": (240, 240, 235),    # белый стакан
}

# Нейтральный белый детектор пропускает по устройству (в кадре со снегом светлые
# нейтральные ячейки Lab плотные), поэтому таблицы печатаются двумя срезами:
# по всей палитре и по цветным предметам отдельно, иначе одна слепая зона
# размазывается по всей кривой.
GROUPS = (("все цвета", None), ("цветные", ("sinij", "biruza", "oranzh")))

TOP_OP = 25        # рабочая точка: столько блобов с кадра уходит в отсмотр
TOP_TIGHT = 10     # ужатый бюджет отсмотра, для второй колонки отчёта
WIN = 48           # полуокно локальных измерений (размытость, шум, класс фона)
SITE_STEP = 64     # шаг сетки кандидатов на врезку
MARGIN = 64        # отступ от края кадра
SS = 4             # суперсэмплинг при отрисовке формы
EDGE_FLOOR = 0.798  # что оценщик края читает на идеально резкой ступеньке
BLUR_RANGE = (0.5, 2.5)  # вне этого диапазона оценка бессмысленна
BLUR_DEFAULT = 1.0       # медиана по кадрам репозитория
MIN_DIR = 5.0      # dE ближе этого - цвет неотличим от фона, ячейка пропускается

IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# --- измерение сцены --------------------------------------------------------


def noise_sigma(gray: np.ndarray) -> float:
    """Оценка шума по Иммеркеру, DN.

    Маска гасит линейные градиенты, поэтому гладкий снежный склон читается как
    ноль и остаётся только зерно. На текстуре оценка завышена, и это безопасная
    сторона: врезка на осыпи получит зерно осыпи, а не стерильную гладкость.
    """
    lap = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, IMMERKAER)
    return float(np.mean(np.abs(lap)) * math.sqrt(math.pi / 2) / 6.0)


def blur_sigma(gray: np.ndarray) -> float:
    """Sigma функции рассеяния точки, пикс., по ширине края.

    Ширина края = (перепад яркости в окне 7x7) / (пиковый градиент); для
    гауссовой ступеньки это sigma*sqrt(2*pi). Оценщик прогонялся по
    синтетическим ступенькам известного размытия: 0.0 -> 0.80, 0.5 -> 0.89,
    1.0 -> 1.24, 2.0 -> 1.82, то есть читает sqrt(sigma^2 + 0.80^2) с занижением
    на дальнем конце. Отсюда вычитание EDGE_FLOOR и зажим диапазона.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    k = np.ones((7, 7), np.uint8)
    step = cv2.dilate(g, k) - cv2.erode(g, k)
    thr = max(float(np.percentile(mag, 99)), 2.0)
    sel = mag >= thr
    if int(sel.sum()) < 30:
        return BLUR_DEFAULT   # краёв нет: ровный снег, мерить нечего
    w = float(np.median(step[sel] / mag[sel])) / math.sqrt(2 * math.pi)
    sigma = math.sqrt(max(w * w - EDGE_FLOOR ** 2, 0.0))
    return min(max(sigma, BLUR_RANGE[0]), BLUR_RANGE[1])


def site_map(img: np.ndarray):
    """{'sneg': [(x, y)...], 'osyp': [...]} - однородные точки врезки.

    Фон решает ответ: на снегу редка любая цветная точка, на осыпи кадр и без
    предмета полон редких ячеек. Поэтому классы меряются раздельно, а точка
    берётся только там, где окно WIN однородно, иначе замер был бы про границу
    снег/камень, а не про фон.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = lab[..., 0].astype(np.int16)
    h, w = L.shape
    out = {"sneg": [], "osyp": []}
    for y in range(MARGIN, h - MARGIN, SITE_STEP):
        for x in range(MARGIN, w - MARGIN, SITE_STEP):
            patch = L[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1]
            med = float(np.median(patch))
            if med >= 200 and float((patch >= 170).mean()) >= 0.9:
                out["sneg"].append((x, y))
            elif med <= 130 and float((patch <= 170).mean()) >= 0.9:
                out["osyp"].append((x, y))
    return out


# --- врезка -----------------------------------------------------------------


def render_blob(size_px: float, blur: float, rng) -> np.ndarray:
    """Маска предмета с пиком 1.0: неровное пятно, свёрнутое с размытием кадра.

    Радиус гуляет двумя низкими гармониками (около 15%): идеальный круг - это
    то, чего в кадре не бывает. Поле берётся с запасом в три sigma, обрезанный
    хвост гаусса - это ступенька, а ступенька для детектора подарок.
    """
    half = int(math.ceil(size_px / 2 + 3 * blur + 2))
    n = 2 * half + 1
    big = np.zeros((n * SS, n * SS), np.float32)
    c = (n * SS - 1) / 2.0
    r = size_px * SS / 2.0
    th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    ph = rng.uniform(0, 2 * math.pi, 2)
    wob = 1.0 + 0.15 * (np.sin(2 * th + ph[0]) + 0.6 * np.sin(3 * th + ph[1]))
    pts = np.stack([c + r * wob * np.cos(th), c + r * wob * np.sin(th)], 1)
    cv2.fillPoly(big, [np.round(pts).astype(np.int32)], 1.0)
    a = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), blur)
    peak = float(a.max())
    return a / peak if peak > 0 else a


def inject(img, gray, x, y, size_px, contrast_de, colour, rng):
    """Вклеить один предмет в (x, y). Возвращает (кадр, факты о врезке) или None.

    None - когда выбранный цвет в этой точке неотличим от фона (белое на снегу):
    тянуть его до заданного dE пришлось бы в случайную сторону, и ячейка мерила
    бы не то, чем подписана.
    """
    blur = blur_sigma(gray[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    noise = noise_sigma(gray[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    alpha = render_blob(size_px, blur, rng)
    half = alpha.shape[0] // 2
    sl = (slice(y - half, y + half + 1), slice(x - half, x + half + 1))

    lab0 = cv2.cvtColor(img[sl], cv2.COLOR_BGR2Lab).astype(np.float32)
    bg = lab0.reshape(-1, 3).mean(axis=0)
    r, g, b = PALETTE[colour]
    obj = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    d = obj - bg
    # Длина направления в единицах CIE: канал L в OpenCV растянут на 0..255.
    dir_de = math.sqrt((d[0] * 100 / 255) ** 2 + d[1] ** 2 + d[2] ** 2)
    if dir_de < MIN_DIR:
        return None
    delta = d * (contrast_de / dir_de)

    # Зерно предмета: тот же шум, что у фона, размытый той же оптикой. Без него
    # врезка - единственное идеально гладкое место в кадре.
    grain = cv2.GaussianBlur(rng.standard_normal(alpha.shape).astype(np.float32),
                             (0, 0), blur)
    sd = float(grain.std())
    if sd > 0:
        grain /= sd

    lab1 = lab0.copy()
    lab1[..., 0] += alpha * delta[0] + noise * alpha * grain
    lab1[..., 1] += alpha * delta[1]
    lab1[..., 2] += alpha * delta[2]
    foot = alpha > 0.05
    clipped = float(((lab1 < 0) | (lab1 > 255)).any(axis=2)[foot].mean())

    out = img.copy()
    out[sl] = cv2.cvtColor(np.clip(lab1, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    # Что получилось на самом деле: dE меряется по записанным пикселям, в точке
    # пика маски (не по максимуму патча - его завышает зерно).
    got = cv2.cvtColor(out[sl], cv2.COLOR_BGR2Lab).astype(np.float32) - lab0
    de = np.sqrt((got[..., 0] * 100 / 255) ** 2 + got[..., 1] ** 2 + got[..., 2] ** 2)
    py, px = np.unravel_index(int(np.argmax(alpha)), alpha.shape)
    facts = dict(blur=blur, noise=noise, clipped=clipped,
                 de_peak=float(de[py, px]),
                 de_mean=float(de[foot].mean()) if foot.any() else 0.0,
                 bg_L=float(bg[0]))
    return out, facts


def jpeg_roundtrip(img, q):
    """Прогон кадра через JPEG: врезка платит за сжатие, как настоящий предмет."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


# --- сведение с выходом детектора -------------------------------------------


def match(blobs, x, y, size_px):
    """(ранг, score) лучшего блоба на врезке или None.

    Блобы уже отсортированы по score, поэтому первое совпадение - лучшее.
    Засчитывается либо близкий центр, либо накрывающая рамка: детектор вправе
    отметить край предмета, а не середину. Мягкость критерия компенсируется
    отбраковкой засвеченных точек (см. sweep): если рамка ловила эту точку и на
    чистом кадре, врезка ничего не доказывает и трайл выбрасывается.
    """
    tol = 6.0 + size_px / 2
    for i, (bx, by, bw, bh, _area, score) in enumerate(blobs):
        near = math.hypot(bx + bw / 2 - x, by + bh / 2 - y) <= tol
        inside = (bx - 2 <= x <= bx + bw + 2) and (by - 2 <= y <= by + bh + 2)
        if near or inside:
            return i, score
    return None


def frame_paths():
    """Кадры репозитория без повторов (часть файлов лежит в обеих папках)."""
    seen, out = {}, []
    for d in FRAME_DIRS:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h not in seen:
                seen[h] = p
                out.append(p)
    return out


def sweep(out_dir: Path, repeats: int, jpeg: int, seed: int, preview: int):
    paths = frame_paths()
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p}: не читается", file=sys.stderr)
            continue
        if jpeg:
            img = jpeg_roundtrip(img, jpeg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(dict(path=p, img=img, gray=gray, sites=site_map(img),
                           blobs=anomaly_blobs(img)))
        print(f"чистый {p.name}: блобов {len(frames[-1]['blobs'])}")

    prev_dir = out_dir / "preview"
    if preview:
        prev_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    colours = list(PALETTE)
    cells = [(bg, s, c) for bg in ("sneg", "osyp") for s in SIZES for c in CONTRASTS]
    for ci, (bg, size_px, contrast) in enumerate(cells):
        pool = [f for f in frames if len(f["sites"][bg]) >= 4]
        done_cell = 0
        for rep in range(repeats):
            rng = np.random.default_rng((seed, ci, rep))
            # Кадр перебирается по кругу (каждый вносит поровну), цвет берётся
            # жребием: при переборе по кругу обоих число кадров и число цветов
            # имеют общий делитель, и цвет намертво срастается с кадром.
            fr = pool[rep % len(pool)]
            colour = colours[int(rng.integers(len(colours)))]
            sites = fr["sites"][bg]
            x, y = sites[int(rng.integers(len(sites)))]
            done = inject(fr["img"], fr["gray"], x, y, size_px, contrast, colour, rng)
            if done is None:
                continue
            planted, facts = done
            if jpeg:
                planted = jpeg_roundtrip(planted, jpeg)
            hit = match(anomaly_blobs(planted), x, y, size_px)
            dirty = match(fr["blobs"], x, y, size_px)
            rows.append(dict(
                frame=fr["path"].name, width=fr["img"].shape[1], bg=bg, x=x, y=y,
                size=size_px, contrast=contrast, colour=colour,
                rank=-1 if hit is None else hit[0],
                score=0.0 if hit is None else hit[1],
                dirty=0 if dirty is None else 1, **facts))
            done_cell += 1
            if preview and rep == 0 and ci % max(1, len(cells) // preview) == 0:
                c = 3 * max(16, int(size_px))
                cv2.imwrite(str(prev_dir / f"{bg}_{size_px:02d}px_dE{contrast:.0f}_"
                                           f"{colour}.jpg"),
                            planted[max(0, y - c):y + c, max(0, x - c):x + c],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Врезок может быть меньше запрошенного: цвет, неотличимый от фона в
        # выпавшей точке (белое на снегу), пропускается, а не подделывается.
        print(f"ячейка {bg} {size_px}px dE{contrast:.0f}: врезок {done_cell}")

    tsv = out_dir / "trials.tsv"
    keys = ["frame", "width", "bg", "x", "y", "size", "contrast", "colour", "rank",
            "score", "dirty", "blur", "noise", "clipped", "de_peak", "de_mean", "bg_L"]
    with tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    return frames, rows


# --- отчёт ------------------------------------------------------------------


def recall(rows, bg, size_px, contrast, top, colours=None):
    """(recall, число зачтённых врезок) в ячейке.

    Засвеченные точки (детектор срабатывал там и на чистом кадре) выбрасываются
    из знаменателя, а не считаются попаданием: врезка в такое место не
    доказывает ни зрячести, ни слепоты.
    """
    sel = [r for r in rows if r["bg"] == bg and r["size"] == size_px
           and r["contrast"] == contrast and not r["dirty"]
           and (colours is None or r["colour"] in colours)]
    if not sel:
        return float("nan"), 0
    ok = sum(1 for r in sel if 0 <= r["rank"] < top)
    return ok / len(sel), len(sel)


def report(frames, rows, top, out_dir: Path, args):
    lines = []
    add = lines.append
    add(f"Порог различимости analysis/detect.py, врезок {len(rows)}, "
        f"кадров {len(frames)}, повторов на ячейку {args.repeats}, "
        f"JPEG-цикл {'q' + str(args.jpeg) if args.jpeg else 'нет'}")
    dirty = sum(r["dirty"] for r in rows)
    add(f"выброшено засвеченных точек: {dirty} ({dirty / max(len(rows), 1):.1%})")
    nb = [len(f["blobs"]) for f in frames]
    add(f"блобов на чистом кадре: медиана {statistics.median(nb):.0f}, "
        f"мин {min(nb)}, макс {max(nb)}; при отсмотре топ-{top} это столько же "
        f"ложных срабатываний на кадр")
    add("")
    for gname, gcol in GROUPS:
        for bg, name in (("sneg", "снег"), ("osyp", "осыпь")):
            add(f"recall, фон '{name}', {gname}, рабочая точка топ-{top} блобов "
                f"с кадра (в скобках - зачтённых врезок)")
            add(f"{'разм, px':<9}" + "".join(f"{c:>13.0f} dE" for c in CONTRASTS))
            for s in SIZES:
                row = [f"{s:<9d}"]
                for c in CONTRASTS:
                    r, n = recall(rows, bg, s, c, top, gcol)
                    row.append(f"{'  n/a' if not n else f'{r:5.2f}'}({n:>3}){'':>4}")
                add("".join(row))
            add("")
    # Разрешение кадра меняет ответ сильнее, чем фон: бюджет отсмотра задан на
    # кадр, а не на площадь, и на кадре 4K тот же предмет в пикселях конкурирует
    # вчетверо большим числом камней. Кадры 1080p - это съёмка с видео, ради
    # которой конвейер и делался; 4K - фотостопы.
    widths = sorted({r["width"] for r in rows})
    add(f"recall по разрешению кадра (цветные, dE >= 40, топ-{top}) и медиана "
        f"ранга врезки")
    add(f"{'разм, px':<9}" + "".join(f"{w:>10d}px" for w in widths for _ in (0,)))
    for s in SIZES:
        row = [f"{s:<9d}"]
        for w in widths:
            sel = [r for r in rows if r["size"] == s and r["width"] == w
                   and r["contrast"] >= 40 and not r["dirty"]
                   and r["colour"] in GROUPS[1][1]]
            if not sel:
                row.append(f"{'n/a':>12}")
                continue
            ok = sum(1 for r in sel if 0 <= r["rank"] < top) / len(sel)
            ranks = [r["rank"] for r in sel if r["rank"] >= 0]
            row.append(f"{f'{ok:.2f}/р{statistics.median(ranks):.0f}' if ranks else f'{ok:.2f}/-':>12}")
        add("".join(row))
    add("")
    # Последние две колонки - проверка самого стенда: пик обязан совпадать с
    # заказанным dE (нормировка после свёртки), а среднее по пятну обязано быть
    # заметно ниже пика у мелких, иначе врезка не размыта как надо.
    add(f"{'разм, px':<9}{'найдено (блоб есть)':>21}{'медиана ранга':>15}"
        f"{f'топ-{TOP_TIGHT}*':>10}{f'топ-{top}*':>10}"
        f"{'пик/заказ':>12}{'средн/пик':>12}{'обрезано':>10}")
    add(f"{'':<9}* цветные, dE >= 40, все разрешения")
    for s in SIZES:
        sel = [r for r in rows if r["size"] == s and not r["dirty"]]
        if not sel:
            continue
        got = [r for r in sel if r["rank"] >= 0]
        ranks = sorted(r["rank"] for r in got)
        op = [r for r in sel if r["contrast"] >= 40 and r["colour"] in GROUPS[1][1]]
        tight = sum(1 for r in op if 0 <= r["rank"] < TOP_TIGHT) / len(op) if op else 0.0
        wide = sum(1 for r in op if 0 <= r["rank"] < top) / len(op) if op else 0.0
        add(f"{s:<9d}{f'{len(got)}/{len(sel)}':>21}"
            f"{(f'{statistics.median(ranks):.0f}' if ranks else '-'):>15}"
            f"{tight:>10.2f}{wide:>10.2f}"
            f"{statistics.median([r['de_peak'] / r['contrast'] for r in sel]):>12.2f}"
            f"{statistics.median([r['de_mean'] / max(r['de_peak'], 1e-6) for r in sel]):>12.2f}"
            f"{statistics.median([r['clipped'] for r in sel]):>10.2f}")
    add("")
    # Диапазон, а не только нижний край: recall по размеру не обязан расти
    # монотонно. Крупный одноцветный предмет способен сам сделать свою ячейку
    # гистограммы частой и перестать быть аномалией.
    add("размеры (px) с recall >= 0.5, по фонам и контрастам")
    for gname, gcol in GROUPS:
        add(f"  {gname}")
        add(f"{'  фон':<11}" + "".join(f"{c:>10.0f} dE" for c in CONTRASTS))
        for bg, name in (("sneg", "снег"), ("osyp", "осыпь")):
            cells = []
            for c in CONTRASTS:
                good = [s for s in SIZES
                        if recall(rows, bg, s, c, top, gcol)[1]
                        and recall(rows, bg, s, c, top, gcol)[0] >= 0.5]
                cells.append(f"{(f'{min(good)}-{max(good)}' if good else 'нет'):>13}")
            add(f"  {name:<9}" + "".join(cells))
    add("")
    add(f"{'цвет':<9}" + "".join(f"{s:>7d} px" for s in SIZES))
    for col in PALETTE:
        cells = []
        for s in SIZES:
            sel = [r for r in rows if r["colour"] == col and r["size"] == s
                   and not r["dirty"]]
            ok = sum(1 for r in sel if 0 <= r["rank"] < top)
            cells.append(f"{(f'{ok / len(sel):.2f}' if sel else 'n/a'):>10}")
        add(f"{col:<9}" + "".join(cells))
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=30,
                    help="врезок на ячейку (размер x контраст x фон)")
    ap.add_argument("--jpeg", type=int, default=0,
                    help="качество JPEG-цикла для кадров; 0 - без цикла")
    ap.add_argument("--top", type=int, default=TOP_OP,
                    help="рабочая точка: сколько блобов с кадра уходит в отсмотр")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0,
                    help="сохранить столько примеров врезок для глазной проверки")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames_, rows_ = sweep(args.out, args.repeats, args.jpeg, args.seed, args.preview)
    report(frames_, rows_, args.top, args.out, args)
