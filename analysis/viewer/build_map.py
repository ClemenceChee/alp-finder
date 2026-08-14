#!/usr/bin/env python3
"""Генератор интерактивной карты кандидатов: map.html.

Слои: спутник/топооснова (онлайн-тайлы), изолинии из локального DEM
(Copernicus GLO-30, data/dem/N39E073.tif), плановый маршрут группы (GPX),
поисковый коридор и линия падения (docs/nakhodki/search_vectors.kml),
линии наискорейшего спуска из зоны интереса (считаются здесь по DEM),
подтверждённые вещи и кандидаты (курируемый список POINTS ниже,
источник — analysis/review/*.md и docs/nakhodki/README.md).

Каждому кандидату при сборке считаются расстояния до линии падения,
до планового маршрута и до ближайшей линии спуска из зоны — видны в попапе.

Сервер тот же, что для index.html:
python3 -m http.server 8077 -d <корень репо>; открывать
http://localhost:8077/analysis/viewer/map.html
"""

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
from geoproject import Dem  # noqa: E402

# Рабочая рамка карты (весь район операции: морена — вершина)
LAT0, LAT1 = 39.455, 39.525
LON0, LON1 = 73.565, 73.635

M_PER_DEG_LAT = 111132.0


def m_per_deg_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


# --- точки: курируемый список --------------------------------------------------
# conf: 5..1 — шкала штаба; "v" — подтверждённая вещь; "x" — кандидат закрыт.
# coord: как получена координата (честность важнее красоты).

GPS = "GPS дрона (объект может быть в стороне до ~145 м)"

# Источник вердикта/статуса по умолчанию для группы; у точки можно переопределить
# полем who=. Требование: в каждой карточке видно, КТО принял решение.
WHO = {
    "veshchi": "подтверждено штабом и TG-волонтёрами (опознание по фото, съёмка в упор)",
    "zona": "наш отсмотр (ИИ-агент + детектор аномалий); штабом не перепроверялось",
    "rayon": "наш отсмотр (ИИ-агент + детектор аномалий); штабом не перепроверялось",
    "pro3": "наш отсмотр (ИИ-агент + детектор аномалий); штабом не перепроверялось",
    "ropes": "вердикт штаба в TG: старые перила прошлых экспедиций",
    "trace": "наш полнокадровый отсмотр (ИИ-агент); штабом не перепроверялось",
    "tg-open": "заявка волонтёра в TG; вердикта никто не выносил",
    "tg-rej": "отклонено пилотами дрона штаба после повторного облёта (тема «Отклонённые»)",
    "closed": "закрыто нашим пиксельным разбором кадров (ИИ-агент); штабу не передавалось",
}
LRF = "лазерный дальномер + GPS (триангуляция штаба)"
PROJ = ("геопроекция луча в рельеф, ±десятки метров при пологом к склону луче "
        "(чувствительность к ошибке DEM — geoprojection.md, аудит 14.08)")

POINTS = [
    # -- подтверждённые вещи --
    dict(g="veshchi", conf="v", lat=39.482656, lon=73.586792, alt=4663,
         name="Синий рюкзак Николая", coord=LRF, video="C0049.MP4 + DJI_20260813184253", tc="0:15–0:21",
         desc="Опознан по фото «Эльбрус с Севера 2025». Склон 46°, зона зарождения лавины. "
              "Крупные планы (вертолёт C0048): красная стропа, зелёная бутылка, ручка второй палки. "
              "Дрон-ракурсы (та же кварцевая полоса): 184253 0:28–0:32, 163855 4:15 — "
              "изначально значились отдельными кандидатами из-за GPS-смещения 90–230 м.",
         imgs=["docs/nakhodki/frames/C0049_00m18s.png",
               "analysis/scans/heli-C0048/crops/t0007_00m55s.jpg",
               "analysis/scans/heli-C0048/crops/t0003_00m52s.jpg",
               "analysis/pilot/check_184253_t148_crop30s.jpg",
               "analysis/pilot/scan-163855/crops/t0131_04m15s.jpg"]),
    dict(g="veshchi", conf="v", lat=39.483176, lon=73.585463, alt=4529,
         name="Палка Komperdell №1", coord=LRF, video="DJI_20260813183140_0012_Z.JPG", tc="фото",
         desc="Пробковая ручка. Рядом палка второй пары — на её древке читается бренд «CAMP».",
         imgs=["data/drive/2026-08-13/drone-part3/DJI_20260813183140_0012_Z.JPG",
               "data/drive/2026-08-13/drone-part3/DJI_20260813183028_0002_Z.JPG",
               "analysis/pilot/scan-183004/crops/t0334_00m56s.jpg"]),
    dict(g="veshchi", conf="v", lat=39.483144, lon=73.585443, alt=4531,
         name="Бирюзовая крышка/миска", coord=LRF, video="DJI_20260813183727_0015_Z", tc="3:32",
         desc="Складная миска, в 4 м от палки.",
         imgs=["docs/nakhodki/frames/DJI_20260813183727_0015_Z_03m32s.jpg",
               "data/drive/2026-08-13/drone-part3/DJI_20260813183028_0002_Z.JPG"]),
    dict(g="veshchi", conf="v", lat=39.483403, lon=73.585137, alt=4552,
         name="Белый стакан, красная маркировка", coord=GPS, video="DJI_20260813183004_0001_Z", tc="3:40",
         desc="Координата — GPS дрона у объекта (оговорка GPS-ловушки).",
         imgs=["docs/nakhodki/frames/DJI_20260813183004_0001_Z_03m40s.jpg"]),
    dict(g="veshchi", conf="v", lat=39.483203, lon=73.585272, alt=4545,
         name="Оранжевый спальник Николая (позиция приблизительная)",
         who="опознание: консенсус TG-волонтёров (#758, #820); отчёт штаба осторожнее: «спальник или куртка»; CC называет «курткой»",
         coord="две оценки расходятся на ~29 м — не выдавать как точку",
         video="C0044/C0045, DJI_20260813163855/183004/183727", tc="крупно 3:26–4:00",
         desc="Стёганая ткань с белой биркой, ~1–3 м. Консенсус TG: спальник Николая — висел под "
              "рюкзаком в компрессионном мешке (Скриншоты #758, #820); документ CC называет «курткой». "
              "Формулировка для штаба: «в том же скоплении, что палка и крышка». "
              "Крупные планы — дрон 183727 3:31 и вертолёт C0044/C0045 (в упор). "
              "Дальний ракурс 3:02 того же видео ранее ошибочно значился отдельным «малым фрагментом» "
              "(ложное пересечение луча с рельефом); подлёт 3:02→3:28 (59 м) показывает один объект.",
         imgs=["analysis/pilot/scan-183727/crops/t0000_03m31s.jpg",
               "analysis/pilot/check_183727_frag302_zoom.jpg",
               "analysis/pilot/scan-163855/crops/t0000_01m49s.jpg",
               "analysis/scans/heli-C0044/crops/t0000_00m16s.jpg",
               "analysis/scans/heli-C0045/crops/t0010_00m35s.jpg",
               "analysis/pilot/scan-183004/crops/t0298_02m24s.jpg"], unc=30),

    # -- зона интереса 5385–5485 --
    dict(g="zona", conf=4, lat=39.47732, lon=73.59203, alt=5443,
         name="Г-образный тёмный предмет", coord=GPS, video="DJI_20260813152839_0005_Z", tc="1:42–1:44",
         who="наш отсмотр: ув. 4; в TG обсуждён с вердиктом волонтёра «99% камни» (#1428) — расхождение не снято, нужен облёт",
         desc="Изолирован на чистом снегу; на камень не похож, на ледоруб не тянет по пропорциям.",
         imgs=["analysis/scans/DJI_20260813152839_0005_Z/crops/t0000_01m44s.jpg",
               "analysis/pilot/check_152839_t0.jpg"]),
    dict(g="zona", conf=3, lat=39.47679, lon=73.59203, alt=5483,
         name="Дорожка отметин по линии падения", coord=GPS, video="DJI_20260811204542_0001_Z", tc="1:21–1:25",
         desc="Пунктирная дорожка тёмных отметин строго вниз на чистом снегу.",
         imgs=["analysis/scans/DJI_20260811204542_0001_Z/crops/t0058_01m25s.jpg",
               "analysis/pilot/check_204542_t58.jpg"]),
    dict(g="zona", conf=3, lat=39.47773, lon=73.59228, alt=5385,
         name="Поле 4–6 предметов с бороздами", coord=GPS, video="DJI_20260811202621_0002_Z (перескан)", tc="0:29–1:00",
         desc="Каждый предмет со своей бороздой-шлейфом; предметы ~0,2–1,3 м (геопроекция). Туман.",
         imgs=["analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0025_00m44s.jpg",
               "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0001_00m31s.jpg"]),
    dict(g="zona", conf=3, lat=39.47753, lon=73.59228, alt=5404,
         name="«Ткань?» — смятый охристо-бурый объект", coord=GPS, video="DJI_20260812143557_0001_Z", tc="5:58",
         desc="Фактура смятой ткани/брезента; может быть камень.",
         imgs=["analysis/scans/DJI_20260812143557_0001_Z/crops/t0016_05m58s.jpg",
               "analysis/pilot/check_143557_t16.jpg"]),
    dict(g="zona", conf=3, lat=39.47679, lon=73.59217, alt=5466,
         name="Россыпь предметов, один сегментированный", coord=GPS, video="DJI_20260812143557_0001_Z", tc="8:20–8:26",
         desc="Вытянутые тёмные предметы с бороздами; форма ледоруба/палки не исключена.",
         imgs=["analysis/scans/DJI_20260812143557_0001_Z/crops/t0079_08m20s.jpg",
               "analysis/scans/DJI_20260812143557_0001_Z/crops/t0292_08m26s.jpg"]),
    dict(g="zona", conf=2, lat=39.47764, lon=73.59227, alt=5395,
         name="Борозды без предметов", coord=GPS, video="DJI_20260811202621_0002_Z (перескан)", tc="1:51–1:59",
         desc="Сужающиеся борозды сквозь плотный туман; старый прогон их не видел.",
         imgs=["analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0041_01m58s.jpg"]),
    dict(g="zona", conf=2, lat=39.47788, lon=73.59225, alt=5366,
         name="Цепочка вмятин с шагом 1–2 м", coord=GPS, video="DJI_20260812143557_0001_Z", tc="3:28",
         desc="Пробоины от камней или заметённые следы; вторая цепочка в 14 м южнее (4:32).",
         imgs=["analysis/fullframe/DJI_20260812143557_0001_Z/enh/f0105.jpg"]),

    # -- кандидаты в районе вещей --
    dict(g="rayon", conf=3, lat=39.48309, lon=73.58512, alt=4556,
         name="Пенка/каремат?", coord=GPS, video="scan-183534", tc="0:54–0:57",
         desc="Бледно-зелёный прямоугольник с прямыми кромками на бурой осыпи; в ~40 м от палки/крышки.",
         imgs=["analysis/pilot/scan-183534/crops/t0237_00m54s.jpg"]),
    dict(g="rayon", conf=3, lat=39.48327, lon=73.58513, alt=4559,
         name="Бирюзовое кольцо + оранжевый фрагмент (сведены 2 наблюдения)",
         coord="проекция центра кадра по телеметрии: наблюдения 1:23 и 2:41 "
               "с разных точек полёта сходятся в одну точку (±30 м DEM)",
         video="scan-163855", tc="1:23, 2:37–2:41",
         desc="Цвет неприродный. Оба наблюдения — один предмет; лежит в кластере "
              "стакан/пенка, а не в 70–150 м севернее, как давал GPS дрона.",
         imgs=["analysis/pilot/scan-163855/crops/t0257_01m23s.jpg",
               "analysis/pilot/scan-163855/crops/t0271_02m41s.jpg",
               "analysis/pilot/scan-163855/crops/t0273_02m37s.jpg"]),
    # -- прочие кандидаты ув. 3 --
    dict(g="pro3", conf=3, lat=39.48547, lon=73.59488, alt=4992,
         name="Воронка удара + предмет в снегу", coord=GPS, video="DJI_20260811161324_0003_Z", tc="0:03–0:04",
         desc="Пятно с 3 расходящимися лучами + тёмный предмет, утопленный в снег. ВАЖНО: с "
              "траекторией вещей НЕ связана — в ~490 м восточнее линий падения и выше рюкзака; "
              "её собственная линия спуска уходит в соседний кулуар (ближе 694 м к рюкзаку не "
              "подходит). Самостоятельный кандидат: в 62 м от нитки маршрута, участок у Camp1.",
         imgs=["analysis/scans/DJI_20260811161324_0003_Z/crops/t0018_00m04s.jpg",
               "analysis/scans/DJI_20260811161324_0003_Z/crops/t0003_00m03s.jpg"]),
    dict(g="pro3", conf=3, lat=39.48597, lon=73.59433, alt=4952,
         name="Каплевидный объект на ровном снегу", coord=GPS, video="DJI_20260811161324_0003_Z", tc="3:14",
         desc="Одиночный тёмный + 2 мелких пятна рядом.",
         imgs=["analysis/scans/DJI_20260811161324_0003_Z/crops/t0002_03m14s.jpg"]),
    dict(g="pro3", conf=3, lat=39.47075, lon=73.58860, alt=5859,
         name="«Трепыхающаяся ткань» на гребне", coord=GPS, video="DJI_20260813095243_0001_Z", tc="0:05–0:21",
         desc="Светлое зеленовато-серое пятно на скальном выступе; совпадает с кандидатом TG (#1176).",
         imgs=["analysis/scans/DJI_20260813095243_0001_Z/crops/t0008_00m15s.jpg"]),
    dict(g="pro3", conf=3, lat=39.47845, lon=73.58840, alt=5879,
         name="Тёмный угловатый объект, светлое включение", coord=GPS, video="DJI_20260813101405_0001_Z", tc="1:32",
         desc="Поверх чистого снега, отчётливая тень, вокруг мелкие отметины.",
         imgs=["analysis/scans/DJI_20260813101405_0001_Z/crops/t0057_01m32s.jpg"]),
    dict(g="pro3", conf=3, lat=39.46882, lon=73.58942, alt=6096,
         name="Объект с бороздой у высоты вершины", coord=GPS, video="DJI_20260813103527_0001_Z", tc="3:10–3:23",
         desc="Тёмный угловатый, позади борозда; рядом плоский вытянутый предмет.",
         imgs=["analysis/scans/DJI_20260813103527_0001_Z/crops/t0017_03m18s.jpg"]),
    dict(g="pro3", conf=3, lat=39.47640, lon=73.59509, alt=5381,
         name="Х-образный объект + борозда на выносе", coord=GPS, video="DJI_20260813130531_0001_Z", tc="0:44, 2:38–2:47",
         desc="У кромки лавинного выноса; длинная прямая борозда с предметом в верхней части.",
         imgs=["analysis/scans/DJI_20260813130531_0001_Z/crops/t0026_02m40s.jpg",
               "analysis/scans/DJI_20260813130531_0001_Z/crops/t0130_00m44s.jpg"]),
    dict(g="pro3", conf=3, lat=39.48138, lon=73.59511, alt=5108,
         name="Кремовый округлый объект (каска?)", coord=GPS, video="DJI_20260813133738_0001_Z", tc="6:26–6:58",
         desc="Гладкий, среди острой серой осыпи.",
         imgs=["analysis/scans/DJI_20260813133738_0001_Z/crops/t0006_06m38s.jpg"]),
    dict(g="pro3", conf=3, lat=39.48064, lon=73.59260, alt=5155,
         name="Объект с жёлтым включением у бергшрунда", coord=GPS, video="DJI_20260813152650_0001_Z", tc="0:00–0:05",
         desc="Тёмный угловатый, у кромки трещины.",
         imgs=["analysis/scans/DJI_20260813152650_0001_Z/crops/t0001_00m02s.jpg",
               "analysis/pilot/check_152650_t1.jpg"]),
    dict(g="pro3", conf=3, lat=39.48247, lon=73.59298, alt=5088,
         name="Овальный предмет на чистом снегу", coord=GPS, video="DJI_20260811204542_0001_Z", tc="4:18",
         desc="Изолированный тёмный гладкий, отчётливая тень, ореола протаивания нет.",
         imgs=["analysis/scans/DJI_20260811204542_0001_Z/crops/t0189_04m18s.jpg"]),

    # -- верёвки и перила --
    dict(g="ropes", conf=5, lat=39.48097, lon=73.59262, alt=5146,
         name="Линия старых перил (0:00–2:18)", coord=GPS, video="DJI_20260812135747_0002_Z", tc="0:00–2:18",
         desc="Верёвка через скальный рог, узлы, станции; там же красная верёвка 0:30 (цвет основной верёвки группы).",
         imgs=["analysis/scans/DJI_20260812135747_0002_Z/crops/t0000_00m16s.jpg",
               "analysis/scans/DJI_20260812135747_0002_Z/crops/t0005_00m33s.jpg"]),
    dict(g="ropes", conf=4, lat=39.47992, lon=73.59255, alt=5211,
         name="Верёвки выше по стене", coord=GPS, video="DJI_20260812140210_0004_Z", tc="0:29–1:09",
         desc="Тонкие прямые линии на скальной стене в тумане.",
         imgs=["analysis/scans/DJI_20260812140210_0004_Z/crops/t0014_00m29s.jpg"]),
    dict(g="ropes", conf=4, lat=39.48558, lon=73.59511, alt=4985,
         name="Верёвка ниже + оранжевая точка", coord=GPS, video="DJI_20260812133415_0006_Z", tc="0:44",
         desc="Тонкая светлая линия поперёк тёмной скалы.",
         imgs=["analysis/scans/DJI_20260812133415_0006_Z/crops/t0073_00m44s.jpg"]),
    dict(g="ropes", conf=4, lat=39.481591, lon=73.592787, alt=5073,
         name="Верёвка на гребне (±48 м)", coord="оценка внешнего анализа (CSV CC)", video="DJI_20260811204542_0001_Z", tc="3:24",
         who="нашли наблюдатели штаба (Подтверждённые #577, #838); геометрию измерила автоматика CC — этот снимок человеком в конвейере CC не отсматривался",
         desc="Нашли наблюдатели штаба, CC измерил: тонкая тёмная линия постоянной ширины (~10 м, "
              "~3 см — как верёвка 10 мм с размытием оптики) идёт через снег и скалу без излома — "
              "рельеф так не умеет. Видна в одиночном кадре без обработки. Проверить, не одна ли "
              "это система перил с линией 5146.",
         imgs=["docs/nezavisimyy-analiz/screens/20260811204542_00-03-24_1_single.png",
               "docs/nezavisimyy-analiz/screens/20260811204542_00-03-24_0_frame.jpg"], unc=48),
    dict(g="ropes", conf="v", lat=39.48181, lon=73.59288, alt=5097,
         name="След срыва (калибровочный, известный)", coord=GPS, video="DJI_20260812135426_0001_Z", tc="0:41",
         who="зафиксирован волонтёрами TG (Подтверждённые #836); у нас служит калибровкой поиска рельефных следов",
         desc="Узкая борозда строго вниз по склону, читается только рельефной тенью. Там же в кадре "
              "верёвка (Подтверждённые #836).",
         imgs=["analysis/fullframe/_calib/t41_track_zoom.jpg"]),
    dict(g="ropes", conf=5, lat=39.48181, lon=73.59288, alt=5097,
         name="Скальный крюк", coord=GPS, video="DJI_20260812135426_0001_Z", tc="0:15–0:18",
         desc="Подтверждён штабом (Подтверждённые #319). Маркер провешенного пути.",
         imgs=["data/telegram/podtverzhdennye/319.jpg"]),
    dict(g="ropes", conf=5, lat=39.48136, lon=73.59294, alt=5180,
         name="Верёвки: «пещера», вокруг скалы, короткие", coord=GPS, video="DJI_20260811202212_0001_Z", tc="0:46–1:20",
         desc="Несколько верёвок: две у «пещеры» (1:06), вокруг скалы (1:18), короткие (0:46–0:50). "
              "Консенсус штаба: перила прошлых экспедиций — у группы было всего 2 верёвки "
              "(Подтверждённые #370–#773).",
         imgs=["data/telegram/podtverzhdennye/732.jpg",
               "data/telegram/podtverzhdennye/731.jpg",
               "data/telegram/podtverzhdennye/419.jpg"]),

    # -- следы/рельеф ув. 2 --
    dict(g="trace", conf=2, lat=39.48247, lon=73.59298, alt=5088,
         name="Провал снежного моста у бергшрунда", coord=GPS, video="DJI_20260811204542_0001_Z", tc="4:16",
         desc="Тёмная полость, вывороченные блоки наста; следов подхода не видно.",
         imgs=["analysis/fullframe/DJI_20260811204542_0001_Z/enh/f0129.jpg"]),
    dict(g="trace", conf=2, lat=39.48406, lon=73.59367, alt=5085,
         name="Две смежные лунки в снегу", coord=GPS, video="DJI_20260811202212_0001_Z", tc="0:39",
         desc="Смазанные края — следы/точки ударов?",
         imgs=["analysis/scans/DJI_20260811202212_0001_Z/crops/t0002_00m39s.jpg"]),
    dict(g="trace", conf=2, lat=39.47775, lon=73.59226, alt=5376,
         name="Вторая цепочка вмятин", coord=GPS, video="DJI_20260812143557_0001_Z", tc="4:32",
         desc="3–4 вытянутых вмятины с равным шагом, ~14 м южнее первой.",
         imgs=["analysis/fullframe/DJI_20260812143557_0001_Z/enh/f0137.jpg"]),

    # -- TG: открытые кандидаты волонтёров --
    dict(g="tg-open", conf=None, lat=39.49260, lon=73.59717, alt=4852,
         name="«Турик?» — каменная пирамидка?", coord=GPS, video="DJI_20260811160637_0001_Z", tc="0:44",
         desc="Заявка волонтёра: «Координаты турика?» (Перепроверка #582). Вердикта в TG нет.",
         imgs=["data/telegram/dlya-pereproverki-dronom/582.jpg"]),
    dict(g="tg-open", conf=2, lat=39.49154, lon=73.59807, alt=4851,
         name="CC: тёмная «линза» 12×5 px на снегу", coord="позиция дрона; CC: «только азимут, точка не выдаётся»",
         video="DJI_20260811160637_0001_Z", tc="1:14",
         who="автоматическая детекция и автоматическая проверка CC; человеком не отсмотрено (пометка самого CC)",
         desc="Кандидат автоматической проверки CC (kurumdy_candidates.csv).",
         imgs=["docs/nezavisimyy-analiz/screens/20260811160637_00-01-14_1_single.png",
               "docs/nezavisimyy-analiz/screens/20260811160637_00-01-14_0_frame.jpg"]),
    dict(g="tg-open", conf=1, lat=39.49130, lon=73.59813, alt=4851,
         name="«Камни?» (Гипотезы #919)", coord=GPS, video="DJI_20260811160637_0001_Z", tc="2:01",
         desc="Заявка волонтёра без вердикта. В нашем отсмотре этот район — ув. ≤2 (зона проталин).",
         imgs=["data/telegram/1-gipotezy/919.jpg"]),
    dict(g="tg-open", conf=2, lat=39.47075, lon=73.58859, alt=5859,
         name="Обледеневший кусок с чётким краем на гребне?", coord=GPS, video="DJI_20260813095243_0001_Z", tc="0:30–0:35",
         desc="«Чёткий край, чёткий перегиб на подветренной части» (Перепроверка #993). Рядом с "
              "кандидатом «трепыхающаяся ткань». Облёт не выполнен (нет точных координат).",
         imgs=["data/telegram/dlya-pereproverki-dronom/993.jpg"]),
    dict(g="tg-open", conf=2, lat=39.48991, lon=73.59731, alt=4858,
         name="Блоки, пробившие снежную корку, у трещины", coord="координаты из заявки (позиция дрона)",
         video="DJI_20260812133237_0005_Z", tc="0:00",
         desc="Группа блоков у трещины, над которой висел дрон (Перепроверка #1019, #1051 — одно место).",
         imgs=["data/telegram/dlya-pereproverki-dronom/1019.jpg",
               "data/telegram/dlya-pereproverki-dronom/1051.jpg"]),
    dict(g="tg-open", conf=2, lat=39.47993, lon=73.59188, alt=5211,
         name="«Дно рюкзака?» — снег необычно лежит на камне", coord=GPS,
         video="DJI_20260813124213_0001_Z", tc="0:26",
         desc="Заявка волонтёра (Перепроверка #1255–#1256). Вердикта в TG нет.",
         imgs=["data/telegram/dlya-pereproverki-dronom/1255.jpg"]),
    dict(g="tg-open", conf=2, lat=39.48171, lon=73.59279, alt=5116,
         name="CC: тёмный объект 15–20 px на снегу", coord="позиция дрона; CC: «только азимут, точка не выдаётся»",
         video="DJI_20260811204542_0001_Z", tc="3:52",
         who="автоматическая детекция и автоматическая проверка CC; человеком не отсмотрено (пометка самого CC)",
         desc="Окно 4:06–4:40 этого видео — краевая трещина ледника: рядом полость, куда предмет "
              "может провалиться.",
         imgs=["docs/nezavisimyy-analiz/screens/20260811204542_00-03-52_1_single.png",
               "docs/nezavisimyy-analiz/screens/20260811204542_00-03-52_0_frame.jpg"]),
    dict(g="tg-open", conf=1, lat=39.47528, lon=73.59176, alt=5589,
         name="CC: тёмный объект 6 px с бороздой", coord="позиция дрона; координат нет и не будет (фокусное не измеримо)",
         video="DJI_20260812141803_0001_Z", tc="0:14",
         who="автоматическая детекция и автоматическая проверка CC; человеком не отсмотрено (пометка самого CC)",
         desc="Весь вылет 141803 (5572–5673 м — рабочие высоты группы) не геопривязывается — "
              "CC и мы рекомендуем переснять.",
         imgs=["docs/nezavisimyy-analiz/screens/20260812141803_00-00-14_1_single.png",
               "docs/nezavisimyy-analiz/screens/20260812141803_00-00-14_0_frame.jpg"]),

    dict(g="tg-open", conf=3, lat=39.48431, lon=73.58402, alt=4673,
         name="Две крупные борозды у района вещей (TG #1752)", coord=GPS + "; камера вниз −38° — объект близко",
         video="DJI_20260813163855_0001_Z", tc="0:53–0:54",
         desc="Две U-образные борозды на снежнике — дрон в этот момент над районом находок вещей. "
              "Рельефный след, цветовой детектор такие не ловит; вердикта в TG нет.",
         imgs=["data/telegram/novye-skrinshoty/1752.jpg",
               "analysis/fullframe/DJI_20260813163855_0001_Z/f0027.jpg"]),
    dict(g="tg-open", conf=2, lat=39.47993, lon=73.59188, alt=5210,
         name="Кулуар 124213: следы скольжения, «след рюкзака?», пещеры", coord=GPS + "; камера ВВЕРХ (+11…24°) — объекты выше 5210 м, в сторону зоны интереса",
         video="DJI_20260813124213_0001_Z", tc="1:58–2:38",
         desc="Пакет заявок волонтёров: два следа скольжения (один «похож на след от рюкзака», "
              "#1294–#1296), изолированный след (#1305), след камня + две «пещеры» (#1258). "
              "Контраргумент в TG: рельефность снега мелкая, рюкзак был бы крупнее (#1301).",
         imgs=["data/telegram/novye-skrinshoty/1296.png",
               "data/telegram/novye-skrinshoty/1305.png",
               "data/telegram/novye-skrinshoty/1258.jpg"]),
    dict(g="tg-open", conf=3, lat=39.478187, lon=73.591547, alt=5296,
         name="Верёвка над «надписью LOOK» (личка Геннадия)",
         coord="геопроекция луча в рельеф с 20 м; вилка фокусного <2 м, но луч полого к склону — "
               "вилка DEM уводит точку до ~16 м вдоль склона",
         video="DJI_20260813124655_0003_Z", tc="1:07–1:09",
         who="находка Геннадия Беге (админ TG-группы), передана в личку 14.08 ~04:50; "
             "«должны сегодня проверить» — облёт планировался штабом на 14.08",
         desc="Тонкая светлая линия ~3.5–4 м (при GSD 0.4 см/пикс толщина 1–2 см — масштаб "
              "верёвки), уходит вниз к камням: «справа верёвка идёт к камню». В ~30 м выше по "
              "склону закрытой «надписи LOOK». Версия Геннадия: если LOOK — надпись, могли "
              "написать с верёвки, а потом подниматься.",
         imgs=["data/telegram/lichka-gennadiy-bege/148979.jpg",
               "data/telegram/lichka-gennadiy-bege/148983.jpg",
               "analysis/fullframe/gennadiy-124655/crop_t67_rope_zoom.png"], unc=30),
    dict(g="tg-open", conf=2, lat=39.478322, lon=73.591417, alt=5268,
         name="Тёмный предмет у «надписи LOOK» (личка Геннадия)",
         coord="надирная геопроекция (подвес −90.7°, дистанция 34 м): к фокусному нечувствительна, ±10 м",
         video="DJI_20260813124655_0003_Z", tc="1:54",
         who="находка Геннадия Беге (админ TG-группы), передана в личку 14.08 ~04:50; "
             "«должны сегодня проверить» — облёт планировался штабом на 14.08",
         desc="Слабый тёмный вытянутый объект на чистом снегу (обводка ~1×0.7 м на местности, сам "
              "предмет меньше; весь надирный кадр покрывает ~10×6 м). Практически в точке "
              "«надписи LOOK». Слова Геннадия: «предмет (возможно камни), похожий на тело "
              "альпиниста».",
         imgs=["data/telegram/lichka-gennadiy-bege/148980.jpg",
               "analysis/fullframe/gennadiy-124655/crop_t114_zoom.png"], unc=10),
    dict(g="tg-open", conf=2, lat=39.47994, lon=73.59188, alt=5210,
         name="Фестончатая дорожка по центру кулуара + «отрыв лавины»", coord=GPS + "; камера ВВЕРХ — объекты выше 5210 м",
         video="DJI_20260813124213_0001_Z", tc="0:57–1:50",
         desc="Непрерывная ~50 кадров дорожка по центру кулуара — следы спуска/падения или русло "
              "камнепада (#1262); на гребне след, похожий на отрыв лавины, ниже — «следы падения» "
              "(#1281). Вердикта в TG нет.",
         imgs=["data/telegram/novye-skrinshoty/1262.jpg",
               "data/telegram/novye-skrinshoty/1281.jpg"]),

    # -- TG: отклонено после облёта (ложные срабатывания) --
    dict(g="tg-rej", conf="x", lat=39.48181, lon=73.59288, alt=5097,
         name="«Тело головой вниз» — наледь", coord=GPS, video="DJI_20260812135426_0001_Z", tc="0:54",
         desc="Пилоты пересмотрели в полном качестве: наледь, «человек не мог так покрыться льдом "
              "и прилипнуть к скале» (General #214–#215; Отклонённые #278).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/278.jpg"]),
    dict(g="tg-rej", conf="x", lat=39.48181, lon=73.59288, alt=5097,
         name="Прямоугольное пятно — маленький лёд", coord=GPS, video="DJI_20260812135426_0001_Z", tc="0:44",
         desc="«И цвет, и форма» — проверили: лёд (Отклонённые #512–#515).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/512.jpg"]),
    dict(g="tg-rej", conf="x", lat=39.48030, lon=73.59272, alt=5242,
         name="Синий объект — камень", coord=GPS, video="DJI_20260811202212_0001_Z", tc="1:20",
         desc="«Чётко виден синий цвет» — проверили: камень (Отклонённые #518–#519).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/518.jpg"]),
    dict(g="tg-rej", conf="x", lat=39.48030, lon=73.59272, alt=5242,
         name="«Фигура у следов» и «след схода» — камни", coord=GPS, video="DJI_20260811202212_0001_Z", tc="1:19–1:21",
         desc="Силуэт, похожий на человека, над следами + протяжённый «след схода снега» — "
              "проверили: камни (Отклонённые #552–#557; Перепроверка #573).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/552.jpg",
               "data/telegram/dlya-pereproverki-dronom/573.jpg"]),
    dict(g="tg-rej", conf="x", lat=39.48080, lon=73.59254, alt=5160,
         name="Подозрительное пятно — снег и камни", coord=GPS, video="DJI_20260812140054_0003_Z", tc="0:27–0:36",
         desc="«Перепроверили — это интересно, завтра полетим» → слетали: снег и камни "
              "(Отклонённые #961–#964).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/961.jpg"]),
    dict(g="tg-rej", conf="x", lat=39.46882, lon=73.58942, alt=6096,
         name="Объект, на который зумился дрон — камни", coord=GPS, video="DJI_20260813103527_0001_Z", tc="1:57",
         desc="«Видно, что дрон пытался зазумиться» — проверили: камни (Отклонённые #995–#996).",
         imgs=["data/telegram/otklonennye-posle-proverki-dronom/995.jpg"]),

    # -- закрытые кандидаты --
    dict(g="closed", conf="x", lat=39.47875, lon=73.59309, alt=5257,
         name="«Красный сегмент» — закрыт: скальные выходы", coord=PROJ, video="DJI_20260813131019_0002_Z", tc="~2:49",
         desc="Не ткань/стропа (4/5): объект холодно-серый, «красность» — артефакт скрина; "
              "вердикт пиксельный, от геопривязки не зависит. Координата заявки (5318 м) была "
              "GPS дрона — расчётная точка на ~170 м в стороне, сама ±десятки метров.",
         imgs=["analysis/review/tg-2026-08-13-framecheck/red/aligned_170.00.png"]),
    dict(g="closed", conf="x", lat=39.47832, lon=73.59142, alt=5268,
         name="«Надпись LOOK» — закрыта: тени на снегу", coord=PROJ, video="DJI_20260813124655_0003_Z", tc="1:59",
         desc="Не надпись (5/5): рисунок теней и неровностей снега, стабилен 5 секунд.",
         imgs=["analysis/review/tg-2026-08-13-framecheck/slope/aligned_118.00.png"]),
]

GROUPS = [
    ("veshchi", "Вещи группы (подтверждённые + оранжевые)", True),
    ("zona", "Зона интереса 5385–5485 м", True),
    ("rayon", "Кандидаты в районе вещей", True),
    ("pro3", "Прочие кандидаты ув. 3", True),
    ("ropes", "Верёвки, перила, след срыва", True),
    ("trace", "Следы/рельеф ув. 2", False),
    ("tg-open", "TG: открытые кандидаты волонтёров", True),
    ("tg-rej", "TG: отклонено после облёта (ложные)", True),
    ("closed", "Закрыто нашим разбором кадров", True),
]

CAMPS = [
    ("ABC", 39.513125, 73.606729, 4049),
    ("Camp1", 39.484587, 73.594884, 5018),
    ("Camp2", 39.476132, 73.592438, 5529),
    ("Camp3", 39.467957, 73.590464, 6062),
    ("Real C1 (из СМС)", 39.483535, 73.593962, None),
    ("Верёвки (из СМС)", 39.480960, 73.592631, None),
]

# Стартовые точки линий наискорейшего спуска (кандидаты зоны + перила)
DESCENT_STARTS = [
    ("от дорожки отметин 5483", 39.47679, 73.59203),
    ("от «ткани?» 5404", 39.47753, 73.59228),
    ("от поля с бороздами 5385", 39.47773, 73.59228),
    ("от Г-образного 5443", 39.47732, 73.59203),
    ("от борозд 5395", 39.47764, 73.59227),
    ("от перил 5146", 39.48097, 73.59262),
]

ZONE_RECT = [[39.4768, 73.5920], [39.4777, 73.5923]]           # зона интереса
IMPACT_RECT = [[39.4855, 73.5943], [39.4860, 73.5949]]         # ударные отметины

# Слепое пятно склона между зоной интереса и вещами: полигон и ячейки покрытия
# считаются analysis/coverage_polygon.py (метод — по центрам кадров, допуск 75 м)
SLOPE_POLY_JSON = ROOT / "analysis/coverage/sklon-poligon-2026-08-14.json"
SLOPE_CELLS_TSV = ROOT / "analysis/coverage/sklon-poligon-2026-08-14.cells.tsv"


def slope_blind():
    """(полигон, слепые ячейки [[lat, lon], ...], доля слепых) или None."""
    if not (SLOPE_POLY_JSON.exists() and SLOPE_CELLS_TSV.exists()):
        return None
    poly = json.loads(SLOPE_POLY_JSON.read_text())
    rows = [l.split("\t") for l in SLOPE_CELLS_TSV.read_text().splitlines()[1:]]
    blind = [[float(r[0]), float(r[1])] for r in rows if r[3] == "0"]
    return dict(poly=poly, blind=blind, share=round(len(blind) / len(rows), 2))


# --- исходники ------------------------------------------------------------------


def source_index():
    files = []
    for pat in ("*.MP4", "*.mp4", "*.JPG", "*.jpg"):
        files += (ROOT / "data/drive").rglob(pat)
    return sorted({p.relative_to(ROOT) for p in files})


SOURCES = source_index()


def source_paths(video_label):
    """Полные пути исходников по подписи точки (в подписи может быть несколько видео).

    Понимает полные имена (DJI_..._Z[.JPG]), вертолётные клипы (C0049) и
    шестизначные времена-сокращения (scan-183534, «.../183004/183727»)."""
    label = video_label or ""
    tokens = set(re.findall(r"DJI_\d{14}_\d{4}_Z(?:\.JPG)?", label))
    tokens |= set(re.findall(r"C\d{4}", label))
    tokens |= set(re.findall(r"(?<![0-9])\d{6}(?![0-9])", label))
    tokens |= {m[-6:] for m in re.findall(r"DJI_(\d{14})(?!\d|_\d{4}_Z)", label)}
    out = []
    for p in SOURCES:
        for t in tokens:
            want_jpg = t.endswith(".JPG")
            if t.removesuffix(".JPG") in p.name and (p.suffix.upper() == ".JPG") == want_jpg:
                out.append(str(p))
                break
    return out


# --- геометрия ------------------------------------------------------------------


def descent_line(dem, lat, lon, step_m=12.0, max_steps=400, min_slope_deg=4.0):
    """Линия наискорейшего спуска по DEM от точки; [(lat, lon), ...]."""
    pts = [(lat, lon)]
    for _ in range(max_steps):
        h = 15.0
        dlat = h / M_PER_DEG_LAT
        dlon = h / m_per_deg_lon(lat)
        try:
            dz_dn = (dem.elev(lat + dlat, lon) - dem.elev(lat - dlat, lon)) / (2 * h)
            dz_de = (dem.elev(lat, lon + dlon) - dem.elev(lat, lon - dlon)) / (2 * h)
        except ValueError:
            break
        g = math.hypot(dz_dn, dz_de)
        if g < math.tan(math.radians(min_slope_deg)):
            break
        de, dn = -dz_de / g, -dz_dn / g
        lat += dn * step_m / M_PER_DEG_LAT
        lon += de * step_m / m_per_deg_lon(lat)
        if not (LAT0 < lat < LAT1 and LON0 < lon < LON1):
            break
        pts.append((round(lat, 6), round(lon, 6)))
    return pts


def dist_to_line_m(lat, lon, line):
    """Мин. расстояние (м) от точки до полилинии [(lat, lon), ...]."""
    kx = m_per_deg_lon(lat)
    px, py = lon * kx, lat * M_PER_DEG_LAT
    best = float("inf")
    for (la1, lo1), (la2, lo2) in zip(line, line[1:]):
        x1, y1 = lo1 * kx, la1 * M_PER_DEG_LAT
        x2, y2 = lo2 * kx, la2 * M_PER_DEG_LAT
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0 if L2 == 0 else max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / L2))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


# --- источники ------------------------------------------------------------------


def strip_ns(tag):
    return tag.rsplit("}", 1)[-1]


def parse_gpx(path):
    """(track=[(lat,lon)...], waypoints=[(name,lat,lon)...])"""
    root = ET.parse(path).getroot()
    track, wpts = [], []
    for el in root.iter():
        t = strip_ns(el.tag)
        if t == "trkpt":
            track.append((float(el.attrib["lat"]), float(el.attrib["lon"])))
        elif t == "wpt":
            name = next((c.text for c in el if strip_ns(c.tag) == "name"), "")
            wpts.append((name, float(el.attrib["lat"]), float(el.attrib["lon"])))
    return track, wpts


def parse_kml_lines(path):
    """{имя placemark: [(lat, lon), ...]} для LineString и Polygon."""
    root = ET.parse(path).getroot()
    out = {}
    for pm in root.iter():
        if strip_ns(pm.tag) != "Placemark":
            continue
        name = next((c.text for c in pm if strip_ns(c.tag) == "name"), "")
        for el in pm.iter():
            if strip_ns(el.tag) == "coordinates" and el.text:
                pts = []
                for tok in el.text.split():
                    lon, lat = tok.split(",")[:2]
                    pts.append((float(lat), float(lon)))
                if len(pts) > 1:
                    out[name] = pts
    return out


def contours_geo(dem, levels):
    """[(уровень, [[(lat, lon), ...], ...]), ...] по рамке карты."""
    import contourpy
    x0 = int((LON0 - dem.lon0) / dem.dlon)
    x1 = int((LON1 - dem.lon0) / dem.dlon) + 1
    y0 = int((dem.lat0 - LAT1) / dem.dlat)
    y1 = int((dem.lat0 - LAT0) / dem.dlat) + 1
    z = dem.z[y0:y1, x0:x1]
    gen = contourpy.contour_generator(z=z)
    out = []
    for lvl in levels:
        lines = []
        for arr in gen.lines(lvl):
            pts = [(round(dem.lat0 - (y0 + p[1]) * dem.dlat, 6),
                    round(dem.lon0 + (x0 + p[0]) * dem.dlon, 6)) for p in arr[::2]]
            if len(pts) > 1:
                lines.append(pts)
        if lines:
            out.append((lvl, lines))
    return out


# --- сборка ----------------------------------------------------------------------


def build():
    dem = Dem()

    track, wpts = parse_gpx(ROOT / "docs/marshrut/plan-track.gpx")
    kml = parse_kml_lines(ROOT / "docs/nakhodki/search_vectors.kml")
    fall = next(v for k, v in kml.items() if k.startswith("Fall line"))
    prio = next(v for k, v in kml.items() if k.startswith("PRIORITY"))
    corridor = next(v for k, v in kml.items() if "corridor" in k)

    descents = [(label, descent_line(dem, la, lo)) for label, la, lo in DESCENT_STARTS]
    contours = contours_geo(dem, range(4100, 6101, 50))

    pts = []
    for p in POINTS:
        d_fall = dist_to_line_m(p["lat"], p["lon"], fall)
        d_route = dist_to_line_m(p["lat"], p["lon"], track)
        d_desc = min(dist_to_line_m(p["lat"], p["lon"], line) for _, line in descents)
        imgs = [i for i in p.get("imgs", []) if (ROOT / i).exists()]
        pts.append({**{k: v for k, v in p.items() if k != "imgs"},
                    "who": p.get("who", WHO[p["g"]]),
                    "imgs": imgs, "srcs": source_paths(p.get("video", "")),
                    "dFall": round(d_fall), "dRoute": round(d_route), "dDesc": round(d_desc),
                    "dTraj": round(min(d_fall, d_desc))})

    data = dict(
        groups=[dict(id=g, title=t, on=on) for g, t, on in GROUPS],
        points=pts,
        camps=[dict(name=n, lat=la, lon=lo, alt=al) for n, la, lo, al in CAMPS],
        route=[[round(la, 6), round(lo, 6)] for la, lo in track],
        wpts=[dict(name=n, lat=la, lon=lo) for n, la, lo in wpts],
        fall=fall, prio=prio, corridor=corridor,
        descents=[dict(label=l, line=v) for l, v in descents],
        contours=[dict(lvl=lvl, lines=lines) for lvl, lines in contours],
        zone=ZONE_RECT, impact=IMPACT_RECT,
        slope=slope_blind(),
    )
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "map.html").write_text(html, "utf-8")
    kb = (OUT / "map.html").stat().st_size // 1024
    print(f"готово: {OUT/'map.html'} ({kb} КБ), точек {len(pts)}, "
          f"изолиний {sum(len(c['lines']) for c in data['contours'])}")


TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карта — Курумды</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root { --bg:#14161a; --card:#1d2026; --text:#e6e8ec; --dim:#9aa3af; --line:#2b2f37; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif; height:100vh;
       display:flex; flex-direction:column; }
header { background:#14161acc; border-bottom:1px solid var(--line); padding:8px 16px; z-index:1001; }
header nav { display:flex; gap:16px; align-items:baseline; }
header .title { font-weight:700; }
header a { color:#4da3ff; text-decoration:none; }
#wrap { flex:1; display:flex; min-height:0; }
#side { width:330px; overflow-y:auto; background:var(--card); border-right:1px solid var(--line);
        padding:10px 12px 40px; flex-shrink:0; }
#map { flex:1; background:#0c0d10; }
#side h3 { margin:14px 0 4px; font-size:13px; color:var(--dim); text-transform:uppercase;
           letter-spacing:.4px; }
.grp { margin-bottom:4px; }
.grp > label { display:flex; gap:8px; align-items:center; font-weight:600; cursor:pointer;
               padding:4px 2px; }
.item { display:flex; gap:6px; align-items:flex-start; padding:2px 0 2px 6px; cursor:pointer;
        border-radius:5px; }
.item:hover { background:#262a31; }
.item input { margin-top:3px; }
.item .nm { font-size:12.5px; }
.item .alt { color:var(--dim); font-size:11px; }
.conf { display:inline-block; min-width:17px; text-align:center; border-radius:4px;
        font-size:10.5px; font-weight:700; padding:0 4px; margin-right:4px; color:#fff; }
.c5{background:#2e7d32}.c4{background:#e65100}.c3{background:#8d6e08}
.c2{background:#455a64}.cv{background:#1565c0}.cx{background:#5d4270}
.lyr { display:flex; gap:8px; align-items:center; padding:3px 2px; cursor:pointer; }
.popup { font:13px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif; max-width:340px; }
.popup b { font-size:13.5px; }
.popup .meta { color:#555; font-size:12px; margin:3px 0; }
.popup .src { color:#666; font-size:11px; font-family:ui-monospace,Menlo,monospace;
              word-break:break-all; margin:1px 0; user-select:all; }
.popup .dist { font-size:12px; margin:4px 0; }
.popup .thumbs { display:flex; gap:6px; margin-top:6px; }
.popup .thumbs img { height:92px; border-radius:5px; cursor:zoom-in; }
#lightbox { position:fixed; inset:0; background:#000d; display:none; z-index:2000;
            align-items:center; justify-content:center; cursor:zoom-out; }
#lightbox img { max-width:96vw; max-height:96vh; }
#lightbox.on { display:flex; }
.leaflet-container { font:12px/1.4 -apple-system,sans-serif; }
.camp-label { background:none; border:none; box-shadow:none; color:#fff; font-weight:700;
              font-size:11px; text-shadow:0 0 3px #000,0 0 3px #000; white-space:nowrap; }
.ctr-label { background:none; border:none; box-shadow:none; color:#ffd54f; font-weight:700;
             font-size:12px; text-align:center;
             text-shadow:0 0 3px #000,0 0 3px #000,0 0 4px #000; }
.note { color:var(--dim); font-size:11.5px; margin:6px 0; }
.legend { border:1px solid var(--line); border-radius:8px; padding:6px 10px; margin-bottom:8px; }
.legend summary { cursor:pointer; padding:2px 0; }
.lrow { display:flex; gap:8px; margin:7px 0; font-size:12px; line-height:1.4; color:#c7cdd6; }
.sw { flex:0 0 16px; height:16px; margin-top:2px; border-radius:3px; }
</style></head>
<body>
<header><nav><span class="title">Курумды — видеоанализ</span>
<a href="index.html">Кандидаты</a>
<a href="montages.html">Все монтажные листы</a>
<a href="map.html"><b>Карта</b></a>
</nav></header>
<div id="wrap">
<div id="side">
  <details class="legend" open><summary><b>Как читать карту</b></summary>
    <div class="lrow"><span class="sw" style="background:#00e5ff"></span>
      <b>Голубой пунктир</b> — плановый маршрут группы с карты штаба; кружки — лагеря.
      Группа спускалась от Camp2 к Camp1.</div>
    <div class="lrow"><span class="sw" style="border:2px solid #ff1744; background:#ff174422"></span>
      <b>Красная рамка</b> — зона интереса 5385–5485 м: участок маршрута сразу под Camp2,
      где за три дня съёмки сходятся находки (предметы с бороздами, «ткань», Г-образный
      предмет). Снята только издалека — предметы меньше ~1 м там неразличимы, нужен
      повторный облёт.</div>
    <div class="lrow"><span class="sw" style="background:#ff7043"></span>
      <b>Оранжевые пунктирные</b> — «линии спуска из зоны»: расчёт по рельефу, куда
      скатится упавшее из каждой точки зоны. Все они приводят к месту найденных вещей —
      срыв в зоне объясняет, откуда вещи взялись.</div>
    <div class="lrow"><span class="sw" style="background:#ffee00"></span>
      <b>Жёлтая линия</b> — «линия падения»: тот же расчёт, но от рюкзака дальше вниз.
      Тяжёлое (человека) уносит ниже лёгких вещей — линия показывает, куда именно.</div>
    <div class="lrow"><span class="sw" style="background:#ff3d00"></span>
      <b>Красный толстый отрезок</b> — часть линии падения ниже 4529 м:
      не осматривалась никем. Самый ценный неосмотренный участок.</div>
    <div class="lrow"><span class="sw" style="background:#ffaa0033; border:1px solid #ffaa00"></span>
      <b>Оранжевая заливка</b> — коридор поиска 400 м вокруг линии падения.</div>
    <div class="lrow"><span class="sw" style="background:#e040fb38; border:1px dashed #e040fb"></span>
      <b>Фиолетовый пунктирный полигон</b> — слепое пятно: склон между зоной интереса и
      вещами. Если сорвавшийся успел тормозить — он остановился здесь, выше вещей.
      Заливка — ячейки 30 м, куда ни разу не смотрел центр кадра ни одного пролёта.</div>
    <div class="lrow"><span class="sw" style="background:#ffd54f"></span>
      <b>Янтарные тонкие</b> — изолинии высоты через 50 м, подписи — на каждой линии,
      кратной 200 м.</div>
    <div class="lrow"><span class="sw" style="background:#1e88e5; border-radius:50%"></span>
      <b>Кружки</b> — синие: подтверждённые вещи; остальные цвета — кандидаты по
      уверенности (зелёный 5 … серый 2, фиолетовый — закрытые). В попапе: способ
      привязки координаты, расстояния до траектории падения и маршрута, кадры,
      полный путь исходника.</div>
    <p class="note">Обе «линии» — модель стока по рельефу (сетка 30 м), а не траектории:
    реальные предметы прыгают и уходят дальше. Это инструмент приоритизации осмотра,
    не точные координаты.</p>
  </details>
  <p class="note">Клик по имени точки — перелёт к ней. Координаты большинства кандидатов —
  GPS дрона (объект может быть в стороне до ~145 м).</p>
  <h3>Слои</h3><div id="layers"></div>
  <h3>Точки</h3><div id="items"></div>
</div>
<div id="map"></div>
</div>
<div id="lightbox"><img alt=""></div>
<script>
const D = __DATA__;

const map = L.map('map', {zoomControl:true});
map.fitBounds([[39.468,73.578],[39.503,73.606]]);
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:19, attribution:'Esri World Imagery'});
const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
  {maxZoom:17, attribution:'OpenTopoMap'});
sat.addTo(map);
L.control.layers({'Спутник (Esri)':sat,'Топокарта':topo}, null, {position:'topright'}).addTo(map);
L.control.scale({imperial:false}).addTo(map);

const CONF_COLOR = {5:'#2e7d32',4:'#e65100',3:'#c9a20b',2:'#607d8b',v:'#1e88e5',x:'#8e6bb0'};
const confCls = c => 'c'+c;

// --- изолинии ---
const contourLayer = L.layerGroup();
for (const c of D.contours) {
  const major = c.lvl % 200 === 0;
  for (const line of c.lines) {
    L.polyline(line, {color:'#ffd54f', weight:major?1.8:0.8,
                      opacity:major?0.95:0.55, interactive:false}).addTo(contourLayer);
    if (major && line.length > 8) {
      // подпись высоты на каждой достаточно длинной линии уровня
      const p = line[Math.floor(line.length/2)];
      L.marker(p, {icon:L.divIcon({className:'ctr-label', html:c.lvl,
                                   iconSize:[44,16], iconAnchor:[22,8]}),
                   interactive:false}).addTo(contourLayer);
    }
  }
}

// --- маршрут и лагеря ---
const routeLayer = L.layerGroup();
L.polyline(D.route, {color:'#00e5ff', weight:2.5, opacity:0.9, dashArray:'6 4'})
  .bindPopup('Плановый маршрут группы (карта штаба, docs/marshrut/)').addTo(routeLayer);
for (const c of D.camps) {
  L.circleMarker([c.lat,c.lon], {radius:5, color:'#00e5ff', fillColor:'#003c8f', fillOpacity:1, weight:2})
    .bindPopup(`<b>${c.name}</b>${c.alt?(' — '+c.alt+' м'):''}`).addTo(routeLayer);
  L.marker([c.lat,c.lon], {icon:L.divIcon({className:'camp-label', html:c.name, iconAnchor:[-8,6]}),
           interactive:false}).addTo(routeLayer);
}

// --- линия падения, приоритет, коридор ---
const fallLayer = L.layerGroup();
L.polygon(D.corridor, {color:'#ffaa00', weight:1, fillColor:'#ffaa00', fillOpacity:0.10})
  .bindPopup('Поисковый коридор 400 м (search_vectors.kml)').addTo(fallLayer);
L.polyline(D.fall, {color:'#ffee00', weight:3, opacity:0.9})
  .bindPopup('<b>Линия падения</b>: расчёт по рельефу, куда скатывается упавшее от рюкзака дальше вниз (4685 → 4150 м). Тяжёлое уносит ниже лёгких вещей. Модель, не траектория (search_vectors.kml).').addTo(fallLayer);
L.polyline(D.prio, {color:'#ff3d00', weight:4, opacity:0.85})
  .bindPopup('<b>ПРИОРИТЕТ</b>: часть линии падения ниже 4529 м — не осматривалась никем. Если человека унесло дальше вещей, он на этом отрезке.').addTo(fallLayer);

// --- линии спуска из зоны ---
const descLayer = L.layerGroup();
for (const d of D.descents)
  L.polyline(d.line, {color:'#ff7043', weight:2, opacity:0.85, dashArray:'2 5'})
    .bindPopup('<b>Линия спуска</b> '+d.label+': расчёт по рельефу, куда скатится упавшее из этой точки зоны интереса. Линии из всей зоны сходятся к месту найденных вещей — срыв в зоне объясняет находки. Модель, не траектория.').addTo(descLayer);

// --- зоны ---
const zonesLayer = L.layerGroup();
L.rectangle(D.zone, {color:'#ff1744', weight:2, fillOpacity:0.12})
  .bindPopup('<b>Зона интереса 5385–5485 м</b><br>5 независимых наблюдений трёх дней; на нитке маршрута сразу под Camp2. Систематически не осмотрена (coverage-gsd).')
  .addTo(zonesLayer);
L.rectangle(D.impact, {color:'#b0bec5', weight:1.5, fillOpacity:0.10, dashArray:'4 4'})
  .bindPopup('Зона ударных отметин ~4950–4990 м (воронка, предметы в снегу)').addTo(zonesLayer);

// --- слепое пятно склона между зоной и вещами ---
const slopeLayer = L.layerGroup();
if (D.slope) {
  const cell = 30 / 111132;  // ячейка сетки, градусы широты
  const kLon = 1 / Math.cos(39.48 * Math.PI / 180);
  const popup = `<b>Слепое пятно склона</b> (расчёт 14.08)<br>
    Склон между зоной интереса и вещами: если сорвавшийся успел тормозить
    (зарубился, зацепился), он остановился здесь, выше вещей.<br>
    ${Math.round(D.slope.share*100)}% площади ни разу не попало в центр кадра
    ни одного пролёта (допуск 75 м). Фиолетовые квадраты — неосмотренные ячейки 30 м.`;
  L.polygon(D.slope.poly, {color:'#e040fb', weight:2, fillOpacity:0.04, dashArray:'6 3'})
    .bindPopup(popup).addTo(slopeLayer);
  for (const [la, lo] of D.slope.blind)
    L.rectangle([[la - cell/2, lo - cell/2*kLon], [la + cell/2, lo + cell/2*kLon]],
                {color:'#e040fb', weight:0, fillColor:'#e040fb', fillOpacity:0.22,
                 interactive:false}).addTo(slopeLayer);
}

// --- точки ---
function popupHtml(p) {
  const conf = p.conf==='v' ? 'вещь' : (p.conf==='x' ? 'закрыт' : 'ув. '+p.conf);
  const thumbs = p.imgs.map(i=>`<img src="/${i}" loading="lazy">`).join('');
  const srcs = (p.srcs||[]).map(s=>`<div class="src">${s}</div>`).join('');
  return `<div class="popup"><b>${p.name}</b> <span class="conf ${confCls(p.conf)}">${conf}</span>
  <div class="meta">${p.video} ${p.tc||''} · ${p.alt} м<br>Привязка: ${p.coord}<br>
  <b>Вердикт:</b> ${p.who}</div>
  ${srcs}
  <div>${p.desc}</div>
  <div class="dist">до траектории падения <b>${p.dTraj} м</b> (линия падения ${p.dFall} м, линии спуска из зоны ${p.dDesc} м) · до маршрута <b>${p.dRoute} м</b></div>
  <div class="thumbs">${thumbs}</div></div>`;
}

const groupLayers = {}, itemsByGroup = {};
for (const g of D.groups) { groupLayers[g.id] = L.layerGroup(); itemsByGroup[g.id] = []; }
D.points.forEach((p, i) => {
  const color = CONF_COLOR[p.conf] || '#999';
  const m = L.circleMarker([p.lat, p.lon],
    {radius: p.conf==='v'?7:6, color:'#111', weight:1.2, fillColor:color, fillOpacity:0.95});
  m.bindPopup(popupHtml(p), {maxWidth:360});
  if (p.unc) {
    // interactive:false — иначе круг перехватывает клики по точкам под ним
    p._circle = L.circle([p.lat,p.lon], {radius:p.unc, color:color, weight:1, fillOpacity:0.07,
                                         dashArray:'3 4', interactive:false});
    p._circle.addTo(groupLayers[p.g]);
  }
  m.addTo(groupLayers[p.g]);
  p._marker = m; p._idx = i;
  itemsByGroup[p.g].push(p);
});

// --- панель слоёв ---
const overlays = [
  ['Изолинии (50 м)', contourLayer, true],
  ['Маршрут и лагеря', routeLayer, true],
  ['Линия падения и коридор', fallLayer, true],
  ['Линии спуска из зоны', descLayer, true],
  ['Зоны (интереса / отметин)', zonesLayer, true],
  ['Слепое пятно склона', slopeLayer, true],
];
const layersDiv = document.getElementById('layers');
for (const [title, layer, on] of overlays) {
  if (on) layer.addTo(map);
  const el = document.createElement('label');
  el.className = 'lyr';
  el.innerHTML = `<input type="checkbox" ${on?'checked':''}> ${title}`;
  el.querySelector('input').onchange = e =>
    e.target.checked ? layer.addTo(map) : map.removeLayer(layer);
  layersDiv.appendChild(el);
}

// --- панель точек ---
const itemsDiv = document.getElementById('items');
for (const g of D.groups) {
  if (g.on) groupLayers[g.id].addTo(map);
  const box = document.createElement('div');
  box.className = 'grp';
  const head = document.createElement('label');
  head.innerHTML = `<input type="checkbox" ${g.on?'checked':''}> ${g.title}
                    <span class="alt">(${itemsByGroup[g.id].length})</span>`;
  const groupCb = head.querySelector('input');
  box.appendChild(head);
  const list = document.createElement('div');
  box.appendChild(list);
  for (const p of itemsByGroup[g.id]) {
    const row = document.createElement('div');
    row.className = 'item';
    const conf = p.conf==='v' ? 'В' : (p.conf==='x' ? '✕' : p.conf);
    row.innerHTML = `<input type="checkbox" checked>
      <span class="nm"><span class="conf ${confCls(p.conf)}">${conf}</span>${p.name}
      <span class="alt">${p.alt} м · траектория ${p.dTraj} м</span></span>`;
    const cb = row.querySelector('input');
    cb.onchange = () => {
      const tgt = groupLayers[p.g];
      if (cb.checked) { p._marker.addTo(tgt); if (p._circle) p._circle.addTo(tgt); }
      else { tgt.removeLayer(p._marker); if (p._circle) tgt.removeLayer(p._circle); }
    };
    row.querySelector('.nm').onclick = () => {
      if (!map.hasLayer(groupLayers[p.g])) { groupCb.checked = true; groupLayers[p.g].addTo(map); }
      if (!cb.checked) { cb.checked = true; cb.onchange(); }
      map.flyTo([p.lat, p.lon], Math.max(map.getZoom(), 16));
      p._marker.openPopup();
    };
    list.appendChild(row);
  }
  groupCb.onchange = () =>
    groupCb.checked ? groupLayers[g.id].addTo(map) : map.removeLayer(groupLayers[g.id]);
  itemsDiv.appendChild(box);
}

// --- лайтбокс ---
document.addEventListener('click', e => {
  const t = e.target;
  if (t.tagName === 'IMG' && t.closest('.popup')) {
    const lb = document.getElementById('lightbox');
    lb.querySelector('img').src = t.src;
    lb.classList.add('on');
  } else if (t.closest('#lightbox')) {
    document.getElementById('lightbox').classList.remove('on');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('on');
});
</script></body></html>"""


if __name__ == "__main__":
    build()
