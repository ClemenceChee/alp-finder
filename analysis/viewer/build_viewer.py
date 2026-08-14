#!/usr/bin/env python3
"""Генератор визуального просмотрщика: index.html (кандидаты) + montages.html (все листы).

Пути к картинкам — абсолютные от корня репозитория, поэтому сервер надо
поднимать из корня: python3 -m http.server 8077 -d /Users/d.razumovskiy/work/alp-finder
и открывать http://localhost:8077/analysis/viewer/index.html

Перезапуск после появления новых сканов (scans-low, новые видео) пересобирает
страницы: montages.html строится обходом файловой системы, index.html — из
списка CANDIDATES ниже (курируется вручную, источник — analysis/review/*.md).
"""

from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# (заголовок секции, пояснение, [карточки])
# Карточка: dict(video, tc, desc, conf, imgs=[(подпись, путь от корня)])
CANDIDATES = [
    (
        "Зона интереса: склон 5385–5485 м",
        "Кандидаты четырёх независимых пролётов трёх дней сходятся в одну полосу склона "
        "(39.4768–39.4777 N, 73.5920–73.5923 E). Статус: приоритетная перепроверка дроном "
        "с близкого расстояния. Координаты — GPS дрона, не объекта.",
        [
            dict(video="DJI_20260811204542_0001_Z", tc="1:21–1:25", conf=3,
                 desc="Пунктирная дорожка тёмных отметин строго вниз по линии падения на чистом снегу (~5483 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260811204542_0001_Z/crops/t0058_01m25s.jpg"),
                       ("полный кадр", "analysis/pilot/check_204542_t58.jpg")]),
            dict(video="DJI_20260811202621_0002_Z", tc="0:45", conf=3,
                 desc="Тёмный предмет на снегу с бороздой-шлейфом вниз по склону (~5385 м, туман)",
                 imgs=[("кроп", "analysis/scans/DJI_20260811202621_0002_Z/crops/t0020_00m45s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z", tc="2:19–2:22", conf=3,
                 desc="Диагональная цепочка отметин + группа предметов с бороздами (~5395 м)",
                 imgs=[("кроп 2:19", "analysis/scans/DJI_20260811202621_0002_Z/crops/t0015_02m19s.jpg"),
                       ("кроп 2:22", "analysis/scans/DJI_20260811202621_0002_Z/crops/t0026_02m22s.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="5:58", conf=3,
                 desc="Охристо-бурый плоский объект на чистом снегу, фактура смятой ткани/брезента (~5404 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260812143557_0001_Z/crops/t0016_05m58s.jpg"),
                       ("полный кадр", "analysis/pilot/check_143557_t16.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="8:20", conf=2,
                 desc="Узкий сегментированный предмет в снегу — форма ледоруба/палки, может быть камень (~5466 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260812143557_0001_Z/crops/t0079_08m20s.jpg"),
                       ("полный кадр", "analysis/pilot/check_143557_t79.jpg")]),
            dict(video="DJI_20260813152839_0005_Z", tc="1:42–1:44", conf=4,
                 desc="Изолированный тёмный предмет на чистом снегу: прямое древко с перпендикулярной "
                      "головкой — силуэт ледоруба; камней рядом нет (~5443 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813152839_0005_Z/crops/t0000_01m44s.jpg"),
                       ("полный кадр", "analysis/pilot/check_152839_t0.jpg"),
                       ("полный кадр 2", "analysis/pilot/check_152839_t103.jpg")]),
        ],
    ),
    (
        "Зона интереса, перескан тумана (CLAHE, порог ×2.7): поле предметов с бороздами",
        "Перескан DJI_20260811202621_0002_Z с контрастированием и ослабленным порогом "
        "(analysis/review/rescan-low.md). На участке 5385 м в окне 0:29–1:00 — 4–6 предметов, "
        "у каждого своя борозда-шлейф; на 5395 м — борозды без предметов, которых старый прогон "
        "не видел. Участок в ~2 м от нитки планового маршрута группы (docs/marshrut/README.md). "
        "Масштаб по геопроекции (оценка вилкой, не одиночной трассировкой — устойчива к ошибке DEM): "
        "дрон висел в 15–60 м от склона, "
        "предметы ~0,2–1,3 м по длинной стороне (t1 до 2,5 м на дальней границе), борозды — метры. "
        "Кадры контрастированы CLAHE — для штаба вырезать исходные без обработки.",
        [
            dict(video="DJI_20260811202621_0002_Z (low)", tc="0:44 (0:44–0:59)", conf=3,
                 desc="Старый t20: компактный тёмный предмет с сужающимся хвостом-бороздой; "
                      "борозда тянется десятки метров вверх по склону (5385 м)",
                 imgs=[("кроп", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0025_00m44s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z (low)", tc="0:31", conf=3,
                 desc="Новое: второй предмет — двухдольный, светлое включение между долями, "
                      "слабая борозда позади (5385 м)",
                 imgs=[("кроп", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0001_00m31s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z (low)", tc="0:50", conf=3,
                 desc="Новое: предмет и ниже — отдельная длинная сужающаяся борозда, "
                      "след скольжения ещё одного тела (5385 м)",
                 imgs=[("кроп", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0056_00m50s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z (low)", tc="0:31–0:38", conf=3,
                 desc="Новое: ещё 3 мелких предмета, у каждого свой диагональный шлейф (5385 м)",
                 imgs=[("кроп 0:36", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0087_00m36s.jpg"),
                       ("кроп 0:31", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0107_00m31s.jpg"),
                       ("кроп 0:38", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0155_00m38s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z (low)", tc="1:53–1:58", conf=2,
                 desc="Новое: сужающиеся борозды без видимого предмета сквозь плотный туман (5395 м)",
                 imgs=[("кроп", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0041_01m58s.jpg"),
                       ("кроп 1:56", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0116_01m56s.jpg")]),
            dict(video="DJI_20260811202621_0002_Z (low)", tc="2:34", conf=2,
                 desc="Округлая вмятина с тёмным ободом — воронка от упавшего тела или протаявший камень (5395 м)",
                 imgs=[("кроп", "analysis/scans-low/DJI_20260811202621_0002_Z/crops/t0139_02m34s.jpg")]),
        ],
    ),
    (
        "Полнокадровый проход: следы и борозды (analysis/review/fullframe-tracks.md)",
        "Цветовой детектор рельефные следы не ловит — 7 приоритетных видео просмотрены по полным "
        "кадрам с CLAHE (шаг 4 с, 100% покрытие). Калибровка: известный след срыва найден.",
        [
            dict(video="DJI_20260812135426_0001_Z", tc="0:41", conf=None,
                 desc="Калибровочный след срыва (известный): узкая борозда строго вниз по склону, "
                      "без цветового отличия, читается только рельефной тенью (~5097 м). "
                      "Образец искомой сигнатуры",
                 imgs=[("кадр CLAHE", "analysis/fullframe/DJI_20260812135426_0001_Z/enh/f0021.jpg"),
                       ("увеличение", "analysis/fullframe/_calib/t41_track_zoom.jpg")]),
            dict(video="DJI_20260813183727_0015_Z", tc="3:00–3:08", conf=None,
                 desc="Дальний ракурс оранжевого спальника (исправлено 14.08: ранняя геопроекция "
                      "«малый фрагмент в 35 м, снят с 11 м» отозвана — ложное пересечение луча; "
                      "подлёт 3:02→3:28 на 59 м показывает тот же объект, что крупно на 3:26–4:00)",
                 imgs=[("кадр", "analysis/fullframe/DJI_20260813183727_0015_Z/f0091.jpg"),
                       ("зум 3:02", "analysis/pilot/check_183727_frag302_zoom.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="3:28", conf=2,
                 desc="Цепочка вмятин вниз по склону с шагом 1–2 м (5366 м, нижняя граница зоны "
                      "интереса) — пробоины от камней или заметённые следы",
                 imgs=[("кадр CLAHE", "analysis/fullframe/DJI_20260812143557_0001_Z/enh/f0105.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="4:32", conf=2,
                 desc="Вторая цепочка из 3–4 вытянутых вмятин с равным шагом, ~14 м южнее первой, "
                      "та же экспозиция склона (5376 м)",
                 imgs=[("кадр CLAHE", "analysis/fullframe/DJI_20260812143557_0001_Z/enh/f0137.jpg")]),
            dict(video="DJI_20260811204542_0001_Z", tc="4:16", conf=2,
                 desc="Локальный провал снежного моста у бергшрунда: тёмная полость, вывороченные "
                      "блоки наста, следов подхода не видно (5088 м)",
                 imgs=[("кадр CLAHE", "analysis/fullframe/DJI_20260811204542_0001_Z/enh/f0129.jpg")]),
        ],
    ),
    (
        "Оранжевый спальник Николая — атрибуция скрина «Вещи #1721»",
        "Крупный оранжевый предмет один (исправлено 14.08: «отдельный малый фрагмент в 35 м» отозван — "
        "ложное пересечение луча на кадре 3:02, подлёт дрона показывает тот же объект). "
        "Позиция ~39.48325–39.48331, 73.58509–73.58518, ~4558 м (в 10–15 м от стакана), как точку "
        "не выдавать. Консенсус TG — спальник Николая (Скриншоты #758, #820); документ CC называет "
        "«курткой». Рядом в кадре 1:28 — мелкий оранжевый фрагмент ~0,2 м.",
        [
            dict(video="DJI_20260813163855_0001_Z", tc="1:22–2:51, крупно 1:41–1:52", conf=None,
                 desc="Оранжевая ткань/чехол с белыми лентами; дрон: 39.48431, 73.58402, 4673 м",
                 imgs=[("кроп", "analysis/pilot/scan-163855/crops/t0000_01m49s.jpg")]),
            dict(video="DJI_20260813183004_0001_Z", tc="0:02, 2:07–2:24", conf=None,
                 desc="Тот же (?) оранжевый предмет; дрон: 39.48340, 73.58514, 4552 м",
                 imgs=[("кроп", "analysis/pilot/scan-183004/crops/t0014_02m24s.jpg")]),
            dict(video="DJI_20260813183727_0015_Z", tc="3:26–4:00", conf=None,
                 desc="Оранжевый предмет крупным планом (топ-1 трек видео); дрон: 39.48335, 73.58502, 4552 м",
                 imgs=[("кроп", "analysis/pilot/scan-183727/crops/t0000_03m31s.jpg")]),
        ],
    ),
    (
        "Подтверждённые вещи группы (реестр штаба, 4529–4663 м)",
        "Все на линии падения ниже лагеря 1. Участок ниже 4529 м не осматривался — "
        "тяжёлые предметы уносит дальше лёгких.",
        [
            dict(video="helicopter/C0049.MP4", tc="0:15–0:21", conf=5,
                 desc="Синий рюкзак Николая; 39.482656, 73.586792, 4663 м, склон 46°",
                 imgs=[("кадр", "docs/nakhodki/frames/C0049_00m18s.png")]),
            dict(video="DJI_20260813183140_0012_Z.JPG", tc="фото", conf=5,
                 desc="Палка Komperdell №1 (пробковая ручка); 39.483176, 73.585463, 4529 м",
                 imgs=[("фото", "data/drive/2026-08-13/drone-part3/DJI_20260813183140_0012_Z.JPG"),
                       ("фото 2", "data/drive/2026-08-13/drone-part3/DJI_20260813183028_0002_Z.JPG")]),
            dict(video="DJI_20260813183101_0008_Z.JPG", tc="фото", conf=5,
                 desc="Палка из второй пары; тот же район",
                 imgs=[("фото", "data/drive/2026-08-13/drone-part3/DJI_20260813183101_0008_Z.JPG")]),
            dict(video="DJI_20260813183727_0015_Z", tc="3:32", conf=5,
                 desc="Бирюзовая крышка / складная миска; 39.483144, 73.585443, 4531 м",
                 imgs=[("кадр", "docs/nakhodki/frames/DJI_20260813183727_0015_Z_03m32s.jpg")]),
            dict(video="DJI_20260813183004_0001_Z", tc="3:40", conf=5,
                 desc="Белый одноразовый стакан с красной маркировкой; 39.483403, 73.585137, 4552 м",
                 imgs=[("кадр", "docs/nakhodki/frames/DJI_20260813183004_0001_Z_03m40s.jpg")]),
        ],
    ),
    (
        "Верёвки и перила (вероятно старые — для карты штаба)",
        "",
        [
            dict(video="DJI_20260812135747_0002_Z", tc="0:00–2:18", conf=5,
                 desc="Сплошная линия старых перил вдоль стены (~5146 м): верёвка через скальный рог, "
                      "узлы и петли станций, куски в трещинах под снегом",
                 imgs=[("якорь 0:16", "analysis/scans/DJI_20260812135747_0002_Z/crops/t0000_00m16s.jpg"),
                       ("станция 1:13", "analysis/scans/DJI_20260812135747_0002_Z/crops/t0008_01m13s.jpg"),
                       ("в трещине 1:46", "analysis/scans/DJI_20260812135747_0002_Z/crops/t0012_01m46s.jpg")]),
            dict(video="DJI_20260812135747_0002_Z", tc="0:30–0:34", conf=5,
                 desc="Красная верёвка (подтверждённая находка, дальний план) + сдвоенная светлая с белым фрагментом",
                 imgs=[("кроп", "analysis/scans/DJI_20260812135747_0002_Z/crops/t0005_00m33s.jpg")]),
            dict(video="DJI_20260812140210_0004_Z", tc="0:29–1:09", conf=4,
                 desc="Тонкие прямые линии на скальной стене в тумане — верёвки/перила выше по стене (~5211 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260812140210_0004_Z/crops/t0014_00m29s.jpg")]),
            dict(video="DJI_20260812133415_0006_Z", tc="0:44", conf=4,
                 desc="Тонкая светлая линия поперёк тёмной скалы + оранжевая точка рядом (~4985 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260812133415_0006_Z/crops/t0073_00m44s.jpg")]),
        ],
    ),
    (
        "Глубокие листы 03–09 приоритетных видео (analysis/review/deep-sheets.md)",
        "Второй заход: раньше отсматривались только топ-3 листа. Здесь кандидаты уверенности 3 "
        "из листов 03–09 одиннадцати приоритетных сканов.",
        [
            dict(video="scan-183534 (район находок)", tc="0:54–0:57", conf=3,
                 desc="Плоский бледно-зелёный прямоугольник с прямыми кромками и мелкозернистой "
                      "текстурой на бурой осыпи — похоже на фрагмент пенки/каремата; в ~40 м от "
                      "кластера палка/крышка (4556 м)",
                 imgs=[("кроп 0:54", "analysis/pilot/scan-183534/crops/t0237_00m54s.jpg"),
                       ("кроп 0:57", "analysis/pilot/scan-183534/crops/t0261_00m57s.jpg")]),
            dict(video="scan-183004 (район находок)", tc="0:56", conf=5,
                 desc="Палка крупным планом: на древке читается бренд «CAMP» с оранжевой отделкой. "
                      "В реестре палки описаны как Komperdell — бренд может помочь штабу опознать "
                      "владельца (проверено: надпись читается)",
                 imgs=[("кроп", "analysis/pilot/scan-183004/crops/t0334_00m56s.jpg")]),
            dict(video="scan-184253 (район находок)", tc="0:28–0:34", conf=5,
                 desc="Синий рюкзак Николая с дрона (перепроверено по полным кадрам: объект один, "
                      "лежит на той же кварцевой полосе, что в вертолётном кадре C0049). "
                      "Новый ракурс известной находки, не новая вещь",
                 imgs=[("кроп", "analysis/pilot/scan-184253/crops/t0148_00m31s.jpg"),
                       ("полный кадр", "analysis/pilot/check_184253_t148.jpg"),
                       ("увеличение", "analysis/pilot/check_184253_t148_crop30s.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="8:26", conf=3,
                 desc="Ещё один вытянутый тёмный предмет с бороздой-шлейфом в россыпи 8:20–8:26 "
                      "(верх зоны интереса, 5466 м); участок — на покадровый просмотр 8:15–8:30",
                 imgs=[("кроп", "analysis/scans/DJI_20260812143557_0001_Z/crops/t0292_08m26s.jpg")]),
            dict(video="DJI_20260812143557_0001_Z", tc="6:21", conf=3,
                 desc="Округлый серый многоугольный предмет изолированно на чистом снегу, ниже — "
                      "намёк на тонкую линию (стропа?) (5404 м, зона интереса)",
                 imgs=[("кроп", "analysis/scans/DJI_20260812143557_0001_Z/crops/t0201_06m21s.jpg")]),
            dict(video="DJI_20260813130531_0001_Z", tc="0:44", conf=3,
                 desc="Длинная прямая борозда вниз по лавинному выносу с тёмным предметом в верхней "
                      "части (5381 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813130531_0001_Z/crops/t0130_00m44s.jpg")]),
            dict(video="DJI_20260811204542_0001_Z", tc="4:18", conf=3,
                 desc="Изолированный тёмный гладкий овальный предмет на чистом снегу, отчётливая "
                      "тень, ореола протаивания нет (5088 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260811204542_0001_Z/crops/t0189_04m18s.jpg")]),
            dict(video="scan-163855 (район находок)", tc="1:23 и 2:41", conf=3,
                 desc="Бирюзовый кольцеобразный объект среди осыпи в двух точках полёта — один "
                      "предмет с двух ракурсов или два (4610–4673 м)",
                 imgs=[("кроп 1:23", "analysis/pilot/scan-163855/crops/t0257_01m23s.jpg"),
                       ("кроп 2:41", "analysis/pilot/scan-163855/crops/t0271_02m41s.jpg")]),
            dict(video="scan-163855 (район находок)", tc="2:37", conf=3,
                 desc="Ярко-оранжевый мелкий фрагмент на тёмной осыпи, рядом по кадру с бирюзовым "
                      "(4610 м)",
                 imgs=[("кроп", "analysis/pilot/scan-163855/crops/t0273_02m37s.jpg")]),
            dict(video="scan-163855 (район находок)", tc="4:15", conf=3,
                 desc="Тёмно-синий гладкий округлый объект в тени скальной полки — похоже на "
                      "ткань/вещь (4610 м)",
                 imgs=[("кроп", "analysis/pilot/scan-163855/crops/t0131_04m15s.jpg")]),
            dict(video="scan-183004 (район находок)", tc="2:24", conf=3,
                 desc="Ярко-оранжевый округлый предмет на тёмной осыпи у края кадра — возможно "
                      "известная крышка (кластер в ~30 м), возможно отдельный предмет (4552 м)",
                 imgs=[("кроп", "analysis/pilot/scan-183004/crops/t0298_02m24s.jpg")]),
        ],
    ),
    (
        "Остальные кандидаты уверенности 3",
        "Кандидаты уверенности 1–2 (~35 треков: свежие камни, следы, лунки) — в analysis/review/*.md.",
        [
            dict(video="DJI_20260811161324_0003_Z", tc="0:03–0:04", conf=3,
                 desc="Тёмный предмет, утопленный в снег, следы вокруг + пятно с 3 расходящимися "
                      "лучами — похоже на воронку удара с выбросом (~4992 м)",
                 imgs=[("воронка", "analysis/scans/DJI_20260811161324_0003_Z/crops/t0018_00m04s.jpg"),
                       ("предмет", "analysis/scans/DJI_20260811161324_0003_Z/crops/t0003_00m03s.jpg")]),
            dict(video="DJI_20260811161324_0003_Z", tc="3:14", conf=3,
                 desc="Одиночный тёмный каплевидный объект на ровном снегу + 2 мелких пятна рядом (~4952 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260811161324_0003_Z/crops/t0002_03m14s.jpg")]),
            dict(video="DJI_20260813095243_0001_Z", tc="0:05–0:21", conf=3,
                 desc="Светлое зеленовато-серое пятно на тёмном скальном выступе гребня (~5859 м); "
                      "совпадает с кандидатом TG «трепыхающаяся ткань»",
                 imgs=[("кроп", "analysis/scans/DJI_20260813095243_0001_Z/crops/t0008_00m15s.jpg")]),
            dict(video="DJI_20260813101405_0001_Z", tc="1:32", conf=3,
                 desc="Тёмный угловатый объект поверх чистого снега, светлое включение, отчётливая тень (~5879 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813101405_0001_Z/crops/t0057_01m32s.jpg")]),
            dict(video="DJI_20260813103527_0001_Z", tc="3:10–3:23", conf=3,
                 desc="Тёмный угловатый объект на снегу с бороздой позади, у высоты вершины (~6096 м); "
                      "рядом плоский вытянутый предмет (3:10)",
                 imgs=[("объект", "analysis/scans/DJI_20260813103527_0001_Z/crops/t0017_03m18s.jpg"),
                       ("предмет 3:10", "analysis/scans/DJI_20260813103527_0001_Z/crops/t0025_03m10s.jpg")]),
            dict(video="DJI_20260813130531_0001_Z", tc="2:38–2:47", conf=3,
                 desc="Х-образный тёмный объект с остроконечными выступами у кромки лавинного выноса (~5381 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813130531_0001_Z/crops/t0026_02m40s.jpg")]),
            dict(video="DJI_20260813133738_0001_Z", tc="6:26–6:58", conf=3,
                 desc="Кремовый гладкий округлый объект среди острой серой осыпи — окатанный камень "
                      "или предмет (каска/канистра?) (~5108 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813133738_0001_Z/crops/t0006_06m38s.jpg")]),
            dict(video="DJI_20260813152650_0001_Z", tc="0:00–0:05", conf=3,
                 desc="Тёмный угловатый объект с жёлтым включением у кромки бергшрунда (~5155 м)",
                 imgs=[("кроп", "analysis/scans/DJI_20260813152650_0001_Z/crops/t0001_00m02s.jpg"),
                       ("полный кадр", "analysis/pilot/check_152650_t1.jpg")]),
        ],
    ),
]

CSS = """
:root { --bg:#14161a; --card:#1d2026; --text:#e6e8ec; --dim:#9aa3af;
        --accent:#4da3ff; --line:#2b2f37; }
* { box-sizing: border-box; }
body { margin:0; background:#14161a; color:#e6e8ec;
       font:15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color:#4da3ff; text-decoration:none; }
a:hover { text-decoration:underline; }
header { position:sticky; top:0; z-index:5; background:#14161acc; backdrop-filter:blur(8px);
         border-bottom:1px solid #2b2f37; padding:10px 24px; }
header nav { display:flex; gap:18px; flex-wrap:wrap; align-items:baseline; }
header .title { font-weight:700; margin-right:8px; }
main { max-width:1500px; margin:0 auto; padding:16px 24px 80px; }
h2 { margin:36px 0 6px; font-size:20px; }
p.note { color:#9aa3af; margin:0 0 14px; max-width:900px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:14px; }
.card { background:#1d2026; border:1px solid #2b2f37; border-radius:10px; padding:12px; }
.card .head { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
.card .vid { font-family:ui-monospace, Menlo, monospace; font-size:12.5px; color:#e6e8ec; }
.card .tc { color:#4da3ff; font-size:13px; white-space:nowrap; }
.card .desc { color:#c7cdd6; font-size:13.5px; margin:6px 0 10px; }
.conf { display:inline-block; min-width:20px; text-align:center; border-radius:5px;
        font-size:12px; font-weight:700; padding:1px 6px; margin-left:6px; }
.c5 { background:#2e7d32; } .c4 { background:#e65100; } .c3 { background:#8d6e08; }
.c2 { background:#455a64; } .c1 { background:#37474f; } .cx { background:#6a1b9a; }
.thumbs { display:flex; gap:8px; flex-wrap:wrap; }
.thumbs figure { margin:0; }
.thumbs img { width:150px; height:150px; object-fit:cover; border-radius:6px;
              border:1px solid #2b2f37; display:block; cursor:zoom-in; }
.thumbs figcaption { font-size:11.5px; color:#9aa3af; text-align:center; margin-top:3px; }
details { border:1px solid #2b2f37; border-radius:10px; margin:10px 0; background:#1d2026; }
details summary { cursor:pointer; padding:10px 14px; font-family:ui-monospace, Menlo, monospace;
                  font-size:13px; }
details .sheets { display:flex; flex-wrap:wrap; gap:10px; padding:0 14px 14px; }
details .sheets figure { margin:0; }
details .sheets img { width:340px; border-radius:6px; border:1px solid #2b2f37; cursor:zoom-in; }
details .sheets figcaption { font-size:12px; color:#9aa3af; text-align:center; }
#lightbox { position:fixed; inset:0; background:#000d; display:none; z-index:50;
            align-items:center; justify-content:center; cursor:zoom-out; }
#lightbox img { max-width:96vw; max-height:96vh; }
#lightbox.on { display:flex; }
.small { color:#9aa3af; font-size:13px; }
.src { color:#7d8590; font-size:11px; font-family:ui-monospace, Menlo, monospace;
       margin-top:2px; word-break:break-all; }
summary .src { margin-left:8px; }
h3.datehead { margin:26px 0 8px; color:#c7cdd6; }
"""

JS = """
document.addEventListener('click', e => {
  const t = e.target;
  if (t.tagName === 'IMG' && t.closest('.thumbs, .sheets')) {
    const lb = document.getElementById('lightbox');
    lb.querySelector('img').src = t.dataset.full || t.src;
    lb.classList.add('on');
  } else if (t.closest('#lightbox')) {
    document.getElementById('lightbox').classList.remove('on');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('on');
});
"""


def esc(s):
    return html.escape(str(s), quote=True)


def source_index():
    """{относительный путь исходника} для всех видео и фото в data/drive."""
    files = []
    for pat in ("*.MP4", "*.mp4", "*.JPG", "*.jpg"):
        files += (ROOT / "data/drive").rglob(pat)
    return [p.relative_to(ROOT) for p in files]


SOURCES = source_index()


def source_path(video_label: str) -> str:
    """Путь исходника по подписи карточки/скана ('' если не нашли)."""
    label = re.sub(r"\s*\(.*\)$", "", video_label.strip())
    if "/" in label:                       # 'helicopter/C0049.MP4'
        label = label.rsplit("/", 1)[1]
    if label.startswith("scan-"):          # 'scan-183534' -> '183534', 'scan-C0049'
        label = label[5:]
    if label.startswith("heli-"):          # 'heli-C0001' -> 'C0001'
        label = label[5:]
    token = label.removesuffix(".MP4").removesuffix(".JPG")
    want_jpg = video_label.strip().endswith(".JPG")
    for p in SOURCES:
        if token in p.name and (p.suffix.upper() == ".JPG") == want_jpg:
            return str(p)
    return ""


def page(title, body):
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body>
<header><nav><span class="title">Курумды — видеоанализ</span>
<a href="index.html">Кандидаты</a>
<a href="montages.html">Все монтажные листы</a>
<a href="map.html">Карта</a>
</nav></header>
<main>{body}</main>
<div id="lightbox"><img alt=""></div>
<script>{JS}</script></body></html>"""


def conf_badge(conf):
    if conf is None:
        return '<span class="conf cx">?</span>'
    return f'<span class="conf c{conf}">{conf}</span>'


def card(c):
    thumbs = "".join(
        f'<figure><img loading="lazy" src="/{esc(p)}" data-full="/{esc(p)}" alt="{esc(lbl)}">'
        f"<figcaption>{esc(lbl)}</figcaption></figure>"
        for lbl, p in c["imgs"] if (ROOT / p).exists()
    )
    missing = [p for _, p in c["imgs"] if not (ROOT / p).exists()]
    warn = f'<div class="small">нет файла: {esc(", ".join(missing))}</div>' if missing else ""
    src = source_path(c["video"])
    src_html = f'<div class="src">{esc(src)}</div>' if src else ""
    return (
        '<div class="card">'
        f'<div class="head"><span class="vid">{esc(c["video"])}</span>'
        f'<span class="tc">{esc(c["tc"])}{conf_badge(c["conf"])}</span></div>'
        f"{src_html}"
        f'<div class="desc">{esc(c["desc"])}</div>'
        f'<div class="thumbs">{thumbs}</div>{warn}</div>'
    )


def build_index():
    parts = [
        "<h1>Кандидаты</h1>",
        '<p class="note">Клик по картинке — увеличение (Esc — закрыть). Цифра — уверенность '
        "по шкале штаба 1–5. Координаты и высоты — позиция дрона, не объекта: камера зумная. "
        "Источники: analysis/review/*.md, docs/video-analysis.md, docs/nakhodki/README.md.</p>",
    ]
    for title, note, cards in CANDIDATES:
        parts.append(f"<h2>{esc(title)}</h2>")
        if note:
            parts.append(f'<p class="note">{esc(note)}</p>')
        parts.append('<div class="grid">' + "".join(card(c) for c in cards) + "</div>")
    (OUT / "index.html").write_text(page("Кандидаты — Курумды", "\n".join(parts)), "utf-8")


def scan_dirs():
    """[(группа, имя, dir)] всех директорий с монтажами."""
    out = []
    for d in sorted((ROOT / "analysis/scans").iterdir()):
        if not (d / "montage_00.jpg").exists():
            continue
        m = re.match(r"DJI_(2026\d{4})", d.name)
        group = f"Дрон {m.group(1)[6:]}.{m.group(1)[4:6]}" if m else "Вертолёт"
        out.append((group, d.name, d))
    low = ROOT / "analysis/scans-low"
    if low.exists():
        for d in sorted(low.iterdir()):
            if (d / "montage_00.jpg").exists():
                out.append(("Перескан с ослабленными порогами", d.name, d))
    pilot = ROOT / "analysis/pilot"
    for d in sorted(pilot.glob("scan-*")):
        if (d / "montage_00.jpg").exists():
            out.append(("Район находок (пилот)", d.name, d))
    return out


def build_montages():
    groups = {}
    for group, name, d in scan_dirs():
        groups.setdefault(group, []).append((name, d))
    parts = [
        "<h1>Все монтажные листы</h1>",
        '<p class="note">Лист — сетка 6×6 кропов, отсортированных по убыванию аномальности '
        "(montage_00 — самые подозрительные треки видео). Жёлтая подпись в ячейке: номер трека "
        "и таймкод. Клик — увеличение.</p>",
    ]
    order = sorted(groups, key=lambda g: (g.startswith("Дрон"), g), reverse=True)
    order = sorted(order)  # стабильный алфавитный порядок: Вертолёт, Дрон ..., Перескан, Район
    for group in order:
        parts.append(f'<h3 class="datehead">{esc(group)}</h3>')
        for name, d in groups[group]:
            sheets = sorted(d.glob("montage_*.jpg"))
            rel = d.relative_to(ROOT)
            figs = "".join(
                f'<figure><img loading="lazy" src="/{esc(rel)}/{esc(s.name)}" '
                f'data-full="/{esc(rel)}/{esc(s.name)}" alt="{esc(s.name)}">'
                f"<figcaption>{esc(s.name)}</figcaption></figure>"
                for s in sheets
            )
            src = source_path(name)
            src_html = f' <span class="src">{esc(src)}</span>' if src else ""
            parts.append(
                f"<details><summary>{esc(name)} — {len(sheets)} лист(ов){src_html}</summary>"
                f'<div class="sheets">{figs}</div></details>'
            )
    (OUT / "montages.html").write_text(page("Монтажи — Курумды", "\n".join(parts)), "utf-8")


if __name__ == "__main__":
    build_index()
    build_montages()
    print(f"готово: {OUT/'index.html'}, {OUT/'montages.html'}")
