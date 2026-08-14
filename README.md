# alp-finder

Поисково-аналитический проект: помощь штабу поисковой операции на пике Курумды
(Кыргызстан, август 2026) в отсмотре съёмок с дрона и вертолёта. Три альпиниста
пропали при спуске после восхождения 2 августа. Задача — алгоритмическая
предобработка десятков часов видео и выявление подозрительных кадров (люди,
снаряжение, следы, палатки, верёвки) с передачей координат штабу.

Точка входа для агентов и людей: `AGENTS.md`. Контекст операции: `docs/operation-context.md`.
Реестр находок с координатами и KML: `docs/nakhodki/`.
Независимый анализ отдельной группы (пробел покрытия, ловушка GPS, кандидаты): `docs/nezavisimyy-analiz/`.
Что уже исключено спутниковым анализом и куда смотреть дальше: `docs/avalanche-assessment.md`.

## Что лежит в репозитории, а что нет

В репо: документация, скрипты, реестр находок с кадрами и картой.
НЕ в репо (создаётся локально скриптами, см. «Развёртывание»):

- `data/drive/` — зеркало видео с Google Drive штаба (~40 ГБ);
- `data/telegram/` и `docs/telegram/` — выгрузка волонтёрской Telegram-группы
  (переписку не публикуем из уважения к участникам — выгружается самостоятельно);
- `data/dem/N39E073.tif` — тайл модели рельефа Copernicus GLO-30 (38 МБ);
- производные артефакты детектора (`analysis/scans/*/crops`, монтажи,
  `analysis/fullframe/`) — регенерируются скриптами из видео;
- секреты `scripts/.tg_env`, `scripts/.tg_session*`.

Готовая публичная копия просмотрщика (без развёртывания чего-либо):
**https://alp-finder.pages.dev** — кандидаты, монтажные листы, интерактивная
карта (`/map`). Из России и Беларуси `*.pages.dev` заблокирован провайдерами —
там открывайте зеркало: **https://darazumovskiy.github.io/alp-finder/**
(карта — `/alp-finder/map`). Локальное развёртывание нужно только для работы
с исходными видео и перегенерации анализа.

## Развёртывание

### 1. Видео с Google Drive

```bash
bash scripts/download_drive.sh
```

Качает по `scripts/manifest.tsv` (путь, drive file id, размер), идемпотентно.
При «quota exceeded» — повторить позже (или поставить в cron: `scripts/drive_cron.sh`).
Опись содержимого Drive: `docs/drive-inventory.md`.

### 2. Выгрузка Telegram-группы волонтёров

Группа «Анализ видео с дронам»: https://t.me/+e-P0uc1EehozOTg0 (вступление по ссылке).

```bash
pip install telethon
# на https://my.telegram.org создать приложение, получить api_id и api_hash
export TG_API_ID=... TG_API_HASH=...
python3 scripts/tg_export.py --login   # одноразовый интерактивный вход
python3 scripts/tg_export.py           # полная выгрузка: docs/telegram/ + data/telegram/
```

Темы группы становятся поддиректориями `docs/telegram/<тема>/messages.md`, медиа
скачиваются в `data/telegram/<тема>/`. Повторные запуски дописывают только новое;
для докачки раз в минуту — `scripts/tg_cron.sh` в crontab. Файл сессии
`scripts/.tg_session` — это вход в вашу учётку Telegram, не публикуйте его.

### 3. Окружение анализа и модель рельефа (DEM)

```bash
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r analysis/requirements.txt

# тайл рельефа Copernicus GLO-30 (нужен geoproject.py, coverage_gsd.py, карте):
mkdir -p data/dem
curl -o data/dem/N39E073.tif \
  https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_E073_00_DEM/Copernicus_DSM_COG_10_N39_00_E073_00_DEM.tif
```

### 4. Просмотрщик (кандидаты, монтажи, карта)

```bash
analysis/.venv/bin/python analysis/viewer/build_viewer.py   # index.html, montages.html
analysis/.venv/bin/python analysis/viewer/build_map.py      # map.html (нужен DEM)
python3 -m http.server 8077 -d .    # из корня репозитория
# открыть http://localhost:8077/analysis/viewer/index.html
```

Превью в карточках ссылаются на кадры в репозитории и на `data/drive/` —
последние появятся после шага 1. Деплой публичной копии — `AGENTS.md`
(раздел «Структура проекта», `analysis/viewer/`).

### 5. Геопривязка видео без SRT

У части DJI-видео нет сайдкар-SRT с телеметрией. Извлечение GPS-трека из
встроенного потока (DJI M30T, protobuf):

```bash
python3 scripts/dji_meta_gps.py data/drive/<путь к видео>.MP4
# рядом появится <видео>.MP4.gps.tsv: время, широта, долгота, высота

python3 scripts/gps_tsv_to_srt.py data/drive/<путь к видео>.MP4
# из tsv соберёт стандартный DJI-SRT — открывается в VLC вместе с видео
```

## Правила работы

- Сначала инвентаризация и документирование, потом обработка.
- Все выводы фиксируются в `docs/` — контекст проекта живёт в файлах, не в чате.
- Кандидаты-находки: файл, таймкод, координаты, скриншот, степень уверенности.
- **Любая находка перепроверяется перед передачей штабу**: ложные срабатывания
  тратят ресурс спасателей, пропущенные кадры — хуже.
