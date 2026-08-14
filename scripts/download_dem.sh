#!/usr/bin/env bash
# Скачивание модели высот Copernicus GLO-30 для района поисков (analysis/geoindex.py).
# Данные открытые, ключей и регистрации не требуют: зеркало AWS Open Data.
# Тайл в репозиторий не кладётся (data/ в .gitignore), каждый качает сам: 37 МБ.
#
# Тайл N39/E073 покрывает 39-40 N, 73-74 E - весь массив Курумды с запасом.
# Другой район: DEM_TILES="Copernicus_DSM_COG_10_N39_00_E074_00_DEM ..." bash scripts/download_dem.sh
set -u
cd "$(dirname "$0")/.."

BASE="https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
DEM_TILES="${DEM_TILES:-Copernicus_DSM_COG_10_N39_00_E073_00_DEM}"
OUT="data/dem"
FAILED=0

mkdir -p "$OUT"
for tile in $DEM_TILES; do
  dest="$OUT/$tile.tif"
  if [ -s "$dest" ]; then
    echo "НА МЕСТЕ $dest"
    continue
  fi
  echo "качаю $tile ..."
  if curl -sSL --fail --retry 3 --connect-timeout 30 \
       "$BASE/$tile/$tile.tif" -o "$dest.part"; then
    # COG начинается с сигнатуры TIFF (II* или MM*). Иначе прилетела страница
    # ошибки, а не растр, и молча положить её рядом значило бы сломать
    # геопривязку с невнятной ошибкой в rasterio.
    if head -c 2 "$dest.part" | grep -qE '^(II|MM)'; then
      mv "$dest.part" "$dest"
      echo "OK $dest ($(wc -c < "$dest") байт)"
    else
      echo "ОШИБКА $tile: скачан не TIFF"
      rm -f "$dest.part"
      FAILED=$((FAILED + 1))
    fi
  else
    echo "ОШИБКА $tile: не скачался"
    rm -f "$dest.part"
    FAILED=$((FAILED + 1))
  fi
done

echo "ИТОГ: ошибок $FAILED"
[ "$FAILED" -eq 0 ]
