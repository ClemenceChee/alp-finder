#!/usr/bin/env bash
# Идемпотентная докачка файлов из публичного Google Drive по scripts/manifest.tsv.
# Формат манифеста: <локальный путь>\t<drive file id>\t<размер в байтах>
# Уже скачанные файлы с совпавшим размером пропускаются. Повторный запуск безопасен.
set -u
cd "$(dirname "$0")/.."

MANIFEST="scripts/manifest.tsv"
FAILED=0
TOTAL=0
DONE=0

while IFS=$'\t' read -r path file_id size; do
  [ -z "$path" ] && continue
  TOTAL=$((TOTAL + 1))
  if [ -f "$path" ] && [ "$(stat -f%z "$path")" = "$size" ]; then
    DONE=$((DONE + 1))
    continue
  fi
  mkdir -p "$(dirname "$path")"
  ok=""
  quota=""
  for attempt in 1 2 3; do
    curl -sSL --fail --retry 2 --connect-timeout 30 \
      "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
      -o "$path.part"
    actual=$(stat -f%z "$path.part" 2>/dev/null || echo 0)
    if [ "$actual" = "$size" ]; then
      mv "$path.part" "$path"
      ok=1
      break
    fi
    # Страница «Quota exceeded» вместо файла: повторять сейчас бессмысленно
    if head -c 100 "$path.part" 2>/dev/null | grep -q '<!DOCTYPE html>'; then
      quota=1
      rm -f "$path.part"
      break
    fi
    echo "RETRY ($attempt) $path: ожидали $size, получили $actual"
    rm -f "$path.part"
    sleep $((attempt * 20))
  done
  if [ -n "$ok" ]; then
    DONE=$((DONE + 1))
    echo "OK [$DONE/$TOTAL] $path"
  elif [ -n "$quota" ]; then
    FAILED=$((FAILED + 1))
    echo "QUOTA $path (id=$file_id)"
  else
    FAILED=$((FAILED + 1))
    echo "FAIL $path (id=$file_id)"
  fi
done < "$MANIFEST"

echo "ИТОГ: скачано/на месте $DONE из $TOTAL, ошибок: $FAILED"
[ "$FAILED" -eq 0 ]
