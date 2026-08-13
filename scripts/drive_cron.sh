#!/bin/bash
# Ежечасный повтор докачки Drive-зеркала (квоты Google сбрасываются со временем):
#   0 * * * * bash /Users/d.razumovskiy/work/alp-finder/scripts/drive_cron.sh
# Скрипт download_drive.sh идемпотентен; guard от параллельного запуска.

cd "$(dirname "$0")/.." || exit 1
pgrep -f download_drive.sh >/dev/null && exit 0
bash scripts/download_drive.sh >> download.log 2>&1
