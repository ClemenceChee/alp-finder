#!/bin/bash
# Одна итерация докачки Telegram-группы; вызывается кроном раз в минуту:
#   * * * * * bash /Users/d.razumovskiy/work/alp-finder/scripts/tg_cron.sh
# Новые сообщения -> logs/tg-updates.log, служебный вывод и ошибки -> logs/tg-watch.log.

cd "$(dirname "$0")/.." || exit 1
. scripts/.tg_env
mkdir -p logs
/usr/bin/python3 scripts/tg_export.py --quiet >> logs/tg-watch.log 2>&1
