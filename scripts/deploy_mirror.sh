#!/usr/bin/env bash
# Публикация analysis/viewer/dist в ветку gh-pages — зеркало просмотрщика
# на GitHub Pages: https://darazumovskiy.github.io/alp-finder/
# Нужно потому, что *.pages.dev заблокирован провайдерами в РФ и РБ.
# Перед запуском собрать dist: python analysis/viewer/build_dist.py
#
# Пуш инкрементальный: клонируем gh-pages без содержимого файлов
# (--filter=blob:none), накладываем dist поверх пустого рабочего дерева и
# пушим только изменённые файлы — полный dist (~0.5 ГБ) одним паком
# соединение с GitHub не переживает.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/analysis/viewer/dist"
[ -f "$DIST/index.html" ] || {
  echo "нет $DIST/index.html — сначала python analysis/viewer/build_dist.py" >&2
  exit 1
}

URL="$(git -C "$ROOT" remote get-url origin)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone -q --filter=blob:none --no-checkout --depth 1 --branch gh-pages "$URL" "$TMP"
git -C "$TMP" read-tree HEAD
cp -R "$DIST/." "$TMP/"
# рабочее дерево = ровно dist, поэтому add -A даёт точное зеркало:
# файлы, исчезнувшие из dist, станут удалениями
git -C "$TMP" add -A
if git -C "$TMP" diff --cached --quiet; then
  echo "gh-pages уже актуальна, пуш не нужен"
  exit 0
fi
git -C "$TMP" \
  -c user.name="$(git -C "$ROOT" config user.name)" \
  -c user.email="$(git -C "$ROOT" config user.email)" \
  commit -q -m "Зеркало просмотрщика (GitHub Pages)"
git -C "$TMP" push -q "$URL" HEAD:gh-pages
echo "готово: https://darazumovskiy.github.io/alp-finder/"
