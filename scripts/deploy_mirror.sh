#!/usr/bin/env bash
# Публикация analysis/viewer/dist в ветку gh-pages — зеркало просмотрщика
# на GitHub Pages: https://darazumovskiy.github.io/alp-finder/
# Нужно потому, что *.pages.dev заблокирован провайдерами в РФ и РБ.
# Перед запуском собрать dist: python analysis/viewer/build_dist.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/analysis/viewer/dist"
[ -f "$DIST/index.html" ] || {
  echo "нет $DIST/index.html — сначала python analysis/viewer/build_dist.py" >&2
  exit 1
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cp -R "$DIST/." "$TMP/"
git -C "$TMP" init -q -b gh-pages
git -C "$TMP" add -A
git -C "$TMP" \
  -c user.name="$(git -C "$ROOT" config user.name)" \
  -c user.email="$(git -C "$ROOT" config user.email)" \
  commit -q -m "Зеркало просмотрщика (GitHub Pages)"
git -C "$TMP" push -f "$(git -C "$ROOT" remote get-url origin)" gh-pages
echo "готово: https://darazumovskiy.github.io/alp-finder/"
