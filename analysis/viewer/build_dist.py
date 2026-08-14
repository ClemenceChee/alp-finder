#!/usr/bin/env python3
"""Сборка публичной версии просмотрщика в analysis/viewer/dist/.

Копирует index.html, map.html, montages.html и все картинки, на которые они
ссылаются (пути от корня репозитория: analysis/..., docs/..., data/...),
в dist/ с сохранением структуры путей. Абсолютные пути картинок (/analysis/...)
при копировании переписываются на относительные: страницы в dist лежат в корне
сайта, поэтому относительные пути работают и в корне домена (Cloudflare Pages),
и в подпапке (зеркало GitHub Pages).

Запуск: python analysis/viewer/build_dist.py
Деплой: npx wrangler pages deploy analysis/viewer/dist --project-name alp-finder
Зеркало для РФ/РБ: bash scripts/deploy_mirror.sh
"""

import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = Path(__file__).resolve().parent
DIST = VIEWER_DIR / "dist"

PAGES = ["index.html", "map.html", "montages.html"]

# Ловим и литеральные src="/analysis/...", и пути в JS-данных ("analysis/.../x.jpg")
ASSET_RE = re.compile(
    r'(?:analysis|docs|data)/[A-Za-z0-9_./\-]+\.(?:jpe?g|png|JPG|PNG)'
)


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    # без него GitHub Pages (Jekyll) выкидывает каталоги с подчёркиванием (_calib)
    (DIST / ".nojekyll").write_text("")

    assets: set[str] = set()
    for page in PAGES:
        src = VIEWER_DIR / page
        if not src.exists():
            print(f"нет страницы {src}, сначала запусти build_viewer.py/build_map.py",
                  file=sys.stderr)
            return 1
        text = src.read_text()
        # локальные страницы смотрят от корня репо (src="/analysis/...");
        # в dist страницы в корне сайта — те же пути делаем относительными
        (DIST / page).write_text(
            text.replace('src="/', 'src="').replace('data-full="/', 'data-full="'),
            "utf-8",
        )
        assets.update(ASSET_RE.findall(text))

    copied = 0
    missing: list[str] = []
    for rel in sorted(assets):
        src = REPO_ROOT / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = DIST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    total_mb = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 2**20
    print(f"dist/: {copied} картинок + {len(PAGES)} страниц, {total_mb:.0f} МБ")
    if missing:
        print(f"не найдено {len(missing)} файлов (страницы будут с битыми превью):",
              file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
