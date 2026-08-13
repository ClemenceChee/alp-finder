#!/usr/bin/env python3
"""Выгрузка Telegram-группы «Анализ видео с дронам» в базу знаний проекта.

Сообщения каждой темы -> docs/telegram/<слаг-темы>/messages.md
Медиа (фото, файлы)   -> data/telegram/<слаг-темы>/<msg_id>_<имя>
Состояние выгрузки    -> data/telegram/.state.json (последний номер сообщения по теме)
Лог добавлений        -> logs/tg-updates.log

Подготовка (один раз):
  1. pip install telethon
  2. На https://my.telegram.org создать приложение, получить api_id и api_hash.
  3. export TG_API_ID=... TG_API_HASH=...
  4. python scripts/tg_export.py --login   # интерактивный вход, создаст .tg_session

Дальше просто: python scripts/tg_export.py
Первый запуск по теме пишет messages.md целиком; дальше новые сообщения дописываются
в конец, медиа докачиваются только отсутствующие. Полная пересборка: --full.
Фоновая докачка раз в минуту: scripts/tg_watch.sh.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from telethon.sync import TelegramClient
from telethon.tl.functions.messages import (
    CheckChatInviteRequest,
    GetForumTopicsRequest,
    ImportChatInviteRequest,
)
from telethon.tl.types import ChatInviteAlready

INVITE_HASH = "e-P0uc1EehozOTg0"  # https://t.me/+e-P0uc1EehozOTg0

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs" / "telegram"
MEDIA_DIR = ROOT / "data" / "telegram"
SESSION = str(ROOT / "scripts" / ".tg_session")
STATE_FILE = MEDIA_DIR / ".state.json"
UPDATES_LOG = ROOT / "logs" / "tg-updates.log"
LOCK_FILE = MEDIA_DIR / ".export.lock"
LOCK_STALE_SEC = 300

# Известные темы; новые темы получают слаг транслитерацией
TOPIC_SLUGS = {
    "General": "general",
    "Новые скриншоты": "novye-skrinshoty",
    "Вещи": "veshchi",
    "Для перепроверки дроном": "dlya-pereproverki-dronom",
    "1 Гипотезы": "1-gipotezy",
    "Архив скриншотов": "arkhiv-skrinshotov",
    "Инструкции и ссылки": "instrukcii-i-ssylki",
    "Подтвержденные": "podtverzhdennye",
    "Отклоненные после проверки дроном": "otklonennye-posle-proverki-dronom",
}

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(title: str) -> str:
    if title in TOPIC_SLUGS:
        return TOPIC_SLUGS[title]
    s = "".join(TRANSLIT.get(ch, ch) for ch in title.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "topic"


def get_client() -> TelegramClient:
    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        sys.exit("Задай переменные окружения TG_API_ID и TG_API_HASH (см. докстринг).")
    return TelegramClient(SESSION, int(api_id), api_hash)


def get_group(client: TelegramClient):
    check = client(CheckChatInviteRequest(INVITE_HASH))
    if isinstance(check, ChatInviteAlready):
        return check.chat
    updates = client(ImportChatInviteRequest(INVITE_HASH))
    return updates.chats[0]


def sender_name(msg) -> str:
    s = msg.sender
    if s is None:
        return "?"
    name = " ".join(filter(None, [getattr(s, "first_name", None), getattr(s, "last_name", None)]))
    return name or getattr(s, "title", None) or getattr(s, "username", None) or str(msg.sender_id)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def acquire_lock() -> bool:
    if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime < LOCK_STALE_SEC:
        return False
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def log_update(title: str, header: str, text: str, media_note: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
    snippet = " ".join(text.split())
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    UPDATES_LOG.parent.mkdir(exist_ok=True)
    with UPDATES_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {title} | {header}{media_note} {snippet}\n")


def export_topic(client, group, topic_id: int, title: str, last_id: int) -> tuple[int, int, int]:
    """Возвращает (новых сообщений, новых медиа, максимальный id)."""
    slug = slugify(title)
    doc_dir = DOCS_DIR / slug
    media_dir = MEDIA_DIR / slug
    doc_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    md_file = doc_dir / "messages.md"

    full = last_id == 0 or not md_file.exists()
    lines = [f"# Тема «{title}» — выгрузка сообщений", ""] if full else []
    n_msgs = n_media = 0
    max_id = last_id

    # В теме General topic_id=1, фильтр reply_to там не работает — берём всё без фильтра
    kwargs = {} if topic_id == 1 else {"reply_to": topic_id}
    if not full:
        kwargs["min_id"] = last_id
    for msg in client.iter_messages(group, reverse=True, **kwargs):
        max_id = max(max_id, msg.id)
        if topic_id == 1 and msg.reply_to and getattr(msg.reply_to, "forum_topic", False):
            continue  # сообщение другой темы
        if msg.action is not None:
            continue  # служебное: создание/переименование темы и т.п.
        n_msgs += 1
        stamp = msg.date.strftime("%Y-%m-%d %H:%M")
        header = f"[{stamp}] #{msg.id} {sender_name(msg)}:"

        media_note = ""
        if msg.media:
            existing = list(media_dir.glob(f"{msg.id}_*")) + list(media_dir.glob(f"{msg.id}.*"))
            if existing:
                path = existing[0]
            else:
                orig = getattr(msg.file, "name", None)
                target = f"{msg.id}_{orig}" if orig else str(msg.id)
                path = msg.download_media(file=str(media_dir / target))
                path = Path(path) if path else None
            if path:
                n_media += 1
                media_note = f" [медиа: data/telegram/{slug}/{path.name}]"

        text = (msg.text or "").strip()
        lines.append(f"{header}{media_note}")
        if text:
            lines.append(text)
        lines.append("")

        if not full:
            log_update(title, header, text, media_note)

    if full:
        md_file.write_text("\n".join(lines), encoding="utf-8")
    elif lines:
        with md_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return n_msgs, n_media, max_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", action="store_true", help="только интерактивный вход")
    parser.add_argument("--topic", help="выгрузить одну тему по названию или слагу")
    parser.add_argument("--full", action="store_true", help="полная пересборка messages.md")
    parser.add_argument("--quiet", action="store_true", help="печатать только темы с новыми сообщениями")
    args = parser.parse_args()

    client = get_client()
    with client:
        if args.login:
            print(f"Вход выполнен: {client.get_me().first_name}. Сессия: {SESSION}")
            return

        if not acquire_lock():
            print("Другая выгрузка ещё идёт (data/telegram/.export.lock), выхожу.")
            return
        try:
            state = load_state()
            group = get_group(client)
            topics = client(GetForumTopicsRequest(
                peer=group, offset_date=None, offset_id=0, offset_topic=0, limit=100,
            )).topics

            for t in topics:
                slug = slugify(t.title)
                if args.topic and args.topic not in (t.title, slug):
                    continue
                last_id = 0 if args.full else state.get(slug, 0)
                n_msgs, n_media, max_id = export_topic(client, group, t.id, t.title, last_id)
                state[slug] = max_id
                save_state(state)
                if n_msgs or not args.quiet:
                    print(f"{t.title}: +{n_msgs} сообщений, +{n_media} медиа -> docs/telegram/{slug}/")
        finally:
            LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
