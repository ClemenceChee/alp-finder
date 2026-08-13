#!/usr/bin/env python3
"""Докачка недостающих файлов манифеста с публичного Яндекс Диска штаба.

Зеркало: https://disk.yandex.ru/d/zyqqUfN8A-iQzg (структура папок отличается от
Google Drive, поэтому сопоставление — по имени файла и точному размеру).
Идемпотентен: файлы, уже лежащие локально с верным размером, пропускаются.
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

PUBLIC_KEY = "https://disk.yandex.ru/d/zyqqUfN8A-iQzg"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api_get(endpoint: str, **params) -> dict:
    params["public_key"] = PUBLIC_KEY
    url = endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def walk(path="/"):
    offset = 0
    while True:
        data = api_get(API, path=path, limit=200, offset=offset)
        emb = data["_embedded"]
        for item in emb["items"]:
            if item["type"] == "dir":
                yield from walk(item["path"])
            else:
                yield item
        offset += len(emb["items"])
        if offset >= emb["total"]:
            break


def main() -> int:
    remote = {}
    for item in walk():
        remote[(os.path.basename(item["path"]), item["size"])] = item["path"]

    missing = []
    with open(os.path.join(ROOT, "scripts/manifest.tsv")) as f:
        for line in f:
            local_path, _file_id, size = line.rstrip("\n").split("\t")
            size = int(size)
            abs_path = os.path.join(ROOT, local_path)
            if os.path.isfile(abs_path) and os.path.getsize(abs_path) == size:
                continue
            missing.append((local_path, size))

    print(f"Недостаёт локально: {len(missing)}")
    done = failed = absent = 0
    for local_path, size in missing:
        key = (os.path.basename(local_path), size)
        if key not in remote:
            absent += 1
            print(f"НЕТ НА ЯНДЕКСЕ {local_path}")
            continue
        href = api_get(API + "/download", path=remote[key])["href"]
        abs_path = os.path.join(ROOT, local_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        part = abs_path + ".part"
        rc = subprocess.run(
            ["curl", "-sSL", "--fail", "--retry", "3", "--connect-timeout", "30",
             "-C", "-", href, "-o", part]).returncode
        if rc == 0 and os.path.exists(part) and os.path.getsize(part) == size:
            os.replace(part, abs_path)
            done += 1
            print(f"OK {local_path}")
        else:
            failed += 1
            actual = os.path.getsize(part) if os.path.exists(part) else 0
            print(f"FAIL {local_path}: rc={rc}, байт {actual} из {size}")
            if os.path.exists(part):
                os.remove(part)
        sys.stdout.flush()

    print(f"ИТОГ ЯНДЕКС: скачано {done}, ошибок {failed}, нет на зеркале {absent}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
