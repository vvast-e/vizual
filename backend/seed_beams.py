"""
Seed-скрипт: скачивает 3 CC0-текстуры фальшбалок с Poly Haven
и добавляет соответствующие записи в таблицу materials.

Идемпотентен — повторный запуск не дублирует материалы.

Запуск (из backend/, с активным venv):
    python seed_beams.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# Добавляем backend/ в PYTHONPATH, чтобы импортировать db.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import SessionLocal
from db.models import Material

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "public", "textures", "beams")

BEAM_TEXTURES = [
    {
        "name": "Балка тёмный орех",
        "file": "beam-dark-walnut.jpg",
        "url": (
            "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k"
            "/dark_wood/dark_wood_diff_2k.jpg"
        ),
    },
    {
        "name": "Балка дуб натуральный",
        "file": "beam-oak-natural.jpg",
        "url": (
            "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k"
            "/wood_table_001/wood_table_001_diff_2k.jpg"
        ),
    },
    {
        "name": "Балка дуб состаренный",
        "file": "beam-oak-aged.jpg",
        "url": (
            "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k"
            "/oak_wood_planks/oak_wood_planks_diff_2k.jpg"
        ),
    },
]

# ---------------------------------------------------------------------------
# Скачивание файлов
# ---------------------------------------------------------------------------


def download_textures() -> None:
    os.makedirs(TEXTURES_DIR, exist_ok=True)
    for entry in BEAM_TEXTURES:
        dest = os.path.join(TEXTURES_DIR, entry["file"])
        if os.path.exists(dest):
            print(f"  [skip] {entry['file']} — уже существует")
            continue
        print(f"  [download] {entry['file']}  ← {entry['url']}")
        try:
            urllib.request.urlretrieve(entry["url"], dest)
            size_kb = Path(dest).stat().st_size // 1024
            print(f"           OK ({size_kb} KB)")
        except Exception as exc:
            print(f"           ОШИБКА: {exc}")
            raise


# ---------------------------------------------------------------------------
# Вставка в БД
# ---------------------------------------------------------------------------


def seed_db() -> None:
    db = SessionLocal()
    try:
        inserted = 0
        for entry in BEAM_TEXTURES:
            file_path = f"beams/{entry['file']}"
            exists = (
                db.query(Material)
                .filter(Material.file_path == file_path)
                .first()
            )
            if exists:
                print(f"  [skip] БД: '{entry['name']}' — уже есть (id={exists.id})")
                continue
            mat = Material(
                name=entry["name"],
                scene_category="interior",
                material_category="beam",
                file_path=file_path,
                is_visible_in_visualizer=True,
            )
            db.add(mat)
            inserted += 1
            print(f"  [insert] '{entry['name']}' → {file_path}")
        db.commit()
        print(f"\nГотово. Вставлено записей: {inserted}.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== seed_beams: скачивание текстур ===")
    download_textures()
    print("\n=== seed_beams: вставка в БД ===")
    seed_db()
