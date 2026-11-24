"""Импорт базы знаний и справочника в БД (Этап 9, методичка Step_6).

Защита от дубликатов: (series_id, task_type) для кейсов, term для терминов —
повторный запуск ничего не дублирует (inserted/skipped в отчёте).

Запуск:
    python scripts/import_data.py                       # оба файла из data/processed
    python scripts/import_data.py --kb only             # только база знаний
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # прямой запуск: python scripts/...

from backend.data import knowledge_base  # noqa: E402
from backend.db.session import ensure_tables_once  # noqa: E402

PROCESSED_DIR = BASE_DIR / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("import_data")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Файл не найден: %s", path)
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def run(args: argparse.Namespace) -> int:
    # таблицы создаются/дополняются при отсутствии (локальный режим без Alembic)
    await ensure_tables_once()

    if args.what in ("all", "kb"):
        cases = read_jsonl(args.kb_path)
        if cases:
            inserted, skipped = await knowledge_base.import_cases(cases)
            logger.info(
                "knowledge_base: %d вставлено, %d пропущено (дубликаты)", inserted, skipped
            )

    if args.what in ("all", "glossary"):
        terms = read_jsonl(args.glossary_path)
        if terms:
            inserted, skipped = await knowledge_base.import_terms(terms)
            logger.info("glossary: %d вставлено, %d пропущено", inserted, skipped)

    counts = await knowledge_base.get_counts()
    logger.info("Итог в БД: %s", counts)

    ok = counts["knowledge_base"] >= 1000 and counts["glossary"] >= 500
    logger.info(
        "Пороги задания (БЗ >= 1000, справочник >= 500): %s",
        "выполнены" if ok else "НЕ выполнены — запустите generate-скрипты",
    )
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Импорт данных в БД")
    parser.add_argument(
        "--what", choices=["all", "kb", "glossary"], default="all"
    )
    parser.add_argument("--kb-path", type=Path, default=PROCESSED_DIR / "knowledge_base.jsonl")
    parser.add_argument(
        "--glossary-path", type=Path, default=PROCESSED_DIR / "glossary.jsonl"
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
