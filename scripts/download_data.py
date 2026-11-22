"""Загрузка метеодатасетов в data/raw (Этап 9, методичка Step_6).

Источники:
- daily-climate (Дели) — kagglehub: sumanthvrao/daily-climate-time-series-data;
- Jena Climate (2009–2016, 10-мин) — публичное S3-зеркало CSV (UCI-датасет);
  Kaggle-зеркала требуют согласия/авторизации, поэтому прямой URL.

Запуск: python scripts/download_data.py [--source all|daily|jena]
"""
import argparse
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

JENA_URL = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
DAILY_SLUG = "sumanthvrao/daily-climate-time-series-data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_data")


def download_daily() -> None:
    """Daily Climate (Дели) через kagglehub — публичный датасет."""
    import kagglehub

    path = Path(kagglehub.dataset_download(DAILY_SLUG))
    for csv in path.glob("*.csv"):
        target = RAW_DIR / csv.name
        target.write_bytes(csv.read_bytes())
        logger.info("daily-climate: %s", target.name)


def download_jena() -> None:
    """Jena Climate: публичное S3-зеркало zip-архива с одним CSV."""
    target = RAW_DIR / "jena_climate_2009_2016.csv"
    if target.exists():
        logger.info("jena: уже скачан, пропускаем (%s)", target.name)
        return

    logger.info("jena: скачиваем %s", JENA_URL)
    with urllib.request.urlopen(JENA_URL, timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = archive.namelist()[0]
    target.write_bytes(archive.read(name))

    import pandas as pd

    frame = pd.read_csv(target)
    logger.info("jena: %s (%d строк, %d колонок)", target.name, *frame.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка метеодатасетов")
    parser.add_argument("--source", choices=["all", "daily", "jena"], default="all")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.source in ("all", "daily"):
        download_daily()
    if args.source in ("all", "jena"):
        download_jena()

    logger.info("Готово. Файлы в %s:", RAW_DIR)
    for path in sorted(RAW_DIR.glob("*.csv")):
        logger.info("  %s (%.1f МБ)", path.name, path.stat().st_size / 1e6)


if __name__ == "__main__":
    sys.exit(main())
