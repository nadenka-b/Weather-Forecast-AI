"""Оценка качества системы (Этап 13, задание MLF: scripts/run_eval.py).

Параметры (по заданию):
    --num-series     20 / 50 / 100 — число тестовых окон (случайная выборка с seed);
    --horizons       горизонты прогноза, напр. 7,14 (в задании: 10,20,50);
    --data-fractions доли обучающих данных: 0.1,0.2,0.5,1.0 (по умолчанию 1.0);
    --mode           один или несколько режимов через запятую: baseline, rag,
                     agents, foundation; all (по умолчанию) — все режимы;
    --fast           сокращённая сетка baseline (naive + ARIMA(1,1,1));
    --no-save-db     не записывать сводку в БД (evaluation_results, model_comparisons).

Режимы: baseline (классика без LLM), rag (выбор модели по кейсам БЗ),
agents (полный цикл агентов), foundation (бонус; без установки — пропуск).
Метрики: MAE/RMSE/MAPE/WAPE/MASE на горизонте + агентные метрики (итерации,
доля HITL, токены LLM, время). Сравнение режимов: Diebold-Mariano, Wilcoxon,
bootstrap-CI, paired t-test.

Выходы в --out-dir (по умолчанию data/eval):
    results.jsonl      построчные результаты (окно × режим × горизонт × доля)
    summary.csv/json   сводная таблица
    comparisons.json   попарные сравнения режимов со стат. значимостью
    meta.json          параметры запуска и время
Логи: eval.log в каталоге результатов.

Запуск:
    python scripts/run_eval.py --num-series 50 --horizons 7,14 --data-fractions 1.0
    python scripts/run_eval.py --num-series 20 --horizons 14 --data-fractions 0.1,0.2,0.5,1.0 --mode baseline
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # прямой запуск: python scripts/...

from backend.evaluation.runner import (  # noqa: E402
    DEFAULT_TEST_SET,
    MODES,
    EvalConfig,
    load_test_windows,
    run_evaluation,
    summary_frame,
)

OUT_DIR = BASE_DIR / "data" / "eval"

logger = logging.getLogger("run_eval")


def parse_list(text: str, cast) -> list:
    return [cast(item) for item in text.split(",") if item.strip()]


def setup_logging(log_path: Path) -> None:
    """Консоль + файл eval.log (для пост-мортем анализа)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка качества прогнозирования")
    parser.add_argument("--num-series", type=int, default=20, help="число тестовых окон")
    parser.add_argument("--horizons", default="14", help="горизонты через запятую")
    parser.add_argument("--data-fractions", default="1.0", help="доли данных через запятую")
    parser.add_argument("--mode", default="all",
                        help="режим(ы) через запятую: baseline, rag, agents, foundation или all")
    parser.add_argument("--fast", action="store_true", help="сокращённая сетка baseline")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-db", action="store_true", help="не писать сводку в БД")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir / "eval.log")

    requested_modes = parse_list(args.mode, str.strip) or ["all"]
    unknown = [mode for mode in requested_modes if mode != "all" and mode not in MODES]
    if unknown:
        parser.error(f"Неизвестные режимы {unknown}; доступны {list(MODES)} и all")
    modes = list(MODES) if "all" in requested_modes else requested_modes
    horizons = parse_list(args.horizons, int)
    fractions = parse_list(args.data_fractions, float)
    if not horizons or not fractions:
        parser.error("--horizons и --data-fractions должны содержать хотя бы одно значение")

    configs = [
        EvalConfig(mode=mode, horizon=horizon, data_fraction=fraction, fast=args.fast)
        for mode in modes
        for horizon in horizons
        for fraction in fractions
    ]

    started = time.perf_counter()
    windows = load_test_windows(args.test_set, num_series=args.num_series, seed=args.seed)
    logger.info(
        "run_eval: %d окон × %d конфигураций (режимы %s, горизонты %s, доли %s, fast=%s)",
        len(windows), len(configs), modes, horizons, fractions, args.fast,
    )

    outcome = asyncio.run(run_evaluation(configs, windows))

    # --- Выходные файлы
    with (args.out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in outcome["records"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    frame = summary_frame(outcome["summary"])
    frame.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(outcome["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "comparisons.json").write_text(
        json.dumps(outcome["comparisons"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "num_series": args.num_series,
                "horizons": horizons,
                "data_fractions": fractions,
                "modes": modes,
                "fast": args.fast,
                "seed": args.seed,
                "wall_time_s": round(time.perf_counter() - started, 2),
                "n_windows": outcome["n_windows"],
                "n_records": outcome["n_records"],
                "n_errors": outcome["n_errors"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- Сводка в БД
    if not args.no_save_db:
        try:
            from backend.db.session import ensure_tables_once
            from backend.evaluation.service import create_run, save_results

            async def persist() -> str:
                await ensure_tables_once()
                run = await create_run(
                    {
                        "modes": modes, "horizons": horizons, "data_fractions": fractions,
                        "num_series": args.num_series, "fast": args.fast, "seed": args.seed,
                        "source": "run_eval.py",
                    }
                )
                await save_results(run["id"], outcome)
                return run["id"]

            run_id = asyncio.run(persist())
            logger.info("Сводка сохранена в БД (evaluation run %s)", run_id)
        except Exception as exc:  # noqa: BLE001 — файлы уже записаны, БД — опционально
            logger.warning("Не удалось сохранить сводку в БД: %s", exc)

    # --- Сводная таблица в лог
    logger.info("Итог (%d записей, ошибок %d, время %.1f с):",
                outcome["n_records"], outcome["n_errors"], outcome["total_time_s"])
    columns = [c for c in ("mode", "horizon", "data_fraction", "n_series", "mae", "rmse",
                           "mape", "wape", "mase", "mean_iterations", "hitl_share") if c in frame]
    logger.info("\n%s", frame[columns].to_string(index=False))
    logger.info("Результаты: %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
