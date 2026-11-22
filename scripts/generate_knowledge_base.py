"""Генерация базы знаний «успешных кейсов» (Этап 9, задание MLF: >=1000 записей).

Схема (синтетическая генерация разрешена и поощряется заданием):
1. Реальные ряды (Jena Climate, Daily Climate Дели) приводятся к дневной частоте;
2. Ряды нарезаются скользящими окнами (окно 365 дней, шаг 10) — окно = «кейс»;
3. Последние holdout_fraction окон каждого ряда уходят в test_set.jsonl
   (не попадают в базу знаний) — отдельный тестовый набор для run_eval;
4. Для каждого обучающего окна гоняется сетка моделей (наивный сезонный,
   ARIMA(1,1,1), SARIMA при сезонности <=31, XGBoost на лагах), лучшая
   по MASE (fallback MAE) становится «успешным кейсом»;
5. Кейс: статистические признаки окна + лучшая модель + гиперпараметры +
   метрики + текстовое описание (для BM25 на Этапе 10). Эмбеддинги (PCA)
   добавит Этап 10 при индексации.

Запуск (полный): python scripts/generate_knowledge_base.py --target-cases 1000
Быстрый прогон:  ... --fast --target-cases 30 --window 365 --step 60
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # прямой запуск: python scripts/...

from backend.core.metrics import naive_forecast, regression_metrics  # noqa: E402
from backend.timeseries.classical import fit_arima, fit_xgboost, time_split  # noqa: E402
from backend.timeseries.preprocessing import prepare_timeseries  # noqa: E402

RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

EVAL_HORIZON = 14  # горизонт оценки сетки (тема: прогноз 7–14 дней)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("generate_kb")


@dataclass
class SeriesSource:
    source: str
    city: str
    variable: str
    series: pd.Series  # дневная частота


# ============================================================================
# Загрузка источников
# ============================================================================
def load_sources(raw_dir: Path) -> list[SeriesSource]:
    sources: list[SeriesSource] = []

    jena_path = raw_dir / "jena_climate_2009_2016.csv"
    if jena_path.exists():
        frame = pd.read_csv(jena_path)
        frame["Date Time"] = pd.to_datetime(
            frame["Date Time"], format="%d.%m.%Y %H:%M:%S"
        )
        frame = frame.set_index("Date Time").replace(-9999.0, np.nan)
        # S3-зеркало (keras) короче UCI-версии: колонка дождей есть не всегда —
        # берём те, что реально присутствуют в файле
        spec = {
            "T (degC)": ("temperature", "mean"),
            "p (mbar)": ("pressure", "mean"),
            "rh (%)": ("humidity", "mean"),
            "rain (mm)": ("precipitation", "sum"),
            "wv (m/s)": ("wind_speed", "mean"),
            "Tdew (degC)": ("dew_point", "mean"),
        }
        used = 0
        for column, (variable, agg) in spec.items():
            if column not in frame.columns:
                logger.info("jena: колонки %s нет в файле — пропускаем", column)
                continue
            daily = frame[column].resample("D").agg(agg).dropna()
            sources.append(SeriesSource("jena", "Jena", variable, daily))
            used += 1
        logger.info("jena: %d дневных рядов", used)
    else:
        logger.warning("jena CSV не найден: %s", jena_path)

    delhi_path = raw_dir / "DailyDelhiClimateTrain.csv"
    if delhi_path.exists():
        frame = pd.read_csv(delhi_path, parse_dates=["date"]).set_index("date")
        for column, variable in (
            ("meantemp", "temperature"),
            ("humidity", "humidity"),
            ("wind_speed", "wind_speed"),
        ):
            sources.append(
                SeriesSource("delhi", "Delhi", variable, frame[column].dropna())
            )
        logger.info("delhi: 3 дневных ряда")
    else:
        logger.warning("delhi CSV не найден: %s", delhi_path)

    return sources


def synthetic_sources(n_series: int = 8, seed: int = 2026) -> list[SeriesSource]:
    """Синтетические метеоряды (добор до target-cases, разрешён заданием)."""
    rng = np.random.default_rng(seed)
    variables = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
    sources = []
    for i in range(n_series):
        period = int(rng.choice([7, 14, 30, 90]))
        trend = float(rng.uniform(-0.005, 0.005))
        noise = float(rng.uniform(0.2, 1.5))
        base = float(rng.uniform(0, 15))
        t = np.arange(730)
        values = (
            base
            + 5 * np.sin(2 * np.pi * t / period)
            + trend * t
            + rng.normal(0, noise, t.size)
        )
        if variables[i % len(variables)] == "precipitation":
            values = np.maximum(values, 0)  # осадки неотрицательны
        index = pd.date_range("2015-01-01", periods=t.size, freq="D")
        sources.append(
            SeriesSource(
                "synthetic",
                f"SynthCity{i}",
                variables[i % len(variables)],
                pd.Series(values, index=index),
            )
        )
    return sources


# ============================================================================
# Окна и оценка сетки моделей
# ============================================================================
def make_windows(series: pd.Series, window: int, step: int) -> list[pd.Series]:
    return [
        series.iloc[start : start + window]
        for start in range(0, len(series) - window + 1, step)
    ]


def _evaluate_naive(window: pd.Series, season: int | None) -> dict:
    train, val = time_split(window, EVAL_HORIZON)
    pred = naive_forecast(train.values, len(val), season=season or 1)
    metrics = regression_metrics(val.values, pred, train.values, season=season or 1)
    return {
        "best_model": "naive_seasonal",
        "hyperparams": {"season": season or 1},
        "metrics": metrics,
    }


def evaluate_window(
    window: pd.Series, source: SeriesSource, fast: bool
) -> dict | None:
    """Полный пайплайн одного окна: подготовка -> сетка моделей -> лучший кейс."""
    prepared = prepare_timeseries(
        window.values.tolist(), dates=list(window.index), target_freq="D"
    )
    series = prepared["series"]
    if len(series) < 60:
        return None
    diagnostics = prepared["diagnostics"]
    season = diagnostics.get("seasonal_period")

    candidates = [_evaluate_naive(series, season)]
    try:
        arima = fit_arima(series, EVAL_HORIZON, order=[1, 1, 1], season=season)
        candidates.append(
            {
                "best_model": arima["model"],
                "hyperparams": {
                    "order": arima["order"],
                    "seasonal_order": arima["seasonal_order"],
                },
                "metrics": arima["metrics"],
            }
        )
    except Exception as exc:  # noqa: BLE001 — окно может не обучиться
        logger.debug("arima failed on window: %s", exc)
    if not fast:
        if season and season <= 31:
            try:
                sarima = fit_arima(
                    series, EVAL_HORIZON, order=[1, 1, 1], seasonal_order=[1, 1, 1, season]
                )
                candidates.append(
                    {
                        "best_model": "sarima",
                        "hyperparams": {
                            "order": sarima["order"],
                            "seasonal_order": sarima["seasonal_order"],
                        },
                        "metrics": sarima["metrics"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("sarima failed on window: %s", exc)
        try:
            xgb = fit_xgboost(series, EVAL_HORIZON, lags=7, n_estimators=100)
            candidates.append(
                {
                    "best_model": "xgboost",
                    "hyperparams": {"lags": 7, "n_estimators": 100},
                    "metrics": xgb["metrics"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("xgboost failed on window: %s", exc)

    def rank(case: dict) -> float:
        metrics = case["metrics"]
        value = metrics.get("mase")
        return value if value is not None and np.isfinite(value) else metrics["mae"] * 10

    best = min(candidates, key=rank)

    stats = diagnostics["stats"]
    adf = diagnostics.get("adf", {})
    stl = diagnostics.get("stl") or {}
    features = {
        "length": stats["length"],
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "acf_lag1": stats["acf_lag1"],
        "adf_pvalue": adf.get("adf_pvalue"),
        "stationary": adf.get("stationary"),
        "seasonal_period": season,
        "trend_strength": stl.get("trend_strength"),
        "seasonal_strength": stl.get("seasonal_strength"),
        "missing_rate": diagnostics.get("raw_missing_rate"),
        "variable": source.variable,
        "city": source.city,
    }
    description = (
        f"Дневной ряд {source.variable} ({source.city}), окно "
        f"{series.index[0]:%Y-%m-%d}..{series.index[-1]:%Y-%m-%d}; "
        f"сезонность {season or 'не выражена'}, тренд {stl.get('trend_strength', 0):.2f}. "
        f"Лучшая модель {best['best_model']}: MASE {best['metrics'].get('mase')}, "
        f"MAE {best['metrics']['mae']}."
    )
    start = series.index[0].strftime("%Y%m%d")
    return {
        "series_id": f"{source.source}|{source.variable}|{start}",
        "task_type": "timeseries",
        "city": source.city,
        "freq": "D",
        "target_column": source.variable,
        "features": features,
        "best_model": best["best_model"],
        "hyperparams": best["hyperparams"],
        "metrics": best["metrics"],
        "description": description,
    }


# ============================================================================
# Основной сценарий
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация базы знаний")
    parser.add_argument("--target-cases", type=int, default=1000)
    parser.add_argument("--window", type=int, default=365)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--fast", action="store_true", help="сетка только naive+arima")
    parser.add_argument("--include-synthetic", action="store_true", default=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sources = load_sources(args.raw_dir)
    if not sources:
        logger.error("Нет данных: сначала запустите scripts/download_data.py")
        return 1

    train_windows: list[tuple[SeriesSource, pd.Series]] = []
    test_windows: list[tuple[SeriesSource, pd.Series]] = []
    for source in sources:
        windows = make_windows(source.series, args.window, args.step)
        if not windows:
            continue
        n_test = max(1, int(len(windows) * args.holdout_fraction))
        source_test = windows[-n_test:]
        # Gap против утечки: train-окна должны ЗАКАНЧИВАТЬСЯ раньше начала
        # первого test-окна, иначе данные пересекаются (окна с шагом < окна)
        first_test_start = source_test[0].index[0]
        source_train = [
            w for w in windows[: -n_test] if w.index[-1] < first_test_start
        ]
        test_windows += [(source, w) for w in source_test]
        train_windows += [(source, w) for w in source_train]
    logger.info(
        "окон: %d обучающих, %d тестовых (holdout %.0f%%)",
        len(train_windows), len(test_windows), args.holdout_fraction * 100,
    )

    if len(train_windows) < args.target_cases and args.include_synthetic:
        need = args.target_cases - len(train_windows)
        for source in synthetic_sources():
            windows = make_windows(source.series, args.window, step=max(10, args.step))
            if not windows:
                continue
            n_test = max(1, int(len(windows) * args.holdout_fraction))
            source_test = windows[-n_test:]
            first_test_start = source_test[0].index[0]
            source_train = [
                w for w in windows[: -n_test] if w.index[-1] < first_test_start
            ]
            test_windows += [(source, w) for w in source_test]
            taken = source_train[: need // 8 + 1]
            train_windows += [(source, w) for w in taken]
            need -= len(taken)
        logger.info("после синтетики: %d обучающих окон", len(train_windows))

    train_windows = train_windows[: args.target_cases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kb_path = args.out_dir / "knowledge_base.jsonl"
    test_path = args.out_dir / "test_set.jsonl"

    written = 0
    with kb_path.open("w", encoding="utf-8") as handle:
        for i, (source, window) in enumerate(train_windows):
            case = evaluate_window(window, source, fast=args.fast)
            if case is None:
                continue
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                logger.info("кейсов готово: %d / %d", written, len(train_windows))
    logger.info("База знаний: %d кейсов -> %s", written, kb_path)

    with test_path.open("w", encoding="utf-8") as handle:
        for source, window in test_windows:
            entry = {
                "series_id": f"{source.source}|{source.variable}|{window.index[0]:%Y%m%d}",
                "city": source.city,
                "variable": source.variable,
                "start": str(window.index[0].date()),
                "end": str(window.index[-1].date()),
                "values": [round(float(v), 3) for v in window.values],
                "dates": [str(ts.date()) for ts in window.index],
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Тестовый набор: %d окон -> %s", len(test_windows), test_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
