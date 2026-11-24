"""Общие помощники тестов оценки (Этап 13): синтетический тестовый набор."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_window(index: int, n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + index)
    t = np.arange(n)
    values = 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return {
        "series_id": f"test|temperature|{index}",
        "city": "TestCity",
        "variable": "temperature",
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "values": [round(float(v), 3) for v in values],
        "dates": [str(d.date()) for d in dates],
    }


def write_test_set(path: Path, n_windows: int = 3, n: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(n_windows):
            handle.write(json.dumps(make_window(index, n=n)) + "\n")
    return path
