"""Тесты scripts/report_tables.py — публикационные таблицы и графики из результатов оценки."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

SUMMARY = [
    {"mode": "baseline", "horizon": 7, "data_fraction": 1.0, "n_series": 3, "mae": 4.0,
     "mae_median": 3.9, "rmse": 4.5, "mape": 110.0, "wape": 28.0, "mase": 0.97},
    {"mode": "agents", "horizon": 7, "data_fraction": 1.0, "n_series": 3, "mae": 3.6,
     "mae_median": 3.5, "rmse": 4.1, "mape": 120.0, "wape": 27.0, "mase": 0.9,
     "agent_metrics": {"mean_iterations": 1.0, "hitl_share": 1.0, "fallback_share": 0.0,
                       "mean_tokens_in": 150.0, "mean_tokens_out": 20.0,
                       "mean_duration_s": 2.5, "mean_critic_score": 5.0}},
]
COMPARISONS = [{
    "models_compared": ["agents", "baseline"], "horizon": 7, "data_fraction": 1.0,
    "winner": "agents", "n_series": 3,
    "significance": {"wilcoxon": {"pvalue": 0.18}, "paired_ttest": {"pvalue": 0.07},
                     "bootstrap": {"mean_diff": -0.4, "ci_low": -0.9, "ci_high": -0.02},
                     "dm_mean_pvalue": 0.5},
}]
RECORDS = [
    {"mode": "baseline", "horizon": 7, "duration_s": 1.5, "metrics": {"mae": 4.0}},
    {"mode": "agents", "horizon": 7, "duration_s": 2.5, "metrics": {"mae": 3.6},
     "agent_metrics": {"tokens_in": 150, "tokens_out": 20}},
]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("report_tables", SCRIPTS / "report_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_tables"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def eval_dir(tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    (root / "comparisons.json").write_text(json.dumps(COMPARISONS), encoding="utf-8")
    (root / "meta.json").write_text(json.dumps({"n_records": 2, "n_windows": 1, "seed": 42}),
                                    encoding="utf-8")
    with (root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record) + "\n")
    fractions = root / "fractions"
    fractions.mkdir()
    (fractions / "summary.json").write_text(json.dumps([
        {"mode": "baseline", "horizon": 14, "data_fraction": 0.1, "n_series": 2,
         "mae": 4.5, "rmse": 5.1, "mase": 1.3},
        {"mode": "baseline", "horizon": 14, "data_fraction": 1.0, "n_series": 2,
         "mae": 4.4, "rmse": 5.0, "mase": 1.2},
    ]), encoding="utf-8")
    return root


def run_main(script, monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["report_tables", *argv])
    return script.main()


class TestReportTables:
    def test_tables_only(self, script, monkeypatch, eval_dir, tmp_path):
        out = tmp_path / "report"
        assert run_main(script, monkeypatch, "--eval-dir", str(eval_dir),
                        "--fractions-dir", str(eval_dir / "fractions"),
                        "--out", str(out), "--no-plots") == 0

        summary = (out / "tables" / "summary.md").read_text(encoding="utf-8")
        assert "| baseline (классика без LLM) | 7 | 3 | 4.000 | 3.900 | 4.500 | 110.0 | 28.0 | 0.970 |" in summary
        assert "Agents" not in summary

        agents = (out / "tables" / "agent_metrics.md").read_text(encoding="utf-8")
        assert "100%" in agents and "150" in agents and "2.50" in agents

        comparisons = (out / "tables" / "comparisons.md").read_text(encoding="utf-8")
        assert "agents vs baseline" in comparisons and "[-0.900; -0.020]" in comparisons

        cost = (out / "tables" / "cost.md").read_text(encoding="utf-8")
        assert "| baseline (классика без LLM) | 7 | 4.000 | 1.50 | 0 |" in cost
        assert "170" in cost  # 150 + 20 токенов на ряд для agents

        fractions = (out / "tables" / "fractions.md").read_text(encoding="utf-8")
        assert "| 0.1 | 2 | 4.500 | 5.100 | 1.300 |" in fractions
        assert not (out / "plots").exists()

    def test_with_plots(self, script, monkeypatch, eval_dir, tmp_path):
        pytest.importorskip("matplotlib", reason="matplotlib не установлен")
        out = tmp_path / "report"
        assert run_main(script, monkeypatch, "--eval-dir", str(eval_dir),
                        "--fractions-dir", str(eval_dir / "fractions"),
                        "--out", str(out)) == 0
        for name in ("mae_by_mode.png", "mase_by_mode.png", "mae_vs_data_fraction.png"):
            assert (out / "plots" / name).stat().st_size > 1000, name

    def test_missing_results_returns_1(self, script, monkeypatch, tmp_path):
        assert run_main(script, monkeypatch, "--eval-dir", str(tmp_path / "missing"),
                        "--out", str(tmp_path / "out")) == 1

    def test_none_values_rendered_as_dash(self, script):
        assert script._fmt(None) == "—" and script._fmt(float("nan")) == "—"
        assert script._fmt(1.23456) == "1.235"
        assert script.comparisons_table([]).startswith("_Сравнения режимов недоступны")
        assert script.agent_metrics_table([{"mode": "baseline", "horizon": 7}]).startswith(
            "_Агентные метрики недоступны"
        )
