"""Публикационные таблицы и графики из результатов оценки (методичка Step_12, п. 12.19).

Читает выходы scripts/run_eval.py и строит материалы для отчёта:
    reports/tables/summary.md        MAE/RMSE/MAPE/WAPE/MASE по режимам x горизонтам
    reports/tables/agent_metrics.md  агентные метрики (итерации, HITL, токены, время)
    reports/tables/fractions.md      точность vs доля обучающих данных
    reports/tables/comparisons.md    стат. значимость (Wilcoxon, bootstrap-CI, t-test, DM)
    reports/tables/cost.md           стоимость агентного подхода (время/токены)
    reports/plots/*.png              графики (matplotlib)

Запуск:
    python scripts/report_tables.py                      # таблицы + графики
    python scripts/report_tables.py --out reports/eval  # другой каталог
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # прямой запуск: python scripts/...

EVAL_DIR = BASE_DIR / "data" / "eval"
FRACTIONS_DIR = EVAL_DIR / "fractions"

MODE_LABELS = {
    "baseline": "baseline (классика без LLM)",
    "rag": "+rag (модель по кейсам БЗ)",
    "agents": "+agents (полный цикл агентов)",
    "foundation": "+foundation (TS Foundation Models)",
}
METRICS = ("mae", "rmse", "mape", "wape", "mase")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def summary_tables(summary: list[dict]) -> tuple[str, pd.DataFrame]:
    frame = pd.DataFrame(summary)
    rows = []
    for _, row in frame.sort_values(["horizon", "mode"]).iterrows():
        rows.append([
            MODE_LABELS.get(row["mode"], row["mode"]), str(int(row["horizon"])),
            str(int(row["n_series"])), _fmt(row.get("mae")), _fmt(row.get("mae_median")),
            _fmt(row.get("rmse")), _fmt(row.get("mape"), ".1f"), _fmt(row.get("wape"), ".1f"),
            _fmt(row.get("mase")),
        ])
    text = md_table(
        ["Режим", "Горизонт", "Рядов", "MAE", "MAE (медиана)", "RMSE", "MAPE, %", "WAPE, %", "MASE"],
        rows,
    )
    return text, frame


def agent_metrics_table(summary: list[dict]) -> str:
    rows = []
    for row in sorted(summary, key=lambda r: (r["horizon"], r["mode"])):
        agent = row.get("agent_metrics")
        if not agent:
            continue
        rows.append([
            MODE_LABELS.get(row["mode"], row["mode"]), str(int(row["horizon"])),
            _fmt(agent.get("mean_iterations"), ".2f"),
            f"{_fmt(agent.get('hitl_share'), '.0%') if agent.get('hitl_share') is not None else '—'}",
            _fmt(agent.get("fallback_share"), ".0%") if agent.get("fallback_share") is not None else "—",
            _fmt(agent.get("mean_tokens_in"), ".0f"), _fmt(agent.get("mean_tokens_out"), ".0f"),
            _fmt(agent.get("mean_duration_s"), ".2f"), _fmt(agent.get("mean_critic_score"), ".2f"),
        ])
    if not rows:
        return "_Агентные метрики недоступны (режим agents не запускался)._\n"
    return md_table(
        ["Режим", "Горизонт", "Итераций Supervisor", "Доля HITL", "Доля fallback",
         "Токены (вход)", "Токены (выход)", "Время на ряд, с", "Critic (среднее)"],
        rows,
    )


def fractions_table(summary: list[dict]) -> str:
    rows = []
    for row in sorted(summary, key=lambda r: (r["mode"], r["horizon"], r["data_fraction"])):
        rows.append([
            MODE_LABELS.get(row["mode"], row["mode"]), str(int(row["horizon"])),
            f"{row['data_fraction']:.1f}", str(int(row["n_series"])),
            _fmt(row.get("mae")), _fmt(row.get("rmse")), _fmt(row.get("mase")),
        ])
    return md_table(["Режим", "Горизонт", "Доля данных", "Рядов", "MAE", "RMSE", "MASE"], rows)


def comparisons_table(comparisons: list[dict]) -> str:
    rows = []
    for pair in comparisons:
        sig = pair.get("significance") or {}
        boot, wilcox, ttest = sig.get("bootstrap") or {}, sig.get("wilcoxon") or {}, sig.get("paired_ttest") or {}
        rows.append([
            " vs ".join(pair["models_compared"]), str(int(pair["horizon"])),
            f"{pair['data_fraction']:.1f}", str(int(pair["n_series"])), pair["winner"],
            _fmt(boot.get("mean_diff")),
            f"[{_fmt(boot.get('ci_low'))}; {_fmt(boot.get('ci_high'))}]",
            _fmt(wilcox.get("pvalue")), _fmt(ttest.get("pvalue")), _fmt(sig.get("dm_mean_pvalue")),
        ])
    if not rows:
        return "_Сравнения режимов недоступны (запущен один режим)._\n"
    return md_table(
        ["Сравнение", "Горизонт", "Доля данных", "Рядов", "Победитель", "Δ MAE", "95% CI (bootstrap)",
         "p (Wilcoxon)", "p (t-test)", "p (DM, среднее)"],
        rows,
    )


def _tokens_used(record: dict) -> int:
    """Токены LLM на ряд (0, если агентных метрик нет — pandas даёт NaN)."""
    agent = record.get("agent_metrics")
    if not isinstance(agent, dict):
        return 0
    return int(agent.get("tokens_in") or 0) + int(agent.get("tokens_out") or 0)


def cost_table(records: list[dict]) -> str:
    """Стоимость агентного подхода: время и токены LLM относительно классики."""
    frame = pd.DataFrame(records)
    if frame.empty or "metrics" not in frame.columns:
        return "_Нет данных._\n"
    frame = frame.dropna(subset=["metrics"])
    rows = []
    for (mode, horizon), group in frame.groupby(["mode", "horizon"]):
        duration = pd.to_numeric(group.get("duration_s"), errors="coerce").mean()
        tokens = sum(_tokens_used(record) for record in group.to_dict("records")) / len(group)
        mae = group["metrics"].apply(lambda m: m.get("mae")).mean()
        rows.append([MODE_LABELS.get(mode, mode), str(int(horizon)), _fmt(mae), _fmt(duration, ".2f"),
                     _fmt(tokens, ".0f")])
    return md_table(["Режим", "Горизонт", "MAE", "Время на ряд, с", "Токены LLM на ряд"], rows)


def build_plots(frame: pd.DataFrame, fractions: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. MAE по режимам для каждого горизонта
    if not frame.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for horizon, group in frame.groupby("horizon"):
            group = group.sort_values("mode")
            ax.bar([f"{m}\n(h={int(horizon)})" for m in group["mode"]],
                   pd.to_numeric(group["mae"], errors="coerce"), label=f"горизонт {int(horizon)}")
        ax.set_title("MAE по режимам")
        ax.set_ylabel("MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "mae_by_mode.png", dpi=150)
        plt.close(fig)
        written.append("plots/mae_by_mode.png")

        # 2. MASE относительно наивного прогноза (линия 1.0 = наивный)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for horizon, group in frame.groupby("horizon"):
            group = group.sort_values("mode")
            ax.plot(group["mode"], pd.to_numeric(group["mase"], errors="coerce"), marker="o",
                    label=f"горизонт {int(horizon)}")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="наивный прогноз")
        ax.set_title("MASE по режимам (ниже 1.0 — лучше наивного)")
        ax.set_ylabel("MASE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "mase_by_mode.png", dpi=150)
        plt.close(fig)
        written.append("plots/mase_by_mode.png")

    # 3. Точность vs доля обучающих данных
    if fractions is not None and not fractions.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (mode, horizon), group in fractions.groupby(["mode", "horizon"]):
            group = group.sort_values("data_fraction")
            ax.plot(group["data_fraction"], pd.to_numeric(group["mae"], errors="coerce"),
                    marker="o", label=f"{mode}, h={int(horizon)}")
        ax.set_title("MAE vs доля обучающих данных")
        ax.set_xlabel("Доля обучающих данных")
        ax.set_ylabel("MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "mae_vs_data_fraction.png", dpi=150)
        plt.close(fig)
        written.append("plots/mae_vs_data_fraction.png")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Таблицы и графики для отчёта")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--fractions-dir", type=Path, default=FRACTIONS_DIR)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "reports")
    parser.add_argument("--no-plots", action="store_true", help="только таблицы (без matplotlib)")
    args = parser.parse_args()

    summary = read_json(args.eval_dir / "summary.json")
    comparisons = read_json(args.eval_dir / "comparisons.json") or []
    records = read_jsonl(args.eval_dir / "results.jsonl")
    fractions_summary = read_json(args.fractions_dir / "summary.json")
    meta = read_json(args.eval_dir / "meta.json") or {}

    if summary is None:
        print(f"Нет результатов оценки: {args.eval_dir}. Запустите scripts/run_eval.py")
        return 1

    tables_dir = args.out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_md, frame = summary_tables(summary)
    sections = [
        ("summary.md", "# Точность по режимам и горизонтам\n\n", summary_md),
        ("agent_metrics.md", "# Агентные метрики\n\n", agent_metrics_table(summary)),
        ("comparisons.md", "# Статистическое сравнение режимов\n\n", comparisons_table(comparisons)),
        ("cost.md", "# Стоимость агентного подхода\n\n", cost_table(records)),
    ]
    if fractions_summary:
        sections.append(("fractions.md", "# Точность vs доля обучающих данных\n\n",
                         fractions_table(fractions_summary)))

    header = (
        f"<!-- Сгенерировано scripts/report_tables.py: {meta.get('n_records', '?')} записей, "
        f"{meta.get('n_windows', '?')} окон, seed={meta.get('seed', '?')} -->\n\n"
    )
    for name, title, table in sections:
        (tables_dir / name).write_text(header + title + table, encoding="utf-8")
        print(f"таблица: {tables_dir / name}")

    plots = [] if args.no_plots else build_plots(
        frame, pd.DataFrame(fractions_summary) if fractions_summary else None, args.out
    )
    for plot in plots:
        print(f"график: {args.out / plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
