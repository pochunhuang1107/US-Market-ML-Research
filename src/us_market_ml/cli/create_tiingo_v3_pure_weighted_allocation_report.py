"""Create the Tiingo benchmark-gap v3 pure weighted-allocation study report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_modern_baseline_sweep import DEFAULT_DATASET_GLOB
from us_market_ml.cli.run_tiingo_v3_pure_weighted_allocation_study import (
    BENCHMARK_GAP_V3_FEATURE_SET_VERSION,
    DATASET_SOURCE_LABEL,
    LABEL_HORIZON_DAYS,
    MIN_HOLDING_DAYS,
    MODEL_TYPE,
    RESULTS_NAME,
    SWITCH_MARGIN,
    TARGET_COL,
    TOP_K,
)
from us_market_ml.cli.run_tiingo_v3_pure_weighted_allocation_study import (
    DEFAULT_OUTPUT_DIR as DEFAULT_STUDY_OUTPUT_DIR,
)

DEFAULT_RESULTS_CSV = DEFAULT_STUDY_OUTPUT_DIR / RESULTS_NAME
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_v3_pure_weighted_allocation_study")

SUMMARY_PATH_NAME = "tiingo_v3_pure_weighted_allocation_summary.csv"
DECISION_MATRIX_PATH_NAME = "tiingo_v3_pure_weighted_allocation_decision_matrix.csv"
REPORT_PATH_NAME = "tiingo_v3_pure_weighted_allocation_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

BASELINE_RULE_NAME = "equal_weight_top5_baseline"

SUMMARY_COLUMNS = [
    "allocation_rule_name",
    "fold_count",
    "average_total_return",
    "median_total_return",
    "positive_window_rate",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_total_transaction_cost",
    "average_mean_daily_turnover",
    "spy_average_total_return",
    "qqq_average_total_return",
    "average_strategy_vs_spy",
    "average_strategy_vs_qqq",
    "spy_underperformance_rate",
    "qqq_underperformance_rate",
    "delta_vs_equal_weight_average_return",
    "delta_vs_equal_weight_strategy_vs_spy",
    "delta_vs_equal_weight_strategy_vs_qqq",
    "delta_vs_equal_weight_worst_drawdown",
    "delta_vs_equal_weight_turnover",
]

DECISION_COLUMNS = [
    *SUMMARY_COLUMNS,
    "decision",
    "decision_reason",
    "rich_replay_priority",
]


@dataclass(frozen=True)
class TiingoV3PureWeightedAllocationReportData:
    """All tables used by the pure weighted-allocation report."""

    results: pd.DataFrame
    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    dataset_source: str
    market_dataset_glob: str
    best_allocation_rule_name: str | None


def run_tiingo_v3_pure_weighted_allocation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the pure weighted-allocation report."""

    args = _parse_args(argv)
    report_data = build_tiingo_v3_pure_weighted_allocation_report(
        results_csv=args.results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_v3_pure_weighted_allocation_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo v3 pure weighted-allocation study report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    print(f"market dataset: {args.market_dataset_glob}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.decision_matrix.head(20).to_string(index=False))
    return 0


def build_tiingo_v3_pure_weighted_allocation_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    market_dataset_glob: str = DEFAULT_DATASET_GLOB,
) -> TiingoV3PureWeightedAllocationReportData:
    """Read result rows and derive allocation summary/decision tables."""

    results = _read_results(results_csv)
    summary = summarize_tiingo_v3_pure_weighted_allocation_results(results)
    decision_matrix = build_tiingo_v3_pure_weighted_allocation_decision_matrix(summary)
    best_allocation_rule_name = _best_allocation_rule_name(decision_matrix)
    dataset_source = (
        str(results["dataset_source"].dropna().iloc[0])
        if "dataset_source" in results.columns and not results.empty
        else DATASET_SOURCE_LABEL
    )
    return TiingoV3PureWeightedAllocationReportData(
        results=results,
        summary=summary,
        decision_matrix=decision_matrix,
        dataset_source=dataset_source,
        market_dataset_glob=market_dataset_glob,
        best_allocation_rule_name=best_allocation_rule_name,
    )


def summarize_tiingo_v3_pure_weighted_allocation_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold-level allocation rows into report metrics."""

    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frame = results.copy()
    numeric_cols = [
        "strategy_total_return",
        "strategy_sharpe_like",
        "strategy_max_drawdown",
        "total_transaction_cost",
        "mean_daily_turnover",
        "spy_total_return",
        "qqq_total_return",
        "strategy_vs_spy",
        "strategy_vs_qqq",
    ]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    summary = (
        frame.groupby("allocation_rule_name", sort=False)
        .agg(
            fold_count=("fold", "nunique"),
            average_total_return=("strategy_total_return", "mean"),
            median_total_return=("strategy_total_return", "median"),
            positive_window_rate=(
                "strategy_total_return",
                lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean()),
            ),
            average_sharpe_like=("strategy_sharpe_like", "mean"),
            average_max_drawdown=("strategy_max_drawdown", "mean"),
            worst_max_drawdown=("strategy_max_drawdown", "min"),
            average_total_transaction_cost=("total_transaction_cost", "mean"),
            average_mean_daily_turnover=("mean_daily_turnover", "mean"),
            spy_average_total_return=("spy_total_return", "mean"),
            qqq_average_total_return=("qqq_total_return", "mean"),
            average_strategy_vs_spy=("strategy_vs_spy", "mean"),
            average_strategy_vs_qqq=("strategy_vs_qqq", "mean"),
            spy_underperformance_rate=(
                "strategy_vs_spy",
                lambda values: float((pd.to_numeric(values, errors="coerce") < 0).mean()),
            ),
            qqq_underperformance_rate=(
                "strategy_vs_qqq",
                lambda values: float((pd.to_numeric(values, errors="coerce") < 0).mean()),
            ),
        )
        .reset_index()
    )

    baseline = _baseline_summary_row(summary)
    for column, baseline_column in (
        ("delta_vs_equal_weight_average_return", "average_total_return"),
        ("delta_vs_equal_weight_strategy_vs_spy", "average_strategy_vs_spy"),
        ("delta_vs_equal_weight_strategy_vs_qqq", "average_strategy_vs_qqq"),
        ("delta_vs_equal_weight_worst_drawdown", "worst_max_drawdown"),
        ("delta_vs_equal_weight_turnover", "average_mean_daily_turnover"),
    ):
        baseline_value = _row_value(baseline, baseline_column)
        summary[column] = pd.to_numeric(summary[baseline_column], errors="coerce") - baseline_value
    return summary.loc[:, SUMMARY_COLUMNS]


def build_tiingo_v3_pure_weighted_allocation_decision_matrix(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Apply research decision logic to allocation summary rows."""

    if summary.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    baseline = _baseline_summary_row(summary)
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        decision, reason, priority = _decision_for_rule(row, baseline)
        rows.append(
            {
                **{column: getattr(row, column) for column in SUMMARY_COLUMNS},
                "decision": decision,
                "decision_reason": reason,
                "rich_replay_priority": priority,
            }
        )
    return (
        pd.DataFrame(rows, columns=DECISION_COLUMNS)
        .sort_values(
            [
                "rich_replay_priority",
                "delta_vs_equal_weight_strategy_vs_qqq",
                "average_strategy_vs_qqq",
                "average_total_return",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_tiingo_v3_pure_weighted_allocation_report(
    *,
    report_data: TiingoV3PureWeightedAllocationReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_v3_pure_weighted_allocation_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_v3_pure_weighted_allocation_markdown(
    report_data: TiingoV3PureWeightedAllocationReportData,
) -> str:
    """Render the pure weighted-allocation study Markdown report."""

    baseline = _baseline_summary_row(report_data.summary)
    best = report_data.best_allocation_rule_name
    qqq_improvers = _qqq_gap_reduction_table(report_data.summary)
    tradeoffs = _drawdown_turnover_table(report_data.summary)
    candidates = report_data.decision_matrix.loc[
        report_data.decision_matrix["decision"].isin(
            [
                "keep_for_rich_replay",
                "mixed_beats_qqq_with_tradeoff",
                "mixed_return_up_underperformance_worse",
                "watchlist_narrows_qqq_gap",
            ]
        )
    ]
    rejected = report_data.decision_matrix.loc[
        report_data.decision_matrix["decision"].str.startswith("reject", na=False)
    ]
    no_qqq_improvement = (
        report_data.summary["delta_vs_equal_weight_strategy_vs_qqq"].dropna().max() <= 0.0
        if not report_data.summary.empty
        else True
    )
    rich_replay_line = (
        f"Run a rich replay for `{best}`."
        if best is not None
        else "No pure allocation rule clears the rich-replay bar yet."
    )
    qqq_only_line = (
        "Prioritize QQQ-only trading strategy research if no pure rule narrows the QQQ gap."
        if no_qqq_improvement
        else "Keep QQQ-only trading strategy research as the next comparison path."
    )
    feature_v4_line = (
        "Defer feature v4 until this pure allocation readout is reviewed."
        if not no_qqq_improvement
        else "Feature v4 can be a later research path if QQQ-only research also stalls."
    )

    lines = [
        "# Tiingo V3 Pure Weighted-Allocation Study Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Why core-overlay path is paused",
        "",
        (
            "The QQQ core 30 overlay was diagnostically useful, but it hard-coded a "
            "benchmark exposure that does not match the desired pure rotation style. "
            "This report excludes QQQ-core, SPY-core, benchmark floors, and production "
            "promotion recommendations."
        ),
        "",
        "## 3. Current v3 equal-weight baseline",
        "",
        _baseline_text(baseline),
        "",
        "## 4. Pure allocation rules tested",
        "",
        _markdown_table(
            report_data.summary,
            [
                "allocation_rule_name",
                "fold_count",
                "average_total_return",
                "average_strategy_vs_spy",
                "average_strategy_vs_qqq",
                "worst_max_drawdown",
                "average_mean_daily_turnover",
            ],
        ),
        "",
        "## 5. SPY/QQQ benchmark comparison",
        "",
        _markdown_table(
            report_data.summary,
            [
                "allocation_rule_name",
                "spy_average_total_return",
                "qqq_average_total_return",
                "average_strategy_vs_spy",
                "average_strategy_vs_qqq",
                "spy_underperformance_rate",
                "qqq_underperformance_rate",
            ],
        ),
        "",
        "## 6. QQQ gap reduction",
        "",
        _markdown_table(
            qqq_improvers,
            [
                "allocation_rule_name",
                "average_strategy_vs_qqq",
                "delta_vs_equal_weight_strategy_vs_qqq",
                "qqq_underperformance_rate",
            ],
        ),
        "",
        "## 7. Drawdown and turnover trade-off",
        "",
        _markdown_table(
            tradeoffs,
            [
                "allocation_rule_name",
                "average_max_drawdown",
                "worst_max_drawdown",
                "delta_vs_equal_weight_worst_drawdown",
                "average_mean_daily_turnover",
                "delta_vs_equal_weight_turnover",
                "average_total_transaction_cost",
            ],
        ),
        "",
        "## 8. Best pure allocation candidates",
        "",
        _markdown_table(
            candidates,
            [
                "allocation_rule_name",
                "decision",
                "decision_reason",
                "average_strategy_vs_spy",
                "delta_vs_equal_weight_strategy_vs_qqq",
                "delta_vs_equal_weight_worst_drawdown",
                "delta_vs_equal_weight_turnover",
            ],
        ),
        "",
        "## 9. Rejected/deprioritized rules",
        "",
        _markdown_table(
            rejected,
            [
                "allocation_rule_name",
                "decision",
                "decision_reason",
                "delta_vs_equal_weight_strategy_vs_qqq",
                "delta_vs_equal_weight_worst_drawdown",
                "delta_vs_equal_weight_turnover",
            ],
        ),
        "",
        "## 10. Recommendation",
        "",
        f"- Rich replay best pure allocation candidate: {rich_replay_line}",
        f"- QQQ-only trading strategy: {qqq_only_line}",
        f"- Feature v4: {feature_v4_line}",
        "- tree_screen: Do not run in this pass.",
        "- MLP: Do not run in this pass.",
        (
            "- Production promotion: Do not mark production-ready and do not export "
            "production signals."
        ),
        (
            "- QQQ-core sizing: Do not recommend QQQ-core sizing from this pure "
            "allocation study."
        ),
        "",
        "## 11. Caveats",
        "",
        (
            f"- Dataset source is `{report_data.dataset_source}` from "
            f"`{report_data.market_dataset_glob}`."
        ),
        "- This is Tiingo-only and research-only.",
        "- No production signal export was produced.",
        "- QQQ remains the key benchmark because the baseline still trails QQQ on average.",
        (
            f"- Candidate lock: `{MODEL_TYPE}` / top{TOP_K} / hold{MIN_HOLDING_DAYS} "
            f"/ margin{SWITCH_MARGIN:g} / `{BENCHMARK_GAP_V3_FEATURE_SET_VERSION}`."
        ),
        f"- Target remains `{TARGET_COL}` with a {LABEL_HORIZON_DAYS}-day label horizon.",
        "",
    ]
    return "\n".join(lines)


def _decision_for_rule(row: object, baseline: pd.Series) -> tuple[str, str, int]:
    rule_name = str(row.allocation_rule_name)
    if rule_name == BASELINE_RULE_NAME:
        return "baseline_reference", "Current selected-candidate equal-weight behavior.", 50

    qqq_delta = _as_float(row.delta_vs_equal_weight_strategy_vs_qqq)
    return_delta = _as_float(row.delta_vs_equal_weight_average_return)
    drawdown_delta = _as_float(row.delta_vs_equal_weight_worst_drawdown)
    turnover_delta = _as_float(row.delta_vs_equal_weight_turnover)
    avg_vs_spy = _as_float(row.average_strategy_vs_spy)
    avg_vs_qqq = _as_float(row.average_strategy_vs_qqq)
    qqq_underperformance = _as_float(row.qqq_underperformance_rate)
    baseline_turnover = _row_value(baseline, "average_mean_daily_turnover")
    baseline_qqq_underperformance = _row_value(baseline, "qqq_underperformance_rate")

    materially_worse_drawdown = np.isfinite(drawdown_delta) and drawdown_delta < -0.02
    turnover_threshold = max(0.05, abs(baseline_turnover) * 0.25)
    materially_worse_turnover = (
        np.isfinite(turnover_delta) and turnover_delta > turnover_threshold
    )
    qqq_underperformance_worse = (
        np.isfinite(qqq_underperformance)
        and np.isfinite(baseline_qqq_underperformance)
        and qqq_underperformance > baseline_qqq_underperformance + 1.0e-12
    )

    if not np.isfinite(qqq_delta) or qqq_delta <= 0.0:
        return "reject_no_qqq_gap_improvement", "Does not narrow the QQQ gap.", 90
    if avg_vs_qqq > 0.0 and (materially_worse_drawdown or materially_worse_turnover):
        return (
            "mixed_beats_qqq_with_tradeoff",
            "Beats QQQ on average but worsens drawdown or turnover.",
            30,
        )
    if return_delta > 0.0 and qqq_underperformance_worse:
        return (
            "mixed_return_up_underperformance_worse",
            "Improves return but worsens the QQQ underperformance rate.",
            35,
        )
    if avg_vs_spy > 0.0 and qqq_delta > 0.0 and not materially_worse_drawdown:
        if not materially_worse_turnover:
            return (
                "keep_for_rich_replay",
                "Beats SPY and narrows QQQ gap without material drawdown or turnover damage.",
                10,
            )
    if materially_worse_drawdown or materially_worse_turnover:
        return (
            "mixed_drawdown_turnover_tradeoff",
            "Narrows QQQ gap but worsens drawdown or turnover.",
            40,
        )
    return (
        "watchlist_narrows_qqq_gap",
        "Narrows QQQ gap but needs stronger benchmark evidence.",
        60,
    )


def _executive_summary(report_data: TiingoV3PureWeightedAllocationReportData) -> str:
    if report_data.summary.empty:
        return "No pure allocation result rows were available."
    baseline = _baseline_summary_row(report_data.summary)
    best = report_data.best_allocation_rule_name
    baseline_vs_spy = _row_value(baseline, "average_strategy_vs_spy")
    baseline_vs_qqq = _row_value(baseline, "average_strategy_vs_qqq")
    if best is None:
        return (
            "No pure allocation rule improves the QQQ gap while preserving the baseline "
            f"profile. The equal-weight v3 baseline averages {_fmt(baseline_vs_spy)} "
            f"vs SPY and {_fmt(baseline_vs_qqq)} vs QQQ."
        )
    best_row = report_data.summary.loc[
        report_data.summary["allocation_rule_name"].eq(best)
    ].iloc[0]
    return (
        f"`{best}` is the best pure allocation candidate. It changes the QQQ gap by "
        f"{_fmt(best_row['delta_vs_equal_weight_strategy_vs_qqq'])} versus equal weight "
        f"with worst-drawdown delta {_fmt(best_row['delta_vs_equal_weight_worst_drawdown'])} "
        f"and turnover delta {_fmt(best_row['delta_vs_equal_weight_turnover'])}."
    )


def _baseline_text(baseline: pd.Series) -> str:
    if baseline.empty:
        return "The equal-weight baseline row is missing from the result set."
    return (
        f"`{BASELINE_RULE_NAME}` covers {int(baseline['fold_count'])} folds with "
        f"average return {_fmt(baseline['average_total_return'])}, SPY gap "
        f"{_fmt(baseline['average_strategy_vs_spy'])}, QQQ gap "
        f"{_fmt(baseline['average_strategy_vs_qqq'])}, worst drawdown "
        f"{_fmt(baseline['worst_max_drawdown'])}, and mean daily turnover "
        f"{_fmt(baseline['average_mean_daily_turnover'])}."
    )


def _best_allocation_rule_name(decision_matrix: pd.DataFrame) -> str | None:
    if decision_matrix.empty:
        return None
    eligible = decision_matrix.loc[
        ~decision_matrix["decision"].isin(["baseline_reference"])
        & ~decision_matrix["decision"].str.startswith("reject", na=False)
        & (pd.to_numeric(decision_matrix["delta_vs_equal_weight_strategy_vs_qqq"]) > 0.0)
    ]
    if eligible.empty:
        return None
    return str(eligible.iloc[0]["allocation_rule_name"])


def _qqq_gap_reduction_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return summary.sort_values(
        ["delta_vs_equal_weight_strategy_vs_qqq", "average_strategy_vs_qqq"],
        ascending=[False, False],
        kind="mergesort",
    )


def _drawdown_turnover_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return summary.sort_values(
        ["delta_vs_equal_weight_worst_drawdown", "average_mean_daily_turnover"],
        ascending=[False, True],
        kind="mergesort",
    )


def _baseline_summary_row(summary: pd.DataFrame) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    baseline = summary.loc[summary["allocation_rule_name"].eq(BASELINE_RULE_NAME)]
    if baseline.empty:
        return pd.Series(dtype=object)
    return baseline.iloc[0]


def _row_value(row: pd.Series, column: str) -> float:
    if row.empty or column not in row:
        return np.nan
    return _as_float(row[column])


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pure weighted-allocation results CSV does not exist: {path}")
    return pd.read_csv(path)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    lines = [
        "| " + " | ".join(visible.columns) + " |",
        "| " + " | ".join("---" for _ in visible.columns) + " |",
    ]
    for _, row in visible.iterrows():
        cells = [_format_cell(row[column]) for column in visible.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _fmt(value: object) -> str:
    numeric = _as_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo v3 pure weighted-allocation study report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


create_tiingo_v3_pure_weighted_allocation_report_from_args = (
    run_tiingo_v3_pure_weighted_allocation_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_v3_pure_weighted_allocation_report_from_args())
