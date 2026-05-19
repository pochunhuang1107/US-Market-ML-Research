"""Create the compact Tiingo-only QQQ timing baseline report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_timing_baseline_study import (
    DATASET_SOURCE_LABEL,
    DEFAULT_DATASET_GLOB,
    RESULTS_NAME,
)
from us_market_ml.cli.run_tiingo_qqq_timing_baseline_study import (
    DEFAULT_OUTPUT_DIR as DEFAULT_STUDY_OUTPUT_DIR,
)
from us_market_ml.qqq_timing import BENCHMARK_STRATEGY_NAMES, RESULT_COLUMNS

DEFAULT_RESULTS_CSV = DEFAULT_STUDY_OUTPUT_DIR / RESULTS_NAME
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_qqq_timing_baseline_study")

SUMMARY_NAME = "tiingo_qqq_timing_baseline_summary.csv"
DECISION_MATRIX_NAME = "tiingo_qqq_timing_baseline_decision_matrix.csv"
REPORT_NAME = "tiingo_qqq_timing_baseline_report.md"
OUTPUT_NAMES = (SUMMARY_NAME, DECISION_MATRIX_NAME, REPORT_NAME)

SUMMARY_COLUMNS: tuple[str, ...] = (
    *RESULT_COLUMNS,
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "materially_improves_qqq_drawdown",
    "turnover_rate",
)

DECISION_COLUMNS: tuple[str, ...] = (
    *SUMMARY_COLUMNS,
    "decision",
    "decision_reason",
    "deeper_sweep_priority",
)

MATERIAL_DRAWDOWN_IMPROVEMENT = 0.05
ACCEPTABLE_DRAWDOWN_TOLERANCE = 0.02
HIGH_AVERAGE_TURNOVER = 0.10


@dataclass(frozen=True)
class TiingoQQQTimingReportData:
    """All tables and metadata used by the QQQ timing report."""

    results: pd.DataFrame
    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    dataset_source: str
    market_dataset_glob: str
    best_candidate: str | None


def run_tiingo_qqq_timing_baseline_report_from_args(argv: list[str] | None = None) -> int:
    """Build and write the QQQ timing baseline report."""

    args = _parse_args(argv)
    report_data = build_tiingo_qqq_timing_baseline_report(
        results_csv=args.results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_qqq_timing_baseline_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo QQQ timing baseline report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    print(f"market dataset: {args.market_dataset_glob}")
    for output_name in OUTPUT_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    if not report_data.decision_matrix.empty:
        preview_cols = [
            "strategy_name",
            "decision",
            "total_return",
            "strategy_vs_qqq",
            "max_drawdown",
            "drawdown_improvement_vs_qqq",
            "transaction_count",
        ]
        print()
        print(report_data.decision_matrix.loc[:, preview_cols].head(20).to_string(index=False))
    return 0


def build_tiingo_qqq_timing_baseline_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    market_dataset_glob: str = DEFAULT_DATASET_GLOB,
) -> TiingoQQQTimingReportData:
    """Read result rows and derive summary/decision tables."""

    results = _read_results(results_csv)
    summary = summarize_tiingo_qqq_timing_baseline_results(results)
    decision_matrix = build_tiingo_qqq_timing_baseline_decision_matrix(summary)
    best_candidate = _best_candidate_name(decision_matrix)
    return TiingoQQQTimingReportData(
        results=results,
        summary=summary,
        decision_matrix=decision_matrix,
        dataset_source=DATASET_SOURCE_LABEL,
        market_dataset_glob=market_dataset_glob,
        best_candidate=best_candidate,
    )


def summarize_tiingo_qqq_timing_baseline_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per QQQ timing strategy."""

    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = results.copy()
    for column in RESULT_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    for column in _numeric_result_columns():
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    summary["beats_qqq_return"] = summary["strategy_vs_qqq"] > 0.0
    summary["beats_qqq_return_to_drawdown"] = (
        summary["return_to_drawdown"] > summary["qqq_buy_hold_return_to_drawdown"]
    )
    summary["materially_improves_qqq_drawdown"] = (
        summary["drawdown_improvement_vs_qqq"] >= MATERIAL_DRAWDOWN_IMPROVEMENT
    )
    summary["turnover_rate"] = summary["transaction_count"] / summary["trading_days"].replace(
        0,
        np.nan,
    )
    return summary.loc[:, SUMMARY_COLUMNS]


def build_tiingo_qqq_timing_baseline_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the QQQ timing research decision rules."""

    if summary.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)

    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        decision, reason, priority = _decision_for_strategy(row)
        rows.append(
            {
                **{column: getattr(row, column) for column in SUMMARY_COLUMNS},
                "decision": decision,
                "decision_reason": reason,
                "deeper_sweep_priority": priority,
            }
        )
    return (
        pd.DataFrame(rows, columns=DECISION_COLUMNS)
        .sort_values(
            [
                "deeper_sweep_priority",
                "strategy_vs_qqq",
                "drawdown_improvement_vs_qqq",
                "return_to_drawdown",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_tiingo_qqq_timing_baseline_report(
    *,
    report_data: TiingoQQQTimingReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }
    report_data.summary.to_csv(outputs[SUMMARY_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    outputs[REPORT_NAME].write_text(
        render_tiingo_qqq_timing_baseline_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_qqq_timing_baseline_markdown(
    report_data: TiingoQQQTimingReportData,
) -> str:
    """Render the QQQ timing baseline Markdown report."""

    summary = report_data.summary
    decision = report_data.decision_matrix
    benchmarks = summary.loc[summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    timing = summary.loc[~summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    risk_off = _risk_off_comparison(timing)
    candidates = decision.loc[
        decision["decision"].isin(
            [
                "best_qqq_timing_candidate_for_deeper_sweep",
                "defensive_candidate",
                "mixed_beats_qqq_with_tradeoff",
            ]
        )
    ]
    rejected = decision.loc[decision["decision"].str.startswith("reject", na=False)]

    lines = [
        "# Tiingo QQQ Timing Baseline Study Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Why QQQ-only timing path",
        "",
        (
            "The pure weighted-allocation sweep did not narrow the QQQ gap versus the "
            "v3 equal-weight rotation baseline. This study pivots to a direct QQQ "
            "risk-on/risk-off question: when should the strategy hold QQQ, and when "
            "should it be defensive or cash-like?"
        ),
        "",
        "## 3. Benchmarks: QQQ buy-and-hold, SPY buy-and-hold, cash",
        "",
        _markdown_table(
            benchmarks,
            [
                "strategy_name",
                "total_return",
                "annualized_return",
                "max_drawdown",
                "daily_volatility",
                "sharpe_like",
            ],
        ),
        "",
        "## 4. QQQ timing strategies tested",
        "",
        _markdown_table(
            timing,
            [
                "strategy_name",
                "risk_off_asset",
                "signal_name",
                "exposure_rate_to_qqq",
                "total_return",
                "strategy_vs_qqq",
            ],
        ),
        "",
        "## 5. Risk-off asset comparison",
        "",
        _markdown_table(
            risk_off,
            [
                "risk_off_asset",
                "strategy_count",
                "average_total_return",
                "average_strategy_vs_qqq",
                "average_max_drawdown",
                "average_transaction_count",
            ],
        ),
        "",
        "## 6. QQQ underperformance / outperformance",
        "",
        _markdown_table(
            summary.sort_values("strategy_vs_qqq", ascending=False, kind="mergesort"),
            [
                "strategy_name",
                "total_return",
                "qqq_buy_hold_total_return",
                "strategy_vs_qqq",
                "qqq_underperformance",
            ],
            max_rows=20,
        ),
        "",
        "## 7. Drawdown comparison",
        "",
        _markdown_table(
            summary.sort_values(
                "drawdown_improvement_vs_qqq",
                ascending=False,
                kind="mergesort",
            ),
            [
                "strategy_name",
                "max_drawdown",
                "qqq_buy_hold_max_drawdown",
                "drawdown_improvement_vs_qqq",
                "return_to_drawdown",
                "qqq_buy_hold_return_to_drawdown",
            ],
            max_rows=20,
        ),
        "",
        "## 8. Trade frequency and transaction costs",
        "",
        _markdown_table(
            summary.sort_values("transaction_count", ascending=True, kind="mergesort"),
            [
                "strategy_name",
                "transaction_count",
                "turnover_rate",
                "average_turnover",
                "estimated_transaction_cost",
            ],
            max_rows=20,
        ),
        "",
        "## 9. Best candidates",
        "",
        _markdown_table(
            candidates,
            [
                "strategy_name",
                "decision",
                "decision_reason",
                "total_return",
                "strategy_vs_qqq",
                "drawdown_improvement_vs_qqq",
            ],
        ),
        "",
        "## 10. Rejected/deprioritized candidates",
        "",
        _markdown_table(
            rejected,
            [
                "strategy_name",
                "decision",
                "decision_reason",
                "strategy_vs_qqq",
                "drawdown_improvement_vs_qqq",
            ],
        ),
        "",
        "## 11. Recommendation",
        "",
        _recommendation_text(report_data),
        "",
        "## 12. Caveats",
        "",
        (
            "Tiingo-only research data was used. This is not production-ready, does "
            "not use S3, does not export production signals, and does not alter the "
            "Market Data repo. QQQ buy-and-hold remains the key benchmark."
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _decision_for_strategy(row: object) -> tuple[str, str, int]:
    strategy_name = str(row.strategy_name)
    if strategy_name in BENCHMARK_STRATEGY_NAMES:
        return "benchmark_reference", "Reference series only; not a candidate.", 90

    beats_qqq_return = bool(row.beats_qqq_return)
    beats_drawdown_adjusted = bool(row.beats_qqq_return_to_drawdown)
    material_drawdown = bool(row.materially_improves_qqq_drawdown)
    max_drawdown = _as_float(row.max_drawdown)
    qqq_drawdown = _as_float(row.qqq_buy_hold_max_drawdown)
    average_turnover = _as_float(row.average_turnover)
    worse_drawdown = max_drawdown < qqq_drawdown - ACCEPTABLE_DRAWDOWN_TOLERANCE
    high_turnover = average_turnover > HIGH_AVERAGE_TURNOVER

    if beats_qqq_return and not worse_drawdown and not high_turnover:
        return (
            "best_qqq_timing_candidate_for_deeper_sweep",
            "Beats QQQ buy-and-hold return with acceptable drawdown and turnover.",
            1,
        )
    if beats_qqq_return:
        return (
            "mixed_beats_qqq_with_tradeoff",
            "Beats QQQ return but has worse drawdown or elevated turnover.",
            2,
        )
    if beats_drawdown_adjusted or material_drawdown:
        return (
            "defensive_candidate",
            "Trails QQQ return but improves drawdown or drawdown-adjusted return.",
            3,
        )
    return (
        "reject_no_qqq_return_or_drawdown_adjusted_edge",
        "Does not beat QQQ return or drawdown-adjusted return.",
        9,
    )


def _risk_off_comparison(timing: pd.DataFrame) -> pd.DataFrame:
    if timing.empty:
        return pd.DataFrame(
            columns=[
                "risk_off_asset",
                "strategy_count",
                "average_total_return",
                "average_strategy_vs_qqq",
                "average_max_drawdown",
                "average_transaction_count",
            ]
        )
    return (
        timing.groupby("risk_off_asset", dropna=False, sort=False)
        .agg(
            strategy_count=("strategy_name", "count"),
            average_total_return=("total_return", "mean"),
            average_strategy_vs_qqq=("strategy_vs_qqq", "mean"),
            average_max_drawdown=("max_drawdown", "mean"),
            average_transaction_count=("transaction_count", "mean"),
        )
        .reset_index()
    )


def _executive_summary(report_data: TiingoQQQTimingReportData) -> str:
    if report_data.summary.empty:
        return "No QQQ timing result rows were available; do not promote."
    best = report_data.best_candidate
    if best is not None:
        return (
            f"`{best}` is the best QQQ timing candidate for a deeper sweep. "
            "It is still research-only and not production-ready."
        )
    defensive_count = int((report_data.decision_matrix["decision"] == "defensive_candidate").sum())
    if defensive_count:
        return (
            f"No timing strategy cleanly beats QQQ buy-and-hold on return, but "
            f"{defensive_count} defensive candidate(s) improve drawdown or "
            "drawdown-adjusted return. Do not promote."
        )
    return "No timing strategy beats QQQ buy-and-hold on return or drawdown-adjusted return."


def _recommendation_text(report_data: TiingoQQQTimingReportData) -> str:
    best = report_data.best_candidate
    defensive_rows = (
        report_data.decision_matrix.loc[
            report_data.decision_matrix["decision"].eq("defensive_candidate")
        ]
        if not report_data.decision_matrix.empty
        else pd.DataFrame()
    )
    if best is not None:
        deeper = f"Run a deeper QQQ timing sweep around `{best}`."
    elif not defensive_rows.empty:
        top_defensive = str(defensive_rows.iloc[0]["strategy_name"])
        deeper = (
            f"Run a deeper defensive QQQ timing sweep around `{top_defensive}` and "
            "compare it honestly to QQQ buy-and-hold."
        )
    else:
        deeper = "Do not run a deeper QQQ timing sweep until a candidate clears QQQ better."
    defensive = (
        "Consider QQQ + defensive rotation only for candidates labeled defensive or mixed."
        if not report_data.decision_matrix.empty
        else "Defer QQQ + defensive rotation until baseline rows exist."
    )
    return "\n".join(
        [
            f"- Deeper QQQ timing sweep: {deeper}",
            f"- QQQ + defensive rotation: {defensive}",
            "- Feature v4: defer; do not add benchmark_gap_features_v4 from this baseline.",
            "- tree_screen: do not run from this baseline.",
            "- MLP: do not run from this baseline.",
            "- Production promotion: do not promote or mark production-ready.",
        ]
    )


def _best_candidate_name(decision_matrix: pd.DataFrame) -> str | None:
    if decision_matrix.empty:
        return None
    best_rows = decision_matrix.loc[
        decision_matrix["decision"].eq("best_qqq_timing_candidate_for_deeper_sweep")
    ]
    if best_rows.empty:
        return None
    return str(best_rows.iloc[0]["strategy_name"])


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"QQQ timing results CSV does not exist: {path}")
    results = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    return results.loc[:, RESULT_COLUMNS]


def _markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    max_rows: int | None = None,
) -> str:
    if df.empty:
        return "_No rows._"
    table = df.loc[:, [column for column in columns if column in df.columns]].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    formatted = table.map(_format_cell)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    numeric = _as_float(value)
    if np.isfinite(numeric) and not isinstance(value, str):
        return f"{numeric:.4f}"
    return str(value)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric_result_columns() -> list[str]:
    return [
        column
        for column in RESULT_COLUMNS
        if column
        not in {
            "strategy_name",
            "risk_on_asset",
            "risk_off_asset",
            "signal_name",
            "start_date",
            "end_date",
        }
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the compact Tiingo-only QQQ timing baseline report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


create_tiingo_qqq_timing_baseline_report_from_args = (
    run_tiingo_qqq_timing_baseline_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_timing_baseline_report_from_args())
