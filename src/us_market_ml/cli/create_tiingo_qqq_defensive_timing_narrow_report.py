"""Create the narrow Tiingo-only QQQ defensive timing sweep report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_narrow_sweep import (
    DATASET_SOURCE_LABEL,
    DEFAULT_DATASET_GLOB,
    RESULTS_NAME,
    SUBPERIOD_RESULTS_NAME,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_narrow_sweep import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SWEEP_OUTPUT_DIR,
)
from us_market_ml.qqq_timing import (
    BENCHMARK_STRATEGY_NAMES,
    DEFENSIVE_RESULT_COLUMNS,
    DEFENSIVE_SUBPERIOD_RESULT_COLUMNS,
    MATERIAL_DRAWDOWN_IMPROVEMENT,
)

DEFAULT_RESULTS_CSV = DEFAULT_SWEEP_OUTPUT_DIR / RESULTS_NAME
DEFAULT_SUBPERIOD_RESULTS_CSV = DEFAULT_SWEEP_OUTPUT_DIR / SUBPERIOD_RESULTS_NAME
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_qqq_defensive_timing_narrow_sweep")

SUMMARY_NAME = "tiingo_qqq_defensive_timing_narrow_summary.csv"
DECISION_MATRIX_NAME = "tiingo_qqq_defensive_timing_narrow_decision_matrix.csv"
SUBPERIOD_SUMMARY_NAME = "tiingo_qqq_defensive_timing_narrow_subperiod_summary.csv"
REPORT_NAME = "tiingo_qqq_defensive_timing_narrow_report.md"
OUTPUT_NAMES = (SUMMARY_NAME, DECISION_MATRIX_NAME, SUBPERIOD_SUMMARY_NAME, REPORT_NAME)

SUMMARY_COLUMNS: tuple[str, ...] = (
    *DEFENSIVE_RESULT_COLUMNS,
    "turnover_rate",
    "qqq_return_capture",
)

SUBPERIOD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "subperiod_count",
    "positive_strategy_vs_qqq_periods",
    "drawdown_improvement_periods",
    "severe_underperformance_periods",
    "avg_strategy_vs_qqq",
    "min_strategy_vs_qqq",
    "worst_subperiod_max_drawdown",
    "avg_drawdown_improvement_vs_qqq",
    "total_subperiod_transactions",
    "avg_exposure_rate_to_qqq",
    "robustness_score",
    "robustness_decision",
)

DECISION_COLUMNS: tuple[str, ...] = (
    *SUMMARY_COLUMNS,
    "subperiod_count",
    "positive_strategy_vs_qqq_periods",
    "drawdown_improvement_periods",
    "severe_underperformance_periods",
    "avg_strategy_vs_qqq",
    "min_strategy_vs_qqq",
    "avg_drawdown_improvement_vs_qqq",
    "robustness_score",
    "robustness_decision",
    "decision",
    "decision_reason",
    "recommendation_priority",
)

NEAR_QQQ_RETURN_GAP = -0.10
REASONABLE_TRANSACTION_COUNT = 50
REASONABLE_AVERAGE_TURNOVER = 0.05
TOO_DEFENSIVE_RETURN_CAPTURE = 0.65
SEVERE_SUBPERIOD_UNDERPERFORMANCE = -0.15


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingNarrowReportData:
    """All tables and metadata used by the narrow defensive timing report."""

    results: pd.DataFrame
    subperiod_results: pd.DataFrame
    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    subperiod_summary: pd.DataFrame
    dataset_source: str
    market_dataset_glob: str
    best_defensive_candidate: str | None


def run_tiingo_qqq_defensive_timing_narrow_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the narrow QQQ defensive timing sweep report."""

    args = _parse_args(argv)
    report_data = build_tiingo_qqq_defensive_timing_narrow_report(
        results_csv=args.results_csv,
        subperiod_results_csv=args.subperiod_results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_qqq_defensive_timing_narrow_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo QQQ defensive timing narrow sweep report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    print(f"subperiod results CSV: {args.subperiod_results_csv}")
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
            "return_to_drawdown",
            "transaction_count",
        ]
        print()
        print(report_data.decision_matrix.loc[:, preview_cols].head(25).to_string(index=False))
    return 0


def build_tiingo_qqq_defensive_timing_narrow_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    subperiod_results_csv: Path = DEFAULT_SUBPERIOD_RESULTS_CSV,
    market_dataset_glob: str = DEFAULT_DATASET_GLOB,
) -> TiingoQQQDefensiveTimingNarrowReportData:
    """Read sweep rows and derive summary, robustness, and decision tables."""

    results = _read_results(results_csv)
    subperiod_results = _read_subperiod_results(subperiod_results_csv)
    summary = summarize_tiingo_qqq_defensive_timing_narrow_results(results)
    subperiod_summary = summarize_tiingo_qqq_defensive_timing_narrow_subperiods(
        subperiod_results,
    )
    decision_matrix = build_tiingo_qqq_defensive_timing_narrow_decision_matrix(
        summary,
        subperiod_summary,
    )
    best_defensive_candidate = _best_defensive_candidate_name(decision_matrix)
    return TiingoQQQDefensiveTimingNarrowReportData(
        results=results,
        subperiod_results=subperiod_results,
        summary=summary,
        decision_matrix=decision_matrix,
        subperiod_summary=subperiod_summary,
        dataset_source=DATASET_SOURCE_LABEL,
        market_dataset_glob=market_dataset_glob,
        best_defensive_candidate=best_defensive_candidate,
    )


def summarize_tiingo_qqq_defensive_timing_narrow_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return one full-period summary row per strategy/config."""

    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = results.copy()
    for column in DEFENSIVE_RESULT_COLUMNS:
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
    summary["qqq_return_capture"] = (
        summary["total_return"] / summary["qqq_buy_hold_total_return"].replace(0, np.nan)
    )
    return summary.loc[:, SUMMARY_COLUMNS]


def summarize_tiingo_qqq_defensive_timing_narrow_subperiods(
    subperiod_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize robustness across non-full subperiods by strategy."""

    if subperiod_results.empty:
        return pd.DataFrame(columns=SUBPERIOD_SUMMARY_COLUMNS)

    subperiods = subperiod_results.copy()
    for column in DEFENSIVE_SUBPERIOD_RESULT_COLUMNS:
        if column not in subperiods.columns:
            subperiods[column] = np.nan
    for column in _numeric_subperiod_columns():
        subperiods[column] = pd.to_numeric(subperiods[column], errors="coerce")
    subperiods["drawdown_improvement_vs_qqq"] = (
        subperiods["max_drawdown"] - subperiods["qqq_max_drawdown"]
    )

    stress_periods = subperiods.loc[subperiods["period_name"].ne("full")].copy()
    if stress_periods.empty:
        stress_periods = subperiods.copy()

    rows: list[dict[str, object]] = []
    for strategy_name, group in stress_periods.groupby("strategy_name", sort=False):
        positive_return_periods = int((group["strategy_vs_qqq"] > 0.0).sum())
        drawdown_improvement_periods = int((group["drawdown_improvement_vs_qqq"] > 0.0).sum())
        severe_underperformance_periods = int(
            (group["strategy_vs_qqq"] <= SEVERE_SUBPERIOD_UNDERPERFORMANCE).sum()
        )
        robustness_score = (
            positive_return_periods
            + drawdown_improvement_periods
            - severe_underperformance_periods
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "subperiod_count": int(len(group)),
                "positive_strategy_vs_qqq_periods": positive_return_periods,
                "drawdown_improvement_periods": drawdown_improvement_periods,
                "severe_underperformance_periods": severe_underperformance_periods,
                "avg_strategy_vs_qqq": float(group["strategy_vs_qqq"].mean()),
                "min_strategy_vs_qqq": float(group["strategy_vs_qqq"].min()),
                "worst_subperiod_max_drawdown": float(group["max_drawdown"].min()),
                "avg_drawdown_improvement_vs_qqq": float(
                    group["drawdown_improvement_vs_qqq"].mean()
                ),
                "total_subperiod_transactions": int(group["transaction_count"].sum()),
                "avg_exposure_rate_to_qqq": float(group["exposure_rate_to_qqq"].mean()),
                "robustness_score": int(robustness_score),
                "robustness_decision": _robustness_decision(
                    subperiod_count=int(len(group)),
                    positive_return_periods=positive_return_periods,
                    drawdown_improvement_periods=drawdown_improvement_periods,
                    severe_underperformance_periods=severe_underperformance_periods,
                ),
            }
        )

    return (
        pd.DataFrame(rows, columns=SUBPERIOD_SUMMARY_COLUMNS)
        .sort_values(
            [
                "robustness_score",
                "drawdown_improvement_periods",
                "avg_strategy_vs_qqq",
            ],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_tiingo_qqq_defensive_timing_narrow_decision_matrix(
    summary: pd.DataFrame,
    subperiod_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the narrow sweep defensive decision rules."""

    if summary.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)

    merged = summary.merge(subperiod_summary, on="strategy_name", how="left")
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        decision, reason, priority = _decision_for_strategy(row)
        rows.append(
            {
                **{column: getattr(row, column, np.nan) for column in SUMMARY_COLUMNS},
                **{
                    column: getattr(row, column, np.nan)
                    for column in SUBPERIOD_SUMMARY_COLUMNS
                    if column != "strategy_name"
                },
                "decision": decision,
                "decision_reason": reason,
                "recommendation_priority": priority,
            }
        )
    return (
        pd.DataFrame(rows, columns=DECISION_COLUMNS)
        .sort_values(
            [
                "recommendation_priority",
                "total_return",
                "return_to_drawdown",
                "drawdown_improvement_vs_qqq",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_tiingo_qqq_defensive_timing_narrow_report(
    *,
    report_data: TiingoQQQDefensiveTimingNarrowReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        SUBPERIOD_SUMMARY_NAME: output_dir / SUBPERIOD_SUMMARY_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }
    report_data.summary.to_csv(outputs[SUMMARY_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    report_data.subperiod_summary.to_csv(outputs[SUBPERIOD_SUMMARY_NAME], index=False)
    outputs[REPORT_NAME].write_text(
        render_tiingo_qqq_defensive_timing_narrow_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_qqq_defensive_timing_narrow_markdown(
    report_data: TiingoQQQDefensiveTimingNarrowReportData,
) -> str:
    """Render the narrow defensive timing sweep Markdown report."""

    summary = report_data.summary
    decision = report_data.decision_matrix
    subperiod_summary = report_data.subperiod_summary
    benchmarks = summary.loc[summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    timing = summary.loc[~summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    best_return = timing.sort_values("total_return", ascending=False, kind="mergesort")
    best_drawdown = timing.sort_values("max_drawdown", ascending=False, kind="mergesort")
    best_rtd = timing.sort_values("return_to_drawdown", ascending=False, kind="mergesort")
    low_turnover = timing.sort_values(
        ["transaction_count", "total_return"],
        ascending=[True, False],
        kind="mergesort",
    )
    robustness = subperiod_summary.loc[
        ~subperiod_summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)
    ]
    robustness = robustness.sort_values(
        ["robustness_score", "avg_strategy_vs_qqq"],
        ascending=[False, False],
        kind="mergesort",
    )

    lines = [
        "# Tiingo QQQ Defensive Timing Narrow Sweep Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Why this is a narrow sweep",
        "",
        _why_narrow_text(),
        "",
        "## 3. Previous best candidate recap",
        "",
        _previous_candidate_recap(),
        "",
        "## 4. Parameter ranges",
        "",
        _parameter_ranges_text(),
        "",
        "## 5. Full-period benchmark comparison",
        "",
        _markdown_table(
            benchmarks,
            [
                "strategy_name",
                "total_return",
                "annualized_return",
                "max_drawdown",
                "return_to_drawdown",
            ],
        ),
        "",
        "## 6. Best raw return candidates",
        "",
        _markdown_table(
            best_return,
            [
                "strategy_name",
                "family",
                "risk_off_asset",
                "total_return",
                "strategy_vs_qqq",
                "max_drawdown",
                "transaction_count",
            ],
            max_rows=15,
        ),
        "",
        "## 7. Best drawdown candidates",
        "",
        _markdown_table(
            best_drawdown,
            [
                "strategy_name",
                "family",
                "risk_off_asset",
                "total_return",
                "max_drawdown",
                "drawdown_improvement_vs_qqq",
                "transaction_count",
            ],
            max_rows=15,
        ),
        "",
        "## 8. Best return-to-drawdown candidates",
        "",
        _markdown_table(
            best_rtd,
            [
                "strategy_name",
                "family",
                "risk_off_asset",
                "return_to_drawdown",
                "qqq_buy_hold_return_to_drawdown",
                "total_return",
                "max_drawdown",
            ],
            max_rows=15,
        ),
        "",
        "## 9. Subperiod robustness",
        "",
        _markdown_table(
            robustness,
            [
                "strategy_name",
                "robustness_score",
                "positive_strategy_vs_qqq_periods",
                "drawdown_improvement_periods",
                "severe_underperformance_periods",
                "avg_strategy_vs_qqq",
                "min_strategy_vs_qqq",
                "robustness_decision",
            ],
            max_rows=20,
        ),
        "",
        "## 10. Turnover / transaction cost analysis",
        "",
        "Best low-turnover candidates:",
        "",
        _markdown_table(
            low_turnover,
            [
                "strategy_name",
                "family",
                "transaction_count",
                "turnover_rate",
                "average_turnover",
                "estimated_transaction_cost",
                "total_return",
            ],
            max_rows=20,
        ),
        "",
        "## 11. Candidate decision matrix",
        "",
        _markdown_table(
            decision,
            [
                "strategy_name",
                "decision",
                "decision_reason",
                "total_return",
                "strategy_vs_qqq",
                "drawdown_improvement_vs_qqq",
                "return_to_drawdown",
                "robustness_decision",
            ],
            max_rows=40,
        ),
        "",
        "## 12. Recommendation",
        "",
        _recommendation_text(report_data),
        "",
        "## 13. Caveats",
        "",
        _caveats_text(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _decision_for_strategy(row: object) -> tuple[str, str, int]:
    strategy_name = str(row.strategy_name)
    if strategy_name in BENCHMARK_STRATEGY_NAMES:
        return "benchmark_reference", "Reference series only; not a candidate.", 90

    strategy_vs_qqq = _as_float(row.strategy_vs_qqq)
    total_return = _as_float(row.total_return)
    qqq_return = _as_float(row.qqq_buy_hold_total_return)
    drawdown_improvement = _as_float(row.drawdown_improvement_vs_qqq)
    return_to_drawdown = _as_float(row.return_to_drawdown)
    qqq_return_to_drawdown = _as_float(row.qqq_buy_hold_return_to_drawdown)
    transaction_count = _as_float(row.transaction_count)
    average_turnover = _as_float(row.average_turnover)
    qqq_return_capture = _as_float(row.qqq_return_capture)
    severe_underperformance_periods = _as_float(row.severe_underperformance_periods)
    positive_periods = _as_float(row.positive_strategy_vs_qqq_periods)
    robustness_decision = str(getattr(row, "robustness_decision", ""))

    near_qqq_return = (
        np.isfinite(strategy_vs_qqq)
        and np.isfinite(total_return)
        and np.isfinite(qqq_return)
        and np.isfinite(qqq_return_capture)
        and qqq_return_capture >= 0.90
        and (strategy_vs_qqq >= NEAR_QQQ_RETURN_GAP or total_return >= qqq_return * 0.95)
    )
    improves_drawdown = np.isfinite(drawdown_improvement) and drawdown_improvement > 0.0
    reasonable_turnover = (
        np.isfinite(transaction_count)
        and transaction_count <= REASONABLE_TRANSACTION_COUNT
        and np.isfinite(average_turnover)
        and average_turnover <= REASONABLE_AVERAGE_TURNOVER
    )
    improves_rtd = (
        np.isfinite(return_to_drawdown)
        and np.isfinite(qqq_return_to_drawdown)
        and return_to_drawdown > qqq_return_to_drawdown
    )
    material_drawdown = (
        np.isfinite(drawdown_improvement)
        and drawdown_improvement >= MATERIAL_DRAWDOWN_IMPROVEMENT
    )
    too_defensive = (
        np.isfinite(qqq_return_capture)
        and qqq_return_capture < TOO_DEFENSIVE_RETURN_CAPTURE
        and improves_drawdown
    )
    one_period_only = (
        np.isfinite(positive_periods)
        and positive_periods <= 1
        and np.isfinite(severe_underperformance_periods)
        and severe_underperformance_periods >= 2
    )

    if one_period_only or robustness_decision == "mixed":
        return (
            "mixed",
            "Subperiod results are concentrated and include severe underperformance.",
            5,
        )
    if near_qqq_return and improves_drawdown and reasonable_turnover:
        return (
            "best_defensive_candidate",
            "Nearly matches QQQ return while improving drawdown with reasonable turnover.",
            1,
        )
    if too_defensive:
        return (
            "too_defensive",
            "Drawdown improves, but the strategy gives up too much QQQ upside.",
            8,
        )
    if improves_rtd and material_drawdown:
        return (
            "defensive_candidate",
            "Trails QQQ return but materially improves drawdown and return-to-drawdown.",
            3,
        )
    if improves_drawdown or improves_rtd:
        return (
            "mixed",
            "Improves one defensive metric but lacks a clean full-period profile.",
            6,
        )
    return (
        "reject",
        "Does not beat QQQ return or improve defensive metrics enough.",
        9,
    )


def _robustness_decision(
    *,
    subperiod_count: int,
    positive_return_periods: int,
    drawdown_improvement_periods: int,
    severe_underperformance_periods: int,
) -> str:
    if subperiod_count == 0:
        return "unknown"
    if severe_underperformance_periods >= 2 and positive_return_periods <= 1:
        return "mixed"
    if drawdown_improvement_periods >= max(3, subperiod_count - 1):
        return "robust_defensive"
    if positive_return_periods >= max(2, subperiod_count // 2):
        return "robust_return"
    return "weak"


def _executive_summary(report_data: TiingoQQQDefensiveTimingNarrowReportData) -> str:
    if report_data.summary.empty:
        return "No narrow defensive timing sweep rows were available; do not promote."
    best = report_data.best_defensive_candidate
    if best is not None:
        return (
            f"`{best}` is the best defensive candidate from this narrow sweep. "
            "It should remain research-only and be checked in rich replay before "
            "any broader use."
        )
    defensive_count = int(
        (report_data.decision_matrix["decision"] == "defensive_candidate").sum()
    )
    if defensive_count:
        return (
            "No narrow configuration cleanly matches QQQ return with lower drawdown, "
            f"but {defensive_count} candidate(s) improve return-to-drawdown and "
            "drawdown. QQQ buy-and-hold remains the benchmark."
        )
    return "No narrow sweep configuration clears the defensive candidate rules."


def _why_narrow_text() -> str:
    return (
        "The grid only explores SMA neighborhoods around the prior best raw-return "
        "SPY defensive candidate and the prior best return-to-drawdown cash "
        "candidate, plus nearby low-turnover crossover variants. It deliberately "
        "does not run tree_screen, MLP, feature v4, S3, or production signal export."
    )


def _previous_candidate_recap() -> str:
    return "\n".join(
        [
            (
                "- Best total-return defensive candidate: "
                "`qqq_sma180_buffer_p0_hold0_to_spy`, total return about `3.7010`, "
                "QQQ buy-hold about `3.7950`, strategy vs QQQ about `-0.0940`, "
                "max drawdown about `-0.3324`, transactions `23`."
            ),
            (
                "- Best return-to-drawdown defensive candidate: "
                "`qqq_sma200_buffer_p1_hold10_cash`, total return about `3.1342`, "
                "max drawdown about `-0.1840`, return-to-drawdown about `17.0342`, "
                "transactions `11`."
            ),
            (
                "- Interpretation: no QQQ timing strategy cleanly beat QQQ buy-and-hold "
                "on raw return, but several improved drawdown or return-to-drawdown."
            ),
        ]
    )


def _parameter_ranges_text() -> str:
    return "\n".join(
        [
            (
                "- Total-return SPY defensive family: QQQ close > slow SMA times "
                "`1 + buffer`; otherwise SPY. Slow SMA 165 to 205, buffers -0.5% "
                "to +1%, min holds 0/3/5/10."
            ),
            (
                "- Drawdown-adjusted cash defensive family: QQQ close > slow SMA "
                "times `1 + buffer`; otherwise cash. Slow SMA 180/190/200/210/220, "
                "buffers -2% to +2%, min holds 5/10/15."
            ),
            (
                "- Low-turnover crossover cash family: fast SMA > slow SMA times "
                "`1 + buffer`; otherwise cash. Fast SMA 60/75/90/100, slow SMA "
                "180/200/220/240, fast < slow only, buffers -1.5% to 0%, min holds "
                "0/5/10."
            ),
            (
                "- Timing convention: signals are computed from close data available "
                "on date T and apply starting the next trading day."
            ),
        ]
    )


def _recommendation_text(report_data: TiingoQQQDefensiveTimingNarrowReportData) -> str:
    best = report_data.best_defensive_candidate
    if best is not None:
        rich_replay = f"run on `{best}` before making any stronger claim."
        deeper = f"only deepen around `{best}` and close SMA/buffer neighbors."
        rotation = "consider research-only QQQ + defensive rotation using the top candidates."
    else:
        rich_replay = "defer unless a candidate survives manual review."
        deeper = "do not broaden the sweep until the decision matrix has a clear winner."
        rotation = "keep as research-only; no allocation change follows from this table."
    return "\n".join(
        [
            f"- Rich replay best QQQ timing candidate: {rich_replay}",
            f"- Deeper QQQ timing sweep: {deeper}",
            f"- QQQ + defensive rotation: {rotation}",
            "- Feature v4: do not add benchmark_gap_features_v4 from this sweep.",
            "- tree_screen: do not run from this sweep.",
            "- MLP: do not run from this sweep.",
            "- Production promotion: do not promote or mark production-ready.",
        ]
    )


def _caveats_text() -> str:
    return "\n".join(
        [
            "- Tiingo-only input data.",
            "- Research-only output.",
            "- No S3/cloud paths were used.",
            "- No production signal export was created or updated.",
            "- QQQ buy-and-hold remains the benchmark.",
        ]
    )


def _best_defensive_candidate_name(decision_matrix: pd.DataFrame) -> str | None:
    if decision_matrix.empty:
        return None
    best_rows = decision_matrix.loc[decision_matrix["decision"].eq("best_defensive_candidate")]
    if best_rows.empty:
        return None
    return str(best_rows.iloc[0]["strategy_name"])


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"QQQ defensive timing narrow results CSV does not exist: {path}")
    results = pd.read_csv(path)
    for column in DEFENSIVE_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    return results.loc[:, DEFENSIVE_RESULT_COLUMNS]


def _read_subperiod_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"QQQ defensive timing narrow subperiod results CSV does not exist: {path}"
        )
    subperiod_results = pd.read_csv(path)
    for column in DEFENSIVE_SUBPERIOD_RESULT_COLUMNS:
        if column not in subperiod_results.columns:
            subperiod_results[column] = np.nan
    return subperiod_results.loc[:, DEFENSIVE_SUBPERIOD_RESULT_COLUMNS]


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
        for column in DEFENSIVE_RESULT_COLUMNS
        if column
        not in {
            "strategy_name",
            "family",
            "risk_on_asset",
            "risk_off_asset",
            "signal_name",
            "start_date",
            "end_date",
            "beats_qqq_return",
            "beats_qqq_return_to_drawdown",
            "materially_improves_qqq_drawdown",
        }
    ]


def _numeric_subperiod_columns() -> list[str]:
    return [
        column
        for column in DEFENSIVE_SUBPERIOD_RESULT_COLUMNS
        if column not in {"strategy_name", "period_name", "period_start", "period_end"}
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the narrow Tiingo-only QQQ defensive timing sweep report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--subperiod-results-csv",
        type=Path,
        default=DEFAULT_SUBPERIOD_RESULTS_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


create_tiingo_qqq_defensive_timing_narrow_report_from_args = (
    run_tiingo_qqq_defensive_timing_narrow_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_narrow_report_from_args())
