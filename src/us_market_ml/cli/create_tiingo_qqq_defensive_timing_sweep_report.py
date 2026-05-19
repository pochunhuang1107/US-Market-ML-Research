"""Create the focused Tiingo-only QQQ defensive timing sweep report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_sweep import (
    DATASET_SOURCE_LABEL,
    DEFAULT_DATASET_GLOB,
    RESULTS_NAME,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_sweep import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SWEEP_OUTPUT_DIR,
)
from us_market_ml.qqq_timing import (
    BENCHMARK_STRATEGY_NAMES,
    DEFENSIVE_RESULT_COLUMNS,
    MATERIAL_DRAWDOWN_IMPROVEMENT,
)

DEFAULT_RESULTS_CSV = DEFAULT_SWEEP_OUTPUT_DIR / RESULTS_NAME
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_qqq_defensive_timing_sweep")

SUMMARY_NAME = "tiingo_qqq_defensive_timing_sweep_summary.csv"
DECISION_MATRIX_NAME = "tiingo_qqq_defensive_timing_sweep_decision_matrix.csv"
REPORT_NAME = "tiingo_qqq_defensive_timing_sweep_report.md"
OUTPUT_NAMES = (SUMMARY_NAME, DECISION_MATRIX_NAME, REPORT_NAME)

SUMMARY_COLUMNS: tuple[str, ...] = (
    *DEFENSIVE_RESULT_COLUMNS,
    "turnover_rate",
)

DECISION_COLUMNS: tuple[str, ...] = (
    *SUMMARY_COLUMNS,
    "decision",
    "decision_reason",
    "deeper_sweep_priority",
)

ACCEPTABLE_DRAWDOWN_TOLERANCE = 0.02
HIGH_AVERAGE_TURNOVER = 0.10
VERY_LOW_QQQ_RETURN_CAPTURE = 0.35


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingSweepReportData:
    """All tables and metadata used by the defensive timing sweep report."""

    results: pd.DataFrame
    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    dataset_source: str
    market_dataset_glob: str
    best_candidate: str | None


def run_tiingo_qqq_defensive_timing_sweep_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the focused QQQ defensive timing sweep report."""

    args = _parse_args(argv)
    report_data = build_tiingo_qqq_defensive_timing_sweep_report(
        results_csv=args.results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_qqq_defensive_timing_sweep_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo QQQ defensive timing sweep report")
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
        print(report_data.decision_matrix.loc[:, preview_cols].head(25).to_string(index=False))
    return 0


def build_tiingo_qqq_defensive_timing_sweep_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    market_dataset_glob: str = DEFAULT_DATASET_GLOB,
) -> TiingoQQQDefensiveTimingSweepReportData:
    """Read result rows and derive summary/decision tables."""

    results = _read_results(results_csv)
    summary = summarize_tiingo_qqq_defensive_timing_sweep_results(results)
    decision_matrix = build_tiingo_qqq_defensive_timing_sweep_decision_matrix(summary)
    best_candidate = _best_candidate_name(decision_matrix)
    return TiingoQQQDefensiveTimingSweepReportData(
        results=results,
        summary=summary,
        decision_matrix=decision_matrix,
        dataset_source=DATASET_SOURCE_LABEL,
        market_dataset_glob=market_dataset_glob,
        best_candidate=best_candidate,
    )


def summarize_tiingo_qqq_defensive_timing_sweep_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return one summary row per defensive timing configuration."""

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
    return summary.loc[:, SUMMARY_COLUMNS]


def build_tiingo_qqq_defensive_timing_sweep_decision_matrix(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the defensive timing sweep decision rules."""

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
                "total_return",
                "drawdown_improvement_vs_qqq",
                "return_to_drawdown",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_tiingo_qqq_defensive_timing_sweep_report(
    *,
    report_data: TiingoQQQDefensiveTimingSweepReportData,
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
        render_tiingo_qqq_defensive_timing_sweep_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_qqq_defensive_timing_sweep_markdown(
    report_data: TiingoQQQDefensiveTimingSweepReportData,
) -> str:
    """Render the defensive timing sweep Markdown report."""

    summary = report_data.summary
    decision = report_data.decision_matrix
    benchmarks = summary.loc[summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    timing = summary.loc[~summary["strategy_name"].isin(BENCHMARK_STRATEGY_NAMES)]
    best_return = timing.sort_values("total_return", ascending=False, kind="mergesort")
    best_drawdown = timing.sort_values("max_drawdown", ascending=False, kind="mergesort")
    best_rtd = timing.sort_values("return_to_drawdown", ascending=False, kind="mergesort")
    rejected = decision.loc[
        decision["decision"].isin(["reject", "too_defensive"])
        if not decision.empty
        else []
    ]
    defensive = decision.loc[
        decision["decision"].eq("defensive_candidate") if not decision.empty else []
    ]

    lines = [
        "# Tiingo QQQ Defensive Timing Sweep Report",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Baseline QQQ timing result recap",
        "",
        _baseline_recap(),
        "",
        "## 3. Sweep families and parameter ranges",
        "",
        _family_parameter_text(),
        "",
        "## 4. Benchmark comparison vs QQQ/SPY/cash",
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
        "## 5. Best total-return candidates",
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
        "## 6. Best drawdown candidates",
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
        "## 7. Best return-to-drawdown candidates",
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
        "## 8. Trade frequency / cost analysis",
        "",
        _markdown_table(
            timing.sort_values("transaction_count", ascending=True, kind="mergesort"),
            [
                "strategy_name",
                "family",
                "transaction_count",
                "turnover_rate",
                "average_turnover",
                "estimated_transaction_cost",
            ],
            max_rows=20,
        ),
        "",
        "## 9. Rejected/deprioritized candidates",
        "",
        _markdown_table(
            rejected,
            [
                "strategy_name",
                "decision",
                "decision_reason",
                "total_return",
                "drawdown_improvement_vs_qqq",
            ],
            max_rows=30,
        ),
        "",
        "## 10. Recommendation",
        "",
        _recommendation_text(report_data, defensive),
        "",
        "## 11. Caveats",
        "",
        (
            "Tiingo-only research data was used. This is research-only, uses no S3, "
            "does not export production signals, and does not alter the Market Data "
            "repo. QQQ buy-and-hold remains the key benchmark."
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
    total_return = _as_float(row.total_return)
    qqq_return = _as_float(row.qqq_buy_hold_total_return)
    max_drawdown = _as_float(row.max_drawdown)
    qqq_drawdown = _as_float(row.qqq_buy_hold_max_drawdown)
    average_turnover = _as_float(row.average_turnover)
    worse_drawdown = max_drawdown < qqq_drawdown - ACCEPTABLE_DRAWDOWN_TOLERANCE
    high_turnover = average_turnover > HIGH_AVERAGE_TURNOVER
    very_low_return = (
        np.isfinite(total_return)
        and np.isfinite(qqq_return)
        and total_return < qqq_return * VERY_LOW_QQQ_RETURN_CAPTURE
    )

    if beats_qqq_return and not worse_drawdown and not high_turnover:
        return (
            "best_candidate",
            "Beats QQQ total return with acceptable drawdown and turnover.",
            1,
        )
    if beats_qqq_return:
        return (
            "mixed_beats_qqq_with_tradeoff",
            "Beats QQQ return but has worse drawdown or elevated turnover.",
            2,
        )
    if very_low_return and material_drawdown:
        return (
            "too_defensive",
            "Materially lowers drawdown but captures too little QQQ upside.",
            8,
        )
    if beats_drawdown_adjusted or material_drawdown:
        return (
            "defensive_candidate",
            "Trails QQQ return but improves drawdown or drawdown-adjusted return.",
            3,
        )
    return (
        "reject",
        "Trails QQQ return and does not improve drawdown-adjusted return.",
        9,
    )


def _baseline_recap() -> str:
    return "\n".join(
        [
            "- QQQ buy-and-hold total return was approximately `3.7950`.",
            "- QQQ buy-and-hold max drawdown was approximately `-0.3512`.",
            (
                "- `qqq_sma50_sma200_cash` improved drawdown with very low "
                "turnover but trailed QQQ return."
            ),
            (
                "- `qqq_sma200_cash` materially improved drawdown and slightly "
                "improved return-to-drawdown in the baseline report."
            ),
            "- No QQQ timing baseline beat QQQ buy-and-hold on raw return.",
        ]
    )


def _family_parameter_text() -> str:
    return "\n".join(
        [
            (
                "- SMA price filter to cash: slow SMA 150/180/200/220/250, "
                "buffer -2% to +2%, min hold 0/5/10."
            ),
            (
                "- SMA fast/slow crossover to cash: fast 20/50/75/100, slow "
                "150/180/200/220/250, buffer -1%/0%/+1%, min hold 0/5/10."
            ),
            (
                "- SMA200-style defensive SPY variant: slow SMA 180/200/220/250, "
                "buffer -1%/0%/+1%, min hold 0/5/10."
            ),
            (
                "- Volatility guard to cash: positive return lookback plus realized "
                "volatility below a rolling percentile threshold."
            ),
            (
                "- Momentum + drawdown guard to cash: positive return lookback plus "
                "drawdown above -5%/-10%/-15%."
            ),
            "- Regime-score defensive variant: qqq_regime_score_v3-like count threshold 1/2/3.",
            "",
            (
                "All signals are computed from close data available on date T and "
                "applied starting the next trading day."
            ),
        ]
    )


def _executive_summary(report_data: TiingoQQQDefensiveTimingSweepReportData) -> str:
    if report_data.summary.empty:
        return "No defensive timing sweep rows were available; do not promote."
    best = report_data.best_candidate
    if best is not None:
        return (
            f"`{best}` is the best raw-return candidate in this defensive sweep. "
            "Treat it as research-only and verify it with a richer replay before any "
            "promotion discussion."
        )
    defensive_count = int((report_data.decision_matrix["decision"] == "defensive_candidate").sum())
    if defensive_count:
        return (
            f"No sweep configuration cleanly beats QQQ buy-and-hold on return, but "
            f"{defensive_count} defensive candidate(s) improve drawdown or "
            "return-to-drawdown. QQQ buy-and-hold remains the main hurdle."
        )
    return "No sweep configuration beats QQQ return or drawdown-adjusted return."


def _recommendation_text(
    report_data: TiingoQQQDefensiveTimingSweepReportData,
    defensive_rows: pd.DataFrame,
) -> str:
    best = report_data.best_candidate
    if best is not None:
        rich_replay = f"Run rich replay on `{best}` before any broader claims."
        deeper = f"Deepen only around `{best}` and close neighbors."
    elif not defensive_rows.empty:
        top_defensive = str(defensive_rows.iloc[0]["strategy_name"])
        rich_replay = f"Rich replay can wait unless `{top_defensive}` survives review."
        deeper = f"Run a narrower defensive QQQ timing sweep around `{top_defensive}`."
    else:
        rich_replay = "Do not run rich replay until a candidate clears this table."
        deeper = "Do not expand the QQQ timing sweep from these results."
    return "\n".join(
        [
            f"- Rich replay best QQQ timing candidate: {rich_replay}",
            f"- Deeper QQQ timing sweep: {deeper}",
            (
                "- QQQ + defensive rotation: consider only for defensive candidates "
                "with sane turnover."
            ),
            "- Feature v4: do not add benchmark_gap_features_v4 from this sweep.",
            "- tree_screen: do not run from this sweep.",
            "- MLP: do not run from this sweep.",
            "- Production promotion: do not promote or mark production-ready.",
        ]
    )


def _best_candidate_name(decision_matrix: pd.DataFrame) -> str | None:
    if decision_matrix.empty:
        return None
    best_rows = decision_matrix.loc[decision_matrix["decision"].eq("best_candidate")]
    if best_rows.empty:
        return None
    return str(best_rows.iloc[0]["strategy_name"])


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"QQQ defensive timing sweep results CSV does not exist: {path}")
    results = pd.read_csv(path)
    for column in DEFENSIVE_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    return results.loc[:, DEFENSIVE_RESULT_COLUMNS]


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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the focused Tiingo-only QQQ defensive timing sweep report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


create_tiingo_qqq_defensive_timing_sweep_report_from_args = (
    run_tiingo_qqq_defensive_timing_sweep_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_sweep_report_from_args())
