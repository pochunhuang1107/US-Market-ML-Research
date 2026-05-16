"""Report on completed longer-window sliding-window shortlist validation results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_sliding_window_benchmark_comparison_report import (
    calculate_buy_and_hold_window_return,
    parse_comma_separated_symbols,
)
from us_market_ml.cli.run_sliding_window_shortlist_benchmark_report import (
    REQUIRED_BENCHMARK_SYMBOLS,
    SHORTLIST_SPECS,
)
from us_market_ml.cli.run_sliding_window_stability_report import (
    parse_sliding_window_fold_name,
)
from us_market_ml.data import read_local_market_data

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/sliding_window_longer_window_shortlist_study_results.csv"
)
DEFAULT_FAILED_KEYS = Path(
    "artifacts/sweeps/sliding_window_longer_window_shortlist_study_failed_keys.txt"
)
DEFAULT_FULL_MARKET_DATA = Path("artifacts/cache/universe_full_2018_2026.parquet")
DEFAULT_BROAD_MACRO_MARKET_DATA = Path(
    "artifacts/cache/universe_broad_macro_2018_2026.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/sliding_window_longer_window_shortlist_study"
)

WINDOW_METRICS_PATH_NAME = "longer_window_shortlist_window_metrics.csv"
CANDIDATE_SUMMARY_PATH_NAME = "longer_window_shortlist_candidate_summary.csv"
BY_TEST_WINDOW_PATH_NAME = "longer_window_shortlist_by_test_window.csv"
BY_CANDIDATE_AND_TEST_WINDOW_PATH_NAME = (
    "longer_window_shortlist_by_candidate_and_test_window.csv"
)
BY_YEAR_PATH_NAME = "longer_window_shortlist_by_year.csv"
RELATIVE_WINDOWS_PATH_NAME = "longer_window_shortlist_relative_windows.csv"
REPORT_PATH_NAME = "longer_window_shortlist_report.md"

OUTPUT_PATH_NAMES = (
    WINDOW_METRICS_PATH_NAME,
    CANDIDATE_SUMMARY_PATH_NAME,
    BY_TEST_WINDOW_PATH_NAME,
    BY_CANDIDATE_AND_TEST_WINDOW_PATH_NAME,
    BY_YEAR_PATH_NAME,
    RELATIVE_WINDOWS_PATH_NAME,
    REPORT_PATH_NAME,
)

REQUIRED_RESULTS_COLUMNS = [
    "fold",
    "split",
    "universe",
    "model_type",
    "top_k",
    "allocation_mode",
    "transaction_cost_bps",
    "total_return",
    "sharpe_like",
    "max_drawdown",
]

WINDOW_METRIC_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "universe",
    "model_type",
    "top_k",
    "allocation_mode",
    "transaction_cost_bps",
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "test_year",
    "strategy_total_return",
    "strategy_sharpe_like",
    "strategy_max_drawdown",
    "spy_total_return",
    "qqq_total_return",
    "strategy_minus_spy",
    "strategy_minus_qqq",
    "strategy_beats_spy",
    "strategy_beats_qqq",
]

SUMMARY_METRIC_COLUMNS = [
    "window_count",
    "mean_total_return",
    "median_total_return",
    "positive_window_rate",
    "mean_sharpe_like",
    "mean_max_drawdown",
    "worst_max_drawdown",
    "spy_mean_total_return",
    "qqq_mean_total_return",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_spy",
    "mean_strategy_minus_qqq",
]

CANDIDATE_SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "universe",
    "model_type",
    "top_k",
    "allocation_mode",
    "transaction_cost_bps",
    *SUMMARY_METRIC_COLUMNS,
    "balanced_score",
    "benchmark_stability_score",
]

BY_TEST_WINDOW_COLUMNS = [
    "test_window_months",
    "window_count",
    "candidate_count",
    "mean_total_return",
    "median_total_return",
    "positive_window_rate",
    "mean_sharpe_like",
    "mean_max_drawdown",
    "worst_max_drawdown",
    "spy_mean_total_return",
    "qqq_mean_total_return",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_spy",
    "mean_strategy_minus_qqq",
]

BY_CANDIDATE_AND_TEST_WINDOW_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "test_window_months",
    *SUMMARY_METRIC_COLUMNS,
    "balanced_score",
    "benchmark_stability_score",
]

BY_YEAR_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "test_year",
    "window_count",
    "mean_total_return",
    "spy_mean_total_return",
    "qqq_mean_total_return",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_spy",
    "mean_strategy_minus_qqq",
]

PREVIOUS_BENCHMARK_RELATIVE_LEADER = (
    "broad_macro:linear_regression:top1:equal_weight:cost10"
)
PREVIOUS_RISK_ADJUSTED_LEADER = "broad_macro:random_forest:top3:equal_weight:cost10"
PREVIOUS_DEFENSIVE_CANDIDATE = "broad_macro:ridge:top5:equal_weight:cost10"
PREVIOUS_FULL_REFERENCE = "full:ridge:top3:equal_weight:cost10"


@dataclass(frozen=True)
class SlidingWindowLongerWindowShortlistReportData:
    """All table outputs produced by the longer-window shortlist report."""

    window_metrics: pd.DataFrame
    candidate_summary: pd.DataFrame
    by_test_window: pd.DataFrame
    by_candidate_and_test_window: pd.DataFrame
    by_year: pd.DataFrame
    relative_windows: pd.DataFrame
    failed_key_count: int
    completed_key_count: int
    result_row_count: int
    test_row_count: int
    shortlist_test_row_count: int
    benchmark_symbols: tuple[str, ...]


def run_sliding_window_longer_window_shortlist_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build a report from completed longer-window shortlist validation rows."""

    args = _parse_args(argv)
    report_data = build_sliding_window_longer_window_shortlist_report(
        results_csv=args.results_csv,
        failed_keys=args.failed_keys,
        full_market_data=args.full_market_data,
        broad_macro_market_data=args.broad_macro_market_data,
        benchmark_symbols=args.benchmark_symbols,
        top_n=args.top_n,
    )

    outputs = write_sliding_window_longer_window_shortlist_report(
        report_data=report_data,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
    )

    print("US Market ML longer-window sliding-window shortlist report")
    print(f"results CSV: {args.results_csv}")
    print(f"failed keys: {args.failed_keys}")
    print(f"full market data: {args.full_market_data}")
    print(f"broad_macro market data: {args.broad_macro_market_data}")
    print(f"benchmarks: {','.join(report_data.benchmark_symbols)}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.candidate_summary.head(args.top_n).to_string(index=False))
    return 0


def build_sliding_window_longer_window_shortlist_report(
    *,
    results_csv: Path,
    failed_keys: Path | None = None,
    full_market_data: Path | pd.DataFrame = DEFAULT_FULL_MARKET_DATA,
    broad_macro_market_data: Path | pd.DataFrame = DEFAULT_BROAD_MACRO_MARKET_DATA,
    benchmark_symbols: Sequence[str] = REQUIRED_BENCHMARK_SYMBOLS,
    top_n: int = 10,
) -> SlidingWindowLongerWindowShortlistReportData:
    """Read completed longer-window shortlist results and return report tables."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    resolved_benchmark_symbols = _normalize_benchmark_symbols(benchmark_symbols)
    market_data_by_universe = {
        "full": _load_market_data(full_market_data),
        "broad_macro": _load_market_data(broad_macro_market_data),
    }

    results = pd.read_csv(results_csv)
    _validate_results_columns(results)
    test_rows = results.loc[results["split"] == "test"].copy().reset_index(drop=True)
    shortlist_rows = _build_shortlist_test_rows(test_rows)
    if shortlist_rows.empty:
        raise ValueError("No split == 'test' rows matched the shortlist candidate specs.")

    window_metrics = _build_window_metrics(
        shortlist_rows,
        market_data_by_universe=market_data_by_universe,
        benchmark_symbols=resolved_benchmark_symbols,
    )
    candidate_summary = _add_scores(_build_candidate_summary(window_metrics))
    candidate_summary = _sort_candidate_summary(candidate_summary).reset_index(drop=True)
    by_test_window = _build_by_test_window(window_metrics)
    by_candidate_and_test_window = _add_scores(
        _build_by_candidate_and_test_window(window_metrics)
    )
    by_candidate_and_test_window = _sort_candidate_test_window_summary(
        by_candidate_and_test_window
    ).reset_index(drop=True)
    by_year = _build_by_year(window_metrics)
    relative_windows = _build_relative_windows(window_metrics)

    failed_key_count = _count_text_file_lines(failed_keys)
    completed_key_count = int(results["run_key"].nunique()) if "run_key" in results else 0

    return SlidingWindowLongerWindowShortlistReportData(
        window_metrics=window_metrics,
        candidate_summary=candidate_summary,
        by_test_window=by_test_window,
        by_candidate_and_test_window=by_candidate_and_test_window,
        by_year=by_year,
        relative_windows=relative_windows,
        failed_key_count=failed_key_count,
        completed_key_count=completed_key_count,
        result_row_count=len(results),
        test_row_count=len(test_rows),
        shortlist_test_row_count=len(shortlist_rows),
        benchmark_symbols=resolved_benchmark_symbols,
    )


def write_sliding_window_longer_window_shortlist_report(
    *,
    report_data: SlidingWindowLongerWindowShortlistReportData,
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, Path]:
    """Write report CSVs and markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        WINDOW_METRICS_PATH_NAME: output_dir / WINDOW_METRICS_PATH_NAME,
        CANDIDATE_SUMMARY_PATH_NAME: output_dir / CANDIDATE_SUMMARY_PATH_NAME,
        BY_TEST_WINDOW_PATH_NAME: output_dir / BY_TEST_WINDOW_PATH_NAME,
        BY_CANDIDATE_AND_TEST_WINDOW_PATH_NAME: (
            output_dir / BY_CANDIDATE_AND_TEST_WINDOW_PATH_NAME
        ),
        BY_YEAR_PATH_NAME: output_dir / BY_YEAR_PATH_NAME,
        RELATIVE_WINDOWS_PATH_NAME: output_dir / RELATIVE_WINDOWS_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }

    report_data.window_metrics.to_csv(outputs[WINDOW_METRICS_PATH_NAME], index=False)
    report_data.candidate_summary.to_csv(
        outputs[CANDIDATE_SUMMARY_PATH_NAME],
        index=False,
    )
    report_data.by_test_window.to_csv(outputs[BY_TEST_WINDOW_PATH_NAME], index=False)
    report_data.by_candidate_and_test_window.to_csv(
        outputs[BY_CANDIDATE_AND_TEST_WINDOW_PATH_NAME],
        index=False,
    )
    report_data.by_year.to_csv(outputs[BY_YEAR_PATH_NAME], index=False)
    report_data.relative_windows.to_csv(
        outputs[RELATIVE_WINDOWS_PATH_NAME],
        index=False,
    )
    outputs[REPORT_PATH_NAME].write_text(
        render_sliding_window_longer_window_shortlist_markdown(
            report_data,
            top_n=top_n,
        ),
        encoding="utf-8",
    )
    return outputs


def render_sliding_window_longer_window_shortlist_markdown(
    report_data: SlidingWindowLongerWindowShortlistReportData,
    *,
    top_n: int = 10,
) -> str:
    """Render the longer-window shortlist report as Markdown."""

    period_start = report_data.window_metrics["test_start"].min()
    period_end = report_data.window_metrics["test_end"].max()
    benchmark_ranking = _build_benchmark_ranking(report_data.candidate_summary).head(top_n)
    year_breakdown = report_data.by_year.head(top_n)
    worst_relative = report_data.relative_windows.head(top_n)

    lines = [
        "# Longer-Window Sliding-Window Shortlist Report",
        "",
        "This report validates the completed shortlist over 6-month and 12-month "
        "rolling test windows. It uses only `split == \"test\"` rows, computes "
        "SPY and QQQ buy-and-hold returns over each matching test window, and "
        "does not rerun sweeps or change model training.",
        "",
        "## Completion Status",
        "",
        f"- Result rows: {report_data.result_row_count}",
        f"- Test rows read: {report_data.test_row_count}",
        f"- Shortlist test rows used: {report_data.shortlist_test_row_count}",
        f"- Completed configs: {report_data.completed_key_count}",
        f"- Failed configs: {report_data.failed_key_count}",
        "- Rolling test period covered: "
        f"{_format_date(period_start)} to {_format_date(period_end)}",
        f"- Benchmarks: `{', '.join(report_data.benchmark_symbols)}`",
        "",
        "## Candidate Shortlist",
        "",
        _markdown_table(_format_shortlist_specs_for_markdown()),
        "",
        "## Overall Candidate Summary",
        "",
        _markdown_table(_format_candidate_summary_for_markdown(report_data.candidate_summary)),
        "",
        "## 6-Month vs 12-Month Comparison",
        "",
        _markdown_table(_format_by_test_window_for_markdown(report_data.by_test_window)),
        "",
        "## Candidate-by-Test-Window Comparison",
        "",
        _markdown_table(
            _format_candidate_test_window_for_markdown(
                report_data.by_candidate_and_test_window.head(top_n * 2)
            )
        ),
        "",
        "## Benchmark-Relative Ranking",
        "",
        _markdown_table(_format_benchmark_ranking_for_markdown(benchmark_ranking)),
        "",
        "## Year Breakdown",
        "",
        _markdown_table(_format_by_year_for_markdown(year_breakdown)),
        "",
        "## Worst Relative Windows",
        "",
        _markdown_table(_format_windows_for_markdown(worst_relative)),
        "",
        "## Final Recommendation",
        "",
        _final_recommendation(report_data),
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a report from completed longer-window sliding-window "
            "shortlist validation results."
        )
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--failed-keys", type=Path, default=DEFAULT_FAILED_KEYS)
    parser.add_argument(
        "--full-market-data",
        type=Path,
        default=DEFAULT_FULL_MARKET_DATA,
    )
    parser.add_argument(
        "--broad-macro-market-data",
        type=Path,
        default=DEFAULT_BROAD_MACRO_MARKET_DATA,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark-symbols",
        type=parse_comma_separated_symbols,
        default=parse_comma_separated_symbols("SPY,QQQ"),
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be positive.")
    return args


def _load_market_data(market_data: Path | pd.DataFrame) -> pd.DataFrame:
    frame = (
        read_local_market_data(market_data)
        if not isinstance(market_data, pd.DataFrame)
        else market_data.copy()
    )
    _validate_market_columns(frame, ["date", "symbol", "close"])
    return frame


def _normalize_benchmark_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol))
    if not normalized:
        raise ValueError("At least one benchmark symbol is required.")
    missing_required = sorted(set(REQUIRED_BENCHMARK_SYMBOLS).difference(normalized))
    if missing_required:
        raise ValueError(
            "benchmark_symbols must include SPY and QQQ for this report schema; "
            f"missing: {missing_required}"
        )
    return normalized


def _validate_results_columns(results: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_RESULTS_COLUMNS).difference(results.columns))
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")


def _build_shortlist_test_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = test_rows.copy()
    numeric_columns = [
        "top_k",
        "transaction_cost_bps",
        "total_return",
        "sharpe_like",
        "max_drawdown",
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if rows[numeric_columns].isna().any().any():
        raise ValueError("Shortlist candidate rows contain non-numeric metric or spec values.")

    mask = pd.Series(False, index=rows.index)
    order_by_id = {
        _candidate_id_from_spec(spec): index for index, spec in enumerate(SHORTLIST_SPECS)
    }
    for spec in SHORTLIST_SPECS:
        mask |= _matches_shortlist_spec(rows, spec)

    shortlist = rows.loc[mask].copy()
    if shortlist.empty:
        return shortlist

    metadata_rows = []
    for fold_name in shortlist["fold"]:
        metadata = parse_sliding_window_fold_name(str(fold_name))
        metadata_rows.append(
            {
                "train_window_years": metadata.train_window_years,
                "validation_window_months": metadata.validation_window_months,
                "test_window_months": metadata.test_window_months,
                "sequence_number": metadata.sequence_number,
                "test_start": metadata.test_start.date().isoformat(),
                "test_end": metadata.test_end.date().isoformat(),
                "test_year": metadata.test_year,
            }
        )

    metadata = pd.DataFrame(metadata_rows, index=shortlist.index)
    shortlist = pd.concat([shortlist, metadata], axis=1)
    shortlist["candidate_id"] = shortlist.apply(_candidate_id_from_row, axis=1)
    shortlist["candidate_label"] = shortlist.apply(_candidate_label_from_row, axis=1)
    shortlist["_candidate_order"] = shortlist["candidate_id"].map(order_by_id)
    return (
        shortlist.sort_values(
            by=[
                "_candidate_order",
                "test_window_months",
                "train_window_years",
                "test_start",
                "sequence_number",
                "fold",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _matches_shortlist_spec(rows: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    return (
        rows["universe"].eq(spec["universe"])
        & rows["model_type"].eq(spec["model_type"])
        & rows["top_k"].astype(int).eq(int(spec["top_k"]))
        & rows["allocation_mode"].eq(spec["allocation_mode"])
        & np.isclose(
            rows["transaction_cost_bps"].astype(float),
            float(spec["transaction_cost_bps"]),
            rtol=0.0,
            atol=1e-9,
        )
    )


def _build_window_metrics(
    shortlist_rows: pd.DataFrame,
    *,
    market_data_by_universe: dict[str, pd.DataFrame],
    benchmark_symbols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, strategy_row in shortlist_rows.iterrows():
        universe = str(strategy_row["universe"])
        market_data = market_data_by_universe[universe]
        benchmark_returns = {
            symbol.lower(): calculate_buy_and_hold_window_return(
                market_data,
                symbol=symbol,
                test_start=strategy_row["test_start"],
                test_end=strategy_row["test_end"],
            )
            for symbol in benchmark_symbols
        }
        strategy_total_return = float(strategy_row["total_return"])
        spy_return = benchmark_returns["spy"]
        qqq_return = benchmark_returns["qqq"]

        rows.append(
            {
                "candidate_id": strategy_row["candidate_id"],
                "candidate_label": strategy_row["candidate_label"],
                "universe": universe,
                "model_type": strategy_row["model_type"],
                "top_k": int(strategy_row["top_k"]),
                "allocation_mode": strategy_row["allocation_mode"],
                "transaction_cost_bps": float(strategy_row["transaction_cost_bps"]),
                "fold": strategy_row["fold"],
                "train_window_years": int(strategy_row["train_window_years"]),
                "validation_window_months": int(
                    strategy_row["validation_window_months"]
                ),
                "test_window_months": int(strategy_row["test_window_months"]),
                "sequence_number": int(strategy_row["sequence_number"]),
                "test_start": strategy_row["test_start"],
                "test_end": strategy_row["test_end"],
                "test_year": int(strategy_row["test_year"]),
                "strategy_total_return": strategy_total_return,
                "strategy_sharpe_like": float(strategy_row["sharpe_like"]),
                "strategy_max_drawdown": float(strategy_row["max_drawdown"]),
                "spy_total_return": spy_return,
                "qqq_total_return": qqq_return,
                "strategy_minus_spy": _difference_or_nan(strategy_total_return, spy_return),
                "strategy_minus_qqq": _difference_or_nan(strategy_total_return, qqq_return),
                "strategy_beats_spy": _beats_or_nan(strategy_total_return, spy_return),
                "strategy_beats_qqq": _beats_or_nan(strategy_total_return, qqq_return),
                "_candidate_order": int(strategy_row["_candidate_order"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "_candidate_order",
                "test_window_months",
                "train_window_years",
                "test_start",
                "sequence_number",
                "fold",
            ],
            kind="mergesort",
        )
        .drop(columns=["_candidate_order"])
        .reset_index(drop=True)[WINDOW_METRIC_COLUMNS]
    )


def _build_candidate_summary(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in window_metrics.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": first["candidate_id"],
                "candidate_label": first["candidate_label"],
                "universe": first["universe"],
                "model_type": first["model_type"],
                "top_k": int(first["top_k"]),
                "allocation_mode": first["allocation_mode"],
                "transaction_cost_bps": float(first["transaction_cost_bps"]),
                **_summarize_metric_group(group),
            }
        )
    return pd.DataFrame(rows)


def _build_by_test_window(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_window_months, group in window_metrics.groupby(
        "test_window_months",
        sort=True,
    ):
        rows.append(
            {
                "test_window_months": int(test_window_months),
                "candidate_count": int(group["candidate_id"].nunique()),
                **_summarize_metric_group(group),
            }
        )
    return pd.DataFrame(rows, columns=BY_TEST_WINDOW_COLUMNS)


def _build_by_candidate_and_test_window(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, test_window_months), group in window_metrics.groupby(
        ["candidate_id", "test_window_months"],
        sort=False,
    ):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": first["candidate_label"],
                "test_window_months": int(test_window_months),
                **_summarize_metric_group(group),
            }
        )
    return pd.DataFrame(rows)


def _build_by_year(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, test_year), group in window_metrics.groupby(
        ["candidate_id", "test_year"],
        sort=False,
    ):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": first["candidate_label"],
                "test_year": int(test_year),
                "window_count": int(len(group)),
                "mean_total_return": group["strategy_total_return"].mean(),
                "spy_mean_total_return": group["spy_total_return"].mean(),
                "qqq_mean_total_return": group["qqq_total_return"].mean(),
                "strategy_beats_spy_rate": _boolean_rate(group["strategy_beats_spy"]),
                "strategy_beats_qqq_rate": _boolean_rate(group["strategy_beats_qqq"]),
                "mean_strategy_minus_spy": group["strategy_minus_spy"].mean(),
                "mean_strategy_minus_qqq": group["strategy_minus_qqq"].mean(),
            }
        )
    return pd.DataFrame(rows, columns=BY_YEAR_COLUMNS)


def _build_relative_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    output = window_metrics.copy()
    output["worst_benchmark_relative_return"] = output[
        ["strategy_minus_spy", "strategy_minus_qqq"]
    ].min(axis=1, skipna=True)
    output["worst_relative_benchmark"] = np.where(
        output["strategy_minus_spy"] <= output["strategy_minus_qqq"],
        "SPY",
        "QQQ",
    )
    both_missing = output[["strategy_minus_spy", "strategy_minus_qqq"]].isna().all(axis=1)
    output.loc[both_missing, "worst_relative_benchmark"] = ""
    return output.sort_values(
        by=[
            "worst_benchmark_relative_return",
            "strategy_minus_spy",
            "strategy_minus_qqq",
            "candidate_id",
            "fold",
        ],
        ascending=[True, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_metric_group(group: pd.DataFrame) -> dict[str, float | int]:
    return {
        "window_count": int(len(group)),
        "mean_total_return": group["strategy_total_return"].mean(),
        "median_total_return": group["strategy_total_return"].median(),
        "positive_window_rate": _positive_rate(group["strategy_total_return"]),
        "mean_sharpe_like": group["strategy_sharpe_like"].mean(),
        "mean_max_drawdown": group["strategy_max_drawdown"].mean(),
        "worst_max_drawdown": group["strategy_max_drawdown"].min(),
        "spy_mean_total_return": group["spy_total_return"].mean(),
        "qqq_mean_total_return": group["qqq_total_return"].mean(),
        "strategy_beats_spy_rate": _boolean_rate(group["strategy_beats_spy"]),
        "strategy_beats_qqq_rate": _boolean_rate(group["strategy_beats_qqq"]),
        "mean_strategy_minus_spy": group["strategy_minus_spy"].mean(),
        "mean_strategy_minus_qqq": group["strategy_minus_qqq"].mean(),
    }


def _add_scores(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["balanced_score"] = (
        _zscore(output["mean_total_return"])
        + _zscore(output["mean_sharpe_like"])
        + _zscore(output["positive_window_rate"])
        + _zscore(output["mean_max_drawdown"])
        + _zscore(output["worst_max_drawdown"])
    ) / 5.0
    output["benchmark_stability_score"] = (
        _zscore(output["strategy_beats_spy_rate"])
        + _zscore(output["strategy_beats_qqq_rate"])
        + _zscore(output["mean_strategy_minus_spy"])
        + _zscore(output["mean_strategy_minus_qqq"])
    ) / 4.0
    return output


def _sort_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        by=[
            "balanced_score",
            "benchmark_stability_score",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "candidate_id",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )[CANDIDATE_SUMMARY_COLUMNS]


def _sort_candidate_test_window_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        by=[
            "test_window_months",
            "balanced_score",
            "benchmark_stability_score",
            "mean_total_return",
            "candidate_id",
        ],
        ascending=[True, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )[BY_CANDIDATE_AND_TEST_WINDOW_COLUMNS]


def _build_benchmark_ranking(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    output = candidate_summary.sort_values(
        by=[
            "benchmark_stability_score",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "mean_strategy_minus_spy",
            "mean_strategy_minus_qqq",
            "balanced_score",
            "candidate_id",
        ],
        ascending=[False, False, False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    return output


def _candidate_id_from_spec(spec: dict[str, object]) -> str:
    return (
        f"{spec['universe']}:{spec['model_type']}:top{int(spec['top_k'])}:"
        f"{spec['allocation_mode']}:cost{_format_compact(spec['transaction_cost_bps'])}"
    )


def _candidate_id_from_row(row: pd.Series) -> str:
    return (
        f"{row['universe']}:{row['model_type']}:top{int(row['top_k'])}:"
        f"{row['allocation_mode']}:cost{_format_compact(row['transaction_cost_bps'])}"
    )


def _candidate_label_from_row(row: pd.Series) -> str:
    return (
        f"{row['universe']} / {row['model_type']} / top{int(row['top_k'])} / "
        f"{row['allocation_mode']} / cost{_format_compact(row['transaction_cost_bps'])}"
    )


def _candidate_label_from_spec(spec: dict[str, object]) -> str:
    return (
        f"{spec['universe']} / {spec['model_type']} / top{int(spec['top_k'])} / "
        f"{spec['allocation_mode']} / cost{_format_compact(spec['transaction_cost_bps'])}"
    )


def _difference_or_nan(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(left) - float(right)


def _beats_or_nan(strategy_return: float, benchmark_return: float) -> object:
    if pd.isna(strategy_return) or pd.isna(benchmark_return):
        return np.nan
    return bool(float(strategy_return) > float(benchmark_return))


def _positive_rate(values: pd.Series) -> float:
    non_missing = pd.to_numeric(values, errors="coerce").dropna()
    if non_missing.empty:
        return float("nan")
    return float((non_missing > 0).mean())


def _boolean_rate(values: pd.Series) -> float:
    non_missing = values.dropna()
    if non_missing.empty:
        return float("nan")
    return float(non_missing.astype(bool).mean())


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std


def _count_text_file_lines(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def _validate_market_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")


def _format_shortlist_specs_for_markdown() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": _candidate_id_from_spec(spec),
                "candidate_label": _candidate_label_from_spec(spec),
            }
            for spec in SHORTLIST_SPECS
        ]
    )


def _format_candidate_summary_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_label",
        "window_count",
        "mean_total_return",
        "positive_window_rate",
        "mean_sharpe_like",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "strategy_beats_spy_rate",
        "strategy_beats_qqq_rate",
        "mean_strategy_minus_spy",
        "mean_strategy_minus_qqq",
        "balanced_score",
        "benchmark_stability_score",
    ]
    output = summary[columns].copy()
    output["balanced_score"] = output["balanced_score"].map(_format_decimal)
    output["benchmark_stability_score"] = output["benchmark_stability_score"].map(
        _format_decimal
    )
    return _format_percent_columns(
        output,
        exclude={
            "candidate_label",
            "window_count",
            "balanced_score",
            "benchmark_stability_score",
        },
    )


def _format_by_test_window_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    return _format_percent_columns(
        output,
        exclude={"test_window_months", "window_count", "candidate_count"},
    )


def _format_candidate_test_window_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_label",
        "test_window_months",
        "window_count",
        "mean_total_return",
        "positive_window_rate",
        "mean_sharpe_like",
        "worst_max_drawdown",
        "strategy_beats_spy_rate",
        "strategy_beats_qqq_rate",
        "balanced_score",
        "benchmark_stability_score",
    ]
    output = summary[columns].copy()
    output["balanced_score"] = output["balanced_score"].map(_format_decimal)
    output["benchmark_stability_score"] = output["benchmark_stability_score"].map(
        _format_decimal
    )
    return _format_percent_columns(
        output,
        exclude={
            "candidate_label",
            "test_window_months",
            "window_count",
            "balanced_score",
            "benchmark_stability_score",
        },
    )


def _format_benchmark_ranking_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "candidate_label",
        "benchmark_stability_score",
        "balanced_score",
        "strategy_beats_spy_rate",
        "strategy_beats_qqq_rate",
        "mean_strategy_minus_spy",
        "mean_strategy_minus_qqq",
    ]
    output = summary[columns].copy()
    output["benchmark_stability_score"] = output["benchmark_stability_score"].map(
        _format_decimal
    )
    output["balanced_score"] = output["balanced_score"].map(_format_decimal)
    return _format_percent_columns(
        output,
        exclude={"rank", "candidate_label", "benchmark_stability_score", "balanced_score"},
    )


def _format_by_year_for_markdown(by_year: pd.DataFrame) -> pd.DataFrame:
    output = by_year.copy()
    return _format_percent_columns(
        output,
        exclude={"candidate_id", "candidate_label", "test_year", "window_count"},
    )


def _format_windows_for_markdown(windows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_label",
        "fold",
        "test_window_months",
        "test_start",
        "test_end",
        "strategy_total_return",
        "spy_total_return",
        "qqq_total_return",
        "strategy_minus_spy",
        "strategy_minus_qqq",
        "worst_relative_benchmark",
        "worst_benchmark_relative_return",
    ]
    output = windows[columns].copy()
    return _format_percent_columns(
        output,
        exclude={
            "candidate_label",
            "fold",
            "test_window_months",
            "test_start",
            "test_end",
            "worst_relative_benchmark",
        },
    )


def _format_percent_columns(
    frame: pd.DataFrame,
    *,
    exclude: set[str],
) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column not in exclude:
            output[column] = output[column].map(_format_percent)
    return output


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    separator = ["---" for _ in headers]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _final_recommendation(
    report_data: SlidingWindowLongerWindowShortlistReportData,
) -> str:
    benchmark_ranking = _build_benchmark_ranking(report_data.candidate_summary)
    benchmark_leader = benchmark_ranking.iloc[0]
    balanced_leader = report_data.candidate_summary.iloc[0]
    six_month_leader = _leader_for_test_window(report_data.by_candidate_and_test_window, 6)
    twelve_month_leader = _leader_for_test_window(report_data.by_candidate_and_test_window, 12)

    benchmark_holds = benchmark_leader["candidate_id"] == PREVIOUS_BENCHMARK_RELATIVE_LEADER
    risk_holds = balanced_leader["candidate_id"] == PREVIOUS_RISK_ADJUSTED_LEADER
    shortlist_result = (
        "The 3-month shortlist result broadly holds"
        if benchmark_holds and risk_holds
        else "The 3-month shortlist result only partially holds"
        if benchmark_holds or risk_holds
        else "The 3-month shortlist leadership does not fully carry over"
    )

    follow_up = _follow_up_candidates(report_data.candidate_summary, benchmark_ranking)
    return (
        "The candidate that survives the longer windows best on the balanced "
        f"return/risk score is `{balanced_leader['candidate_label']}`. "
        f"{shortlist_result}: the benchmark-relative leader is "
        f"`{benchmark_leader['candidate_label']}`, so broad_macro linear top1 "
        f"{'remains' if benchmark_holds else 'does not remain'} the "
        "benchmark-relative leader under 6-month and 12-month windows. "
        f"The best risk-adjusted candidate is `{balanced_leader['candidate_label']}`, "
        f"so broad_macro random_forest top3 "
        f"{'remains' if risk_holds else 'does not remain'} best by this longer-window "
        "return/risk score. "
        f"Across window lengths, the 6-month balanced leader is {_format_leader(six_month_leader)} "
        f"and the 12-month balanced leader is {_format_leader(twelve_month_leader)}. "
        f"Deeper follow-up is warranted for {follow_up}, with special attention to "
        "whether their benchmark-relative edge survives adjacent regimes and "
        "portfolio-construction variants. Do not call any candidate production-ready "
        "from this report."
    )


def _leader_for_test_window(
    by_candidate_and_test_window: pd.DataFrame,
    test_window_months: int,
) -> pd.Series | None:
    rows = by_candidate_and_test_window.loc[
        by_candidate_and_test_window["test_window_months"] == test_window_months
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _format_leader(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    return f"`{row['candidate_label']}`"


def _follow_up_candidates(
    candidate_summary: pd.DataFrame,
    benchmark_ranking: pd.DataFrame,
) -> str:
    candidate_ids = [
        str(candidate_summary.iloc[0]["candidate_id"]),
        str(benchmark_ranking.iloc[0]["candidate_id"]),
        PREVIOUS_DEFENSIVE_CANDIDATE,
        PREVIOUS_FULL_REFERENCE,
    ]
    rows = candidate_summary.set_index("candidate_id")
    labels = []
    for candidate_id in candidate_ids:
        if candidate_id in rows.index:
            label = str(rows.loc[candidate_id, "candidate_label"])
            if label not in labels:
                labels.append(label)
    return _join_markdown_labels(labels)


def _join_markdown_labels(labels: Sequence[str]) -> str:
    formatted = [f"`{label}`" for label in labels]
    if not formatted:
        return "no candidates"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return " and ".join(formatted)
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_decimal(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _format_compact(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def main() -> int:
    return run_sliding_window_longer_window_shortlist_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
