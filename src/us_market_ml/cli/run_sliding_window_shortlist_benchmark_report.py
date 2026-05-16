"""Compare shortlist sliding-window candidates against SPY and QQQ."""

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
from us_market_ml.cli.run_sliding_window_stability_report import (
    parse_sliding_window_fold_name,
)
from us_market_ml.data import read_local_market_data

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/sliding_window_overnight_candidate_study_results.csv"
)
DEFAULT_FAILED_KEYS = Path(
    "artifacts/sweeps/sliding_window_overnight_candidate_study_failed_keys.txt"
)
DEFAULT_FULL_MARKET_DATA = Path("artifacts/cache/universe_full_2018_2026.parquet")
DEFAULT_BROAD_MACRO_MARKET_DATA = Path(
    "artifacts/cache/universe_broad_macro_2018_2026.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/sliding_window_shortlist_benchmark")
REQUIRED_BENCHMARK_SYMBOLS = ("SPY", "QQQ")

WINDOW_METRICS_PATH_NAME = "sliding_window_shortlist_window_metrics.csv"
CANDIDATE_SUMMARY_PATH_NAME = "sliding_window_shortlist_candidate_summary.csv"
BENCHMARK_SUMMARY_PATH_NAME = "sliding_window_shortlist_benchmark_summary.csv"
BY_YEAR_PATH_NAME = "sliding_window_shortlist_by_year.csv"
RELATIVE_WINDOWS_PATH_NAME = "sliding_window_shortlist_relative_windows.csv"
REPORT_PATH_NAME = "sliding_window_shortlist_benchmark_report.md"

OUTPUT_PATH_NAMES = (
    WINDOW_METRICS_PATH_NAME,
    CANDIDATE_SUMMARY_PATH_NAME,
    BENCHMARK_SUMMARY_PATH_NAME,
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

CANDIDATE_SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "universe",
    "model_type",
    "top_k",
    "allocation_mode",
    "transaction_cost_bps",
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
    "benchmark_stability_score",
    "return_risk_score",
]

BENCHMARK_SUMMARY_COLUMNS = [
    "rank",
    "candidate_id",
    "candidate_label",
    "window_count",
    "benchmark_stability_score",
    "return_risk_score",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_spy",
    "mean_strategy_minus_qqq",
    "mean_total_return",
    "mean_sharpe_like",
    "worst_max_drawdown",
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

SHORTLIST_SPECS = (
    {
        "universe": "broad_macro",
        "model_type": "linear_regression",
        "top_k": 1,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
    {
        "universe": "broad_macro",
        "model_type": "linear_regression",
        "top_k": 2,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
    {
        "universe": "broad_macro",
        "model_type": "random_forest",
        "top_k": 3,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
    {
        "universe": "broad_macro",
        "model_type": "ridge",
        "top_k": 5,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
    {
        "universe": "full",
        "model_type": "ridge",
        "top_k": 3,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
    {
        "universe": "full",
        "model_type": "ridge",
        "top_k": 5,
        "allocation_mode": "equal_weight",
        "transaction_cost_bps": 10.0,
    },
)


@dataclass(frozen=True)
class SlidingWindowShortlistBenchmarkReportData:
    """All table outputs produced by the shortlist benchmark report."""

    window_metrics: pd.DataFrame
    candidate_summary: pd.DataFrame
    benchmark_summary: pd.DataFrame
    by_year: pd.DataFrame
    relative_windows: pd.DataFrame
    failed_key_count: int
    completed_key_count: int
    result_row_count: int
    test_row_count: int
    shortlist_test_row_count: int
    benchmark_symbols: tuple[str, ...]


def run_sliding_window_shortlist_benchmark_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build a benchmark comparison report for the shortlist candidate set."""

    args = _parse_args(argv)
    report_data = build_sliding_window_shortlist_benchmark_report(
        results_csv=args.results_csv,
        failed_keys=args.failed_keys,
        full_market_data=args.full_market_data,
        broad_macro_market_data=args.broad_macro_market_data,
        benchmark_symbols=args.benchmark_symbols,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir)
    outputs = write_sliding_window_shortlist_benchmark_report(
        report_data=report_data,
        output_dir=output_dir,
        top_n=args.top_n,
    )

    print("US Market ML sliding-window shortlist benchmark report")
    print(f"results CSV: {args.results_csv}")
    print(f"failed keys: {args.failed_keys}")
    print(f"full market data: {args.full_market_data}")
    print(f"broad_macro market data: {args.broad_macro_market_data}")
    print(f"benchmarks: {','.join(report_data.benchmark_symbols)}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.benchmark_summary.head(args.top_n).to_string(index=False))
    return 0


def build_sliding_window_shortlist_benchmark_report(
    *,
    results_csv: Path,
    failed_keys: Path | None = None,
    full_market_data: Path | pd.DataFrame = DEFAULT_FULL_MARKET_DATA,
    broad_macro_market_data: Path | pd.DataFrame = DEFAULT_BROAD_MACRO_MARKET_DATA,
    benchmark_symbols: Sequence[str] = REQUIRED_BENCHMARK_SYMBOLS,
    top_n: int = 10,
) -> SlidingWindowShortlistBenchmarkReportData:
    """Read completed candidate rows and return shortlist-vs-benchmark tables."""

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
    candidate_summary = _build_candidate_summary(window_metrics)
    benchmark_summary = _build_benchmark_summary(candidate_summary)
    by_year = _build_by_year(window_metrics)
    relative_windows = _build_relative_windows(window_metrics)

    failed_key_count = _count_text_file_lines(failed_keys)
    completed_key_count = int(results["run_key"].nunique()) if "run_key" in results else 0

    return SlidingWindowShortlistBenchmarkReportData(
        window_metrics=window_metrics,
        candidate_summary=candidate_summary,
        benchmark_summary=benchmark_summary,
        by_year=by_year,
        relative_windows=relative_windows,
        failed_key_count=failed_key_count,
        completed_key_count=completed_key_count,
        result_row_count=len(results),
        test_row_count=len(test_rows),
        shortlist_test_row_count=len(shortlist_rows),
        benchmark_symbols=resolved_benchmark_symbols,
    )


def write_sliding_window_shortlist_benchmark_report(
    *,
    report_data: SlidingWindowShortlistBenchmarkReportData,
    output_dir: Path,
    top_n: int = 10,
) -> dict[str, Path]:
    """Write report CSVs and markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        WINDOW_METRICS_PATH_NAME: output_dir / WINDOW_METRICS_PATH_NAME,
        CANDIDATE_SUMMARY_PATH_NAME: output_dir / CANDIDATE_SUMMARY_PATH_NAME,
        BENCHMARK_SUMMARY_PATH_NAME: output_dir / BENCHMARK_SUMMARY_PATH_NAME,
        BY_YEAR_PATH_NAME: output_dir / BY_YEAR_PATH_NAME,
        RELATIVE_WINDOWS_PATH_NAME: output_dir / RELATIVE_WINDOWS_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }

    report_data.window_metrics.to_csv(outputs[WINDOW_METRICS_PATH_NAME], index=False)
    report_data.candidate_summary.to_csv(
        outputs[CANDIDATE_SUMMARY_PATH_NAME],
        index=False,
    )
    report_data.benchmark_summary.to_csv(
        outputs[BENCHMARK_SUMMARY_PATH_NAME],
        index=False,
    )
    report_data.by_year.to_csv(outputs[BY_YEAR_PATH_NAME], index=False)
    report_data.relative_windows.to_csv(
        outputs[RELATIVE_WINDOWS_PATH_NAME],
        index=False,
    )
    outputs[REPORT_PATH_NAME].write_text(
        render_sliding_window_shortlist_benchmark_markdown(
            report_data,
            top_n=top_n,
        ),
        encoding="utf-8",
    )
    return outputs


def render_sliding_window_shortlist_benchmark_markdown(
    report_data: SlidingWindowShortlistBenchmarkReportData,
    *,
    top_n: int = 10,
) -> str:
    """Render the shortlist benchmark report as Markdown."""

    period_start = report_data.window_metrics["test_start"].min()
    period_end = report_data.window_metrics["test_end"].max()
    benchmark_ranking = report_data.benchmark_summary.head(top_n)
    year_breakdown = report_data.by_year.head(top_n)
    worst_relative = report_data.relative_windows.head(top_n)

    lines = [
        "# Sliding-Window Shortlist Benchmark Report",
        "",
        "This report compares the pre-selected research-stage shortlist against SPY "
        "and QQQ over the same rolling test windows. It uses only "
        '`split == "test"` rows from the completed overnight candidate study and '
        "does not rerun sweeps or modify model training.",
        "",
        "## Completion Status",
        "",
        f"- Result rows: {report_data.result_row_count}",
        f"- Test rows read: {report_data.test_row_count}",
        f"- Shortlist test rows used: {report_data.shortlist_test_row_count}",
        f"- Completed keys: {report_data.completed_key_count}",
        f"- Failed keys: {report_data.failed_key_count}",
        "- Rolling test period covered: "
        f"{_format_date(period_start)} to {_format_date(period_end)}",
        f"- Benchmarks: `{', '.join(report_data.benchmark_symbols)}`",
        "",
        "## Shortlist Candidates",
        "",
        _markdown_table(_format_shortlist_specs_for_markdown()),
        "",
        "## Candidate Summary",
        "",
        _markdown_table(_format_candidate_summary_for_markdown(report_data.candidate_summary)),
        "",
        "## Benchmark-Relative Ranking",
        "",
        _markdown_table(_format_benchmark_summary_for_markdown(benchmark_ranking)),
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
            "Generate a benchmark comparison report from completed sliding-window "
            "overnight shortlist candidate results."
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
            by=["_candidate_order", "train_window_years", "test_start", "sequence_number", "fold"],
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
                "_sequence_number": int(strategy_row["sequence_number"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "_candidate_order",
                "train_window_years",
                "test_start",
                "_sequence_number",
                "fold",
            ],
            kind="mergesort",
        )
        .drop(columns=["_candidate_order", "_sequence_number"])
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
        )

    summary = pd.DataFrame(rows)
    summary["benchmark_stability_score"] = (
        summary["strategy_beats_spy_rate"] + summary["strategy_beats_qqq_rate"]
    ) / 2.0
    summary["return_risk_score"] = (
        _zscore(summary["mean_total_return"])
        + _zscore(summary["mean_sharpe_like"])
        + _zscore(summary["mean_max_drawdown"])
        + _zscore(summary["worst_max_drawdown"])
    ) / 4.0
    return summary[CANDIDATE_SUMMARY_COLUMNS]


def _build_benchmark_summary(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    output = candidate_summary.sort_values(
        by=[
            "benchmark_stability_score",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "mean_strategy_minus_spy",
            "mean_strategy_minus_qqq",
            "return_risk_score",
            "candidate_id",
        ],
        ascending=[False, False, False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    return output[BENCHMARK_SUMMARY_COLUMNS]


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
        "worst_max_drawdown",
        "strategy_beats_spy_rate",
        "strategy_beats_qqq_rate",
        "mean_strategy_minus_spy",
        "mean_strategy_minus_qqq",
    ]
    output = summary[columns].copy()
    return _format_percent_columns(
        output,
        exclude={"candidate_label", "window_count"},
    )


def _format_benchmark_summary_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary[
        [
            "rank",
            "candidate_label",
            "benchmark_stability_score",
            "return_risk_score",
            "strategy_beats_spy_rate",
            "strategy_beats_qqq_rate",
            "mean_strategy_minus_spy",
            "mean_strategy_minus_qqq",
        ]
    ].copy()
    output["benchmark_stability_score"] = output["benchmark_stability_score"].map(
        _format_decimal
    )
    output["return_risk_score"] = output["return_risk_score"].map(_format_decimal)
    return _format_percent_columns(
        output,
        exclude={"rank", "candidate_label", "benchmark_stability_score", "return_risk_score"},
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


def _final_recommendation(report_data: SlidingWindowShortlistBenchmarkReportData) -> str:
    benchmark_leader = report_data.benchmark_summary.iloc[0]
    risk_leader = report_data.candidate_summary.sort_values(
        by=[
            "return_risk_score",
            "mean_total_return",
            "mean_sharpe_like",
            "worst_max_drawdown",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).iloc[0]

    follow_up = _follow_up_candidates(report_data.candidate_summary)
    return (
        "Best by benchmark-relative stability is "
        f"`{benchmark_leader['candidate_label']}`: it has a benchmark stability score "
        f"of {_format_decimal(benchmark_leader['benchmark_stability_score'])}, beats "
        f"SPY in {_format_percent(benchmark_leader['strategy_beats_spy_rate'])} of "
        f"windows, and beats QQQ in "
        f"{_format_percent(benchmark_leader['strategy_beats_qqq_rate'])} of windows. "
        "Best return/risk tradeoff is "
        f"`{risk_leader['candidate_label']}` by the return/risk score, which combines "
        "mean return, Sharpe-like score, average drawdown, and worst drawdown. "
        f"Deeper follow-up is warranted for {follow_up}; that follow-up should test "
        "whether the benchmark-relative edge survives longer windows and adjacent "
        "market regimes. Do not call any shortlist candidate production-ready from "
        "this report."
    )


def _follow_up_candidates(candidate_summary: pd.DataFrame) -> str:
    leaders = []
    benchmark_leader = _build_benchmark_summary(candidate_summary).iloc[0]
    risk_leader = candidate_summary.sort_values(
        by=["return_risk_score", "mean_total_return", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    for row in [benchmark_leader, risk_leader]:
        label = str(row["candidate_label"])
        if label not in leaders:
            leaders.append(label)
    return " and ".join(f"`{label}`" for label in leaders)


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
    return run_sliding_window_shortlist_benchmark_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
