"""Compare completed sliding-window champion windows against SPY and QQQ."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_sliding_window_stability_report import (
    CHAMPION_NAME,
    CHAMPION_SPEC,
    parse_sliding_window_fold_name,
)
from us_market_ml.data import read_local_market_data

DEFAULT_RESULTS_CSV = Path("artifacts/sweeps/sliding_window_champion_study_results.csv")
DEFAULT_FAILED_KEYS = Path(
    "artifacts/sweeps/sliding_window_champion_study_failed_keys.txt"
)
DEFAULT_MARKET_DATA = Path("artifacts/cache/universe_full_2018_2026.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/sliding_window_benchmark_comparison")
REQUIRED_BENCHMARK_SYMBOLS = ("SPY", "QQQ")

WINDOW_METRIC_COLUMNS = [
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

SUMMARY_COLUMNS = [
    "window_count",
    "strategy_mean_total_return",
    "strategy_median_total_return",
    "spy_mean_total_return",
    "spy_median_total_return",
    "qqq_mean_total_return",
    "qqq_median_total_return",
    "strategy_positive_window_rate",
    "spy_positive_window_rate",
    "qqq_positive_window_rate",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_spy",
    "mean_strategy_minus_qqq",
]

GROUP_COLUMNS = [
    "count",
    "strategy_mean_total_return",
    "spy_mean_total_return",
    "qqq_mean_total_return",
    "strategy_beats_spy_rate",
    "strategy_beats_qqq_rate",
]


@dataclass(frozen=True)
class SlidingWindowBenchmarkComparisonReportData:
    """All table outputs produced by the benchmark comparison report."""

    window_metrics: pd.DataFrame
    summary: pd.DataFrame
    by_train_window: pd.DataFrame
    by_year: pd.DataFrame
    relative_windows: pd.DataFrame
    failed_key_count: int
    completed_key_count: int
    result_row_count: int
    benchmark_symbols: tuple[str, ...]


def run_sliding_window_benchmark_comparison_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build a benchmark comparison report from completed sliding-window rows."""

    args = _parse_args(argv)
    report_data = build_sliding_window_benchmark_comparison_report(
        results_csv=args.results_csv,
        market_data=args.market_data,
        failed_keys=args.failed_keys,
        benchmark_symbols=args.benchmark_symbols,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_metrics_path = output_dir / "sliding_window_benchmark_window_metrics.csv"
    summary_path = output_dir / "sliding_window_benchmark_summary.csv"
    by_train_window_path = output_dir / "sliding_window_benchmark_by_train_window.csv"
    by_year_path = output_dir / "sliding_window_benchmark_by_year.csv"
    relative_windows_path = output_dir / "sliding_window_benchmark_relative_windows.csv"
    markdown_path = output_dir / "sliding_window_benchmark_comparison_report.md"

    report_data.window_metrics.to_csv(window_metrics_path, index=False)
    report_data.summary.to_csv(summary_path, index=False)
    report_data.by_train_window.to_csv(by_train_window_path, index=False)
    report_data.by_year.to_csv(by_year_path, index=False)
    report_data.relative_windows.to_csv(relative_windows_path, index=False)
    markdown_path.write_text(
        render_sliding_window_benchmark_comparison_markdown(
            report_data,
            top_n=args.top_n,
        ),
        encoding="utf-8",
    )

    print("US Market ML sliding-window benchmark comparison report")
    print(f"champion: {CHAMPION_NAME}")
    print(f"results CSV: {args.results_csv}")
    print(f"market data: {args.market_data}")
    print(f"failed keys: {args.failed_keys}")
    print(f"benchmarks: {','.join(report_data.benchmark_symbols)}")
    print(f"output window metrics CSV: {window_metrics_path}")
    print(f"output summary CSV: {summary_path}")
    print(f"output train-window CSV: {by_train_window_path}")
    print(f"output year CSV: {by_year_path}")
    print(f"output relative windows CSV: {relative_windows_path}")
    print(f"output markdown report: {markdown_path}")
    print()
    print(report_data.summary.to_string(index=False))

    return 0


def calculate_buy_and_hold_window_return(
    market_data: pd.DataFrame,
    *,
    symbol: str,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> float:
    """Return simple close-to-close buy-and-hold return for one test window."""

    _validate_market_columns(market_data, [date_col, symbol_col, price_col])
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    if start > end:
        raise ValueError("test_start must be earlier than or equal to test_end.")

    date_values = pd.to_datetime(market_data[date_col])
    symbol_mask = market_data[symbol_col] == symbol
    date_mask = (date_values >= start) & (date_values <= end)
    prices = (
        market_data.loc[symbol_mask & date_mask, [date_col, price_col]]
        .copy()
        .dropna(subset=[price_col])
        .sort_values(date_col, kind="mergesort")
        .reset_index(drop=True)
    )
    if prices.empty:
        return float("nan")

    first_close = float(prices.loc[0, price_col])
    last_close = float(prices.loc[len(prices) - 1, price_col])
    if not np.isfinite(first_close) or not np.isfinite(last_close) or first_close == 0.0:
        return float("nan")
    return last_close / first_close - 1.0


def build_sliding_window_benchmark_comparison_report(
    *,
    results_csv: Path,
    market_data: Path | pd.DataFrame,
    failed_keys: Path | None = None,
    benchmark_symbols: Sequence[str] = REQUIRED_BENCHMARK_SYMBOLS,
    top_n: int = 10,
) -> SlidingWindowBenchmarkComparisonReportData:
    """Read completed sweep results and return strategy-vs-benchmark tables."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    resolved_benchmark_symbols = _normalize_benchmark_symbols(benchmark_symbols)
    market_frame = (
        read_local_market_data(market_data)
        if not isinstance(market_data, pd.DataFrame)
        else market_data.copy()
    )
    _validate_market_columns(market_frame, ["date", "symbol", "close"])

    results = pd.read_csv(results_csv)
    _validate_results_columns(results)
    strategy_rows = _build_enriched_strategy_test_rows(results)
    if strategy_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    window_metrics = _build_window_metrics(
        strategy_rows,
        market_frame,
        benchmark_symbols=resolved_benchmark_symbols,
    )
    summary = _build_summary(window_metrics)
    by_train_window = _build_train_window_summary(window_metrics)
    by_year = _build_year_summary(window_metrics)
    relative_windows = _build_relative_windows(window_metrics)

    failed_key_count = _count_text_file_lines(failed_keys)
    completed_key_count = int(results["run_key"].nunique()) if "run_key" in results else 0

    return SlidingWindowBenchmarkComparisonReportData(
        window_metrics=window_metrics,
        summary=summary,
        by_train_window=by_train_window,
        by_year=by_year,
        relative_windows=relative_windows,
        failed_key_count=failed_key_count,
        completed_key_count=completed_key_count,
        result_row_count=len(results),
        benchmark_symbols=resolved_benchmark_symbols,
    )


def render_sliding_window_benchmark_comparison_markdown(
    report_data: SlidingWindowBenchmarkComparisonReportData,
    *,
    top_n: int = 10,
) -> str:
    """Render the benchmark comparison report as Markdown."""

    summary_row = report_data.summary.iloc[0]
    period_start = report_data.window_metrics["test_start"].min()
    period_end = report_data.window_metrics["test_end"].max()
    worst_relative = report_data.relative_windows.head(top_n)

    lines = [
        "# Sliding-Window Benchmark Comparison Report",
        "",
        "## Completion Status",
        "",
        f"- Result rows: {report_data.result_row_count}",
        f"- Completed keys: {report_data.completed_key_count}",
        f"- Failed keys: {report_data.failed_key_count}",
        f"- Rolling test windows compared: {int(summary_row['window_count'])}",
        f"- Test period covered: {_format_date(period_start)} to {_format_date(period_end)}",
        f"- Benchmarks: `{', '.join(report_data.benchmark_symbols)}`",
        "",
        "## Current Champion",
        "",
        f"- Candidate name: `{CHAMPION_NAME}`",
        f"- Candidate spec: `{CHAMPION_SPEC}`",
        "- Model family: `baseline`",
        "- Model type: `linear_regression`",
        "- Feature set: `default_features_v1`",
        "- Signal: `top_k=1`, `min_holding_days=10`, `switch_margin=0.002`",
        "- Costs and target: `transaction_cost_bps=10`, "
        "`target_col=excess_return_vs_spy_20d`, `label_horizon_days=20`",
        "",
        "## Comparison Method",
        "",
        "This report compares the completed sliding-window champion study against SPY "
        "and QQQ over the identical rolling test windows. For each strategy test "
        "row, the benchmark return uses the first available close inside that same "
        "test_start/test_end range and the last available close inside that range.",
        "",
        "## Overall Strategy vs SPY/QQQ Summary",
        "",
        _markdown_table(_format_summary_for_markdown(report_data.summary)),
        "",
        "## Train-Window Comparison",
        "",
        _markdown_table(_format_group_for_markdown(report_data.by_train_window)),
        "",
        "## Year Comparison",
        "",
        _markdown_table(_format_group_for_markdown(report_data.by_year)),
        "",
        "## Worst Relative Windows",
        "",
        _markdown_table(_format_windows_for_markdown(worst_relative)),
        "",
        "## Final Recommendation",
        "",
        _final_recommendation(report_data.window_metrics, summary_row),
        "",
    ]
    return "\n".join(lines)


def parse_comma_separated_symbols(value: str) -> tuple[str, ...]:
    """Parse comma-separated ticker symbols."""

    symbols = tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())
    if not symbols:
        raise argparse.ArgumentTypeError("At least one symbol is required.")
    return symbols


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a report-only benchmark comparison from completed "
            "sliding-window champion study results."
        )
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--market-data", type=Path, default=DEFAULT_MARKET_DATA)
    parser.add_argument("--failed-keys", type=Path, default=DEFAULT_FAILED_KEYS)
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
    required_columns = {"fold", "split", "total_return", "sharpe_like", "max_drawdown"}
    missing = sorted(required_columns.difference(results.columns))
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")


def _build_enriched_strategy_test_rows(results: pd.DataFrame) -> pd.DataFrame:
    test_rows = results.loc[results["split"] == "test"].copy()
    if test_rows.empty:
        return test_rows

    metadata_rows = []
    for fold_name in test_rows["fold"]:
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

    metadata = pd.DataFrame(metadata_rows, index=test_rows.index)
    test_rows = pd.concat([test_rows, metadata], axis=1)
    for column in ["total_return", "sharpe_like", "max_drawdown"]:
        test_rows[column] = pd.to_numeric(test_rows[column], errors="coerce")

    return (
        test_rows.sort_values(
            by=["train_window_years", "test_start", "fold"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _build_window_metrics(
    strategy_rows: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    benchmark_symbols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, strategy_row in strategy_rows.iterrows():
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
                "fold": strategy_row["fold"],
                "train_window_years": int(strategy_row["train_window_years"]),
                "test_start": strategy_row["test_start"],
                "test_end": strategy_row["test_end"],
                "test_year": int(strategy_row["test_year"]),
                "strategy_total_return": strategy_total_return,
                "strategy_sharpe_like": strategy_row["sharpe_like"],
                "strategy_max_drawdown": strategy_row["max_drawdown"],
                "spy_total_return": spy_return,
                "qqq_total_return": qqq_return,
                "strategy_minus_spy": _difference_or_nan(strategy_total_return, spy_return),
                "strategy_minus_qqq": _difference_or_nan(strategy_total_return, qqq_return),
                "strategy_beats_spy": _beats_or_nan(strategy_total_return, spy_return),
                "strategy_beats_qqq": _beats_or_nan(strategy_total_return, qqq_return),
                "_sequence_number": int(strategy_row["sequence_number"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["train_window_years", "test_start", "_sequence_number", "fold"],
            kind="mergesort",
        )
        .drop(columns=["_sequence_number"])
        .reset_index(drop=True)[WINDOW_METRIC_COLUMNS]
    )


def _build_summary(window_metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_count": int(len(window_metrics)),
                "strategy_mean_total_return": window_metrics[
                    "strategy_total_return"
                ].mean(),
                "strategy_median_total_return": window_metrics[
                    "strategy_total_return"
                ].median(),
                "spy_mean_total_return": window_metrics["spy_total_return"].mean(),
                "spy_median_total_return": window_metrics["spy_total_return"].median(),
                "qqq_mean_total_return": window_metrics["qqq_total_return"].mean(),
                "qqq_median_total_return": window_metrics["qqq_total_return"].median(),
                "strategy_positive_window_rate": _positive_rate(
                    window_metrics["strategy_total_return"]
                ),
                "spy_positive_window_rate": _positive_rate(
                    window_metrics["spy_total_return"]
                ),
                "qqq_positive_window_rate": _positive_rate(
                    window_metrics["qqq_total_return"]
                ),
                "strategy_beats_spy_rate": _boolean_rate(
                    window_metrics["strategy_beats_spy"]
                ),
                "strategy_beats_qqq_rate": _boolean_rate(
                    window_metrics["strategy_beats_qqq"]
                ),
                "mean_strategy_minus_spy": window_metrics["strategy_minus_spy"].mean(),
                "mean_strategy_minus_qqq": window_metrics["strategy_minus_qqq"].mean(),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _build_train_window_summary(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"train_window_years": name, **_aggregate_group(group).to_dict()}
        for name, group in window_metrics.groupby("train_window_years", sort=True)
    ]
    output = pd.DataFrame(rows, columns=["train_window_years", *GROUP_COLUMNS])
    return _cast_count_columns(output, ["train_window_years", "count"])


def _build_year_summary(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"test_year": name, **_aggregate_group(group).to_dict()}
        for name, group in window_metrics.groupby("test_year", sort=True)
    ]
    output = pd.DataFrame(rows, columns=["test_year", *GROUP_COLUMNS])
    return _cast_count_columns(output, ["test_year", "count"])


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "count": int(len(group)),
            "strategy_mean_total_return": group["strategy_total_return"].mean(),
            "spy_mean_total_return": group["spy_total_return"].mean(),
            "qqq_mean_total_return": group["qqq_total_return"].mean(),
            "strategy_beats_spy_rate": _boolean_rate(group["strategy_beats_spy"]),
            "strategy_beats_qqq_rate": _boolean_rate(group["strategy_beats_qqq"]),
        }
    )


def _build_relative_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        window_metrics.sort_values(
            by=["strategy_minus_spy", "strategy_minus_qqq", "fold"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
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


def _count_text_file_lines(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def _cast_count_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        output[column] = output[column].astype(int)
    return output


def _validate_market_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")


def _format_summary_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    for column in output.columns:
        if column != "window_count":
            output[column] = output[column].map(_format_percent)
    return output


def _format_group_for_markdown(group: pd.DataFrame) -> pd.DataFrame:
    output = group.copy()
    for column in output.columns:
        if column not in {"train_window_years", "test_year", "count"}:
            output[column] = output[column].map(_format_percent)
    return output


def _format_windows_for_markdown(windows: pd.DataFrame) -> pd.DataFrame:
    output = windows[
        [
            "fold",
            "train_window_years",
            "test_start",
            "test_end",
            "strategy_total_return",
            "spy_total_return",
            "qqq_total_return",
            "strategy_minus_spy",
            "strategy_minus_qqq",
            "strategy_beats_spy",
            "strategy_beats_qqq",
        ]
    ].copy()
    for column in [
        "strategy_total_return",
        "spy_total_return",
        "qqq_total_return",
        "strategy_minus_spy",
        "strategy_minus_qqq",
    ]:
        output[column] = output[column].map(_format_percent)
    for column in ["strategy_beats_spy", "strategy_beats_qqq"]:
        output[column] = output[column].map(_format_boolean)
    return output


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    separator = ["---" for _ in headers]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _final_recommendation(window_metrics: pd.DataFrame, summary_row: pd.Series) -> str:
    spy_beat_rate = float(summary_row["strategy_beats_spy_rate"])
    qqq_beat_rate = float(summary_row["strategy_beats_qqq_rate"])
    strategy_mean = float(summary_row["strategy_mean_total_return"])
    strategy_median = float(summary_row["strategy_median_total_return"])
    positive_rate = float(summary_row["strategy_positive_window_rate"])

    benchmark_stability_confirmed = spy_beat_rate > 0.55 and qqq_beat_rate > 0.55
    absolute_returns_weak = (
        strategy_mean <= 0.01 or strategy_median <= 0.0 or positive_rate <= 0.55
    )
    difficult_window_note = ""
    if absolute_returns_weak and _beats_benchmarks_in_difficult_windows(window_metrics):
        difficult_window_note = (
            " The weak absolute rolling returns may partly be market-regime related, "
            "because the strategy still beat the benchmarks in difficult benchmark "
            "windows."
        )

    if not benchmark_stability_confirmed:
        return (
            "Benchmark-relative stability is not confirmed. The champion should remain "
            "a research-stage candidate, not production-ready."
            f"{difficult_window_note} Next research should prioritize longer test "
            "windows and a regime filter, then compare defensive references; if the "
            "relative gaps persist, move toward ranking/regime research."
        )

    return (
        "Benchmark-relative results are directionally encouraging, but the champion "
        "should remain a research-stage candidate, not production-ready."
        f"{difficult_window_note} Next research should still test longer windows, add "
        "a regime filter, compare defensive references, and continue ranking/regime "
        "research before making any deployment decision."
    )


def _beats_benchmarks_in_difficult_windows(window_metrics: pd.DataFrame) -> bool:
    difficult = window_metrics.loc[
        (window_metrics["spy_total_return"] <= 0) & (window_metrics["qqq_total_return"] <= 0)
    ]
    if difficult.empty:
        return False
    spy_rate = _boolean_rate(difficult["strategy_beats_spy"])
    qqq_rate = _boolean_rate(difficult["strategy_beats_qqq"])
    if pd.isna(spy_rate) or pd.isna(qqq_rate):
        return False
    return spy_rate > 0.5 and qqq_rate > 0.5


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_boolean(value: object) -> str:
    if pd.isna(value):
        return ""
    return "yes" if bool(value) else "no"


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def main() -> int:
    return run_sliding_window_benchmark_comparison_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
