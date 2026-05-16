"""Generate a report from completed sliding-window champion study results."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CHAMPION_NAME = "linear_smooth10_margin002_v1"
CHAMPION_SPEC = "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002"
DEFAULT_RESULTS_CSV = Path("artifacts/sweeps/sliding_window_champion_study_results.csv")
DEFAULT_FAILED_KEYS = Path(
    "artifacts/sweeps/sliding_window_champion_study_failed_keys.txt"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/sliding_window_champion_study")

SUMMARY_COLUMNS = [
    "test_window_count",
    "mean_total_return",
    "median_total_return",
    "min_total_return",
    "max_total_return",
    "positive_window_rate",
    "mean_sharpe_like",
    "median_sharpe_like",
    "mean_max_drawdown",
    "worst_max_drawdown",
    "mean_daily_volatility",
]

GROUP_COLUMNS = [
    "count",
    "mean_total_return",
    "median_total_return",
    "positive_window_rate",
    "mean_sharpe_like",
    "worst_max_drawdown",
    "mean_daily_volatility",
]

YEAR_GROUP_COLUMNS = [
    "count",
    "mean_total_return",
    "median_total_return",
    "positive_window_rate",
    "mean_sharpe_like",
    "worst_max_drawdown",
]

WINDOW_COLUMNS = [
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "test_year",
    "total_return",
    "sharpe_like",
    "max_drawdown",
    "daily_volatility",
]

_FOLD_NAME_RE = re.compile(
    r"^tw(?P<train_window_years>\d+)y"
    r"_v(?P<validation_window_months>\d+)m"
    r"_t(?P<test_window_months>\d+)m"
    r"_q(?P<sequence_number>\d+)"
    r"_(?P<test_start>\d{8})"
    r"_(?P<test_end>\d{8})$"
)


@dataclass(frozen=True)
class SlidingWindowFoldMetadata:
    """Metadata encoded in one sliding-window fold name."""

    train_window_years: int
    validation_window_months: int
    test_window_months: int
    sequence_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def test_year(self) -> int:
        return int(self.test_start.year)


@dataclass(frozen=True)
class SlidingWindowReportData:
    """All table outputs produced by the report builder."""

    summary: pd.DataFrame
    by_train_window: pd.DataFrame
    by_year: pd.DataFrame
    worst_windows: pd.DataFrame
    best_windows: pd.DataFrame
    test_rows: pd.DataFrame
    failed_key_count: int
    completed_key_count: int
    result_row_count: int


def run_sliding_window_stability_report_from_args(argv: list[str] | None = None) -> int:
    """Build a local report pack from completed sliding-window study CSV rows."""

    args = _parse_args(argv)
    report_data = build_sliding_window_stability_report(
        results_csv=args.results_csv,
        failed_keys=args.failed_keys,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "sliding_window_summary.csv"
    by_train_window_path = output_dir / "sliding_window_by_train_window.csv"
    by_year_path = output_dir / "sliding_window_by_year.csv"
    worst_windows_path = output_dir / "sliding_window_worst_windows.csv"
    best_windows_path = output_dir / "sliding_window_best_windows.csv"
    markdown_path = output_dir / "sliding_window_stability_report.md"

    report_data.summary.to_csv(summary_path, index=False)
    report_data.by_train_window.to_csv(by_train_window_path, index=False)
    report_data.by_year.to_csv(by_year_path, index=False)
    report_data.worst_windows.to_csv(worst_windows_path, index=False)
    report_data.best_windows.to_csv(best_windows_path, index=False)
    markdown_path.write_text(
        render_sliding_window_stability_markdown(report_data),
        encoding="utf-8",
    )

    print("US Market ML sliding-window stability report")
    print(f"champion: {CHAMPION_NAME}")
    print(f"results CSV: {args.results_csv}")
    print(f"failed keys: {args.failed_keys}")
    print(f"output summary CSV: {summary_path}")
    print(f"output train-window CSV: {by_train_window_path}")
    print(f"output year CSV: {by_year_path}")
    print(f"output worst windows CSV: {worst_windows_path}")
    print(f"output best windows CSV: {best_windows_path}")
    print(f"output markdown report: {markdown_path}")
    print()
    print(report_data.summary.to_string(index=False))

    return 0


def parse_sliding_window_fold_name(fold_name: str) -> SlidingWindowFoldMetadata:
    """Parse a sliding-window fold name into report metadata."""

    match = _FOLD_NAME_RE.match(fold_name)
    if match is None:
        raise ValueError(
            "Sliding-window fold must use format "
            "tw{years}y_v{months}m_t{months}m_q{sequence}_{YYYYMMDD}_{YYYYMMDD}."
        )

    parts = match.groupdict()
    test_start = pd.to_datetime(parts["test_start"], format="%Y%m%d")
    test_end = pd.to_datetime(parts["test_end"], format="%Y%m%d")
    if test_start > test_end:
        raise ValueError(f"Fold {fold_name!r} has test_start after test_end.")

    return SlidingWindowFoldMetadata(
        train_window_years=int(parts["train_window_years"]),
        validation_window_months=int(parts["validation_window_months"]),
        test_window_months=int(parts["test_window_months"]),
        sequence_number=int(parts["sequence_number"]),
        test_start=test_start,
        test_end=test_end,
    )


def build_sliding_window_stability_report(
    *,
    results_csv: Path,
    failed_keys: Path | None = None,
    top_n: int = 10,
) -> SlidingWindowReportData:
    """Read completed sweep results and return stability report tables."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    results = pd.read_csv(results_csv)
    _validate_results_columns(results)
    test_rows = _build_enriched_test_rows(results)
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    summary = _build_summary(test_rows)
    by_train_window = _build_train_window_summary(test_rows)
    by_year = _build_year_summary(test_rows)
    worst_windows = _build_worst_windows(test_rows, top_n)
    best_windows = _build_best_windows(test_rows, top_n)

    failed_key_count = _count_text_file_lines(failed_keys)
    completed_key_count = int(results["run_key"].nunique()) if "run_key" in results else 0

    return SlidingWindowReportData(
        summary=summary,
        by_train_window=by_train_window,
        by_year=by_year,
        worst_windows=worst_windows,
        best_windows=best_windows,
        test_rows=test_rows,
        failed_key_count=failed_key_count,
        completed_key_count=completed_key_count,
        result_row_count=len(results),
    )


def render_sliding_window_stability_markdown(report_data: SlidingWindowReportData) -> str:
    """Render the stability report as Markdown."""

    summary_row = report_data.summary.iloc[0]
    positive_rate = float(summary_row["positive_window_rate"])
    median_return = float(summary_row["median_total_return"])
    mean_return = float(summary_row["mean_total_return"])
    verdict = _stability_verdict(positive_rate, median_return)
    period_start = report_data.test_rows["test_start"].min()
    period_end = report_data.test_rows["test_end"].max()

    lines = [
        "# Sliding-Window Champion Stability Report",
        "",
        "## Completion Status",
        "",
        f"- Result rows: {report_data.result_row_count}",
        f"- Completed keys: {report_data.completed_key_count}",
        f"- Failed keys: {report_data.failed_key_count}",
        f"- Rolling test windows: {int(summary_row['test_window_count'])}",
        f"- Test period covered: {_format_date(period_start)} to {_format_date(period_end)}",
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
        "## Interpretation Note",
        "",
        "Each rolling test window is short, generally about three months and sometimes "
        "shorter for the partial final window. These total returns are useful for "
        "stability diagnostics, but they are not directly comparable to multi-year "
        "fixed-fold returns.",
        "",
        "## Overall Rolling-Window Stability Result",
        "",
        f"{verdict} Mean rolling test total return is "
        f"{_format_percent(mean_return)}, median is {_format_percent(median_return)}, "
        f"and the positive-window rate is {_format_percent(positive_rate)}.",
        "",
        _markdown_table(_format_summary_for_markdown(report_data.summary)),
        "",
        "## Train-Window Comparison",
        "",
        _markdown_table(_format_group_for_markdown(report_data.by_train_window)),
        "",
        "## Year/Period Breakdown",
        "",
        _markdown_table(_format_group_for_markdown(report_data.by_year)),
        "",
        "## Worst Windows",
        "",
        _markdown_table(_format_windows_for_markdown(report_data.worst_windows)),
        "",
        "## Best Windows",
        "",
        _markdown_table(_format_windows_for_markdown(report_data.best_windows)),
        "",
        "## Final Recommendation",
        "",
        _final_recommendation(positive_rate, median_return),
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a report-only stability summary from completed "
            "sliding-window champion study results."
        )
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--failed-keys", type=Path, default=DEFAULT_FAILED_KEYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be positive.")
    return args


def _validate_results_columns(results: pd.DataFrame) -> None:
    required_columns = {
        "fold",
        "split",
        "total_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
    }
    missing = sorted(required_columns.difference(results.columns))
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")


def _build_enriched_test_rows(results: pd.DataFrame) -> pd.DataFrame:
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
    for column in ["total_return", "daily_volatility", "sharpe_like", "max_drawdown"]:
        test_rows[column] = pd.to_numeric(test_rows[column], errors="coerce")
    return (
        test_rows.sort_values(
            by=["train_window_years", "test_start", "fold"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _build_summary(test_rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_window_count": int(len(test_rows)),
                "mean_total_return": test_rows["total_return"].mean(),
                "median_total_return": test_rows["total_return"].median(),
                "min_total_return": test_rows["total_return"].min(),
                "max_total_return": test_rows["total_return"].max(),
                "positive_window_rate": (test_rows["total_return"] > 0).mean(),
                "mean_sharpe_like": test_rows["sharpe_like"].mean(),
                "median_sharpe_like": test_rows["sharpe_like"].median(),
                "mean_max_drawdown": test_rows["max_drawdown"].mean(),
                "worst_max_drawdown": test_rows["max_drawdown"].min(),
                "mean_daily_volatility": test_rows["daily_volatility"].mean(),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _build_train_window_summary(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"train_window_years": name, **_aggregate_group(group).to_dict()}
        for name, group in test_rows.groupby("train_window_years", sort=True)
    ]
    output = pd.DataFrame(rows, columns=["train_window_years", *GROUP_COLUMNS])
    return _cast_count_columns(output, ["train_window_years", "count"])


def _build_year_summary(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"test_year": name, **_aggregate_year_group(group).to_dict()}
        for name, group in test_rows.groupby("test_year", sort=True)
    ]
    output = pd.DataFrame(rows, columns=["test_year", *YEAR_GROUP_COLUMNS])
    return _cast_count_columns(output, ["test_year", "count"])


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "count": int(len(group)),
            "mean_total_return": group["total_return"].mean(),
            "median_total_return": group["total_return"].median(),
            "positive_window_rate": (group["total_return"] > 0).mean(),
            "mean_sharpe_like": group["sharpe_like"].mean(),
            "worst_max_drawdown": group["max_drawdown"].min(),
            "mean_daily_volatility": group["daily_volatility"].mean(),
        }
    )


def _aggregate_year_group(group: pd.DataFrame) -> pd.Series:
    aggregated = _aggregate_group(group)
    return aggregated.drop(labels=["mean_daily_volatility"])


def _build_worst_windows(test_rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    worst_return = (
        test_rows.sort_values(["total_return", "fold"], ascending=[True, True])
        .head(top_n)
        .assign(ranking="worst_total_return")
    )
    worst_drawdown = (
        test_rows.sort_values(["max_drawdown", "fold"], ascending=[True, True])
        .head(top_n)
        .assign(ranking="worst_max_drawdown")
    )
    return pd.concat([worst_return, worst_drawdown], ignore_index=True)[
        ["ranking", *WINDOW_COLUMNS]
    ]


def _build_best_windows(test_rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return (
        test_rows.sort_values(["total_return", "fold"], ascending=[False, True])
        .head(top_n)
        .assign(ranking="best_total_return")[["ranking", *WINDOW_COLUMNS]]
        .reset_index(drop=True)
    )


def _count_text_file_lines(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def _cast_count_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        output[column] = output[column].astype(int)
    return output


def _format_summary_for_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    percent_columns = [
        "mean_total_return",
        "median_total_return",
        "min_total_return",
        "max_total_return",
        "positive_window_rate",
        "mean_sharpe_like",
        "median_sharpe_like",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_daily_volatility",
    ]
    for column in percent_columns:
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
            "ranking",
            "fold",
            "train_window_years",
            "test_start",
            "test_end",
            "total_return",
            "sharpe_like",
            "max_drawdown",
            "daily_volatility",
        ]
    ].copy()
    for column in ["total_return", "sharpe_like", "max_drawdown", "daily_volatility"]:
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


def _stability_verdict(positive_rate: float, median_return: float) -> str:
    if 0.45 <= positive_rate <= 0.55 and median_return <= 0.01:
        return "Sliding-window stability is not confirmed."
    if positive_rate < 0.6 or median_return <= 0:
        return "Sliding-window stability remains inconclusive."
    return "Sliding-window stability is directionally encouraging but still research-stage."


def _final_recommendation(positive_rate: float, median_return: float) -> str:
    if 0.45 <= positive_rate <= 0.55 and median_return <= 0.01:
        return (
            "Treat this champion as a research-stage candidate, not production-ready. "
            "Because the rolling distribution has an approximately 50% positive-window "
            "rate and a median return near or below zero, stability under rolling "
            "retraining is not confirmed. Next steps should compare the same windows "
            "against SPY and QQQ, test longer test windows, add a regime filter, or "
            "move the work toward ranking/regime research."
        )
    return (
        "Keep the champion in research-stage evaluation. Before increasing confidence, "
        "compare the same windows against SPY and QQQ, test longer test windows, add a "
        "regime filter, or move the work toward ranking/regime research."
    )


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def main() -> int:
    return run_sliding_window_stability_report_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
