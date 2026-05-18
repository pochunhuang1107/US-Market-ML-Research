"""Create the Tiingo benchmark-gap feature study report."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_modern_baseline_sweep_report import (
    DATASET_SOURCE_LABEL,
    REQUIRED_RESULTS_COLUMNS,
    _as_float,
    _format_for_markdown,
    _markdown_table,
    _positive_rate,
    _validate_columns,
)
from us_market_ml.cli.run_tiingo_benchmark_gap_feature_study import (
    BENCHMARK_GAP_FEATURE_SET_VERSION,
    BENCHMARK_GAP_V2_FEATURE_SET_VERSION,
    DEFAULT_DATASET_GLOB,
    DEFAULT_FEATURE_SET_VERSION,
)
from us_market_ml.data import read_local_market_data

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_benchmark_gap_feature_study/"
    "tiingo_benchmark_gap_feature_study_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_benchmark_gap_feature_study")

SUMMARY_PATH_NAME = "tiingo_benchmark_gap_feature_summary.csv"
DECISION_MATRIX_PATH_NAME = "tiingo_benchmark_gap_feature_decision_matrix.csv"
REPORT_PATH_NAME = "tiingo_benchmark_gap_feature_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

BENCHMARK_SYMBOLS = ("SPY", "QQQ")
SELECTED_CANDIDATE_ROLE = "selected_candidate"
COMPARISON_FEATURE_SET_VERSIONS = (
    DEFAULT_FEATURE_SET_VERSION,
    BENCHMARK_GAP_FEATURE_SET_VERSION,
    BENCHMARK_GAP_V2_FEATURE_SET_VERSION,
)
DRAWDOWN_SEVERE_WORSENING_THRESHOLD = -0.05

SUMMARY_GROUP_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "candidate_family",
    "candidate_role",
    "model_family",
    "model_type",
    "feature_set_version",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
]

SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_key",
    "candidate_label",
    *SUMMARY_GROUP_COLUMNS,
    "window_count",
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
    "delta_average_return_vs_default",
    "delta_strategy_vs_spy_vs_default",
    "delta_strategy_vs_qqq_vs_default",
    "delta_average_return_vs_benchmark_gap_v1",
    "delta_strategy_vs_spy_vs_benchmark_gap_v1",
    "delta_strategy_vs_qqq_vs_benchmark_gap_v1",
    "delta_worst_max_drawdown_vs_benchmark_gap_v1",
    "benchmark_gap_change",
    "benchmark_gap_change_vs_benchmark_gap_v1",
    "benchmark_status",
]

DECISION_GROUP_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "candidate_family",
    "candidate_role",
    "model_family",
    "model_type",
    "feature_set_version",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_key",
    "candidate_label",
    *DECISION_GROUP_COLUMNS,
    "window_count",
    "average_total_return",
    "positive_window_rate",
    "average_sharpe_like",
    "worst_max_drawdown",
    "spy_average_total_return",
    "qqq_average_total_return",
    "average_strategy_vs_spy",
    "average_strategy_vs_qqq",
    "spy_underperformance_rate",
    "qqq_underperformance_rate",
    "delta_average_return_vs_default",
    "delta_strategy_vs_spy_vs_default",
    "delta_strategy_vs_qqq_vs_default",
    "delta_average_return_vs_benchmark_gap_v1",
    "delta_strategy_vs_spy_vs_benchmark_gap_v1",
    "delta_strategy_vs_qqq_vs_benchmark_gap_v1",
    "delta_worst_max_drawdown_vs_benchmark_gap_v1",
    "benchmark_gap_change",
    "benchmark_gap_change_vs_benchmark_gap_v1",
    "benchmark_status",
    "feature_decision",
    "next_action",
    "rationale",
]

_FOLD_NAME_RE = re.compile(
    r"tw(?P<train_window_years>\d+)y"
    r"_v(?P<validation_window_months>\d+)m"
    r"_t(?P<test_window_months>\d+)m"
    r"_q(?P<sequence_number>\d+)"
    r"_(?P<test_start>\d{8})"
    r"_(?P<test_end>\d{8})"
)


@dataclass(frozen=True)
class BenchmarkGapFeatureReportData:
    """Tables and metadata produced by the benchmark-gap feature report."""

    feature_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    dataset_source: str = DATASET_SOURCE_LABEL


def run_tiingo_benchmark_gap_feature_report_from_args(argv: list[str] | None = None) -> int:
    """Build and write the benchmark-gap feature study report."""

    args = _parse_args(argv)
    report_data = build_tiingo_benchmark_gap_feature_report(
        results_csv=args.results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_benchmark_gap_feature_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo benchmark-gap feature study report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    print(f"market dataset: {args.market_dataset_glob}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.decision_matrix.head(20).to_string(index=False))
    return 0


def build_tiingo_benchmark_gap_feature_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    market_dataset_glob: str | Path = DEFAULT_DATASET_GLOB,
) -> BenchmarkGapFeatureReportData:
    """Read completed feature-study results and create report tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(
        results,
        [*REQUIRED_RESULTS_COLUMNS, "feature_set_version"],
        path=results_csv,
    )
    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    benchmark_by_fold = _benchmark_metrics_by_fold(
        raw_df=read_local_market_data(market_dataset_glob),
        fold_names=test_rows["fold"].astype(str).dropna().unique().tolist(),
    )
    enriched = _enrich_test_rows(test_rows).merge(
        benchmark_by_fold,
        on="fold",
        how="left",
        validate="many_to_one",
    )
    enriched = _add_benchmark_gap_columns(enriched)
    included = enriched.loc[
        enriched["model_family"].astype(str).eq("baseline")
        & enriched["feature_set_version"]
        .astype(str)
        .isin(COMPARISON_FEATURE_SET_VERSIONS)
    ].copy()
    if included.empty:
        raise ValueError(
            "No default/benchmark-gap baseline test rows were found in the results CSV."
        )

    feature_summary = _sort_summary(
        _add_default_deltas(_aggregate(included, SUMMARY_GROUP_COLUMNS))
    )
    decision_matrix = _sort_decision_matrix(
        _add_decision_columns(
            _add_default_deltas(_aggregate(included, DECISION_GROUP_COLUMNS))
        )
    )

    return BenchmarkGapFeatureReportData(
        feature_summary=feature_summary,
        decision_matrix=decision_matrix,
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
    )


def write_tiingo_benchmark_gap_feature_report(
    *,
    report_data: BenchmarkGapFeatureReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.feature_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_benchmark_gap_feature_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_benchmark_gap_feature_markdown(
    report_data: BenchmarkGapFeatureReportData,
) -> str:
    """Render the benchmark-gap feature study Markdown report."""

    decision = report_data.decision_matrix
    selected_rows = decision.loc[
        decision["candidate_role"].astype(str).eq(SELECTED_CANDIDATE_ROLE)
    ].copy()
    selected_table = _format_for_markdown(
        selected_rows[
            [
                "feature_set_version",
                "average_total_return",
                "spy_average_total_return",
                "qqq_average_total_return",
                "average_strategy_vs_spy",
                "average_strategy_vs_qqq",
                "spy_underperformance_rate",
                "qqq_underperformance_rate",
                "delta_strategy_vs_spy_vs_default",
                "delta_strategy_vs_qqq_vs_default",
                "delta_strategy_vs_spy_vs_benchmark_gap_v1",
                "delta_strategy_vs_qqq_vs_benchmark_gap_v1",
                "delta_worst_max_drawdown_vs_benchmark_gap_v1",
                "benchmark_gap_change",
                "benchmark_gap_change_vs_benchmark_gap_v1",
                "benchmark_status",
            ]
        ],
        exclude={
            "feature_set_version",
            "benchmark_gap_change",
            "benchmark_gap_change_vs_benchmark_gap_v1",
            "benchmark_status",
        },
    )
    v1_candidate = _feature_row(selected_rows, BENCHMARK_GAP_FEATURE_SET_VERSION)
    v2_candidate = _feature_row(selected_rows, BENCHMARK_GAP_V2_FEATURE_SET_VERSION)
    benchmark_candidate = v2_candidate if v2_candidate is not None else v1_candidate
    default_candidate = _feature_row(selected_rows, DEFAULT_FEATURE_SET_VERSION)
    status = (
        str(benchmark_candidate.get("benchmark_status", "unknown"))
        if benchmark_candidate is not None
        else "benchmark-gap feature row missing"
    )
    gap_change = (
        str(
            benchmark_candidate.get(
                "benchmark_gap_change_vs_benchmark_gap_v1"
                if v2_candidate is not None
                else "benchmark_gap_change",
                "unknown",
            )
        )
        if benchmark_candidate is not None
        else "unknown"
    )
    default_return = (
        _fmt(default_candidate["average_total_return"])
        if default_candidate is not None
        else "nan"
    )
    feature_return = (
        _fmt(benchmark_candidate["average_total_return"])
        if benchmark_candidate is not None
        else "nan"
    )
    v1_return = (
        _fmt(v1_candidate["average_total_return"]) if v1_candidate is not None else "nan"
    )

    reference_rows = decision.loc[
        ~decision["candidate_role"].astype(str).eq(SELECTED_CANDIDATE_ROLE)
    ].copy()
    reference_table = _format_for_markdown(
        reference_rows[
            [
                "candidate_role",
                "feature_set_version",
                "candidate_label",
                "average_total_return",
                "average_strategy_vs_spy",
                "average_strategy_vs_qqq",
                "benchmark_gap_change",
                "benchmark_gap_change_vs_benchmark_gap_v1",
                "feature_decision",
            ]
        ].head(30),
        exclude={
            "candidate_role",
            "feature_set_version",
            "candidate_label",
            "benchmark_gap_change",
            "benchmark_gap_change_vs_benchmark_gap_v1",
            "feature_decision",
        },
    )

    lines = [
        "# Tiingo Benchmark-Gap Feature Study Report",
        "",
        "## Executive Summary",
        "",
        "This report compares `default_features_v1`, `benchmark_gap_features_v1`, "
        "and `benchmark_gap_features_v2` for the current selected Tiingo-only "
        "ridge top5 candidate. The study is local-only, baseline-only, and "
        "intentionally excludes tree models, MLPs, S3, and production signal export.",
        "",
        f"Input result rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included feature-study test rows: "
        f"`{report_data.included_test_row_count}`.",
        "",
        f"Selected default average return: `{default_return}`. Selected v1 average "
        f"return: `{v1_return}`. Selected latest benchmark-gap average return: "
        f"`{feature_return}`.",
        "",
        f"Benchmark-gap feature verdict: `{gap_change}`. Benchmark status: `{status}`.",
        "",
        "## Selected Candidate Feature Comparison",
        "",
        _markdown_table(selected_table),
        "",
        "## Decision Matrix",
        "",
        _markdown_table(
            _format_for_markdown(
                decision[
                    [
                        "candidate_label",
                        "feature_set_version",
                        "average_total_return",
                        "positive_window_rate",
                        "average_sharpe_like",
                        "worst_max_drawdown",
                        "average_strategy_vs_spy",
                        "average_strategy_vs_qqq",
                        "delta_strategy_vs_spy_vs_benchmark_gap_v1",
                        "delta_strategy_vs_qqq_vs_benchmark_gap_v1",
                        "benchmark_gap_change",
                        "benchmark_gap_change_vs_benchmark_gap_v1",
                        "benchmark_status",
                        "feature_decision",
                        "next_action",
                    ]
                ].head(30),
                exclude={
                    "candidate_label",
                    "feature_set_version",
                    "benchmark_gap_change",
                    "benchmark_gap_change_vs_benchmark_gap_v1",
                    "benchmark_status",
                    "feature_decision",
                    "next_action",
                },
            )
        ),
        "",
        "## Reference Rows",
        "",
        _markdown_table(reference_table),
        "",
        "## Interpretation",
        "",
        "- `average_strategy_vs_spy` and `average_strategy_vs_qqq` are simple average "
        "test-window return gaps versus buy-and-hold benchmarks over the same folds.",
        "- `underperformance_rate` is the share of test windows where the strategy "
        "return was below the benchmark return.",
        "- A positive feature delta is useful evidence, but it is not enough by itself "
        "to mark the candidate production-ready.",
        "",
        "## Deferrals And Caveats",
        "",
        "- `tree_screen` and MLP remain deferred.",
        "- This report does not export production signals.",
        "- Tiingo-only research source; Massive app data may not match these results exactly.",
        "- If SPY/QQQ still win, the honest next step is more feature engineering or "
        "risk-aware allocation work, not promotion.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo benchmark-gap feature study report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


def _enrich_test_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    output = test_rows.copy()
    output["dataset_source"] = DATASET_SOURCE_LABEL
    for column, default in [
        ("sweep_profile", "benchmark_gap_feature_study"),
        ("candidate_family", "unknown"),
        ("candidate_role", "unknown"),
        ("allocation_mode", "equal_weight"),
    ]:
        if column not in output.columns:
            output[column] = default
        else:
            output[column] = output[column].fillna(default)

    for column in [
        "top_k",
        "min_holding_days",
        "switch_margin",
        "transaction_cost_bps",
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "total_transaction_cost",
        "mean_daily_turnover",
    ]:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")

    for metadata_column in [
        "train_window_years",
        "validation_window_months",
        "test_window_months",
        "sequence_number",
        "test_start",
        "test_end",
    ]:
        if metadata_column not in output.columns:
            output[metadata_column] = np.nan
    output["test_start"] = output["test_start"].astype("object")
    output["test_end"] = output["test_end"].astype("object")

    parsed_metadata = output["fold"].astype(str).map(_parse_fold_metadata)
    for index, metadata in parsed_metadata.items():
        if metadata is None:
            continue
        for column, value in metadata.items():
            if column in output.columns and pd.isna(output.at[index, column]):
                output.at[index, column] = value

    for column in [
        "top_k",
        "min_holding_days",
        "train_window_years",
        "validation_window_months",
        "test_window_months",
    ]:
        output[column] = output[column].astype("Int64")

    output["candidate_key"] = output.apply(_candidate_key_from_row, axis=1)
    output["candidate_id"] = output.apply(_candidate_id_from_row, axis=1)
    output["candidate_label"] = output.apply(_candidate_label_from_row, axis=1)
    return output.reset_index(drop=True)


def _benchmark_metrics_by_fold(raw_df: pd.DataFrame, fold_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_name in fold_names:
        metadata = _parse_fold_metadata(fold_name)
        row: dict[str, object] = {"fold": fold_name}
        if metadata is None:
            for symbol in BENCHMARK_SYMBOLS:
                prefix = symbol.lower()
                row[f"{prefix}_total_return"] = np.nan
                row[f"{prefix}_sharpe_like"] = np.nan
                row[f"{prefix}_max_drawdown"] = np.nan
            rows.append(row)
            continue
        test_start = pd.Timestamp(str(metadata["test_start"]))
        test_end = pd.Timestamp(str(metadata["test_end"]))
        for symbol in BENCHMARK_SYMBOLS:
            metrics = _benchmark_metrics(raw_df, symbol, test_start, test_end)
            prefix = symbol.lower()
            row[f"{prefix}_total_return"] = metrics["total_return"]
            row[f"{prefix}_sharpe_like"] = metrics["sharpe_like"]
            row[f"{prefix}_max_drawdown"] = metrics["max_drawdown"]
        rows.append(row)
    return pd.DataFrame(rows)


def _benchmark_metrics(
    raw_df: pd.DataFrame,
    symbol: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict[str, float]:
    benchmark = raw_df.loc[
        (raw_df["symbol"].astype("string").str.upper() == symbol)
        & (pd.to_datetime(raw_df["date"]) >= test_start)
        & (pd.to_datetime(raw_df["date"]) <= test_end),
        ["date", "close"],
    ].copy()
    if benchmark.empty:
        return {"total_return": np.nan, "sharpe_like": np.nan, "max_drawdown": np.nan}
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = (
        benchmark.sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    returns = pd.to_numeric(benchmark["close"], errors="coerce").pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(returns.std())
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe_like": (
            float(returns.mean() / volatility)
            if volatility > 0.0 and not np.isnan(volatility)
            else np.nan
        ),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
    }


def _add_benchmark_gap_columns(test_rows: pd.DataFrame) -> pd.DataFrame:
    output = test_rows.copy()
    output["strategy_vs_spy"] = output["total_return"] - output["spy_total_return"]
    output["strategy_vs_qqq"] = output["total_return"] - output["qqq_total_return"]
    output["underperformed_spy"] = np.where(
        output["strategy_vs_spy"].isna(),
        np.nan,
        output["strategy_vs_spy"] < 0.0,
    )
    output["underperformed_qqq"] = np.where(
        output["strategy_vs_qqq"].isna(),
        np.nan,
        output["strategy_vs_qqq"] < 0.0,
    )
    return output


def _aggregate(test_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in test_rows.groupby(group_cols, dropna=False, sort=False):
        first = group.iloc[0]
        row = {
            "candidate_id": first["candidate_id"],
            "candidate_key": first["candidate_key"],
            "candidate_label": first["candidate_label"],
            **dict(zip(group_cols, group_values, strict=True)),
            "window_count": int(len(group)),
            "average_total_return": group["total_return"].mean(),
            "median_total_return": group["total_return"].median(),
            "positive_window_rate": _positive_rate(group["total_return"]),
            "average_sharpe_like": group["sharpe_like"].mean(),
            "average_max_drawdown": group["max_drawdown"].mean(),
            "worst_max_drawdown": group["max_drawdown"].min(),
            "average_total_transaction_cost": group["total_transaction_cost"].mean(),
            "average_mean_daily_turnover": group["mean_daily_turnover"].mean(),
            "spy_average_total_return": group["spy_total_return"].mean(),
            "qqq_average_total_return": group["qqq_total_return"].mean(),
            "average_strategy_vs_spy": group["strategy_vs_spy"].mean(),
            "average_strategy_vs_qqq": group["strategy_vs_qqq"].mean(),
            "spy_underperformance_rate": _rate(group["underperformed_spy"]),
            "qqq_underperformance_rate": _rate(group["underperformed_qqq"]),
        }
        row["benchmark_status"] = _benchmark_status(row)
        rows.append(row)
    columns = SUMMARY_COLUMNS if group_cols == SUMMARY_GROUP_COLUMNS else DECISION_MATRIX_COLUMNS
    selected_columns = [column for column in columns if column in rows[0]] if rows else columns
    return pd.DataFrame(rows).reindex(columns=selected_columns)


def _add_default_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    default_metric_map = {
        "average_total_return": "delta_average_return_vs_default",
        "average_strategy_vs_spy": "delta_strategy_vs_spy_vs_default",
        "average_strategy_vs_qqq": "delta_strategy_vs_qqq_vs_default",
    }
    v1_metric_map = {
        "average_total_return": "delta_average_return_vs_benchmark_gap_v1",
        "average_strategy_vs_spy": "delta_strategy_vs_spy_vs_benchmark_gap_v1",
        "average_strategy_vs_qqq": "delta_strategy_vs_qqq_vs_benchmark_gap_v1",
        "worst_max_drawdown": "delta_worst_max_drawdown_vs_benchmark_gap_v1",
    }
    for output_column in [*default_metric_map.values(), *v1_metric_map.values()]:
        output[output_column] = np.nan

    default_lookup = {
        _default_comparison_key(pd.Series(row._asdict())): row
        for row in output.loc[
            output["feature_set_version"].astype(str).eq(DEFAULT_FEATURE_SET_VERSION)
        ].itertuples(index=False)
    }
    v1_lookup = {
        _default_comparison_key(pd.Series(row._asdict())): row
        for row in output.loc[
            output["feature_set_version"].astype(str).eq(BENCHMARK_GAP_FEATURE_SET_VERSION)
        ].itertuples(index=False)
    }
    for index, row in output.iterrows():
        default_row = default_lookup.get(_default_comparison_key(row))
        if default_row is not None:
            for metric_column, output_column in default_metric_map.items():
                output.at[index, output_column] = _as_float(row[metric_column]) - _as_float(
                    getattr(default_row, metric_column)
                )
        v1_row = v1_lookup.get(_default_comparison_key(row))
        if v1_row is not None:
            for metric_column, output_column in v1_metric_map.items():
                output.at[index, output_column] = _as_float(row[metric_column]) - _as_float(
                    getattr(v1_row, metric_column)
                )
    output["benchmark_gap_change"] = output.apply(_benchmark_gap_change, axis=1)
    output["benchmark_gap_change_vs_benchmark_gap_v1"] = output.apply(
        _benchmark_gap_change_vs_v1,
        axis=1,
    )
    return output


def _default_comparison_key(row: pd.Series) -> tuple[object, ...]:
    key_columns = [
        "candidate_key",
        "train_window_years",
        "validation_window_months",
        "test_window_months",
    ]
    return tuple(row[column] for column in key_columns if column in row.index)


def _add_decision_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["feature_decision"] = output.apply(_feature_decision, axis=1)
    output["next_action"] = output.apply(_next_action, axis=1)
    output["rationale"] = output.apply(_rationale, axis=1)
    return output


def _benchmark_gap_change(row: pd.Series) -> str:
    if str(row["feature_set_version"]) == DEFAULT_FEATURE_SET_VERSION:
        return "baseline comparator"
    spy_delta = _as_float(row.get("delta_strategy_vs_spy_vs_default"))
    qqq_delta = _as_float(row.get("delta_strategy_vs_qqq_vs_default"))
    if pd.isna(spy_delta) or pd.isna(qqq_delta):
        return "unknown"
    if spy_delta > 0.0 and qqq_delta > 0.0:
        return "improved versus SPY and QQQ"
    if spy_delta > 0.0 or qqq_delta > 0.0:
        return "mixed"
    return "not improved"


def _benchmark_gap_change_vs_v1(row: pd.Series) -> str:
    feature_set_version = str(row["feature_set_version"])
    if feature_set_version == DEFAULT_FEATURE_SET_VERSION:
        return "baseline comparator"
    if feature_set_version == BENCHMARK_GAP_FEATURE_SET_VERSION:
        return "v1 comparator"
    spy_delta = _as_float(row.get("delta_strategy_vs_spy_vs_benchmark_gap_v1"))
    qqq_delta = _as_float(row.get("delta_strategy_vs_qqq_vs_benchmark_gap_v1"))
    if pd.isna(spy_delta) or pd.isna(qqq_delta):
        return "unknown"
    if spy_delta > 0.0 and qqq_delta > 0.0:
        return "improved versus v1 SPY and QQQ gaps"
    if spy_delta > 0.0 or qqq_delta > 0.0:
        return "mixed versus v1"
    return "not improved versus v1"


def _benchmark_status(row: pd.Series | dict[str, object]) -> str:
    vs_spy = _as_float(row["average_strategy_vs_spy"])
    vs_qqq = _as_float(row["average_strategy_vs_qqq"])
    if pd.isna(vs_spy) or pd.isna(vs_qqq):
        return "benchmark data unavailable"
    if vs_spy > 0.0 and vs_qqq > 0.0:
        return "strategy beats SPY and QQQ on average"
    if vs_spy > 0.0:
        return "strategy beats SPY on average; QQQ still wins"
    if vs_qqq > 0.0:
        return "strategy beats QQQ on average; SPY still wins"
    return "SPY and QQQ still win on average"


def _feature_decision(row: pd.Series) -> str:
    feature_set_version = str(row["feature_set_version"])
    if feature_set_version == DEFAULT_FEATURE_SET_VERSION:
        return "current baseline comparator"
    if str(row["candidate_role"]) != SELECTED_CANDIDATE_ROLE:
        return "reference only"
    if feature_set_version == BENCHMARK_GAP_V2_FEATURE_SET_VERSION:
        spy_delta = _as_float(row.get("delta_strategy_vs_spy_vs_benchmark_gap_v1"))
        qqq_delta = _as_float(row.get("delta_strategy_vs_qqq_vs_benchmark_gap_v1"))
        drawdown_delta = _as_float(row.get("delta_worst_max_drawdown_vs_benchmark_gap_v1"))
        if qqq_delta > 0.0 and drawdown_delta <= DRAWDOWN_SEVERE_WORSENING_THRESHOLD:
            return "mixed; QQQ gap improved but drawdown worsened"
        if spy_delta > 0.0 and qqq_delta > 0.0:
            return "keep v2 for next feature-engineering iteration"
        if spy_delta > 0.0 or qqq_delta > 0.0:
            return "mixed; inspect v2 window drivers"
        return "reject/deprioritize v2"
    gap_change = str(row["benchmark_gap_change"])
    status = str(row["benchmark_status"])
    if gap_change == "improved versus SPY and QQQ":
        if status == "strategy beats SPY and QQQ on average":
            return "keep feature set for deeper validation"
        return "keep feature set for next feature-engineering iteration"
    if gap_change == "mixed":
        return "inconclusive; inspect window drivers"
    return "do not promote as-is"


def _next_action(row: pd.Series) -> str:
    decision = str(row.get("feature_decision", ""))
    if decision == "keep feature set for deeper validation":
        return "run richer replay before any promotion"
    if decision == "keep feature set for next feature-engineering iteration":
        return "continue benchmark-gap feature engineering"
    if decision == "keep v2 for next feature-engineering iteration":
        return "keep v2 in the feature-engineering queue; no production promotion"
    if decision == "mixed; QQQ gap improved but drawdown worsened":
        return "inspect drawdown-heavy folds before keeping v2"
    if decision == "mixed; inspect v2 window drivers":
        return "inspect fold-level winners and losers versus v1"
    if decision == "reject/deprioritize v2":
        return "deprioritize v2 unless fold diagnostics reveal a targeted fix"
    if decision == "inconclusive; inspect window drivers":
        return "inspect fold-level winners and losers"
    if decision == "reference only":
        return "reference only"
    return "feature engineering"


def _rationale(row: pd.Series) -> str:
    return (
        f"gap_change={row['benchmark_gap_change']}; status={row['benchmark_status']}; "
        f"gap_change_vs_v1={row['benchmark_gap_change_vs_benchmark_gap_v1']}; "
        f"avg_return={_fmt(row['average_total_return'])}; "
        f"vs_spy={_fmt(row['average_strategy_vs_spy'])}; "
        f"vs_qqq={_fmt(row['average_strategy_vs_qqq'])}; "
        f"delta_v1_spy={_fmt(row['delta_strategy_vs_spy_vs_benchmark_gap_v1'])}; "
        f"delta_v1_qqq={_fmt(row['delta_strategy_vs_qqq_vs_benchmark_gap_v1'])}; "
        f"spy_under={_fmt(row['spy_underperformance_rate'])}; "
        f"qqq_under={_fmt(row['qqq_underperformance_rate'])}"
    )


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.reindex(columns=SUMMARY_COLUMNS).sort_values(
        by=[
            "candidate_role",
            "model_type",
            "top_k",
            "min_holding_days",
            "switch_margin",
            "transaction_cost_bps",
            "train_window_years",
            "test_window_months",
            "feature_set_version",
        ],
        ascending=[False, True, True, True, True, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _sort_decision_matrix(decision_matrix: pd.DataFrame) -> pd.DataFrame:
    output = decision_matrix.reindex(columns=DECISION_MATRIX_COLUMNS)
    feature_order = {
        DEFAULT_FEATURE_SET_VERSION: 0,
        BENCHMARK_GAP_FEATURE_SET_VERSION: 1,
        BENCHMARK_GAP_V2_FEATURE_SET_VERSION: 2,
    }
    output["_feature_order"] = output["feature_set_version"].map(feature_order).fillna(9)
    return output.sort_values(
        by=[
            "candidate_role",
            "model_type",
            "top_k",
            "min_holding_days",
            "switch_margin",
            "transaction_cost_bps",
            "_feature_order",
            "average_total_return",
        ],
        ascending=[False, True, True, True, True, True, True, False],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_feature_order"]).reset_index(drop=True)


def _parse_fold_metadata(fold_name: str) -> dict[str, object] | None:
    match = _FOLD_NAME_RE.search(fold_name)
    if match is None:
        return None
    parts = match.groupdict()
    return {
        "train_window_years": int(parts["train_window_years"]),
        "validation_window_months": int(parts["validation_window_months"]),
        "test_window_months": int(parts["test_window_months"]),
        "sequence_number": int(parts["sequence_number"]),
        "test_start": pd.Timestamp(parts["test_start"]).date().isoformat(),
        "test_end": pd.Timestamp(parts["test_end"]).date().isoformat(),
    }


def _candidate_key_from_row(row: pd.Series) -> str:
    return (
        f"{row['candidate_role']}:{row['model_type']}:top{int(row['top_k'])}:"
        f"hold{int(row['min_holding_days'])}:margin{float(row['switch_margin']):g}:"
        f"cost{float(row['transaction_cost_bps']):g}"
    )


def _candidate_id_from_row(row: pd.Series) -> str:
    return f"{row['candidate_key']}:features:{row['feature_set_version']}"


def _candidate_label_from_row(row: pd.Series) -> str:
    return (
        f"{row['candidate_role']} / {row['model_type']} / top{int(row['top_k'])} / "
        f"hold {int(row['min_holding_days'])} / margin {float(row['switch_margin']):g} / "
        f"cost {float(row['transaction_cost_bps']):g}bps"
    )


def _rate(values: pd.Series) -> float:
    non_null = values.dropna()
    if non_null.empty:
        return float("nan")
    return float(non_null.astype(bool).mean())


def _feature_row(frame: pd.DataFrame, feature_set_version: str) -> pd.Series | None:
    rows = frame.loc[frame["feature_set_version"].astype(str).eq(feature_set_version)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fmt(value: object) -> str:
    number = _as_float(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.4f}"


create_tiingo_benchmark_gap_feature_report_from_args = (
    run_tiingo_benchmark_gap_feature_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_benchmark_gap_feature_report_from_args())
