"""Create replay artifact tables for the Tiingo selected candidate."""

from __future__ import annotations

import argparse
import glob
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.cli.create_tiingo_modern_baseline_sweep_report import (
    DATASET_SOURCE_LABEL,
    REQUIRED_RESULTS_COLUMNS,
    _as_float,
    _enrich_test_rows,
    _format_for_markdown,
    _markdown_table,
    _positive_rate,
    _validate_columns,
)
from us_market_ml.cli.create_tiingo_selected_candidate_evaluation_bundle import (
    COMPARISON_COLUMNS as SELECTED_COMPARISON_COLUMNS,
)
from us_market_ml.cli.create_tiingo_selected_candidate_evaluation_bundle import (
    _adjusted_close_column,
    _parse_benchmark_symbols,
    _path_mentions_benchmark_symbol,
)
from us_market_ml.data.local_reader import read_csv_files, read_parquet_files

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_focused_linear_ridge_confirmation/"
    "tiingo_focused_linear_ridge_confirmation_results.csv"
)
DEFAULT_SELECTED_SUMMARY_CSV = Path(
    "artifacts/reports/tiingo_selected_candidate_evaluation/selected_candidate_summary.csv"
)
DEFAULT_SELECTED_COMPARISON_CSV = Path(
    "artifacts/reports/tiingo_selected_candidate_evaluation/selected_candidate_comparison.csv"
)
DEFAULT_MARKET_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_selected_candidate_replay_artifacts")
DEFAULT_BENCHMARK_SYMBOLS = ("SPY", "QQQ")

FEATURE_SET_VERSION = "default_features_v1"
TARGET_COL = "excess_return_vs_spy_20d"
DECISION_ANCHOR_BPS = 10.0

SUMMARY_PATH_NAME = "selected_candidate_replay_summary.csv"
FOLD_RETURNS_PATH_NAME = "selected_candidate_fold_returns.csv"
EQUITY_CURVES_PATH_NAME = "selected_candidate_equity_curves.csv"
DRAWDOWNS_PATH_NAME = "selected_candidate_drawdowns.csv"
BENCHMARK_COMPARISON_PATH_NAME = "selected_candidate_benchmark_comparison.csv"
UNDERPERFORMANCE_WINDOWS_PATH_NAME = "selected_candidate_underperformance_windows.csv"
TURNOVER_DIAGNOSTICS_PATH_NAME = "selected_candidate_turnover_diagnostics.csv"
HOLDINGS_SUMMARY_PATH_NAME = "selected_candidate_holdings_summary.csv"
REPORT_PATH_NAME = "selected_candidate_replay_report.md"
OUTPUT_PATH_NAMES = (
    SUMMARY_PATH_NAME,
    FOLD_RETURNS_PATH_NAME,
    EQUITY_CURVES_PATH_NAME,
    DRAWDOWNS_PATH_NAME,
    BENCHMARK_COMPARISON_PATH_NAME,
    UNDERPERFORMANCE_WINDOWS_PATH_NAME,
    TURNOVER_DIAGNOSTICS_PATH_NAME,
    HOLDINGS_SUMMARY_PATH_NAME,
    REPORT_PATH_NAME,
)

SELECTED_SUMMARY_REQUIRED_COLUMNS = ["candidate_role", "candidate_label"]
SELECTED_COMPARISON_REQUIRED_COLUMNS = [
    "comparison_name",
    "primary_candidate_label",
    "comparison_candidate_label",
    "comparison_average_test_return_10bps",
    "delta_average_test_return_10bps",
]

FOLD_RETURN_COLUMNS = [
    "series_role",
    "series_name",
    "benchmark_symbol",
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "first_observation_date",
    "last_observation_date",
    "observation_count",
    "total_return",
    "mean_daily_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "run_key",
    "source",
    "reconstruction_level",
]

EQUITY_CURVE_COLUMNS = [
    "series_role",
    "series_name",
    "benchmark_symbol",
    "fold",
    "date",
    "daily_return",
    "equity",
    "drawdown",
    "fold_total_return",
    "fold_max_drawdown",
    "source",
    "reconstruction_level",
]

DRAWDOWN_COLUMNS = [
    "series_role",
    "series_name",
    "benchmark_symbol",
    "fold",
    "date",
    "equity",
    "drawdown",
    "fold_max_drawdown",
    "source",
    "reconstruction_level",
]

BENCHMARK_COMPARISON_COLUMNS = [
    *SELECTED_COMPARISON_COLUMNS,
    "reference_type",
    "benchmark_symbol",
    "primary_beats_comparison_on_average_return",
    "comparison_beats_primary_on_raw_return",
    "raw_return_caveat",
]

UNDERPERFORMANCE_COLUMNS = [
    "benchmark_symbol",
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "primary_total_return",
    "benchmark_total_return",
    "excess_return_vs_benchmark",
    "underperformed_benchmark",
    "underperformance_severity",
]

TURNOVER_COLUMNS = [
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "run_key",
    "source",
    "reconstruction_level",
]

HOLDINGS_SUMMARY_COLUMNS = [
    "symbol",
    "selected_days",
    "avg_target_weight",
    "avg_prediction",
    "avg_signal_rank",
    "first_selected_date",
    "last_selected_date",
    "source",
    "reconstruction_level",
    "caveat",
]

REPLAY_SUMMARY_COLUMNS = [
    "candidate_role",
    "candidate_label",
    "dataset_source",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "feature_set_version",
    "target_col",
    "fold_count",
    "average_total_return",
    "median_total_return",
    "positive_window_rate",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_total_transaction_cost",
    "average_mean_daily_turnover",
    "benchmark_symbols",
    "spy_average_total_return",
    "qqq_average_total_return",
    "best_raw_return_reference",
    "benchmark_gate",
    "decision",
    "next_action",
    "artifact_caveats",
]

_FOLD_WINDOW_RE = re.compile(
    r"tw(?P<train_window_years>\d+)y"
    r"_v(?P<validation_window_months>\d+)m"
    r"_t(?P<test_window_months>\d+)m"
    r"_q(?P<sequence_number>\d+)"
    r"_(?P<test_start>\d{8})"
    r"_(?P<test_end>\d{8})"
)


@dataclass(frozen=True)
class ReplayCandidateSpec:
    """One modeled candidate/reference available from the focused result CSV."""

    role: str
    label: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float


@dataclass(frozen=True)
class FoldWindow:
    """Parsed walk-forward fold metadata."""

    fold: str
    train_window_years: int
    validation_window_months: int
    test_window_months: int
    sequence_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class TiingoSelectedCandidateReplayArtifacts:
    """All replay artifact tables and reconstruction warnings."""

    replay_summary: pd.DataFrame
    fold_returns: pd.DataFrame
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    underperformance_windows: pd.DataFrame
    turnover_diagnostics: pd.DataFrame
    holdings_summary: pd.DataFrame
    warnings: tuple[str, ...]
    unavailable_artifacts: tuple[str, ...]
    result_row_count: int
    test_row_count: int
    primary_fold_count: int
    benchmark_symbols: tuple[str, ...]
    dataset_source: str = DATASET_SOURCE_LABEL


PRIMARY_SPEC = ReplayCandidateSpec(
    role="primary_candidate",
    label="ridge / top5 / hold 20 / margin 0.005",
    model_type="ridge",
    top_k=5,
    min_holding_days=20,
    switch_margin=0.005,
)
RETURN_REFERENCE_SPEC = ReplayCandidateSpec(
    role="return_reference",
    label="ridge / top1 / hold 20 / margin 0.005",
    model_type="ridge",
    top_k=1,
    min_holding_days=20,
    switch_margin=0.005,
)
DRAWDOWN_GUARDRAIL_SPEC = ReplayCandidateSpec(
    role="drawdown_guardrail",
    label="ridge / top5 / hold 5 / margin 0.002",
    model_type="ridge",
    top_k=5,
    min_holding_days=5,
    switch_margin=0.002,
)
MODELED_REFERENCE_SPECS = (
    PRIMARY_SPEC,
    RETURN_REFERENCE_SPEC,
    DRAWDOWN_GUARDRAIL_SPEC,
)


def run_tiingo_selected_candidate_replay_artifacts_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write Tiingo selected-candidate replay artifacts."""

    args = _parse_args(argv)
    report_data = build_tiingo_selected_candidate_replay_artifacts(
        results_csv=args.results_csv,
        selected_summary_csv=args.selected_summary_csv,
        selected_comparison_csv=args.selected_comparison_csv,
        market_dataset_glob=args.market_dataset_glob,
        benchmark_symbols=args.benchmark_symbols,
    )
    outputs = write_tiingo_selected_candidate_replay_artifacts(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo selected-candidate replay artifacts")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"decision anchor: {DECISION_ANCHOR_BPS:g}bps")
    print(f"results CSV: {args.results_csv}")
    print(f"selected summary CSV: {args.selected_summary_csv}")
    print(f"selected comparison CSV: {args.selected_comparison_csv}")
    print(f"market dataset glob: {args.market_dataset_glob}")
    print(f"benchmark symbols: {', '.join(report_data.benchmark_symbols)}")
    print(f"output dir: {args.output_dir}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    if report_data.unavailable_artifacts:
        print()
        print("Unavailable replay details:")
        for artifact in report_data.unavailable_artifacts:
            print(f"- {artifact}")
    if report_data.warnings:
        print()
        print("Warnings:")
        for warning in report_data.warnings:
            print(f"- {warning}")
    print()
    print(report_data.replay_summary.to_string(index=False))
    return 0


def build_tiingo_selected_candidate_replay_artifacts(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    selected_summary_csv: Path = DEFAULT_SELECTED_SUMMARY_CSV,
    selected_comparison_csv: Path = DEFAULT_SELECTED_COMPARISON_CSV,
    market_dataset_glob: str | Path = DEFAULT_MARKET_DATASET_GLOB,
    benchmark_symbols: str | Sequence[str] = DEFAULT_BENCHMARK_SYMBOLS,
) -> TiingoSelectedCandidateReplayArtifacts:
    """Read prior outputs and build selected-candidate replay artifact tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    selected_summary = _read_selected_summary(selected_summary_csv)
    selected_comparison = _read_selected_comparison(selected_comparison_csv)
    resolved_benchmark_symbols = _parse_benchmark_symbols(benchmark_symbols)

    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _add_fold_dates(_enrich_test_rows(test_rows))
    modeled_rows = _filter_modeled_reference_rows(enriched)
    primary_rows = _match_spec(modeled_rows, PRIMARY_SPEC)
    if primary_rows.empty:
        raise ValueError(
            "Primary Tiingo selected candidate rows were not found in the results CSV: "
            f"{PRIMARY_SPEC.label} at {DECISION_ANCHOR_BPS:g}bps."
        )

    warnings: list[str] = []
    unavailable = list(_unavailable_artifacts(results))
    if _historical_champion_missing(enriched):
        unavailable.append(
            "Historical champion fold-level replay rows cannot be reconstructed "
            "from the provided inputs; only selected summary/comparison metrics are available."
        )
    market_data, market_warnings = _read_benchmark_market_data(
        market_dataset_glob,
        resolved_benchmark_symbols,
    )
    warnings.extend(market_warnings)

    benchmark_curves, benchmark_fold_returns, benchmark_warnings = _build_benchmark_artifacts(
        market_data=market_data,
        primary_rows=primary_rows,
        benchmark_symbols=resolved_benchmark_symbols,
    )
    warnings.extend(benchmark_warnings)

    fold_returns = _build_fold_returns(
        modeled_rows=modeled_rows,
        benchmark_fold_returns=benchmark_fold_returns,
    )
    equity_curves = _build_equity_curves(
        primary_rows=primary_rows,
        benchmark_curves=benchmark_curves,
    )
    drawdowns = _build_drawdowns(equity_curves)
    benchmark_comparison = _build_benchmark_comparison(selected_comparison)
    underperformance_windows = _build_underperformance_windows(
        primary_rows=primary_rows,
        benchmark_fold_returns=benchmark_fold_returns,
    )
    turnover_diagnostics = _build_turnover_diagnostics(primary_rows)
    holdings_summary = _build_holdings_summary(results)
    replay_summary = _build_replay_summary(
        primary_rows=primary_rows,
        selected_summary=selected_summary,
        benchmark_fold_returns=benchmark_fold_returns,
        benchmark_comparison=benchmark_comparison,
        benchmark_symbols=resolved_benchmark_symbols,
        unavailable_artifacts=unavailable,
    )

    return TiingoSelectedCandidateReplayArtifacts(
        replay_summary=replay_summary,
        fold_returns=fold_returns,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
        benchmark_comparison=benchmark_comparison,
        underperformance_windows=underperformance_windows,
        turnover_diagnostics=turnover_diagnostics,
        holdings_summary=holdings_summary,
        warnings=tuple(warnings),
        unavailable_artifacts=tuple(unavailable),
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        primary_fold_count=int(len(primary_rows)),
        benchmark_symbols=resolved_benchmark_symbols,
    )


def write_tiingo_selected_candidate_replay_artifacts(
    *,
    report_data: TiingoSelectedCandidateReplayArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all selected-candidate replay artifact files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        FOLD_RETURNS_PATH_NAME: output_dir / FOLD_RETURNS_PATH_NAME,
        EQUITY_CURVES_PATH_NAME: output_dir / EQUITY_CURVES_PATH_NAME,
        DRAWDOWNS_PATH_NAME: output_dir / DRAWDOWNS_PATH_NAME,
        BENCHMARK_COMPARISON_PATH_NAME: output_dir / BENCHMARK_COMPARISON_PATH_NAME,
        UNDERPERFORMANCE_WINDOWS_PATH_NAME: output_dir / UNDERPERFORMANCE_WINDOWS_PATH_NAME,
        TURNOVER_DIAGNOSTICS_PATH_NAME: output_dir / TURNOVER_DIAGNOSTICS_PATH_NAME,
        HOLDINGS_SUMMARY_PATH_NAME: output_dir / HOLDINGS_SUMMARY_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.replay_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.fold_returns.to_csv(outputs[FOLD_RETURNS_PATH_NAME], index=False)
    report_data.equity_curves.to_csv(outputs[EQUITY_CURVES_PATH_NAME], index=False)
    report_data.drawdowns.to_csv(outputs[DRAWDOWNS_PATH_NAME], index=False)
    report_data.benchmark_comparison.to_csv(
        outputs[BENCHMARK_COMPARISON_PATH_NAME],
        index=False,
    )
    report_data.underperformance_windows.to_csv(
        outputs[UNDERPERFORMANCE_WINDOWS_PATH_NAME],
        index=False,
    )
    report_data.turnover_diagnostics.to_csv(
        outputs[TURNOVER_DIAGNOSTICS_PATH_NAME],
        index=False,
    )
    report_data.holdings_summary.to_csv(outputs[HOLDINGS_SUMMARY_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_selected_candidate_replay_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_selected_candidate_replay_markdown(
    report_data: TiingoSelectedCandidateReplayArtifacts,
) -> str:
    """Render a Markdown report for the replay artifact bundle."""

    summary = report_data.replay_summary.iloc[0]
    comparison = report_data.benchmark_comparison
    fold_returns = report_data.fold_returns
    primary_folds = fold_returns.loc[fold_returns["series_role"].eq(PRIMARY_SPEC.role)].copy()
    benchmark_rows = fold_returns.loc[fold_returns["series_role"].eq("benchmark_reference")]
    underperformance = report_data.underperformance_windows
    turnover = report_data.turnover_diagnostics
    holdings = report_data.holdings_summary
    qqq_caveat = _raw_return_sentence(comparison, "QQQ")
    spy_caveat = _raw_return_sentence(comparison, "SPY")

    lines = [
        "# Tiingo Selected Candidate Replay Artifacts",
        "",
        "## Executive summary",
        "",
        f"Primary candidate: `{summary['candidate_label']}`.",
        "",
        "This bundle replays the available completed-result artifacts for the "
        "Tiingo-only selected candidate. It does not run a new sweep, add tree "
        "models, add MLP, mutate Market Data, change S3 readers, update production "
        "signal export, or mark anything production-ready.",
        "",
        f"Primary fold count at `{DECISION_ANCHOR_BPS:g}bps`: "
        f"`{int(summary['fold_count'])}`. Average fold return: "
        f"`{_format_number(summary['average_total_return'])}`. Worst fold drawdown: "
        f"`{_format_number(summary['worst_max_drawdown'])}`.",
        "",
        f"Replay decision: `{summary['decision']}`.",
        "",
        "## Primary candidate vs SPY/QQQ",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_subset(
                    comparison,
                    ("primary_vs_spy_buy_hold", "primary_vs_qqq_buy_hold"),
                ),
                exclude={
                    "comparison_name",
                    "primary_candidate_label",
                    "comparison_candidate_label",
                    "interpretation",
                    "reference_type",
                    "benchmark_symbol",
                    "raw_return_caveat",
                },
            )
        ),
        "",
        spy_caveat,
        "",
        qqq_caveat,
        "",
        "## Primary candidate vs historical champion",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_subset(comparison, ("primary_vs_historical_champion",)),
                exclude={
                    "comparison_name",
                    "primary_candidate_label",
                    "comparison_candidate_label",
                    "interpretation",
                    "reference_type",
                    "benchmark_symbol",
                    "raw_return_caveat",
                },
            )
        ),
        "",
        "The historical ML champion remains a continuity reference only. Promotion "
        "requires benchmark-aware evidence against SPY and QQQ, not only an internal "
        "ML-to-ML win.",
        "",
        "## Primary candidate vs return reference",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_subset(comparison, ("primary_vs_return_reference",)),
                exclude={
                    "comparison_name",
                    "primary_candidate_label",
                    "comparison_candidate_label",
                    "interpretation",
                    "reference_type",
                    "benchmark_symbol",
                    "raw_return_caveat",
                },
            )
        ),
        "",
        "The return reference is retained to keep raw-return pressure visible while "
        "the selected top5 candidate is evaluated on return, drawdown, and turnover.",
        "",
        "## Primary candidate vs drawdown guardrail",
        "",
        _markdown_table(
            _format_for_markdown(
                _comparison_subset(comparison, ("primary_vs_drawdown_guardrail",)),
                exclude={
                    "comparison_name",
                    "primary_candidate_label",
                    "comparison_candidate_label",
                    "interpretation",
                    "reference_type",
                    "benchmark_symbol",
                    "raw_return_caveat",
                },
            )
        ),
        "",
        "The drawdown guardrail remains useful as a shorter-hold risk check, but "
        "turnover and cost pressure still matter.",
        "",
        "## Fold-by-fold performance",
        "",
        _markdown_table(
            _format_for_markdown(
                primary_folds[
                    [
                        "series_name",
                        "fold",
                        "test_start",
                        "test_end",
                        "total_return",
                        "sharpe_like",
                        "max_drawdown",
                        "total_transaction_cost",
                        "mean_daily_turnover",
                    ]
                ].head(80),
                exclude={"series_name", "fold", "test_start", "test_end"},
            )
        ),
        "",
        "## Equity/drawdown diagnostics",
        "",
        "Benchmark daily equity and drawdown series are reconstructed from market "
        "prices. The selected candidate source CSV contains fold-level metrics, not "
        "daily strategy equity rows, so candidate equity is represented only by "
        "fold-terminal metric rows.",
        "",
        _markdown_table(
            _format_for_markdown(
                benchmark_rows[
                    [
                        "series_name",
                        "fold",
                        "test_start",
                        "test_end",
                        "total_return",
                        "max_drawdown",
                        "observation_count",
                    ]
                ].head(40),
                exclude={"series_name", "fold", "test_start", "test_end"},
            )
        ),
        "",
        "## Turnover diagnostics",
        "",
        "Turnover diagnostics are available at fold-summary granularity only.",
        "",
        _markdown_table(
            _format_for_markdown(
                turnover[
                    [
                        "fold",
                        "test_start",
                        "test_end",
                        "total_transaction_cost",
                        "mean_daily_turnover",
                        "max_daily_turnover",
                        "reconstruction_level",
                    ]
                ].head(60),
                exclude={"fold", "test_start", "test_end", "reconstruction_level"},
            )
        ),
        "",
        "## Holdings / selected ETF summary if available",
        "",
        _holdings_markdown_sentence(holdings),
        "",
        _markdown_table(
            _format_for_markdown(
                holdings.head(20),
                exclude={"symbol", "source", "reconstruction_level", "caveat"},
            )
        ),
        "",
        "## Underperformance windows versus SPY and QQQ",
        "",
        _markdown_table(
            _format_for_markdown(
                underperformance.sort_values(
                    ["underperformed_benchmark", "excess_return_vs_benchmark"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(80),
                exclude={
                    "benchmark_symbol",
                    "fold",
                    "test_start",
                    "test_end",
                    "underperformed_benchmark",
                    "underperformance_severity",
                },
            )
        ),
        "",
        "## Decision",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "Decision rule applied: do not promote only because the selected candidate "
        "beats the historical ML champion. Promotion needs benchmark-aware evidence "
        "against both SPY and QQQ.",
        "",
        "## Next action",
        "",
        f"Next action: `{summary['next_action']}`.",
        "",
        "Allowed next-action menu considered: `tree_screen`, `MLP`, "
        "`feature engineering`, `richer replay run`, `model-card draft`.",
        "",
        "## Caveats",
        "",
        "- Existing results are fold-summary artifacts, not a full daily replay.",
        "- Daily selected ETF holdings cannot be reconstructed from the provided CSVs.",
        "- Daily strategy equity and exact strategy drawdown paths cannot be reconstructed.",
        "- QQQ buy-and-hold can be stronger on raw return even when the selected candidate "
        "beats older ML references.",
        "- This is research evidence only and is not production-ready.",
        "",
    ]
    if report_data.unavailable_artifacts:
        lines.extend(["## Unavailable artifacts", ""])
        lines.extend(f"- {artifact}" for artifact in report_data.unavailable_artifacts)
        lines.append("")
    if report_data.warnings:
        lines.extend(["## Input warnings", ""])
        lines.extend(f"- {warning}" for warning in report_data.warnings)
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create replay artifacts for the Tiingo selected candidate."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--selected-summary-csv",
        type=Path,
        default=DEFAULT_SELECTED_SUMMARY_CSV,
    )
    parser.add_argument(
        "--selected-comparison-csv",
        type=Path,
        default=DEFAULT_SELECTED_COMPARISON_CSV,
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=DEFAULT_MARKET_DATASET_GLOB,
    )
    parser.add_argument(
        "--benchmark-symbols",
        nargs="+",
        default=[",".join(DEFAULT_BENCHMARK_SYMBOLS)],
        help="Comma-separated or space-separated benchmark symbols, default SPY,QQQ.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _read_selected_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    _validate_columns(summary, SELECTED_SUMMARY_REQUIRED_COLUMNS, path=path)
    return summary


def _read_selected_comparison(path: Path) -> pd.DataFrame:
    comparison = pd.read_csv(path)
    _validate_columns(comparison, SELECTED_COMPARISON_REQUIRED_COLUMNS, path=path)
    return comparison


def _filter_modeled_reference_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    output = test_rows.loc[
        test_rows["model_family"].astype(str).eq("baseline")
        & _float_eq(test_rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)
        & test_rows["allocation_mode"].astype(str).eq("equal_weight")
        & _series_in_values(test_rows["train_window_years"], (3, 5))
        & _series_in_values(test_rows["validation_window_months"], (12,))
        & _series_in_values(test_rows["test_window_months"], (3, 6, 12))
    ].copy()
    if "feature_set_version" in output.columns:
        output = output.loc[output["feature_set_version"].astype(str).eq(FEATURE_SET_VERSION)]
    if "target_col" in output.columns:
        output = output.loc[output["target_col"].astype(str).eq(TARGET_COL)]
    return output.reset_index(drop=True)


def _match_spec(rows: pd.DataFrame, spec: ReplayCandidateSpec) -> pd.DataFrame:
    return rows.loc[
        rows["model_type"].astype(str).eq(spec.model_type)
        & _int_eq(rows, "top_k", spec.top_k)
        & _int_eq(rows, "min_holding_days", spec.min_holding_days)
        & _float_eq(rows, "switch_margin", spec.switch_margin)
    ].copy()


def _add_fold_dates(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    for column in ["test_start", "test_end"]:
        if column not in output.columns:
            output[column] = pd.NaT
    parsed = output["fold"].astype(str).map(_parse_fold_window)
    for index, window in parsed.items():
        if window is None:
            continue
        for column in [
            "train_window_years",
            "validation_window_months",
            "test_window_months",
            "sequence_number",
        ]:
            if column in output.columns and pd.isna(output.at[index, column]):
                output.at[index, column] = getattr(window, column)
        output.at[index, "test_start"] = window.test_start
        output.at[index, "test_end"] = window.test_end
    return output


def _parse_fold_window(fold_name: str) -> FoldWindow | None:
    match = _FOLD_WINDOW_RE.search(fold_name)
    if match is None:
        return None
    parts = match.groupdict()
    test_start = pd.to_datetime(parts["test_start"], format="%Y%m%d")
    test_end = pd.to_datetime(parts["test_end"], format="%Y%m%d")
    if test_start > test_end:
        raise ValueError(f"Fold {fold_name!r} has test_start after test_end.")
    return FoldWindow(
        fold=fold_name,
        train_window_years=int(parts["train_window_years"]),
        validation_window_months=int(parts["validation_window_months"]),
        test_window_months=int(parts["test_window_months"]),
        sequence_number=int(parts["sequence_number"]),
        test_start=test_start,
        test_end=test_end,
    )


def _read_benchmark_market_data(
    market_dataset_glob: str | Path,
    benchmark_symbols: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    warnings: list[str] = []
    pattern = str(market_dataset_glob)
    matched_paths = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    file_paths = [path for path in matched_paths if path.is_file()]
    if not file_paths:
        raise FileNotFoundError(f"Market dataset glob matched no files: {pattern}")

    symbol_paths = [
        path for path in file_paths if _path_mentions_benchmark_symbol(path, benchmark_symbols)
    ]
    read_paths = symbol_paths if symbol_paths else file_paths
    if not symbol_paths:
        warnings.append(
            "Market dataset glob was not symbol-partitioned for requested benchmarks; "
            "all matched files were scanned."
        )

    parquet_paths = [path for path in read_paths if path.suffix.lower() == ".parquet"]
    if parquet_paths:
        return read_parquet_files(parquet_paths), tuple(warnings)

    csv_paths = [path for path in read_paths if path.suffix.lower() == ".csv"]
    if csv_paths:
        return read_csv_files(csv_paths), tuple(warnings)

    raise ValueError("Benchmark market data glob matched no supported .parquet or .csv files.")


def _build_benchmark_artifacts(
    *,
    market_data: pd.DataFrame,
    primary_rows: pd.DataFrame,
    benchmark_symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    price_column = _adjusted_close_column(market_data)
    warnings: list[str] = []
    curve_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    windows = [_parse_fold_window(str(fold)) for fold in primary_rows["fold"]]
    windows = [window for window in windows if window is not None]

    for symbol in benchmark_symbols:
        symbol_rows = market_data.loc[
            market_data["symbol"].astype(str).str.upper().eq(symbol.upper())
        ].copy()
        if symbol_rows.empty:
            warnings.append(f"Benchmark symbol {symbol.upper()} was not present in market data.")
            continue
        for window in windows:
            window_curve = _benchmark_curve_for_window(
                symbol_rows=symbol_rows,
                symbol=symbol,
                price_column=price_column,
                window=window,
            )
            if window_curve.empty:
                warnings.append(
                    f"Benchmark {symbol.upper()} skipped fold {window.fold}: fewer than two "
                    "adjusted-close observations were available inside the test window."
                )
                continue
            curve_rows.extend(window_curve.to_dict("records"))
            fold_rows.append(_benchmark_fold_return_row(window_curve, symbol, window))

    curves = pd.DataFrame(curve_rows, columns=EQUITY_CURVE_COLUMNS)
    fold_returns = pd.DataFrame(fold_rows, columns=FOLD_RETURN_COLUMNS)
    return curves, fold_returns, tuple(warnings)


def _benchmark_curve_for_window(
    *,
    symbol_rows: pd.DataFrame,
    symbol: str,
    price_column: str,
    window: FoldWindow,
) -> pd.DataFrame:
    prices = symbol_rows[["date", price_column]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices[price_column] = pd.to_numeric(prices[price_column], errors="coerce")
    window_prices = (
        prices.loc[(prices["date"] >= window.test_start) & (prices["date"] <= window.test_end)]
        .dropna(subset=[price_column])
        .loc[lambda frame: frame[price_column] > 0.0]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(window_prices) < 2:
        return pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)

    daily_return = window_prices[price_column].pct_change().fillna(0.0)
    equity = (1.0 + daily_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    fold_total_return = float(equity.iloc[-1] - 1.0)
    fold_max_drawdown = float(drawdown.min())
    output = pd.DataFrame(
        {
            "series_role": "benchmark_reference",
            "series_name": f"{symbol.upper()} buy-and-hold",
            "benchmark_symbol": symbol.upper(),
            "fold": window.fold,
            "date": window_prices["date"],
            "daily_return": daily_return,
            "equity": equity,
            "drawdown": drawdown,
            "fold_total_return": fold_total_return,
            "fold_max_drawdown": fold_max_drawdown,
            "source": "market_data_adjusted_close",
            "reconstruction_level": "daily_benchmark_from_market_data",
        }
    )
    return output.loc[:, EQUITY_CURVE_COLUMNS]


def _benchmark_fold_return_row(
    curve: pd.DataFrame,
    symbol: str,
    window: FoldWindow,
) -> dict[str, object]:
    daily_returns = pd.to_numeric(curve["daily_return"], errors="coerce")
    equity = pd.to_numeric(curve["equity"], errors="coerce")
    daily_volatility = float(daily_returns.std())
    sharpe_like = (
        float(daily_returns.mean()) / daily_volatility
        if daily_volatility > 0.0 and not np.isnan(daily_volatility)
        else np.nan
    )
    return {
        "series_role": "benchmark_reference",
        "series_name": f"{symbol.upper()} buy-and-hold",
        "benchmark_symbol": symbol.upper(),
        "fold": window.fold,
        "train_window_years": window.train_window_years,
        "validation_window_months": window.validation_window_months,
        "test_window_months": window.test_window_months,
        "sequence_number": window.sequence_number,
        "test_start": window.test_start.date().isoformat(),
        "test_end": window.test_end.date().isoformat(),
        "first_observation_date": pd.Timestamp(curve["date"].iloc[0]).date().isoformat(),
        "last_observation_date": pd.Timestamp(curve["date"].iloc[-1]).date().isoformat(),
        "observation_count": int(len(curve)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mean_daily_return": float(daily_returns.mean()),
        "daily_volatility": daily_volatility,
        "sharpe_like": sharpe_like,
        "max_drawdown": calculate_max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
        "total_transaction_cost": 0.0,
        "mean_daily_turnover": 0.0,
        "max_daily_turnover": 0.0,
        "run_key": f"{symbol.upper()}:{window.fold}:buy_and_hold",
        "source": "market_data_adjusted_close",
        "reconstruction_level": "daily_benchmark_from_market_data",
    }


def _build_fold_returns(
    *,
    modeled_rows: pd.DataFrame,
    benchmark_fold_returns: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in MODELED_REFERENCE_SPECS:
        spec_rows = _match_spec(modeled_rows, spec)
        for _, row in spec_rows.sort_values("fold", kind="mergesort").iterrows():
            rows.append(_modeled_fold_return_row(row, spec))
    modeled_fold_returns = pd.DataFrame(rows, columns=FOLD_RETURN_COLUMNS)
    output = pd.concat(
        [modeled_fold_returns, benchmark_fold_returns],
        ignore_index=True,
    )
    if output.empty:
        return pd.DataFrame(columns=FOLD_RETURN_COLUMNS)
    return (
        output.loc[:, FOLD_RETURN_COLUMNS]
        .sort_values(["fold", "series_role", "series_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def _modeled_fold_return_row(row: pd.Series, spec: ReplayCandidateSpec) -> dict[str, object]:
    test_start = pd.to_datetime(row.get("test_start"))
    test_end = pd.to_datetime(row.get("test_end"))
    return {
        "series_role": spec.role,
        "series_name": spec.label,
        "benchmark_symbol": "",
        "fold": row["fold"],
        "train_window_years": _as_int(row.get("train_window_years")),
        "validation_window_months": _as_int(row.get("validation_window_months")),
        "test_window_months": _as_int(row.get("test_window_months")),
        "sequence_number": _as_int(row.get("sequence_number")),
        "test_start": _date_string(test_start),
        "test_end": _date_string(test_end),
        "first_observation_date": "",
        "last_observation_date": "",
        "observation_count": np.nan,
        "total_return": _as_float(row.get("total_return")),
        "mean_daily_return": _as_float(row.get("mean_daily_return")),
        "daily_volatility": _as_float(row.get("daily_volatility")),
        "sharpe_like": _as_float(row.get("sharpe_like")),
        "max_drawdown": _as_float(row.get("max_drawdown")),
        "final_equity": _as_float(row.get("final_equity")),
        "total_transaction_cost": _as_float(row.get("total_transaction_cost")),
        "mean_daily_turnover": _as_float(row.get("mean_daily_turnover")),
        "max_daily_turnover": _as_float(row.get("max_daily_turnover")),
        "run_key": row.get("run_key", ""),
        "source": "results_csv_fold_metric",
        "reconstruction_level": "fold_metric_only",
    }


def _build_equity_curves(
    *,
    primary_rows: pd.DataFrame,
    benchmark_curves: pd.DataFrame,
) -> pd.DataFrame:
    primary_curve = _build_primary_fold_terminal_equity_rows(primary_rows)
    output = pd.concat([primary_curve, benchmark_curves], ignore_index=True)
    if output.empty:
        return pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    return (
        output.loc[:, EQUITY_CURVE_COLUMNS]
        .sort_values(["fold", "series_role", "series_name", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_primary_fold_terminal_equity_rows(primary_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in primary_rows.sort_values("fold", kind="mergesort").iterrows():
        rows.append(
            {
                "series_role": PRIMARY_SPEC.role,
                "series_name": PRIMARY_SPEC.label,
                "benchmark_symbol": "",
                "fold": row["fold"],
                "date": row.get("test_end"),
                "daily_return": np.nan,
                "equity": _as_float(row.get("final_equity")),
                "drawdown": np.nan,
                "fold_total_return": _as_float(row.get("total_return")),
                "fold_max_drawdown": _as_float(row.get("max_drawdown")),
                "source": "results_csv_fold_metric",
                "reconstruction_level": "fold_terminal_only_not_daily_equity",
            }
        )
    return pd.DataFrame(rows, columns=EQUITY_CURVE_COLUMNS)


def _build_drawdowns(equity_curves: pd.DataFrame) -> pd.DataFrame:
    if equity_curves.empty:
        return pd.DataFrame(columns=DRAWDOWN_COLUMNS)
    output = equity_curves[
        [
            "series_role",
            "series_name",
            "benchmark_symbol",
            "fold",
            "date",
            "equity",
            "drawdown",
            "fold_max_drawdown",
            "source",
            "reconstruction_level",
        ]
    ].copy()
    return output.loc[:, DRAWDOWN_COLUMNS].reset_index(drop=True)


def _build_benchmark_comparison(selected_comparison: pd.DataFrame) -> pd.DataFrame:
    output = selected_comparison.copy()
    output["reference_type"] = output["comparison_name"].map(_reference_type_from_comparison_name)
    output["benchmark_symbol"] = output["comparison_name"].map(_benchmark_symbol_from_name)
    output["primary_beats_comparison_on_average_return"] = (
        pd.to_numeric(output["delta_average_test_return_10bps"], errors="coerce") > 0.0
    )
    output["comparison_beats_primary_on_raw_return"] = (
        pd.to_numeric(output["delta_average_test_return_10bps"], errors="coerce") < 0.0
    )
    output["raw_return_caveat"] = [
        _raw_return_caveat(row) for _, row in output.iterrows()
    ]
    for column in BENCHMARK_COMPARISON_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output.loc[:, BENCHMARK_COMPARISON_COLUMNS]


def _build_underperformance_windows(
    *,
    primary_rows: pd.DataFrame,
    benchmark_fold_returns: pd.DataFrame,
) -> pd.DataFrame:
    if benchmark_fold_returns.empty:
        return pd.DataFrame(columns=UNDERPERFORMANCE_COLUMNS)
    primary_by_fold = primary_rows.set_index("fold", drop=False)
    rows: list[dict[str, object]] = []
    for _, benchmark in benchmark_fold_returns.iterrows():
        fold = str(benchmark["fold"])
        if fold not in primary_by_fold.index:
            continue
        primary = primary_by_fold.loc[fold]
        primary_return = _as_float(primary["total_return"])
        benchmark_return = _as_float(benchmark["total_return"])
        excess = primary_return - benchmark_return
        rows.append(
            {
                "benchmark_symbol": benchmark["benchmark_symbol"],
                "fold": fold,
                "train_window_years": _as_int(primary.get("train_window_years")),
                "validation_window_months": _as_int(primary.get("validation_window_months")),
                "test_window_months": _as_int(primary.get("test_window_months")),
                "sequence_number": _as_int(primary.get("sequence_number")),
                "test_start": _date_string(primary.get("test_start")),
                "test_end": _date_string(primary.get("test_end")),
                "primary_total_return": primary_return,
                "benchmark_total_return": benchmark_return,
                "excess_return_vs_benchmark": excess,
                "underperformed_benchmark": bool(excess < 0.0),
                "underperformance_severity": _underperformance_severity(excess),
            }
        )
    return pd.DataFrame(rows, columns=UNDERPERFORMANCE_COLUMNS).sort_values(
        ["benchmark_symbol", "test_start", "fold"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_turnover_diagnostics(primary_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in primary_rows.sort_values("fold", kind="mergesort").iterrows():
        rows.append(
            {
                "fold": row["fold"],
                "train_window_years": _as_int(row.get("train_window_years")),
                "validation_window_months": _as_int(row.get("validation_window_months")),
                "test_window_months": _as_int(row.get("test_window_months")),
                "sequence_number": _as_int(row.get("sequence_number")),
                "test_start": _date_string(row.get("test_start")),
                "test_end": _date_string(row.get("test_end")),
                "total_transaction_cost": _as_float(row.get("total_transaction_cost")),
                "mean_daily_turnover": _as_float(row.get("mean_daily_turnover")),
                "max_daily_turnover": _as_float(row.get("max_daily_turnover")),
                "run_key": row.get("run_key", ""),
                "source": "results_csv_fold_metric",
                "reconstruction_level": "fold_metric_only",
            }
        )
    return pd.DataFrame(rows, columns=TURNOVER_COLUMNS)


def _build_holdings_summary(results: pd.DataFrame) -> pd.DataFrame:
    if set(HOLDINGS_SUMMARY_COLUMNS).issubset(results.columns):
        holdings = results.loc[:, HOLDINGS_SUMMARY_COLUMNS].copy()
        return holdings.sort_values(["selected_days", "symbol"], ascending=[False, True])

    return pd.DataFrame(
        [
            {
                "symbol": "not_reconstructable",
                "selected_days": 0,
                "avg_target_weight": np.nan,
                "avg_prediction": np.nan,
                "avg_signal_rank": np.nan,
                "first_selected_date": "",
                "last_selected_date": "",
                "source": "results_csv_fold_metric",
                "reconstruction_level": "not_available",
                "caveat": (
                    "Daily selected ETF holdings are not present in the provided "
                    "fold-level result CSVs."
                ),
            }
        ],
        columns=HOLDINGS_SUMMARY_COLUMNS,
    )


def _build_replay_summary(
    *,
    primary_rows: pd.DataFrame,
    selected_summary: pd.DataFrame,
    benchmark_fold_returns: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    benchmark_symbols: Sequence[str],
    unavailable_artifacts: Sequence[str],
) -> pd.DataFrame:
    primary_summary = _selected_summary_row(selected_summary)
    primary_metrics = _fold_metrics(primary_rows)
    benchmark_means = _benchmark_average_returns(benchmark_fold_returns)
    best_reference = _best_raw_return_reference(benchmark_comparison)
    decision, next_action, benchmark_gate = _decision_and_next_action(
        primary_metrics=primary_metrics,
        benchmark_comparison=benchmark_comparison,
        unavailable_artifacts=unavailable_artifacts,
    )
    row = {
        "candidate_role": "primary_candidate",
        "candidate_label": PRIMARY_SPEC.label,
        "dataset_source": DATASET_SOURCE_LABEL,
        "model_type": PRIMARY_SPEC.model_type,
        "top_k": PRIMARY_SPEC.top_k,
        "allocation_mode": "equal_weight",
        "min_holding_days": PRIMARY_SPEC.min_holding_days,
        "switch_margin": PRIMARY_SPEC.switch_margin,
        "transaction_cost_bps": DECISION_ANCHOR_BPS,
        "train_window_years": "3,5",
        "validation_window_months": "12",
        "test_window_months": "3,6,12",
        "feature_set_version": FEATURE_SET_VERSION,
        "target_col": TARGET_COL,
        "fold_count": primary_metrics["fold_count"],
        "average_total_return": primary_metrics["average_total_return"],
        "median_total_return": primary_metrics["median_total_return"],
        "positive_window_rate": primary_metrics["positive_window_rate"],
        "average_sharpe_like": primary_metrics["average_sharpe_like"],
        "average_max_drawdown": primary_metrics["average_max_drawdown"],
        "worst_max_drawdown": primary_metrics["worst_max_drawdown"],
        "average_total_transaction_cost": primary_metrics[
            "average_total_transaction_cost"
        ],
        "average_mean_daily_turnover": primary_metrics["average_mean_daily_turnover"],
        "benchmark_symbols": ",".join(benchmark_symbols),
        "spy_average_total_return": benchmark_means.get("SPY", np.nan),
        "qqq_average_total_return": benchmark_means.get("QQQ", np.nan),
        "best_raw_return_reference": best_reference,
        "benchmark_gate": benchmark_gate,
        "decision": decision,
        "next_action": next_action,
        "artifact_caveats": "; ".join(unavailable_artifacts),
    }
    if primary_summary is not None:
        for source_col, target_col in [
            ("recommendation", "source_recommendation"),
            ("next_step", "source_next_step"),
        ]:
            if source_col in primary_summary.index:
                row[target_col] = primary_summary[source_col]
    extra_columns = sorted(set(row) - set(REPLAY_SUMMARY_COLUMNS))
    return pd.DataFrame([row], columns=[*REPLAY_SUMMARY_COLUMNS, *extra_columns])


def _fold_metrics(primary_rows: pd.DataFrame) -> dict[str, object]:
    return {
        "fold_count": int(len(primary_rows)),
        "average_total_return": _mean(primary_rows, "total_return"),
        "median_total_return": _median(primary_rows, "total_return"),
        "positive_window_rate": _positive_rate(primary_rows["total_return"]),
        "average_sharpe_like": _mean(primary_rows, "sharpe_like"),
        "average_max_drawdown": _mean(primary_rows, "max_drawdown"),
        "worst_max_drawdown": _min(primary_rows, "max_drawdown"),
        "average_total_transaction_cost": _mean(primary_rows, "total_transaction_cost"),
        "average_mean_daily_turnover": _mean(primary_rows, "mean_daily_turnover"),
    }


def _selected_summary_row(selected_summary: pd.DataFrame) -> pd.Series | None:
    if "candidate_role" not in selected_summary.columns:
        return None
    matches = selected_summary.loc[
        selected_summary["candidate_role"].astype(str).eq("primary_balanced_candidate")
    ]
    if matches.empty:
        matches = selected_summary.loc[
            selected_summary["candidate_label"].astype(str).eq(PRIMARY_SPEC.label)
        ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _benchmark_average_returns(benchmark_fold_returns: pd.DataFrame) -> dict[str, float]:
    if benchmark_fold_returns.empty:
        return {}
    means: dict[str, float] = {}
    for symbol, group in benchmark_fold_returns.groupby("benchmark_symbol", dropna=False):
        means[str(symbol).upper()] = _mean(group, "total_return")
    return means


def _best_raw_return_reference(benchmark_comparison: pd.DataFrame) -> str:
    if benchmark_comparison.empty:
        return "unknown"
    rows = benchmark_comparison.copy()
    rows["comparison_average_test_return_10bps"] = pd.to_numeric(
        rows["comparison_average_test_return_10bps"],
        errors="coerce",
    )
    rows = rows.dropna(subset=["comparison_average_test_return_10bps"])
    if rows.empty:
        return "unknown"
    best = rows.sort_values(
        ["comparison_average_test_return_10bps", "comparison_candidate_label"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(best["comparison_candidate_label"])


def _decision_and_next_action(
    *,
    primary_metrics: dict[str, object],
    benchmark_comparison: pd.DataFrame,
    unavailable_artifacts: Sequence[str],
) -> tuple[str, str, str]:
    benchmark_rows = benchmark_comparison.loc[
        benchmark_comparison["reference_type"].eq("benchmark")
    ].copy()
    benchmark_deltas = pd.to_numeric(
        benchmark_rows["delta_average_test_return_10bps"],
        errors="coerce",
    ).dropna()
    if benchmark_deltas.empty:
        return (
            "keep as selected candidate but not target",
            "richer replay run",
            "missing benchmark comparison",
        )

    benchmark_gate = "pass" if bool((benchmark_deltas >= 0.0).all()) else "watch"
    positive_rate = _as_float(primary_metrics["positive_window_rate"])
    average_return = _as_float(primary_metrics["average_total_return"])
    if benchmark_gate == "pass" and average_return > 0.0 and positive_rate >= 0.55:
        if unavailable_artifacts:
            return (
                "keep as selected candidate but not target",
                "richer replay run",
                "pass but daily replay artifacts missing",
            )
        return ("promote to current research baseline target", "model-card draft", "pass")
    if average_return <= 0.0 or positive_rate < 0.45:
        return ("reject/deprioritize", "feature engineering", benchmark_gate)
    if unavailable_artifacts:
        return ("keep as selected candidate but not target", "richer replay run", benchmark_gate)
    return (
        "keep as selected candidate but not target",
        "feature engineering",
        benchmark_gate,
    )


def _unavailable_artifacts(results: pd.DataFrame) -> tuple[str, ...]:
    unavailable: list[str] = []
    daily_equity_columns = {"date", "portfolio_return", "equity"}
    holdings_columns = {"date", "symbol", "selected", "target_weight", "prediction"}
    if not daily_equity_columns.issubset(results.columns):
        unavailable.append(
            "Daily selected-candidate strategy equity curve cannot be reconstructed "
            "from the fold-level results CSV."
        )
        unavailable.append(
            "Daily selected-candidate drawdown path cannot be reconstructed; only "
            "fold max_drawdown metrics are available."
        )
    if not holdings_columns.issubset(results.columns):
        unavailable.append(
            "Daily selected ETF holdings and selected-symbol summaries cannot be "
            "reconstructed from the provided CSVs."
        )
    return tuple(unavailable)


def _historical_champion_missing(test_rows: pd.DataFrame) -> bool:
    return test_rows.loc[
        test_rows["model_type"].astype(str).eq("linear_regression")
        & _int_eq(test_rows, "top_k", 1)
        & _int_eq(test_rows, "min_holding_days", 10)
        & _float_eq(test_rows, "switch_margin", 0.002)
        & _float_eq(test_rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)
    ].empty


def _reference_type_from_comparison_name(name: object) -> str:
    text = str(name).lower()
    if "spy" in text or "qqq" in text or "buy_hold" in text:
        return "benchmark"
    if "historical" in text:
        return "historical_champion"
    if "return_reference" in text:
        return "return_reference"
    if "drawdown_guardrail" in text:
        return "drawdown_guardrail"
    return "reference"


def _benchmark_symbol_from_name(name: object) -> str:
    text = str(name).lower()
    if "spy" in text:
        return "SPY"
    if "qqq" in text:
        return "QQQ"
    return ""


def _raw_return_caveat(row: pd.Series) -> str:
    comparison_label = str(row["comparison_candidate_label"])
    delta = _as_float(row["delta_average_test_return_10bps"])
    if pd.isna(delta):
        return "raw return comparison is unavailable"
    if delta < 0.0:
        return f"{comparison_label} is stronger on raw average return"
    if delta > 0.0:
        return "primary is stronger on raw average return"
    return "raw average return is tied"


def _raw_return_sentence(comparison: pd.DataFrame, symbol: str) -> str:
    rows = comparison.loc[comparison["benchmark_symbol"].astype(str).str.upper().eq(symbol)]
    if rows.empty:
        return f"{symbol} buy-and-hold comparison is unavailable."
    row = rows.iloc[0]
    caveat = str(row["raw_return_caveat"])
    delta = _format_number(row["delta_average_test_return_10bps"])
    return f"{symbol} raw-return note: {caveat}; primary-minus-{symbol} delta `{delta}`."


def _comparison_subset(comparison: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return comparison.loc[comparison["comparison_name"].isin(names)].copy()


def _holdings_markdown_sentence(holdings_summary: pd.DataFrame) -> str:
    if holdings_summary.empty:
        return "No holdings summary rows are available."
    if set(holdings_summary["reconstruction_level"].astype(str)) == {"not_available"}:
        return (
            "Holdings are not available because the provided inputs contain fold metrics, "
            "not daily selected-symbol rows."
        )
    return "Holdings summary was available in the provided inputs."


def _underperformance_severity(excess_return: float) -> str:
    if pd.isna(excess_return) or excess_return >= 0.0:
        return "none"
    if excess_return <= -0.05:
        return "material"
    if excess_return <= -0.02:
        return "moderate"
    return "minor"


def _series_in_values(series: pd.Series, values: Sequence[int]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    for value in values:
        mask |= numeric.eq(value)
    return mask


def _float_eq(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.isclose(numeric, value, atol=1e-9), index=frame.index)


def _int_eq(frame: pd.DataFrame, column: str, value: int) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").eq(int(value))


def _mean(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return float("nan")
    return float(pd.to_numeric(rows[column], errors="coerce").mean())


def _median(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return float("nan")
    return float(pd.to_numeric(rows[column], errors="coerce").median())


def _min(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return float("nan")
    return float(pd.to_numeric(rows[column], errors="coerce").min())


def _as_int(value: object) -> int | float:
    numeric = _as_float(value)
    if pd.isna(numeric):
        return np.nan
    return int(numeric)


def _date_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _format_number(value: object) -> str:
    numeric = _as_float(value)
    if pd.isna(numeric):
        return "nan"
    return f"{numeric:.4f}"


create_tiingo_selected_candidate_replay_artifacts_from_args = (
    run_tiingo_selected_candidate_replay_artifacts_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_selected_candidate_replay_artifacts_from_args())
