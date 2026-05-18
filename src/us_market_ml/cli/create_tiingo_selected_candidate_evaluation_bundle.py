"""Create the Tiingo selected-candidate evaluation bundle."""

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
from us_market_ml.data.local_reader import read_csv_files, read_parquet_files

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_focused_linear_ridge_confirmation/"
    "tiingo_focused_linear_ridge_confirmation_results.csv"
)
DEFAULT_SUMMARY_CSV = Path(
    "artifacts/reports/tiingo_focused_linear_ridge_confirmation/"
    "tiingo_focused_linear_ridge_confirmation_summary.csv"
)
DEFAULT_DECISION_MATRIX_CSV = Path(
    "artifacts/reports/tiingo_focused_linear_ridge_confirmation/"
    "tiingo_focused_linear_ridge_confirmation_decision_matrix.csv"
)
DEFAULT_CORE_LINEAR_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_modern_baseline_sweep/"
    "tiingo_modern_baseline_sweep_core_linear_results.csv"
)
DEFAULT_CORE_LINEAR_SUMMARY_CSV = Path(
    "artifacts/reports/tiingo_modern_baseline_sweep_core_linear/"
    "tiingo_modern_baseline_summary.csv"
)
DEFAULT_MARKET_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_BENCHMARK_SYMBOLS = ("SPY", "QQQ")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_selected_candidate_evaluation")

SUMMARY_PATH_NAME = "selected_candidate_summary.csv"
DECISION_MATRIX_PATH_NAME = "selected_candidate_decision_matrix.csv"
WINDOW_STABILITY_PATH_NAME = "selected_candidate_window_stability.csv"
COST_SENSITIVITY_PATH_NAME = "selected_candidate_cost_sensitivity.csv"
COMPARISON_PATH_NAME = "selected_candidate_comparison.csv"
BENCHMARK_REFERENCES_PATH_NAME = "selected_candidate_benchmark_references.csv"
HISTORICAL_CHAMPION_REFERENCE_PATH_NAME = (
    "selected_candidate_historical_champion_reference.csv"
)
REPORT_PATH_NAME = "selected_candidate_evaluation_report.md"
MODEL_CARD_DRAFT_PATH_NAME = "selected_candidate_model_card_draft.md"
OUTPUT_PATH_NAMES = (
    SUMMARY_PATH_NAME,
    DECISION_MATRIX_PATH_NAME,
    WINDOW_STABILITY_PATH_NAME,
    COST_SENSITIVITY_PATH_NAME,
    COMPARISON_PATH_NAME,
    BENCHMARK_REFERENCES_PATH_NAME,
    HISTORICAL_CHAMPION_REFERENCE_PATH_NAME,
    REPORT_PATH_NAME,
    MODEL_CARD_DRAFT_PATH_NAME,
)

DECISION_ANCHOR_BPS = 10.0
SENSITIVITY_BPS = 20.0
SENSITIVITY_COSTS_BPS = (DECISION_ANCHOR_BPS, SENSITIVITY_BPS)

SUMMARY_COLUMNS = [
    "candidate_role",
    "candidate_classification",
    "candidate_key",
    "candidate_label",
    "dataset_source",
    "sweep_profile",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "benchmark_symbol",
    "min_holding_days",
    "switch_margin",
    "decision_anchor_bps",
    "transaction_cost_bps",
    "window_count_10bps",
    "average_test_return_10bps",
    "median_test_return_10bps",
    "positive_window_rate_10bps",
    "average_sharpe_like_10bps",
    "average_max_drawdown_10bps",
    "worst_max_drawdown_10bps",
    "average_total_transaction_cost_10bps",
    "average_mean_daily_turnover_10bps",
    "window_count_20bps",
    "average_test_return_20bps",
    "median_test_return_20bps",
    "positive_window_rate_20bps",
    "average_sharpe_like_20bps",
    "average_max_drawdown_20bps",
    "worst_max_drawdown_20bps",
    "average_total_transaction_cost_20bps",
    "average_mean_daily_turnover_20bps",
    "cost_delta_20_vs_10",
    "return_strength",
    "positive_window_strength",
    "drawdown_control",
    "cost_resilience",
    "evidence_scope",
    "source_decision_recommendation",
    "source_decision_rationale",
    "recommendation",
    "next_step",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_role",
    "candidate_classification",
    "candidate_label",
    "decision_anchor_bps",
    "average_test_return_gate",
    "positive_window_gate",
    "cost_sensitivity_gate",
    "drawdown_vs_top1_gate",
    "historical_champion_gate",
    "benchmark_reference_gate",
    "evidence_scope_gate",
    "return_strength",
    "positive_window_strength",
    "drawdown_control",
    "cost_resilience",
    "drawdown_vs_return_reference",
    "cost_resilience_vs_drawdown_guardrail",
    "historical_champion_profile",
    "benchmark_reference_profile",
    "evidence_scope",
    "promote_primary_candidate",
    "recommendation",
    "next_action",
    "rationale",
]

WINDOW_STABILITY_COLUMNS = [
    "candidate_role",
    "candidate_label",
    "decision_anchor_bps",
    "stability_dimension",
    "stability_value",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "window_count",
    "average_test_return",
    "median_test_return",
    "positive_window_rate",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_total_transaction_cost",
    "average_mean_daily_turnover",
    "stability_rating",
]

COST_SENSITIVITY_COLUMNS = [
    "candidate_role",
    "candidate_label",
    "transaction_cost_bps",
    "is_decision_anchor",
    "is_20bps_sensitivity",
    "window_count",
    "average_test_return",
    "median_test_return",
    "positive_window_rate",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_total_transaction_cost",
    "average_mean_daily_turnover",
    "return_delta_vs_10bps",
    "positive_rate_delta_vs_10bps",
    "worst_drawdown_delta_vs_10bps",
]

COMPARISON_COLUMNS = [
    "comparison_name",
    "primary_candidate_label",
    "comparison_candidate_label",
    "decision_anchor_bps",
    "primary_average_test_return_10bps",
    "comparison_average_test_return_10bps",
    "delta_average_test_return_10bps",
    "primary_positive_window_rate_10bps",
    "comparison_positive_window_rate_10bps",
    "delta_positive_window_rate_10bps",
    "primary_average_max_drawdown_10bps",
    "comparison_average_max_drawdown_10bps",
    "delta_average_max_drawdown_10bps",
    "primary_worst_max_drawdown_10bps",
    "comparison_worst_max_drawdown_10bps",
    "delta_worst_max_drawdown_10bps",
    "primary_average_test_return_20bps",
    "comparison_average_test_return_20bps",
    "delta_average_test_return_20bps",
    "primary_cost_delta_20_vs_10",
    "comparison_cost_delta_20_vs_10",
    "delta_cost_delta_20_vs_10",
    "primary_average_total_transaction_cost_10bps",
    "comparison_average_total_transaction_cost_10bps",
    "delta_average_total_transaction_cost_10bps",
    "primary_average_mean_daily_turnover_10bps",
    "comparison_average_mean_daily_turnover_10bps",
    "delta_average_mean_daily_turnover_10bps",
    "interpretation",
]

BENCHMARK_REFERENCE_COLUMNS = [
    "candidate_role",
    "candidate_classification",
    "candidate_label",
    "dataset_source",
    "sweep_profile",
    "model_family",
    "model_type",
    "allocation_mode",
    "benchmark_symbol",
    "transaction_cost_bps",
    "price_column",
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "sequence_number",
    "test_start",
    "test_end",
    "first_price_date",
    "last_price_date",
    "observation_count",
    "total_return",
    "mean_daily_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "run_key",
]

PRIMARY_ROLE = "primary_balanced_candidate"
RETURN_REFERENCE_ROLE = "return_reference"
DRAWDOWN_GUARDRAIL_ROLE = "drawdown_guardrail"
HISTORICAL_CHAMPION_ROLE = "historical_return_champion_reference"
BENCHMARK_REFERENCE_ROLE = "benchmark_reference"

_FOLD_WINDOW_RE = re.compile(
    r"tw(?P<train_window_years>\d+)y"
    r"_v(?P<validation_window_months>\d+)m"
    r"_t(?P<test_window_months>\d+)m"
    r"_q(?P<sequence_number>\d+)"
    r"_(?P<test_start>\d{8})"
    r"_(?P<test_end>\d{8})"
)


@dataclass(frozen=True)
class SelectedCandidateSpec:
    """One confirmed selected-candidate evaluation role."""

    role: str
    classification: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float

    @property
    def key(self) -> str:
        return (
            f"{self.role}:{self.model_type}:top{self.top_k}:hold"
            f"{self.min_holding_days}:margin{self.switch_margin:g}"
        )

    @property
    def label(self) -> str:
        return (
            f"{self.model_type} / top{self.top_k} / hold {self.min_holding_days} / "
            f"margin {self.switch_margin:g}"
        )


@dataclass(frozen=True)
class TiingoSelectedCandidateEvaluationData:
    """All tables and metadata for the selected-candidate evaluation bundle."""

    candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    window_stability: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    comparison: pd.DataFrame
    benchmark_references: pd.DataFrame
    historical_champion_reference: pd.DataFrame
    warnings: tuple[str, ...]
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    source_summary_row_count: int
    source_decision_matrix_row_count: int
    dataset_source: str = DATASET_SOURCE_LABEL


@dataclass(frozen=True)
class BenchmarkTestWindow:
    """One focused confirmation test window used for benchmark references."""

    fold: str
    train_window_years: float
    validation_window_months: float
    test_window_months: float
    sequence_number: float
    test_start: pd.Timestamp
    test_end: pd.Timestamp


SELECTED_CANDIDATE_SPECS = (
    SelectedCandidateSpec(
        role=PRIMARY_ROLE,
        classification="primary research baseline candidate",
        model_type="ridge",
        top_k=5,
        min_holding_days=20,
        switch_margin=0.005,
    ),
    SelectedCandidateSpec(
        role=RETURN_REFERENCE_ROLE,
        classification="return reference",
        model_type="ridge",
        top_k=1,
        min_holding_days=20,
        switch_margin=0.005,
    ),
    SelectedCandidateSpec(
        role=DRAWDOWN_GUARDRAIL_ROLE,
        classification="drawdown guardrail",
        model_type="ridge",
        top_k=5,
        min_holding_days=5,
        switch_margin=0.002,
    ),
)

HISTORICAL_CHAMPION_SPEC = SelectedCandidateSpec(
    role=HISTORICAL_CHAMPION_ROLE,
    classification="historical return champion reference",
    model_type="linear_regression",
    top_k=1,
    min_holding_days=10,
    switch_margin=0.002,
)

MODELED_REFERENCE_SPECS = (*SELECTED_CANDIDATE_SPECS, HISTORICAL_CHAMPION_SPEC)


def run_tiingo_selected_candidate_evaluation_bundle_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the selected-candidate evaluation bundle."""

    args = _parse_args(argv)
    report_data = build_tiingo_selected_candidate_evaluation_bundle(
        results_csv=args.results_csv,
        summary_csv=args.summary_csv,
        decision_matrix_csv=args.decision_matrix_csv,
        core_linear_results_csv=args.core_linear_results_csv,
        core_linear_summary_csv=args.core_linear_summary_csv,
        market_dataset_glob=args.market_dataset_glob,
        benchmark_symbols=args.benchmark_symbols,
        skip_missing_references=args.skip_missing_references,
    )
    outputs = write_tiingo_selected_candidate_evaluation_bundle(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo selected-candidate evaluation bundle")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"decision anchor: {DECISION_ANCHOR_BPS:g}bps")
    print(f"20bps sensitivity: {SENSITIVITY_BPS:g}bps")
    print(f"results CSV: {args.results_csv}")
    print(f"summary CSV: {args.summary_csv}")
    print(f"decision matrix CSV: {args.decision_matrix_csv}")
    print(f"core linear results CSV: {args.core_linear_results_csv}")
    print(f"core linear summary CSV: {args.core_linear_summary_csv}")
    print(f"market dataset glob: {args.market_dataset_glob}")
    print(f"benchmark symbols: {', '.join(_parse_benchmark_symbols(args.benchmark_symbols))}")
    print(f"output dir: {args.output_dir}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    if report_data.warnings:
        print()
        print("Warnings:")
        for warning in report_data.warnings:
            print(f"- {warning}")
    print()
    print(report_data.decision_matrix.to_string(index=False))
    return 0


def build_tiingo_selected_candidate_evaluation_bundle(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    decision_matrix_csv: Path = DEFAULT_DECISION_MATRIX_CSV,
    core_linear_results_csv: Path = DEFAULT_CORE_LINEAR_RESULTS_CSV,
    core_linear_summary_csv: Path = DEFAULT_CORE_LINEAR_SUMMARY_CSV,
    market_dataset_glob: str | Path = DEFAULT_MARKET_DATASET_GLOB,
    benchmark_symbols: str | Sequence[str] = DEFAULT_BENCHMARK_SYMBOLS,
    skip_missing_references: bool = False,
) -> TiingoSelectedCandidateEvaluationData:
    """Read focused confirmation outputs and build selected-candidate tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    source_summary = pd.read_csv(summary_csv)
    source_decision_matrix = pd.read_csv(decision_matrix_csv)
    resolved_benchmark_symbols = _parse_benchmark_symbols(benchmark_symbols)

    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _enrich_test_rows(test_rows)
    included = enriched.loc[
        enriched["model_family"].astype(str).eq("baseline")
        & _is_selected_candidate_spec(enriched)
        & _series_in_float_values(enriched["transaction_cost_bps"], SENSITIVITY_COSTS_BPS)
    ].copy()
    if included.empty:
        raise ValueError("No selected Tiingo ridge candidate test rows were found.")

    warnings = list(_source_input_warnings(source_summary, source_decision_matrix))
    historical_rows, historical_summary, historical_warnings = (
        _build_historical_champion_reference_summary(
            core_linear_results_csv=core_linear_results_csv,
            core_linear_summary_csv=core_linear_summary_csv,
            skip_missing_references=skip_missing_references,
        )
    )
    warnings.extend(historical_warnings)

    candidate_summary = _build_candidate_summary(
        included,
        source_decision_matrix=source_decision_matrix,
        specs=SELECTED_CANDIDATE_SPECS,
    )
    if not historical_summary.empty:
        candidate_summary = pd.concat(
            [candidate_summary, historical_summary],
            ignore_index=True,
        )
    missing_roles = [
        spec.role
        for spec in SELECTED_CANDIDATE_SPECS
        if spec.role not in set(candidate_summary["candidate_role"])
    ]
    if missing_roles:
        raise ValueError(
            "Requested selected candidates were not found in the results CSV: "
            + ", ".join(missing_roles)
        )

    benchmark_references, benchmark_summary, benchmark_warnings = _build_benchmark_references(
        market_dataset_glob=market_dataset_glob,
        benchmark_symbols=resolved_benchmark_symbols,
        focused_test_rows=included,
        skip_missing_references=skip_missing_references,
    )
    warnings.extend(benchmark_warnings)
    if not benchmark_summary.empty:
        candidate_summary = pd.concat(
            [candidate_summary, benchmark_summary],
            ignore_index=True,
        )

    modeled_rows = pd.concat([included, historical_rows], ignore_index=True)
    cost_sensitivity = _build_cost_sensitivity(
        modeled_rows,
        candidate_summary,
        specs=MODELED_REFERENCE_SPECS,
    )
    window_stability = _build_window_stability(
        modeled_rows,
        specs=MODELED_REFERENCE_SPECS,
    )
    comparison = _build_comparison(candidate_summary)
    decision_matrix = _build_decision_matrix(candidate_summary)
    candidate_summary = _apply_decision_recommendations(candidate_summary, decision_matrix)
    historical_champion_reference = candidate_summary.loc[
        candidate_summary["candidate_role"].astype(str).eq(HISTORICAL_CHAMPION_ROLE)
    ].copy()

    return TiingoSelectedCandidateEvaluationData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        window_stability=window_stability,
        cost_sensitivity=cost_sensitivity,
        comparison=comparison,
        benchmark_references=benchmark_references,
        historical_champion_reference=historical_champion_reference,
        warnings=tuple(warnings),
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
        source_summary_row_count=int(len(source_summary)),
        source_decision_matrix_row_count=int(len(source_decision_matrix)),
    )


def write_tiingo_selected_candidate_evaluation_bundle(
    *,
    report_data: TiingoSelectedCandidateEvaluationData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the selected-candidate evaluation bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        WINDOW_STABILITY_PATH_NAME: output_dir / WINDOW_STABILITY_PATH_NAME,
        COST_SENSITIVITY_PATH_NAME: output_dir / COST_SENSITIVITY_PATH_NAME,
        COMPARISON_PATH_NAME: output_dir / COMPARISON_PATH_NAME,
        BENCHMARK_REFERENCES_PATH_NAME: output_dir / BENCHMARK_REFERENCES_PATH_NAME,
        HISTORICAL_CHAMPION_REFERENCE_PATH_NAME: (
            output_dir / HISTORICAL_CHAMPION_REFERENCE_PATH_NAME
        ),
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
        MODEL_CARD_DRAFT_PATH_NAME: output_dir / MODEL_CARD_DRAFT_PATH_NAME,
    }
    report_data.candidate_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    report_data.window_stability.to_csv(outputs[WINDOW_STABILITY_PATH_NAME], index=False)
    report_data.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_PATH_NAME], index=False)
    report_data.comparison.to_csv(outputs[COMPARISON_PATH_NAME], index=False)
    report_data.benchmark_references.to_csv(
        outputs[BENCHMARK_REFERENCES_PATH_NAME],
        index=False,
    )
    report_data.historical_champion_reference.to_csv(
        outputs[HISTORICAL_CHAMPION_REFERENCE_PATH_NAME],
        index=False,
    )
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_selected_candidate_evaluation_markdown(report_data),
        encoding="utf-8",
    )
    outputs[MODEL_CARD_DRAFT_PATH_NAME].write_text(
        render_selected_candidate_model_card_draft(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_selected_candidate_evaluation_markdown(
    report_data: TiingoSelectedCandidateEvaluationData,
) -> str:
    """Render the selected-candidate evaluation Markdown report."""

    primary = _summary_row(report_data.candidate_summary, PRIMARY_ROLE)
    return_reference = _summary_row(report_data.candidate_summary, RETURN_REFERENCE_ROLE)
    drawdown_guardrail = _summary_row(report_data.candidate_summary, DRAWDOWN_GUARDRAIL_ROLE)
    historical_champion = _summary_row_optional(
        report_data.candidate_summary,
        HISTORICAL_CHAMPION_ROLE,
    )
    benchmark_rows = _benchmark_summary_rows(report_data.candidate_summary)
    primary_decision = _decision_row(report_data.decision_matrix, PRIMARY_ROLE)
    recommended_next_step = str(primary_decision["next_action"])
    window_count = int(primary["window_count_10bps"])

    primary_metrics = _primary_metric_table(primary)
    benchmark_table = _format_for_markdown(
        benchmark_rows[
            [
                "candidate_label",
                "benchmark_symbol",
                "average_test_return_10bps",
                "positive_window_rate_10bps",
                "average_sharpe_like_10bps",
                "average_max_drawdown_10bps",
                "worst_max_drawdown_10bps",
                "window_count_10bps",
            ]
        ]
        if not benchmark_rows.empty
        else benchmark_rows,
        exclude={"candidate_label", "benchmark_symbol"},
    )
    historical_table = _format_for_markdown(
        pd.DataFrame([historical_champion])
        [
            [
                "candidate_label",
                "average_test_return_10bps",
                "positive_window_rate_10bps",
                "average_sharpe_like_10bps",
                "average_max_drawdown_10bps",
                "worst_max_drawdown_10bps",
                "average_test_return_20bps",
                "cost_delta_20_vs_10",
                "average_mean_daily_turnover_10bps",
            ]
        ]
        if historical_champion is not None
        else pd.DataFrame(),
        exclude={"candidate_label"},
    )
    comparison_table = _format_for_markdown(
        report_data.comparison,
        exclude={"comparison_name", "primary_candidate_label", "comparison_candidate_label"},
    )
    stability_table = _format_for_markdown(
        report_data.window_stability.head(80),
        exclude={"candidate_role", "candidate_label", "stability_dimension", "stability_rating"},
    )
    cost_table = _format_for_markdown(
        report_data.cost_sensitivity,
        exclude={"candidate_role", "candidate_label", "is_decision_anchor", "is_20bps_sensitivity"},
    )
    decision_table = _format_for_markdown(
        report_data.decision_matrix[
            [
                "candidate_label",
                "candidate_classification",
                "average_test_return_gate",
                "positive_window_gate",
                "cost_sensitivity_gate",
                "drawdown_vs_top1_gate",
                "historical_champion_gate",
                "benchmark_reference_gate",
                "evidence_scope_gate",
                "recommendation",
                "next_action",
            ]
        ],
        exclude={
            "candidate_label",
            "candidate_classification",
            "average_test_return_gate",
            "positive_window_gate",
            "cost_sensitivity_gate",
            "drawdown_vs_top1_gate",
            "historical_champion_gate",
            "benchmark_reference_gate",
            "evidence_scope_gate",
            "recommendation",
            "next_action",
        },
    )

    lines = [
        "# Tiingo Selected Candidate Evaluation Bundle",
        "",
        "## Executive Summary",
        "",
        f"Current primary candidate: `{primary['candidate_label']}`.",
        "",
        "This is a research candidate evaluation bundle, not a production-ready "
        "model approval. It does not run a new sweep, `tree_screen`, MLP, signal "
        "export, S3 readers, or any production model-card update.",
        "",
        f"The decision anchor is `{DECISION_ANCHOR_BPS:g}bps`; `{SENSITIVITY_BPS:g}bps` "
        "is included as cost sensitivity. `5bps` is not used as the main decision anchor.",
        "",
        f"Input result rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included selected-candidate test rows: "
        f"`{report_data.included_test_row_count}`.",
        "",
        f"Primary candidate decision: `{primary_decision['recommendation']}`.",
        "",
        "## Why This Candidate Was Selected",
        "",
        f"- Stronger balanced profile than the top1 return reference "
        f"`{return_reference['candidate_label']}`.",
        "- Better drawdown than top1 at the 10bps anchor, especially on worst "
        "max drawdown.",
        f"- Better cost resilience than shorter holding / higher turnover variants, "
        f"including the guardrail `{drawdown_guardrail['candidate_label']}`.",
        f"- Broad `{window_count}`-window confirmation evidence at the 10bps anchor.",
        "",
        "## Primary Candidate Metrics",
        "",
        _markdown_table(_format_for_markdown(primary_metrics, exclude={"metric"})),
        "",
        "## Benchmark References",
        "",
        "SPY and QQQ buy-and-hold references are included so the selected ridge "
        "candidate is not judged only against internal ridge variants.",
        "",
        "Benchmark calculation caveat: buy-and-hold metrics are computed from "
        "adjusted close over the focused confirmation test windows, using the "
        "first and last available trading-day closes inside each window. No "
        "transaction cost is applied to benchmark rows; `transaction_cost_bps` "
        "is marked as `0`.",
        "",
        _markdown_table(benchmark_table),
        "",
        "## Historical Champion Reference",
        "",
        "The original historical fixed-fold / earlier return champion is retained "
        "for continuity: `linear_regression / top1 / hold 10 / margin 0.002`. "
        "It is not the current primary candidate.",
        "",
        _markdown_table(historical_table),
        "",
        "## Primary vs Return Reference",
        "",
        _markdown_table(
            comparison_table.loc[
                comparison_table["comparison_name"].eq("primary_vs_return_reference")
            ]
        ),
        "",
        "The top1 return reference can edge raw return, but it is not promoted on "
        "return alone. The selected top5 candidate keeps the decision focused on a "
        "more balanced return, drawdown, cost, and turnover profile.",
        "",
        "## Primary vs Drawdown Guardrail",
        "",
        _markdown_table(
            comparison_table.loc[
                comparison_table["comparison_name"].eq("primary_vs_drawdown_guardrail")
            ]
        ),
        "",
        "The drawdown guardrail remains useful for risk-control comparisons, but "
        "the shorter holding configuration has higher observed transaction cost "
        "and turnover pressure.",
        "",
        "## Primary vs Historical Champion",
        "",
        _markdown_table(
            comparison_table.loc[
                comparison_table["comparison_name"].eq("primary_vs_historical_champion")
            ]
        ),
        "",
        _historical_markdown_sentence(primary, historical_champion),
        "",
        "## Primary vs SPY Buy-and-Hold",
        "",
        _markdown_table(
            comparison_table.loc[
                comparison_table["comparison_name"].eq("primary_vs_spy_buy_hold")
            ]
        ),
        "",
        _benchmark_markdown_sentence(primary, benchmark_rows, "SPY"),
        "",
        "## Primary vs QQQ Buy-and-Hold",
        "",
        _markdown_table(
            comparison_table.loc[
                comparison_table["comparison_name"].eq("primary_vs_qqq_buy_hold")
            ]
        ),
        "",
        _benchmark_markdown_sentence(primary, benchmark_rows, "QQQ"),
        "",
        "## Window Stability",
        "",
        "The stability table breaks the selected candidates down by "
        "`train_window_years`, `test_window_months`, and "
        "`validation_window_months` when present. Test-window stability is still "
        "weak in the current report, so this is a research baseline target rather "
        "than production evidence.",
        "",
        _markdown_table(stability_table),
        "",
        "## Cost Sensitivity",
        "",
        "The cost table compares the 10bps decision anchor against the 20bps "
        "sensitivity setting.",
        "",
        _markdown_table(cost_table),
        "",
        "## Failure / Weakness Section",
        "",
        "- Test-window stability is still weak in the current report.",
        "- ETF survivorship bias is not fully eliminated.",
        "- Tiingo-only results may not match Massive app data.",
        "- This is not production-ready.",
        "",
        "## Decision",
        "",
        _markdown_table(decision_table),
        "",
        "## Updated Decision",
        "",
        _updated_decision_sentence(primary_decision),
        "",
        "- Classification: `primary research baseline candidate`.",
        "- Return reference: `ridge / top1 / hold 20 / margin 0.005`.",
        "- Drawdown guardrail: `ridge / top5 / hold 5 / margin 0.002`.",
        "- Historical champion reference: `linear_regression / top1 / hold 10 / "
        "margin 0.002`.",
        "- Benchmark references: `SPY buy-and-hold`, `QQQ buy-and-hold`.",
        "",
        "## Next-Step Recommendation",
        "",
        f"Recommended next step: {recommended_next_step}",
        "",
        _next_step_markdown_sentence(primary_decision),
        "",
    ]
    if report_data.warnings:
        lines.extend(["## Input Warnings", ""])
        lines.extend(f"- {warning}" for warning in report_data.warnings)
        lines.append("")
    return "\n".join(lines)


def render_selected_candidate_model_card_draft(
    report_data: TiingoSelectedCandidateEvaluationData,
) -> str:
    """Render a lightweight research-only model-card draft for the primary candidate."""

    primary = _summary_row(report_data.candidate_summary, PRIMARY_ROLE)
    primary_decision = _decision_row(report_data.decision_matrix, PRIMARY_ROLE)
    rows = [
        ("Candidate", str(primary["candidate_label"])),
        ("Dataset source", report_data.dataset_source),
        ("Stage", "research candidate only; not production-ready"),
        ("Classification", str(primary["candidate_classification"])),
        ("Decision anchor", f"{DECISION_ANCHOR_BPS:g}bps"),
        ("20bps sensitivity return", _format_number(primary["average_test_return_20bps"])),
        ("Research decision", str(primary_decision["recommendation"])),
        ("Next action", str(primary_decision["next_action"])),
    ]
    metric_rows = [
        ("average_test_return_10bps", _format_number(primary["average_test_return_10bps"])),
        ("median_test_return_10bps", _format_number(primary["median_test_return_10bps"])),
        (
            "positive_window_rate_10bps",
            _format_number(primary["positive_window_rate_10bps"]),
        ),
        ("average_sharpe_like_10bps", _format_number(primary["average_sharpe_like_10bps"])),
        ("average_max_drawdown_10bps", _format_number(primary["average_max_drawdown_10bps"])),
        ("worst_max_drawdown_10bps", _format_number(primary["worst_max_drawdown_10bps"])),
        (
            "average_total_transaction_cost_10bps",
            _format_number(primary["average_total_transaction_cost_10bps"]),
        ),
        (
            "average_mean_daily_turnover_10bps",
            _format_number(primary["average_mean_daily_turnover_10bps"]),
        ),
    ]
    lines = [
        "# Selected Candidate Model Card Draft",
        "",
        "This is a draft for research review only. It does not mark the candidate "
        "as production-ready.",
        "",
        "## Overview",
        "",
        _two_column_markdown_table("Field", "Value", rows),
        "",
        "## Primary Metrics",
        "",
        _two_column_markdown_table("Metric", "Value", metric_rows),
        "",
        "## Caveats",
        "",
        "- Test-window stability is still weak.",
        "- ETF survivorship bias is not fully eliminated.",
        "- Tiingo-only results may not match Massive app data.",
        "- Production export and production model-card promotion remain out of scope.",
        "",
    ]
    return "\n".join(lines)


def _historical_markdown_sentence(
    primary: pd.Series,
    historical_champion: pd.Series | None,
) -> str:
    if historical_champion is None:
        return "Historical champion metrics were unavailable in the provided reference inputs."
    return_delta = _delta(primary, historical_champion, "average_test_return_10bps")
    drawdown_delta = _delta(primary, historical_champion, "worst_max_drawdown_10bps")
    turnover_delta = _delta(
        historical_champion,
        primary,
        "average_mean_daily_turnover_10bps",
    )
    return (
        "Primary vs historical champion: "
        f"average-return delta `{_format_number(return_delta)}`, "
        f"worst-drawdown delta `{_format_number(drawdown_delta)}`, "
        f"turnover improvement `{_format_number(turnover_delta)}`. "
        "Positive drawdown and turnover deltas favor the primary candidate."
    )


def _benchmark_markdown_sentence(
    primary: pd.Series,
    benchmark_rows: pd.DataFrame,
    symbol: str,
) -> str:
    matches = benchmark_rows.loc[
        benchmark_rows["benchmark_symbol"].astype(str).str.upper().eq(symbol.upper())
    ]
    if matches.empty:
        return f"{symbol.upper()} buy-and-hold metrics were unavailable."
    benchmark = matches.iloc[0]
    return_delta = _delta(primary, benchmark, "average_test_return_10bps")
    sharpe_delta = _delta(primary, benchmark, "average_sharpe_like_10bps")
    drawdown_delta = _delta(primary, benchmark, "worst_max_drawdown_10bps")
    if _valid_number(return_delta) and return_delta < -0.02:
        raw_return_note = f"{symbol.upper()} is materially stronger on raw return"
    elif _valid_number(return_delta) and return_delta >= 0.0:
        raw_return_note = "primary is stronger on raw return"
    else:
        raw_return_note = "primary is close on raw return"
    return (
        f"Primary vs {symbol.upper()} buy-and-hold: {raw_return_note}; "
        f"average-return delta `{_format_number(return_delta)}`, "
        f"Sharpe-like delta `{_format_number(sharpe_delta)}`, "
        f"worst-drawdown delta `{_format_number(drawdown_delta)}`."
    )


def _updated_decision_sentence(primary_decision: pd.Series) -> str:
    if str(primary_decision["recommendation"]) == "primary research baseline candidate":
        return (
            "After adding SPY/QQQ buy-and-hold and the historical champion, the "
            "ridge top5 hold20 margin0.005 candidate remains the current research "
            "baseline target to beat. This remains a research decision, not a "
            "production-ready designation."
        )
    return (
        "After adding SPY/QQQ buy-and-hold and the historical champion, the ridge "
        "top5 hold20 margin0.005 candidate needs further review before it can be "
        "treated as the current research baseline target to beat."
    )


def _next_step_markdown_sentence(primary_decision: pd.Series) -> str:
    if str(primary_decision["recommendation"]) == "primary research baseline candidate":
        return (
            "Do not run `tree_screen` immediately. Do not run MLP immediately. The "
            "ridge top5 primary is strong enough to become the current research "
            "baseline target to beat, while still needing model-card review and "
            "feature-engineering challengers before heavier model families."
        )
    return (
        "Do not run `tree_screen` immediately. Do not run MLP immediately. Because "
        "the benchmark gate is still a watch item, keep this as a selected research "
        "candidate and replay it with richer benchmark-aware artifacts before "
        "treating it as the baseline target to beat."
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo selected-candidate evaluation bundle."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--decision-matrix-csv",
        type=Path,
        default=DEFAULT_DECISION_MATRIX_CSV,
    )
    parser.add_argument(
        "--core-linear-results-csv",
        type=Path,
        default=DEFAULT_CORE_LINEAR_RESULTS_CSV,
    )
    parser.add_argument(
        "--core-linear-summary-csv",
        type=Path,
        default=DEFAULT_CORE_LINEAR_SUMMARY_CSV,
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
    parser.add_argument(
        "--skip-missing-references",
        action="store_true",
        help=(
            "Build the selected-candidate-only bundle when historical or benchmark "
            "reference inputs are missing, recording warnings instead of failing."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _build_historical_champion_reference_summary(
    *,
    core_linear_results_csv: Path,
    core_linear_summary_csv: Path,
    skip_missing_references: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    warnings: list[str] = []
    core_results = _read_required_reference_csv(
        core_linear_results_csv,
        label="core linear results CSV",
        skip_missing_references=skip_missing_references,
        warnings=warnings,
    )
    core_summary = _read_required_reference_csv(
        core_linear_summary_csv,
        label="core linear summary CSV",
        skip_missing_references=skip_missing_references,
        warnings=warnings,
    )
    if core_results is None or core_summary is None:
        return pd.DataFrame(), pd.DataFrame(columns=SUMMARY_COLUMNS), tuple(warnings)

    _validate_columns(core_results, REQUIRED_RESULTS_COLUMNS, path=core_linear_results_csv)
    test_rows = core_results.loc[core_results["split"].astype(str).eq("test")].copy()
    enriched = _enrich_test_rows(test_rows)
    historical_rows = _match_spec(enriched, HISTORICAL_CHAMPION_SPEC).loc[
        lambda frame: frame["model_family"].astype(str).eq("baseline")
        & _series_in_float_values(frame["transaction_cost_bps"], SENSITIVITY_COSTS_BPS)
    ].copy()
    if historical_rows.empty or historical_rows.loc[
        _float_eq(historical_rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)
    ].empty:
        message = (
            "Historical champion reference was not found at the 10bps decision anchor "
            f"in core linear results CSV {core_linear_results_csv}: "
            f"{HISTORICAL_CHAMPION_SPEC.label}."
        )
        if skip_missing_references:
            warnings.append(message)
            return pd.DataFrame(), pd.DataFrame(columns=SUMMARY_COLUMNS), tuple(warnings)
        raise ValueError(message)

    historical_summary = _build_candidate_summary(
        historical_rows,
        source_decision_matrix=core_summary,
        specs=(HISTORICAL_CHAMPION_SPEC,),
    )
    return historical_rows, historical_summary, tuple(warnings)


def _read_required_reference_csv(
    path: Path,
    *,
    label: str,
    skip_missing_references: bool,
    warnings: list[str],
) -> pd.DataFrame | None:
    resolved_path = Path(path)
    if resolved_path.exists():
        return pd.read_csv(resolved_path)
    message = (
        f"Required reference input is missing: {label} at {resolved_path}. "
        f"Pass --{label.replace(' ', '-').lower()} with a valid path"
        " or use --skip-missing-references to build without external references."
    )
    if skip_missing_references:
        warnings.append(message)
        return None
    raise FileNotFoundError(message)


def _build_benchmark_references(
    *,
    market_dataset_glob: str | Path,
    benchmark_symbols: Sequence[str],
    focused_test_rows: pd.DataFrame,
    skip_missing_references: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    warnings: list[str] = []
    if not benchmark_symbols:
        return (
            pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS),
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            ("No benchmark symbols were requested.",),
        )

    try:
        market_data = _read_benchmark_market_data(market_dataset_glob, benchmark_symbols)
    except (FileNotFoundError, ValueError) as exc:
        message = (
            f"Benchmark reference input could not be loaded from {market_dataset_glob}: "
            f"{exc}"
        )
        if skip_missing_references:
            warnings.append(message)
            return (
                pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS),
                pd.DataFrame(columns=SUMMARY_COLUMNS),
                tuple(warnings),
            )
        raise FileNotFoundError(message) from exc

    price_column = _adjusted_close_column(market_data)
    windows = _extract_benchmark_test_windows(focused_test_rows)
    benchmark_rows: list[dict[str, object]] = []
    for symbol in benchmark_symbols:
        symbol_rows = market_data.loc[
            market_data["symbol"].astype(str).str.upper().eq(symbol.upper())
        ].copy()
        if symbol_rows.empty:
            warnings.append(f"Benchmark symbol {symbol} was not present in market data.")
            continue
        for window in windows:
            metrics = _benchmark_metrics_for_window(
                symbol_rows=symbol_rows,
                symbol=symbol,
                price_column=price_column,
                window=window,
            )
            if metrics is None:
                warnings.append(
                    f"Benchmark {symbol} skipped fold {window.fold}: fewer than two "
                    f"adjusted-close observations between "
                    f"{window.test_start.date().isoformat()} and "
                    f"{window.test_end.date().isoformat()}."
                )
                continue
            benchmark_rows.append(metrics)

    benchmark_references = pd.DataFrame(benchmark_rows, columns=BENCHMARK_REFERENCE_COLUMNS)
    benchmark_summary = _build_benchmark_summary(benchmark_references, benchmark_symbols)
    return benchmark_references, benchmark_summary, tuple(warnings)


def _read_benchmark_market_data(
    market_dataset_glob: str | Path,
    benchmark_symbols: Sequence[str],
) -> pd.DataFrame:
    pattern = str(market_dataset_glob)
    matched_paths = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    file_paths = [path for path in matched_paths if path.is_file()]
    if not file_paths:
        raise FileNotFoundError(f"Market dataset glob matched no files: {pattern}")

    symbol_filtered_paths = [
        path
        for path in file_paths
        if _path_mentions_benchmark_symbol(path, benchmark_symbols)
    ]
    read_paths = symbol_filtered_paths if symbol_filtered_paths else file_paths
    parquet_paths = [path for path in read_paths if path.suffix.lower() == ".parquet"]
    if parquet_paths:
        return read_parquet_files(parquet_paths)

    csv_paths = [path for path in read_paths if path.suffix.lower() == ".csv"]
    if csv_paths:
        return read_csv_files(csv_paths)

    raise ValueError("Benchmark market data glob matched no supported .parquet or .csv files.")


def _path_mentions_benchmark_symbol(path: Path, benchmark_symbols: Sequence[str]) -> bool:
    parts = {part.upper() for part in path.parts}
    for symbol in benchmark_symbols:
        upper_symbol = symbol.upper()
        if upper_symbol in parts or f"SYMBOL={upper_symbol}" in parts:
            return True
    return False


def _adjusted_close_column(market_data: pd.DataFrame) -> str:
    for column in ["adj_close", "adjusted_close", "adjClose", "adjustedClose"]:
        if column in market_data.columns:
            return column
    raise ValueError(
        "Benchmark market data is missing an adjusted-close column; expected one of "
        "adj_close, adjusted_close, adjClose, adjustedClose."
    )


def _extract_benchmark_test_windows(rows: pd.DataFrame) -> list[BenchmarkTestWindow]:
    if rows.empty:
        return []
    windows: list[BenchmarkTestWindow] = []
    for _, row in rows.drop_duplicates(subset=["fold"], keep="first").iterrows():
        windows.append(_benchmark_test_window_from_row(row))
    return sorted(windows, key=lambda window: (window.test_start, window.fold))


def _benchmark_test_window_from_row(row: pd.Series) -> BenchmarkTestWindow:
    fold = str(row["fold"])
    match = _FOLD_WINDOW_RE.search(fold)
    if match is None:
        raise ValueError(
            "Focused confirmation fold names must include test start/end dates in "
            f"tw{{years}}y_v{{months}}m_t{{months}}m_q{{sequence}}_YYYYMMDD_YYYYMMDD format: {fold}"
        )
    parts = match.groupdict()
    test_start = pd.to_datetime(parts["test_start"], format="%Y%m%d")
    test_end = pd.to_datetime(parts["test_end"], format="%Y%m%d")
    if test_start > test_end:
        raise ValueError(f"Fold {fold!r} has test_start after test_end.")
    return BenchmarkTestWindow(
        fold=fold,
        train_window_years=float(parts["train_window_years"]),
        validation_window_months=float(parts["validation_window_months"]),
        test_window_months=float(parts["test_window_months"]),
        sequence_number=float(parts["sequence_number"]),
        test_start=test_start,
        test_end=test_end,
    )


def _benchmark_metrics_for_window(
    *,
    symbol_rows: pd.DataFrame,
    symbol: str,
    price_column: str,
    window: BenchmarkTestWindow,
) -> dict[str, object] | None:
    prices = symbol_rows[[price_column, "date"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices[price_column] = pd.to_numeric(prices[price_column], errors="coerce")
    mask = (prices["date"] >= window.test_start) & (prices["date"] <= window.test_end)
    window_prices = (
        prices.loc[mask]
        .dropna(subset=[price_column])
        .loc[lambda frame: frame[price_column] > 0.0]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(window_prices) < 2:
        return None

    daily_returns = window_prices[price_column].pct_change().fillna(0.0)
    equity = (1.0 + daily_returns).cumprod()
    final_equity = float(equity.iloc[-1])
    mean_daily_return = float(daily_returns.mean())
    daily_volatility = float(daily_returns.std())
    sharpe_like = (
        mean_daily_return / daily_volatility
        if daily_volatility > 0.0 and not np.isnan(daily_volatility)
        else np.nan
    )
    label = f"{symbol.upper()} buy-and-hold"
    return {
        "candidate_role": BENCHMARK_REFERENCE_ROLE,
        "candidate_classification": "benchmark buy-and-hold reference",
        "candidate_label": label,
        "dataset_source": DATASET_SOURCE_LABEL,
        "sweep_profile": "focused_confirmation_test_windows",
        "model_family": "benchmark",
        "model_type": "buy_and_hold",
        "allocation_mode": "buy_and_hold",
        "benchmark_symbol": symbol.upper(),
        "transaction_cost_bps": 0.0,
        "price_column": price_column,
        "fold": window.fold,
        "train_window_years": window.train_window_years,
        "validation_window_months": window.validation_window_months,
        "test_window_months": window.test_window_months,
        "sequence_number": window.sequence_number,
        "test_start": window.test_start.date().isoformat(),
        "test_end": window.test_end.date().isoformat(),
        "first_price_date": pd.Timestamp(window_prices["date"].iloc[0]).date().isoformat(),
        "last_price_date": pd.Timestamp(window_prices["date"].iloc[-1]).date().isoformat(),
        "observation_count": int(len(window_prices)),
        "total_return": final_equity - 1.0,
        "mean_daily_return": mean_daily_return,
        "daily_volatility": daily_volatility,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": calculate_max_drawdown(equity),
        "final_equity": final_equity,
        "total_transaction_cost": 0.0,
        "mean_daily_turnover": 0.0,
        "run_key": f"{symbol.upper()}:{window.fold}:buy_and_hold",
    }


def _build_benchmark_summary(
    benchmark_references: pd.DataFrame,
    benchmark_symbols: Sequence[str],
) -> pd.DataFrame:
    if benchmark_references.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for symbol in benchmark_symbols:
        symbol_rows = benchmark_references.loc[
            benchmark_references["benchmark_symbol"].astype(str).str.upper().eq(symbol.upper())
        ].copy()
        if symbol_rows.empty:
            continue
        metrics = _metrics_from_rows(symbol_rows)
        row = {
            "candidate_role": BENCHMARK_REFERENCE_ROLE,
            "candidate_classification": "benchmark buy-and-hold reference",
            "candidate_key": f"{BENCHMARK_REFERENCE_ROLE}:buy_and_hold:{symbol.upper()}",
            "candidate_label": f"{symbol.upper()} buy-and-hold",
            "dataset_source": DATASET_SOURCE_LABEL,
            "sweep_profile": "focused_confirmation_test_windows",
            "model_family": "benchmark",
            "model_type": "buy_and_hold",
            "top_k": np.nan,
            "allocation_mode": "buy_and_hold",
            "benchmark_symbol": symbol.upper(),
            "min_holding_days": np.nan,
            "switch_margin": np.nan,
            "decision_anchor_bps": 0.0,
            "transaction_cost_bps": 0.0,
            **_prefix_metrics(metrics, "10bps"),
            **_prefix_metrics(metrics, "20bps"),
            "cost_delta_20_vs_10": 0.0,
            "return_strength": _return_strength(metrics),
            "positive_window_strength": _positive_window_strength(metrics),
            "drawdown_control": _drawdown_control(metrics),
            "cost_resilience": "not_applicable",
            "evidence_scope": _evidence_scope(metrics["window_count"]),
            "source_decision_recommendation": "",
            "source_decision_rationale": "",
            "recommendation": "",
            "next_step": "",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _parse_benchmark_symbols(symbols: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        raw_parts = [symbols]
    else:
        raw_parts = [str(symbol) for symbol in symbols]
    parsed: list[str] = []
    for raw_part in raw_parts:
        for part in raw_part.split(","):
            symbol = part.strip().upper()
            if symbol and symbol not in parsed:
                parsed.append(symbol)
    return tuple(parsed)


def _build_candidate_summary(
    rows: pd.DataFrame,
    *,
    source_decision_matrix: pd.DataFrame,
    specs: Sequence[SelectedCandidateSpec],
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        spec_rows = _match_spec(rows, spec)
        anchor_rows = spec_rows.loc[
            _float_eq(spec_rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)
        ]
        sensitivity_rows = spec_rows.loc[
            _float_eq(spec_rows, "transaction_cost_bps", SENSITIVITY_BPS)
        ]
        if anchor_rows.empty:
            continue
        first = anchor_rows.iloc[0]
        metrics_10 = _metrics_from_rows(anchor_rows)
        metrics_20 = _metrics_from_rows(sensitivity_rows)
        cost_delta = _as_float(metrics_20["average_test_return"]) - _as_float(
            metrics_10["average_test_return"]
        )
        source_decision = _lookup_source_decision(source_decision_matrix, spec)
        row = {
            "candidate_role": spec.role,
            "candidate_classification": spec.classification,
            "candidate_key": spec.key,
            "candidate_label": spec.label,
            "dataset_source": DATASET_SOURCE_LABEL,
            "sweep_profile": str(first.get("sweep_profile", "unknown")),
            "model_family": str(first["model_family"]),
            "model_type": spec.model_type,
            "top_k": spec.top_k,
            "allocation_mode": str(first.get("allocation_mode", "unknown")),
            "benchmark_symbol": "",
            "min_holding_days": spec.min_holding_days,
            "switch_margin": spec.switch_margin,
            "decision_anchor_bps": DECISION_ANCHOR_BPS,
            "transaction_cost_bps": DECISION_ANCHOR_BPS,
            **_prefix_metrics(metrics_10, "10bps"),
            **_prefix_metrics(metrics_20, "20bps"),
            "cost_delta_20_vs_10": cost_delta,
            "return_strength": _return_strength(metrics_10),
            "positive_window_strength": _positive_window_strength(metrics_10),
            "drawdown_control": _drawdown_control(metrics_10),
            "cost_resilience": _cost_resilience(metrics_10, metrics_20, cost_delta),
            "evidence_scope": _evidence_scope(metrics_10["window_count"]),
            "source_decision_recommendation": source_decision.get("recommendation", ""),
            "source_decision_rationale": source_decision.get("rationale", ""),
            "recommendation": "",
            "next_step": "",
        }
        summary_rows.append(row)
    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def _build_decision_matrix(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    primary = _summary_row(candidate_summary, PRIMARY_ROLE)
    return_reference = _summary_row(candidate_summary, RETURN_REFERENCE_ROLE)
    drawdown_guardrail = _summary_row(candidate_summary, DRAWDOWN_GUARDRAIL_ROLE)
    historical_champion = _summary_row_optional(candidate_summary, HISTORICAL_CHAMPION_ROLE)
    benchmark_references = _benchmark_summary_rows(candidate_summary)
    drawdown_vs_top1 = _drawdown_vs_reference(primary, return_reference)
    cost_vs_guardrail = _cost_resilience_vs_guardrail(primary, drawdown_guardrail)
    historical_profile = _historical_champion_profile(primary, historical_champion)
    benchmark_profile = _benchmark_reference_profile(primary, benchmark_references)

    rows: list[dict[str, object]] = []
    for _, row in candidate_summary.iterrows():
        is_primary = str(row["candidate_role"]) == PRIMARY_ROLE
        return_gate = _gate(row["return_strength"] == "strong")
        positive_gate = _gate(row["positive_window_strength"] == "strong")
        cost_gate = _gate(str(row["cost_resilience"]) in {"strong", "moderate"})
        drawdown_gate = _gate(
            not is_primary or drawdown_vs_top1 in {"strong", "moderate"}
        )
        historical_gate = _gate(not is_primary or historical_profile == "strong")
        benchmark_gate = _gate(
            not is_primary or benchmark_profile in {"strong", "moderate"}
        )
        evidence_gate = _gate(str(row["evidence_scope"]) == "broad multi-window evidence")
        promote_primary = (
            is_primary
            and return_gate == "pass"
            and positive_gate == "pass"
            and cost_gate == "pass"
            and drawdown_gate == "pass"
            and historical_gate == "pass"
            and benchmark_gate == "pass"
            and evidence_gate == "pass"
        )
        recommendation = _candidate_decision_recommendation(
            role=str(row["candidate_role"]),
            promote_primary=promote_primary,
        )
        next_action = _candidate_next_action(
            role=str(row["candidate_role"]),
            promote_primary=promote_primary,
        )
        rows.append(
            {
                "candidate_role": row["candidate_role"],
                "candidate_classification": row["candidate_classification"],
                "candidate_label": row["candidate_label"],
                "decision_anchor_bps": row["decision_anchor_bps"],
                "average_test_return_gate": return_gate if is_primary else "reference",
                "positive_window_gate": positive_gate if is_primary else "reference",
                "cost_sensitivity_gate": cost_gate if is_primary else "reference",
                "drawdown_vs_top1_gate": drawdown_gate if is_primary else "reference",
                "historical_champion_gate": historical_gate if is_primary else "reference",
                "benchmark_reference_gate": benchmark_gate if is_primary else "reference",
                "evidence_scope_gate": evidence_gate if is_primary else "reference",
                "return_strength": row["return_strength"],
                "positive_window_strength": row["positive_window_strength"],
                "drawdown_control": row["drawdown_control"],
                "cost_resilience": row["cost_resilience"],
                "drawdown_vs_return_reference": drawdown_vs_top1 if is_primary else "reference",
                "cost_resilience_vs_drawdown_guardrail": (
                    cost_vs_guardrail if is_primary else "reference"
                ),
                "historical_champion_profile": (
                    historical_profile if is_primary else "reference"
                ),
                "benchmark_reference_profile": (
                    benchmark_profile if is_primary else "reference"
                ),
                "evidence_scope": row["evidence_scope"],
                "promote_primary_candidate": bool(promote_primary),
                "recommendation": recommendation,
                "next_action": next_action,
                "rationale": _decision_rationale(
                    row,
                    drawdown_vs_top1=drawdown_vs_top1 if is_primary else "reference",
                    cost_vs_guardrail=cost_vs_guardrail if is_primary else "reference",
                    historical_profile=historical_profile if is_primary else "reference",
                    benchmark_profile=benchmark_profile if is_primary else "reference",
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def _apply_decision_recommendations(
    candidate_summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    decisions = {
        (str(row["candidate_role"]), str(row["candidate_label"])): row
        for _, row in decision_matrix.iterrows()
    }
    output = candidate_summary.copy()
    output["recommendation"] = [
        decisions.get((str(row["candidate_role"]), str(row["candidate_label"])), {}).get(
            "recommendation",
            "",
        )
        for _, row in output.iterrows()
    ]
    output["next_step"] = [
        decisions.get((str(row["candidate_role"]), str(row["candidate_label"])), {}).get(
            "next_action",
            "",
        )
        for _, row in output.iterrows()
    ]
    return output[SUMMARY_COLUMNS]


def _build_cost_sensitivity(
    rows: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    *,
    specs: Sequence[SelectedCandidateSpec],
) -> pd.DataFrame:
    summary_by_role = candidate_summary.set_index("candidate_role")
    output_rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.role not in summary_by_role.index:
            continue
        spec_rows = _match_spec(rows, spec)
        if spec_rows.empty:
            continue
        anchor_metrics = _metrics_from_rows(
            spec_rows.loc[_float_eq(spec_rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)]
        )
        for cost_bps in SENSITIVITY_COSTS_BPS:
            cost_rows = spec_rows.loc[_float_eq(spec_rows, "transaction_cost_bps", cost_bps)]
            metrics = _metrics_from_rows(cost_rows)
            output_rows.append(
                {
                    "candidate_role": spec.role,
                    "candidate_label": str(summary_by_role.loc[spec.role, "candidate_label"]),
                    "transaction_cost_bps": cost_bps,
                    "is_decision_anchor": bool(np.isclose(cost_bps, DECISION_ANCHOR_BPS)),
                    "is_20bps_sensitivity": bool(np.isclose(cost_bps, SENSITIVITY_BPS)),
                    **metrics,
                    "return_delta_vs_10bps": _as_float(metrics["average_test_return"])
                    - _as_float(anchor_metrics["average_test_return"]),
                    "positive_rate_delta_vs_10bps": _as_float(metrics["positive_window_rate"])
                    - _as_float(anchor_metrics["positive_window_rate"]),
                    "worst_drawdown_delta_vs_10bps": _as_float(metrics["worst_max_drawdown"])
                    - _as_float(anchor_metrics["worst_max_drawdown"]),
                }
            )
    return pd.DataFrame(output_rows, columns=COST_SENSITIVITY_COLUMNS)


def _build_window_stability(
    rows: pd.DataFrame,
    *,
    specs: Sequence[SelectedCandidateSpec],
) -> pd.DataFrame:
    anchor = rows.loc[_float_eq(rows, "transaction_cost_bps", DECISION_ANCHOR_BPS)].copy()
    dimensions = ["train_window_years", "test_window_months"]
    if "validation_window_months" in anchor.columns:
        valid_validation = pd.to_numeric(
            anchor["validation_window_months"],
            errors="coerce",
        ).notna()
        if valid_validation.any():
            dimensions.append("validation_window_months")

    output_rows: list[dict[str, object]] = []
    for spec in specs:
        spec_rows = _match_spec(anchor, spec)
        if spec_rows.empty:
            continue
        for dimension in dimensions:
            dimension_rows: list[dict[str, object]] = []
            for value, group in spec_rows.groupby(dimension, dropna=False, sort=True):
                metrics = _metrics_from_rows(group)
                row = {
                    "candidate_role": spec.role,
                    "candidate_label": spec.label,
                    "decision_anchor_bps": DECISION_ANCHOR_BPS,
                    "stability_dimension": dimension,
                    "stability_value": value,
                    "train_window_years": (
                        value if dimension == "train_window_years" else np.nan
                    ),
                    "validation_window_months": (
                        value if dimension == "validation_window_months" else np.nan
                    ),
                    "test_window_months": (
                        value if dimension == "test_window_months" else np.nan
                    ),
                    **metrics,
                }
                dimension_rows.append(row)
            rating = _sensitivity_rating(pd.DataFrame(dimension_rows))
            for row in dimension_rows:
                row["stability_rating"] = rating
                output_rows.append(row)
    return pd.DataFrame(output_rows, columns=WINDOW_STABILITY_COLUMNS)


def _build_comparison(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    primary = _summary_row(candidate_summary, PRIMARY_ROLE)
    rows = [
        _comparison_row(
            comparison_name="primary_vs_return_reference",
            primary=primary,
            comparison=_summary_row(candidate_summary, RETURN_REFERENCE_ROLE),
            interpretation=(
                "Primary trades a small amount of raw return for materially better "
                "drawdown and lower turnover than the top1 return reference."
            ),
        ),
        _comparison_row(
            comparison_name="primary_vs_drawdown_guardrail",
            primary=primary,
            comparison=_summary_row(candidate_summary, DRAWDOWN_GUARDRAIL_ROLE),
            interpretation=(
                "Primary keeps stronger cost and turnover resilience than the shorter-hold "
                "drawdown guardrail while preserving the guardrail as a risk-control check."
            ),
        ),
    ]
    historical_champion = _summary_row_optional(candidate_summary, HISTORICAL_CHAMPION_ROLE)
    if historical_champion is not None:
        rows.append(
            _comparison_row(
                comparison_name="primary_vs_historical_champion",
                primary=primary,
                comparison=historical_champion,
                interpretation=_historical_comparison_interpretation(
                    primary,
                    historical_champion,
                ),
            )
        )
    for _, benchmark in _benchmark_summary_rows(candidate_summary).iterrows():
        symbol = str(benchmark["benchmark_symbol"]).lower()
        rows.append(
            _comparison_row(
                comparison_name=f"primary_vs_{symbol}_buy_hold",
                primary=primary,
                comparison=benchmark,
                interpretation=_benchmark_comparison_interpretation(primary, benchmark),
            )
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _comparison_row(
    *,
    comparison_name: str,
    primary: pd.Series,
    comparison: pd.Series,
    interpretation: str,
) -> dict[str, object]:
    return {
        "comparison_name": comparison_name,
        "primary_candidate_label": primary["candidate_label"],
        "comparison_candidate_label": comparison["candidate_label"],
        "decision_anchor_bps": DECISION_ANCHOR_BPS,
        "primary_average_test_return_10bps": primary["average_test_return_10bps"],
        "comparison_average_test_return_10bps": comparison["average_test_return_10bps"],
        "delta_average_test_return_10bps": _delta(
            primary,
            comparison,
            "average_test_return_10bps",
        ),
        "primary_positive_window_rate_10bps": primary["positive_window_rate_10bps"],
        "comparison_positive_window_rate_10bps": comparison["positive_window_rate_10bps"],
        "delta_positive_window_rate_10bps": _delta(
            primary,
            comparison,
            "positive_window_rate_10bps",
        ),
        "primary_average_max_drawdown_10bps": primary["average_max_drawdown_10bps"],
        "comparison_average_max_drawdown_10bps": comparison["average_max_drawdown_10bps"],
        "delta_average_max_drawdown_10bps": _delta(
            primary,
            comparison,
            "average_max_drawdown_10bps",
        ),
        "primary_worst_max_drawdown_10bps": primary["worst_max_drawdown_10bps"],
        "comparison_worst_max_drawdown_10bps": comparison["worst_max_drawdown_10bps"],
        "delta_worst_max_drawdown_10bps": _delta(
            primary,
            comparison,
            "worst_max_drawdown_10bps",
        ),
        "primary_average_test_return_20bps": primary["average_test_return_20bps"],
        "comparison_average_test_return_20bps": comparison["average_test_return_20bps"],
        "delta_average_test_return_20bps": _delta(
            primary,
            comparison,
            "average_test_return_20bps",
        ),
        "primary_cost_delta_20_vs_10": primary["cost_delta_20_vs_10"],
        "comparison_cost_delta_20_vs_10": comparison["cost_delta_20_vs_10"],
        "delta_cost_delta_20_vs_10": _delta(primary, comparison, "cost_delta_20_vs_10"),
        "primary_average_total_transaction_cost_10bps": primary[
            "average_total_transaction_cost_10bps"
        ],
        "comparison_average_total_transaction_cost_10bps": comparison[
            "average_total_transaction_cost_10bps"
        ],
        "delta_average_total_transaction_cost_10bps": _delta(
            primary,
            comparison,
            "average_total_transaction_cost_10bps",
        ),
        "primary_average_mean_daily_turnover_10bps": primary[
            "average_mean_daily_turnover_10bps"
        ],
        "comparison_average_mean_daily_turnover_10bps": comparison[
            "average_mean_daily_turnover_10bps"
        ],
        "delta_average_mean_daily_turnover_10bps": _delta(
            primary,
            comparison,
            "average_mean_daily_turnover_10bps",
        ),
        "interpretation": interpretation,
    }


def _primary_metric_table(primary: pd.Series) -> pd.DataFrame:
    rows = [
        "average_test_return_10bps",
        "median_test_return_10bps",
        "positive_window_rate_10bps",
        "average_sharpe_like_10bps",
        "average_max_drawdown_10bps",
        "worst_max_drawdown_10bps",
        "average_test_return_20bps",
        "positive_window_rate_20bps",
        "worst_max_drawdown_20bps",
        "cost_delta_20_vs_10",
        "average_total_transaction_cost_10bps",
        "average_mean_daily_turnover_10bps",
    ]
    return pd.DataFrame(
        [{"metric": metric, "value": primary[metric]} for metric in rows],
        columns=["metric", "value"],
    )


def _metrics_from_rows(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "window_count": 0,
            "average_test_return": float("nan"),
            "median_test_return": float("nan"),
            "positive_window_rate": float("nan"),
            "average_sharpe_like": float("nan"),
            "average_max_drawdown": float("nan"),
            "worst_max_drawdown": float("nan"),
            "average_total_transaction_cost": float("nan"),
            "average_mean_daily_turnover": float("nan"),
        }
    return {
        "window_count": int(rows["run_key"].nunique() if "run_key" in rows.columns else len(rows)),
        "average_test_return": _mean(rows, "total_return"),
        "median_test_return": _median(rows, "total_return"),
        "positive_window_rate": _positive_rate(rows["total_return"]),
        "average_sharpe_like": _mean(rows, "sharpe_like"),
        "average_max_drawdown": _mean(rows, "max_drawdown"),
        "worst_max_drawdown": _min(rows, "max_drawdown"),
        "average_total_transaction_cost": _mean(rows, "total_transaction_cost"),
        "average_mean_daily_turnover": _mean(rows, "mean_daily_turnover"),
    }


def _prefix_metrics(metrics: dict[str, object], suffix: str) -> dict[str, object]:
    return {f"{key}_{suffix}": value for key, value in metrics.items()}


def _return_strength(metrics: dict[str, object] | pd.Series) -> str:
    average_return = _as_float(metrics["average_test_return"])
    positive_rate = _as_float(metrics["positive_window_rate"])
    if pd.isna(average_return) or pd.isna(positive_rate):
        return "unknown"
    if average_return >= 0.06 and positive_rate >= 0.65:
        return "strong"
    if average_return > 0.0 and positive_rate >= 0.50:
        return "moderate"
    return "weak"


def _positive_window_strength(metrics: dict[str, object] | pd.Series) -> str:
    positive_rate = _as_float(metrics["positive_window_rate"])
    if pd.isna(positive_rate):
        return "unknown"
    if positive_rate >= 0.65:
        return "strong"
    if positive_rate >= 0.50:
        return "moderate"
    return "weak"


def _drawdown_control(metrics: dict[str, object] | pd.Series) -> str:
    worst = _as_float(metrics["worst_max_drawdown"])
    if pd.isna(worst):
        return "unknown"
    if worst >= -0.15:
        return "strong"
    if worst >= -0.30:
        return "moderate"
    return "weak"


def _cost_resilience(
    metrics_10: dict[str, object],
    metrics_20: dict[str, object],
    cost_delta: float,
) -> str:
    avg_10 = _as_float(metrics_10["average_test_return"])
    avg_20 = _as_float(metrics_20["average_test_return"])
    positive_20 = _as_float(metrics_20["positive_window_rate"])
    if pd.isna(avg_10) or pd.isna(avg_20) or pd.isna(positive_20) or pd.isna(cost_delta):
        return "unknown"
    if avg_20 > 0.0 and positive_20 >= 0.55 and cost_delta >= -0.04:
        return "strong"
    if avg_20 > 0.0 and positive_20 >= 0.45 and avg_10 > 0.0 and cost_delta >= -0.08:
        return "moderate"
    return "weak"


def _drawdown_vs_reference(primary: pd.Series, reference: pd.Series) -> str:
    worst_delta = _delta(primary, reference, "worst_max_drawdown_10bps")
    average_delta = _delta(primary, reference, "average_max_drawdown_10bps")
    if pd.isna(worst_delta) and pd.isna(average_delta):
        return "unknown"
    if worst_delta >= 0.03 or average_delta >= 0.01:
        return "strong"
    if worst_delta > 0.0 or average_delta > 0.0:
        return "moderate"
    return "weak"


def _cost_resilience_vs_guardrail(primary: pd.Series, guardrail: pd.Series) -> str:
    primary_cost_delta = _as_float(primary["cost_delta_20_vs_10"])
    guardrail_cost_delta = _as_float(guardrail["cost_delta_20_vs_10"])
    checks = [
        primary_cost_delta >= guardrail_cost_delta,
        _as_float(primary["average_total_transaction_cost_10bps"])
        <= _as_float(guardrail["average_total_transaction_cost_10bps"]),
        _as_float(primary["average_mean_daily_turnover_10bps"])
        <= _as_float(guardrail["average_mean_daily_turnover_10bps"]),
    ]
    valid_checks = [bool(check) for check in checks if not pd.isna(check)]
    if not valid_checks:
        return "unknown"
    passed = sum(valid_checks)
    if passed == len(valid_checks):
        return "strong"
    if passed >= 2:
        return "moderate"
    return "weak"


def _historical_champion_profile(
    primary: pd.Series,
    historical_champion: pd.Series | None,
) -> str:
    if historical_champion is None:
        return "unknown"
    worst_drawdown_delta = _delta(primary, historical_champion, "worst_max_drawdown_10bps")
    average_drawdown_delta = _delta(
        primary,
        historical_champion,
        "average_max_drawdown_10bps",
    )
    turnover_delta = _delta(
        historical_champion,
        primary,
        "average_mean_daily_turnover_10bps",
    )
    cost_delta = _delta(primary, historical_champion, "cost_delta_20_vs_10")
    drawdown_materially_better = (
        _valid_number(worst_drawdown_delta)
        and worst_drawdown_delta >= 0.03
    ) or (
        _valid_number(average_drawdown_delta)
        and average_drawdown_delta >= 0.01
    )
    turnover_better = _valid_number(turnover_delta) and turnover_delta >= 0.0
    cost_no_worse = not _valid_number(cost_delta) or cost_delta >= -0.01
    if drawdown_materially_better and turnover_better and cost_no_worse:
        return "strong"
    if drawdown_materially_better and turnover_better:
        return "moderate"
    return "weak"


def _benchmark_reference_profile(
    primary: pd.Series,
    benchmark_references: pd.DataFrame,
) -> str:
    if benchmark_references.empty:
        return "unknown"
    profiles = [
        _single_benchmark_profile(primary, row)
        for _, row in benchmark_references.iterrows()
    ]
    if profiles and all(profile == "strong" for profile in profiles):
        return "strong"
    if profiles and all(profile in {"strong", "moderate"} for profile in profiles):
        return "moderate"
    return "weak"


def _single_benchmark_profile(primary: pd.Series, benchmark: pd.Series) -> str:
    return_delta = _delta(primary, benchmark, "average_test_return_10bps")
    positive_delta = _delta(primary, benchmark, "positive_window_rate_10bps")
    sharpe_delta = _delta(primary, benchmark, "average_sharpe_like_10bps")
    worst_drawdown_delta = _delta(primary, benchmark, "worst_max_drawdown_10bps")
    average_drawdown_delta = _delta(primary, benchmark, "average_max_drawdown_10bps")
    benchmark_return = abs(_as_float(benchmark["average_test_return_10bps"]))
    raw_close_threshold = max(0.02, 0.25 * benchmark_return)
    raw_close_or_better = _valid_number(return_delta) and return_delta >= -raw_close_threshold
    risk_adjusted_close = (
        _valid_number(sharpe_delta)
        and sharpe_delta >= -0.02
    ) or (
        _valid_number(_as_float(primary["average_sharpe_like_10bps"]))
        and _valid_number(_as_float(benchmark["average_sharpe_like_10bps"]))
        and _as_float(primary["average_sharpe_like_10bps"])
        >= 0.8 * _as_float(benchmark["average_sharpe_like_10bps"])
    )
    drawdown_benefit = (
        _valid_number(worst_drawdown_delta)
        and worst_drawdown_delta >= 0.03
    ) or (
        _valid_number(average_drawdown_delta)
        and average_drawdown_delta >= 0.01
    )
    if (
        _valid_number(return_delta)
        and return_delta >= 0.02
        and (not _valid_number(positive_delta) or positive_delta >= -0.10)
    ):
        return "strong"
    if _valid_number(return_delta) and return_delta >= 0.0 and (
        risk_adjusted_close or drawdown_benefit
    ):
        return "strong"
    if raw_close_or_better and (risk_adjusted_close or drawdown_benefit):
        return "strong"
    if drawdown_benefit and (not _valid_number(positive_delta) or positive_delta >= -0.25):
        return "moderate"
    return "weak"


def _sensitivity_rating(table: pd.DataFrame) -> str:
    if table.empty or "average_test_return" not in table.columns:
        return "unknown"
    valid = table.loc[table["average_test_return"].notna()].copy()
    if len(valid) < 2:
        return "unknown"
    return_range = float(valid["average_test_return"].max() - valid["average_test_return"].min())
    min_positive = float(valid["positive_window_rate"].min())
    if return_range <= 0.025 and min_positive >= 0.55:
        return "strong"
    if return_range <= 0.07 and min_positive >= 0.45:
        return "moderate"
    return "weak"


def _evidence_scope(window_count: object) -> str:
    count = int(_as_float(window_count)) if not pd.isna(_as_float(window_count)) else 0
    if count <= 5:
        return "narrow evidence"
    if count < 12:
        return "moderate multi-window evidence"
    return "broad multi-window evidence"


def _candidate_decision_recommendation(*, role: str, promote_primary: bool) -> str:
    if role == PRIMARY_ROLE:
        if promote_primary:
            return "primary research baseline candidate"
        return "replay selected candidate with richer artifacts before promotion"
    if role == RETURN_REFERENCE_ROLE:
        return "return reference"
    if role == DRAWDOWN_GUARDRAIL_ROLE:
        return "drawdown guardrail"
    if role == HISTORICAL_CHAMPION_ROLE:
        return "historical fixed-fold return champion reference"
    if role == BENCHMARK_REFERENCE_ROLE:
        return "buy-and-hold benchmark reference"
    return "monitor only"


def _candidate_next_action(*, role: str, promote_primary: bool) -> str:
    if role == PRIMARY_ROLE:
        if promote_primary:
            return (
                "Create the research model-card draft and start feature-engineering "
                "experiments against this candidate as the baseline to beat; keep "
                "tree_screen and MLP deferred."
            )
        return (
            "Replay selected candidate with richer artifacts before running "
            "tree_screen or MLP."
        )
    if role == RETURN_REFERENCE_ROLE:
        return "Keep as the return reference; do not promote by raw return alone."
    if role == DRAWDOWN_GUARDRAIL_ROLE:
        return "Keep as the drawdown-control guardrail for future challenger checks."
    if role == HISTORICAL_CHAMPION_ROLE:
        return (
            "Keep visible as the historical fixed-fold / earlier return champion "
            "reference for continuity checks."
        )
    if role == BENCHMARK_REFERENCE_ROLE:
        return "Keep as an external buy-and-hold benchmark reference."
    return "Monitor only."


def _decision_rationale(
    row: pd.Series,
    *,
    drawdown_vs_top1: str,
    cost_vs_guardrail: str,
    historical_profile: str,
    benchmark_profile: str,
) -> str:
    return (
        f"return={row['return_strength']}; positive_windows="
        f"{row['positive_window_strength']}; drawdown={row['drawdown_control']}; "
        f"cost={row['cost_resilience']}; drawdown_vs_top1={drawdown_vs_top1}; "
        f"cost_vs_guardrail={cost_vs_guardrail}; "
        f"historical_champion={historical_profile}; benchmarks={benchmark_profile}; "
        f"evidence={row['evidence_scope']}"
    )


def _source_input_warnings(
    source_summary: pd.DataFrame,
    source_decision_matrix: pd.DataFrame,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if source_summary.empty:
        warnings.append("Focused confirmation summary CSV was empty.")
    if source_decision_matrix.empty:
        warnings.append("Focused confirmation decision matrix CSV was empty.")
    for label, frame in [
        ("summary", source_summary),
        ("decision matrix", source_decision_matrix),
    ]:
        required = {"model_type", "top_k", "min_holding_days", "switch_margin"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            warnings.append(f"Focused confirmation {label} is missing lookup columns: {missing}.")
    return tuple(warnings)


def _lookup_source_decision(
    source_decision_matrix: pd.DataFrame,
    spec: SelectedCandidateSpec,
) -> dict[str, object]:
    required = {"model_type", "top_k", "min_holding_days", "switch_margin"}
    if not required.issubset(source_decision_matrix.columns):
        return {}
    matches = source_decision_matrix.loc[
        source_decision_matrix["model_type"].astype(str).eq(spec.model_type)
        & _int_eq(source_decision_matrix, "top_k", spec.top_k)
        & _int_eq(source_decision_matrix, "min_holding_days", spec.min_holding_days)
        & _float_eq(source_decision_matrix, "switch_margin", spec.switch_margin)
    ].copy()
    if "decision_anchor_bps" in matches.columns:
        anchor_matches = matches.loc[
            _float_eq(matches, "decision_anchor_bps", DECISION_ANCHOR_BPS)
        ]
        if not anchor_matches.empty:
            matches = anchor_matches
    elif "transaction_cost_bps" in matches.columns:
        anchor_matches = matches.loc[
            _float_eq(matches, "transaction_cost_bps", DECISION_ANCHOR_BPS)
        ]
        if not anchor_matches.empty:
            matches = anchor_matches
    if matches.empty:
        return {}
    first = matches.iloc[0]
    return {
        "recommendation": first.get("recommendation", ""),
        "rationale": first.get("rationale", ""),
    }


def _is_selected_candidate_spec(rows: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=rows.index)
    for spec in SELECTED_CANDIDATE_SPECS:
        mask |= (
            rows["model_type"].astype(str).eq(spec.model_type)
            & _int_eq(rows, "top_k", spec.top_k)
            & _int_eq(rows, "min_holding_days", spec.min_holding_days)
            & _float_eq(rows, "switch_margin", spec.switch_margin)
        )
    return mask


def _match_spec(rows: pd.DataFrame, spec: SelectedCandidateSpec) -> pd.DataFrame:
    return rows.loc[
        rows["model_type"].astype(str).eq(spec.model_type)
        & _int_eq(rows, "top_k", spec.top_k)
        & _int_eq(rows, "min_holding_days", spec.min_holding_days)
        & _float_eq(rows, "switch_margin", spec.switch_margin)
    ].copy()


def _summary_row(summary: pd.DataFrame, role: str) -> pd.Series:
    matches = summary.loc[summary["candidate_role"].astype(str).eq(role)]
    if matches.empty:
        raise ValueError(f"Missing selected-candidate summary row for role: {role}")
    return matches.iloc[0]


def _summary_row_optional(summary: pd.DataFrame, role: str) -> pd.Series | None:
    matches = summary.loc[summary["candidate_role"].astype(str).eq(role)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _benchmark_summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    matches = summary.loc[
        summary["candidate_role"].astype(str).eq(BENCHMARK_REFERENCE_ROLE)
    ].copy()
    if matches.empty:
        return pd.DataFrame(columns=summary.columns)
    return matches.reset_index(drop=True)


def _decision_row(decision: pd.DataFrame, role: str) -> pd.Series:
    matches = decision.loc[decision["candidate_role"].astype(str).eq(role)]
    if matches.empty:
        raise ValueError(f"Missing selected-candidate decision row for role: {role}")
    return matches.iloc[0]


def _delta(left: pd.Series, right: pd.Series, column: str) -> float:
    return _as_float(left[column]) - _as_float(right[column])


def _historical_comparison_interpretation(
    primary: pd.Series,
    historical_champion: pd.Series,
) -> str:
    drawdown_delta = _delta(primary, historical_champion, "worst_max_drawdown_10bps")
    turnover_delta = _delta(
        historical_champion,
        primary,
        "average_mean_daily_turnover_10bps",
    )
    return_delta = _delta(primary, historical_champion, "average_test_return_10bps")
    pieces: list[str] = []
    if _valid_number(return_delta) and return_delta < 0.0:
        pieces.append("Historical champion is stronger on average raw return.")
    elif _valid_number(return_delta):
        pieces.append("Primary is stronger on average raw return.")
    if _valid_number(drawdown_delta) and drawdown_delta > 0.0:
        pieces.append("Primary improves worst drawdown versus the historical top1 reference.")
    if _valid_number(turnover_delta) and turnover_delta >= 0.0:
        pieces.append("Primary carries lower observed turnover.")
    return " ".join(pieces) or "Comparison is limited by missing historical metrics."


def _benchmark_comparison_interpretation(
    primary: pd.Series,
    benchmark: pd.Series,
) -> str:
    symbol = str(benchmark["benchmark_symbol"])
    return_delta = _delta(primary, benchmark, "average_test_return_10bps")
    drawdown_delta = _delta(primary, benchmark, "worst_max_drawdown_10bps")
    sharpe_delta = _delta(primary, benchmark, "average_sharpe_like_10bps")
    pieces: list[str] = []
    if _valid_number(return_delta) and return_delta < -0.02:
        pieces.append(f"{symbol} buy-and-hold is materially stronger on raw return.")
    elif _valid_number(return_delta) and return_delta >= 0.0:
        pieces.append(f"Primary beats {symbol} buy-and-hold on average raw return.")
    elif _valid_number(return_delta):
        pieces.append(f"Primary is close to {symbol} buy-and-hold on average raw return.")
    if _valid_number(sharpe_delta) and sharpe_delta >= 0.0:
        pieces.append("Primary is no worse on the daily Sharpe-like metric.")
    if _valid_number(drawdown_delta) and drawdown_delta > 0.0:
        pieces.append("Primary has the better worst drawdown.")
    elif _valid_number(drawdown_delta):
        pieces.append("Benchmark has the better worst drawdown.")
    return " ".join(pieces) or f"{symbol} benchmark comparison has incomplete metrics."


def _gate(value: bool) -> str:
    return "pass" if value else "watch"


def _series_in_float_values(series: pd.Series, values: Sequence[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    for value in values:
        mask |= pd.Series(np.isclose(numeric, value, atol=1e-9), index=series.index)
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


def _format_number(value: object) -> str:
    numeric = _as_float(value)
    if pd.isna(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _valid_number(value: object) -> bool:
    return not pd.isna(_as_float(value))


def _two_column_markdown_table(
    left_header: str,
    right_header: str,
    rows: Sequence[tuple[object, object]],
) -> str:
    table_rows = [
        [left_header, right_header],
        ["---", "---"],
        *[[str(left), str(right)] for left, right in rows],
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


create_tiingo_selected_candidate_evaluation_bundle_from_args = (
    run_tiingo_selected_candidate_evaluation_bundle_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_selected_candidate_evaluation_bundle_from_args())
