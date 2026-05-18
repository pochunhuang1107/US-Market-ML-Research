"""Create the Tiingo selected-candidate evaluation bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_selected_candidate_evaluation")

SUMMARY_PATH_NAME = "selected_candidate_summary.csv"
DECISION_MATRIX_PATH_NAME = "selected_candidate_decision_matrix.csv"
WINDOW_STABILITY_PATH_NAME = "selected_candidate_window_stability.csv"
COST_SENSITIVITY_PATH_NAME = "selected_candidate_cost_sensitivity.csv"
COMPARISON_PATH_NAME = "selected_candidate_comparison.csv"
REPORT_PATH_NAME = "selected_candidate_evaluation_report.md"
MODEL_CARD_DRAFT_PATH_NAME = "selected_candidate_model_card_draft.md"
OUTPUT_PATH_NAMES = (
    SUMMARY_PATH_NAME,
    DECISION_MATRIX_PATH_NAME,
    WINDOW_STABILITY_PATH_NAME,
    COST_SENSITIVITY_PATH_NAME,
    COMPARISON_PATH_NAME,
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
    "min_holding_days",
    "switch_margin",
    "decision_anchor_bps",
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
    "evidence_scope_gate",
    "return_strength",
    "positive_window_strength",
    "drawdown_control",
    "cost_resilience",
    "drawdown_vs_return_reference",
    "cost_resilience_vs_drawdown_guardrail",
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

PRIMARY_ROLE = "primary_balanced_candidate"
RETURN_REFERENCE_ROLE = "return_reference"
DRAWDOWN_GUARDRAIL_ROLE = "drawdown_guardrail"


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
    warnings: tuple[str, ...]
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    source_summary_row_count: int
    source_decision_matrix_row_count: int
    dataset_source: str = DATASET_SOURCE_LABEL


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


def run_tiingo_selected_candidate_evaluation_bundle_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the selected-candidate evaluation bundle."""

    args = _parse_args(argv)
    report_data = build_tiingo_selected_candidate_evaluation_bundle(
        results_csv=args.results_csv,
        summary_csv=args.summary_csv,
        decision_matrix_csv=args.decision_matrix_csv,
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
) -> TiingoSelectedCandidateEvaluationData:
    """Read focused confirmation outputs and build selected-candidate tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    source_summary = pd.read_csv(summary_csv)
    source_decision_matrix = pd.read_csv(decision_matrix_csv)

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
    candidate_summary = _build_candidate_summary(
        included,
        source_decision_matrix=source_decision_matrix,
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

    cost_sensitivity = _build_cost_sensitivity(included, candidate_summary)
    window_stability = _build_window_stability(included)
    comparison = _build_comparison(candidate_summary)
    decision_matrix = _build_decision_matrix(candidate_summary)
    candidate_summary = _apply_decision_recommendations(candidate_summary, decision_matrix)

    return TiingoSelectedCandidateEvaluationData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        window_stability=window_stability,
        cost_sensitivity=cost_sensitivity,
        comparison=comparison,
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
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
        MODEL_CARD_DRAFT_PATH_NAME: output_dir / MODEL_CARD_DRAFT_PATH_NAME,
    }
    report_data.candidate_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    report_data.window_stability.to_csv(outputs[WINDOW_STABILITY_PATH_NAME], index=False)
    report_data.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_PATH_NAME], index=False)
    report_data.comparison.to_csv(outputs[COMPARISON_PATH_NAME], index=False)
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
    primary_decision = _decision_row(report_data.decision_matrix, PRIMARY_ROLE)
    recommended_next_step = str(primary_decision["next_action"])
    window_count = int(primary["window_count_10bps"])

    primary_metrics = _primary_metric_table(primary)
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
        "- Classification: `primary research baseline candidate`.",
        "- Return reference: `ridge / top1 / hold 20 / margin 0.005`.",
        "- Drawdown guardrail: `ridge / top5 / hold 5 / margin 0.002`.",
        "",
        "## Next-Step Recommendation",
        "",
        f"Recommended next step: {recommended_next_step}",
        "",
        "Do not run `tree_screen` immediately. Do not run MLP immediately. The "
        "ridge top5 primary is strong enough to become the current research "
        "baseline target to beat, while still needing model-card review and "
        "feature-engineering challengers before heavier model families.",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _build_candidate_summary(
    rows: pd.DataFrame,
    *,
    source_decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for spec in SELECTED_CANDIDATE_SPECS:
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
            "min_holding_days": spec.min_holding_days,
            "switch_margin": spec.switch_margin,
            "decision_anchor_bps": DECISION_ANCHOR_BPS,
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
    drawdown_vs_top1 = _drawdown_vs_reference(primary, return_reference)
    cost_vs_guardrail = _cost_resilience_vs_guardrail(primary, drawdown_guardrail)

    rows: list[dict[str, object]] = []
    for _, row in candidate_summary.iterrows():
        is_primary = str(row["candidate_role"]) == PRIMARY_ROLE
        return_gate = _gate(row["return_strength"] == "strong")
        positive_gate = _gate(row["positive_window_strength"] == "strong")
        cost_gate = _gate(str(row["cost_resilience"]) in {"strong", "moderate"})
        drawdown_gate = _gate(
            not is_primary or drawdown_vs_top1 in {"strong", "moderate"}
        )
        evidence_gate = _gate(str(row["evidence_scope"]) == "broad multi-window evidence")
        promote_primary = (
            is_primary
            and return_gate == "pass"
            and positive_gate == "pass"
            and cost_gate == "pass"
            and drawdown_gate == "pass"
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
                "decision_anchor_bps": DECISION_ANCHOR_BPS,
                "average_test_return_gate": return_gate if is_primary else "reference",
                "positive_window_gate": positive_gate if is_primary else "reference",
                "cost_sensitivity_gate": cost_gate if is_primary else "reference",
                "drawdown_vs_top1_gate": drawdown_gate if is_primary else "reference",
                "evidence_scope_gate": evidence_gate if is_primary else "reference",
                "return_strength": row["return_strength"],
                "positive_window_strength": row["positive_window_strength"],
                "drawdown_control": row["drawdown_control"],
                "cost_resilience": row["cost_resilience"],
                "drawdown_vs_return_reference": drawdown_vs_top1 if is_primary else "reference",
                "cost_resilience_vs_drawdown_guardrail": (
                    cost_vs_guardrail if is_primary else "reference"
                ),
                "evidence_scope": row["evidence_scope"],
                "promote_primary_candidate": bool(promote_primary),
                "recommendation": recommendation,
                "next_action": next_action,
                "rationale": _decision_rationale(
                    row,
                    drawdown_vs_top1=drawdown_vs_top1 if is_primary else "reference",
                    cost_vs_guardrail=cost_vs_guardrail if is_primary else "reference",
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def _apply_decision_recommendations(
    candidate_summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    decisions = decision_matrix.set_index("candidate_role")
    output = candidate_summary.copy()
    output["recommendation"] = output["candidate_role"].map(decisions["recommendation"])
    output["next_step"] = output["candidate_role"].map(decisions["next_action"])
    return output[SUMMARY_COLUMNS]


def _build_cost_sensitivity(
    rows: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary_by_role = candidate_summary.set_index("candidate_role")
    output_rows: list[dict[str, object]] = []
    for spec in SELECTED_CANDIDATE_SPECS:
        spec_rows = _match_spec(rows, spec)
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


def _build_window_stability(rows: pd.DataFrame) -> pd.DataFrame:
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
    for spec in SELECTED_CANDIDATE_SPECS:
        spec_rows = _match_spec(anchor, spec)
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
    return "Monitor only."


def _decision_rationale(
    row: pd.Series,
    *,
    drawdown_vs_top1: str,
    cost_vs_guardrail: str,
) -> str:
    return (
        f"return={row['return_strength']}; positive_windows="
        f"{row['positive_window_strength']}; drawdown={row['drawdown_control']}; "
        f"cost={row['cost_resilience']}; drawdown_vs_top1={drawdown_vs_top1}; "
        f"cost_vs_guardrail={cost_vs_guardrail}; evidence={row['evidence_scope']}"
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


def _decision_row(decision: pd.DataFrame, role: str) -> pd.Series:
    matches = decision.loc[decision["candidate_role"].astype(str).eq(role)]
    if matches.empty:
        raise ValueError(f"Missing selected-candidate decision row for role: {role}")
    return matches.iloc[0]


def _delta(left: pd.Series, right: pd.Series, column: str) -> float:
    return _as_float(left[column]) - _as_float(right[column])


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
