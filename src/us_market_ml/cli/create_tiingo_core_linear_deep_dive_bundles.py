"""Create Tiingo core-linear candidate deep-dive bundles from a completed sweep."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_modern_baseline_sweep_report import (
    DATASET_SOURCE_LABEL,
    REQUIRED_RESULTS_COLUMNS,
    SUMMARY_GROUP_COLUMNS,
    _as_float,
    _enrich_test_rows,
    _format_for_markdown,
    _markdown_table,
    _positive_rate,
    _validate_columns,
)

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_modern_baseline_sweep/"
    "tiingo_modern_baseline_sweep_core_linear_results.csv"
)
DEFAULT_SUMMARY_CSV = Path(
    "artifacts/reports/tiingo_modern_baseline_sweep_core_linear/"
    "tiingo_modern_baseline_summary.csv"
)
DEFAULT_DECISION_MATRIX_CSV = Path(
    "artifacts/reports/tiingo_modern_baseline_sweep_core_linear/"
    "tiingo_modern_baseline_decision_matrix.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_core_linear_deep_dive_bundles")

SUMMARY_PATH_NAME = "tiingo_core_linear_deep_dive_summary.csv"
DECISION_MATRIX_PATH_NAME = "tiingo_core_linear_deep_dive_decision_matrix.csv"
REPORT_PATH_NAME = "tiingo_core_linear_deep_dive_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

DECISION_ANCHOR_BPS = 10.0
COST_SENSITIVITY_BPS = (5.0, 10.0, 20.0)
COMMON_ALLOCATION_MODE = "equal_weight"
CORE_LINEAR_PROFILE = "core_linear"

SUMMARY_COLUMNS = [
    "candidate_key",
    "candidate_id",
    "candidate_label",
    "selection_groups",
    "selection_reasons",
    "dataset_source",
    "sweep_profile",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "decision_anchor_bps",
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
    "cost_sensitivity_rating",
    "smoothing_sensitivity_rating",
    "window_stability_rating",
    "evidence_scope",
    "baseline_sweep_recommendation",
    "baseline_decision_rationale",
    "recommendation_label",
    "recommendation_rationale",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_key",
    "candidate_label",
    "selection_groups",
    "model_type",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "window_count",
    "average_test_return",
    "positive_window_rate",
    "average_sharpe_like",
    "worst_max_drawdown",
    "return_strength",
    "positive_window_strength",
    "drawdown_control",
    "cost_sensitivity_rating",
    "smoothing_sensitivity_rating",
    "window_stability_rating",
    "evidence_scope",
    "balanced_score",
    "recommendation_label",
    "next_action",
    "rationale",
]

FOLD_METADATA_COLUMNS = [
    "candidate_key",
    "candidate_label",
    "fold",
    "sequence_number",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "test_start",
    "test_end",
]

FOLD_RESULT_COLUMNS = [
    "candidate_key",
    "candidate_label",
    "fold",
    "sequence_number",
    "test_start",
    "test_end",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "total_return",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "run_key",
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
class CandidateTemplate:
    """Static shortlist candidate requested for the Tiingo core-linear deep dive."""

    selection_group: str
    selection_reason: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass
class CandidateSelection:
    """One unique 10bps candidate selected for the deep-dive shortlist."""

    sort_order: int
    primary_group: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float
    transaction_cost_bps: float
    train_window_years: int
    validation_window_months: int
    test_window_months: int
    selection_groups: list[str] = field(default_factory=list)
    selection_reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int, float, float, int, int, int]:
        return (
            self.model_type,
            self.top_k,
            self.min_holding_days,
            round(self.switch_margin, 9),
            round(self.transaction_cost_bps, 9),
            self.train_window_years,
            self.validation_window_months,
            self.test_window_months,
        )


@dataclass(frozen=True)
class CandidateBundle:
    """Prepared per-candidate diagnostic bundle tables."""

    selection: CandidateSelection
    candidate_key: str
    candidate_id: str
    candidate_label: str
    candidate_summary: pd.DataFrame
    fold_results: pd.DataFrame
    window_metadata: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    smoothing_sensitivity: pd.DataFrame
    window_stability: pd.DataFrame


@dataclass(frozen=True)
class TiingoCoreLinearDeepDiveReportData:
    """All tables and metadata for the Tiingo core-linear deep-dive bundles."""

    candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    bundles: tuple[CandidateBundle, ...]
    warnings: tuple[str, ...]
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    source_summary_row_count: int | None
    source_decision_matrix_row_count: int | None
    dataset_source: str = DATASET_SOURCE_LABEL


EXPLICIT_CANDIDATES = (
    CandidateTemplate(
        selection_group="return leader",
        selection_reason=(
            "Highest requested raw-return ridge top1 reference at the 10bps anchor."
        ),
        model_type="ridge",
        top_k=1,
        min_holding_days=20,
        switch_margin=0.005,
        train_window_years=5,
        validation_window_months=12,
        test_window_months=12,
    ),
    CandidateTemplate(
        selection_group="linear return reference",
        selection_reason=(
            "Linear top1 return reference for separating model-family effects from top_k."
        ),
        model_type="linear_regression",
        top_k=1,
        min_holding_days=20,
        switch_margin=0.002,
        train_window_years=5,
        validation_window_months=12,
        test_window_months=12,
    ),
    CandidateTemplate(
        selection_group="balanced reference",
        selection_reason="Ridge top3 diversified return and balanced-risk reference.",
        model_type="ridge",
        top_k=3,
        min_holding_days=10,
        switch_margin=0.0,
        train_window_years=5,
        validation_window_months=12,
        test_window_months=12,
    ),
    CandidateTemplate(
        selection_group="balanced reference",
        selection_reason="Ridge top5 diversified baseline direction.",
        model_type="ridge",
        top_k=5,
        min_holding_days=10,
        switch_margin=0.0,
        train_window_years=5,
        validation_window_months=12,
        test_window_months=12,
    ),
    CandidateTemplate(
        selection_group="balanced reference",
        selection_reason="Ridge top5 longer-hold diversified reference.",
        model_type="ridge",
        top_k=5,
        min_holding_days=20,
        switch_margin=0.005,
        train_window_years=5,
        validation_window_months=12,
        test_window_months=12,
    ),
    CandidateTemplate(
        selection_group="drawdown-control reference",
        selection_reason="Linear top5 drawdown-sensitive reference from the 3-month test window.",
        model_type="linear_regression",
        top_k=5,
        min_holding_days=10,
        switch_margin=0.005,
        train_window_years=3,
        validation_window_months=12,
        test_window_months=3,
    ),
    CandidateTemplate(
        selection_group="drawdown-control reference",
        selection_reason="Ridge top5 drawdown-sensitive reference from the 3-month test window.",
        model_type="ridge",
        top_k=5,
        min_holding_days=5,
        switch_margin=0.002,
        train_window_years=3,
        validation_window_months=12,
        test_window_months=3,
    ),
)


def run_tiingo_core_linear_deep_dive_bundles_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write Tiingo core-linear deep-dive bundles."""

    args = _parse_args(argv)
    report_data = build_tiingo_core_linear_deep_dive_bundles(
        results_csv=args.results_csv,
        summary_csv=args.summary_csv,
        decision_matrix_csv=args.decision_matrix_csv,
    )
    outputs = write_tiingo_core_linear_deep_dive_bundles(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo core-linear deep-dive bundles")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"decision anchor: {DECISION_ANCHOR_BPS:g}bps")
    print(f"results CSV: {args.results_csv}")
    print(f"output dir: {args.output_dir}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    for bundle in report_data.bundles:
        print(f"candidate bundle: {args.output_dir / bundle.candidate_key}")
    if report_data.warnings:
        print()
        print("Warnings:")
        for warning in report_data.warnings:
            print(f"- {warning}")
    print()
    print(report_data.candidate_summary.to_string(index=False))
    return 0


def build_tiingo_core_linear_deep_dive_bundles(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    summary_csv: Path | None = DEFAULT_SUMMARY_CSV,
    decision_matrix_csv: Path | None = DEFAULT_DECISION_MATRIX_CSV,
) -> TiingoCoreLinearDeepDiveReportData:
    """Read completed sweep outputs and build overall plus per-candidate tables."""

    warnings: list[str] = []
    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)

    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _add_fold_dates(_enrich_test_rows(test_rows))
    included = enriched.loc[
        enriched["model_family"].astype(str).eq("baseline")
        & enriched["sweep_profile"].astype(str).eq(CORE_LINEAR_PROFILE)
    ].copy()
    if included.empty:
        raise ValueError("No baseline/core_linear test rows were found in the results CSV.")

    source_summary = _read_optional_csv(summary_csv, label="summary", warnings=warnings)
    source_decision_matrix = _read_optional_csv(
        decision_matrix_csv,
        label="decision matrix",
        warnings=warnings,
    )
    all_candidate_metrics = _build_candidate_metrics(included)
    selections = _build_candidate_selections(all_candidate_metrics)

    bundles: list[CandidateBundle] = []
    summary_rows: list[dict[str, object]] = []
    missing_required: list[str] = []

    for selection in selections:
        candidate_rows = _match_candidate_rows(included, selection)
        if candidate_rows.empty:
            missing_required.append(_selection_label(selection))
            continue
        bundle = _build_candidate_bundle(
            selection=selection,
            candidate_rows=candidate_rows,
            all_test_rows=included,
            source_summary=source_summary,
            source_decision_matrix=source_decision_matrix,
        )
        bundles.append(bundle)
        summary_rows.append(bundle.candidate_summary.iloc[0].to_dict())

    if missing_required:
        raise ValueError(
            "Requested Tiingo core-linear candidates were not found in the results CSV: "
            + "; ".join(missing_required)
        )

    candidate_summary = pd.DataFrame(summary_rows)
    candidate_summary = _sort_shortlist(candidate_summary)
    decision_matrix = _build_decision_matrix(candidate_summary)

    return TiingoCoreLinearDeepDiveReportData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        bundles=tuple(
            sorted(
                bundles,
                key=lambda bundle: (
                    int(bundle.candidate_summary.iloc[0]["sort_order"])
                    if "sort_order" in bundle.candidate_summary.columns
                    else bundle.selection.sort_order
                ),
            )
        ),
        warnings=tuple(warnings),
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
        source_summary_row_count=None if source_summary is None else int(len(source_summary)),
        source_decision_matrix_row_count=(
            None if source_decision_matrix is None else int(len(source_decision_matrix))
        ),
    )


def write_tiingo_core_linear_deep_dive_bundles(
    *,
    report_data: TiingoCoreLinearDeepDiveReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the overall report and all per-candidate bundles."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.candidate_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_core_linear_deep_dive_markdown(report_data),
        encoding="utf-8",
    )

    for bundle in report_data.bundles:
        _write_candidate_bundle(bundle=bundle, output_dir=output_dir / bundle.candidate_key)

    return outputs


def render_tiingo_core_linear_deep_dive_markdown(
    report_data: TiingoCoreLinearDeepDiveReportData,
) -> str:
    """Render the overall Tiingo core-linear deep-dive Markdown report."""

    summary = report_data.candidate_summary
    decision = report_data.decision_matrix
    return_leaders = summary.loc[
        summary["selection_groups"].str.contains("return", case=False, regex=False)
    ]
    drawdown = summary.loc[
        summary["selection_groups"].str.contains("drawdown-control", regex=False)
    ]
    topk_table = _topk_comparison_table(summary)
    model_table = _model_comparison_table(summary)
    cost_table = _combined_bundle_table(
        report_data.bundles,
        table_name="cost_sensitivity",
        max_rows=40,
    )
    smoothing_table = _combined_bundle_table(
        report_data.bundles,
        table_name="smoothing_sensitivity",
        max_rows=40,
    )
    window_table = _combined_bundle_table(
        report_data.bundles,
        table_name="window_stability",
        max_rows=40,
    )

    best_return = summary.sort_values(
        by=["average_test_return", "positive_window_rate", "worst_max_drawdown"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    balanced_decision = decision.loc[
        decision["recommendation_label"].eq("keep as balanced reference")
    ].copy()
    if balanced_decision.empty:
        balanced_decision = decision
    best_balanced = balanced_decision.sort_values(
        by=["balanced_score", "average_test_return"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]

    lines = [
        "# Tiingo Core-Linear Deep-Dive Bundles",
        "",
        "## Executive Summary",
        "",
        "This report deep-dives selected Tiingo-only `core_linear` baseline "
        "candidates from the completed sweep. The decision anchor is "
        f"`{DECISION_ANCHOR_BPS:g}bps`; all shortlist rows are evaluated at that "
        "transaction-cost setting, with 5/10/20bps cost sensitivity shown separately.",
        "",
        f"Input rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included baseline/core_linear test rows: "
        f"`{report_data.included_test_row_count}`.",
        "",
        f"Highest average return in this shortlist: `{best_return['candidate_label']}`. "
        "It is kept as a return reference, not automatically promoted.",
        "",
        f"Best balanced-reference score in this shortlist: "
        f"`{best_balanced['candidate_label']}`.",
        "",
        "The practical next step is a narrower linear/ridge confirmation around the "
        "ridge top5 and drawdown-control directions before spending cycles on "
        "`tree_screen` or MLP.",
        "",
        "## Why These Candidates Were Selected",
        "",
        _markdown_table(
            _format_for_markdown(
                summary[
                    [
                        "candidate_label",
                        "selection_groups",
                        "selection_reasons",
                        "average_test_return",
                        "positive_window_rate",
                        "worst_max_drawdown",
                        "recommendation_label",
                    ]
                ],
                exclude={
                    "candidate_label",
                    "selection_groups",
                    "selection_reasons",
                    "recommendation_label",
                },
            )
        ),
        "",
        "## Return Leader Analysis",
        "",
        _markdown_table(
            _format_for_markdown(
                return_leaders[
                    [
                        "candidate_label",
                        "average_test_return",
                        "positive_window_rate",
                        "average_sharpe_like",
                        "worst_max_drawdown",
                        "evidence_scope",
                        "recommendation_label",
                    ]
                ],
                exclude={"candidate_label", "evidence_scope", "recommendation_label"},
            )
        ),
        "",
        "The top raw-return configuration remains useful as a stress reference, but "
        "the decision logic penalizes drawdown, cost sensitivity, and narrow "
        "5-window evidence before recommending the next run.",
        "",
        "## Top1 vs top3 vs top5 comparison",
        "",
        _markdown_table(
            _format_for_markdown(
                topk_table,
                exclude={"model_type", "candidate_count", "top_k"},
            )
        ),
        "",
        "## Ridge vs linear_regression comparison",
        "",
        _markdown_table(
            _format_for_markdown(
                model_table,
                exclude={"model_type", "candidate_count"},
            )
        ),
        "",
        "## Cost Sensitivity At 5/10/20bps",
        "",
        _markdown_table(
            _format_for_markdown(
                cost_table,
                exclude={"candidate_label", "is_anchor_cost"},
            )
        ),
        "",
        "## Smoothing Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                smoothing_table,
                exclude={"candidate_label", "is_candidate_smoothing"},
            )
        ),
        "",
        "## Train-Window And Test-Window Stability",
        "",
        _markdown_table(
            _format_for_markdown(
                window_table,
                exclude={"candidate_label", "is_candidate_window"},
            )
        ),
        "",
        "## Drawdown/Risk-Control Candidates",
        "",
        _markdown_table(
            _format_for_markdown(
                drawdown[
                    [
                        "candidate_label",
                        "average_test_return",
                        "positive_window_rate",
                        "average_max_drawdown",
                        "worst_max_drawdown",
                        "recommendation_label",
                    ]
                ],
                exclude={"candidate_label", "recommendation_label"},
            )
        ),
        "",
        "## Candidate Shortlist Recommendation",
        "",
        _markdown_table(
            _format_for_markdown(
                decision[
                    [
                        "candidate_label",
                        "return_strength",
                        "drawdown_control",
                        "cost_sensitivity_rating",
                        "window_stability_rating",
                        "recommendation_label",
                        "next_action",
                    ]
                ],
                exclude={
                    "candidate_label",
                    "return_strength",
                    "drawdown_control",
                    "cost_sensitivity_rating",
                    "window_stability_rating",
                    "recommendation_label",
                    "next_action",
                },
            )
        ),
        "",
        "## Next-Step Recommendation",
        "",
        "- First run an adjacent focused linear/ridge confirmation centered on ridge "
        "top5, top3, and the linear/ridge drawdown-control references.",
        "- Run `tree_screen` only after that confirmation shows the balanced/top5 "
        "direction is not just a narrow-window artifact.",
        "- Do not run MLP yet; keep it deferred until classical baselines have a "
        "confirmed, drawdown-aware target to beat.",
        "",
        "## Caveats",
        "",
        "- Tiingo-only research source.",
        "- Massive app data may not match research results exactly.",
        "- ETF survivorship bias is reduced by dynamic eligibility but not fully eliminated.",
        "- Candidates are not production-ready.",
        "",
    ]
    if report_data.warnings:
        lines.extend(["## Input Warnings", ""])
        lines.extend(f"- {warning}" for warning in report_data.warnings)
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Tiingo core-linear candidate deep-dive bundles from a completed "
            "baseline sweep CSV."
        )
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


def _read_optional_csv(
    path: Path | None,
    *,
    label: str,
    warnings: list[str],
) -> pd.DataFrame | None:
    if path is None:
        warnings.append(f"Optional {label} CSV was not provided.")
        return None
    if not path.exists():
        warnings.append(f"Optional {label} CSV is missing: {path}")
        return None
    return pd.read_csv(path)


def _build_candidate_selections(
    all_candidate_metrics: pd.DataFrame,
) -> tuple[CandidateSelection, ...]:
    selections: dict[tuple[str, int, int, float, float, int, int, int], CandidateSelection] = {}
    sort_order = 0
    for template in EXPLICIT_CANDIDATES:
        sort_order += 1
        selection = CandidateSelection(
            sort_order=sort_order,
            primary_group=template.selection_group,
            model_type=template.model_type,
            top_k=template.top_k,
            min_holding_days=template.min_holding_days,
            switch_margin=template.switch_margin,
            transaction_cost_bps=DECISION_ANCHOR_BPS,
            train_window_years=template.train_window_years,
            validation_window_months=template.validation_window_months,
            test_window_months=template.test_window_months,
            selection_groups=[template.selection_group],
            selection_reasons=[template.selection_reason],
        )
        _merge_selection(selections, selection)

    anchor = all_candidate_metrics.loc[
        _float_eq(all_candidate_metrics, "transaction_cost_bps", DECISION_ANCHOR_BPS)
    ].copy()
    if anchor.empty:
        raise ValueError("No 10bps candidate rows were found for optional deep-dive picks.")

    sort_order += 1
    best_sharpe = anchor.sort_values(
        by=["average_sharpe_like", "average_test_return", "positive_window_rate"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    _merge_selection(
        selections,
        _selection_from_metric_row(
            best_sharpe,
            sort_order=sort_order,
            selection_group="best 10bps average_sharpe_like",
            selection_reason="Best candidate by average_sharpe_like at the 10bps anchor.",
        ),
    )

    sort_order += 1
    best_drawdown = anchor.sort_values(
        by=["worst_max_drawdown", "average_test_return", "positive_window_rate"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    _merge_selection(
        selections,
        _selection_from_metric_row(
            best_drawdown,
            sort_order=sort_order,
            selection_group="best 10bps worst_drawdown",
            selection_reason="Best candidate by worst_max_drawdown at the 10bps anchor.",
        ),
    )

    return tuple(sorted(selections.values(), key=lambda selection: selection.sort_order))


def _merge_selection(
    selections: dict[tuple[str, int, int, float, float, int, int, int], CandidateSelection],
    candidate: CandidateSelection,
) -> None:
    existing = selections.get(candidate.key)
    if existing is None:
        selections[candidate.key] = candidate
        return
    existing.sort_order = min(existing.sort_order, candidate.sort_order)
    for group in candidate.selection_groups:
        if group not in existing.selection_groups:
            existing.selection_groups.append(group)
    for reason in candidate.selection_reasons:
        if reason not in existing.selection_reasons:
            existing.selection_reasons.append(reason)


def _selection_from_metric_row(
    row: pd.Series,
    *,
    sort_order: int,
    selection_group: str,
    selection_reason: str,
) -> CandidateSelection:
    return CandidateSelection(
        sort_order=sort_order,
        primary_group=selection_group,
        model_type=str(row["model_type"]),
        top_k=int(row["top_k"]),
        min_holding_days=int(row["min_holding_days"]),
        switch_margin=float(row["switch_margin"]),
        transaction_cost_bps=float(row["transaction_cost_bps"]),
        train_window_years=int(row["train_window_years"]),
        validation_window_months=int(row["validation_window_months"]),
        test_window_months=int(row["test_window_months"]),
        selection_groups=[selection_group],
        selection_reasons=[selection_reason],
    )


def _build_candidate_bundle(
    *,
    selection: CandidateSelection,
    candidate_rows: pd.DataFrame,
    all_test_rows: pd.DataFrame,
    source_summary: pd.DataFrame | None,
    source_decision_matrix: pd.DataFrame | None,
) -> CandidateBundle:
    sorted_candidate_rows = _sort_fold_rows(candidate_rows)
    first = sorted_candidate_rows.iloc[0]
    candidate_key = _candidate_key(selection)
    candidate_id = str(first["candidate_id"])
    candidate_label = str(first["candidate_label"])
    cost_sensitivity = _build_cost_sensitivity(all_test_rows, selection, candidate_label)
    smoothing_sensitivity = _build_smoothing_sensitivity(
        all_test_rows,
        selection,
        candidate_label,
    )
    window_stability = _build_window_stability(all_test_rows, selection, candidate_label)

    metrics = _metrics_from_rows(sorted_candidate_rows)
    cost_rating = _sensitivity_rating(cost_sensitivity)
    smoothing_rating = _sensitivity_rating(smoothing_sensitivity)
    window_rating = _sensitivity_rating(window_stability)
    baseline_recommendation = _lookup_value(
        source_summary,
        candidate_id=candidate_id,
        column="recommendation",
    )
    baseline_rationale = _lookup_value(
        source_decision_matrix,
        candidate_id=candidate_id,
        column="rationale",
    )
    recommendation_label, recommendation_rationale = _recommendation(
        metrics=metrics,
        selection=selection,
        cost_rating=cost_rating,
        smoothing_rating=smoothing_rating,
        window_rating=window_rating,
    )
    summary_row = {
        "candidate_key": candidate_key,
        "candidate_id": candidate_id,
        "candidate_label": candidate_label,
        "selection_groups": _join_unique(selection.selection_groups),
        "selection_reasons": _join_unique(selection.selection_reasons),
        "dataset_source": DATASET_SOURCE_LABEL,
        "sweep_profile": CORE_LINEAR_PROFILE,
        "model_family": "baseline",
        "model_type": selection.model_type,
        "top_k": selection.top_k,
        "allocation_mode": COMMON_ALLOCATION_MODE,
        "min_holding_days": selection.min_holding_days,
        "switch_margin": selection.switch_margin,
        "transaction_cost_bps": selection.transaction_cost_bps,
        "decision_anchor_bps": DECISION_ANCHOR_BPS,
        "train_window_years": selection.train_window_years,
        "validation_window_months": selection.validation_window_months,
        "test_window_months": selection.test_window_months,
        **metrics,
        "cost_sensitivity_rating": cost_rating,
        "smoothing_sensitivity_rating": smoothing_rating,
        "window_stability_rating": window_rating,
        "evidence_scope": _evidence_scope(metrics["window_count"]),
        "baseline_sweep_recommendation": baseline_recommendation,
        "baseline_decision_rationale": baseline_rationale,
        "recommendation_label": recommendation_label,
        "recommendation_rationale": recommendation_rationale,
        "sort_order": selection.sort_order,
    }
    candidate_summary = pd.DataFrame([summary_row])
    fold_results = _candidate_fold_results(
        sorted_candidate_rows,
        candidate_key=candidate_key,
        candidate_label=candidate_label,
    )
    window_metadata = fold_results[FOLD_METADATA_COLUMNS].drop_duplicates().reset_index(drop=True)
    return CandidateBundle(
        selection=selection,
        candidate_key=candidate_key,
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        candidate_summary=candidate_summary,
        fold_results=fold_results,
        window_metadata=window_metadata,
        cost_sensitivity=cost_sensitivity,
        smoothing_sensitivity=smoothing_sensitivity,
        window_stability=window_stability,
    )


def _write_candidate_bundle(*, bundle: CandidateBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = bundle.candidate_key
    bundle.candidate_summary.drop(columns=["sort_order"], errors="ignore").to_csv(
        output_dir / f"{prefix}_candidate_summary.csv",
        index=False,
    )
    bundle.fold_results.to_csv(output_dir / f"{prefix}_fold_test_results.csv", index=False)
    bundle.window_metadata.to_csv(output_dir / f"{prefix}_window_metadata.csv", index=False)
    bundle.cost_sensitivity.to_csv(output_dir / f"{prefix}_cost_sensitivity.csv", index=False)
    bundle.smoothing_sensitivity.to_csv(
        output_dir / f"{prefix}_smoothing_sensitivity.csv",
        index=False,
    )
    bundle.window_stability.to_csv(output_dir / f"{prefix}_window_stability.csv", index=False)
    (output_dir / f"{prefix}_report.md").write_text(
        _render_candidate_markdown(bundle),
        encoding="utf-8",
    )


def _render_candidate_markdown(bundle: CandidateBundle) -> str:
    row = bundle.candidate_summary.iloc[0]
    lines = [
        f"# {bundle.candidate_label}",
        "",
        "## Candidate Summary",
        "",
        f"Recommendation: `{row['recommendation_label']}`.",
        "",
        str(row["recommendation_rationale"]),
        "",
        _markdown_table(
            _format_for_markdown(
                bundle.candidate_summary.drop(columns=["sort_order"], errors="ignore")[
                    [
                        "candidate_label",
                        "selection_groups",
                        "average_test_return",
                        "positive_window_rate",
                        "average_sharpe_like",
                        "worst_max_drawdown",
                        "cost_sensitivity_rating",
                        "smoothing_sensitivity_rating",
                        "window_stability_rating",
                        "evidence_scope",
                        "recommendation_label",
                    ]
                ],
                exclude={
                    "candidate_label",
                    "selection_groups",
                    "cost_sensitivity_rating",
                    "smoothing_sensitivity_rating",
                    "window_stability_rating",
                    "evidence_scope",
                    "recommendation_label",
                },
            )
        ),
        "",
        "## Fold-Level Test Results",
        "",
        _markdown_table(
            _format_for_markdown(
                bundle.fold_results[
                    [
                        "fold",
                        "total_return",
                        "sharpe_like",
                        "max_drawdown",
                        "final_equity",
                        "mean_daily_turnover",
                    ]
                ],
                exclude={"fold"},
            )
        ),
        "",
        "## Window Metadata",
        "",
        _markdown_table(_format_for_markdown(bundle.window_metadata, exclude={"fold"})),
        "",
        "## Cost Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                bundle.cost_sensitivity,
                exclude={"candidate_label", "is_anchor_cost"},
            )
        ),
        "",
        "## Smoothing Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                bundle.smoothing_sensitivity.head(20),
                exclude={"candidate_label", "is_candidate_smoothing"},
            )
        ),
        "",
        "## Window Stability",
        "",
        _markdown_table(
            _format_for_markdown(
                bundle.window_stability,
                exclude={"candidate_label", "is_candidate_window"},
            )
        ),
        "",
    ]
    return "\n".join(lines)


def _build_candidate_metrics(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = list(SUMMARY_GROUP_COLUMNS)
    for group_values, group in test_rows.groupby(group_cols, dropna=False, sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": first["candidate_id"],
                "candidate_label": first["candidate_label"],
                **dict(zip(group_cols, group_values, strict=True)),
                **_metrics_from_rows(group),
            }
        )
    return pd.DataFrame(rows)


def _build_cost_sensitivity(
    all_test_rows: pd.DataFrame,
    selection: CandidateSelection,
    candidate_label: str,
) -> pd.DataFrame:
    rows = all_test_rows.loc[
        all_test_rows["model_type"].astype(str).eq(selection.model_type)
        & _int_eq(all_test_rows, "top_k", selection.top_k)
        & _int_eq(all_test_rows, "min_holding_days", selection.min_holding_days)
        & _float_eq(all_test_rows, "switch_margin", selection.switch_margin)
        & _int_eq(all_test_rows, "train_window_years", selection.train_window_years)
        & _int_eq(
            all_test_rows,
            "validation_window_months",
            selection.validation_window_months,
        )
        & _int_eq(all_test_rows, "test_window_months", selection.test_window_months)
        & all_test_rows["allocation_mode"].astype(str).eq(COMMON_ALLOCATION_MODE)
        & _series_in_float_values(
            all_test_rows["transaction_cost_bps"],
            COST_SENSITIVITY_BPS,
        )
    ].copy()
    table = _aggregate(rows, ["transaction_cost_bps"])
    if table.empty:
        return table
    table.insert(0, "candidate_label", candidate_label)
    table["is_anchor_cost"] = table["transaction_cost_bps"].map(
        lambda value: np.isclose(float(value), selection.transaction_cost_bps)
    )
    return table.sort_values(by=["transaction_cost_bps"], kind="mergesort").reset_index(drop=True)


def _build_smoothing_sensitivity(
    all_test_rows: pd.DataFrame,
    selection: CandidateSelection,
    candidate_label: str,
) -> pd.DataFrame:
    rows = all_test_rows.loc[
        all_test_rows["model_type"].astype(str).eq(selection.model_type)
        & _int_eq(all_test_rows, "top_k", selection.top_k)
        & _float_eq(all_test_rows, "transaction_cost_bps", selection.transaction_cost_bps)
        & _int_eq(all_test_rows, "train_window_years", selection.train_window_years)
        & _int_eq(
            all_test_rows,
            "validation_window_months",
            selection.validation_window_months,
        )
        & _int_eq(all_test_rows, "test_window_months", selection.test_window_months)
        & all_test_rows["allocation_mode"].astype(str).eq(COMMON_ALLOCATION_MODE)
    ].copy()
    table = _aggregate(rows, ["min_holding_days", "switch_margin"])
    if table.empty:
        return table
    table.insert(0, "candidate_label", candidate_label)
    table["is_candidate_smoothing"] = table.apply(
        lambda row: int(row["min_holding_days"]) == selection.min_holding_days
        and np.isclose(float(row["switch_margin"]), selection.switch_margin),
        axis=1,
    )
    return table.sort_values(
        by=["average_test_return", "positive_window_rate", "worst_max_drawdown"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_window_stability(
    all_test_rows: pd.DataFrame,
    selection: CandidateSelection,
    candidate_label: str,
) -> pd.DataFrame:
    rows = all_test_rows.loc[
        all_test_rows["model_type"].astype(str).eq(selection.model_type)
        & _int_eq(all_test_rows, "top_k", selection.top_k)
        & _int_eq(all_test_rows, "min_holding_days", selection.min_holding_days)
        & _float_eq(all_test_rows, "switch_margin", selection.switch_margin)
        & _float_eq(all_test_rows, "transaction_cost_bps", selection.transaction_cost_bps)
        & all_test_rows["allocation_mode"].astype(str).eq(COMMON_ALLOCATION_MODE)
    ].copy()
    table = _aggregate(
        rows,
        ["train_window_years", "validation_window_months", "test_window_months"],
    )
    if table.empty:
        return table
    table.insert(0, "candidate_label", candidate_label)
    table["is_candidate_window"] = table.apply(
        lambda row: int(row["train_window_years"]) == selection.train_window_years
        and int(row["validation_window_months"]) == selection.validation_window_months
        and int(row["test_window_months"]) == selection.test_window_months,
        axis=1,
    )
    return table.sort_values(
        by=["train_window_years", "validation_window_months", "test_window_months"],
        kind="mergesort",
    ).reset_index(drop=True)


def _aggregate(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    if rows.empty:
        return pd.DataFrame(columns=[*group_cols, *_metric_column_names()])
    for values, group in rows.groupby(group_cols, dropna=False, sort=True):
        group_values = values if isinstance(values, tuple) else (values,)
        output_rows.append(
            {
                **dict(zip(group_cols, group_values, strict=True)),
                **_metrics_from_rows(group),
            }
        )
    return pd.DataFrame(output_rows, columns=[*group_cols, *_metric_column_names()])


def _metrics_from_rows(rows: pd.DataFrame) -> dict[str, object]:
    return {
        "window_count": int(len(rows)),
        "average_test_return": _mean(rows, "total_return"),
        "median_test_return": _median(rows, "total_return"),
        "positive_window_rate": _positive_rate(rows["total_return"]),
        "average_sharpe_like": _mean(rows, "sharpe_like"),
        "average_max_drawdown": _mean(rows, "max_drawdown"),
        "worst_max_drawdown": _min(rows, "max_drawdown"),
        "average_total_transaction_cost": _mean(rows, "total_transaction_cost"),
        "average_mean_daily_turnover": _mean(rows, "mean_daily_turnover"),
    }


def _metric_column_names() -> list[str]:
    return [
        "window_count",
        "average_test_return",
        "median_test_return",
        "positive_window_rate",
        "average_sharpe_like",
        "average_max_drawdown",
        "worst_max_drawdown",
        "average_total_transaction_cost",
        "average_mean_daily_turnover",
    ]


def _candidate_fold_results(
    rows: pd.DataFrame,
    *,
    candidate_key: str,
    candidate_label: str,
) -> pd.DataFrame:
    output = rows.copy()
    output["candidate_key"] = candidate_key
    output["candidate_label"] = candidate_label
    for column in FOLD_RESULT_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output[FOLD_RESULT_COLUMNS].reset_index(drop=True)


def _build_decision_matrix(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in candidate_summary.iterrows():
        return_strength = _return_strength(row)
        positive_strength = _positive_window_strength(row)
        drawdown_control = _drawdown_control(row)
        balanced_score = _balanced_score(row)
        rows.append(
            {
                "candidate_key": row["candidate_key"],
                "candidate_label": row["candidate_label"],
                "selection_groups": row["selection_groups"],
                "model_type": row["model_type"],
                "top_k": int(row["top_k"]),
                "min_holding_days": int(row["min_holding_days"]),
                "switch_margin": float(row["switch_margin"]),
                "transaction_cost_bps": float(row["transaction_cost_bps"]),
                "train_window_years": int(row["train_window_years"]),
                "validation_window_months": int(row["validation_window_months"]),
                "test_window_months": int(row["test_window_months"]),
                "window_count": int(row["window_count"]),
                "average_test_return": float(row["average_test_return"]),
                "positive_window_rate": float(row["positive_window_rate"]),
                "average_sharpe_like": float(row["average_sharpe_like"]),
                "worst_max_drawdown": float(row["worst_max_drawdown"]),
                "return_strength": return_strength,
                "positive_window_strength": positive_strength,
                "drawdown_control": drawdown_control,
                "cost_sensitivity_rating": row["cost_sensitivity_rating"],
                "smoothing_sensitivity_rating": row["smoothing_sensitivity_rating"],
                "window_stability_rating": row["window_stability_rating"],
                "evidence_scope": row["evidence_scope"],
                "balanced_score": balanced_score,
                "recommendation_label": row["recommendation_label"],
                "next_action": _next_action(row),
                "rationale": _decision_rationale(
                    row,
                    return_strength=return_strength,
                    positive_strength=positive_strength,
                    drawdown_control=drawdown_control,
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def _recommendation(
    *,
    metrics: dict[str, object],
    selection: CandidateSelection,
    cost_rating: str,
    smoothing_rating: str,
    window_rating: str,
) -> tuple[str, str]:
    row = {
        **metrics,
        "cost_sensitivity_rating": cost_rating,
        "smoothing_sensitivity_rating": smoothing_rating,
        "window_stability_rating": window_rating,
        "top_k": selection.top_k,
    }
    return_strength = _return_strength(row)
    positive_strength = _positive_window_strength(row)
    drawdown_control = _drawdown_control(row)
    groups = set(selection.selection_groups)
    sensitivity_warning = sum(
        rating == "weak" for rating in [cost_rating, smoothing_rating, window_rating]
    )
    if "drawdown-control reference" in groups or "best 10bps worst_drawdown" in groups:
        if drawdown_control == "strong" and _as_float(metrics["average_test_return"]) > 0.0:
            return (
                "keep as drawdown-control reference",
                "Positive average return with the strongest drawdown profile in the shortlist.",
            )
    if "balanced reference" in groups:
        if (
            return_strength in {"strong", "moderate"}
            and positive_strength != "weak"
            and drawdown_control != "weak"
            and cost_rating != "weak"
            and sensitivity_warning <= 2
        ):
            return (
                "keep as balanced reference",
                "Diversified top_k candidate with acceptable return, hit rate, "
                "and drawdown; confirm window stability next.",
            )
    if groups.intersection(
        {
            "return leader",
            "linear return reference",
            "best 10bps average_sharpe_like",
        }
    ):
        if (
            _as_float(metrics["average_test_return"]) > 0.0
            and positive_strength != "weak"
            and drawdown_control != "weak"
        ):
            return (
                "keep as return reference",
                "Useful return-side benchmark, but not promoted solely on raw return.",
            )
    if drawdown_control == "strong" and selection.top_k >= 5:
        return (
            "keep as drawdown-control reference",
            "Drawdown profile is strong enough to preserve as a risk-control comparison.",
        )
    return (
        "deprioritize",
        "Metrics do not clear enough return, hit-rate, drawdown, or sensitivity checks.",
    )


def _return_strength(row: pd.Series | dict[str, object]) -> str:
    average_return = _as_float(row["average_test_return"])
    positive_rate = _as_float(row["positive_window_rate"])
    if pd.isna(average_return) or pd.isna(positive_rate):
        return "unknown"
    if average_return >= 0.10 and positive_rate >= 0.75:
        return "strong"
    if average_return >= 0.03 and positive_rate >= 0.55:
        return "moderate"
    return "weak"


def _positive_window_strength(row: pd.Series | dict[str, object]) -> str:
    positive_rate = _as_float(row["positive_window_rate"])
    if pd.isna(positive_rate):
        return "unknown"
    if positive_rate >= 0.75:
        return "strong"
    if positive_rate >= 0.55:
        return "moderate"
    return "weak"


def _drawdown_control(row: pd.Series | dict[str, object]) -> str:
    worst = _as_float(row["worst_max_drawdown"])
    if pd.isna(worst):
        return "unknown"
    if worst >= -0.15:
        return "strong"
    if worst >= -0.30:
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
    if return_range <= 0.02 and min_positive >= 0.55:
        return "strong"
    if return_range <= 0.06 and min_positive >= 0.45:
        return "moderate"
    return "weak"


def _balanced_score(row: pd.Series | dict[str, object]) -> float:
    score = _as_float(row["average_test_return"])
    score += 0.04 * _as_float(row["positive_window_rate"])
    score += 0.02 * _as_float(row["average_sharpe_like"])
    score += 0.08 * max(_as_float(row["worst_max_drawdown"]), -1.0)
    if _as_float(row["window_count"]) <= 5:
        score -= 0.01
    for column in [
        "cost_sensitivity_rating",
        "smoothing_sensitivity_rating",
        "window_stability_rating",
    ]:
        score += {"strong": 0.01, "moderate": 0.0, "weak": -0.02}.get(
            str(row[column]),
            -0.005,
        )
    if int(row.get("top_k", 1)) >= 5:
        score += 0.01
    return float(score)


def _decision_rationale(
    row: pd.Series,
    *,
    return_strength: str,
    positive_strength: str,
    drawdown_control: str,
) -> str:
    return (
        f"return={return_strength}; positive_windows={positive_strength}; "
        f"drawdown={drawdown_control}; cost={row['cost_sensitivity_rating']}; "
        f"smoothing={row['smoothing_sensitivity_rating']}; "
        f"windows={row['window_stability_rating']}; evidence={row['evidence_scope']}"
    )


def _next_action(row: pd.Series) -> str:
    label = str(row["recommendation_label"])
    if label == "keep as balanced reference":
        return "Include in adjacent focused linear/ridge confirmation."
    if label == "keep as drawdown-control reference":
        return "Carry into confirmation as drawdown-control guardrail."
    if label == "keep as return reference":
        return "Keep for return-side comparison; do not promote by return alone."
    return "Deprioritize unless a later confirmation needs a contrast case."


def _topk_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_k, group in summary.groupby("top_k", sort=True):
        rows.append(
            {
                "top_k": int(top_k),
                "candidate_count": int(len(group)),
                "average_test_return": group["average_test_return"].mean(),
                "positive_window_rate": group["positive_window_rate"].mean(),
                "average_worst_max_drawdown": group["worst_max_drawdown"].mean(),
                "best_candidate": group.sort_values(
                    by=["average_test_return", "worst_max_drawdown"],
                    ascending=[False, False],
                    kind="mergesort",
                ).iloc[0]["candidate_label"],
            }
        )
    return pd.DataFrame(rows)


def _model_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_type, group in summary.groupby("model_type", sort=True):
        rows.append(
            {
                "model_type": model_type,
                "candidate_count": int(len(group)),
                "average_test_return": group["average_test_return"].mean(),
                "positive_window_rate": group["positive_window_rate"].mean(),
                "average_worst_max_drawdown": group["worst_max_drawdown"].mean(),
                "best_candidate": group.sort_values(
                    by=["balanced_score" if "balanced_score" in group else "average_test_return"],
                    ascending=[False],
                    kind="mergesort",
                ).iloc[0]["candidate_label"],
            }
        )
    return pd.DataFrame(rows)


def _combined_bundle_table(
    bundles: Iterable[CandidateBundle],
    *,
    table_name: str,
    max_rows: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = getattr(bundle, table_name).copy()
        if frame.empty:
            continue
        frame.insert(0, "candidate_key", bundle.candidate_key)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).head(max_rows)


def _sort_shortlist(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    sort_order = pd.to_numeric(output.pop("sort_order"), errors="coerce")
    output.insert(0, "sort_order", sort_order)
    output = output.sort_values(
        by=["sort_order", "average_test_return"],
        ascending=[True, False],
        kind="mergesort",
    ).drop(columns=["sort_order"])
    return output[SUMMARY_COLUMNS].reset_index(drop=True)


def _match_candidate_rows(rows: pd.DataFrame, selection: CandidateSelection) -> pd.DataFrame:
    return rows.loc[
        rows["model_type"].astype(str).eq(selection.model_type)
        & _int_eq(rows, "top_k", selection.top_k)
        & _int_eq(rows, "min_holding_days", selection.min_holding_days)
        & _float_eq(rows, "switch_margin", selection.switch_margin)
        & _float_eq(rows, "transaction_cost_bps", selection.transaction_cost_bps)
        & _int_eq(rows, "train_window_years", selection.train_window_years)
        & _int_eq(rows, "validation_window_months", selection.validation_window_months)
        & _int_eq(rows, "test_window_months", selection.test_window_months)
        & rows["allocation_mode"].astype(str).eq(COMMON_ALLOCATION_MODE)
    ].copy()


def _add_fold_dates(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    for column in ["test_start", "test_end"]:
        if column not in output.columns:
            output[column] = pd.Series([None] * len(output), index=output.index, dtype=object)
        else:
            output[column] = output[column].astype(object)
    for index, fold_name in output["fold"].astype(str).items():
        metadata = _parse_fold_metadata(fold_name)
        if metadata is None:
            continue
        for column, value in metadata.items():
            output.at[index, column] = value
    return output


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
        "test_start": _format_fold_date(parts["test_start"]),
        "test_end": _format_fold_date(parts["test_end"]),
    }


def _format_fold_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _sort_fold_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(
        by=["test_start", "sequence_number", "fold"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _candidate_key(selection: CandidateSelection) -> str:
    return _slug(
        "__".join(
            [
                selection.primary_group,
                selection.model_type,
                f"top{selection.top_k}",
                f"hold{selection.min_holding_days}",
                f"margin{selection.switch_margin:g}",
                f"tw{selection.train_window_years}",
                f"v{selection.validation_window_months}",
                f"t{selection.test_window_months}",
            ]
        )
    )


def _selection_label(selection: CandidateSelection) -> str:
    return (
        f"{selection.model_type} / top{selection.top_k} / "
        f"hold {selection.min_holding_days} / margin {selection.switch_margin:g} / "
        f"cost {selection.transaction_cost_bps:g}bps / "
        f"tw{selection.train_window_years}y-v{selection.validation_window_months}m-"
        f"t{selection.test_window_months}m"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _lookup_value(
    frame: pd.DataFrame | None,
    *,
    candidate_id: str,
    column: str,
) -> object:
    if frame is None or "candidate_id" not in frame.columns or column not in frame.columns:
        return ""
    matches = frame.loc[frame["candidate_id"].astype(str).eq(candidate_id)]
    if matches.empty:
        return ""
    return matches.iloc[0][column]


def _evidence_scope(window_count: object) -> str:
    count = _as_float(window_count)
    if pd.isna(count):
        return "unknown evidence"
    if count <= 5:
        return "narrow 5-window evidence"
    return "broader multi-window evidence"


def _join_unique(values: Iterable[str]) -> str:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return "; ".join(unique)


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


def _int_eq(frame: pd.DataFrame, column: str, value: int) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").eq(value)


def _float_eq(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.isclose(numeric, value, atol=1e-9), index=frame.index)


def _series_in_float_values(series: pd.Series, values: Sequence[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    for value in values:
        mask |= pd.Series(np.isclose(numeric, value, atol=1e-9), index=series.index)
    return mask


create_tiingo_core_linear_deep_dive_bundles_from_args = (
    run_tiingo_core_linear_deep_dive_bundles_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_core_linear_deep_dive_bundles_from_args())
