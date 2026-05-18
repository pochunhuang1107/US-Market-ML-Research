"""Create the Tiingo focused linear/ridge confirmation report."""

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
from us_market_ml.cli.run_tiingo_focused_linear_ridge_confirmation import (
    EXPLICIT_CANDIDATE_SPECS,
)

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_focused_linear_ridge_confirmation/"
    "tiingo_focused_linear_ridge_confirmation_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_focused_linear_ridge_confirmation")

SUMMARY_PATH_NAME = "tiingo_focused_linear_ridge_confirmation_summary.csv"
DECISION_MATRIX_PATH_NAME = (
    "tiingo_focused_linear_ridge_confirmation_decision_matrix.csv"
)
REPORT_PATH_NAME = "tiingo_focused_linear_ridge_confirmation_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

DECISION_ANCHOR_BPS = 10.0
SENSITIVITY_BPS = 20.0

SPEC_COLUMNS = [
    "candidate_family",
    "candidate_role",
    "model_type",
    "top_k",
    "min_holding_days",
    "switch_margin",
]

SUMMARY_GROUP_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "candidate_family",
    "candidate_role",
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
]

SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    *SUMMARY_GROUP_COLUMNS,
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
    "train_window_stability_rating",
    "test_window_stability_rating",
    "evidence_scope",
    "recommendation",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_key",
    "candidate_label",
    "candidate_family",
    "candidate_role",
    "model_type",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "decision_anchor_bps",
    "window_count_10bps",
    "average_test_return_10bps",
    "positive_window_rate_10bps",
    "average_sharpe_like_10bps",
    "worst_max_drawdown_10bps",
    "average_test_return_20bps",
    "positive_window_rate_20bps",
    "worst_max_drawdown_20bps",
    "cost_delta_20_vs_10",
    "return_strength",
    "drawdown_control",
    "cost_resilience",
    "train_window_stability",
    "test_window_stability",
    "evidence_scope",
    "balanced_score",
    "recommendation",
    "next_action",
    "rationale",
]


@dataclass(frozen=True)
class FocusedConfirmationReportData:
    """Tables and metadata produced by the focused confirmation report."""

    candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    train_window_stability: pd.DataFrame
    test_window_stability: pd.DataFrame
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    dataset_source: str = DATASET_SOURCE_LABEL


def run_tiingo_focused_linear_ridge_confirmation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the focused linear/ridge confirmation report."""

    args = _parse_args(argv)
    report_data = build_tiingo_focused_linear_ridge_confirmation_report(
        results_csv=args.results_csv,
    )
    outputs = write_tiingo_focused_linear_ridge_confirmation_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo focused linear/ridge confirmation report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.decision_matrix.head(20).to_string(index=False))
    return 0


def build_tiingo_focused_linear_ridge_confirmation_report(
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
) -> FocusedConfirmationReportData:
    """Read completed focused confirmation results and create report tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _add_candidate_metadata(_enrich_test_rows(test_rows))
    included = enriched.loc[
        enriched["model_family"].astype(str).eq("baseline")
        & enriched["candidate_role"].astype(str).ne("unknown")
    ].copy()
    if included.empty:
        raise ValueError(
            "No focused linear/ridge candidate test rows were found in the results CSV."
        )

    cost_sensitivity = _aggregate(included, [*SPEC_COLUMNS, "transaction_cost_bps"])
    train_window_stability = _aggregate(
        included,
        [*SPEC_COLUMNS, "transaction_cost_bps", "train_window_years"],
    )
    test_window_stability = _aggregate(
        included,
        [*SPEC_COLUMNS, "transaction_cost_bps", "test_window_months"],
    )
    candidate_summary = _sort_candidate_summary(
        _build_candidate_summary(
            included,
            cost_sensitivity=cost_sensitivity,
            train_window_stability=train_window_stability,
            test_window_stability=test_window_stability,
        )
    )
    decision_matrix = _sort_decision_matrix(
        _build_decision_matrix(
            included,
            cost_sensitivity=cost_sensitivity,
            train_window_stability=train_window_stability,
            test_window_stability=test_window_stability,
        )
    )

    return FocusedConfirmationReportData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        cost_sensitivity=cost_sensitivity,
        train_window_stability=train_window_stability,
        test_window_stability=test_window_stability,
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
    )


def write_tiingo_focused_linear_ridge_confirmation_report(
    *,
    report_data: FocusedConfirmationReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.candidate_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_focused_linear_ridge_confirmation_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_focused_linear_ridge_confirmation_markdown(
    report_data: FocusedConfirmationReportData,
) -> str:
    """Render the focused confirmation Markdown report."""

    decision = report_data.decision_matrix
    best_balanced = _best_role(decision, "balanced_reference")
    best_return = _best_role(decision, "return_reference")
    best_drawdown = _best_role(decision, "drawdown_control_reference")

    family_table = (
        report_data.candidate_summary[
            [
                "candidate_family",
                "candidate_role",
                "model_type",
                "top_k",
                "min_holding_days",
                "switch_margin",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by=["candidate_role", "candidate_family", "model_type", "top_k"],
            kind="mergesort",
        )
    )
    return_refs = decision.loc[decision["candidate_role"].eq("return_reference")]
    balanced = decision.loc[decision["candidate_role"].eq("balanced_reference")]
    drawdown = decision.loc[decision["candidate_role"].eq("drawdown_control_reference")]

    lines = [
        "# Tiingo Focused Linear/Ridge Confirmation Report",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the Tiingo-only focused linear/ridge confirmation "
        "around the shortlisted core-linear candidates. It is local-only, "
        "uses the `tiingo_only` source label, and intentionally excludes tree "
        "models, MLP, S3, and GPU work.",
        "",
        f"Input rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included focused candidate test rows: "
        f"`{report_data.included_test_row_count}`.",
        "",
        f"Best balanced top3/top5 candidate: `{best_balanced['candidate_label']}`.",
        "",
        f"Return comparison anchor: `{best_return['candidate_label']}`.",
        "",
        f"Drawdown-control guardrail: `{best_drawdown['candidate_label']}`.",
        "",
        "## Candidate Families Tested",
        "",
        _markdown_table(
            _format_for_markdown(
                family_table,
                exclude={"candidate_family", "candidate_role", "model_type"},
            )
        ),
        "",
        "## Return References",
        "",
        _markdown_table(
            _format_for_markdown(
                return_refs[
                    [
                        "candidate_label",
                        "average_test_return_10bps",
                        "positive_window_rate_10bps",
                        "worst_max_drawdown_10bps",
                        "cost_resilience",
                        "recommendation",
                    ]
                ],
                exclude={"candidate_label", "cost_resilience", "recommendation"},
            )
        ),
        "",
        "Top1 return references remain comparison anchors; the decision logic does "
        "not promote a raw-return leader on return alone.",
        "",
        "## Balanced Top3/Top5 Candidates",
        "",
        _markdown_table(
            _format_for_markdown(
                balanced[
                    [
                        "candidate_label",
                        "average_test_return_10bps",
                        "positive_window_rate_10bps",
                        "worst_max_drawdown_10bps",
                        "cost_resilience",
                        "train_window_stability",
                        "test_window_stability",
                        "recommendation",
                    ]
                ],
                exclude={
                    "candidate_label",
                    "cost_resilience",
                    "train_window_stability",
                    "test_window_stability",
                    "recommendation",
                },
            )
        ),
        "",
        "## Drawdown-Control Candidates",
        "",
        _markdown_table(
            _format_for_markdown(
                drawdown[
                    [
                        "candidate_label",
                        "average_test_return_10bps",
                        "positive_window_rate_10bps",
                        "worst_max_drawdown_10bps",
                        "drawdown_control",
                        "recommendation",
                    ]
                ],
                exclude={"candidate_label", "drawdown_control", "recommendation"},
            )
        ),
        "",
        "## Cost Sensitivity 10bps vs 20bps",
        "",
        _markdown_table(
            _format_for_markdown(
                decision[
                    [
                        "candidate_label",
                        "average_test_return_10bps",
                        "average_test_return_20bps",
                        "positive_window_rate_20bps",
                        "cost_delta_20_vs_10",
                        "cost_resilience",
                    ]
                ],
                exclude={"candidate_label", "cost_resilience"},
            )
        ),
        "",
        "## Test-Window Stability 3m vs 6m vs 12m",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.test_window_stability.loc[
                    _float_eq(
                        report_data.test_window_stability,
                        "transaction_cost_bps",
                        DECISION_ANCHOR_BPS,
                    )
                ].head(60),
                exclude={"candidate_family", "candidate_role", "model_type"},
            )
        ),
        "",
        "## Train-Window Stability 3y vs 5y",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.train_window_stability.loc[
                    _float_eq(
                        report_data.train_window_stability,
                        "transaction_cost_bps",
                        DECISION_ANCHOR_BPS,
                    )
                ].head(60),
                exclude={"candidate_family", "candidate_role", "model_type"},
            )
        ),
        "",
        "## Decision Matrix",
        "",
        _markdown_table(
            _format_for_markdown(
                decision[
                    [
                        "candidate_label",
                        "candidate_role",
                        "return_strength",
                        "drawdown_control",
                        "cost_resilience",
                        "train_window_stability",
                        "test_window_stability",
                        "recommendation",
                        "next_action",
                    ]
                ],
                exclude={
                    "candidate_label",
                    "candidate_role",
                    "return_strength",
                    "drawdown_control",
                    "cost_resilience",
                    "train_window_stability",
                    "test_window_stability",
                    "recommendation",
                    "next_action",
                },
            )
        ),
        "",
        "## Recommendation",
        "",
        f"- Candidate to deep-dive further: `{best_balanced['candidate_label']}`.",
        f"- Candidate to keep as return reference: `{best_return['candidate_label']}`.",
        f"- Candidate to keep as drawdown-control guardrail: "
        f"`{best_drawdown['candidate_label']}`.",
        f"- Whether `tree_screen` should run next: {_tree_screen_recommendation(decision)}",
        "- Whether MLP should still wait: yes. Keep MLP deferred until the "
        "classical baseline family has a confirmed target to beat.",
        "",
        "## Caveats",
        "",
        "- Tiingo-only source.",
        "- Massive app data may not match.",
        "- ETF survivorship bias is not fully eliminated.",
        "- Not production-ready.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo focused linear/ridge confirmation report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _add_candidate_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    metadata = output.apply(_candidate_metadata_from_row, axis=1, result_type="expand")
    output["candidate_family"] = metadata["candidate_family"]
    output["candidate_role"] = metadata["candidate_role"]
    return output


def _candidate_metadata_from_row(row: pd.Series) -> dict[str, str]:
    spec = _candidate_metadata_lookup().get(
        (
            str(row["model_type"]),
            int(row["top_k"]),
            int(row["min_holding_days"]),
            round(float(row["switch_margin"]), 9),
        )
    )
    if spec is None:
        return {"candidate_family": "unknown", "candidate_role": "unknown"}
    return {
        "candidate_family": spec.candidate_family,
        "candidate_role": spec.candidate_role,
    }


def _candidate_metadata_lookup() -> dict[tuple[str, int, int, float], object]:
    return {
        (
            candidate.model_type,
            candidate.top_k,
            candidate.min_holding_days,
            round(candidate.switch_margin, 9),
        ): candidate
        for candidate in EXPLICIT_CANDIDATE_SPECS
    }


def _build_candidate_summary(
    rows: pd.DataFrame,
    *,
    cost_sensitivity: pd.DataFrame,
    train_window_stability: pd.DataFrame,
    test_window_stability: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for group_values, group in rows.groupby(SUMMARY_GROUP_COLUMNS, dropna=False, sort=False):
        first = group.iloc[0]
        cost_spec_filter = _spec_filter(cost_sensitivity, first)
        train_spec_filter = _spec_filter(train_window_stability, first)
        test_spec_filter = _spec_filter(test_window_stability, first)
        train_filter = train_spec_filter & _float_eq(
            train_window_stability,
            "transaction_cost_bps",
            float(first["transaction_cost_bps"]),
        )
        test_filter = test_spec_filter & _float_eq(
            test_window_stability,
            "transaction_cost_bps",
            float(first["transaction_cost_bps"]),
        )
        metrics = _metrics_from_rows(group)
        row = {
            "candidate_id": first["candidate_id"],
            "candidate_label": first["candidate_label"],
            **dict(zip(SUMMARY_GROUP_COLUMNS, group_values, strict=True)),
            **metrics,
            "cost_sensitivity_rating": _sensitivity_rating(
                cost_sensitivity.loc[cost_spec_filter]
            ),
            "train_window_stability_rating": _sensitivity_rating(
                train_window_stability.loc[train_filter]
            ),
            "test_window_stability_rating": _sensitivity_rating(
                test_window_stability.loc[test_filter]
            ),
            "evidence_scope": _evidence_scope(metrics["window_count"]),
        }
        row["recommendation"] = _summary_recommendation(row)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def _build_decision_matrix(
    rows: pd.DataFrame,
    *,
    cost_sensitivity: pd.DataFrame,
    train_window_stability: pd.DataFrame,
    test_window_stability: pd.DataFrame,
) -> pd.DataFrame:
    decision_rows: list[dict[str, object]] = []
    del cost_sensitivity
    for values, group in rows.groupby(SPEC_COLUMNS, dropna=False, sort=False):
        group_values = dict(zip(SPEC_COLUMNS, values, strict=True))
        anchor_rows = group.loc[_float_eq(group, "transaction_cost_bps", DECISION_ANCHOR_BPS)]
        if anchor_rows.empty:
            anchor_rows = group.copy()
        sensitivity_rows = group.loc[_float_eq(group, "transaction_cost_bps", SENSITIVITY_BPS)]

        first = anchor_rows.iloc[0]
        metrics_10 = _metrics_from_rows(anchor_rows)
        metrics_20 = _metrics_from_rows(sensitivity_rows)
        cost_delta = _as_float(metrics_20["average_test_return"]) - _as_float(
            metrics_10["average_test_return"]
        )
        train_filter = _spec_filter(train_window_stability, group_values) & _float_eq(
            train_window_stability,
            "transaction_cost_bps",
            DECISION_ANCHOR_BPS,
        )
        test_filter = _spec_filter(test_window_stability, group_values) & _float_eq(
            test_window_stability,
            "transaction_cost_bps",
            DECISION_ANCHOR_BPS,
        )
        return_strength = _return_strength(metrics_10)
        drawdown_control = _drawdown_control(metrics_10)
        cost_resilience = _cost_resilience(metrics_10, metrics_20, cost_delta)
        train_rating = _sensitivity_rating(train_window_stability.loc[train_filter])
        test_rating = _sensitivity_rating(test_window_stability.loc[test_filter])
        evidence_scope = _evidence_scope(metrics_10["window_count"])
        balanced_score = _balanced_score(
            metrics=metrics_10,
            role=str(group_values["candidate_role"]),
            cost_resilience=cost_resilience,
            train_stability=train_rating,
            test_stability=test_rating,
        )
        recommendation = _decision_recommendation(
            role=str(group_values["candidate_role"]),
            return_strength=return_strength,
            drawdown_control=drawdown_control,
            cost_resilience=cost_resilience,
            train_stability=train_rating,
            test_stability=test_rating,
            evidence_scope=evidence_scope,
            average_return=_as_float(metrics_10["average_test_return"]),
        )
        decision_rows.append(
            {
                "candidate_key": _candidate_key(group_values),
                "candidate_label": _candidate_label(first),
                **group_values,
                "decision_anchor_bps": DECISION_ANCHOR_BPS,
                "window_count_10bps": metrics_10["window_count"],
                "average_test_return_10bps": metrics_10["average_test_return"],
                "positive_window_rate_10bps": metrics_10["positive_window_rate"],
                "average_sharpe_like_10bps": metrics_10["average_sharpe_like"],
                "worst_max_drawdown_10bps": metrics_10["worst_max_drawdown"],
                "average_test_return_20bps": metrics_20["average_test_return"],
                "positive_window_rate_20bps": metrics_20["positive_window_rate"],
                "worst_max_drawdown_20bps": metrics_20["worst_max_drawdown"],
                "cost_delta_20_vs_10": cost_delta,
                "return_strength": return_strength,
                "drawdown_control": drawdown_control,
                "cost_resilience": cost_resilience,
                "train_window_stability": train_rating,
                "test_window_stability": test_rating,
                "evidence_scope": evidence_scope,
                "balanced_score": balanced_score,
                "recommendation": recommendation,
                "next_action": _next_action(recommendation),
                "rationale": _decision_rationale(
                    return_strength=return_strength,
                    drawdown_control=drawdown_control,
                    cost_resilience=cost_resilience,
                    train_stability=train_rating,
                    test_stability=test_rating,
                    evidence_scope=evidence_scope,
                ),
            }
        )
    return pd.DataFrame(decision_rows, columns=DECISION_MATRIX_COLUMNS)


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


def _summary_recommendation(row: dict[str, object]) -> str:
    role = str(row["candidate_role"])
    return_strength = _return_strength(row)
    drawdown = _drawdown_control(row)
    if role == "return_reference":
        return "keep as return reference"
    if role == "drawdown_control_reference" and drawdown in {"strong", "moderate"}:
        return "keep as drawdown-control guardrail"
    if (
        role == "balanced_reference"
        and return_strength in {"strong", "moderate"}
        and drawdown != "weak"
    ):
        return "candidate to deep-dive further"
    return "monitor only"


def _decision_recommendation(
    *,
    role: str,
    return_strength: str,
    drawdown_control: str,
    cost_resilience: str,
    train_stability: str,
    test_stability: str,
    evidence_scope: str,
    average_return: float,
) -> str:
    stability_warning = train_stability == "weak" or test_stability == "weak"
    if role == "return_reference":
        return "keep as return reference"
    if role == "drawdown_control_reference":
        if drawdown_control in {"strong", "moderate"} and average_return > 0.0:
            return "keep as drawdown-control guardrail"
        return "monitor only"
    if role == "balanced_reference":
        if (
            return_strength in {"strong", "moderate"}
            and drawdown_control != "weak"
            and cost_resilience != "weak"
            and evidence_scope != "narrow evidence"
            and not stability_warning
        ):
            return "candidate to deep-dive further"
        if return_strength in {"strong", "moderate"} and drawdown_control != "weak":
            return "keep as balanced candidate"
    return "monitor only"


def _return_strength(row: pd.Series | dict[str, object]) -> str:
    average_return = _as_float(row["average_test_return"])
    positive_rate = _as_float(row["positive_window_rate"])
    if pd.isna(average_return) or pd.isna(positive_rate):
        return "unknown"
    if average_return >= 0.06 and positive_rate >= 0.65:
        return "strong"
    if average_return > 0.0 and positive_rate >= 0.50:
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


def _cost_resilience(
    metrics_10: dict[str, object],
    metrics_20: dict[str, object],
    cost_delta: float,
) -> str:
    avg_20 = _as_float(metrics_20["average_test_return"])
    positive_20 = _as_float(metrics_20["positive_window_rate"])
    avg_10 = _as_float(metrics_10["average_test_return"])
    if pd.isna(avg_20) or pd.isna(positive_20) or pd.isna(cost_delta):
        return "unknown"
    if avg_20 > 0.0 and positive_20 >= 0.55 and cost_delta >= -0.04:
        return "strong"
    if avg_20 > 0.0 and positive_20 >= 0.45 and avg_10 > 0.0 and cost_delta >= -0.08:
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


def _balanced_score(
    *,
    metrics: dict[str, object],
    role: str,
    cost_resilience: str,
    train_stability: str,
    test_stability: str,
) -> float:
    score = _as_float(metrics["average_test_return"])
    score += 0.04 * _as_float(metrics["positive_window_rate"])
    score += 0.02 * _as_float(metrics["average_sharpe_like"])
    score += 0.08 * max(_as_float(metrics["worst_max_drawdown"]), -1.0)
    score += {"strong": 0.015, "moderate": 0.0, "weak": -0.025}.get(cost_resilience, -0.005)
    score += {"strong": 0.01, "moderate": 0.0, "weak": -0.02}.get(train_stability, -0.005)
    score += {"strong": 0.01, "moderate": 0.0, "weak": -0.02}.get(test_stability, -0.005)
    if role == "balanced_reference":
        score += 0.015
    if role == "drawdown_control_reference":
        score += 0.005
    return float(score)


def _evidence_scope(window_count: object) -> str:
    count = int(_as_float(window_count)) if not pd.isna(_as_float(window_count)) else 0
    if count <= 5:
        return "narrow evidence"
    if count < 12:
        return "moderate multi-window evidence"
    return "broad multi-window evidence"


def _decision_rationale(
    *,
    return_strength: str,
    drawdown_control: str,
    cost_resilience: str,
    train_stability: str,
    test_stability: str,
    evidence_scope: str,
) -> str:
    return (
        f"return={return_strength}; drawdown={drawdown_control}; "
        f"cost={cost_resilience}; train_windows={train_stability}; "
        f"test_windows={test_stability}; evidence={evidence_scope}"
    )


def _next_action(recommendation: str) -> str:
    if recommendation == "candidate to deep-dive further":
        return "Deep-dive this family before treating it as the candidate to beat."
    if recommendation == "keep as balanced candidate":
        return "Keep in the balanced shortlist; inspect weak sensitivity before promotion."
    if recommendation == "keep as return reference":
        return "Keep for return-side comparison; do not promote by raw return alone."
    if recommendation == "keep as drawdown-control guardrail":
        return "Carry forward as the drawdown-control guardrail."
    return "Do not promote from this confirmation alone."


def _tree_screen_recommendation(decision: pd.DataFrame) -> str:
    balanced = decision.loc[decision["candidate_role"].eq("balanced_reference")]
    if balanced.empty:
        return "not yet; no balanced top3/top5 candidate was available."
    best = balanced.sort_values(
        by=["balanced_score", "average_test_return_10bps"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    if str(best["recommendation"]) == "candidate to deep-dive further":
        return "yes, after reviewing the focused diagnostics, run `tree_screen` as the next classical challenger."
    return "wait; keep tree_screen deferred until the balanced ridge direction clears stability and cost checks."


def _best_role(decision: pd.DataFrame, role: str) -> pd.Series:
    subset = decision.loc[decision["candidate_role"].eq(role)].copy()
    if subset.empty:
        subset = decision.copy()
    return subset.sort_values(
        by=["balanced_score", "average_test_return_10bps", "worst_max_drawdown_10bps"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]


def _sort_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        by=[
            "candidate_role",
            "average_test_return",
            "positive_window_rate",
            "worst_max_drawdown",
            "candidate_id",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _sort_decision_matrix(decision: pd.DataFrame) -> pd.DataFrame:
    role_order = {
        "balanced_reference": 0,
        "return_reference": 1,
        "drawdown_control_reference": 2,
    }
    output = decision.copy()
    output["_role_order"] = output["candidate_role"].map(role_order).fillna(99)
    output = output.sort_values(
        by=["_role_order", "balanced_score", "average_test_return_10bps"],
        ascending=[True, False, False],
        kind="mergesort",
    ).drop(columns=["_role_order"])
    return output.reset_index(drop=True)


def _candidate_key(values: dict[str, object]) -> str:
    return (
        f"{values['candidate_role']}:{values['model_type']}:top{int(values['top_k'])}:"
        f"hold{int(values['min_holding_days'])}:"
        f"margin{float(values['switch_margin']):g}"
    )


def _candidate_label(row: pd.Series) -> str:
    return (
        f"{row['model_type']} / top{int(row['top_k'])} / "
        f"hold {int(row['min_holding_days'])} / margin {float(row['switch_margin']):g}"
    )


def _spec_filter(frame: pd.DataFrame, values: pd.Series | dict[str, object]) -> pd.Series:
    mask = (
        frame["candidate_family"].astype(str).eq(str(_value(values, "candidate_family")))
        & frame["candidate_role"].astype(str).eq(str(_value(values, "candidate_role")))
        & frame["model_type"].astype(str).eq(str(_value(values, "model_type")))
        & _int_eq(frame, "top_k", int(_value(values, "top_k")))
        & _int_eq(frame, "min_holding_days", int(_value(values, "min_holding_days")))
        & _float_eq(frame, "switch_margin", float(_value(values, "switch_margin")))
    )
    return mask


def _value(values: pd.Series | dict[str, object], key: str) -> object:
    return values[key]


def _float_eq(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").sub(float(value)).abs().le(1e-9)


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


create_tiingo_focused_linear_ridge_confirmation_report_from_args = (
    run_tiingo_focused_linear_ridge_confirmation_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_focused_linear_ridge_confirmation_report_from_args())
