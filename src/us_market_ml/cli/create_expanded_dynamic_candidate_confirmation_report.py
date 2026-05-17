"""Create the expanded dynamic candidate confirmation report."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/expanded_dynamic_candidate_confirmation_study/"
    "expanded_dynamic_candidate_confirmation_study_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/expanded_dynamic_candidate_confirmation")

SUMMARY_PATH_NAME = "expanded_dynamic_candidate_summary.csv"
DECISION_MATRIX_PATH_NAME = "expanded_dynamic_candidate_decision_matrix.csv"
REPORT_PATH_NAME = "expanded_dynamic_candidate_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

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

SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "role_label",
    "model_type",
    "top_k",
    "allocation_mode",
    "window_count",
    "average_test_return",
    "median_test_return",
    "positive_window_rate",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_test_return_cost10",
    "average_test_return_cost20",
    "cost20_minus_cost10",
    "train_window_count",
    "min_train_window_positive_rate",
    "train_window_mean_return_range",
    "test_window_count",
    "min_test_window_positive_rate",
    "test_window_mean_return_range",
    "spy_mean_total_return",
    "strategy_beats_spy_rate",
    "mean_strategy_minus_spy",
    "qqq_mean_total_return",
    "strategy_beats_qqq_rate",
    "mean_strategy_minus_qqq",
    "recommendation",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "role_label",
    "return_strength",
    "train_window_consistency",
    "test_window_consistency",
    "cost_sensitivity",
    "drawdown_risk",
    "benchmark_relative_behavior",
    "recommendation",
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

CANDIDATE_ROLE_LABELS = {
    ("linear_regression", 1): "dynamic version of fixed-fold return/reference/champion family",
    ("linear_regression", 3): "champion-family diversified reference",
    ("ridge", 3): "balanced reference",
    ("ridge", 5): "longer-window balanced candidate",
}


@dataclass(frozen=True)
class ExpandedDynamicCandidateConfirmationReportData:
    """Tables and metadata produced by the expanded dynamic confirmation report."""

    candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    by_train_window: pd.DataFrame
    by_test_window: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    benchmark_columns_available: bool
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    excluded_test_row_count: int


def run_expanded_dynamic_candidate_confirmation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the expanded dynamic candidate confirmation report."""

    args = _parse_args(argv)
    report_data = build_expanded_dynamic_candidate_confirmation_report(
        results_csv=args.results_csv,
    )
    outputs = write_expanded_dynamic_candidate_confirmation_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML expanded dynamic candidate confirmation report")
    print(f"results CSV: {args.results_csv}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.candidate_summary.to_string(index=False))
    return 0


def build_expanded_dynamic_candidate_confirmation_report(
    *,
    results_csv: Path,
) -> ExpandedDynamicCandidateConfirmationReportData:
    """Read completed expanded dynamic study results and create report tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _enrich_test_rows(test_rows)
    included = _filter_intended_candidates(enriched)
    if included.empty:
        raise ValueError("No test rows matched the expanded dynamic candidate specs.")

    candidate_summary = _sort_candidate_summary(_build_candidate_summary(included))
    decision_matrix = _build_decision_matrix(candidate_summary)
    by_train_window = _build_window_stability(included, window_col="train_window_years")
    by_test_window = _build_window_stability(included, window_col="test_window_months")
    cost_sensitivity = _build_cost_sensitivity(included)

    return ExpandedDynamicCandidateConfirmationReportData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        by_train_window=by_train_window,
        by_test_window=by_test_window,
        cost_sensitivity=cost_sensitivity,
        benchmark_columns_available=_has_benchmark_columns(included),
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
        excluded_test_row_count=int(len(test_rows) - len(included)),
    )


def write_expanded_dynamic_candidate_confirmation_report(
    *,
    report_data: ExpandedDynamicCandidateConfirmationReportData,
    output_dir: Path,
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
        render_expanded_dynamic_candidate_confirmation_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_expanded_dynamic_candidate_confirmation_markdown(
    report_data: ExpandedDynamicCandidateConfirmationReportData,
) -> str:
    """Render the expanded dynamic candidate confirmation report."""

    leader = report_data.candidate_summary.iloc[0]
    benchmark_note = (
        "Benchmark-relative columns were available and are included in the decision matrix."
        if report_data.benchmark_columns_available
        else "Benchmark-relative columns were not available in the results CSV."
    )
    ranking_table = _format_for_markdown(
        report_data.candidate_summary[
            [
                "candidate_label",
                "role_label",
                "average_test_return",
                "positive_window_rate",
                "average_sharpe_like",
                "worst_max_drawdown",
                "cost20_minus_cost10",
                "recommendation",
            ]
        ],
        exclude={"candidate_label", "role_label", "recommendation"},
    )
    drawdown_table = _format_for_markdown(
        report_data.candidate_summary[
            [
                "candidate_label",
                "average_max_drawdown",
                "worst_max_drawdown",
                "average_sharpe_like",
            ]
        ],
        exclude={"candidate_label"},
    )

    lines = [
        "# Expanded Dynamic Candidate Confirmation Report",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the focused dynamic-universe candidate confirmation "
        "study. It uses only the intended baseline candidate directions, equal-weight "
        "allocation, fold-train feature selection, and point-in-time dynamic universe "
        "eligibility.",
        "",
        f"The current ranking leader by average test return is "
        f"`{leader['candidate_label']}`. {benchmark_note}",
        "",
        f"Input result rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included candidate test rows: "
        f"`{report_data.included_test_row_count}`. Excluded test rows: "
        f"`{report_data.excluded_test_row_count}`.",
        "",
        "## Candidate Ranking Table",
        "",
        _markdown_table(ranking_table),
        "",
        "## Decision Matrix",
        "",
        _markdown_table(report_data.decision_matrix),
        "",
        "## Window Stability",
        "",
        "Train-window consistency:",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.by_train_window,
                exclude={"candidate_label", "train_window_years"},
            )
        ),
        "",
        "Test-window consistency:",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.by_test_window,
                exclude={"candidate_label", "test_window_months"},
            )
        ),
        "",
        "## Cost Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.cost_sensitivity,
                exclude={"candidate_label", "transaction_cost_bps"},
            )
        ),
        "",
        "## Drawdown",
        "",
        _markdown_table(drawdown_table),
        "",
        "## Recommendation",
        "",
        _markdown_table(
            report_data.decision_matrix[
                ["candidate_label", "role_label", "recommendation", "rationale"]
            ]
        ),
        "",
        "Recommendation labels are intentionally research-stage only: keep as "
        "reference, promote for further deep dive, reject/deprioritize, or "
        "inconclusive.",
        "",
        "## Caveat",
        "",
        "This is still based on the current extended ETF dataset and does not fully "
        "solve historical ETF survivorship bias.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the expanded dynamic candidate confirmation report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _enrich_test_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    output = test_rows.copy()
    for column in [
        "top_k",
        "transaction_cost_bps",
        "total_return",
        "sharpe_like",
        "max_drawdown",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    for metadata_column in [
        "train_window_years",
        "validation_window_months",
        "test_window_months",
        "sequence_number",
    ]:
        if metadata_column not in output.columns:
            output[metadata_column] = np.nan

    parsed_metadata = output["fold"].astype(str).map(_parse_fold_metadata)
    for index, metadata in parsed_metadata.items():
        if metadata is None:
            continue
        for column, value in metadata.items():
            if column in output.columns and pd.isna(output.at[index, column]):
                output.at[index, column] = value

    output["top_k"] = output["top_k"].astype("Int64")
    output["candidate_id"] = output.apply(_candidate_id_from_row, axis=1)
    output["candidate_label"] = output.apply(_candidate_label_from_row, axis=1)
    output["role_label"] = output.apply(_role_label_from_row, axis=1)
    output = _add_benchmark_relative_columns(output)
    return output


def _filter_intended_candidates(enriched: pd.DataFrame) -> pd.DataFrame:
    pairs = set(CANDIDATE_ROLE_LABELS)
    mask = enriched.apply(
        lambda row: (str(row["model_type"]), int(row["top_k"])) in pairs
        and str(row["allocation_mode"]) == "equal_weight",
        axis=1,
    )
    return enriched.loc[mask].copy().reset_index(drop=True)


def _build_candidate_summary(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in test_rows.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        train_consistency = _consistency_metrics(group, "train_window_years")
        test_consistency = _consistency_metrics(group, "test_window_months")
        cost_metrics = _candidate_cost_metrics(group)
        benchmark_metrics = _candidate_benchmark_metrics(group)
        summary_row = {
            "candidate_id": first["candidate_id"],
            "candidate_label": first["candidate_label"],
            "role_label": first["role_label"],
            "model_type": first["model_type"],
            "top_k": int(first["top_k"]),
            "allocation_mode": first["allocation_mode"],
            "window_count": int(len(group)),
            "average_test_return": group["total_return"].mean(),
            "median_test_return": group["total_return"].median(),
            "positive_window_rate": _positive_rate(group["total_return"]),
            "average_sharpe_like": group["sharpe_like"].mean(),
            "average_max_drawdown": group["max_drawdown"].mean(),
            "worst_max_drawdown": group["max_drawdown"].min(),
            **cost_metrics,
            **train_consistency,
            **test_consistency,
            **benchmark_metrics,
        }
        summary_row["recommendation"] = _recommendation(summary_row)
        rows.append(summary_row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _build_decision_matrix(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidate_summary.iterrows():
        return_strength = _return_strength(row)
        train_consistency = _window_consistency_rating(
            row,
            min_positive_rate_col="min_train_window_positive_rate",
            return_range_col="train_window_mean_return_range",
        )
        test_consistency = _window_consistency_rating(
            row,
            min_positive_rate_col="min_test_window_positive_rate",
            return_range_col="test_window_mean_return_range",
        )
        cost_sensitivity = _cost_sensitivity_rating(row)
        drawdown_risk = _drawdown_risk(row)
        benchmark_behavior = _benchmark_relative_behavior(row)
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_label": row["candidate_label"],
                "role_label": row["role_label"],
                "return_strength": return_strength,
                "train_window_consistency": train_consistency,
                "test_window_consistency": test_consistency,
                "cost_sensitivity": cost_sensitivity,
                "drawdown_risk": drawdown_risk,
                "benchmark_relative_behavior": benchmark_behavior,
                "recommendation": row["recommendation"],
                "rationale": _decision_rationale(
                    return_strength=return_strength,
                    train_consistency=train_consistency,
                    test_consistency=test_consistency,
                    cost_sensitivity=cost_sensitivity,
                    drawdown_risk=drawdown_risk,
                    benchmark_behavior=benchmark_behavior,
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def _build_window_stability(test_rows: pd.DataFrame, *, window_col: str) -> pd.DataFrame:
    label_col = "train_window_years" if window_col == "train_window_years" else "test_window_months"
    available = test_rows.dropna(subset=[window_col]).copy()
    if available.empty:
        return pd.DataFrame(columns=["candidate_label", label_col, "window_count"])

    rows = []
    for (candidate_id, window_value), group in available.groupby(
        ["candidate_id", window_col],
        sort=True,
    ):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": first["candidate_label"],
                label_col: int(window_value),
                "window_count": int(len(group)),
                "average_test_return": group["total_return"].mean(),
                "median_test_return": group["total_return"].median(),
                "positive_window_rate": _positive_rate(group["total_return"]),
                "average_sharpe_like": group["sharpe_like"].mean(),
                "worst_max_drawdown": group["max_drawdown"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["candidate_label", label_col],
        kind="mergesort",
    )


def _build_cost_sensitivity(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, cost), group in test_rows.groupby(
        ["candidate_id", "transaction_cost_bps"],
        sort=True,
    ):
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": first["candidate_label"],
                "transaction_cost_bps": float(cost),
                "window_count": int(len(group)),
                "average_test_return": group["total_return"].mean(),
                "median_test_return": group["total_return"].median(),
                "positive_window_rate": _positive_rate(group["total_return"]),
                "average_sharpe_like": group["sharpe_like"].mean(),
                "worst_max_drawdown": group["max_drawdown"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["candidate_label", "transaction_cost_bps"],
        kind="mergesort",
    )


def _candidate_cost_metrics(group: pd.DataFrame) -> dict[str, float]:
    cost_means = (
        group.groupby("transaction_cost_bps", sort=True)["total_return"]
        .mean()
        .to_dict()
    )
    cost10 = _lookup_cost(cost_means, 10.0)
    cost20 = _lookup_cost(cost_means, 20.0)
    return {
        "average_test_return_cost10": cost10,
        "average_test_return_cost20": cost20,
        "cost20_minus_cost10": (
            float(cost20) - float(cost10)
            if not pd.isna(cost10) and not pd.isna(cost20)
            else float("nan")
        ),
    }


def _consistency_metrics(group: pd.DataFrame, column: str) -> dict[str, float | int]:
    prefix = "train" if column == "train_window_years" else "test"
    available = group.dropna(subset=[column]).copy()
    if available.empty:
        return {
            f"{prefix}_window_count": 0,
            f"min_{prefix}_window_positive_rate": float("nan"),
            f"{prefix}_window_mean_return_range": float("nan"),
        }
    by_window = available.groupby(column, sort=True)["total_return"].agg(
        mean_total_return="mean",
        positive_window_rate=_positive_rate,
    )
    return {
        f"{prefix}_window_count": int(len(by_window)),
        f"min_{prefix}_window_positive_rate": float(
            by_window["positive_window_rate"].min()
        ),
        f"{prefix}_window_mean_return_range": float(
            by_window["mean_total_return"].max() - by_window["mean_total_return"].min()
        ),
    }


def _candidate_benchmark_metrics(group: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for symbol in ("spy", "qqq"):
        output[f"{symbol}_mean_total_return"] = group[f"{symbol}_total_return"].mean()
        output[f"strategy_beats_{symbol}_rate"] = _positive_rate(
            group[f"strategy_beats_{symbol}"].astype(float)
        )
        output[f"mean_strategy_minus_{symbol}"] = group[f"strategy_minus_{symbol}"].mean()
    return output


def _add_benchmark_relative_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    lower_columns = {column.lower(): column for column in output.columns}
    for symbol in ("spy", "qqq"):
        benchmark_col = lower_columns.get(f"{symbol}_total_return")
        minus_col = lower_columns.get(f"strategy_minus_{symbol}")
        beats_col = lower_columns.get(f"strategy_beats_{symbol}")

        if benchmark_col is not None:
            output[f"{symbol}_total_return"] = pd.to_numeric(
                output[benchmark_col],
                errors="coerce",
            )
        else:
            output[f"{symbol}_total_return"] = np.nan

        if minus_col is not None:
            output[f"strategy_minus_{symbol}"] = pd.to_numeric(
                output[minus_col],
                errors="coerce",
            )
        else:
            output[f"strategy_minus_{symbol}"] = (
                output["total_return"] - output[f"{symbol}_total_return"]
            )

        if beats_col is not None:
            output[f"strategy_beats_{symbol}"] = _coerce_boolish(output[beats_col])
        else:
            output[f"strategy_beats_{symbol}"] = np.where(
                output[f"strategy_minus_{symbol}"].notna(),
                output[f"strategy_minus_{symbol}"] > 0.0,
                np.nan,
            )
    return output


def _sort_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        by=[
            "average_test_return",
            "positive_window_rate",
            "average_sharpe_like",
            "worst_max_drawdown",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _recommendation(row: dict[str, object]) -> str:
    return_strength = _return_strength(row)
    test_consistency = _window_consistency_rating(
        row,
        min_positive_rate_col="min_test_window_positive_rate",
        return_range_col="test_window_mean_return_range",
    )
    drawdown = _drawdown_risk(row)
    role = str(row["role_label"])

    if return_strength == "weak" and float(row["positive_window_rate"]) < 0.40:
        return "reject/deprioritize"
    if (
        role == CANDIDATE_ROLE_LABELS[("ridge", 5)]
        and return_strength in {"strong", "moderate"}
        and test_consistency != "weak"
        and drawdown != "weak"
    ):
        return "promote for further deep dive"
    if return_strength == "strong" and test_consistency != "weak" and drawdown != "weak":
        return "promote for further deep dive"
    if "reference" in role or "champion" in role:
        return "keep as reference"
    return "inconclusive"


def _return_strength(row: pd.Series | dict[str, object]) -> str:
    average_return = _as_float(row["average_test_return"])
    positive_rate = _as_float(row["positive_window_rate"])
    if pd.isna(average_return) or pd.isna(positive_rate):
        return "unknown"
    if average_return > 0.03 and positive_rate >= 0.55:
        return "strong"
    if average_return > 0.0 and positive_rate >= 0.45:
        return "moderate"
    return "weak"


def _window_consistency_rating(
    row: pd.Series | dict[str, object],
    *,
    min_positive_rate_col: str,
    return_range_col: str,
) -> str:
    min_positive = _as_float(row[min_positive_rate_col])
    return_range = _as_float(row[return_range_col])
    if pd.isna(min_positive) or pd.isna(return_range):
        return "unknown"
    if min_positive >= 0.55 and return_range <= 0.05:
        return "strong"
    if min_positive >= 0.45:
        return "moderate"
    return "weak"


def _cost_sensitivity_rating(row: pd.Series | dict[str, object]) -> str:
    delta = _as_float(row["cost20_minus_cost10"])
    if pd.isna(delta):
        return "unknown"
    if delta >= -0.005:
        return "strong"
    if delta >= -0.02:
        return "moderate"
    return "weak"


def _drawdown_risk(row: pd.Series | dict[str, object]) -> str:
    worst = _as_float(row["worst_max_drawdown"])
    if pd.isna(worst):
        return "unknown"
    if worst >= -0.15:
        return "strong"
    if worst >= -0.35:
        return "moderate"
    return "weak"


def _benchmark_relative_behavior(row: pd.Series | dict[str, object]) -> str:
    spy_rate = _as_float(row["strategy_beats_spy_rate"])
    qqq_rate = _as_float(row["strategy_beats_qqq_rate"])
    spy_delta = _as_float(row["mean_strategy_minus_spy"])
    qqq_delta = _as_float(row["mean_strategy_minus_qqq"])
    values = [spy_rate, qqq_rate, spy_delta, qqq_delta]
    if all(pd.isna(value) for value in values):
        return "unknown"
    if spy_rate >= 0.55 and qqq_rate >= 0.55 and spy_delta > 0 and qqq_delta > 0:
        return "strong"
    if spy_rate >= 0.45 and qqq_rate >= 0.45:
        return "moderate"
    return "weak"


def _decision_rationale(
    *,
    return_strength: str,
    train_consistency: str,
    test_consistency: str,
    cost_sensitivity: str,
    drawdown_risk: str,
    benchmark_behavior: str,
) -> str:
    return (
        f"return={return_strength}; train_windows={train_consistency}; "
        f"test_windows={test_consistency}; cost={cost_sensitivity}; "
        f"drawdown={drawdown_risk}; benchmark_relative={benchmark_behavior}"
    )


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
    }


def _candidate_id_from_row(row: pd.Series) -> str:
    return (
        f"{row['model_type']}:top{int(row['top_k'])}:"
        f"{row['allocation_mode']}"
    )


def _candidate_label_from_row(row: pd.Series) -> str:
    return f"{row['model_type']} / top{int(row['top_k'])} / {row['allocation_mode']}"


def _role_label_from_row(row: pd.Series) -> str:
    return CANDIDATE_ROLE_LABELS.get(
        (str(row["model_type"]), int(row["top_k"])),
        "outside expanded dynamic candidate plan",
    )


def _lookup_cost(cost_means: dict[object, object], cost: float) -> float:
    for key, value in cost_means.items():
        if np.isclose(float(key), cost, rtol=0.0, atol=1e-9):
            return float(value)
    return float("nan")


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0).mean())


def _coerce_boolish(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(float)
    text = values.astype(str).str.lower()
    mapped = text.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    numeric = pd.to_numeric(values, errors="coerce")
    return mapped.fillna(numeric)


def _has_benchmark_columns(frame: pd.DataFrame) -> bool:
    benchmark_cols = [
        "spy_total_return",
        "qqq_total_return",
        "strategy_minus_spy",
        "strategy_minus_qqq",
    ]
    return any(frame[column].notna().any() for column in benchmark_cols)


def _validate_columns(frame: pd.DataFrame, required_columns: Sequence[str], *, path: Path) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Report input {path} is missing required columns: {missing}")


def _format_for_markdown(frame: pd.DataFrame, *, exclude: set[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column in exclude:
            continue
        output[column] = output[column].map(_format_value)
    return output


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    separator = ["---" for _ in headers]
    table_rows = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


create_expanded_dynamic_candidate_confirmation_report_from_args = (
    run_expanded_dynamic_candidate_confirmation_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_expanded_dynamic_candidate_confirmation_report_from_args())
