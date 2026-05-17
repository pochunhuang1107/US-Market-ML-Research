"""Create the Tiingo-only modern baseline sweep report."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_SOURCE_LABEL = "tiingo_only"
DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_modern_baseline_sweep/"
    "tiingo_modern_baseline_sweep_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_modern_baseline_sweep")

SUMMARY_PATH_NAME = "tiingo_modern_baseline_summary.csv"
DECISION_MATRIX_PATH_NAME = "tiingo_modern_baseline_decision_matrix.csv"
REPORT_PATH_NAME = "tiingo_modern_baseline_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

REQUIRED_RESULTS_COLUMNS = [
    "fold",
    "split",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "total_return",
    "sharpe_like",
    "max_drawdown",
]

SUMMARY_GROUP_COLUMNS = [
    "dataset_source",
    "sweep_profile",
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
    "cost_sensitivity",
    "smoothing_sensitivity",
    "train_window_stability",
    "test_window_stability",
    "recommendation",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "dataset_source",
    "sweep_profile",
    "model_family",
    "model_type",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "return_strength",
    "risk_adjusted_strength",
    "drawdown_risk",
    "cost_sensitivity",
    "smoothing_sensitivity",
    "train_window_stability",
    "test_window_stability",
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


@dataclass(frozen=True)
class TiingoModernBaselineReportData:
    """Tables and metadata produced by the Tiingo-only baseline report."""

    candidate_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    smoothing_sensitivity: pd.DataFrame
    train_window_stability: pd.DataFrame
    test_window_stability: pd.DataFrame
    result_row_count: int
    test_row_count: int
    included_test_row_count: int
    dataset_source: str = DATASET_SOURCE_LABEL


def run_tiingo_modern_baseline_sweep_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the Tiingo-only modern baseline sweep report."""

    args = _parse_args(argv)
    report_data = build_tiingo_modern_baseline_sweep_report(
        results_csv=args.results_csv,
    )
    outputs = write_tiingo_modern_baseline_sweep_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo-only modern baseline sweep report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.candidate_summary.head(20).to_string(index=False))
    return 0


def build_tiingo_modern_baseline_sweep_report(
    *,
    results_csv: Path,
) -> TiingoModernBaselineReportData:
    """Read completed Tiingo baseline sweep results and create report tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    test_rows = results.loc[results["split"].astype(str).eq("test")].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows were found in the results CSV.")

    enriched = _enrich_test_rows(test_rows)
    included = enriched.loc[enriched["model_family"].astype(str).eq("baseline")].copy()
    if included.empty:
        raise ValueError("No baseline test rows were found in the results CSV.")

    cost_sensitivity = _build_dimension_table(
        included,
        [
            "sweep_profile",
            "model_family",
            "model_type",
            "top_k",
            "allocation_mode",
            "transaction_cost_bps",
        ],
    )
    smoothing_sensitivity = _build_dimension_table(
        included,
        [
            "sweep_profile",
            "model_family",
            "model_type",
            "top_k",
            "min_holding_days",
            "switch_margin",
        ],
    )
    train_window_stability = _build_dimension_table(
        included,
        ["sweep_profile", "model_family", "model_type", "top_k", "train_window_years"],
    )
    test_window_stability = _build_dimension_table(
        included,
        ["sweep_profile", "model_family", "model_type", "top_k", "test_window_months"],
    )
    candidate_summary = _sort_candidate_summary(
        _build_candidate_summary(
            included,
            cost_sensitivity=cost_sensitivity,
            smoothing_sensitivity=smoothing_sensitivity,
            train_window_stability=train_window_stability,
            test_window_stability=test_window_stability,
        )
    )
    decision_matrix = _build_decision_matrix(candidate_summary)

    return TiingoModernBaselineReportData(
        candidate_summary=candidate_summary,
        decision_matrix=decision_matrix,
        cost_sensitivity=cost_sensitivity,
        smoothing_sensitivity=smoothing_sensitivity,
        train_window_stability=train_window_stability,
        test_window_stability=test_window_stability,
        result_row_count=int(len(results)),
        test_row_count=int(len(test_rows)),
        included_test_row_count=int(len(included)),
    )


def write_tiingo_modern_baseline_sweep_report(
    *,
    report_data: TiingoModernBaselineReportData,
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
        render_tiingo_modern_baseline_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_modern_baseline_markdown(
    report_data: TiingoModernBaselineReportData,
) -> str:
    """Render the Tiingo-only modern baseline sweep Markdown report."""

    best_return = report_data.candidate_summary.iloc[0]
    best_balanced = _best_balanced_candidate(report_data.candidate_summary)
    return_table = _format_for_markdown(
        report_data.candidate_summary.head(20)[
            [
                "candidate_label",
                "average_test_return",
                "median_test_return",
                "positive_window_rate",
                "average_sharpe_like",
                "worst_max_drawdown",
                "recommendation",
            ]
        ],
        exclude={"candidate_label", "recommendation"},
    )
    balanced_table = _format_for_markdown(
        _sort_balanced(report_data.candidate_summary).head(20)[
            [
                "candidate_label",
                "average_test_return",
                "positive_window_rate",
                "average_sharpe_like",
                "worst_max_drawdown",
                "cost_sensitivity",
                "smoothing_sensitivity",
                "recommendation",
            ]
        ],
        exclude={"candidate_label", "cost_sensitivity", "smoothing_sensitivity", "recommendation"},
    )
    drawdown_table = _format_for_markdown(
        report_data.candidate_summary.sort_values(
            by=["worst_max_drawdown", "average_test_return"],
            ascending=[False, False],
            kind="mergesort",
        ).head(20)[
            [
                "candidate_label",
                "average_test_return",
                "average_sharpe_like",
                "average_max_drawdown",
                "worst_max_drawdown",
            ]
        ],
        exclude={"candidate_label"},
    )

    lines = [
        "# Tiingo Modern Baseline Sweep Report",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the Tiingo-only modern classical baseline sweep. "
        "The research source label is `tiingo_only`; the sweep is local-only, "
        "baseline-only, and intentionally excludes neural networks.",
        "",
        f"Input result rows: `{report_data.result_row_count}`. Test rows: "
        f"`{report_data.test_row_count}`. Included baseline test rows: "
        f"`{report_data.included_test_row_count}`.",
        "",
        f"Best classical baseline candidate by average test return: "
        f"`{best_return['candidate_label']}`.",
        "",
        f"Best overall candidate by balanced risk/return score: "
        f"`{best_balanced['candidate_label']}`.",
        "",
        "## Best Candidates By Return",
        "",
        _markdown_table(return_table),
        "",
        "## Best Candidates By Balanced Risk/Return",
        "",
        _markdown_table(balanced_table),
        "",
        "## Drawdown-Sensitive Candidates",
        "",
        _markdown_table(drawdown_table),
        "",
        "## Cost Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.cost_sensitivity.head(40),
                exclude={"model_family", "model_type", "allocation_mode"},
            )
        ),
        "",
        "## Smoothing Sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.smoothing_sensitivity.head(40),
                exclude={"model_family", "model_type"},
            )
        ),
        "",
        "## Window Stability",
        "",
        "Train-window stability:",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.train_window_stability.head(40),
                exclude={"model_family", "model_type"},
            )
        ),
        "",
        "Test-window stability:",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.test_window_stability.head(40),
                exclude={"model_family", "model_type"},
            )
        ),
        "",
        "## Recommended Baseline Shortlist For Deep Dive",
        "",
        _markdown_table(
            report_data.decision_matrix.loc[
                report_data.decision_matrix["recommendation"].isin(
                    ["promote for deep dive", "keep as reference"]
                ),
                ["candidate_label", "recommendation", "rationale"],
            ].head(25)
        ),
        "",
        "## MLP Deferral Note",
        "",
        "MLP is intentionally deferred until after this Tiingo-only baseline run "
        "identifies competitive classical candidates.",
        "",
        "## Caveats",
        "",
        "- Tiingo-only research source.",
        "- Massive app data may not match research results exactly.",
        "- Still not production-ready.",
        "- ETF survivorship bias is reduced by dynamic eligibility but not fully "
        "eliminated unless the dataset includes delisted/closed ETFs.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo-only modern baseline sweep report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _enrich_test_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    output = test_rows.copy()
    output["dataset_source"] = DATASET_SOURCE_LABEL
    if "sweep_profile" not in output.columns:
        output["sweep_profile"] = "unknown"
    else:
        output["sweep_profile"] = output["sweep_profile"].fillna("unknown")
    for column in [
        "top_k",
        "min_holding_days",
        "switch_margin",
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
    output["min_holding_days"] = output["min_holding_days"].astype("Int64")
    output["train_window_years"] = output["train_window_years"].astype("Int64")
    output["validation_window_months"] = output["validation_window_months"].astype("Int64")
    output["test_window_months"] = output["test_window_months"].astype("Int64")
    output["candidate_id"] = output.apply(_candidate_id_from_row, axis=1)
    output["candidate_label"] = output.apply(_candidate_label_from_row, axis=1)
    return output.reset_index(drop=True)


def _build_candidate_summary(
    test_rows: pd.DataFrame,
    *,
    cost_sensitivity: pd.DataFrame,
    smoothing_sensitivity: pd.DataFrame,
    train_window_stability: pd.DataFrame,
    test_window_stability: pd.DataFrame,
) -> pd.DataFrame:
    family_metrics = _build_family_metrics(
        cost_sensitivity=cost_sensitivity,
        smoothing_sensitivity=smoothing_sensitivity,
        train_window_stability=train_window_stability,
        test_window_stability=test_window_stability,
    )
    rows = []
    for group_values, group in test_rows.groupby(SUMMARY_GROUP_COLUMNS, dropna=False, sort=False):
        first = group.iloc[0]
        family_key = _family_key(first)
        row = {
            "candidate_id": first["candidate_id"],
            "candidate_label": first["candidate_label"],
            **dict(zip(SUMMARY_GROUP_COLUMNS, group_values, strict=True)),
            "window_count": int(len(group)),
            "average_test_return": group["total_return"].mean(),
            "median_test_return": group["total_return"].median(),
            "positive_window_rate": _positive_rate(group["total_return"]),
            "average_sharpe_like": group["sharpe_like"].mean(),
            "average_max_drawdown": group["max_drawdown"].mean(),
            "worst_max_drawdown": group["max_drawdown"].min(),
            **family_metrics.get(family_key, {}),
        }
        row.setdefault("cost_sensitivity", "unknown")
        row.setdefault("smoothing_sensitivity", "unknown")
        row.setdefault("train_window_stability", "unknown")
        row.setdefault("test_window_stability", "unknown")
        row["recommendation"] = _recommendation(row)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _build_decision_matrix(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidate_summary.iterrows():
        return_strength = _return_strength(row)
        risk_adjusted_strength = _risk_adjusted_strength(row)
        drawdown_risk = _drawdown_risk(row)
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_label": row["candidate_label"],
                "dataset_source": row["dataset_source"],
                "sweep_profile": row["sweep_profile"],
                "model_family": row["model_family"],
                "model_type": row["model_type"],
                "top_k": int(row["top_k"]),
                "min_holding_days": int(row["min_holding_days"]),
                "switch_margin": float(row["switch_margin"]),
                "transaction_cost_bps": float(row["transaction_cost_bps"]),
                "train_window_years": int(row["train_window_years"]),
                "validation_window_months": int(row["validation_window_months"]),
                "test_window_months": int(row["test_window_months"]),
                "return_strength": return_strength,
                "risk_adjusted_strength": risk_adjusted_strength,
                "drawdown_risk": drawdown_risk,
                "cost_sensitivity": row["cost_sensitivity"],
                "smoothing_sensitivity": row["smoothing_sensitivity"],
                "train_window_stability": row["train_window_stability"],
                "test_window_stability": row["test_window_stability"],
                "recommendation": row["recommendation"],
                "rationale": _decision_rationale(
                    return_strength=return_strength,
                    risk_adjusted_strength=risk_adjusted_strength,
                    drawdown_risk=drawdown_risk,
                    cost_sensitivity=str(row["cost_sensitivity"]),
                    smoothing_sensitivity=str(row["smoothing_sensitivity"]),
                    train_window_stability=str(row["train_window_stability"]),
                    test_window_stability=str(row["test_window_stability"]),
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def _build_dimension_table(test_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = (
        test_rows.groupby(group_cols, dropna=False, sort=True)
        .agg(
            window_count=(
                ("run_key", "count")
                if "run_key" in test_rows.columns
                else ("fold", "count")
            ),
            average_test_return=("total_return", "mean"),
            median_test_return=("total_return", "median"),
            positive_window_rate=("total_return", _positive_rate),
            average_sharpe_like=("sharpe_like", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
    )
    return table.sort_values(
        by=[*group_cols, "average_test_return"],
        ascending=[True] * len(group_cols) + [False],
        kind="mergesort",
    )


def _build_family_metrics(
    *,
    cost_sensitivity: pd.DataFrame,
    smoothing_sensitivity: pd.DataFrame,
    train_window_stability: pd.DataFrame,
    test_window_stability: pd.DataFrame,
) -> dict[tuple[str, str, str, int], dict[str, str]]:
    output: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for frame, metric_name in [
        (cost_sensitivity, "cost_sensitivity"),
        (smoothing_sensitivity, "smoothing_sensitivity"),
        (train_window_stability, "train_window_stability"),
        (test_window_stability, "test_window_stability"),
    ]:
        for key, group in frame.groupby(
            ["sweep_profile", "model_family", "model_type", "top_k"],
            sort=False,
        ):
            family_key = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
            output.setdefault(family_key, {})[metric_name] = _sensitivity_rating(group)
    return output


def _sensitivity_rating(group: pd.DataFrame) -> str:
    if group.empty or group["average_test_return"].isna().all():
        return "unknown"
    return_range = float(group["average_test_return"].max() - group["average_test_return"].min())
    min_positive = float(group["positive_window_rate"].min())
    if return_range <= 0.015 and min_positive >= 0.55:
        return "strong"
    if return_range <= 0.04 and min_positive >= 0.45:
        return "moderate"
    return "weak"


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


def _sort_balanced(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["balanced_score"] = output.apply(_balanced_score, axis=1)
    return output.sort_values(
        by=["balanced_score", "average_test_return", "worst_max_drawdown"],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _best_balanced_candidate(summary: pd.DataFrame) -> pd.Series:
    return _sort_balanced(summary).iloc[0]


def _balanced_score(row: pd.Series) -> float:
    score = _as_float(row["average_test_return"])
    score += 0.02 * _as_float(row["positive_window_rate"])
    score += 0.01 * _as_float(row["average_sharpe_like"])
    score += 0.05 * max(_as_float(row["worst_max_drawdown"]), -1.0)
    for column in [
        "cost_sensitivity",
        "smoothing_sensitivity",
        "train_window_stability",
        "test_window_stability",
    ]:
        score += {"strong": 0.01, "moderate": 0.0, "weak": -0.02}.get(str(row[column]), -0.005)
    return float(score)


def _recommendation(row: dict[str, object]) -> str:
    return_strength = _return_strength(row)
    risk_strength = _risk_adjusted_strength(row)
    drawdown = _drawdown_risk(row)
    weak_sensitivity_count = sum(
        str(row[column]) == "weak"
        for column in [
            "cost_sensitivity",
            "smoothing_sensitivity",
            "train_window_stability",
            "test_window_stability",
        ]
    )
    if return_strength == "strong" and risk_strength != "weak" and drawdown != "weak":
        if weak_sensitivity_count <= 1:
            return "promote for deep dive"
    if return_strength in {"strong", "moderate"} and drawdown != "weak":
        return "keep as reference"
    if return_strength == "weak" and drawdown == "weak":
        return "reject/deprioritize"
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


def _risk_adjusted_strength(row: pd.Series | dict[str, object]) -> str:
    sharpe_like = _as_float(row["average_sharpe_like"])
    if pd.isna(sharpe_like):
        return "unknown"
    if sharpe_like >= 0.75:
        return "strong"
    if sharpe_like >= 0.25:
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


def _decision_rationale(
    *,
    return_strength: str,
    risk_adjusted_strength: str,
    drawdown_risk: str,
    cost_sensitivity: str,
    smoothing_sensitivity: str,
    train_window_stability: str,
    test_window_stability: str,
) -> str:
    return (
        f"return={return_strength}; risk_adjusted={risk_adjusted_strength}; "
        f"drawdown={drawdown_risk}; cost={cost_sensitivity}; "
        f"smoothing={smoothing_sensitivity}; train_windows={train_window_stability}; "
        f"test_windows={test_window_stability}"
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
    profile = str(row.get("sweep_profile", "unknown"))
    profile_part = "" if profile == "unknown" else f"profile:{profile}:"
    return (
        f"{profile_part}{row['model_family']}:{row['model_type']}:top{int(row['top_k'])}:"
        f"hold{int(row['min_holding_days'])}:margin{float(row['switch_margin']):g}:"
        f"cost{float(row['transaction_cost_bps']):g}:tw{int(row['train_window_years'])}:"
        f"v{int(row['validation_window_months'])}:t{int(row['test_window_months'])}"
    )


def _candidate_label_from_row(row: pd.Series) -> str:
    return (
        f"{row['model_type']} / top{int(row['top_k'])} / "
        f"hold {int(row['min_holding_days'])} / margin {float(row['switch_margin']):g} / "
        f"cost {float(row['transaction_cost_bps']):g}bps / "
        f"tw{int(row['train_window_years'])}y-v{int(row['validation_window_months'])}m-"
        f"t{int(row['test_window_months'])}m"
    )


def _family_key(row: pd.Series) -> tuple[str, str, str, int]:
    return (
        str(row["sweep_profile"]),
        str(row["model_family"]),
        str(row["model_type"]),
        int(row["top_k"]),
    )


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0).mean())


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


create_tiingo_modern_baseline_sweep_report_from_args = (
    run_tiingo_modern_baseline_sweep_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_modern_baseline_sweep_report_from_args())
