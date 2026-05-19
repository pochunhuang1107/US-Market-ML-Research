"""Create the long-history QQQ-relative model-zoo report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/long_history_qqq_relative_model_zoo/"
    "long_history_qqq_relative_model_zoo_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/long_history_qqq_relative_model_zoo")

SUMMARY_PATH_NAME = "long_history_qqq_relative_model_zoo_summary.csv"
DECISION_MATRIX_PATH_NAME = "long_history_qqq_relative_model_zoo_decision_matrix.csv"
REPORT_PATH_NAME = "long_history_qqq_relative_model_zoo_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)

REQUIRED_RESULTS_COLUMNS = [
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_like",
    "return_to_drawdown",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "drawdown_improvement_vs_qqq",
]

SUMMARY_COLUMNS = [
    *REQUIRED_RESULTS_COLUMNS,
    "decision",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "average_turnover",
    "decision",
    "rationale",
]


@dataclass(frozen=True)
class LongHistoryModelZooReportData:
    """Tables and metadata produced by the long-history model-zoo report."""

    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    result_row_count: int
    candidate_count: int
    qqq_beat_count: int
    return_to_drawdown_beat_count: int


def run_long_history_qqq_relative_model_zoo_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the long-history model-zoo report."""

    args = _parse_args(argv)
    report_data = build_long_history_qqq_relative_model_zoo_report(
        results_csv=args.results_csv,
    )
    outputs = write_long_history_qqq_relative_model_zoo_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )
    print("US Market ML long-history QQQ-relative model-zoo report")
    print(f"results CSV: {args.results_csv}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.decision_matrix.head(20).to_string(index=False))
    return 0


def build_long_history_qqq_relative_model_zoo_report(
    *,
    results_csv: Path,
) -> LongHistoryModelZooReportData:
    """Read model-zoo results and create summary/decision tables."""

    results = pd.read_csv(results_csv)
    _validate_columns(results, REQUIRED_RESULTS_COLUMNS, path=results_csv)
    if results.empty:
        raise ValueError("No result rows were found in the model-zoo results CSV.")

    summary = results.copy()
    for column in REQUIRED_RESULTS_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    summary["beats_qqq_return"] = _as_bool(summary["beats_qqq_return"])
    summary["beats_qqq_return_to_drawdown"] = _as_bool(
        summary["beats_qqq_return_to_drawdown"]
    )
    summary["decision"] = summary.apply(classify_long_history_decision, axis=1)
    summary = _sort_summary(summary.loc[:, SUMMARY_COLUMNS])
    decision_matrix = _build_decision_matrix(summary)

    return LongHistoryModelZooReportData(
        summary=summary,
        decision_matrix=decision_matrix,
        result_row_count=int(len(results)),
        candidate_count=int(len(summary)),
        qqq_beat_count=int(summary["beats_qqq_return"].sum()),
        return_to_drawdown_beat_count=int(summary["beats_qqq_return_to_drawdown"].sum()),
    )


def write_long_history_qqq_relative_model_zoo_report(
    *,
    report_data: LongHistoryModelZooReportData,
    output_dir: Path,
) -> dict[str, Path]:
    """Write summary CSVs and Markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_long_history_model_zoo_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def classify_long_history_decision(row: pd.Series) -> str:
    """Classify one model-zoo row without ever marking it production-ready."""

    beats_return = bool(row.get("beats_qqq_return", False))
    beats_rtd = bool(row.get("beats_qqq_return_to_drawdown", False))
    drawdown_improvement = _float(row.get("drawdown_improvement_vs_qqq"))
    average_turnover = _float(row.get("average_turnover"))

    materially_worse_drawdown = drawdown_improvement < -0.05
    turnover_too_high = bool(np.isfinite(average_turnover) and average_turnover > 0.75)

    if beats_return and not materially_worse_drawdown and not turnover_too_high:
        return "best_candidate"
    if not beats_return and beats_rtd and drawdown_improvement > 0.0:
        return "risk_adjusted_candidate"
    if beats_return or beats_rtd or drawdown_improvement > 0.0:
        return "mixed"
    return "reject"


def render_long_history_model_zoo_markdown(
    report_data: LongHistoryModelZooReportData,
) -> str:
    """Render the Markdown report."""

    best = report_data.decision_matrix.iloc[0]
    best_candidates = report_data.decision_matrix.loc[
        report_data.decision_matrix["decision"].isin(
            ["best_candidate", "risk_adjusted_candidate"]
        )
    ]
    qqq_answer = (
        "Yes" if report_data.qqq_beat_count > 0 else "No completed configuration did."
    )
    rtd_answer = (
        "Yes"
        if report_data.return_to_drawdown_beat_count > 0
        else "No completed configuration did."
    )
    stop_or_deep_dive = (
        "Deep dive the candidates below before any production discussion."
        if not best_candidates.empty
        else "Stop or revise the research design; this batch does not justify a deep dive."
    )
    return "\n".join(
        [
            "# Long-History QQQ-Relative Model Zoo Report",
            "",
            "This is a research report only. It does not mark any configuration production-ready.",
            "",
            "## Headline Answers",
            "",
            f"- Did any configuration beat QQQ return? `{qqq_answer}`",
            f"- Did any configuration improve return-to-drawdown? `{rtd_answer}`",
            f"- Most promising configuration: `{best['candidate_label']}`",
            f"- Suggested next step: `{stop_or_deep_dive}`",
            "",
            "## Best Candidates",
            "",
            _markdown_table(
                report_data.decision_matrix.head(20),
                [
                    "candidate_label",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "return_to_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "average_turnover",
                    "decision",
                ],
            ),
            "",
            "## Training Mode Readout",
            "",
            _markdown_table(
                _training_mode_readout(report_data.summary),
                [
                    "training_mode",
                    "candidate_count",
                    "best_strategy_vs_qqq",
                    "best_return_to_drawdown",
                    "qqq_beat_count",
                    "candidate_decisions",
                ],
            ),
            "",
            "## Guardrails",
            "",
            "- Primary benchmark: `QQQ buy-and-hold`",
            "- Secondary references: `SPY buy-and-hold`, `cash`",
            "- Production export/model card/rich replay/deep-dive docs: `not created`",
            "",
        ]
    )


def _build_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in summary.reset_index(drop=True).iterrows():
        candidate_id = f"candidate_{index + 1:04d}"
        candidate_label = (
            f"{row['model_type']} | {row['training_mode']} | "
            f"{row['target_variant']} | {row['allocation_rule']} | "
            f"{row['rebalance_frequency']}"
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": candidate_label,
                **{column: row[column] for column in DECISION_MATRIX_COLUMNS if column in row},
                "rationale": _decision_rationale(row),
            }
        )
    matrix = pd.DataFrame(rows)
    for column in DECISION_MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = np.nan
    return matrix.loc[:, DECISION_MATRIX_COLUMNS]


def _decision_rationale(row: pd.Series) -> str:
    parts = []
    if bool(row["beats_qqq_return"]):
        parts.append("beats QQQ return")
    else:
        parts.append("trails QQQ return")
    if bool(row["beats_qqq_return_to_drawdown"]):
        parts.append("beats QQQ return-to-drawdown")
    else:
        parts.append("does not beat QQQ return-to-drawdown")
    if _float(row["drawdown_improvement_vs_qqq"]) > 0.0:
        parts.append("improves drawdown")
    elif _float(row["drawdown_improvement_vs_qqq"]) < -0.05:
        parts.append("materially worsens drawdown")
    if _float(row["average_turnover"]) > 0.75:
        parts.append("turnover is high")
    return "; ".join(parts)


def _training_mode_readout(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby("training_mode", sort=False, dropna=False)
        .agg(
            candidate_count=("training_mode", "size"),
            best_strategy_vs_qqq=("strategy_vs_qqq", "max"),
            best_return_to_drawdown=("return_to_drawdown", "max"),
            qqq_beat_count=("beats_qqq_return", "sum"),
            candidate_decisions=("decision", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["best_strategy_vs_qqq", "best_return_to_drawdown"],
        ascending=[False, False],
        kind="mergesort",
    )


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    decision_rank = {
        "best_candidate": 0,
        "risk_adjusted_candidate": 1,
        "mixed": 2,
        "reject": 3,
    }
    out = summary.copy()
    out["_decision_rank"] = out["decision"].map(decision_rank).fillna(9)
    return (
        out.sort_values(
            [
                "_decision_rank",
                "strategy_vs_qqq",
                "return_to_drawdown",
                "drawdown_improvement_vs_qqq",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )


def _validate_columns(df: pd.DataFrame, columns: list[str], *, path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("\n", " ") for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the long-history QQQ-relative model-zoo report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_relative_model_zoo_report_from_args())
