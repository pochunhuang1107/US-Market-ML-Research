"""Create the long-history QQQ shortlist validation Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.long_history_shortlist_validation import (
    CANDIDATES_NAME,
    COST_SENSITIVITY_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    REPORT_NAME,
    ROLLING_WINDOWS_NAME,
    SELECTED_SYMBOLS_NAME,
    SUBPERIODS_NAME,
    SUMMARY_NAME,
    SYMBOL_EXPOSURE_NAME,
    TRADES_NAME,
)


@dataclass(frozen=True)
class LongHistoryQQQShortlistValidationReportData:
    """CSV-backed data for the shortlist validation report."""

    candidates: pd.DataFrame
    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    decision_matrix: pd.DataFrame


def run_long_history_qqq_shortlist_validation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the shortlist validation Markdown report."""

    args = _parse_args(argv)
    report_data = read_long_history_qqq_shortlist_validation_artifacts(args.output_dir)
    report = render_long_history_qqq_shortlist_validation_markdown(report_data)
    report_path = (
        args.report_path if args.report_path is not None else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("US Market ML long-history QQQ shortlist validation report")
    print(f"output dir: {args.output_dir}")
    print(f"report path: {report_path}")
    print(f"candidates: {len(report_data.candidates)}")
    print(f"rolling windows: {len(report_data.rolling_windows)}")
    return 0


def read_long_history_qqq_shortlist_validation_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> LongHistoryQQQShortlistValidationReportData:
    """Read validation CSV artifacts from ``output_dir``."""

    return LongHistoryQQQShortlistValidationReportData(
        candidates=_read_required_csv(output_dir / CANDIDATES_NAME),
        summary=_read_required_csv(output_dir / SUMMARY_NAME),
        daily_equity=_read_required_csv(output_dir / DAILY_EQUITY_NAME),
        trades=_read_required_csv(output_dir / TRADES_NAME),
        selected_symbols=_read_required_csv(output_dir / SELECTED_SYMBOLS_NAME),
        symbol_exposure=_read_required_csv(output_dir / SYMBOL_EXPOSURE_NAME),
        subperiods=_read_required_csv(output_dir / SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(output_dir / ROLLING_WINDOWS_NAME),
        cost_sensitivity=_read_required_csv(output_dir / COST_SENSITIVITY_NAME),
        decision_matrix=_read_required_csv(output_dir / DECISION_MATRIX_NAME),
    )


def render_long_history_qqq_shortlist_validation_markdown(
    report_data: LongHistoryQQQShortlistValidationReportData,
) -> str:
    """Render the focused shortlist validation Markdown report."""

    best = _best_deep_dive_candidates(report_data.decision_matrix)
    rejected = report_data.decision_matrix.loc[
        report_data.decision_matrix["decision"].astype(str).eq("reject_deprioritize")
    ]
    lines = [
        "# Long-History QQQ Shortlist Validation Report",
        "",
        "This is a research-only validation bundle. It does not mark any candidate "
        "production-ready and does not create production signal exports.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Candidate shortlist",
        "",
        _markdown_table(
            _format_for_markdown(report_data.candidates),
            [
                "candidate_id",
                "candidate_label",
                "switch_margin",
                "total_return",
                "strategy_vs_qqq",
                "return_to_drawdown",
                "selection_reason",
            ],
        ),
        "",
        "## 3. Full-period comparison versus QQQ/SPY/cash",
        "",
        _markdown_table(_full_period_table(report_data.summary)),
        "",
        "## 4. Selected symbols and concentration",
        "",
        _markdown_table(
            _format_for_markdown(report_data.symbol_exposure),
            [
                "candidate_id",
                "top_symbol",
                "top_symbol_exposure_pct",
                "top3_symbol_exposure_pct",
                "unique_symbol_count",
                "qqq_exposure_pct",
                "spy_exposure_pct",
            ],
        ),
        "",
        "Top selected symbols by candidate:",
        "",
        _markdown_table(
            _format_for_markdown(
                report_data.selected_symbols.sort_values(
                    ["candidate_id", "exposure_pct"],
                    ascending=[True, False],
                    kind="mergesort",
                ).groupby("candidate_id", sort=False).head(5)
            ),
            [
                "candidate_id",
                "symbol",
                "selected_rebalance_count",
                "exposure_days",
                "exposure_pct",
                "total_return_contribution",
            ],
        ),
        "",
        "## 5. Subperiod robustness",
        "",
        _markdown_table(
            _format_for_markdown(report_data.subperiods),
            [
                "candidate_id",
                "period_name",
                "total_return",
                "qqq_total_return",
                "strategy_vs_qqq",
                "max_drawdown",
                "qqq_max_drawdown",
                "drawdown_improvement_vs_qqq",
                "transaction_count",
                "top_symbol",
            ],
        ),
        "",
        "## 6. Rolling-window robustness",
        "",
        _rolling_window_text(report_data.rolling_windows, report_data.decision_matrix),
        "",
        "## 7. Cost sensitivity",
        "",
        _markdown_table(
            _format_for_markdown(report_data.cost_sensitivity),
            [
                "candidate_id",
                "transaction_cost_bps",
                "total_return",
                "strategy_vs_qqq",
                "max_drawdown",
                "return_to_drawdown",
                "decision_at_cost",
            ],
        ),
        "",
        "## 8. Best candidates to deep dive",
        "",
        _markdown_table(
            _format_for_markdown(best),
            [
                "candidate_id",
                "candidate_label",
                "strategy_vs_qqq",
                "positive_vs_qqq_window_rate",
                "top_symbol_exposure_pct",
                "decision",
                "rationale",
            ],
        ),
        "",
        "## 9. Rejected/deprioritized candidates",
        "",
        _markdown_table(
            _format_for_markdown(rejected),
            [
                "candidate_id",
                "candidate_label",
                "strategy_vs_qqq",
                "positive_vs_qqq_window_rate",
                "top_symbol_exposure_pct",
                "decision",
                "rationale",
            ],
        ),
        "",
        "## 10. Recommendation",
        "",
        _recommendation_text(report_data.decision_matrix),
        "",
        "## 11. Caveats",
        "",
        _caveats_text(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(report_data: LongHistoryQQQShortlistValidationReportData) -> str:
    if report_data.summary.empty:
        return "No summary rows were available; keep this research-only."
    decisions = report_data.decision_matrix["decision"].value_counts().to_dict()
    best = _best_deep_dive_candidates(report_data.decision_matrix).head(1)
    best_text = (
        f"Top validation candidate: `{best.iloc[0]['candidate_id']}` "
        f"(`{best.iloc[0]['decision']}`)."
        if not best.empty
        else "No candidate cleared the deep-dive threshold."
    )
    qqq_beats = int(_as_bool(report_data.summary["beats_qqq_return"]).sum())
    return (
        f"Validated `{len(report_data.summary)}` deduplicated shortlist candidates against "
        "QQQ buy-and-hold as the primary benchmark, with SPY and cash as references. "
        f"`{qqq_beats}` candidates beat QQQ on full-period return. "
        f"Decision counts: `{decisions}`. {best_text} "
        "The output is a validation/deep-dive bundle only, not a production promotion."
    )


def _full_period_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for row in summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "strategy",
                    "total_return": row.total_return,
                    "max_drawdown": row.max_drawdown,
                    "return_to_drawdown": row.return_to_drawdown,
                    "vs_qqq": row.strategy_vs_qqq,
                },
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "QQQ buy-and-hold",
                    "total_return": row.qqq_buy_hold_total_return,
                    "max_drawdown": getattr(row, "qqq_buy_hold_max_drawdown", np.nan),
                    "return_to_drawdown": getattr(
                        row,
                        "qqq_buy_hold_return_to_drawdown",
                        np.nan,
                    ),
                    "vs_qqq": 0.0,
                },
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "SPY buy-and-hold",
                    "total_return": row.spy_buy_hold_total_return,
                    "max_drawdown": getattr(row, "spy_buy_hold_max_drawdown", np.nan),
                    "return_to_drawdown": getattr(
                        row,
                        "spy_buy_hold_return_to_drawdown",
                        np.nan,
                    ),
                    "vs_qqq": row.spy_buy_hold_total_return - row.qqq_buy_hold_total_return,
                },
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "cash",
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "return_to_drawdown": np.nan,
                    "vs_qqq": -row.qqq_buy_hold_total_return,
                },
            ]
        )
    return _format_for_markdown(pd.DataFrame(rows))


def _rolling_window_text(rolling_windows: pd.DataFrame, decision_matrix: pd.DataFrame) -> str:
    if rolling_windows.empty:
        return "_No rolling-window rows were available._"
    summary_cols = [
        "candidate_id",
        "rolling_window_count",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "average_strategy_vs_qqq",
        "worst_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
        "worst_window_label",
    ]
    worst = rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(10)
    return "\n\n".join(
        [
            "Rolling summary:",
            _markdown_table(_format_for_markdown(decision_matrix), summary_cols),
            "Weakest QQQ-relative rolling windows:",
            _markdown_table(
                _format_for_markdown(worst),
                [
                    "candidate_id",
                    "window_label",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "underperformed_qqq",
                ],
            ),
        ]
    )


def _best_deep_dive_candidates(decision_matrix: pd.DataFrame) -> pd.DataFrame:
    if decision_matrix.empty:
        return decision_matrix
    keep = decision_matrix.loc[
        decision_matrix["decision"].astype(str).isin(
            ["robust_candidate", "keep_for_deep_dive"]
        )
    ]
    if keep.empty:
        return keep
    return keep.sort_values(
        ["decision", "strategy_vs_qqq", "positive_vs_qqq_window_rate"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def _recommendation_text(decision_matrix: pd.DataFrame) -> str:
    best = _best_deep_dive_candidates(decision_matrix)
    candidate_list = (
        ", ".join(best["candidate_id"].astype(str).tolist())
        if not best.empty
        else "none"
    )
    mixed_count = int(decision_matrix["decision"].astype(str).eq("mixed_candidate").sum())
    return "\n".join(
        [
            f"- Rich replay / deep dive candidate(s): `{candidate_list}`.",
            (
                "- Additional targeted sweep: `consider only around robust/mixed shortlist "
                "families after reviewing concentration and rolling-window failures`."
            ),
            (
                "- MLP batch: `defer`; this validation intentionally did not run MLP. "
                f"Mixed candidates: `{mixed_count}`."
            ),
            "- Production promotion: `no`; no candidate is marked production-ready.",
        ]
    )


def _caveats_text() -> str:
    return "\n".join(
        [
            "- Research-only artifact bundle.",
            "- Tiingo-only local market data.",
            "- No production signal export, model card, or cloud write.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Transaction costs are a simple turnover-times-bps approximation.",
        ]
    )


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required validation artifact is missing: {path}")
    return pd.read_csv(path)


def _format_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(_format_number)
    return out


def _format_number(value: object) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number):
        return ""
    return f"{number:.4f}"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    if columns is None:
        columns = table.columns.tolist()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(_markdown_cell(value) for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the long-history QQQ shortlist validation report."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_shortlist_validation_report_from_args())
