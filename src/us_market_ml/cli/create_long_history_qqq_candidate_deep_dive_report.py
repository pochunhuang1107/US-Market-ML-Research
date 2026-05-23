"""Create the long-history QQQ candidate deep-dive Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.long_history_candidate_deep_dive import (
    COST_SENSITIVITY_NAME,
    DAILY_DRAWDOWN_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    REPORT_NAME,
    ROLLING_WINDOWS_NAME,
    SELECTED_SYMBOLS_NAME,
    SUBPERIODS_NAME,
    SUMMARY_NAME,
    SYMBOL_EXPOSURE_NAME,
    SYMBOL_PERIODS_NAME,
    TRADES_NAME,
)


@dataclass(frozen=True)
class LongHistoryQQQCandidateDeepDiveReportData:
    """CSV-backed data for the candidate deep-dive report."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    trades: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    symbol_periods: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    decision_matrix: pd.DataFrame


def run_long_history_qqq_candidate_deep_dive_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the candidate deep-dive Markdown report."""

    args = _parse_args(argv)
    report_data = read_long_history_qqq_candidate_deep_dive_artifacts(args.output_dir)
    report = render_long_history_qqq_candidate_deep_dive_markdown(report_data)
    report_path = (
        args.report_path if args.report_path is not None else args.output_dir / REPORT_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("US Market ML long-history QQQ candidate deep-dive report")
    print(f"output dir: {args.output_dir}")
    print(f"report path: {report_path}")
    print(f"candidates: {len(report_data.summary)}")
    print(f"rolling windows: {len(report_data.rolling_windows)}")
    return 0


def read_long_history_qqq_candidate_deep_dive_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> LongHistoryQQQCandidateDeepDiveReportData:
    """Read deep-dive CSV artifacts from ``output_dir``."""

    return LongHistoryQQQCandidateDeepDiveReportData(
        summary=_read_required_csv(output_dir / SUMMARY_NAME),
        daily_equity=_read_required_csv(output_dir / DAILY_EQUITY_NAME),
        daily_drawdown=_read_required_csv(output_dir / DAILY_DRAWDOWN_NAME),
        trades=_read_required_csv(output_dir / TRADES_NAME),
        selected_symbols=_read_required_csv(output_dir / SELECTED_SYMBOLS_NAME),
        symbol_exposure=_read_required_csv(output_dir / SYMBOL_EXPOSURE_NAME),
        symbol_periods=_read_required_csv(output_dir / SYMBOL_PERIODS_NAME),
        subperiods=_read_required_csv(output_dir / SUBPERIODS_NAME),
        rolling_windows=_read_required_csv(output_dir / ROLLING_WINDOWS_NAME),
        cost_sensitivity=_read_required_csv(output_dir / COST_SENSITIVITY_NAME),
        decision_matrix=_read_required_csv(output_dir / DECISION_MATRIX_NAME),
    )


def render_long_history_qqq_candidate_deep_dive_markdown(
    report_data: LongHistoryQQQCandidateDeepDiveReportData,
) -> str:
    """Render the rich candidate deep-dive Markdown report."""

    lines = [
        "# Long-History QQQ Candidate Deep Dive Report",
        "",
        "This is a research-only deep-dive bundle. It does not mark any candidate "
        "production-ready and does not create production signal exports.",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(report_data),
        "",
        "## 2. Candidate configurations",
        "",
        _markdown_table(
            _format_for_markdown(report_data.summary),
            [
                "candidate_id",
                "model_type",
                "training_mode",
                "target_variant",
                "allocation_rule",
                "rebalance_frequency",
                "min_holding_days",
                "switch_margin",
                "transaction_cost_bps",
                "shortlist_decision",
            ],
        ),
        "",
        "## 3. Full-period performance versus QQQ/SPY/cash",
        "",
        _markdown_table(_full_period_table(report_data.summary)),
        "",
        "## 4. Equity curve and drawdown summary",
        "",
        _markdown_table(_equity_drawdown_summary(report_data.daily_equity)),
        "",
        "## 5. Trade and rebalance timeline",
        "",
        _markdown_table(
            _format_for_markdown(report_data.trades),
            [
                "date",
                "candidate_id",
                "selected_symbol",
                "previous_symbol",
                "new_symbol",
                "predicted_score",
                "rank",
                "turnover",
                "estimated_transaction_cost",
                "reason",
            ],
        ),
        "",
        "## 6. Symbol exposure and concentration",
        "",
        _markdown_table(
            _format_for_markdown(report_data.symbol_exposure),
            [
                "candidate_id",
                "symbol",
                "rank_by_exposure",
                "selected_rebalance_count",
                "exposure_days",
                "exposure_pct",
                "total_return_contribution",
                "is_top_symbol",
                "is_top3_symbol",
            ],
        ),
        "",
        "## 7. Continuous holding segments",
        "",
        _markdown_table(
            _format_for_markdown(report_data.symbol_periods),
            [
                "candidate_id",
                "symbol",
                "start_date",
                "end_date",
                "trading_days",
                "segment_return",
                "qqq_segment_return",
                "segment_vs_qqq",
                "max_drawdown_in_segment",
            ],
        ),
        "",
        "## 8. Subperiod robustness",
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
        "## 9. Rolling-window robustness",
        "",
        _rolling_window_section(report_data.rolling_windows, report_data.decision_matrix),
        "",
        "## 10. Cost sensitivity",
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
        "## 11. Candidate comparison: candidate_0001 vs candidate_0003",
        "",
        _comparison_text(report_data.decision_matrix),
        "",
        "## 12. Failure modes",
        "",
        _failure_modes_text(report_data),
        "",
        "## 13. Recommendation",
        "",
        _recommendation_text(report_data.decision_matrix),
        "",
        "## 14. Caveats",
        "",
        _caveats_text(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(report_data: LongHistoryQQQCandidateDeepDiveReportData) -> str:
    if report_data.summary.empty:
        return "No candidate rows were available; keep this research-only."
    decisions = report_data.decision_matrix["decision"].value_counts().to_dict()
    beat_count = int(_as_bool(report_data.summary["beats_qqq_return"]).sum())
    production_ready = int(_as_bool(report_data.decision_matrix["production_ready"]).sum())
    candidate_ids = ", ".join(report_data.summary["candidate_id"].astype(str).tolist())
    return (
        f"Replayed `{len(report_data.summary)}` candidates ({candidate_ids}) against QQQ "
        "buy-and-hold as the primary benchmark, with SPY and cash as references. "
        f"`{beat_count}` candidates beat QQQ on full-period return. "
        f"Deep-dive decision counts: `{decisions}`. "
        f"Production-ready candidates: `{production_ready}`."
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
                    "strategy_name": "candidate_strategy",
                    "total_return": row.total_return,
                    "max_drawdown": row.max_drawdown,
                    "return_to_drawdown": row.return_to_drawdown,
                    "vs_qqq": row.strategy_vs_qqq,
                },
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "QQQ buy-and-hold",
                    "total_return": row.qqq_buy_hold_total_return,
                    "max_drawdown": np.nan,
                    "return_to_drawdown": np.nan,
                    "vs_qqq": 0.0,
                },
                {
                    "candidate_id": row.candidate_id,
                    "strategy_name": "SPY buy-and-hold",
                    "total_return": row.spy_buy_hold_total_return,
                    "max_drawdown": np.nan,
                    "return_to_drawdown": np.nan,
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


def _equity_drawdown_summary(daily_equity: pd.DataFrame) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame()
    rows = []
    for (candidate_id, strategy_name), group in daily_equity.groupby(
        ["candidate_id", "strategy_name"],
        sort=False,
    ):
        ordered = group.sort_values("date", kind="mergesort")
        rows.append(
            {
                "candidate_id": candidate_id,
                "strategy_name": strategy_name,
                "start_date": ordered["date"].iloc[0],
                "end_date": ordered["date"].iloc[-1],
                "ending_equity": ordered["equity"].iloc[-1],
                "max_drawdown": pd.to_numeric(ordered["drawdown"], errors="coerce").min(),
                "mean_daily_turnover": pd.to_numeric(
                    ordered.get("turnover", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean(),
            }
        )
    return _format_for_markdown(pd.DataFrame(rows))


def _rolling_window_section(
    rolling_windows: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    if rolling_windows.empty:
        return "_No rolling-window rows were available._"
    summary_cols = [
        "candidate_id",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "decision",
    ]
    weakest = rolling_windows.sort_values("strategy_vs_qqq", kind="mergesort").head(10)
    return "\n\n".join(
        [
            "Rolling summary:",
            _markdown_table(_format_for_markdown(decision_matrix), summary_cols),
            "Weakest QQQ-relative rolling windows:",
            _markdown_table(
                _format_for_markdown(weakest),
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


def _comparison_text(decision_matrix: pd.DataFrame) -> str:
    if decision_matrix.empty:
        return "_No decision rows were available._"
    comparison = decision_matrix.loc[
        decision_matrix["candidate_id"].astype(str).isin(["candidate_0001", "candidate_0003"])
    ]
    table = _markdown_table(
        _format_for_markdown(comparison),
        [
            "candidate_id",
            "strategy_vs_qqq",
            "positive_vs_qqq_window_rate",
            "worst_strategy_vs_qqq",
            "top_symbol",
            "top_symbol_exposure_pct",
            "cost_30bps_strategy_vs_qqq",
            "decision",
        ],
    )
    if comparison.empty or len(comparison) < 2:
        return table
    rows = comparison.set_index("candidate_id")
    c1 = rows.loc["candidate_0001"] if "candidate_0001" in rows.index else None
    c3 = rows.loc["candidate_0003"] if "candidate_0003" in rows.index else None
    notes: list[str] = []
    if c1 is not None:
        notes.append(
            "`candidate_0001` has the larger full-period QQQ-relative return, "
            "but its XLE/XLU/VUG/XLRE concentration and AI-bull lag need targeted controls."
        )
    if c3 is not None:
        notes.append(
            "`candidate_0003` has stronger rolling breadth, but ARKK exposure and the "
            "recent drawdown path need separate review."
        )
    return "\n\n".join([table, *notes])


def _failure_modes_text(report_data: LongHistoryQQQCandidateDeepDiveReportData) -> str:
    rows = []
    for row in report_data.decision_matrix.itertuples(index=False):
        weak_periods = []
        for column, label in [
            ("covid_crash_recovery_strategy_vs_qqq", "covid_crash_recovery"),
            ("ai_bull_2023_2024_strategy_vs_qqq", "ai_bull_2023_2024"),
            ("recent_2025_2026_strategy_vs_qqq", "recent_2025_2026"),
        ]:
            value = getattr(row, column, np.nan)
            if _float(value) < 0.0:
                weak_periods.append(label)
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "top_symbol": row.top_symbol,
                "top_symbol_exposure_pct": row.top_symbol_exposure_pct,
                "top3_symbol_exposure_pct": row.top3_symbol_exposure_pct,
                "weak_subperiods": ", ".join(weak_periods) if weak_periods else "none",
                "rationale": row.rationale,
            }
        )
    return _markdown_table(_format_for_markdown(pd.DataFrame(rows)))


def _recommendation_text(decision_matrix: pd.DataFrame) -> str:
    c1_decision = _decision_for(decision_matrix, "candidate_0001")
    c3_decision = _decision_for(decision_matrix, "candidate_0003")
    return "\n".join(
        [
            f"- Targeted sweep around candidate_0001 family: `{_yes_no(c1_decision)}`.",
            f"- Targeted sweep around candidate_0003 family: `{_yes_no(c3_decision)}`.",
            (
                "- Concentration-controlled variant: `yes`; cap or penalize "
                "single-symbol/top3 exposure before any signal review."
            ),
            "- MLP batch: `defer`; this bundle intentionally did not run MLP.",
            "- Production promotion: `no`; production_ready is always false.",
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


def _decision_for(decision_matrix: pd.DataFrame, candidate_id: str) -> str:
    if decision_matrix.empty:
        return ""
    match = decision_matrix.loc[decision_matrix["candidate_id"].astype(str).eq(candidate_id)]
    if match.empty:
        return ""
    return str(match["decision"].iloc[0])


def _yes_no(decision: str) -> str:
    if decision in {"keep_for_targeted_sweep", "keep_for_signal_review"}:
        return "yes"
    if decision == "mixed_candidate":
        return "maybe"
    if decision == "reject_deprioritize":
        return "no"
    return "not_applicable"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required deep-dive artifact is missing: {path}")
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


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


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
        description="Create the long-history QQQ candidate deep-dive report."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_candidate_deep_dive_report_from_args())
