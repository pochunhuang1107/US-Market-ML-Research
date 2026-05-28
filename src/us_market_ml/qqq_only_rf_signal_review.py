"""Signal-behavior review for the strongest QQQ-only RF timing family.

This module is research-only and reads completed RF confirmation artifacts.
It does not train models, run sweeps, export production signals, create model
cards, or mark any row production-ready.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.qqq_only_best_candidate_deep_dive import (
    build_rolling_windows,
    confirm_held_assets_are_qqq_cash_only,
)

OUTPUT_PREFIX = "qqq_only_rf_signal_review"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_signal_review")

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
CANDIDATE_SET_NAME = f"{OUTPUT_PREFIX}_candidate_set.csv"
EXPOSURE_SEGMENTS_NAME = f"{OUTPUT_PREFIX}_exposure_segments.csv"
CASH_TIMING_NAME = f"{OUTPUT_PREFIX}_cash_timing.csv"
MISSED_RALLIES_NAME = f"{OUTPUT_PREFIX}_missed_rallies.csv"
DRAWDOWN_AVOIDANCE_NAME = f"{OUTPUT_PREFIX}_drawdown_avoidance.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = f"{OUTPUT_PREFIX}_equity_curve.png"
DRAWDOWN_CHART_NAME = f"{OUTPUT_PREFIX}_drawdown_curve.png"
EXPOSURE_CHART_NAME = f"{OUTPUT_PREFIX}_exposure_timeline.png"
CASH_RETURNS_CHART_NAME = f"{OUTPUT_PREFIX}_cash_vs_qqq_returns.png"
ROLLING_CHART_NAME = f"{OUTPUT_PREFIX}_rolling_edge.png"
THRESHOLD_CHART_NAME = f"{OUTPUT_PREFIX}_threshold_comparison.png"

REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. What improved versus the original best candidate.",
    "## 3. Candidate set reviewed.",
    "## 4. Equity and drawdown behavior.",
    "## 5. QQQ/cash exposure path.",
    "## 6. Cash timing review.",
    "## 7. Missed rally review.",
    "## 8. Drawdown avoidance review.",
    "## 9. Rolling-window behavior.",
    "## 10. Threshold comparison.",
    "## 11. Final recommendation:",
    "## 12. Caveats:",
)

BASE_FAMILY = {
    "model_type": "random_forest",
    "training_mode": "rolling_5y",
    "target_name": "risk_on_20d",
    "feature_set_name": "qqq_self_cross_market",
    "rebalance_frequency": "weekly",
}

REGIME_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("covid_crash_recovery", "2020-01-01", "2020-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_bull_2023_2024", "2023-01-01", "2024-12-31"),
    ("recent_2025_2026", "2025-01-01", "2026-05-15"),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "review_name",
    "best_strategy_id",
    "best_strategy_name",
    "best_threshold",
    "best_min_holding_days",
    "best_transaction_cost_bps",
    "best_total_return",
    "qqq_buy_hold_total_return",
    "best_strategy_vs_qqq",
    "best_max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "best_drawdown_improvement_vs_qqq",
    "best_return_to_drawdown",
    "best_positive_vs_qqq_window_rate",
    "best_worst_strategy_vs_qqq",
    "best_drawdown_improvement_window_rate",
    "best_average_exposure",
    "best_cash_days_pct",
    "best_trade_count",
    "selected_candidate_count",
    "cash_segment_count",
    "useful_risk_off_cash_segments",
    "missed_rally_cash_segments",
    "largest_missed_rally",
    "second_confirmation_recommended",
    "paper_tracking_candidate_review_recommended",
    "production_ready",
)

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "candidate_rank",
    "candidate_role",
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "feature_set_name",
    "exposure_mode",
    "rebalance_frequency",
    "transaction_cost_bps",
    "threshold",
    "min_holding_days",
    "switch_margin",
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "positive_vs_qqq_window_rate",
    "worst_strategy_vs_qqq",
    "best_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "decision_label",
    "production_ready",
)

SEGMENT_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "segment_type",
    "start_date",
    "end_date",
    "trading_days",
    "average_exposure",
    "qqq_return_during_segment",
    "strategy_return_during_segment",
    "qqq_max_drawdown_during_segment",
    "segment_interpretation",
)

CASH_TIMING_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "start_date",
    "end_date",
    "trading_days",
    "average_exposure",
    "did_cash_avoid_qqq_drawdown",
    "did_cash_miss_qqq_rally",
    "qqq_return_during_cash",
    "strategy_return_during_cash",
    "max_qqq_drawdown_during_cash",
    "cash_timing_label",
)

MISSED_RALLY_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "start_date",
    "end_date",
    "trading_days",
    "average_exposure",
    "qqq_return",
    "strategy_return",
    "missed_return",
    "explains_underperformance_windows",
    "underperformance_window_label",
)

DRAWDOWN_AVOIDANCE_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "period_name",
    "period_start",
    "period_end",
    "strategy_return",
    "qqq_return",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "strategy_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "interpretation",
)

DECISION_COLUMNS: tuple[str, ...] = (
    "candidate_rank",
    "candidate_role",
    "strategy_id",
    "strategy_name",
    "threshold",
    "min_holding_days",
    "transaction_cost_bps",
    "exposure_mode",
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "positive_vs_qqq_window_rate",
    "worst_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "cash_timing_summary",
    "missed_rally_count",
    "decision_label",
    "recommendation",
    "production_ready",
)


@dataclass(frozen=True)
class QQQOnlyRFSignalReviewArtifacts:
    """In-memory artifacts for the signal behavior review."""

    summary: pd.DataFrame
    candidate_set: pd.DataFrame
    exposure_segments: pd.DataFrame
    cash_timing: pd.DataFrame
    missed_rallies: pd.DataFrame
    drawdown_avoidance: pd.DataFrame
    rolling_windows: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def build_qqq_only_rf_signal_review_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    decision_matrix: pd.DataFrame | None = None,
) -> QQQOnlyRFSignalReviewArtifacts:
    """Build the focused signal-review tables from completed artifacts."""

    source = summary.copy() if summary is not None and not summary.empty else results.copy()
    candidate_set = select_signal_review_candidate_set(source, decision_matrix=decision_matrix)
    selected_ids = candidate_set["strategy_id"].astype(str).tolist()
    selected_equity = _filter_strategy_ids(daily_equity, selected_ids)
    selected_drawdown = _filter_strategy_ids(daily_drawdown, selected_ids)
    selected_exposure = _filter_strategy_ids(exposure_log, selected_ids)
    confirm_held_assets_are_qqq_cash_only(selected_exposure)

    segments = build_exposure_segments(selected_exposure, selected_equity)
    cash_timing = build_cash_timing_review(segments)
    rolling = build_rolling_windows(selected_equity)
    rolling = add_recent_window_behavior(rolling)
    missed_rallies = build_missed_rallies(segments, rolling)
    drawdown_avoidance = build_drawdown_avoidance(selected_equity, selected_exposure)
    decisions = build_signal_review_decision_matrix(
        candidate_set=candidate_set,
        cash_timing=cash_timing,
        missed_rallies=missed_rallies,
    )
    summary_table = build_signal_review_summary(
        candidate_set=candidate_set,
        cash_timing=cash_timing,
        missed_rallies=missed_rallies,
    )
    report = render_qqq_only_rf_signal_review_markdown(
        summary=summary_table,
        candidate_set=candidate_set,
        exposure_segments=segments,
        cash_timing=cash_timing,
        missed_rallies=missed_rallies,
        drawdown_avoidance=drawdown_avoidance,
        rolling_windows=rolling,
        decision_matrix=decisions,
    )
    return QQQOnlyRFSignalReviewArtifacts(
        summary=summary_table,
        candidate_set=candidate_set,
        exposure_segments=segments,
        cash_timing=cash_timing,
        missed_rallies=missed_rallies,
        drawdown_avoidance=drawdown_avoidance,
        rolling_windows=rolling,
        decision_matrix=decisions,
        report_markdown=report,
    )


def select_signal_review_candidate_set(
    source_rows: pd.DataFrame,
    *,
    decision_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select the requested narrow RF family and the QQQ benchmark."""

    frame = source_rows.copy()
    if frame.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    if "switch_margin" not in frame.columns:
        frame["switch_margin"] = 0.0

    base = pd.Series(True, index=frame.index)
    for column, value in BASE_FAMILY.items():
        base &= frame[column].astype(str).eq(value)
    switch_zero = np.isclose(pd.to_numeric(frame["switch_margin"], errors="coerce").fillna(0.0), 0.0)
    cost = pd.to_numeric(frame["transaction_cost_bps"], errors="coerce")
    threshold = pd.to_numeric(frame["threshold"], errors="coerce")
    min_hold = pd.to_numeric(frame["min_holding_days"], errors="coerce").fillna(0).astype(int)
    exposure = frame["exposure_mode"].astype(str)

    primary_045 = (
        base
        & switch_zero
        & exposure.eq("binary_0_100")
        & np.isclose(threshold, 0.45)
        & min_hold.eq(0)
        & cost.isin([0.0, 5.0, 10.0, 20.0])
    )
    min_hold_045 = (
        base
        & switch_zero
        & exposure.eq("binary_0_100")
        & np.isclose(threshold, 0.45)
        & min_hold.eq(10)
        & cost.isin([0.0, 10.0])
    )
    threshold_050 = (
        base
        & switch_zero
        & exposure.eq("binary_0_100")
        & np.isclose(threshold, 0.50)
        & min_hold.eq(0)
        & cost.isin([0.0, 10.0])
    )
    threshold_refs = (
        base
        & switch_zero
        & exposure.eq("binary_0_100")
        & threshold.isin([0.55, 0.60])
        & min_hold.eq(0)
        & cost.isin([0.0, 10.0])
    )
    smoother_refs = (
        base
        & switch_zero
        & exposure.eq("partial_50_100")
        & np.isclose(threshold, 0.45)
    )
    benchmark = frame["strategy_name"].astype(str).eq("qqq_buy_hold")
    selected = frame.loc[
        primary_045 | min_hold_045 | threshold_050 | threshold_refs | smoother_refs | benchmark
    ].copy()

    if selected.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    selected = selected.drop_duplicates("strategy_id", keep="first")
    selected["candidate_role"] = selected.apply(_candidate_role, axis=1)
    selected = _sort_candidate_set(selected)
    selected["candidate_rank"] = np.arange(1, len(selected) + 1)
    selected["production_ready"] = False
    selected["decision_label"] = selected.apply(_review_decision_label, axis=1)
    selected = _merge_decision_context(selected, decision_matrix)
    return _ensure_columns(selected, CANDIDATE_COLUMNS)


def build_exposure_segments(
    exposure_log: pd.DataFrame,
    daily_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Group each selected strategy into continuous QQQ/reduced-risk segments."""

    rows: list[dict[str, object]] = []
    for strategy_id, exposure in exposure_log.groupby("strategy_id", sort=False):
        strategy_id = str(strategy_id)
        equity = _strategy_frame(daily_equity, strategy_id)
        if exposure.empty or equity.empty:
            continue
        exp = exposure.copy()
        exp["date"] = pd.to_datetime(exp["date"])
        exp = exp.sort_values("date", kind="mergesort").reset_index(drop=True)
        exp_values = pd.to_numeric(exp["effective_exposure"], errors="coerce").fillna(0.0)
        exp["segment_type"] = np.where(exp_values >= 1.0 - 1e-9, "QQQ", "cash")
        segment_key = exp["segment_type"].ne(exp["segment_type"].shift()).cumsum()
        for _, group in exp.groupby(segment_key, sort=False):
            start = pd.Timestamp(group["date"].iloc[0])
            end = pd.Timestamp(group["date"].iloc[-1])
            eq = _date_slice(equity, start, end)
            if eq.empty:
                continue
            qqq_return = _segment_total_return(eq["qqq_buy_hold_equity"])
            strategy_return = _segment_total_return(eq["equity"])
            qqq_dd = _segment_max_drawdown(eq["qqq_buy_hold_equity"])
            avg_exp = float(pd.to_numeric(group["effective_exposure"], errors="coerce").mean())
            segment_type = str(group["segment_type"].iloc[0])
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": str(group["strategy_name"].iloc[0]),
                    "segment_type": segment_type,
                    "start_date": str(start.date()),
                    "end_date": str(end.date()),
                    "trading_days": int(len(group)),
                    "average_exposure": avg_exp,
                    "qqq_return_during_segment": qqq_return,
                    "strategy_return_during_segment": strategy_return,
                    "qqq_max_drawdown_during_segment": qqq_dd,
                    "segment_interpretation": _segment_interpretation(
                        segment_type=segment_type,
                        average_exposure=avg_exp,
                        qqq_return=qqq_return,
                        strategy_return=strategy_return,
                        qqq_drawdown=qqq_dd,
                    ),
                }
            )
    return _ensure_columns(pd.DataFrame(rows), SEGMENT_COLUMNS)


def build_cash_timing_review(exposure_segments: pd.DataFrame) -> pd.DataFrame:
    """Classify reduced-risk segments as useful, missed-rally, neutral, or whipsaw."""

    cash = exposure_segments.loc[exposure_segments["segment_type"].astype(str).eq("cash")].copy()
    rows: list[dict[str, object]] = []
    for row in cash.to_dict("records"):
        qqq_return = _float(row.get("qqq_return_during_segment"))
        strategy_return = _float(row.get("strategy_return_during_segment"))
        qqq_dd = _float(row.get("qqq_max_drawdown_during_segment"))
        days = int(_float(row.get("trading_days")))
        avoided = bool(qqq_return <= -0.02 or qqq_dd <= -0.05)
        missed = bool(qqq_return >= 0.03 and strategy_return < qqq_return)
        if avoided and strategy_return >= qqq_return:
            label = "useful_risk_off"
        elif missed:
            label = "missed_rally"
        elif days <= 10 and abs(qqq_return) < 0.02:
            label = "whipsaw"
        else:
            label = "neutral"
        rows.append(
            {
                "strategy_id": row.get("strategy_id"),
                "strategy_name": row.get("strategy_name"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "trading_days": days,
                "average_exposure": row.get("average_exposure"),
                "did_cash_avoid_qqq_drawdown": avoided,
                "did_cash_miss_qqq_rally": missed,
                "qqq_return_during_cash": qqq_return,
                "strategy_return_during_cash": strategy_return,
                "max_qqq_drawdown_during_cash": qqq_dd,
                "cash_timing_label": label,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["strategy_id", "start_date"],
            ascending=[True, True],
            kind="mergesort",
        )
    return _ensure_columns(out, CASH_TIMING_COLUMNS)


def build_missed_rallies(
    exposure_segments: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    *,
    min_qqq_rally: float = 0.03,
) -> pd.DataFrame:
    """Find larger QQQ rallies during reduced-risk segments."""

    cash = exposure_segments.loc[exposure_segments["segment_type"].astype(str).eq("cash")].copy()
    if cash.empty:
        return pd.DataFrame(columns=MISSED_RALLY_COLUMNS)
    cash["missed_return"] = (
        pd.to_numeric(cash["qqq_return_during_segment"], errors="coerce")
        - pd.to_numeric(cash["strategy_return_during_segment"], errors="coerce")
    )
    rallies = cash.loc[
        (pd.to_numeric(cash["qqq_return_during_segment"], errors="coerce") >= min_qqq_rally)
        & (pd.to_numeric(cash["missed_return"], errors="coerce") > 0.0)
    ].copy()
    rows: list[dict[str, object]] = []
    for row in rallies.to_dict("records"):
        overlap = _overlapping_underperformance_window(
            rolling_windows=rolling_windows,
            strategy_id=str(row["strategy_id"]),
            start_date=str(row["start_date"]),
            end_date=str(row["end_date"]),
        )
        rows.append(
            {
                "strategy_id": row["strategy_id"],
                "strategy_name": row["strategy_name"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "trading_days": row["trading_days"],
                "average_exposure": row["average_exposure"],
                "qqq_return": row["qqq_return_during_segment"],
                "strategy_return": row["strategy_return_during_segment"],
                "missed_return": row["missed_return"],
                "explains_underperformance_windows": bool(overlap),
                "underperformance_window_label": overlap,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("missed_return", ascending=False, kind="mergesort")
    return _ensure_columns(out, MISSED_RALLY_COLUMNS)


def build_drawdown_avoidance(
    daily_equity: pd.DataFrame,
    exposure_log: pd.DataFrame,
    *,
    periods: Sequence[tuple[str, str, str]] = REGIME_PERIODS,
) -> pd.DataFrame:
    """Measure return/drawdown/exposure behavior in named market periods."""

    rows: list[dict[str, object]] = []
    max_date = _max_date(daily_equity)
    for strategy_id, equity in daily_equity.groupby("strategy_id", sort=False):
        strategy_id = str(strategy_id)
        exposure = _strategy_frame(exposure_log, strategy_id)
        for period_name, start, end in periods:
            period_end = min(pd.Timestamp(end), max_date) if max_date is not None else pd.Timestamp(end)
            eq = _date_slice(equity, pd.Timestamp(start), period_end)
            if eq.empty:
                continue
            exp = _date_slice(exposure, pd.Timestamp(start), period_end)
            metrics = _period_metrics(eq)
            exp_values = pd.to_numeric(
                exp.get("effective_exposure", pd.Series(dtype=float)),
                errors="coerce",
            )
            avg_exp = float(exp_values.mean()) if exp_values.notna().any() else np.nan
            cash_pct = float((exp_values < 1.0 - 1e-9).mean()) if not exp_values.empty else np.nan
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": str(eq["strategy_name"].iloc[0]),
                    "period_name": period_name,
                    "period_start": start,
                    "period_end": str(period_end.date()),
                    "strategy_return": metrics["strategy_return"],
                    "qqq_return": metrics["qqq_return"],
                    "strategy_max_drawdown": metrics["strategy_max_drawdown"],
                    "qqq_max_drawdown": metrics["qqq_max_drawdown"],
                    "drawdown_improvement_vs_qqq": metrics["drawdown_improvement_vs_qqq"],
                    "strategy_vs_qqq": metrics["strategy_vs_qqq"],
                    "average_exposure": avg_exp,
                    "cash_days_pct": cash_pct,
                    "interpretation": _drawdown_period_interpretation(
                        period_name=period_name,
                        strategy_vs_qqq=metrics["strategy_vs_qqq"],
                        drawdown_improvement=metrics["drawdown_improvement_vs_qqq"],
                        average_exposure=avg_exp,
                    ),
                }
            )
    return _ensure_columns(pd.DataFrame(rows), DRAWDOWN_AVOIDANCE_COLUMNS)


def add_recent_window_behavior(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    """Add a compact recent-window label to rolling-window rows."""

    if rolling_windows.empty:
        out = rolling_windows.copy()
        out["recent_window_behavior"] = np.nan
        return out
    out = rolling_windows.copy()
    out["recent_window_behavior"] = "historical_window"
    out["window_end_dt"] = pd.to_datetime(out["window_end"])
    for (strategy_id, years), group in out.groupby(["strategy_id", "window_years"], sort=False):
        latest_idx = group["window_end_dt"].idxmax()
        edge = _float(out.loc[latest_idx, "strategy_vs_qqq"])
        dd_edge = _float(out.loc[latest_idx, "drawdown_improvement_vs_qqq"])
        if edge > 0.0 and dd_edge > 0.0:
            label = "recent_positive_edge_and_drawdown_improvement"
        elif edge > 0.0:
            label = "recent_positive_edge"
        elif dd_edge > 0.0:
            label = "recent_drawdown_improvement_only"
        else:
            label = "recent_underperformance"
        out.loc[latest_idx, "recent_window_behavior"] = label
    return out.drop(columns=["window_end_dt"])


def build_signal_review_decision_matrix(
    *,
    candidate_set: pd.DataFrame,
    cash_timing: pd.DataFrame,
    missed_rallies: pd.DataFrame,
) -> pd.DataFrame:
    """Apply research-only second-confirmation recommendations."""

    timing = _cash_timing_summary(cash_timing)
    missed = _missed_rally_summary(missed_rallies)
    merged = candidate_set.merge(timing, on="strategy_id", how="left").merge(
        missed,
        on="strategy_id",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        label = str(row.get("decision_label"))
        role = str(row.get("candidate_role"))
        if role == "benchmark":
            recommendation = "Benchmark only; QQQ buy-and-hold remains the primary comparison."
        elif label == "keep_for_second_confirmation":
            recommendation = "Carry forward for a narrow second confirmation; research-only."
        elif label == "keep_for_paper_tracking_candidate_review":
            recommendation = "Consider only after second confirmation; do not export production signals."
        elif label == "keep_as_threshold_reference":
            recommendation = "Keep as a threshold or smoother reference, not as the lead setting."
        else:
            recommendation = "Reject or deprioritize for this RF signal-behavior review."
        rows.append(
            {
                "candidate_rank": row.get("candidate_rank"),
                "candidate_role": role,
                "strategy_id": row.get("strategy_id"),
                "strategy_name": row.get("strategy_name"),
                "threshold": row.get("threshold"),
                "min_holding_days": row.get("min_holding_days"),
                "transaction_cost_bps": row.get("transaction_cost_bps"),
                "exposure_mode": row.get("exposure_mode"),
                "total_return": row.get("total_return"),
                "qqq_buy_hold_total_return": row.get("qqq_buy_hold_total_return"),
                "strategy_vs_qqq": row.get("strategy_vs_qqq"),
                "max_drawdown": row.get("max_drawdown"),
                "qqq_buy_hold_max_drawdown": row.get("qqq_buy_hold_max_drawdown"),
                "drawdown_improvement_vs_qqq": row.get("drawdown_improvement_vs_qqq"),
                "positive_vs_qqq_window_rate": row.get("positive_vs_qqq_window_rate"),
                "worst_strategy_vs_qqq": row.get("worst_strategy_vs_qqq"),
                "drawdown_improvement_window_rate": row.get("drawdown_improvement_window_rate"),
                "average_exposure": row.get("average_exposure"),
                "cash_days_pct": row.get("cash_days_pct"),
                "trade_count": row.get("trade_count"),
                "cash_timing_summary": row.get("cash_timing_summary", ""),
                "missed_rally_count": int(_float(row.get("missed_rally_count"))) if pd.notna(row.get("missed_rally_count")) else 0,
                "decision_label": label,
                "recommendation": recommendation,
                "production_ready": False,
            }
        )
    return _ensure_columns(pd.DataFrame(rows), DECISION_COLUMNS)


def build_signal_review_summary(
    *,
    candidate_set: pd.DataFrame,
    cash_timing: pd.DataFrame,
    missed_rallies: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row executive summary."""

    best = candidate_set.loc[candidate_set["candidate_role"].astype(str).eq("best_setting")].head(1)
    if best.empty:
        best = candidate_set.head(1)
    useful = int(cash_timing["cash_timing_label"].astype(str).eq("useful_risk_off").sum()) if not cash_timing.empty else 0
    missed = int(cash_timing["cash_timing_label"].astype(str).eq("missed_rally").sum()) if not cash_timing.empty else 0
    largest_missed = (
        float(pd.to_numeric(missed_rallies["missed_return"], errors="coerce").max())
        if not missed_rallies.empty
        else 0.0
    )
    row = {
        "review_name": OUTPUT_PREFIX,
        "best_strategy_id": best["strategy_id"].iloc[0] if not best.empty else "",
        "best_strategy_name": best["strategy_name"].iloc[0] if not best.empty else "",
        "best_threshold": best["threshold"].iloc[0] if not best.empty else np.nan,
        "best_min_holding_days": best["min_holding_days"].iloc[0] if not best.empty else np.nan,
        "best_transaction_cost_bps": best["transaction_cost_bps"].iloc[0] if not best.empty else np.nan,
        "best_total_return": best["total_return"].iloc[0] if not best.empty else np.nan,
        "qqq_buy_hold_total_return": best["qqq_buy_hold_total_return"].iloc[0] if not best.empty else np.nan,
        "best_strategy_vs_qqq": best["strategy_vs_qqq"].iloc[0] if not best.empty else np.nan,
        "best_max_drawdown": best["max_drawdown"].iloc[0] if not best.empty else np.nan,
        "qqq_buy_hold_max_drawdown": best["qqq_buy_hold_max_drawdown"].iloc[0] if not best.empty else np.nan,
        "best_drawdown_improvement_vs_qqq": best["drawdown_improvement_vs_qqq"].iloc[0] if not best.empty else np.nan,
        "best_return_to_drawdown": best["return_to_drawdown"].iloc[0] if not best.empty else np.nan,
        "best_positive_vs_qqq_window_rate": best["positive_vs_qqq_window_rate"].iloc[0] if not best.empty else np.nan,
        "best_worst_strategy_vs_qqq": best["worst_strategy_vs_qqq"].iloc[0] if not best.empty else np.nan,
        "best_drawdown_improvement_window_rate": best["drawdown_improvement_window_rate"].iloc[0] if not best.empty else np.nan,
        "best_average_exposure": best["average_exposure"].iloc[0] if not best.empty else np.nan,
        "best_cash_days_pct": best["cash_days_pct"].iloc[0] if not best.empty else np.nan,
        "best_trade_count": best["trade_count"].iloc[0] if not best.empty else np.nan,
        "selected_candidate_count": len(candidate_set),
        "cash_segment_count": len(cash_timing),
        "useful_risk_off_cash_segments": useful,
        "missed_rally_cash_segments": missed,
        "largest_missed_rally": largest_missed,
        "second_confirmation_recommended": bool(not best.empty),
        "paper_tracking_candidate_review_recommended": True,
        "production_ready": False,
    }
    return _ensure_columns(pd.DataFrame([row]), SUMMARY_COLUMNS)


def write_qqq_only_rf_signal_review_artifacts(
    *,
    artifacts: QQQOnlyRFSignalReviewArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write required CSV, Markdown, and PNG artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.candidate_set.to_csv(paths[CANDIDATE_SET_NAME], index=False)
    artifacts.exposure_segments.to_csv(paths[EXPOSURE_SEGMENTS_NAME], index=False)
    artifacts.cash_timing.to_csv(paths[CASH_TIMING_NAME], index=False)
    artifacts.missed_rallies.to_csv(paths[MISSED_RALLIES_NAME], index=False)
    artifacts.drawdown_avoidance.to_csv(paths[DRAWDOWN_AVOIDANCE_NAME], index=False)
    artifacts.rolling_windows.to_csv(paths[ROLLING_WINDOWS_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_signal_review_charts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        paths=paths,
    )
    return paths


def output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all expected output paths."""

    names = (
        SUMMARY_NAME,
        CANDIDATE_SET_NAME,
        EXPOSURE_SEGMENTS_NAME,
        CASH_TIMING_NAME,
        MISSED_RALLIES_NAME,
        DRAWDOWN_AVOIDANCE_NAME,
        ROLLING_WINDOWS_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        EXPOSURE_CHART_NAME,
        CASH_RETURNS_CHART_NAME,
        ROLLING_CHART_NAME,
        THRESHOLD_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def render_qqq_only_rf_signal_review_markdown(
    *,
    summary: pd.DataFrame,
    candidate_set: pd.DataFrame,
    exposure_segments: pd.DataFrame,
    cash_timing: pd.DataFrame,
    missed_rallies: pd.DataFrame,
    drawdown_avoidance: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the required signal-behavior review report."""

    best_id = _best_strategy_id(candidate_set)
    best = candidate_set.loc[candidate_set["strategy_id"].astype(str).eq(best_id)].head(1)
    best_segments = exposure_segments.loc[exposure_segments["strategy_id"].astype(str).eq(best_id)]
    best_cash = cash_timing.loc[cash_timing["strategy_id"].astype(str).eq(best_id)]
    best_missed = missed_rallies.loc[missed_rallies["strategy_id"].astype(str).eq(best_id)]
    best_drawdown = drawdown_avoidance.loc[drawdown_avoidance["strategy_id"].astype(str).eq(best_id)]
    best_rolling = rolling_windows.loc[rolling_windows["strategy_id"].astype(str).eq(best_id)]
    threshold_table = _threshold_comparison_table(candidate_set)

    return "\n".join(
        [
            "# QQQ-Only RF Signal Review",
            "",
            "## 1. Executive summary.",
            "",
            _markdown_table(
                summary,
                [
                    "best_strategy_name",
                    "best_threshold",
                    "best_total_return",
                    "qqq_buy_hold_total_return",
                    "best_strategy_vs_qqq",
                    "best_max_drawdown",
                    "qqq_buy_hold_max_drawdown",
                    "best_drawdown_improvement_vs_qqq",
                    "best_positive_vs_qqq_window_rate",
                    "production_ready",
                ],
            ),
            "",
            (
                "The threshold 0.45 binary weekly RF setting remains the strongest "
                "current row. It is more risk-on than the 0.50/0.55/0.60 references, "
                "but still improves maximum drawdown versus QQQ."
            ),
            "",
            "## 2. What improved versus the original best candidate.",
            "",
            (
                "The original RF candidate used threshold 0.50. The threshold 0.45 "
                "variant raises full-period edge versus QQQ while keeping drawdown "
                "improvement positive, suggesting the family benefits from staying "
                "risk-on more often instead of using the stricter cut point."
            ),
            "",
            _markdown_table(
                candidate_set.loc[
                    candidate_set["candidate_role"].isin(["best_setting", "threshold_050_reference"])
                ],
                [
                    "candidate_role",
                    "threshold",
                    "transaction_cost_bps",
                    "total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "average_exposure",
                    "cash_days_pct",
                ],
            ),
            "",
            "## 3. Candidate set reviewed.",
            "",
            _markdown_table(
                candidate_set,
                [
                    "candidate_rank",
                    "candidate_role",
                    "strategy_name",
                    "threshold",
                    "min_holding_days",
                    "transaction_cost_bps",
                    "exposure_mode",
                    "strategy_vs_qqq",
                    "decision_label",
                    "production_ready",
                ],
            ),
            "",
            "## 4. Equity and drawdown behavior.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "candidate_role",
                    "threshold",
                    "transaction_cost_bps",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "drawdown_improvement_vs_qqq",
                ],
            ),
            "",
            f"![Equity curve]({EQUITY_CHART_NAME})",
            "",
            f"![Drawdown curve]({DRAWDOWN_CHART_NAME})",
            "",
            "## 5. QQQ/cash exposure path.",
            "",
            _markdown_table(
                best_segments.head(24),
                [
                    "segment_type",
                    "start_date",
                    "end_date",
                    "trading_days",
                    "average_exposure",
                    "qqq_return_during_segment",
                    "strategy_return_during_segment",
                    "segment_interpretation",
                ],
            ),
            "",
            f"![Exposure timeline]({EXPOSURE_CHART_NAME})",
            "",
            "## 6. Cash timing review.",
            "",
            _markdown_table(
                best_cash,
                [
                    "start_date",
                    "end_date",
                    "trading_days",
                    "qqq_return_during_cash",
                    "strategy_return_during_cash",
                    "max_qqq_drawdown_during_cash",
                    "cash_timing_label",
                ],
            ),
            "",
            f"![Cash timing]({CASH_RETURNS_CHART_NAME})",
            "",
            "## 7. Missed rally review.",
            "",
            _markdown_table(
                best_missed.head(12),
                [
                    "start_date",
                    "end_date",
                    "trading_days",
                    "qqq_return",
                    "strategy_return",
                    "missed_return",
                    "explains_underperformance_windows",
                    "underperformance_window_label",
                ],
            ),
            "",
            "## 8. Drawdown avoidance review.",
            "",
            _markdown_table(
                best_drawdown,
                [
                    "period_name",
                    "strategy_return",
                    "qqq_return",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "average_exposure",
                    "cash_days_pct",
                    "interpretation",
                ],
            ),
            "",
            "## 9. Rolling-window behavior.",
            "",
            _markdown_table(
                _rolling_summary(best_rolling),
                [
                    "window_years",
                    "rolling_window_count",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "best_strategy_vs_qqq",
                    "drawdown_improvement_window_rate",
                    "recent_window_behavior",
                ],
            ),
            "",
            f"![Rolling edge]({ROLLING_CHART_NAME})",
            "",
            "## 10. Threshold comparison.",
            "",
            _markdown_table(
                threshold_table,
                [
                    "threshold",
                    "transaction_cost_bps",
                    "total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "average_exposure",
                    "trade_count",
                ],
            ),
            "",
            f"![Threshold comparison]({THRESHOLD_CHART_NAME})",
            "",
            "## 11. Final recommendation:",
            "",
            "- Best setting to carry forward: `threshold 0.45, min_holding_days 0, binary_0_100, weekly`.",
            "- Whether to run another confirmation: `yes, narrow second confirmation only`.",
            "- Whether to prepare paper-tracking: `candidate review only after second confirmation`.",
            "- Whether to promote to production: `no`.",
            "- Production-ready: `false` for every row.",
            "",
            "## 12. Caveats:",
            "",
            "- Research-only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ/cash only.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Other ETFs/reference series are features only.",
            "- The reviewed RF family may be a selected winner, so this is not independent validation.",
            "",
        ]
    )


def _write_signal_review_charts(
    *,
    artifacts: QQQOnlyRFSignalReviewArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_id = _best_strategy_id(artifacts.candidate_set)
    best_equity = _strategy_frame(daily_equity, best_id)
    comparison_ids = _chart_strategy_ids(artifacts.candidate_set)

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in comparison_ids:
        subset = _strategy_frame(daily_equity, strategy_id)
        if subset.empty:
            continue
        ax.plot(pd.to_datetime(subset["date"]), subset["equity"], label=_short_label(subset["strategy_name"].iloc[0]))
    if not best_equity.empty:
        ax.plot(
            pd.to_datetime(best_equity["date"]),
            best_equity["qqq_buy_hold_equity"],
            color="black",
            linewidth=2,
            label="QQQ buy-and-hold",
        )
    ax.set_title("QQQ-only RF signal review: equity curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[EQUITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in comparison_ids:
        subset = _strategy_frame(daily_drawdown, strategy_id)
        if subset.empty:
            continue
        ax.plot(pd.to_datetime(subset["date"]), subset["drawdown"], label=_short_label(subset["strategy_name"].iloc[0]))
    best_dd = _strategy_frame(daily_drawdown, best_id)
    if not best_dd.empty:
        ax.plot(
            pd.to_datetime(best_dd["date"]),
            best_dd["qqq_buy_hold_drawdown"],
            color="black",
            linewidth=2,
            label="QQQ buy-and-hold",
        )
    ax.set_title("QQQ-only RF signal review: drawdown curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for strategy_id in comparison_ids[:4]:
        subset = _strategy_frame(exposure_log, strategy_id)
        if subset.empty:
            continue
        ax.step(
            pd.to_datetime(subset["date"]),
            pd.to_numeric(subset["effective_exposure"], errors="coerce"),
            where="post",
            label=_short_label(subset["strategy_name"].iloc[0]),
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("QQQ-only RF signal review: exposure timeline")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cash = artifacts.cash_timing.loc[artifacts.cash_timing["strategy_id"].astype(str).eq(best_id)].copy()
    if not cash.empty:
        colors = cash["cash_timing_label"].map(
            {
                "useful_risk_off": "#2a9d8f",
                "missed_rally": "#e76f51",
                "neutral": "#6c757d",
                "whipsaw": "#f4a261",
            }
        ).fillna("#6c757d")
        ax.scatter(
            cash["qqq_return_during_cash"],
            cash["strategy_return_during_cash"],
            c=colors,
            s=60,
            alpha=0.85,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Cash/reduced-risk segments: strategy return vs QQQ")
    ax.set_xlabel("QQQ return during segment")
    ax.set_ylabel("Strategy return during segment")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[CASH_RETURNS_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    rolling = artifacts.rolling_windows.loc[artifacts.rolling_windows["strategy_id"].astype(str).eq(best_id)]
    if not rolling.empty:
        for years, group in rolling.groupby("window_years", sort=True):
            ax.plot(
                pd.to_datetime(group["window_end"]),
                pd.to_numeric(group["strategy_vs_qqq"], errors="coerce"),
                marker="o",
                linewidth=1.3,
                label=f"{int(years)}y windows",
            )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("QQQ-only RF signal review: rolling edge vs QQQ")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[ROLLING_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    threshold_table = _threshold_comparison_table(artifacts.candidate_set)
    if not threshold_table.empty:
        labels = threshold_table["threshold"].map(lambda value: f"{value:.2f}") + " / " + threshold_table[
            "transaction_cost_bps"
        ].map(lambda value: f"{value:.0f}bps")
        x = np.arange(len(labels))
        ax.bar(x, threshold_table["strategy_vs_qqq"], color="#457b9d")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Threshold comparison: strategy edge vs QQQ")
    ax.set_ylabel("Strategy vs QQQ total return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[THRESHOLD_CHART_NAME], dpi=140)
    plt.close(fig)


def _candidate_role(row: pd.Series) -> str:
    if str(row.get("strategy_name")) == "qqq_buy_hold":
        return "benchmark"
    exposure = str(row.get("exposure_mode"))
    threshold = _float(row.get("threshold"))
    min_hold = int(_float(row.get("min_holding_days")) or 0)
    cost = _float(row.get("transaction_cost_bps"))
    if exposure == "binary_0_100" and np.isclose(threshold, 0.45) and min_hold == 0 and np.isclose(cost, 0.0):
        return "best_setting"
    if exposure == "binary_0_100" and np.isclose(threshold, 0.45) and min_hold == 0:
        return "primary_cost_sensitivity"
    if exposure == "binary_0_100" and np.isclose(threshold, 0.45) and min_hold == 10:
        return "min_holding_reference"
    if exposure == "binary_0_100" and np.isclose(threshold, 0.50):
        return "threshold_050_reference"
    if exposure == "partial_50_100":
        return "smoother_reference"
    return "threshold_reference"


def _review_decision_label(row: pd.Series) -> str:
    role = str(row.get("candidate_role"))
    if role == "benchmark":
        return "keep_as_threshold_reference"
    if role == "best_setting":
        return "keep_for_second_confirmation"
    if role in {"primary_cost_sensitivity", "min_holding_reference"}:
        return "keep_for_second_confirmation"
    if role == "smoother_reference":
        return "keep_for_paper_tracking_candidate_review"
    if role in {"threshold_050_reference", "threshold_reference"}:
        return "keep_as_threshold_reference"
    return "reject_deprioritize"


def _merge_decision_context(
    selected: pd.DataFrame,
    decision_matrix: pd.DataFrame | None,
) -> pd.DataFrame:
    if decision_matrix is None or decision_matrix.empty or "strategy_id" not in decision_matrix.columns:
        return selected
    context_columns = [
        column
        for column in ["strategy_id", "recommendation", "cost_10bps_strategy_vs_qqq"]
        if column in decision_matrix.columns
    ]
    if len(context_columns) <= 1:
        return selected
    return selected.merge(
        decision_matrix.loc[:, context_columns].drop_duplicates("strategy_id"),
        on="strategy_id",
        how="left",
        suffixes=("", "_confirmation"),
    )


def _sort_candidate_set(frame: pd.DataFrame) -> pd.DataFrame:
    role_rank = {
        "best_setting": 0,
        "primary_cost_sensitivity": 1,
        "min_holding_reference": 2,
        "threshold_050_reference": 3,
        "threshold_reference": 4,
        "smoother_reference": 5,
        "benchmark": 6,
    }
    out = frame.copy()
    out["_role_rank"] = out["candidate_role"].map(role_rank).fillna(9)
    return out.sort_values(
        ["_role_rank", "threshold", "min_holding_days", "transaction_cost_bps", "strategy_vs_qqq"],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    ).drop(columns=["_role_rank"]).reset_index(drop=True)


def _segment_interpretation(
    *,
    segment_type: str,
    average_exposure: float,
    qqq_return: float,
    strategy_return: float,
    qqq_drawdown: float,
) -> str:
    if segment_type == "QQQ":
        return "risk_on_participation"
    if qqq_return <= -0.02 or qqq_drawdown <= -0.05:
        return "reduced_exposure_helped_drawdown"
    if qqq_return >= 0.03 and strategy_return < qqq_return:
        return "reduced_exposure_missed_rally"
    if average_exposure > 0.0:
        return "partial_defensive_exposure"
    return "cash_neutral"


def _drawdown_period_interpretation(
    *,
    period_name: str,
    strategy_vs_qqq: float,
    drawdown_improvement: float,
    average_exposure: float,
) -> str:
    if drawdown_improvement > 0.02 and strategy_vs_qqq >= 0.0:
        return f"{period_name}: reduced risk while preserving or improving return"
    if drawdown_improvement > 0.02:
        return f"{period_name}: defensive benefit but return lagged QQQ"
    if strategy_vs_qqq > 0.0 and average_exposure > 0.85:
        return f"{period_name}: risk-on participation drove edge"
    if strategy_vs_qqq < 0.0 and average_exposure < 0.85:
        return f"{period_name}: lower exposure likely missed upside"
    return f"{period_name}: broadly benchmark-like behavior"


def _period_metrics(frame: pd.DataFrame) -> dict[str, float]:
    strategy_total = _segment_total_return(frame["equity"])
    qqq_total = _segment_total_return(frame["qqq_buy_hold_equity"])
    strategy_dd = _segment_max_drawdown(frame["equity"])
    qqq_dd = _segment_max_drawdown(frame["qqq_buy_hold_equity"])
    return {
        "strategy_return": strategy_total,
        "qqq_return": qqq_total,
        "strategy_vs_qqq": strategy_total - qqq_total,
        "strategy_max_drawdown": strategy_dd,
        "qqq_max_drawdown": qqq_dd,
        "drawdown_improvement_vs_qqq": strategy_dd - qqq_dd,
    }


def _cash_timing_summary(cash_timing: pd.DataFrame) -> pd.DataFrame:
    if cash_timing.empty:
        return pd.DataFrame(columns=["strategy_id", "cash_timing_summary"])
    rows = []
    for strategy_id, group in cash_timing.groupby("strategy_id", sort=False):
        counts = group["cash_timing_label"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "strategy_id": strategy_id,
                "cash_timing_summary": ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
            }
        )
    return pd.DataFrame(rows)


def _missed_rally_summary(missed_rallies: pd.DataFrame) -> pd.DataFrame:
    if missed_rallies.empty:
        return pd.DataFrame(columns=["strategy_id", "missed_rally_count"])
    rows = []
    for strategy_id, group in missed_rallies.groupby("strategy_id", sort=False):
        rows.append({"strategy_id": strategy_id, "missed_rally_count": int(len(group))})
    return pd.DataFrame(rows)


def _rolling_summary(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    if rolling_windows.empty:
        return pd.DataFrame()
    rows = []
    for years, group in rolling_windows.groupby("window_years", sort=True):
        strategy_vs = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        dd = pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
        recent = group.loc[pd.to_datetime(group["window_end"]).idxmax(), "recent_window_behavior"]
        rows.append(
            {
                "window_years": years,
                "rolling_window_count": int(strategy_vs.notna().sum()),
                "positive_vs_qqq_window_rate": float((strategy_vs > 0.0).mean()),
                "worst_strategy_vs_qqq": float(strategy_vs.min()),
                "best_strategy_vs_qqq": float(strategy_vs.max()),
                "drawdown_improvement_window_rate": float((dd > 0.0).mean()),
                "recent_window_behavior": recent,
            }
        )
    return pd.DataFrame(rows)


def _threshold_comparison_table(candidate_set: pd.DataFrame) -> pd.DataFrame:
    if candidate_set.empty:
        return pd.DataFrame()
    table = candidate_set.loc[
        candidate_set["exposure_mode"].astype(str).eq("binary_0_100")
        & pd.to_numeric(candidate_set["min_holding_days"], errors="coerce").fillna(0).eq(0)
        & candidate_set["candidate_role"].astype(str).ne("benchmark")
    ].copy()
    table = table.loc[pd.to_numeric(table["transaction_cost_bps"], errors="coerce").isin([0.0, 10.0])]
    return table.sort_values(["threshold", "transaction_cost_bps"], kind="mergesort")


def _overlapping_underperformance_window(
    *,
    rolling_windows: pd.DataFrame,
    strategy_id: str,
    start_date: str,
    end_date: str,
) -> str:
    if rolling_windows.empty:
        return ""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    subset = rolling_windows.loc[
        rolling_windows["strategy_id"].astype(str).eq(strategy_id)
        & (pd.to_numeric(rolling_windows["strategy_vs_qqq"], errors="coerce") < 0.0)
    ].copy()
    if subset.empty:
        return ""
    subset["window_start_dt"] = pd.to_datetime(subset["window_start"])
    subset["window_end_dt"] = pd.to_datetime(subset["window_end"])
    overlap = subset.loc[(subset["window_start_dt"] <= end) & (subset["window_end_dt"] >= start)]
    if overlap.empty:
        return ""
    worst_idx = pd.to_numeric(overlap["strategy_vs_qqq"], errors="coerce").idxmin()
    return str(overlap.loc[worst_idx, "window_label"])


def _chart_strategy_ids(candidate_set: pd.DataFrame) -> list[str]:
    roles = ["best_setting", "threshold_050_reference", "threshold_reference", "smoother_reference"]
    ids = []
    for role in roles:
        row = candidate_set.loc[
            candidate_set["candidate_role"].astype(str).eq(role)
            & np.isclose(pd.to_numeric(candidate_set["transaction_cost_bps"], errors="coerce"), 0.0)
        ].head(1)
        if not row.empty:
            ids.append(str(row["strategy_id"].iloc[0]))
    return ids


def _best_strategy_id(candidate_set: pd.DataFrame) -> str:
    if candidate_set.empty:
        return ""
    best = candidate_set.loc[candidate_set["candidate_role"].astype(str).eq("best_setting")].head(1)
    if best.empty:
        best = candidate_set.head(1)
    return str(best["strategy_id"].iloc[0])


def _filter_strategy_ids(frame: pd.DataFrame, strategy_ids: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.loc[frame["strategy_id"].astype(str).isin(set(strategy_ids))].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _strategy_frame(frame: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    if frame.empty or "strategy_id" not in frame.columns:
        return pd.DataFrame()
    out = frame.loc[frame["strategy_id"].astype(str).eq(strategy_id)].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date", kind="mergesort")
    return out.reset_index(drop=True)


def _date_slice(frame: pd.DataFrame, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.loc[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].copy()


def _segment_total_return(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] == 0.0:
        return np.nan
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def _segment_max_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty or values.iloc[0] == 0.0:
        return np.nan
    return calculate_max_drawdown(values / values.iloc[0])


def _max_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    return pd.Timestamp(dates.max()) if not dates.empty else None


def _short_label(value: object) -> str:
    text = str(value)
    return (
        text.replace("random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_", "")
        .replace("_weekly", "")
        .replace("binary_0_100", "binary")
        .replace("partial_50_100", "partial")
    )


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, [*columns, *[column for column in out.columns if column not in columns]]]


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{float(value):.6g}"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    return str(value).replace("\n", " ")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
