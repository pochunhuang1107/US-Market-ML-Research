"""Focused deep dive for the best QQQ-only timing candidate.

This module reads completed QQQ-only model-zoo artifacts and writes a
confirmation-oriented diagnostic bundle. It deliberately does not train models,
run a broad sweep, export production signals, or create model cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown

OUTPUT_PREFIX = "qqq_only_best_candidate_deep_dive"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_best_candidate_deep_dive")
BEST_STRATEGY_NAME = (
    "random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_"
    "binary_0_100_weekly_cost0"
)
COST_SENSITIVITY_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
DAILY_DRAWDOWN_NAME = f"{OUTPUT_PREFIX}_daily_drawdown.csv"
EXPOSURE_LOG_NAME = f"{OUTPUT_PREFIX}_exposure_log.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
COST_SENSITIVITY_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = "qqq_only_best_candidate_equity_curve.png"
DRAWDOWN_CHART_NAME = "qqq_only_best_candidate_drawdown_curve.png"
EXPOSURE_CHART_NAME = "qqq_only_best_candidate_exposure_timeline.png"
ROLLING_CHART_NAME = "qqq_only_best_candidate_rolling_vs_qqq.png"
SUBPERIOD_RETURNS_CHART_NAME = "qqq_only_best_candidate_subperiod_returns.png"
SUBPERIOD_DRAWDOWNS_CHART_NAME = "qqq_only_best_candidate_subperiod_drawdowns.png"

REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Why this candidate matters.",
    "## 3. Candidate configuration.",
    "## 4. Full-period performance versus QQQ.",
    "## 5. Equity and drawdown charts.",
    "## 6. Rolling-window behavior.",
    "## 7. Subperiod behavior.",
    "## 8. QQQ/cash exposure path.",
    "## 9. Cost sensitivity.",
    "## 10. Candidate comparison against other QQQ timing candidates.",
    "## 11. Decision and recommendation.",
    "## 12. Caveats:",
)

SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("full", "2019-01-01", "2026-05-15"),
    ("covid_crash_recovery", "2020-01-01", "2020-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_bull_2023_2024", "2023-01-01", "2024-12-31"),
    ("recent_2025_2026", "2025-01-01", "2026-05-15"),
)

ROLLING_SPECS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (2, 3),
    (3, 6),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "candidate_rank",
    "candidate_role",
    "source_candidate_id",
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
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "return_to_drawdown_vs_qqq",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "qqq_days",
    "cash_days",
    "trade_count",
    "average_holding_days",
    "longest_qqq_period_days",
    "longest_cash_period_days",
    "cash_segments_avoiding_major_qqq_drawdown",
    "cash_segments_missing_major_qqq_rally",
    "decision_label",
    "production_ready",
)

ROLLING_WINDOW_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "window_years",
    "step_months",
    "window_label",
    "window_start",
    "window_end",
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "underperformed_qqq",
    "drawdown_improved",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "best_strategy_vs_qqq",
    "worst_window_label",
    "best_window_label",
    "drawdown_improvement_window_rate",
)

SUBPERIOD_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "period_name",
    "period_start",
    "period_end",
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
)

COST_SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "transaction_cost_bps",
    "total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "return_to_drawdown",
    "estimated_transaction_cost",
    "trade_count",
    "cost_method",
    "decision_at_cost",
)

DECISION_MATRIX_COLUMNS: tuple[str, ...] = (
    "candidate_rank",
    "candidate_role",
    "strategy_id",
    "strategy_name",
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "return_to_drawdown_vs_qqq",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "best_strategy_vs_qqq",
    "worst_window_label",
    "best_window_label",
    "drawdown_improvement_window_rate",
    "cost_10bps_strategy_vs_qqq",
    "beats_qqq_return",
    "improves_max_drawdown",
    "improves_return_to_drawdown",
    "positive_edge_after_10bps",
    "decision",
    "recommendation",
    "production_ready",
)


@dataclass(frozen=True)
class QQQOnlyBestCandidateDeepDiveArtifacts:
    """All in-memory tables for the focused deep-dive bundle."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    exposure_log: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def build_qqq_only_best_candidate_deep_dive_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    decision_matrix: pd.DataFrame | None = None,
    top_n_candidates: int = 10,
    cost_sensitivity_bps: Sequence[float] = COST_SENSITIVITY_BPS,
) -> QQQOnlyBestCandidateDeepDiveArtifacts:
    """Build the focused deep-dive tables and report from completed artifacts."""

    source_results = summary.copy() if summary is not None and not summary.empty else results.copy()
    selected = select_best_candidate_rows(
        source_results,
        decision_matrix=decision_matrix,
        top_n_candidates=top_n_candidates,
    )
    selected_ids = selected["strategy_id"].astype(str).tolist()
    selected_equity = _filter_strategy_ids(daily_equity, selected_ids)
    selected_drawdown = _filter_strategy_ids(daily_drawdown, selected_ids)
    selected_exposure = _filter_strategy_ids(exposure_log, selected_ids)
    confirm_held_assets_are_qqq_cash_only(selected_exposure)

    subperiods = build_subperiods(selected_equity, selected_exposure)
    rolling = build_rolling_windows(selected_equity)
    costs = build_cost_sensitivity(
        selected_equity,
        cost_sensitivity_bps=cost_sensitivity_bps,
    )
    summary_table = build_deep_dive_summary(selected, selected_equity, selected_exposure)
    decisions = build_deep_dive_decision_matrix(
        summary=summary_table,
        rolling_windows=rolling,
        cost_sensitivity=costs,
    )
    summary_table = summary_table.merge(
        decisions.loc[:, ["strategy_id", "decision"]],
        on="strategy_id",
        how="left",
    )
    report = render_qqq_only_best_candidate_deep_dive_markdown(
        summary=summary_table,
        subperiods=subperiods,
        rolling_windows=rolling,
        cost_sensitivity=costs,
        decision_matrix=decisions,
    )
    return QQQOnlyBestCandidateDeepDiveArtifacts(
        summary=summary_table,
        daily_equity=selected_equity,
        daily_drawdown=selected_drawdown,
        exposure_log=selected_exposure,
        subperiods=subperiods,
        rolling_windows=rolling,
        cost_sensitivity=costs,
        decision_matrix=decisions,
        report_markdown=report,
    )


def select_best_candidate_rows(
    results: pd.DataFrame,
    *,
    decision_matrix: pd.DataFrame | None = None,
    top_n_candidates: int = 10,
) -> pd.DataFrame:
    """Select the best row, top QQQ timing candidates, and one defensive context row."""

    frame = results.copy()
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for column in (
        "strategy_id",
        "strategy_name",
        "decision_label",
        "strategy_vs_qqq",
        "return_to_drawdown",
        "drawdown_improvement_vs_qqq",
    ):
        if column not in frame.columns:
            frame[column] = np.nan
    if decision_matrix is not None and not decision_matrix.empty:
        ids = decision_matrix.loc[:, [c for c in ("strategy_id", "candidate_id") if c in decision_matrix]]
        if {"strategy_id", "candidate_id"}.issubset(ids.columns):
            frame = frame.merge(
                ids.rename(columns={"candidate_id": "source_candidate_id"}),
                on="strategy_id",
                how="left",
            )
    if "source_candidate_id" not in frame.columns:
        frame["source_candidate_id"] = ""

    not_qqq_bh = ~frame["strategy_name"].astype(str).eq("qqq_buy_hold")
    timing = frame.loc[not_qqq_bh & frame["decision_label"].astype(str).eq("qqq_timing_candidate")]
    timing = timing.sort_values(
        ["strategy_vs_qqq", "return_to_drawdown", "drawdown_improvement_vs_qqq"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(top_n_candidates)

    selected_parts = []
    best = frame.loc[frame["strategy_name"].astype(str).eq(BEST_STRATEGY_NAME)].head(1)
    if best.empty:
        best = timing.head(1)
    if not best.empty:
        selected_parts.append(best.assign(candidate_role="best_candidate"))
    if not timing.empty:
        selected_parts.append(timing.assign(candidate_role="top_timing_candidate"))

    defensive = frame.loc[
        not_qqq_bh
        & frame["decision_label"].astype(str).eq("defensive_reference")
        & (pd.to_numeric(frame["strategy_vs_qqq"], errors="coerce") >= -0.10)
        & (pd.to_numeric(frame["drawdown_improvement_vs_qqq"], errors="coerce") >= 0.05)
    ].sort_values(
        ["strategy_vs_qqq", "drawdown_improvement_vs_qqq"],
        ascending=[False, False],
        kind="mergesort",
    )
    if not defensive.empty:
        selected_parts.append(defensive.head(1).assign(candidate_role="defensive_context"))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    role_order = {"best_candidate": 0, "top_timing_candidate": 1, "defensive_context": 2}
    selected["_role_order"] = selected["candidate_role"].map(role_order).fillna(9)
    selected = selected.sort_values(
        ["_role_order", "strategy_vs_qqq", "return_to_drawdown", "drawdown_improvement_vs_qqq"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop_duplicates("strategy_id", keep="first")
    selected = selected.drop(columns=["_role_order"]).reset_index(drop=True)
    selected["candidate_rank"] = np.arange(1, len(selected) + 1)
    selected["production_ready"] = False
    return _ensure_columns(selected, SUMMARY_COLUMNS)


def confirm_held_assets_are_qqq_cash_only(exposure_log: pd.DataFrame) -> None:
    """Raise if the selected exposure log holds anything other than QQQ and cash."""

    if "held_assets" not in exposure_log.columns:
        raise ValueError("Exposure log is missing held_assets.")
    allowed = {"QQQ", "cash", "QQQ,cash", "QQQ/cash", ""}
    held = set(exposure_log["held_assets"].fillna("").astype(str).unique())
    bad = sorted(held.difference(allowed))
    if bad:
        raise ValueError("Selected deep-dive rows hold non-QQQ/cash assets: " + ", ".join(bad))
    weights = pd.to_numeric(exposure_log.get("qqq_weight", np.nan), errors="coerce") + pd.to_numeric(
        exposure_log.get("cash_weight", np.nan),
        errors="coerce",
    )
    if weights.notna().any() and not np.allclose(weights.dropna(), 1.0):
        raise ValueError("Selected QQQ/cash weights do not sum to 1.0.")


def build_deep_dive_summary(
    selected_rows: pd.DataFrame,
    daily_equity: pd.DataFrame,
    exposure_log: pd.DataFrame,
) -> pd.DataFrame:
    """Add exposure-review diagnostics to selected source rows."""

    rows: list[dict[str, object]] = []
    for source in selected_rows.to_dict("records"):
        strategy_id = str(source["strategy_id"])
        equity = _strategy_frame(daily_equity, strategy_id)
        exposure = _strategy_frame(exposure_log, strategy_id)
        exposure_values = pd.to_numeric(exposure.get("effective_exposure", pd.Series(dtype=float)), errors="coerce")
        qqq_days = int((exposure_values >= 1.0 - 1e-9).sum())
        cash_days = int((exposure_values <= 1e-9).sum())
        periods = _exposure_periods(exposure, equity)
        qqq_periods = [p for p in periods if p["exposure_bucket"] == "QQQ"]
        cash_periods = [p for p in periods if p["exposure_bucket"] == "cash"]
        source.update(
            {
                "qqq_days": qqq_days,
                "cash_days": cash_days,
                "longest_qqq_period_days": max(
                    [int(p["trading_days"]) for p in qqq_periods],
                    default=0,
                ),
                "longest_cash_period_days": max(
                    [int(p["trading_days"]) for p in cash_periods],
                    default=0,
                ),
                "cash_segments_avoiding_major_qqq_drawdown": sum(
                    1
                    for p in cash_periods
                    if float(p["qqq_segment_return"]) <= -0.05
                    or float(p["qqq_segment_max_drawdown"]) <= -0.10
                ),
                "cash_segments_missing_major_qqq_rally": sum(
                    1 for p in cash_periods if float(p["qqq_segment_return"]) >= 0.05
                ),
                "production_ready": False,
            }
        )
        rows.append(source)
    return _ensure_columns(pd.DataFrame(rows), SUMMARY_COLUMNS)


def build_subperiods(
    daily_equity: pd.DataFrame,
    exposure_log: pd.DataFrame,
    *,
    subperiods: Sequence[tuple[str, str, str]] = SUBPERIODS,
) -> pd.DataFrame:
    """Build required subperiod comparison rows for every selected strategy."""

    rows: list[dict[str, object]] = []
    for strategy_id, strategy_equity in daily_equity.groupby("strategy_id", sort=False):
        strategy_id = str(strategy_id)
        strategy_name = str(strategy_equity["strategy_name"].iloc[0])
        strategy_exposure = _strategy_frame(exposure_log, strategy_id)
        for period_name, period_start, period_end in subperiods:
            period_equity = _date_slice(strategy_equity, period_start, period_end)
            period_exposure = _date_slice(strategy_exposure, period_start, period_end)
            if period_equity.empty:
                continue
            metrics = _period_metrics(period_equity)
            exposure_values = pd.to_numeric(
                period_exposure.get("effective_exposure", pd.Series(dtype=float)),
                errors="coerce",
            )
            turnover = pd.to_numeric(
                period_equity.get("turnover", pd.Series(dtype=float)),
                errors="coerce",
            )
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "period_name": period_name,
                    "period_start": period_start,
                    "period_end": period_end,
                    "strategy_total_return": metrics["strategy_total_return"],
                    "qqq_total_return": metrics["qqq_total_return"],
                    "strategy_vs_qqq": metrics["strategy_vs_qqq"],
                    "strategy_max_drawdown": metrics["strategy_max_drawdown"],
                    "qqq_max_drawdown": metrics["qqq_max_drawdown"],
                    "drawdown_improvement_vs_qqq": metrics["drawdown_improvement_vs_qqq"],
                    "average_exposure": float(exposure_values.mean()) if not exposure_values.empty else np.nan,
                    "cash_days_pct": float((exposure_values <= 1e-9).mean()) if not exposure_values.empty else np.nan,
                    "trade_count": int((turnover > 1e-9).sum()) if not turnover.empty else 0,
                }
            )
    return _ensure_columns(pd.DataFrame(rows), SUBPERIOD_COLUMNS)


def build_rolling_windows(
    daily_equity: pd.DataFrame,
    *,
    rolling_specs: Sequence[tuple[int, int]] = ROLLING_SPECS,
) -> pd.DataFrame:
    """Build 1y/2y/3y rolling-window rows and repeated aggregate diagnostics."""

    rows: list[dict[str, object]] = []
    for strategy_id, strategy_equity in daily_equity.groupby("strategy_id", sort=False):
        strategy_id = str(strategy_id)
        strategy_name = str(strategy_equity["strategy_name"].iloc[0])
        frame = strategy_equity.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
        if frame.empty:
            continue
        first_date = pd.Timestamp(frame["date"].iloc[0])
        last_date = pd.Timestamp(frame["date"].iloc[-1])
        for years, step_months in rolling_specs:
            cursor = first_date
            while cursor + pd.DateOffset(years=years) <= last_date + pd.Timedelta(days=1):
                target_end = cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1)
                subset = frame.loc[(frame["date"] >= cursor) & (frame["date"] <= target_end)]
                if len(subset) >= max(60, int(126 * years)):
                    metrics = _period_metrics(subset)
                    start_label = str(pd.Timestamp(subset["date"].iloc[0]).date())
                    end_label = str(pd.Timestamp(subset["date"].iloc[-1]).date())
                    label = f"{start_label} to {end_label} ({years}y)"
                    rows.append(
                        {
                            "strategy_id": strategy_id,
                            "strategy_name": strategy_name,
                            "window_years": years,
                            "step_months": step_months,
                            "window_label": label,
                            "window_start": start_label,
                            "window_end": end_label,
                            "strategy_total_return": metrics["strategy_total_return"],
                            "qqq_total_return": metrics["qqq_total_return"],
                            "strategy_vs_qqq": metrics["strategy_vs_qqq"],
                            "strategy_max_drawdown": metrics["strategy_max_drawdown"],
                            "qqq_max_drawdown": metrics["qqq_max_drawdown"],
                            "drawdown_improvement_vs_qqq": metrics["drawdown_improvement_vs_qqq"],
                            "underperformed_qqq": bool(metrics["strategy_vs_qqq"] < 0.0),
                            "drawdown_improved": bool(metrics["drawdown_improvement_vs_qqq"] > 0.0),
                        }
                    )
                cursor = cursor + pd.DateOffset(months=step_months)
    rolling = pd.DataFrame(rows)
    if rolling.empty:
        return _ensure_columns(rolling, ROLLING_WINDOW_COLUMNS)
    aggregate_rows = []
    for (strategy_id, years), group in rolling.groupby(["strategy_id", "window_years"], sort=False):
        worst_idx = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce").idxmin()
        best_idx = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce").idxmax()
        aggregate_rows.append(
            {
                "strategy_id": strategy_id,
                "window_years": years,
                "positive_vs_qqq_window_rate": float((group["strategy_vs_qqq"] > 0.0).mean()),
                "qqq_underperformance_rate": float((group["strategy_vs_qqq"] < 0.0).mean()),
                "worst_strategy_vs_qqq": float(group["strategy_vs_qqq"].min()),
                "best_strategy_vs_qqq": float(group["strategy_vs_qqq"].max()),
                "worst_window_label": str(group.loc[worst_idx, "window_label"]),
                "best_window_label": str(group.loc[best_idx, "window_label"]),
                "drawdown_improvement_window_rate": float(
                    (group["drawdown_improvement_vs_qqq"] > 0.0).mean()
                ),
            }
        )
    aggregates = pd.DataFrame(aggregate_rows)
    rolling = rolling.merge(aggregates, on=["strategy_id", "window_years"], how="left")
    return _ensure_columns(rolling, ROLLING_WINDOW_COLUMNS)


def build_cost_sensitivity(
    daily_equity: pd.DataFrame,
    *,
    cost_sensitivity_bps: Sequence[float] = COST_SENSITIVITY_BPS,
) -> pd.DataFrame:
    """Replay selected exposure paths at requested transaction-cost assumptions."""

    rows: list[dict[str, object]] = []
    for strategy_id, strategy_equity in daily_equity.groupby("strategy_id", sort=False):
        strategy_id = str(strategy_id)
        strategy_name = str(strategy_equity["strategy_name"].iloc[0])
        frame = strategy_equity.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
        exposure = pd.to_numeric(frame["effective_exposure"], errors="coerce").fillna(0.0)
        qqq_return = pd.to_numeric(frame["qqq_return"], errors="coerce").fillna(0.0)
        turnover = exposure.diff().abs().fillna(0.0)
        trade_count = int((turnover > 1e-9).sum())
        qqq_equity = (1.0 + qqq_return).cumprod()
        qqq_total = float(qqq_equity.iloc[-1] - 1.0) if not qqq_equity.empty else np.nan
        for bps in cost_sensitivity_bps:
            transaction_cost = turnover * float(bps) / 10000.0
            daily_return = exposure * qqq_return - transaction_cost
            equity = (1.0 + daily_return).cumprod()
            total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan
            max_drawdown = calculate_max_drawdown(equity) if not equity.empty else np.nan
            strategy_vs_qqq = total_return - qqq_total
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "transaction_cost_bps": float(bps),
                    "total_return": total_return,
                    "qqq_total_return": qqq_total,
                    "strategy_vs_qqq": strategy_vs_qqq,
                    "max_drawdown": max_drawdown,
                    "return_to_drawdown": _return_to_drawdown(total_return, max_drawdown),
                    "estimated_transaction_cost": float(transaction_cost.sum()),
                    "trade_count": trade_count,
                    "cost_method": "replayed_from_existing_exposure_path",
                    "decision_at_cost": "edge_positive" if strategy_vs_qqq > 0.0 else "edge_lost",
                }
            )
    return _ensure_columns(pd.DataFrame(rows), COST_SENSITIVITY_COLUMNS)


def build_deep_dive_decision_matrix(
    *,
    summary: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Apply focused confirmation decision logic without marking production-ready."""

    rolling_summary = _summarize_rolling_all_windows(rolling_windows)
    cost_10 = (
        cost_sensitivity.loc[
            np.isclose(
                pd.to_numeric(cost_sensitivity["transaction_cost_bps"], errors="coerce"),
                10.0,
            ),
            ["strategy_id", "strategy_vs_qqq"],
        ]
        .rename(columns={"strategy_vs_qqq": "cost_10bps_strategy_vs_qqq"})
        .drop_duplicates("strategy_id")
    )
    rows = []
    merged = summary.merge(rolling_summary, on="strategy_id", how="left").merge(
        cost_10,
        on="strategy_id",
        how="left",
    )
    for row in merged.to_dict("records"):
        beats = _float(row.get("strategy_vs_qqq")) > 0.0
        improves_dd = _float(row.get("drawdown_improvement_vs_qqq")) > 0.0
        improves_rtd = _float(row.get("return_to_drawdown_vs_qqq")) > 0.0
        cost_edge = _float(row.get("cost_10bps_strategy_vs_qqq")) > 0.0
        rolling_rate = _float(row.get("positive_vs_qqq_window_rate_roll"))
        worst = _float(row.get("worst_strategy_vs_qqq_roll"))
        if beats and improves_dd and improves_rtd and cost_edge:
            decision = "keep_for_confirmation"
            recommendation = "Focused confirmation study; do not promote to production."
        elif beats and (improves_dd or improves_rtd) and (rolling_rate >= 0.45 or worst > -0.35):
            decision = "keep_for_signal_review"
            recommendation = "Review signal behavior before any confirmation run."
        else:
            decision = "reject_deprioritize"
            recommendation = "Do not prioritize unless new confirmation evidence appears."
        out = {
            "candidate_rank": row.get("candidate_rank"),
            "candidate_role": row.get("candidate_role"),
            "strategy_id": row.get("strategy_id"),
            "strategy_name": row.get("strategy_name"),
            "total_return": row.get("total_return"),
            "qqq_buy_hold_total_return": row.get("qqq_buy_hold_total_return"),
            "strategy_vs_qqq": row.get("strategy_vs_qqq"),
            "max_drawdown": row.get("max_drawdown"),
            "drawdown_improvement_vs_qqq": row.get("drawdown_improvement_vs_qqq"),
            "return_to_drawdown": row.get("return_to_drawdown"),
            "return_to_drawdown_vs_qqq": row.get("return_to_drawdown_vs_qqq"),
            "rolling_window_count": row.get("rolling_window_count"),
            "positive_vs_qqq_window_rate": row.get("positive_vs_qqq_window_rate_roll"),
            "qqq_underperformance_rate": row.get("qqq_underperformance_rate_roll"),
            "worst_strategy_vs_qqq": row.get("worst_strategy_vs_qqq_roll"),
            "best_strategy_vs_qqq": row.get("best_strategy_vs_qqq"),
            "worst_window_label": row.get("worst_window_label"),
            "best_window_label": row.get("best_window_label"),
            "drawdown_improvement_window_rate": row.get("drawdown_improvement_window_rate"),
            "cost_10bps_strategy_vs_qqq": row.get("cost_10bps_strategy_vs_qqq"),
            "beats_qqq_return": beats,
            "improves_max_drawdown": improves_dd,
            "improves_return_to_drawdown": improves_rtd,
            "positive_edge_after_10bps": cost_edge,
            "decision": decision,
            "recommendation": recommendation,
            "production_ready": False,
        }
        rows.append(out)
    return _ensure_columns(pd.DataFrame(rows), DECISION_MATRIX_COLUMNS)


def write_qqq_only_best_candidate_deep_dive_artifacts(
    *,
    artifacts: QQQOnlyBestCandidateDeepDiveArtifacts,
    output_dir: Path,
) -> dict[str, Path]:
    """Write CSVs, Markdown, and PNG charts for the focused deep dive."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.daily_equity.to_csv(paths[DAILY_EQUITY_NAME], index=False)
    artifacts.daily_drawdown.to_csv(paths[DAILY_DRAWDOWN_NAME], index=False)
    artifacts.exposure_log.to_csv(paths[EXPOSURE_LOG_NAME], index=False)
    artifacts.subperiods.to_csv(paths[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(paths[ROLLING_WINDOWS_NAME], index=False)
    artifacts.cost_sensitivity.to_csv(paths[COST_SENSITIVITY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    write_charts(artifacts=artifacts, output_dir=output_dir)
    return paths


def output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all expected output paths."""

    names = (
        SUMMARY_NAME,
        DAILY_EQUITY_NAME,
        DAILY_DRAWDOWN_NAME,
        EXPOSURE_LOG_NAME,
        SUBPERIODS_NAME,
        ROLLING_WINDOWS_NAME,
        COST_SENSITIVITY_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        EXPOSURE_CHART_NAME,
        ROLLING_CHART_NAME,
        SUBPERIOD_RETURNS_CHART_NAME,
        SUBPERIOD_DRAWDOWNS_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def write_charts(
    *,
    artifacts: QQQOnlyBestCandidateDeepDiveArtifacts,
    output_dir: Path,
) -> None:
    """Write required PNG charts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = output_paths(output_dir)
    best_id = _best_strategy_id(artifacts.summary)

    equity = _strategy_frame(artifacts.daily_equity, best_id)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not equity.empty:
        dates = pd.to_datetime(equity["date"])
        ax.plot(dates, equity["equity"], label="Selected strategy", linewidth=2)
        ax.plot(dates, equity["qqq_buy_hold_equity"], label="QQQ buy-and-hold", color="black")
    ax.set_title("Best QQQ-only timing candidate: equity curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[EQUITY_CHART_NAME], dpi=140)
    plt.close(fig)

    drawdown = _strategy_frame(artifacts.daily_drawdown, best_id)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not drawdown.empty:
        dates = pd.to_datetime(drawdown["date"])
        ax.plot(dates, drawdown["drawdown"], label="Selected strategy", linewidth=2)
        ax.plot(dates, drawdown["qqq_buy_hold_drawdown"], label="QQQ buy-and-hold", color="black")
    ax.set_title("Best QQQ-only timing candidate: drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    exposure = _strategy_frame(artifacts.exposure_log, best_id)
    fig, ax = plt.subplots(figsize=(11, 4))
    if not exposure.empty:
        ax.step(
            pd.to_datetime(exposure["date"]),
            pd.to_numeric(exposure["effective_exposure"], errors="coerce"),
            where="post",
            label="QQQ exposure",
            linewidth=1.7,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Best QQQ-only timing candidate: QQQ/cash exposure")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_CHART_NAME], dpi=140)
    plt.close(fig)

    rolling = artifacts.rolling_windows.loc[
        artifacts.rolling_windows["strategy_id"].astype(str).eq(best_id)
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
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
    ax.set_title("Best QQQ-only timing candidate: rolling return edge vs QQQ")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[ROLLING_CHART_NAME], dpi=140)
    plt.close(fig)

    sub = artifacts.subperiods.loc[
        artifacts.subperiods["strategy_id"].astype(str).eq(best_id)
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not sub.empty:
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["strategy_total_return"], width, label="Strategy")
        ax.bar(x + width / 2, sub["qqq_total_return"], width, label="QQQ")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["period_name"], rotation=25, ha="right")
    ax.set_title("Subperiod total returns")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[SUBPERIOD_RETURNS_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not sub.empty:
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["strategy_max_drawdown"], width, label="Strategy")
        ax.bar(x + width / 2, sub["qqq_max_drawdown"], width, label="QQQ")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["period_name"], rotation=25, ha="right")
    ax.set_title("Subperiod max drawdowns")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[SUBPERIOD_DRAWDOWNS_CHART_NAME], dpi=140)
    plt.close(fig)


def render_qqq_only_best_candidate_deep_dive_markdown(
    *,
    summary: pd.DataFrame,
    subperiods: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the focused confirmation report."""

    best_id = _best_strategy_id(summary)
    best = summary.loc[summary["strategy_id"].astype(str).eq(best_id)].head(1)
    best_decision = decision_matrix.loc[decision_matrix["strategy_id"].astype(str).eq(best_id)].head(1)
    best_name = str(best["strategy_name"].iloc[0]) if not best.empty else "none"
    decision = str(best_decision["decision"].iloc[0]) if not best_decision.empty else "none"
    selected_count = len(summary)
    best_subperiods = subperiods.loc[subperiods["strategy_id"].astype(str).eq(best_id)]
    best_costs = cost_sensitivity.loc[cost_sensitivity["strategy_id"].astype(str).eq(best_id)]
    best_rolling = rolling_windows.loc[rolling_windows["strategy_id"].astype(str).eq(best_id)]

    return "\n".join(
        [
            "# QQQ-Only Best Candidate Deep Dive",
            "",
            "## 1. Executive summary.",
            "",
            f"- Best strategy: `{best_name}`",
            f"- Selected rows reviewed: `{selected_count}`",
            f"- Decision: `{decision}`",
            "- Production-ready: `false`",
            (
                "- Interpretation: the best random-forest QQQ/cash timing row improves "
                "full-period return and drawdown versus QQQ, but rolling behavior still "
                "requires focused confirmation."
            ),
            "",
            "## 2. Why this candidate matters.",
            "",
            (
                "The selected row is the strongest QQQ-only timing candidate from the "
                "completed cross-market model zoo. It matters because it improves QQQ "
                "buy-and-hold by moving only between QQQ and cash; cross-market series "
                "remain inputs, not holdings."
            ),
            "",
            "## 3. Candidate configuration.",
            "",
            _markdown_table(
                best,
                [
                    "strategy_name",
                    "model_type",
                    "training_mode",
                    "target_name",
                    "feature_set_name",
                    "exposure_mode",
                    "rebalance_frequency",
                    "threshold",
                    "transaction_cost_bps",
                ],
            ),
            "",
            "## 4. Full-period performance versus QQQ.",
            "",
            _markdown_table(
                best,
                [
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "spy_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "qqq_buy_hold_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "return_to_drawdown",
                    "qqq_buy_hold_return_to_drawdown",
                ],
            ),
            "",
            "## 5. Equity and drawdown charts.",
            "",
            f"![Equity curve]({EQUITY_CHART_NAME})",
            "",
            f"![Drawdown curve]({DRAWDOWN_CHART_NAME})",
            "",
            "## 6. Rolling-window behavior.",
            "",
            _markdown_table(
                _summarize_rolling_all_windows(best_rolling),
                [
                    "rolling_window_count",
                    "positive_vs_qqq_window_rate_roll",
                    "qqq_underperformance_rate_roll",
                    "worst_strategy_vs_qqq_roll",
                    "best_strategy_vs_qqq",
                    "worst_window_label",
                    "best_window_label",
                    "drawdown_improvement_window_rate",
                ],
            ),
            "",
            f"![Rolling edge versus QQQ]({ROLLING_CHART_NAME})",
            "",
            "## 7. Subperiod behavior.",
            "",
            _markdown_table(
                best_subperiods,
                [
                    "period_name",
                    "strategy_total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "average_exposure",
                    "cash_days_pct",
                    "trade_count",
                ],
            ),
            "",
            f"![Subperiod returns]({SUBPERIOD_RETURNS_CHART_NAME})",
            "",
            f"![Subperiod drawdowns]({SUBPERIOD_DRAWDOWNS_CHART_NAME})",
            "",
            "## 8. QQQ/cash exposure path.",
            "",
            _markdown_table(
                best,
                [
                    "qqq_days",
                    "cash_days",
                    "average_exposure",
                    "cash_days_pct",
                    "trade_count",
                    "average_holding_days",
                    "longest_qqq_period_days",
                    "longest_cash_period_days",
                    "cash_segments_avoiding_major_qqq_drawdown",
                    "cash_segments_missing_major_qqq_rally",
                ],
            ),
            "",
            f"![Exposure timeline]({EXPOSURE_CHART_NAME})",
            "",
            "## 9. Cost sensitivity.",
            "",
            (
                "Costs are replayed from the existing exposure path and daily QQQ "
                "returns; this is not a retraining run."
            ),
            "",
            _markdown_table(
                best_costs,
                [
                    "transaction_cost_bps",
                    "total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "return_to_drawdown",
                    "decision_at_cost",
                ],
            ),
            "",
            "## 10. Candidate comparison against other QQQ timing candidates.",
            "",
            _markdown_table(
                summary,
                [
                    "candidate_rank",
                    "candidate_role",
                    "strategy_name",
                    "total_return",
                    "strategy_vs_qqq",
                    "drawdown_improvement_vs_qqq",
                    "return_to_drawdown",
                    "positive_vs_qqq_window_rate",
                    "decision",
                ],
            ),
            "",
            "## 11. Decision and recommendation.",
            "",
            _markdown_table(
                decision_matrix,
                [
                    "candidate_rank",
                    "candidate_role",
                    "strategy_name",
                    "cost_10bps_strategy_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "decision",
                    "production_ready",
                ],
            ),
            "",
            (
                "Recommendation: keep the best random-forest row for focused "
                "confirmation. Do not treat it as production-ready; the next step is "
                "a narrow confirmation study, not production export."
            ),
            "",
            "## 12. Caveats:",
            "",
            "- Research-only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ is the only tradable risky asset.",
            "- Other ETFs/reference series are features only.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- `production_ready` is always false.",
            "",
        ]
    )


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
    out = frame.loc[frame["strategy_id"].astype(str).eq(str(strategy_id))].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date", kind="mergesort")
    return out.reset_index(drop=True)


def _date_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.loc[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].copy()


def _period_metrics(frame: pd.DataFrame) -> dict[str, float]:
    strategy_equity = pd.to_numeric(frame["equity"], errors="coerce")
    qqq_equity = pd.to_numeric(frame["qqq_buy_hold_equity"], errors="coerce")
    strategy_total = _segment_total_return(strategy_equity)
    qqq_total = _segment_total_return(qqq_equity)
    strategy_dd = _segment_max_drawdown(strategy_equity)
    qqq_dd = _segment_max_drawdown(qqq_equity)
    return {
        "strategy_total_return": strategy_total,
        "qqq_total_return": qqq_total,
        "strategy_vs_qqq": strategy_total - qqq_total,
        "strategy_max_drawdown": strategy_dd,
        "qqq_max_drawdown": qqq_dd,
        "drawdown_improvement_vs_qqq": strategy_dd - qqq_dd,
    }


def _segment_total_return(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] == 0.0:
        return np.nan
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def _segment_max_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty or values.iloc[0] == 0.0:
        return np.nan
    rebased = values / values.iloc[0]
    return calculate_max_drawdown(rebased)


def _exposure_periods(exposure_log: pd.DataFrame, daily_equity: pd.DataFrame) -> list[dict[str, object]]:
    if exposure_log.empty:
        return []
    exposure = exposure_log.copy()
    exposure["date"] = pd.to_datetime(exposure["date"])
    exposure["bucket"] = [
        "cash" if value <= 1e-9 else "QQQ" if value >= 1.0 - 1e-9 else "QQQ,cash"
        for value in pd.to_numeric(exposure["effective_exposure"], errors="coerce").fillna(0.0)
    ]
    change = exposure["bucket"].ne(exposure["bucket"].shift()).cumsum()
    periods = []
    equity = daily_equity.copy()
    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"])
    for _, group in exposure.groupby(change, sort=False):
        start = pd.Timestamp(group["date"].iloc[0])
        end = pd.Timestamp(group["date"].iloc[-1])
        eq_slice = equity.loc[(equity["date"] >= start) & (equity["date"] <= end)]
        qqq_return = _segment_total_return(eq_slice["qqq_buy_hold_equity"]) if not eq_slice.empty else np.nan
        qqq_dd = _segment_max_drawdown(eq_slice["qqq_buy_hold_equity"]) if not eq_slice.empty else np.nan
        periods.append(
            {
                "exposure_bucket": str(group["bucket"].iloc[0]),
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "trading_days": int(len(group)),
                "qqq_segment_return": qqq_return,
                "qqq_segment_max_drawdown": qqq_dd,
            }
        )
    return periods


def _summarize_rolling_all_windows(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    if rolling_windows.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "rolling_window_count",
                "positive_vs_qqq_window_rate_roll",
                "qqq_underperformance_rate_roll",
                "worst_strategy_vs_qqq_roll",
                "best_strategy_vs_qqq",
                "worst_window_label",
                "best_window_label",
                "drawdown_improvement_window_rate",
            ]
        )
    rows = []
    for strategy_id, group in rolling_windows.groupby("strategy_id", sort=False):
        worst_idx = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce").idxmin()
        best_idx = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce").idxmax()
        rows.append(
            {
                "strategy_id": strategy_id,
                "rolling_window_count": int(len(group)),
                "positive_vs_qqq_window_rate_roll": float((group["strategy_vs_qqq"] > 0.0).mean()),
                "qqq_underperformance_rate_roll": float((group["strategy_vs_qqq"] < 0.0).mean()),
                "worst_strategy_vs_qqq_roll": float(group["strategy_vs_qqq"].min()),
                "best_strategy_vs_qqq": float(group["strategy_vs_qqq"].max()),
                "worst_window_label": str(group.loc[worst_idx, "window_label"]),
                "best_window_label": str(group.loc[best_idx, "window_label"]),
                "drawdown_improvement_window_rate": float(
                    (group["drawdown_improvement_vs_qqq"] > 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_strategy_id(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    best = summary.loc[summary["candidate_role"].astype(str).eq("best_candidate")]
    if best.empty:
        best = summary.head(1)
    return str(best["strategy_id"].iloc[0])


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown) or max_drawdown >= 0.0:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, list(columns)]


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_bool_dtype(table[column]):
            table[column] = table[column].map(lambda value: str(bool(value)).lower())
        elif pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    table = table.fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])
