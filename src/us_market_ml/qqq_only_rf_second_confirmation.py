"""Second confirmation study for the locked QQQ-only RF timing family.

This module is research-only and local-only. It narrows around the selected
threshold 0.45 random-forest QQQ/cash family and tests robustness across
random seeds, weekly rebalance weekdays, transaction costs, and compact RF
hyperparameter variants. Simulated holdings are always QQQ and cash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.qqq_only_best_candidate_deep_dive import (
    build_cost_sensitivity,
    build_rolling_windows,
    build_subperiods,
    confirm_held_assets_are_qqq_cash_only,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    EXPOSURE_LOG_COLUMNS,
    QQQOnlyRunSpec,
    _baseline_specs,
)
from us_market_ml.qqq_only_rf_signal_review import (
    add_recent_window_behavior,
    build_cash_timing_review,
    build_exposure_segments,
    build_missed_rallies,
)

OUTPUT_PREFIX = "qqq_only_rf_second_confirmation"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_second_confirmation")

RANDOM_STATES: tuple[int, ...] = (7, 11, 17, 23, 42, 101, 202, 777, 1009, 2026)
REBALANCE_WEEKDAYS: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")
TRANSACTION_COST_BPS_VALUES: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0)
HYPER_N_ESTIMATORS: tuple[int, ...] = (200, 500)
HYPER_MAX_DEPTHS: tuple[int | None, ...] = (None, 4, 6)
HYPER_MIN_SAMPLES_LEAF: tuple[int, ...] = (5, 20)

DEFAULT_RF_N_ESTIMATORS = 60
DEFAULT_RF_MAX_DEPTH: int | None = None
DEFAULT_RF_MIN_SAMPLES_LEAF = 5

BASE_RF_CONFIG: dict[str, object] = {
    "model_type": "random_forest",
    "training_mode": "rolling_5y",
    "target_name": "risk_on_20d",
    "target_horizon": 20,
    "feature_set_name": "qqq_self_cross_market",
    "exposure_mode": "binary_0_100",
    "rebalance_frequency": "weekly",
    "rebalance_weekday": "FRI",
    "random_state": 42,
    "n_estimators": DEFAULT_RF_N_ESTIMATORS,
    "max_depth": DEFAULT_RF_MAX_DEPTH,
    "min_samples_leaf": DEFAULT_RF_MIN_SAMPLES_LEAF,
    "transaction_cost_bps": 0.0,
    "threshold": 0.45,
    "min_holding_days": 0,
    "switch_margin": 0.0,
}

BASELINE_RULES: tuple[str, ...] = ("qqq_buy_hold", "qqq_sma200_timing")

SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("covid_crash_recovery", "2020-01-01", "2020-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_bull_2023_2024", "2023-01-01", "2024-12-31"),
    ("recent_2025_2026", "2025-01-01", "2026-05-15"),
)

RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "target_horizon",
    "feature_set_name",
    "feature_count",
    "exposure_mode",
    "rebalance_frequency",
    "rebalance_weekday",
    "random_state",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "transaction_cost_bps",
    "threshold",
    "min_holding_days",
    "switch_margin",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "daily_volatility",
    "sharpe_like",
    "return_to_drawdown",
    "qqq_buy_hold_total_return",
    "qqq_buy_hold_max_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "strategy_vs_qqq",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown_vs_qqq",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "best_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "average_holding_days",
    "estimated_transaction_cost",
    "cash_segment_count",
    "useful_risk_off_cash_segments",
    "missed_rally_cash_segments",
    "cash_timing_quality",
    "missed_rally_count",
    "missed_rally_penalty",
    "cost_10bps_strategy_vs_qqq",
    "cost_20bps_strategy_vs_qqq",
    "cost_30bps_strategy_vs_qqq",
    "beats_qqq_return",
    "improves_max_drawdown",
    "improves_return_to_drawdown",
    "positive_edge_after_10bps",
    "positive_edge_after_20bps",
    "second_confirmation_pass_raw",
    "strong_second_confirmation_raw",
    "decision_label",
    "recommendation",
    "production_ready",
)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
SEED_STABILITY_NAME = f"{OUTPUT_PREFIX}_seed_stability.csv"
REBALANCE_DAY_STABILITY_NAME = f"{OUTPUT_PREFIX}_rebalance_day_stability.csv"
COST_STABILITY_NAME = f"{OUTPUT_PREFIX}_cost_stability.csv"
PARAMETER_STABILITY_NAME = f"{OUTPUT_PREFIX}_parameter_stability.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = f"{OUTPUT_PREFIX}_equity_curve_top_candidates.png"
DRAWDOWN_CHART_NAME = f"{OUTPUT_PREFIX}_drawdown_curve_top_candidates.png"
SEED_CHART_NAME = f"{OUTPUT_PREFIX}_seed_stability.png"
REBALANCE_DAY_CHART_NAME = f"{OUTPUT_PREFIX}_rebalance_day_stability.png"
COST_CHART_NAME = f"{OUTPUT_PREFIX}_cost_stability.png"
PARAMETER_CHART_NAME = f"{OUTPUT_PREFIX}_parameter_stability.png"

REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Why second confirmation.",
    "## 3. Locked candidate family.",
    "## 4. Seed stability.",
    "## 5. Rebalance-day stability.",
    "## 6. Cost stability.",
    "## 7. Parameter stability.",
    "## 8. Full-period QQQ comparison.",
    "## 9. Rolling-window consistency.",
    "## 10. Drawdown behavior.",
    "## 11. Exposure behavior.",
    "## 12. Best confirmed rows.",
    "## 13. Paper-tracking readiness.",
    "## 14. Recommendation:",
    "## 15. Caveats:",
)


@dataclass(frozen=True)
class QQQOnlyRFSecondConfirmationArtifacts:
    """In-memory artifacts for the second confirmation report."""

    summary: pd.DataFrame
    seed_stability: pd.DataFrame
    rebalance_day_stability: pd.DataFrame
    cost_stability: pd.DataFrame
    parameter_stability: pd.DataFrame
    decision_matrix: pd.DataFrame
    rolling_windows: pd.DataFrame
    subperiods: pd.DataFrame
    cash_timing: pd.DataFrame
    missed_rallies: pd.DataFrame
    report_markdown: str


def build_qqq_only_rf_second_confirmation_specs() -> tuple[QQQOnlyRunSpec, ...]:
    """Build the narrow locked-candidate second-confirmation grid."""

    specs_by_key: dict[str, QQQOnlyRunSpec] = {}
    for spec in _baseline_specs((0.0,)):
        if spec.baseline_rule in BASELINE_RULES:
            specs_by_key[spec.run_key] = spec

    def add_spec(**overrides: object) -> None:
        config = {**BASE_RF_CONFIG, **overrides}
        spec = _rf_spec(config)
        specs_by_key[spec.run_key] = spec

    # Seed stability plus actual transaction-cost variants for the locked base.
    for random_state in RANDOM_STATES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(random_state=random_state, transaction_cost_bps=cost)

    # Rebalance weekday sensitivity for the base seed.
    for weekday in REBALANCE_WEEKDAYS:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(
                rebalance_weekday=weekday,
                random_state=42,
                transaction_cost_bps=cost,
            )

    # Locked references selected by the signal review.
    for cost in TRANSACTION_COST_BPS_VALUES:
        add_spec(transaction_cost_bps=cost, min_holding_days=10)
        add_spec(transaction_cost_bps=cost, exposure_mode="partial_50_100")
        add_spec(transaction_cost_bps=cost, threshold=0.50)

    # Compact RF hyperparameter stability slice. Cost 20 is included so the
    # strong-confirmation cost criterion is visible without a broad grid.
    for n_estimators in HYPER_N_ESTIMATORS:
        for max_depth in HYPER_MAX_DEPTHS:
            for min_samples_leaf in HYPER_MIN_SAMPLES_LEAF:
                for cost in (0.0, 10.0, 20.0):
                    add_spec(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        transaction_cost_bps=cost,
                    )

    return tuple(specs_by_key.values())


def add_second_confirmation_metrics_to_result(
    result: dict[str, object],
    *,
    spec: QQQOnlyRunSpec,
    daily_equity: pd.DataFrame,
) -> dict[str, object]:
    """Add second-confirmation metadata and rolling diagnostics to one result."""

    out = dict(result)
    out.update(
        {
            "rebalance_weekday": spec.rebalance_weekday,
            "random_state": spec.random_state,
            "n_estimators": spec.n_estimators,
            "max_depth": spec.max_depth,
            "min_samples_leaf": spec.min_samples_leaf,
            "production_ready": False,
        }
    )
    rolling_diff = _rolling_strategy_vs_qqq(daily_equity)
    if not rolling_diff.empty:
        out["best_strategy_vs_qqq"] = float(rolling_diff.max())
        out["worst_strategy_vs_qqq"] = float(rolling_diff.min())
        out["positive_vs_qqq_window_rate"] = float((rolling_diff > 0.0).mean())
        out["qqq_underperformance_rate"] = float((rolling_diff < 0.0).mean())
    else:
        out["best_strategy_vs_qqq"] = out.get("strategy_vs_qqq", np.nan)
    out["drawdown_improvement_window_rate"] = _rolling_drawdown_improvement_rate(daily_equity)
    return _ensure_result_dict(out)


def build_qqq_only_rf_second_confirmation_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
) -> QQQOnlyRFSecondConfirmationArtifacts:
    """Build all required second-confirmation tables, report text, and chart inputs."""

    confirm_held_assets_are_qqq_cash_only(exposure_log)
    summary = _ensure_columns(results.copy(), RESULT_COLUMNS)
    summary["production_ready"] = False

    rolling_windows = build_rolling_windows(daily_equity)
    rolling_summary = _summarize_rolling_windows(rolling_windows)
    cost_sensitivity = build_cost_sensitivity(
        daily_equity,
        cost_sensitivity_bps=TRANSACTION_COST_BPS_VALUES,
    )
    cost_edges = _cost_edge_columns(cost_sensitivity)
    subperiods = build_subperiods(daily_equity, exposure_log, subperiods=SUBPERIODS)
    exposure_segments = build_exposure_segments(exposure_log, daily_equity)
    cash_timing = build_cash_timing_review(exposure_segments)
    recent_rolling = add_recent_window_behavior(rolling_windows)
    missed_rallies = build_missed_rallies(exposure_segments, recent_rolling)
    cash_quality = _summarize_cash_timing(cash_timing, missed_rallies)

    summary = summary.merge(rolling_summary, on="strategy_id", how="left", suffixes=("", "_roll"))
    for column in (
        "rolling_window_count",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "best_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
    ):
        roll_column = f"{column}_roll"
        if roll_column in summary.columns:
            summary[column] = summary[roll_column].combine_first(summary[column])
    summary = _merge_replacing_columns(summary, cost_edges, on="strategy_id")
    summary = _merge_replacing_columns(summary, cash_quality, on="strategy_id")
    summary = _add_decision_inputs(summary)

    seed_stability = build_seed_stability(summary)
    rebalance_day_stability = build_rebalance_day_stability(summary)
    seed_stable_rate = _stability_rate(seed_stability)
    weekday_stable_count = _stable_count(rebalance_day_stability)
    summary = _apply_decision_labels(
        summary,
        seed_stable_rate=seed_stable_rate,
        weekday_stable_count=weekday_stable_count,
    )
    summary = _sort_summary(summary)
    summary["candidate_rank"] = np.arange(1, len(summary) + 1)

    seed_stability = build_seed_stability(summary)
    rebalance_day_stability = build_rebalance_day_stability(summary)
    cost_stability = build_cost_stability(cost_sensitivity, summary)
    parameter_stability = build_parameter_stability(summary)
    decision_matrix = _build_decision_matrix(summary)
    report = render_qqq_only_rf_second_confirmation_markdown(
        summary=summary,
        seed_stability=seed_stability,
        rebalance_day_stability=rebalance_day_stability,
        cost_stability=cost_stability,
        parameter_stability=parameter_stability,
        decision_matrix=decision_matrix,
        rolling_windows=rolling_windows,
        subperiods=subperiods,
    )

    return QQQOnlyRFSecondConfirmationArtifacts(
        summary=summary,
        seed_stability=seed_stability,
        rebalance_day_stability=rebalance_day_stability,
        cost_stability=cost_stability,
        parameter_stability=parameter_stability,
        decision_matrix=decision_matrix,
        rolling_windows=rolling_windows,
        subperiods=subperiods,
        cash_timing=cash_timing,
        missed_rallies=missed_rallies,
        report_markdown=report,
    )


def write_qqq_only_rf_second_confirmation_report_artifacts(
    *,
    artifacts: QQQOnlyRFSecondConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write required report CSVs, Markdown, and PNG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = report_output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.seed_stability.to_csv(paths[SEED_STABILITY_NAME], index=False)
    artifacts.rebalance_day_stability.to_csv(paths[REBALANCE_DAY_STABILITY_NAME], index=False)
    artifacts.cost_stability.to_csv(paths[COST_STABILITY_NAME], index=False)
    artifacts.parameter_stability.to_csv(paths[PARAMETER_STABILITY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_second_confirmation_charts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        paths=paths,
    )
    return paths


def report_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all required report output paths."""

    names = (
        SUMMARY_NAME,
        SEED_STABILITY_NAME,
        REBALANCE_DAY_STABILITY_NAME,
        COST_STABILITY_NAME,
        PARAMETER_STABILITY_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        SEED_CHART_NAME,
        REBALANCE_DAY_CHART_NAME,
        COST_CHART_NAME,
        PARAMETER_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def build_seed_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability of the locked base candidate across random seeds."""

    base = summary.loc[_base_locked_mask(summary) & _is_cost(summary, 0.0) & _is_weekday(summary, "FRI")].copy()
    rows: list[dict[str, object]] = []
    for seed, group in base.groupby("random_state", dropna=False, sort=True):
        row = group.iloc[0]
        stable = bool(row.get("second_confirmation_pass_raw", False))
        rows.append(
            {
                "random_state": int(_float(seed)) if np.isfinite(_float(seed)) else seed,
                "config_count": len(group),
                "stable_count": int(stable),
                "stable_rate": float(stable),
                "total_return": _float(row.get("total_return")),
                "strategy_vs_qqq": _float(row.get("strategy_vs_qqq")),
                "max_drawdown": _float(row.get("max_drawdown")),
                "drawdown_improvement_vs_qqq": _float(row.get("drawdown_improvement_vs_qqq")),
                "positive_vs_qqq_window_rate": _float(row.get("positive_vs_qqq_window_rate")),
                "worst_strategy_vs_qqq": _float(row.get("worst_strategy_vs_qqq")),
                "decision_label": row.get("decision_label", ""),
            }
        )
    return pd.DataFrame(rows)


def build_rebalance_day_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability of the locked base candidate across weekly weekdays."""

    base = summary.loc[_base_locked_mask(summary) & _is_cost(summary, 0.0) & _is_seed(summary, 42)].copy()
    rows: list[dict[str, object]] = []
    for weekday, group in base.groupby("rebalance_weekday", dropna=False, sort=True):
        row = group.iloc[0]
        stable = bool(row.get("second_confirmation_pass_raw", False))
        rows.append(
            {
                "rebalance_weekday": str(weekday),
                "config_count": len(group),
                "stable_count": int(stable),
                "stable_rate": float(stable),
                "total_return": _float(row.get("total_return")),
                "strategy_vs_qqq": _float(row.get("strategy_vs_qqq")),
                "max_drawdown": _float(row.get("max_drawdown")),
                "drawdown_improvement_vs_qqq": _float(row.get("drawdown_improvement_vs_qqq")),
                "positive_vs_qqq_window_rate": _float(row.get("positive_vs_qqq_window_rate")),
                "worst_strategy_vs_qqq": _float(row.get("worst_strategy_vs_qqq")),
                "decision_label": row.get("decision_label", ""),
            }
        )
    order = {weekday: index for index, weekday in enumerate(REBALANCE_WEEKDAYS)}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["rebalance_weekday"].map(order).fillna(99)
        out = out.sort_values("_order", kind="mergesort").drop(columns="_order")
    return out


def build_cost_stability(cost_sensitivity: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize replayed transaction-cost sensitivity by bps assumption."""

    if cost_sensitivity.empty:
        return pd.DataFrame()
    metadata = summary.loc[
        _base_locked_mask(summary),
        ["strategy_id", "random_state", "rebalance_weekday", "n_estimators", "max_depth", "min_samples_leaf"],
    ].drop_duplicates("strategy_id")
    merged = cost_sensitivity.merge(metadata, on="strategy_id", how="inner")
    rows: list[dict[str, object]] = []
    for bps, group in merged.groupby("transaction_cost_bps", sort=True):
        edge = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        rows.append(
            {
                "transaction_cost_bps": float(bps),
                "config_count": len(group),
                "positive_edge_count": int((edge > 0.0).sum()),
                "positive_edge_rate": float((edge > 0.0).mean()) if len(group) else np.nan,
                "median_strategy_vs_qqq": float(edge.median()) if edge.notna().any() else np.nan,
                "min_strategy_vs_qqq": float(edge.min()) if edge.notna().any() else np.nan,
                "max_strategy_vs_qqq": float(edge.max()) if edge.notna().any() else np.nan,
                "median_estimated_transaction_cost": _median(group["estimated_transaction_cost"]),
            }
        )
    return pd.DataFrame(rows)


def build_parameter_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability across locked candidate parameter values."""

    model_rows = summary.loc[summary["model_type"].astype(str).eq("random_forest")].copy()
    dimensions = (
        "threshold",
        "min_holding_days",
        "exposure_mode",
        "transaction_cost_bps",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
    )
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        if dimension not in model_rows.columns:
            continue
        values = model_rows[dimension].where(model_rows[dimension].notna(), "default")
        for value, group in model_rows.groupby(values, sort=True, dropna=False):
            labels = group["decision_label"].astype(str)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "config_count": len(group),
                    "second_confirmation_pass_count": int(labels.eq("second_confirmation_pass").sum()),
                    "strong_second_confirmation_count": int(labels.eq("strong_second_confirmation").sum()),
                    "paper_tracking_candidate_count": int(labels.eq("paper_tracking_candidate").sum()),
                    "reject_count": int(labels.eq("reject_deprioritize").sum()),
                    "median_strategy_vs_qqq": _median(group["strategy_vs_qqq"]),
                    "min_strategy_vs_qqq": _min(group["strategy_vs_qqq"]),
                    "max_strategy_vs_qqq": _max(group["strategy_vs_qqq"]),
                    "median_positive_vs_qqq_window_rate": _median(
                        group["positive_vs_qqq_window_rate"]
                    ),
                    "median_drawdown_improvement_vs_qqq": _median(
                        group["drawdown_improvement_vs_qqq"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_qqq_only_rf_second_confirmation_markdown(
    *,
    summary: pd.DataFrame,
    seed_stability: pd.DataFrame,
    rebalance_day_stability: pd.DataFrame,
    cost_stability: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    subperiods: pd.DataFrame,
) -> str:
    """Render the second-confirmation Markdown report."""

    paper_count = int(summary["decision_label"].astype(str).eq("paper_tracking_candidate").sum())
    strong_count = int(summary["decision_label"].astype(str).eq("strong_second_confirmation").sum())
    pass_count = int(summary["decision_label"].astype(str).eq("second_confirmation_pass").sum())
    defensive_count = int(summary["decision_label"].astype(str).eq("defensive_reference").sum())
    seed_rate = _stability_rate(seed_stability)
    weekday_stable = _stable_count(rebalance_day_stability)
    keep_threshold = paper_count + strong_count + pass_count > 0
    prepare_paper_tracking = paper_count > 0
    one_more_confirmation = keep_threshold
    promote_production = False
    best_rows = decision_matrix.loc[
        decision_matrix["decision_label"].isin(
            ["paper_tracking_candidate", "strong_second_confirmation", "second_confirmation_pass"]
        )
    ].head(12)
    top_ids = decision_matrix.head(8)["strategy_id"].astype(str).tolist()
    top_subperiods = subperiods.loc[subperiods["strategy_id"].astype(str).isin(top_ids)]
    rolling_table = _rolling_report_table(rolling_windows, top_ids)

    return "\n".join(
        [
            "# QQQ-Only RF Second Confirmation Report",
            "",
            "## 1. Executive summary.",
            "",
            f"- Completed rows reviewed: `{len(summary)}`",
            f"- Second-confirmation pass rows: `{pass_count}`",
            f"- Strong second-confirmation rows: `{strong_count}`",
            f"- Paper-tracking candidate rows: `{paper_count}`",
            f"- Defensive references: `{defensive_count}`",
            f"- Locked-base seed stable rate: `{seed_rate:.3f}`",
            f"- Locked-base rebalance weekdays stable: `{weekday_stable}` of `5`",
            "- Production-ready: `false`",
            "",
            "## 2. Why second confirmation.",
            "",
            (
                "The signal review selected threshold 0.45 as the strongest QQQ-only "
                "random-forest timing family. This study is a stricter robustness "
                "confirmation, not a broad model zoo or production promotion."
            ),
            "",
            "## 3. Locked candidate family.",
            "",
            _markdown_table(
                _locked_family_table(),
                ["field", "locked_value"],
            ),
            "",
            "## 4. Seed stability.",
            "",
            _markdown_table(
                seed_stability,
                [
                    "random_state",
                    "stable_count",
                    "strategy_vs_qqq",
                    "drawdown_improvement_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Seed stability]({SEED_CHART_NAME})",
            "",
            "## 5. Rebalance-day stability.",
            "",
            _markdown_table(
                rebalance_day_stability,
                [
                    "rebalance_weekday",
                    "stable_count",
                    "strategy_vs_qqq",
                    "drawdown_improvement_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Rebalance-day stability]({REBALANCE_DAY_CHART_NAME})",
            "",
            "## 6. Cost stability.",
            "",
            _markdown_table(
                cost_stability,
                [
                    "transaction_cost_bps",
                    "config_count",
                    "positive_edge_count",
                    "positive_edge_rate",
                    "median_strategy_vs_qqq",
                    "min_strategy_vs_qqq",
                    "median_estimated_transaction_cost",
                ],
            ),
            "",
            f"![Cost stability]({COST_CHART_NAME})",
            "",
            "## 7. Parameter stability.",
            "",
            _markdown_table(
                parameter_stability.sort_values(
                    [
                        "paper_tracking_candidate_count",
                        "strong_second_confirmation_count",
                        "second_confirmation_pass_count",
                        "median_strategy_vs_qqq",
                    ],
                    ascending=[False, False, False, False],
                    kind="mergesort",
                ).head(20),
                [
                    "dimension",
                    "value",
                    "config_count",
                    "second_confirmation_pass_count",
                    "strong_second_confirmation_count",
                    "paper_tracking_candidate_count",
                    "median_strategy_vs_qqq",
                    "median_positive_vs_qqq_window_rate",
                ],
            ),
            "",
            f"![Parameter stability]({PARAMETER_CHART_NAME})",
            "",
            "## 8. Full-period QQQ comparison.",
            "",
            _markdown_table(
                decision_matrix.head(14),
                [
                    "candidate_rank",
                    "strategy_name",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "qqq_buy_hold_max_drawdown",
                    "return_to_drawdown_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Equity curve]({EQUITY_CHART_NAME})",
            "",
            "## 9. Rolling-window consistency.",
            "",
            _markdown_table(
                rolling_table,
                [
                    "strategy_name",
                    "window_years",
                    "rolling_window_count",
                    "positive_vs_qqq_window_rate",
                    "worst_strategy_vs_qqq",
                    "best_strategy_vs_qqq",
                    "drawdown_improvement_window_rate",
                ],
            ),
            "",
            "## 10. Drawdown behavior.",
            "",
            _markdown_table(
                top_subperiods,
                [
                    "strategy_name",
                    "period_name",
                    "strategy_total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                ],
            ),
            "",
            f"![Drawdown curve]({DRAWDOWN_CHART_NAME})",
            "",
            "## 11. Exposure behavior.",
            "",
            _markdown_table(
                decision_matrix.head(14),
                [
                    "strategy_name",
                    "average_exposure",
                    "cash_days_pct",
                    "trade_count",
                    "average_holding_days",
                    "cash_timing_quality",
                    "missed_rally_penalty",
                ],
            ),
            "",
            "## 12. Best confirmed rows.",
            "",
            _markdown_table(
                best_rows,
                [
                    "candidate_rank",
                    "strategy_name",
                    "random_state",
                    "rebalance_weekday",
                    "transaction_cost_bps",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "strategy_vs_qqq",
                    "cost_20bps_strategy_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            "## 13. Paper-tracking readiness.",
            "",
            (
                f"Paper-tracking readiness is `{str(prepare_paper_tracking).lower()}`. "
                f"The locked base is stable across `{seed_rate:.1%}` of observed seed rows "
                f"and `{weekday_stable}` of `5` weekday rows."
            ),
            "",
            "## 14. Recommendation:",
            "",
            f"- Whether to keep threshold 0.45: `{str(keep_threshold).lower()}`",
            f"- Whether to prepare research paper-tracking: `{str(prepare_paper_tracking).lower()}`",
            f"- Whether to run one more out-of-time confirmation: `{str(one_more_confirmation).lower()}`",
            f"- Whether to promote to production: `{str(promote_production).lower()}`",
            "",
            "## 15. Caveats:",
            "",
            "- Research-only.",
            "- QQQ/cash only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Other ETFs/reference series are features only.",
            "- `production_ready` is always false.",
            "",
        ]
    )


def _rf_spec(config: dict[str, object]) -> QQQOnlyRunSpec:
    payload = {
        "study": OUTPUT_PREFIX,
        "model_type": str(config["model_type"]),
        "training_mode": str(config["training_mode"]),
        "target_name": str(config["target_name"]),
        "feature_set_name": str(config["feature_set_name"]),
        "exposure_mode": str(config["exposure_mode"]),
        "rebalance_frequency": str(config["rebalance_frequency"]),
        "rebalance_weekday": str(config["rebalance_weekday"]).upper(),
        "random_state": int(config["random_state"]),
        "n_estimators": int(config["n_estimators"]),
        "max_depth": _max_depth_payload(config["max_depth"]),
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "transaction_cost_bps": float(config["transaction_cost_bps"]),
        "threshold": float(config["threshold"]),
        "min_holding_days": int(config["min_holding_days"]),
        "switch_margin": float(config["switch_margin"]),
    }
    run_key = _run_key(payload)
    depth_label = _max_depth_label(config["max_depth"])
    strategy_name = (
        "random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_"
        f"{config['exposure_mode']}_weekly_{payload['rebalance_weekday'].lower()}_"
        f"seed{payload['random_state']}_trees{payload['n_estimators']}_"
        f"depth{depth_label}_leaf{payload['min_samples_leaf']}_"
        f"thr{float(config['threshold']):.2f}_hold{payload['min_holding_days']}_"
        f"cost{float(config['transaction_cost_bps']):g}"
    )
    return QQQOnlyRunSpec(
        run_key=run_key,
        strategy_id=run_key,
        strategy_name=strategy_name,
        model_type=str(config["model_type"]),
        training_mode=str(config["training_mode"]),
        target_name=str(config["target_name"]),
        target_horizon=int(config["target_horizon"]),
        feature_set_name=str(config["feature_set_name"]),
        exposure_mode=str(config["exposure_mode"]),
        rebalance_frequency=str(config["rebalance_frequency"]),
        transaction_cost_bps=float(config["transaction_cost_bps"]),
        threshold=float(config["threshold"]),
        min_holding_days=int(config["min_holding_days"]),
        switch_margin=float(config["switch_margin"]),
        rebalance_weekday=payload["rebalance_weekday"],
        random_state=payload["random_state"],
        n_estimators=payload["n_estimators"],
        max_depth=None if config["max_depth"] is None or pd.isna(config["max_depth"]) else int(config["max_depth"]),
        min_samples_leaf=payload["min_samples_leaf"],
    )


def _run_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    parts = [
        "rf2",
        f"seed{payload['random_state']}",
        str(payload["rebalance_weekday"]).lower(),
        f"trees{payload['n_estimators']}",
        f"depth{payload['max_depth']}",
        f"leaf{payload['min_samples_leaf']}",
        f"thr{str(payload['threshold']).replace('.', 'p')}",
        f"hold{payload['min_holding_days']}",
        f"cost{str(payload['transaction_cost_bps']).replace('.', 'p')}",
    ]
    return "__".join(parts) + f"__{digest}"


def _ensure_result_dict(result: dict[str, object]) -> dict[str, object]:
    out = dict(result)
    for column in RESULT_COLUMNS:
        out.setdefault(column, np.nan)
    return {column: out[column] for column in RESULT_COLUMNS}


def _add_decision_inputs(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for column in (
        "cost_10bps_strategy_vs_qqq",
        "cost_20bps_strategy_vs_qqq",
        "cost_30bps_strategy_vs_qqq",
        "cash_segment_count",
        "useful_risk_off_cash_segments",
        "missed_rally_cash_segments",
        "cash_timing_quality",
        "missed_rally_count",
        "missed_rally_penalty",
    ):
        if column not in out.columns:
            out[column] = np.nan
    out["beats_qqq_return"] = pd.to_numeric(out["strategy_vs_qqq"], errors="coerce") > 0.0
    out["improves_max_drawdown"] = (
        pd.to_numeric(out["drawdown_improvement_vs_qqq"], errors="coerce") > 0.0
    )
    out["improves_return_to_drawdown"] = (
        pd.to_numeric(out["return_to_drawdown_vs_qqq"], errors="coerce") > 0.0
    )
    out["positive_edge_after_10bps"] = (
        pd.to_numeric(out["cost_10bps_strategy_vs_qqq"], errors="coerce") > 0.0
    )
    out["positive_edge_after_20bps"] = (
        pd.to_numeric(out["cost_20bps_strategy_vs_qqq"], errors="coerce") > 0.0
    )
    out["second_confirmation_pass_raw"] = (
        out["beats_qqq_return"]
        & out["improves_max_drawdown"]
        & out["improves_return_to_drawdown"]
        & (pd.to_numeric(out["positive_vs_qqq_window_rate"], errors="coerce") >= 0.65)
        & (pd.to_numeric(out["worst_strategy_vs_qqq"], errors="coerce") > -0.20)
        & out["positive_edge_after_10bps"]
    )
    out["strong_second_confirmation_raw"] = (
        out["second_confirmation_pass_raw"]
        & (pd.to_numeric(out["positive_vs_qqq_window_rate"], errors="coerce") >= 0.70)
        & (pd.to_numeric(out["drawdown_improvement_window_rate"], errors="coerce") >= 0.70)
        & out["positive_edge_after_20bps"]
    )
    return out


def _apply_decision_labels(
    summary: pd.DataFrame,
    *,
    seed_stable_rate: float,
    weekday_stable_count: int,
) -> pd.DataFrame:
    out = summary.copy()
    seed_ok = seed_stable_rate >= 0.60
    weekday_ok = weekday_stable_count >= 3
    labels = []
    recommendations = []
    for row in out.to_dict("records"):
        label = _decision_label(row, seed_ok=seed_ok, weekday_ok=weekday_ok)
        labels.append(label)
        recommendations.append(_recommendation(label))
    out["decision_label"] = labels
    out["recommendation"] = recommendations
    out["production_ready"] = False
    return out


def _decision_label(row: dict[str, object], *, seed_ok: bool, weekday_ok: bool) -> str:
    if str(row.get("strategy_name")) == "qqq_buy_hold":
        return "benchmark"
    strong = bool(row.get("strong_second_confirmation_raw", False))
    passed = bool(row.get("second_confirmation_pass_raw", False))
    defensive = (
        _float(row.get("strategy_vs_qqq")) <= 0.0
        and _float(row.get("drawdown_improvement_vs_qqq")) >= 0.05
    )
    if strong and seed_ok and weekday_ok:
        return "paper_tracking_candidate"
    if strong:
        return "strong_second_confirmation"
    if passed:
        return "second_confirmation_pass"
    if defensive:
        return "defensive_reference"
    return "reject_deprioritize"


def _recommendation(label: str) -> str:
    if label == "paper_tracking_candidate":
        return "Prepare research paper-tracking candidate; do not promote to production."
    if label == "strong_second_confirmation":
        return "Keep threshold 0.45 and require one more out-of-time confirmation."
    if label == "second_confirmation_pass":
        return "Keep as a confirmed research row; continue robustness review."
    if label == "defensive_reference":
        return "Use only as defensive context against QQQ buy-and-hold."
    if label == "benchmark":
        return "Primary benchmark only."
    return "Reject or deprioritize within this second confirmation."


def _build_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "candidate_rank",
        "strategy_id",
        "strategy_name",
        "model_type",
        "training_mode",
        "target_name",
        "feature_set_name",
        "exposure_mode",
        "rebalance_frequency",
        "rebalance_weekday",
        "random_state",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
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
        "qqq_buy_hold_return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "positive_vs_qqq_window_rate",
        "worst_strategy_vs_qqq",
        "best_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
        "average_exposure",
        "cash_days_pct",
        "trade_count",
        "average_holding_days",
        "estimated_transaction_cost",
        "cash_timing_quality",
        "missed_rally_penalty",
        "cost_10bps_strategy_vs_qqq",
        "cost_20bps_strategy_vs_qqq",
        "beats_qqq_return",
        "improves_max_drawdown",
        "improves_return_to_drawdown",
        "positive_edge_after_10bps",
        "positive_edge_after_20bps",
        "decision_label",
        "recommendation",
        "production_ready",
    )
    return _ensure_columns(summary, columns).loc[:, list(columns)]


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rank = {
        "paper_tracking_candidate": 0,
        "strong_second_confirmation": 1,
        "second_confirmation_pass": 2,
        "defensive_reference": 3,
        "benchmark": 4,
        "reject_deprioritize": 5,
    }
    out = summary.copy()
    out["_rank"] = out["decision_label"].map(rank).fillna(9)
    out = out.sort_values(
        [
            "_rank",
            "strategy_vs_qqq",
            "cost_20bps_strategy_vs_qqq",
            "return_to_drawdown_vs_qqq",
            "drawdown_improvement_vs_qqq",
        ],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    return out.drop(columns=["_rank"]).reset_index(drop=True)


def _summarize_rolling_windows(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    if rolling_windows.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "rolling_window_count",
                "positive_vs_qqq_window_rate",
                "qqq_underperformance_rate",
                "worst_strategy_vs_qqq",
                "best_strategy_vs_qqq",
                "drawdown_improvement_window_rate",
            ]
        )
    rows: list[dict[str, object]] = []
    for strategy_id, group in rolling_windows.groupby("strategy_id", sort=False):
        strategy_vs = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        dd_improvement = pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
        rows.append(
            {
                "strategy_id": strategy_id,
                "rolling_window_count": int(strategy_vs.notna().sum()),
                "positive_vs_qqq_window_rate": float((strategy_vs > 0.0).mean()),
                "qqq_underperformance_rate": float((strategy_vs < 0.0).mean()),
                "worst_strategy_vs_qqq": float(strategy_vs.min()),
                "best_strategy_vs_qqq": float(strategy_vs.max()),
                "drawdown_improvement_window_rate": float((dd_improvement > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _cost_edge_columns(cost_sensitivity: pd.DataFrame) -> pd.DataFrame:
    if cost_sensitivity.empty:
        return pd.DataFrame(columns=["strategy_id"])
    pieces = []
    for bps in (10.0, 20.0, 30.0):
        subset = cost_sensitivity.loc[
            np.isclose(pd.to_numeric(cost_sensitivity["transaction_cost_bps"], errors="coerce"), bps),
            ["strategy_id", "strategy_vs_qqq"],
        ].copy()
        pieces.append(
            subset.rename(columns={"strategy_vs_qqq": f"cost_{int(bps)}bps_strategy_vs_qqq"})
        )
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="strategy_id", how="outer")
    return out.drop_duplicates("strategy_id")


def _merge_replacing_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
) -> pd.DataFrame:
    if right.empty:
        return left
    out = left.copy()
    replace_columns = [column for column in right.columns if column != on and column in out.columns]
    if replace_columns:
        out = out.drop(columns=replace_columns)
    return out.merge(right, on=on, how="left")


def _summarize_cash_timing(
    cash_timing: pd.DataFrame,
    missed_rallies: pd.DataFrame,
) -> pd.DataFrame:
    ids = set(cash_timing.get("strategy_id", pd.Series(dtype=str)).astype(str))
    ids.update(missed_rallies.get("strategy_id", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, object]] = []
    for strategy_id in sorted(ids):
        cash = cash_timing.loc[cash_timing.get("strategy_id", pd.Series(dtype=str)).astype(str).eq(strategy_id)]
        missed = missed_rallies.loc[
            missed_rallies.get("strategy_id", pd.Series(dtype=str)).astype(str).eq(strategy_id)
        ]
        useful_count = int(cash["cash_timing_label"].astype(str).eq("useful_risk_off").sum()) if not cash.empty else 0
        missed_count = int(cash["cash_timing_label"].astype(str).eq("missed_rally").sum()) if not cash.empty else 0
        cash_count = len(cash)
        rows.append(
            {
                "strategy_id": strategy_id,
                "cash_segment_count": cash_count,
                "useful_risk_off_cash_segments": useful_count,
                "missed_rally_cash_segments": missed_count,
                "cash_timing_quality": (useful_count - missed_count) / cash_count
                if cash_count
                else 0.0,
                "missed_rally_count": len(missed),
                "missed_rally_penalty": _max(missed["missed_return"]) if not missed.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _rolling_strategy_vs_qqq(daily_equity: pd.DataFrame, *, window: int = 63) -> pd.Series:
    strategy = pd.to_numeric(daily_equity["equity"], errors="coerce").pct_change(window)
    qqq = pd.to_numeric(daily_equity["qqq_buy_hold_equity"], errors="coerce").pct_change(window)
    return (strategy - qqq).dropna()


def _rolling_drawdown_improvement_rate(
    daily_equity: pd.DataFrame,
    *,
    window: int = 63,
) -> float:
    if len(daily_equity) <= window:
        return np.nan
    strategy = pd.to_numeric(daily_equity["equity"], errors="coerce").reset_index(drop=True)
    qqq = pd.to_numeric(daily_equity["qqq_buy_hold_equity"], errors="coerce").reset_index(drop=True)
    improved: list[bool] = []
    for end in range(window, len(strategy) + 1):
        strategy_window = strategy.iloc[end - window : end]
        qqq_window = qqq.iloc[end - window : end]
        if strategy_window.notna().sum() < window or qqq_window.notna().sum() < window:
            continue
        strategy_dd = calculate_max_drawdown(strategy_window / strategy_window.iloc[0])
        qqq_dd = calculate_max_drawdown(qqq_window / qqq_window.iloc[0])
        improved.append(bool(strategy_dd > qqq_dd))
    return float(np.mean(improved)) if improved else np.nan


def _write_second_confirmation_charts(
    *,
    artifacts: QQQOnlyRFSecondConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_ids = artifacts.decision_matrix.head(6)["strategy_id"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in top_ids:
        subset = _strategy_frame(daily_equity, strategy_id)
        if subset.empty:
            continue
        ax.plot(pd.to_datetime(subset["date"]), subset["equity"], label=subset["strategy_name"].iloc[0])
    if not daily_equity.empty:
        first = daily_equity.groupby("date", sort=True).head(1)
        ax.plot(
            pd.to_datetime(first["date"]),
            first["qqq_buy_hold_equity"],
            color="black",
            linewidth=2,
            label="QQQ buy-and-hold",
        )
    ax.set_title("QQQ-only RF second confirmation: top candidate equity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[EQUITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in top_ids:
        subset = _strategy_frame(daily_drawdown, strategy_id)
        if subset.empty:
            continue
        ax.plot(pd.to_datetime(subset["date"]), subset["drawdown"], label=subset["strategy_name"].iloc[0])
    if not daily_drawdown.empty:
        first = daily_drawdown.groupby("date", sort=True).head(1)
        ax.plot(
            pd.to_datetime(first["date"]),
            first["qqq_buy_hold_drawdown"],
            color="black",
            linewidth=2,
            label="QQQ buy-and-hold",
        )
    ax.set_title("QQQ-only RF second confirmation: top candidate drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    seed = artifacts.seed_stability.copy()
    if seed.empty:
        ax.text(0.5, 0.5, "No seed rows", ha="center", va="center")
    else:
        ax.bar(seed["random_state"].astype(str), seed["strategy_vs_qqq"], color="#457b9d")
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Locked base: random seed stability")
    ax.set_ylabel("Strategy vs QQQ")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[SEED_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    days = artifacts.rebalance_day_stability.copy()
    if days.empty:
        ax.text(0.5, 0.5, "No rebalance weekday rows", ha="center", va="center")
    else:
        ax.bar(days["rebalance_weekday"].astype(str), days["strategy_vs_qqq"], color="#2a9d8f")
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Locked base: weekly rebalance-day stability")
    ax.set_ylabel("Strategy vs QQQ")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[REBALANCE_DAY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cost = artifacts.cost_stability.copy()
    if cost.empty:
        ax.text(0.5, 0.5, "No cost rows", ha="center", va="center")
    else:
        ax.plot(
            cost["transaction_cost_bps"],
            cost["median_strategy_vs_qqq"],
            marker="o",
            label="median edge",
        )
        ax.plot(
            cost["transaction_cost_bps"],
            cost["min_strategy_vs_qqq"],
            marker="o",
            label="min edge",
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.legend(fontsize=8)
    ax.set_title("Locked family: transaction-cost stability")
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Strategy vs QQQ")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[COST_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    stability = artifacts.parameter_stability.copy()
    if stability.empty:
        ax.text(0.5, 0.5, "No parameter stability rows", ha="center", va="center")
    else:
        plot_rows = stability.sort_values(
            [
                "paper_tracking_candidate_count",
                "strong_second_confirmation_count",
                "second_confirmation_pass_count",
                "median_strategy_vs_qqq",
            ],
            ascending=[False, False, False, False],
            kind="mergesort",
        ).head(18)
        labels = plot_rows["dimension"].astype(str) + "=" + plot_rows["value"].astype(str)
        ax.barh(labels, plot_rows["median_strategy_vs_qqq"], color="#6a994e")
        ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("QQQ-only RF second confirmation: parameter stability")
    ax.set_xlabel("Median strategy vs QQQ")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[PARAMETER_CHART_NAME], dpi=140)
    plt.close(fig)


def _locked_family_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("model_type", "random_forest"),
            ("training_mode", "rolling_5y"),
            ("target_name", "risk_on_20d"),
            ("feature_set_name", "qqq_self_cross_market"),
            ("exposure_mode", "binary_0_100 base, partial_50_100 locked reference"),
            ("rebalance_frequency", "weekly"),
            ("threshold", "0.45 base, 0.50 prior-best reference"),
            ("tradable_holdings", "QQQ, cash"),
            ("primary_benchmark", "QQQ buy-and-hold"),
        ],
        columns=["field", "locked_value"],
    )


def _rolling_report_table(rolling_windows: pd.DataFrame, strategy_ids: Sequence[str]) -> pd.DataFrame:
    if rolling_windows.empty:
        return pd.DataFrame()
    rows = []
    subset = rolling_windows.loc[rolling_windows["strategy_id"].astype(str).isin(strategy_ids)]
    for (strategy_id, years), group in subset.groupby(["strategy_id", "window_years"], sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": first.get("strategy_name"),
                "window_years": years,
                "rolling_window_count": len(group),
                "positive_vs_qqq_window_rate": float(
                    (pd.to_numeric(group["strategy_vs_qqq"], errors="coerce") > 0.0).mean()
                ),
                "worst_strategy_vs_qqq": _min(group["strategy_vs_qqq"]),
                "best_strategy_vs_qqq": _max(group["strategy_vs_qqq"]),
                "drawdown_improvement_window_rate": float(
                    (
                        pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
                        > 0.0
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _base_locked_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return (
        frame["model_type"].astype(str).eq("random_forest")
        & frame["training_mode"].astype(str).eq("rolling_5y")
        & frame["target_name"].astype(str).eq("risk_on_20d")
        & frame["feature_set_name"].astype(str).eq("qqq_self_cross_market")
        & frame["exposure_mode"].astype(str).eq("binary_0_100")
        & frame["rebalance_frequency"].astype(str).eq("weekly")
        & np.isclose(pd.to_numeric(frame["threshold"], errors="coerce"), 0.45)
        & pd.to_numeric(frame["min_holding_days"], errors="coerce").fillna(-1).astype(int).eq(0)
        & np.isclose(pd.to_numeric(frame["switch_margin"], errors="coerce").fillna(0.0), 0.0)
        & pd.to_numeric(frame["n_estimators"], errors="coerce").fillna(DEFAULT_RF_N_ESTIMATORS).eq(
            DEFAULT_RF_N_ESTIMATORS
        )
        & _max_depth_is_default(frame["max_depth"])
        & pd.to_numeric(frame["min_samples_leaf"], errors="coerce")
        .fillna(DEFAULT_RF_MIN_SAMPLES_LEAF)
        .eq(DEFAULT_RF_MIN_SAMPLES_LEAF)
    )


def _is_cost(frame: pd.DataFrame, cost: float) -> pd.Series:
    return np.isclose(pd.to_numeric(frame["transaction_cost_bps"], errors="coerce"), cost)


def _is_seed(frame: pd.DataFrame, seed: int) -> pd.Series:
    return pd.to_numeric(frame["random_state"], errors="coerce").eq(seed)


def _is_weekday(frame: pd.DataFrame, weekday: str) -> pd.Series:
    return frame["rebalance_weekday"].astype(str).str.upper().eq(weekday)


def _max_depth_is_default(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    return values.isna() | text.isin({"", "nan", "none", "default", "<na>"})


def _max_depth_payload(value: object) -> str | int:
    if value is None or pd.isna(value):
        return "default"
    return int(value)


def _max_depth_label(value: object) -> str:
    return "default" if value is None or pd.isna(value) else str(int(value))


def _stability_rate(stability: pd.DataFrame) -> float:
    if stability.empty or "stable_count" not in stability.columns:
        return 0.0
    stable = pd.to_numeric(stability["stable_count"], errors="coerce")
    return float((stable > 0).mean()) if len(stability) else 0.0


def _stable_count(stability: pd.DataFrame) -> int:
    if stability.empty or "stable_count" not in stability.columns:
        return 0
    return int((pd.to_numeric(stability["stable_count"], errors="coerce") > 0).sum())


def _strategy_frame(frame: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    if frame.empty or "strategy_id" not in frame.columns:
        return pd.DataFrame()
    out = frame.loc[frame["strategy_id"].astype(str).eq(str(strategy_id))].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    return out


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
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.median()) if numeric.notna().any() else np.nan


def _min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min()) if numeric.notna().any() else np.nan


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else np.nan
