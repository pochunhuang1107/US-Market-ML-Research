"""Focused random-forest confirmation study for QQQ-only timing.

This module is research-only and local-only. It confirms a narrow random
forest QQQ/cash timing family around the best QQQ-only model-zoo candidate.
Other ETFs and reference series can be features, but simulated holdings are
always QQQ and cash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.qqq_only_best_candidate_deep_dive import (
    build_cost_sensitivity,
    build_rolling_windows,
    confirm_held_assets_are_qqq_cash_only,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    CLASSIFICATION_TARGETS,
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    EXPOSURE_LOG_COLUMNS,
    QQQOnlyRunSpec,
    _baseline_specs,
    _model_spec,
)

OUTPUT_PREFIX = "qqq_only_rf_confirmation"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_confirmation")

BEST_STRATEGY_NAME = (
    "random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_"
    "binary_0_100_weekly_cost0"
)
CURRENT_BEST_POSITIVE_VS_QQQ_WINDOW_RATE = 0.6140
CURRENT_BEST_DRAWDOWN_IMPROVEMENT_WINDOW_RATE = 0.7544
CURRENT_BEST_COST_10BPS_STRATEGY_VS_QQQ = 0.6074

TRAINING_MODES: tuple[str, ...] = (
    "rolling_3y",
    "rolling_5y",
    "recency_weighted_expanding",
)
TARGET_NAMES: tuple[str, ...] = (
    "risk_on_20d",
    "qqq_forward_return_20d",
    "qqq_forward_return_63d",
)
FEATURE_SET_NAMES: tuple[str, ...] = (
    "qqq_self_cross_market",
    "qqq_self_cross_market_reference",
)
EXPOSURE_MODES: tuple[str, ...] = (
    "binary_0_100",
    "partial_50_100",
    "ternary_0_50_100",
)
REBALANCE_FREQUENCIES: tuple[str, ...] = ("weekly", "monthly")
CLASSIFICATION_THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
REGRESSION_THRESHOLDS: tuple[float, ...] = (0.0, 0.002, 0.005)
MIN_HOLDING_DAYS_VALUES: tuple[int, ...] = (0, 5, 10, 20)
SWITCH_MARGIN_VALUES: tuple[float, ...] = (0.0, 0.002, 0.005)
TRANSACTION_COST_BPS_VALUES: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
BASELINE_RULES: tuple[str, ...] = ("qqq_buy_hold", "qqq_sma200_timing")

BASE_RF_CONFIG: dict[str, object] = {
    "model_type": "random_forest",
    "training_mode": "rolling_5y",
    "target_name": "risk_on_20d",
    "feature_set_name": "qqq_self_cross_market",
    "exposure_mode": "binary_0_100",
    "rebalance_frequency": "weekly",
    "transaction_cost_bps": 0.0,
    "threshold": 0.50,
    "min_holding_days": 0,
    "switch_margin": 0.0,
}

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
    "production_ready",
    "decision_label",
)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = f"{OUTPUT_PREFIX}_equity_curve_top_candidates.png"
DRAWDOWN_CHART_NAME = f"{OUTPUT_PREFIX}_drawdown_curve_top_candidates.png"
EXPOSURE_CHART_NAME = f"{OUTPUT_PREFIX}_exposure_timeline_top_candidates.png"
ROLLING_CHART_NAME = f"{OUTPUT_PREFIX}_rolling_vs_qqq_top_candidates.png"
COST_SENSITIVITY_CHART_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.png"
PARAMETER_STABILITY_CHART_NAME = f"{OUTPUT_PREFIX}_parameter_stability.png"

REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Why focused confirmation.",
    "## 3. Baseline candidate recap.",
    "## 4. Parameter grid.",
    "## 5. Full-period QQQ comparison.",
    "## 6. Cost sensitivity.",
    "## 7. Rolling-window consistency.",
    "## 8. Drawdown behavior.",
    "## 9. Exposure behavior.",
    "## 10. Parameter stability.",
    "## 11. Best confirmed candidates.",
    "## 12. What failed.",
    "## 13. Recommendation:",
    "## 14. Caveats:",
)


@dataclass(frozen=True)
class QQQOnlyRFConfirmationArtifacts:
    """In-memory report artifacts for the focused confirmation study."""

    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    parameter_stability: pd.DataFrame
    report_markdown: str


def build_qqq_only_rf_confirmation_specs() -> tuple[QQQOnlyRunSpec, ...]:
    """Build a narrow, deterministic confirmation grid around the RF candidate."""

    specs_by_key: dict[str, QQQOnlyRunSpec] = {}
    for spec in _baseline_specs((0.0,)):
        if spec.baseline_rule in BASELINE_RULES:
            specs_by_key[spec.run_key] = spec

    def add_spec(**overrides: object) -> None:
        config = {**BASE_RF_CONFIG, **overrides}
        spec = _model_spec(
            model_type=str(config["model_type"]),
            training_mode=str(config["training_mode"]),
            target_name=str(config["target_name"]),
            feature_set_name=str(config["feature_set_name"]),
            exposure_mode=str(config["exposure_mode"]),
            rebalance_frequency=str(config["rebalance_frequency"]),
            transaction_cost_bps=float(config["transaction_cost_bps"]),
            threshold=float(config["threshold"]),
            min_holding_days=int(config["min_holding_days"]),
            switch_margin=float(config["switch_margin"]),
        )
        specs_by_key[spec.run_key] = spec

    for cost in TRANSACTION_COST_BPS_VALUES:
        add_spec(transaction_cost_bps=cost)

    for training_mode in TRAINING_MODES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(training_mode=training_mode, transaction_cost_bps=cost)

    for target_name in TARGET_NAMES:
        for threshold in _thresholds_for_target(target_name):
            for cost in TRANSACTION_COST_BPS_VALUES:
                add_spec(target_name=target_name, threshold=threshold, transaction_cost_bps=cost)

    for feature_set_name in FEATURE_SET_NAMES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(feature_set_name=feature_set_name, transaction_cost_bps=cost)

    for exposure_mode in EXPOSURE_MODES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(exposure_mode=exposure_mode, transaction_cost_bps=cost)

    for rebalance_frequency in REBALANCE_FREQUENCIES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(rebalance_frequency=rebalance_frequency, transaction_cost_bps=cost)

    for min_holding_days in MIN_HOLDING_DAYS_VALUES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(min_holding_days=min_holding_days, transaction_cost_bps=cost)

    for switch_margin in SWITCH_MARGIN_VALUES:
        for cost in TRANSACTION_COST_BPS_VALUES:
            add_spec(switch_margin=switch_margin, transaction_cost_bps=cost)

    # Two local interaction slices around the best row. They reuse the base
    # prediction path and test whether the edge is just a threshold artifact.
    for threshold in CLASSIFICATION_THRESHOLDS:
        for min_holding_days in MIN_HOLDING_DAYS_VALUES:
            for cost in (0.0, 10.0):
                add_spec(
                    threshold=threshold,
                    min_holding_days=min_holding_days,
                    transaction_cost_bps=cost,
                )
    for exposure_mode in EXPOSURE_MODES:
        for threshold in CLASSIFICATION_THRESHOLDS:
            for cost in (0.0, 10.0):
                add_spec(
                    exposure_mode=exposure_mode,
                    threshold=threshold,
                    transaction_cost_bps=cost,
                )

    return tuple(specs_by_key.values())


def add_confirmation_metrics_to_result(
    result: dict[str, object],
    *,
    daily_equity: pd.DataFrame,
) -> dict[str, object]:
    """Add confirmation-specific metrics missing from the shared zoo runner."""

    out = dict(result)
    rolling_diff = _rolling_strategy_vs_qqq(daily_equity)
    if not rolling_diff.empty:
        out["best_strategy_vs_qqq"] = float(rolling_diff.max())
        out["worst_strategy_vs_qqq"] = float(rolling_diff.min())
        out["positive_vs_qqq_window_rate"] = float((rolling_diff > 0.0).mean())
        out["qqq_underperformance_rate"] = float((rolling_diff < 0.0).mean())
    else:
        out["best_strategy_vs_qqq"] = out.get("strategy_vs_qqq", np.nan)
    out["drawdown_improvement_window_rate"] = _rolling_drawdown_improvement_rate(daily_equity)
    out["production_ready"] = False
    return _ensure_result_columns(out)


def build_qqq_only_rf_confirmation_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    prior_decision_matrix: pd.DataFrame | None = None,
) -> QQQOnlyRFConfirmationArtifacts:
    """Build the focused confirmation report tables and Markdown."""

    confirm_held_assets_are_qqq_cash_only(exposure_log)
    summary = _ensure_columns(results.copy(), RESULT_COLUMNS)
    summary["production_ready"] = False
    summary["is_prior_best_candidate"] = summary.apply(_is_prior_best_candidate, axis=1)
    rolling_windows = build_rolling_windows(daily_equity)
    rolling_summary = _summarize_rolling_windows(rolling_windows)
    cost_sensitivity = build_cost_sensitivity(daily_equity)
    cost_10 = _cost_at_bps(cost_sensitivity, 10.0)

    summary = summary.merge(rolling_summary, on="strategy_id", how="left", suffixes=("", "_roll"))
    for column in (
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "best_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
    ):
        roll_column = f"{column}_roll"
        if roll_column in summary.columns:
            summary[column] = summary[roll_column].combine_first(summary[column])
    summary = summary.merge(cost_10, on="strategy_id", how="left")
    if "cost_10bps_strategy_vs_qqq" not in summary.columns:
        summary["cost_10bps_strategy_vs_qqq"] = np.nan

    current_best_rate, current_best_cost_10 = _current_best_thresholds(prior_decision_matrix)
    decisions = []
    for row in summary.to_dict("records"):
        decision = _classify_confirmation_row(
            row,
            current_best_positive_rate=current_best_rate,
        )
        decisions.append({**row, **decision})
    summary = pd.DataFrame(decisions)
    summary = _sort_confirmation_rows(summary)
    summary["candidate_rank"] = np.arange(1, len(summary) + 1)
    decision_matrix = _build_decision_matrix(summary)
    parameter_stability = build_parameter_stability(summary)
    report = render_qqq_only_rf_confirmation_markdown(
        summary=summary,
        decision_matrix=decision_matrix,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        parameter_stability=parameter_stability,
        current_best_positive_rate=current_best_rate,
        current_best_cost_10=current_best_cost_10,
    )
    return QQQOnlyRFConfirmationArtifacts(
        summary=summary,
        decision_matrix=decision_matrix,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        parameter_stability=parameter_stability,
        report_markdown=report,
    )


def write_qqq_only_rf_confirmation_report_artifacts(
    *,
    artifacts: QQQOnlyRFConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write required report CSVs, Markdown, and PNG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = report_output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_confirmation_charts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        paths=paths,
    )
    return paths


def report_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all required report output paths."""

    names = (
        SUMMARY_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        EXPOSURE_CHART_NAME,
        ROLLING_CHART_NAME,
        COST_SENSITIVITY_CHART_NAME,
        PARAMETER_STABILITY_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def build_parameter_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize local stability by parameter value."""

    rows: list[dict[str, object]] = []
    dimensions = (
        "training_mode",
        "target_name",
        "feature_set_name",
        "exposure_mode",
        "rebalance_frequency",
        "transaction_cost_bps",
        "threshold",
        "min_holding_days",
        "switch_margin",
    )
    model_rows = summary.loc[summary["model_type"].astype(str).eq("random_forest")].copy()
    for dimension in dimensions:
        if dimension not in model_rows.columns:
            continue
        for value, group in model_rows.groupby(dimension, dropna=False, sort=True):
            decisions = group["decision_label"].astype(str)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "config_count": len(group),
                    "confirmed_count": int(decisions.eq("confirmed_candidate").sum()),
                    "robust_count": int(decisions.eq("robust_confirmation_candidate").sum()),
                    "median_strategy_vs_qqq": _median(group["strategy_vs_qqq"]),
                    "max_strategy_vs_qqq": _max(group["strategy_vs_qqq"]),
                    "median_drawdown_improvement_vs_qqq": _median(
                        group["drawdown_improvement_vs_qqq"]
                    ),
                    "median_positive_vs_qqq_window_rate": _median(
                        group["positive_vs_qqq_window_rate"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_qqq_only_rf_confirmation_markdown(
    *,
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    current_best_positive_rate: float = CURRENT_BEST_POSITIVE_VS_QQQ_WINDOW_RATE,
    current_best_cost_10: float = CURRENT_BEST_COST_10BPS_STRATEGY_VS_QQQ,
) -> str:
    """Render the focused confirmation Markdown report."""

    completed = len(summary)
    robust_count = int(summary["decision_label"].astype(str).eq("robust_confirmation_candidate").sum())
    confirmed_count = int(summary["decision_label"].astype(str).eq("confirmed_candidate").sum())
    defensive_count = int(summary["decision_label"].astype(str).eq("defensive_reference").sum())
    prior_clone = summary.loc[summary["is_prior_best_candidate"].fillna(False)].head(1)
    best_rows = decision_matrix.loc[
        decision_matrix["decision_label"].isin(
            ["robust_confirmation_candidate", "confirmed_candidate"]
        )
    ].head(10)
    failed = decision_matrix.loc[decision_matrix["decision_label"].eq("reject_deprioritize")].tail(10)
    keep_family = robust_count + confirmed_count > 0
    second_confirmation = "yes" if keep_family else "no"
    paper_tracking = "research paper tracking only after a second confirmation" if keep_family else "no"

    return "\n".join(
        [
            "# QQQ-Only Random Forest Confirmation Report",
            "",
            "## 1. Executive summary.",
            "",
            f"- Completed rows reviewed: `{completed}`",
            f"- Confirmed candidates: `{confirmed_count}`",
            f"- Robust confirmation candidates: `{robust_count}`",
            f"- Defensive references: `{defensive_count}`",
            "- Production-ready: `false`",
            "",
            "## 2. Why focused confirmation.",
            "",
            (
                "This is a narrow robustness check around the best random-forest "
                "QQQ/cash timing family, not a new model zoo. The tradable portfolio "
                "is always QQQ plus cash."
            ),
            "",
            "## 3. Baseline candidate recap.",
            "",
            _markdown_table(
                prior_clone,
                [
                    "strategy_name",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "qqq_buy_hold_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "cost_10bps_strategy_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            (
                f"Prior decision-matrix reference: rolling positive rate approximately "
                f"`{current_best_positive_rate:.4f}` and 10 bps edge approximately "
                f"`{current_best_cost_10:.4f}`."
            ),
            "",
            "## 4. Parameter grid.",
            "",
            _markdown_table(
                _grid_table(summary),
                ["parameter", "values"],
            ),
            "",
            "## 5. Full-period QQQ comparison.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "candidate_rank",
                    "strategy_name",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "return_to_drawdown_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Equity curve]({EQUITY_CHART_NAME})",
            "",
            "## 6. Cost sensitivity.",
            "",
            _markdown_table(
                _cost_sensitivity_preview(cost_sensitivity, decision_matrix),
                [
                    "strategy_name",
                    "transaction_cost_bps",
                    "strategy_vs_qqq",
                    "estimated_transaction_cost",
                    "decision_at_cost",
                ],
            ),
            "",
            f"![Cost sensitivity]({COST_SENSITIVITY_CHART_NAME})",
            "",
            "## 7. Rolling-window consistency.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "strategy_name",
                    "positive_vs_qqq_window_rate",
                    "qqq_underperformance_rate",
                    "worst_strategy_vs_qqq",
                    "best_strategy_vs_qqq",
                    "drawdown_improvement_window_rate",
                ],
            ),
            "",
            f"![Rolling edge versus QQQ]({ROLLING_CHART_NAME})",
            "",
            "## 8. Drawdown behavior.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "strategy_name",
                    "max_drawdown",
                    "qqq_buy_hold_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "drawdown_improvement_window_rate",
                ],
            ),
            "",
            f"![Drawdown curve]({DRAWDOWN_CHART_NAME})",
            "",
            "## 9. Exposure behavior.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "strategy_name",
                    "average_exposure",
                    "cash_days_pct",
                    "trade_count",
                    "average_holding_days",
                ],
            ),
            "",
            f"![Exposure timeline]({EXPOSURE_CHART_NAME})",
            "",
            "## 10. Parameter stability.",
            "",
            _markdown_table(
                parameter_stability.sort_values(
                    ["robust_count", "confirmed_count", "max_strategy_vs_qqq"],
                    ascending=[False, False, False],
                    kind="mergesort",
                ).head(16),
                [
                    "dimension",
                    "value",
                    "config_count",
                    "confirmed_count",
                    "robust_count",
                    "median_strategy_vs_qqq",
                    "max_strategy_vs_qqq",
                    "median_positive_vs_qqq_window_rate",
                ],
            ),
            "",
            f"![Parameter stability]({PARAMETER_STABILITY_CHART_NAME})",
            "",
            "## 11. Best confirmed candidates.",
            "",
            _markdown_table(
                best_rows,
                [
                    "candidate_rank",
                    "strategy_name",
                    "strategy_vs_qqq",
                    "cost_10bps_strategy_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "drawdown_improvement_window_rate",
                    "decision_label",
                ],
            ),
            "",
            "## 12. What failed.",
            "",
            _markdown_table(
                failed,
                [
                    "strategy_name",
                    "strategy_vs_qqq",
                    "cost_10bps_strategy_vs_qqq",
                    "positive_vs_qqq_window_rate",
                    "return_to_drawdown_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            "## 13. Recommendation:",
            "",
            f"- Whether to keep this random forest family: `{str(keep_family).lower()}`",
            f"- Whether to run a second confirmation: `{second_confirmation}`",
            "- Whether to add features: `not from this run; keep additions for a separate study`",
            f"- Whether to move toward paper tracking: `{paper_tracking}`",
            "- Whether to promote to production: `no`",
            "",
            "## 14. Caveats:",
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


def _thresholds_for_target(target_name: str) -> tuple[float, ...]:
    return CLASSIFICATION_THRESHOLDS if target_name in CLASSIFICATION_TARGETS else REGRESSION_THRESHOLDS


def _ensure_result_columns(result: dict[str, object]) -> dict[str, object]:
    out = dict(result)
    for column in RESULT_COLUMNS:
        out.setdefault(column, np.nan)
    return {column: out[column] for column in RESULT_COLUMNS}


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


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, [*columns, *[column for column in out.columns if column not in columns]]]


def _is_prior_best_candidate(row: pd.Series) -> bool:
    return (
        str(row.get("strategy_name")) == BEST_STRATEGY_NAME
        and str(row.get("model_type")) == "random_forest"
        and str(row.get("training_mode")) == "rolling_5y"
        and str(row.get("target_name")) == "risk_on_20d"
        and str(row.get("feature_set_name")) == "qqq_self_cross_market"
        and str(row.get("exposure_mode")) == "binary_0_100"
        and str(row.get("rebalance_frequency")) == "weekly"
        and np.isclose(_float(row.get("transaction_cost_bps")), 0.0)
        and np.isclose(_float(row.get("threshold")), 0.5)
        and int(_float(row.get("min_holding_days"))) == 0
        and np.isclose(_float(row.get("switch_margin")), 0.0)
    )


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


def _cost_at_bps(cost_sensitivity: pd.DataFrame, bps: float) -> pd.DataFrame:
    if cost_sensitivity.empty:
        return pd.DataFrame(columns=["strategy_id", "cost_10bps_strategy_vs_qqq"])
    subset = cost_sensitivity.loc[
        np.isclose(pd.to_numeric(cost_sensitivity["transaction_cost_bps"], errors="coerce"), bps),
        ["strategy_id", "strategy_vs_qqq"],
    ].copy()
    return subset.rename(columns={"strategy_vs_qqq": "cost_10bps_strategy_vs_qqq"})


def _current_best_thresholds(
    prior_decision_matrix: pd.DataFrame | None,
) -> tuple[float, float]:
    if (
        prior_decision_matrix is not None
        and not prior_decision_matrix.empty
        and "strategy_name" in prior_decision_matrix.columns
    ):
        hits = prior_decision_matrix.loc[
            prior_decision_matrix.get("strategy_name", pd.Series(dtype=str))
            .astype(str)
            .eq(BEST_STRATEGY_NAME)
        ]
        if not hits.empty:
            rate = _float(hits.iloc[0].get("positive_vs_qqq_window_rate"))
            cost_10 = _float(hits.iloc[0].get("cost_10bps_strategy_vs_qqq"))
            if np.isfinite(rate) and np.isfinite(cost_10):
                return rate, cost_10
    return CURRENT_BEST_POSITIVE_VS_QQQ_WINDOW_RATE, CURRENT_BEST_COST_10BPS_STRATEGY_VS_QQQ


def _classify_confirmation_row(
    row: dict[str, object],
    *,
    current_best_positive_rate: float,
) -> dict[str, object]:
    beats_return = _float(row.get("strategy_vs_qqq")) > 0.0
    improves_drawdown = _float(row.get("drawdown_improvement_vs_qqq")) > 0.0
    improves_rtd = _float(row.get("return_to_drawdown_vs_qqq")) > 0.0
    cost_edge = _float(row.get("cost_10bps_strategy_vs_qqq")) > 0.0
    rolling_rate = _float(row.get("positive_vs_qqq_window_rate"))
    rolling_ok = rolling_rate >= current_best_positive_rate - 1e-9
    materially_defensive = (
        (not beats_return) and _float(row.get("drawdown_improvement_vs_qqq")) >= 0.05
    )
    if beats_return and improves_drawdown and improves_rtd and cost_edge and rolling_ok:
        if rolling_rate >= 0.65:
            label = "robust_confirmation_candidate"
        else:
            label = "confirmed_candidate"
    elif materially_defensive:
        label = "defensive_reference"
    else:
        label = "reject_deprioritize"
    return {
        "beats_qqq_return": beats_return,
        "improves_max_drawdown": improves_drawdown,
        "improves_return_to_drawdown": improves_rtd,
        "positive_edge_after_10bps": cost_edge,
        "rolling_consistency_not_worse_than_current_best": rolling_ok,
        "decision_label": label,
        "production_ready": False,
        "recommendation": _recommendation_for_label(label),
    }


def _recommendation_for_label(label: str) -> str:
    if label == "robust_confirmation_candidate":
        return "Keep for a second confirmation; still not production-ready."
    if label == "confirmed_candidate":
        return "Keep the RF family for focused follow-up; do not promote."
    if label == "defensive_reference":
        return "Use only as defensive context against QQQ buy-and-hold."
    return "Reject or deprioritize within this RF confirmation study."


def _sort_confirmation_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rank = {
        "robust_confirmation_candidate": 0,
        "confirmed_candidate": 1,
        "defensive_reference": 2,
        "reject_deprioritize": 3,
    }
    out = summary.copy()
    out["_rank"] = out["decision_label"].map(rank).fillna(9)
    out = out.sort_values(
        [
            "_rank",
            "strategy_vs_qqq",
            "cost_10bps_strategy_vs_qqq",
            "return_to_drawdown_vs_qqq",
            "drawdown_improvement_vs_qqq",
        ],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    return out.drop(columns=["_rank"]).reset_index(drop=True)


def _build_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_rank",
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
        "qqq_buy_hold_return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "rolling_window_count",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "best_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
        "cost_10bps_strategy_vs_qqq",
        "beats_qqq_return",
        "improves_max_drawdown",
        "improves_return_to_drawdown",
        "positive_edge_after_10bps",
        "rolling_consistency_not_worse_than_current_best",
        "average_exposure",
        "cash_days_pct",
        "trade_count",
        "average_holding_days",
        "decision_label",
        "recommendation",
        "production_ready",
    ]
    return _ensure_columns(summary, columns).loc[:, columns]


def _grid_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in (
        "training_mode",
        "target_name",
        "feature_set_name",
        "exposure_mode",
        "rebalance_frequency",
        "threshold",
        "min_holding_days",
        "switch_margin",
        "transaction_cost_bps",
    ):
        values = sorted(summary.loc[summary["model_type"].eq("random_forest"), parameter].dropna().unique())
        rows.append({"parameter": parameter, "values": ", ".join(str(value) for value in values)})
    return pd.DataFrame(rows)


def _cost_sensitivity_preview(
    cost_sensitivity: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if cost_sensitivity.empty or decision_matrix.empty:
        return pd.DataFrame()
    top_ids = decision_matrix.head(6)["strategy_id"].astype(str).tolist()
    names = decision_matrix.set_index("strategy_id")["strategy_name"].astype(str).to_dict()
    subset = cost_sensitivity.loc[cost_sensitivity["strategy_id"].astype(str).isin(top_ids)].copy()
    subset["strategy_name"] = subset["strategy_id"].map(names).fillna(subset["strategy_name"])
    return subset.sort_values(["strategy_name", "transaction_cost_bps"], kind="mergesort").head(24)


def _write_confirmation_charts(
    *,
    artifacts: QQQOnlyRFConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_ids = artifacts.decision_matrix.head(6)["strategy_id"].astype(str).tolist()
    qqq_label = "QQQ buy-and-hold"

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in top_ids:
        subset = _strategy_frame(daily_equity, strategy_id)
        if subset.empty:
            continue
        ax.plot(pd.to_datetime(subset["date"]), subset["equity"], label=subset["strategy_name"].iloc[0])
    if not daily_equity.empty:
        first = daily_equity.groupby("date", sort=True).head(1)
        ax.plot(pd.to_datetime(first["date"]), first["qqq_buy_hold_equity"], color="black", linewidth=2, label=qqq_label)
    ax.set_title("QQQ-only RF confirmation: top candidate equity")
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
        ax.plot(pd.to_datetime(first["date"]), first["qqq_buy_hold_drawdown"], color="black", linewidth=2, label=qqq_label)
    ax.set_title("QQQ-only RF confirmation: top candidate drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for strategy_id in top_ids[:4]:
        subset = _strategy_frame(exposure_log, strategy_id)
        if subset.empty:
            continue
        ax.step(
            pd.to_datetime(subset["date"]),
            pd.to_numeric(subset["effective_exposure"], errors="coerce"),
            where="post",
            label=subset["strategy_name"].iloc[0],
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("QQQ-only RF confirmation: QQQ/cash exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy_id in top_ids:
        subset = _strategy_frame(daily_equity, strategy_id)
        if subset.empty:
            continue
        diff = _rolling_strategy_vs_qqq(subset)
        ax.plot(pd.to_datetime(subset.loc[diff.index, "date"]), diff, label=subset["strategy_name"].iloc[0])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("QQQ-only RF confirmation: rolling 63-day edge vs QQQ")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[ROLLING_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    cost = artifacts.cost_sensitivity.copy()
    if not cost.empty:
        for strategy_id in top_ids[:5]:
            subset = cost.loc[cost["strategy_id"].astype(str).eq(strategy_id)].sort_values(
                "transaction_cost_bps"
            )
            if subset.empty:
                continue
            name = str(subset["strategy_name"].iloc[0])
            ax.plot(
                subset["transaction_cost_bps"],
                subset["strategy_vs_qqq"],
                marker="o",
                label=name,
            )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("QQQ-only RF confirmation: cost sensitivity")
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Strategy vs QQQ total return")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[COST_SENSITIVITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    stability = artifacts.parameter_stability.copy()
    if stability.empty:
        ax.text(0.5, 0.5, "No parameter stability rows", ha="center", va="center")
    else:
        plot_rows = stability.sort_values(
            ["robust_count", "confirmed_count", "max_strategy_vs_qqq"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(18)
        labels = plot_rows["dimension"].astype(str) + "=" + plot_rows["value"].astype(str)
        ax.barh(labels, plot_rows["max_strategy_vs_qqq"], color="#457b9d")
        ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("QQQ-only RF confirmation: parameter stability")
    ax.set_xlabel("Best strategy vs QQQ")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[PARAMETER_STABILITY_CHART_NAME], dpi=140)
    plt.close(fig)


def _strategy_frame(frame: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    if frame.empty or "strategy_id" not in frame.columns:
        return pd.DataFrame()
    out = frame.loc[frame["strategy_id"].astype(str).eq(str(strategy_id))].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    return out


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


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else np.nan
