"""Out-of-time confirmation for the locked QQQ-only RF timing family.

This module is research-only and local-only. It evaluates the pre-registered
threshold 0.45 QQQ/cash random-forest family on stricter later holdout
periods. Other ETFs and reference series may be features, but simulated
holdings remain QQQ and cash only.
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
    confirm_held_assets_are_qqq_cash_only,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    QQQOnlyRunSpec,
    _baseline_specs,
)

OUTPUT_PREFIX = "qqq_only_rf_out_of_time_confirmation"
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_out_of_time_confirmation")

DEFAULT_RF_N_ESTIMATORS = 60
DEFAULT_RF_MAX_DEPTH: int | None = None
DEFAULT_RF_MIN_SAMPLES_LEAF = 5

PRIMARY_HOLDOUT_NAME = "primary_2025_2026"
PRIMARY_HOLDOUT_START = "2025-01-01"
PRIMARY_HOLDOUT_END = "2026-05-15"
ADDITIONAL_HOLDOUT_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("readout_2024_2026", "2024-01-01", "2026-05-15"),
    ("calendar_2025", "2025-01-01", "2025-12-31"),
    ("ytd_2026", "2026-01-01", "2026-05-15"),
)
HOLDOUT_PERIODS: tuple[tuple[str, str, str], ...] = (
    (PRIMARY_HOLDOUT_NAME, PRIMARY_HOLDOUT_START, PRIMARY_HOLDOUT_END),
    *ADDITIONAL_HOLDOUT_PERIODS,
)

COST_LEVELS: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
WEEKDAY_ROBUSTNESS_REFERENCES: tuple[str, ...] = ("MON", "TUE", "THU", "FRI")
BASELINE_RULES: tuple[str, ...] = ("qqq_buy_hold", "qqq_sma200_timing")
FAIL_BADLY_20BPS_EDGE_FLOOR = -0.05
DEFENSIVE_DRAWDOWN_IMPROVEMENT = 0.05

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

RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "candidate_family_key",
    "holdout_period_name",
    "period_start",
    "period_end",
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
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "strategy_return_to_drawdown",
    "qqq_return_to_drawdown",
    "return_to_drawdown_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "cost_10bps_strategy_vs_qqq",
    "cost_20bps_strategy_vs_qqq",
    "cost_30bps_strategy_vs_qqq",
    "beats_qqq_return",
    "improves_max_drawdown",
    "improves_return_to_drawdown",
    "positive_edge_after_10bps",
    "does_not_fail_badly_after_20bps",
    "not_contradictory_2024_2026",
    "not_contradictory_2025",
    "not_contradictory_2026",
    "additional_holdout_contradiction_count",
    "out_of_time_pass",
    "paper_tracking_candidate",
    "decision_label",
    "recommendation",
    "production_ready",
)

SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
HOLDOUT_PERIODS_NAME = f"{OUTPUT_PREFIX}_holdout_periods.csv"
COST_STABILITY_NAME = f"{OUTPUT_PREFIX}_cost_stability.csv"
CANDIDATE_STABILITY_NAME = f"{OUTPUT_PREFIX}_candidate_stability.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = "qqq_only_rf_out_of_time_equity_curve_top_candidates.png"
DRAWDOWN_CHART_NAME = "qqq_only_rf_out_of_time_drawdown_curve_top_candidates.png"
HOLDOUT_RETURNS_CHART_NAME = "qqq_only_rf_out_of_time_holdout_returns.png"
HOLDOUT_DRAWDOWNS_CHART_NAME = "qqq_only_rf_out_of_time_holdout_drawdowns.png"
COST_STABILITY_CHART_NAME = "qqq_only_rf_out_of_time_cost_stability.png"
EXPOSURE_TIMELINE_CHART_NAME = "qqq_only_rf_out_of_time_exposure_timeline.png"

REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Why out-of-time confirmation.",
    "## 3. Locked candidate set.",
    "## 4. Primary holdout: 2025-01-01 to 2026-05-15.",
    "## 5. Additional holdout readouts.",
    "## 6. Cost stability.",
    "## 7. Candidate stability.",
    "## 8. Exposure behavior during holdout.",
    "## 9. QQQ buy-and-hold comparison.",
    "## 10. Paper-tracking readiness.",
    "## 11. Recommendation:",
    "## 12. Caveats:",
)


@dataclass(frozen=True)
class QQQOnlyRFOutOfTimeConfirmationArtifacts:
    """In-memory artifacts for the out-of-time confirmation report."""

    summary: pd.DataFrame
    holdout_periods: pd.DataFrame
    cost_stability: pd.DataFrame
    candidate_stability: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str


def build_holdout_periods(
    *,
    primary_start: str = PRIMARY_HOLDOUT_START,
    primary_end: str = PRIMARY_HOLDOUT_END,
) -> tuple[tuple[str, str, str], ...]:
    """Return the primary holdout plus fixed additional readouts."""

    return ((PRIMARY_HOLDOUT_NAME, primary_start, primary_end), *ADDITIONAL_HOLDOUT_PERIODS)


def build_qqq_only_rf_out_of_time_confirmation_specs() -> tuple[QQQOnlyRunSpec, ...]:
    """Build the compact, pre-registered out-of-time candidate set."""

    specs_by_key: dict[str, QQQOnlyRunSpec] = {}
    for spec in _baseline_specs((0.0,)):
        if spec.baseline_rule in BASELINE_RULES:
            specs_by_key[spec.run_key] = spec

    def add_spec(**overrides: object) -> None:
        config = {**BASE_RF_CONFIG, **overrides}
        spec = _rf_spec(config)
        specs_by_key[spec.run_key] = spec

    for cost in COST_LEVELS:
        add_spec(transaction_cost_bps=cost)
        add_spec(transaction_cost_bps=cost, min_holding_days=10)
        add_spec(transaction_cost_bps=cost, exposure_mode="partial_50_100")
        add_spec(transaction_cost_bps=cost, threshold=0.50)

    for weekday in WEEKDAY_ROBUSTNESS_REFERENCES:
        add_spec(rebalance_weekday=weekday, transaction_cost_bps=10.0)

    return tuple(specs_by_key.values())


def build_qqq_only_rf_out_of_time_confirmation_result_rows(
    result: dict[str, object],
    *,
    spec: QQQOnlyRunSpec,
    daily_equity: pd.DataFrame,
    exposure_log: pd.DataFrame,
    holdout_periods: Sequence[tuple[str, str, str]] = HOLDOUT_PERIODS,
) -> list[dict[str, object]]:
    """Build one metric row per pre-defined holdout period for a completed spec."""

    rows: list[dict[str, object]] = []
    for period_name, period_start, period_end in holdout_periods:
        period_equity = _date_slice(daily_equity, period_start, period_end)
        if period_equity.empty:
            continue
        period_exposure = _date_slice(exposure_log, period_start, period_end)
        metrics = _period_metrics(period_equity, period_exposure)
        row: dict[str, object] = {
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "candidate_family_key": _candidate_family_key_from_spec(spec),
            "holdout_period_name": period_name,
            "period_start": period_start,
            "period_end": period_end,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "target_horizon": spec.target_horizon,
            "feature_set_name": spec.feature_set_name,
            "feature_count": result.get("feature_count", np.nan),
            "exposure_mode": spec.exposure_mode,
            "rebalance_frequency": spec.rebalance_frequency,
            "rebalance_weekday": spec.rebalance_weekday,
            "random_state": spec.random_state,
            "n_estimators": spec.n_estimators,
            "max_depth": spec.max_depth,
            "min_samples_leaf": spec.min_samples_leaf,
            "transaction_cost_bps": spec.transaction_cost_bps,
            "threshold": spec.threshold,
            "min_holding_days": spec.min_holding_days,
            "switch_margin": spec.switch_margin,
            "production_ready": False,
            "decision_label": "unclassified",
            "recommendation": "Pending report-level out-of-time decision.",
        }
        row.update(metrics)
        rows.append(_ensure_result_dict(row))
    return rows


def build_qqq_only_rf_out_of_time_confirmation_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
) -> QQQOnlyRFOutOfTimeConfirmationArtifacts:
    """Build all required out-of-time tables, report text, and chart inputs."""

    confirm_held_assets_are_qqq_cash_only(exposure_log)
    holdout_periods = _ensure_columns(results.copy(), RESULT_COLUMNS)
    holdout_periods["production_ready"] = False
    holdout_periods["candidate_family_key"] = holdout_periods.apply(
        _candidate_family_key_from_row,
        axis=1,
    )
    holdout_periods = _apply_out_of_time_decisions(holdout_periods)
    summary = _build_summary(holdout_periods)
    cost_stability = build_cost_stability(holdout_periods)
    candidate_stability = build_candidate_stability(holdout_periods, summary)
    decision_matrix = _build_decision_matrix(summary)
    report = render_qqq_only_rf_out_of_time_confirmation_markdown(
        summary=summary,
        holdout_periods=holdout_periods,
        cost_stability=cost_stability,
        candidate_stability=candidate_stability,
        decision_matrix=decision_matrix,
    )
    return QQQOnlyRFOutOfTimeConfirmationArtifacts(
        summary=summary,
        holdout_periods=holdout_periods,
        cost_stability=cost_stability,
        candidate_stability=candidate_stability,
        decision_matrix=decision_matrix,
        report_markdown=report,
    )


def write_qqq_only_rf_out_of_time_confirmation_report_artifacts(
    *,
    artifacts: QQQOnlyRFOutOfTimeConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write required report CSVs, Markdown, and PNG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = report_output_paths(output_dir)
    artifacts.summary.to_csv(paths[SUMMARY_NAME], index=False)
    artifacts.holdout_periods.to_csv(paths[HOLDOUT_PERIODS_NAME], index=False)
    artifacts.cost_stability.to_csv(paths[COST_STABILITY_NAME], index=False)
    artifacts.candidate_stability.to_csv(paths[CANDIDATE_STABILITY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_out_of_time_charts(
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
        HOLDOUT_PERIODS_NAME,
        COST_STABILITY_NAME,
        CANDIDATE_STABILITY_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        HOLDOUT_RETURNS_CHART_NAME,
        HOLDOUT_DRAWDOWNS_CHART_NAME,
        COST_STABILITY_CHART_NAME,
        EXPOSURE_TIMELINE_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def build_cost_stability(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary-holdout cost behavior for every model candidate family."""

    primary = _primary_rows(holdout_periods)
    model_rows = primary.loc[primary["model_type"].astype(str).eq("random_forest")].copy()
    rows: list[dict[str, object]] = []
    for family_key, group in model_rows.groupby("candidate_family_key", sort=False):
        base_name = _family_display_name(group)
        for _, row in group.sort_values("transaction_cost_bps", kind="mergesort").iterrows():
            rows.append(
                {
                    "candidate_family_key": family_key,
                    "candidate_family_name": base_name,
                    "strategy_id": row["strategy_id"],
                    "strategy_name": row["strategy_name"],
                    "transaction_cost_bps": row["transaction_cost_bps"],
                    "holdout_period_name": row["holdout_period_name"],
                    "strategy_total_return": row["strategy_total_return"],
                    "qqq_total_return": row["qqq_total_return"],
                    "strategy_vs_qqq": row["strategy_vs_qqq"],
                    "positive_edge": _float(row["strategy_vs_qqq"]) > 0.0,
                    "strategy_max_drawdown": row["strategy_max_drawdown"],
                    "drawdown_improvement_vs_qqq": row["drawdown_improvement_vs_qqq"],
                    "trade_count": row["trade_count"],
                    "decision_label": row["decision_label"],
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def build_candidate_stability(
    holdout_periods: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each config across the primary and additional holdout readouts."""

    label_lookup = summary.set_index("strategy_id")["decision_label"].to_dict()
    rank_lookup = summary.set_index("strategy_id")["candidate_rank"].to_dict()
    rows: list[dict[str, object]] = []
    for strategy_id, group in holdout_periods.groupby("strategy_id", sort=False):
        group = group.copy()
        edge = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        dd = pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
        contradictions = [
            not _period_not_contradictory(row)
            for row in group.to_dict("records")
            if row.get("holdout_period_name") != PRIMARY_HOLDOUT_NAME
        ]
        first = group.iloc[0]
        rows.append(
            {
                "candidate_rank": rank_lookup.get(strategy_id, np.nan),
                "strategy_id": strategy_id,
                "strategy_name": first.get("strategy_name"),
                "candidate_family_key": first.get("candidate_family_key"),
                "model_type": first.get("model_type"),
                "exposure_mode": first.get("exposure_mode"),
                "threshold": first.get("threshold"),
                "min_holding_days": first.get("min_holding_days"),
                "rebalance_weekday": first.get("rebalance_weekday"),
                "transaction_cost_bps": first.get("transaction_cost_bps"),
                "period_count": len(group),
                "positive_edge_period_count": int((edge > 0.0).sum()),
                "positive_edge_period_rate": float((edge > 0.0).mean()) if len(edge) else np.nan,
                "drawdown_improved_period_count": int((dd >= 0.0).sum()),
                "drawdown_improved_period_rate": float((dd >= 0.0).mean()) if len(dd) else np.nan,
                "worst_strategy_vs_qqq": float(edge.min()) if edge.notna().any() else np.nan,
                "best_strategy_vs_qqq": float(edge.max()) if edge.notna().any() else np.nan,
                "additional_holdout_contradiction_count": int(sum(contradictions)),
                "decision_label": label_lookup.get(strategy_id, "unclassified"),
                "production_ready": False,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["candidate_rank", "worst_strategy_vs_qqq"],
            ascending=[True, False],
            kind="mergesort",
        )
    return out.reset_index(drop=True)


def render_qqq_only_rf_out_of_time_confirmation_markdown(
    *,
    summary: pd.DataFrame,
    holdout_periods: pd.DataFrame,
    cost_stability: pd.DataFrame,
    candidate_stability: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the out-of-time confirmation Markdown report."""

    paper_count = int(summary["paper_tracking_candidate"].fillna(False).sum())
    pass_count = int(summary["out_of_time_pass"].fillna(False).sum())
    defensive_count = int(summary["decision_label"].astype(str).eq("defensive_reference").sum())
    best_rows = decision_matrix.loc[
        decision_matrix["decision_label"].isin(["paper_tracking_candidate", "out_of_time_pass"])
    ].head(12)
    top_ids = decision_matrix.head(8)["strategy_id"].astype(str).tolist()
    top_holdouts = holdout_periods.loc[holdout_periods["strategy_id"].astype(str).isin(top_ids)]
    primary_top = summary.head(14)
    threshold_045_tracking = _threshold_045_paper_tracking(summary)
    partial_keep = _partial_reference_should_continue(summary)

    return "\n".join(
        [
            "# QQQ-Only RF Out-of-Time Confirmation Report",
            "",
            "## 1. Executive summary.",
            "",
            f"- Completed primary-holdout rows reviewed: `{len(summary)}`",
            f"- Out-of-time pass rows: `{pass_count}`",
            f"- Paper-tracking candidate rows: `{paper_count}`",
            f"- Defensive reference rows: `{defensive_count}`",
            "- Primary benchmark: `QQQ buy-and-hold`",
            "- Tradable holdings: `QQQ`, `cash`",
            "- Production-ready: `false`",
            "",
            "## 2. Why out-of-time confirmation.",
            "",
            (
                "The second confirmation kept threshold 0.45 but required one more "
                "pre-defined later-period check before paper-tracking. This study "
                "does not select new parameters from the holdout and does not expand "
                "the model zoo."
            ),
            "",
            "## 3. Locked candidate set.",
            "",
            _markdown_table(_locked_candidate_table(), ["field", "locked_value"]),
            "",
            "## 4. Primary holdout: 2025-01-01 to 2026-05-15.",
            "",
            _markdown_table(
                primary_top,
                [
                    "candidate_rank",
                    "strategy_name",
                    "transaction_cost_bps",
                    "strategy_total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "return_to_drawdown_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Primary holdout equity]({EQUITY_CHART_NAME})",
            "",
            f"![Primary holdout drawdown]({DRAWDOWN_CHART_NAME})",
            "",
            "## 5. Additional holdout readouts.",
            "",
            _markdown_table(
                top_holdouts,
                [
                    "strategy_name",
                    "holdout_period_name",
                    "strategy_total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            f"![Holdout returns]({HOLDOUT_RETURNS_CHART_NAME})",
            "",
            f"![Holdout drawdowns]({HOLDOUT_DRAWDOWNS_CHART_NAME})",
            "",
            "## 6. Cost stability.",
            "",
            _markdown_table(
                cost_stability.head(24),
                [
                    "candidate_family_name",
                    "transaction_cost_bps",
                    "strategy_vs_qqq",
                    "positive_edge",
                    "drawdown_improvement_vs_qqq",
                    "trade_count",
                    "decision_label",
                ],
            ),
            "",
            f"![Cost stability]({COST_STABILITY_CHART_NAME})",
            "",
            "## 7. Candidate stability.",
            "",
            _markdown_table(
                candidate_stability.head(20),
                [
                    "strategy_name",
                    "period_count",
                    "positive_edge_period_count",
                    "drawdown_improved_period_count",
                    "worst_strategy_vs_qqq",
                    "additional_holdout_contradiction_count",
                    "decision_label",
                ],
            ),
            "",
            "## 8. Exposure behavior during holdout.",
            "",
            _markdown_table(
                primary_top,
                [
                    "strategy_name",
                    "average_exposure",
                    "cash_days_pct",
                    "trade_count",
                    "transaction_cost_bps",
                    "decision_label",
                ],
            ),
            "",
            f"![Exposure timeline]({EXPOSURE_TIMELINE_CHART_NAME})",
            "",
            "## 9. QQQ buy-and-hold comparison.",
            "",
            _markdown_table(
                _qqq_comparison_table(summary),
                [
                    "strategy_name",
                    "strategy_total_return",
                    "qqq_total_return",
                    "strategy_vs_qqq",
                    "strategy_max_drawdown",
                    "qqq_max_drawdown",
                    "return_to_drawdown_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            "## 10. Paper-tracking readiness.",
            "",
            (
                f"Paper-tracking readiness is `{str(paper_count > 0).lower()}`. "
                "Rows must pass the primary holdout, retain positive primary return "
                "edge, avoid worse primary drawdown, keep a positive 10 bps edge, "
                "and avoid contradictory additional readouts."
            ),
            "",
            _markdown_table(
                best_rows,
                [
                    "candidate_rank",
                    "strategy_name",
                    "transaction_cost_bps",
                    "strategy_vs_qqq",
                    "drawdown_improvement_vs_qqq",
                    "cost_10bps_strategy_vs_qqq",
                    "cost_20bps_strategy_vs_qqq",
                    "decision_label",
                    "recommendation",
                ],
            ),
            "",
            "## 11. Recommendation:",
            "",
            (
                "- Whether threshold 0.45 should move to research paper-tracking: "
                f"`{threshold_045_tracking}`"
            ),
            f"- Whether partial_50_100 should be kept as smoother reference: `{partial_keep}`",
            "- Whether another broad sweep is needed: `false`",
            "- Whether to promote to production: `false`",
            "",
            "## 12. Caveats:",
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
        max_depth=None
        if config["max_depth"] is None or pd.isna(config["max_depth"])
        else int(config["max_depth"]),
        min_samples_leaf=payload["min_samples_leaf"],
    )


def _run_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    parts = [
        "rf_oot",
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


def _candidate_family_key_from_spec(spec: QQQOnlyRunSpec) -> str:
    payload = {
        "model_type": spec.model_type,
        "training_mode": spec.training_mode,
        "target_name": spec.target_name,
        "feature_set_name": spec.feature_set_name,
        "exposure_mode": spec.exposure_mode,
        "rebalance_frequency": spec.rebalance_frequency,
        "rebalance_weekday": spec.rebalance_weekday,
        "random_state": spec.random_state,
        "n_estimators": spec.n_estimators,
        "max_depth": "default" if spec.max_depth is None else spec.max_depth,
        "min_samples_leaf": spec.min_samples_leaf,
        "threshold": spec.threshold,
        "min_holding_days": spec.min_holding_days,
        "switch_margin": spec.switch_margin,
        "baseline_rule": spec.baseline_rule or "",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _candidate_family_key_from_row(row: pd.Series) -> str:
    payload = {
        "model_type": row.get("model_type"),
        "training_mode": row.get("training_mode"),
        "target_name": row.get("target_name"),
        "feature_set_name": row.get("feature_set_name"),
        "exposure_mode": row.get("exposure_mode"),
        "rebalance_frequency": row.get("rebalance_frequency"),
        "rebalance_weekday": row.get("rebalance_weekday"),
        "random_state": _maybe_int(row.get("random_state")),
        "n_estimators": _maybe_int(row.get("n_estimators")),
        "max_depth": _max_depth_payload(row.get("max_depth")),
        "min_samples_leaf": _maybe_int(row.get("min_samples_leaf")),
        "threshold": _maybe_float(row.get("threshold")),
        "min_holding_days": _maybe_int(row.get("min_holding_days")),
        "switch_margin": _maybe_float(row.get("switch_margin")),
        "baseline_rule": "qqq_buy_hold"
        if str(row.get("strategy_name")) == "qqq_buy_hold"
        else "qqq_sma200_timing"
        if str(row.get("strategy_name")) == "qqq_sma200_timing"
        else "",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _period_metrics(
    period_equity: pd.DataFrame,
    period_exposure: pd.DataFrame,
) -> dict[str, object]:
    strategy_returns = pd.to_numeric(period_equity["daily_return"], errors="coerce").fillna(0.0)
    qqq_returns = pd.to_numeric(period_equity["qqq_return"], errors="coerce").fillna(0.0)
    strategy_equity = (1.0 + strategy_returns).cumprod()
    qqq_equity = (1.0 + qqq_returns).cumprod()
    strategy_total = float(strategy_equity.iloc[-1] - 1.0) if not strategy_equity.empty else np.nan
    qqq_total = float(qqq_equity.iloc[-1] - 1.0) if not qqq_equity.empty else np.nan
    strategy_dd = calculate_max_drawdown(strategy_equity) if not strategy_equity.empty else np.nan
    qqq_dd = calculate_max_drawdown(qqq_equity) if not qqq_equity.empty else np.nan
    exposure_values = pd.to_numeric(
        period_exposure.get("effective_exposure", period_equity["effective_exposure"]),
        errors="coerce",
    )
    turnover = pd.to_numeric(period_equity.get("turnover", pd.Series(dtype=float)), errors="coerce")
    strategy_rtd = _return_to_drawdown(strategy_total, strategy_dd)
    qqq_rtd = _return_to_drawdown(qqq_total, qqq_dd)
    return {
        "strategy_total_return": strategy_total,
        "qqq_total_return": qqq_total,
        "strategy_vs_qqq": strategy_total - qqq_total,
        "strategy_max_drawdown": strategy_dd,
        "qqq_max_drawdown": qqq_dd,
        "drawdown_improvement_vs_qqq": strategy_dd - qqq_dd,
        "strategy_return_to_drawdown": strategy_rtd,
        "qqq_return_to_drawdown": qqq_rtd,
        "return_to_drawdown_vs_qqq": strategy_rtd - qqq_rtd,
        "average_exposure": (
            float(exposure_values.mean()) if exposure_values.notna().any() else np.nan
        ),
        "cash_days_pct": float((exposure_values <= 1e-9).mean())
        if exposure_values.notna().any()
        else np.nan,
        "trade_count": int((turnover > 1e-9).sum()) if not turnover.empty else 0,
    }


def _apply_out_of_time_decisions(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    out = holdout_periods.copy()
    out = _merge_replacing_columns(out, _primary_cost_edges(out), on="candidate_family_key")
    out = _merge_replacing_columns(out, _additional_holdout_flags(out), on="strategy_id")

    primary_decisions: dict[str, dict[str, object]] = {}
    for row in _primary_rows(out).to_dict("records"):
        primary_decisions[str(row["strategy_id"])] = _decision_for_primary_row(row)

    for column in (
        "beats_qqq_return",
        "improves_max_drawdown",
        "improves_return_to_drawdown",
        "positive_edge_after_10bps",
        "does_not_fail_badly_after_20bps",
        "out_of_time_pass",
        "paper_tracking_candidate",
        "decision_label",
        "recommendation",
    ):
        out[column] = out["strategy_id"].astype(str).map(
            {strategy_id: values.get(column) for strategy_id, values in primary_decisions.items()}
        )
    out["production_ready"] = False
    return _ensure_columns(out, RESULT_COLUMNS)


def _decision_for_primary_row(row: dict[str, object]) -> dict[str, object]:
    strategy_name = str(row.get("strategy_name"))
    if strategy_name == "qqq_buy_hold":
        return _decision_payload(
            label="benchmark",
            recommendation="Primary benchmark only.",
            row=row,
            out_of_time_pass=False,
            paper_tracking_candidate=False,
        )

    beats = _float(row.get("strategy_vs_qqq")) > 0.0
    improves_dd = _float(row.get("drawdown_improvement_vs_qqq")) > 0.0
    improves_rtd = _float(row.get("return_to_drawdown_vs_qqq")) > 0.0
    positive_10 = _float(row.get("cost_10bps_strategy_vs_qqq")) > 0.0
    cost_20 = _float(row.get("cost_20bps_strategy_vs_qqq"))
    not_bad_20 = np.isfinite(cost_20) and cost_20 >= FAIL_BADLY_20BPS_EDGE_FLOOR
    model_row = str(row.get("model_type")) == "random_forest"
    out_of_time_pass = bool(
        model_row and beats and (improves_dd or improves_rtd) and positive_10 and not_bad_20
    )
    additional_ok = (
        bool(row.get("not_contradictory_2024_2026", False))
        and bool(row.get("not_contradictory_2025", False))
        and bool(row.get("not_contradictory_2026", False))
    )
    paper = bool(
        out_of_time_pass
        and _float(row.get("strategy_vs_qqq")) > 0.0
        and _float(row.get("drawdown_improvement_vs_qqq")) >= 0.0
        and additional_ok
    )
    defensive = (
        _float(row.get("strategy_vs_qqq")) <= 0.0
        and _float(row.get("drawdown_improvement_vs_qqq")) >= DEFENSIVE_DRAWDOWN_IMPROVEMENT
    )
    if paper:
        label = "paper_tracking_candidate"
        recommendation = "Move to research paper-tracking; do not promote to production."
    elif out_of_time_pass:
        label = "out_of_time_pass"
        recommendation = "Keep as confirmed out-of-time research row."
    elif defensive:
        label = "defensive_reference"
        recommendation = "Use only as defensive context against QQQ buy-and-hold."
    else:
        label = "reject_deprioritize"
        recommendation = "Reject or deprioritize within this out-of-time confirmation."
    return _decision_payload(
        label=label,
        recommendation=recommendation,
        row=row,
        out_of_time_pass=out_of_time_pass,
        paper_tracking_candidate=paper,
    )


def _decision_payload(
    *,
    label: str,
    recommendation: str,
    row: dict[str, object],
    out_of_time_pass: bool,
    paper_tracking_candidate: bool,
) -> dict[str, object]:
    return {
        "beats_qqq_return": _float(row.get("strategy_vs_qqq")) > 0.0,
        "improves_max_drawdown": _float(row.get("drawdown_improvement_vs_qqq")) > 0.0,
        "improves_return_to_drawdown": _float(row.get("return_to_drawdown_vs_qqq")) > 0.0,
        "positive_edge_after_10bps": _float(row.get("cost_10bps_strategy_vs_qqq")) > 0.0,
        "does_not_fail_badly_after_20bps": _float(row.get("cost_20bps_strategy_vs_qqq"))
        >= FAIL_BADLY_20BPS_EDGE_FLOOR,
        "out_of_time_pass": out_of_time_pass,
        "paper_tracking_candidate": paper_tracking_candidate,
        "decision_label": label,
        "recommendation": recommendation,
    }


def _primary_cost_edges(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    primary = _primary_rows(holdout_periods)
    pieces = []
    for bps in (10.0, 20.0, 30.0):
        subset = primary.loc[
            np.isclose(pd.to_numeric(primary["transaction_cost_bps"], errors="coerce"), bps),
            ["candidate_family_key", "strategy_vs_qqq"],
        ].copy()
        pieces.append(
            subset.rename(columns={"strategy_vs_qqq": f"cost_{int(bps)}bps_strategy_vs_qqq"})
        )
    out = pd.DataFrame({"candidate_family_key": primary["candidate_family_key"].drop_duplicates()})
    for piece in pieces:
        out = out.merge(
            piece.drop_duplicates("candidate_family_key"),
            on="candidate_family_key",
            how="left",
        )
    return out


def _additional_holdout_flags(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_id, group in holdout_periods.groupby("strategy_id", sort=False):
        flags = {
            "not_contradictory_2024_2026": True,
            "not_contradictory_2025": True,
            "not_contradictory_2026": True,
        }
        for _, row in group.iterrows():
            period = str(row.get("holdout_period_name"))
            if period == "readout_2024_2026":
                flags["not_contradictory_2024_2026"] = _period_not_contradictory(row)
            elif period == "calendar_2025":
                flags["not_contradictory_2025"] = _period_not_contradictory(row)
            elif period == "ytd_2026":
                flags["not_contradictory_2026"] = _period_not_contradictory(row)
        contradiction_count = sum(1 for value in flags.values() if not value)
        rows.append(
            {
                "strategy_id": strategy_id,
                **flags,
                "additional_holdout_contradiction_count": contradiction_count,
            }
        )
    return pd.DataFrame(rows)


def _period_not_contradictory(row: pd.Series | dict[str, object]) -> bool:
    edge = _float(row.get("strategy_vs_qqq"))
    dd = _float(row.get("drawdown_improvement_vs_qqq"))
    rtd = _float(row.get("return_to_drawdown_vs_qqq"))
    return not ((edge < -0.05 and dd < 0.0) or (edge < -0.10 and rtd < 0.0))


def _build_summary(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    summary = _primary_rows(holdout_periods).copy()
    rank = {
        "paper_tracking_candidate": 0,
        "out_of_time_pass": 1,
        "defensive_reference": 2,
        "benchmark": 3,
        "reject_deprioritize": 4,
    }
    summary["_rank"] = summary["decision_label"].map(rank).fillna(9)
    summary = summary.sort_values(
        [
            "_rank",
            "strategy_vs_qqq",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown_vs_qqq",
        ],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop(columns="_rank")
    summary = summary.reset_index(drop=True)
    summary["candidate_rank"] = np.arange(1, len(summary) + 1)
    summary["production_ready"] = False
    return summary


def _build_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "candidate_rank",
        "strategy_id",
        "strategy_name",
        "candidate_family_key",
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
        "holdout_period_name",
        "period_start",
        "period_end",
        "strategy_total_return",
        "qqq_total_return",
        "strategy_vs_qqq",
        "strategy_max_drawdown",
        "qqq_max_drawdown",
        "drawdown_improvement_vs_qqq",
        "strategy_return_to_drawdown",
        "qqq_return_to_drawdown",
        "return_to_drawdown_vs_qqq",
        "average_exposure",
        "cash_days_pct",
        "trade_count",
        "cost_10bps_strategy_vs_qqq",
        "cost_20bps_strategy_vs_qqq",
        "cost_30bps_strategy_vs_qqq",
        "beats_qqq_return",
        "improves_max_drawdown",
        "improves_return_to_drawdown",
        "positive_edge_after_10bps",
        "does_not_fail_badly_after_20bps",
        "not_contradictory_2024_2026",
        "not_contradictory_2025",
        "not_contradictory_2026",
        "out_of_time_pass",
        "paper_tracking_candidate",
        "decision_label",
        "recommendation",
        "production_ready",
    )
    return _ensure_columns(summary, columns).loc[:, list(columns)]


def _write_out_of_time_charts(
    *,
    artifacts: QQQOnlyRFOutOfTimeConfirmationArtifacts,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_ids = artifacts.decision_matrix.head(6)["strategy_id"].astype(str).tolist()
    primary_start, primary_end = _primary_period_bounds(artifacts.holdout_periods)

    fig, ax = plt.subplots(figsize=(11, 6))
    qqq_plotted = False
    for strategy_id in top_ids:
        subset = _strategy_period_equity(daily_equity, strategy_id, primary_start, primary_end)
        if subset.empty:
            continue
        ax.plot(subset["date"], subset["equity"], label=subset["strategy_name"].iloc[0])
        if not qqq_plotted:
            ax.plot(
                subset["date"],
                subset["qqq_equity"],
                color="black",
                linewidth=2,
                label="QQQ buy-and-hold",
            )
            qqq_plotted = True
    ax.set_title("QQQ-only RF out-of-time confirmation: primary-holdout equity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[EQUITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    qqq_plotted = False
    for strategy_id in top_ids:
        subset = _strategy_period_equity(daily_drawdown, strategy_id, primary_start, primary_end)
        if subset.empty:
            continue
        ax.plot(subset["date"], subset["drawdown"], label=subset["strategy_name"].iloc[0])
        if not qqq_plotted:
            ax.plot(
                subset["date"],
                subset["qqq_drawdown"],
                color="black",
                linewidth=2,
                label="QQQ buy-and-hold",
            )
            qqq_plotted = True
    ax.set_title("QQQ-only RF out-of-time confirmation: primary-holdout drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    chart_rows = artifacts.holdout_periods.loc[
        artifacts.holdout_periods["strategy_id"].astype(str).isin(top_ids[:4])
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    if chart_rows.empty:
        ax.text(0.5, 0.5, "No holdout rows", ha="center", va="center")
    else:
        pivot = chart_rows.pivot_table(
            index="holdout_period_name",
            columns="strategy_name",
            values="strategy_total_return",
            aggfunc="first",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Out-of-time holdout total returns")
    ax.set_ylabel("Total return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[HOLDOUT_RETURNS_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    if chart_rows.empty:
        ax.text(0.5, 0.5, "No holdout rows", ha="center", va="center")
    else:
        pivot = chart_rows.pivot_table(
            index="holdout_period_name",
            columns="strategy_name",
            values="strategy_max_drawdown",
            aggfunc="first",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Out-of-time holdout max drawdowns")
    ax.set_ylabel("Max drawdown")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[HOLDOUT_DRAWDOWNS_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    cost = artifacts.cost_stability.copy()
    if cost.empty:
        ax.text(0.5, 0.5, "No cost rows", ha="center", va="center")
    else:
        for family_name, group in cost.groupby("candidate_family_name", sort=False):
            group = group.sort_values("transaction_cost_bps", kind="mergesort")
            ax.plot(
                group["transaction_cost_bps"],
                group["strategy_vs_qqq"],
                marker="o",
                linewidth=1.4,
                label=str(family_name)[:80],
            )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.legend(fontsize=6)
    ax.set_title("Primary holdout cost stability")
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Strategy vs QQQ")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[COST_STABILITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for strategy_id in top_ids[:5]:
        subset = _strategy_period_frame(exposure_log, strategy_id, primary_start, primary_end)
        if subset.empty:
            continue
        ax.step(
            pd.to_datetime(subset["date"]),
            pd.to_numeric(subset["effective_exposure"], errors="coerce"),
            where="post",
            linewidth=1.4,
            label=subset["strategy_name"].iloc[0],
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Primary holdout QQQ exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_TIMELINE_CHART_NAME], dpi=140)
    plt.close(fig)


def _strategy_period_equity(
    frame: pd.DataFrame,
    strategy_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    subset = _strategy_period_frame(frame, strategy_id, start, end)
    if subset.empty:
        return subset
    if "daily_return" in subset.columns:
        strategy_return = pd.to_numeric(subset["daily_return"], errors="coerce").fillna(0.0)
        qqq_return = pd.to_numeric(subset["qqq_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + strategy_return).cumprod()
        qqq_equity = (1.0 + qqq_return).cumprod()
        subset["equity"] = equity.to_numpy()
        subset["qqq_equity"] = qqq_equity.to_numpy()
        subset["drawdown"] = (equity / equity.cummax() - 1.0).to_numpy()
        subset["qqq_drawdown"] = (qqq_equity / qqq_equity.cummax() - 1.0).to_numpy()
        return subset
    equity = pd.to_numeric(subset["equity"], errors="coerce")
    qqq_equity = pd.to_numeric(subset["qqq_buy_hold_equity"], errors="coerce")
    subset["equity"] = (equity / equity.iloc[0]).to_numpy()
    subset["qqq_equity"] = (qqq_equity / qqq_equity.iloc[0]).to_numpy()
    subset["drawdown"] = (subset["equity"] / subset["equity"].cummax() - 1.0).to_numpy()
    subset["qqq_drawdown"] = (subset["qqq_equity"] / subset["qqq_equity"].cummax() - 1.0).to_numpy()
    return subset


def _strategy_period_frame(
    frame: pd.DataFrame,
    strategy_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if frame.empty or "strategy_id" not in frame.columns:
        return pd.DataFrame()
    out = frame.loc[frame["strategy_id"].astype(str).eq(str(strategy_id))].copy()
    return _date_slice(out, start, end)


def _primary_rows(holdout_periods: pd.DataFrame) -> pd.DataFrame:
    return holdout_periods.loc[
        holdout_periods["holdout_period_name"].astype(str).eq(PRIMARY_HOLDOUT_NAME)
    ].copy()


def _primary_period_bounds(holdout_periods: pd.DataFrame) -> tuple[str, str]:
    primary = _primary_rows(holdout_periods)
    if primary.empty:
        return PRIMARY_HOLDOUT_START, PRIMARY_HOLDOUT_END
    return str(primary["period_start"].iloc[0]), str(primary["period_end"].iloc[0])


def _date_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.loc[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].copy()


def _locked_candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("model_type", "random_forest"),
            ("training_mode", "rolling_5y"),
            ("target_name", "risk_on_20d"),
            ("feature_set_name", "qqq_self_cross_market"),
            ("base_exposure_mode", "binary_0_100"),
            ("reference_exposure_mode", "partial_50_100"),
            ("rebalance_frequency", "weekly"),
            ("base_rebalance_weekday", "FRI"),
            ("thresholds", "0.45 locked family, 0.50 prior-best reference"),
            ("min_holding_days", "0 base, 10 locked reference"),
            ("rf_params", "random_state=42, n_estimators=60, max_depth=None, min_samples_leaf=5"),
            ("cost_levels_bps", "0, 10, 20, 30"),
            ("tradable_holdings", "QQQ, cash"),
            ("primary_benchmark", "QQQ buy-and-hold"),
        ],
        columns=["field", "locked_value"],
    )


def _qqq_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    benchmark = summary.loc[summary["strategy_name"].astype(str).eq("qqq_buy_hold")]
    selected = summary.loc[
        summary["decision_label"].isin(
            ["paper_tracking_candidate", "out_of_time_pass", "defensive_reference"]
        )
    ].head(10)
    out = pd.concat([benchmark, selected], ignore_index=True).drop_duplicates("strategy_id")
    return out if not out.empty else summary.head(10)


def _threshold_045_paper_tracking(summary: pd.DataFrame) -> str:
    model = summary.loc[
        summary["model_type"].astype(str).eq("random_forest")
        & np.isclose(pd.to_numeric(summary["threshold"], errors="coerce"), 0.45)
        & summary["paper_tracking_candidate"].fillna(False)
    ]
    return str(not model.empty).lower()


def _partial_reference_should_continue(summary: pd.DataFrame) -> str:
    partial = summary.loc[
        summary["exposure_mode"].astype(str).eq("partial_50_100")
        & ~summary["decision_label"].astype(str).eq("reject_deprioritize")
    ]
    return str(not partial.empty).lower()


def _family_display_name(group: pd.DataFrame) -> str:
    row = group.sort_values("transaction_cost_bps", kind="mergesort").iloc[0]
    return (
        f"{row['exposure_mode']} thr={_format_number(row['threshold'])} "
        f"hold={_format_number(row['min_holding_days'])} "
        f"{row['rebalance_weekday']} leaf={_format_number(row['min_samples_leaf'])}"
    )


def _ensure_result_dict(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    for column in RESULT_COLUMNS:
        out.setdefault(column, np.nan)
    return {column: out[column] for column in RESULT_COLUMNS}


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, [*columns, *[column for column in out.columns if column not in columns]]]


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


def _format_number(value: object) -> str:
    number = _float(value)
    if not np.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _max_depth_payload(value: object) -> str | int:
    if value is None:
        return "default"
    try:
        if pd.isna(value):
            return "default"
    except (TypeError, ValueError):
        pass
    text = str(value).lower()
    if text in {"", "nan", "none", "default", "<na>"}:
        return "default"
    return int(float(value))


def _max_depth_label(value: object) -> str:
    return "default" if _max_depth_payload(value) == "default" else str(int(float(value)))


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown) or max_drawdown >= 0.0:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _maybe_float(value: object) -> float | str:
    number = _float(value)
    return number if np.isfinite(number) else ""


def _maybe_int(value: object) -> int | str:
    number = _float(value)
    return int(number) if np.isfinite(number) else ""
