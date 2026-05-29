"""Research-only paper-tracking package for locked QQQ-only RF timing rows.

This module is local-only and research-only. It packages the locked QQQ/cash
random-forest candidates selected after out-of-time confirmation for forward
paper tracking. It does not promote anything to production, export production
signals, create a model card, or run a broad sweep.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.qqq_only_best_candidate_deep_dive import (
    confirm_held_assets_are_qqq_cash_only,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    QQQOnlyRunSpec,
    QQQOnlySpecOutput,
    rebalance_dates_for_frequency,
)
from us_market_ml.qqq_only_rf_out_of_time_confirmation import (
    PRIMARY_HOLDOUT_END,
    PRIMARY_HOLDOUT_NAME,
    PRIMARY_HOLDOUT_START,
    build_qqq_only_rf_out_of_time_confirmation_specs,
)

OUTPUT_PREFIX = "qqq_only_rf_paper_tracking"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_paper_tracking")

PRIMARY_CANDIDATE_ID = "primary_rf_threshold_050"
SECONDARY_CANDIDATE_ID = "secondary_rf_threshold_045_hold10"
REJECTED_REFERENCE_ID = "rejected_rf_threshold_045_hold0"
CANDIDATE_IDS: tuple[str, ...] = (
    PRIMARY_CANDIDATE_ID,
    SECONDARY_CANDIDATE_ID,
    REJECTED_REFERENCE_ID,
)
COST_LEVELS_BPS: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)

CANDIDATE_REGISTRY_NAME = f"{OUTPUT_PREFIX}_candidate_registry.csv"
CANDIDATE_REGISTRY_JSON_NAME = f"{OUTPUT_PREFIX}_candidate_registry.json"
INITIAL_SIGNAL_NAME = f"{OUTPUT_PREFIX}_initial_signal.csv"
INITIAL_SIGNAL_JSON_NAME = f"{OUTPUT_PREFIX}_initial_signal.json"
REPLAY_SUMMARY_NAME = f"{OUTPUT_PREFIX}_replay_summary.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

EQUITY_CHART_NAME = f"{OUTPUT_PREFIX}_equity_curve.png"
DRAWDOWN_CHART_NAME = f"{OUTPUT_PREFIX}_drawdown_curve.png"
EXPOSURE_TIMELINE_CHART_NAME = f"{OUTPUT_PREFIX}_exposure_timeline.png"
HOLDOUT_COMPARISON_CHART_NAME = f"{OUTPUT_PREFIX}_holdout_comparison.png"

REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Why paper-tracking now.",
    "## 3. Why this is still not production.",
    "## 4. Primary candidate.",
    "## 5. Secondary reference.",
    "## 6. Rejected/deprioritized reference.",
    "## 7. Out-of-time confirmation recap.",
    "## 8. Initial paper-tracking signal snapshot.",
    "## 9. Monitoring rules.",
    "## 10. When to revisit the candidate.",
    "## 11. Recommendation:",
    "## 12. Caveats:",
)

REGISTRY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "role",
    "model_type",
    "training_mode",
    "target_name",
    "target_horizon",
    "feature_set_name",
    "exposure_mode",
    "rebalance_frequency",
    "rebalance_weekday",
    "random_state",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "threshold",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_tracking_bps",
    "tradable_holdings",
    "primary_benchmark",
    "other_etfs_role",
    "status",
    "production_ready",
)

INITIAL_SIGNAL_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "as_of_date",
    "latest_available_price_date",
    "target_exposure",
    "effective_exposure",
    "qqq_weight",
    "cash_weight",
    "held_assets",
    "raw_score",
    "threshold",
    "rebalance_weekday",
    "is_rebalance_day",
    "next_action_interpretation",
    "production_ready",
)

DECISION_MATRIX_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "role",
    "cost_view_bps",
    "strategy_id",
    "strategy_name",
    "holdout_period_name",
    "period_start",
    "period_end",
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "out_of_time_pass",
    "paper_tracking_candidate",
    "out_of_time_decision_label",
    "package_decision_label",
    "package_recommendation",
    "production_ready",
)

REPLAY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "role",
    "status",
    "holdout_period_name",
    "strategy_vs_qqq_0bps",
    "strategy_vs_qqq_10bps",
    "strategy_vs_qqq_20bps",
    "strategy_vs_qqq_30bps",
    "strategy_max_drawdown_0bps",
    "qqq_max_drawdown_0bps",
    "drawdown_improvement_vs_qqq_0bps",
    "trade_count_0bps",
    "latest_signal_target_exposure",
    "latest_signal_effective_exposure",
    "latest_signal_held_assets",
    "out_of_time_decision_label",
    "package_recommendation",
    "production_ready",
)


@dataclass(frozen=True)
class QQQOnlyRFPaperTrackingArtifacts:
    """In-memory artifacts for the paper-tracking package."""

    candidate_registry: pd.DataFrame
    initial_signal: pd.DataFrame
    replay_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    report_markdown: str
    replay_daily_equity: pd.DataFrame
    replay_daily_drawdown: pd.DataFrame
    replay_exposure_log: pd.DataFrame


def build_qqq_only_rf_paper_tracking_candidate_registry() -> pd.DataFrame:
    """Return the locked three-row research candidate registry."""

    rows = [
        _registry_row(
            candidate_id=PRIMARY_CANDIDATE_ID,
            role="primary_paper_tracking_candidate",
            threshold=0.50,
            min_holding_days=0,
            status="start research paper-tracking",
        ),
        _registry_row(
            candidate_id=SECONDARY_CANDIDATE_ID,
            role="secondary_paper_tracking_reference",
            threshold=0.45,
            min_holding_days=10,
            status="keep as secondary paper-tracking reference",
        ),
        _registry_row(
            candidate_id=REJECTED_REFERENCE_ID,
            role="rejected_out_of_time_reference",
            threshold=0.45,
            min_holding_days=0,
            status="deprioritized after out-of-time confirmation",
        ),
    ]
    registry = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    _validate_candidate_registry(registry)
    return registry


def build_qqq_only_rf_paper_tracking_specs() -> tuple[QQQOnlyRunSpec, ...]:
    """Return cost-0 replay specs for the locked registry candidates."""

    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    all_specs = build_qqq_only_rf_out_of_time_confirmation_specs()
    specs: list[QQQOnlyRunSpec] = []
    for row in registry.to_dict("records"):
        specs.append(_find_locked_spec(row, all_specs))
    return tuple(specs)


def build_qqq_only_rf_paper_tracking_artifacts(
    *,
    out_of_time_summary: pd.DataFrame,
    out_of_time_decision_matrix: pd.DataFrame,
    out_of_time_results: pd.DataFrame,
    out_of_time_exposure_log: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
) -> QQQOnlyRFPaperTrackingArtifacts:
    """Build all paper-tracking tables, report text, and chart inputs."""

    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    _validate_candidate_registry(registry)
    _validate_prior_artifacts(
        summary=out_of_time_summary,
        decision_matrix=out_of_time_decision_matrix,
        results=out_of_time_results,
        exposure_log=out_of_time_exposure_log,
    )
    _validate_replay_outputs(registry, replay_outputs)

    replay_exposure_log = _combine_output_frames(replay_outputs, "exposure_log")
    confirm_held_assets_are_qqq_cash_only(out_of_time_exposure_log)
    confirm_held_assets_are_qqq_cash_only(replay_exposure_log)

    initial_signal = build_initial_signal_snapshot(registry, replay_outputs)
    decision_matrix = build_paper_tracking_decision_matrix(
        registry=registry,
        out_of_time_decision_matrix=out_of_time_decision_matrix,
    )
    replay_summary = build_paper_tracking_replay_summary(
        registry=registry,
        decision_matrix=decision_matrix,
        initial_signal=initial_signal,
    )
    replay_daily_equity = _combine_output_frames(replay_outputs, "daily_equity")
    replay_daily_drawdown = _combine_output_frames(replay_outputs, "daily_drawdown")
    report = render_qqq_only_rf_paper_tracking_markdown(
        registry=registry,
        initial_signal=initial_signal,
        replay_summary=replay_summary,
        decision_matrix=decision_matrix,
        out_of_time_summary=out_of_time_summary,
        out_of_time_results=out_of_time_results,
    )
    return QQQOnlyRFPaperTrackingArtifacts(
        candidate_registry=registry,
        initial_signal=initial_signal,
        replay_summary=replay_summary,
        decision_matrix=decision_matrix,
        report_markdown=report,
        replay_daily_equity=replay_daily_equity,
        replay_daily_drawdown=replay_daily_drawdown,
        replay_exposure_log=replay_exposure_log,
    )


def build_initial_signal_snapshot(
    registry: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
) -> pd.DataFrame:
    """Build one research-only latest-signal row per registry candidate."""

    rows: list[dict[str, object]] = []
    for registry_row in registry.to_dict("records"):
        candidate_id = str(registry_row["candidate_id"])
        output = replay_outputs[candidate_id]
        exposure_log = output.exposure_log.copy()
        if exposure_log.empty:
            raise ValueError(f"Replay exposure log is empty for {candidate_id}.")
        exposure_log["date"] = pd.to_datetime(exposure_log["date"])
        latest = exposure_log.sort_values("date", kind="mergesort").iloc[-1]
        latest_date = pd.Timestamp(latest["date"])
        frame_dates = pd.DatetimeIndex(exposure_log["date"]).sort_values().unique()
        rebalance_dates = rebalance_dates_for_frequency(
            pd.DatetimeIndex(frame_dates),
            start_date=frame_dates[0],
            end_date=frame_dates[-1],
            frequency=str(registry_row["rebalance_frequency"]),
            weekday=str(registry_row["rebalance_weekday"]),
        )
        is_rebalance_day = latest_date.normalize() in {
            pd.Timestamp(date).normalize() for date in rebalance_dates
        }
        target_exposure = _float(latest.get("target_exposure"))
        effective_exposure = _float(latest.get("effective_exposure"))
        qqq_weight = _float(latest.get("qqq_weight", effective_exposure))
        cash_weight = _float(latest.get("cash_weight", 1.0 - qqq_weight))
        held_assets = _held_assets_from_exposure(effective_exposure)
        rows.append(
            {
                "candidate_id": candidate_id,
                "as_of_date": latest_date.date().isoformat(),
                "latest_available_price_date": latest_date.date().isoformat(),
                "target_exposure": target_exposure,
                "effective_exposure": effective_exposure,
                "qqq_weight": qqq_weight,
                "cash_weight": cash_weight,
                "held_assets": held_assets,
                "raw_score": _float(latest.get("raw_score")),
                "threshold": _float(registry_row.get("threshold")),
                "rebalance_weekday": registry_row["rebalance_weekday"],
                "is_rebalance_day": bool(is_rebalance_day),
                "next_action_interpretation": _next_action_interpretation(
                    is_rebalance_day=bool(is_rebalance_day),
                    target_exposure=target_exposure,
                    effective_exposure=effective_exposure,
                ),
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=INITIAL_SIGNAL_COLUMNS)


def build_paper_tracking_decision_matrix(
    *,
    registry: pd.DataFrame,
    out_of_time_decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build the package decision matrix from prior out-of-time cost views."""

    rows: list[dict[str, object]] = []
    for registry_row in registry.to_dict("records"):
        matched = _candidate_cost_rows(out_of_time_decision_matrix, registry_row)
        for _, row in matched.sort_values("transaction_cost_bps", kind="mergesort").iterrows():
            candidate_id = str(registry_row["candidate_id"])
            package_label, package_recommendation = _package_decision(registry_row)
            rejected_reference = str(registry_row["role"]) == "rejected_out_of_time_reference"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": registry_row["role"],
                    "cost_view_bps": _float(row.get("transaction_cost_bps")),
                    "strategy_id": row.get("strategy_id"),
                    "strategy_name": row.get("strategy_name"),
                    "holdout_period_name": row.get("holdout_period_name"),
                    "period_start": row.get("period_start"),
                    "period_end": row.get("period_end"),
                    "strategy_total_return": row.get("strategy_total_return"),
                    "qqq_total_return": row.get("qqq_total_return"),
                    "strategy_vs_qqq": row.get("strategy_vs_qqq"),
                    "strategy_max_drawdown": row.get("strategy_max_drawdown"),
                    "qqq_max_drawdown": row.get("qqq_max_drawdown"),
                    "drawdown_improvement_vs_qqq": row.get("drawdown_improvement_vs_qqq"),
                    "return_to_drawdown_vs_qqq": row.get("return_to_drawdown_vs_qqq"),
                    "average_exposure": row.get("average_exposure"),
                    "cash_days_pct": row.get("cash_days_pct"),
                    "trade_count": row.get("trade_count"),
                    "out_of_time_pass": False
                    if rejected_reference
                    else _bool(row.get("out_of_time_pass", False)),
                    "paper_tracking_candidate": False
                    if rejected_reference
                    else _bool(row.get("paper_tracking_candidate", False)),
                    "out_of_time_decision_label": "reject_deprioritize"
                    if rejected_reference
                    else row.get("decision_label"),
                    "package_decision_label": package_label,
                    "package_recommendation": package_recommendation,
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows, columns=DECISION_MATRIX_COLUMNS)


def build_paper_tracking_replay_summary(
    *,
    registry: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    initial_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Build one compact summary row per locked registry candidate."""

    signal_lookup = initial_signal.set_index("candidate_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for registry_row in registry.to_dict("records"):
        candidate_id = str(registry_row["candidate_id"])
        candidate_rows = decision_matrix.loc[decision_matrix["candidate_id"].eq(candidate_id)]
        zero = candidate_rows.loc[np.isclose(candidate_rows["cost_view_bps"], 0.0)]
        if zero.empty:
            raise ValueError(f"Missing 0 bps decision row for {candidate_id}.")
        base = zero.iloc[0]
        signal = signal_lookup[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "role": registry_row["role"],
                "status": registry_row["status"],
                "holdout_period_name": base.get("holdout_period_name"),
                "strategy_vs_qqq_0bps": _edge_at_cost(candidate_rows, 0.0),
                "strategy_vs_qqq_10bps": _edge_at_cost(candidate_rows, 10.0),
                "strategy_vs_qqq_20bps": _edge_at_cost(candidate_rows, 20.0),
                "strategy_vs_qqq_30bps": _edge_at_cost(candidate_rows, 30.0),
                "strategy_max_drawdown_0bps": base.get("strategy_max_drawdown"),
                "qqq_max_drawdown_0bps": base.get("qqq_max_drawdown"),
                "drawdown_improvement_vs_qqq_0bps": base.get(
                    "drawdown_improvement_vs_qqq"
                ),
                "trade_count_0bps": base.get("trade_count"),
                "latest_signal_target_exposure": signal["target_exposure"],
                "latest_signal_effective_exposure": signal["effective_exposure"],
                "latest_signal_held_assets": signal["held_assets"],
                "out_of_time_decision_label": base.get("out_of_time_decision_label"),
                "package_recommendation": base.get("package_recommendation"),
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=REPLAY_SUMMARY_COLUMNS)


def render_qqq_only_rf_paper_tracking_markdown(
    *,
    registry: pd.DataFrame,
    initial_signal: pd.DataFrame,
    replay_summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    out_of_time_summary: pd.DataFrame,
    out_of_time_results: pd.DataFrame,
) -> str:
    """Render the QQQ-only RF paper-tracking Markdown report."""

    primary = replay_summary.loc[replay_summary["candidate_id"].eq(PRIMARY_CANDIDATE_ID)]
    secondary = replay_summary.loc[replay_summary["candidate_id"].eq(SECONDARY_CANDIDATE_ID)]
    rejected = replay_summary.loc[replay_summary["candidate_id"].eq(REJECTED_REFERENCE_ID)]
    paper_count = int(out_of_time_summary.get("paper_tracking_candidate", pd.Series()).sum())
    result_count = len(out_of_time_results)
    decision_count = len(decision_matrix)
    return "\n".join(
        [
            "# QQQ-Only RF Paper-Tracking Package",
            "",
            "## 1. Executive summary.",
            "",
            "- Package type: `research-only paper-tracking`",
            "- Primary benchmark: `QQQ buy-and-hold`",
            "- Tradable holdings: `QQQ`, `cash`",
            "- Production-ready: `false`",
            "- Production signal export: `false`",
            "- Model card: `false`",
            f"- Locked registry rows: `{len(registry)}`",
            f"- Package decision rows: `{decision_count}`",
            "",
            "## 2. Why paper-tracking now.",
            "",
            (
                "The out-of-time confirmation changed the candidate ranking and "
                "identified the threshold 0.50, min_holding_days 0 RF row as the "
                "primary research paper-tracking candidate. This package locks that "
                "row and records a first local signal snapshot without opening a new "
                "model sweep."
            ),
            "",
            "## 3. Why this is still not production.",
            "",
            (
                "This package is for monitoring only. It does not create production "
                "signals, does not write to `artifacts/model_signals`, does not create "
                "a model card, and leaves `production_ready` as `false` for every row."
            ),
            "",
            "## 4. Primary candidate.",
            "",
            _markdown_table(primary, REPLAY_SUMMARY_COLUMNS),
            "",
            f"![Equity curve]({EQUITY_CHART_NAME})",
            "",
            "## 5. Secondary reference.",
            "",
            _markdown_table(secondary, REPLAY_SUMMARY_COLUMNS),
            "",
            "## 6. Rejected/deprioritized reference.",
            "",
            _markdown_table(rejected, REPLAY_SUMMARY_COLUMNS),
            "",
            "## 7. Out-of-time confirmation recap.",
            "",
            f"- Prior out-of-time result rows read: `{result_count}`",
            f"- Prior paper-tracking candidate rows: `{paper_count}`",
            f"- Primary holdout: `{PRIMARY_HOLDOUT_START}` to `{PRIMARY_HOLDOUT_END}`",
            "- Confirmation result: threshold 0.50 became the primary paper-tracking row.",
            "",
            _markdown_table(
                decision_matrix,
                [
                    "candidate_id",
                    "role",
                    "cost_view_bps",
                    "strategy_vs_qqq",
                    "drawdown_improvement_vs_qqq",
                    "out_of_time_decision_label",
                    "package_decision_label",
                ],
            ),
            "",
            f"![Holdout comparison]({HOLDOUT_COMPARISON_CHART_NAME})",
            "",
            "## 8. Initial paper-tracking signal snapshot.",
            "",
            _markdown_table(initial_signal, INITIAL_SIGNAL_COLUMNS),
            "",
            f"![Exposure timeline]({EXPOSURE_TIMELINE_CHART_NAME})",
            "",
            "## 9. Monitoring rules.",
            "",
            "- Monitor the primary threshold 0.50 row first.",
            "- Keep the threshold 0.45 with min_holding_days 10 row as a secondary reference.",
            "- Track QQQ/cash exposure, raw score, turnover, drawdown, and QQQ-relative return.",
            "- Treat all non-QQQ ETFs and reference series as features only.",
            "- Keep QQQ buy-and-hold as the primary benchmark.",
            "",
            "## 10. When to revisit the candidate.",
            "",
            "- Revisit if the primary candidate persistently trails QQQ buy-and-hold.",
            "- Revisit if drawdown behavior becomes worse than the out-of-time profile.",
            "- Revisit if signal flips become unstable or economically hard to justify.",
            "- Revisit before any production discussion; do not promote from this package.",
            "",
            "## 11. Recommendation:",
            "",
            "- Start research paper-tracking for primary threshold 0.50.",
            "- Keep threshold 0.45 with min_holding_days 10 as secondary reference.",
            "- Do not promote to production.",
            "- Do not run another broad sweep yet.",
            "",
            "## 12. Caveats:",
            "",
            "- Research-only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ/cash only.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Other ETFs/reference series are features only.",
            "- `production_ready` is always false.",
            "",
            f"![Drawdown curve]({DRAWDOWN_CHART_NAME})",
            "",
        ]
    )


def write_qqq_only_rf_paper_tracking_artifacts(
    *,
    artifacts: QQQOnlyRFPaperTrackingArtifacts,
    output_dir: Path,
) -> dict[str, Path]:
    """Write required CSV, JSON, Markdown, and PNG report artifacts."""

    _ensure_not_model_signals_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = report_output_paths(output_dir)
    artifacts.candidate_registry.to_csv(paths[CANDIDATE_REGISTRY_NAME], index=False)
    _write_json_records(artifacts.candidate_registry, paths[CANDIDATE_REGISTRY_JSON_NAME])
    artifacts.initial_signal.to_csv(paths[INITIAL_SIGNAL_NAME], index=False)
    _write_json_records(artifacts.initial_signal, paths[INITIAL_SIGNAL_JSON_NAME])
    artifacts.replay_summary.to_csv(paths[REPLAY_SUMMARY_NAME], index=False)
    artifacts.decision_matrix.to_csv(paths[DECISION_MATRIX_NAME], index=False)
    paths[REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_paper_tracking_charts(artifacts=artifacts, paths=paths)
    return paths


def report_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all required paper-tracking output paths."""

    names = (
        CANDIDATE_REGISTRY_NAME,
        CANDIDATE_REGISTRY_JSON_NAME,
        INITIAL_SIGNAL_NAME,
        INITIAL_SIGNAL_JSON_NAME,
        REPLAY_SUMMARY_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
        EQUITY_CHART_NAME,
        DRAWDOWN_CHART_NAME,
        EXPOSURE_TIMELINE_CHART_NAME,
        HOLDOUT_COMPARISON_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def _registry_row(
    *,
    candidate_id: str,
    role: str,
    threshold: float,
    min_holding_days: int,
    status: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "role": role,
        "model_type": "random_forest",
        "training_mode": "rolling_5y",
        "target_name": "risk_on_20d",
        "target_horizon": 20,
        "feature_set_name": "qqq_self_cross_market",
        "exposure_mode": "binary_0_100",
        "rebalance_frequency": "weekly",
        "rebalance_weekday": "FRI",
        "random_state": 42,
        "n_estimators": 60,
        "max_depth": None,
        "min_samples_leaf": 5,
        "threshold": threshold,
        "min_holding_days": min_holding_days,
        "switch_margin": 0.0,
        "transaction_cost_tracking_bps": "0,10,20,30",
        "tradable_holdings": "QQQ,cash",
        "primary_benchmark": "QQQ buy-and-hold",
        "other_etfs_role": "features_only",
        "status": status,
        "production_ready": False,
    }


def _find_locked_spec(
    registry_row: Mapping[str, object],
    specs: Sequence[QQQOnlyRunSpec],
) -> QQQOnlyRunSpec:
    matches = [
        spec
        for spec in specs
        if not spec.is_baseline
        and spec.model_type == registry_row["model_type"]
        and spec.training_mode == registry_row["training_mode"]
        and spec.target_name == registry_row["target_name"]
        and spec.feature_set_name == registry_row["feature_set_name"]
        and spec.exposure_mode == registry_row["exposure_mode"]
        and spec.rebalance_frequency == registry_row["rebalance_frequency"]
        and spec.rebalance_weekday == registry_row["rebalance_weekday"]
        and spec.random_state == int(registry_row["random_state"])
        and spec.n_estimators == int(registry_row["n_estimators"])
        and spec.max_depth is None
        and spec.min_samples_leaf == int(registry_row["min_samples_leaf"])
        and np.isclose(spec.threshold, float(registry_row["threshold"]))
        and spec.min_holding_days == int(registry_row["min_holding_days"])
        and np.isclose(spec.switch_margin, float(registry_row["switch_margin"]))
        and np.isclose(spec.transaction_cost_bps, 0.0)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one locked out-of-time spec for "
            f"{registry_row['candidate_id']}; found {len(matches)}."
        )
    return matches[0]


def _candidate_cost_rows(
    decision_matrix: pd.DataFrame,
    registry_row: Mapping[str, object],
) -> pd.DataFrame:
    matrix = decision_matrix.copy()
    required = {
        "model_type",
        "training_mode",
        "target_name",
        "feature_set_name",
        "exposure_mode",
        "rebalance_frequency",
        "rebalance_weekday",
        "random_state",
        "n_estimators",
        "min_samples_leaf",
        "threshold",
        "min_holding_days",
        "switch_margin",
        "transaction_cost_bps",
        "holdout_period_name",
        "max_depth",
    }
    missing = required.difference(matrix.columns)
    if missing:
        raise ValueError(
            "Out-of-time decision matrix is missing required columns: "
            + ", ".join(sorted(missing))
        )
    mask = (
        matrix["model_type"].astype(str).eq(str(registry_row["model_type"]))
        & matrix["training_mode"].astype(str).eq(str(registry_row["training_mode"]))
        & matrix["target_name"].astype(str).eq(str(registry_row["target_name"]))
        & matrix["feature_set_name"].astype(str).eq(str(registry_row["feature_set_name"]))
        & matrix["exposure_mode"].astype(str).eq(str(registry_row["exposure_mode"]))
        & matrix["rebalance_frequency"].astype(str).eq(str(registry_row["rebalance_frequency"]))
        & matrix["rebalance_weekday"].astype(str).eq(str(registry_row["rebalance_weekday"]))
        & np.isclose(pd.to_numeric(matrix["random_state"], errors="coerce"), 42)
        & np.isclose(pd.to_numeric(matrix["n_estimators"], errors="coerce"), 60)
        & _max_depth_is_default(matrix["max_depth"])
        & np.isclose(pd.to_numeric(matrix["min_samples_leaf"], errors="coerce"), 5)
        & np.isclose(
            pd.to_numeric(matrix["threshold"], errors="coerce"),
            float(registry_row["threshold"]),
        )
        & np.isclose(
            pd.to_numeric(matrix["min_holding_days"], errors="coerce"),
            int(registry_row["min_holding_days"]),
        )
        & np.isclose(pd.to_numeric(matrix["switch_margin"], errors="coerce"), 0.0)
        & matrix["holdout_period_name"].astype(str).eq(PRIMARY_HOLDOUT_NAME)
    )
    out = matrix.loc[mask].copy()
    out = out.loc[
        pd.to_numeric(out["transaction_cost_bps"], errors="coerce").isin(COST_LEVELS_BPS)
    ].copy()
    found_costs = set(pd.to_numeric(out["transaction_cost_bps"], errors="coerce").astype(float))
    expected = set(COST_LEVELS_BPS)
    if found_costs != expected:
        raise ValueError(
            f"Missing required cost views for {registry_row['candidate_id']}: "
            f"expected {sorted(expected)}, found {sorted(found_costs)}."
        )
    return out


def _max_depth_is_default(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.lower().fillna("")
    return text.isin({"", "nan", "none", "default", "<na>"})


def _package_decision(registry_row: Mapping[str, object]) -> tuple[str, str]:
    role = str(registry_row["role"])
    if role == "primary_paper_tracking_candidate":
        return role, "Start research paper-tracking; do not promote to production."
    if role == "secondary_paper_tracking_reference":
        return role, "Keep as secondary/reference row beside the primary."
    return "rejected_deprioritized_reference", "Reference only; do not use as candidate."


def _edge_at_cost(candidate_rows: pd.DataFrame, cost_bps: float) -> float:
    row = candidate_rows.loc[np.isclose(candidate_rows["cost_view_bps"], cost_bps)]
    if row.empty:
        return np.nan
    return _float(row.iloc[0].get("strategy_vs_qqq"))


def _combine_output_frames(
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
    attribute: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for candidate_id in CANDIDATE_IDS:
        output = replay_outputs[candidate_id]
        frame = getattr(output, attribute).copy()
        frame.insert(0, "candidate_id", candidate_id)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _validate_candidate_registry(registry: pd.DataFrame) -> None:
    if tuple(registry["candidate_id"].astype(str)) != CANDIDATE_IDS:
        raise ValueError("Candidate registry must contain exactly the locked candidate IDs.")
    if not registry["production_ready"].eq(False).all():
        raise ValueError("Candidate registry production_ready must always be false.")
    holdings = set(registry["tradable_holdings"].astype(str))
    if holdings != {"QQQ,cash"}:
        raise ValueError("Candidate registry tradable holdings must be QQQ,cash only.")


def _validate_prior_artifacts(
    *,
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    results: pd.DataFrame,
    exposure_log: pd.DataFrame,
) -> None:
    if summary.empty:
        raise ValueError("Out-of-time summary is empty.")
    if decision_matrix.empty:
        raise ValueError("Out-of-time decision matrix is empty.")
    if results.empty:
        raise ValueError("Out-of-time results are empty.")
    if exposure_log.empty:
        raise ValueError("Out-of-time exposure log is empty.")
    if "production_ready" in summary and not summary["production_ready"].eq(False).all():
        raise ValueError("Out-of-time summary contains production_ready values other than false.")
    if "production_ready" in decision_matrix and not decision_matrix["production_ready"].eq(
        False
    ).all():
        raise ValueError(
            "Out-of-time decision matrix contains production_ready values other than false."
        )


def _validate_replay_outputs(
    registry: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
) -> None:
    missing = [
        candidate_id
        for candidate_id in registry["candidate_id"]
        if candidate_id not in replay_outputs
    ]
    if missing:
        raise ValueError("Missing replay outputs for: " + ", ".join(missing))


def _next_action_interpretation(
    *,
    is_rebalance_day: bool,
    target_exposure: float,
    effective_exposure: float,
) -> str:
    target = _format_pct(target_exposure)
    effective = _format_pct(effective_exposure)
    if not is_rebalance_day:
        return (
            "Research-only snapshot: not a scheduled rebalance day; keep the "
            f"paper-tracking effective exposure at {effective} QQQ until the next FRI check."
        )
    if np.isclose(target_exposure, effective_exposure):
        return (
            "Research-only FRI rebalance check: no exposure change; maintain "
            f"{effective} QQQ."
        )
    if target_exposure > effective_exposure:
        return (
            "Research-only FRI rebalance check: target increases QQQ exposure "
            f"from {effective} to {target}."
        )
    return (
        "Research-only FRI rebalance check: target reduces QQQ exposure "
        f"from {effective} to {target}."
    )


def _held_assets_from_exposure(exposure: float) -> str:
    if exposure <= 1e-9:
        return "cash"
    if exposure >= 1.0 - 1e-9:
        return "QQQ"
    return "QQQ,cash"


def _write_paper_tracking_charts(
    *,
    artifacts: QQQOnlyRFPaperTrackingArtifacts,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = artifacts.replay_daily_equity.copy()
    drawdown = artifacts.replay_daily_drawdown.copy()
    exposure = artifacts.replay_exposure_log.copy()
    decision = artifacts.decision_matrix.copy()
    registry = artifacts.candidate_registry.set_index("candidate_id")

    fig, ax = plt.subplots(figsize=(11, 6))
    qqq_plotted = False
    if equity.empty:
        ax.text(0.5, 0.5, "No replay equity rows", ha="center", va="center")
    else:
        equity["date"] = pd.to_datetime(equity["date"])
        for candidate_id, group in equity.groupby("candidate_id", sort=False):
            group = group.sort_values("date", kind="mergesort")
            strategy_equity = pd.to_numeric(group["equity"], errors="coerce")
            if strategy_equity.empty:
                continue
            ax.plot(
                group["date"],
                strategy_equity / strategy_equity.iloc[0],
                label=_chart_label(candidate_id, registry),
            )
            if not qqq_plotted:
                qqq_equity = pd.to_numeric(group["qqq_buy_hold_equity"], errors="coerce")
                ax.plot(
                    group["date"],
                    qqq_equity / qqq_equity.iloc[0],
                    color="black",
                    linewidth=2,
                    label="QQQ buy-and-hold",
                )
                qqq_plotted = True
    ax.set_title("QQQ-only RF paper-tracking replay equity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[EQUITY_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    qqq_plotted = False
    if drawdown.empty:
        ax.text(0.5, 0.5, "No replay drawdown rows", ha="center", va="center")
    else:
        drawdown["date"] = pd.to_datetime(drawdown["date"])
        for candidate_id, group in drawdown.groupby("candidate_id", sort=False):
            group = group.sort_values("date", kind="mergesort")
            ax.plot(
                group["date"],
                pd.to_numeric(group["drawdown"], errors="coerce"),
                label=_chart_label(candidate_id, registry),
            )
            if not qqq_plotted:
                ax.plot(
                    group["date"],
                    pd.to_numeric(group["qqq_buy_hold_drawdown"], errors="coerce"),
                    color="black",
                    linewidth=2,
                    label="QQQ buy-and-hold",
                )
                qqq_plotted = True
    ax.set_title("QQQ-only RF paper-tracking replay drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[DRAWDOWN_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    if exposure.empty:
        ax.text(0.5, 0.5, "No replay exposure rows", ha="center", va="center")
    else:
        exposure["date"] = pd.to_datetime(exposure["date"])
        for candidate_id, group in exposure.groupby("candidate_id", sort=False):
            group = group.sort_values("date", kind="mergesort")
            ax.step(
                group["date"],
                pd.to_numeric(group["effective_exposure"], errors="coerce"),
                where="post",
                linewidth=1.5,
                label=_chart_label(candidate_id, registry),
            )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("QQQ-only RF paper-tracking effective exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_TIMELINE_CHART_NAME], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if decision.empty:
        ax.text(0.5, 0.5, "No holdout comparison rows", ha="center", va="center")
    else:
        pivot = decision.pivot_table(
            index="candidate_id",
            columns="cost_view_bps",
            values="strategy_vs_qqq",
            aggfunc="first",
        )
        pivot = pivot.reindex(CANDIDATE_IDS)
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Primary holdout strategy vs QQQ by cost view")
    ax.set_ylabel("Strategy vs QQQ")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[HOLDOUT_COMPARISON_CHART_NAME], dpi=140)
    plt.close(fig)


def _chart_label(candidate_id: str, registry: pd.DataFrame) -> str:
    if candidate_id not in registry.index:
        return candidate_id
    row = registry.loc[candidate_id]
    return (
        f"{candidate_id} "
        f"thr={_format_number(row['threshold'])} hold={_format_number(row['min_holding_days'])}"
    )


def _write_json_records(frame: pd.DataFrame, path: Path) -> None:
    cleaned = frame.replace({np.nan: None})
    path.write_text(
        json.dumps(cleaned.to_dict(orient="records"), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _ensure_not_model_signals_path(path: Path) -> None:
    parts = tuple(part.lower() for part in Path(path).parts)
    for index, part in enumerate(parts[:-1]):
        if part == "artifacts" and parts[index + 1] == "model_signals":
            raise ValueError("Paper-tracking package must not write under artifacts/model_signals.")


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


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    return f"{value:.0%}"


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
