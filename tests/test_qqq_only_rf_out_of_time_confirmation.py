from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.qqq_only_rf_out_of_time_confirmation import (
    HOLDOUT_PERIODS,
    REPORT_SECTIONS,
    RESULT_COLUMNS,
    build_qqq_only_rf_out_of_time_confirmation_artifacts,
    build_qqq_only_rf_out_of_time_confirmation_result_rows,
    build_qqq_only_rf_out_of_time_confirmation_specs,
    report_output_paths,
    write_qqq_only_rf_out_of_time_confirmation_report_artifacts,
)


def test_out_of_time_grid_is_locked_and_compact() -> None:
    specs = build_qqq_only_rf_out_of_time_confirmation_specs()
    model_specs = [spec for spec in specs if not spec.is_baseline]
    baselines = [spec for spec in specs if spec.is_baseline]

    assert len(model_specs) < 50
    assert len(baselines) == 2
    assert {spec.baseline_rule for spec in baselines} == {"qqq_buy_hold", "qqq_sma200_timing"}
    assert {spec.model_type for spec in model_specs} == {"random_forest"}
    assert {spec.training_mode for spec in model_specs} == {"rolling_5y"}
    assert {spec.target_name for spec in model_specs} == {"risk_on_20d"}
    assert {spec.feature_set_name for spec in model_specs} == {"qqq_self_cross_market"}
    assert {spec.rebalance_frequency for spec in model_specs} == {"weekly"}
    assert {spec.random_state for spec in model_specs} == {42}
    assert {spec.n_estimators for spec in model_specs} == {60}
    assert {spec.max_depth for spec in model_specs} == {None}
    assert {spec.min_samples_leaf for spec in model_specs} == {5}
    assert {spec.transaction_cost_bps for spec in model_specs} == {0.0, 10.0, 20.0, 30.0}
    assert {spec.rebalance_weekday for spec in model_specs} == {"MON", "TUE", "THU", "FRI"}
    assert set(spec.exposure_mode for spec in model_specs).issubset(
        {"binary_0_100", "partial_50_100"}
    )


def test_out_of_time_artifacts_write_required_tables_and_charts(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log = _out_of_time_inputs()

    artifacts = build_qqq_only_rf_out_of_time_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )

    assert set(RESULT_COLUMNS).issubset(artifacts.holdout_periods.columns)
    assert set(artifacts.holdout_periods["holdout_period_name"]) == {
        name for name, _, _ in HOLDOUT_PERIODS
    }
    assert artifacts.summary["holdout_period_name"].eq("primary_2025_2026").all()
    assert artifacts.summary["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["decision_label"].isin(
        ["paper_tracking_candidate", "out_of_time_pass"]
    ).any()
    assert set(exposure_log["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert not artifacts.cost_stability.empty
    assert not artifacts.candidate_stability.empty
    for section in REPORT_SECTIONS:
        assert section in artifacts.report_markdown
    assert "No production signal export" in artifacts.report_markdown
    assert "No model card" in artifacts.report_markdown

    paths = write_qqq_only_rf_out_of_time_confirmation_report_artifacts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        output_dir=tmp_path,
    )

    assert paths.keys() == report_output_paths(tmp_path).keys()
    for name in [
        "qqq_only_rf_out_of_time_confirmation_summary.csv",
        "qqq_only_rf_out_of_time_confirmation_holdout_periods.csv",
        "qqq_only_rf_out_of_time_confirmation_cost_stability.csv",
        "qqq_only_rf_out_of_time_confirmation_candidate_stability.csv",
        "qqq_only_rf_out_of_time_confirmation_decision_matrix.csv",
        "qqq_only_rf_out_of_time_confirmation_report.md",
        "qqq_only_rf_out_of_time_equity_curve_top_candidates.png",
        "qqq_only_rf_out_of_time_drawdown_curve_top_candidates.png",
        "qqq_only_rf_out_of_time_holdout_returns.png",
        "qqq_only_rf_out_of_time_holdout_drawdowns.png",
        "qqq_only_rf_out_of_time_cost_stability.png",
        "qqq_only_rf_out_of_time_exposure_timeline.png",
    ]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def _out_of_time_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = build_qqq_only_rf_out_of_time_confirmation_specs()
    dates = pd.bdate_range("2019-01-02", "2026-05-15")
    qqq_returns = pd.Series(0.0007, index=dates)
    qqq_returns.loc["2025-03-03":"2025-03-28"] = -0.018
    qqq_returns.loc["2026-02-02":"2026-02-20"] = -0.010

    result_rows: list[dict[str, object]] = []
    daily_parts = []
    drawdown_parts = []
    exposure_parts = []
    for spec in specs:
        exposure = pd.Series(1.0, index=dates)
        if spec.strategy_name != "qqq_buy_hold":
            defensive_exposure = 0.5 if spec.exposure_mode == "partial_50_100" else 0.0
            exposure.loc["2025-03-03":"2025-03-28"] = defensive_exposure
            exposure.loc["2026-02-02":"2026-02-20"] = defensive_exposure
        daily, drawdown, log, row = _strategy_outputs(
            spec=spec,
            qqq_returns=qqq_returns,
            exposure=exposure,
        )
        result_rows.extend(
            build_qqq_only_rf_out_of_time_confirmation_result_rows(
                row,
                spec=spec,
                daily_equity=daily,
                exposure_log=log,
            )
        )
        daily_parts.append(daily)
        drawdown_parts.append(drawdown)
        exposure_parts.append(log)
    return (
        pd.DataFrame(result_rows),
        pd.concat(daily_parts, ignore_index=True),
        pd.concat(drawdown_parts, ignore_index=True),
        pd.concat(exposure_parts, ignore_index=True),
    )


def _strategy_outputs(
    *,
    spec,
    qqq_returns: pd.Series,
    exposure: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    turnover = exposure.diff().abs().fillna(0.0)
    transaction_cost = turnover * spec.transaction_cost_bps / 10000.0
    daily_return = exposure * qqq_returns - transaction_cost
    equity = (1.0 + daily_return).cumprod()
    qqq_equity = (1.0 + qqq_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    qqq_drawdown = qqq_equity / qqq_equity.cummax() - 1.0
    dates = qqq_returns.index
    daily = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "exposure_mode": spec.exposure_mode,
            "rebalance_frequency": spec.rebalance_frequency,
            "transaction_cost_bps": spec.transaction_cost_bps,
            "target_exposure": exposure.to_numpy(),
            "effective_exposure": exposure.to_numpy(),
            "qqq_weight": exposure.to_numpy(),
            "cash_weight": (1.0 - exposure).to_numpy(),
            "qqq_return": qqq_returns.to_numpy(),
            "gross_return": (exposure * qqq_returns).to_numpy(),
            "transaction_cost": transaction_cost.to_numpy(),
            "turnover": turnover.to_numpy(),
            "daily_return": daily_return.to_numpy(),
            "equity": equity.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
        }
    )
    drawdown_frame = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "equity": equity.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
            "qqq_buy_hold_drawdown": qqq_drawdown.to_numpy(),
        }
    )
    log = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "exposure_mode": spec.exposure_mode,
            "rebalance_frequency": spec.rebalance_frequency,
            "target_exposure": exposure.to_numpy(),
            "effective_exposure": exposure.to_numpy(),
            "qqq_weight": exposure.to_numpy(),
            "cash_weight": (1.0 - exposure).to_numpy(),
            "held_assets": [_held_assets(value) for value in exposure],
            "raw_score": np.where(exposure.to_numpy() > 0.0, 0.7, 0.2),
            "threshold": spec.threshold,
        }
    )
    row = {
        "strategy_id": spec.strategy_id,
        "strategy_name": spec.strategy_name,
        "feature_count": 0 if spec.is_baseline else 18,
        "production_ready": False,
    }
    return daily, drawdown_frame, log, row


def _held_assets(exposure: float) -> str:
    if exposure <= 1e-9:
        return "cash"
    if exposure >= 1.0 - 1e-9:
        return "QQQ"
    return "QQQ,cash"
