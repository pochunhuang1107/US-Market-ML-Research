from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.qqq_only_best_candidate_deep_dive import (
    BEST_STRATEGY_NAME,
    COST_SENSITIVITY_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    REQUIRED_REPORT_SECTIONS,
    ROLLING_WINDOW_COLUMNS,
    SUBPERIOD_COLUMNS,
    build_qqq_only_best_candidate_deep_dive_artifacts,
    confirm_held_assets_are_qqq_cash_only,
    output_paths,
    select_best_candidate_rows,
    write_qqq_only_best_candidate_deep_dive_artifacts,
)


def test_selects_best_candidate_top_timing_rows_and_defensive_context() -> None:
    results, _, _, _, _, decision = _synthetic_inputs()

    selected = select_best_candidate_rows(results, decision_matrix=decision, top_n_candidates=2)

    assert selected["strategy_name"].iloc[0] == BEST_STRATEGY_NAME
    assert "qqq_buy_hold" not in set(selected["strategy_name"])
    assert selected["candidate_role"].tolist() == [
        "best_candidate",
        "top_timing_candidate",
        "defensive_context",
    ]
    assert selected["source_candidate_id"].iloc[0] == "candidate_0001"
    assert selected["production_ready"].eq(False).all()


def test_confirm_held_assets_are_qqq_cash_only_accepts_expected_assets() -> None:
    exposure = pd.DataFrame(
        {
            "held_assets": ["QQQ", "cash", "QQQ,cash"],
            "qqq_weight": [1.0, 0.0, 0.5],
            "cash_weight": [0.0, 1.0, 0.5],
        }
    )

    confirm_held_assets_are_qqq_cash_only(exposure)


def test_artifact_builder_outputs_required_shapes_decisions_and_report_sections() -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _synthetic_inputs()

    artifacts = build_qqq_only_best_candidate_deep_dive_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        summary=summary,
        decision_matrix=decision,
        top_n_candidates=2,
    )

    assert set(SUBPERIOD_COLUMNS).issubset(artifacts.subperiods.columns)
    assert set(ROLLING_WINDOW_COLUMNS).issubset(artifacts.rolling_windows.columns)
    assert set(COST_SENSITIVITY_COLUMNS).issubset(artifacts.cost_sensitivity.columns)
    assert set(DECISION_MATRIX_COLUMNS).issubset(artifacts.decision_matrix.columns)
    assert set(artifacts.cost_sensitivity["transaction_cost_bps"]) == {0.0, 5.0, 10.0, 20.0}
    assert artifacts.summary["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert (
        artifacts.decision_matrix.loc[
            artifacts.decision_matrix["candidate_role"].eq("best_candidate"),
            "decision",
        ].item()
        == "keep_for_confirmation"
    )
    assert set(artifacts.exposure_log["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    for section in REQUIRED_REPORT_SECTIONS:
        assert section in artifacts.report_markdown


def test_writer_creates_required_chart_files(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _synthetic_inputs()
    artifacts = build_qqq_only_best_candidate_deep_dive_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        summary=summary,
        decision_matrix=decision,
        top_n_candidates=2,
    )

    paths = write_qqq_only_best_candidate_deep_dive_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    expected = output_paths(tmp_path)
    assert paths.keys() == expected.keys()
    for name in [
        "qqq_only_best_candidate_equity_curve.png",
        "qqq_only_best_candidate_drawdown_curve.png",
        "qqq_only_best_candidate_exposure_timeline.png",
        "qqq_only_best_candidate_rolling_vs_qqq.png",
        "qqq_only_best_candidate_subperiod_returns.png",
        "qqq_only_best_candidate_subperiod_drawdowns.png",
    ]:
        assert expected[name].exists()
        assert expected[name].stat().st_size > 0
    assert (tmp_path / "qqq_only_best_candidate_deep_dive_report.md").exists()


def _synthetic_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.bdate_range("2019-01-02", "2026-05-15")
    qqq_returns = pd.Series(0.001, index=dates)
    qqq_returns.iloc[::50] = -0.003
    qqq_returns.iloc[90:125] = -0.030
    qqq_returns.iloc[760:790] = -0.025
    qqq_returns.iloc[1180:1210] = 0.006

    strategies = [
        (
            "best_id",
            BEST_STRATEGY_NAME,
            "qqq_timing_candidate",
            _exposure_series(dates, [(90, 125), (760, 790)]),
            "random_forest",
        ),
        (
            "second_id",
            "second_candidate",
            "qqq_timing_candidate",
            _exposure_series(dates, [(760, 790)]),
            "lasso",
        ),
        (
            "defensive_id",
            "defensive_reference",
            "defensive_reference",
            _exposure_series(dates, [(90, 150), (760, 830), (1180, 1210)]),
            "elasticnet",
        ),
        (
            "qqq_buy_hold_id",
            "qqq_buy_hold",
            "reject_deprioritize",
            pd.Series(1.0, index=dates),
            "baseline",
        ),
    ]
    daily_parts = []
    drawdown_parts = []
    exposure_parts = []
    result_rows = []
    for strategy_id, strategy_name, label, exposure, model_type in strategies:
        daily, drawdown, log, row = _strategy_outputs(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            decision_label=label,
            model_type=model_type,
            qqq_returns=qqq_returns,
            exposure=exposure,
        )
        daily_parts.append(daily)
        drawdown_parts.append(drawdown)
        exposure_parts.append(log)
        result_rows.append(row)
    results = pd.DataFrame(result_rows)
    decision = pd.DataFrame(
        {
            "candidate_id": ["candidate_0001", "candidate_0002", "candidate_0003"],
            "strategy_id": ["best_id", "second_id", "defensive_id"],
        }
    )
    return (
        results,
        pd.concat(daily_parts, ignore_index=True),
        pd.concat(drawdown_parts, ignore_index=True),
        pd.concat(exposure_parts, ignore_index=True),
        results.copy(),
        decision,
    )


def _exposure_series(dates: pd.DatetimeIndex, cash_slices: list[tuple[int, int]]) -> pd.Series:
    exposure = pd.Series(1.0, index=dates)
    for start, end in cash_slices:
        exposure.iloc[start:end] = 0.0
    return exposure


def _strategy_outputs(
    *,
    strategy_id: str,
    strategy_name: str,
    decision_label: str,
    model_type: str,
    qqq_returns: pd.Series,
    exposure: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    turnover = exposure.diff().abs().fillna(0.0)
    daily_return = exposure * qqq_returns
    equity = (1.0 + daily_return).cumprod()
    qqq_equity = (1.0 + qqq_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    qqq_drawdown = qqq_equity / qqq_equity.cummax() - 1.0
    dates = qqq_returns.index
    daily = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "model_type": model_type,
            "training_mode": "rolling_5y",
            "target_name": "risk_on_20d",
            "feature_set_name": "qqq_self_cross_market",
            "exposure_mode": "binary_0_100",
            "rebalance_frequency": "weekly",
            "transaction_cost_bps": 0.0,
            "target_exposure": exposure.to_numpy(),
            "effective_exposure": exposure.to_numpy(),
            "qqq_weight": exposure.to_numpy(),
            "cash_weight": (1.0 - exposure).to_numpy(),
            "qqq_return": qqq_returns.to_numpy(),
            "gross_return": (exposure * qqq_returns).to_numpy(),
            "transaction_cost": np.zeros(len(dates)),
            "turnover": turnover.to_numpy(),
            "daily_return": daily_return.to_numpy(),
            "equity": equity.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
        }
    )
    drawdown_frame = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "model_type": model_type,
            "training_mode": "rolling_5y",
            "target_name": "risk_on_20d",
            "feature_set_name": "qqq_self_cross_market",
            "equity": equity.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
            "qqq_buy_hold_drawdown": qqq_drawdown.to_numpy(),
        }
    )
    log = pd.DataFrame(
        {
            "date": dates,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "model_type": model_type,
            "training_mode": "rolling_5y",
            "target_name": "risk_on_20d",
            "feature_set_name": "qqq_self_cross_market",
            "exposure_mode": "binary_0_100",
            "rebalance_frequency": "weekly",
            "target_exposure": exposure.to_numpy(),
            "effective_exposure": exposure.to_numpy(),
            "qqq_weight": exposure.to_numpy(),
            "cash_weight": (1.0 - exposure).to_numpy(),
            "held_assets": ["QQQ" if value == 1.0 else "cash" for value in exposure],
            "raw_score": exposure.to_numpy(),
            "threshold": 0.5,
        }
    )
    total = float(equity.iloc[-1] - 1.0)
    qqq_total = float(qqq_equity.iloc[-1] - 1.0)
    max_dd = float(drawdown.min())
    qqq_dd = float(qqq_drawdown.min())
    row = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "model_type": model_type,
        "training_mode": "rolling_5y",
        "target_name": "risk_on_20d",
        "target_horizon": 20,
        "feature_set_name": "qqq_self_cross_market",
        "feature_count": 10,
        "exposure_mode": "binary_0_100",
        "rebalance_frequency": "weekly",
        "transaction_cost_bps": 0.0,
        "threshold": 0.5,
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "total_return": total,
        "annualized_return": total,
        "max_drawdown": max_dd,
        "daily_volatility": float(daily_return.std()),
        "sharpe_like": 0.0,
        "return_to_drawdown": total / abs(max_dd) if max_dd < 0.0 else np.nan,
        "qqq_buy_hold_total_return": qqq_total,
        "qqq_buy_hold_max_drawdown": qqq_dd,
        "qqq_buy_hold_return_to_drawdown": qqq_total / abs(qqq_dd),
        "strategy_vs_qqq": total - qqq_total,
        "drawdown_improvement_vs_qqq": max_dd - qqq_dd,
        "return_to_drawdown_vs_qqq": (total / abs(max_dd)) - (qqq_total / abs(qqq_dd))
        if max_dd < 0.0
        else np.nan,
        "positive_vs_qqq_window_rate": 0.58,
        "qqq_underperformance_rate": 0.40,
        "worst_strategy_vs_qqq": -0.12,
        "average_exposure": float(exposure.mean()),
        "cash_days_pct": float((exposure == 0.0).mean()),
        "trade_count": int((turnover > 0.0).sum()),
        "average_holding_days": 30.0,
        "estimated_transaction_cost": 0.0,
        "production_ready": False,
        "decision_label": decision_label,
        "high_consistency": False,
        "very_high_consistency": False,
        "spy_buy_hold_total_return": qqq_total * 0.7,
        "spy_buy_hold_max_drawdown": qqq_dd * 0.9,
    }
    if decision_label == "defensive_reference":
        row["strategy_vs_qqq"] = -0.05
        row["total_return"] = qqq_total - 0.05
    return daily, drawdown_frame, log, row
