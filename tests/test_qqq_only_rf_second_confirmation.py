from __future__ import annotations

from pathlib import Path

import pandas as pd

from test_qqq_only_best_candidate_deep_dive import _synthetic_inputs

from us_market_ml.qqq_only_rf_second_confirmation import (
    REPORT_SECTIONS,
    RESULT_COLUMNS,
    build_qqq_only_rf_second_confirmation_artifacts,
    build_qqq_only_rf_second_confirmation_specs,
    report_output_paths,
    write_qqq_only_rf_second_confirmation_report_artifacts,
)


def test_second_confirmation_grid_is_locked_and_compact() -> None:
    specs = build_qqq_only_rf_second_confirmation_specs()
    model_specs = [spec for spec in specs if not spec.is_baseline]
    baselines = [spec for spec in specs if spec.is_baseline]

    assert 100 <= len(specs) <= 250
    assert len(baselines) == 2
    assert {spec.baseline_rule for spec in baselines} == {"qqq_buy_hold", "qqq_sma200_timing"}
    assert {spec.model_type for spec in model_specs} == {"random_forest"}
    assert {spec.training_mode for spec in model_specs} == {"rolling_5y"}
    assert {spec.target_name for spec in model_specs} == {"risk_on_20d"}
    assert {spec.feature_set_name for spec in model_specs} == {"qqq_self_cross_market"}
    assert {spec.rebalance_frequency for spec in model_specs} == {"weekly"}
    assert {spec.rebalance_weekday for spec in model_specs} == {"MON", "TUE", "WED", "THU", "FRI"}
    assert {spec.random_state for spec in model_specs}.issuperset(
        {7, 11, 17, 23, 42, 101, 202, 777, 1009, 2026}
    )
    assert {spec.transaction_cost_bps for spec in model_specs} == {0.0, 5.0, 10.0, 20.0, 30.0}
    assert {spec.n_estimators for spec in model_specs}.issuperset({60, 200, 500})
    assert {spec.max_depth for spec in model_specs}.issuperset({None, 4, 6})
    assert {spec.min_samples_leaf for spec in model_specs}.issuperset({5, 20})
    assert set(spec.exposure_mode for spec in model_specs).issubset(
        {"binary_0_100", "partial_50_100"}
    )


def test_second_confirmation_artifacts_write_required_tables_and_charts(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log = _second_confirmation_inputs()

    artifacts = build_qqq_only_rf_second_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )

    assert set(RESULT_COLUMNS).issubset(artifacts.summary.columns)
    assert artifacts.summary["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert "second_confirmation_pass" in set(artifacts.decision_matrix["decision_label"])
    assert set(exposure_log["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert not artifacts.seed_stability.empty
    assert not artifacts.rebalance_day_stability.empty
    assert set(artifacts.cost_stability["transaction_cost_bps"]) == {0.0, 5.0, 10.0, 20.0, 30.0}
    for section in REPORT_SECTIONS:
        assert section in artifacts.report_markdown
    assert "No production signal export" in artifacts.report_markdown
    assert "No model card" in artifacts.report_markdown

    paths = write_qqq_only_rf_second_confirmation_report_artifacts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        output_dir=tmp_path,
    )

    assert paths.keys() == report_output_paths(tmp_path).keys()
    for name in [
        "qqq_only_rf_second_confirmation_summary.csv",
        "qqq_only_rf_second_confirmation_seed_stability.csv",
        "qqq_only_rf_second_confirmation_rebalance_day_stability.csv",
        "qqq_only_rf_second_confirmation_cost_stability.csv",
        "qqq_only_rf_second_confirmation_parameter_stability.csv",
        "qqq_only_rf_second_confirmation_decision_matrix.csv",
        "qqq_only_rf_second_confirmation_report.md",
        "qqq_only_rf_second_confirmation_equity_curve_top_candidates.png",
        "qqq_only_rf_second_confirmation_drawdown_curve_top_candidates.png",
        "qqq_only_rf_second_confirmation_seed_stability.png",
        "qqq_only_rf_second_confirmation_rebalance_day_stability.png",
        "qqq_only_rf_second_confirmation_cost_stability.png",
        "qqq_only_rf_second_confirmation_parameter_stability.png",
    ]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def _second_confirmation_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, daily_equity, daily_drawdown, exposure_log, _, _ = _synthetic_inputs()
    best_row = results.loc[results["strategy_id"].eq("best_id")].iloc[0].to_dict()
    qqq_row = results.loc[results["strategy_name"].eq("qqq_buy_hold")].iloc[0].to_dict()
    configs = [
        ("seed7", "binary_0_100", 0.45, 0, 0.0, "FRI", 7, 60, None, 5),
        ("seed11", "binary_0_100", 0.45, 0, 0.0, "FRI", 11, 60, None, 5),
        ("seed42", "binary_0_100", 0.45, 0, 0.0, "FRI", 42, 60, None, 5),
        ("mon42", "binary_0_100", 0.45, 0, 0.0, "MON", 42, 60, None, 5),
        ("tue42", "binary_0_100", 0.45, 0, 0.0, "TUE", 42, 60, None, 5),
        ("wed42", "binary_0_100", 0.45, 0, 0.0, "WED", 42, 60, None, 5),
        ("thu42", "binary_0_100", 0.45, 0, 0.0, "THU", 42, 60, None, 5),
        ("hold10", "binary_0_100", 0.45, 10, 0.0, "FRI", 42, 60, None, 5),
        ("partial", "partial_50_100", 0.45, 0, 0.0, "FRI", 42, 60, None, 5),
        ("thr50", "binary_0_100", 0.50, 0, 0.0, "FRI", 42, 60, None, 5),
        ("trees200", "binary_0_100", 0.45, 0, 0.0, "FRI", 42, 200, None, 5),
        ("depth4", "binary_0_100", 0.45, 0, 0.0, "FRI", 42, 200, 4, 20),
    ]
    rows = []
    daily_parts = []
    drawdown_parts = []
    exposure_parts = []
    for strategy_id, exposure_mode, threshold, min_hold, cost, weekday, seed, trees, depth, leaf in configs:
        name = (
            "random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_"
            f"{exposure_mode}_weekly_{weekday.lower()}_seed{seed}_trees{trees}_"
            f"depth{'default' if depth is None else depth}_leaf{leaf}_"
            f"thr{threshold:.2f}_hold{min_hold}_cost{cost:g}"
        )
        row = best_row.copy()
        row.update(
            {
                "strategy_id": strategy_id,
                "strategy_name": name,
                "model_type": "random_forest",
                "training_mode": "rolling_5y",
                "target_name": "risk_on_20d",
                "feature_set_name": "qqq_self_cross_market",
                "exposure_mode": exposure_mode,
                "rebalance_frequency": "weekly",
                "rebalance_weekday": weekday,
                "random_state": seed,
                "n_estimators": trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "transaction_cost_bps": cost,
                "threshold": threshold,
                "min_holding_days": min_hold,
                "switch_margin": 0.0,
                "production_ready": False,
            }
        )
        rows.append(row)
        daily_parts.append(_clone_strategy_frame(daily_equity, "best_id", strategy_id, name, exposure_mode, threshold))
        drawdown_parts.append(
            _clone_strategy_frame(daily_drawdown, "best_id", strategy_id, name, exposure_mode, threshold)
        )
        exposure_parts.append(
            _clone_strategy_frame(exposure_log, "best_id", strategy_id, name, exposure_mode, threshold)
        )
    qqq_row.update(
        {
            "rebalance_weekday": "FRI",
            "random_state": 42,
            "n_estimators": 60,
            "max_depth": None,
            "min_samples_leaf": 5,
        }
    )
    rows.append(qqq_row)
    daily_parts.append(daily_equity.loc[daily_equity["strategy_name"].eq("qqq_buy_hold")].copy())
    drawdown_parts.append(daily_drawdown.loc[daily_drawdown["strategy_name"].eq("qqq_buy_hold")].copy())
    exposure_parts.append(exposure_log.loc[exposure_log["strategy_name"].eq("qqq_buy_hold")].copy())

    result_frame = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in result_frame.columns:
            result_frame[column] = pd.NA
    return (
        result_frame,
        pd.concat(daily_parts, ignore_index=True),
        pd.concat(drawdown_parts, ignore_index=True),
        pd.concat(exposure_parts, ignore_index=True),
    )


def _clone_strategy_frame(
    frame: pd.DataFrame,
    source_id: str,
    strategy_id: str,
    strategy_name: str,
    exposure_mode: str,
    threshold: float,
) -> pd.DataFrame:
    out = frame.loc[frame["strategy_id"].eq(source_id)].copy()
    out["strategy_id"] = strategy_id
    out["strategy_name"] = strategy_name
    if "exposure_mode" in out.columns:
        out["exposure_mode"] = exposure_mode
    if "threshold" in out.columns:
        out["threshold"] = threshold
    return out
