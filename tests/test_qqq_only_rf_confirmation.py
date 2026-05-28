from __future__ import annotations

from pathlib import Path

import pandas as pd

from test_qqq_only_best_candidate_deep_dive import _synthetic_inputs

from us_market_ml.qqq_only_rf_confirmation import (
    BEST_STRATEGY_NAME,
    REPORT_SECTIONS,
    RESULT_COLUMNS,
    build_qqq_only_rf_confirmation_artifacts,
    build_qqq_only_rf_confirmation_specs,
    report_output_paths,
    write_qqq_only_rf_confirmation_report_artifacts,
)


def test_confirmation_grid_is_focused_and_includes_exact_prior_best_clone() -> None:
    specs = build_qqq_only_rf_confirmation_specs()
    model_specs = [spec for spec in specs if not spec.is_baseline]
    baselines = [spec for spec in specs if spec.is_baseline]

    assert 80 <= len(specs) <= 180
    assert len(baselines) == 2
    assert {spec.baseline_rule for spec in baselines} == {"qqq_buy_hold", "qqq_sma200_timing"}
    assert {spec.model_type for spec in model_specs} == {"random_forest"}
    assert {spec.training_mode for spec in model_specs} == {
        "rolling_3y",
        "rolling_5y",
        "recency_weighted_expanding",
    }
    assert {spec.target_name for spec in model_specs} == {
        "risk_on_20d",
        "qqq_forward_return_20d",
        "qqq_forward_return_63d",
    }
    assert {spec.feature_set_name for spec in model_specs} == {
        "qqq_self_cross_market",
        "qqq_self_cross_market_reference",
    }
    assert {spec.exposure_mode for spec in model_specs} == {
        "binary_0_100",
        "partial_50_100",
        "ternary_0_50_100",
    }
    assert {spec.rebalance_frequency for spec in model_specs} == {"weekly", "monthly"}
    assert {spec.transaction_cost_bps for spec in model_specs} == {0.0, 5.0, 10.0, 20.0}
    assert {spec.min_holding_days for spec in model_specs} == {0, 5, 10, 20}
    assert {spec.switch_margin for spec in model_specs} == {0.0, 0.002, 0.005}

    exact = [
        spec
        for spec in model_specs
        if spec.strategy_name == BEST_STRATEGY_NAME
        and spec.training_mode == "rolling_5y"
        and spec.target_name == "risk_on_20d"
        and spec.feature_set_name == "qqq_self_cross_market"
        and spec.exposure_mode == "binary_0_100"
        and spec.rebalance_frequency == "weekly"
        and spec.transaction_cost_bps == 0.0
        and spec.threshold == 0.5
        and spec.min_holding_days == 0
        and spec.switch_margin == 0.0
    ]
    assert len(exact) == 1


def test_report_artifacts_apply_confirmation_labels_and_write_required_charts(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log, _, decision = _synthetic_inputs()
    results = _ensure_confirmation_result_columns(results)

    artifacts = build_qqq_only_rf_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        prior_decision_matrix=decision,
    )

    assert artifacts.summary["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert set(exposure_log["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert "confirmed_candidate" in set(artifacts.decision_matrix["decision_label"]) or (
        "robust_confirmation_candidate" in set(artifacts.decision_matrix["decision_label"])
    )
    for section in REPORT_SECTIONS:
        assert section in artifacts.report_markdown

    paths = write_qqq_only_rf_confirmation_report_artifacts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        output_dir=tmp_path,
    )

    assert paths.keys() == report_output_paths(tmp_path).keys()
    assert (tmp_path / "qqq_only_rf_confirmation_summary.csv").exists()
    assert (tmp_path / "qqq_only_rf_confirmation_decision_matrix.csv").exists()
    for name in [
        "qqq_only_rf_confirmation_equity_curve_top_candidates.png",
        "qqq_only_rf_confirmation_drawdown_curve_top_candidates.png",
        "qqq_only_rf_confirmation_exposure_timeline_top_candidates.png",
        "qqq_only_rf_confirmation_rolling_vs_qqq_top_candidates.png",
        "qqq_only_rf_confirmation_cost_sensitivity.png",
        "qqq_only_rf_confirmation_parameter_stability.png",
    ]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def _ensure_confirmation_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    for column in RESULT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["best_strategy_vs_qqq"] = out["strategy_vs_qqq"]
    out["drawdown_improvement_window_rate"] = 0.75
    return out.loc[:, RESULT_COLUMNS]
