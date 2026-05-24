from __future__ import annotations

import pandas as pd

from us_market_ml.long_history_concentration_control import (
    COST_SENSITIVITY_COLUMNS,
    RESULT_COLUMNS,
    ROLLING_WINDOW_COLUMNS,
    SELECTED_SYMBOL_COLUMNS,
    SUBPERIOD_COLUMNS,
    SYMBOL_EXPOSURE_COLUMNS,
    build_concentration_control_candidates,
    build_control_rebalance_weights,
    build_control_selected_symbols_log,
    build_symbol_exposure_rows,
    build_weighted_symbol_exposure_summary,
    classify_concentration_control_decision,
    replay_concentration_control_candidate,
    select_concentration_control_source_candidates,
)


def test_control_config_generation_count_for_two_default_sources() -> None:
    source = _source_candidates()

    controls = build_concentration_control_candidates(source)

    assert len(controls) == 34
    assert controls.groupby("candidate_source_id").size().to_dict() == {
        "candidate_0001": 17,
        "candidate_0003": 17,
    }
    assert "baseline_clone" in set(controls["concentration_rule"])
    assert "top3_score_weighted_cap050" in set(controls["concentration_rule"])


def test_source_candidate_selection_uses_deep_dive_decision_metadata() -> None:
    summary = _source_candidates().drop(columns=["source_decision"])
    decisions = pd.DataFrame(
        {
            "candidate_id": ["candidate_0001", "candidate_0003"],
            "decision": ["keep_for_targeted_sweep", "keep_for_signal_review"],
            "rationale": ["one", "three"],
        }
    )

    selected = select_concentration_control_source_candidates(
        deep_dive_summary=summary,
        deep_dive_decision_matrix=decisions,
    )

    assert selected["candidate_id"].tolist() == ["candidate_0001", "candidate_0003"]
    assert selected["source_decision"].tolist() == [
        "keep_for_targeted_sweep",
        "keep_for_signal_review",
    ]


def test_max_consecutive_rebalance_rule_uses_only_prior_choices() -> None:
    predictions = _repeated_rankings()
    candidate = _control_candidate(
        "top1_max_consecutive_rebalances",
        max_consecutive_rebalances=2,
    )

    weights = build_control_rebalance_weights(
        predictions=predictions,
        candidate=candidate,
        eval_dates=pd.date_range("2020-01-01", "2020-01-10", freq="D"),
    )
    perturbed_future = predictions.copy()
    perturbed_future.loc[
        perturbed_future["date"].eq(pd.Timestamp("2020-01-10"))
        & perturbed_future["symbol"].eq("BBB"),
        "prediction",
    ] = 2.0
    perturbed_weights = build_control_rebalance_weights(
        predictions=perturbed_future,
        candidate=candidate,
        eval_dates=pd.date_range("2020-01-01", "2020-01-10", freq="D"),
    )

    assert weights["symbol"].tolist() == ["AAA", "AAA", "BBB", "AAA"]
    assert perturbed_weights["symbol"].head(3).tolist() == weights["symbol"].head(3).tolist()


def test_trailing_exposure_cap_rule_uses_only_realized_prior_exposure() -> None:
    predictions = _repeated_rankings(symbols=("AAA", "BBB", "CCC"))
    candidate = _control_candidate(
        "top1_trailing_exposure_cap",
        trailing_exposure_cap=0.35,
    )

    weights = build_control_rebalance_weights(
        predictions=predictions,
        candidate=candidate,
        eval_dates=pd.date_range("2020-01-01", "2020-01-10", freq="D"),
    )
    perturbed_future = predictions.copy()
    perturbed_future.loc[
        perturbed_future["date"].eq(pd.Timestamp("2020-01-10"))
        & perturbed_future["symbol"].eq("CCC"),
        "prediction",
    ] = 2.0
    perturbed_weights = build_control_rebalance_weights(
        predictions=perturbed_future,
        candidate=candidate,
        eval_dates=pd.date_range("2020-01-01", "2020-01-10", freq="D"),
    )

    assert weights["symbol"].tolist() == ["AAA", "BBB", "CCC", "AAA"]
    assert perturbed_weights["symbol"].head(3).tolist() == weights["symbol"].head(3).tolist()


def test_topk_and_weighted_controls_sum_to_one() -> None:
    predictions = _one_date_rankings()
    cases = {
        "top2_equal_weight": [0.5, 0.5],
        "top3_equal_weight": [1 / 3, 1 / 3, 1 / 3],
        "top3_rank_weighted": [0.5, 0.3, 0.2],
        "top3_score_weighted_cap050": None,
    }

    for rule, expected in cases.items():
        weights = build_control_rebalance_weights(
            predictions=predictions,
            candidate=_control_candidate(rule),
            eval_dates=pd.date_range("2020-01-01", periods=1),
        )

        assert abs(weights["target_weight"].sum() - 1.0) < 1.0e-12
        if expected is not None:
            assert weights["target_weight"].round(10).tolist() == [
                round(value, 10) for value in expected
            ]
        if rule == "top3_score_weighted_cap050":
            assert weights["target_weight"].max() <= 0.5 + 1.0e-12


def test_exclusion_sanity_controls_exclude_expected_symbols() -> None:
    controls = build_concentration_control_candidates(_source_candidates())
    predictions = pd.DataFrame(
        [
            {"date": "2020-01-01", "symbol": "XLE", "prediction": 1.0},
            {"date": "2020-01-01", "symbol": "XLU", "prediction": 0.9},
            {"date": "2020-01-01", "symbol": "VUG", "prediction": 0.8},
            {"date": "2020-01-01", "symbol": "AAA", "prediction": 0.7},
        ]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    top_symbol = controls.loc[
        controls["candidate_source_id"].eq("candidate_0001")
        & controls["concentration_rule"].eq("top1_exclude_top_symbol_sanity")
    ].iloc[0]
    top3 = controls.loc[
        controls["candidate_source_id"].eq("candidate_0001")
        & controls["concentration_rule"].eq("top1_exclude_top3_symbols_sanity")
    ].iloc[0]

    top_symbol_weights = build_control_rebalance_weights(
        predictions=predictions,
        candidate=top_symbol,
        eval_dates=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]),
    )
    top3_weights = build_control_rebalance_weights(
        predictions=predictions,
        candidate=top3,
        eval_dates=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]),
    )

    assert top_symbol_weights["symbol"].tolist() == ["XLU"]
    assert top3_weights["symbol"].tolist() == ["AAA"]


def test_selected_symbol_and_weighted_exposure_outputs() -> None:
    candidate = _control_candidate("top3_rank_weighted")
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "target_weight": [0.5, 0.5, 1.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "prediction": [0.9, 0.8, 1.0],
        }
    )
    daily = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4)})
    positions = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "lagged_target_weight": [0.0, 0.0, 0.5, 0.5],
            "position_return": [0.0, 0.0, 0.01, 0.02],
        }
    )

    selected = build_control_selected_symbols_log(
        candidate=candidate,
        weights=weights,
        predictions=predictions,
    )
    exposure = build_symbol_exposure_rows(
        build_weighted_symbol_exposure_summary(
            candidate=candidate,
            weights=weights,
            daily_equity=daily,
            position_returns=positions,
        )
    )

    assert set(SELECTED_SYMBOL_COLUMNS).issubset(selected.columns)
    assert set(SYMBOL_EXPOSURE_COLUMNS).issubset(exposure.columns)
    assert exposure["exposure_pct"].sum() <= 1.0
    assert exposure["is_top_symbol"].sum() == 1


def test_replay_outputs_subperiod_rolling_and_cost_sensitivity() -> None:
    raw = _market_frame("2020-01-01", "2022-12-31", symbols=("QQQ", "SPY", "AAA", "BBB"))
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "prediction": [0.9, 0.8],
        }
    )
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "symbol": ["AAA"],
            "target_weight": [1.0],
        }
    )

    replay = replay_concentration_control_candidate(
        raw_df=raw,
        predictions=predictions,
        weights=weights,
        candidate=_control_candidate("baseline_clone"),
        start_date="2020-01-01",
        end_date="2022-12-31",
        cost_sensitivity_bps=(10.0, 20.0, 30.0),
    )

    assert set(SUBPERIOD_COLUMNS).issubset(replay.subperiods.columns)
    assert set(ROLLING_WINDOW_COLUMNS).issubset(replay.rolling_windows.columns)
    assert set(COST_SENSITIVITY_COLUMNS).issubset(replay.cost_sensitivity.columns)
    assert not replay.rolling_windows.empty
    assert replay.cost_sensitivity["transaction_cost_bps"].tolist() == [10.0, 20.0, 30.0]


def test_decision_logic_classifies_keep_robust_mixed_and_reject() -> None:
    keep = pd.Series(_result_row(strategy_vs_qqq=0.8, positive_rate=0.56, top=0.25, top3=0.55))
    robust = pd.Series(_result_row(strategy_vs_qqq=0.8, positive_rate=0.70, top=0.50, top3=0.75))
    mixed = pd.Series(_result_row(strategy_vs_qqq=0.2, positive_rate=0.50, top=0.25, top3=0.55))
    reject = pd.Series(_result_row(strategy_vs_qqq=-0.1, positive_rate=0.70, top=0.25, top3=0.55))

    assert classify_concentration_control_decision(keep)[0] == "keep_for_deep_dive"
    assert classify_concentration_control_decision(robust)[0] == "robust_candidate"
    assert classify_concentration_control_decision(mixed)[0] == "concentration_improved_mixed"
    assert classify_concentration_control_decision(reject)[0] == "reject_deprioritize"


def test_result_columns_include_required_metrics() -> None:
    assert {
        "candidate_source_id",
        "concentration_rule",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "return_to_drawdown",
        "transaction_count",
        "average_turnover",
        "estimated_transaction_cost",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "strategy_vs_qqq",
        "strategy_vs_spy",
        "qqq_return_capture",
        "beats_qqq_return",
        "beats_qqq_return_to_drawdown",
        "drawdown_improvement_vs_qqq",
        "positive_vs_qqq_window_rate",
        "qqq_underperformance_rate",
        "worst_strategy_vs_qqq",
        "drawdown_improvement_window_rate",
        "top_symbol",
        "top_symbol_exposure_pct",
        "top3_symbol_exposure_pct",
        "unique_symbol_count",
        "cost_20bps_strategy_vs_qqq",
        "cost_30bps_strategy_vs_qqq",
    }.issubset(RESULT_COLUMNS)


def _source_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _source_row("candidate_0001", "elasticnet", "XLE", 0.44, 0.86, 4),
            _source_row("candidate_0003", "ridge", "ARKK", 0.34, 0.73, 8),
        ]
    )


def _source_row(
    candidate_id: str,
    model_type: str,
    top_symbol: str,
    top_exposure: float,
    top3_exposure: float,
    unique_count: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": f"{candidate_id} label",
        "model_type": model_type,
        "training_mode": "recency_weighted_expanding"
        if model_type == "elasticnet"
        else "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_126d"
        if model_type == "elasticnet"
        else "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20 if model_type == "elasticnet" else 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "recency_decay_profile": "medium_decay" if model_type == "elasticnet" else pd.NA,
        "base_weight": pd.NA,
        "recent_window_years": pd.NA,
        "finetune_window_years": pd.NA,
        "finetune_lr_scale": pd.NA,
        "total_return": 6.0,
        "strategy_vs_qqq": 2.0,
        "return_to_drawdown": 12.0,
        "positive_vs_qqq_window_rate": 0.56,
        "worst_strategy_vs_qqq": -0.4,
        "top_symbol": top_symbol,
        "top_symbol_exposure_pct": top_exposure,
        "top3_symbol_exposure_pct": top3_exposure,
        "unique_symbol_count": unique_count,
        "source_decision": "keep_for_targeted_sweep",
    }


def _control_candidate(rule: str, **overrides: object) -> pd.Series:
    payload = {
        **_source_row("candidate_0001", "elasticnet", "XLE", 0.44, 0.86, 4),
        "candidate_id": "candidate_0001_cc_test",
        "candidate_source_id": "candidate_0001",
        "concentration_rule": rule,
        "concentration_rule_family": rule,
        "control_value": "",
        "excluded_symbols": "",
        "diagnostic_control": False,
        "allocation_rule": rule if rule.startswith("top3") or rule.startswith("top2") else "top1",
        "source_strategy_vs_qqq": 2.0,
        "source_return_to_drawdown": 12.0,
        "source_positive_vs_qqq_window_rate": 0.56,
        "source_worst_strategy_vs_qqq": -0.4,
        "source_top_symbol": "XLE",
        "source_top_symbol_exposure_pct": 0.44,
        "source_top3_symbol_exposure_pct": 0.86,
        "source_unique_symbol_count": 4,
    }
    payload.update(overrides)
    return pd.Series(payload)


def _one_date_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "prediction": [3.0, 2.0, 1.0],
        }
    )


def _repeated_rankings(symbols: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2020-01-01", "2020-01-04", "2020-01-07", "2020-01-10"])
    for date in dates:
        for rank, symbol in enumerate(symbols):
            rows.append({"date": date, "symbol": symbol, "prediction": 1.0 - rank * 0.1})
    return pd.DataFrame(rows)


def _market_frame(start: str, end: str, symbols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, end, freq="D")
    for symbol_index, symbol in enumerate(symbols):
        for day, date in enumerate(dates):
            close = 100.0 + symbol_index * 10.0 + day * (1.0 + symbol_index * 0.05)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _result_row(
    *,
    strategy_vs_qqq: float,
    positive_rate: float,
    top: float,
    top3: float,
) -> dict[str, object]:
    return {
        "concentration_rule": "top3_equal_weight",
        "strategy_vs_qqq": strategy_vs_qqq,
        "beats_qqq_return": strategy_vs_qqq > 0,
        "beats_qqq_return_to_drawdown": strategy_vs_qqq > 0,
        "cost_20bps_strategy_vs_qqq": strategy_vs_qqq - 0.01,
        "cost_30bps_strategy_vs_qqq": strategy_vs_qqq - 0.02,
        "source_strategy_vs_qqq": 1.0,
        "source_positive_vs_qqq_window_rate": 0.56,
        "positive_vs_qqq_window_rate": positive_rate,
        "source_top_symbol_exposure_pct": 0.50,
        "source_top3_symbol_exposure_pct": 0.80,
        "source_unique_symbol_count": 4,
        "top_symbol_exposure_pct": top,
        "top3_symbol_exposure_pct": top3,
        "unique_symbol_count": 6,
    }
