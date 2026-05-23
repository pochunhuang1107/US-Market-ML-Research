from __future__ import annotations

import pandas as pd

from us_market_ml.long_history_shortlist_validation import (
    COST_SENSITIVITY_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    ROLLING_WINDOW_COLUMNS,
    SELECTED_SYMBOL_COLUMNS,
    SUBPERIOD_COLUMNS,
    SYMBOL_EXPOSURE_COLUMNS,
    build_candidate_daily_equity,
    build_cost_sensitivity,
    build_rolling_window_robustness,
    build_selected_symbol_summary,
    build_subperiod_robustness,
    build_symbol_exposure_summary,
    build_validation_decision_matrix,
    select_shortlist_candidates,
    summarize_rolling_windows,
)


def test_shortlist_candidate_selection_deduplicates_switch_margin_and_includes_important() -> None:
    results = pd.DataFrame(
        [
            _result_row("generic_top", strategy_vs_qqq=2.0, switch_margin=0.0),
            _result_row("generic_dup_a", strategy_vs_qqq=1.5, switch_margin=0.0),
            _result_row("generic_dup_b", strategy_vs_qqq=1.5, switch_margin=0.002),
            _result_row(
                "important_drawdown",
                model_type="ridge",
                training_mode="base_recent_blend",
                target_variant="return_drawdown_score_vs_qqq_63d",
                allocation_rule="top1",
                rebalance_frequency="quarterly",
                min_holding_days=20,
                strategy_vs_qqq=0.2,
                switch_margin=0.0,
            ),
        ]
    )
    decision_matrix = _decision_matrix_for_results(results)

    selected = select_shortlist_candidates(
        results=results,
        decision_matrix=decision_matrix,
        top_n=1,
    )

    assert len(selected) == 2
    assert selected["selection_reason"].str.contains("top_n_deduplicated").any()
    assert selected["selection_reason"].str.contains("important_candidate_drawdown_control").any()
    assert not (
        selected[["model_type", "training_mode", "target_variant", "switch_margin"]]
        .duplicated(subset=["model_type", "training_mode", "target_variant"])
        .any()
    )


def test_candidate_daily_equity_output_shape() -> None:
    candidate = _candidate_series("candidate_0001")
    raw = _market_frame("2020-01-01", periods=10)
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "symbol": ["AAA"],
            "target_weight": [1.0],
        }
    )

    daily, positions = build_candidate_daily_equity(
        raw_df=raw,
        rebalance_weights=weights,
        candidate=candidate,
        start_date="2020-01-01",
        end_date="2020-01-10",
        transaction_cost_bps=10.0,
    )

    assert set(DAILY_EQUITY_COLUMNS).issubset(daily.columns)
    assert len(daily) == 10
    assert set(daily["candidate_id"]) == {"candidate_0001"}
    assert "qqq_equity" in daily.columns
    assert "cash_equity" in daily.columns
    assert not positions.empty


def test_selected_symbol_exposure_summary() -> None:
    candidate = _candidate_series("candidate_0001")
    raw = _market_frame("2020-01-01", periods=10)
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-06")],
            "symbol": ["AAA", "BBB"],
            "target_weight": [1.0, 1.0],
        }
    )
    daily, positions = build_candidate_daily_equity(
        raw_df=raw,
        rebalance_weights=weights,
        candidate=candidate,
        start_date="2020-01-01",
        end_date="2020-01-10",
        transaction_cost_bps=10.0,
    )

    selected = build_selected_symbol_summary(
        candidate=candidate,
        rebalance_weights=weights,
        daily_equity=daily,
        position_returns=positions,
    )
    exposure = build_symbol_exposure_summary(candidate=candidate, selected_symbols=selected)

    assert set(SELECTED_SYMBOL_COLUMNS).issubset(selected.columns)
    assert set(SYMBOL_EXPOSURE_COLUMNS).issubset(exposure.columns)
    assert set(selected["symbol"]) == {"AAA", "BBB"}
    assert exposure["unique_symbol_count"].iloc[0] == 2
    assert exposure["top_symbol_exposure_pct"].iloc[0] > 0.0


def test_subperiod_robustness_output() -> None:
    candidate = _candidate_series("candidate_0001")
    daily = _daily_equity_frame("2020-01-01", periods=12)
    trades = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "candidate_id": ["candidate_0001"],
            "previous_symbols": [""],
            "new_symbols": ["AAA"],
            "selected_symbol": ["AAA"],
            "rank_score": [1.0],
            "turnover": [1.0],
            "estimated_transaction_cost": [0.001],
        }
    )

    subperiods = build_subperiod_robustness(
        candidate=candidate,
        daily_equity=daily,
        trades=trades,
        subperiods=(("sample", "2020-01-01", "2020-12-31"),),
    )

    assert set(SUBPERIOD_COLUMNS).issubset(subperiods.columns)
    assert subperiods["period_name"].tolist() == ["sample"]
    assert subperiods["transaction_count"].iloc[0] == 1
    assert subperiods["top_symbol"].iloc[0] == "AAA"


def test_rolling_window_summary_output() -> None:
    candidate = _candidate_series("candidate_0001")
    daily = _daily_equity_frame("2020-01-01", periods=900)

    rolling = build_rolling_window_robustness(
        candidate=candidate,
        daily_equity=daily,
        start_date="2020-01-01",
        end_date="2022-12-31",
    )
    summary = summarize_rolling_windows(rolling)

    assert set(ROLLING_WINDOW_COLUMNS).issubset(rolling.columns)
    assert not rolling.empty
    assert summary["rolling_window_count"].iloc[0] == len(rolling)
    assert "positive_vs_qqq_window_rate" in summary.columns


def test_cost_sensitivity_output() -> None:
    candidate = _candidate_series("candidate_0001")
    raw = _market_frame("2020-01-01", periods=20)
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "symbol": ["AAA"],
            "target_weight": [1.0],
        }
    )

    cost = build_cost_sensitivity(
        raw_df=raw,
        rebalance_weights=weights,
        candidate=candidate,
        start_date="2020-01-01",
        end_date="2020-01-20",
        cost_sensitivity_bps=(0.0, 5.0, 10.0, 20.0),
    )

    assert set(COST_SENSITIVITY_COLUMNS).issubset(cost.columns)
    assert cost["transaction_cost_bps"].tolist() == [0.0, 5.0, 10.0, 20.0]
    assert cost["decision_at_cost"].notna().all()


def test_validation_decision_logic_robust_and_concentration_reject() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate_series("candidate_0001").to_dict(),
                "selection_reason": "top_n_deduplicated",
                "decision": "best_candidate",
            },
            {
                **_candidate_series("candidate_0002").to_dict(),
                "selection_reason": "top_n_deduplicated",
                "decision": "best_candidate",
            },
        ]
    )
    summary = pd.DataFrame(
        [
            _summary_row("candidate_0001", beats=True),
            _summary_row("candidate_0002", beats=True),
        ]
    )
    rolling_summary = pd.DataFrame(
        [
            _rolling_summary_row("candidate_0001", positive_rate=0.75),
            _rolling_summary_row("candidate_0002", positive_rate=0.75),
        ]
    )
    exposure = pd.DataFrame(
        [
            _exposure_row("candidate_0001", top_exposure=0.50, unique_count=4),
            _exposure_row("candidate_0002", top_exposure=0.95, unique_count=1),
        ]
    )
    cost = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "transaction_cost_bps": 20.0,
                "strategy_vs_qqq": 0.2,
            },
            {
                "candidate_id": "candidate_0002",
                "transaction_cost_bps": 20.0,
                "strategy_vs_qqq": 0.2,
            },
        ]
    )

    matrix = build_validation_decision_matrix(
        candidates=candidates,
        summary=summary,
        symbol_exposure=exposure,
        rolling_summary=rolling_summary,
        cost_sensitivity=cost,
    )

    assert set(DECISION_MATRIX_COLUMNS).issubset(matrix.columns)
    assert matrix.loc[matrix["candidate_id"].eq("candidate_0001"), "model_type"].item() == "ridge"
    assert (
        matrix.loc[matrix["candidate_id"].eq("candidate_0001"), "decision"].item()
        == "robust_candidate"
    )
    assert (
        matrix.loc[matrix["candidate_id"].eq("candidate_0002"), "decision"].item()
        == "reject_deprioritize"
    )
    assert matrix["production_ready"].eq(False).all()


def _result_row(
    run_key: str,
    *,
    model_type: str = "ridge",
    training_mode: str = "frozen_base_1999_2018",
    target_variant: str = "excess_return_vs_qqq_63d",
    allocation_rule: str = "top1",
    rebalance_frequency: str = "quarterly",
    min_holding_days: int = 0,
    switch_margin: float = 0.0,
    strategy_vs_qqq: float = 1.0,
) -> dict[str, object]:
    qqq_return = 1.0
    total_return = qqq_return + strategy_vs_qqq
    max_drawdown = -0.3
    return {
        "run_key": run_key,
        "model_type": model_type,
        "training_mode": training_mode,
        "target_variant": target_variant,
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": allocation_rule,
        "rebalance_frequency": rebalance_frequency,
        "min_holding_days": min_holding_days,
        "switch_margin": switch_margin,
        "transaction_cost_bps": 10.0,
        "recency_decay_profile": pd.NA,
        "base_weight": 0.5 if training_mode == "base_recent_blend" else pd.NA,
        "recent_window_years": 3 if training_mode == "base_recent_blend" else pd.NA,
        "finetune_window_years": pd.NA,
        "finetune_lr_scale": pd.NA,
        "total_return": total_return,
        "annualized_return": 0.2,
        "max_drawdown": max_drawdown,
        "sharpe_like": 0.1,
        "return_to_drawdown": total_return / abs(max_drawdown),
        "transaction_count": 10,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": 0.8,
        "strategy_vs_qqq": strategy_vs_qqq,
        "strategy_vs_spy": total_return - 0.8,
        "qqq_return_capture": total_return / qqq_return,
        "beats_qqq_return": strategy_vs_qqq > 0,
        "beats_qqq_return_to_drawdown": True,
        "drawdown_improvement_vs_qqq": 0.02,
        "qqq_buy_hold_max_drawdown": -0.35,
        "spy_buy_hold_max_drawdown": -0.3,
        "qqq_buy_hold_return_to_drawdown": 2.8,
        "spy_buy_hold_return_to_drawdown": 2.6,
        "decision": "unclassified",
    }


def _decision_matrix_for_results(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"candidate_{index + 1:04d}" for index in range(len(results))],
            "candidate_label": [
                f"candidate {index + 1}" for index in range(len(results))
            ],
            "decision": ["best_candidate"] * len(results),
        }
    )


def _candidate_series(candidate_id: str) -> pd.Series:
    return pd.Series(
        {
            "candidate_id": candidate_id,
            "candidate_label": f"{candidate_id} label",
            "model_type": "ridge",
            "training_mode": "frozen_base_1999_2018",
            "target_variant": "excess_return_vs_qqq_63d",
            "feature_set_version": "long_history_regime_features_v1",
            "allocation_rule": "top1",
            "rebalance_frequency": "quarterly",
            "min_holding_days": 0,
            "switch_margin": 0.0,
            "transaction_cost_bps": 10.0,
            "recency_decay_profile": pd.NA,
            "base_weight": pd.NA,
            "recent_window_years": pd.NA,
            "finetune_window_years": pd.NA,
            "finetune_lr_scale": pd.NA,
        }
    )


def _market_frame(start: str, periods: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, periods=periods, freq="D")
    for index, symbol in enumerate(["QQQ", "SPY", "AAA", "BBB"]):
        base = 100.0 + index * 20.0
        for day, date in enumerate(dates):
            close = base + day * (1.0 + index * 0.2)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _daily_equity_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    strategy_returns = pd.Series([0.002] * periods)
    qqq_returns = pd.Series([0.001] * periods)
    spy_returns = pd.Series([0.0008] * periods)
    equity = (1.0 + strategy_returns).cumprod()
    qqq_equity = (1.0 + qqq_returns).cumprod()
    spy_equity = (1.0 + spy_returns).cumprod()
    return pd.DataFrame(
        {
            "date": dates,
            "candidate_id": "candidate_0001",
            "candidate_label": "candidate_0001 label",
            "strategy_name": "candidate_0001 label",
            "active_symbol": "AAA",
            "active_symbols": "AAA",
            "daily_return": strategy_returns,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "qqq_daily_return": qqq_returns,
            "qqq_equity": qqq_equity,
            "qqq_drawdown": qqq_equity / qqq_equity.cummax() - 1.0,
            "spy_daily_return": spy_returns,
            "spy_equity": spy_equity,
            "spy_drawdown": spy_equity / spy_equity.cummax() - 1.0,
            "cash_daily_return": 0.0,
            "cash_equity": 1.0,
            "cash_drawdown": 0.0,
            "gross_return": strategy_returns,
            "transaction_cost": 0.0,
            "turnover": 0.0,
        }
    )


def _summary_row(candidate_id: str, *, beats: bool) -> dict[str, object]:
    row = _candidate_series(candidate_id).to_dict()
    row.update(
        {
            "total_return": 2.0 if beats else 0.5,
            "annualized_return": 0.2,
            "max_drawdown": -0.3,
            "sharpe_like": 0.1,
            "return_to_drawdown": 6.6 if beats else 1.6,
            "transaction_count": 10,
            "average_turnover": 0.01,
            "estimated_transaction_cost": 0.02,
            "qqq_buy_hold_total_return": 1.0,
            "spy_buy_hold_total_return": 0.8,
            "strategy_vs_qqq": 1.0 if beats else -0.5,
            "strategy_vs_spy": 1.2 if beats else -0.3,
            "qqq_return_capture": 2.0 if beats else 0.5,
            "beats_qqq_return": beats,
            "beats_qqq_return_to_drawdown": beats,
            "drawdown_improvement_vs_qqq": 0.02,
        }
    )
    return row


def _rolling_summary_row(candidate_id: str, *, positive_rate: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rolling_window_count": 10,
        "positive_vs_qqq_window_rate": positive_rate,
        "qqq_underperformance_rate": 1.0 - positive_rate,
        "average_strategy_vs_qqq": 0.2,
        "median_strategy_vs_qqq": 0.1,
        "worst_strategy_vs_qqq": -0.1,
        "drawdown_improvement_window_rate": 0.5,
        "average_drawdown_improvement_vs_qqq": 0.01,
        "worst_window_label": "1y_20200101_20201231",
    }


def _exposure_row(
    candidate_id: str,
    *,
    top_exposure: float,
    unique_count: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": f"{candidate_id} label",
        "top_symbol": "AAA",
        "top_symbol_exposure_pct": top_exposure,
        "top3_symbol_exposure_pct": min(1.0, top_exposure + 0.2),
        "unique_symbol_count": unique_count,
        "qqq_exposure_pct": 0.0,
        "spy_exposure_pct": 0.0,
    }
