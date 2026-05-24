from __future__ import annotations

import pandas as pd

from us_market_ml.long_history_spy_core_sleeve import (
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    RESULT_COLUMNS,
    ROLLING_WINDOW_COLUMNS,
    SUBPERIOD_COLUMNS,
    build_core_sleeve_daily_equity,
    build_long_history_spy_core_sleeve_artifacts,
    build_result_row,
    build_rolling_window_rows,
    build_spy_core_sleeve_decision_matrix,
    build_subperiod_rows,
    classify_core_sleeve_decision,
    extract_candidate_reference_returns,
    generate_core_sleeve_weight_grid,
)


def test_core_sleeve_weight_grid_generation() -> None:
    grid = generate_core_sleeve_weight_grid()

    assert grid["spy_core_weight"].tolist() == [
        0.0,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
    ]
    assert grid["candidate_sleeve_weight"].round(10).tolist() == [
        1.0,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.0,
    ]
    assert (grid.sum(axis=1).round(10) == 1.0).all()


def test_portfolio_daily_return_calculation() -> None:
    aligned = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "candidate_daily_return": [0.0, 0.10],
            "candidate_equity": [1.0, 1.10],
            "candidate_drawdown": [0.0, 0.0],
            "candidate_turnover": [0.0, 1.0],
            "candidate_transaction_cost": [0.0, 0.001],
            "qqq_daily_return": [0.0, 0.03],
            "qqq_equity": [1.0, 1.03],
            "qqq_drawdown": [0.0, 0.0],
            "spy_daily_return": [0.0, 0.02],
            "spy_equity": [1.0, 1.02],
            "spy_drawdown": [0.0, 0.0],
        }
    )

    daily = build_core_sleeve_daily_equity(
        aligned_returns=aligned,
        candidate=pd.Series(_source_row("candidate_0001")),
        spy_core_weight=0.4,
    )

    assert set(DAILY_EQUITY_COLUMNS).issubset(daily.columns)
    assert daily["daily_return"].round(6).tolist() == [0.0, 0.068]
    assert daily["equity"].round(6).tolist() == [1.0, 1.068]
    assert daily["rebalance_policy"].eq("daily_rebalanced_fixed_weights").all()


def test_qqq_spy_reference_extraction_from_daily_equity() -> None:
    daily_equity = _daily_equity_frame("2020-01-01", "2020-01-05")

    extracted = extract_candidate_reference_returns(
        daily_equity,
        "candidate_0001",
        start_date="2020-01-01",
        end_date="2020-01-05",
    )

    assert {
        "candidate_daily_return",
        "qqq_daily_return",
        "spy_daily_return",
        "candidate_turnover",
        "candidate_transaction_cost",
    }.issubset(extracted.columns)
    assert len(extracted) == 5
    assert extracted["qqq_daily_return"].notna().all()
    assert extracted["spy_daily_return"].notna().all()


def test_summary_metrics_subperiod_and_rolling_outputs() -> None:
    daily_equity = _daily_equity_frame("2020-01-01", "2023-12-31")
    daily_drawdown = daily_equity.loc[
        ~daily_equity["strategy_name"].eq("cash"),
        ["date", "candidate_id", "candidate_label", "strategy_name", "equity", "drawdown"],
    ].copy()
    summary = pd.DataFrame(
        [
            _source_row("candidate_0001", model_type="elasticnet"),
            _source_row("candidate_0003", model_type="ridge"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision_row("candidate_0001", 0.45, 0.40),
            _decision_row("candidate_0003", 0.30, 0.25),
        ]
    )

    artifacts = build_long_history_spy_core_sleeve_artifacts(
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        deep_dive_summary=summary,
        deep_dive_decision_matrix=decisions,
        candidate_ids=("candidate_0001", "candidate_0003"),
        start_date="2020-01-01",
        end_date="2023-12-31",
    )

    assert set(RESULT_COLUMNS).issubset(artifacts.results.columns)
    assert set(DAILY_EQUITY_COLUMNS).issubset(artifacts.daily_equity.columns)
    assert set(DAILY_DRAWDOWN_COLUMNS).issubset(artifacts.daily_drawdown.columns)
    assert set(SUBPERIOD_COLUMNS).issubset(artifacts.subperiods.columns)
    assert set(ROLLING_WINDOW_COLUMNS).issubset(artifacts.rolling_windows.columns)
    assert set(DECISION_MATRIX_COLUMNS).issubset(artifacts.decision_matrix.columns)
    assert len(artifacts.results) == 18
    assert not artifacts.subperiods.empty
    assert not artifacts.rolling_windows.empty
    assert artifacts.decision_matrix["production_ready"].eq(False).all()


def test_result_row_includes_summary_metrics_and_inherited_cost_estimates() -> None:
    aligned = extract_candidate_reference_returns(
        _daily_equity_frame("2020-01-01", "2020-12-31"),
        "candidate_0001",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    candidate = pd.Series(_source_row("candidate_0001"))
    candidate["source_cost_20bps_strategy_vs_qqq"] = 0.30
    candidate["source_cost_30bps_strategy_vs_qqq"] = 0.20
    daily = build_core_sleeve_daily_equity(
        aligned_returns=aligned,
        candidate=candidate,
        spy_core_weight=0.5,
    )
    rolling = build_rolling_window_rows(
        daily_equity=daily,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    result = build_result_row(
        daily_equity=daily,
        candidate=candidate,
        rolling_summary=rolling.iloc[0:0],
    )
    subperiods = build_subperiod_rows(daily_equity=daily)
    decision = build_spy_core_sleeve_decision_matrix(result)

    assert result["total_return"].notna().all()
    assert result["qqq_buy_hold_total_return"].notna().all()
    assert result["spy_buy_hold_total_return"].notna().all()
    assert result["cost_20bps_strategy_vs_qqq"].notna().all()
    assert not subperiods.empty
    assert decision["production_ready"].eq(False).all()


def test_decision_logic_classifies_required_categories() -> None:
    best = pd.Series(_decision_payload(beats=True, capture=1.1, dd=0.03, rtd=1.0))
    balanced = pd.Series(_decision_payload(beats=False, capture=0.95, dd=0.03, rtd=1.0))
    return_only = pd.Series(_decision_payload(beats=True, capture=1.1, dd=-0.01, rtd=1.0))
    defensive = pd.Series(_decision_payload(beats=False, capture=0.8, dd=0.05, rtd=1.0))
    reject = pd.Series(_decision_payload(beats=False, capture=0.5, dd=-0.01, rtd=-1.0))

    assert classify_core_sleeve_decision(best)[0] == "best_core_sleeve_candidate"
    assert classify_core_sleeve_decision(balanced)[0] == "balanced_core_sleeve_candidate"
    assert classify_core_sleeve_decision(return_only)[0] == "return_candidate"
    assert classify_core_sleeve_decision(defensive)[0] == "defensive_candidate"
    assert classify_core_sleeve_decision(reject)[0] == "reject_deprioritize"


def _daily_equity_frame(start: str, end: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, end, freq="D")
    for candidate_id in ("candidate_0001", "candidate_0003"):
        candidate_drift = 0.0018 if candidate_id == "candidate_0001" else 0.0015
        candidate_returns = _return_series(len(dates), candidate_drift)
        qqq_returns = _return_series(len(dates), 0.0012)
        spy_returns = _return_series(len(dates), 0.0008, shock=-0.012)
        cash_returns = [0.0] * len(dates)
        specs = [
            ("candidate_strategy", candidate_returns),
            ("QQQ buy-and-hold", qqq_returns),
            ("SPY buy-and-hold", spy_returns),
            ("cash", cash_returns),
        ]
        for strategy_name, returns in specs:
            equity = pd.Series(returns).add(1.0).cumprod()
            drawdown = equity / equity.cummax() - 1.0
            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "candidate_id": candidate_id,
                        "candidate_label": f"{candidate_id} label",
                        "strategy_name": strategy_name,
                        "active_symbol": "AAA" if strategy_name == "candidate_strategy" else "",
                        "active_symbols": "AAA" if strategy_name == "candidate_strategy" else "",
                        "daily_return": returns[idx],
                        "equity": equity.iloc[idx],
                        "drawdown": drawdown.iloc[idx],
                        "gross_return": returns[idx],
                        "transaction_cost": 0.001
                        if strategy_name == "candidate_strategy" and idx % 90 == 0
                        else 0.0,
                        "turnover": 1.0
                        if strategy_name == "candidate_strategy" and idx % 90 == 0
                        else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _return_series(length: int, drift: float, shock: float = -0.02) -> list[float]:
    return [0.0 if idx == 0 else (shock if idx % 45 == 0 else drift) for idx in range(length)]


def _source_row(candidate_id: str, model_type: str = "elasticnet") -> dict[str, object]:
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
        "total_return": 1.0,
        "max_drawdown": -0.3,
        "return_to_drawdown": 3.3,
        "transaction_count": 5,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 0.8,
        "spy_buy_hold_total_return": 0.6,
        "strategy_vs_qqq": 0.2,
        "strategy_vs_spy": 0.4,
        "positive_vs_qqq_window_rate": 0.6,
        "qqq_underperformance_rate": 0.4,
        "worst_strategy_vs_qqq": -0.1,
        "top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "shortlist_decision": "keep_for_deep_dive",
    }


def _decision_row(candidate_id: str, cost20: float, cost30: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "decision": "keep_for_targeted_sweep",
        "rationale": "research only",
        "cost_20bps_strategy_vs_qqq": cost20,
        "cost_30bps_strategy_vs_qqq": cost30,
    }


def _decision_payload(
    *,
    beats: bool,
    capture: float,
    dd: float,
    rtd: float,
) -> dict[str, object]:
    return {
        "beats_qqq_return": beats,
        "captures_90pct_qqq_return": capture >= 0.9,
        "drawdown_improvement_vs_qqq": dd,
        "improves_drawdown_vs_qqq": dd > 0.0,
        "return_to_drawdown_vs_qqq": rtd,
        "improves_return_to_drawdown_vs_qqq": rtd > 0.0,
        "is_core_sleeve": True,
    }
