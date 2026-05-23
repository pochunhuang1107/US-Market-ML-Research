from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import us_market_ml.long_history_candidate_deep_dive as deep
from us_market_ml.long_history_candidate_deep_dive import (
    COST_SENSITIVITY_COLUMNS,
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    ROLLING_WINDOW_COLUMNS,
    SELECTED_SYMBOL_COLUMNS,
    SUBPERIOD_COLUMNS,
    SYMBOL_EXPOSURE_COLUMNS,
    SYMBOL_PERIOD_COLUMNS,
    TRADE_COLUMNS,
    build_daily_drawdown_rows,
    build_daily_equity_rows,
    build_deep_dive_decision_matrix,
    build_long_history_candidate_deep_dive_artifacts,
    build_selected_symbols_log,
    build_symbol_exposure_rows,
    build_symbol_periods,
    build_trade_rebalance_log,
    select_deep_dive_candidates,
)


def test_select_deep_dive_candidates_defaults_to_candidate_0001_and_0003() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row("candidate_0001"),
            _candidate_row("candidate_0002"),
            _candidate_row("candidate_0003", model_type="ridge"),
        ]
    )
    decision_matrix = pd.DataFrame(
        {
            "candidate_id": ["candidate_0001", "candidate_0002", "candidate_0003"],
            "decision": ["keep_for_deep_dive", "mixed_candidate", "robust_candidate"],
            "rationale": ["one", "two", "three"],
        }
    )

    selected = select_deep_dive_candidates(
        shortlist_candidates=candidates,
        shortlist_decision_matrix=decision_matrix,
    )

    assert selected["candidate_id"].tolist() == ["candidate_0001", "candidate_0003"]
    assert selected["shortlist_decision"].tolist() == [
        "keep_for_deep_dive",
        "robust_candidate",
    ]


def test_daily_equity_and_drawdown_output_shapes() -> None:
    wide = _wide_daily_equity("candidate_0001")

    daily = build_daily_equity_rows(wide)
    drawdown = build_daily_drawdown_rows(wide)

    assert set(DAILY_EQUITY_COLUMNS).issubset(daily.columns)
    assert len(daily) == len(wide) * 4
    assert set(daily["strategy_name"]) == {
        "candidate_strategy",
        "QQQ buy-and-hold",
        "SPY buy-and-hold",
        "cash",
    }
    assert set(DAILY_DRAWDOWN_COLUMNS).issubset(drawdown.columns)
    assert len(drawdown) == len(wide) * 3
    assert "cash" not in set(drawdown["strategy_name"])


def test_trade_rebalance_and_selected_symbol_logs() -> None:
    replay = _fake_replay("candidate_0001")

    selected = build_selected_symbols_log(replay)
    trades = build_trade_rebalance_log(replay)

    assert set(SELECTED_SYMBOL_COLUMNS).issubset(selected.columns)
    assert selected["rank"].tolist() == [1, 1]
    assert selected["predicted_score"].notna().all()
    assert set(TRADE_COLUMNS).issubset(trades.columns)
    assert trades["reason"].tolist() == ["initial_selection", "symbol_change"]
    assert trades["previous_symbol"].tolist() == ["", "AAA"]
    assert trades["new_symbol"].tolist() == ["AAA", "BBB"]


def test_symbol_exposure_and_period_construction() -> None:
    replay = _fake_replay("candidate_0001")

    exposure = build_symbol_exposure_rows(replay.selected_symbols)
    periods = build_symbol_periods(replay.daily_equity)

    assert set(SYMBOL_EXPOSURE_COLUMNS).issubset(exposure.columns)
    assert exposure["symbol"].tolist() == ["AAA", "BBB"]
    assert exposure["is_top_symbol"].tolist() == [True, False]
    assert set(SYMBOL_PERIOD_COLUMNS).issubset(periods.columns)
    assert periods["symbol"].tolist() == ["AAA", "BBB"]
    assert periods["trading_days"].tolist() == [2, 2]
    assert periods["segment_vs_qqq"].notna().all()


def test_artifact_builder_assembles_subperiod_rolling_cost_and_decision_outputs(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate_row("candidate_0001"),
                "shortlist_decision": "keep_for_deep_dive",
            },
            {
                **_candidate_row("candidate_0003", model_type="ridge"),
                "shortlist_decision": "robust_candidate",
            },
        ]
    )
    replay_by_id = {
        "candidate_0001": _fake_replay("candidate_0001"),
        "candidate_0003": _fake_replay("candidate_0003", top_symbol="ARKK"),
    }

    def fake_replay_shortlist_candidate(**kwargs):
        candidate_id = kwargs["candidate"]["candidate_id"]
        return replay_by_id[candidate_id]

    monkeypatch.setattr(deep, "replay_shortlist_candidate", fake_replay_shortlist_candidate)

    artifacts = build_long_history_candidate_deep_dive_artifacts(
        raw_df=pd.DataFrame(),
        selected_candidates=candidates,
    )

    assert set(SUBPERIOD_COLUMNS).issubset(artifacts.subperiods.columns)
    assert set(ROLLING_WINDOW_COLUMNS).issubset(artifacts.rolling_windows.columns)
    assert set(COST_SENSITIVITY_COLUMNS).issubset(artifacts.cost_sensitivity.columns)
    assert set(DECISION_MATRIX_COLUMNS).issubset(artifacts.decision_matrix.columns)
    assert set(artifacts.cost_sensitivity["transaction_cost_bps"]) == {
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
    }
    assert artifacts.decision_matrix["production_ready"].eq(False).all()


def test_deep_dive_decision_logic_targeted_sweep_signal_review_and_reject() -> None:
    summary = pd.DataFrame(
        [
            _summary_row(
                "candidate_0001",
                positive_rate=0.56,
                top_symbol="XLE",
                top_exposure=0.44,
                top3_exposure=0.86,
            ),
            _summary_row(
                "candidate_0003",
                positive_rate=0.62,
                top_symbol="ARKK",
                top_exposure=0.34,
                top3_exposure=0.73,
            ),
            _summary_row(
                "candidate_0009",
                positive_rate=0.30,
                top_symbol="AAA",
                top_exposure=0.20,
                top3_exposure=0.40,
                beats=False,
            ),
        ]
    )
    cost = pd.DataFrame(
        [
            _cost_row("candidate_0001", 20.0, 0.8),
            _cost_row("candidate_0001", 30.0, 0.7),
            _cost_row("candidate_0003", 20.0, 0.6),
            _cost_row("candidate_0003", 30.0, 0.5),
            _cost_row("candidate_0009", 20.0, -0.1),
            _cost_row("candidate_0009", 30.0, -0.2),
        ]
    )
    subperiods = pd.DataFrame(
        [
            _subperiod_row(candidate_id, period_name, edge)
            for candidate_id, edges in {
                "candidate_0001": {
                    "covid_crash_recovery": -0.1,
                    "ai_bull_2023_2024": -0.2,
                    "recent_2025_2026": 0.1,
                },
                "candidate_0003": {
                    "covid_crash_recovery": 0.1,
                    "ai_bull_2023_2024": 0.2,
                    "recent_2025_2026": -0.1,
                },
                "candidate_0009": {
                    "covid_crash_recovery": -0.1,
                    "ai_bull_2023_2024": -0.2,
                    "recent_2025_2026": -0.3,
                },
            }.items()
            for period_name, edge in edges.items()
        ]
    )

    matrix = build_deep_dive_decision_matrix(
        summary=summary,
        subperiods=subperiods,
        cost_sensitivity=cost,
    )

    assert (
        matrix.loc[matrix["candidate_id"].eq("candidate_0001"), "decision"].item()
        == "keep_for_targeted_sweep"
    )
    assert (
        matrix.loc[matrix["candidate_id"].eq("candidate_0003"), "decision"].item()
        == "keep_for_signal_review"
    )
    assert (
        matrix.loc[matrix["candidate_id"].eq("candidate_0009"), "decision"].item()
        == "reject_deprioritize"
    )
    assert matrix["production_ready"].eq(False).all()


def _candidate_row(
    candidate_id: str,
    *,
    model_type: str = "elasticnet",
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
        "decision": "best_candidate",
    }


def _fake_replay(
    candidate_id: str,
    *,
    top_symbol: str = "AAA",
) -> SimpleNamespace:
    candidate = pd.Series(_candidate_row(candidate_id))
    candidate["shortlist_decision"] = (
        "robust_candidate" if candidate_id == "candidate_0003" else "keep_for_deep_dive"
    )
    return SimpleNamespace(
        candidate=candidate,
        metrics=SimpleNamespace(
            total_return=2.0,
            annualized_return=0.2,
            max_drawdown=-0.3,
            sharpe_like=0.1,
            return_to_drawdown=6.7,
            transaction_count=2,
            average_turnover=0.01,
            estimated_transaction_cost=0.02,
            qqq_buy_hold_total_return=1.0,
            spy_buy_hold_total_return=0.8,
            strategy_vs_qqq=1.0,
            strategy_vs_spy=1.2,
            qqq_return_capture=2.0,
            beats_qqq_return=True,
            beats_qqq_return_to_drawdown=True,
            drawdown_improvement_vs_qqq=0.02,
        ),
        daily_equity=_wide_daily_equity(candidate_id),
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
                "symbol": ["AAA", "BBB", "BBB"],
                "prediction": [0.9, 0.1, 0.8],
            }
        ),
        rebalance_weights=pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                "symbol": ["AAA", "BBB"],
                "target_weight": [1.0, 1.0],
            }
        ),
        trades=pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                "candidate_id": [candidate_id, candidate_id],
                "previous_symbols": ["", "AAA"],
                "new_symbols": ["AAA", "BBB"],
                "selected_symbol": ["AAA", "BBB"],
                "rank_score": [0.9, 0.8],
                "turnover": [0.0, 2.0],
                "estimated_transaction_cost": [0.0, 0.002],
            }
        ),
        selected_symbols=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_id,
                    "candidate_label": f"{candidate_id} label",
                    "symbol": top_symbol,
                    "selected_rebalance_count": 2,
                    "exposure_days": 3,
                    "exposure_pct": 0.75,
                    "total_return_contribution": 0.5,
                },
                {
                    "candidate_id": candidate_id,
                    "candidate_label": f"{candidate_id} label",
                    "symbol": "BBB",
                    "selected_rebalance_count": 1,
                    "exposure_days": 1,
                    "exposure_pct": 0.25,
                    "total_return_contribution": 0.1,
                },
            ]
        ),
        symbol_exposure=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_id,
                    "candidate_label": f"{candidate_id} label",
                    "top_symbol": top_symbol,
                    "top_symbol_exposure_pct": 0.34
                    if candidate_id == "candidate_0003"
                    else 0.44,
                    "top3_symbol_exposure_pct": 0.73
                    if candidate_id == "candidate_0003"
                    else 0.86,
                    "unique_symbol_count": 4,
                    "qqq_exposure_pct": 0.0,
                    "spy_exposure_pct": 0.0,
                }
            ]
        ),
        subperiods=pd.DataFrame(
            [
                _subperiod_row(candidate_id, "full", 1.0),
                _subperiod_row(candidate_id, "covid_crash_recovery", -0.1),
                _subperiod_row(candidate_id, "bear_2022", 0.2),
                _subperiod_row(candidate_id, "ai_bull_2023_2024", -0.2),
                _subperiod_row(candidate_id, "recent_2025_2026", 0.1),
            ]
        ),
        rolling_windows=pd.DataFrame(
            [
                _rolling_row(candidate_id, "1y_20200101_20201231", 0.2),
                _rolling_row(candidate_id, "1y_20200401_20210331", -0.1),
                _rolling_row(candidate_id, "2y_20200101_20211231", 0.3),
            ]
        ),
        cost_sensitivity=pd.DataFrame(
            [
                _cost_row(candidate_id, 0.0, 1.1),
                _cost_row(candidate_id, 5.0, 1.05),
                _cost_row(candidate_id, 10.0, 1.0),
                _cost_row(candidate_id, 20.0, 0.9),
                _cost_row(candidate_id, 30.0, 0.8),
            ]
        ),
    )


def _wide_daily_equity(candidate_id: str) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    active = ["AAA", "AAA", "BBB", "BBB"]
    returns = [0.0, 0.02, -0.01, 0.03]
    equity = pd.Series(returns).add(1.0).cumprod()
    qqq_returns = [0.0, 0.01, 0.01, 0.01]
    qqq_equity = pd.Series(qqq_returns).add(1.0).cumprod()
    return pd.DataFrame(
        {
            "date": dates,
            "candidate_id": candidate_id,
            "candidate_label": f"{candidate_id} label",
            "strategy_name": f"{candidate_id} label",
            "active_symbol": active,
            "active_symbols": active,
            "daily_return": returns,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "qqq_daily_return": qqq_returns,
            "qqq_equity": qqq_equity,
            "qqq_drawdown": qqq_equity / qqq_equity.cummax() - 1.0,
            "spy_daily_return": [0.0, 0.005, 0.005, 0.005],
            "spy_equity": [1.0, 1.005, 1.010025, 1.015075125],
            "spy_drawdown": [0.0, 0.0, 0.0, 0.0],
            "cash_daily_return": [0.0, 0.0, 0.0, 0.0],
            "cash_equity": [1.0, 1.0, 1.0, 1.0],
            "cash_drawdown": [0.0, 0.0, 0.0, 0.0],
            "gross_return": returns,
            "transaction_cost": [0.0, 0.0, 0.001, 0.0],
            "turnover": [0.0, 0.0, 1.0, 0.0],
        }
    )


def _summary_row(
    candidate_id: str,
    *,
    positive_rate: float,
    top_symbol: str,
    top_exposure: float,
    top3_exposure: float,
    beats: bool = True,
) -> dict[str, object]:
    row = _candidate_row(candidate_id)
    row.update(
        {
            "total_return": 2.0 if beats else 0.5,
            "annualized_return": 0.2,
            "max_drawdown": -0.3,
            "sharpe_like": 0.1,
            "return_to_drawdown": 6.7,
            "transaction_count": 10,
            "average_turnover": 0.01,
            "estimated_transaction_cost": 0.02,
            "qqq_buy_hold_total_return": 1.0,
            "spy_buy_hold_total_return": 0.8,
            "strategy_vs_qqq": 1.0 if beats else -0.5,
            "strategy_vs_spy": 1.2,
            "qqq_return_capture": 2.0,
            "beats_qqq_return": beats,
            "beats_qqq_return_to_drawdown": beats,
            "drawdown_improvement_vs_qqq": 0.02,
            "rolling_window_count": 57,
            "positive_vs_qqq_window_rate": positive_rate,
            "qqq_underperformance_rate": 1.0 - positive_rate,
            "average_strategy_vs_qqq": 0.1,
            "worst_strategy_vs_qqq": -0.2,
            "top_symbol": top_symbol,
            "top_symbol_exposure_pct": top_exposure,
            "top3_symbol_exposure_pct": top3_exposure,
            "unique_symbol_count": 4,
            "shortlist_decision": "keep_for_deep_dive",
        }
    )
    return row


def _rolling_row(candidate_id: str, label: str, edge: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "window_label": label,
        "window_start": "2020-01-01",
        "window_end": "2020-12-31",
        "strategy_total_return": 0.3,
        "qqq_total_return": 0.3 - edge,
        "strategy_vs_qqq": edge,
        "strategy_max_drawdown": -0.2,
        "qqq_max_drawdown": -0.25,
        "drawdown_improvement_vs_qqq": 0.05,
        "underperformed_qqq": edge < 0.0,
        "materially_improved_drawdown": True,
    }


def _subperiod_row(candidate_id: str, period_name: str, edge: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "period_name": period_name,
        "period_start": "2020-01-01",
        "period_end": "2020-12-31",
        "total_return": 0.3,
        "qqq_total_return": 0.3 - edge,
        "spy_total_return": 0.2,
        "strategy_vs_qqq": edge,
        "max_drawdown": -0.2,
        "qqq_max_drawdown": -0.25,
        "drawdown_improvement_vs_qqq": 0.05,
        "transaction_count": 1,
        "top_symbol": "AAA",
    }


def _cost_row(candidate_id: str, cost_bps: float, edge: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "transaction_cost_bps": cost_bps,
        "total_return": 1.0 + edge,
        "strategy_vs_qqq": edge,
        "max_drawdown": -0.3,
        "return_to_drawdown": 6.0,
        "decision_at_cost": "keep_for_deep_dive" if edge > 0 else "reject_deprioritize",
    }
