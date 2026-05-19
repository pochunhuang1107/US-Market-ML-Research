from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.qqq_timing import (
    QQQTimingStrategySpec,
    build_qqq_timing_indicators,
    build_qqq_timing_signal,
    default_qqq_timing_strategy_specs,
    run_qqq_timing_strategy,
)


def test_sma_timing_signal_applies_next_day_without_future_data() -> None:
    qqq = [100.0] * 199 + [120.0, 121.0]
    spy = [100.0 + index * 0.1 for index in range(len(qqq))]
    prices = _price_matrix({"QQQ": qqq, "SPY": spy})
    indicators = build_qqq_timing_indicators(prices)
    spec = QQQTimingStrategySpec(
        "qqq_sma200_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma200",
    )

    _, daily_equity, _ = run_qqq_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
        indicators=indicators,
    )

    assert bool(daily_equity.loc[199, "raw_signal_risk_on"])
    assert not bool(daily_equity.loc[199, "effective_signal_risk_on"])
    assert daily_equity.loc[199, "active_asset"] == "cash"
    assert bool(daily_equity.loc[200, "effective_signal_risk_on"])
    assert daily_equity.loc[200, "active_asset"] == "QQQ"


def test_momentum_signal_uses_no_future_data() -> None:
    prices = _price_matrix(
        {
            "QQQ": np.linspace(100.0, 150.0, 180),
            "SPY": np.linspace(100.0, 120.0, 180),
        }
    )
    mutated = prices.copy()
    cutoff = prices.index[140]
    mutated.loc[mutated.index > cutoff, "QQQ"] = 10.0

    signal = build_qqq_timing_signal(
        build_qqq_timing_indicators(prices),
        "qqq_return126_positive",
    )
    mutated_signal = build_qqq_timing_signal(
        build_qqq_timing_indicators(mutated),
        "qqq_return126_positive",
    )

    pd.testing.assert_series_equal(signal.loc[:cutoff], mutated_signal.loc[:cutoff])


def test_drawdown_guard_requires_momentum_and_drawdown_limit() -> None:
    qqq = [100.0] * 34 + [130.0] * 50 + [115.0] * 76
    spy = [100.0 + index * 0.01 for index in range(len(qqq))]
    prices = _price_matrix({"QQQ": qqq, "SPY": spy})
    safe_prices = prices.copy()
    safe_prices.iloc[-1, safe_prices.columns.get_loc("QQQ")] = 120.0

    guarded_signal = build_qqq_timing_signal(
        build_qqq_timing_indicators(prices),
        "qqq_momentum_drawdown_guard",
    )
    safe_signal = build_qqq_timing_signal(
        build_qqq_timing_indicators(safe_prices),
        "qqq_momentum_drawdown_guard",
    )

    assert not bool(guarded_signal.iloc[-1])
    assert bool(safe_signal.iloc[-1])


def test_cash_risk_off_has_zero_return_when_signal_stays_off() -> None:
    prices = _price_matrix(
        {
            "QQQ": np.linspace(150.0, 100.0, 180),
            "SPY": np.linspace(100.0, 120.0, 180),
        }
    )
    spec = QQQTimingStrategySpec(
        "qqq_return126_positive_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_return126_positive",
    )

    metrics, daily_equity, _ = run_qqq_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    assert metrics["total_return"] == pytest.approx(0.0)
    assert daily_equity["gross_return"].abs().sum() == pytest.approx(0.0)
    assert set(daily_equity["active_asset"]) == {"cash"}


def test_spy_ief_tlt_risk_off_variants_use_risk_off_asset_returns() -> None:
    prices = _price_matrix(
        {
            "QQQ": np.linspace(150.0, 100.0, 180),
            "SPY": np.linspace(100.0, 130.0, 180),
            "IEF": np.linspace(80.0, 88.0, 180),
            "TLT": np.linspace(90.0, 99.0, 180),
        }
    )
    cases = [
        (
            QQQTimingStrategySpec(
                "qqq_return126_positive_to_spy",
                risk_on_asset="QQQ",
                risk_off_asset="SPY",
                signal_name="qqq_return126_positive",
            ),
            "SPY",
        ),
        (
            QQQTimingStrategySpec(
                "qqq_momentum_drawdown_guard_to_ief",
                risk_on_asset="QQQ",
                risk_off_asset="IEF",
                signal_name="qqq_momentum_drawdown_guard",
            ),
            "IEF",
        ),
        (
            QQQTimingStrategySpec(
                "qqq_momentum_drawdown_guard_to_tlt",
                risk_on_asset="QQQ",
                risk_off_asset="TLT",
                signal_name="qqq_momentum_drawdown_guard",
            ),
            "TLT",
        ),
    ]

    asset_returns = prices.pct_change().fillna(0.0)
    for spec, asset in cases:
        _, daily_equity, _ = run_qqq_timing_strategy(
            prices,
            spec,
            transaction_cost_bps=0.0,
        )
        check_date = prices.index[140]
        row = daily_equity.loc[daily_equity["date"].eq(check_date)].iloc[0]
        assert row["active_asset"] == asset
        assert row["gross_return"] == pytest.approx(asset_returns.loc[check_date, asset])


def test_transaction_costs_are_applied_on_position_changes() -> None:
    qqq = np.concatenate(
        [
            np.linspace(100.0, 130.0, 140),
            np.linspace(129.0, 80.0, 160),
        ]
    )
    prices = _price_matrix(
        {
            "QQQ": qqq,
            "SPY": np.linspace(100.0, 120.0, len(qqq)),
        }
    )
    spec = QQQTimingStrategySpec(
        "qqq_return126_positive_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_return126_positive",
    )

    metrics, daily_equity, trades = run_qqq_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=10.0,
    )

    assert metrics["transaction_count"] >= 2
    assert metrics["estimated_transaction_cost"] > 0.0
    assert not trades.empty
    assert trades["estimated_transaction_cost"].sum() == pytest.approx(
        daily_equity["transaction_cost"].sum()
    )


def test_missing_optional_risk_off_symbol_skips_only_that_strategy() -> None:
    specs, warnings = default_qqq_timing_strategy_specs({"QQQ", "SPY", "TLT"})
    names = {spec.strategy_name for spec in specs}

    assert "qqq_momentum_drawdown_guard_to_ief" not in names
    assert "qqq_momentum_drawdown_guard_to_tlt" in names
    assert any("IEF" in warning for warning in warnings)
    assert not any("TLT" in warning for warning in warnings)


def _price_matrix(symbol_prices: dict[str, object]) -> pd.DataFrame:
    length = len(next(iter(symbol_prices.values())))
    dates = pd.bdate_range("2020-01-01", periods=length)
    return pd.DataFrame(symbol_prices, index=dates, dtype=float)
