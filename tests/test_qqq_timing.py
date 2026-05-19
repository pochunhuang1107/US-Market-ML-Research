from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from us_market_ml.qqq_timing import (
    QQQDefensiveTimingStrategySpec,
    QQQTimingStrategySpec,
    build_qqq_defensive_timing_signal,
    build_qqq_timing_indicators,
    build_qqq_timing_signal,
    calculate_qqq_defensive_timing_subperiod_results,
    default_qqq_defensive_timing_narrow_sweep_specs,
    default_qqq_defensive_timing_sweep_specs,
    default_qqq_timing_strategy_specs,
    run_qqq_defensive_timing_strategy,
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


def test_defensive_sweep_parameter_grid_count_is_focused() -> None:
    specs, warnings, reductions = default_qqq_defensive_timing_sweep_specs(
        {"QQQ", "SPY", "IEF", "TLT", "SHY"}
    )
    timing_specs = [spec for spec in specs if not spec.is_benchmark]
    family_counts = Counter(spec.family for spec in timing_specs)

    assert len(specs) == 492
    assert len(timing_specs) == 489
    assert family_counts == {
        "sma_price_filter_to_cash": 75,
        "sma_crossover_to_cash": 180,
        "sma_price_filter_to_spy": 36,
        "volatility_guard_to_cash": 108,
        "momentum_drawdown_guard_to_cash": 81,
        "regime_score_defensive_to_cash": 9,
    }
    assert warnings == ()
    assert reductions == ()


def test_narrow_defensive_sweep_parameter_grid_count_matches_requested_ranges() -> None:
    specs, warnings = default_qqq_defensive_timing_narrow_sweep_specs(
        {"QQQ", "SPY", "IEF", "TLT", "SHY"}
    )
    timing_specs = [spec for spec in specs if not spec.is_benchmark]
    family_counts = Counter(spec.family for spec in timing_specs)
    names = {spec.strategy_name for spec in timing_specs}

    assert len(specs) == 429
    assert len(timing_specs) == 426
    assert family_counts == {
        "total_return_spy_defensive": 144,
        "drawdown_adjusted_cash_defensive": 90,
        "low_turnover_crossover_cash": 192,
    }
    assert "qqq_sma180_buffer_p0_hold0_to_spy" in names
    assert "qqq_sma200_buffer_p1_hold10_cash" in names
    assert "qqq_sma75_sma200_buffer_m1_hold0_cash" in names
    assert "qqq_sma100_sma200_buffer_m1_hold0_cash" in names
    assert any("buffer_p0p5" in name for name in names)
    assert all(
        spec.fast_sma_days < spec.slow_sma_days
        for spec in timing_specs
        if spec.family == "low_turnover_crossover_cash"
    )
    assert warnings == ()


def test_defensive_sma_price_filter_applies_next_day_without_future_data() -> None:
    qqq = [100.0] * 149 + [120.0, 121.0]
    prices = _price_matrix({"QQQ": qqq, "SPY": np.linspace(100.0, 110.0, len(qqq))})
    spec = QQQDefensiveTimingStrategySpec(
        "qqq_sma150_buffer_p0_hold0_cash",
        family="sma_price_filter_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=150,
        buffer_pct=0.0,
        min_hold_days=0,
    )

    _, daily_equity, _ = run_qqq_defensive_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    assert bool(daily_equity.loc[149, "raw_signal_risk_on"])
    assert not bool(daily_equity.loc[149, "effective_signal_risk_on"])
    assert daily_equity.loc[149, "active_asset"] == "cash"
    assert bool(daily_equity.loc[150, "effective_signal_risk_on"])
    assert daily_equity.loc[150, "active_asset"] == "QQQ"


def test_narrow_sma_price_filter_family_applies_next_day_without_future_data() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 100.0, 120.0, 121.0],
            "SPY": [100.0, 101.0, 102.0, 103.0],
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "qqq_sma3_buffer_p0_hold0_to_spy",
        family="total_return_spy_defensive",
        risk_on_asset="QQQ",
        risk_off_asset="SPY",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=3,
        buffer_pct=0.0,
        min_hold_days=0,
    )

    _, daily_equity, _ = run_qqq_defensive_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    assert bool(daily_equity.loc[2, "raw_signal_risk_on"])
    assert not bool(daily_equity.loc[2, "effective_signal_risk_on"])
    assert daily_equity.loc[2, "active_asset"] == "SPY"
    assert bool(daily_equity.loc[3, "effective_signal_risk_on"])
    assert daily_equity.loc[3, "active_asset"] == "QQQ"


def test_defensive_sma_crossover_applies_next_day_without_future_data() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 100.0, 120.0, 121.0],
            "SPY": [100.0, 100.1, 100.2, 100.3],
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "qqq_sma2_sma3_buffer_p0_hold0_cash",
        family="sma_crossover_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_crossover",
        fast_sma_days=2,
        slow_sma_days=3,
        buffer_pct=0.0,
        min_hold_days=0,
    )

    _, daily_equity, _ = run_qqq_defensive_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    assert bool(daily_equity.loc[2, "raw_signal_risk_on"])
    assert not bool(daily_equity.loc[2, "effective_signal_risk_on"])
    assert daily_equity.loc[2, "active_asset"] == "cash"
    assert bool(daily_equity.loc[3, "effective_signal_risk_on"])
    assert daily_equity.loc[3, "active_asset"] == "QQQ"


def test_narrow_crossover_family_applies_next_day_without_future_data() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 100.0, 120.0, 121.0],
            "SPY": [100.0, 100.1, 100.2, 100.3],
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "qqq_sma2_sma3_buffer_p0_hold0_cash",
        family="low_turnover_crossover_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_crossover",
        fast_sma_days=2,
        slow_sma_days=3,
        buffer_pct=0.0,
        min_hold_days=0,
    )

    _, daily_equity, _ = run_qqq_defensive_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    assert bool(daily_equity.loc[2, "raw_signal_risk_on"])
    assert not bool(daily_equity.loc[2, "effective_signal_risk_on"])
    assert daily_equity.loc[2, "active_asset"] == "cash"
    assert bool(daily_equity.loc[3, "effective_signal_risk_on"])
    assert daily_equity.loc[3, "active_asset"] == "QQQ"


def test_defensive_sma_buffer_requires_extra_clearance() -> None:
    prices = _price_matrix({"QQQ": [100.0, 100.0, 101.0], "SPY": [100.0, 100.0, 100.0]})
    loose_spec = QQQDefensiveTimingStrategySpec(
        "loose",
        family="sma_price_filter_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=3,
        buffer_pct=0.0,
    )
    buffered_spec = QQQDefensiveTimingStrategySpec(
        "buffered",
        family="sma_price_filter_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=3,
        buffer_pct=0.01,
    )

    loose_signal = build_qqq_defensive_timing_signal(prices, loose_spec)
    buffered_signal = build_qqq_defensive_timing_signal(prices, buffered_spec)

    assert bool(loose_signal.iloc[-1])
    assert not bool(buffered_signal.iloc[-1])


def test_defensive_min_hold_delays_signal_flips() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 120.0, 80.0, 130.0, 70.0, 140.0],
            "SPY": [100.0] * 6,
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "hold3",
        family="sma_price_filter_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=2,
        buffer_pct=0.0,
        min_hold_days=3,
    )

    signal = build_qqq_defensive_timing_signal(prices, spec)

    assert signal.tolist() == [False, False, False, True, True, True]


def test_defensive_volatility_percentile_guard_requires_low_realized_volatility() -> None:
    calm_prices = _price_matrix(
        {"QQQ": _volatility_guard_prices(calm_tail=True), "SPY": [100.0] * 850}
    )
    spiky_prices = _price_matrix(
        {"QQQ": _volatility_guard_prices(calm_tail=False), "SPY": [100.0] * 850}
    )
    spec = QQQDefensiveTimingStrategySpec(
        "vol_guard",
        family="volatility_guard_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_volatility_percentile_guard",
        return_lookback_days=63,
        volatility_lookback_days=21,
        volatility_percentile_window_days=504,
        volatility_percentile_threshold=0.80,
    )

    calm_signal = build_qqq_defensive_timing_signal(calm_prices, spec)
    spiky_signal = build_qqq_defensive_timing_signal(spiky_prices, spec)

    assert bool(calm_signal.iloc[-1])
    assert not bool(spiky_signal.iloc[-1])


def test_defensive_drawdown_guard_requires_drawdown_above_threshold() -> None:
    risky_tail = [100.0] * 20 + [130.0] * 20 + list(np.linspace(112.0, 116.0, 20))
    safe_tail = [100.0] * 20 + [130.0] * 20 + list(np.linspace(121.0, 125.0, 20))
    risky_prices = _price_matrix({"QQQ": risky_tail, "SPY": [100.0] * len(risky_tail)})
    safe_prices = _price_matrix({"QQQ": safe_tail, "SPY": [100.0] * len(safe_tail)})
    spec = QQQDefensiveTimingStrategySpec(
        "drawdown_guard",
        family="momentum_drawdown_guard_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_momentum_drawdown_guard_parameterized",
        return_lookback_days=10,
        drawdown_lookback_days=25,
        drawdown_threshold=-0.05,
    )

    risky_signal = build_qqq_defensive_timing_signal(risky_prices, spec)
    safe_signal = build_qqq_defensive_timing_signal(safe_prices, spec)

    assert not bool(risky_signal.iloc[-1])
    assert bool(safe_signal.iloc[-1])


def test_defensive_transaction_costs_are_applied_on_position_changes() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 120.0, 80.0, 130.0, 70.0, 140.0, 60.0],
            "SPY": [100.0] * 7,
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "flip_costs",
        family="sma_price_filter_to_cash",
        risk_on_asset="QQQ",
        risk_off_asset="cash",
        signal_name="qqq_sma_price_filter",
        slow_sma_days=2,
        buffer_pct=0.0,
    )

    metrics, daily_equity, trades = run_qqq_defensive_timing_strategy(
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


def test_defensive_subperiod_metrics_reset_first_period_return() -> None:
    prices = _price_matrix(
        {
            "QQQ": [100.0, 110.0, 99.0, 120.0],
            "SPY": [100.0, 105.0, 105.0, 110.0],
        }
    )
    spec = QQQDefensiveTimingStrategySpec(
        "qqq_buy_hold",
        family="benchmark",
        risk_on_asset="QQQ",
        risk_off_asset="",
        signal_name="buy_hold",
        is_benchmark=True,
    )
    _, daily_equity, _ = run_qqq_defensive_timing_strategy(
        prices,
        spec,
        transaction_cost_bps=0.0,
    )

    subperiods = calculate_qqq_defensive_timing_subperiod_results(
        daily_equity=daily_equity,
        prices=prices,
        subperiods=(
            ("full", "2020-01-01", "2020-01-06"),
            ("tail", "2020-01-02", "2020-01-06"),
        ),
    )

    full = subperiods.loc[subperiods["period_name"].eq("full")].iloc[0]
    tail = subperiods.loc[subperiods["period_name"].eq("tail")].iloc[0]

    assert full["total_return"] == pytest.approx(0.20)
    assert tail["total_return"] == pytest.approx(120.0 / 110.0 - 1.0)
    assert tail["qqq_buy_hold_total_return"] == pytest.approx(120.0 / 110.0 - 1.0)
    assert tail["strategy_vs_qqq"] == pytest.approx(0.0)
    assert tail["transaction_count"] == 0
    assert tail["exposure_rate_to_qqq"] == pytest.approx(1.0)


def _price_matrix(symbol_prices: dict[str, object]) -> pd.DataFrame:
    length = len(next(iter(symbol_prices.values())))
    dates = pd.bdate_range("2020-01-01", periods=length)
    return pd.DataFrame(symbol_prices, index=dates, dtype=float)


def _volatility_guard_prices(*, calm_tail: bool) -> np.ndarray:
    returns = np.empty(850)
    returns[:700] = np.tile([0.025, -0.015], 350)
    if calm_tail:
        returns[700:] = 0.0015
    else:
        returns[700:] = np.tile([0.08, -0.06], 75)
    return 100.0 * np.cumprod(1.0 + returns)
