"""Compact leakage-safe QQQ timing baseline research utilities.

Signals in this module are calculated from close data available on date T and
applied to the close-to-close return on the next trading day. The implementation
is intentionally local and deterministic: it does not train models, read cloud
storage, export production signals, or add feature-set versions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown

REQUIRED_SYMBOLS: tuple[str, ...] = ("QQQ", "SPY")
OPTIONAL_RISK_OFF_SYMBOLS: tuple[str, ...] = ("IEF", "TLT", "SHY")
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ASSET = "cash"

BENCHMARK_STRATEGY_NAMES: frozenset[str] = frozenset(
    {"qqq_buy_hold", "spy_buy_hold", "cash"}
)

RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "start_date",
    "end_date",
    "trading_days",
    "exposure_rate_to_qqq",
    "exposure_rate_to_risk_off",
    "total_return",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "worst_1m_return",
    "worst_3m_return",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_underperformance",
    "spy_underperformance",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
)

DAILY_EQUITY_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_name",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "signal_date",
    "raw_signal_risk_on",
    "effective_signal_risk_on",
    "active_asset",
    "gross_return",
    "transaction_cost",
    "turnover",
    "daily_return",
    "equity",
)

TRADE_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_name",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "signal_date",
    "previous_asset",
    "new_asset",
    "turnover",
    "estimated_transaction_cost",
)

DEFENSIVE_PARAMETER_COLUMNS: tuple[str, ...] = (
    "slow_sma_days",
    "fast_sma_days",
    "buffer_pct",
    "min_hold_days",
    "return_lookback_days",
    "volatility_lookback_days",
    "volatility_percentile_window_days",
    "volatility_percentile_threshold",
    "drawdown_lookback_days",
    "drawdown_threshold",
    "regime_score_threshold",
)

DEFENSIVE_RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "family",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    *DEFENSIVE_PARAMETER_COLUMNS,
    "start_date",
    "end_date",
    "trading_days",
    "exposure_rate_to_qqq",
    "exposure_rate_to_risk_off",
    "total_return",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "worst_1m_return",
    "worst_3m_return",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_underperformance",
    "spy_underperformance",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "materially_improves_qqq_drawdown",
)

DEFENSIVE_DAILY_EQUITY_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_name",
    "family",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    *DEFENSIVE_PARAMETER_COLUMNS,
    "signal_date",
    "raw_signal_risk_on",
    "effective_signal_risk_on",
    "active_asset",
    "gross_return",
    "transaction_cost",
    "turnover",
    "daily_return",
    "equity",
)

DEFENSIVE_TRADE_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_name",
    "family",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    *DEFENSIVE_PARAMETER_COLUMNS,
    "signal_date",
    "previous_asset",
    "new_asset",
    "turnover",
    "estimated_transaction_cost",
)

DEFENSIVE_TIMING_FAMILIES: tuple[str, ...] = (
    "sma_price_filter_to_cash",
    "sma_crossover_to_cash",
    "sma_price_filter_to_spy",
    "volatility_guard_to_cash",
    "momentum_drawdown_guard_to_cash",
    "regime_score_defensive_to_cash",
)

NARROW_DEFENSIVE_TIMING_FAMILIES: tuple[str, ...] = (
    "total_return_spy_defensive",
    "drawdown_adjusted_cash_defensive",
    "low_turnover_crossover_cash",
)

DEFENSIVE_TIMING_SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("full", "2019-01-01", "2026-05-15"),
    ("covid_crash_recovery", "2020-01-01", "2020-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_bull_2023_2024", "2023-01-01", "2024-12-31"),
    ("recent_2025_2026", "2025-01-01", "2026-05-15"),
)

DEFENSIVE_SUBPERIOD_RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "period_name",
    "period_start",
    "period_end",
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "max_drawdown",
    "qqq_max_drawdown",
    "spy_max_drawdown",
    "transaction_count",
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
)

MATERIAL_DRAWDOWN_IMPROVEMENT = 0.05
SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME = "qqq_sma190_buffer_p0p5_hold0_to_spy"

_SMA_PRICE_FILTER_FAMILIES = {
    "sma_price_filter_to_cash",
    "sma_price_filter_to_spy",
    "total_return_spy_defensive",
    "drawdown_adjusted_cash_defensive",
}

_SMA_CROSSOVER_FAMILIES = {
    "sma_crossover_to_cash",
    "low_turnover_crossover_cash",
}


@dataclass(frozen=True)
class QQQTimingStrategySpec:
    """A single QQQ timing or benchmark strategy definition."""

    strategy_name: str
    risk_on_asset: str
    risk_off_asset: str
    signal_name: str
    is_benchmark: bool = False


@dataclass(frozen=True)
class QQQDefensiveTimingStrategySpec:
    """A single parameterized QQQ defensive timing sweep configuration."""

    strategy_name: str
    family: str
    risk_on_asset: str
    risk_off_asset: str
    signal_name: str
    slow_sma_days: int | None = None
    fast_sma_days: int | None = None
    buffer_pct: float | None = None
    min_hold_days: int = 0
    return_lookback_days: int | None = None
    volatility_lookback_days: int | None = None
    volatility_percentile_window_days: int | None = None
    volatility_percentile_threshold: float | None = None
    drawdown_lookback_days: int | None = None
    drawdown_threshold: float | None = None
    regime_score_threshold: int | None = None
    is_benchmark: bool = False


@dataclass(frozen=True)
class QQQTimingStudyResult:
    """Outputs from a QQQ timing baseline run."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    skipped_strategies: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class QQQDefensiveTimingSweepResult:
    """Outputs from a focused QQQ defensive timing sweep."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    skipped_strategies: tuple[str, ...]
    warnings: tuple[str, ...]
    reductions: tuple[str, ...]
    timing_config_count: int
    total_strategy_count: int


@dataclass(frozen=True)
class QQQDefensiveTimingNarrowSweepResult:
    """Outputs from the narrow QQQ defensive timing robustness sweep."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    subperiod_results: pd.DataFrame
    skipped_strategies: tuple[str, ...]
    warnings: tuple[str, ...]
    timing_config_count: int
    total_strategy_count: int


@dataclass(frozen=True)
class QQQDefensiveTimingSelectedReplayResult:
    """Outputs from the selected QQQ defensive timing replay bundle."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    subperiod_results: pd.DataFrame
    skipped_strategies: tuple[str, ...]
    warnings: tuple[str, ...]


def default_qqq_timing_strategy_specs(
    available_symbols: set[str] | None = None,
    *,
    include_regime_score: bool = True,
) -> tuple[tuple[QQQTimingStrategySpec, ...], tuple[str, ...]]:
    """Return the compact baseline strategy menu, skipping missing optional ETFs."""

    specs: list[QQQTimingStrategySpec] = [
        QQQTimingStrategySpec(
            "qqq_buy_hold",
            risk_on_asset="QQQ",
            risk_off_asset="",
            signal_name="buy_hold",
            is_benchmark=True,
        ),
        QQQTimingStrategySpec(
            "spy_buy_hold",
            risk_on_asset="SPY",
            risk_off_asset="",
            signal_name="buy_hold",
            is_benchmark=True,
        ),
        QQQTimingStrategySpec(
            "cash",
            risk_on_asset=RISK_FREE_ASSET,
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="cash",
            is_benchmark=True,
        ),
        QQQTimingStrategySpec(
            "qqq_sma200_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_sma200",
        ),
        QQQTimingStrategySpec(
            "qqq_sma50_sma200_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_sma50_sma200",
        ),
        QQQTimingStrategySpec(
            "qqq_return126_positive_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_return126_positive",
        ),
        QQQTimingStrategySpec(
            "qqq_return252_positive_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_return252_positive",
        ),
        QQQTimingStrategySpec(
            "qqq_momentum_drawdown_guard_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_momentum_drawdown_guard",
        ),
        QQQTimingStrategySpec(
            "qqq_volatility_guard_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_volatility_guard",
        ),
        QQQTimingStrategySpec(
            "qqq_vs_spy_relative_strength_cash",
            risk_on_asset="QQQ",
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="qqq_vs_spy_relative_strength",
        ),
        QQQTimingStrategySpec(
            "qqq_sma200_to_spy",
            risk_on_asset="QQQ",
            risk_off_asset="SPY",
            signal_name="qqq_sma200",
        ),
        QQQTimingStrategySpec(
            "qqq_return126_positive_to_spy",
            risk_on_asset="QQQ",
            risk_off_asset="SPY",
            signal_name="qqq_return126_positive",
        ),
    ]

    warnings: list[str] = []
    specs.extend(
        _optional_risk_off_spec(
            available_symbols=available_symbols,
            strategy_name="qqq_momentum_drawdown_guard_to_ief",
            risk_off_asset="IEF",
            signal_name="qqq_momentum_drawdown_guard",
            warnings=warnings,
        )
    )
    specs.extend(
        _optional_risk_off_spec(
            available_symbols=available_symbols,
            strategy_name="qqq_momentum_drawdown_guard_to_tlt",
            risk_off_asset="TLT",
            signal_name="qqq_momentum_drawdown_guard",
            warnings=warnings,
        )
    )

    if include_regime_score:
        specs.extend(
            [
                QQQTimingStrategySpec(
                    "qqq_regime_score_v3_cash",
                    risk_on_asset="QQQ",
                    risk_off_asset=RISK_FREE_ASSET,
                    signal_name="qqq_regime_score_v3",
                ),
                QQQTimingStrategySpec(
                    "qqq_regime_score_v3_to_spy",
                    risk_on_asset="QQQ",
                    risk_off_asset="SPY",
                    signal_name="qqq_regime_score_v3",
                ),
            ]
        )

    return tuple(specs), tuple(warnings)


def default_qqq_defensive_timing_sweep_specs(
    available_symbols: set[str] | None = None,
    *,
    max_timing_configs: int = 500,
) -> tuple[tuple[QQQDefensiveTimingStrategySpec, ...], tuple[str, ...], tuple[str, ...]]:
    """Return the focused Tiingo QQQ defensive timing sweep grid.

    The default grid contains 489 non-benchmark timing configurations and three
    benchmark references. Optional fixed-income symbols are only summarized by
    the CLIs; this focused grid intentionally tests cash and SPY risk-off legs.
    """

    warnings: list[str] = []
    reductions: list[str] = []
    if available_symbols is not None:
        for symbol in OPTIONAL_RISK_OFF_SYMBOLS:
            if symbol not in available_symbols:
                warnings.append(
                    f"Optional risk-off asset {symbol} is missing; no {symbol} variants "
                    "are included in this focused cash/SPY sweep."
                )

    benchmark_specs = _defensive_benchmark_specs()
    timing_specs = _raw_defensive_timing_sweep_specs()
    if len(timing_specs) > max_timing_configs:
        timing_specs, reductions = _reduce_defensive_timing_sweep(
            timing_specs,
            max_timing_configs=max_timing_configs,
        )

    return (
        (*benchmark_specs, *timing_specs),
        tuple(dict.fromkeys(warnings)),
        tuple(reductions),
    )


def default_qqq_defensive_timing_narrow_sweep_specs(
    available_symbols: set[str] | None = None,
) -> tuple[tuple[QQQDefensiveTimingStrategySpec, ...], tuple[str, ...]]:
    """Return the narrow Tiingo QQQ defensive timing sweep grid.

    The grid intentionally stays close to the two strongest prior defensive
    timing candidates and their nearest SMA/crossover neighbors. It contains
    426 non-benchmark timing configurations plus QQQ, SPY, and cash references.
    """

    warnings: list[str] = []
    if available_symbols is not None:
        for symbol in OPTIONAL_RISK_OFF_SYMBOLS:
            if symbol not in available_symbols:
                warnings.append(
                    f"Optional risk-off asset {symbol} is missing; no {symbol} variants "
                    "are included in this narrow cash/SPY sweep."
                )

    return (
        (*_defensive_benchmark_specs(), *_raw_defensive_timing_narrow_sweep_specs()),
        tuple(dict.fromkeys(warnings)),
    )


def default_qqq_defensive_timing_selected_replay_spec(
    *,
    strategy_name: str = SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME,
    slow_sma_days: int = 190,
    buffer_pct: float = 0.005,
    min_hold_days: int = 0,
    risk_off_asset: str = "SPY",
) -> QQQDefensiveTimingStrategySpec:
    """Return the selected narrow-sweep QQQ defensive replay configuration."""

    return QQQDefensiveTimingStrategySpec(
        strategy_name=strategy_name,
        family="total_return_spy_defensive",
        risk_on_asset="QQQ",
        risk_off_asset=risk_off_asset,
        signal_name="qqq_sma_price_filter",
        slow_sma_days=slow_sma_days,
        buffer_pct=buffer_pct,
        min_hold_days=min_hold_days,
    )


def prepare_qqq_timing_prices(
    market_data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    required_symbols: Sequence[str] = REQUIRED_SYMBOLS,
) -> pd.DataFrame:
    """Pivot normalized OHLCV rows into a close-price matrix for timing tests."""

    missing_columns = {"date", "symbol", "close"}.difference(market_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market data is missing required QQQ timing columns: {missing}")

    frame = market_data.loc[:, ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = frame.loc[frame["date"].between(start, end)]
    prices = (
        frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="first")
        .sort_index(kind="mergesort")
        .rename_axis(columns=None)
    )

    required = [symbol.upper() for symbol in required_symbols]
    missing_symbols = [symbol for symbol in required if symbol not in prices.columns]
    if missing_symbols:
        missing = ", ".join(missing_symbols)
        raise ValueError(f"Missing required QQQ timing symbols: {missing}")

    prices = prices.dropna(subset=required)
    if prices.empty:
        raise ValueError("No QQQ timing price rows remain after date and required-symbol filters.")
    return prices


def build_qqq_timing_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Build trailing indicators used by the compact QQQ timing strategies."""

    _validate_price_symbols(prices, REQUIRED_SYMBOLS)
    qqq_close = pd.to_numeric(prices["QQQ"], errors="coerce")
    spy_close = pd.to_numeric(prices["SPY"], errors="coerce")
    qqq_daily_return = qqq_close.pct_change()

    indicators = pd.DataFrame(index=prices.index)
    indicators["qqq_close"] = qqq_close
    indicators["spy_close"] = spy_close
    indicators["qqq_sma_50"] = qqq_close.rolling(window=50, min_periods=50).mean()
    indicators["qqq_sma_200"] = qqq_close.rolling(window=200, min_periods=200).mean()
    indicators["qqq_return_63d"] = qqq_close.pct_change(periods=63)
    indicators["qqq_return_126d"] = qqq_close.pct_change(periods=126)
    indicators["qqq_return_252d"] = qqq_close.pct_change(periods=252)
    indicators["spy_return_126d"] = spy_close.pct_change(periods=126)

    rolling_peak_126d = qqq_close.rolling(window=126, min_periods=126).max()
    indicators["qqq_drawdown_126d"] = qqq_close / rolling_peak_126d - 1.0
    indicators["qqq_volatility_63d"] = qqq_daily_return.rolling(
        window=63,
        min_periods=63,
    ).std()
    indicators["qqq_volatility_63d_p75_756d"] = indicators[
        "qqq_volatility_63d"
    ].rolling(window=756, min_periods=1).quantile(0.75)

    for lookback in (63, 126, 252):
        return_col = f"qqq_return_{lookback}d"
        flag_col = f"{return_col}_positive_flag"
        indicators[flag_col] = _positive_flag(indicators[return_col])

    for lookback in (126, 252):
        percentile = indicators[f"qqq_return_{lookback}d"].rolling(
            window=756,
            min_periods=max(lookback, 126),
        ).apply(_percentile_rank_of_last_value, raw=True)
        top_quartile_col = f"qqq_return_{lookback}d_top_quartile_756d"
        indicators[top_quartile_col] = _positive_flag(percentile - 0.75)

    regime_score_columns = [
        "qqq_return_63d_positive_flag",
        "qqq_return_126d_positive_flag",
        "qqq_return_252d_positive_flag",
        "qqq_return_126d_top_quartile_756d",
        "qqq_return_252d_top_quartile_756d",
    ]
    indicators["qqq_strong_regime_score"] = indicators[regime_score_columns].mean(
        axis=1,
        skipna=True,
    )
    indicators["qqq_strong_regime_score_count"] = indicators[regime_score_columns].sum(
        axis=1,
        skipna=True,
    )
    return indicators


def build_qqq_timing_signal(indicators: pd.DataFrame, signal_name: str) -> pd.Series:
    """Return a same-date close-based risk-on signal for one timing rule."""

    if signal_name == "qqq_sma200":
        signal = indicators["qqq_close"] > indicators["qqq_sma_200"]
    elif signal_name == "qqq_sma50_sma200":
        signal = indicators["qqq_sma_50"] > indicators["qqq_sma_200"]
    elif signal_name == "qqq_return126_positive":
        signal = indicators["qqq_return_126d"] > 0.0
    elif signal_name == "qqq_return252_positive":
        signal = indicators["qqq_return_252d"] > 0.0
    elif signal_name == "qqq_momentum_drawdown_guard":
        signal = (indicators["qqq_return_126d"] > 0.0) & (
            indicators["qqq_drawdown_126d"] > -0.10
        )
    elif signal_name == "qqq_volatility_guard":
        signal = (
            (indicators["qqq_return_126d"] > 0.0)
            & indicators["qqq_volatility_63d"].notna()
            & (
                indicators["qqq_volatility_63d"]
                < indicators["qqq_volatility_63d_p75_756d"]
            )
        )
    elif signal_name == "qqq_vs_spy_relative_strength":
        signal = (indicators["qqq_return_126d"] > indicators["spy_return_126d"]) & (
            indicators["qqq_return_126d"] > 0.0
        )
    elif signal_name == "qqq_regime_score_v3":
        signal = indicators["qqq_strong_regime_score"] >= 0.60
    else:
        raise ValueError(f"Unsupported QQQ timing signal_name: {signal_name}")

    return signal.fillna(False).astype(bool)


def build_qqq_defensive_timing_signal(
    prices: pd.DataFrame,
    spec: QQQDefensiveTimingStrategySpec,
) -> pd.Series:
    """Return a same-date close-based risk-on signal for one sweep config."""

    if spec.is_benchmark:
        if spec.strategy_name == "qqq_buy_hold":
            return pd.Series(True, index=prices.index, dtype=bool)
        return pd.Series(False, index=prices.index, dtype=bool)

    _validate_price_symbols(prices, REQUIRED_SYMBOLS)
    qqq_close = pd.to_numeric(prices["QQQ"], errors="coerce")
    qqq_daily_return = qqq_close.pct_change()

    if spec.family in _SMA_PRICE_FILTER_FAMILIES:
        slow_sma_days = _require_int(spec.slow_sma_days, "slow_sma_days", spec)
        buffer_pct = _require_float(spec.buffer_pct, "buffer_pct", spec)
        slow_sma = qqq_close.rolling(window=slow_sma_days, min_periods=slow_sma_days).mean()
        signal = qqq_close > slow_sma * (1.0 + buffer_pct)
    elif spec.family in _SMA_CROSSOVER_FAMILIES:
        fast_sma_days = _require_int(spec.fast_sma_days, "fast_sma_days", spec)
        slow_sma_days = _require_int(spec.slow_sma_days, "slow_sma_days", spec)
        if fast_sma_days >= slow_sma_days:
            raise ValueError(
                f"{spec.strategy_name} has fast_sma_days >= slow_sma_days."
            )
        buffer_pct = _require_float(spec.buffer_pct, "buffer_pct", spec)
        fast_sma = qqq_close.rolling(window=fast_sma_days, min_periods=fast_sma_days).mean()
        slow_sma = qqq_close.rolling(window=slow_sma_days, min_periods=slow_sma_days).mean()
        signal = fast_sma > slow_sma * (1.0 + buffer_pct)
    elif spec.family == "volatility_guard_to_cash":
        return_lookback_days = _require_int(
            spec.return_lookback_days,
            "return_lookback_days",
            spec,
        )
        volatility_lookback_days = _require_int(
            spec.volatility_lookback_days,
            "volatility_lookback_days",
            spec,
        )
        volatility_percentile_window_days = _require_int(
            spec.volatility_percentile_window_days,
            "volatility_percentile_window_days",
            spec,
        )
        volatility_percentile_threshold = _require_float(
            spec.volatility_percentile_threshold,
            "volatility_percentile_threshold",
            spec,
        )
        momentum = qqq_close.pct_change(periods=return_lookback_days) > 0.0
        realized_volatility = qqq_daily_return.rolling(
            window=volatility_lookback_days,
            min_periods=volatility_lookback_days,
        ).std()
        percentile_min_periods = min(
            volatility_percentile_window_days,
            max(63, volatility_lookback_days),
        )
        volatility_threshold = realized_volatility.rolling(
            window=volatility_percentile_window_days,
            min_periods=percentile_min_periods,
        ).quantile(volatility_percentile_threshold)
        signal = (
            momentum
            & realized_volatility.notna()
            & volatility_threshold.notna()
            & (realized_volatility < volatility_threshold)
        )
    elif spec.family == "momentum_drawdown_guard_to_cash":
        return_lookback_days = _require_int(
            spec.return_lookback_days,
            "return_lookback_days",
            spec,
        )
        drawdown_lookback_days = _require_int(
            spec.drawdown_lookback_days,
            "drawdown_lookback_days",
            spec,
        )
        drawdown_threshold = _require_float(
            spec.drawdown_threshold,
            "drawdown_threshold",
            spec,
        )
        momentum = qqq_close.pct_change(periods=return_lookback_days) > 0.0
        rolling_peak = qqq_close.rolling(
            window=drawdown_lookback_days,
            min_periods=drawdown_lookback_days,
        ).max()
        drawdown = qqq_close / rolling_peak - 1.0
        signal = momentum & drawdown.notna() & (drawdown > drawdown_threshold)
    elif spec.family == "regime_score_defensive_to_cash":
        threshold = _require_int(spec.regime_score_threshold, "regime_score_threshold", spec)
        indicators = build_qqq_timing_indicators(prices)
        signal = indicators["qqq_strong_regime_score_count"] >= threshold
    else:
        raise ValueError(f"Unsupported QQQ defensive timing family: {spec.family}")

    return _apply_min_hold(signal.fillna(False).astype(bool), spec.min_hold_days)


def run_qqq_timing_strategy(
    prices: pd.DataFrame,
    spec: QQQTimingStrategySpec,
    *,
    transaction_cost_bps: float = 10.0,
    indicators: pd.DataFrame | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run one QQQ timing strategy over an already prepared price matrix."""

    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be greater than or equal to 0.")
    if prices.empty:
        raise ValueError("At least one QQQ timing price row is required.")

    required_assets = [asset for asset in (spec.risk_on_asset, spec.risk_off_asset) if asset]
    required_price_assets = [
        asset for asset in required_assets if asset != RISK_FREE_ASSET and asset in REQUIRED_SYMBOLS
    ]
    _validate_price_symbols(prices, required_price_assets)
    if spec.risk_off_asset not in {"", RISK_FREE_ASSET} and spec.risk_off_asset not in prices:
        raise ValueError(
            f"Risk-off asset {spec.risk_off_asset} is missing for {spec.strategy_name}."
        )
    if spec.risk_on_asset not in {RISK_FREE_ASSET} and spec.risk_on_asset not in prices:
        raise ValueError(
            f"Risk-on asset {spec.risk_on_asset} is missing for {spec.strategy_name}."
        )

    dates = prices.index
    asset_returns = prices.pct_change().fillna(0.0)
    signal_frame = indicators if indicators is not None else build_qqq_timing_indicators(prices)

    raw_signal = _raw_signal_for_spec(signal_frame, spec)
    effective_signal = raw_signal.shift(1).fillna(False).astype(bool)
    active_asset = _active_asset_for_spec(spec, effective_signal, dates)
    signal_dates = pd.Series(dates, index=dates).shift(1)
    gross_return = _gross_returns_for_active_assets(active_asset, asset_returns)
    turnover = _turnover_for_active_assets(active_asset)
    transaction_cost = turnover * transaction_cost_bps / 10000.0
    daily_return = gross_return - transaction_cost
    equity = (1.0 + daily_return).cumprod()

    daily_equity = pd.DataFrame(
        {
            "date": dates,
            "strategy_name": spec.strategy_name,
            "risk_on_asset": spec.risk_on_asset,
            "risk_off_asset": spec.risk_off_asset,
            "signal_name": spec.signal_name,
            "signal_date": signal_dates.values,
            "raw_signal_risk_on": raw_signal.values,
            "effective_signal_risk_on": effective_signal.values,
            "active_asset": active_asset.values,
            "gross_return": gross_return.values,
            "transaction_cost": transaction_cost.values,
            "turnover": turnover.values,
            "daily_return": daily_return.values,
            "equity": equity.values,
        },
        columns=DAILY_EQUITY_COLUMNS,
    )

    trades = _build_trade_frame(daily_equity, spec)
    metrics = _strategy_metrics(
        spec=spec,
        daily_equity=daily_equity,
        prices=prices,
    )
    return metrics, daily_equity, trades


def run_qqq_defensive_timing_strategy(
    prices: pd.DataFrame,
    spec: QQQDefensiveTimingStrategySpec,
    *,
    transaction_cost_bps: float = 10.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run one parameterized QQQ defensive timing sweep configuration."""

    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be greater than or equal to 0.")
    if prices.empty:
        raise ValueError("At least one QQQ timing price row is required.")

    required_assets = [asset for asset in (spec.risk_on_asset, spec.risk_off_asset) if asset]
    required_price_assets = [
        asset for asset in required_assets if asset != RISK_FREE_ASSET and asset in REQUIRED_SYMBOLS
    ]
    _validate_price_symbols(prices, required_price_assets)
    if spec.risk_off_asset not in {"", RISK_FREE_ASSET} and spec.risk_off_asset not in prices:
        raise ValueError(
            f"Risk-off asset {spec.risk_off_asset} is missing for {spec.strategy_name}."
        )
    if spec.risk_on_asset not in {RISK_FREE_ASSET} and spec.risk_on_asset not in prices:
        raise ValueError(
            f"Risk-on asset {spec.risk_on_asset} is missing for {spec.strategy_name}."
        )

    dates = prices.index
    asset_returns = prices.pct_change().fillna(0.0)
    raw_signal = build_qqq_defensive_timing_signal(prices, spec)
    effective_signal = raw_signal.shift(1).fillna(False).astype(bool)
    active_asset = _active_asset_for_spec(spec, effective_signal, dates)
    signal_dates = pd.Series(dates, index=dates).shift(1)
    gross_return = _gross_returns_for_active_assets(active_asset, asset_returns)
    turnover = _turnover_for_active_assets(active_asset)
    transaction_cost = turnover * transaction_cost_bps / 10000.0
    daily_return = gross_return - transaction_cost
    equity = (1.0 + daily_return).cumprod()
    parameter_values = _defensive_parameter_values(spec)

    daily_equity = pd.DataFrame(
        {
            "date": dates,
            "strategy_name": spec.strategy_name,
            "family": spec.family,
            "risk_on_asset": spec.risk_on_asset,
            "risk_off_asset": spec.risk_off_asset,
            "signal_name": spec.signal_name,
            **parameter_values,
            "signal_date": signal_dates.values,
            "raw_signal_risk_on": raw_signal.values,
            "effective_signal_risk_on": effective_signal.values,
            "active_asset": active_asset.values,
            "gross_return": gross_return.values,
            "transaction_cost": transaction_cost.values,
            "turnover": turnover.values,
            "daily_return": daily_return.values,
            "equity": equity.values,
        },
        columns=DEFENSIVE_DAILY_EQUITY_COLUMNS,
    )

    trades = _build_defensive_trade_frame(daily_equity, spec)
    metrics = _defensive_strategy_metrics(
        spec=spec,
        daily_equity=daily_equity,
        prices=prices,
    )
    return metrics, daily_equity, trades


def run_qqq_defensive_timing_sweep(
    market_data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    max_timing_configs: int = 500,
) -> QQQDefensiveTimingSweepResult:
    """Run the focused Tiingo-only QQQ defensive timing sweep."""

    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=start_date,
        end_date=end_date,
    )
    available_symbols = set(prices.columns[prices.notna().any()].astype(str))
    specs, strategy_warnings, reductions = default_qqq_defensive_timing_sweep_specs(
        available_symbols,
        max_timing_configs=max_timing_configs,
    )

    result_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    warnings = list(strategy_warnings)

    for spec in specs:
        if _strategy_has_missing_price_rows(prices, spec):
            skipped.append(spec.strategy_name)
            warnings.append(
                f"Skipping {spec.strategy_name}: {spec.risk_off_asset} has missing prices "
                "inside the sweep window."
            )
            continue
        metrics, daily_equity, trades = run_qqq_defensive_timing_strategy(
            prices,
            spec,
            transaction_cost_bps=transaction_cost_bps,
        )
        result_rows.append(metrics)
        daily_frames.append(daily_equity)
        trade_frames.append(trades)

    results = pd.DataFrame(result_rows, columns=DEFENSIVE_RESULT_COLUMNS)
    daily_equity = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=DEFENSIVE_DAILY_EQUITY_COLUMNS)
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=DEFENSIVE_TRADE_COLUMNS)
    )
    timing_config_count = sum(1 for spec in specs if not spec.is_benchmark)
    return QQQDefensiveTimingSweepResult(
        results=results,
        daily_equity=daily_equity,
        trades=trades,
        skipped_strategies=tuple(skipped),
        warnings=tuple(dict.fromkeys(warnings)),
        reductions=tuple(reductions),
        timing_config_count=timing_config_count,
        total_strategy_count=len(specs),
    )


def run_qqq_defensive_timing_narrow_sweep(
    market_data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    subperiods: Sequence[tuple[str, str, str]] = DEFENSIVE_TIMING_SUBPERIODS,
) -> QQQDefensiveTimingNarrowSweepResult:
    """Run the narrow Tiingo-only QQQ defensive timing robustness sweep."""

    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=start_date,
        end_date=end_date,
    )
    available_symbols = set(prices.columns[prices.notna().any()].astype(str))
    specs, strategy_warnings = default_qqq_defensive_timing_narrow_sweep_specs(
        available_symbols,
    )

    result_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    warnings = list(strategy_warnings)

    for spec in specs:
        if _strategy_has_missing_price_rows(prices, spec):
            skipped.append(spec.strategy_name)
            warnings.append(
                f"Skipping {spec.strategy_name}: {spec.risk_off_asset} has missing prices "
                "inside the narrow sweep window."
            )
            continue
        metrics, daily_equity, trades = run_qqq_defensive_timing_strategy(
            prices,
            spec,
            transaction_cost_bps=transaction_cost_bps,
        )
        result_rows.append(metrics)
        daily_frames.append(daily_equity)
        trade_frames.append(trades)

    results = pd.DataFrame(result_rows, columns=DEFENSIVE_RESULT_COLUMNS)
    daily_equity = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=DEFENSIVE_DAILY_EQUITY_COLUMNS)
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=DEFENSIVE_TRADE_COLUMNS)
    )
    subperiod_results = calculate_qqq_defensive_timing_subperiod_results(
        daily_equity=daily_equity,
        prices=prices,
        subperiods=subperiods,
    )
    timing_config_count = sum(1 for spec in specs if not spec.is_benchmark)
    return QQQDefensiveTimingNarrowSweepResult(
        results=results,
        daily_equity=daily_equity,
        trades=trades,
        subperiod_results=subperiod_results,
        skipped_strategies=tuple(skipped),
        warnings=tuple(dict.fromkeys(warnings)),
        timing_config_count=timing_config_count,
        total_strategy_count=len(specs),
    )


def run_qqq_defensive_timing_selected_replay(
    market_data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    selected_spec: QQQDefensiveTimingStrategySpec | None = None,
    subperiods: Sequence[tuple[str, str, str]] = DEFENSIVE_TIMING_SUBPERIODS,
) -> QQQDefensiveTimingSelectedReplayResult:
    """Run the selected QQQ defensive timing candidate plus benchmark references."""

    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=start_date,
        end_date=end_date,
    )
    spec = selected_spec or default_qqq_defensive_timing_selected_replay_spec()
    specs = (spec, *_defensive_benchmark_specs())

    result_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    warnings: list[str] = []

    for strategy_spec in specs:
        if _strategy_has_missing_price_rows(prices, strategy_spec):
            skipped.append(strategy_spec.strategy_name)
            warnings.append(
                f"Skipping {strategy_spec.strategy_name}: required asset has missing prices "
                "inside the selected replay window."
            )
            continue
        metrics, daily_equity, trades = run_qqq_defensive_timing_strategy(
            prices,
            strategy_spec,
            transaction_cost_bps=transaction_cost_bps,
        )
        result_rows.append(metrics)
        daily_frames.append(daily_equity)
        trade_frames.append(trades)

    results = pd.DataFrame(result_rows, columns=DEFENSIVE_RESULT_COLUMNS)
    daily_equity = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=DEFENSIVE_DAILY_EQUITY_COLUMNS)
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=DEFENSIVE_TRADE_COLUMNS)
    )
    subperiod_results = calculate_qqq_defensive_timing_subperiod_results(
        daily_equity=daily_equity,
        prices=prices,
        subperiods=subperiods,
    )
    return QQQDefensiveTimingSelectedReplayResult(
        results=results,
        daily_equity=daily_equity,
        trades=trades,
        subperiod_results=subperiod_results,
        skipped_strategies=tuple(skipped),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def calculate_qqq_defensive_timing_subperiod_results(
    *,
    daily_equity: pd.DataFrame,
    prices: pd.DataFrame,
    subperiods: Sequence[tuple[str, str, str]] = DEFENSIVE_TIMING_SUBPERIODS,
) -> pd.DataFrame:
    """Calculate leakage-safe strategy and benchmark metrics by subperiod."""

    if daily_equity.empty:
        return pd.DataFrame(columns=DEFENSIVE_SUBPERIOD_RESULT_COLUMNS)

    _validate_price_symbols(prices, REQUIRED_SYMBOLS)
    price_frame = prices.copy()
    price_frame.index = pd.to_datetime(price_frame.index)
    daily = daily_equity.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["strategy_name", "date"], kind="mergesort")

    rows: list[dict[str, object]] = []
    for strategy_name, strategy_daily in daily.groupby("strategy_name", sort=False):
        for period_name, period_start, period_end in subperiods:
            start = pd.Timestamp(period_start)
            end = pd.Timestamp(period_end)
            period_daily = strategy_daily.loc[strategy_daily["date"].between(start, end)]
            period_prices = price_frame.loc[price_frame.index.to_series().between(start, end)]
            if period_daily.empty or period_prices.empty:
                continue
            rows.append(
                _defensive_subperiod_metric_row(
                    strategy_name=str(strategy_name),
                    period_name=period_name,
                    period_start=period_start,
                    period_end=period_end,
                    period_daily=period_daily,
                    period_prices=period_prices,
                )
            )

    return pd.DataFrame(rows, columns=DEFENSIVE_SUBPERIOD_RESULT_COLUMNS)


def run_qqq_timing_study(
    market_data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    include_regime_score: bool = True,
) -> QQQTimingStudyResult:
    """Run the compact QQQ timing baseline study on local market data."""

    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=start_date,
        end_date=end_date,
    )
    available_symbols = set(prices.columns[prices.notna().any()].astype(str))
    specs, strategy_warnings = default_qqq_timing_strategy_specs(
        available_symbols,
        include_regime_score=include_regime_score,
    )
    indicators = build_qqq_timing_indicators(prices)

    result_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    warnings = list(strategy_warnings)

    for spec in specs:
        if _strategy_has_missing_price_rows(prices, spec):
            skipped.append(spec.strategy_name)
            warnings.append(
                f"Skipping {spec.strategy_name}: {spec.risk_off_asset} has missing prices "
                "inside the study window."
            )
            continue
        metrics, daily_equity, trades = run_qqq_timing_strategy(
            prices,
            spec,
            transaction_cost_bps=transaction_cost_bps,
            indicators=indicators,
        )
        result_rows.append(metrics)
        daily_frames.append(daily_equity)
        trade_frames.append(trades)

    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    daily_equity = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=DAILY_EQUITY_COLUMNS)
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    return QQQTimingStudyResult(
        results=results,
        daily_equity=daily_equity,
        trades=trades,
        skipped_strategies=tuple(skipped),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def available_symbol_summary(
    prices: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return required/optional present and missing symbol summaries."""

    present = set(prices.columns[prices.notna().any()].astype(str))
    required_present = tuple(symbol for symbol in REQUIRED_SYMBOLS if symbol in present)
    required_missing = tuple(symbol for symbol in REQUIRED_SYMBOLS if symbol not in present)
    optional_present = tuple(symbol for symbol in OPTIONAL_RISK_OFF_SYMBOLS if symbol in present)
    optional_missing = tuple(
        symbol for symbol in OPTIONAL_RISK_OFF_SYMBOLS if symbol not in present
    )
    return required_present, required_missing, optional_present, optional_missing


def _optional_risk_off_spec(
    *,
    available_symbols: set[str] | None,
    strategy_name: str,
    risk_off_asset: str,
    signal_name: str,
    warnings: list[str],
) -> tuple[QQQTimingStrategySpec, ...]:
    if available_symbols is not None and risk_off_asset not in available_symbols:
        warnings.append(
            f"Skipping {strategy_name}: optional risk-off asset {risk_off_asset} is missing."
        )
        return ()
    return (
        QQQTimingStrategySpec(
            strategy_name,
            risk_on_asset="QQQ",
            risk_off_asset=risk_off_asset,
            signal_name=signal_name,
        ),
    )


def _defensive_benchmark_specs() -> tuple[QQQDefensiveTimingStrategySpec, ...]:
    return (
        QQQDefensiveTimingStrategySpec(
            "qqq_buy_hold",
            family="benchmark",
            risk_on_asset="QQQ",
            risk_off_asset="",
            signal_name="buy_hold",
            is_benchmark=True,
        ),
        QQQDefensiveTimingStrategySpec(
            "spy_buy_hold",
            family="benchmark",
            risk_on_asset="SPY",
            risk_off_asset="",
            signal_name="buy_hold",
            is_benchmark=True,
        ),
        QQQDefensiveTimingStrategySpec(
            "cash",
            family="benchmark",
            risk_on_asset=RISK_FREE_ASSET,
            risk_off_asset=RISK_FREE_ASSET,
            signal_name="cash",
            is_benchmark=True,
        ),
    )


def _raw_defensive_timing_sweep_specs() -> tuple[QQQDefensiveTimingStrategySpec, ...]:
    specs: list[QQQDefensiveTimingStrategySpec] = []

    for slow_sma_days in (150, 180, 200, 220, 250):
        for buffer_pct in (-0.02, -0.01, 0.0, 0.01, 0.02):
            for min_hold_days in (0, 5, 10):
                specs.append(
                    QQQDefensiveTimingStrategySpec(
                        strategy_name=(
                            f"qqq_sma{slow_sma_days}_buffer_"
                            f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_cash"
                        ),
                        family="sma_price_filter_to_cash",
                        risk_on_asset="QQQ",
                        risk_off_asset=RISK_FREE_ASSET,
                        signal_name="qqq_sma_price_filter",
                        slow_sma_days=slow_sma_days,
                        buffer_pct=buffer_pct,
                        min_hold_days=min_hold_days,
                    )
                )

    for fast_sma_days in (20, 50, 75, 100):
        for slow_sma_days in (150, 180, 200, 220, 250):
            if fast_sma_days >= slow_sma_days:
                continue
            for buffer_pct in (-0.01, 0.0, 0.01):
                for min_hold_days in (0, 5, 10):
                    specs.append(
                        QQQDefensiveTimingStrategySpec(
                            strategy_name=(
                                f"qqq_sma{fast_sma_days}_sma{slow_sma_days}_buffer_"
                                f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_cash"
                            ),
                            family="sma_crossover_to_cash",
                            risk_on_asset="QQQ",
                            risk_off_asset=RISK_FREE_ASSET,
                            signal_name="qqq_sma_crossover",
                            slow_sma_days=slow_sma_days,
                            fast_sma_days=fast_sma_days,
                            buffer_pct=buffer_pct,
                            min_hold_days=min_hold_days,
                        )
                    )

    for slow_sma_days in (180, 200, 220, 250):
        for buffer_pct in (-0.01, 0.0, 0.01):
            for min_hold_days in (0, 5, 10):
                specs.append(
                    QQQDefensiveTimingStrategySpec(
                        strategy_name=(
                            f"qqq_sma{slow_sma_days}_buffer_"
                            f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_to_spy"
                        ),
                        family="sma_price_filter_to_spy",
                        risk_on_asset="QQQ",
                        risk_off_asset="SPY",
                        signal_name="qqq_sma_price_filter",
                        slow_sma_days=slow_sma_days,
                        buffer_pct=buffer_pct,
                        min_hold_days=min_hold_days,
                    )
                )

    for return_lookback_days in (63, 126, 252):
        for volatility_lookback_days in (21, 63):
            for volatility_percentile_window_days in (504, 756):
                for volatility_percentile_threshold in (0.60, 0.70, 0.80):
                    for min_hold_days in (0, 5, 10):
                        specs.append(
                            QQQDefensiveTimingStrategySpec(
                                strategy_name=(
                                    f"qqq_ret{return_lookback_days}_vol"
                                    f"{volatility_lookback_days}_pctl"
                                    f"{volatility_percentile_window_days}_thr"
                                    f"{int(volatility_percentile_threshold * 100)}_"
                                    f"hold{min_hold_days}_cash"
                                ),
                                family="volatility_guard_to_cash",
                                risk_on_asset="QQQ",
                                risk_off_asset=RISK_FREE_ASSET,
                                signal_name="qqq_volatility_percentile_guard",
                                min_hold_days=min_hold_days,
                                return_lookback_days=return_lookback_days,
                                volatility_lookback_days=volatility_lookback_days,
                                volatility_percentile_window_days=(
                                    volatility_percentile_window_days
                                ),
                                volatility_percentile_threshold=(
                                    volatility_percentile_threshold
                                ),
                            )
                        )

    for return_lookback_days in (63, 126, 252):
        for drawdown_lookback_days in (63, 126, 252):
            for drawdown_threshold in (-0.05, -0.10, -0.15):
                for min_hold_days in (0, 5, 10):
                    specs.append(
                        QQQDefensiveTimingStrategySpec(
                            strategy_name=(
                                f"qqq_ret{return_lookback_days}_dd"
                                f"{drawdown_lookback_days}_thr_"
                                f"{_parameter_token(drawdown_threshold)}_"
                                f"hold{min_hold_days}_cash"
                            ),
                            family="momentum_drawdown_guard_to_cash",
                            risk_on_asset="QQQ",
                            risk_off_asset=RISK_FREE_ASSET,
                            signal_name="qqq_momentum_drawdown_guard_parameterized",
                            min_hold_days=min_hold_days,
                            return_lookback_days=return_lookback_days,
                            drawdown_lookback_days=drawdown_lookback_days,
                            drawdown_threshold=drawdown_threshold,
                        )
                    )

    for regime_score_threshold in (1, 2, 3):
        for min_hold_days in (0, 5, 10):
            specs.append(
                QQQDefensiveTimingStrategySpec(
                    strategy_name=(
                        f"qqq_regime_score_ge{regime_score_threshold}_"
                        f"hold{min_hold_days}_cash"
                    ),
                    family="regime_score_defensive_to_cash",
                    risk_on_asset="QQQ",
                    risk_off_asset=RISK_FREE_ASSET,
                    signal_name="qqq_regime_score_v3_count",
                    min_hold_days=min_hold_days,
                    regime_score_threshold=regime_score_threshold,
                )
            )

    return tuple(specs)


def _raw_defensive_timing_narrow_sweep_specs() -> tuple[QQQDefensiveTimingStrategySpec, ...]:
    specs: list[QQQDefensiveTimingStrategySpec] = []

    for slow_sma_days in (165, 170, 175, 180, 185, 190, 195, 200, 205):
        for buffer_pct in (-0.005, 0.0, 0.005, 0.01):
            for min_hold_days in (0, 3, 5, 10):
                specs.append(
                    QQQDefensiveTimingStrategySpec(
                        strategy_name=(
                            f"qqq_sma{slow_sma_days}_buffer_"
                            f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_to_spy"
                        ),
                        family="total_return_spy_defensive",
                        risk_on_asset="QQQ",
                        risk_off_asset="SPY",
                        signal_name="qqq_sma_price_filter",
                        slow_sma_days=slow_sma_days,
                        buffer_pct=buffer_pct,
                        min_hold_days=min_hold_days,
                    )
                )

    for slow_sma_days in (180, 190, 200, 210, 220):
        for buffer_pct in (-0.02, -0.01, 0.0, 0.005, 0.01, 0.02):
            for min_hold_days in (5, 10, 15):
                specs.append(
                    QQQDefensiveTimingStrategySpec(
                        strategy_name=(
                            f"qqq_sma{slow_sma_days}_buffer_"
                            f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_cash"
                        ),
                        family="drawdown_adjusted_cash_defensive",
                        risk_on_asset="QQQ",
                        risk_off_asset=RISK_FREE_ASSET,
                        signal_name="qqq_sma_price_filter",
                        slow_sma_days=slow_sma_days,
                        buffer_pct=buffer_pct,
                        min_hold_days=min_hold_days,
                    )
                )

    for fast_sma_days in (60, 75, 90, 100):
        for slow_sma_days in (180, 200, 220, 240):
            if fast_sma_days >= slow_sma_days:
                continue
            for buffer_pct in (-0.015, -0.01, -0.005, 0.0):
                for min_hold_days in (0, 5, 10):
                    specs.append(
                        QQQDefensiveTimingStrategySpec(
                            strategy_name=(
                                f"qqq_sma{fast_sma_days}_sma{slow_sma_days}_buffer_"
                                f"{_parameter_token(buffer_pct)}_hold{min_hold_days}_cash"
                            ),
                            family="low_turnover_crossover_cash",
                            risk_on_asset="QQQ",
                            risk_off_asset=RISK_FREE_ASSET,
                            signal_name="qqq_sma_crossover",
                            slow_sma_days=slow_sma_days,
                            fast_sma_days=fast_sma_days,
                            buffer_pct=buffer_pct,
                            min_hold_days=min_hold_days,
                        )
                    )

    return tuple(specs)


def _reduce_defensive_timing_sweep(
    specs: tuple[QQQDefensiveTimingStrategySpec, ...],
    *,
    max_timing_configs: int,
) -> tuple[tuple[QQQDefensiveTimingStrategySpec, ...], list[str]]:
    reductions: list[str] = []
    reduced = list(specs)
    for family in (
        "regime_score_defensive_to_cash",
        "volatility_guard_to_cash",
        "momentum_drawdown_guard_to_cash",
        "sma_crossover_to_cash",
    ):
        if len(reduced) <= max_timing_configs:
            break
        before = len(reduced)
        reduced = [spec for spec in reduced if spec.family != family]
        reductions.append(
            f"Reduced {family}: removed {before - len(reduced)} configs to keep "
            f"the focused sweep under {max_timing_configs} timing configs."
        )
    if len(reduced) > max_timing_configs:
        reduced = reduced[:max_timing_configs]
        reductions.append(
            f"Truncated remaining grid to the first {max_timing_configs} configs."
        )
    return tuple(reduced), reductions


def _parameter_token(value: float) -> str:
    scaled = round(float(value) * 100, 3)
    if scaled < 0:
        return f"m{_format_parameter_magnitude(abs(scaled))}"
    if scaled > 0:
        return f"p{_format_parameter_magnitude(scaled)}"
    return "p0"


def _format_parameter_magnitude(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _validate_price_symbols(prices: pd.DataFrame, symbols: Sequence[str]) -> None:
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns for symbols: {', '.join(missing)}")


def _positive_flag(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out.loc[values.notna()] = (values.loc[values.notna()] > 0.0).astype(float)
    return out


def _require_int(
    value: int | None,
    name: str,
    spec: QQQDefensiveTimingStrategySpec,
) -> int:
    if value is None:
        raise ValueError(f"{spec.strategy_name} is missing required parameter {name}.")
    return int(value)


def _require_float(
    value: float | None,
    name: str,
    spec: QQQDefensiveTimingStrategySpec,
) -> float:
    if value is None:
        raise ValueError(f"{spec.strategy_name} is missing required parameter {name}.")
    return float(value)


def _apply_min_hold(raw_signal: pd.Series, min_hold_days: int) -> pd.Series:
    if min_hold_days < 0:
        raise ValueError("min_hold_days must be greater than or equal to 0.")
    signal = raw_signal.fillna(False).astype(bool)
    if signal.empty or min_hold_days == 0:
        return signal

    values = signal.to_numpy(dtype=bool)
    held_values = np.empty(len(values), dtype=bool)
    current = bool(values[0])
    days_in_state = 1
    held_values[0] = current
    for index in range(1, len(values)):
        desired = bool(values[index])
        if desired != current and days_in_state >= min_hold_days:
            current = desired
            days_in_state = 1
        else:
            days_in_state += 1
        held_values[index] = current
    return pd.Series(held_values, index=signal.index, dtype=bool)


def _percentile_rank_of_last_value(values: np.ndarray) -> float:
    current = values[-1]
    if np.isnan(current):
        return np.nan
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    return float((valid <= current).mean())


def _raw_signal_for_spec(
    indicators: pd.DataFrame,
    spec: QQQTimingStrategySpec,
) -> pd.Series:
    if spec.strategy_name == "qqq_buy_hold":
        return pd.Series(True, index=indicators.index, dtype=bool)
    if spec.strategy_name in {"spy_buy_hold", "cash"}:
        return pd.Series(False, index=indicators.index, dtype=bool)
    return build_qqq_timing_signal(indicators, spec.signal_name)


def _active_asset_for_spec(
    spec: QQQTimingStrategySpec,
    effective_signal: pd.Series,
    dates: pd.Index,
) -> pd.Series:
    if spec.strategy_name == "qqq_buy_hold":
        return pd.Series("QQQ", index=dates, dtype="object")
    if spec.strategy_name == "spy_buy_hold":
        return pd.Series("SPY", index=dates, dtype="object")
    if spec.strategy_name == "cash":
        return pd.Series(RISK_FREE_ASSET, index=dates, dtype="object")

    risk_off_asset = spec.risk_off_asset or RISK_FREE_ASSET
    active_values = np.where(effective_signal.to_numpy(), spec.risk_on_asset, risk_off_asset)
    return pd.Series(active_values, index=dates, dtype="object")


def _gross_returns_for_active_assets(
    active_asset: pd.Series,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    gross_return = pd.Series(0.0, index=active_asset.index, dtype=float)
    for asset in sorted(set(active_asset.dropna().astype(str))):
        if asset == RISK_FREE_ASSET:
            continue
        if asset not in asset_returns.columns:
            raise ValueError(f"Active asset {asset} is missing from asset return frame.")
        mask = active_asset.eq(asset)
        gross_return.loc[mask] = pd.to_numeric(asset_returns.loc[mask, asset], errors="coerce")
    return gross_return.fillna(0.0)


def _turnover_for_active_assets(active_asset: pd.Series) -> pd.Series:
    invested_assets = sorted(
        asset for asset in set(active_asset.dropna()) if asset != RISK_FREE_ASSET
    )
    if not invested_assets:
        return pd.Series(0.0, index=active_asset.index, dtype=float)

    weights = pd.DataFrame(0.0, index=active_asset.index, columns=invested_assets)
    for asset in invested_assets:
        weights.loc[active_asset.eq(asset), asset] = 1.0
    turnover = weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = 0.0
    return turnover.astype(float)


def _build_trade_frame(daily_equity: pd.DataFrame, spec: QQQTimingStrategySpec) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    active_asset = daily_equity["active_asset"].astype(str)
    previous_asset = active_asset.shift(1)
    changed = previous_asset.notna() & active_asset.ne(previous_asset)
    trades = daily_equity.loc[
        changed,
        [
            "date",
            "strategy_name",
            "risk_on_asset",
            "risk_off_asset",
            "signal_name",
            "signal_date",
            "turnover",
            "transaction_cost",
        ],
    ].copy()
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    trades["previous_asset"] = previous_asset.loc[changed].values
    trades["new_asset"] = active_asset.loc[changed].values
    trades = trades.rename(columns={"transaction_cost": "estimated_transaction_cost"})
    return trades.loc[:, TRADE_COLUMNS].assign(strategy_name=spec.strategy_name)


def _build_defensive_trade_frame(
    daily_equity: pd.DataFrame,
    spec: QQQDefensiveTimingStrategySpec,
) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame(columns=DEFENSIVE_TRADE_COLUMNS)

    active_asset = daily_equity["active_asset"].astype(str)
    previous_asset = active_asset.shift(1)
    changed = previous_asset.notna() & active_asset.ne(previous_asset)
    trades = daily_equity.loc[
        changed,
        [
            "date",
            "strategy_name",
            "family",
            "risk_on_asset",
            "risk_off_asset",
            "signal_name",
            *DEFENSIVE_PARAMETER_COLUMNS,
            "signal_date",
            "turnover",
            "transaction_cost",
        ],
    ].copy()
    if trades.empty:
        return pd.DataFrame(columns=DEFENSIVE_TRADE_COLUMNS)

    trades["previous_asset"] = previous_asset.loc[changed].values
    trades["new_asset"] = active_asset.loc[changed].values
    trades = trades.rename(columns={"transaction_cost": "estimated_transaction_cost"})
    return trades.loc[:, DEFENSIVE_TRADE_COLUMNS].assign(strategy_name=spec.strategy_name)


def _defensive_subperiod_metric_row(
    *,
    strategy_name: str,
    period_name: str,
    period_start: str,
    period_end: str,
    period_daily: pd.DataFrame,
    period_prices: pd.DataFrame,
) -> dict[str, object]:
    returns = pd.to_numeric(period_daily["daily_return"], errors="coerce").fillna(0.0).copy()
    if not returns.empty:
        returns.iloc[0] = 0.0
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan
    max_drawdown = calculate_max_drawdown(equity) if not equity.empty else np.nan

    qqq_daily = pd.to_numeric(period_prices["QQQ"], errors="coerce").pct_change().fillna(0.0)
    spy_daily = pd.to_numeric(period_prices["SPY"], errors="coerce").pct_change().fillna(0.0)
    qqq_equity = (1.0 + qqq_daily).cumprod()
    spy_equity = (1.0 + spy_daily).cumprod()
    qqq_total_return = float(qqq_equity.iloc[-1] - 1.0) if not qqq_equity.empty else np.nan
    spy_total_return = float(spy_equity.iloc[-1] - 1.0) if not spy_equity.empty else np.nan
    qqq_max_drawdown = calculate_max_drawdown(qqq_equity) if not qqq_equity.empty else np.nan
    spy_max_drawdown = calculate_max_drawdown(spy_equity) if not spy_equity.empty else np.nan
    active_asset = period_daily["active_asset"].astype(str)

    return {
        "strategy_name": strategy_name,
        "period_name": period_name,
        "period_start": period_start,
        "period_end": period_end,
        "total_return": total_return,
        "qqq_buy_hold_total_return": qqq_total_return,
        "spy_buy_hold_total_return": spy_total_return,
        "strategy_vs_qqq": total_return - qqq_total_return,
        "strategy_vs_spy": total_return - spy_total_return,
        "max_drawdown": max_drawdown,
        "qqq_max_drawdown": qqq_max_drawdown,
        "spy_max_drawdown": spy_max_drawdown,
        "transaction_count": int(active_asset.ne(active_asset.shift(1)).iloc[1:].sum()),
        "exposure_rate_to_qqq": float(active_asset.eq("QQQ").mean()),
        "exposure_rate_to_spy": float(active_asset.eq("SPY").mean()),
    }


def _strategy_metrics(
    *,
    spec: QQQTimingStrategySpec,
    daily_equity: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, object]:
    returns = pd.to_numeric(daily_equity["daily_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(daily_equity["equity"], errors="coerce")
    trading_days = int(len(daily_equity))
    total_return = float(equity.iloc[-1] - 1.0) if trading_days else np.nan
    annualized_return = _annualized_return(total_return, trading_days)
    daily_volatility = float(returns.std())
    sharpe_like = (
        float(returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_volatility > 0.0 and not np.isnan(daily_volatility)
        else np.nan
    )
    max_drawdown = calculate_max_drawdown(equity)
    qqq_daily = prices["QQQ"].pct_change().fillna(0.0)
    spy_daily = prices["SPY"].pct_change().fillna(0.0)
    qqq_equity = (1.0 + qqq_daily).cumprod()
    spy_equity = (1.0 + spy_daily).cumprod()
    qqq_total_return = float(qqq_equity.iloc[-1] - 1.0)
    spy_total_return = float(spy_equity.iloc[-1] - 1.0)
    qqq_max_drawdown = calculate_max_drawdown(qqq_equity)
    spy_max_drawdown = calculate_max_drawdown(spy_equity)
    transaction_cost = pd.to_numeric(
        daily_equity["transaction_cost"],
        errors="coerce",
    ).fillna(0.0)
    turnover = pd.to_numeric(daily_equity["turnover"], errors="coerce").fillna(0.0)
    active_asset = daily_equity["active_asset"].astype(str)
    risk_off_asset = spec.risk_off_asset or ""
    risk_off_mask = (
        active_asset.ne("QQQ")
        if risk_off_asset == RISK_FREE_ASSET
        else active_asset.eq(risk_off_asset)
    )

    return {
        "strategy_name": spec.strategy_name,
        "risk_on_asset": spec.risk_on_asset,
        "risk_off_asset": spec.risk_off_asset,
        "signal_name": spec.signal_name,
        "start_date": str(pd.Timestamp(daily_equity["date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(daily_equity["date"].iloc[-1]).date()),
        "trading_days": trading_days,
        "exposure_rate_to_qqq": float(active_asset.eq("QQQ").mean()),
        "exposure_rate_to_risk_off": float(risk_off_mask.mean()),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "daily_volatility": daily_volatility,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "worst_1m_return": _worst_period_return(equity, 21),
        "worst_3m_return": _worst_period_return(equity, 63),
        "transaction_count": int(active_asset.ne(active_asset.shift(1)).iloc[1:].sum()),
        "average_turnover": float(turnover.mean()),
        "estimated_transaction_cost": float(transaction_cost.sum()),
        "qqq_buy_hold_total_return": qqq_total_return,
        "spy_buy_hold_total_return": spy_total_return,
        "strategy_vs_qqq": total_return - qqq_total_return,
        "strategy_vs_spy": total_return - spy_total_return,
        "qqq_underperformance": qqq_total_return - total_return,
        "spy_underperformance": spy_total_return - total_return,
        "qqq_buy_hold_max_drawdown": qqq_max_drawdown,
        "spy_buy_hold_max_drawdown": spy_max_drawdown,
        "drawdown_improvement_vs_qqq": max_drawdown - qqq_max_drawdown,
        "return_to_drawdown": _return_to_drawdown(total_return, max_drawdown),
        "qqq_buy_hold_return_to_drawdown": _return_to_drawdown(
            qqq_total_return,
            qqq_max_drawdown,
        ),
    }


def _defensive_strategy_metrics(
    *,
    spec: QQQDefensiveTimingStrategySpec,
    daily_equity: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, object]:
    base_metrics = _strategy_metrics(
        spec=spec,
        daily_equity=daily_equity,
        prices=prices,
    )
    return_to_drawdown = _as_finite_float(base_metrics["return_to_drawdown"])
    qqq_return_to_drawdown = _as_finite_float(
        base_metrics["qqq_buy_hold_return_to_drawdown"]
    )
    strategy_vs_qqq = _as_finite_float(base_metrics["strategy_vs_qqq"])
    drawdown_improvement = _as_finite_float(base_metrics["drawdown_improvement_vs_qqq"])
    metrics = {
        **base_metrics,
        "family": spec.family,
        **_defensive_parameter_values(spec),
        "beats_qqq_return": strategy_vs_qqq > 0.0,
        "beats_qqq_return_to_drawdown": return_to_drawdown > qqq_return_to_drawdown,
        "materially_improves_qqq_drawdown": (
            drawdown_improvement >= MATERIAL_DRAWDOWN_IMPROVEMENT
        ),
    }
    return {column: metrics.get(column, np.nan) for column in DEFENSIVE_RESULT_COLUMNS}


def _defensive_parameter_values(
    spec: QQQDefensiveTimingStrategySpec,
) -> dict[str, object]:
    return {
        "slow_sma_days": spec.slow_sma_days if spec.slow_sma_days is not None else np.nan,
        "fast_sma_days": spec.fast_sma_days if spec.fast_sma_days is not None else np.nan,
        "buffer_pct": spec.buffer_pct if spec.buffer_pct is not None else np.nan,
        "min_hold_days": spec.min_hold_days,
        "return_lookback_days": (
            spec.return_lookback_days if spec.return_lookback_days is not None else np.nan
        ),
        "volatility_lookback_days": (
            spec.volatility_lookback_days
            if spec.volatility_lookback_days is not None
            else np.nan
        ),
        "volatility_percentile_window_days": (
            spec.volatility_percentile_window_days
            if spec.volatility_percentile_window_days is not None
            else np.nan
        ),
        "volatility_percentile_threshold": (
            spec.volatility_percentile_threshold
            if spec.volatility_percentile_threshold is not None
            else np.nan
        ),
        "drawdown_lookback_days": (
            spec.drawdown_lookback_days if spec.drawdown_lookback_days is not None else np.nan
        ),
        "drawdown_threshold": (
            spec.drawdown_threshold if spec.drawdown_threshold is not None else np.nan
        ),
        "regime_score_threshold": (
            spec.regime_score_threshold if spec.regime_score_threshold is not None else np.nan
        ),
    }


def _as_finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric


def _annualized_return(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1.0)


def _worst_period_return(equity: pd.Series, periods: int) -> float:
    period_returns = equity.pct_change(periods=periods)
    if period_returns.dropna().empty:
        return np.nan
    return float(period_returns.min())


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown) or max_drawdown == 0.0:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _strategy_has_missing_price_rows(prices: pd.DataFrame, spec: QQQTimingStrategySpec) -> bool:
    assets = [
        asset
        for asset in (spec.risk_on_asset, spec.risk_off_asset)
        if asset and asset != RISK_FREE_ASSET
    ]
    return any(asset not in prices.columns or prices[asset].isna().any() for asset in assets)


__all__ = [
    "BENCHMARK_STRATEGY_NAMES",
    "DAILY_EQUITY_COLUMNS",
    "DEFENSIVE_DAILY_EQUITY_COLUMNS",
    "DEFENSIVE_PARAMETER_COLUMNS",
    "DEFENSIVE_RESULT_COLUMNS",
    "DEFENSIVE_SUBPERIOD_RESULT_COLUMNS",
    "DEFENSIVE_TIMING_SUBPERIODS",
    "DEFENSIVE_TIMING_FAMILIES",
    "DEFENSIVE_TRADE_COLUMNS",
    "NARROW_DEFENSIVE_TIMING_FAMILIES",
    "OPTIONAL_RISK_OFF_SYMBOLS",
    "QQQDefensiveTimingStrategySpec",
    "QQQDefensiveTimingNarrowSweepResult",
    "QQQDefensiveTimingSelectedReplayResult",
    "QQQDefensiveTimingSweepResult",
    "QQQTimingStrategySpec",
    "QQQTimingStudyResult",
    "REQUIRED_SYMBOLS",
    "RESULT_COLUMNS",
    "SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME",
    "TRADE_COLUMNS",
    "available_symbol_summary",
    "build_qqq_defensive_timing_signal",
    "build_qqq_timing_indicators",
    "build_qqq_timing_signal",
    "calculate_qqq_defensive_timing_subperiod_results",
    "default_qqq_defensive_timing_narrow_sweep_specs",
    "default_qqq_defensive_timing_selected_replay_spec",
    "default_qqq_defensive_timing_sweep_specs",
    "default_qqq_timing_strategy_specs",
    "prepare_qqq_timing_prices",
    "run_qqq_defensive_timing_narrow_sweep",
    "run_qqq_defensive_timing_selected_replay",
    "run_qqq_defensive_timing_strategy",
    "run_qqq_defensive_timing_sweep",
    "run_qqq_timing_strategy",
    "run_qqq_timing_study",
]
