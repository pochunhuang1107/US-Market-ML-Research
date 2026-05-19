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


@dataclass(frozen=True)
class QQQTimingStrategySpec:
    """A single QQQ timing or benchmark strategy definition."""

    strategy_name: str
    risk_on_asset: str
    risk_off_asset: str
    signal_name: str
    is_benchmark: bool = False


@dataclass(frozen=True)
class QQQTimingStudyResult:
    """Outputs from a QQQ timing baseline run."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
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


def _validate_price_symbols(prices: pd.DataFrame, symbols: Sequence[str]) -> None:
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns for symbols: {', '.join(missing)}")


def _positive_flag(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out.loc[values.notna()] = (values.loc[values.notna()] > 0.0).astype(float)
    return out


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
    "OPTIONAL_RISK_OFF_SYMBOLS",
    "QQQTimingStrategySpec",
    "QQQTimingStudyResult",
    "REQUIRED_SYMBOLS",
    "RESULT_COLUMNS",
    "TRADE_COLUMNS",
    "available_symbol_summary",
    "build_qqq_timing_indicators",
    "build_qqq_timing_signal",
    "default_qqq_timing_strategy_specs",
    "prepare_qqq_timing_prices",
    "run_qqq_timing_strategy",
    "run_qqq_timing_study",
]
