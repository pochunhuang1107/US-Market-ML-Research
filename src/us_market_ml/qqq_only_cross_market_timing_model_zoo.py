"""QQQ-only cross-market timing model-zoo research utilities.

This module is deliberately research-only and local-only. It builds features
from QQQ, cross-market ETFs, and reference macro/volatility series, but the
simulated portfolio can only hold QQQ and cash. It does not export production
signals, create model cards, access S3/cloud storage, or mutate upstream data.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown

TRADING_DAYS_PER_YEAR = 252
OUTPUT_PREFIX = "qqq_only_cross_market_timing_model_zoo"
PRICE_MODE = "tiingo_adjusted"
DATASET_SOURCE_LABEL = "tiingo_only"

EXPECTED_REFERENCE_SYMBOLS: tuple[str, ...] = (
    "DFF",
    "DGS2",
    "T10Y2Y",
    "VIX",
    "VIX9D",
    "VXEEM",
)
EXCLUDED_REFERENCE_SYMBOLS: tuple[str, ...] = ("DGS10",)
DEFAULT_REFERENCE_SERIES_GLOBS: tuple[str, ...] = (
    "/Users/casper/Projects/Research/Market Data/data/research/reference_series/**/*.parquet",
    "/Users/casper/Projects/Research/Market Data/data/curated/reference_series/**/*.parquet",
    "/Users/casper/Projects/Research/Market Data/data/reference_series/**/*.parquet",
)

CANDIDATE_FEATURE_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "IWM",
    "DIA",
    "VTI",
    "XLK",
    "XLY",
    "XLP",
    "XLF",
    "XLE",
    "XLU",
    "XLV",
    "XLRE",
    "TLT",
    "IEF",
    "HYG",
    "LQD",
    "GLD",
    "SLV",
    "MTUM",
    "QUAL",
    "VUG",
    "VTV",
    "ARKK",
)
SECTOR_SYMBOLS: tuple[str, ...] = (
    "XLK",
    "XLY",
    "XLP",
    "XLF",
    "XLE",
    "XLU",
    "XLV",
    "XLRE",
)

FEATURE_SET_NAMES: tuple[str, ...] = (
    "qqq_self",
    "qqq_self_cross_market",
    "qqq_self_cross_market_reference",
)
EXPOSURE_MODES: tuple[str, ...] = (
    "binary_0_100",
    "partial_50_100",
    "ternary_0_50_100",
)
REBALANCE_FREQUENCIES: tuple[str, ...] = ("weekly", "monthly")
TRANSACTION_COST_BPS_VALUES: tuple[float, ...] = (0.0, 5.0, 10.0)
PROBABILITY_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60)
REGRESSION_THRESHOLDS: tuple[float, ...] = (0.0, 0.002, 0.005)
MIN_HOLDING_DAYS_VALUES: tuple[int, ...] = (0, 10, 20)
SWITCH_MARGIN_VALUES: tuple[float, ...] = (0.0, 0.002, 0.005)

CLASSIFICATION_TARGETS: frozenset[str] = frozenset(
    {"risk_on_20d", "risk_on_63d", "drawdown_safe_20d"}
)
REGRESSION_TARGETS: frozenset[str] = frozenset(
    {
        "qqq_forward_return_20d",
        "qqq_forward_return_63d",
        "return_drawdown_score_20d",
    }
)

BASELINE_RULES: tuple[str, ...] = (
    "qqq_buy_hold",
    "cash",
    "qqq_sma200_timing",
    "qqq_sma50_sma200_trend",
    "vix_defensive_25",
    "vix_defensive_30",
    "trend_vix_25",
    "trend_vix_30",
)

RESULT_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "target_horizon",
    "feature_set_name",
    "feature_count",
    "exposure_mode",
    "rebalance_frequency",
    "transaction_cost_bps",
    "threshold",
    "min_holding_days",
    "switch_margin",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "daily_volatility",
    "sharpe_like",
    "return_to_drawdown",
    "qqq_buy_hold_total_return",
    "qqq_buy_hold_max_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "strategy_vs_qqq",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown_vs_qqq",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "average_exposure",
    "cash_days_pct",
    "trade_count",
    "average_holding_days",
    "estimated_transaction_cost",
    "production_ready",
    "decision_label",
    "high_consistency",
    "very_high_consistency",
    "spy_buy_hold_total_return",
    "spy_buy_hold_max_drawdown",
)

DAILY_EQUITY_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "feature_set_name",
    "exposure_mode",
    "rebalance_frequency",
    "transaction_cost_bps",
    "target_exposure",
    "effective_exposure",
    "qqq_weight",
    "cash_weight",
    "qqq_return",
    "gross_return",
    "transaction_cost",
    "turnover",
    "daily_return",
    "equity",
    "qqq_buy_hold_equity",
)

DAILY_DRAWDOWN_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "feature_set_name",
    "equity",
    "drawdown",
    "qqq_buy_hold_equity",
    "qqq_buy_hold_drawdown",
)

EXPOSURE_LOG_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_id",
    "strategy_name",
    "model_type",
    "training_mode",
    "target_name",
    "feature_set_name",
    "exposure_mode",
    "rebalance_frequency",
    "target_exposure",
    "effective_exposure",
    "qqq_weight",
    "cash_weight",
    "held_assets",
    "raw_score",
    "threshold",
)

FEATURE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "feature_family",
    "source_symbol",
    "available_rows",
    "missing_pct",
    "first_valid_date",
    "last_valid_date",
    "used_in_feature_sets",
)


class SkippedQQQOnlyRun(RuntimeError):
    """Raised when a config should be checkpointed as skipped."""


@dataclass(frozen=True)
class ReferenceSeriesDiscovery:
    """Reference-series path and symbol discovery details."""

    selected_glob: str
    candidate_globs: tuple[str, ...]
    matched_paths: tuple[Path, ...]
    expected_symbols: tuple[str, ...]
    found_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    explicit_glob: bool


@dataclass(frozen=True)
class QQQOnlyRunSpec:
    """One baseline or model-driven QQQ/cash timing configuration."""

    run_key: str
    strategy_id: str
    strategy_name: str
    model_type: str
    training_mode: str
    target_name: str
    target_horizon: int
    feature_set_name: str
    exposure_mode: str
    rebalance_frequency: str
    transaction_cost_bps: float
    threshold: float
    min_holding_days: int
    switch_margin: float
    is_baseline: bool = False
    baseline_rule: str | None = None
    rebalance_weekday: str = "FRI"
    random_state: int = 42
    n_estimators: int = 60
    max_depth: int | None = None
    min_samples_leaf: int = 5


@dataclass(frozen=True)
class QQQOnlyProfile:
    """Named sweep profile."""

    name: str
    purpose: str
    specs: tuple[QQQOnlyRunSpec, ...]


@dataclass(frozen=True)
class QQQOnlyFeatureDataset:
    """Feature/target frame and metadata used by the QQQ-only model zoo."""

    frame: pd.DataFrame
    feature_summary: pd.DataFrame
    feature_sets: dict[str, list[str]]
    available_etf_symbols: tuple[str, ...]
    available_reference_symbols: tuple[str, ...]
    missing_reference_symbols: tuple[str, ...]


@dataclass(frozen=True)
class QQQOnlySpecOutput:
    """Outputs for one completed spec."""

    result: dict[str, object]
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    exposure_log: pd.DataFrame


def discover_reference_series(
    reference_series_glob: str | None = None,
    *,
    expected_symbols: Sequence[str] = EXPECTED_REFERENCE_SYMBOLS,
    default_globs: Sequence[str] = DEFAULT_REFERENCE_SERIES_GLOBS,
) -> ReferenceSeriesDiscovery:
    """Discover local reference-series files without reading cloud storage."""

    explicit = reference_series_glob is not None and str(reference_series_glob).strip() != ""
    candidates = (str(reference_series_glob),) if explicit else tuple(default_globs)
    selected = candidates[0] if candidates else ""
    matched: list[Path] = []
    for candidate in candidates:
        paths = _glob_supported_files(candidate)
        if paths:
            selected = candidate
            matched = paths
            break
    found = _discover_reference_symbols_from_paths(matched)
    expected = tuple(symbol.upper() for symbol in expected_symbols)
    missing = tuple(symbol for symbol in expected if symbol not in found)
    return ReferenceSeriesDiscovery(
        selected_glob=selected,
        candidate_globs=tuple(candidates),
        matched_paths=tuple(matched),
        expected_symbols=expected,
        found_symbols=tuple(sorted(found)),
        missing_symbols=missing,
        explicit_glob=explicit,
    )


def load_reference_series(
    path_or_glob: str | Path,
    *,
    expected_symbols: Sequence[str] = EXPECTED_REFERENCE_SYMBOLS,
) -> pd.DataFrame:
    """Load and pivot local reference series into date-indexed wide form.

    Supported inputs are CSV/parquet files with either a ``symbol`` column or a
    Hive-style path segment such as ``symbol=VIX``. Future observations are
    never backfilled; callers should align with ``reindex(...).ffill()``.
    """

    paths = _glob_supported_files(str(path_or_glob))
    if not paths:
        raise FileNotFoundError(f"Reference series glob matched no local files: {path_or_glob}")

    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_reference_file(path)
        if frame.empty:
            continue
        rows.append(frame)
    if not rows:
        raise ValueError(f"Reference series files contained no usable rows: {path_or_glob}")

    long = pd.concat(rows, ignore_index=True)
    long["symbol"] = long["symbol"].astype("string").str.upper()
    long = long.loc[~long["symbol"].isin(EXCLUDED_REFERENCE_SYMBOLS)].copy()
    expected = [symbol.upper() for symbol in expected_symbols]
    long = long.loc[long["symbol"].isin(expected)].copy()
    if long.empty:
        raise ValueError(
            "Reference series files did not contain any expected symbols: " + ", ".join(expected)
        )

    pivot = (
        long.pivot_table(index="date", columns="symbol", values="value", aggfunc="last")
        .sort_index(kind="mergesort")
        .rename_axis(columns=None)
    )
    for symbol in expected:
        if symbol not in pivot.columns:
            pivot[symbol] = np.nan
    return pivot.loc[:, expected]


def build_qqq_only_feature_dataset(
    market_data: pd.DataFrame,
    reference_series: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    candidate_feature_symbols: Sequence[str] = CANDIDATE_FEATURE_SYMBOLS,
    expected_reference_symbols: Sequence[str] = EXPECTED_REFERENCE_SYMBOLS,
) -> QQQOnlyFeatureDataset:
    """Build leakage-safe same-date features and future QQQ labels."""

    prices = _prepare_price_matrix(market_data, end_date=end_date)
    if "QQQ" not in prices.columns:
        raise ValueError("QQQ is required for the QQQ-only timing model zoo.")
    if "SPY" not in prices.columns:
        raise ValueError("SPY is required as a reference benchmark/relative feature.")

    dates = prices.index
    prices = prices.reindex(dates).ffill()
    qqq = pd.to_numeric(prices["QQQ"], errors="coerce")
    qqq_returns = qqq.pct_change()
    features = pd.DataFrame(index=dates)
    families: dict[str, tuple[str, str]] = {}

    def add_feature(name: str, values: pd.Series, family: str, source: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            features[name] = pd.to_numeric(values.reindex(dates), errors="coerce")
        families[name] = (family, source)

    for lookback in (5, 20, 60, 126):
        add_feature(f"qqq_return_{lookback}d", qqq.pct_change(lookback), "qqq_self", "QQQ")
    for window in (20, 50, 200):
        sma = qqq.rolling(window=window, min_periods=window).mean()
        add_feature(f"qqq_close_to_sma_{window}", qqq / sma - 1.0, "qqq_self", "QQQ")
        if window in {50, 200}:
            features[f"qqq_sma_{window}"] = sma
            families[f"qqq_sma_{window}"] = ("qqq_self_helper", "QQQ")
    add_feature(
        "qqq_sma_50_to_sma_200",
        features["qqq_sma_50"] / features["qqq_sma_200"] - 1.0,
        "qqq_self",
        "QQQ",
    )
    for window in (20, 60):
        add_feature(
            f"qqq_realized_vol_{window}d",
            qqq_returns.rolling(window=window, min_periods=window).std()
            * math.sqrt(TRADING_DAYS_PER_YEAR),
            "qqq_self",
            "QQQ",
        )
        add_feature(
            f"qqq_rolling_drawdown_{window}d",
            qqq / qqq.rolling(window=window, min_periods=window).max() - 1.0,
            "qqq_self",
            "QQQ",
        )
    add_feature("qqq_rsi_14", _rsi(qqq, window=14), "qqq_self", "QQQ")

    available_etfs = tuple(
        symbol
        for symbol in dict.fromkeys(symbol.upper() for symbol in candidate_feature_symbols)
        if symbol in prices.columns and symbol != "QQQ"
    )
    for symbol in available_etfs:
        close = pd.to_numeric(prices[symbol], errors="coerce")
        returns = close.pct_change()
        prefix = symbol.lower()
        add_feature(f"{prefix}_return_20d", close.pct_change(20), "cross_market_etf", symbol)
        add_feature(f"{prefix}_return_60d", close.pct_change(60), "cross_market_etf", symbol)
        sma50 = close.rolling(window=50, min_periods=50).mean()
        sma200 = close.rolling(window=200, min_periods=200).mean()
        add_feature(f"{prefix}_close_to_sma_50", close / sma50 - 1.0, "cross_market_etf", symbol)
        add_feature(
            f"{prefix}_close_to_sma_200",
            close / sma200 - 1.0,
            "cross_market_etf",
            symbol,
        )
        add_feature(
            f"{prefix}_realized_vol_20d",
            returns.rolling(window=20, min_periods=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR),
            "cross_market_etf",
            symbol,
        )
        add_feature(
            f"{prefix}_drawdown_60d",
            close / close.rolling(window=60, min_periods=60).max() - 1.0,
            "cross_market_etf",
            symbol,
        )

    for left, right, lookback, name in (
        ("QQQ", "SPY", 20, "qqq_spy_return_ratio_20d"),
        ("QQQ", "SPY", 60, "qqq_spy_return_ratio_60d"),
        ("IWM", "SPY", 20, "iwm_spy_return_ratio_20d"),
        ("XLK", "SPY", 20, "xlk_spy_return_ratio_20d"),
        ("XLY", "XLP", 20, "xly_xlp_return_ratio_20d"),
        ("HYG", "LQD", 20, "hyg_lqd_return_ratio_20d"),
        ("TLT", "SPY", 20, "tlt_spy_return_ratio_20d"),
        ("GLD", "SPY", 20, "gld_spy_return_ratio_20d"),
        ("XLE", "SPY", 20, "xle_spy_return_ratio_20d"),
    ):
        if left in prices.columns and right in prices.columns:
            relative = (prices[left] / prices[right]).pct_change(lookback)
            add_feature(name, relative, "relative_regime", f"{left}/{right}")

    present_sectors = [symbol for symbol in SECTOR_SYMBOLS if symbol in prices.columns]
    if len(present_sectors) >= 3:
        sector_returns = prices[present_sectors].pct_change(20)
        add_feature(
            "sector_dispersion_20d",
            sector_returns.std(axis=1, skipna=True),
            "relative_regime",
            "sectors",
        )

    reference = _align_reference_series(
        reference_series,
        dates=dates,
        expected_symbols=expected_reference_symbols,
    )
    available_refs = tuple(
        symbol for symbol in expected_reference_symbols if symbol in reference.columns
    )
    for symbol in available_refs:
        level = pd.to_numeric(reference[symbol], errors="coerce")
        prefix = symbol.lower()
        add_feature(f"{prefix}_level", level, "reference_macro_vol", symbol)
        for lookback in (5, 20, 60):
            add_feature(
                f"{prefix}_change_{lookback}d",
                level.diff(lookback),
                "reference_macro_vol",
                symbol,
            )
        sma20 = level.rolling(window=20, min_periods=20).mean()
        add_feature(f"{prefix}_sma_20_gap", level - sma20, "reference_macro_vol", symbol)
        rolling_mean = level.rolling(window=60, min_periods=20).mean()
        rolling_std = level.rolling(window=60, min_periods=20).std()
        add_feature(
            f"{prefix}_zscore_60d",
            (level - rolling_mean) / rolling_std.replace(0.0, np.nan),
            "reference_macro_vol",
            symbol,
        )

    if "VIX" in reference.columns:
        add_feature("vix_above_20", (reference["VIX"] > 20.0).astype(float), "regime_flag", "VIX")
        add_feature("vix_above_30", (reference["VIX"] > 30.0).astype(float), "regime_flag", "VIX")
    if {"VIX", "VIX9D"}.issubset(reference.columns):
        add_feature(
            "vix9d_above_vix",
            (reference["VIX9D"] > reference["VIX"]).astype(float),
            "regime_flag",
            "VIX9D/VIX",
        )
    if "T10Y2Y" in reference.columns:
        add_feature(
            "yield_curve_inverted",
            (reference["T10Y2Y"] < 0.0).astype(float),
            "regime_flag",
            "T10Y2Y",
        )
    if "DGS2" in reference.columns:
        add_feature(
            "dgs2_rising_20d",
            (reference["DGS2"].diff(20) > 0.0).astype(float),
            "regime_flag",
            "DGS2",
        )
    if "DFF" in reference.columns:
        add_feature(
            "dff_rising_60d",
            (reference["DFF"].diff(60) > 0.0).astype(float),
            "regime_flag",
            "DFF",
        )
    if "VXEEM" in reference.columns and "vxeem_zscore_60d" in features.columns:
        add_feature(
            "vxeem_stress_high",
            (features["vxeem_zscore_60d"] > 1.0).astype(float),
            "regime_flag",
            "VXEEM",
        )
    if "hyg_lqd_return_ratio_20d" in features.columns:
        add_feature(
            "credit_stress_hyg_lqd_weak",
            (features["hyg_lqd_return_ratio_20d"] < 0.0).astype(float),
            "regime_flag",
            "HYG/LQD",
        )

    frame = features.copy()
    frame["date"] = frame.index
    frame["qqq_close"] = qqq
    frame["spy_close"] = prices["SPY"]
    frame["qqq_daily_return"] = qqq.pct_change().fillna(0.0)
    if "SPY" in prices.columns:
        frame["spy_daily_return"] = prices["SPY"].pct_change().fillna(0.0)
    for horizon in (20, 63):
        forward_return = qqq.shift(-horizon) / qqq - 1.0
        frame[f"qqq_forward_return_{horizon}d"] = forward_return
        frame[f"risk_on_{horizon}d"] = np.where(
            forward_return.notna(),
            forward_return > 0.0,
            np.nan,
        )
        frame[f"target_available_date_{horizon}d"] = pd.Series(dates, index=dates).shift(-horizon)

    future_drawdown = _future_drawdown(qqq, horizon=20)
    frame["drawdown_safe_20d"] = np.where(
        future_drawdown.notna(),
        future_drawdown > -0.10,
        np.nan,
    )
    frame["return_drawdown_score_20d"] = frame["qqq_forward_return_20d"] / (
        future_drawdown.abs().clip(lower=0.01)
    )

    qqq_self_cols = _feature_columns_by_family(families, {"qqq_self"})
    cross_cols = _feature_columns_by_family(
        families,
        {"qqq_self", "cross_market_etf", "relative_regime"},
    )
    reference_cols = _feature_columns_by_family(
        families,
        {
            "qqq_self",
            "cross_market_etf",
            "relative_regime",
            "reference_macro_vol",
            "regime_flag",
        },
    )
    feature_sets = {
        "qqq_self": qqq_self_cols,
        "qqq_self_cross_market": cross_cols,
        "qqq_self_cross_market_reference": reference_cols,
    }
    feature_summary = _build_feature_summary(
        frame=frame,
        families=families,
        feature_sets=feature_sets,
    )

    expected_refs = tuple(symbol.upper() for symbol in expected_reference_symbols)
    found_refs = tuple(symbol for symbol in expected_refs if symbol in reference.columns)
    missing_refs = tuple(symbol for symbol in expected_refs if symbol not in found_refs)
    return QQQOnlyFeatureDataset(
        frame=frame.reset_index(drop=True),
        feature_summary=feature_summary,
        feature_sets=feature_sets,
        available_etf_symbols=tuple(prices.columns.astype(str)),
        available_reference_symbols=found_refs,
        missing_reference_symbols=missing_refs,
    )


def build_qqq_only_sweep_specs(profile_name: str) -> tuple[QQQOnlyRunSpec, ...]:
    """Build deterministic baseline and model specs for a sweep profile."""

    profile = profile_name.lower()
    if profile == "smoke":
        costs = (5.0,)
        specs = [
            _model_spec(
                model_type="logistic_regression",
                training_mode="expanding",
                target_name="risk_on_20d",
                feature_set_name="qqq_self_cross_market_reference",
                exposure_mode="binary_0_100",
                rebalance_frequency="weekly",
                transaction_cost_bps=5.0,
                threshold=0.55,
                min_holding_days=0,
                switch_margin=0.0,
            )
        ]
        return (*_baseline_specs(costs), *specs)

    if profile == "core":
        model_blueprints = (
            ("logistic_regression", "expanding", "risk_on_20d", "binary_0_100", 0.55),
            ("ridge", "expanding", "qqq_forward_return_20d", "partial_50_100", 0.002),
            ("elasticnet", "rolling_3y", "qqq_forward_return_63d", "ternary_0_50_100", 0.002),
            ("random_forest", "rolling_5y", "risk_on_20d", "binary_0_100", 0.55),
            ("extra_trees", "expanding", "return_drawdown_score_20d", "ternary_0_50_100", 0.0),
            (
                "hist_gradient_boosting",
                "recency_weighted_expanding",
                "risk_on_63d",
                "partial_50_100",
                0.55,
            ),
        )
        specs = [
            _model_spec(
                model_type=model_type,
                training_mode=training_mode,
                target_name=target_name,
                feature_set_name="qqq_self_cross_market_reference",
                exposure_mode=exposure_mode,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=cost,
                threshold=threshold,
                min_holding_days=10 if rebalance_frequency == "weekly" else 0,
                switch_margin=0.002,
            )
            for model_type, training_mode, target_name, exposure_mode, threshold in model_blueprints
            for rebalance_frequency in REBALANCE_FREQUENCIES
            for cost in TRANSACTION_COST_BPS_VALUES
        ]
        return (*_baseline_specs(TRANSACTION_COST_BPS_VALUES), *specs)

    if profile == "full":
        model_types = (
            "logistic_regression",
            "ridge",
            "elasticnet",
            "lasso",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "simple_mlp",
        )
        training_modes = (
            "expanding",
            "rolling_3y",
            "rolling_5y",
            "frozen_base_1999_2018",
            "recency_weighted_expanding",
        )
        feature_sets = ("qqq_self_cross_market", "qqq_self_cross_market_reference")
        specs: list[QQQOnlyRunSpec] = []
        counter = 0
        for model_type in model_types:
            targets = (
                ("risk_on_20d", "risk_on_63d")
                if model_type == "logistic_regression"
                else (
                    "risk_on_20d",
                    "qqq_forward_return_20d",
                    "qqq_forward_return_63d",
                    "return_drawdown_score_20d",
                )
            )
            for training_mode in training_modes:
                for target_name in targets:
                    if not _model_target_is_supported(model_type, target_name):
                        continue
                    for feature_set_name in feature_sets:
                        for rebalance_frequency in REBALANCE_FREQUENCIES:
                            exposure = EXPOSURE_MODES[counter % len(EXPOSURE_MODES)]
                            cost = TRANSACTION_COST_BPS_VALUES[
                                counter % len(TRANSACTION_COST_BPS_VALUES)
                            ]
                            threshold_values = (
                                PROBABILITY_THRESHOLDS
                                if target_name in CLASSIFICATION_TARGETS
                                else REGRESSION_THRESHOLDS
                            )
                            threshold = threshold_values[counter % len(threshold_values)]
                            min_hold = MIN_HOLDING_DAYS_VALUES[
                                counter % len(MIN_HOLDING_DAYS_VALUES)
                            ]
                            margin = SWITCH_MARGIN_VALUES[counter % len(SWITCH_MARGIN_VALUES)]
                            specs.append(
                                _model_spec(
                                    model_type=model_type,
                                    training_mode=training_mode,
                                    target_name=target_name,
                                    feature_set_name=feature_set_name,
                                    exposure_mode=exposure,
                                    rebalance_frequency=rebalance_frequency,
                                    transaction_cost_bps=cost,
                                    threshold=threshold,
                                    min_holding_days=min_hold,
                                    switch_margin=margin,
                                )
                            )
                            counter += 1
                            if len(specs) >= 300:
                                return (*_baseline_specs(TRANSACTION_COST_BPS_VALUES), *specs)
        return (*_baseline_specs(TRANSACTION_COST_BPS_VALUES), *specs)

    raise ValueError(f"Unsupported SWEEP_PROFILE={profile_name!r}; expected smoke, core, or full.")


def run_qqq_only_spec(
    dataset: QQQOnlyFeatureDataset,
    spec: QQQOnlyRunSpec,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    prediction_cache: dict[tuple[object, ...], pd.DataFrame] | None = None,
    min_train_rows: int = 252,
) -> QQQOnlySpecOutput:
    """Run one QQQ/cash timing config and return result/equity artifacts."""

    frame = dataset.frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if spec.is_baseline:
        raw = baseline_exposure(frame, spec.baseline_rule or spec.strategy_name)
        predictions = pd.DataFrame(
            {
                "date": frame["date"],
                "raw_score": raw,
                "target_exposure": raw.clip(0.0, 1.0),
            }
        )
    else:
        cache = prediction_cache if prediction_cache is not None else {}
        cache_key = (
            spec.model_type,
            spec.training_mode,
            spec.target_name,
            spec.feature_set_name,
            spec.rebalance_frequency,
            spec.rebalance_weekday,
            spec.random_state,
            spec.n_estimators,
            spec.max_depth,
            spec.min_samples_leaf,
            str(pd.Timestamp(start_date).date()),
            str(pd.Timestamp(end_date).date()),
        )
        if cache_key not in cache:
            cache[cache_key] = walk_forward_predictions(
                dataset,
                spec,
                start_date=start_date,
                end_date=end_date,
                min_train_rows=min_train_rows,
            )
        predictions = build_exposure_schedule(
            cache[cache_key],
            frame_dates=pd.DatetimeIndex(frame["date"]),
            spec=spec,
            start_date=start_date,
            end_date=end_date,
        )

    output = simulate_qqq_cash_strategy(
        frame,
        predictions,
        spec=spec,
        start_date=start_date,
        end_date=end_date,
    )
    output.result["feature_count"] = (
        0 if spec.is_baseline else len(dataset.feature_sets.get(spec.feature_set_name, []))
    )
    return output


def baseline_exposure(frame: pd.DataFrame, baseline_rule: str) -> pd.Series:
    """Return same-date target exposure for one QQQ/cash baseline rule."""

    rule = baseline_rule.lower()
    if rule == "qqq_buy_hold":
        return pd.Series(1.0, index=frame.index, dtype=float)
    if rule == "cash":
        return pd.Series(0.0, index=frame.index, dtype=float)
    if rule == "qqq_sma200_timing":
        return (frame["qqq_close"] > frame["qqq_sma_200"]).fillna(False).astype(float)
    if rule == "qqq_sma50_sma200_trend":
        return (frame["qqq_sma_50"] > frame["qqq_sma_200"]).fillna(False).astype(float)
    if rule == "vix_defensive_25":
        return (frame["vix_level"] < 25.0).fillna(False).astype(float)
    if rule == "vix_defensive_30":
        return (frame["vix_level"] < 30.0).fillna(False).astype(float)
    if rule == "trend_vix_25":
        return (
            ((frame["qqq_close"] > frame["qqq_sma_200"]) & (frame["vix_level"] < 25.0))
            .fillna(False)
            .astype(float)
        )
    if rule == "trend_vix_30":
        return (
            ((frame["qqq_close"] > frame["qqq_sma_200"]) & (frame["vix_level"] < 30.0))
            .fillna(False)
            .astype(float)
        )
    raise ValueError(f"Unsupported QQQ-only baseline rule: {baseline_rule!r}")


def walk_forward_predictions(
    dataset: QQQOnlyFeatureDataset,
    spec: QQQOnlyRunSpec,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    min_train_rows: int = 252,
) -> pd.DataFrame:
    """Generate leakage-safe walk-forward predictions on rebalance dates."""

    if spec.target_name in CLASSIFICATION_TARGETS and not _model_supports_classification(
        spec.model_type
    ):
        raise SkippedQQQOnlyRun(
            "Model "
            f"{spec.model_type!r} does not support classification target "
            f"{spec.target_name!r}."
        )
    if spec.target_name in REGRESSION_TARGETS and not _model_supports_regression(spec.model_type):
        raise SkippedQQQOnlyRun(
            f"Model {spec.model_type!r} does not support regression target {spec.target_name!r}."
        )
    if spec.training_mode == "recency_weighted_expanding" and spec.model_type == "simple_mlp":
        raise SkippedQQQOnlyRun("simple_mlp sample_weight support is unavailable.")

    frame = dataset.frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    feature_cols = list(dataset.feature_sets.get(spec.feature_set_name, []))
    if not feature_cols:
        raise SkippedQQQOnlyRun(f"No feature columns are available for {spec.feature_set_name}.")
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise SkippedQQQOnlyRun("Feature columns missing from dataset: " + ", ".join(missing[:20]))
    if spec.target_name not in frame.columns:
        raise SkippedQQQOnlyRun(f"Target column is missing: {spec.target_name}")

    rebalance_dates = rebalance_dates_for_frequency(
        pd.DatetimeIndex(frame["date"]),
        start_date=start_date,
        end_date=end_date,
        frequency=spec.rebalance_frequency,
        weekday=spec.rebalance_weekday,
    )
    if len(rebalance_dates) == 0:
        raise SkippedQQQOnlyRun("No rebalance dates were available.")

    predictions: list[dict[str, object]] = []
    for predict_date in rebalance_dates:
        train_df = _training_frame_for_prediction(
            frame,
            predict_date=predict_date,
            spec=spec,
        )
        selected_features = _select_usable_features(train_df, feature_cols)
        if len(train_df) < min_train_rows or not selected_features:
            continue
        prediction_row = frame.loc[frame["date"].eq(predict_date)].tail(1)
        if prediction_row.empty:
            continue
        score = _fit_predict_score(
            train_df=train_df,
            prediction_row=prediction_row,
            feature_cols=selected_features,
            spec=spec,
        )
        predictions.append(
            {
                "date": predict_date,
                "raw_score": score,
                "selected_feature_count": len(selected_features),
                "train_rows": len(train_df),
            }
        )

    if not predictions:
        raise SkippedQQQOnlyRun("No predictions were generated for the spec.")
    return pd.DataFrame(predictions).sort_values("date", kind="mergesort")


def build_exposure_schedule(
    predictions: pd.DataFrame,
    *,
    frame_dates: pd.DatetimeIndex,
    spec: QQQOnlyRunSpec,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Convert rebalance-date scores to a daily target-exposure schedule."""

    if predictions.empty:
        raise SkippedQQQOnlyRun("Cannot build exposure schedule from empty predictions.")
    eval_dates = _evaluation_dates(frame_dates, start_date=start_date, end_date=end_date)
    if len(eval_dates) == 0:
        raise SkippedQQQOnlyRun("No evaluation dates were available.")

    rebalance = predictions.copy()
    rebalance["date"] = pd.to_datetime(rebalance["date"])
    rebalance = rebalance.loc[rebalance["date"].isin(eval_dates)].sort_values(
        "date",
        kind="mergesort",
    )
    if rebalance.empty:
        raise SkippedQQQOnlyRun("No prediction rows overlap the evaluation window.")

    current_exposure: float | None = None
    last_change_date: pd.Timestamp | None = None
    rows: list[dict[str, object]] = []
    for row in rebalance.itertuples(index=False):
        date = pd.Timestamp(row.date)
        score = float(row.raw_score)
        proposed = _score_to_exposure(score, spec)
        if current_exposure is None:
            chosen = proposed
            last_change_date = date
        else:
            chosen = proposed
            if not math.isclose(proposed, current_exposure, abs_tol=1e-12):
                days_held = _trading_days_between(eval_dates, last_change_date, date)
                near_threshold = abs(score - spec.threshold) < spec.switch_margin
                if days_held < spec.min_holding_days or near_threshold:
                    chosen = current_exposure
                else:
                    last_change_date = date
            else:
                chosen = current_exposure
        current_exposure = float(chosen)
        rows.append({"date": date, "raw_score": score, "target_exposure": current_exposure})

    exposure = pd.DataFrame(rows)
    daily = (
        exposure.set_index("date")
        .reindex(eval_dates)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    floor = _minimum_exposure_for_mode(spec.exposure_mode)
    daily["raw_score"] = pd.to_numeric(daily["raw_score"], errors="coerce")
    daily["target_exposure"] = (
        pd.to_numeric(daily["target_exposure"], errors="coerce")
        .ffill()
        .fillna(floor)
        .clip(0.0, 1.0)
    )
    return daily


def simulate_qqq_cash_strategy(
    frame: pd.DataFrame,
    exposure_schedule: pd.DataFrame,
    *,
    spec: QQQOnlyRunSpec,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> QQQOnlySpecOutput:
    """Simulate close-to-close QQQ/cash exposure and summarize performance."""

    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"])
    source = source.sort_values("date", kind="mergesort").set_index("date")
    eval_dates = _evaluation_dates(source.index, start_date=start_date, end_date=end_date)
    if len(eval_dates) == 0:
        raise SkippedQQQOnlyRun("No evaluation rows remain for simulation.")

    qqq_close = pd.to_numeric(source.loc[eval_dates, "qqq_close"], errors="coerce")
    qqq_returns = qqq_close.pct_change().fillna(0.0)
    qqq_equity = (1.0 + qqq_returns).cumprod()
    qqq_drawdown = qqq_equity / qqq_equity.cummax() - 1.0

    exposure = exposure_schedule.copy()
    exposure["date"] = pd.to_datetime(exposure["date"])
    exposure = exposure.set_index("date").reindex(eval_dates).ffill()
    floor = _minimum_exposure_for_mode(spec.exposure_mode)
    target_exposure = (
        pd.to_numeric(exposure["target_exposure"], errors="coerce")
        .ffill()
        .fillna(floor)
        .clip(0.0, 1.0)
    )
    raw_score = pd.to_numeric(
        exposure.get("raw_score", pd.Series(index=eval_dates)), errors="coerce"
    )
    effective_exposure = target_exposure.shift(1)
    if not effective_exposure.empty:
        effective_exposure.iloc[0] = target_exposure.iloc[0]
    effective_exposure = effective_exposure.fillna(floor).clip(0.0, 1.0)
    turnover = effective_exposure.diff().abs().fillna(0.0)
    transaction_cost = turnover * spec.transaction_cost_bps / 10000.0
    gross_return = effective_exposure * qqq_returns
    daily_return = gross_return - transaction_cost
    equity = (1.0 + daily_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    daily_equity = pd.DataFrame(
        {
            "date": eval_dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "exposure_mode": spec.exposure_mode,
            "rebalance_frequency": spec.rebalance_frequency,
            "transaction_cost_bps": spec.transaction_cost_bps,
            "target_exposure": target_exposure.to_numpy(),
            "effective_exposure": effective_exposure.to_numpy(),
            "qqq_weight": effective_exposure.to_numpy(),
            "cash_weight": (1.0 - effective_exposure).to_numpy(),
            "qqq_return": qqq_returns.to_numpy(),
            "gross_return": gross_return.to_numpy(),
            "transaction_cost": transaction_cost.to_numpy(),
            "turnover": turnover.to_numpy(),
            "daily_return": daily_return.to_numpy(),
            "equity": equity.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
        },
        columns=DAILY_EQUITY_COLUMNS,
    )
    daily_drawdown = pd.DataFrame(
        {
            "date": eval_dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "equity": equity.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "qqq_buy_hold_equity": qqq_equity.to_numpy(),
            "qqq_buy_hold_drawdown": qqq_drawdown.to_numpy(),
        },
        columns=DAILY_DRAWDOWN_COLUMNS,
    )
    exposure_log = pd.DataFrame(
        {
            "date": eval_dates,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "model_type": spec.model_type,
            "training_mode": spec.training_mode,
            "target_name": spec.target_name,
            "feature_set_name": spec.feature_set_name,
            "exposure_mode": spec.exposure_mode,
            "rebalance_frequency": spec.rebalance_frequency,
            "target_exposure": target_exposure.to_numpy(),
            "effective_exposure": effective_exposure.to_numpy(),
            "qqq_weight": effective_exposure.to_numpy(),
            "cash_weight": (1.0 - effective_exposure).to_numpy(),
            "held_assets": [_held_assets(value) for value in effective_exposure],
            "raw_score": raw_score.to_numpy(),
            "threshold": spec.threshold,
        },
        columns=EXPOSURE_LOG_COLUMNS,
    )
    result = _result_row(
        spec=spec,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        source=source,
        feature_count=0 if spec.is_baseline else len(_feature_count_columns(frame, spec)),
    )
    return QQQOnlySpecOutput(
        result=result,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )


def rebalance_dates_for_frequency(
    dates: pd.DatetimeIndex,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    frequency: str,
    weekday: str | None = None,
) -> pd.DatetimeIndex:
    """Return QQQ trading dates used as same-date signal calculation points."""

    eval_dates = _evaluation_dates(dates, start_date=start_date, end_date=end_date)
    if len(eval_dates) == 0:
        return pd.DatetimeIndex([])
    if frequency == "weekly":
        selected = _weekly_rebalance_dates(eval_dates, weekday=weekday)
    elif frequency == "monthly":
        periods = eval_dates.to_period("M")
        selected = pd.Series(eval_dates, index=eval_dates).groupby(periods).max()
    else:
        raise ValueError(f"Unsupported rebalance_frequency: {frequency!r}")
    values = pd.DatetimeIndex([eval_dates[0], *selected.tolist()]).unique().sort_values()
    return values[(values >= eval_dates[0]) & (values <= eval_dates[-1])]


def _weekly_rebalance_dates(
    eval_dates: pd.DatetimeIndex,
    *,
    weekday: str | None = None,
) -> pd.Series:
    """Return one trading date per week, optionally anchored to a weekday."""

    weekday_lookup = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
    }
    target = weekday_lookup.get(str(weekday or "FRI").upper())
    periods = eval_dates.to_period("W-FRI")
    grouped = pd.Series(eval_dates, index=eval_dates).groupby(periods)
    if target is None or target == 4:
        return grouped.max()

    selected: list[pd.Timestamp] = []
    for _, group in grouped:
        week_dates = pd.DatetimeIndex(group.tolist()).sort_values()
        exact = week_dates[week_dates.weekday == target]
        if len(exact) > 0:
            selected.append(pd.Timestamp(exact[0]))
            continue
        after = week_dates[week_dates.weekday > target]
        if len(after) > 0:
            selected.append(pd.Timestamp(after[0]))
            continue
        selected.append(pd.Timestamp(week_dates[-1]))
    return pd.Series(selected, index=grouped.size().index)


def classify_qqq_only_decision(row: pd.Series | dict[str, object]) -> str:
    """Classify a completed result row without marking production-ready."""

    total = _float_value(row.get("total_return"))
    qqq_total = _float_value(row.get("qqq_buy_hold_total_return"))
    drawdown_improvement = _float_value(row.get("drawdown_improvement_vs_qqq"))
    rtd_delta = _float_value(row.get("return_to_drawdown_vs_qqq"))
    positive_rate = _float_value(row.get("positive_vs_qqq_window_rate"))
    worst = _float_value(row.get("worst_strategy_vs_qqq"))
    beats_return = np.isfinite(total) and np.isfinite(qqq_total) and total > qqq_total

    if beats_return and (drawdown_improvement > 0.0 or rtd_delta > 0.0):
        return "qqq_timing_candidate"
    if positive_rate >= 0.70 and worst >= -0.10:
        return "consistency_candidate"
    if (not beats_return) and drawdown_improvement >= 0.05:
        return "defensive_reference"
    if beats_return and drawdown_improvement <= 0.0:
        return "return_reference"
    return "reject_deprioritize"


def build_qqq_only_report_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sorted summary and decision matrix tables."""

    summary = results.copy()
    for column in RESULT_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    summary["production_ready"] = False
    summary["decision_label"] = summary.apply(classify_qqq_only_decision, axis=1)
    summary["high_consistency"] = (
        pd.to_numeric(summary["positive_vs_qqq_window_rate"], errors="coerce") >= 0.70
    )
    summary["very_high_consistency"] = (
        pd.to_numeric(summary["positive_vs_qqq_window_rate"], errors="coerce") >= 0.80
    )
    summary = _sort_results(summary.loc[:, RESULT_COLUMNS])

    rows: list[dict[str, object]] = []
    for idx, row in summary.reset_index(drop=True).iterrows():
        rows.append(
            {
                "candidate_id": f"candidate_{idx + 1:04d}",
                "strategy_id": row["strategy_id"],
                "strategy_name": row["strategy_name"],
                "model_type": row["model_type"],
                "training_mode": row["training_mode"],
                "target_name": row["target_name"],
                "feature_set_name": row["feature_set_name"],
                "exposure_mode": row["exposure_mode"],
                "rebalance_frequency": row["rebalance_frequency"],
                "transaction_cost_bps": row["transaction_cost_bps"],
                "threshold": row["threshold"],
                "total_return": row["total_return"],
                "qqq_buy_hold_total_return": row["qqq_buy_hold_total_return"],
                "strategy_vs_qqq": row["strategy_vs_qqq"],
                "max_drawdown": row["max_drawdown"],
                "drawdown_improvement_vs_qqq": row["drawdown_improvement_vs_qqq"],
                "return_to_drawdown_vs_qqq": row["return_to_drawdown_vs_qqq"],
                "positive_vs_qqq_window_rate": row["positive_vs_qqq_window_rate"],
                "worst_strategy_vs_qqq": row["worst_strategy_vs_qqq"],
                "average_exposure": row["average_exposure"],
                "trade_count": row["trade_count"],
                "decision_label": row["decision_label"],
                "production_ready": False,
                "rationale": _decision_rationale(row),
            }
        )
    return summary, pd.DataFrame(rows)


def write_qqq_only_report_artifacts(
    *,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write report CSVs, Markdown, and required PNG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary, decision = build_qqq_only_report_tables(results)
    paths = {
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.csv",
        "decision_matrix": output_dir / f"{OUTPUT_PREFIX}_decision_matrix.csv",
        "report": output_dir / f"{OUTPUT_PREFIX}_report.md",
        "equity_chart": output_dir / "qqq_only_cross_market_timing_equity_curve_top_candidates.png",
        "drawdown_chart": output_dir
        / "qqq_only_cross_market_timing_drawdown_curve_top_candidates.png",
        "exposure_chart": output_dir
        / "qqq_only_cross_market_timing_exposure_timeline_top_candidates.png",
        "rolling_chart": output_dir
        / "qqq_only_cross_market_timing_rolling_vs_qqq_top_candidates.png",
        "family_chart": output_dir / "qqq_only_cross_market_timing_model_family_comparison.png",
        "feature_chart": output_dir / "qqq_only_cross_market_timing_feature_coverage.png",
    }
    summary.to_csv(paths["summary"], index=False)
    decision.to_csv(paths["decision_matrix"], index=False)
    _write_charts(
        summary=summary,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        feature_summary=feature_summary,
        paths=paths,
    )
    paths["report"].write_text(
        render_qqq_only_report_markdown(summary=summary, decision_matrix=decision),
        encoding="utf-8",
    )
    return paths


def render_qqq_only_report_markdown(
    *,
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> str:
    """Render the QQQ-only timing research report."""

    if summary.empty:
        best_name = "none"
        beat_count = 0
        defensive_count = 0
        continuation = "No completed configurations are available; rerun after fixing inputs."
    else:
        best_name = str(decision_matrix.iloc[0]["strategy_name"])
        beat_count = int((summary["strategy_vs_qqq"] > 0.0).sum())
        defensive_count = int((summary["drawdown_improvement_vs_qqq"] > 0.0).sum())
        continuation = (
            "Continue with focused confirmation only for rows labelled "
            "`qqq_timing_candidate` or `consistency_candidate`."
        )
    return "\n".join(
        [
            "# QQQ-Only Cross-Market Timing Model Zoo Report",
            "",
            "## 1. Executive summary.",
            "",
            f"- Completed strategies/configs: `{len(summary)}`",
            f"- Configurations beating QQQ total return: `{beat_count}`",
            f"- Configurations improving drawdown vs QQQ: `{defensive_count}`",
            f"- Top ranked row: `{best_name}`",
            "",
            "## 2. Why QQQ-only timing.",
            "",
            (
                "The research question is whether QQQ buy-and-hold can be improved "
                "by controlling QQQ exposure, not by rotating among ETFs."
            ),
            "",
            "## 3. Inputs and feature families.",
            "",
            (
                "Inputs are QQQ self features, cross-market ETF regime features, "
                "and reference macro/volatility features. Cross-market ETFs and "
                "reference series are features only."
            ),
            "",
            "## 4. QQQ-only rule: holdings are QQQ/cash only.",
            "",
            (
                "Every simulated row is expressed as QQQ weight plus cash weight. "
                "SPY, sectors, bonds, commodities, ARKK, and reference series are "
                "never holdings."
            ),
            "",
            "## 5. Baseline rules.",
            "",
            (
                "Baselines include QQQ buy-and-hold, cash, QQQ SMA200 timing, "
                "QQQ SMA50/SMA200 trend, VIX defensive thresholds, and trend plus "
                "VIX thresholds."
            ),
            "",
            "## 6. Model zoo design.",
            "",
            (
                "Models are trained with walk-forward chronological folds only. "
                "Labels may use future QQQ returns; feature rows use only same-date "
                "or prior observations."
            ),
            "",
            "## 7. Full-period performance vs QQQ buy-and-hold.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "strategy_name",
                    "model_type",
                    "training_mode",
                    "total_return",
                    "qqq_buy_hold_total_return",
                    "strategy_vs_qqq",
                    "decision_label",
                ],
            ),
            "",
            "## 8. Drawdown behavior.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                [
                    "strategy_name",
                    "max_drawdown",
                    "drawdown_improvement_vs_qqq",
                    "return_to_drawdown_vs_qqq",
                ],
            ),
            "",
            "## 9. Rolling-window consistency vs QQQ.",
            "",
            (
                "The study tracks positive rolling-window rates without requiring "
                "95% consistency. Values at or above 0.90 should be treated as "
                "suspicious until validated."
            ),
            "",
            "## 10. Exposure behavior.",
            "",
            _markdown_table(
                decision_matrix.head(12),
                ["strategy_name", "exposure_mode", "average_exposure", "trade_count"],
            ),
            "",
            "## 11. Best QQQ-only timing candidates.",
            "",
            _markdown_table(
                decision_matrix.loc[
                    decision_matrix["decision_label"].isin(
                        ["qqq_timing_candidate", "consistency_candidate"]
                    )
                ].head(12),
                ["strategy_name", "strategy_vs_qqq", "drawdown_improvement_vs_qqq", "rationale"],
            ),
            "",
            "## 12. What failed.",
            "",
            _markdown_table(
                decision_matrix.loc[
                    decision_matrix["decision_label"].eq("reject_deprioritize")
                ].head(12),
                ["strategy_name", "strategy_vs_qqq", "return_to_drawdown_vs_qqq", "rationale"],
            ),
            "",
            "## 13. Recommendation:",
            "",
            f"- Whether QQQ-only timing should continue: `{continuation}`",
            (
                "- Whether to run focused confirmation: `yes, only for the strongest "
                "research candidates`"
            ),
            (
                "- Whether to expand model family or features: `only after focused "
                "confirmation shows durable value`"
            ),
            (
                "- Whether to stop ETF rotation as the main path: `do not decide "
                "from this batch alone; compare after confirmation`"
            ),
            "- Whether to promote to production: `no`",
            "",
            "## 14. Caveats:",
            "",
            "- Research-only.",
            "- Local-only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ is the only tradable risky asset.",
            "- Other ETFs and reference series are features only.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- `production_ready` is always false.",
            "",
        ]
    )


def _baseline_specs(costs: Sequence[float]) -> tuple[QQQOnlyRunSpec, ...]:
    specs: list[QQQOnlyRunSpec] = []
    for rule in BASELINE_RULES:
        for cost in costs:
            payload = {"baseline_rule": rule, "transaction_cost_bps": cost}
            run_key = _run_key(payload)
            specs.append(
                QQQOnlyRunSpec(
                    run_key=run_key,
                    strategy_id=f"{rule}__cost_{cost:g}bps",
                    strategy_name=rule,
                    model_type="baseline",
                    training_mode="rule",
                    target_name="none",
                    target_horizon=0,
                    feature_set_name="baseline_rule",
                    exposure_mode="binary_0_100",
                    rebalance_frequency="daily",
                    transaction_cost_bps=float(cost),
                    threshold=np.nan,
                    min_holding_days=0,
                    switch_margin=0.0,
                    is_baseline=True,
                    baseline_rule=rule,
                )
            )
    return tuple(specs)


def _model_spec(
    *,
    model_type: str,
    training_mode: str,
    target_name: str,
    feature_set_name: str,
    exposure_mode: str,
    rebalance_frequency: str,
    transaction_cost_bps: float,
    threshold: float,
    min_holding_days: int,
    switch_margin: float,
    rebalance_weekday: str = "FRI",
    random_state: int = 42,
    n_estimators: int = 60,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
) -> QQQOnlyRunSpec:
    horizon = _target_horizon(target_name)
    payload = {
        "model_type": model_type,
        "training_mode": training_mode,
        "target_name": target_name,
        "feature_set_name": feature_set_name,
        "exposure_mode": exposure_mode,
        "rebalance_frequency": rebalance_frequency,
        "transaction_cost_bps": transaction_cost_bps,
        "threshold": threshold,
        "min_holding_days": min_holding_days,
        "switch_margin": switch_margin,
    }
    if str(rebalance_weekday).upper() != "FRI":
        payload["rebalance_weekday"] = str(rebalance_weekday).upper()
    if int(random_state) != 42:
        payload["random_state"] = int(random_state)
    if int(n_estimators) != 60:
        payload["n_estimators"] = int(n_estimators)
    if max_depth is not None:
        payload["max_depth"] = int(max_depth)
    if int(min_samples_leaf) != 5:
        payload["min_samples_leaf"] = int(min_samples_leaf)
    run_key = _run_key(payload)
    suffix = ""
    if str(rebalance_weekday).upper() != "FRI":
        suffix += f"_{str(rebalance_weekday).upper().lower()}"
    if int(random_state) != 42:
        suffix += f"_seed{int(random_state)}"
    if int(n_estimators) != 60 or max_depth is not None or int(min_samples_leaf) != 5:
        depth_label = "none" if max_depth is None else str(int(max_depth))
        suffix += f"_trees{int(n_estimators)}_depth{depth_label}_leaf{int(min_samples_leaf)}"
    strategy_name = (
        f"{model_type}_{training_mode}_{target_name}_{feature_set_name}_"
        f"{exposure_mode}_{rebalance_frequency}{suffix}_cost{transaction_cost_bps:g}"
    )
    return QQQOnlyRunSpec(
        run_key=run_key,
        strategy_id=run_key,
        strategy_name=strategy_name,
        model_type=model_type,
        training_mode=training_mode,
        target_name=target_name,
        target_horizon=horizon,
        feature_set_name=feature_set_name,
        exposure_mode=exposure_mode,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=float(transaction_cost_bps),
        threshold=float(threshold),
        min_holding_days=int(min_holding_days),
        switch_margin=float(switch_margin),
        rebalance_weekday=str(rebalance_weekday).upper(),
        random_state=int(random_state),
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
    )


def _run_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    prefix = "__".join(f"{_slug(key)}-{_slug(str(value))}" for key, value in payload.items())
    return f"{prefix}__{digest}"


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "_")


def _glob_supported_files(path_or_glob: str) -> list[Path]:
    matches = glob.glob(path_or_glob, recursive=True) if glob.has_magic(path_or_glob) else []
    if not matches:
        path = Path(path_or_glob)
        if path.is_dir():
            matches = [str(path) for path in path.rglob("*")]
        elif path.is_file():
            matches = [str(path)]
    paths = [
        Path(path)
        for path in matches
        if Path(path).is_file() and Path(path).suffix.lower() in {".parquet", ".csv"}
    ]
    return sorted(paths)


def _discover_reference_symbols_from_paths(paths: Sequence[Path]) -> set[str]:
    symbols: set[str] = set()
    for path in paths:
        symbol = _symbol_from_path(path)
        if symbol:
            symbols.add(symbol)
            continue
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=["symbol"])
            else:
                frame = pd.read_csv(path, usecols=["symbol"])
        except Exception:
            continue
        symbols.update(frame["symbol"].dropna().astype(str).str.upper().unique().tolist())
    return {symbol for symbol in symbols if symbol not in EXCLUDED_REFERENCE_SYMBOLS}


def _read_reference_file(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "symbol", "value"])
    date_col = _first_present(raw.columns, ("date", "observation_date", "timestamp", "datetime"))
    if date_col is None:
        raise ValueError(f"Reference file is missing a date column: {path}")
    symbol_col = _first_present(raw.columns, ("symbol", "series_id", "ticker"))
    value_col = _first_present(
        raw.columns,
        ("value", "level", "close", "adj_close", "rate", "observation_value"),
    )
    if value_col is None:
        numeric_cols = [
            column
            for column in raw.columns
            if column not in {date_col, symbol_col, "year", "month"}
            and pd.api.types.is_numeric_dtype(raw[column])
        ]
        if not numeric_cols:
            raise ValueError(f"Reference file is missing a numeric value column: {path}")
        value_col = numeric_cols[0]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "symbol": (
                raw[symbol_col].astype(str).str.upper()
                if symbol_col is not None
                else _symbol_from_path(path)
            ),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "symbol", "value"])
    return out


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]
    return None


def _symbol_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.lower().startswith("symbol="):
            return part.split("=", maxsplit=1)[1].upper()
    return None


def _prepare_price_matrix(
    market_data: pd.DataFrame,
    *,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    missing = {"date", "symbol", "close"}.difference(market_data.columns)
    if missing:
        raise ValueError("Market data is missing required columns: " + ", ".join(sorted(missing)))
    frame = market_data.loc[:, ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    prices = (
        frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index(kind="mergesort")
        .rename_axis(columns=None)
    )
    return prices.dropna(how="all")


def _align_reference_series(
    reference_series: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    expected_symbols: Sequence[str],
) -> pd.DataFrame:
    if reference_series.empty:
        return pd.DataFrame(index=dates)
    reference = reference_series.copy()
    reference.index = pd.to_datetime(reference.index)
    reference = reference.sort_index(kind="mergesort")
    expected = [symbol.upper() for symbol in expected_symbols]
    cols = [symbol for symbol in expected if symbol in reference.columns]
    aligned = reference.loc[:, cols].reindex(dates).ffill()
    return aligned


def _rsi(close: pd.Series, *, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window=window, min_periods=window).mean()
    losses = (-delta.clip(upper=0.0)).rolling(window=window, min_periods=window).mean()
    relative_strength = gains / losses.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _future_drawdown(close: pd.Series, *, horizon: int) -> pd.Series:
    future_min = (
        close.shift(-1).iloc[::-1].rolling(window=horizon, min_periods=horizon).min().iloc[::-1]
    )
    return future_min / close - 1.0


def _feature_columns_by_family(
    families: dict[str, tuple[str, str]],
    selected_families: set[str],
) -> list[str]:
    return [
        column
        for column, (family, _source) in sorted(families.items())
        if family in selected_families and not column.startswith("qqq_sma_")
    ]


def _build_feature_summary(
    *,
    frame: pd.DataFrame,
    families: dict[str, tuple[str, str]],
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_name, (family, source) in sorted(families.items()):
        if feature_name not in frame.columns or family.endswith("_helper"):
            continue
        values = frame[feature_name]
        valid = values.notna()
        used_in = [name for name, columns in feature_sets.items() if feature_name in set(columns)]
        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": family,
                "source_symbol": source,
                "available_rows": int(valid.sum()),
                "missing_pct": float(1.0 - valid.mean()) if len(valid) else np.nan,
                "first_valid_date": (
                    str(pd.Timestamp(frame.loc[valid, "date"].iloc[0]).date())
                    if valid.any()
                    else ""
                ),
                "last_valid_date": (
                    str(pd.Timestamp(frame.loc[valid, "date"].iloc[-1]).date())
                    if valid.any()
                    else ""
                ),
                "used_in_feature_sets": ",".join(used_in),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_SUMMARY_COLUMNS)


def _training_frame_for_prediction(
    frame: pd.DataFrame,
    *,
    predict_date: pd.Timestamp,
    spec: QQQOnlyRunSpec,
) -> pd.DataFrame:
    horizon = spec.target_horizon
    availability_col = f"target_available_date_{horizon}d"
    if availability_col not in frame.columns:
        raise SkippedQQQOnlyRun(f"Target availability column is missing: {availability_col}")
    train = frame.loc[
        (frame["date"] < predict_date)
        & (pd.to_datetime(frame[availability_col]) <= predict_date)
        & frame[spec.target_name].notna()
    ].copy()
    if spec.training_mode == "expanding" or spec.training_mode == "recency_weighted_expanding":
        return train
    if spec.training_mode == "rolling_3y":
        return train.loc[train["date"] >= predict_date - pd.DateOffset(years=3)].copy()
    if spec.training_mode == "rolling_5y":
        return train.loc[train["date"] >= predict_date - pd.DateOffset(years=5)].copy()
    if spec.training_mode == "frozen_base_1999_2018":
        frozen_end = pd.Timestamp("2018-12-31")
        frozen = frame.loc[
            (frame["date"] <= frozen_end)
            & (pd.to_datetime(frame[availability_col]) <= frozen_end)
            & frame[spec.target_name].notna()
        ].copy()
        return frozen
    raise SkippedQQQOnlyRun(f"Unsupported training mode: {spec.training_mode}")


def _select_usable_features(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> list[str]:
    selected = []
    for column in feature_cols:
        values = pd.to_numeric(train_df[column], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) > 1:
            selected.append(column)
    return selected


def _fit_predict_score(
    *,
    train_df: pd.DataFrame,
    prediction_row: pd.DataFrame,
    feature_cols: list[str],
    spec: QQQOnlyRunSpec,
) -> float:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = train_df.loc[:, feature_cols]
    y_train = pd.to_numeric(train_df[spec.target_name], errors="coerce")
    usable = y_train.notna()
    x_train = x_train.loc[usable]
    y_train = y_train.loc[usable]
    if x_train.empty:
        raise SkippedQQQOnlyRun("No usable training rows after target filtering.")

    is_classifier = spec.target_name in CLASSIFICATION_TARGETS
    if is_classifier:
        y_class = (y_train > 0.0).astype(int)
        if y_class.nunique() < 2:
            return float(y_class.mean())
        model: Any
        if spec.model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif spec.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_samples_leaf,
                random_state=spec.random_state,
                n_jobs=1,
            )
        elif spec.model_type == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=60,
                min_samples_leaf=5,
                random_state=spec.random_state,
                n_jobs=1,
            )
        elif spec.model_type == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=spec.random_state,
            )
        elif spec.model_type == "simple_mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=120,
                random_state=spec.random_state,
                learning_rate_init=0.001,
            )
        else:
            raise SkippedQQQOnlyRun(f"Unsupported classifier model: {spec.model_type}")
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        fit_kwargs = _fit_kwargs_for_sample_weight(spec, train_df.loc[usable, "date"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(x_train, y_class, **fit_kwargs)
        probabilities = pipeline.predict_proba(prediction_row.loc[:, feature_cols])
        return float(probabilities[0, 1])

    if spec.model_type == "ridge":
        regressor: Any = Ridge(alpha=1.0, random_state=spec.random_state)
    elif spec.model_type == "elasticnet":
        regressor = ElasticNet(
            alpha=0.001, l1_ratio=0.5, random_state=spec.random_state, max_iter=10000
        )
    elif spec.model_type == "lasso":
        regressor = Lasso(alpha=0.001, random_state=spec.random_state, max_iter=10000)
    elif spec.model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            min_samples_leaf=spec.min_samples_leaf,
            random_state=spec.random_state,
            n_jobs=1,
        )
    elif spec.model_type == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=60,
            min_samples_leaf=5,
            random_state=spec.random_state,
            n_jobs=1,
        )
    elif spec.model_type == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=spec.random_state,
        )
    elif spec.model_type == "simple_mlp":
        regressor = MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=120,
            random_state=spec.random_state,
            learning_rate_init=0.001,
        )
    else:
        raise SkippedQQQOnlyRun(f"Unsupported regression model: {spec.model_type}")

    if spec.model_type == "simple_mlp":
        regressor = TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", regressor),
        ]
    )
    fit_kwargs = _fit_kwargs_for_sample_weight(spec, train_df.loc[usable, "date"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(x_train, y_train, **fit_kwargs)
    return float(pipeline.predict(prediction_row.loc[:, feature_cols])[0])


def _fit_kwargs_for_sample_weight(spec: QQQOnlyRunSpec, train_dates: pd.Series) -> dict[str, Any]:
    if spec.training_mode != "recency_weighted_expanding":
        return {}
    if spec.model_type == "simple_mlp":
        return {}
    dates = pd.to_datetime(train_dates)
    train_end = dates.max()
    ages = (train_end - dates).dt.days.clip(lower=0)
    weights = np.exp(-ages / 504.0)
    weights = weights / np.nanmean(weights)
    return {"model__sample_weight": weights.to_numpy()}


def _score_to_exposure(score: float, spec: QQQOnlyRunSpec) -> float:
    if spec.exposure_mode == "binary_0_100":
        return 1.0 if score >= spec.threshold else 0.0
    if spec.exposure_mode == "partial_50_100":
        return 1.0 if score >= spec.threshold else 0.5
    if spec.exposure_mode == "ternary_0_50_100":
        band = max(
            abs(spec.switch_margin), 0.05 if spec.target_name in CLASSIFICATION_TARGETS else 0.002
        )
        if score >= spec.threshold + band:
            return 1.0
        if score >= spec.threshold - band:
            return 0.5
        return 0.0
    raise ValueError(f"Unsupported exposure mode: {spec.exposure_mode}")


def _minimum_exposure_for_mode(exposure_mode: str) -> float:
    return 0.5 if exposure_mode == "partial_50_100" else 0.0


def _evaluation_dates(
    dates: pd.DatetimeIndex,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return index[(index >= start) & (index <= end)]


def _trading_days_between(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> int:
    if start is None:
        return 10**9
    return int(((dates > start) & (dates <= end)).sum())


def _held_assets(exposure: float) -> str:
    if exposure <= 1e-9:
        return "cash"
    if exposure >= 1.0 - 1e-9:
        return "QQQ"
    return "QQQ,cash"


def _result_row(
    *,
    spec: QQQOnlyRunSpec,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    source: pd.DataFrame,
    feature_count: int,
) -> dict[str, object]:
    days = max(len(daily_equity), 1)
    total_return = float(daily_equity["equity"].iloc[-1] - 1.0)
    daily_returns = pd.to_numeric(daily_equity["daily_return"], errors="coerce")
    volatility = float(daily_returns.std())
    sharpe = float(daily_returns.mean() / volatility) if volatility > 0.0 else np.nan
    max_drawdown = float(daily_drawdown["drawdown"].min())
    qqq_total = float(daily_equity["qqq_buy_hold_equity"].iloc[-1] - 1.0)
    qqq_dd = float(daily_drawdown["qqq_buy_hold_drawdown"].min())
    rtd = _return_to_drawdown(total_return, max_drawdown)
    qqq_rtd = _return_to_drawdown(qqq_total, qqq_dd)
    rolling_diff = _rolling_vs_qqq(daily_equity)
    positive_rate = float((rolling_diff > 0.0).mean()) if not rolling_diff.empty else np.nan
    under_rate = float((rolling_diff < 0.0).mean()) if not rolling_diff.empty else np.nan
    worst_vs_qqq = float(rolling_diff.min()) if not rolling_diff.empty else total_return - qqq_total
    trade_count = int((pd.to_numeric(daily_equity["turnover"], errors="coerce") > 1e-9).sum())
    result = {
        "strategy_id": spec.strategy_id,
        "strategy_name": spec.strategy_name,
        "model_type": spec.model_type,
        "training_mode": spec.training_mode,
        "target_name": spec.target_name,
        "target_horizon": spec.target_horizon,
        "feature_set_name": spec.feature_set_name,
        "feature_count": feature_count,
        "exposure_mode": spec.exposure_mode,
        "rebalance_frequency": spec.rebalance_frequency,
        "transaction_cost_bps": spec.transaction_cost_bps,
        "threshold": spec.threshold,
        "min_holding_days": spec.min_holding_days,
        "switch_margin": spec.switch_margin,
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, days),
        "max_drawdown": max_drawdown,
        "daily_volatility": volatility,
        "sharpe_like": sharpe,
        "return_to_drawdown": rtd,
        "qqq_buy_hold_total_return": qqq_total,
        "qqq_buy_hold_max_drawdown": qqq_dd,
        "qqq_buy_hold_return_to_drawdown": qqq_rtd,
        "strategy_vs_qqq": total_return - qqq_total,
        "drawdown_improvement_vs_qqq": max_drawdown - qqq_dd,
        "return_to_drawdown_vs_qqq": rtd - qqq_rtd,
        "positive_vs_qqq_window_rate": positive_rate,
        "qqq_underperformance_rate": under_rate,
        "worst_strategy_vs_qqq": worst_vs_qqq,
        "average_exposure": float(daily_equity["effective_exposure"].mean()),
        "cash_days_pct": float((daily_equity["effective_exposure"] <= 1e-9).mean()),
        "trade_count": trade_count,
        "average_holding_days": _average_holding_days(daily_equity["effective_exposure"]),
        "estimated_transaction_cost": float(daily_equity["transaction_cost"].sum()),
        "production_ready": False,
        "decision_label": "unclassified",
        "high_consistency": bool(positive_rate >= 0.70) if np.isfinite(positive_rate) else False,
        "very_high_consistency": bool(positive_rate >= 0.80)
        if np.isfinite(positive_rate)
        else False,
        "spy_buy_hold_total_return": np.nan,
        "spy_buy_hold_max_drawdown": np.nan,
    }
    if "spy_close" in source.columns:
        spy = pd.to_numeric(
            source.loc[pd.to_datetime(daily_equity["date"]), "spy_close"], errors="coerce"
        )
        spy_returns = spy.pct_change().fillna(0.0)
        spy_equity = (1.0 + spy_returns).cumprod()
        result["spy_buy_hold_total_return"] = float(spy_equity.iloc[-1] - 1.0)
        result["spy_buy_hold_max_drawdown"] = calculate_max_drawdown(spy_equity)
    result["decision_label"] = classify_qqq_only_decision(result)
    return result


def _feature_count_columns(frame: pd.DataFrame, spec: QQQOnlyRunSpec) -> list[str]:
    if spec.feature_set_name == "baseline_rule":
        return []
    excluded = {
        "date",
        "qqq_close",
        "spy_close",
        "qqq_daily_return",
        "spy_daily_return",
        *CLASSIFICATION_TARGETS,
        *REGRESSION_TARGETS,
        "target_available_date_20d",
        "target_available_date_63d",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith("target_available_date")
        and not column.startswith("qqq_forward_return")
        and not column.startswith("risk_on_")
    ]


def _rolling_vs_qqq(daily_equity: pd.DataFrame, *, window: int = 63) -> pd.Series:
    strategy = pd.to_numeric(daily_equity["equity"], errors="coerce").pct_change(window)
    qqq = pd.to_numeric(daily_equity["qqq_buy_hold_equity"], errors="coerce").pct_change(window)
    return (strategy - qqq).dropna()


def _annualized_return(total_return: float, days: int) -> float:
    if days <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0)


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown) or max_drawdown >= 0.0:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _average_holding_days(exposure: pd.Series) -> float:
    values = pd.to_numeric(exposure, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    change_idx = np.flatnonzero(values.diff().abs().fillna(0.0).to_numpy() > 1e-9)
    boundaries = [0, *change_idx.tolist(), len(values)]
    durations = [
        end - start
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=False)
        if end > start
    ]
    return float(np.mean(durations)) if durations else float(len(values))


def _target_horizon(target_name: str) -> int:
    if target_name.endswith("_63d"):
        return 63
    if target_name.endswith("_20d"):
        return 20
    return 0


def _model_target_is_supported(model_type: str, target_name: str) -> bool:
    if target_name in CLASSIFICATION_TARGETS:
        return _model_supports_classification(model_type)
    if target_name in REGRESSION_TARGETS:
        return _model_supports_regression(model_type)
    return False


def _model_supports_classification(model_type: str) -> bool:
    return model_type in {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "simple_mlp",
    }


def _model_supports_regression(model_type: str) -> bool:
    return model_type in {
        "ridge",
        "elasticnet",
        "lasso",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "simple_mlp",
    }


def _sort_results(results: pd.DataFrame) -> pd.DataFrame:
    decision_rank = {
        "qqq_timing_candidate": 0,
        "consistency_candidate": 1,
        "return_reference": 2,
        "defensive_reference": 3,
        "reject_deprioritize": 4,
    }
    out = results.copy()
    out["_decision_rank"] = out["decision_label"].map(decision_rank).fillna(9)
    out = out.sort_values(
        [
            "_decision_rank",
            "strategy_vs_qqq",
            "return_to_drawdown_vs_qqq",
            "drawdown_improvement_vs_qqq",
        ],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    return out.drop(columns=["_decision_rank"])


def _decision_rationale(row: pd.Series) -> str:
    parts = []
    parts.append(
        "beats QQQ return" if _float_value(row["strategy_vs_qqq"]) > 0 else "trails QQQ return"
    )
    if _float_value(row["drawdown_improvement_vs_qqq"]) > 0:
        parts.append("improves drawdown")
    else:
        parts.append("does not improve drawdown")
    if _float_value(row["positive_vs_qqq_window_rate"]) >= 0.80:
        parts.append("very high rolling consistency; validate for leakage")
    elif _float_value(row["positive_vs_qqq_window_rate"]) >= 0.70:
        parts.append("high rolling consistency")
    return "; ".join(parts)


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _write_charts(
    *,
    summary: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    feature_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_ids = summary.head(6)["strategy_id"].astype(str).tolist()
    qqq_label = "QQQ buy-and-hold"

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in top_ids:
        subset = daily_equity.loc[daily_equity["strategy_id"].astype(str).eq(strategy_id)]
        if subset.empty:
            continue
        ax.plot(
            pd.to_datetime(subset["date"]), subset["equity"], label=subset["strategy_name"].iloc[0]
        )
    if not daily_equity.empty:
        first = daily_equity.groupby("date", sort=True).head(1)
        ax.plot(
            pd.to_datetime(first["date"]),
            first["qqq_buy_hold_equity"],
            label=qqq_label,
            color="black",
            linewidth=2,
        )
    ax.set_title("Top QQQ-only timing candidates: equity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["equity_chart"], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy_id in top_ids:
        subset = daily_drawdown.loc[daily_drawdown["strategy_id"].astype(str).eq(strategy_id)]
        if subset.empty:
            continue
        ax.plot(
            pd.to_datetime(subset["date"]),
            subset["drawdown"],
            label=subset["strategy_name"].iloc[0],
        )
    if not daily_drawdown.empty:
        first = daily_drawdown.groupby("date", sort=True).head(1)
        ax.plot(
            pd.to_datetime(first["date"]),
            first["qqq_buy_hold_drawdown"],
            label=qqq_label,
            color="black",
            linewidth=2,
        )
    ax.set_title("Top QQQ-only timing candidates: drawdown")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["drawdown_chart"], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for strategy_id in top_ids[:4]:
        subset = exposure_log.loc[exposure_log["strategy_id"].astype(str).eq(strategy_id)]
        if subset.empty:
            continue
        ax.step(
            pd.to_datetime(subset["date"]),
            subset["effective_exposure"],
            where="post",
            label=subset["strategy_name"].iloc[0],
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Top QQQ-only timing candidates: QQQ exposure")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["exposure_chart"], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy_id in top_ids[:5]:
        subset = daily_equity.loc[daily_equity["strategy_id"].astype(str).eq(strategy_id)].copy()
        if subset.empty:
            continue
        diff = _rolling_vs_qqq(subset)
        ax.plot(
            pd.to_datetime(subset.loc[diff.index, "date"]),
            diff,
            label=subset["strategy_name"].iloc[0],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Rolling 63-day return difference vs QQQ")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["rolling_chart"], dpi=140)
    plt.close(fig)

    family = (
        summary.groupby("model_type", dropna=False)["strategy_vs_qqq"]
        .max()
        .sort_values(ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    family.plot(kind="bar", ax=ax, color="#2a9d8f")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Best strategy vs QQQ by model family")
    ax.set_ylabel("Total return difference")
    fig.tight_layout()
    fig.savefig(paths["family_chart"], dpi=140)
    plt.close(fig)

    coverage = feature_summary.copy()
    if not coverage.empty:
        coverage["coverage"] = 1.0 - pd.to_numeric(coverage["missing_pct"], errors="coerce")
        by_family = coverage.groupby("feature_family")["coverage"].mean().sort_values()
    else:
        by_family = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    if by_family.empty:
        ax.text(0.5, 0.5, "No feature coverage rows", ha="center", va="center")
    else:
        by_family.plot(kind="barh", ax=ax, color="#457b9d")
    ax.set_xlim(0, 1)
    ax.set_title("Average feature coverage by family")
    fig.tight_layout()
    fig.savefig(paths["feature_chart"], dpi=140)
    plt.close(fig)
