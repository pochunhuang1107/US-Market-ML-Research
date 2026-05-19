"""Feature pipeline for curated ETF daily OHLCV research rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from us_market_ml.features.benchmark_gap import (
    add_benchmark_gap_features,
    add_benchmark_gap_v2_features,
    add_benchmark_gap_v3_features,
)
from us_market_ml.features.cross_asset import (
    add_cross_asset_features,
    add_technical_v2_cross_sectional_features,
)
from us_market_ml.features.technical import add_technical_features, add_technical_v2_features

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"date", "symbol", "open", "high", "low", "close", "volume", "vwap"}
)
DEFAULT_FEATURE_SET_VERSION = "default_features_v1"
TECHNICAL_FEATURE_SET_VERSION = "technical_features_v2"
BENCHMARK_GAP_FEATURE_SET_VERSION = "benchmark_gap_features_v1"
BENCHMARK_GAP_V2_FEATURE_SET_VERSION = "benchmark_gap_features_v2"
BENCHMARK_GAP_V3_FEATURE_SET_VERSION = "benchmark_gap_features_v3"
SUPPORTED_FEATURE_SET_VERSIONS: frozenset[str] = frozenset(
    {
        DEFAULT_FEATURE_SET_VERSION,
        TECHNICAL_FEATURE_SET_VERSION,
        BENCHMARK_GAP_FEATURE_SET_VERSION,
        BENCHMARK_GAP_V2_FEATURE_SET_VERSION,
        BENCHMARK_GAP_V3_FEATURE_SET_VERSION,
    }
)


@dataclass(frozen=True)
class FeatureSetConfig:
    """Configuration for leakage-safe feature generation."""

    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION


def build_feature_frame(
    df: pd.DataFrame,
    config: FeatureSetConfig | str | None = None,
) -> pd.DataFrame:
    """Build a leakage-safe ETF feature frame from daily OHLCV rows.

    The input is copied, sorted by ``symbol`` and ``date``, and never filtered.
    Per-symbol features are calculated with pandas groupby operations so rolling
    windows, lagged returns, RSI, MACD, and volume statistics only use rows from
    the same symbol at the current date or earlier dates. Cross-asset features
    are same-date differences between already trailing 20-day returns and are
    joined back by date without forward-fill or backward-fill.
    """

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    resolved_config = _resolve_feature_set_config(config)

    out = df.copy()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    out = add_technical_features(out)
    out = add_cross_asset_features(out)
    if resolved_config.feature_set_version == TECHNICAL_FEATURE_SET_VERSION:
        out = add_technical_v2_features(out)
        out = add_technical_v2_cross_sectional_features(out)
    if resolved_config.feature_set_version == BENCHMARK_GAP_FEATURE_SET_VERSION:
        out = add_benchmark_gap_features(out)
    if resolved_config.feature_set_version == BENCHMARK_GAP_V2_FEATURE_SET_VERSION:
        out = add_benchmark_gap_v2_features(out)
    if resolved_config.feature_set_version == BENCHMARK_GAP_V3_FEATURE_SET_VERSION:
        out = add_benchmark_gap_v3_features(out)
    return out


def validate_feature_set_version(feature_set_version: str) -> str:
    """Validate and normalize a supported feature set version."""

    if feature_set_version not in SUPPORTED_FEATURE_SET_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FEATURE_SET_VERSIONS))
        raise ValueError(
            f"Unsupported feature_set_version {feature_set_version!r}; "
            f"supported versions: {supported}."
        )
    return feature_set_version


def _resolve_feature_set_config(config: FeatureSetConfig | str | None) -> FeatureSetConfig:
    if config is None:
        return FeatureSetConfig()
    if isinstance(config, str):
        return FeatureSetConfig(validate_feature_set_version(config))
    validate_feature_set_version(config.feature_set_version)
    return config
