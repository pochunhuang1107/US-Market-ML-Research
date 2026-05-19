"""Feature generation interfaces for leakage-safe research datasets."""

from us_market_ml.features.benchmark_gap import (
    BENCHMARK_GAP_ALL_FEATURE_COLUMNS,
    BENCHMARK_GAP_FEATURE_COLUMNS,
    BENCHMARK_GAP_V2_FEATURE_COLUMNS,
    BENCHMARK_GAP_V3_FEATURE_COLUMNS,
    add_benchmark_gap_features,
    add_benchmark_gap_v2_features,
    add_benchmark_gap_v3_features,
)
from us_market_ml.features.cross_asset import add_cross_asset_features
from us_market_ml.features.feature_pipeline import (
    BENCHMARK_GAP_FEATURE_SET_VERSION,
    BENCHMARK_GAP_V2_FEATURE_SET_VERSION,
    BENCHMARK_GAP_V3_FEATURE_SET_VERSION,
    DEFAULT_FEATURE_SET_VERSION,
    SUPPORTED_FEATURE_SET_VERSIONS,
    TECHNICAL_FEATURE_SET_VERSION,
    FeatureSetConfig,
    build_feature_frame,
    validate_feature_set_version,
)

__all__ = [
    "BENCHMARK_GAP_FEATURE_COLUMNS",
    "BENCHMARK_GAP_FEATURE_SET_VERSION",
    "BENCHMARK_GAP_ALL_FEATURE_COLUMNS",
    "BENCHMARK_GAP_V2_FEATURE_COLUMNS",
    "BENCHMARK_GAP_V2_FEATURE_SET_VERSION",
    "BENCHMARK_GAP_V3_FEATURE_COLUMNS",
    "BENCHMARK_GAP_V3_FEATURE_SET_VERSION",
    "DEFAULT_FEATURE_SET_VERSION",
    "SUPPORTED_FEATURE_SET_VERSIONS",
    "TECHNICAL_FEATURE_SET_VERSION",
    "FeatureSetConfig",
    "add_benchmark_gap_features",
    "add_benchmark_gap_v2_features",
    "add_benchmark_gap_v3_features",
    "add_cross_asset_features",
    "build_feature_frame",
    "validate_feature_set_version",
]
