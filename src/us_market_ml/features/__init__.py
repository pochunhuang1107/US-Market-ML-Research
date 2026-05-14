"""Feature generation interfaces for leakage-safe research datasets."""

from us_market_ml.features.cross_asset import add_cross_asset_features
from us_market_ml.features.feature_pipeline import (
    DEFAULT_FEATURE_SET_VERSION,
    SUPPORTED_FEATURE_SET_VERSIONS,
    TECHNICAL_FEATURE_SET_VERSION,
    FeatureSetConfig,
    build_feature_frame,
    validate_feature_set_version,
)

__all__ = [
    "DEFAULT_FEATURE_SET_VERSION",
    "SUPPORTED_FEATURE_SET_VERSIONS",
    "TECHNICAL_FEATURE_SET_VERSION",
    "FeatureSetConfig",
    "add_cross_asset_features",
    "build_feature_frame",
    "validate_feature_set_version",
]
