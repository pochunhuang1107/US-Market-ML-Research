"""Data interfaces for curated market research datasets."""

from us_market_ml.data.dataset_builder import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    drop_rows_with_missing_model_values,
    get_default_non_feature_columns,
    get_feature_columns,
    get_target_columns,
)
from us_market_ml.data.local_reader import (
    LocalMarketDataConfig,
    normalize_market_data_frame,
    read_csv_files,
    read_local_market_data,
    read_parquet_files,
    validate_market_data_columns,
)
from us_market_ml.data.s3_reader import (
    S3MarketDataConfig,
    build_curated_etf_s3_key,
    build_curated_etf_s3_uri,
    build_s3_storage_options,
    read_s3_curated_etf_data,
    read_s3_parquet_files,
)

__all__ = [
    "LocalMarketDataConfig",
    "S3MarketDataConfig",
    "SupervisedDatasetConfig",
    "build_curated_etf_s3_key",
    "build_curated_etf_s3_uri",
    "build_s3_storage_options",
    "build_supervised_dataset",
    "drop_rows_with_missing_model_values",
    "get_default_non_feature_columns",
    "get_feature_columns",
    "get_target_columns",
    "normalize_market_data_frame",
    "read_csv_files",
    "read_local_market_data",
    "read_parquet_files",
    "read_s3_curated_etf_data",
    "read_s3_parquet_files",
    "validate_market_data_columns",
]
