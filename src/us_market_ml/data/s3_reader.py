"""S3 readers for curated ETF daily parquet research input data.

These utilities load curated ETF daily parquet partitions produced by the
upstream ``US-Market-Data`` pipeline into the normalized market data frame
shape expected by downstream dataset, feature, label, and modeling code. AWS
dependencies are optional and imported only when an S3 read is requested.

Missing symbol/year partitions are skipped by default so sparse requests can
still succeed, but a request where every partition is missing raises
``FileNotFoundError``. This module only reads curated ETF parquet data; it does
not build features or labels, write artifacts, or mutate source data.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from us_market_ml.data.local_reader import LocalMarketDataConfig, normalize_market_data_frame


@dataclass(frozen=True)
class S3MarketDataConfig:
    """Configuration for curated ETF daily parquet reads from S3.

    The defaults match the upstream curated ETF layout:
    ``curated/etf_daily_prices/symbol=SYMBOL/year=YYYY/data.parquet``.
    ``boto3`` is an optional project dependency and is imported only by read
    functions. Extra curated columns such as ``year``, ``month``, and
    ``transactions`` are preserved for downstream research use.
    """

    bucket: str
    curated_prefix: str = "curated/etf_daily_prices"
    date_col: str = "date"
    symbol_col: str = "symbol"
    required_columns: tuple[str, ...] = (
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
    )
    aws_profile: str | None = None
    region_name: str | None = None


def build_curated_etf_s3_key(
    symbol: str,
    year: int,
    config: S3MarketDataConfig,
) -> str:
    """Build the S3 key for one curated ETF daily parquet partition."""

    prefix = config.curated_prefix.strip("/")
    partition_key = f"symbol={symbol}/year={year}/data.parquet"
    if not prefix:
        return partition_key
    return f"{prefix}/{partition_key}"


def build_curated_etf_s3_uri(
    symbol: str,
    year: int,
    config: S3MarketDataConfig,
) -> str:
    """Build the S3 URI for one curated ETF daily parquet partition."""

    return f"s3://{config.bucket}/{build_curated_etf_s3_key(symbol, year, config)}"


def build_s3_storage_options(config: S3MarketDataConfig) -> dict[str, object] | None:
    """Build legacy ``s3fs`` storage options for compatibility.

    S3 parquet reads no longer use ``s3fs`` because path-based parquet reads
    can trigger Hive partition inference. This helper remains part of the
    public API for callers that may still depend on it.
    """

    storage_options: dict[str, object] = {}
    if config.aws_profile is not None:
        storage_options["profile"] = config.aws_profile
    if config.region_name is not None:
        storage_options["client_kwargs"] = {"region_name": config.region_name}
    return storage_options or None


def read_s3_curated_etf_data(
    symbols: Sequence[str],
    years: Sequence[int],
    config: S3MarketDataConfig,
) -> pd.DataFrame:
    """Read requested curated ETF daily parquet partitions from S3.

    Symbols and years are expanded into upstream partition URIs, missing
    symbol/year parquet files are skipped, and the loaded rows are normalized
    into the same shape as the local market data reader. Feature generation,
    label construction, and modeling are handled downstream.
    """

    symbol_values = list(symbols)
    if not symbol_values:
        raise ValueError("At least one ETF symbol is required for S3 market data loading.")

    year_values = list(years)
    if not year_values:
        raise ValueError("At least one year is required for S3 market data loading.")

    s3_uris = [
        build_curated_etf_s3_uri(symbol=symbol, year=year, config=config)
        for symbol in symbol_values
        for year in year_values
    ]
    return read_s3_parquet_files(s3_uris, config)


def read_s3_parquet_files(
    s3_uris: Sequence[str],
    config: S3MarketDataConfig,
) -> pd.DataFrame:
    """Read S3 parquet files and normalize them for research pipelines.

    AWS dependencies are optional for the package and are checked lazily here.
    Individual missing S3 parquet files raise ``FileNotFoundError`` internally
    and are skipped; if no requested file can be loaded, ``FileNotFoundError``
    is raised for the whole request.
    """

    uri_values = list(s3_uris)
    if not uri_values:
        raise ValueError("At least one S3 parquet URI is required.")

    frames: list[pd.DataFrame] = []
    missing_uris: list[str] = []
    for uri in uri_values:
        try:
            frames.append(read_s3_parquet_object(uri, config))
        except FileNotFoundError:
            missing_uris.append(uri)

    if not frames:
        missing = ", ".join(missing_uris or uri_values)
        raise FileNotFoundError(f"No requested S3 parquet files could be loaded: {missing}")

    local_config = LocalMarketDataConfig(
        date_col=config.date_col,
        symbol_col=config.symbol_col,
        required_columns=config.required_columns,
    )
    return normalize_market_data_frame(pd.concat(frames, ignore_index=True), local_config)


def read_s3_parquet_object(
    s3_uri: str,
    config: S3MarketDataConfig,
) -> pd.DataFrame:
    """Read one S3 parquet object through boto3 and return its raw DataFrame.

    The object is fetched as bytes and handed to pandas through ``BytesIO`` so
    parquet loading sees only file contents, not the Hive-like S3 key path.
    """

    bucket, key = _parse_s3_uri(s3_uri)
    boto3, client_error = _load_optional_aws_modules()

    session_kwargs: dict[str, str] = {}
    if config.aws_profile is not None:
        session_kwargs["profile_name"] = config.aws_profile
    if config.region_name is not None:
        session_kwargs["region_name"] = config.region_name

    session = boto3.Session(**session_kwargs)
    s3_client = session.client("s3")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except client_error as exc:
        if _is_missing_s3_object_error(exc):
            raise FileNotFoundError(f"S3 parquet object does not exist: {s3_uri}") from exc
        raise

    body = response["Body"]
    data = body.read()
    return pd.read_parquet(BytesIO(data))


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Expected an S3 URI like s3://bucket/key, got: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _load_optional_aws_modules() -> tuple[Any, type[Exception]]:
    """Return lazily imported AWS modules or raise a clear install error."""

    try:
        boto3 = importlib.import_module("boto3")
        exceptions = importlib.import_module("botocore.exceptions")
    except ImportError as exc:
        raise ImportError(
            "S3 market data readers require optional AWS dependencies. "
            "Install them with: python -m pip install -e \".[aws]\" "
            "or python -m pip install -e \".[dev,aws]\"."
        ) from exc

    return boto3, exceptions.ClientError


def _is_missing_s3_object_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False

    error = response.get("Error")
    if not isinstance(error, dict):
        return False

    code = error.get("Code")
    return str(code) in {"NoSuchKey", "404", "NotFound", "NoSuchBucket"}


__all__ = [
    "S3MarketDataConfig",
    "build_curated_etf_s3_key",
    "build_curated_etf_s3_uri",
    "build_s3_storage_options",
    "read_s3_curated_etf_data",
    "read_s3_parquet_object",
    "read_s3_parquet_files",
]
