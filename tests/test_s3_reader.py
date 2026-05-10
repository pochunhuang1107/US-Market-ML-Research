from __future__ import annotations

import builtins
import importlib
import sys
import types
from collections.abc import Callable
from io import BytesIO

import pandas as pd
import pytest

from us_market_ml.data import (
    S3MarketDataConfig,
    build_curated_etf_s3_key,
    build_curated_etf_s3_uri,
    build_s3_storage_options,
    read_s3_curated_etf_data,
    read_s3_parquet_files,
    read_s3_parquet_object,
    s3_reader,
)


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


def _sample_ohlcv(symbol: str = "SPY", date: str = "2024-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "year": 2024,
                "month": 1,
                "symbol": symbol,
                "open": 201.0,
                "high": 203.0,
                "low": 200.0,
                "close": 202.0,
                "volume": 2_200_000,
                "vwap": 201.5,
                "transactions": 22_000,
            }
        ]
    )


def _sample_parquet_bytes(symbol: str = "SPY", date: str = "2024-01-02") -> bytes:
    buffer = BytesIO()
    _sample_ohlcv(symbol=symbol, date=date).to_parquet(buffer, index=False)
    return buffer.getvalue()


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
    objects: dict[tuple[str, str], bytes | Exception],
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    requests: list[tuple[str, str]] = []
    session_kwargs: list[dict[str, str]] = []

    class FakeS3Client:
        def get_object(self, Bucket: str, Key: str) -> dict[str, BytesIO]:
            requests.append((Bucket, Key))
            value = objects[(Bucket, Key)]
            if isinstance(value, Exception):
                raise value
            return {"Body": BytesIO(value)}

    class FakeSession:
        def __init__(self, **kwargs: str) -> None:
            session_kwargs.append(kwargs)

        def client(self, service_name: str) -> FakeS3Client:
            assert service_name == "s3"
            return FakeS3Client()

    boto3_module = types.ModuleType("boto3")
    boto3_module.Session = FakeSession

    botocore_module = types.ModuleType("botocore")
    exceptions_module = types.ModuleType("botocore.exceptions")
    exceptions_module.ClientError = FakeClientError
    botocore_module.exceptions = exceptions_module

    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_module)

    return requests, session_kwargs


def test_build_curated_etf_s3_key_returns_expected_key() -> None:
    config = S3MarketDataConfig(bucket="example-bucket")

    result = build_curated_etf_s3_key("SPY", 2024, config)

    assert result == "curated/etf_daily_prices/symbol=SPY/year=2024/data.parquet"


def test_build_curated_etf_s3_uri_returns_expected_uri() -> None:
    config = S3MarketDataConfig(bucket="example-bucket")

    result = build_curated_etf_s3_uri("QQQ", 2023, config)

    assert result == "s3://example-bucket/curated/etf_daily_prices/symbol=QQQ/year=2023/data.parquet"


def test_read_s3_curated_etf_data_builds_expected_uris(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = S3MarketDataConfig(bucket="example-bucket")
    objects = {
        (
            "example-bucket",
            "curated/etf_daily_prices/symbol=SPY/year=2023/data.parquet",
        ): _sample_parquet_bytes(symbol="SPY"),
        (
            "example-bucket",
            "curated/etf_daily_prices/symbol=SPY/year=2024/data.parquet",
        ): _sample_parquet_bytes(symbol="SPY"),
        (
            "example-bucket",
            "curated/etf_daily_prices/symbol=QQQ/year=2023/data.parquet",
        ): _sample_parquet_bytes(symbol="QQQ"),
        (
            "example-bucket",
            "curated/etf_daily_prices/symbol=QQQ/year=2024/data.parquet",
        ): _sample_parquet_bytes(symbol="QQQ"),
    }
    requests, _session_kwargs = _install_fake_boto3(monkeypatch, objects)

    result = read_s3_curated_etf_data(["SPY", "QQQ"], [2023, 2024], config)

    assert [
        f"s3://{bucket}/{key}"
        for bucket, key in requests
    ] == [
        "s3://example-bucket/curated/etf_daily_prices/symbol=SPY/year=2023/data.parquet",
        "s3://example-bucket/curated/etf_daily_prices/symbol=SPY/year=2024/data.parquet",
        "s3://example-bucket/curated/etf_daily_prices/symbol=QQQ/year=2023/data.parquet",
        "s3://example-bucket/curated/etf_daily_prices/symbol=QQQ/year=2024/data.parquet",
    ]
    assert result["symbol"].tolist() == ["QQQ", "QQQ", "SPY", "SPY"]


def test_read_s3_parquet_object_reads_parquet_bytes_from_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        ("bucket", "path/to/data.parquet"): _sample_parquet_bytes(symbol="SPY"),
    }
    requests, session_kwargs = _install_fake_boto3(monkeypatch, objects)
    config = S3MarketDataConfig(
        bucket="unused-config-bucket",
        aws_profile="research",
        region_name="us-west-2",
    )

    result = read_s3_parquet_object("s3://bucket/path/to/data.parquet", config)

    assert requests == [("bucket", "path/to/data.parquet")]
    assert session_kwargs == [{"profile_name": "research", "region_name": "us-west-2"}]
    assert result["symbol"].tolist() == ["SPY"]


def test_read_s3_parquet_files_normalizes_and_concatenates_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        ("bucket", "second.parquet"): _sample_parquet_bytes(symbol="SPY", date="2024-01-03"),
        ("bucket", "first.parquet"): _sample_parquet_bytes(symbol="QQQ", date="2024-01-02"),
    }
    _install_fake_boto3(monkeypatch, objects)

    result = read_s3_parquet_files(
        ["s3://bucket/second.parquet", "s3://bucket/first.parquet"],
        S3MarketDataConfig(bucket="bucket"),
    )

    assert result["symbol"].tolist() == ["QQQ", "SPY"]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert {"year", "month", "transactions"}.issubset(result.columns)


def test_read_s3_parquet_files_skips_individual_missing_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        ("bucket", "missing.parquet"): FakeClientError("NoSuchKey"),
        ("bucket", "present.parquet"): _sample_parquet_bytes(symbol="SPY"),
    }
    _install_fake_boto3(monkeypatch, objects)

    result = read_s3_parquet_files(
        ["s3://bucket/missing.parquet", "s3://bucket/present.parquet"],
        S3MarketDataConfig(bucket="bucket"),
    )

    assert len(result) == 1
    assert result["symbol"].tolist() == ["SPY"]


def test_read_s3_parquet_files_raises_when_all_files_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        ("bucket", "missing-a.parquet"): FakeClientError("NoSuchKey"),
        ("bucket", "missing-b.parquet"): FakeClientError("404"),
    }
    _install_fake_boto3(monkeypatch, objects)

    with pytest.raises(FileNotFoundError, match="No requested S3 parquet files"):
        read_s3_parquet_files(
            ["s3://bucket/missing-a.parquet", "s3://bucket/missing-b.parquet"],
            S3MarketDataConfig(bucket="bucket"),
        )


def test_read_s3_parquet_files_preserves_unrelated_aws_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        ("bucket", "forbidden.parquet"): FakeClientError("AccessDenied"),
    }
    _install_fake_boto3(monkeypatch, objects)

    with pytest.raises(FakeClientError, match="AccessDenied"):
        read_s3_parquet_files(
            ["s3://bucket/forbidden.parquet"],
            S3MarketDataConfig(bucket="bucket"),
        )


def test_empty_symbols_raises_value_error() -> None:
    with pytest.raises(ValueError, match="symbol"):
        read_s3_curated_etf_data([], [2024], S3MarketDataConfig(bucket="bucket"))


def test_empty_years_raises_value_error() -> None:
    with pytest.raises(ValueError, match="year"):
        read_s3_curated_etf_data(["SPY"], [], S3MarketDataConfig(bucket="bucket"))


def test_empty_s3_uris_raises_value_error() -> None:
    with pytest.raises(ValueError, match="S3 parquet URI"):
        read_s3_parquet_files([], S3MarketDataConfig(bucket="bucket"))


def test_storage_options_include_profile_and_region() -> None:
    config = S3MarketDataConfig(
        bucket="bucket",
        aws_profile="research",
        region_name="us-west-2",
    )

    assert build_s3_storage_options(config) == {
        "profile": "research",
        "client_kwargs": {"region_name": "us-west-2"},
    }


def test_storage_options_are_none_when_unconfigured() -> None:
    assert build_s3_storage_options(S3MarketDataConfig(bucket="bucket")) is None


def test_package_import_works_without_requiring_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import: Callable[..., object] = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"boto3", "botocore"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reloaded = importlib.reload(s3_reader)

    assert reloaded.S3MarketDataConfig(bucket="bucket").bucket == "bucket"


def test_read_s3_parquet_files_raises_clear_import_error_when_aws_deps_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(package_name: str) -> object:
        if package_name in {"boto3", "botocore.exceptions"}:
            raise ImportError(package_name)
        return types.ModuleType(package_name)

    monkeypatch.setattr(s3_reader.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"\.\[dev,aws\]"):
        read_s3_parquet_files(
            ["s3://bucket/present.parquet"],
            S3MarketDataConfig(bucket="bucket"),
        )
