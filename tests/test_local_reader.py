from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.data import (
    LocalMarketDataConfig,
    normalize_market_data_frame,
    read_csv_files,
    read_local_market_data,
    read_parquet_files,
)


def _sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "year": 2024,
                "month": 1,
                "symbol": "QQQ",
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1_200_000,
                "vwap": 101.5,
                "transactions": 12_000,
            },
            {
                "date": "2024-01-02",
                "year": 2024,
                "month": 1,
                "symbol": "SPY",
                "open": 201.0,
                "high": 203.0,
                "low": 200.0,
                "close": 202.0,
                "volume": 2_200_000,
                "vwap": 201.5,
                "transactions": 22_000,
            },
        ]
    )


def test_read_one_csv_file(tmp_path) -> None:
    path = tmp_path / "etf_daily.csv"
    _sample_ohlcv().to_csv(path, index=False)

    result = read_local_market_data(path)

    assert len(result) == 2
    assert result["symbol"].tolist() == ["QQQ", "SPY"]


def test_read_one_parquet_file(tmp_path) -> None:
    path = tmp_path / "etf_daily.parquet"
    _sample_ohlcv().to_parquet(path, index=False)

    result = read_local_market_data(path)

    assert len(result) == 2
    assert result["close"].tolist() == [102.0, 202.0]


def test_read_directory_with_multiple_parquet_files(tmp_path) -> None:
    first = tmp_path / "year=2024" / "part-1.parquet"
    second = tmp_path / "year=2024" / "part-2.parquet"
    first.parent.mkdir()
    _sample_ohlcv().iloc[[0]].to_parquet(first, index=False)
    _sample_ohlcv().iloc[[1]].to_parquet(second, index=False)

    result = read_local_market_data(tmp_path)

    assert len(result) == 2
    assert result["symbol"].tolist() == ["QQQ", "SPY"]


def test_read_parquet_glob_with_spaces_in_path(tmp_path) -> None:
    data_dir = tmp_path / "Market Data" / "year=2024"
    data_dir.mkdir(parents=True)
    first = data_dir / "part-1.parquet"
    second = data_dir / "part-2.parquet"
    _sample_ohlcv().iloc[[0]].to_parquet(first, index=False)
    _sample_ohlcv().iloc[[1]].to_parquet(second, index=False)

    result = read_local_market_data(str(tmp_path / "Market Data" / "**" / "*.parquet"))

    assert len(result) == 2
    assert result["symbol"].tolist() == ["QQQ", "SPY"]


def test_directory_falls_back_to_csv_when_no_parquet_files_exist(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _sample_ohlcv().to_csv(nested / "etf_daily.csv", index=False)

    result = read_local_market_data(tmp_path)

    assert len(result) == 2
    assert result["symbol"].tolist() == ["QQQ", "SPY"]


def test_normalize_sorts_by_symbol_and_date() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "symbol": "SPY",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "vwap": 1,
            },
            {
                "date": "2024-01-02",
                "symbol": "QQQ",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "vwap": 1,
            },
            {
                "date": "2024-01-01",
                "symbol": "SPY",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "vwap": 1,
            },
        ]
    )

    result = normalize_market_data_frame(df)

    assert list(zip(result["symbol"], result["date"], strict=True)) == [
        ("QQQ", pd.Timestamp("2024-01-02")),
        ("SPY", pd.Timestamp("2024-01-01")),
        ("SPY", pd.Timestamp("2024-01-03")),
    ]


def test_date_column_is_parsed_to_datetime(tmp_path) -> None:
    path = tmp_path / "etf_daily.csv"
    _sample_ohlcv().to_csv(path, index=False)

    result = read_csv_files([path])

    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_extra_columns_are_preserved(tmp_path) -> None:
    path = tmp_path / "etf_daily.parquet"
    _sample_ohlcv().to_parquet(path, index=False)

    result = read_parquet_files([path])

    assert {"year", "month", "transactions"}.issubset(result.columns)


def test_missing_required_columns_raise_value_error() -> None:
    df = _sample_ohlcv().drop(columns=["vwap"])

    with pytest.raises(ValueError, match="vwap"):
        normalize_market_data_frame(df)


def test_custom_required_columns_are_supported() -> None:
    df = _sample_ohlcv().drop(columns=["vwap"])
    config = LocalMarketDataConfig(
        required_columns=("date", "symbol", "open", "high", "low", "close", "volume")
    )

    result = normalize_market_data_frame(df, config)

    assert "vwap" not in result.columns


def test_unsupported_extension_raises_value_error(tmp_path) -> None:
    path = tmp_path / "etf_daily.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        read_local_market_data(path)


def test_missing_path_raises_file_not_found_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_local_market_data(tmp_path / "missing.parquet")


def test_unmatched_glob_raises_file_not_found_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="glob matched no files"):
        read_local_market_data(str(tmp_path / "**" / "*.parquet"))


def test_empty_directory_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="no supported"):
        read_local_market_data(tmp_path)


def test_empty_file_sequence_raises_value_error() -> None:
    with pytest.raises(ValueError, match="At least one"):
        read_csv_files([])

    with pytest.raises(ValueError, match="At least one"):
        read_parquet_files([])
