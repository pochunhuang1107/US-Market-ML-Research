from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    drop_rows_with_missing_model_values,
    get_default_non_feature_columns,
    get_feature_columns,
    get_target_columns,
)


def _synthetic_etf_ohlcv(symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=90, freq="D")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 15.0
        drift = 1.0 + symbol_index * 0.1
        for day_index, date in enumerate(dates):
            close = base_price + day_index * drift
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.4,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 50_000 + day_index * 1_000,
                    "vwap": close - 0.1,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                    "source_text": f"{symbol}-{day_index}",
                }
            )

    return pd.DataFrame(rows)


def test_build_supervised_dataset_adds_feature_and_label_columns() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    assert "return_1d" in dataset.columns
    assert "close_to_sma_20" in dataset.columns
    assert "forward_return_5d" in dataset.columns
    assert "excess_return_vs_spy_20d" in dataset.columns
    assert "rank_excess_return_vs_spy_20d" in dataset.columns


def test_build_supervised_dataset_preserves_row_count() -> None:
    df = _synthetic_etf_ohlcv()

    dataset = build_supervised_dataset(df)

    assert len(dataset) == len(df)


def test_default_config_creates_expected_forward_return_columns() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    assert {"forward_return_5d", "forward_return_10d", "forward_return_20d"}.issubset(
        dataset.columns
    )


def test_default_config_creates_excess_return_vs_spy_20d() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    assert "excess_return_vs_spy_20d" in dataset.columns
    spy_excess = dataset.loc[dataset["symbol"] == "SPY", "excess_return_vs_spy_20d"]
    assert np.isclose(spy_excess.dropna(), 0.0).all()


def test_default_config_creates_rank_excess_return_vs_spy_20d() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    assert "rank_excess_return_vs_spy_20d" in dataset.columns
    day_one_ranks = dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-01"),
        "rank_excess_return_vs_spy_20d",
    ]
    assert day_one_ranks.notna().all()


def test_get_target_columns_finds_forward_excess_and_rank_columns() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.0],
            "forward_return_5d": [0.01],
            "excess_return_vs_spy_20d": [0.02],
            "rank_excess_return_vs_spy_20d": [0.5],
        }
    )

    assert get_target_columns(df) == [
        "forward_return_5d",
        "excess_return_vs_spy_20d",
        "rank_excess_return_vs_spy_20d",
    ]


def test_get_feature_columns_excludes_metadata_ohlcv_and_targets() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    feature_cols = get_feature_columns(dataset)

    excluded = get_default_non_feature_columns().union(get_target_columns(dataset))
    assert excluded.isdisjoint(feature_cols)
    assert "return_1d" in feature_cols
    assert "close_to_sma_20" in feature_cols


def test_get_feature_columns_returns_only_numeric_columns() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    feature_cols = get_feature_columns(dataset)

    assert "source_text" not in feature_cols
    assert feature_cols
    assert all(pd.api.types.is_numeric_dtype(dataset[column]) for column in feature_cols)


def test_get_feature_columns_honors_extra_exclude() -> None:
    dataset = build_supervised_dataset(_synthetic_etf_ohlcv())

    feature_cols = get_feature_columns(dataset, extra_exclude={"return_1d"})

    assert "return_1d" not in feature_cols


def test_drop_rows_with_missing_model_values_drops_only_selected_missing_values() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, np.nan, 3.0, 4.0],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
            "unused_feature": [np.nan, np.nan, np.nan, np.nan],
            "target": [0.1, 0.2, np.nan, 0.4],
        }
    )

    dropped = drop_rows_with_missing_model_values(df, ["feature_a", "feature_b"], "target")

    assert dropped.index.tolist() == [0, 3]
    assert "unused_feature" in dropped.columns
    assert len(df) == 4


def test_custom_config_can_disable_rank_generation() -> None:
    config = SupervisedDatasetConfig(rank_target_col=None)

    dataset = build_supervised_dataset(_synthetic_etf_ohlcv(), config)

    assert "rank_excess_return_vs_spy_20d" not in dataset.columns
    assert "excess_return_vs_spy_20d" in dataset.columns


def test_missing_configured_rank_target_raises_value_error() -> None:
    config = SupervisedDatasetConfig(rank_target_col="missing_target")

    with pytest.raises(ValueError, match="missing_target"):
        build_supervised_dataset(_synthetic_etf_ohlcv(), config)
