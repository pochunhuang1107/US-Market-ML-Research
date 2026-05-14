from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.data import get_feature_columns, get_target_columns
from us_market_ml.features import add_cross_asset_features, build_feature_frame

EXPECTED_FEATURE_COLUMNS = {
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "volume_zscore_20d",
    "vwap_gap",
    "qqq_spy_return_ratio_20d",
    "iwm_spy_return_ratio_20d",
    "tlt_spy_return_ratio_20d",
    "gld_spy_return_ratio_20d",
    "hyg_lqd_return_ratio_20d",
    "xlk_xlf_return_ratio_20d",
}
EXPECTED_TECHNICAL_V2_COLUMNS = {
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_20d",
    "rolling_max_drawdown_20d",
    "rolling_max_drawdown_60d",
    "close_to_sma_10",
    "close_to_sma_20",
    "close_to_sma_50",
    "sma_10_to_sma_50",
    "sma_20_to_sma_50",
    "volume_change_5d",
    "volume_change_20d",
    "dollar_volume_20d",
    "dollar_volume_change_20d",
    "return_20d_rank_pct",
    "return_60d_rank_pct",
    "volatility_20d_rank_pct",
    "close_to_sma_20_rank_pct",
    "dollar_volume_20d_rank_pct",
    "relative_strength_vs_spy_20d",
    "relative_strength_vs_qqq_20d",
    "beta_to_spy_60d",
    "correlation_to_spy_60d",
}


def _ohlcv_frame(symbols: list[str], periods: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 20.0
        for day_index, date in enumerate(dates):
            close = base_price + day_index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 10_000 + day_index * 100,
                    "vwap": close - 0.25,
                    "source_column": f"{symbol}-{day_index}",
                }
            )

    return pd.DataFrame(rows)


def test_build_feature_frame_adds_expected_columns_and_preserves_input_columns() -> None:
    df = _ohlcv_frame(["SPY", "QQQ"], periods=80)

    features = build_feature_frame(df)

    assert EXPECTED_FEATURE_COLUMNS.issubset(features.columns)
    assert "source_column" in features.columns
    assert len(features) == len(df)


def test_default_feature_set_version_keeps_existing_feature_columns() -> None:
    df = _ohlcv_frame(["SPY", "QQQ"], periods=80)

    default_features = build_feature_frame(df)
    explicit_default_features = build_feature_frame(df, "default_features_v1")

    assert list(explicit_default_features.columns) == list(default_features.columns)
    assert "return_3d" not in explicit_default_features.columns
    assert "volatility_5d" not in explicit_default_features.columns

    v2_features = build_feature_frame(df, "technical_features_v2")
    default_feature_cols = get_feature_columns(v2_features)
    assert "return_3d" not in default_feature_cols
    assert "volatility_5d" not in default_feature_cols


def test_technical_features_v2_adds_expected_columns() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=90)

    features = build_feature_frame(df, "technical_features_v2")

    assert EXPECTED_TECHNICAL_V2_COLUMNS.issubset(features.columns)
    assert len(features) == len(df)


def test_technical_features_v2_feature_columns_exclude_targets_and_labels() -> None:
    features = build_feature_frame(
        _ohlcv_frame(["SPY", "QQQ"], periods=90),
        "technical_features_v2",
    )
    features["forward_return_20d"] = 0.1
    features["excess_return_vs_spy_20d"] = 0.2
    features["rank_excess_return_vs_spy_20d"] = 1.0

    feature_cols = get_feature_columns(features, feature_set_version="technical_features_v2")

    assert get_target_columns(features)
    assert set(get_target_columns(features)).isdisjoint(feature_cols)
    assert "forward_return_20d" not in feature_cols
    assert "excess_return_vs_spy_20d" not in feature_cols
    assert "rank_excess_return_vs_spy_20d" not in feature_cols


def test_technical_features_v2_cross_sectional_rank_pct_columns_are_bounded() -> None:
    features = build_feature_frame(
        _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=90),
        "technical_features_v2",
    )
    rank_columns = [
        "return_20d_rank_pct",
        "return_60d_rank_pct",
        "volatility_20d_rank_pct",
        "close_to_sma_20_rank_pct",
        "dollar_volume_20d_rank_pct",
    ]

    for column in rank_columns:
        non_null_values = features[column].dropna()
        assert not non_null_values.empty
        assert non_null_values.between(0.0, 1.0).all()


def test_return_features_do_not_leak_across_symbols() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "symbol": "AAA",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 100.0,
                "volume": 100.0,
                "vwap": 100.0,
            },
            {
                "date": "2024-01-01",
                "symbol": "BBB",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 500.0,
                "volume": 100.0,
                "vwap": 500.0,
            },
            {
                "date": "2024-01-02",
                "symbol": "AAA",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 110.0,
                "volume": 100.0,
                "vwap": 110.0,
            },
            {
                "date": "2024-01-02",
                "symbol": "BBB",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 450.0,
                "volume": 100.0,
                "vwap": 450.0,
            },
        ]
    )

    features = build_feature_frame(df)

    first_by_symbol = features.groupby("symbol", sort=False).head(1)
    assert first_by_symbol["return_1d"].isna().all()
    second_aaa = features.loc[
        (features["symbol"] == "AAA") & (features["date"] == "2024-01-02"), "return_1d"
    ].item()
    second_bbb = features.loc[
        (features["symbol"] == "BBB") & (features["date"] == "2024-01-02"), "return_1d"
    ].item()
    assert second_aaa == pytest.approx(0.10)
    assert second_bbb == pytest.approx(-0.10)


def test_warmup_periods_produce_nan_for_rolling_features() -> None:
    df = _ohlcv_frame(["SPY"], periods=25)

    features = build_feature_frame(df)

    assert features.loc[:18, "close_to_sma_20"].isna().all()
    assert pd.notna(features.loc[19, "close_to_sma_20"])
    assert features.loc[:9, "realized_vol_10d"].isna().all()
    assert pd.notna(features.loc[10, "realized_vol_10d"])
    assert features.loc[:18, "volume_zscore_20d"].isna().all()
    assert pd.notna(features.loc[19, "volume_zscore_20d"])


def test_vwap_gap_math() -> None:
    df = _ohlcv_frame(["SPY"], periods=3)
    df.loc[0, "close"] = 105.0
    df.loc[0, "vwap"] = 100.0

    features = build_feature_frame(df)

    assert features.loc[0, "vwap_gap"] == pytest.approx(0.05)


def test_cross_asset_features_are_added_and_joined_by_date() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "SPY", "return_20d": 0.10},
            {"date": "2024-01-01", "symbol": "QQQ", "return_20d": 0.15},
            {"date": "2024-01-01", "symbol": "IWM", "return_20d": 0.03},
            {"date": "2024-01-02", "symbol": "SPY", "return_20d": 0.20},
            {"date": "2024-01-02", "symbol": "QQQ", "return_20d": 0.18},
            {"date": "2024-01-02", "symbol": "IWM", "return_20d": 0.11},
        ]
    )

    features = add_cross_asset_features(df)

    day_one_values = features.loc[features["date"] == "2024-01-01", "qqq_spy_return_ratio_20d"]
    day_two_values = features.loc[features["date"] == "2024-01-02", "qqq_spy_return_ratio_20d"]
    assert np.allclose(day_one_values, 0.05)
    assert np.allclose(day_two_values, -0.02)
    assert features.loc[
        features["date"] == "2024-01-01", "iwm_spy_return_ratio_20d"
    ].eq(-0.07).all()


def test_missing_cross_asset_symbols_produce_nan_columns() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "SPY", "return_20d": 0.10},
            {"date": "2024-01-02", "symbol": "SPY", "return_20d": 0.20},
        ]
    )

    features = add_cross_asset_features(df)

    assert "qqq_spy_return_ratio_20d" in features.columns
    assert features["qqq_spy_return_ratio_20d"].isna().all()
