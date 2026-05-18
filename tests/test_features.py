from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.data import get_feature_columns, get_target_columns
from us_market_ml.features import (
    BENCHMARK_GAP_FEATURE_COLUMNS,
    BENCHMARK_GAP_V2_FEATURE_COLUMNS,
    add_cross_asset_features,
    build_feature_frame,
)

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
EXPECTED_BENCHMARK_GAP_COLUMNS = set(BENCHMARK_GAP_FEATURE_COLUMNS)
EXPECTED_BENCHMARK_GAP_V2_COLUMNS = set(BENCHMARK_GAP_V2_FEATURE_COLUMNS)


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


def test_benchmark_gap_features_v1_adds_expected_columns() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=280)

    features = build_feature_frame(df, "benchmark_gap_features_v1")

    assert EXPECTED_BENCHMARK_GAP_COLUMNS.issubset(features.columns)
    assert "return_3d" not in features.columns
    assert "volatility_5d" not in features.columns
    feature_cols = get_feature_columns(
        features,
        feature_set_version="benchmark_gap_features_v1",
    )
    assert "spy_return_63d" in feature_cols
    assert "symbol_return_252d_zscore_vs_expanding_history" in feature_cols
    assert "return_3d" not in feature_cols


def test_benchmark_gap_features_v2_adds_expected_columns_and_pipeline_selects_them() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=820)

    v1_features = build_feature_frame(df, "benchmark_gap_features_v1")
    v2_features = build_feature_frame(df, "benchmark_gap_features_v2")

    assert EXPECTED_BENCHMARK_GAP_COLUMNS.issubset(v2_features.columns)
    assert EXPECTED_BENCHMARK_GAP_V2_COLUMNS.issubset(v2_features.columns)
    assert "return_3d" not in v2_features.columns
    assert "volatility_5d" not in v2_features.columns

    v1_feature_cols = get_feature_columns(
        v1_features,
        feature_set_version="benchmark_gap_features_v1",
    )
    v2_feature_cols = get_feature_columns(
        v2_features,
        feature_set_version="benchmark_gap_features_v2",
    )
    assert "qqq_vs_spy_return_126d" in v2_feature_cols
    assert "is_qqq_x_qqq_return_126d_zscore_756d" in v2_feature_cols
    assert "spy_return_63d" in v2_feature_cols
    assert "return_3d" not in v2_feature_cols
    assert len(v2_feature_cols) > len(v1_feature_cols)


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


def test_non_benchmark_gap_feature_sets_exclude_benchmark_gap_columns() -> None:
    features = build_feature_frame(
        _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=820),
        "benchmark_gap_features_v2",
    )

    default_feature_cols = get_feature_columns(features, feature_set_version="default_features_v1")
    technical_v2_feature_cols = get_feature_columns(
        features,
        feature_set_version="technical_features_v2",
    )

    assert "spy_return_63d" not in default_feature_cols
    assert "symbol_return_252d" not in default_feature_cols
    assert "qqq_vs_spy_return_126d" not in default_feature_cols
    assert "spy_return_63d" not in technical_v2_feature_cols
    assert "symbol_return_252d" not in technical_v2_feature_cols
    assert "qqq_vs_spy_return_126d" not in technical_v2_feature_cols


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


def test_benchmark_gap_spy_qqq_features_use_same_date_trailing_history() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=280)
    df.loc[df["symbol"].eq("SPY"), "close"] = np.arange(100.0, 380.0)
    df.loc[df["symbol"].eq("QQQ"), "close"] = np.arange(200.0, 480.0) * 1.2
    df.loc[df["symbol"].eq("IWM"), "close"] = np.arange(50.0, 330.0) * 0.8

    features = build_feature_frame(df, "benchmark_gap_features_v1")
    iwm_day = features.loc[
        (features["symbol"] == "IWM") & (features["date"] == pd.Timestamp("2024-03-04"))
    ].iloc[0]
    qqq_day = features.loc[
        (features["symbol"] == "QQQ") & (features["date"] == pd.Timestamp("2024-03-04"))
    ].iloc[0]

    assert iwm_day["spy_return_63d"] == pytest.approx(163.0 / 100.0 - 1.0)
    assert iwm_day["qqq_return_63d"] == pytest.approx((263.0 * 1.2) / (200.0 * 1.2) - 1.0)
    assert iwm_day["qqq_minus_spy_return_63d"] == pytest.approx(
        iwm_day["qqq_return_63d"] - iwm_day["spy_return_63d"]
    )
    assert iwm_day["symbol_return_63d_minus_spy_return_63d"] == pytest.approx(
        ((113.0 * 0.8) / (50.0 * 0.8) - 1.0) - iwm_day["spy_return_63d"]
    )
    assert qqq_day["qqq_div_spy_price_ratio_return_63d"] == pytest.approx(
        ((263.0 * 1.2) / 163.0) / ((200.0 * 1.2) / 100.0) - 1.0
    )


def test_benchmark_gap_v2_ratio_features_are_date_aligned() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=300)
    df.loc[df["symbol"].eq("SPY"), "close"] = np.arange(100.0, 400.0)
    df.loc[df["symbol"].eq("QQQ"), "close"] = np.arange(200.0, 500.0) * 1.5
    df.loc[df["symbol"].eq("IWM"), "close"] = np.arange(50.0, 350.0) * 0.7

    features = build_feature_frame(df, "benchmark_gap_features_v2")
    date = pd.Timestamp("2024-03-04")
    iwm_day = features.loc[(features["symbol"] == "IWM") & (features["date"] == date)].iloc[0]
    qqq_day = features.loc[(features["symbol"] == "QQQ") & (features["date"] == date)].iloc[0]

    expected_ratio_return = ((263.0 * 1.5) / 163.0) / ((200.0 * 1.5) / 100.0) - 1.0
    assert iwm_day["qqq_vs_spy_return_63d"] == pytest.approx(expected_ratio_return)
    assert qqq_day["qqq_vs_spy_return_63d"] == pytest.approx(expected_ratio_return)
    assert iwm_day["qqq_vs_spy_return_63d"] == pytest.approx(
        iwm_day["qqq_div_spy_price_ratio_return_63d"]
    )


def test_benchmark_gap_v2_features_do_not_use_future_rows() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=900)
    cutoff_date = pd.Timestamp("2025-12-01")
    future_df = df.copy()
    future_df.loc[future_df["date"].gt(cutoff_date), "close"] *= 10.0

    base_features = build_feature_frame(df, "benchmark_gap_features_v2")
    changed_future_features = build_feature_frame(future_df, "benchmark_gap_features_v2")

    checked_columns = [
        "qqq_vs_spy_return_zscore_252d",
        "spy_return_63d_zscore_756d",
        "qqq_return_126d_percentile_756d",
        "symbol_minus_max_spy_qqq_return_126d",
    ]
    base_slice = base_features.loc[
        base_features["symbol"].eq("IWM") & base_features["date"].le(cutoff_date),
        checked_columns,
    ].reset_index(drop=True)
    changed_slice = changed_future_features.loc[
        changed_future_features["symbol"].eq("IWM")
        & changed_future_features["date"].le(cutoff_date),
        checked_columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(base_slice, changed_slice)


def test_benchmark_gap_v2_identity_flags_are_correct() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "XLK", "XLU", "GLD", "IWM"], periods=300)

    features = build_feature_frame(df, "benchmark_gap_features_v2")
    latest = (
        features.sort_values(["symbol", "date"], kind="mergesort")
        .groupby("symbol", sort=False)
        .tail(1)
        .set_index("symbol")
    )

    assert latest.loc["SPY", "is_spy"] == 1
    assert latest.loc["SPY", "is_spy_or_qqq"] == 1
    assert latest.loc["QQQ", "is_qqq"] == 1
    assert latest.loc["QQQ", "is_growth_like_anchor"] == 1
    assert latest.loc["XLK", "is_growth_like_anchor"] == 1
    assert latest.loc["XLU", "is_defensive_or_bond_like"] == 1
    assert latest.loc["GLD", "is_commodity_like"] == 1
    assert latest.loc["IWM", "is_spy_or_qqq"] == 0
    assert latest.loc["IWM", "is_growth_like_anchor"] == 0


def test_benchmark_gap_v2_symbol_minus_max_spy_qqq_return_values() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "IWM"], periods=300)
    day_numbers = np.arange(300, dtype=float)
    df.loc[df["symbol"].eq("SPY"), "close"] = 100.0 + day_numbers
    df.loc[df["symbol"].eq("QQQ"), "close"] = 100.0 + day_numbers * 2.0
    df.loc[df["symbol"].eq("IWM"), "close"] = 100.0 + day_numbers * 3.0

    features = build_feature_frame(df, "benchmark_gap_features_v2")
    iwm_day = features.loc[
        (features["symbol"] == "IWM") & (features["date"] == pd.Timestamp("2024-03-04"))
    ].iloc[0]

    symbol_return = (100.0 + 63.0 * 3.0) / 100.0 - 1.0
    spy_return = (100.0 + 63.0) / 100.0 - 1.0
    qqq_return = (100.0 + 63.0 * 2.0) / 100.0 - 1.0
    assert iwm_day["symbol_minus_max_spy_qqq_return_63d"] == pytest.approx(
        symbol_return - max(spy_return, qqq_return)
    )
    assert iwm_day["symbol_return_63d_rank_vs_spy_qqq"] == pytest.approx(1.0)


def test_benchmark_gap_expanding_return_history_is_shifted() -> None:
    df = _ohlcv_frame(["SPY", "QQQ", "AAA"], periods=280)
    aaa_mask = df["symbol"].eq("AAA")
    day_numbers = np.arange(int(aaa_mask.sum()), dtype=float)
    df.loc[aaa_mask, "close"] = 100.0 + day_numbers + day_numbers**2 * 0.01

    features = build_feature_frame(df, "benchmark_gap_features_v1")
    aaa = features.loc[features["symbol"].eq("AAA")].reset_index(drop=True)
    return_252 = aaa.loc[252, "symbol_return_252d"]
    return_253 = aaa.loc[253, "symbol_return_252d"]

    assert pd.isna(aaa.loc[252, "symbol_expanding_mean_return_252d_shifted"])
    assert aaa.loc[253, "symbol_expanding_mean_return_252d_shifted"] == pytest.approx(
        return_252
    )
    assert aaa.loc[254, "symbol_expanding_mean_return_252d_shifted"] == pytest.approx(
        (return_252 + return_253) / 2.0
    )
    assert aaa.loc[253, "symbol_return_252d_minus_expanding_mean"] == pytest.approx(
        return_253 - return_252
    )
    assert pd.isna(aaa.loc[253, "symbol_return_252d_zscore_vs_expanding_history"])
    assert pd.notna(aaa.loc[254, "symbol_return_252d_zscore_vs_expanding_history"])


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
