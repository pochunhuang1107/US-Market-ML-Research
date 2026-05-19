from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.data import SupervisedDatasetConfig, build_supervised_dataset, get_feature_columns
from us_market_ml.qqq_relative_targets import (
    add_qqq_relative_targets,
    target_horizon_from_variant,
)


def test_add_qqq_relative_targets_creates_excess_return_vs_qqq() -> None:
    df = _price_frame(days=8)

    out = add_qqq_relative_targets(
        df,
        excess_return_horizons=(2,),
        return_drawdown_horizons=(),
    )
    aaa_day0 = out.loc[out["symbol"].eq("AAA")].iloc[0]

    aaa_forward = 104.0 / 100.0 - 1.0
    qqq_forward = 202.0 / 200.0 - 1.0
    assert aaa_day0["excess_return_vs_qqq_2d"] == pytest.approx(
        aaa_forward - qqq_forward
    )
    assert pd.Timestamp(aaa_day0["forward_return_end_date_2d"]) == pd.Timestamp(
        "2020-01-03"
    )


def test_qqq_relative_target_horizon_does_not_use_later_future_prices() -> None:
    base = _price_frame(days=8)
    changed_late_future = base.copy()
    changed_late_future.loc[
        changed_late_future["date"].gt(pd.Timestamp("2020-01-03")),
        "close",
    ] *= 10.0

    base_targets = add_qqq_relative_targets(
        base,
        excess_return_horizons=(2,),
        return_drawdown_horizons=(),
    )
    changed_targets = add_qqq_relative_targets(
        changed_late_future,
        excess_return_horizons=(2,),
        return_drawdown_horizons=(),
    )

    base_day0 = base_targets.loc[
        base_targets["symbol"].eq("AAA") & base_targets["date"].eq(pd.Timestamp("2020-01-01")),
        "excess_return_vs_qqq_2d",
    ].item()
    changed_day0 = changed_targets.loc[
        changed_targets["symbol"].eq("AAA")
        & changed_targets["date"].eq(pd.Timestamp("2020-01-01")),
        "excess_return_vs_qqq_2d",
    ].item()
    assert changed_day0 == pytest.approx(base_day0)


def test_qqq_return_drawdown_score_and_class_target_are_created() -> None:
    df = _price_frame(days=8)

    out = add_qqq_relative_targets(
        df,
        excess_return_horizons=(3,),
        return_drawdown_horizons=(3,),
        classification_horizons=(3,),
    )

    assert "return_drawdown_score_vs_qqq_3d" in out.columns
    assert "qqq_outperformance_class_3d" in out.columns
    assert out.loc[out["symbol"].eq("AAA"), "qqq_outperformance_class_3d"].dropna().isin(
        [0, 1]
    ).all()


def test_dataset_builder_excludes_qqq_relative_targets_from_feature_columns() -> None:
    dataset = build_supervised_dataset(
        _ohlcv_frame(days=90),
        SupervisedDatasetConfig(
            benchmark_symbol="QQQ",
            excess_return_horizons=(20,),
            rank_target_col=None,
            qqq_relative_return_drawdown_horizons=(20,),
        ),
    )

    feature_cols = get_feature_columns(dataset)

    assert "excess_return_vs_qqq_20d" not in feature_cols
    assert "return_drawdown_score_vs_qqq_20d" not in feature_cols


def test_target_horizon_from_variant_parses_supported_names() -> None:
    assert target_horizon_from_variant("excess_return_vs_qqq_63d") == 63
    assert target_horizon_from_variant("return_drawdown_score_vs_qqq_126d") == 126


def _price_frame(days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    for symbol, start, step in [("QQQ", 200.0, 1.0), ("AAA", 100.0, 2.0)]:
        for day, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": start + day * step})
    return pd.DataFrame(rows)


def _ohlcv_frame(days: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    for index, symbol in enumerate(["QQQ", "SPY", "AAA"]):
        for day, date in enumerate(dates):
            close = 100.0 + index * 20.0 + day * (1.0 + index * 0.1)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000 + day,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)
