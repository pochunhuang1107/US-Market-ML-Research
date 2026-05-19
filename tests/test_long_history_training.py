from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.features import (
    LONG_HISTORY_REGIME_FEATURE_COLUMNS,
    build_feature_frame,
)
from us_market_ml.long_history_training import (
    build_long_history_dataset,
    build_long_history_run_specs,
    build_rebalance_weights,
    build_training_windows,
    dataset_start_for_required_symbols,
    preflight_counts_for_spec,
    run_spec_is_structurally_valid,
)


def test_long_history_feature_columns_exist_and_are_selectable() -> None:
    features = build_feature_frame(_market_frame(days=900), "long_history_regime_features_v1")

    assert set(LONG_HISTORY_REGIME_FEATURE_COLUMNS).issubset(features.columns)
    assert "qqq_return_252d_zscore_expanding" in features.columns
    assert "symbol_minus_qqq_return_63d" in features.columns


def test_long_history_expanding_features_do_not_use_future_rows() -> None:
    df = _market_frame(days=900)
    cutoff = pd.Timestamp("2021-12-01")
    future_changed = df.copy()
    future_changed.loc[future_changed["date"].gt(cutoff), "close"] *= 10.0

    base = build_feature_frame(df, "long_history_regime_features_v1")
    changed = build_feature_frame(future_changed, "long_history_regime_features_v1")
    checked = [
        "qqq_return_252d_zscore_expanding",
        "symbol_return_252d_zscore_expanding",
        "symbol_252d_return_minus_symbol_long_run_mean",
    ]

    pd.testing.assert_frame_equal(
        base.loc[base["date"].le(cutoff), checked].reset_index(drop=True),
        changed.loc[changed["date"].le(cutoff), checked].reset_index(drop=True),
    )


def test_dataset_start_for_required_symbols_uses_common_qqq_spy_coverage() -> None:
    df = _market_frame(days=20)
    df = df.loc[~(df["symbol"].eq("SPY") & df["date"].lt(pd.Timestamp("2020-01-05")))]

    assert dataset_start_for_required_symbols(df) == pd.Timestamp("2020-01-05")


def test_build_training_windows_for_required_modes() -> None:
    dataset_start = pd.Timestamp("1999-03-10")

    frozen = build_training_windows(
        "frozen_base_1999_2018",
        dataset_start=dataset_start,
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    yearly = build_training_windows(
        "anchored_expanding_yearly",
        dataset_start=dataset_start,
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    quarterly = build_training_windows(
        "anchored_expanding_quarterly",
        dataset_start=dataset_start,
        start_date="2019-01-01",
        end_date="2019-06-30",
    )

    assert frozen[0].train_start == dataset_start
    assert frozen[0].train_end == pd.Timestamp("2018-12-31")
    assert [window.name for window in yearly] == ["year_2019", "year_2020"]
    assert yearly[1].train_end == pd.Timestamp("2019-12-31")
    assert [window.name for window in quarterly] == ["quarter_2019_q1", "quarter_2019_q2"]


def test_smoke_config_generation_covers_training_mode_families() -> None:
    specs = build_long_history_run_specs(sweep_profile="smoke")

    assert len(specs) == 24
    assert {
        "frozen_base_1999_2018",
        "anchored_expanding_yearly",
        "anchored_expanding_quarterly",
        "recency_weighted_expanding",
        "base_recent_blend",
        "mlp_pretrain_finetune",
    }.issubset({spec.training_mode for spec in specs})


def test_recency_weighted_and_base_blend_specs_are_structurally_valid() -> None:
    specs = build_long_history_run_specs(sweep_profile="smoke")
    recency = next(spec for spec in specs if spec.training_mode == "recency_weighted_expanding")
    blend = next(spec for spec in specs if spec.training_mode == "base_recent_blend")
    mlp = next(spec for spec in specs if spec.training_mode == "mlp_pretrain_finetune")

    assert run_spec_is_structurally_valid(recency) == (True, "")
    assert run_spec_is_structurally_valid(blend) == (True, "")
    valid, reason = run_spec_is_structurally_valid(mlp)
    assert valid is False
    assert "mlp_pretrain_finetune_intentionally_limited" in reason


def test_preflight_counts_use_dynamic_point_in_time_eligibility() -> None:
    raw = _market_frame(days=820, start="2017-01-01", symbols=("QQQ", "SPY", "AAA", "LATE"))
    raw = raw.loc[
        ~(
            raw["symbol"].eq("LATE")
            & raw["date"].lt(pd.Timestamp("2018-06-01"))
        )
    ].copy()
    dataset = build_long_history_dataset(raw)
    spec = next(
        spec
        for spec in build_long_history_run_specs(sweep_profile="smoke")
        if spec.training_mode == "frozen_base_1999_2018"
        and spec.target_variant == "excess_return_vs_qqq_20d"
        and spec.allocation_rule == "top1"
    )

    counts = preflight_counts_for_spec(
        dataset,
        spec,
        min_history_rows=252,
        min_train_eligible_rows=1,
        start_date="2019-01-01",
        end_date="2019-03-31",
    )

    assert counts["train_eligible_row_count"] > 0
    assert counts["train_symbol_count"] >= 2
    assert counts["train_symbol_count"] < raw["symbol"].nunique()
    assert counts["valid_config"] is True


def test_build_rebalance_weights_supports_requested_allocation_rules() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    rank_weighted = build_rebalance_weights(
        predictions,
        allocation_rule="top5_rank_weighted_mild",
    )
    capped = build_rebalance_weights(
        predictions,
        allocation_rule="top5_score_weighted_cap035",
    )

    assert rank_weighted["target_weight"].sum() == pytest.approx(1.0)
    assert rank_weighted.sort_values("target_weight", ascending=False)["symbol"].iloc[0] == "A"
    assert capped["target_weight"].sum() == pytest.approx(1.0)
    assert capped["target_weight"].max() <= 0.350001


def _market_frame(
    days: int,
    start: str = "2020-01-01",
    symbols: tuple[str, ...] = ("QQQ", "SPY", "AAA", "XLK", "IWM", "TLT", "GLD"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, periods=days, freq="D")
    for symbol_index, symbol in enumerate(symbols):
        base = 100.0 + symbol_index * 15.0
        slope = 0.08 + symbol_index * 0.01
        for day, date in enumerate(dates):
            close = base + day * slope + (day % 31) * 0.02 * (symbol_index + 1)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 1000 + day,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)
