from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.universe import (
    DynamicEligibilityConfig,
    apply_dynamic_universe_eligibility,
    dynamic_eligibility_mask,
    min_symbol_count_per_date,
    select_fold_train_feature_columns,
)


def test_dynamic_eligibility_excludes_symbols_until_enough_lookback() -> None:
    df = pd.DataFrame(
        [
            _row("2020-01-01", "AAA", 1.0, 0.01),
            _row("2020-01-02", "AAA", 1.0, 0.01),
            _row("2020-01-03", "AAA", 1.0, 0.01),
            _row("2020-01-04", "AAA", 1.0, 0.01),
            _row("2020-01-04", "LATE", 1.0, 0.02),
            _row("2020-01-05", "LATE", 1.0, 0.02),
            _row("2020-01-06", "LATE", 1.0, 0.02),
        ]
    )

    eligible = apply_dynamic_universe_eligibility(
        df,
        feature_cols=["feature"],
        target_col="target",
        require_target=True,
        config=DynamicEligibilityConfig(min_history_rows=3),
    )

    assert eligible[["date", "symbol"]].to_records(index=False).tolist() == [
        (pd.Timestamp("2020-01-03"), "AAA"),
        (pd.Timestamp("2020-01-04"), "AAA"),
        (pd.Timestamp("2020-01-06"), "LATE"),
    ]


def test_dynamic_eligibility_uses_full_history_for_scoring_split() -> None:
    history_df = pd.DataFrame(
        [
            _row("2020-01-01", "AAA", 1.0, 0.01),
            _row("2020-01-02", "AAA", 1.0, 0.01),
            _row("2020-01-03", "AAA", 1.0, 0.01),
            _row("2020-01-04", "AAA", 1.0, None),
        ]
    )
    scoring_df = history_df.loc[history_df["date"] == pd.Timestamp("2020-01-04")].copy()

    mask = dynamic_eligibility_mask(
        scoring_df,
        feature_cols=["feature"],
        target_col="target",
        require_target=False,
        config=DynamicEligibilityConfig(min_history_rows=3),
        history_df=history_df,
    )

    assert mask.tolist() == [True]


def test_training_eligibility_requires_target_when_requested() -> None:
    df = pd.DataFrame(
        [
            _row("2020-01-01", "AAA", 1.0, 0.01),
            _row("2020-01-02", "AAA", 1.0, None),
        ]
    )

    train_mask = dynamic_eligibility_mask(
        df,
        feature_cols=["feature"],
        target_col="target",
        require_target=True,
        config=DynamicEligibilityConfig(min_history_rows=1),
    )
    scoring_mask = dynamic_eligibility_mask(
        df,
        feature_cols=["feature"],
        target_col="target",
        require_target=False,
        config=DynamicEligibilityConfig(min_history_rows=1),
    )

    assert train_mask.tolist() == [True, False]
    assert scoring_mask.tolist() == [True, True]


def test_fold_train_feature_selection_uses_only_training_availability() -> None:
    train_df = pd.DataFrame(
        {
            "stable_feature": [1.0, 2.0, 3.0],
            "future_only_feature": [None, None, None],
            "sparse_feature": [1.0, None, None],
        }
    )

    selected = select_fold_train_feature_columns(
        train_df,
        ["stable_feature", "future_only_feature", "sparse_feature"],
        min_non_null_ratio=0.5,
    )

    assert selected == ["stable_feature"]


def test_min_symbol_count_per_date_checks_top_k_support() -> None:
    eligible = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02"],
            ),
            "symbol": ["AAA", "BBB", "AAA"],
        }
    )

    assert min_symbol_count_per_date(eligible) == 1
    assert min_symbol_count_per_date(eligible) < 2


def test_invalid_min_history_rows_raise_value_error() -> None:
    with pytest.raises(ValueError, match="min_history_rows"):
        DynamicEligibilityConfig(min_history_rows=0)


def _row(date: str, symbol: str, feature: float, target: float | None) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "feature": feature,
        "target": target,
    }
