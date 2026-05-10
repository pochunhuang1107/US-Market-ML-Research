from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.splits import (
    WalkForwardFold,
    assign_split_labels,
    make_annual_walk_forward_folds,
)


def _example_fold() -> WalkForwardFold:
    return WalkForwardFold(
        name="example",
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-12-31"),
        validation_start=pd.Timestamp("2021-01-01"),
        validation_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )


def test_walk_forward_fold_split_returns_expected_rows() -> None:
    df = pd.DataFrame(
        [
            {"date": "2019-12-31", "symbol": "AAA", "feature": 0},
            {"date": "2020-06-01", "symbol": "AAA", "feature": 1},
            {"date": "2021-06-01", "symbol": "AAA", "feature": 2},
            {"date": "2022-06-01", "symbol": "AAA", "feature": 3},
            {"date": "2023-01-01", "symbol": "AAA", "feature": 4},
        ]
    )

    train, validation, test = _example_fold().split(df)

    assert train["feature"].tolist() == [1]
    assert validation["feature"].tolist() == [2]
    assert test["feature"].tolist() == [3]
    assert list(train.columns) == list(df.columns)


def test_walk_forward_fold_boundaries_are_inclusive() -> None:
    df = pd.DataFrame(
        [
            {"date": "2020-01-01", "row": "train_start"},
            {"date": "2020-12-31", "row": "train_end"},
            {"date": "2021-01-01", "row": "validation_start"},
            {"date": "2021-12-31", "row": "validation_end"},
            {"date": "2022-01-01", "row": "test_start"},
            {"date": "2022-12-31", "row": "test_end"},
        ]
    )

    train, validation, test = _example_fold().split(df)

    assert train["row"].tolist() == ["train_start", "train_end"]
    assert validation["row"].tolist() == ["validation_start", "validation_end"]
    assert test["row"].tolist() == ["test_start", "test_end"]


def test_walk_forward_fold_split_preserves_row_order_within_outputs() -> None:
    df = pd.DataFrame(
        [
            {"date": "2022-03-01", "row": "test_later"},
            {"date": "2020-05-01", "row": "train_later"},
            {"date": "2021-08-01", "row": "validation_later"},
            {"date": "2020-02-01", "row": "train_earlier"},
            {"date": "2022-01-15", "row": "test_earlier"},
            {"date": "2021-03-01", "row": "validation_earlier"},
        ]
    )

    train, validation, test = _example_fold().split(df)

    assert train["row"].tolist() == ["train_later", "train_earlier"]
    assert validation["row"].tolist() == ["validation_later", "validation_earlier"]
    assert test["row"].tolist() == ["test_later", "test_earlier"]


def test_walk_forward_fold_invalid_overlapping_ranges_raise_value_error() -> None:
    with pytest.raises(ValueError, match="train_end must be earlier than validation_start"):
        WalkForwardFold(
            name="invalid",
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2021-01-01"),
            validation_start=pd.Timestamp("2021-01-01"),
            validation_end=pd.Timestamp("2021-12-31"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2022-12-31"),
        )


def test_make_annual_walk_forward_folds_produces_expected_three_folds() -> None:
    folds = make_annual_walk_forward_folds(
        start_year=2018,
        first_validation_year=2022,
        last_test_year=2025,
    )

    assert folds == [
        WalkForwardFold(
            name="fold_1",
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2021-12-31"),
            validation_start=pd.Timestamp("2022-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
            test_start=pd.Timestamp("2023-01-01"),
            test_end=pd.Timestamp("2023-12-31"),
        ),
        WalkForwardFold(
            name="fold_2",
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2022-12-31"),
            validation_start=pd.Timestamp("2023-01-01"),
            validation_end=pd.Timestamp("2023-12-31"),
            test_start=pd.Timestamp("2024-01-01"),
            test_end=pd.Timestamp("2024-12-31"),
        ),
        WalkForwardFold(
            name="fold_3",
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2023-12-31"),
            validation_start=pd.Timestamp("2024-01-01"),
            validation_end=pd.Timestamp("2024-12-31"),
            test_start=pd.Timestamp("2025-01-01"),
            test_end=pd.Timestamp("2025-12-31"),
        ),
    ]


def test_make_annual_walk_forward_folds_requires_at_least_one_fold() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        make_annual_walk_forward_folds(
            start_year=2020,
            first_validation_year=2022,
            last_test_year=2022,
        )


def test_assign_split_labels_marks_fold_rows_and_outside_rows() -> None:
    df = pd.DataFrame(
        [
            {"date": "2019-12-31", "row": "before"},
            {"date": "2020-06-01", "row": "train"},
            {"date": "2021-06-01", "row": "validation"},
            {"date": "2022-06-01", "row": "test"},
            {"date": "2023-01-01", "row": "after"},
        ]
    )

    labeled = assign_split_labels(df, _example_fold())

    assert labeled["split"].tolist()[:4] == [pd.NA, "train", "validation", "test"]
    assert pd.isna(labeled.loc[4, "split"])
    assert "split" not in df.columns


def test_missing_date_col_raises_value_error() -> None:
    df = pd.DataFrame([{"timestamp": "2020-01-01", "row": "missing_date"}])

    with pytest.raises(ValueError, match="date"):
        _example_fold().split(df)

    with pytest.raises(ValueError, match="trade_date"):
        assign_split_labels(df, _example_fold(), date_col="trade_date")
