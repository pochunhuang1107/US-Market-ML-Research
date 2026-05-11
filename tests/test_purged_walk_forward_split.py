from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.splits import (
    PurgeConfig,
    WalkForwardFold,
    assign_purged_split_labels,
    purge_fold_boundaries,
    summarize_purged_split,
)


def _fold() -> WalkForwardFold:
    return WalkForwardFold(
        name="purge_example",
        train_start=pd.Timestamp("2021-01-01"),
        train_end=pd.Timestamp("2021-01-10"),
        validation_start=pd.Timestamp("2021-01-11"),
        validation_end=pd.Timestamp("2021-01-20"),
        test_start=pd.Timestamp("2021-01-21"),
        test_end=pd.Timestamp("2021-01-30"),
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=35, freq="D"),
            "row": range(35),
        }
    )


def test_purge_horizon_zero_matches_normal_split() -> None:
    df = _df()
    fold = _fold()

    purged = purge_fold_boundaries(df, fold, PurgeConfig(label_horizon_days=0))
    normal = fold.split(df)

    for purged_split, normal_split in zip(purged, normal, strict=True):
        pd.testing.assert_frame_equal(purged_split, normal_split)
    assert "split" not in df.columns


def test_purge_train_removes_rows_near_train_end() -> None:
    train, validation, test = purge_fold_boundaries(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3),
    )

    assert train["date"].max() == pd.Timestamp("2021-01-07")
    assert len(train) == 7
    assert len(validation) == 7
    assert len(test) == 10


def test_purge_validation_removes_rows_near_validation_end() -> None:
    _, validation, _ = purge_fold_boundaries(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3),
    )

    assert validation["date"].max() == pd.Timestamp("2021-01-17")
    assert validation["row"].tolist() == list(range(10, 17))


def test_purge_test_false_keeps_test_rows_unchanged_by_default() -> None:
    _, _, test = purge_fold_boundaries(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3),
    )

    assert test["date"].min() == pd.Timestamp("2021-01-21")
    assert test["date"].max() == pd.Timestamp("2021-01-30")
    assert len(test) == 10


def test_purge_test_true_removes_rows_near_test_end() -> None:
    _, _, test = purge_fold_boundaries(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3, purge_test=True),
    )

    assert test["date"].max() == pd.Timestamp("2021-01-27")
    assert len(test) == 7


def test_assign_purged_split_labels_marks_purged_rows_as_na() -> None:
    labeled = assign_purged_split_labels(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3),
    )

    assert labeled.loc[labeled["date"] == pd.Timestamp("2021-01-07"), "split"].item() == "train"
    assert pd.isna(labeled.loc[labeled["date"] == pd.Timestamp("2021-01-08"), "split"].item())
    assert (
        labeled.loc[labeled["date"] == pd.Timestamp("2021-01-17"), "split"].item()
        == "validation"
    )
    assert pd.isna(labeled.loc[labeled["date"] == pd.Timestamp("2021-01-18"), "split"].item())
    assert labeled.loc[labeled["date"] == pd.Timestamp("2021-01-30"), "split"].item() == "test"


def test_summarize_purged_split_returns_before_after_and_purged_counts() -> None:
    summary = summarize_purged_split(
        _df(),
        _fold(),
        PurgeConfig(label_horizon_days=3, purge_test=True),
    )

    assert summary == {
        "train_rows_before": 10,
        "train_rows_after": 7,
        "validation_rows_before": 10,
        "validation_rows_after": 7,
        "test_rows_before": 10,
        "test_rows_after": 7,
        "train_rows_purged": 3,
        "validation_rows_purged": 3,
        "test_rows_purged": 3,
    }


def test_missing_date_col_raises_value_error_for_purged_split() -> None:
    df = pd.DataFrame([{"timestamp": "2021-01-01", "row": 1}])

    with pytest.raises(ValueError, match="date"):
        purge_fold_boundaries(df, _fold(), PurgeConfig(label_horizon_days=3))


def test_purge_config_validates_horizon_and_date_col() -> None:
    with pytest.raises(ValueError, match="label_horizon_days"):
        PurgeConfig(label_horizon_days=-1)

    with pytest.raises(ValueError, match="date_col"):
        PurgeConfig(date_col="")
