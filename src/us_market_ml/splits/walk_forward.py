"""Leakage-safe walk-forward splits for ETF ML research.

Random row splits are not appropriate for this project because adjacent market
rows are time ordered and supervised labels may intentionally point to future
returns. Walk-forward folds keep model selection and evaluation chronological:
training rows come first, validation rows follow, and test rows remain last.

Future-looking labels are valid supervised targets, but split boundaries still
define which rows are available during training, validation, and final test
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    """A single chronological train/validation/test fold.

    Date ranges are inclusive on both start and end. The fold enforces strictly
    increasing train, validation, and test windows so future rows cannot leak
    into earlier training data.
    """

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        self.train_start = pd.Timestamp(self.train_start)
        self.train_end = pd.Timestamp(self.train_end)
        self.validation_start = pd.Timestamp(self.validation_start)
        self.validation_end = pd.Timestamp(self.validation_end)
        self.test_start = pd.Timestamp(self.test_start)
        self.test_end = pd.Timestamp(self.test_end)
        self._validate_ranges()

    def train_mask(self, df: pd.DataFrame, date_col: str = "date") -> pd.Series:
        """Return an inclusive boolean mask for this fold's training rows."""

        return _date_range_mask(df, self.train_start, self.train_end, date_col)

    def validation_mask(self, df: pd.DataFrame, date_col: str = "date") -> pd.Series:
        """Return an inclusive boolean mask for this fold's validation rows."""

        return _date_range_mask(df, self.validation_start, self.validation_end, date_col)

    def test_mask(self, df: pd.DataFrame, date_col: str = "date") -> pd.Series:
        """Return an inclusive boolean mask for this fold's test rows."""

        return _date_range_mask(df, self.test_start, self.test_end, date_col)

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return train, validation, and test subsets for this fold.

        The input columns and row order within each subset are preserved. The
        original date column is only converted on a temporary series used for
        comparison, so callers do not get hidden dtype mutations.
        """

        train = df.loc[self.train_mask(df, date_col)].copy()
        validation = df.loc[self.validation_mask(df, date_col)].copy()
        test = df.loc[self.test_mask(df, date_col)].copy()
        return train, validation, test

    def _validate_ranges(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError("train_start must be less than or equal to train_end.")
        if self.validation_start > self.validation_end:
            raise ValueError("validation_start must be less than or equal to validation_end.")
        if self.test_start > self.test_end:
            raise ValueError("test_start must be less than or equal to test_end.")
        if self.train_end >= self.validation_start:
            raise ValueError("train_end must be earlier than validation_start.")
        if self.validation_end >= self.test_start:
            raise ValueError("validation_end must be earlier than test_start.")


def make_annual_walk_forward_folds(
    start_year: int,
    first_validation_year: int,
    last_test_year: int,
    train_start_month_day: str = "01-01",
    validation_month_day_start: str = "01-01",
    validation_month_day_end: str = "12-31",
    test_month_day_start: str = "01-01",
    test_month_day_end: str = "12-31",
) -> list[WalkForwardFold]:
    """Create expanding-window annual walk-forward folds.

    Each fold trains from ``start_year`` through the day before that fold's
    validation window, validates on one calendar-style year, and tests on the
    following year. This chronological structure avoids the future training
    leakage that random row splits would introduce in time-series market data.
    """

    if first_validation_year >= last_test_year:
        raise ValueError(
            "Inputs must allow at least one fold: first_validation_year must be "
            "earlier than last_test_year."
        )

    folds: list[WalkForwardFold] = []
    for fold_number, validation_year in enumerate(
        range(first_validation_year, last_test_year),
        start=1,
    ):
        validation_start = _timestamp_from_year_month_day(
            validation_year,
            validation_month_day_start,
            "validation_month_day_start",
        )
        train_start = _timestamp_from_year_month_day(
            start_year,
            train_start_month_day,
            "train_start_month_day",
        )
        train_end = validation_start - pd.Timedelta(days=1)
        validation_end = _timestamp_from_year_month_day(
            validation_year,
            validation_month_day_end,
            "validation_month_day_end",
        )
        test_start = _timestamp_from_year_month_day(
            validation_year + 1,
            test_month_day_start,
            "test_month_day_start",
        )
        test_end = _timestamp_from_year_month_day(
            validation_year + 1,
            test_month_day_end,
            "test_month_day_end",
        )

        folds.append(
            WalkForwardFold(
                name=f"fold_{fold_number}",
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    if not folds:
        raise ValueError("Inputs cannot produce at least one walk-forward fold.")

    return folds


def assign_split_labels(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    date_col: str = "date",
    output_col: str = "split",
) -> pd.DataFrame:
    """Return a copy with rows labeled as train, validation, test, or missing.

    The labels describe availability under one walk-forward fold. Rows outside
    the fold receive ``pd.NA`` and should not be treated as in-fold training or
    evaluation data.
    """

    _validate_date_col(df, date_col)
    out = df.copy()
    out[output_col] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[fold.train_mask(out, date_col), output_col] = "train"
    out.loc[fold.validation_mask(out, date_col), output_col] = "validation"
    out.loc[fold.test_mask(out, date_col), output_col] = "test"
    return out


def _date_range_mask(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str,
) -> pd.Series:
    _validate_date_col(df, date_col)
    dates = pd.to_datetime(df[date_col])
    return (dates >= start) & (dates <= end)


def _validate_date_col(df: pd.DataFrame, date_col: str) -> None:
    if date_col not in df.columns:
        raise ValueError(f"Input DataFrame is missing required column: {date_col}")


def _timestamp_from_year_month_day(year: int, month_day: str, parameter_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(f"{year}-{month_day}")
    except ValueError as exc:
        raise ValueError(
            f"{parameter_name} must be a valid MM-DD date fragment; got {month_day!r}."
        ) from exc


__all__ = [
    "WalkForwardFold",
    "assign_split_labels",
    "make_annual_walk_forward_folds",
]
