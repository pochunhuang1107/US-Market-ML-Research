"""Sliding-window split generation for local stability studies."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from us_market_ml.splits.walk_forward import WalkForwardFold


def make_sliding_window_folds(
    *,
    data_start: str | pd.Timestamp,
    data_end: str | pd.Timestamp,
    train_window_years: int,
    validation_window_months: int,
    test_window_months: int,
    step_months: int = 3,
    include_partial_final_test: bool = True,
    min_partial_test_days: int = 30,
) -> list[WalkForwardFold]:
    """Create deterministic fixed-length rolling train/validation/test folds.

    The generated ranges are inclusive and chronological. For each fold,
    training uses the previous ``train_window_years`` ending the day before
    validation, validation then spans ``validation_window_months``, and test
    follows for ``test_window_months``. The fold anchor rolls forward by
    ``step_months``.
    """

    _validate_positive("train_window_years", train_window_years)
    _validate_positive("validation_window_months", validation_window_months)
    _validate_positive("test_window_months", test_window_months)
    _validate_positive("step_months", step_months)
    _validate_positive("min_partial_test_days", min_partial_test_days)

    start = pd.Timestamp(data_start).normalize()
    end = pd.Timestamp(data_end).normalize()
    if start > end:
        raise ValueError("data_start must be earlier than or equal to data_end.")

    validation_start = start + pd.DateOffset(years=train_window_years)
    folds: list[WalkForwardFold] = []
    fold_number = 1
    while True:
        train_start = validation_start - pd.DateOffset(years=train_window_years)
        train_end = validation_start - pd.Timedelta(days=1)
        validation_end = (
            validation_start
            + pd.DateOffset(months=validation_window_months)
            - pd.Timedelta(days=1)
        )
        test_start = validation_end + pd.Timedelta(days=1)
        full_test_end = (
            test_start + pd.DateOffset(months=test_window_months) - pd.Timedelta(days=1)
        )

        if test_start > end:
            break
        if full_test_end > end and not include_partial_final_test:
            break

        test_end = min(full_test_end, end)
        test_days = int((test_end - test_start).days) + 1
        if test_days < min_partial_test_days:
            break

        folds.append(
            WalkForwardFold(
                name=_fold_name(
                    train_window_years=train_window_years,
                    validation_window_months=validation_window_months,
                    test_window_months=test_window_months,
                    fold_number=fold_number,
                    test_start=test_start,
                    test_end=test_end,
                ),
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_number += 1
        validation_start = validation_start + pd.DateOffset(months=step_months)

    return folds


def make_sliding_window_folds_for_train_windows(
    *,
    data_start: str | pd.Timestamp,
    data_end: str | pd.Timestamp,
    train_window_years_values: Iterable[int],
    validation_window_months: int,
    test_window_months: int,
    step_months: int = 3,
    include_partial_final_test: bool = True,
    min_partial_test_days: int = 30,
) -> list[WalkForwardFold]:
    """Create sliding-window folds for multiple train-window lengths."""

    folds: list[WalkForwardFold] = []
    for train_window_years in train_window_years_values:
        folds.extend(
            make_sliding_window_folds(
                data_start=data_start,
                data_end=data_end,
                train_window_years=train_window_years,
                validation_window_months=validation_window_months,
                test_window_months=test_window_months,
                step_months=step_months,
                include_partial_final_test=include_partial_final_test,
                min_partial_test_days=min_partial_test_days,
            )
        )
    return folds


def _fold_name(
    *,
    train_window_years: int,
    validation_window_months: int,
    test_window_months: int,
    fold_number: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> str:
    return (
        f"tw{train_window_years}y"
        f"_v{validation_window_months}m"
        f"_t{test_window_months}m"
        f"_q{fold_number:02d}"
        f"_{test_start.strftime('%Y%m%d')}"
        f"_{test_end.strftime('%Y%m%d')}"
    )


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive.")


__all__ = [
    "make_sliding_window_folds",
    "make_sliding_window_folds_for_train_windows",
]
