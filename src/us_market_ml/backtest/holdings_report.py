"""Selected holdings and contribution reports for walk-forward fold results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class _SignalBacktestLike(Protocol):
    @property
    def position_returns(self) -> pd.DataFrame: ...


class _E2EFoldResultLike(Protocol):
    @property
    def fold_name(self) -> str: ...

    @property
    def test_signals(self) -> pd.DataFrame: ...

    @property
    def test_backtest(self) -> _SignalBacktestLike: ...


SELECTED_HOLDINGS_COLUMNS = [
    "fold_name",
    "date",
    "symbol",
    "prediction",
    "selected",
    "signal_rank",
    "target_weight",
    "close",
]
OPTIONAL_SELECTED_HOLDINGS_COLUMNS = [
    "symbol_return",
    "lagged_target_weight",
    "position_return",
]
POSITION_CONTRIBUTION_COLUMNS = [
    "fold_name",
    "date",
    "symbol",
    "selected",
    "target_weight",
    "lagged_target_weight",
    "symbol_return",
    "position_return",
    "close",
]
HOLDINGS_SUMMARY_COLUMNS = [
    "symbol",
    "selected_days",
    "avg_target_weight",
    "avg_prediction",
    "avg_signal_rank",
    "first_selected_date",
    "last_selected_date",
]
CONTRIBUTION_SUMMARY_COLUMNS = [
    "symbol",
    "total_position_return",
    "avg_position_return",
    "positive_contribution_days",
    "negative_contribution_days",
    "avg_lagged_target_weight",
    "max_lagged_target_weight",
]
DAILY_SELECTION_SUMMARY_COLUMNS = [
    "date",
    "selected_symbols",
    "selected_count",
    "total_target_weight",
    "avg_prediction",
]


def build_selected_holdings_frame(
    fold_results: Sequence[_E2EFoldResultLike],
) -> pd.DataFrame:
    """Return selected test-signal holdings across walk-forward folds.

    The output includes one row per selected symbol/date/fold from each fold's
    test signals. Train and validation signals are intentionally ignored.
    """

    rows: list[pd.DataFrame] = []
    output_columns = _selected_holdings_output_columns(fold_results)
    required_signal_columns = SELECTED_HOLDINGS_COLUMNS[1:]

    for fold_result in fold_results:
        test_signals = fold_result.test_signals
        _validate_required_columns(test_signals, required_signal_columns, "test_signals")

        selected_mask = test_signals["selected"].fillna(False).astype(bool)
        selected = test_signals.loc[selected_mask, required_signal_columns].copy()
        if selected.empty:
            continue

        for optional_column in output_columns:
            if optional_column not in selected.columns and optional_column in test_signals.columns:
                selected[optional_column] = test_signals.loc[selected_mask, optional_column]
            elif optional_column not in selected.columns and optional_column != "fold_name":
                selected[optional_column] = pd.NA
        selected.insert(0, "fold_name", fold_result.fold_name)
        selected["date"] = pd.to_datetime(selected["date"])
        rows.append(selected)

    if not rows:
        return pd.DataFrame(columns=output_columns)

    selected_holdings = pd.concat(rows, ignore_index=True)
    return (
        selected_holdings.loc[:, output_columns]
        .sort_values(["date", "signal_rank", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_position_contribution_frame(
    fold_results: Sequence[_E2EFoldResultLike],
) -> pd.DataFrame:
    """Return test-period position contribution rows across folds."""

    rows: list[pd.DataFrame] = []

    for fold_result in fold_results:
        position_returns = fold_result.test_backtest.position_returns
        _validate_required_columns(
            position_returns,
            POSITION_CONTRIBUTION_COLUMNS[1:],
            "test_backtest.position_returns",
        )

        contribution = position_returns.loc[:, POSITION_CONTRIBUTION_COLUMNS[1:]].copy()
        if contribution.empty:
            continue

        contribution.insert(0, "fold_name", fold_result.fold_name)
        contribution["date"] = pd.to_datetime(contribution["date"])
        rows.append(contribution)

    if not rows:
        return pd.DataFrame(columns=POSITION_CONTRIBUTION_COLUMNS)

    position_contributions = pd.concat(rows, ignore_index=True)
    return (
        position_contributions.loc[:, POSITION_CONTRIBUTION_COLUMNS]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def summarize_holdings(
    selected_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected holdings by symbol."""

    _validate_required_columns(
        selected_holdings,
        ["date", "symbol", "target_weight", "prediction", "signal_rank"],
        "selected_holdings",
    )
    if selected_holdings.empty:
        return pd.DataFrame(columns=HOLDINGS_SUMMARY_COLUMNS)

    holdings = selected_holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    summary = (
        holdings.groupby("symbol", sort=False, dropna=False)
        .agg(
            selected_days=("symbol", "size"),
            avg_target_weight=("target_weight", "mean"),
            avg_prediction=("prediction", "mean"),
            avg_signal_rank=("signal_rank", "mean"),
            first_selected_date=("date", "min"),
            last_selected_date=("date", "max"),
        )
        .reset_index()
    )

    return (
        summary.loc[:, HOLDINGS_SUMMARY_COLUMNS]
        .sort_values(["selected_days", "symbol"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def summarize_position_contributions(
    position_contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize position returns by symbol."""

    _validate_required_columns(
        position_contributions,
        ["symbol", "lagged_target_weight", "position_return"],
        "position_contributions",
    )
    if position_contributions.empty:
        return pd.DataFrame(columns=CONTRIBUTION_SUMMARY_COLUMNS)

    summary = (
        position_contributions.groupby("symbol", sort=False, dropna=False)
        .agg(
            total_position_return=("position_return", "sum"),
            avg_position_return=("position_return", "mean"),
            positive_contribution_days=("position_return", lambda values: int((values > 0).sum())),
            negative_contribution_days=("position_return", lambda values: int((values < 0).sum())),
            avg_lagged_target_weight=("lagged_target_weight", "mean"),
            max_lagged_target_weight=("lagged_target_weight", "max"),
        )
        .reset_index()
    )

    return (
        summary.loc[:, CONTRIBUTION_SUMMARY_COLUMNS]
        .sort_values(
            ["total_position_return", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_daily_selection_summary(
    selected_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Return one ranked selected-symbol summary row per date."""

    _validate_required_columns(
        selected_holdings,
        ["date", "symbol", "signal_rank", "target_weight", "prediction"],
        "selected_holdings",
    )
    if selected_holdings.empty:
        return pd.DataFrame(columns=DAILY_SELECTION_SUMMARY_COLUMNS)

    holdings = selected_holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    rows: list[dict[str, object]] = []
    for date_value, date_group in holdings.groupby("date", sort=True, dropna=False):
        ranked = date_group.sort_values(["signal_rank", "symbol"], kind="mergesort")
        rows.append(
            {
                "date": date_value,
                "selected_symbols": ",".join(ranked["symbol"].astype(str).tolist()),
                "selected_count": int(len(ranked)),
                "total_target_weight": float(ranked["target_weight"].sum()),
                "avg_prediction": float(ranked["prediction"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=DAILY_SELECTION_SUMMARY_COLUMNS)


def _selected_holdings_output_columns(
    fold_results: Sequence[_E2EFoldResultLike],
) -> list[str]:
    columns = list(SELECTED_HOLDINGS_COLUMNS)
    for optional_column in OPTIONAL_SELECTED_HOLDINGS_COLUMNS:
        if any(optional_column in fold_result.test_signals.columns for fold_result in fold_results):
            columns.append(optional_column)
    return columns


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required holdings report columns: {missing}")


__all__ = [
    "build_daily_selection_summary",
    "build_position_contribution_frame",
    "build_selected_holdings_frame",
    "summarize_holdings",
    "summarize_position_contributions",
]
