from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from us_market_ml.backtest import (
    build_daily_selection_summary,
    build_position_contribution_frame,
    build_selected_holdings_frame,
    summarize_holdings,
    summarize_position_contributions,
)


def _fold_result(
    fold_name: str,
    test_signals: pd.DataFrame,
    position_returns: pd.DataFrame | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        fold_name=fold_name,
        test_signals=test_signals,
        test_backtest=SimpleNamespace(
            position_returns=(
                position_returns if position_returns is not None else test_signals.copy()
            )
        ),
    )


def _test_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "BBB",
                "prediction": 0.40,
                "selected": True,
                "signal_rank": 1,
                "target_weight": 0.5,
                "close": 45.0,
            },
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "AAA",
                "prediction": 0.30,
                "selected": True,
                "signal_rank": 2,
                "target_weight": 0.5,
                "close": 100.0,
                "symbol_return": 0.0,
                "lagged_target_weight": 0.0,
                "position_return": 0.0,
            },
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "BBB",
                "prediction": 0.50,
                "selected": True,
                "signal_rank": 1,
                "target_weight": 0.5,
                "close": 50.0,
                "symbol_return": 0.0,
                "lagged_target_weight": 0.0,
                "position_return": 0.0,
            },
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "AAA",
                "prediction": 0.10,
                "selected": False,
                "signal_rank": pd.NA,
                "target_weight": 0.0,
                "close": 110.0,
            },
        ]
    )


def _position_returns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "BBB",
                "selected": True,
                "target_weight": 1.0,
                "lagged_target_weight": 0.0,
                "symbol_return": -0.10,
                "position_return": 0.0,
                "close": 45.0,
            },
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "AAA",
                "selected": True,
                "target_weight": 1.0,
                "lagged_target_weight": 0.0,
                "symbol_return": None,
                "position_return": 0.0,
                "close": 100.0,
            },
            {
                "date": pd.Timestamp("2021-01-03"),
                "symbol": "AAA",
                "selected": False,
                "target_weight": 0.0,
                "lagged_target_weight": 1.0,
                "symbol_return": 0.10,
                "position_return": 0.10,
                "close": 121.0,
            },
            {
                "date": pd.Timestamp("2021-01-03"),
                "symbol": "BBB",
                "selected": False,
                "target_weight": 0.0,
                "lagged_target_weight": 1.0,
                "symbol_return": -0.05,
                "position_return": -0.05,
                "close": 42.75,
            },
        ]
    )


def test_build_selected_holdings_frame_uses_test_signals_only_and_sorts() -> None:
    fold = _fold_result("fold_1", _test_signals())
    fold.train_signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "symbol": "ZZZ",
                "prediction": 1.0,
                "selected": True,
                "signal_rank": 1,
                "target_weight": 1.0,
                "close": 1.0,
            }
        ]
    )

    selected = build_selected_holdings_frame([fold])

    assert selected["fold_name"].tolist() == ["fold_1", "fold_1", "fold_1"]
    assert selected["date"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-02"),
    ]
    assert selected["symbol"].tolist() == ["BBB", "AAA", "BBB"]
    assert selected["selected"].tolist() == [True, True, True]
    assert "ZZZ" not in set(selected["symbol"])
    assert {"symbol_return", "lagged_target_weight", "position_return"}.issubset(
        selected.columns
    )


def test_build_selected_holdings_frame_returns_empty_expected_columns() -> None:
    signals = _test_signals()
    signals["selected"] = False

    selected = build_selected_holdings_frame([_fold_result("fold_1", signals)])

    assert selected.empty
    assert selected.columns.tolist() == [
        "fold_name",
        "date",
        "symbol",
        "prediction",
        "selected",
        "signal_rank",
        "target_weight",
        "close",
        "symbol_return",
        "lagged_target_weight",
        "position_return",
    ]


def test_build_position_contribution_frame_preserves_all_rows_and_sorts() -> None:
    contributions = build_position_contribution_frame(
        [_fold_result("fold_1", _test_signals(), _position_returns())]
    )

    assert contributions["fold_name"].tolist() == ["fold_1", "fold_1", "fold_1", "fold_1"]
    assert contributions["date"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-03"),
        pd.Timestamp("2021-01-03"),
    ]
    assert contributions["symbol"].tolist() == ["AAA", "BBB", "AAA", "BBB"]
    assert contributions["position_return"].tolist() == [0.0, 0.0, 0.10, -0.05]


def test_build_position_contribution_frame_missing_columns_raise_value_error() -> None:
    missing_close = _position_returns().drop(columns=["close"])

    with pytest.raises(ValueError, match="close"):
        build_position_contribution_frame(
            [_fold_result("fold_1", _test_signals(), missing_close)]
        )


def test_summarize_holdings_returns_symbol_level_stats() -> None:
    selected = build_selected_holdings_frame([_fold_result("fold_1", _test_signals())])

    summary = summarize_holdings(selected)

    assert summary["symbol"].tolist() == ["BBB", "AAA"]
    assert summary["selected_days"].tolist() == [2, 1]
    assert summary.loc[0, "avg_target_weight"] == pytest.approx(0.5)
    assert summary.loc[0, "avg_prediction"] == pytest.approx(0.45)
    assert summary.loc[0, "avg_signal_rank"] == pytest.approx(1.0)
    assert summary.loc[0, "first_selected_date"] == pd.Timestamp("2021-01-01")
    assert summary.loc[0, "last_selected_date"] == pd.Timestamp("2021-01-02")


def test_summarize_position_contributions_returns_symbol_level_stats() -> None:
    contributions = build_position_contribution_frame(
        [_fold_result("fold_1", _test_signals(), _position_returns())]
    )

    summary = summarize_position_contributions(contributions)

    assert summary["symbol"].tolist() == ["AAA", "BBB"]
    assert summary.loc[0, "total_position_return"] == pytest.approx(0.10)
    assert summary.loc[1, "total_position_return"] == pytest.approx(-0.05)
    assert summary.loc[0, "positive_contribution_days"] == 1
    assert summary.loc[1, "negative_contribution_days"] == 1
    assert summary.loc[0, "avg_lagged_target_weight"] == pytest.approx(0.5)
    assert summary.loc[0, "max_lagged_target_weight"] == pytest.approx(1.0)


def test_build_daily_selection_summary_returns_ranked_symbols_by_date() -> None:
    selected = build_selected_holdings_frame([_fold_result("fold_1", _test_signals())])

    summary = build_daily_selection_summary(selected)

    assert summary["date"].tolist() == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    assert summary["selected_symbols"].tolist() == ["BBB,AAA", "BBB"]
    assert summary["selected_count"].tolist() == [2, 1]
    assert summary["total_target_weight"].tolist() == [1.0, 0.5]
    assert summary["avg_prediction"].tolist() == pytest.approx([0.40, 0.40])
