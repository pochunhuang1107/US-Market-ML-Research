from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.backtest import (
    build_concentration_report,
    calculate_hhi,
    effective_number_from_hhi,
    summarize_contribution_concentration,
    summarize_selection_concentration,
)


def test_calculate_hhi_normalizes_absolute_non_null_weights() -> None:
    hhi = calculate_hhi(pd.Series([0.50, 0.25, -0.25, None]))

    assert hhi == pytest.approx(0.375)
    assert pd.isna(calculate_hhi(pd.Series([0.0, None])))
    assert pd.isna(calculate_hhi(pd.Series(dtype="float64")))


def test_effective_number_from_hhi_returns_inverse_for_positive_hhi() -> None:
    assert effective_number_from_hhi(0.25) == pytest.approx(4.0)
    assert pd.isna(effective_number_from_hhi(0.0))
    assert pd.isna(effective_number_from_hhi(-0.10))
    assert pd.isna(effective_number_from_hhi(float("nan")))


def test_summarize_selection_concentration_returns_symbol_level_diagnostics() -> None:
    holdings_summary = pd.DataFrame(
        [
            {"symbol": "AAA", "selected_days": 6, "avg_target_weight": 0.50},
            {"symbol": "BBB", "selected_days": 3, "avg_target_weight": 0.30},
            {"symbol": "CCC", "selected_days": 1, "avg_target_weight": 0.20},
        ]
    )

    summary = summarize_selection_concentration(holdings_summary)

    assert summary["total_selected_days"] == pytest.approx(10.0)
    assert summary["unique_selected_symbols"] == pytest.approx(3.0)
    assert summary["top_symbol_by_selection"] == "AAA"
    assert summary["top_symbol_selected_days"] == pytest.approx(6.0)
    assert summary["top_symbol_selected_share"] == pytest.approx(0.60)
    assert summary["top_3_selected_share"] == pytest.approx(1.0)
    assert summary["selection_hhi"] == pytest.approx(0.46)
    assert summary["effective_selected_symbols"] == pytest.approx(1.0 / 0.46)
    assert summary["max_avg_target_weight"] == pytest.approx(0.50)


def test_summarize_contribution_concentration_returns_positive_and_absolute_diagnostics() -> None:
    contribution_summary = pd.DataFrame(
        [
            {"symbol": "AAA", "total_position_return": 0.12},
            {"symbol": "BBB", "total_position_return": 0.06},
            {"symbol": "CCC", "total_position_return": -0.04},
            {"symbol": "DDD", "total_position_return": 0.02},
        ]
    )

    summary = summarize_contribution_concentration(contribution_summary)

    assert summary["total_positive_contribution"] == pytest.approx(0.20)
    assert summary["total_absolute_contribution"] == pytest.approx(0.24)
    assert summary["top_symbol_by_contribution"] == "AAA"
    assert summary["top_symbol_contribution"] == pytest.approx(0.12)
    assert summary["top_symbol_positive_contribution_share"] == pytest.approx(0.60)
    assert summary["top_3_positive_contribution_share"] == pytest.approx(1.0)
    assert summary["contribution_hhi"] == pytest.approx(0.3472222222)
    assert summary["effective_contributing_symbols"] == pytest.approx(1.0 / 0.3472222222)


def test_build_concentration_report_returns_one_row_with_warnings() -> None:
    holdings_summary = pd.DataFrame(
        [
            {"symbol": "AAA", "selected_days": 6, "avg_target_weight": 0.50},
            {"symbol": "BBB", "selected_days": 3, "avg_target_weight": 0.30},
            {"symbol": "CCC", "selected_days": 1, "avg_target_weight": 0.20},
        ]
    )
    contribution_summary = pd.DataFrame(
        [
            {"symbol": "AAA", "total_position_return": 0.12},
            {"symbol": "BBB", "total_position_return": 0.06},
            {"symbol": "CCC", "total_position_return": -0.04},
            {"symbol": "DDD", "total_position_return": 0.02},
        ]
    )

    report = build_concentration_report(holdings_summary, contribution_summary)

    assert len(report) == 1
    assert report.loc[0, "top_symbol_by_selection"] == "AAA"
    assert report.loc[0, "top_symbol_by_contribution"] == "AAA"
    assert report.loc[0, "selection_hhi"] == pytest.approx(0.46)
    assert bool(report.loc[0, "concentration_warning"]) is True
    assert bool(report.loc[0, "contribution_warning"]) is True
