from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.backtest import (
    CONCENTRATION_REPORT_COLUMNS,
    build_concentration_report,
    calculate_hhi,
    effective_number_from_hhi,
    score_concentration_risk,
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


def test_score_concentration_risk_returns_low_for_diversified_metrics() -> None:
    report = _risk_input()

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "low"
    assert scored.loc[0, "contribution_concentration_risk"] == "low"
    assert scored.loc[0, "overall_concentration_risk"] == "low"
    assert scored.loc[0, "risk_score"] == 1


def test_score_concentration_risk_returns_medium_selection_risk() -> None:
    report = _risk_input(top_symbol_selected_share=0.20)

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "medium"
    assert scored.loc[0, "contribution_concentration_risk"] == "low"
    assert scored.loc[0, "overall_concentration_risk"] == "medium"
    assert scored.loc[0, "risk_score"] == 2


def test_score_concentration_risk_returns_high_selection_risk() -> None:
    report = _risk_input(
        top_symbol_selected_share=0.25,
        max_avg_target_weight=0.95,
    )

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "high"
    assert scored.loc[0, "contribution_concentration_risk"] == "low"
    assert scored.loc[0, "overall_concentration_risk"] == "high"
    assert scored.loc[0, "risk_score"] == 3


def test_score_concentration_risk_returns_medium_contribution_risk() -> None:
    report = _risk_input(top_symbol_positive_contribution_share=0.25)

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "low"
    assert scored.loc[0, "contribution_concentration_risk"] == "medium"
    assert scored.loc[0, "overall_concentration_risk"] == "medium"
    assert scored.loc[0, "risk_score"] == 2


def test_score_concentration_risk_returns_high_contribution_risk() -> None:
    report = _risk_input(top_3_positive_contribution_share=0.80)

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "low"
    assert scored.loc[0, "contribution_concentration_risk"] == "high"
    assert scored.loc[0, "overall_concentration_risk"] == "high"
    assert scored.loc[0, "risk_score"] == 3


def test_score_concentration_risk_uses_max_risk_for_overall_score() -> None:
    report = _risk_input(
        top_symbol_selected_share=0.20,
        effective_contributing_symbols=3.4,
    )

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "medium"
    assert scored.loc[0, "contribution_concentration_risk"] == "high"
    assert scored.loc[0, "overall_concentration_risk"] == "high"
    assert scored.loc[0, "risk_score"] == 3


def test_score_concentration_risk_does_not_treat_nan_as_high_risk() -> None:
    report = _risk_input(
        top_symbol_selected_share=float("nan"),
        top_3_selected_share=float("nan"),
        effective_selected_symbols=float("nan"),
        max_avg_target_weight=float("nan"),
        top_symbol_positive_contribution_share=float("nan"),
        top_3_positive_contribution_share=float("nan"),
        effective_contributing_symbols=float("nan"),
    )

    scored = score_concentration_risk(report)

    assert scored.loc[0, "selection_concentration_risk"] == "low"
    assert scored.loc[0, "contribution_concentration_risk"] == "low"
    assert scored.loc[0, "overall_concentration_risk"] == "low"
    assert scored.loc[0, "risk_score"] == 1


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
    assert report.loc[0, "selection_concentration_risk"] == "high"
    assert report.loc[0, "contribution_concentration_risk"] == "high"
    assert report.loc[0, "overall_concentration_risk"] == "high"
    assert report.loc[0, "risk_score"] == 3
    assert {
        "selection_concentration_risk",
        "contribution_concentration_risk",
        "overall_concentration_risk",
        "risk_score",
    }.issubset(report.columns)
    assert list(report.columns) == CONCENTRATION_REPORT_COLUMNS


def _risk_input(**overrides: float) -> pd.DataFrame:
    row = {
        "top_symbol_selected_share": 0.15,
        "top_3_selected_share": 0.45,
        "effective_selected_symbols": 8.0,
        "max_avg_target_weight": 0.50,
        "top_symbol_positive_contribution_share": 0.20,
        "top_3_positive_contribution_share": 0.50,
        "effective_contributing_symbols": 7.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])
