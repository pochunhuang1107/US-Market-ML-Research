from __future__ import annotations

import pandas as pd

from us_market_ml.reports.model_card import build_candidate_model_card


def _metrics(
    *,
    candidate_return: float = 0.24,
    candidate_sharpe: float = 0.08,
    candidate_drawdown: float = -0.18,
    spy_return: float = 0.12,
    spy_sharpe: float = 0.04,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_name": "linear_smooth10_margin002",
                "total_return": candidate_return,
                "mean_daily_return": 0.001,
                "daily_volatility": 0.0125,
                "sharpe_like": candidate_sharpe,
                "max_drawdown": candidate_drawdown,
                "final_equity": 1.24,
                "total_transaction_cost": 0.015,
                "mean_daily_turnover": 0.10,
                "max_daily_turnover": 1.0,
            },
            {
                "series_name": "SPY",
                "total_return": spy_return,
                "mean_daily_return": 0.0005,
                "daily_volatility": 0.0125,
                "sharpe_like": spy_sharpe,
                "max_drawdown": -0.20,
                "final_equity": 1.12,
            },
        ]
    )


def _fold_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_name": "fold_1",
                "test_start": "2022-01-01",
                "test_end": "2022-12-31",
                "strategy_total_return": 0.10,
                "strategy_max_drawdown": -0.08,
                "strategy_final_equity": 1.10,
                "strategy_total_transaction_cost": 0.004,
                "strategy_mean_daily_turnover": 0.08,
            },
            {
                "fold_name": "fold_2",
                "test_start": "2023-01-01",
                "test_end": "2023-12-31",
                "strategy_total_return": 0.12,
                "strategy_max_drawdown": -0.10,
                "strategy_final_equity": 1.12,
                "strategy_total_transaction_cost": 0.005,
                "strategy_mean_daily_turnover": 0.11,
            },
        ]
    )


def _concentration(risk: str = "low") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "top_symbol_by_selection": "SPY",
                "top_symbol_selected_share": 0.22,
                "top_3_selected_share": 0.55,
                "top_symbol_by_contribution": "QQQ",
                "top_symbol_positive_contribution_share": 0.25,
                "effective_selected_symbols": 8.0,
                "effective_contributing_symbols": 7.5,
                "overall_concentration_risk": risk,
                "risk_score": {"low": 1, "medium": 2, "high": 3}[risk],
            }
        ]
    )


def _holding_period_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "avg_holding_days": 9.5,
                "median_holding_days": 7.0,
                "single_day_holding_share": 0.12,
                "mean_turnover": 0.10,
                "max_turnover": 1.0,
            }
        ]
    )


def _holdings_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "SPY", "selected_days": 40, "avg_target_weight": 1.0},
            {"symbol": "QQQ", "selected_days": 35, "avg_target_weight": 1.0},
            {"symbol": "TLT", "selected_days": 20, "avg_target_weight": 1.0},
        ]
    )


def _contribution_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "QQQ", "total_position_return": 0.16},
            {"symbol": "SPY", "total_position_return": 0.07},
            {"symbol": "TLT", "total_position_return": -0.02},
        ]
    )


def _cost_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": "linear_smooth10_margin002",
                "cost_bps": 0.0,
                "total_return": 0.26,
                "sharpe_like": 0.09,
                "max_drawdown": -0.17,
                "final_equity": 1.26,
                "total_transaction_cost": 0.0,
                "mean_daily_turnover": 0.10,
                "return_decay_vs_0bps": 0.0,
            },
            {
                "candidate_name": "linear_smooth10_margin002",
                "cost_bps": 10.0,
                "total_return": 0.20,
                "sharpe_like": 0.07,
                "max_drawdown": -0.19,
                "final_equity": 1.20,
                "total_transaction_cost": 0.03,
                "mean_daily_turnover": 0.10,
                "return_decay_vs_0bps": -0.06,
            },
            {
                "candidate_name": "linear_smooth10_margin002",
                "cost_bps": 20.0,
                "total_return": 0.12,
                "sharpe_like": 0.05,
                "max_drawdown": -0.22,
                "final_equity": 1.12,
                "total_transaction_cost": 0.06,
                "mean_daily_turnover": 0.10,
                "return_decay_vs_0bps": -0.14,
            },
        ]
    )


def _card(
    *,
    metrics: pd.DataFrame | None = None,
    concentration: pd.DataFrame | None = None,
    cost_sensitivity: pd.DataFrame | None = None,
) -> str:
    return build_candidate_model_card(
        candidate_name="linear_smooth10_margin002",
        candidate_family="baseline",
        model_type="linear_regression",
        config={"top_k": 1, "min_holding_days": 10, "switch_margin": 0.002},
        metrics=_metrics() if metrics is None else metrics,
        fold_summary=_fold_summary(),
        concentration_report=_concentration() if concentration is None else concentration,
        holding_period_summary=_holding_period_summary(),
        holdings_summary=_holdings_summary(),
        contribution_summary=_contribution_summary(),
        cost_sensitivity=_cost_sensitivity() if cost_sensitivity is None else cost_sensitivity,
    )


def test_build_candidate_model_card_includes_expected_sections() -> None:
    card = _card()

    expected_sections = [
        "# Candidate Model Card: linear_smooth10_margin002",
        "## Candidate Overview",
        "## Strategy Configuration",
        "## Out-of-Sample Performance",
        "## Benchmark Comparison",
        "## Fold-by-Fold Performance",
        "## Cost Sensitivity",
        "## Holdings and Contributors",
        "## Concentration Risk",
        "## Holding Period / Turnover",
        "## Strengths",
        "## Weaknesses / Risks",
        "## Research Verdict",
        "## Disclaimer: research only, not financial advice",
    ]
    for section in expected_sections:
        assert section in card
    assert "promising_for_further_research" in card
    assert "SPY: selected days 40" in card
    assert "QQQ: total position return 16.00%" in card


def test_build_candidate_model_card_verdict_promising() -> None:
    assert "promising_for_further_research" in _card()


def test_build_candidate_model_card_verdict_needs_more_validation_for_medium_risk() -> None:
    card = _card(concentration=_concentration("medium"))

    assert "needs_more_validation" in card


def test_build_candidate_model_card_verdict_not_currently_promising() -> None:
    card = _card(
        metrics=_metrics(
            candidate_return=0.08,
            candidate_sharpe=0.03,
            spy_return=0.12,
            spy_sharpe=0.04,
        )
    )

    assert "not_currently_promising" in card


def test_build_candidate_model_card_renders_cost_sensitivity_table() -> None:
    card = _card()
    expected_header = (
        "| cost_bps | total_return | sharpe_like | max_drawdown | final_equity | "
        "total_transaction_cost | mean_daily_turnover |"
    )

    assert expected_header in card
    assert "| 0 | 26.00% | 0.0900 | -17.00% | 1.2600 | 0.0000 | 10.00% |" in card
    assert "| 10 | 20.00% | 0.0700 | -19.00% | 1.2000 | 0.0300 | 10.00% |" in card
    assert "| 20 | 12.00% | 0.0500 | -22.00% | 1.1200 | 0.0600 | 10.00% |" in card
    assert "| highest_positive_cost_bps | 20 |" in card
    assert "| first_negative_cost_bps | not observed |" in card
    assert "| return_at_10bps | 20.00% |" in card
    assert "| return_at_20bps | 12.00% |" in card
    assert "Cost robustness label: **high**" in card


def test_build_candidate_model_card_filters_cost_sensitivity_by_candidate_name() -> None:
    cost_sensitivity = pd.concat(
        [
            _cost_sensitivity(),
            pd.DataFrame(
                [
                    {
                        "candidate_name": "other_candidate",
                        "cost_bps": 5.0,
                        "total_return": -0.50,
                        "sharpe_like": -0.20,
                        "max_drawdown": -0.70,
                        "final_equity": 0.50,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    card = _card(cost_sensitivity=cost_sensitivity)

    assert "other_candidate" not in card
    assert "-50.00%" not in card
    assert (
        "| 20 | 12.00% | 0.0500 | -22.00% | 1.1200 | 0.0600 | 10.00% |" in card
    )


def test_build_candidate_model_card_missing_cost_sensitivity_keeps_fallback_text() -> None:
    card = _card(cost_sensitivity=pd.DataFrame())

    assert "No separate transaction-cost sensitivity table was provided." in card
    assert "Cost robustness label" not in card


def test_build_candidate_model_card_cost_robustness_labels() -> None:
    base_rows = [
        {"cost_bps": 5.0, "sharpe_like": 0.01, "max_drawdown": -0.10, "final_equity": 1.01},
        {"cost_bps": 10.0, "sharpe_like": 0.01, "max_drawdown": -0.10, "final_equity": 1.01},
        {"cost_bps": 20.0, "sharpe_like": 0.01, "max_drawdown": -0.10, "final_equity": 1.01},
    ]

    cases = [
        ([0.03, 0.02, 0.01], "high"),
        ([0.03, 0.02, -0.01], "medium"),
        ([0.03, -0.01, -0.02], "low"),
        ([-0.01, -0.02, -0.03], "fragile"),
    ]
    for returns, expected_label in cases:
        sensitivity = pd.DataFrame(
            [
                {
                    **row,
                    "candidate_name": "linear_smooth10_margin002",
                    "total_return": total_return,
                }
                for row, total_return in zip(base_rows, returns, strict=True)
            ]
        )

        card = _card(cost_sensitivity=sensitivity)

        assert f"Cost robustness label: **{expected_label}**" in card
