from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_model_card import parse_config_json, run_model_card_from_args


def test_parse_config_json() -> None:
    assert parse_config_json('{"top_k": 1, "switch_margin": 0.002}') == {
        "top_k": 1,
        "switch_margin": 0.002,
    }


def test_parse_config_json_rejects_non_object() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_config_json("[1, 2, 3]")


def test_run_model_card_from_args_writes_markdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_name = "linear_smooth10_margin002"
    candidate_dir = tmp_path / candidate_name
    output_path = tmp_path / "cards" / f"{candidate_name}_model_card.md"
    cost_sensitivity_path = tmp_path / "cost_sensitivity.csv"
    candidate_dir.mkdir()

    pd.DataFrame(
        [
            {
                "series_name": candidate_name,
                "total_return": 0.24,
                "sharpe_like": 0.08,
                "max_drawdown": -0.18,
                "final_equity": 1.24,
                "total_transaction_cost": 0.015,
                "mean_daily_turnover": 0.10,
            },
            {
                "series_name": "SPY",
                "total_return": 0.12,
                "sharpe_like": 0.04,
                "max_drawdown": -0.20,
                "final_equity": 1.12,
            },
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_name": "fold_1",
                "test_start": "2022-01-01",
                "test_end": "2022-12-31",
                "strategy_total_return": 0.10,
                "strategy_max_drawdown": -0.08,
                "strategy_final_equity": 1.10,
            }
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_fold_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "overall_concentration_risk": "low",
                "risk_score": 1,
                "top_symbol_by_selection": "SPY",
                "top_symbol_selected_share": 0.22,
                "top_symbol_by_contribution": "QQQ",
                "top_symbol_positive_contribution_share": 0.25,
            }
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_concentration_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "avg_holding_days": 9.5,
                "median_holding_days": 7.0,
                "single_day_holding_share": 0.12,
            }
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_holding_period_summary.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "SPY", "selected_days": 40, "avg_target_weight": 1.0},
            {"symbol": "QQQ", "selected_days": 35, "avg_target_weight": 1.0},
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_holdings_summary.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "QQQ", "total_position_return": 0.16},
            {"symbol": "SPY", "total_position_return": 0.07},
        ]
    ).to_csv(candidate_dir / f"{candidate_name}_contribution_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_name": candidate_name,
                "cost_bps": 0.0,
                "total_return": 0.26,
                "sharpe_like": 0.09,
                "max_drawdown": -0.17,
                "final_equity": 1.26,
                "total_transaction_cost": 0.0,
                "mean_daily_turnover": 0.10,
            }
        ]
    ).to_csv(cost_sensitivity_path, index=False)

    return_code = run_model_card_from_args(
        [
            "--candidate-dir",
            str(candidate_dir),
            "--candidate-name",
            candidate_name,
            "--candidate-family",
            "baseline",
            "--model-type",
            "linear_regression",
            "--output-path",
            str(output_path),
            "--config-json",
            '{"top_k": 1, "min_holding_days": 10, "switch_margin": 0.002}',
            "--cost-sensitivity-path",
            str(cost_sensitivity_path),
        ]
    )

    captured = capsys.readouterr()
    markdown = output_path.read_text(encoding="utf-8")

    assert return_code == 0
    assert "output model card Markdown" in captured.out
    assert output_path.exists()
    assert "# Candidate Model Card: linear_smooth10_margin002" in markdown
    assert "promising_for_further_research" in markdown
    assert "Disclaimer: research only, not financial advice" in markdown
