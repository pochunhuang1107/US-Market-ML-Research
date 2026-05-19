from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_qqq_defensive_timing_model_card import (
    MODEL_CARD_NAME,
    RESEARCH_DECISION_NAME,
    run_create_qqq_defensive_timing_model_card_from_args,
)
from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    DECISION_MATRIX_NAME,
    SUBPERIOD_SUMMARY_NAME,
    SUMMARY_NAME,
)


def test_qqq_defensive_timing_model_card_cli_writes_required_docs(
    tmp_path: Path,
    capsys,
) -> None:
    replay_dir = tmp_path / "replay"
    output_dir = tmp_path / "docs" / "research"
    replay_dir.mkdir()
    _summary().to_csv(replay_dir / SUMMARY_NAME, index=False)
    _subperiod_summary().to_csv(replay_dir / SUBPERIOD_SUMMARY_NAME, index=False)
    _decision_matrix().to_csv(replay_dir / DECISION_MATRIX_NAME, index=False)

    exit_code = run_create_qqq_defensive_timing_model_card_from_args(
        [
            "--replay-dir",
            str(replay_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    model_card_path = output_dir / MODEL_CARD_NAME
    decision_path = output_dir / RESEARCH_DECISION_NAME
    model_card = model_card_path.read_text(encoding="utf-8")
    decision = decision_path.read_text(encoding="utf-8")
    combined = model_card + "\n" + decision

    assert exit_code == 0
    assert "QQQ defensive timing research docs" in captured.out
    assert model_card_path.exists()
    assert decision_path.exists()
    assert "qqq_sma190_buffer_p0p5_hold0_to_spy" in combined
    assert "production_ready = false" in combined
    assert "QQQ buy-and-hold comparison" in model_card
    assert "Next Validation Steps" in model_card
    assert "longer-history" in combined
    assert "walk-forward" in combined
    assert "No tree_screen from this result" in decision
    assert "No MLP from this result" in decision
    assert "No production export" in decision
    assert "s3://" not in combined.lower()
    assert "read_s3" not in combined
    assert "run_mlp" not in combined
    assert "tree_screen(" not in combined
    assert "run_model_signal_export" not in combined
    assert "run_daily_signal_export" not in combined


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_name": "qqq_sma190_buffer_p0p5_hold0_to_spy",
                "family": "total_return_spy_defensive",
                "risk_on_asset": "QQQ",
                "risk_off_asset": "SPY",
                "signal_name": "qqq_sma_price_filter",
                "slow_sma_days": 190,
                "buffer_pct": 0.005,
                "min_hold_days": 0,
                "transaction_cost_bps": 10,
                "start_date": "2019-01-02",
                "end_date": "2026-05-15",
                "trading_days": 1853,
                "exposure_rate_to_qqq": 0.7129,
                "exposure_rate_to_spy": 0.2871,
                "total_return": 3.7883,
                "qqq_buy_hold_total_return": 3.7950,
                "spy_buy_hold_total_return": 2.2962,
                "strategy_vs_qqq": -0.0067,
                "strategy_vs_spy": 1.4921,
                "annualized_return": 0.2374,
                "daily_volatility": 0.0139,
                "sharpe_like": 1.0746,
                "max_drawdown": -0.3324,
                "qqq_buy_hold_max_drawdown": -0.3512,
                "spy_buy_hold_max_drawdown": -0.3370,
                "drawdown_improvement_vs_qqq": 0.0189,
                "return_to_drawdown": 11.3985,
                "qqq_buy_hold_return_to_drawdown": 10.8050,
                "transaction_count": 23,
                "estimated_transaction_cost": 0.046,
                "qqq_return_capture": 0.9982,
            }
        ]
    )


def _subperiod_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _subperiod_row("full", "2019-01-01", "2026-05-15", 3.7883, 3.7950, -0.0067),
            _subperiod_row(
                "covid_crash_recovery",
                "2020-01-01",
                "2020-12-31",
                0.4309,
                0.4618,
                -0.0309,
            ),
            _subperiod_row("bear_2022", "2022-01-01", "2022-12-31", -0.2237, -0.3322, 0.1086),
            _subperiod_row(
                "ai_bull_2023_2024",
                "2023-01-01",
                "2024-12-31",
                0.8203,
                0.9578,
                -0.1375,
            ),
            _subperiod_row(
                "recent_2025_2026",
                "2025-01-01",
                "2026-05-15",
                0.3505,
                0.3981,
                -0.0477,
            ),
        ]
    )


def _subperiod_row(
    period_name: str,
    period_start: str,
    period_end: str,
    strategy_return: float,
    qqq_return: float,
    strategy_vs_qqq: float,
) -> dict[str, object]:
    return {
        "strategy_name": "qqq_sma190_buffer_p0p5_hold0_to_spy",
        "period_name": period_name,
        "period_start": period_start,
        "period_end": period_end,
        "strategy_total_return": strategy_return,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": 0.25,
        "strategy_vs_qqq": strategy_vs_qqq,
        "strategy_vs_spy": strategy_return - 0.25,
        "strategy_max_drawdown": -0.20,
        "qqq_max_drawdown": -0.25,
        "spy_max_drawdown": -0.18,
        "exposure_rate_to_qqq": 0.70,
        "exposure_rate_to_spy": 0.30,
        "transaction_count": 3,
    }


def _decision_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_name": "qqq_sma190_buffer_p0p5_hold0_to_spy",
                "decision": "keep_as_best_qqq_defensive_timing_candidate",
                "research_status": "research_only_defensive_candidate",
                "robustness_decision": "robust_defensive",
                "decision_reason": "Nearly matches QQQ return while improving drawdown.",
                "next_action": "model-card draft; longer-history validation",
                "production_ready": False,
            }
        ]
    )
