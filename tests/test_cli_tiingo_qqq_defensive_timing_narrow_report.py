from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_qqq_defensive_timing_narrow_report import (
    build_tiingo_qqq_defensive_timing_narrow_decision_matrix,
    build_tiingo_qqq_defensive_timing_narrow_report,
    summarize_tiingo_qqq_defensive_timing_narrow_results,
    summarize_tiingo_qqq_defensive_timing_narrow_subperiods,
    write_tiingo_qqq_defensive_timing_narrow_report,
)
from us_market_ml.qqq_timing import DEFENSIVE_RESULT_COLUMNS, DEFENSIVE_SUBPERIOD_RESULT_COLUMNS


def test_defensive_timing_narrow_report_identifies_candidate_decisions(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    subperiod_csv = tmp_path / "subperiod.csv"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_subperiod_results().to_csv(subperiod_csv, index=False)

    report_data = build_tiingo_qqq_defensive_timing_narrow_report(
        results_csv=results_csv,
        subperiod_results_csv=subperiod_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    assert report_data.best_defensive_candidate == "qqq_near_match"
    decisions = report_data.decision_matrix.set_index("strategy_name")["decision"]

    assert decisions["qqq_near_match"] == "best_defensive_candidate"
    assert decisions["qqq_defensive"] == "defensive_candidate"
    assert decisions["qqq_mixed"] == "mixed"
    assert decisions["qqq_too_defensive"] == "too_defensive"
    assert decisions["qqq_reject"] == "reject"


def test_defensive_timing_narrow_report_writes_required_files_and_sections(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    subperiod_csv = tmp_path / "subperiod.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_subperiod_results().to_csv(subperiod_csv, index=False)
    report_data = build_tiingo_qqq_defensive_timing_narrow_report(
        results_csv=results_csv,
        subperiod_results_csv=subperiod_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    outputs = write_tiingo_qqq_defensive_timing_narrow_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "tiingo_qqq_defensive_timing_narrow_summary.csv",
        "tiingo_qqq_defensive_timing_narrow_decision_matrix.csv",
        "tiingo_qqq_defensive_timing_narrow_subperiod_summary.csv",
        "tiingo_qqq_defensive_timing_narrow_report.md",
    }
    summary = pd.read_csv(output_dir / "tiingo_qqq_defensive_timing_narrow_summary.csv")
    decision = pd.read_csv(
        output_dir / "tiingo_qqq_defensive_timing_narrow_decision_matrix.csv"
    )
    subperiod_summary = pd.read_csv(
        output_dir / "tiingo_qqq_defensive_timing_narrow_subperiod_summary.csv"
    )
    markdown = (output_dir / "tiingo_qqq_defensive_timing_narrow_report.md").read_text(
        encoding="utf-8"
    )

    assert not summary.empty
    assert not decision.empty
    assert not subperiod_summary.empty
    for section in [
        "Executive summary",
        "Why this is a narrow sweep",
        "Previous best candidate recap",
        "Parameter ranges",
        "Full-period benchmark comparison",
        "Best raw return candidates",
        "Best drawdown candidates",
        "Best return-to-drawdown candidates",
        "Subperiod robustness",
        "Turnover / transaction cost analysis",
        "Candidate decision matrix",
        "Recommendation",
        "Caveats",
    ]:
        assert section in markdown
    assert "Best low-turnover candidates" in markdown
    assert "qqq_sma180_buffer_p0_hold0_to_spy" in markdown
    assert "qqq_sma200_buffer_p1_hold10_cash" in markdown
    assert "close data available on date T" in markdown
    assert "Rich replay best QQQ timing candidate" in markdown
    assert "Deeper QQQ timing sweep" in markdown
    assert "QQQ + defensive rotation" in markdown
    assert "Feature v4: do not add benchmark_gap_features_v4" in markdown
    assert "tree_screen: do not run" in markdown
    assert "MLP: do not run" in markdown
    assert "Production promotion: do not promote" in markdown
    assert "No S3/cloud paths were used" in markdown
    assert "No production signal export" in markdown
    assert "QQQ buy-and-hold remains the benchmark" in markdown


def test_defensive_timing_narrow_summary_and_decision_handle_empty_results() -> None:
    empty_results = pd.DataFrame(columns=DEFENSIVE_RESULT_COLUMNS)
    empty_subperiods = pd.DataFrame(columns=DEFENSIVE_SUBPERIOD_RESULT_COLUMNS)

    summary = summarize_tiingo_qqq_defensive_timing_narrow_results(empty_results)
    subperiod_summary = summarize_tiingo_qqq_defensive_timing_narrow_subperiods(
        empty_subperiods,
    )
    decision = build_tiingo_qqq_defensive_timing_narrow_decision_matrix(
        summary,
        subperiod_summary,
    )

    assert summary.empty
    assert subperiod_summary.empty
    assert decision.empty


def _synthetic_results() -> pd.DataFrame:
    rows = [
        _row("qqq_buy_hold", "benchmark", "QQQ", "", 0.10, -0.20, 0, 0.0),
        _row("spy_buy_hold", "benchmark", "SPY", "", 0.06, -0.15, 0, 0.0),
        _row("cash", "benchmark", "cash", "cash", 0.0, 0.0, 0, 0.0),
        _row(
            "qqq_near_match",
            "total_return_spy_defensive",
            "QQQ",
            "SPY",
            0.095,
            -0.15,
            8,
            0.01,
        ),
        _row(
            "qqq_defensive",
            "drawdown_adjusted_cash_defensive",
            "QQQ",
            "cash",
            0.075,
            -0.07,
            6,
            0.01,
        ),
        _row(
            "qqq_mixed",
            "low_turnover_crossover_cash",
            "QQQ",
            "cash",
            0.08,
            -0.18,
            5,
            0.01,
        ),
        _row(
            "qqq_too_defensive",
            "drawdown_adjusted_cash_defensive",
            "QQQ",
            "cash",
            0.04,
            -0.05,
            4,
            0.01,
        ),
        _row("qqq_reject", "total_return_spy_defensive", "QQQ", "SPY", 0.05, -0.22, 12, 0.01),
    ]
    return pd.DataFrame(rows, columns=DEFENSIVE_RESULT_COLUMNS)


def _synthetic_subperiod_results() -> pd.DataFrame:
    period_names = [
        "full",
        "covid_crash_recovery",
        "bear_2022",
        "ai_bull_2023_2024",
        "recent_2025_2026",
    ]
    strategy_patterns = {
        "qqq_buy_hold": ([0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
        "spy_buy_hold": ([-0.01, -0.01, -0.01, -0.01, -0.01], [0.01, 0.01, 0.01, 0.01, 0.01]),
        "cash": ([-0.05, -0.05, -0.05, -0.05, -0.05], [0.10, 0.10, 0.10, 0.10, 0.10]),
        "qqq_near_match": ([-0.01, 0.02, -0.02, 0.01, -0.01], [0.05, 0.03, 0.04, 0.02, 0.03]),
        "qqq_defensive": ([-0.03, -0.02, -0.04, -0.01, -0.02], [0.12, 0.10, 0.09, 0.08, 0.11]),
        "qqq_mixed": ([-0.02, 0.04, -0.20, -0.18, -0.16], [0.01, 0.03, -0.01, 0.00, -0.02]),
        "qqq_too_defensive": ([-0.04, -0.06, -0.07, -0.08, -0.05], [0.15, 0.13, 0.12, 0.14, 0.11]),
        "qqq_reject": ([-0.05, -0.04, -0.06, -0.03, -0.05], [-0.02, -0.01, -0.03, -0.02, -0.01]),
    }
    rows: list[dict[str, object]] = []
    for strategy_name, (return_gaps, drawdown_improvements) in strategy_patterns.items():
        for period_name, return_gap, drawdown_improvement in zip(
            period_names,
            return_gaps,
            drawdown_improvements,
            strict=True,
        ):
            rows.append(
                _subperiod_row(
                    strategy_name=strategy_name,
                    period_name=period_name,
                    strategy_vs_qqq=return_gap,
                    drawdown_improvement=drawdown_improvement,
                )
            )
    return pd.DataFrame(rows, columns=DEFENSIVE_SUBPERIOD_RESULT_COLUMNS)


def _row(
    strategy_name: str,
    family: str,
    risk_on_asset: str,
    risk_off_asset: str,
    total_return: float,
    max_drawdown: float,
    transaction_count: int,
    average_turnover: float,
) -> dict[str, object]:
    qqq_return = 0.10
    spy_return = 0.06
    qqq_drawdown = -0.20
    spy_drawdown = -0.15
    return_to_drawdown = (
        total_return / abs(max_drawdown) if max_drawdown != 0.0 else np.nan
    )
    qqq_return_to_drawdown = qqq_return / abs(qqq_drawdown)
    return {
        "strategy_name": strategy_name,
        "family": family,
        "risk_on_asset": risk_on_asset,
        "risk_off_asset": risk_off_asset,
        "signal_name": "synthetic",
        "slow_sma_days": np.nan,
        "fast_sma_days": np.nan,
        "buffer_pct": np.nan,
        "min_hold_days": 0,
        "return_lookback_days": np.nan,
        "volatility_lookback_days": np.nan,
        "volatility_percentile_window_days": np.nan,
        "volatility_percentile_threshold": np.nan,
        "drawdown_lookback_days": np.nan,
        "drawdown_threshold": np.nan,
        "regime_score_threshold": np.nan,
        "start_date": "2019-01-01",
        "end_date": "2026-05-15",
        "trading_days": 252,
        "exposure_rate_to_qqq": 1.0 if risk_on_asset == "QQQ" else 0.0,
        "exposure_rate_to_risk_off": 0.0,
        "total_return": total_return,
        "annualized_return": total_return,
        "daily_volatility": 0.01,
        "sharpe_like": 0.5,
        "max_drawdown": max_drawdown,
        "worst_1m_return": -0.03,
        "worst_3m_return": -0.06,
        "transaction_count": transaction_count,
        "average_turnover": average_turnover,
        "estimated_transaction_cost": transaction_count * 0.001,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": spy_return,
        "strategy_vs_qqq": total_return - qqq_return,
        "strategy_vs_spy": total_return - spy_return,
        "qqq_underperformance": qqq_return - total_return,
        "spy_underperformance": spy_return - total_return,
        "qqq_buy_hold_max_drawdown": qqq_drawdown,
        "spy_buy_hold_max_drawdown": spy_drawdown,
        "drawdown_improvement_vs_qqq": max_drawdown - qqq_drawdown,
        "return_to_drawdown": return_to_drawdown,
        "qqq_buy_hold_return_to_drawdown": qqq_return_to_drawdown,
        "beats_qqq_return": total_return > qqq_return,
        "beats_qqq_return_to_drawdown": return_to_drawdown > qqq_return_to_drawdown,
        "materially_improves_qqq_drawdown": max_drawdown - qqq_drawdown >= 0.05,
    }


def _subperiod_row(
    *,
    strategy_name: str,
    period_name: str,
    strategy_vs_qqq: float,
    drawdown_improvement: float,
) -> dict[str, object]:
    qqq_return = 0.05
    qqq_drawdown = -0.10
    total_return = qqq_return + strategy_vs_qqq
    max_drawdown = qqq_drawdown + drawdown_improvement
    return {
        "strategy_name": strategy_name,
        "period_name": period_name,
        "period_start": "2019-01-01",
        "period_end": "2026-05-15",
        "total_return": total_return,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": 0.04,
        "strategy_vs_qqq": strategy_vs_qqq,
        "max_drawdown": max_drawdown,
        "qqq_max_drawdown": qqq_drawdown,
        "transaction_count": 2,
        "exposure_rate_to_qqq": 0.8,
    }
