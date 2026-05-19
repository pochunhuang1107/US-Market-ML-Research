from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.create_tiingo_qqq_defensive_timing_sweep_report import (
    build_tiingo_qqq_defensive_timing_sweep_decision_matrix,
    build_tiingo_qqq_defensive_timing_sweep_report,
    summarize_tiingo_qqq_defensive_timing_sweep_results,
    write_tiingo_qqq_defensive_timing_sweep_report,
)
from us_market_ml.qqq_timing import DEFENSIVE_RESULT_COLUMNS


def test_defensive_timing_sweep_report_identifies_defensive_candidates(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report_data = build_tiingo_qqq_defensive_timing_sweep_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    assert report_data.best_candidate == "qqq_sma200_cash"
    best = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_sma200_cash")
    ].iloc[0]
    defensive = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_sma180_cash")
    ].iloc[0]
    too_defensive = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_low_return_cash")
    ].iloc[0]
    rejected = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_reject_cash")
    ].iloc[0]

    assert best["decision"] == "best_candidate"
    assert defensive["decision"] == "defensive_candidate"
    assert too_defensive["decision"] == "too_defensive"
    assert rejected["decision"] == "reject"


def test_defensive_timing_sweep_report_writes_required_files_and_sections(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)
    report_data = build_tiingo_qqq_defensive_timing_sweep_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    outputs = write_tiingo_qqq_defensive_timing_sweep_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "tiingo_qqq_defensive_timing_sweep_summary.csv",
        "tiingo_qqq_defensive_timing_sweep_decision_matrix.csv",
        "tiingo_qqq_defensive_timing_sweep_report.md",
    }
    summary = pd.read_csv(output_dir / "tiingo_qqq_defensive_timing_sweep_summary.csv")
    decision = pd.read_csv(
        output_dir / "tiingo_qqq_defensive_timing_sweep_decision_matrix.csv"
    )
    markdown = (output_dir / "tiingo_qqq_defensive_timing_sweep_report.md").read_text(
        encoding="utf-8"
    )

    assert not summary.empty
    assert not decision.empty
    for section in [
        "Executive summary",
        "Baseline QQQ timing result recap",
        "Sweep families and parameter ranges",
        "Benchmark comparison vs QQQ/SPY/cash",
        "Best total-return candidates",
        "Best drawdown candidates",
        "Best return-to-drawdown candidates",
        "Trade frequency / cost analysis",
        "Rejected/deprioritized candidates",
        "Recommendation",
        "Caveats",
    ]:
        assert section in markdown
    assert "qqq_sma200_cash" in markdown
    assert "QQQ buy-and-hold total return was approximately `3.7950`" in markdown
    assert "Rich replay best QQQ timing candidate" in markdown
    assert "Deeper QQQ timing sweep" in markdown
    assert "QQQ + defensive rotation" in markdown
    assert "Feature v4: do not add benchmark_gap_features_v4" in markdown
    assert "tree_screen: do not run" in markdown
    assert "MLP: do not run" in markdown
    assert "Production promotion: do not promote" in markdown
    assert "uses no S3" in markdown
    assert "does not export production signals" in markdown


def test_defensive_timing_sweep_summary_and_decision_handle_empty_results() -> None:
    empty = pd.DataFrame(columns=DEFENSIVE_RESULT_COLUMNS)

    summary = summarize_tiingo_qqq_defensive_timing_sweep_results(empty)
    decision = build_tiingo_qqq_defensive_timing_sweep_decision_matrix(summary)

    assert summary.empty
    assert decision.empty


def _synthetic_results() -> pd.DataFrame:
    rows = [
        _row("qqq_buy_hold", "benchmark", "QQQ", "", 0.10, -0.20, 0, 0.0),
        _row("spy_buy_hold", "benchmark", "SPY", "", 0.06, -0.15, 0, 0.0),
        _row("cash", "benchmark", "cash", "cash", 0.0, 0.0, 0, 0.0),
        _row("qqq_sma200_cash", "sma_price_filter_to_cash", "QQQ", "cash", 0.13, -0.18, 4, 0.01),
        _row("qqq_sma180_cash", "sma_price_filter_to_cash", "QQQ", "cash", 0.08, -0.08, 3, 0.01),
        _row(
            "qqq_low_return_cash",
            "volatility_guard_to_cash",
            "QQQ",
            "cash",
            0.02,
            -0.02,
            2,
            0.01,
        ),
        _row("qqq_reject_cash", "sma_crossover_to_cash", "QQQ", "cash", 0.05, -0.22, 2, 0.01),
    ]
    return pd.DataFrame(rows, columns=DEFENSIVE_RESULT_COLUMNS)


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
        "end_date": "2020-01-01",
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
