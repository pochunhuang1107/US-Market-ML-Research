from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_tiingo_qqq_timing_baseline_report import (
    build_tiingo_qqq_timing_baseline_decision_matrix,
    build_tiingo_qqq_timing_baseline_report,
    summarize_tiingo_qqq_timing_baseline_results,
    write_tiingo_qqq_timing_baseline_report,
)
from us_market_ml.qqq_timing import RESULT_COLUMNS


def test_qqq_timing_report_identifies_best_candidate(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report_data = build_tiingo_qqq_timing_baseline_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    assert report_data.best_candidate == "qqq_sma200_cash"
    best = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_sma200_cash")
    ].iloc[0]
    defensive = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_return126_positive_cash")
    ].iloc[0]
    rejected = report_data.decision_matrix.loc[
        report_data.decision_matrix["strategy_name"].eq("qqq_return252_positive_cash")
    ].iloc[0]

    assert best["decision"] == "best_qqq_timing_candidate_for_deeper_sweep"
    assert defensive["decision"] == "defensive_candidate"
    assert rejected["decision"] == "reject_no_qqq_return_or_drawdown_adjusted_edge"


def test_qqq_timing_report_writes_required_files_and_sections(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)
    report_data = build_tiingo_qqq_timing_baseline_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    outputs = write_tiingo_qqq_timing_baseline_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "tiingo_qqq_timing_baseline_summary.csv",
        "tiingo_qqq_timing_baseline_decision_matrix.csv",
        "tiingo_qqq_timing_baseline_report.md",
    }
    summary = pd.read_csv(output_dir / "tiingo_qqq_timing_baseline_summary.csv")
    decision = pd.read_csv(output_dir / "tiingo_qqq_timing_baseline_decision_matrix.csv")
    markdown = (output_dir / "tiingo_qqq_timing_baseline_report.md").read_text(
        encoding="utf-8"
    )

    assert not summary.empty
    assert not decision.empty
    for section in [
        "Executive summary",
        "Why QQQ-only timing path",
        "Benchmarks: QQQ buy-and-hold, SPY buy-and-hold, cash",
        "QQQ timing strategies tested",
        "Risk-off asset comparison",
        "QQQ underperformance / outperformance",
        "Drawdown comparison",
        "Trade frequency and transaction costs",
        "Best candidates",
        "Rejected/deprioritized candidates",
        "Recommendation",
        "Caveats",
    ]:
        assert section in markdown
    assert "qqq_sma200_cash" in markdown
    assert "Feature v4: defer" in markdown
    assert "tree_screen: do not run" in markdown
    assert "MLP: do not run" in markdown
    assert "Production promotion: do not promote" in markdown
    assert "not production-ready" in markdown
    assert "does not use S3" in markdown


def test_qqq_timing_summary_and_decision_handle_empty_results() -> None:
    empty = pd.DataFrame(columns=RESULT_COLUMNS)

    summary = summarize_tiingo_qqq_timing_baseline_results(empty)
    decision = build_tiingo_qqq_timing_baseline_decision_matrix(summary)

    assert summary.empty
    assert decision.empty


def _synthetic_results() -> pd.DataFrame:
    rows = [
        _row("qqq_buy_hold", "QQQ", "", "buy_hold", 0.10, -0.20, 0, 0.0),
        _row("spy_buy_hold", "SPY", "", "buy_hold", 0.06, -0.15, 0, 0.0),
        _row("cash", "cash", "cash", "cash", 0.0, 0.0, 0, 0.0),
        _row("qqq_sma200_cash", "QQQ", "cash", "qqq_sma200", 0.13, -0.18, 4, 0.01),
        _row(
            "qqq_return126_positive_cash",
            "QQQ",
            "cash",
            "qqq_return126_positive",
            0.07,
            -0.08,
            3,
            0.01,
        ),
        _row(
            "qqq_return252_positive_cash",
            "QQQ",
            "cash",
            "qqq_return252_positive",
            0.05,
            -0.22,
            2,
            0.01,
        ),
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _row(
    strategy_name: str,
    risk_on_asset: str,
    risk_off_asset: str,
    signal_name: str,
    total_return: float,
    max_drawdown: float,
    transaction_count: int,
    average_turnover: float,
) -> dict[str, object]:
    qqq_return = 0.10
    spy_return = 0.06
    qqq_drawdown = -0.20
    spy_drawdown = -0.15
    return {
        "strategy_name": strategy_name,
        "risk_on_asset": risk_on_asset,
        "risk_off_asset": risk_off_asset,
        "signal_name": signal_name,
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
        "return_to_drawdown": (
            total_return / abs(max_drawdown) if max_drawdown != 0.0 else float("nan")
        ),
        "qqq_buy_hold_return_to_drawdown": qqq_return / abs(qqq_drawdown),
    }
