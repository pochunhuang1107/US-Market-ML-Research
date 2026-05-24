from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_core_sleeve_decision_report import (
    run_long_history_qqq_core_sleeve_decision_report_from_args,
)
from us_market_ml.long_history_core_sleeve_decision import (
    DECISION_MATRIX_NAME,
    RECOMMENDATIONS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    DECISION_MATRIX_NAME as REFINEMENT_DECISION_MATRIX_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    RESULTS_NAME as REFINEMENT_RESULTS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    ROLLING_WINDOWS_NAME as REFINEMENT_ROLLING_WINDOWS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    SUBPERIODS_NAME as REFINEMENT_SUBPERIODS_NAME,
)


def test_cli_core_sleeve_decision_report_writes_required_files(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    refinement_report_dir = tmp_path / "refinement_report"
    output_dir = tmp_path / "decision_report"
    sweep_dir.mkdir()
    refinement_report_dir.mkdir()
    results = _results()

    results.to_csv(sweep_dir / REFINEMENT_RESULTS_NAME, index=False)
    _subperiods().to_csv(sweep_dir / REFINEMENT_SUBPERIODS_NAME, index=False)
    pd.DataFrame([{"candidate_id": "candidate_0001"}]).to_csv(
        sweep_dir / REFINEMENT_ROLLING_WINDOWS_NAME,
        index=False,
    )
    results.to_csv(sweep_dir / REFINEMENT_DECISION_MATRIX_NAME, index=False)
    results.to_csv(
        refinement_report_dir / "long_history_qqq_core_sleeve_refinement_summary.csv",
        index=False,
    )
    results.to_csv(
        refinement_report_dir / "long_history_qqq_core_sleeve_refinement_decision_matrix.csv",
        index=False,
    )

    exit_code = run_long_history_qqq_core_sleeve_decision_report_from_args(
        [
            "--refinement-sweep-dir",
            str(sweep_dir),
            "--refinement-report-dir",
            str(refinement_report_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / SUMMARY_NAME).exists()
    assert (output_dir / DECISION_MATRIX_NAME).exists()
    assert (output_dir / RECOMMENDATIONS_NAME).exists()
    assert (output_dir / REPORT_NAME).exists()
    assert pd.read_csv(output_dir / DECISION_MATRIX_NAME)["production_ready"].eq(False).all()
    assert pd.read_csv(output_dir / RECOMMENDATIONS_NAME)["production_ready"].eq(False).all()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "Track next:" in markdown
    assert "Run another broad sweep: `no`" in markdown
    assert "Promote to production: `no`" in markdown


def _results() -> pd.DataFrame:
    rows = [
        _row("candidate_0001", 0.0, "daily_rebalanced_fixed_weights", 1.6, -0.3, 5.3),
        _row("candidate_0001", 0.35, "monthly_rebalanced_weights", 1.28, -0.24, 5.3),
        _row("candidate_0001", 0.35, "quarterly_rebalanced_weights", 1.26, -0.245, 5.1),
        _row("candidate_0001", 0.4, "monthly_rebalanced_weights", 1.18, -0.23, 5.1),
        _row(
            "candidate_0001",
            0.4,
            "candidate_rebalance_aligned_weights",
            1.2,
            -0.228,
            5.2,
        ),
        _row("candidate_0003", 0.0, "daily_rebalanced_fixed_weights", 1.35, -0.34, 4.2),
        _row("candidate_0003", 0.25, "monthly_rebalanced_weights", 1.18, -0.235, 5.0),
        _row("candidate_0003", 0.25, "quarterly_rebalanced_weights", 1.16, -0.236, 4.9),
        _row("candidate_0003", 0.3, "monthly_rebalanced_weights", 1.12, -0.225, 5.0),
        _row(
            "candidate_0003",
            0.3,
            "candidate_rebalance_aligned_weights",
            1.14,
            -0.222,
            5.1,
        ),
        _row("candidate_0003", 0.5, "daily_rebalanced_fixed_weights", 0.92, -0.205, 4.5),
    ]
    return pd.DataFrame(rows)


def _row(
    candidate_id: str,
    spy_weight: float,
    mode: str,
    total_return: float,
    max_drawdown: float,
    return_to_drawdown: float,
) -> dict[str, object]:
    qqq_return = 1.0
    qqq_drawdown = -0.25
    qqq_rtd = 4.0
    sleeve_weight = round(1.0 - spy_weight, 10)
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": sleeve_weight,
        "portfolio_accounting_mode": mode,
        "portfolio_name": f"{candidate_id} SPY {spy_weight:.0%} | {mode}",
        "is_core_sleeve": 0.0 < spy_weight < 1.0,
        "total_return": total_return,
        "strategy_vs_qqq": total_return - qqq_return,
        "qqq_return_capture": total_return / qqq_return,
        "max_drawdown": max_drawdown,
        "drawdown_improvement_vs_qqq": max_drawdown - qqq_drawdown,
        "return_to_drawdown": return_to_drawdown,
        "return_to_drawdown_vs_qqq": return_to_drawdown - qqq_rtd,
        "positive_vs_qqq_window_rate": 0.6,
        "worst_strategy_vs_qqq": -0.1,
        "beats_qqq_return": total_return > qqq_return,
        "improves_drawdown_vs_qqq": max_drawdown > qqq_drawdown,
        "improves_return_to_drawdown_vs_qqq": return_to_drawdown > qqq_rtd,
        "captures_90pct_qqq_return": total_return >= 0.9 * qqq_return,
        "production_ready": False,
    }


def _subperiods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "spy_core_weight": 0.35,
                "candidate_sleeve_weight": 0.65,
                "portfolio_accounting_mode": "monthly_rebalanced_weights",
                "period_name": "covid_crash_recovery",
                "total_return": 0.2,
                "qqq_total_return": 0.15,
                "strategy_vs_qqq": 0.05,
                "max_drawdown": -0.1,
                "drawdown_improvement_vs_qqq": 0.02,
            }
        ]
    )
