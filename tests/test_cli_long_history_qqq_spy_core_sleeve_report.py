from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_spy_core_sleeve_report import (
    LongHistoryQQQSpyCoreSleeveReportData,
    render_long_history_qqq_spy_core_sleeve_markdown,
    run_long_history_qqq_spy_core_sleeve_report_from_args,
)
from us_market_ml.long_history_spy_core_sleeve import (
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    RESULT_COLUMNS,
    ROLLING_WINDOW_COLUMNS,
    SUBPERIOD_COLUMNS,
    build_spy_core_sleeve_decision_matrix,
)


def test_spy_core_sleeve_report_writes_required_sections(tmp_path: Path) -> None:
    report_data = _report_data()
    report = render_long_history_qqq_spy_core_sleeve_markdown(report_data)

    for section in [
        "## 1. Executive summary",
        "## 2. Why SPY core + candidate sleeve",
        "## 3. Candidate source recap",
        "## 4. Core weight grid",
        "## 5. Full-period comparison versus QQQ/SPY/standalone candidate",
        "## 6. Best total-return portfolios",
        "## 7. Best drawdown-controlled portfolios",
        "## 8. Best return-to-drawdown portfolios",
        "## 9. Subperiod robustness",
        "## 10. Rolling-window robustness",
        "## 11. Candidate_0001 sleeve results",
        "## 12. Candidate_0003 sleeve results",
        "## 13. Recommendation",
        "## 14. Caveats",
    ]:
        assert section in report
    assert "Rich replay best core-sleeve candidate" in report
    assert "Target sweep around core weight" in report
    assert "Continue standalone candidate" in report
    assert "MLP batch: `no`" in report
    assert "Production promotion: `no`" in report
    assert "QQQ buy-and-hold remains the primary benchmark" in report
    assert "daily rebalanced fixed weights" in report

    sweep_dir = tmp_path / "sweep"
    output_dir = tmp_path / "report"
    _write_sweep_artifacts(sweep_dir, report_data)
    exit_code = run_long_history_qqq_spy_core_sleeve_report_from_args(
        ["--sweep-output-dir", str(sweep_dir), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    assert (output_dir / "long_history_qqq_spy_core_sleeve_summary.csv").exists()
    assert (output_dir / "long_history_qqq_spy_core_sleeve_decision_matrix.csv").exists()
    assert (output_dir / "long_history_qqq_spy_core_sleeve_report.md").exists()
    decision = pd.read_csv(output_dir / "long_history_qqq_spy_core_sleeve_decision_matrix.csv")
    assert decision["production_ready"].eq(False).all()


def _report_data() -> LongHistoryQQQSpyCoreSleeveReportData:
    results = pd.DataFrame(
        [
            _result_row("candidate_0001", 0.0, 1.0, 1.30, -0.35, 3.7),
            _result_row("candidate_0001", 0.5, 0.5, 1.00, -0.20, 5.0),
            _result_row("candidate_0003", 0.5, 0.5, 0.95, -0.18, 5.3),
        ],
        columns=RESULT_COLUMNS,
    )
    decision = build_spy_core_sleeve_decision_matrix(results)
    daily_equity = pd.DataFrame([_daily_row()], columns=DAILY_EQUITY_COLUMNS)
    daily_drawdown = daily_equity.loc[
        :,
        [
            "date",
            "candidate_id",
            "candidate_label",
            "spy_core_weight",
            "candidate_sleeve_weight",
            "portfolio_name",
            "strategy_name",
            "equity",
            "drawdown",
            "rebalance_policy",
        ],
    ].copy()
    daily_drawdown = daily_drawdown.loc[:, DAILY_DRAWDOWN_COLUMNS]
    subperiods = pd.DataFrame(
        [_subperiod_row("candidate_0001", 0.5), _subperiod_row("candidate_0003", 0.5)],
        columns=SUBPERIOD_COLUMNS,
    )
    rolling_windows = pd.DataFrame(
        [_rolling_row("candidate_0001", 0.5), _rolling_row("candidate_0003", 0.5)],
        columns=ROLLING_WINDOW_COLUMNS,
    )
    return LongHistoryQQQSpyCoreSleeveReportData(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        decision_matrix=decision,
    )


def _write_sweep_artifacts(
    output_dir: Path,
    report_data: LongHistoryQQQSpyCoreSleeveReportData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "long_history_qqq_spy_core_sleeve_results.csv": report_data.results,
        "long_history_qqq_spy_core_sleeve_daily_equity.csv": report_data.daily_equity,
        "long_history_qqq_spy_core_sleeve_daily_drawdown.csv": report_data.daily_drawdown,
        "long_history_qqq_spy_core_sleeve_subperiods.csv": report_data.subperiods,
        "long_history_qqq_spy_core_sleeve_rolling_windows.csv": report_data.rolling_windows,
        "long_history_qqq_spy_core_sleeve_decision_matrix.csv": report_data.decision_matrix,
    }
    for name, frame in mapping.items():
        frame.to_csv(output_dir / name, index=False)


def _result_row(
    candidate_id: str,
    spy_weight: float,
    candidate_weight: float,
    total_return: float,
    max_drawdown: float,
    return_to_drawdown: float,
) -> dict[str, object]:
    row = {column: pd.NA for column in RESULT_COLUMNS}
    qqq_return = 1.0
    qqq_drawdown = -0.25
    qqq_rtd = 4.0
    row.update(
        {
            "candidate_id": candidate_id,
            "candidate_label": candidate_id,
            "model_type": "ridge",
            "training_mode": "frozen_base_1999_2018",
            "target_variant": "excess_return_vs_qqq_63d",
            "feature_set_version": "long_history_regime_features_v1",
            "allocation_rule": "top1",
            "rebalance_frequency": "quarterly",
            "min_holding_days": 0,
            "switch_margin": 0.0,
            "transaction_cost_bps": 10.0,
            "spy_core_weight": spy_weight,
            "candidate_sleeve_weight": candidate_weight,
            "portfolio_name": f"{candidate_id} {spy_weight}",
            "rebalance_policy": "daily_rebalanced_fixed_weights",
            "is_core_sleeve": 0.0 < spy_weight < 1.0,
            "total_return": total_return,
            "annualized_return": 0.2,
            "daily_volatility": 0.01,
            "sharpe_like": 0.1,
            "max_drawdown": max_drawdown,
            "return_to_drawdown": return_to_drawdown,
            "qqq_buy_hold_total_return": qqq_return,
            "spy_buy_hold_total_return": 0.7,
            "standalone_candidate_total_return": 1.3,
            "strategy_vs_qqq": total_return - qqq_return,
            "strategy_vs_spy": total_return - 0.7,
            "qqq_return_capture": total_return / qqq_return,
            "spy_return_capture": total_return / 0.7,
            "drawdown_improvement_vs_qqq": max_drawdown - qqq_drawdown,
            "drawdown_improvement_vs_candidate": max_drawdown - (-0.35),
            "return_to_drawdown_vs_qqq": return_to_drawdown - qqq_rtd,
            "transaction_count": 5,
            "average_turnover": 0.01,
            "estimated_transaction_cost": 0.02,
            "positive_vs_qqq_window_rate": 0.6,
            "qqq_underperformance_rate": 0.4,
            "worst_strategy_vs_qqq": -0.1,
            "cost_20bps_strategy_vs_qqq": total_return - qqq_return - 0.02,
            "cost_30bps_strategy_vs_qqq": total_return - qqq_return - 0.03,
            "qqq_buy_hold_max_drawdown": qqq_drawdown,
            "spy_buy_hold_max_drawdown": -0.15,
            "standalone_candidate_max_drawdown": -0.35,
            "qqq_buy_hold_return_to_drawdown": qqq_rtd,
            "spy_buy_hold_return_to_drawdown": 4.7,
            "standalone_candidate_return_to_drawdown": 3.7,
            "beats_qqq_return": total_return > qqq_return,
            "improves_drawdown_vs_qqq": max_drawdown > qqq_drawdown,
            "improves_return_to_drawdown_vs_qqq": return_to_drawdown > qqq_rtd,
            "captures_90pct_qqq_return": total_return / qqq_return >= 0.9,
            "source_strategy_vs_qqq": 0.3,
            "source_return_to_drawdown": 3.7,
            "source_positive_vs_qqq_window_rate": 0.6,
            "source_worst_strategy_vs_qqq": -0.1,
            "source_top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
            "source_top_symbol_exposure_pct": 0.44,
            "source_top3_symbol_exposure_pct": 0.86,
            "source_decision": "keep_for_targeted_sweep",
            "source_rationale": "source",
        }
    )
    return row


def _daily_row() -> dict[str, object]:
    row = {column: pd.NA for column in DAILY_EQUITY_COLUMNS}
    row.update(
        {
            "date": "2020-01-01",
            "candidate_id": "candidate_0001",
            "candidate_label": "candidate_0001",
            "spy_core_weight": 0.5,
            "candidate_sleeve_weight": 0.5,
            "portfolio_name": "candidate_0001 0.5",
            "strategy_name": "SPY core + candidate sleeve",
            "daily_return": 0.0,
            "equity": 1.0,
            "drawdown": 0.0,
            "candidate_daily_return": 0.0,
            "candidate_equity": 1.0,
            "candidate_drawdown": 0.0,
            "qqq_daily_return": 0.0,
            "qqq_equity": 1.0,
            "qqq_drawdown": 0.0,
            "spy_daily_return": 0.0,
            "spy_equity": 1.0,
            "spy_drawdown": 0.0,
            "rebalance_policy": "daily_rebalanced_fixed_weights",
        }
    )
    return row


def _subperiod_row(candidate_id: str, spy_weight: float) -> dict[str, object]:
    row = {column: pd.NA for column in SUBPERIOD_COLUMNS}
    row.update(
        {
            "candidate_id": candidate_id,
            "candidate_label": candidate_id,
            "spy_core_weight": spy_weight,
            "candidate_sleeve_weight": 1.0 - spy_weight,
            "portfolio_name": f"{candidate_id} {spy_weight}",
            "period_name": "bear_2022",
            "period_start": "2022-01-01",
            "period_end": "2022-12-31",
            "total_return": 0.1,
            "qqq_total_return": -0.2,
            "spy_total_return": -0.1,
            "standalone_candidate_total_return": 0.2,
            "strategy_vs_qqq": 0.3,
            "strategy_vs_spy": 0.2,
            "max_drawdown": -0.1,
            "qqq_max_drawdown": -0.25,
            "spy_max_drawdown": -0.2,
            "standalone_candidate_max_drawdown": -0.3,
            "drawdown_improvement_vs_qqq": 0.15,
            "drawdown_improvement_vs_candidate": 0.2,
            "return_to_drawdown": 1.0,
            "qqq_return_to_drawdown": -0.8,
            "spy_return_to_drawdown": -0.5,
            "standalone_candidate_return_to_drawdown": 0.7,
            "transaction_count": 1,
        }
    )
    return row


def _rolling_row(candidate_id: str, spy_weight: float) -> dict[str, object]:
    row = {column: pd.NA for column in ROLLING_WINDOW_COLUMNS}
    row.update(
        {
            "candidate_id": candidate_id,
            "candidate_label": candidate_id,
            "spy_core_weight": spy_weight,
            "candidate_sleeve_weight": 1.0 - spy_weight,
            "portfolio_name": f"{candidate_id} {spy_weight}",
            "window_label": "1y_20200101_20201231",
            "window_years": 1,
            "window_start": "2020-01-01",
            "window_end": "2020-12-31",
            "strategy_total_return": 0.2,
            "qqq_total_return": 0.1,
            "spy_total_return": 0.05,
            "standalone_candidate_total_return": 0.3,
            "strategy_vs_qqq": 0.1,
            "strategy_max_drawdown": -0.15,
            "qqq_max_drawdown": -0.25,
            "spy_max_drawdown": -0.2,
            "standalone_candidate_max_drawdown": -0.3,
            "drawdown_improvement_vs_qqq": 0.1,
            "drawdown_improvement_vs_candidate": 0.15,
            "underperformed_qqq": False,
            "materially_improved_drawdown": True,
        }
    )
    return row
