from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_tiingo_selected_candidate_rich_replay_report import (
    read_rich_replay_artifacts,
    render_tiingo_selected_candidate_rich_replay_markdown,
    run_tiingo_selected_candidate_rich_replay_report_from_args,
)


def test_rich_replay_report_reads_artifacts_and_includes_decision_caveats(
    tmp_path: Path,
    capsys,
) -> None:
    _write_report_inputs(tmp_path)

    return_code = run_tiingo_selected_candidate_rich_replay_report_from_args(
        ["--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    markdown = (tmp_path / "rich_replay_report.md").read_text(encoding="utf-8")

    assert return_code == 0
    assert "rich replay report" in captured.out
    assert "SPY rows:" in captured.out
    assert "QQQ rows:" in captured.out
    assert "SPY" in markdown
    assert "QQQ" in markdown
    assert "Decision:" in markdown
    assert "keep selected candidate but not target" in markdown
    assert "Artifact limitations" in markdown
    assert "not production-ready" in markdown
    assert "QQQ buy-and-hold remains stronger" in markdown


def test_render_rich_replay_report_from_loaded_frames(tmp_path: Path) -> None:
    _write_report_inputs(tmp_path)

    artifact_data = read_rich_replay_artifacts(tmp_path)
    markdown = render_tiingo_selected_candidate_rich_replay_markdown(artifact_data)

    assert "Executive summary" in markdown
    assert "Strategy configuration" in markdown
    assert "Benchmark comparison vs SPY/QQQ" in markdown
    assert "Fold-by-fold performance" in markdown
    assert "Underperformance windows" in markdown
    assert "Holdings / selected symbols" in markdown
    assert "Turnover diagnostics" in markdown
    assert "historical ML champion" in markdown


def _write_report_inputs(base: Path) -> None:
    pd.DataFrame(
        [
            {
                "date": "2021-02-01",
                "fold": "fold_1",
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "strategy_equity": 1.00,
                "strategy_daily_return": 0.0,
                "strategy_drawdown": 0.0,
                "spy_equity": 1.00,
                "spy_daily_return": 0.0,
                "spy_drawdown": 0.0,
                "qqq_equity": 1.00,
                "qqq_daily_return": 0.0,
                "qqq_drawdown": 0.0,
            },
            {
                "date": "2021-02-02",
                "fold": "fold_1",
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "strategy_equity": 1.01,
                "strategy_daily_return": 0.01,
                "strategy_drawdown": 0.0,
                "spy_equity": 1.005,
                "spy_daily_return": 0.005,
                "spy_drawdown": 0.0,
                "qqq_equity": 1.02,
                "qqq_daily_return": 0.02,
                "qqq_drawdown": 0.0,
            },
        ]
    ).to_csv(base / "daily_strategy_equity.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2021-02-01",
                "fold": "fold_1",
                "symbol": "QQQ",
                "target_weight": 0.2,
                "prediction": 0.1,
                "signal_rank": 1,
                "selected": True,
                "rebalance_flag": True,
            }
        ]
    ).to_csv(base / "daily_holdings.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2021-02-01",
                "fold": "fold_1",
                "selected_symbols": "QQQ,SPY",
                "previous_symbols": "",
                "added_symbols": "QQQ,SPY",
                "removed_symbols": "",
                "turnover": 0.0,
                "transaction_cost": 0.0,
            }
        ]
    ).to_csv(base / "rebalance_log.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": "fold_1",
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "rebalance_count": 1,
                "average_turnover": 0.1,
                "max_turnover": 0.2,
                "total_transaction_cost": 0.001,
                "average_holding_days_if_available": 12.0,
            }
        ]
    ).to_csv(base / "trade_turnover_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": "fold_1",
                "train_window_years": 1,
                "validation_window_months": 1,
                "test_window_months": 3,
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "strategy_total_return": 0.01,
                "spy_total_return": 0.005,
                "qqq_total_return": 0.02,
                "strategy_vs_spy": 0.005,
                "strategy_vs_qqq": -0.01,
                "strategy_sharpe_like": 1.0,
                "spy_sharpe_like": 0.5,
                "qqq_sharpe_like": 1.5,
                "strategy_max_drawdown": -0.01,
                "spy_max_drawdown": -0.02,
                "qqq_max_drawdown": -0.03,
                "total_transaction_cost": 0.001,
                "mean_daily_turnover": 0.1,
            }
        ]
    ).to_csv(base / "fold_performance.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": "fold_1",
                "benchmark_symbol": "SPY",
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "strategy_total_return": 0.01,
                "benchmark_total_return": 0.005,
                "excess_return_vs_benchmark": 0.005,
                "underperformed_benchmark": False,
                "underperformance_severity": "none",
            },
            {
                "fold": "fold_1",
                "benchmark_symbol": "QQQ",
                "test_start": "2021-02-01",
                "test_end": "2021-04-30",
                "strategy_total_return": 0.01,
                "benchmark_total_return": 0.02,
                "excess_return_vs_benchmark": -0.01,
                "underperformed_benchmark": True,
                "underperformance_severity": "mild",
            },
        ]
    ).to_csv(base / "underperformance_windows.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": "QQQ",
                "selected_days": 12,
                "selected_fold_count": 1,
                "avg_target_weight": 0.2,
                "avg_prediction": 0.1,
                "avg_signal_rank": 1.0,
                "first_selected_date": "2021-02-01",
                "last_selected_date": "2021-02-12",
            }
        ]
    ).to_csv(base / "selected_symbol_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_source": "tiingo_only",
                "model_family": "baseline",
                "model_type": "ridge",
                "top_k": 5,
                "allocation_mode": "equal_weight",
                "min_holding_days": 20,
                "switch_margin": 0.005,
                "transaction_cost_bps": 10.0,
                "train_window_years": "1",
                "validation_window_months": "1",
                "test_window_months": "3",
                "universe_mode": "dynamic",
                "feature_selection_scope": "fold_train",
                "min_history_rows": 5,
                "min_train_eligible_rows": 1,
                "feature_set_version": "default_features_v1",
                "target_col": "excess_return_vs_spy_20d",
                "label_horizon_days": 20,
                "fold_count": 1,
                "daily_equity_rows": 2,
                "strategy_average_total_return": 0.01,
                "spy_average_total_return": 0.005,
                "qqq_average_total_return": 0.02,
                "average_strategy_vs_spy": 0.005,
                "average_strategy_vs_qqq": -0.01,
                "spy_underperformance_rate": 0.0,
                "qqq_underperformance_rate": 1.0,
                "strategy_average_sharpe_like": 1.0,
                "strategy_average_max_drawdown": -0.01,
                "strategy_worst_max_drawdown": -0.01,
                "spy_average_max_drawdown": -0.02,
                "qqq_average_max_drawdown": -0.03,
                "decision": "keep selected candidate but not target",
                "next_action": "feature engineering",
                "artifact_caveats": "not production-ready",
            }
        ]
    ).to_csv(base / "rich_replay_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "reference_role": "historical_champion",
                "reference_label": "linear_regression / top1 / hold10 / margin0.002 / cost10",
                "model_type": "linear_regression",
                "top_k": 1,
                "min_holding_days": 10,
                "switch_margin": 0.002,
                "fold_count": 1,
                "average_total_return": 0.005,
                "average_sharpe_like": 0.5,
                "average_max_drawdown": -0.02,
                "worst_max_drawdown": -0.02,
                "average_total_transaction_cost": 0.001,
                "average_mean_daily_turnover": 0.1,
            }
        ]
    ).to_csv(base / "reference_performance.csv", index=False)
