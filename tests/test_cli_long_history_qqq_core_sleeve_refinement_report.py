from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_core_sleeve_refinement_report import (
    REPORT_DECISION_MATRIX_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    run_long_history_qqq_core_sleeve_refinement_report_from_args,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    build_long_history_core_sleeve_refinement_artifacts,
    write_long_history_core_sleeve_refinement_artifacts,
)


def test_core_sleeve_refinement_report_writes_required_sections(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    report_dir = tmp_path / "report"
    previous_path = tmp_path / "previous_results.csv"
    artifacts = build_long_history_core_sleeve_refinement_artifacts(
        daily_equity=_daily_equity_frame("2020-01-01", "2023-12-31"),
        deep_dive_summary=pd.DataFrame(
            [
                _source_row("candidate_0001", "elasticnet"),
                _source_row("candidate_0003", "ridge"),
            ]
        ),
        deep_dive_decision_matrix=pd.DataFrame(
            [
                _decision_row("candidate_0001"),
                _decision_row("candidate_0003"),
            ]
        ),
        candidate_ids=("candidate_0001", "candidate_0003"),
        start_date="2020-01-01",
        end_date="2023-12-31",
    )
    write_long_history_core_sleeve_refinement_artifacts(
        artifacts=artifacts,
        output_dir=sweep_dir,
    )
    _previous_results().to_csv(previous_path, index=False)

    exit_code = run_long_history_qqq_core_sleeve_refinement_report_from_args(
        [
            "--sweep-output-dir",
            str(sweep_dir),
            "--output-dir",
            str(report_dir),
            "--previous-spy-core-sleeve-results",
            str(previous_path),
        ]
    )

    summary = pd.read_csv(report_dir / SUMMARY_NAME)
    decision = pd.read_csv(report_dir / REPORT_DECISION_MATRIX_NAME)
    markdown = (report_dir / REPORT_NAME).read_text(encoding="utf-8")

    assert exit_code == 0
    assert not summary.empty
    assert not decision.empty
    assert decision["production_ready"].eq(False).all()
    for section in [
        "## 1. Executive summary",
        "## 2. Why refinement was needed",
        "## 3. Prior coarse SPY core-sleeve recap",
        "## 4. Accounting modes tested",
        "## 5. Fine core-weight grid",
        "## 6. Full-period results",
        "## 7. Best drawdown-controlled results",
        "## 8. Best return-to-drawdown results",
        "## 9. Subperiod robustness",
        "## 10. Rolling-window robustness",
        "## 11. Candidate_0001 refined result",
        "## 12. Candidate_0003 refined result",
        "## 13. Recommendation",
        "## 14. Caveats",
    ]:
        assert section in markdown
    assert "Choose refined core-sleeve candidate for deep dive" in markdown
    assert "Continue standalone candidate" in markdown
    assert "Test QQQ core instead" in markdown
    assert "MLP batch: `no`" in markdown
    assert "Production promotion: `no`" in markdown
    assert "QQQ buy-and-hold remains the primary benchmark" in markdown
    assert "No production signal export" in markdown


def _previous_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "spy_core_weight": 0.4,
                "candidate_sleeve_weight": 0.6,
                "rebalance_policy": "daily_rebalanced_fixed_weights",
                "total_return": 4.5827,
                "strategy_vs_qqq": 0.7877,
                "max_drawdown": -0.3485,
                "return_to_drawdown": 13.1485,
                "decision": "best_core_sleeve_candidate",
            },
            {
                "candidate_id": "candidate_0003",
                "spy_core_weight": 0.3,
                "candidate_sleeve_weight": 0.7,
                "rebalance_policy": "daily_rebalanced_fixed_weights",
                "total_return": 4.3832,
                "strategy_vs_qqq": 0.5882,
                "max_drawdown": -0.3381,
                "return_to_drawdown": 12.9647,
                "decision": "best_core_sleeve_candidate",
            },
        ]
    )


def _daily_equity_frame(start: str, end: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, end, freq="D")
    for candidate_id in ("candidate_0001", "candidate_0003"):
        candidate_drift = 0.0018 if candidate_id == "candidate_0001" else 0.0015
        candidate_returns = _return_series(len(dates), candidate_drift)
        qqq_returns = _return_series(len(dates), 0.0012)
        spy_returns = _return_series(len(dates), 0.0008, shock=-0.012)
        specs = [
            ("candidate_strategy", candidate_returns),
            ("QQQ buy-and-hold", qqq_returns),
            ("SPY buy-and-hold", spy_returns),
        ]
        for strategy_name, returns in specs:
            equity = pd.Series(returns).add(1.0).cumprod()
            drawdown = equity / equity.cummax() - 1.0
            for idx, date in enumerate(dates):
                is_turnover = strategy_name == "candidate_strategy" and idx > 0 and idx % 90 == 0
                rows.append(
                    {
                        "date": date,
                        "candidate_id": candidate_id,
                        "candidate_label": f"{candidate_id} label",
                        "strategy_name": strategy_name,
                        "active_symbol": "AAA" if strategy_name == "candidate_strategy" else "",
                        "active_symbols": "AAA" if strategy_name == "candidate_strategy" else "",
                        "daily_return": returns[idx],
                        "equity": equity.iloc[idx],
                        "drawdown": drawdown.iloc[idx],
                        "gross_return": returns[idx],
                        "transaction_cost": 0.001 if is_turnover else 0.0,
                        "turnover": 1.0 if is_turnover else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _return_series(length: int, drift: float, shock: float = -0.02) -> list[float]:
    return [0.0 if idx == 0 else (shock if idx % 45 == 0 else drift) for idx in range(length)]


def _source_row(candidate_id: str, model_type: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": f"{candidate_id} label",
        "model_type": model_type,
        "training_mode": "recency_weighted_expanding"
        if model_type == "elasticnet"
        else "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_126d"
        if model_type == "elasticnet"
        else "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20 if model_type == "elasticnet" else 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "recency_decay_profile": "medium_decay" if model_type == "elasticnet" else pd.NA,
        "base_weight": pd.NA,
        "recent_window_years": pd.NA,
        "finetune_window_years": pd.NA,
        "finetune_lr_scale": pd.NA,
        "total_return": 1.0,
        "max_drawdown": -0.3,
        "return_to_drawdown": 3.3,
        "transaction_count": 5,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 0.8,
        "spy_buy_hold_total_return": 0.6,
        "strategy_vs_qqq": 0.2,
        "strategy_vs_spy": 0.4,
        "positive_vs_qqq_window_rate": 0.6,
        "qqq_underperformance_rate": 0.4,
        "worst_strategy_vs_qqq": -0.1,
        "top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "shortlist_decision": "keep_for_deep_dive",
    }


def _decision_row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "decision": "keep_for_targeted_sweep",
        "rationale": "research only",
        "cost_20bps_strategy_vs_qqq": 0.4,
        "cost_30bps_strategy_vs_qqq": 0.3,
    }
