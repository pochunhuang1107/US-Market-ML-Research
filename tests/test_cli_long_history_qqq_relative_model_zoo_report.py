from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_relative_model_zoo_report import (
    DECISION_MATRIX_PATH_NAME,
    REPORT_PATH_NAME,
    SUMMARY_PATH_NAME,
    build_long_history_qqq_relative_model_zoo_report,
    classify_long_history_decision,
    run_long_history_qqq_relative_model_zoo_report_from_args,
)


def test_long_history_report_decision_logic() -> None:
    best = _result_row(total_return=0.9, qqq_return=0.8, max_drawdown=-0.25)
    risk = _result_row(
        total_return=0.6,
        qqq_return=0.8,
        max_drawdown=-0.10,
        return_to_drawdown=6.0,
        qqq_return_to_drawdown=2.0,
        drawdown_improvement=0.20,
    )
    mixed = _result_row(
        total_return=0.9,
        qqq_return=0.8,
        max_drawdown=-0.45,
        drawdown_improvement=-0.10,
    )
    reject = _result_row(total_return=0.4, qqq_return=0.8, max_drawdown=-0.30)

    assert classify_long_history_decision(pd.Series(best)) == "best_candidate"
    assert classify_long_history_decision(pd.Series(risk)) == "risk_adjusted_candidate"
    assert classify_long_history_decision(pd.Series(mixed)) == "mixed"
    assert classify_long_history_decision(pd.Series(reject)) == "reject"


def test_long_history_report_writes_summary_decision_matrix_and_markdown(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(
        [
            _result_row(
                model_type="ridge",
                training_mode="anchored_expanding_yearly",
                target_variant="excess_return_vs_qqq_63d",
                total_return=0.9,
                qqq_return=0.8,
                max_drawdown=-0.25,
            ),
            _result_row(
                model_type="elasticnet",
                training_mode="frozen_base_1999_2018",
                target_variant="return_drawdown_score_vs_qqq_63d",
                total_return=0.5,
                qqq_return=0.8,
                max_drawdown=-0.12,
                return_to_drawdown=4.0,
                qqq_return_to_drawdown=2.0,
                drawdown_improvement=0.18,
            ),
        ]
    ).to_csv(results_csv, index=False)

    exit_code = run_long_history_qqq_relative_model_zoo_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    output_dir = tmp_path / "report"
    summary = pd.read_csv(output_dir / SUMMARY_PATH_NAME)
    decision = pd.read_csv(output_dir / DECISION_MATRIX_PATH_NAME)
    markdown = (output_dir / REPORT_PATH_NAME).read_text(encoding="utf-8")

    assert exit_code == 0
    assert summary["decision"].tolist() == [
        "best_candidate",
        "risk_adjusted_candidate",
    ]
    assert decision.loc[0, "candidate_label"].startswith("ridge | anchored_expanding_yearly")
    assert "Did any configuration beat QQQ return? `Yes`" in markdown
    assert "Production export/model card/rich replay/deep-dive docs: `not created`" in markdown


def test_build_long_history_report_counts_qqq_and_rtd_beats(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(
        [
            _result_row(total_return=0.9, qqq_return=0.8, max_drawdown=-0.25),
            _result_row(
                total_return=0.5,
                qqq_return=0.8,
                max_drawdown=-0.12,
                return_to_drawdown=4.0,
                qqq_return_to_drawdown=2.0,
                drawdown_improvement=0.18,
            ),
        ]
    ).to_csv(results_csv, index=False)

    report = build_long_history_qqq_relative_model_zoo_report(results_csv=results_csv)

    assert report.result_row_count == 2
    assert report.qqq_beat_count == 1
    assert report.return_to_drawdown_beat_count == 2


def _result_row(
    *,
    model_type: str = "ridge",
    training_mode: str = "anchored_expanding_yearly",
    target_variant: str = "excess_return_vs_qqq_63d",
    total_return: float = 0.7,
    qqq_return: float = 0.8,
    max_drawdown: float = -0.25,
    return_to_drawdown: float | None = None,
    qqq_return_to_drawdown: float = 2.0,
    drawdown_improvement: float = 0.0,
) -> dict[str, object]:
    resolved_rtd = return_to_drawdown
    if resolved_rtd is None:
        resolved_rtd = total_return / abs(max_drawdown)
    return {
        "model_type": model_type,
        "training_mode": training_mode,
        "target_variant": target_variant,
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top3_equal_weight",
        "rebalance_frequency": "monthly",
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "total_return": total_return,
        "annualized_return": 0.1,
        "max_drawdown": max_drawdown,
        "sharpe_like": 0.05,
        "return_to_drawdown": resolved_rtd,
        "transaction_count": 12,
        "average_turnover": 0.25,
        "estimated_transaction_cost": 0.01,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": 0.5,
        "strategy_vs_qqq": total_return - qqq_return,
        "strategy_vs_spy": total_return - 0.5,
        "qqq_return_capture": total_return / qqq_return,
        "beats_qqq_return": total_return > qqq_return,
        "beats_qqq_return_to_drawdown": resolved_rtd > qqq_return_to_drawdown,
        "drawdown_improvement_vs_qqq": drawdown_improvement,
        "qqq_buy_hold_return_to_drawdown": qqq_return_to_drawdown,
    }
