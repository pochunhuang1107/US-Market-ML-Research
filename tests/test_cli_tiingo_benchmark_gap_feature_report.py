from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_tiingo_benchmark_gap_feature_report import (
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    build_tiingo_benchmark_gap_feature_report,
    run_tiingo_benchmark_gap_feature_report_from_args,
)
from us_market_ml.cli.create_tiingo_benchmark_gap_feature_v2_report import (
    run_tiingo_benchmark_gap_feature_v2_report_from_args,
)


def test_benchmark_gap_feature_report_builds_summary_and_decision_matrix(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    market_csv = tmp_path / "market.csv"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_market_data().to_csv(market_csv, index=False)

    report_data = build_tiingo_benchmark_gap_feature_report(
        results_csv=results_csv,
        market_dataset_glob=market_csv,
    )

    assert list(report_data.feature_summary.columns) == SUMMARY_COLUMNS
    assert list(report_data.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert report_data.dataset_source == "tiingo_only"
    assert set(report_data.feature_summary["feature_set_version"]) == {
        "default_features_v1",
        "benchmark_gap_features_v1",
    }
    assert set(report_data.decision_matrix["candidate_role"]) == {
        "return_reference",
        "selected_candidate",
    }

    selected_feature = report_data.decision_matrix.loc[
        (report_data.decision_matrix["candidate_role"] == "selected_candidate")
        & (
            report_data.decision_matrix["feature_set_version"]
            == "benchmark_gap_features_v1"
        )
    ].iloc[0]
    assert selected_feature["delta_average_return_vs_default"] == pytest.approx(0.02)
    assert selected_feature["delta_strategy_vs_spy_vs_default"] == pytest.approx(0.02)
    assert selected_feature["delta_strategy_vs_qqq_vs_default"] == pytest.approx(0.02)
    assert selected_feature["benchmark_gap_change"] == "improved versus SPY and QQQ"
    assert "still win" in selected_feature["benchmark_status"]
    assert selected_feature["feature_decision"] == (
        "keep feature set for next feature-engineering iteration"
    )


def test_benchmark_gap_feature_report_cli_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    market_csv = tmp_path / "market.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_market_data().to_csv(market_csv, index=False)

    return_code = run_tiingo_benchmark_gap_feature_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--market-dataset-glob",
            str(market_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    summary_path = output_dir / "tiingo_benchmark_gap_feature_summary.csv"
    decision_path = output_dir / "tiingo_benchmark_gap_feature_decision_matrix.csv"
    markdown_path = output_dir / "tiingo_benchmark_gap_feature_report.md"

    assert return_code == 0
    assert "Tiingo benchmark-gap feature study report" in captured.out
    assert "dataset source: tiingo_only" in captured.out
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    matrix = pd.read_csv(decision_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert set(summary["dataset_source"]) == {"tiingo_only"}
    assert set(matrix["dataset_source"]) == {"tiingo_only"}
    assert "Executive Summary" in markdown
    assert "Selected Candidate Feature Comparison" in markdown
    assert "Decision Matrix" in markdown
    assert "Reference Rows" in markdown
    assert "Benchmark-gap feature verdict" in markdown
    assert "SPY and QQQ still win on average" in markdown
    assert "`tree_screen` and MLP remain deferred" in markdown
    assert "does not export production signals" in markdown


def test_benchmark_gap_feature_v2_report_compares_default_v1_and_v2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    market_csv = tmp_path / "market.csv"
    output_dir = tmp_path / "report"
    _synthetic_results(
        {
            "default_features_v1": 0.03,
            "benchmark_gap_features_v1": 0.05,
            "benchmark_gap_features_v2": 0.07,
        }
    ).to_csv(results_csv, index=False)
    _synthetic_market_data().to_csv(market_csv, index=False)

    report_data = build_tiingo_benchmark_gap_feature_report(
        results_csv=results_csv,
        market_dataset_glob=market_csv,
    )
    v2_selected = report_data.decision_matrix.loc[
        (report_data.decision_matrix["candidate_role"] == "selected_candidate")
        & (
            report_data.decision_matrix["feature_set_version"]
            == "benchmark_gap_features_v2"
        )
    ].iloc[0]

    assert set(report_data.feature_summary["feature_set_version"]) == {
        "default_features_v1",
        "benchmark_gap_features_v1",
        "benchmark_gap_features_v2",
    }
    assert v2_selected["delta_strategy_vs_spy_vs_default"] == pytest.approx(0.04)
    assert v2_selected["delta_strategy_vs_qqq_vs_default"] == pytest.approx(0.04)
    assert v2_selected["delta_strategy_vs_spy_vs_benchmark_gap_v1"] == pytest.approx(0.02)
    assert v2_selected["delta_strategy_vs_qqq_vs_benchmark_gap_v1"] == pytest.approx(0.02)
    assert v2_selected["benchmark_gap_change_vs_benchmark_gap_v1"] == (
        "improved versus v1 SPY and QQQ gaps"
    )
    assert v2_selected["feature_decision"] == (
        "keep v2 for next feature-engineering iteration"
    )

    return_code = run_tiingo_benchmark_gap_feature_v2_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--market-dataset-glob",
            str(market_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    summary_path = output_dir / "tiingo_benchmark_gap_feature_v2_summary.csv"
    decision_path = output_dir / "tiingo_benchmark_gap_feature_v2_decision_matrix.csv"
    markdown_path = output_dir / "tiingo_benchmark_gap_feature_v2_report.md"

    assert return_code == 0
    assert "Tiingo benchmark-gap feature v2 study report" in captured.out
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()
    assert "Tiingo Benchmark-Gap Feature V2 Study Report" in markdown_path.read_text(
        encoding="utf-8"
    )


def _synthetic_results(
    feature_returns: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    folds = [
        ("tw1y_v1m_t3m_q01_20210101_20210110", 0.0),
        ("tw1y_v1m_t3m_q02_20210201_20210210", 0.01),
    ]
    specs = [
        ("selected_ridge_top5", "selected_candidate", "ridge", 5, 20, 0.005),
        ("ridge_top1_return_reference", "return_reference", "ridge", 1, 20, 0.005),
    ]
    if feature_returns is None:
        feature_returns = {
            "default_features_v1": 0.03,
            "benchmark_gap_features_v1": 0.05,
        }
    for candidate_family, candidate_role, model_type, top_k, hold, margin in specs:
        role_adjustment = 0.0 if candidate_role == "selected_candidate" else -0.01
        for feature_set_version, base_return in feature_returns.items():
            for fold_name, fold_adjustment in folds:
                total_return = base_return + fold_adjustment + role_adjustment
                rows.append(
                    {
                        "run_key": (
                            f"{candidate_role}-{feature_set_version}-{fold_name}"
                        ),
                        "universe": "tiingo_benchmark_gap_feature_dynamic",
                        "data_path": "/tmp/tiingo.csv",
                        "sweep_profile": "benchmark_gap_feature_study",
                        "candidate_family": candidate_family,
                        "candidate_role": candidate_role,
                        "study_stage": "feature_engineering",
                        "fold": fold_name,
                        "split": "test",
                        "model_family": "baseline",
                        "model_type": model_type,
                        "feature_set_version": feature_set_version,
                        "feature_count": _synthetic_feature_count(feature_set_version),
                        "hidden_units": "none",
                        "dropout": "",
                        "scale_target": False,
                        "learning_rate": "",
                        "weight_decay": "",
                        "batch_size": "",
                        "max_epochs": "",
                        "patience": "",
                        "random_state": 42,
                        "top_k": top_k,
                        "min_holding_days": hold,
                        "switch_margin": margin,
                        "allocation_mode": "equal_weight",
                        "score_temperature": 1.0,
                        "max_weight": 1.0,
                        "transaction_cost_bps": 10.0,
                        "target_col": "excess_return_vs_spy_20d",
                        "label_horizon_days": 20,
                        "total_return": total_return,
                        "mean_daily_return": total_return / 10,
                        "daily_volatility": 0.01,
                        "sharpe_like": total_return / 0.05,
                        "max_drawdown": -0.05,
                        "final_equity": 1.0 + total_return,
                        "total_transaction_cost": 0.001,
                        "mean_daily_turnover": 0.10,
                        "max_daily_turnover": 0.50,
                        "epochs_trained": "",
                    }
                )
    rows.append({**rows[0], "split": "train", "run_key": "ignored-train-row"})
    return pd.DataFrame(rows)


def _synthetic_feature_count(feature_set_version: str) -> int:
    if feature_set_version == "default_features_v1":
        return 20
    if feature_set_version == "benchmark_gap_features_v1":
        return 54
    return 88


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2021-01-01", "2021-02-10", freq="D")
    for symbol, base_price, daily_step in [
        ("SPY", 100.0, 1.0),
        ("QQQ", 100.0, 1.5),
        ("IWM", 50.0, 0.2),
    ]:
        for day_index, date in enumerate(dates):
            close = base_price + day_index * daily_step
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000 + day_index,
                    "vwap": close,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
