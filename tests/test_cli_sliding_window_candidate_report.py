from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_candidate_report import (
    build_sliding_window_candidate_report,
    run_sliding_window_candidate_report_from_args,
)


def test_candidate_report_groups_and_aggregates_test_rows_only(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    failed_keys = tmp_path / "failed.txt"
    _synthetic_results().to_csv(results_csv, index=False)
    failed_keys.write_text("", encoding="utf-8")

    report = build_sliding_window_candidate_report(
        results_csv=results_csv,
        failed_keys=failed_keys,
        top_n=2,
    )

    metrics = report.metrics
    champion = metrics.loc[metrics["is_reference_champion"]].iloc[0]
    broad_linear = metrics.loc[
        metrics["candidate_id"].str.startswith(
            "broad_macro:baseline:linear_regression"
        )
    ].iloc[0]

    assert report.result_row_count == 9
    assert report.test_row_count == 6
    assert report.completed_key_count == 6
    assert report.failed_key_count == 0
    assert len(metrics) == 3

    assert champion["runs"] == 2
    assert champion["avg_test_return"] == pytest.approx(0.03)
    assert champion["median_test_return"] == pytest.approx(0.03)
    assert champion["min_test_return"] == pytest.approx(-0.02)
    assert champion["max_test_return"] == pytest.approx(0.08)
    assert champion["avg_sharpe_like"] == pytest.approx(0.025)
    assert champion["median_sharpe_like"] == pytest.approx(0.025)
    assert champion["avg_max_drawdown"] == pytest.approx(-0.125)
    assert champion["worst_max_drawdown"] == pytest.approx(-0.20)
    assert champion["avg_daily_volatility"] == pytest.approx(0.03)
    assert champion["positive_rate"] == pytest.approx(0.5)

    assert broad_linear["avg_test_return"] == pytest.approx(0.05)
    assert broad_linear["positive_rate"] == pytest.approx(1.0)


def test_candidate_report_identifies_champion_and_calculates_deltas(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_candidate_report(results_csv=results_csv, top_n=2)
    metrics = report.metrics
    champion = metrics.loc[metrics["is_reference_champion"]].iloc[0]
    broad_linear = metrics.loc[
        metrics["candidate_id"].str.startswith(
            "broad_macro:baseline:linear_regression"
        )
    ].iloc[0]

    assert champion["universe"] == "full"
    assert champion["model_type"] == "linear_regression"
    assert champion["top_k"] == 1
    assert champion["allocation_mode"] == "equal_weight"
    assert champion["transaction_cost_bps"] == pytest.approx(10.0)
    assert champion["feature_set_version"] == "default_features_v1"
    assert champion["random_state"] == 42

    assert champion["delta_avg_test_return"] == pytest.approx(0.0)
    assert champion["delta_avg_sharpe_like"] == pytest.approx(0.0)
    assert champion["avg_drawdown_improvement"] == pytest.approx(0.0)
    assert champion["worst_drawdown_improvement"] == pytest.approx(0.0)
    assert champion["delta_positive_rate"] == pytest.approx(0.0)

    assert broad_linear["delta_avg_test_return"] == pytest.approx(0.02)
    assert broad_linear["delta_avg_sharpe_like"] == pytest.approx(0.015)
    assert broad_linear["avg_drawdown_improvement"] == pytest.approx(0.045)
    assert broad_linear["worst_drawdown_improvement"] == pytest.approx(0.08)
    assert broad_linear["delta_positive_rate"] == pytest.approx(0.5)


def test_candidate_report_balanced_score_and_group_tables_exist(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_candidate_report(results_csv=results_csv, top_n=2)

    assert pd.notna(report.metrics["balanced_score"]).all()
    assert len(report.top_return) == 2
    assert len(report.top_sharpe) == 2
    assert len(report.top_drawdown) == 2
    assert len(report.top_balanced) == 2
    assert set(report.by_universe["universe"]) == {"full", "broad_macro"}
    assert set(report.by_model_type["model_type"]) == {"linear_regression", "ridge"}
    assert set(report.by_topk["top_k"]) == {1, 3}
    assert set(report.by_allocation_mode["allocation_mode"]) == {"equal_weight"}


def test_candidate_report_cli_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    failed_keys = tmp_path / "failed.txt"
    output_dir = tmp_path / "reports"
    _synthetic_results().to_csv(results_csv, index=False)
    failed_keys.write_text("failed-key\n\n", encoding="utf-8")

    return_code = run_sliding_window_candidate_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--failed-keys",
            str(failed_keys),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "sliding-window candidate report" in captured.out

    expected_files = [
        "sliding_window_candidate_metrics.csv",
        "sliding_window_candidate_top_return.csv",
        "sliding_window_candidate_top_sharpe.csv",
        "sliding_window_candidate_top_drawdown.csv",
        "sliding_window_candidate_top_balanced.csv",
        "sliding_window_candidate_by_universe.csv",
        "sliding_window_candidate_by_model_type.csv",
        "sliding_window_candidate_by_topk.csv",
        "sliding_window_candidate_by_allocation_mode.csv",
        "sliding_window_candidate_report.md",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    metrics = pd.read_csv(output_dir / "sliding_window_candidate_metrics.csv")
    top_return = pd.read_csv(output_dir / "sliding_window_candidate_top_return.csv")
    markdown = (output_dir / "sliding_window_candidate_report.md").read_text(
        encoding="utf-8"
    )

    assert len(metrics) == 3
    assert top_return.loc[0, "universe"] == "broad_macro"
    assert "Completed configs: 6" in markdown
    assert "Failed configs: 1" in markdown
    assert "not the sliding-window robustness leader" in markdown
    assert "Do not call any candidate production-ready" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [
        {
            "key": "champ",
            "universe": "full",
            "model_type": "linear_regression",
            "top_k": 1,
            "returns": [0.08, -0.02],
            "sharpes": [0.06, -0.01],
            "drawdowns": [-0.05, -0.20],
            "vols": [0.02, 0.04],
        },
        {
            "key": "broad-linear",
            "universe": "broad_macro",
            "model_type": "linear_regression",
            "top_k": 1,
            "returns": [0.06, 0.04],
            "sharpes": [0.05, 0.03],
            "drawdowns": [-0.04, -0.12],
            "vols": [0.02, 0.03],
        },
        {
            "key": "full-ridge",
            "universe": "full",
            "model_type": "ridge",
            "top_k": 3,
            "returns": [0.03, 0.02],
            "sharpes": [0.07, 0.04],
            "drawdowns": [-0.03, -0.10],
            "vols": [0.02, 0.02],
        },
    ]
    folds = [
        "tw2y_v6m_t3m_q01_20210101_20210331",
        "tw2y_v6m_t3m_q02_20210401_20210630",
    ]

    for candidate in candidates:
        rows.append(
            _row(
                f"{candidate['key']}-0",
                folds[0],
                "train",
                candidate["universe"],
                candidate["model_type"],
                candidate["top_k"],
                total_return=9.0,
                sharpe_like=9.0,
                max_drawdown=-9.0,
                daily_volatility=9.0,
            )
        )
        for index, fold in enumerate(folds):
            rows.append(
                _row(
                    f"{candidate['key']}-{index}",
                    fold,
                    "test",
                    candidate["universe"],
                    candidate["model_type"],
                    candidate["top_k"],
                    total_return=candidate["returns"][index],
                    sharpe_like=candidate["sharpes"][index],
                    max_drawdown=candidate["drawdowns"][index],
                    daily_volatility=candidate["vols"][index],
                )
            )

    return pd.DataFrame(rows)


def _row(
    run_key: str,
    fold: str,
    split: str,
    universe: str,
    model_type: str,
    top_k: int,
    *,
    total_return: float,
    sharpe_like: float,
    max_drawdown: float,
    daily_volatility: float,
) -> dict[str, object]:
    return {
        "run_key": run_key,
        "universe": universe,
        "data_path": f"artifacts/cache/universe_{universe}_2018_2026.parquet",
        "fold": fold,
        "split": split,
        "model_family": "baseline",
        "model_type": model_type,
        "feature_set_version": "default_features_v1",
        "feature_count": 23,
        "hidden_units": "none",
        "dropout": None,
        "scale_target": False,
        "learning_rate": None,
        "weight_decay": None,
        "batch_size": None,
        "max_epochs": None,
        "patience": None,
        "random_state": 42,
        "top_k": top_k,
        "min_holding_days": 10,
        "switch_margin": 0.002,
        "allocation_mode": "equal_weight",
        "score_temperature": 1.0,
        "max_weight": 1.0,
        "transaction_cost_bps": 10.0,
        "target_col": "excess_return_vs_spy_20d",
        "label_horizon_days": 20,
        "total_return": total_return,
        "mean_daily_return": 0.001,
        "daily_volatility": daily_volatility,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "final_equity": 1.0 + total_return,
        "epochs_trained": None,
    }
