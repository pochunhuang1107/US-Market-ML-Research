from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_stability_report import (
    CHAMPION_NAME,
    build_sliding_window_stability_report,
    parse_sliding_window_fold_name,
    run_sliding_window_stability_report_from_args,
)


def test_parse_sliding_window_fold_name_extracts_metadata() -> None:
    cases = [
        (
            "tw2y_v6m_t3m_q01_20200701_20200930",
            2,
            6,
            3,
            1,
            "2020-07-01",
            "2020-09-30",
            2020,
        ),
        (
            "tw3y_v6m_t3m_q18_20251001_20251231",
            3,
            6,
            3,
            18,
            "2025-10-01",
            "2025-12-31",
            2025,
        ),
        (
            "tw5y_v6m_t3m_q12_20260401_20260508",
            5,
            6,
            3,
            12,
            "2026-04-01",
            "2026-05-08",
            2026,
        ),
    ]

    for (
        fold_name,
        train_years,
        validation_months,
        test_months,
        sequence_number,
        test_start,
        test_end,
        test_year,
    ) in cases:
        metadata = parse_sliding_window_fold_name(fold_name)

        assert metadata.train_window_years == train_years
        assert metadata.validation_window_months == validation_months
        assert metadata.test_window_months == test_months
        assert metadata.sequence_number == sequence_number
        assert metadata.test_start == pd.Timestamp(test_start)
        assert metadata.test_end == pd.Timestamp(test_end)
        assert metadata.test_year == test_year


@pytest.mark.parametrize(
    "fold_name",
    [
        "fold_1",
        "tw2_v6m_t3m_q01_20200701_20200930",
        "tw2y_v6m_t3m_q01_20201001_20200930",
    ],
)
def test_parse_sliding_window_fold_name_validates(fold_name: str) -> None:
    with pytest.raises(ValueError):
        parse_sliding_window_fold_name(fold_name)


def test_build_sliding_window_stability_report_aggregates_test_rows_only(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    failed_keys = tmp_path / "failed.txt"
    _synthetic_results().to_csv(results_csv, index=False)
    failed_keys.write_text("", encoding="utf-8")

    report = build_sliding_window_stability_report(
        results_csv=results_csv,
        failed_keys=failed_keys,
        top_n=2,
    )

    summary = report.summary.iloc[0]
    assert report.result_row_count == 12
    assert report.completed_key_count == 4
    assert report.failed_key_count == 0
    assert summary["test_window_count"] == 4
    assert summary["mean_total_return"] == pytest.approx(0.015)
    assert summary["median_total_return"] == pytest.approx(0.01)
    assert summary["min_total_return"] == pytest.approx(-0.06)
    assert summary["max_total_return"] == pytest.approx(0.10)
    assert summary["positive_window_rate"] == pytest.approx(0.5)
    assert summary["mean_sharpe_like"] == pytest.approx(0.0175)
    assert summary["median_sharpe_like"] == pytest.approx(0.015)
    assert summary["mean_max_drawdown"] == pytest.approx(-0.125)
    assert summary["worst_max_drawdown"] == pytest.approx(-0.20)
    assert summary["mean_daily_volatility"] == pytest.approx(0.025)

    by_train_window = report.by_train_window.set_index("train_window_years")
    assert by_train_window.loc[2, "count"] == 2
    assert by_train_window.loc[2, "mean_total_return"] == pytest.approx(0.01)
    assert by_train_window.loc[2, "positive_window_rate"] == pytest.approx(0.5)
    assert by_train_window.loc[3, "mean_total_return"] == pytest.approx(0.02)
    assert by_train_window.loc[3, "worst_max_drawdown"] == pytest.approx(-0.20)

    by_year = report.by_year.set_index("test_year")
    assert by_year.loc[2020, "count"] == 2
    assert by_year.loc[2020, "mean_total_return"] == pytest.approx(0.01)
    assert by_year.loc[2021, "mean_total_return"] == pytest.approx(0.02)


def test_run_sliding_window_stability_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    failed_keys = tmp_path / "failed.txt"
    output_dir = tmp_path / "reports"
    _synthetic_results().to_csv(results_csv, index=False)
    failed_keys.write_text("failed-key-1\n\n", encoding="utf-8")

    return_code = run_sliding_window_stability_report_from_args(
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
    summary_path = output_dir / "sliding_window_summary.csv"
    by_train_window_path = output_dir / "sliding_window_by_train_window.csv"
    by_year_path = output_dir / "sliding_window_by_year.csv"
    worst_windows_path = output_dir / "sliding_window_worst_windows.csv"
    best_windows_path = output_dir / "sliding_window_best_windows.csv"
    markdown_path = output_dir / "sliding_window_stability_report.md"

    assert return_code == 0
    assert "sliding-window stability report" in captured.out
    assert summary_path.exists()
    assert by_train_window_path.exists()
    assert by_year_path.exists()
    assert worst_windows_path.exists()
    assert best_windows_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    worst_windows = pd.read_csv(worst_windows_path)
    best_windows = pd.read_csv(best_windows_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert summary.loc[0, "test_window_count"] == 4
    assert set(worst_windows["ranking"]) == {"worst_total_return", "worst_max_drawdown"}
    assert len(worst_windows) == 4
    assert best_windows["ranking"].tolist() == ["best_total_return", "best_total_return"]
    assert len(best_windows) == 2
    assert CHAMPION_NAME in markdown
    assert "Sliding-window stability is not confirmed" in markdown
    assert "not production-ready" in markdown
    assert "Failed keys: 1" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_specs = [
        ("tw2y_v6m_t3m_q01_20200701_20200930", 0.04, 0.04, -0.10, 0.02),
        ("tw2y_v6m_t3m_q02_20201001_20201231", -0.02, -0.01, -0.12, 0.03),
        ("tw3y_v6m_t3m_q01_20210701_20210930", 0.10, 0.08, -0.08, 0.01),
        ("tw3y_v6m_t3m_q02_20211001_20211231", -0.06, -0.04, -0.20, 0.04),
    ]

    for index, (fold, total_return, sharpe_like, max_drawdown, volatility) in enumerate(
        test_specs,
        start=1,
    ):
        for split in ["train", "validation", "test"]:
            multiplier = 1.0 if split == "test" else 10.0
            rows.append(
                {
                    "run_key": f"run-{index}",
                    "universe": "full",
                    "fold": fold,
                    "split": split,
                    "model_family": "baseline",
                    "model_type": "linear_regression",
                    "feature_set_version": "default_features_v1",
                    "top_k": 1,
                    "min_holding_days": 10,
                    "switch_margin": 0.002,
                    "allocation_mode": "equal_weight",
                    "score_temperature": 1.0,
                    "max_weight": 1.0,
                    "transaction_cost_bps": 10.0,
                    "target_col": "excess_return_vs_spy_20d",
                    "label_horizon_days": 20,
                    "total_return": total_return * multiplier,
                    "mean_daily_return": 0.001 * multiplier,
                    "daily_volatility": volatility * multiplier,
                    "sharpe_like": sharpe_like * multiplier,
                    "max_drawdown": max_drawdown * multiplier,
                    "final_equity": 1.0 + total_return * multiplier,
                }
            )

    return pd.DataFrame(rows)
