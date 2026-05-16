from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_longer_window_shortlist_report import (
    build_sliding_window_longer_window_shortlist_report,
    calculate_buy_and_hold_window_return,
    run_sliding_window_longer_window_shortlist_report_from_args,
)


def test_fold_parsing_extracts_6_and_12_month_metadata(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_longer_window_shortlist_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    metrics = report.window_metrics.set_index(["candidate_id", "test_window_months"])
    six_month = metrics.loc[
        ("broad_macro:linear_regression:top1:equal_weight:cost10", 6)
    ]
    twelve_month = metrics.loc[
        ("broad_macro:linear_regression:top1:equal_weight:cost10", 12)
    ]

    assert six_month["fold"] == "tw2y_v6m_t6m_q01_20210101_20210630"
    assert six_month["train_window_years"] == 2
    assert six_month["validation_window_months"] == 6
    assert six_month["sequence_number"] == 1
    assert six_month["test_start"] == "2021-01-01"
    assert six_month["test_end"] == "2021-06-30"
    assert six_month["test_year"] == 2021

    assert twelve_month["fold"] == "tw3y_v6m_t12m_q02_20210101_20211231"
    assert twelve_month["train_window_years"] == 3
    assert twelve_month.name[1] == 12
    assert twelve_month["sequence_number"] == 2
    assert twelve_month["test_end"] == "2021-12-31"


def test_benchmark_return_calculation_uses_first_and_last_close() -> None:
    market_data = pd.DataFrame(
        [
            {"date": "2021-06-30", "symbol": "SPY", "close": 115.0},
            {"date": "2021-01-01", "symbol": "SPY", "close": 100.0},
            {"date": "2021-03-31", "symbol": "SPY", "close": 90.0},
        ]
    )

    result = calculate_buy_and_hold_window_return(
        market_data,
        symbol="SPY",
        test_start="2021-01-01",
        test_end="2021-06-30",
    )
    missing = calculate_buy_and_hold_window_return(
        market_data,
        symbol="QQQ",
        test_start="2021-01-01",
        test_end="2021-06-30",
    )

    assert result == pytest.approx(0.15)
    assert pd.isna(missing)


def test_candidate_summary_aggregates_longer_window_rows(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_longer_window_shortlist_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    summary = report.candidate_summary.set_index("candidate_id")
    linear = summary.loc["broad_macro:linear_regression:top1:equal_weight:cost10"]
    forest = summary.loc["broad_macro:random_forest:top3:equal_weight:cost10"]

    assert report.result_row_count == 7
    assert report.test_row_count == 6
    assert report.shortlist_test_row_count == 4
    assert linear["window_count"] == 2
    assert linear["mean_total_return"] == pytest.approx(0.08)
    assert linear["median_total_return"] == pytest.approx(0.08)
    assert linear["positive_window_rate"] == pytest.approx(1.0)
    assert linear["mean_sharpe_like"] == pytest.approx(0.40)
    assert linear["mean_max_drawdown"] == pytest.approx(-0.055)
    assert linear["worst_max_drawdown"] == pytest.approx(-0.08)
    assert linear["spy_mean_total_return"] == pytest.approx(0.15)
    assert linear["qqq_mean_total_return"] == pytest.approx(0.05)
    assert linear["strategy_beats_spy_rate"] == pytest.approx(0.5)
    assert linear["strategy_beats_qqq_rate"] == pytest.approx(0.5)
    assert linear["mean_strategy_minus_spy"] == pytest.approx(-0.07)
    assert linear["mean_strategy_minus_qqq"] == pytest.approx(0.03)

    assert forest["window_count"] == 2
    assert forest["mean_total_return"] == pytest.approx(0.11)
    assert forest["balanced_score"] > linear["balanced_score"]
    assert "benchmark_stability_score" in report.candidate_summary.columns


def test_candidate_by_test_window_aggregation(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_longer_window_shortlist_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    by_window = report.by_test_window.set_index("test_window_months")
    by_candidate_window = report.by_candidate_and_test_window.set_index(
        ["candidate_id", "test_window_months"]
    )

    assert set(by_window.index) == {6, 12}
    assert by_window.loc[6, "window_count"] == 2
    assert by_window.loc[6, "candidate_count"] == 2
    assert by_window.loc[6, "mean_total_return"] == pytest.approx(0.10)
    assert by_window.loc[12, "mean_total_return"] == pytest.approx(0.09)

    forest_t12 = by_candidate_window.loc[
        ("broad_macro:random_forest:top3:equal_weight:cost10", 12)
    ]
    assert forest_t12["window_count"] == 1
    assert forest_t12["mean_total_return"] == pytest.approx(0.14)
    assert forest_t12["strategy_beats_spy_rate"] == pytest.approx(0.0)
    assert forest_t12["strategy_beats_qqq_rate"] == pytest.approx(0.0)


def test_run_longer_window_shortlist_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    full_market_data = tmp_path / "full.parquet"
    broad_market_data = tmp_path / "broad.parquet"
    failed_keys = tmp_path / "failed.txt"
    output_dir = tmp_path / "reports"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_market_data("full", include_ohlcv=True).to_parquet(
        full_market_data,
        index=False,
    )
    _synthetic_market_data("broad_macro", include_ohlcv=True).to_parquet(
        broad_market_data,
        index=False,
    )
    failed_keys.write_text("", encoding="utf-8")

    return_code = run_sliding_window_longer_window_shortlist_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--failed-keys",
            str(failed_keys),
            "--full-market-data",
            str(full_market_data),
            "--broad-macro-market-data",
            str(broad_market_data),
            "--output-dir",
            str(output_dir),
            "--benchmark-symbols",
            "SPY,QQQ",
            "--top-n",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "longer-window sliding-window shortlist report" in captured.out

    expected_files = [
        "longer_window_shortlist_window_metrics.csv",
        "longer_window_shortlist_candidate_summary.csv",
        "longer_window_shortlist_by_test_window.csv",
        "longer_window_shortlist_by_candidate_and_test_window.csv",
        "longer_window_shortlist_by_year.csv",
        "longer_window_shortlist_relative_windows.csv",
        "longer_window_shortlist_report.md",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    window_metrics = pd.read_csv(output_dir / expected_files[0])
    candidate_summary = pd.read_csv(output_dir / expected_files[1])
    by_candidate_window = pd.read_csv(output_dir / expected_files[3])
    markdown = (output_dir / "longer_window_shortlist_report.md").read_text(
        encoding="utf-8"
    )

    assert len(window_metrics) == 4
    assert len(candidate_summary) == 2
    assert len(by_candidate_window) == 4
    assert "Completion Status" in markdown
    assert "6-Month vs 12-Month Comparison" in markdown
    assert "Benchmark-Relative Ranking" in markdown
    assert "Do not call any candidate production-ready" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    folds = [
        "tw2y_v6m_t6m_q01_20210101_20210630",
        "tw3y_v6m_t12m_q02_20210101_20211231",
    ]
    candidates = [
        {
            "key": "broad-linear",
            "universe": "broad_macro",
            "model_type": "linear_regression",
            "top_k": 1,
            "returns": [0.12, 0.04],
            "sharpes": [0.50, 0.30],
            "drawdowns": [-0.03, -0.08],
        },
        {
            "key": "broad-forest",
            "universe": "broad_macro",
            "model_type": "random_forest",
            "top_k": 3,
            "returns": [0.08, 0.14],
            "sharpes": [0.45, 0.55],
            "drawdowns": [-0.04, -0.06],
        },
        {
            "key": "non-shortlist",
            "universe": "full",
            "model_type": "linear_regression",
            "top_k": 1,
            "returns": [9.0, 9.0],
            "sharpes": [9.0, 9.0],
            "drawdowns": [-9.0, -9.0],
        },
    ]
    rows.append(
        _row(
            "broad-linear-train",
            folds[0],
            "train",
            "broad_macro",
            "linear_regression",
            1,
            total_return=99.0,
            sharpe_like=99.0,
            max_drawdown=-99.0,
        )
    )
    for candidate in candidates:
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
        "daily_volatility": 0.02,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "final_equity": 1.0 + total_return,
        "epochs_trained": None,
    }


def _synthetic_market_data(
    universe: str,
    *,
    include_ohlcv: bool = False,
) -> pd.DataFrame:
    close_values = {
        "broad_macro": {
            "SPY": {
                "2021-01-01": 100.0,
                "2021-06-30": 110.0,
                "2021-12-31": 120.0,
            },
            "QQQ": {
                "2021-01-01": 200.0,
                "2021-06-30": 180.0,
                "2021-12-31": 240.0,
            },
        },
        "full": {
            "SPY": {
                "2021-01-01": 100.0,
                "2021-06-30": 105.0,
                "2021-12-31": 115.0,
            },
            "QQQ": {
                "2021-01-01": 100.0,
                "2021-06-30": 100.0,
                "2021-12-31": 125.0,
            },
        },
    }
    rows: list[dict[str, object]] = []
    for symbol, prices_by_date in close_values[universe].items():
        for date, close in prices_by_date.items():
            row: dict[str, object] = {
                "date": date,
                "symbol": symbol,
                "close": close,
            }
            if include_ohlcv:
                row.update(
                    {
                        "open": close - 0.25,
                        "high": close + 0.50,
                        "low": close - 0.75,
                        "volume": 1_000_000,
                        "vwap": close - 0.05,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)
