from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_shortlist_benchmark_report import (
    build_sliding_window_shortlist_benchmark_report,
    calculate_buy_and_hold_window_return,
    run_sliding_window_shortlist_benchmark_report_from_args,
)


def test_calculate_buy_and_hold_window_return_handles_missing_symbol() -> None:
    market_data = pd.DataFrame(
        [
            {"date": "2021-01-03", "symbol": "SPY", "close": 121.0},
            {"date": "2021-01-01", "symbol": "SPY", "close": 100.0},
            {"date": "2021-01-02", "symbol": "SPY", "close": 110.0},
        ]
    )

    result = calculate_buy_and_hold_window_return(
        market_data,
        symbol="SPY",
        test_start="2021-01-01",
        test_end="2021-01-03",
    )
    missing = calculate_buy_and_hold_window_return(
        market_data,
        symbol="QQQ",
        test_start="2021-01-01",
        test_end="2021-01-03",
    )

    assert result == pytest.approx(0.21)
    assert pd.isna(missing)


def test_shortlist_filtering_uses_test_rows_and_prespecified_candidates(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_shortlist_benchmark_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    assert report.result_row_count == 9
    assert report.test_row_count == 6
    assert report.shortlist_test_row_count == 4
    assert set(report.window_metrics["candidate_id"]) == {
        "broad_macro:linear_regression:top1:equal_weight:cost10",
        "full:ridge:top3:equal_weight:cost10",
    }
    assert not report.window_metrics["strategy_total_return"].eq(9.0).any()


def test_candidate_window_rows_merge_with_matching_universe_benchmarks(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_shortlist_benchmark_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    metrics = report.window_metrics.set_index(["candidate_id", "fold"])
    fold = "tw2y_v6m_t3m_q01_20210101_20210103"
    broad_key = ("broad_macro:linear_regression:top1:equal_weight:cost10", fold)
    full_key = ("full:ridge:top3:equal_weight:cost10", fold)

    assert metrics.loc[broad_key, "strategy_total_return"] == pytest.approx(0.12)
    assert metrics.loc[broad_key, "spy_total_return"] == pytest.approx(0.10)
    assert metrics.loc[broad_key, "qqq_total_return"] == pytest.approx(-0.05)
    assert metrics.loc[broad_key, "strategy_minus_spy"] == pytest.approx(0.02)
    assert bool(metrics.loc[broad_key, "strategy_beats_spy"])
    assert bool(metrics.loc[broad_key, "strategy_beats_qqq"])

    assert metrics.loc[full_key, "spy_total_return"] == pytest.approx(0.20)
    assert metrics.loc[full_key, "qqq_total_return"] == pytest.approx(0.10)
    assert metrics.loc[full_key, "strategy_minus_spy"] == pytest.approx(-0.15)
    assert not bool(metrics.loc[full_key, "strategy_beats_spy"])
    assert not bool(metrics.loc[full_key, "strategy_beats_qqq"])


def test_candidate_summary_aggregates_shortlist_windows(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_shortlist_benchmark_report(
        results_csv=results_csv,
        full_market_data=_synthetic_market_data("full"),
        broad_macro_market_data=_synthetic_market_data("broad_macro"),
    )

    summary = report.candidate_summary.set_index("candidate_id")
    broad = summary.loc["broad_macro:linear_regression:top1:equal_weight:cost10"]
    full = summary.loc["full:ridge:top3:equal_weight:cost10"]

    assert broad["window_count"] == 2
    assert broad["mean_total_return"] == pytest.approx(0.05)
    assert broad["median_total_return"] == pytest.approx(0.05)
    assert broad["positive_window_rate"] == pytest.approx(0.5)
    assert broad["mean_sharpe_like"] == pytest.approx(0.15)
    assert broad["mean_max_drawdown"] == pytest.approx(-0.075)
    assert broad["worst_max_drawdown"] == pytest.approx(-0.12)
    assert broad["spy_mean_total_return"] == pytest.approx(0.0)
    assert broad["qqq_mean_total_return"] == pytest.approx(0.025)
    assert broad["strategy_beats_spy_rate"] == pytest.approx(1.0)
    assert broad["strategy_beats_qqq_rate"] == pytest.approx(0.5)
    assert broad["mean_strategy_minus_spy"] == pytest.approx(0.05)
    assert broad["mean_strategy_minus_qqq"] == pytest.approx(0.025)

    assert full["window_count"] == 2
    assert full["mean_total_return"] == pytest.approx(0.065)
    assert report.benchmark_summary.loc[0, "candidate_id"] == (
        "broad_macro:linear_regression:top1:equal_weight:cost10"
    )

    by_year = report.by_year.set_index(["candidate_id", "test_year"])
    assert by_year.loc[
        ("broad_macro:linear_regression:top1:equal_weight:cost10", 2021),
        "window_count",
    ] == 2


def test_run_shortlist_report_from_args_writes_outputs(
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
    failed_keys.write_text("failed-key\n\n", encoding="utf-8")

    return_code = run_sliding_window_shortlist_benchmark_report_from_args(
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
    assert "sliding-window shortlist benchmark report" in captured.out

    expected_files = [
        "sliding_window_shortlist_window_metrics.csv",
        "sliding_window_shortlist_candidate_summary.csv",
        "sliding_window_shortlist_benchmark_summary.csv",
        "sliding_window_shortlist_by_year.csv",
        "sliding_window_shortlist_relative_windows.csv",
        "sliding_window_shortlist_benchmark_report.md",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    window_metrics = pd.read_csv(
        output_dir / "sliding_window_shortlist_window_metrics.csv"
    )
    candidate_summary = pd.read_csv(
        output_dir / "sliding_window_shortlist_candidate_summary.csv"
    )
    benchmark_summary = pd.read_csv(
        output_dir / "sliding_window_shortlist_benchmark_summary.csv"
    )
    markdown = (output_dir / "sliding_window_shortlist_benchmark_report.md").read_text(
        encoding="utf-8"
    )

    assert len(window_metrics) == 4
    assert len(candidate_summary) == 2
    assert len(benchmark_summary) == 2
    assert "Shortlist Candidates" in markdown
    assert "Benchmark-Relative Ranking" in markdown
    assert "Failed keys: 1" in markdown
    assert "Do not call any shortlist candidate production-ready" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    folds = [
        "tw2y_v6m_t3m_q01_20210101_20210103",
        "tw2y_v6m_t3m_q02_20210104_20210106",
    ]
    candidates = [
        {
            "key": "broad-linear",
            "universe": "broad_macro",
            "model_type": "linear_regression",
            "top_k": 1,
            "returns": [0.12, -0.02],
            "sharpes": [0.40, -0.10],
            "drawdowns": [-0.03, -0.12],
        },
        {
            "key": "full-ridge",
            "universe": "full",
            "model_type": "ridge",
            "top_k": 3,
            "returns": [0.05, 0.08],
            "sharpes": [0.20, 0.30],
            "drawdowns": [-0.06, -0.04],
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
    for candidate in candidates:
        rows.append(
            _row(
                f"{candidate['key']}-train",
                folds[0],
                "train",
                candidate["universe"],
                candidate["model_type"],
                candidate["top_k"],
                total_return=99.0,
                sharpe_like=99.0,
                max_drawdown=-99.0,
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
                "2021-01-02": 105.0,
                "2021-01-03": 110.0,
                "2021-01-04": 100.0,
                "2021-01-05": 95.0,
                "2021-01-06": 90.0,
            },
            "QQQ": {
                "2021-01-01": 200.0,
                "2021-01-02": 195.0,
                "2021-01-03": 190.0,
                "2021-01-04": 190.0,
                "2021-01-05": 200.0,
                "2021-01-06": 209.0,
            },
        },
        "full": {
            "SPY": {
                "2021-01-01": 100.0,
                "2021-01-02": 110.0,
                "2021-01-03": 120.0,
                "2021-01-04": 100.0,
                "2021-01-05": 95.0,
                "2021-01-06": 90.0,
            },
            "QQQ": {
                "2021-01-01": 100.0,
                "2021-01-02": 105.0,
                "2021-01-03": 110.0,
                "2021-01-04": 100.0,
                "2021-01-05": 100.0,
                "2021-01-06": 100.0,
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
