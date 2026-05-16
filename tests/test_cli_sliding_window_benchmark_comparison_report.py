from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_benchmark_comparison_report import (
    build_sliding_window_benchmark_comparison_report,
    calculate_buy_and_hold_window_return,
    run_sliding_window_benchmark_comparison_report_from_args,
)
from us_market_ml.cli.run_sliding_window_stability_report import (
    parse_sliding_window_fold_name,
)


def test_calculate_buy_and_hold_window_return_uses_first_and_last_available_close() -> None:
    market_data = pd.DataFrame(
        [
            {"date": "2021-01-03", "symbol": "SPY", "close": 121.0},
            {"date": "2021-01-01", "symbol": "SPY", "close": 100.0},
            {"date": "2021-01-02", "symbol": "SPY", "close": 110.0},
            {"date": "2021-01-01", "symbol": "QQQ", "close": 200.0},
            {"date": "2021-01-03", "symbol": "QQQ", "close": 190.0},
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
        symbol="IWM",
        test_start="2021-01-01",
        test_end="2021-01-03",
    )

    assert result == pytest.approx(0.21)
    assert pd.isna(missing)


def test_fold_parsing_is_compatible_with_stability_report_parser() -> None:
    metadata = parse_sliding_window_fold_name("tw3y_v6m_t3m_q02_20210104_20210106")

    assert metadata.train_window_years == 3
    assert metadata.validation_window_months == 6
    assert metadata.test_window_months == 3
    assert metadata.sequence_number == 2
    assert metadata.test_start == pd.Timestamp("2021-01-04")
    assert metadata.test_end == pd.Timestamp("2021-01-06")
    assert metadata.test_year == 2021


def test_build_report_compares_strategy_test_windows_to_benchmarks(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_benchmark_comparison_report(
        results_csv=results_csv,
        market_data=_synthetic_market_data(),
        failed_keys=None,
        benchmark_symbols=("SPY", "QQQ"),
        top_n=2,
    )

    window_metrics = report.window_metrics.set_index("fold")
    first_fold = "tw2y_v6m_t3m_q01_20210101_20210103"
    second_fold = "tw2y_v6m_t3m_q02_20210104_20210106"

    assert len(window_metrics) == 4
    assert window_metrics.loc[first_fold, "strategy_total_return"] == pytest.approx(0.10)
    assert window_metrics.loc[first_fold, "spy_total_return"] == pytest.approx(0.10)
    assert window_metrics.loc[first_fold, "qqq_total_return"] == pytest.approx(-0.05)
    assert window_metrics.loc[first_fold, "strategy_minus_spy"] == pytest.approx(0.0)
    assert not bool(window_metrics.loc[first_fold, "strategy_beats_spy"])
    assert bool(window_metrics.loc[first_fold, "strategy_beats_qqq"])

    assert window_metrics.loc[second_fold, "spy_total_return"] == pytest.approx(-0.10)
    assert window_metrics.loc[second_fold, "qqq_total_return"] == pytest.approx(0.10)
    assert window_metrics.loc[second_fold, "strategy_minus_spy"] == pytest.approx(0.05)
    assert window_metrics.loc[second_fold, "strategy_minus_qqq"] == pytest.approx(-0.15)


def test_build_report_aggregates_summary_and_group_tables(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_sliding_window_benchmark_comparison_report(
        results_csv=results_csv,
        market_data=_synthetic_market_data(),
        benchmark_symbols=("SPY", "QQQ"),
    )

    summary = report.summary.iloc[0]
    assert summary["window_count"] == 4
    assert summary["strategy_mean_total_return"] == pytest.approx(0.015)
    assert summary["strategy_median_total_return"] == pytest.approx(0.005)
    assert summary["spy_mean_total_return"] == pytest.approx(0.0125)
    assert summary["spy_median_total_return"] == pytest.approx(0.025)
    assert summary["qqq_mean_total_return"] == pytest.approx(-0.0125)
    assert summary["qqq_median_total_return"] == pytest.approx(-0.025)
    assert summary["strategy_positive_window_rate"] == pytest.approx(0.5)
    assert summary["spy_positive_window_rate"] == pytest.approx(0.5)
    assert summary["qqq_positive_window_rate"] == pytest.approx(0.25)
    assert summary["strategy_beats_spy_rate"] == pytest.approx(0.25)
    assert summary["strategy_beats_qqq_rate"] == pytest.approx(0.75)
    assert summary["mean_strategy_minus_spy"] == pytest.approx(0.0025)
    assert summary["mean_strategy_minus_qqq"] == pytest.approx(0.0275)

    by_train_window = report.by_train_window.set_index("train_window_years")
    assert by_train_window.loc[2, "count"] == 2
    assert by_train_window.loc[2, "strategy_mean_total_return"] == pytest.approx(0.025)
    assert by_train_window.loc[3, "strategy_mean_total_return"] == pytest.approx(0.005)
    assert by_train_window.loc[3, "strategy_beats_qqq_rate"] == pytest.approx(1.0)

    by_year = report.by_year.set_index("test_year")
    assert by_year.loc[2021, "count"] == 2
    assert by_year.loc[2022, "count"] == 2

    assert report.relative_windows["fold"].tolist() == [
        "tw3y_v6m_t3m_q01_20220101_20220103",
        "tw3y_v6m_t3m_q02_20220104_20220106",
        "tw2y_v6m_t3m_q01_20210101_20210103",
        "tw2y_v6m_t3m_q02_20210104_20210106",
    ]


def test_run_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    market_data = tmp_path / "market.parquet"
    failed_keys = tmp_path / "failed.txt"
    output_dir = tmp_path / "reports"
    _synthetic_results().to_csv(results_csv, index=False)
    _synthetic_market_data(include_ohlcv=True).to_parquet(market_data, index=False)
    failed_keys.write_text("failed-key\n\n", encoding="utf-8")

    return_code = run_sliding_window_benchmark_comparison_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--market-data",
            str(market_data),
            "--failed-keys",
            str(failed_keys),
            "--output-dir",
            str(output_dir),
            "--benchmark-symbols",
            "SPY,QQQ",
            "--top-n",
            "2",
        ]
    )

    captured = capsys.readouterr()
    window_metrics_path = output_dir / "sliding_window_benchmark_window_metrics.csv"
    summary_path = output_dir / "sliding_window_benchmark_summary.csv"
    by_train_window_path = output_dir / "sliding_window_benchmark_by_train_window.csv"
    by_year_path = output_dir / "sliding_window_benchmark_by_year.csv"
    relative_windows_path = output_dir / "sliding_window_benchmark_relative_windows.csv"
    markdown_path = output_dir / "sliding_window_benchmark_comparison_report.md"

    assert return_code == 0
    assert "sliding-window benchmark comparison report" in captured.out
    assert window_metrics_path.exists()
    assert summary_path.exists()
    assert by_train_window_path.exists()
    assert by_year_path.exists()
    assert relative_windows_path.exists()
    assert markdown_path.exists()

    window_metrics = pd.read_csv(window_metrics_path)
    summary = pd.read_csv(summary_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert len(window_metrics) == 4
    assert summary.loc[0, "window_count"] == 4
    assert "linear_smooth10_margin002_v1" in markdown
    assert "identical rolling test windows" in markdown
    assert "Benchmark-relative stability is not confirmed" in markdown
    assert "not production-ready" in markdown
    assert "Failed keys: 1" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_specs = [
        ("tw2y_v6m_t3m_q01_20210101_20210103", 0.10, 0.50, -0.02),
        ("tw2y_v6m_t3m_q02_20210104_20210106", -0.05, -0.20, -0.08),
        ("tw3y_v6m_t3m_q01_20220101_20220103", 0.02, 0.10, -0.03),
        ("tw3y_v6m_t3m_q02_20220104_20220106", -0.01, -0.05, -0.04),
    ]
    for index, (fold, total_return, sharpe_like, max_drawdown) in enumerate(
        test_specs,
        start=1,
    ):
        for split in ["train", "validation", "test"]:
            multiplier = 1.0 if split == "test" else 10.0
            rows.append(
                {
                    "run_key": f"run-{index}",
                    "fold": fold,
                    "split": split,
                    "total_return": total_return * multiplier,
                    "sharpe_like": sharpe_like * multiplier,
                    "max_drawdown": max_drawdown * multiplier,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_market_data(*, include_ohlcv: bool = False) -> pd.DataFrame:
    close_values = {
        "SPY": {
            "2021-01-01": 100.0,
            "2021-01-02": 105.0,
            "2021-01-03": 110.0,
            "2021-01-04": 110.0,
            "2021-01-05": 104.0,
            "2021-01-06": 99.0,
            "2022-01-01": 100.0,
            "2022-01-02": 103.0,
            "2022-01-03": 105.0,
            "2022-01-04": 105.0,
            "2022-01-05": 104.0,
            "2022-01-06": 105.0,
        },
        "QQQ": {
            "2021-01-01": 200.0,
            "2021-01-02": 195.0,
            "2021-01-03": 190.0,
            "2021-01-04": 190.0,
            "2021-01-05": 200.0,
            "2021-01-06": 209.0,
            "2022-01-01": 100.0,
            "2022-01-02": 99.0,
            "2022-01-03": 100.0,
            "2022-01-04": 100.0,
            "2022-01-05": 95.0,
            "2022-01-06": 90.0,
        },
    }
    rows: list[dict[str, object]] = []
    for symbol, prices_by_date in close_values.items():
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
