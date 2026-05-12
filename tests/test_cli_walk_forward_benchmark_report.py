from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_walk_forward_benchmark_report import (
    run_walk_forward_benchmark_report_from_args,
)


def _synthetic_etf_ohlcv() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=760, freq="D")
    symbols = ("SPY", "QQQ", "TLT")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 25.0
        drift = 0.09 + symbol_index * 0.02
        cycle = symbol_index + 2
        for day_index, date in enumerate(dates):
            close = base_price + day_index * drift + (day_index % 17) * cycle * 0.03
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.25,
                    "high": close + 0.75,
                    "low": close - 0.8,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 80_000 + day_index * 1_500,
                    "vwap": close - 0.05,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )

    return pd.DataFrame(rows)


def test_run_walk_forward_benchmark_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_walk_forward_benchmark_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "wf",
            "--fold",
            "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
            "--fold",
            "fold_2:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-05-31",
            "--top-k",
            "2",
            "--hidden-units",
            "16",
            "--max-epochs",
            "5",
            "--patience",
            "2",
            "--batch-size",
            "16",
            "--transaction-cost-bps",
            "5",
        ]
    )

    captured = capsys.readouterr()
    combined_path = output_dir / "wf_combined_equity.csv"
    metrics_path = output_dir / "wf_metrics.csv"
    plot_path = output_dir / "wf_equity.png"
    fold_summary_path = output_dir / "wf_fold_summary.csv"
    selected_holdings_path = output_dir / "wf_selected_holdings.csv"
    position_contributions_path = output_dir / "wf_position_contributions.csv"
    holdings_summary_path = output_dir / "wf_holdings_summary.csv"
    contribution_summary_path = output_dir / "wf_contribution_summary.csv"
    daily_selection_summary_path = output_dir / "wf_daily_selection_summary.csv"
    selection_frequency_path = output_dir / "wf_selection_frequency.png"
    contribution_by_symbol_path = output_dir / "wf_contribution_by_symbol.png"
    monthly_selection_heatmap_path = output_dir / "wf_monthly_selection_heatmap.png"
    combined = pd.read_csv(combined_path)
    metrics = pd.read_csv(metrics_path)
    fold_summary = pd.read_csv(fold_summary_path)

    assert return_code == 0
    assert "walk-forward benchmark report" in captured.out
    assert combined_path.exists()
    assert metrics_path.exists()
    assert plot_path.exists()
    assert fold_summary_path.exists()
    assert selected_holdings_path.exists()
    assert position_contributions_path.exists()
    assert holdings_summary_path.exists()
    assert contribution_summary_path.exists()
    assert daily_selection_summary_path.exists()
    assert selection_frequency_path.exists()
    assert contribution_by_symbol_path.exists()
    assert monthly_selection_heatmap_path.exists()
    assert plot_path.stat().st_size > 0
    assert selection_frequency_path.stat().st_size > 0
    assert contribution_by_symbol_path.stat().st_size > 0
    assert monthly_selection_heatmap_path.stat().st_size > 0
    assert set(combined["series_name"]) == {"mlp_strategy", "SPY", "QQQ", "equal_weight"}
    assert set(metrics["series_name"]) == {"mlp_strategy", "SPY", "QQQ", "equal_weight"}
    assert fold_summary["fold_name"].tolist() == ["fold_1", "fold_2"]
    assert {
        "strategy_total_return",
        "strategy_max_drawdown",
        "strategy_final_equity",
        "strategy_total_transaction_cost",
        "strategy_mean_daily_turnover",
        "epochs_trained",
    }.issubset(fold_summary.columns)
    assert {
        "total_transaction_cost",
        "mean_daily_turnover",
        "max_daily_turnover",
    }.issubset(metrics.columns)
