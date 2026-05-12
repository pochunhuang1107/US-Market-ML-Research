from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_candidate_deep_dive_report import (
    run_candidate_deep_dive_report_from_args,
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


def test_run_candidate_deep_dive_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_candidate_deep_dive_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "deep_dive",
            "--fold",
            "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
            "--fold",
            "fold_2:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-05-31",
            "--candidate",
            "baseline:linear_top1:linear_regression:1:42",
            "--candidate",
            "mlp:mlp_top1:1:16:0.001:0.0:43",
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
    candidate_metrics_path = output_dir / "deep_dive_candidate_metrics.csv"
    candidate_rank_path = output_dir / "deep_dive_candidate_rank.csv"
    candidate_equity_path = output_dir / "deep_dive_candidate_equity.csv"
    candidate_equity_plot_path = output_dir / "deep_dive_candidate_equity.png"
    concentration_summary_path = output_dir / "deep_dive_concentration_summary.csv"
    fold_summary_path = output_dir / "deep_dive_fold_summary.csv"
    candidate_metrics = pd.read_csv(candidate_metrics_path)
    candidate_rank = pd.read_csv(candidate_rank_path)
    concentration_summary = pd.read_csv(concentration_summary_path)
    fold_summary = pd.read_csv(fold_summary_path)

    assert return_code == 0
    assert "candidate deep-dive report" in captured.out
    assert candidate_metrics_path.exists()
    assert candidate_rank_path.exists()
    assert candidate_equity_path.exists()
    assert candidate_equity_plot_path.exists()
    assert concentration_summary_path.exists()
    assert fold_summary_path.exists()
    assert candidate_equity_plot_path.stat().st_size > 0
    assert set(candidate_metrics["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert set(candidate_rank["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert set(candidate_rank["candidate_family"]) == {"baseline", "mlp"}
    assert set(concentration_summary["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert set(concentration_summary["candidate_family"]) == {"baseline", "mlp"}
    assert {
        "top_symbol_by_selection",
        "top_symbol_by_contribution",
        "concentration_warning",
        "contribution_warning",
        "overall_concentration_risk",
        "risk_score",
    }.issubset(concentration_summary.columns)
    assert set(concentration_summary["overall_concentration_risk"]).issubset(
        {"low", "medium", "high"}
    )
    assert set(concentration_summary["risk_score"]).issubset({1, 2, 3})
    assert set(fold_summary["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert set(fold_summary["candidate_family"]) == {"baseline", "mlp"}
    assert {
        "strategy_total_transaction_cost",
        "strategy_mean_daily_turnover",
    }.issubset(fold_summary.columns)
    assert {
        "total_transaction_cost",
        "mean_daily_turnover",
        "max_daily_turnover",
    }.issubset(candidate_metrics.columns)
    assert fold_summary.loc[
        fold_summary["candidate_family"] == "baseline", "epochs_trained"
    ].isna().all()

    for candidate_name in ("linear_top1", "mlp_top1"):
        candidate_dir = output_dir / candidate_name
        combined_equity_path = candidate_dir / f"{candidate_name}_combined_equity.csv"
        metrics_path = candidate_dir / f"{candidate_name}_metrics.csv"
        selected_holdings_path = candidate_dir / f"{candidate_name}_selected_holdings.csv"
        position_contributions_path = (
            candidate_dir / f"{candidate_name}_position_contributions.csv"
        )
        holdings_summary_path = candidate_dir / f"{candidate_name}_holdings_summary.csv"
        contribution_summary_path = candidate_dir / f"{candidate_name}_contribution_summary.csv"
        concentration_report_path = candidate_dir / f"{candidate_name}_concentration_report.csv"
        daily_selection_summary_path = (
            candidate_dir / f"{candidate_name}_daily_selection_summary.csv"
        )
        equity_path = candidate_dir / f"{candidate_name}_equity.png"
        selection_frequency_path = candidate_dir / f"{candidate_name}_selection_frequency.png"
        contribution_by_symbol_path = (
            candidate_dir / f"{candidate_name}_contribution_by_symbol.png"
        )
        monthly_selection_heatmap_path = (
            candidate_dir / f"{candidate_name}_monthly_selection_heatmap.png"
        )

        assert candidate_dir.exists()
        assert combined_equity_path.exists()
        assert metrics_path.exists()
        assert selected_holdings_path.exists()
        assert position_contributions_path.exists()
        assert holdings_summary_path.exists()
        assert contribution_summary_path.exists()
        assert concentration_report_path.exists()
        assert daily_selection_summary_path.exists()
        assert equity_path.exists()
        assert selection_frequency_path.exists()
        assert contribution_by_symbol_path.exists()
        assert monthly_selection_heatmap_path.exists()
        assert equity_path.stat().st_size > 0
        assert selection_frequency_path.stat().st_size > 0
        assert contribution_by_symbol_path.stat().st_size > 0
        assert monthly_selection_heatmap_path.stat().st_size > 0
