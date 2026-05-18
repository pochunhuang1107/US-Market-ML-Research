from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_tiingo_selected_candidate_replay_artifacts import (
    OUTPUT_PATH_NAMES,
    build_tiingo_selected_candidate_replay_artifacts,
    run_tiingo_selected_candidate_replay_artifacts_from_args,
)


def test_selected_candidate_replay_builds_available_artifacts(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)

    report = build_tiingo_selected_candidate_replay_artifacts(
        results_csv=paths["results"],
        selected_summary_csv=paths["summary"],
        selected_comparison_csv=paths["comparison"],
        market_dataset_glob=paths["market_glob"],
        benchmark_symbols=("SPY", "QQQ"),
    )

    summary = report.replay_summary.iloc[0]

    assert summary["candidate_label"] == "ridge / top5 / hold 20 / margin 0.005"
    assert summary["fold_count"] == 2
    assert summary["decision"] == "keep as selected candidate but not target"
    assert summary["next_action"] == "richer replay run"
    assert set(report.fold_returns["series_role"]) == {
        "primary_candidate",
        "return_reference",
        "drawdown_guardrail",
        "benchmark_reference",
    }
    assert set(
        report.fold_returns.loc[
            report.fold_returns["series_role"].eq("benchmark_reference"),
            "benchmark_symbol",
        ]
    ) == {"SPY", "QQQ"}
    assert set(report.benchmark_comparison["benchmark_symbol"].dropna()) >= {"SPY", "QQQ"}
    assert not report.underperformance_windows.empty
    assert set(report.underperformance_windows["benchmark_symbol"]) == {"SPY", "QQQ"}
    assert report.underperformance_windows["underperformed_benchmark"].any()
    assert set(report.holdings_summary["reconstruction_level"]) == {"not_available"}
    assert report.unavailable_artifacts
    assert "fold_terminal_only_not_daily_equity" in set(
        report.equity_curves["reconstruction_level"]
    )
    assert "daily_benchmark_from_market_data" in set(
        report.equity_curves["reconstruction_level"]
    )


def test_selected_candidate_replay_cli_writes_outputs_and_report(
    tmp_path: Path,
    capsys,
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "replay"

    return_code = run_tiingo_selected_candidate_replay_artifacts_from_args(
        [
            "--results-csv",
            str(paths["results"]),
            "--selected-summary-csv",
            str(paths["summary"]),
            "--selected-comparison-csv",
            str(paths["comparison"]),
            "--market-dataset-glob",
            str(paths["market_glob"]),
            "--benchmark-symbols",
            "SPY,QQQ",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert "Tiingo selected-candidate replay artifacts" in captured.out
    for filename in OUTPUT_PATH_NAMES:
        assert (output_dir / filename).exists()

    benchmark_comparison = pd.read_csv(
        output_dir / "selected_candidate_benchmark_comparison.csv"
    )
    underperformance = pd.read_csv(
        output_dir / "selected_candidate_underperformance_windows.csv"
    )
    markdown = (output_dir / "selected_candidate_replay_report.md").read_text(
        encoding="utf-8"
    )

    assert {"SPY", "QQQ"}.issubset(set(benchmark_comparison["benchmark_symbol"].dropna()))
    assert not underperformance.empty
    assert "Decision:" in markdown
    assert "keep as selected candidate but not target" in markdown
    assert "richer replay run" in markdown
    assert "Daily selected ETF holdings cannot be reconstructed" in markdown
    assert "QQQ raw-return note" in markdown
    assert "not production-ready" in markdown


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path | str]:
    results = tmp_path / "focused_results.csv"
    summary = tmp_path / "selected_summary.csv"
    comparison = tmp_path / "selected_comparison.csv"
    market_glob = _write_synthetic_market_data(tmp_path)

    _synthetic_results().to_csv(results, index=False)
    _synthetic_selected_summary().to_csv(summary, index=False)
    _synthetic_selected_comparison().to_csv(comparison, index=False)

    return {
        "results": results,
        "summary": summary,
        "comparison": comparison,
        "market_glob": market_glob,
    }


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    folds = [
        ("tw3y_v12m_t3m_q01_20210101_20210331", 0.04),
        ("tw5y_v12m_t6m_q02_20210401_20210930", 0.06),
    ]
    specs = [
        ("ridge", 5, 20, 0.005, "primary", 0.0, -0.12, 0.03),
        ("ridge", 1, 20, 0.005, "return", 0.01, -0.18, 0.05),
        ("ridge", 5, 5, 0.002, "guardrail", -0.005, -0.10, 0.12),
    ]
    for model_type, top_k, hold_days, margin, role, adjustment, drawdown, turnover in specs:
        for index, (fold, base_return) in enumerate(folds, start=1):
            total_return = base_return + adjustment
            rows.append(
                {
                    "run_key": f"{role}-{fold}",
                    "universe": "tiingo_focused_linear_ridge_dynamic",
                    "data_path": "/tmp/fake.parquet",
                    "sweep_profile": "focused_linear_ridge_confirmation",
                    "fold": fold,
                    "split": "test",
                    "model_family": "baseline",
                    "model_type": model_type,
                    "feature_set_version": "default_features_v1",
                    "feature_count": 3,
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
                    "min_holding_days": hold_days,
                    "switch_margin": margin,
                    "allocation_mode": "equal_weight",
                    "score_temperature": 1.0,
                    "max_weight": 1.0,
                    "transaction_cost_bps": 10.0,
                    "target_col": "excess_return_vs_spy_20d",
                    "label_horizon_days": 20,
                    "total_return": total_return,
                    "mean_daily_return": total_return / 63,
                    "daily_volatility": 0.01,
                    "sharpe_like": 0.05 + total_return,
                    "max_drawdown": drawdown - index * 0.01,
                    "final_equity": 1.0 + total_return,
                    "total_transaction_cost": 0.002 + turnover / 100,
                    "mean_daily_turnover": turnover,
                    "max_daily_turnover": 0.50,
                    "epochs_trained": "",
                }
            )
    rows.append({**rows[0], "split": "train", "run_key": "ignored-train"})
    rows.append(
        {
            **rows[0],
            "model_family": "mlp",
            "model_type": "mlp",
            "run_key": "ignored-mlp",
        }
    )
    return pd.DataFrame(rows)


def _synthetic_selected_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_role": "primary_balanced_candidate",
                "candidate_label": "ridge / top5 / hold 20 / margin 0.005",
                "recommendation": "replay selected candidate with richer artifacts",
                "next_step": "Replay selected candidate with richer artifacts.",
            }
        ]
    )


def _synthetic_selected_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _comparison_row(
                "primary_vs_return_reference",
                "ridge / top1 / hold 20 / margin 0.005",
                0.06,
                -0.01,
            ),
            _comparison_row(
                "primary_vs_drawdown_guardrail",
                "ridge / top5 / hold 5 / margin 0.002",
                0.045,
                0.005,
            ),
            _comparison_row(
                "primary_vs_historical_champion",
                "linear_regression / top1 / hold 10 / margin 0.002",
                0.03,
                0.02,
            ),
            _comparison_row("primary_vs_spy_buy_hold", "SPY buy-and-hold", 0.025, 0.025),
            _comparison_row("primary_vs_qqq_buy_hold", "QQQ buy-and-hold", 0.12, -0.07),
        ]
    )


def _comparison_row(
    name: str,
    comparison_label: str,
    comparison_return: float,
    delta_return: float,
) -> dict[str, object]:
    return {
        "comparison_name": name,
        "primary_candidate_label": "ridge / top5 / hold 20 / margin 0.005",
        "comparison_candidate_label": comparison_label,
        "comparison_average_test_return_10bps": comparison_return,
        "delta_average_test_return_10bps": delta_return,
        "interpretation": "synthetic comparison",
    }


def _write_synthetic_market_data(tmp_path: Path) -> str:
    market_dir = tmp_path / "market"
    dates = pd.bdate_range("2021-01-01", "2021-09-30")
    for symbol, daily_return in [("SPY", 0.00020), ("QQQ", 0.00150)]:
        price = 100.0
        rows: list[dict[str, object]] = []
        for date in dates:
            price *= 1.0 + daily_return
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                    "vwap": price,
                    "adj_close": price,
                }
            )
        path = market_dir / f"symbol={symbol}" / "prices.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    return str(market_dir / "**" / "*.csv")
