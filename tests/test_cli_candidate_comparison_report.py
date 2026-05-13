from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_candidate_comparison_report import (
    CandidateConfig,
    _build_smoothing_config,
    _parse_args,
    parse_candidate_config,
    run_candidate_comparison_report_from_args,
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


def test_parse_candidate_config_new_mlp_format() -> None:
    candidate = parse_candidate_config("mlp:broad_top3:3:256,128:0.0005:0.2:42")

    assert candidate == CandidateConfig(
        family="mlp",
        name="broad_top3",
        top_k=3,
        random_state=42,
        hidden_units=(256, 128),
        learning_rate=0.0005,
        dropout=0.2,
    )


def test_parse_candidate_config_baseline_format() -> None:
    candidate = parse_candidate_config("baseline:linear_full_top1:linear_regression:1:42")

    assert candidate == CandidateConfig(
        family="baseline",
        name="linear_full_top1",
        top_k=1,
        random_state=42,
        model_type="linear_regression",
    )


def test_parse_candidate_config_extended_baseline_format() -> None:
    candidate = parse_candidate_config(
        "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002"
    )

    assert candidate == CandidateConfig(
        family="baseline",
        name="linear_smooth10_margin002",
        top_k=1,
        random_state=42,
        model_type="linear_regression",
        min_holding_days=10,
        switch_margin=0.002,
    )


def test_parse_candidate_config_extended_mlp_format() -> None:
    candidate = parse_candidate_config(
        "mlp:broad_smooth10_margin002:3:256,128:0.0005:0.2:42:10:0.002"
    )

    assert candidate == CandidateConfig(
        family="mlp",
        name="broad_smooth10_margin002",
        top_k=3,
        random_state=42,
        hidden_units=(256, 128),
        learning_rate=0.0005,
        dropout=0.2,
        min_holding_days=10,
        switch_margin=0.002,
    )


def test_parse_candidate_config_legacy_mlp_format() -> None:
    candidate = parse_candidate_config("broad_top3:3:256,128:0.0005:0.2:42")

    assert candidate == CandidateConfig(
        family="mlp",
        name="broad_top3",
        top_k=3,
        random_state=42,
        hidden_units=(256, 128),
        learning_rate=0.0005,
        dropout=0.2,
    )


@pytest.mark.parametrize(
    "value,match",
    [
        ("missing:parts", "Candidate family must be one"),
        ("cnn:bad_family:1:16:0.001:0.0:42", "Candidate family must be one"),
        ("baseline:bad_family:ridge:1:42:extra", "Invalid candidate value"),
        ("bad_top_k:0:16:0.001:0.0:42", "top_k must be positive"),
        ("bad_hidden:1:16,0:0.001:0.0:42", "hidden units must be positive"),
        ("bad_lr:1:16:0.0:0.0:42", "learning_rate must be positive"),
        ("bad_dropout:1:16:0.001:1.0:42", "dropout must satisfy"),
        ("baseline:bad_top_k:ridge:0:42", "top_k must be positive"),
        ("baseline:bad_min_holding_days:ridge:1:42:0:0.0", "min_holding_days"),
        ("baseline:bad_switch_margin:ridge:1:42:1:-0.001", "switch_margin"),
        ("mlp:bad_min_holding_days:1:16:0.001:0.0:42:0:0.0", "min_holding_days"),
        ("mlp:bad_switch_margin:1:16:0.001:0.0:42:1:-0.001", "switch_margin"),
    ],
)
def test_parse_candidate_config_validates(value: str, match: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        parse_candidate_config(value)


def test_candidate_specific_smoothing_overrides_global_smoothing() -> None:
    global_args = argparse.Namespace(min_holding_days=10, switch_margin=0.002)
    unsmoothed_candidate = parse_candidate_config(
        "baseline:linear_unsmoothed_top1:linear_regression:1:42:1:0.0"
    )
    smoothed_candidate = parse_candidate_config(
        "baseline:linear_smooth10_margin002:linear_regression:1:42:5:0.001"
    )
    fallback_candidate = parse_candidate_config(
        "baseline:linear_global_smoothing:linear_regression:1:42"
    )

    assert _build_smoothing_config(unsmoothed_candidate, global_args) is None

    candidate_smoothing = _build_smoothing_config(smoothed_candidate, global_args)
    assert candidate_smoothing is not None
    assert candidate_smoothing.min_holding_days == 5
    assert candidate_smoothing.switch_margin == pytest.approx(0.001)

    fallback_smoothing = _build_smoothing_config(fallback_candidate, global_args)
    assert fallback_smoothing is not None
    assert fallback_smoothing.min_holding_days == 10
    assert fallback_smoothing.switch_margin == pytest.approx(0.002)


def test_duplicate_candidate_names_fail(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"

    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--data-path",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--fold",
                "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
                "--candidate",
                "mlp:candidate:1:16:0.001:0.0:42",
                "--candidate",
                "baseline:candidate:linear_regression:2:43",
            ]
        )


def test_run_candidate_comparison_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_candidate_comparison_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "candidates",
            "--fold",
            "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
            "--fold",
            "fold_2:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-05-31",
            "--candidate",
            "mlp:mlp_top1:1:16:0.001:0.0:42",
            "--candidate",
            "baseline:linear_top1:linear_regression:1:42",
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
    combined_path = output_dir / "candidates_combined_equity.csv"
    metrics_path = output_dir / "candidates_metrics.csv"
    candidate_rank_path = output_dir / "candidates_candidate_rank.csv"
    fold_summary_path = output_dir / "candidates_fold_summary.csv"
    plot_path = output_dir / "candidates_equity.png"
    combined = pd.read_csv(combined_path)
    metrics = pd.read_csv(metrics_path)
    candidate_rank = pd.read_csv(candidate_rank_path)
    fold_summary = pd.read_csv(fold_summary_path)

    assert return_code == 0
    assert "candidate comparison report" in captured.out
    assert combined_path.exists()
    assert metrics_path.exists()
    assert candidate_rank_path.exists()
    assert fold_summary_path.exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert set(combined["series_name"]) == {"mlp_top1", "linear_top1", "SPY", "QQQ", "equal_weight"}
    assert set(metrics["series_name"]) == {"mlp_top1", "linear_top1", "SPY", "QQQ", "equal_weight"}
    assert set(candidate_rank["candidate_name"]) == {"mlp_top1", "linear_top1"}
    assert {
        "candidate_name",
        "candidate_family",
        "model_type",
        "min_holding_days",
        "switch_margin",
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
        "rank_by_total_return",
        "rank_by_sharpe_like",
        "rank_by_max_drawdown",
    }.issubset(candidate_rank.columns)
    assert set(candidate_rank["candidate_family"]) == {"mlp", "baseline"}
    assert set(fold_summary["candidate_name"]) == {"mlp_top1", "linear_top1"}
    assert {
        "candidate_name",
        "candidate_family",
        "model_type",
        "fold_name",
        "test_start",
        "test_end",
        "selected_top_k",
        "hidden_units",
        "learning_rate",
        "dropout",
        "random_state",
        "min_holding_days",
        "switch_margin",
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
    assert {
        "total_transaction_cost",
        "mean_daily_turnover",
        "max_daily_turnover",
    }.issubset(candidate_rank.columns)
    assert candidate_rank["total_transaction_cost"].notna().all()
    assert set(fold_summary["candidate_family"]) == {"mlp", "baseline"}
    assert fold_summary.loc[
        fold_summary["candidate_family"] == "baseline", "epochs_trained"
    ].isna().all()
    assert fold_summary["fold_name"].tolist() == ["fold_1", "fold_2", "fold_1", "fold_2"]


def test_candidate_comparison_report_compares_unsmoothed_and_smoothed_baselines(
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_candidate_comparison_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "baseline_smoothing_candidates",
            "--fold",
            "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
            "--candidate",
            "baseline:linear_unsmoothed_top1:linear_regression:1:42:1:0.0",
            "--candidate",
            "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
        ]
    )

    candidate_rank = pd.read_csv(
        output_dir / "baseline_smoothing_candidates_candidate_rank.csv"
    ).sort_values("candidate_name")
    fold_summary = pd.read_csv(
        output_dir / "baseline_smoothing_candidates_fold_summary.csv"
    ).sort_values("candidate_name")

    assert return_code == 0
    assert candidate_rank["candidate_name"].tolist() == [
        "linear_smooth10_margin002",
        "linear_unsmoothed_top1",
    ]
    assert candidate_rank["min_holding_days"].tolist() == [10, 1]
    assert candidate_rank["switch_margin"].tolist() == pytest.approx([0.002, 0.0])
    assert fold_summary["min_holding_days"].tolist() == [10, 1]
    assert fold_summary["switch_margin"].tolist() == pytest.approx([0.002, 0.0])
