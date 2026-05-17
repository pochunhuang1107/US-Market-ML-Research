from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import us_market_ml.cli.run_extended_history_overnight_study as extended_cli
from us_market_ml.cli.run_extended_history_overnight_study import (
    ExtendedFold,
    build_extended_history_folds,
    build_extended_history_report,
    build_extended_history_sweep_config,
    run_extended_history_overnight_study_from_args,
)
from us_market_ml.experiments import (
    SweepFoldConfig,
)


def test_extended_history_config_uses_dataset_glob_and_focused_allocations(tmp_path: Path) -> None:
    folds = build_extended_history_folds("1993-01-29", "2026-05-15")
    config = build_extended_history_sweep_config(
        market_dataset_glob="/tmp/Market Data/**/*.parquet",
        folds=[item.fold for item in folds],
        output_dir=tmp_path,
        run_name="extended",
        resume=True,
        max_workers=4,
    )

    allocation_modes = {(signal.top_k, signal.allocation_mode) for signal in config.signals}

    assert str(config.universes[0].data_path) == "/tmp/Market Data/**/*.parquet"
    assert config.models[0].model_family == "baseline"
    assert config.models[0].model_type == "linear_regression"
    assert config.max_workers == 4
    assert config.transaction_cost_bps_values == [10.0, 20.0]
    assert (1, "equal_weight") in allocation_modes
    assert (3, "equal_weight") in allocation_modes
    assert (5, "equal_weight") in allocation_modes
    assert (3, "capped_score_weighted") in allocation_modes
    assert (5, "softmax_score_weighted") in allocation_modes
    assert any(item.regime == "dot_com" for item in folds)
    assert any(item.study_design == "annual_longer_history" for item in folds)


def test_extended_history_report_writes_rank_and_regime_outputs(tmp_path: Path) -> None:
    results_csv = tmp_path / "extended_results.csv"
    completed = tmp_path / "extended_completed_keys.txt"
    failed = tmp_path / "extended_failed_keys.txt"
    fold_catalog = tmp_path / "extended_history_fold_catalog.csv"
    completed.write_text("one\ntwo\n", encoding="utf-8")
    failed.write_text("", encoding="utf-8")
    fold_catalog.write_text(
        "\n".join(
            [
                "fold,study_design,regime,train_start,train_end,validation_start,"
                "validation_end,test_start,test_end",
                "regime_dot_com,fixed_regime,dot_com,1993-01-29,1998-12-31,"
                "1999-01-01,1999-12-31,2000-01-01,2002-12-31",
                "regime_gfc,fixed_regime,gfc,1999-01-01,2005-12-31,"
                "2006-01-01,2006-12-31,2007-01-01,2009-12-31",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _result_row("regime_dot_com", 1, "equal_weight", 1.0, 1.0, 10.0, 0.10, -0.20),
            _result_row("regime_gfc", 1, "equal_weight", 1.0, 1.0, 10.0, -0.05, -0.30),
            _result_row("regime_dot_com", 3, "equal_weight", 1.0, 1.0, 10.0, 0.08, -0.12),
            _result_row("regime_gfc", 3, "equal_weight", 1.0, 1.0, 10.0, 0.02, -0.18),
        ]
    ).to_csv(results_csv, index=False)

    outputs = build_extended_history_report(
        results_csv=results_csv,
        completed_keys=completed,
        failed_keys=failed,
        fold_catalog=fold_catalog,
        output_dir=tmp_path,
    )

    ranking = pd.read_csv(outputs["rank CSV"])
    regime = pd.read_csv(outputs["regime CSV"])

    assert {"candidate_spec", "allocation_label", "total_return", "mean_daily_turnover"}.issubset(
        ranking.columns
    )
    assert not regime.empty
    assert outputs["markdown report"].read_text(encoding="utf-8").startswith(
        "# Extended-History Overnight Study Report"
    )


def test_extended_history_cli_dry_run_includes_skipped_estimate_and_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    folds = [
        ExtendedFold(
            fold=SweepFoldConfig(
                name="valid_fold",
                train_start="2020-06-01",
                train_end="2020-10-31",
                validation_start="2020-11-01",
                validation_end="2020-11-30",
                test_start="2020-12-01",
                test_end="2020-12-31",
            ),
            study_design="test",
            regime="ordinary",
        ),
        ExtendedFold(
            fold=SweepFoldConfig(
                name="invalid_empty_fold",
                train_start="1999-01-01",
                train_end="1999-12-31",
                validation_start="2000-01-01",
                validation_end="2000-01-31",
                test_start="2000-02-01",
                test_end="2000-02-29",
            ),
            study_design="test",
            regime="dot_com",
        ),
    ]
    monkeypatch.setattr(extended_cli, "build_extended_history_folds", lambda *_: folds)

    exit_code = run_extended_history_overnight_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "extended_history_overnight_study",
        ]
    )

    captured = capsys.readouterr()
    report = tmp_path / "study" / "extended_history_invalid_fold_report.csv"
    skipped = tmp_path / "study" / "extended_history_overnight_study_skipped_keys.txt"
    invalid_report = pd.read_csv(report)

    assert exit_code == 0
    assert "estimated independent configs: 36" in captured.out
    assert "estimated valid configs: 18" in captured.out
    assert "estimated skipped invalid configs: 18" in captured.out
    assert "fold count: 2" in captured.out
    assert "signal count: 9" in captured.out
    assert f"skipped checkpoint: {skipped}" in captured.out
    assert f"invalid fold report: {report}" in captured.out
    assert len(skipped.read_text(encoding="utf-8").splitlines()) == 18
    assert set(invalid_report["reason"]) == {
        "skipped_invalid_fold: insufficient_usable_train_rows"
    }


def test_extended_history_shell_progress_includes_skipped_and_total() -> None:
    script = Path("scripts/run_extended_history_overnight_study.sh").read_text(encoding="utf-8")

    assert "completed=%s skipped=%s failed=%s result_rows=%s total=%s" in script
    assert 'SKIPPED_KEYS="$OUTPUT_DIR/${RUN_NAME}_skipped_keys.txt"' in script


def _result_row(
    fold: str,
    top_k: int,
    allocation_mode: str,
    score_temperature: float,
    max_weight: float,
    cost_bps: float,
    total_return: float,
    max_drawdown: float,
) -> dict[str, object]:
    return {
        "run_key": f"{fold}-{top_k}-{allocation_mode}",
        "universe": "extended_full",
        "data_path": "/tmp/Market Data/**/*.parquet",
        "fold": fold,
        "split": "test",
        "model_family": "baseline",
        "model_type": "linear_regression",
        "feature_set_version": "default_features_v1",
        "feature_count": 10,
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
        "allocation_mode": allocation_mode,
        "score_temperature": score_temperature,
        "max_weight": max_weight,
        "transaction_cost_bps": cost_bps,
        "target_col": "excess_return_vs_spy_20d",
        "label_horizon_days": 20,
        "total_return": total_return,
        "mean_daily_return": total_return / 252,
        "daily_volatility": 0.01,
        "sharpe_like": total_return,
        "max_drawdown": max_drawdown,
        "final_equity": 1.0 + total_return,
        "total_transaction_cost": 0.01,
        "mean_daily_turnover": 0.10,
        "max_daily_turnover": 1.0,
        "epochs_trained": None,
    }


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    for symbol_index, symbol in enumerate(("SPY", "QQQ", "TLT")):
        base_price = 100.0 + symbol_index * 10.0
        for day_index, date in enumerate(dates):
            close = base_price + day_index * (0.1 + symbol_index * 0.02)
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
