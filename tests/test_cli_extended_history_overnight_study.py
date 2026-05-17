from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.run_extended_history_overnight_study import (
    build_extended_history_folds,
    build_extended_history_report,
    build_extended_history_sweep_config,
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
