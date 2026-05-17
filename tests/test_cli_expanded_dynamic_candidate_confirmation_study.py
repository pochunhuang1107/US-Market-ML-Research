from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli import run_expanded_dynamic_candidate_confirmation_study as study
from us_market_ml.cli.run_expanded_dynamic_candidate_confirmation_study import (
    build_expanded_dynamic_candidate_confirmation_folds,
    build_expanded_dynamic_candidate_confirmation_plan,
    run_expanded_dynamic_candidate_confirmation_study_from_args,
)
from us_market_ml.experiments import SweepFoldConfig


def test_default_expanded_window_grid_is_larger_than_smoke_study() -> None:
    fold_plans = build_expanded_dynamic_candidate_confirmation_folds()

    assert len(fold_plans) == 243
    assert len(fold_plans) * len(study.CANDIDATE_SPECS) * 2 == 1944
    assert len(fold_plans) * len(study.CANDIDATE_SPECS) * 2 > 132
    assert {fold.train_window_years for fold in fold_plans} == {3, 5, 10}
    assert {fold.validation_window_months for fold in fold_plans} == {6, 12}
    assert {fold.test_window_months for fold in fold_plans} == {3, 6, 12}


def test_expanded_dynamic_plan_uses_exact_candidate_pairs_and_dynamic_defaults(
    tmp_path: Path,
) -> None:
    plan = build_expanded_dynamic_candidate_confirmation_plan(
        market_dataset_glob="/tmp/Market Data/**/*.parquet",
        fold_plans=[_fold_plan()],
        output_dir=tmp_path,
        run_name="expanded",
        resume=True,
        max_workers=4,
    )

    pairs = [
        (config.models[0].model_type, config.signals[0].top_k)
        for config in plan.run_configs
    ]

    assert pairs == [
        ("linear_regression", 1),
        ("linear_regression", 3),
        ("ridge", 3),
        ("ridge", 5),
    ]
    assert all(config.universe_mode == "dynamic" for config in plan.run_configs)
    assert all(config.feature_selection_scope == "fold_train" for config in plan.run_configs)
    assert all(config.min_history_rows == 252 for config in plan.run_configs)
    assert all(config.target_col == "excess_return_vs_spy_20d" for config in plan.run_configs)
    assert all(config.label_horizon_days == 20 for config in plan.run_configs)
    assert all(
        config.feature_set_versions == ["default_features_v1"]
        for config in plan.run_configs
    )
    assert all(config.transaction_cost_bps_values == [10.0, 20.0] for config in plan.run_configs)
    assert {config.signals[0].allocation_mode for config in plan.run_configs} == {
        "equal_weight"
    }
    assert plan.estimated_total_configs == 8
    assert plan.min_train_eligible_rows == 5000


def test_expanded_dynamic_cli_dry_run_writes_preflight_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_expanded_dynamic_candidate_confirmation_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "expanded_dynamic_candidate_confirmation_study",
            "--data-start",
            "2020-01-01",
            "--data-end",
            "2021-09-30",
            "--train-window-years",
            "1",
            "--validation-window-months",
            "1",
            "--test-window-months",
            "3",
            "--step-months",
            "3",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "1",
        ]
    )

    captured = capsys.readouterr()
    preflight_csv = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_preflight.csv"
    )
    preflight_md = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_preflight.md"
    )
    skipped = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_study_skipped_keys.txt"
    )
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert skipped.exists()
    assert "universe_mode: dynamic" in captured.out
    assert "feature_selection_scope: fold_train" in captured.out
    assert "generated fold count: 3" in captured.out
    assert "estimated total configs: 24" in captured.out
    assert len(preflight) == 24
    assert set(zip(preflight["model_type"], preflight["top_k"], strict=True)) == {
        ("linear_regression", 1),
        ("linear_regression", 3),
        ("ridge", 3),
        ("ridge", 5),
    }
    assert set(preflight["allocation_mode"]) == {"equal_weight"}
    assert set(preflight["transaction_cost_bps"]) == {10.0, 20.0}
    assert preflight["selected_feature_count"].min() > 0
    assert preflight["valid_config"].all()
    assert set(preflight["min_train_eligible_rows"]) == {1}
    markdown = preflight_md.read_text(encoding="utf-8")
    assert markdown.startswith("# Expanded Dynamic Candidate Confirmation Preflight")
    assert "Universe mode: `dynamic`" in markdown
    assert "Feature selection scope: `fold_train`" in markdown
    assert "Minimum train eligible rows: `1`" in markdown
    assert "Invalid/Thin Fold Preview" in markdown


def test_expanded_dynamic_cli_dry_run_marks_low_row_folds_invalid_by_default(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_expanded_dynamic_candidate_confirmation_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "expanded_dynamic_candidate_confirmation_study",
            "--data-start",
            "2020-01-01",
            "--data-end",
            "2021-09-30",
            "--train-window-years",
            "1",
            "--validation-window-months",
            "1",
            "--test-window-months",
            "3",
            "--step-months",
            "3",
            "--min-history-rows",
            "5",
        ]
    )

    preflight_csv = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_preflight.csv"
    )
    preflight_md = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_preflight.md"
    )
    skipped = (
        tmp_path
        / "study"
        / "expanded_dynamic_candidate_confirmation_study_skipped_keys.txt"
    )
    preflight = pd.read_csv(preflight_csv)
    skipped_keys = [
        line
        for line in skipped.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert exit_code == 0
    assert set(preflight["min_train_eligible_rows"]) == {5000}
    assert (preflight["train_eligible_row_count"] < 5000).all()
    assert not preflight["valid_config"].any()
    assert preflight["invalid_reason"].str.contains(
        "train_eligible_row_count below min_train_eligible_rows",
        regex=False,
    ).all()
    assert len(skipped_keys) == len(preflight)

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Minimum train eligible rows: `5000`" in markdown
    assert "Total configs: `24`" in markdown
    assert "Valid configs: `0`" in markdown
    assert "Skipped configs: `24`" in markdown
    assert "Invalid/Thin Fold Preview" in markdown
    assert "train_eligible_row_count below min_train_eligible_rows" in markdown


def test_expanded_dynamic_shell_runner_has_required_overrides_and_progress() -> None:
    script_path = Path("scripts/run_expanded_dynamic_candidate_confirmation_study.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "RUN_NAME=" in script
    assert "MAX_WORKERS=" in script
    assert "RESUME=" in script
    assert "DRY_RUN=" in script
    assert "MIN_HISTORY_ROWS=" in script
    assert "MIN_TRAIN_ELIGIBLE_ROWS=" in script
    assert '--min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"' in script
    assert "STEP_MONTHS=" in script
    assert "sleep 60" in script
    assert "completed=%s skipped=%s failed=%s result_rows=%s" in script
    assert "us_market_ml.cli.run_expanded_dynamic_candidate_confirmation_study" in script


def _fold_plan() -> study.ExpandedFoldPlan:
    return study.ExpandedFoldPlan(
        fold=SweepFoldConfig(
            name="tw1y_v1m_t1m_q01_20210201_20210228",
            train_start="2020-01-01",
            train_end="2020-12-31",
            validation_start="2021-01-01",
            validation_end="2021-01-31",
            test_start="2021-02-01",
            test_end="2021-02-28",
        ),
        train_window_years=1,
        validation_window_months=1,
        test_window_months=1,
    )


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", "2021-09-30", freq="D")
    symbols = ("SPY", "QQQ", "TLT", "IWM", "GLD", "VNQ")
    for symbol_index, symbol in enumerate(symbols):
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
                    "volume": 1_000_000 + symbol_index * 10_000 + day_index,
                    "vwap": close,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
