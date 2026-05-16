from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_champion_study import (
    CHAMPION_NAME,
    build_sliding_window_champion_config,
    build_sliding_window_champion_folds,
    format_fold_arg,
    run_sliding_window_champion_study_from_args,
)
from us_market_ml.splits import make_sliding_window_folds


def test_sliding_window_fold_generation_is_deterministic_and_named() -> None:
    folds = make_sliding_window_folds(
        data_start="2018-01-01",
        data_end="2021-05-08",
        train_window_years=2,
        validation_window_months=6,
        test_window_months=3,
        step_months=3,
    )

    assert [fold.name for fold in folds] == [
        "tw2y_v6m_t3m_q01_20200701_20200930",
        "tw2y_v6m_t3m_q02_20201001_20201231",
        "tw2y_v6m_t3m_q03_20210101_20210331",
        "tw2y_v6m_t3m_q04_20210401_20210508",
    ]
    first = folds[0]
    assert first.train_start == pd.Timestamp("2018-01-01")
    assert first.train_end == pd.Timestamp("2019-12-31")
    assert first.validation_start == pd.Timestamp("2020-01-01")
    assert first.validation_end == pd.Timestamp("2020-06-30")
    assert first.test_start == pd.Timestamp("2020-07-01")
    assert first.test_end == pd.Timestamp("2020-09-30")


def test_sliding_window_folds_do_not_overlap_within_each_fold() -> None:
    folds = make_sliding_window_folds(
        data_start="2018-01-01",
        data_end="2026-05-08",
        train_window_years=3,
        validation_window_months=6,
        test_window_months=3,
        step_months=3,
    )

    assert folds
    for fold in folds:
        assert fold.train_start <= fold.train_end
        assert fold.train_end < fold.validation_start
        assert fold.validation_start <= fold.validation_end
        assert fold.validation_end < fold.test_start
        assert fold.test_start <= fold.test_end
        assert fold.test_end <= pd.Timestamp("2026-05-08")


def test_sliding_window_test_periods_move_forward_monotonically() -> None:
    folds = make_sliding_window_folds(
        data_start="2018-01-01",
        data_end="2023-12-31",
        train_window_years=5,
        validation_window_months=6,
        test_window_months=3,
        step_months=3,
        include_partial_final_test=False,
    )

    test_starts = [fold.test_start for fold in folds]
    test_ends = [fold.test_end for fold in folds]
    assert test_starts == sorted(test_starts)
    assert all(left < right for left, right in zip(test_starts, test_starts[1:], strict=False))
    assert all(left < right for left, right in zip(test_ends, test_ends[1:], strict=False))


def test_champion_folds_use_expected_format_and_date_ordering() -> None:
    folds = build_sliding_window_champion_folds(
        data_start="2018-01-01",
        data_end="2023-09-30",
    )

    assert folds[0].name == "tw2y_v6m_t3m_q01_20200701_20200930"
    assert folds[-1].name == "tw5y_v6m_t3m_q01_20230701_20230930"
    assert format_fold_arg(folds[0]) == (
        "tw2y_v6m_t3m_q01_20200701_20200930:"
        "2018-01-01:2019-12-31:2020-01-01:2020-06-30:2020-07-01:2020-09-30"
    )
    assert len({fold.name for fold in folds}) == len(folds)
    for fold in folds:
        assert fold.train_end < fold.validation_start
        assert fold.validation_end < fold.test_start


def test_champion_config_keeps_fixed_current_champion_definition(tmp_path: Path) -> None:
    folds = build_sliding_window_champion_folds(
        data_start="2018-01-01",
        data_end="2020-09-30",
    )
    config = build_sliding_window_champion_config(
        folds=folds,
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    assert config.universes[0].name == "full"
    assert str(config.universes[0].data_path) == "artifacts/cache/universe_full_2018_2026.parquet"
    assert config.models[0].model_family == "baseline"
    assert config.models[0].model_type == "linear_regression"
    assert config.models[0].random_state == 42
    assert config.feature_set_versions == ["default_features_v1"]
    assert config.target_col == "excess_return_vs_spy_20d"
    assert config.label_horizon_days == 20
    assert config.transaction_cost_bps_values == [10.0]
    assert config.max_workers == 4
    signal = config.signals[0]
    assert signal.top_k == 1
    assert signal.min_holding_days == 10
    assert signal.switch_margin == 0.002
    assert signal.allocation_mode == "equal_weight"
    assert signal.score_temperature == 1.0
    assert signal.max_weight == 1.0


def test_sliding_window_champion_cli_dry_run_does_not_run_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called(config: Any) -> object:
        raise AssertionError("dry-run should not execute the sweep")

    monkeypatch.setattr(
        "us_market_ml.cli.run_sliding_window_champion_study.run_sweep",
        fail_if_called,
    )

    return_code = run_sliding_window_champion_study_from_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-09-30",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "dry run" in captured.out
    assert CHAMPION_NAME in captured.out
    assert "Generated folds:" in captured.out
    assert "Equivalent sweep command:" in captured.out
    assert "tw2y_v6m_t3m_q01_20200701_20200930" in captured.out


def test_sliding_window_champion_cli_runs_with_fixed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_configs: list[Any] = []

    def fake_run_sweep(config: Any) -> pd.DataFrame:
        observed_configs.append(config)
        return pd.DataFrame()

    monkeypatch.setattr(
        "us_market_ml.cli.run_sliding_window_champion_study.run_sweep",
        fake_run_sweep,
    )

    return_code = run_sliding_window_champion_study_from_args(
        [
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-09-30",
            "--max-workers",
            "3",
        ]
    )

    assert return_code == 0
    assert len(observed_configs) == 1
    config = observed_configs[0]
    assert config.run_name == "sliding_window_champion_study"
    assert config.max_workers == 3
    assert config.models[0].model_type == "linear_regression"
    assert config.signals[0].top_k == 1
