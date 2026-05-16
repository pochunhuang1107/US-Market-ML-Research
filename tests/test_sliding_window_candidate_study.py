from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_candidate_study import (
    CHAMPION_CANDIDATE_NAME,
    DEFAULT_RUN_NAME,
    build_candidate_run_keys,
    build_candidate_universes,
    build_sliding_window_candidate_folds,
    build_sliding_window_candidate_phase_configs,
    count_phase_independent_configs,
    run_sliding_window_candidate_study_from_args,
)
from us_market_ml.cli.run_sliding_window_champion_study import (
    build_sliding_window_champion_folds,
)


def _phase_counts(phases: Any) -> dict[str, int]:
    return {
        phase.name: sum(
            len(config.universes)
            * len(config.folds)
            * len(config.models)
            * len(config.signals)
            * len(config.feature_set_versions)
            * len(config.transaction_cost_bps_values)
            for config in phase.configs
        )
        for phase in phases
    }


def test_candidate_folds_reuse_champion_sliding_window_generation() -> None:
    candidate_folds = build_sliding_window_candidate_folds(
        data_start="2018-01-01",
        data_end="2023-09-30",
    )
    champion_folds = build_sliding_window_champion_folds(
        data_start="2018-01-01",
        data_end="2023-09-30",
    )

    assert candidate_folds == champion_folds
    assert candidate_folds[0].name == "tw2y_v6m_t3m_q01_20200701_20200930"
    assert candidate_folds[-1].name == "tw5y_v6m_t3m_q01_20230701_20230930"


def test_broad_macro_missing_file_warns_and_skips(tmp_path: Path) -> None:
    warnings: list[str] = []

    universes = build_candidate_universes(
        full_universe_path=tmp_path / "full.parquet",
        broad_macro_universe_path=tmp_path / "missing_broad_macro.parquet",
        warn=warnings.append,
    )

    assert [universe.name for universe in universes] == ["full"]
    assert len(warnings) == 1
    assert "broad_macro universe file not found" in warnings[0]


def test_candidate_phase_config_generation_counts_default_grid(tmp_path: Path) -> None:
    broad_macro_path = tmp_path / "broad_macro.parquet"
    broad_macro_path.touch()
    folds = build_sliding_window_candidate_folds(
        data_start="2018-01-01",
        data_end="2026-05-08",
    )
    universes = build_candidate_universes(
        full_universe_path=tmp_path / "full.parquet",
        broad_macro_universe_path=broad_macro_path,
    )

    phases = build_sliding_window_candidate_phase_configs(
        folds=folds,
        universes=universes,
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    assert [phase.name for phase in phases] == ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    assert len(folds) == 56
    assert _phase_counts(phases) == {
        "Phase 1": 112,
        "Phase 2": 4480,
        "Phase 3": 3136,
        "Phase 4": 784,
    }
    assert count_phase_independent_configs(phases) == 8512
    assert phases[3].notes == ("Unsupported baseline model types skipped: lasso, elastic_net",)


def test_candidate_grid_uses_phase_specific_models_and_signals(tmp_path: Path) -> None:
    folds = build_sliding_window_candidate_folds(
        data_start="2018-01-01",
        data_end="2020-09-30",
    )
    universes = [build_candidate_universes(include_broad_macro=False)[0]]

    phases = build_sliding_window_candidate_phase_configs(
        folds=folds,
        universes=universes,
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    phase1_config = phases[0].configs[0]
    assert phase1_config.models[0].model_type == "linear_regression"
    assert phase1_config.signals[0].top_k == 1
    assert phase1_config.signals[0].allocation_mode == "equal_weight"
    assert phase1_config.transaction_cost_bps_values == [10.0]

    phase2_signals = phases[1].configs[0].signals
    assert len(phase2_signals) == 20
    assert {signal.top_k for signal in phase2_signals} == {2, 3, 5}
    assert all(
        signal.max_weight != 0.4
        for signal in phase2_signals
        if signal.top_k == 2 and signal.allocation_mode == "capped_score_weighted"
    )

    phase4_model_types = [
        model.model_type
        for config in phases[3].configs
        for model in config.models
    ]
    assert phase4_model_types == [
        "ridge",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    ]


def test_candidate_study_has_no_duplicate_run_keys(tmp_path: Path) -> None:
    broad_macro_path = tmp_path / "broad_macro.parquet"
    broad_macro_path.touch()
    phases = build_sliding_window_candidate_phase_configs(
        folds=build_sliding_window_candidate_folds(
            data_start="2018-01-01",
            data_end="2021-06-30",
        ),
        universes=build_candidate_universes(
            full_universe_path=tmp_path / "full.parquet",
            broad_macro_universe_path=broad_macro_path,
        ),
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    run_keys = build_candidate_run_keys(phases)

    assert len(run_keys) == len(set(run_keys))


def test_candidate_cli_dry_run_does_not_run_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called(config: Any) -> pd.DataFrame:
        raise AssertionError("dry-run should not execute the sweep")

    monkeypatch.setattr(
        "us_market_ml.cli.run_sliding_window_candidate_study.run_sweep",
        fail_if_called,
    )

    return_code = run_sliding_window_candidate_study_from_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-09-30",
            "--no-include-broad-macro",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "dry_run: True" in captured.out
    assert CHAMPION_CANDIDATE_NAME in captured.out
    assert "fold count: 1" in captured.out
    assert "Phase 1: current return champion sanity" in captured.out
    assert "Phase 4: model-family robustness baseline" in captured.out
    assert "Command that would run:" in captured.out


def test_candidate_cli_count_only_prints_independent_config_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = run_sliding_window_candidate_study_from_args(
        [
            "--count-only",
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-09-30",
            "--no-include-broad-macro",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.out.strip() == "76"
    assert captured.err == ""
    assert DEFAULT_RUN_NAME == "sliding_window_overnight_candidate_study"
