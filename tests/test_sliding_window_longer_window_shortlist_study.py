from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from us_market_ml.cli.run_sliding_window_longer_window_shortlist_study import (
    DEFAULT_RUN_NAME,
    build_available_shortlist_candidates,
    build_longer_window_shortlist_config_blocks,
    build_longer_window_shortlist_folds,
    build_longer_window_shortlist_universes,
    build_shortlist_run_keys,
    count_independent_configs,
    run_sliding_window_longer_window_shortlist_study_from_args,
)


def test_longer_window_folds_include_6_month_and_12_month_test_windows() -> None:
    folds = build_longer_window_shortlist_folds(
        data_start="2018-01-01",
        data_end="2026-05-08",
    )

    six_month_folds = [fold for fold in folds if "_t6m_" in fold.name]
    twelve_month_folds = [fold for fold in folds if "_t12m_" in fold.name]

    assert len(folds) == 84
    assert len(six_month_folds) == 56
    assert len(twelve_month_folds) == 28
    assert six_month_folds[0].name == "tw2y_v6m_t6m_q01_20200701_20201231"
    assert six_month_folds[-1].name == "tw5y_v6m_t6m_q12_20260401_20260508"
    assert twelve_month_folds[0].name == "tw2y_v6m_t12m_q01_20200701_20210630"
    assert twelve_month_folds[-1].name == "tw5y_v6m_t12m_q06_20260101_20260508"

    for fold in folds:
        assert fold.train_start <= fold.train_end
        assert fold.train_end < fold.validation_start
        assert fold.validation_start <= fold.validation_end
        assert fold.validation_end < fold.test_start
        assert fold.test_start <= fold.test_end
        assert fold.test_end <= "2026-05-08"


def test_shortlist_config_generation_uses_exact_candidate_set(tmp_path: Path) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    full_path.touch()
    broad_path.touch()
    folds = build_longer_window_shortlist_folds(
        data_start="2018-01-01",
        data_end="2020-12-31",
    )
    universes = build_longer_window_shortlist_universes(
        full_universe_path=full_path,
        broad_macro_universe_path=broad_path,
    )
    candidates = build_available_shortlist_candidates(universes)

    blocks = build_longer_window_shortlist_config_blocks(
        folds=folds,
        universes=universes,
        candidates=candidates,
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    assert [candidate.label for candidate in candidates] == [
        "broad_macro / linear_regression / top1 / equal_weight / cost10",
        "broad_macro / linear_regression / top2 / equal_weight / cost10",
        "broad_macro / random_forest / top3 / equal_weight / cost10",
        "broad_macro / ridge / top5 / equal_weight / cost10",
        "full / ridge / top3 / equal_weight / cost10",
        "full / ridge / top5 / equal_weight / cost10",
    ]
    assert len(blocks) == 4
    assert count_independent_configs(blocks) == len(folds) * 6
    assert {block.config.models[0].model_type for block in blocks} == {
        "linear_regression",
        "random_forest",
        "ridge",
    }
    assert all(block.config.models[0].model_family == "baseline" for block in blocks)
    assert all(block.config.feature_set_versions == ["default_features_v1"] for block in blocks)
    assert all(block.config.transaction_cost_bps_values == [10.0] for block in blocks)
    assert all(block.config.max_workers == 4 for block in blocks)


def test_longer_window_shortlist_study_has_no_duplicate_run_keys(tmp_path: Path) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    full_path.touch()
    broad_path.touch()
    folds = build_longer_window_shortlist_folds(
        data_start="2018-01-01",
        data_end="2021-06-30",
    )
    universes = build_longer_window_shortlist_universes(
        full_universe_path=full_path,
        broad_macro_universe_path=broad_path,
    )
    candidates = build_available_shortlist_candidates(universes)
    blocks = build_longer_window_shortlist_config_blocks(
        folds=folds,
        universes=universes,
        candidates=candidates,
        output_dir=tmp_path,
        run_name="study",
        resume=True,
        max_workers=4,
    )

    run_keys = build_shortlist_run_keys(blocks)

    assert len(run_keys) == len(set(run_keys))


def test_count_only_prints_independent_config_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    full_path.touch()
    broad_path.touch()

    return_code = run_sliding_window_longer_window_shortlist_study_from_args(
        [
            "--count-only",
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-12-31",
            "--full-universe",
            str(full_path),
            "--broad-macro-universe",
            str(broad_path),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.out.strip() == "18"
    assert captured.err == ""
    assert DEFAULT_RUN_NAME == "sliding_window_longer_window_shortlist_study"


def test_missing_broad_macro_file_warns_and_skips_broad_candidates(tmp_path: Path) -> None:
    warnings: list[str] = []
    full_path = tmp_path / "full.parquet"
    full_path.touch()

    universes = build_longer_window_shortlist_universes(
        full_universe_path=full_path,
        broad_macro_universe_path=tmp_path / "missing_broad.parquet",
        warn=warnings.append,
    )
    candidates = build_available_shortlist_candidates(universes)

    assert [universe.name for universe in universes] == ["full"]
    assert [candidate.label for candidate in candidates] == [
        "full / ridge / top3 / equal_weight / cost10",
        "full / ridge / top5 / equal_weight / cost10",
    ]
    assert len(warnings) == 1
    assert "skipping broad_macro shortlist candidates" in warnings[0]


def test_missing_full_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required full universe file not found"):
        build_longer_window_shortlist_universes(
            full_universe_path=tmp_path / "missing_full.parquet",
            broad_macro_universe_path=tmp_path / "missing_broad.parquet",
        )


def test_dry_run_prints_plan_and_does_not_run_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    full_path = tmp_path / "full.parquet"
    broad_path = tmp_path / "broad.parquet"
    full_path.touch()
    broad_path.touch()

    def fail_if_called(config: Any) -> pd.DataFrame:
        raise AssertionError("dry-run should not execute the sweep")

    monkeypatch.setattr(
        "us_market_ml.cli.run_sliding_window_longer_window_shortlist_study.run_sweep",
        fail_if_called,
    )

    return_code = run_sliding_window_longer_window_shortlist_study_from_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--data-end",
            "2020-12-31",
            "--full-universe",
            str(full_path),
            "--broad-macro-universe",
            str(broad_path),
            "--max-workers",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "dry_run: True" in captured.out
    assert "fold count: 3" in captured.out
    assert "candidate count: 6" in captured.out
    assert "estimated independent config count: 18" in captured.out
    assert "max workers: 3" in captured.out
    assert "Equivalent command:" in captured.out
    assert "broad_macro / random_forest / top3 / equal_weight / cost10" in captured.out
