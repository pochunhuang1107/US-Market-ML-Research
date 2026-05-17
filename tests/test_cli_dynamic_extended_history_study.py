from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import us_market_ml.cli.run_dynamic_extended_history_study as dynamic_cli
from us_market_ml.cli.run_dynamic_extended_history_study import (
    build_dynamic_extended_history_sweep_config,
    run_dynamic_extended_history_study_from_args,
)
from us_market_ml.experiments import SweepFoldConfig


def test_dynamic_extended_history_config_uses_dynamic_fold_train_defaults(
    tmp_path: Path,
) -> None:
    fold = _fold()
    config = build_dynamic_extended_history_sweep_config(
        market_dataset_glob="/tmp/Market Data/**/*.parquet",
        folds=[fold],
        output_dir=tmp_path,
        run_name="dynamic",
        resume=True,
        max_workers=4,
    )

    assert str(config.universes[0].data_path) == "/tmp/Market Data/**/*.parquet"
    assert config.universes[0].name == "dynamic_extended"
    assert config.universe_mode == "dynamic"
    assert config.feature_selection_scope == "fold_train"
    assert config.min_history_rows == 252
    assert [model.model_type for model in config.models] == ["linear_regression", "ridge"]
    assert [signal.top_k for signal in config.signals] == [1, 3, 5]
    assert {signal.allocation_mode for signal in config.signals} == {"equal_weight"}
    assert config.transaction_cost_bps_values == [10.0, 20.0]
    assert config.min_holding_days == 10
    assert config.switch_margin == pytest.approx(0.002)
    assert config.label_horizon_days == 20
    assert config.feature_set_versions == ["default_features_v1"]


def test_dynamic_cli_dry_run_writes_preflight_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    monkeypatch.setattr(dynamic_cli, "build_dynamic_extended_history_folds", lambda **_: [_fold()])

    exit_code = run_dynamic_extended_history_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "dynamic_extended_history_study",
            "--min-history-rows",
            "5",
        ]
    )

    captured = capsys.readouterr()
    preflight_csv = tmp_path / "study" / "dynamic_extended_history_preflight.csv"
    preflight_md = tmp_path / "study" / "dynamic_extended_history_preflight.md"
    skipped = tmp_path / "study" / "dynamic_extended_history_study_skipped_keys.txt"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert skipped.exists()
    assert "estimated total configs: 12" in captured.out
    assert "estimated valid configs: 8" in captured.out
    assert "estimated skipped configs: 4" in captured.out
    assert len(preflight) == 12
    assert set(preflight["top_k"]) == {1, 3, 5}
    assert preflight["selected_feature_count"].min() > 0
    assert preflight.loc[preflight["top_k"] == 5, "supports_top_k"].eq(False).all()
    assert preflight.loc[preflight["top_k"] == 5, "valid_config"].eq(False).all()
    assert set(preflight.loc[preflight["top_k"] == 5, "invalid_reason"]) == {
        "unsupported_top_k"
    }
    assert len(skipped.read_text(encoding="utf-8").splitlines()) == 4
    assert preflight_md.read_text(encoding="utf-8").startswith(
        "# Dynamic Extended-History Preflight"
    )


def test_dynamic_shell_runner_has_required_overrides_and_progress() -> None:
    script = Path("scripts/run_dynamic_extended_history_study.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "MAX_WORKERS=" in script
    assert "RESUME=" in script
    assert "DRY_RUN=" in script
    assert "sleep 60" in script
    assert "completed=%s skipped=%s failed=%s result_rows=%s" in script


def _fold() -> SweepFoldConfig:
    return SweepFoldConfig(
        name="synthetic_fold",
        train_start="2020-01-01",
        train_end="2020-02-15",
        validation_start="2020-02-16",
        validation_end="2020-02-29",
        test_start="2020-03-01",
        test_end="2020-03-25",
    )


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
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
                    "volume": 1_000_000 + symbol_index * 10_000 + day_index,
                    "vwap": close,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
