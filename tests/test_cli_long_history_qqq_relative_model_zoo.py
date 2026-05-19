from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli import run_long_history_qqq_relative_model_zoo as cli
from us_market_ml.cli.run_long_history_qqq_relative_model_zoo import (
    DATASET_SOURCE_LABEL,
    build_long_history_qqq_relative_model_zoo_plan,
    run_long_history_qqq_relative_model_zoo_from_args,
)


def test_long_history_model_zoo_plan_smoke_profile_shape(tmp_path: Path) -> None:
    plan = build_long_history_qqq_relative_model_zoo_plan(
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
        output_dir=tmp_path,
        run_name="long_history_qqq_relative_model_zoo",
        sweep_profile="smoke",
        max_workers=4,
    )

    assert plan.sweep_profile == "smoke"
    assert len(plan.specs) == 24
    assert plan.max_workers == 1
    assert {spec.feature_set_version for spec in plan.specs} == {
        "long_history_regime_features_v1"
    }
    assert {spec.training_mode for spec in plan.specs}.issuperset(
        {
            "frozen_base_1999_2018",
            "anchored_expanding_yearly",
            "anchored_expanding_quarterly",
            "recency_weighted_expanding",
            "base_recent_blend",
        }
    )


def test_long_history_model_zoo_cli_dry_run_writes_preflight_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_long_history_qqq_relative_model_zoo_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--sweep-profile",
            "smoke",
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-06-30",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "1",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "study"
    preflight_csv = output_dir / "long_history_qqq_relative_model_zoo_preflight.csv"
    preflight_md = output_dir / "long_history_qqq_relative_model_zoo_preflight.md"
    results_csv = output_dir / "long_history_qqq_relative_model_zoo_results.csv"
    skipped = output_dir / "long_history_qqq_relative_model_zoo_skipped_keys.txt"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert results_csv.exists()
    assert skipped.exists()
    assert len(preflight) == 24
    assert set(preflight["dataset_source"]) == {DATASET_SOURCE_LABEL}
    assert set(preflight["sweep_profile"]) == {"smoke"}
    assert set(preflight["feature_set_version"]) == {"long_history_regime_features_v1"}
    assert "QQQ buy-and-hold" in captured.out
    assert "total configs/runs: 24" in captured.out
    assert "valid configs/runs: 20" in captured.out
    assert "skipped configs/runs: 4" in captured.out
    assert "pending configs/runs: 20" in captured.out
    assert "model types:" in captured.out
    assert "training modes:" in captured.out
    assert "target variants:" in captured.out
    assert "allocation rules:" in captured.out
    assert "results CSV path:" in captured.out

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Primary benchmark: `QQQ buy-and-hold`" in markdown
    assert "Production signal export: `false`" in markdown
    assert "Rich replay/model card/paper tracking: `false`" in markdown


def test_long_history_model_zoo_shell_runner_defaults_to_dry_run_and_is_ctrl_c_safe() -> None:
    script_path = Path("scripts/run_long_history_qqq_relative_model_zoo.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert 'DRY_RUN="${DRY_RUN:-1}"' in script
    assert "SWEEP_PROFILE=" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "RESUME=" in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script
    assert "cleanup_on_interrupt" in script
    assert "forwarding SIGINT to study process" in script
    assert "sending SIGTERM" in script
    assert "sending SIGKILL" in script
    assert "sleep 60" in script
    assert "completed %s / pending %s / total %s" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "rich_replay" not in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "LONG_HISTORY_QQQ_MODEL_ZOO_TEST_HELPERS=1; "
                "source scripts/run_long_history_qqq_relative_model_zoo.sh; "
                "declare -f terminate_process_tree >/dev/null; "
                "declare -f wait_with_timeout >/dev/null"
            ),
        ],
        check=True,
    )


def test_long_history_model_zoo_missing_default_path_fails_helpfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_default = str(tmp_path / "missing" / "**" / "*.parquet")
    monkeypatch.delenv("MARKET_DATASET_GLOB", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_DATASET_GLOB", missing_default)

    with pytest.raises(FileNotFoundError, match="Default Tiingo-only dataset"):
        run_long_history_qqq_relative_model_zoo_from_args(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path / "study"),
            ]
        )


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2017-01-01", "2019-08-31", freq="D")
    symbols = ("SPY", "QQQ", "AAA", "BBB", "XLK", "IWM", "TLT", "GLD")
    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 10.0
        for day_index, date in enumerate(dates):
            close = base_price + day_index * (0.1 + symbol_index * 0.01)
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
