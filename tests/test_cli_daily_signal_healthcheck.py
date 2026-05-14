from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_daily_signal_healthcheck import run_daily_signal_healthcheck_from_args
from us_market_ml.exports.model_signal_manifest import build_model_signal_manifest


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_daily_signal_healthcheck_cli_loads_artifact_and_writes_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
) -> None:
    artifact = _valid_artifact()
    artifact_path = tmp_path / f"signals{suffix}"
    manifest_path = tmp_path / "model_signal_manifest.json"
    output_path = tmp_path / "healthcheck.csv"
    warnings_output_path = tmp_path / "warnings.txt"
    _write_artifact(artifact, artifact_path)
    manifest_path.write_text(json.dumps(_manifest(artifact, artifact_path)), encoding="utf-8")

    return_code = run_daily_signal_healthcheck_from_args(
        [
            "--artifact-path",
            str(artifact_path),
            "--manifest-path",
            str(manifest_path),
            "--model-version",
            "linear_smooth10_margin002_v1",
            "--expected-latest-date",
            "2024-01-03",
            "--min-symbol-count",
            "2",
            "--output-path",
            str(output_path),
            "--warnings-output-path",
            str(warnings_output_path),
        ]
    )

    captured = capsys.readouterr()
    summary = pd.read_csv(output_path)
    assert return_code == 0
    assert "status: ready" in captured.out
    assert summary.loc[0, "status"] == "ready"
    assert summary.loc[0, "row_count"] == 3
    assert summary.loc[0, "top_signal_count"] == 2
    assert warnings_output_path.read_text(encoding="utf-8") == ""


def test_daily_signal_healthcheck_cli_failed_validation_returns_nonzero(tmp_path: Path) -> None:
    artifact = pd.concat([_valid_artifact(), _valid_artifact().iloc[[0]]], ignore_index=True)
    artifact_path = tmp_path / "signals.csv"
    manifest_path = tmp_path / "model_signal_manifest.json"
    artifact.to_csv(artifact_path, index=False)
    manifest_path.write_text(json.dumps(_manifest(artifact, artifact_path)), encoding="utf-8")

    return_code = run_daily_signal_healthcheck_from_args(
        [
            "--artifact-path",
            str(artifact_path),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert return_code == 1


def _valid_artifact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["SPY", "SPY", "QQQ"],
            "model_version": [
                "linear_smooth10_margin002_v1",
                "linear_smooth10_margin002_v1",
                "linear_smooth10_margin002_v1",
            ],
            "score": [0.10, 0.20, 0.05],
            "rank": [1, 1, 2],
            "signal_bucket": ["top", "top", "other"],
            "feature_set_version": [
                "default_features_v1",
                "default_features_v1",
                "default_features_v1",
            ],
            "label_horizon_days": [20, 20, 20],
        }
    )


def _manifest(artifact: pd.DataFrame, artifact_path: Path) -> dict[str, object]:
    return build_model_signal_manifest(
        artifact,
        model_version="linear_smooth10_margin002_v1",
        artifact_path=str(artifact_path),
        candidate_id="baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
        feature_set_version="default_features_v1",
        label_horizon_days=20,
        status="ready",
        warnings=[],
        export_created_at="2024-01-03T00:00:00Z",
    )


def _write_artifact(artifact: pd.DataFrame, artifact_path: Path) -> None:
    if artifact_path.suffix == ".csv":
        artifact.to_csv(artifact_path, index=False)
    else:
        artifact.to_parquet(artifact_path, index=False)
