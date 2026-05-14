from __future__ import annotations

import pandas as pd

from us_market_ml.exports import run_model_signal_healthcheck
from us_market_ml.exports.model_signal_manifest import build_model_signal_manifest


def test_model_signal_healthcheck_ready_case() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        model_version="linear_smooth10_margin002_v1",
        expected_latest_date="2024-01-03",
        min_symbol_count=2,
        artifact_path="artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/signals.parquet",
    )

    row = summary.iloc[0]
    assert row["status"] == "ready"
    assert row["latest_signal_date"] == "2024-01-03"
    assert row["row_count"] == 3
    assert row["symbol_count"] == 2
    assert row["top_signal_count"] == 2
    assert row["other_signal_count"] == 1
    assert row["duplicate_count"] == 0
    assert row["forbidden_column_count"] == 0
    assert row["export_created_date"] == "2024-01-03"
    assert warnings == []


def test_model_signal_healthcheck_catches_duplicate_rows() -> None:
    artifact = pd.concat([_valid_artifact(), _valid_artifact().iloc[[0]]], ignore_index=True)
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(artifact, manifest)

    assert summary.iloc[0]["status"] == "failed"
    assert summary.iloc[0]["duplicate_count"] == 2
    assert warnings == []


def test_model_signal_healthcheck_catches_model_version_mismatch() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, _ = run_model_signal_healthcheck(
        artifact,
        manifest,
        model_version="different_model_v1",
    )

    assert summary.iloc[0]["status"] == "failed"


def test_model_signal_healthcheck_warns_on_expected_latest_date_mismatch() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        expected_latest_date="2024-01-04",
    )

    assert summary.iloc[0]["status"] == "warning"
    assert any("does not match expected 2024-01-04" in warning for warning in warnings)


def test_model_signal_healthcheck_ready_when_signal_age_within_threshold() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        max_signal_age_days=2,
        reference_date="2024-01-05",
    )

    row = summary.iloc[0]
    assert row["status"] == "ready"
    assert row["reference_date"] == "2024-01-05"
    assert row["signal_age_days"] == 2
    assert row["max_signal_age_days"] == 2
    assert not bool(row["stale_as_failure"])
    assert not bool(row["stale_signal"])
    assert warnings == []


def test_model_signal_healthcheck_does_not_fail_when_export_created_later_than_signal() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)
    manifest["export_created_at"] = "2024-01-07T12:00:00Z"

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        max_signal_age_days=2,
        reference_date="2024-01-05",
    )

    row = summary.iloc[0]
    assert row["status"] == "ready"
    assert row["latest_signal_date"] == "2024-01-03"
    assert row["export_created_date"] == "2024-01-07"
    assert row["signal_age_days"] == 2
    assert warnings == []


def test_model_signal_healthcheck_warns_when_signal_age_exceeds_threshold() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        max_signal_age_days=3,
        reference_date="2024-01-07",
    )

    row = summary.iloc[0]
    assert row["status"] == "warning"
    assert row["signal_age_days"] == 4
    assert bool(row["stale_signal"])
    assert warnings == [
        "latest_signal_date is stale: age 4 days exceeds max_signal_age_days 3"
    ]


def test_model_signal_healthcheck_fails_when_stale_as_failure() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        max_signal_age_days=3,
        reference_date="2024-01-07",
        stale_as_failure=True,
    )

    row = summary.iloc[0]
    assert row["status"] == "failed"
    assert bool(row["stale_as_failure"])
    assert bool(row["stale_signal"])
    assert warnings == [
        "latest_signal_date is stale: age 4 days exceeds max_signal_age_days 3"
    ]


def test_model_signal_healthcheck_staleness_works_without_expected_latest_date() -> None:
    artifact = _valid_artifact()
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        max_signal_age_days=0,
        reference_date="2024-01-04",
    )

    assert summary.iloc[0]["status"] == "warning"
    assert summary.iloc[0]["signal_age_days"] == 1
    assert bool(summary.iloc[0]["stale_signal"])
    assert warnings == [
        "latest_signal_date is stale: age 1 days exceeds max_signal_age_days 0"
    ]


def test_model_signal_healthcheck_warns_when_no_top_signal() -> None:
    artifact = _valid_artifact()
    artifact["signal_bucket"] = "other"
    manifest = _manifest(artifact)

    summary, warnings = run_model_signal_healthcheck(artifact, manifest)

    assert summary.iloc[0]["status"] == "warning"
    assert "No rows have signal_bucket == 'top'." in warnings


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


def _manifest(artifact: pd.DataFrame) -> dict[str, object]:
    return build_model_signal_manifest(
        artifact,
        model_version="linear_smooth10_margin002_v1",
        artifact_path=(
            "artifacts/model_signals_daily/"
            "model_version=linear_smooth10_margin002_v1/signals.parquet"
        ),
        candidate_id="baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
        feature_set_version="default_features_v1",
        label_horizon_days=20,
        status="ready",
        warnings=[],
        export_created_at="2024-01-03T00:00:00Z",
    )
