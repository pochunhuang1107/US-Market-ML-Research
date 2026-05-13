from __future__ import annotations

import pandas as pd

from us_market_ml.exports.model_signal_manifest import build_model_signal_manifest


def test_build_model_signal_manifest_counts_latest_date_and_symbols() -> None:
    artifact = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "symbol": "SPY",
                "model_version": "linear_smooth10_margin002_v1",
                "score": 0.1,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "symbol": "QQQ",
                "model_version": "linear_smooth10_margin002_v1",
                "score": 0.2,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "symbol": "SPY",
                "model_version": "linear_smooth10_margin002_v1",
                "score": 0.3,
            },
        ]
    )

    manifest = build_model_signal_manifest(
        artifact,
        model_version="linear_smooth10_margin002_v1",
        artifact_path="artifacts/model_signals/model_version=linear_smooth10_margin002_v1/signals.parquet",
        candidate_id="baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
        feature_set_version="default_features_v1",
        label_horizon_days=20,
        training_window={"start": "2018-01-01", "end": "2022-12-31"},
        validation_window={"start": "2023-01-01", "end": "2023-12-31"},
        tracking_window={"start": "2024-01-03", "end": "2024-01-03"},
        warnings=["example warning"],
        export_created_at="2024-01-03T00:00:00Z",
    )

    assert manifest["model_version"] == "linear_smooth10_margin002_v1"
    assert manifest["feature_set_version"] == "default_features_v1"
    assert manifest["label_horizon_days"] == 20
    assert manifest["latest_signal_date"] == "2024-01-03"
    assert manifest["row_count"] == 3
    assert manifest["symbol_count"] == 2
    assert manifest["candidate_id"] == (
        "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002"
    )
    assert manifest["status"] == "ready"
    assert manifest["warnings"] == ["example warning"]
    assert isinstance(manifest["checksum_sha256"], str)
    assert len(manifest["checksum_sha256"]) == 64
