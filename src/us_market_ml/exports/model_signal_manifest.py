"""Manifest builder for local model signal artifacts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pandas as pd


def build_model_signal_manifest(
    artifact: pd.DataFrame,
    model_version: str,
    artifact_path: str,
    candidate_id: str,
    feature_set_version: str | None = None,
    label_horizon_days: int | None = None,
    training_window: dict[str, str] | None = None,
    validation_window: dict[str, str] | None = None,
    tracking_window: dict[str, str] | None = None,
    status: str = "ready",
    warnings: list[str] | None = None,
    export_created_at: str | None = None,
) -> dict[str, object]:
    """Build a compact manifest describing a written model signal artifact."""

    if not model_version:
        raise ValueError("model_version must be non-empty.")
    if not artifact_path:
        raise ValueError("artifact_path must be non-empty.")
    if not candidate_id:
        raise ValueError("candidate_id must be non-empty.")

    latest_signal_date = None
    if not artifact.empty:
        latest_signal_date = pd.Timestamp(pd.to_datetime(artifact["date"]).max()).date().isoformat()

    manifest: dict[str, object] = {
        "model_version": model_version,
        "feature_set_version": feature_set_version,
        "label_horizon_days": label_horizon_days,
        "latest_signal_date": latest_signal_date,
        "row_count": int(len(artifact)),
        "symbol_count": int(artifact["symbol"].nunique()) if "symbol" in artifact else 0,
        "export_created_at": export_created_at or _utc_now_isoformat(),
        "training_window": training_window,
        "validation_window": validation_window,
        "tracking_window": tracking_window,
        "candidate_id": candidate_id,
        "artifact_path": artifact_path,
        "status": status,
        "warnings": list(warnings or []),
    }
    manifest["checksum_sha256"] = _checksum_artifact_rows(artifact)
    return manifest


def _checksum_artifact_rows(artifact: pd.DataFrame) -> str:
    if artifact.empty:
        return hashlib.sha256(b"").hexdigest()

    sorted_artifact = artifact.copy()
    for column in sorted_artifact.columns:
        if pd.api.types.is_datetime64_any_dtype(sorted_artifact[column]):
            continue
        if sorted_artifact[column].dtype == "object":
            sorted_artifact[column] = sorted_artifact[column].astype("string")
    sorted_artifact = sorted_artifact.sort_values(
        by=list(sorted_artifact.columns),
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(sorted_artifact, index=False)
    return hashlib.sha256(row_hashes.to_numpy(dtype="uint64").tobytes()).hexdigest()


def _utc_now_isoformat() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
