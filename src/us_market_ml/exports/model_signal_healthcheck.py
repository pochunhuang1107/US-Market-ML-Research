"""Healthcheck reporting for local daily model signal artifacts."""

from __future__ import annotations

import pandas as pd

from us_market_ml.exports.model_signals import (
    FORBIDDEN_LOOKAHEAD_COLUMNS,
    FORBIDDEN_LOOKAHEAD_PREFIXES,
    MODEL_SIGNAL_REQUIRED_COLUMNS,
    validate_model_signal_artifact,
)

HEALTHCHECK_COLUMNS: tuple[str, ...] = (
    "status",
    "model_version",
    "latest_signal_date",
    "row_count",
    "symbol_count",
    "top_signal_count",
    "other_signal_count",
    "duplicate_count",
    "forbidden_column_count",
    "manifest_status",
    "manifest_latest_signal_date",
    "artifact_path",
    "warnings_count",
)


def run_model_signal_healthcheck(
    artifact: pd.DataFrame,
    manifest: dict[str, object],
    model_version: str | None = None,
    expected_latest_date: str | None = None,
    min_symbol_count: int | None = None,
    require_top_signal: bool = True,
    artifact_path: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate a local model signal artifact and return a compact summary row."""

    errors: list[str] = []
    warnings: list[str] = []
    required_columns_present = all(
        column in artifact.columns for column in MODEL_SIGNAL_REQUIRED_COLUMNS
    )

    missing_columns = [
        column for column in MODEL_SIGNAL_REQUIRED_COLUMNS if column not in artifact.columns
    ]
    if missing_columns:
        errors.append(
            "Artifact is missing required model signal columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )

    if required_columns_present:
        try:
            validate_model_signal_artifact(artifact)
        except ValueError as exc:
            errors.append(str(exc))

    artifact_model_versions = (
        _non_empty_strings(artifact["model_version"]) if "model_version" in artifact else []
    )
    resolved_model_version = _resolve_model_version(
        artifact_model_versions,
        model_version,
        manifest,
    )
    if not resolved_model_version:
        errors.append("model_version must be non-empty.")
    if len(set(artifact_model_versions)) > 1:
        errors.append("Artifact contains more than one model_version.")
    if model_version is not None:
        requested_model_version = model_version.strip()
        if not requested_model_version:
            errors.append("model_version argument must be non-empty when provided.")
        elif set(artifact_model_versions) != {requested_model_version}:
            errors.append(
                f"Artifact model_version values do not match requested {requested_model_version}."
            )

    latest_signal_date = _latest_signal_date(artifact)
    duplicate_count = _duplicate_count(artifact)
    forbidden_column_count = _forbidden_column_count(artifact)
    top_signal_count = _signal_bucket_count(artifact, "top")
    other_signal_count = _signal_bucket_count(artifact, "other")
    symbol_count = int(artifact["symbol"].nunique()) if "symbol" in artifact else 0
    row_count = int(len(artifact))

    manifest_model_version = _manifest_text(manifest, "model_version")
    if not manifest_model_version:
        errors.append("Manifest model_version must be non-empty.")
    elif resolved_model_version and manifest_model_version != resolved_model_version:
        errors.append(
            f"Manifest model_version {manifest_model_version} does not match artifact "
            f"model_version {resolved_model_version}."
        )

    manifest_row_count = manifest.get("row_count")
    if manifest_row_count is not None and _coerce_int(manifest_row_count) != row_count:
        errors.append(
            f"Manifest row_count {manifest_row_count} does not match artifact "
            f"row_count {row_count}."
        )

    manifest_latest_signal_date = _manifest_text(manifest, "latest_signal_date")
    if (
        manifest_latest_signal_date
        and latest_signal_date
        and manifest_latest_signal_date != latest_signal_date
    ):
        errors.append(
            f"Manifest latest_signal_date {manifest_latest_signal_date} does not match artifact "
            f"latest_signal_date {latest_signal_date}."
        )

    manifest_status = _manifest_text(manifest, "status")
    if manifest_status != "ready":
        if manifest_status == "failed":
            errors.append("Manifest status is failed.")
        else:
            warnings.append(f"Manifest status is {manifest_status or 'missing'}, expected ready.")

    if expected_latest_date is not None:
        expected = pd.Timestamp(expected_latest_date).date().isoformat()
        if latest_signal_date != expected:
            warnings.append(
                f"Latest signal date {latest_signal_date or 'missing'} does not match "
                f"expected {expected}."
            )
    if min_symbol_count is not None and symbol_count < min_symbol_count:
        warnings.append(f"Symbol count {symbol_count} is below minimum {min_symbol_count}.")
    if require_top_signal and top_signal_count == 0:
        warnings.append("No rows have signal_bucket == 'top'.")
    if artifact_path is not None:
        manifest_artifact_path = _manifest_text(manifest, "artifact_path")
        if manifest_artifact_path and manifest_artifact_path != artifact_path:
            warnings.append(
                f"Manifest artifact_path {manifest_artifact_path} differs from checked "
                f"path {artifact_path}."
            )

    warnings.extend(_manifest_warnings(manifest))

    status = "failed" if errors else "warning" if warnings else "ready"
    summary = pd.DataFrame(
        [
            {
                "status": status,
                "model_version": resolved_model_version,
                "latest_signal_date": latest_signal_date,
                "row_count": row_count,
                "symbol_count": symbol_count,
                "top_signal_count": top_signal_count,
                "other_signal_count": other_signal_count,
                "duplicate_count": duplicate_count,
                "forbidden_column_count": forbidden_column_count,
                "manifest_status": manifest_status,
                "manifest_latest_signal_date": manifest_latest_signal_date,
                "artifact_path": artifact_path or _manifest_text(manifest, "artifact_path"),
                "warnings_count": len(warnings),
            }
        ],
        columns=list(HEALTHCHECK_COLUMNS),
    )
    return summary, warnings


def _non_empty_strings(series: pd.Series) -> list[str]:
    values = series.astype("string")
    return sorted({str(value).strip() for value in values.dropna() if str(value).strip()})


def _resolve_model_version(
    artifact_model_versions: list[str],
    requested_model_version: str | None,
    manifest: dict[str, object],
) -> str:
    if requested_model_version is not None and requested_model_version.strip():
        return requested_model_version.strip()
    if len(artifact_model_versions) == 1:
        return artifact_model_versions[0]
    return _manifest_text(manifest, "model_version")


def _latest_signal_date(artifact: pd.DataFrame) -> str:
    if artifact.empty or "date" not in artifact:
        return ""
    parsed = pd.to_datetime(artifact["date"], errors="coerce")
    if parsed.dropna().empty:
        return ""
    return str(pd.Timestamp(parsed.max()).date().isoformat())


def _duplicate_count(artifact: pd.DataFrame) -> int:
    if not all(column in artifact.columns for column in MODEL_SIGNAL_REQUIRED_COLUMNS):
        return 0
    parsed_dates = pd.to_datetime(artifact["date"], errors="coerce")
    duplicate_mask = (
        artifact.assign(__model_signal_date=parsed_dates)
        .duplicated(subset=["__model_signal_date", "symbol", "model_version"], keep=False)
    )
    return int(duplicate_mask.sum())


def _forbidden_column_count(artifact: pd.DataFrame) -> int:
    return len(
        [
            column
            for column in artifact.columns
            if column in FORBIDDEN_LOOKAHEAD_COLUMNS
            or any(column.startswith(prefix) for prefix in FORBIDDEN_LOOKAHEAD_PREFIXES)
        ]
    )


def _signal_bucket_count(artifact: pd.DataFrame, bucket: str) -> int:
    if "signal_bucket" not in artifact:
        return 0
    return int((artifact["signal_bucket"].astype("string") == bucket).sum())


def _manifest_text(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _manifest_warnings(manifest: dict[str, object]) -> list[str]:
    raw_warnings = manifest.get("warnings")
    if raw_warnings is None:
        return []
    if isinstance(raw_warnings, list):
        return [str(warning) for warning in raw_warnings if str(warning).strip()]
    text = str(raw_warnings).strip()
    return [text] if text else []


__all__ = ["HEALTHCHECK_COLUMNS", "run_model_signal_healthcheck"]
