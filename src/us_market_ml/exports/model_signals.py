"""Model signal artifact builders for platform runtime consumption.

The simulation platform consumes point-in-time model scores as a signal layer.
This module intentionally exports only runtime signal fields, not supervised
training labels, future returns, realized returns, or execution decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_SIGNAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "model_version",
    "score",
)

MODEL_SIGNAL_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "predicted_return",
    "rank",
    "confidence",
    "signal_bucket",
    "feature_set_version",
    "label_horizon_days",
    "created_at",
)

FORBIDDEN_LOOKAHEAD_COLUMNS: tuple[str, ...] = (
    "future_return",
    "forward_return",
    "realized_return",
    "target_return",
)

FORBIDDEN_LOOKAHEAD_PREFIXES: tuple[str, ...] = tuple(
    f"{column}_" for column in FORBIDDEN_LOOKAHEAD_COLUMNS
)


def build_model_signal_artifact(
    signals: pd.DataFrame,
    model_version: str,
    feature_set_version: str | None = None,
    label_horizon_days: int | None = None,
    created_at: str | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "prediction",
    rank_col: str = "signal_rank",
) -> pd.DataFrame:
    """Build a platform-compatible model signal artifact from experiment signals."""

    _validate_non_empty_text(model_version, "model_version")
    _validate_required_input_columns(signals, [date_col, symbol_col, score_col])
    if label_horizon_days is not None and label_horizon_days < 0:
        raise ValueError("label_horizon_days must be greater than or equal to 0.")

    out = pd.DataFrame(
        {
            "date": signals[date_col],
            "symbol": signals[symbol_col],
            "model_version": model_version,
            "score": signals[score_col],
            "predicted_return": signals[score_col],
        }
    )

    if rank_col in signals.columns:
        out["rank"] = signals[rank_col]
    if "confidence" in signals.columns:
        out["confidence"] = signals["confidence"]
    if "selected" in signals.columns:
        selected = signals["selected"].fillna(False).astype(bool)
        out["signal_bucket"] = np.where(selected, "top", "other")
    elif "signal_bucket" in signals.columns:
        out["signal_bucket"] = signals["signal_bucket"]
    if feature_set_version is not None:
        _validate_non_empty_text(feature_set_version, "feature_set_version")
        out["feature_set_version"] = feature_set_version
    if label_horizon_days is not None:
        out["label_horizon_days"] = label_horizon_days
    out["created_at"] = created_at or pd.Timestamp.now(tz="UTC").isoformat()

    output_columns = [
        *MODEL_SIGNAL_REQUIRED_COLUMNS,
        *[column for column in MODEL_SIGNAL_OPTIONAL_COLUMNS if column in out.columns],
    ]
    artifact = out[output_columns].copy()
    validate_model_signal_artifact(artifact)
    return artifact


def validate_model_signal_artifact(df: pd.DataFrame) -> None:
    """Validate a model signal artifact against the platform v1 contract."""

    _validate_required_input_columns(df, list(MODEL_SIGNAL_REQUIRED_COLUMNS))
    _validate_forbidden_lookahead_columns(df)

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("date must parse as datetime/date for every row.")

    _validate_non_empty_series(df["symbol"], "symbol")
    _validate_non_empty_series(df["model_version"], "model_version")

    score = pd.to_numeric(df["score"], errors="coerce")
    if score.isna().any():
        raise ValueError("score must be numeric for every row.")

    duplicate_mask = (
        df.assign(__model_signal_date=parsed_dates)
        .duplicated(subset=["__model_signal_date", "symbol", "model_version"], keep=False)
    )
    if duplicate_mask.any():
        raise ValueError("Artifact contains duplicate date/symbol/model_version rows.")

    if "rank" in df.columns:
        _validate_optional_numeric_series(df["rank"], "rank")

    if "confidence" in df.columns:
        confidence = _optional_numeric_series(df["confidence"], "confidence")
        non_null_confidence = confidence.dropna()
        if ((non_null_confidence < 0.0) | (non_null_confidence > 1.0)).any():
            raise ValueError("confidence must be between 0 and 1 when present.")


def _validate_required_input_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"DataFrame is missing required model signal columns: {missing}")


def _validate_forbidden_lookahead_columns(df: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in df.columns
        if column in FORBIDDEN_LOOKAHEAD_COLUMNS
        or any(column.startswith(prefix) for prefix in FORBIDDEN_LOOKAHEAD_PREFIXES)
    ]
    if forbidden:
        columns = ", ".join(sorted(forbidden))
        raise ValueError(f"Artifact contains forbidden lookahead columns: {columns}")


def _validate_non_empty_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be stable and non-empty.")


def _validate_non_empty_series(series: pd.Series, name: str) -> None:
    values = series.astype("string")
    missing_or_empty = values.isna() | values.str.fullmatch(r"\s*")
    if missing_or_empty.any():
        raise ValueError(f"{name} must be non-empty for every row.")


def _validate_optional_numeric_series(series: pd.Series, name: str) -> None:
    _optional_numeric_series(series, name)


def _optional_numeric_series(series: pd.Series, name: str) -> pd.Series:
    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} must be numeric when present.")
    return pd.to_numeric(series, errors="coerce")


__all__ = [
    "MODEL_SIGNAL_OPTIONAL_COLUMNS",
    "MODEL_SIGNAL_REQUIRED_COLUMNS",
    "build_model_signal_artifact",
    "validate_model_signal_artifact",
]
