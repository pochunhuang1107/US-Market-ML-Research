"""Runtime-safe model signal artifact export helpers.

These helpers convert post-smoothed research signal frames into compact model
signal artifacts for downstream platform validation/loading. They intentionally
exclude realized, forward, target, and backtest-only columns.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

MODEL_SIGNAL_ARTIFACT_COLUMNS: list[str] = [
    "date",
    "symbol",
    "model_version",
    "score",
    "predicted_return",
    "rank",
    "signal_bucket",
    "feature_set_version",
    "label_horizon_days",
    "created_at",
]

REQUIRED_MODEL_SIGNAL_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "model_version",
    "score",
)

FORBIDDEN_ARTIFACT_COLUMN_TOKENS: tuple[str, ...] = (
    "future",
    "forward",
    "realized",
    "target",
    "excess_return_vs_",
    "rank_excess_return_vs_",
)


def build_model_signal_artifact(
    signals: pd.DataFrame,
    model_version: str,
    *,
    feature_set_version: str | None = None,
    label_horizon_days: int | None = None,
    created_at: str | None = None,
) -> pd.DataFrame:
    """Build a compact runtime-safe signal artifact from post-smoothed signals."""

    _validate_required_columns(signals, ["date", "symbol", "prediction"])
    if not model_version:
        raise ValueError("model_version must be non-empty.")

    artifact = pd.DataFrame(
        {
            "date": pd.to_datetime(signals["date"]),
            "symbol": signals["symbol"].astype(str),
            "model_version": model_version,
            "score": signals["prediction"],
            "predicted_return": signals["prediction"],
            "rank": _rank_column(signals),
            "signal_bucket": _signal_bucket_column(signals),
            "feature_set_version": feature_set_version,
            "label_horizon_days": label_horizon_days,
            "created_at": created_at,
        },
        columns=MODEL_SIGNAL_ARTIFACT_COLUMNS,
    )
    artifact = artifact.sort_values(
        by=["date", "rank", "symbol"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    validate_model_signal_artifact(artifact)
    return artifact


def validate_model_signal_artifact(artifact: pd.DataFrame) -> None:
    """Validate the v1 runtime-safe model signal artifact schema."""

    _validate_required_columns(artifact, REQUIRED_MODEL_SIGNAL_COLUMNS)
    forbidden_columns = _forbidden_runtime_columns(artifact.columns)
    if forbidden_columns:
        forbidden = ", ".join(forbidden_columns)
        raise ValueError(f"Model signal artifact contains forbidden runtime columns: {forbidden}")

    if artifact.empty:
        raise ValueError("Model signal artifact must contain at least one row.")
    if artifact["date"].isna().any():
        raise ValueError("Model signal artifact contains missing date values.")
    if artifact["symbol"].isna().any() or (artifact["symbol"].astype(str).str.len() == 0).any():
        raise ValueError("Model signal artifact contains missing symbol values.")
    if artifact["model_version"].isna().any():
        raise ValueError("Model signal artifact contains missing model_version values.")
    if artifact["score"].isna().any():
        raise ValueError("Model signal artifact contains missing score values.")

    duplicate_mask = artifact.duplicated(["date", "symbol", "model_version"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Model signal artifact contains duplicate date/symbol/model_version rows.")


def _rank_column(signals: pd.DataFrame) -> pd.Series:
    if "signal_rank" not in signals:
        return pd.Series(pd.NA, index=signals.index, dtype="Int64")
    return pd.to_numeric(signals["signal_rank"], errors="coerce").astype("Int64")


def _signal_bucket_column(signals: pd.DataFrame) -> pd.Series:
    if "selected" not in signals:
        return pd.Series("unselected", index=signals.index, dtype="object")
    selected = signals["selected"].fillna(False).astype(bool)
    return pd.Series("unselected", index=signals.index, dtype="object").mask(selected, "top")


def _validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")


def _forbidden_runtime_columns(columns: Sequence[str]) -> list[str]:
    allowed_columns = set(MODEL_SIGNAL_ARTIFACT_COLUMNS)
    forbidden: list[str] = []
    for column in columns:
        if column in allowed_columns:
            continue
        normalized = column.lower()
        if any(token in normalized for token in FORBIDDEN_ARTIFACT_COLUMN_TOKENS):
            forbidden.append(column)
    return forbidden
