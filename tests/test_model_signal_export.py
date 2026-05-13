from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.exports import build_model_signal_artifact, validate_model_signal_artifact


def test_build_model_signal_artifact_maps_prediction_to_score_and_predicted_return() -> None:
    signals = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01"],
            "symbol": ["SPY", "QQQ"],
            "prediction": [0.03, 0.01],
            "signal_rank": [1, pd.NA],
            "selected": [True, False],
        }
    )

    artifact = build_model_signal_artifact(
        signals,
        model_version="linear_smooth10_margin002_v1",
        feature_set_version="default_features_v1",
        label_horizon_days=20,
        created_at="2026-05-13T00:00:00+00:00",
    )

    assert artifact["score"].tolist() == [0.03, 0.01]
    assert artifact["predicted_return"].tolist() == [0.03, 0.01]
    assert artifact["model_version"].tolist() == [
        "linear_smooth10_margin002_v1",
        "linear_smooth10_margin002_v1",
    ]
    assert artifact["feature_set_version"].tolist() == [
        "default_features_v1",
        "default_features_v1",
    ]
    assert artifact["label_horizon_days"].tolist() == [20, 20]
    assert artifact["created_at"].tolist() == [
        "2026-05-13T00:00:00+00:00",
        "2026-05-13T00:00:00+00:00",
    ]


def test_build_model_signal_artifact_creates_top_other_bucket_from_selected() -> None:
    signals = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01"],
            "symbol": ["SPY", "QQQ"],
            "prediction": [0.02, 0.01],
            "selected": [True, False],
        }
    )

    artifact = build_model_signal_artifact(
        signals,
        model_version="model_v1",
        created_at="2026-05-13T00:00:00+00:00",
    )

    assert artifact["signal_bucket"].tolist() == ["top", "other"]


def test_validate_model_signal_artifact_catches_missing_required_columns() -> None:
    df = pd.DataFrame({"date": ["2021-01-01"], "symbol": ["SPY"], "score": [0.1]})

    with pytest.raises(ValueError, match="model_version"):
        validate_model_signal_artifact(df)


def test_validate_model_signal_artifact_catches_duplicate_keys() -> None:
    df = _valid_artifact()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date/symbol/model_version"):
        validate_model_signal_artifact(df)


def test_validate_model_signal_artifact_catches_forbidden_lookahead_columns() -> None:
    df = _valid_artifact().assign(future_return_20d=[0.1, 0.2])

    with pytest.raises(ValueError, match="forbidden lookahead"):
        validate_model_signal_artifact(df)


def test_validate_model_signal_artifact_catches_non_numeric_score() -> None:
    df = _valid_artifact()
    df["score"] = df["score"].astype(object)
    df.loc[0, "score"] = "not-a-number"

    with pytest.raises(ValueError, match="score must be numeric"):
        validate_model_signal_artifact(df)


def test_validate_model_signal_artifact_catches_confidence_outside_zero_to_one() -> None:
    df = _valid_artifact().assign(confidence=[0.9, 1.2])

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        validate_model_signal_artifact(df)


def _valid_artifact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01"],
            "symbol": ["SPY", "QQQ"],
            "model_version": ["model_v1", "model_v1"],
            "score": [0.1, 0.2],
        }
    )
