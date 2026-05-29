from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_qqq_only_cross_market_timing_model_zoo import (
    _synthetic_market_data,
    _synthetic_reference_wide,
)
from test_qqq_only_rf_out_of_time_confirmation import _out_of_time_inputs

from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    build_qqq_only_feature_dataset,
    run_qqq_only_spec,
)
from us_market_ml.qqq_only_rf_out_of_time_confirmation import (
    build_qqq_only_rf_out_of_time_confirmation_artifacts,
)
from us_market_ml.qqq_only_rf_paper_tracking import (
    CANDIDATE_IDS,
    INITIAL_SIGNAL_COLUMNS,
    PRIMARY_CANDIDATE_ID,
    REJECTED_REFERENCE_ID,
    REPORT_SECTIONS,
    SECONDARY_CANDIDATE_ID,
    build_qqq_only_rf_paper_tracking_artifacts,
    build_qqq_only_rf_paper_tracking_candidate_registry,
    build_qqq_only_rf_paper_tracking_specs,
    report_output_paths,
    write_qqq_only_rf_paper_tracking_artifacts,
)


def test_candidate_registry_contains_exact_locked_rows() -> None:
    registry = build_qqq_only_rf_paper_tracking_candidate_registry()

    assert tuple(registry["candidate_id"]) == CANDIDATE_IDS
    assert registry["production_ready"].eq(False).all()
    assert set(registry["tradable_holdings"]) == {"QQQ,cash"}
    assert set(registry["other_etfs_role"]) == {"features_only"}

    primary = registry.set_index("candidate_id").loc[PRIMARY_CANDIDATE_ID]
    secondary = registry.set_index("candidate_id").loc[SECONDARY_CANDIDATE_ID]
    rejected = registry.set_index("candidate_id").loc[REJECTED_REFERENCE_ID]

    assert primary["role"] == "primary_paper_tracking_candidate"
    assert primary["threshold"] == 0.50
    assert primary["min_holding_days"] == 0
    assert secondary["role"] == "secondary_paper_tracking_reference"
    assert secondary["threshold"] == 0.45
    assert secondary["min_holding_days"] == 10
    assert rejected["role"] == "rejected_out_of_time_reference"
    assert rejected["threshold"] == 0.45
    assert rejected["min_holding_days"] == 0
    assert "deprioritized" in rejected["status"]


def test_specs_are_locked_to_cost_zero_replay_versions() -> None:
    specs = build_qqq_only_rf_paper_tracking_specs()

    assert len(specs) == 3
    assert [spec.threshold for spec in specs] == [0.50, 0.45, 0.45]
    assert [spec.min_holding_days for spec in specs] == [0, 10, 0]
    assert {spec.transaction_cost_bps for spec in specs} == {0.0}
    assert {spec.exposure_mode for spec in specs} == {"binary_0_100"}
    assert {spec.rebalance_weekday for spec in specs} == {"FRI"}


def test_paper_tracking_artifacts_write_required_outputs_and_guardrails(
    tmp_path: Path,
) -> None:
    artifacts = _paper_tracking_artifacts()

    assert tuple(artifacts.candidate_registry["candidate_id"]) == CANDIDATE_IDS
    assert tuple(artifacts.initial_signal["candidate_id"]) == CANDIDATE_IDS
    assert set(INITIAL_SIGNAL_COLUMNS).issubset(artifacts.initial_signal.columns)
    assert artifacts.initial_signal["production_ready"].eq(False).all()
    assert artifacts.replay_summary["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert set(artifacts.initial_signal["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert set(artifacts.replay_exposure_log["held_assets"]).issubset(
        {"QQQ", "cash", "QQQ,cash"}
    )
    assert "SPY" not in set(artifacts.initial_signal["held_assets"])

    rejected_rows = artifacts.decision_matrix.loc[
        artifacts.decision_matrix["candidate_id"].eq(REJECTED_REFERENCE_ID)
    ]
    assert set(np.round(rejected_rows["cost_view_bps"].astype(float), 6)) == {
        0.0,
        10.0,
        20.0,
        30.0,
    }
    assert rejected_rows["out_of_time_decision_label"].eq("reject_deprioritize").all()
    assert rejected_rows["package_decision_label"].eq("rejected_deprioritized_reference").all()

    for section in REPORT_SECTIONS:
        assert section in artifacts.report_markdown
    assert "No production signal export" in artifacts.report_markdown
    assert "No model card" in artifacts.report_markdown

    paths = write_qqq_only_rf_paper_tracking_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    assert paths.keys() == report_output_paths(tmp_path).keys()
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
        assert "artifacts/model_signals" not in path.as_posix()

    registry_json = tmp_path / "qqq_only_rf_paper_tracking_candidate_registry.json"
    assert PRIMARY_CANDIDATE_ID in registry_json.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts/model_signals"):
        write_qqq_only_rf_paper_tracking_artifacts(
            artifacts=artifacts,
            output_dir=Path("artifacts/model_signals/qqq_only_rf_paper_tracking"),
        )


def _paper_tracking_artifacts():
    results, daily_equity, daily_drawdown, exposure_log = _out_of_time_inputs()
    out_of_time = build_qqq_only_rf_out_of_time_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )
    dataset = build_qqq_only_feature_dataset(
        _synthetic_market_data(),
        _synthetic_reference_wide(),
        start_date="2019-01-01",
        end_date="2020-06-30",
    )
    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    specs = build_qqq_only_rf_paper_tracking_specs()
    cache: dict[tuple[object, ...], pd.DataFrame] = {}
    replay_outputs = {}
    for candidate_id, spec in zip(registry["candidate_id"].astype(str), specs, strict=True):
        replay_outputs[candidate_id] = run_qqq_only_spec(
            dataset,
            spec,
            start_date="2019-01-01",
            end_date="2020-06-30",
            prediction_cache=cache,
            min_train_rows=40,
        )
    return build_qqq_only_rf_paper_tracking_artifacts(
        out_of_time_summary=out_of_time.summary,
        out_of_time_decision_matrix=out_of_time.decision_matrix,
        out_of_time_results=results,
        out_of_time_exposure_log=exposure_log,
        replay_outputs=replay_outputs,
    )
