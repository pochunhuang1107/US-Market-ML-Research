from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_market_ml.qqq_only_cross_market_timing_model_zoo import QQQOnlySpecOutput
from us_market_ml.qqq_only_rf_paper_tracking import (
    CANDIDATE_IDS,
    CANDIDATE_REGISTRY_JSON_NAME,
    CANDIDATE_REGISTRY_NAME,
    INITIAL_SIGNAL_COLUMNS,
    INITIAL_SIGNAL_NAME,
    PRIMARY_CANDIDATE_ID,
    REPORT_NAME,
    build_qqq_only_rf_paper_tracking_candidate_registry,
)
from us_market_ml.qqq_only_rf_paper_tracking_update import (
    UPDATE_REPORT_SECTIONS,
    UPDATE_SIGNAL_COLUMNS,
    build_qqq_only_rf_paper_tracking_update_artifacts,
    read_locked_paper_tracking_registry,
    report_output_paths,
    write_qqq_only_rf_paper_tracking_update_artifacts,
)


def test_update_reads_locked_registry_and_rejects_changed_rows(tmp_path: Path) -> None:
    package_dir = _write_package_inputs(tmp_path)

    registry = read_locked_paper_tracking_registry(package_dir)

    assert tuple(registry["candidate_id"]) == CANDIDATE_IDS
    assert registry["production_ready"].eq(False).all()

    changed = registry.copy()
    changed.loc[0, "threshold"] = 0.55
    changed.to_csv(package_dir / CANDIDATE_REGISTRY_NAME, index=False)

    with pytest.raises(ValueError, match="threshold"):
        read_locked_paper_tracking_registry(package_dir)


def test_update_artifacts_required_outputs_guardrails_and_history(tmp_path: Path) -> None:
    package_dir = _write_package_inputs(tmp_path)
    output_dir = tmp_path / "updates"
    registry = read_locked_paper_tracking_registry(package_dir)

    artifacts = build_qqq_only_rf_paper_tracking_update_artifacts(
        locked_registry=registry,
        replay_outputs=_replay_outputs(primary_exposure=1.0),
        package_dir=package_dir,
        output_dir=output_dir,
    )

    assert set(UPDATE_SIGNAL_COLUMNS).issubset(artifacts.update_signal.columns)
    assert artifacts.update_signal["production_ready"].eq(False).all()
    assert artifacts.update_summary["production_ready"].eq(False).all()
    assert artifacts.update_decision_matrix["production_ready"].eq(False).all()
    assert set(artifacts.update_signal["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})

    primary = artifacts.update_signal.set_index("candidate_id").loc[PRIMARY_CANDIDATE_ID]
    assert primary["previous_effective_exposure"] == 0.0
    assert bool(primary["exposure_changed"]) is True
    assert "exposure_change_watch" in set(artifacts.update_decision_matrix["decision_label"])

    for section in UPDATE_REPORT_SECTIONS:
        assert section in artifacts.report_markdown
    assert "No production signal export" in artifacts.report_markdown
    assert "No model card" in artifacts.report_markdown
    assert len(artifacts.update_history) == 3

    paths = write_qqq_only_rf_paper_tracking_update_artifacts(
        artifacts=artifacts,
        output_dir=output_dir,
    )

    assert paths.keys() == report_output_paths(output_dir).keys()
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
        assert "artifacts/model_signals" not in path.as_posix()

    rerun = build_qqq_only_rf_paper_tracking_update_artifacts(
        locked_registry=registry,
        replay_outputs=_replay_outputs(primary_exposure=1.0),
        package_dir=package_dir,
        output_dir=output_dir,
    )
    assert len(rerun.update_history) == 3

    next_run = build_qqq_only_rf_paper_tracking_update_artifacts(
        locked_registry=registry,
        replay_outputs=_replay_outputs(primary_exposure=0.0, date="2020-01-13"),
        package_dir=package_dir,
        output_dir=output_dir,
    )
    assert len(next_run.update_history) == 6


def test_update_rejects_model_signal_output_path(tmp_path: Path) -> None:
    package_dir = _write_package_inputs(tmp_path)
    registry = read_locked_paper_tracking_registry(package_dir)

    with pytest.raises(ValueError, match="artifacts/model_signals"):
        build_qqq_only_rf_paper_tracking_update_artifacts(
            locked_registry=registry,
            replay_outputs=_replay_outputs(),
            package_dir=package_dir,
            output_dir=Path("artifacts/model_signals/qqq_only_rf_paper_tracking_updates"),
        )


def test_update_rejects_non_qqq_cash_holdings(tmp_path: Path) -> None:
    package_dir = _write_package_inputs(tmp_path)
    registry = read_locked_paper_tracking_registry(package_dir)
    outputs = _replay_outputs()
    outputs[PRIMARY_CANDIDATE_ID].exposure_log.loc[:, "held_assets"] = "SPY"

    with pytest.raises(ValueError, match="non-QQQ/cash"):
        build_qqq_only_rf_paper_tracking_update_artifacts(
            locked_registry=registry,
            replay_outputs=outputs,
            package_dir=package_dir,
            output_dir=tmp_path / "updates",
        )


def _write_package_inputs(tmp_path: Path) -> Path:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    registry.to_csv(package_dir / CANDIDATE_REGISTRY_NAME, index=False)
    registry.to_json(package_dir / CANDIDATE_REGISTRY_JSON_NAME, orient="records", indent=2)
    _initial_signal().to_csv(package_dir / INITIAL_SIGNAL_NAME, index=False)
    (package_dir / REPORT_NAME).write_text("# package\n", encoding="utf-8")
    return package_dir


def _initial_signal() -> pd.DataFrame:
    rows = []
    for candidate_id in CANDIDATE_IDS:
        exposure = 0.0 if candidate_id == PRIMARY_CANDIDATE_ID else 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "as_of_date": "2020-01-03",
                "latest_available_price_date": "2020-01-03",
                "target_exposure": exposure,
                "effective_exposure": exposure,
                "qqq_weight": exposure,
                "cash_weight": 1.0 - exposure,
                "held_assets": "QQQ" if exposure else "cash",
                "raw_score": 0.60 if exposure else 0.40,
                "threshold": 0.50,
                "rebalance_weekday": "FRI",
                "is_rebalance_day": True,
                "next_action_interpretation": "initial research-only snapshot",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=INITIAL_SIGNAL_COLUMNS)


def _replay_outputs(
    *,
    primary_exposure: float = 1.0,
    date: str = "2020-01-10",
) -> dict[str, QQQOnlySpecOutput]:
    exposures = {
        CANDIDATE_IDS[0]: primary_exposure,
        CANDIDATE_IDS[1]: 1.0,
        CANDIDATE_IDS[2]: 0.0,
    }
    return {
        candidate_id: _output(candidate_id=candidate_id, exposure=exposure, date=date)
        for candidate_id, exposure in exposures.items()
    }


def _output(*, candidate_id: str, exposure: float, date: str) -> QQQOnlySpecOutput:
    held = "QQQ" if np.isclose(exposure, 1.0) else "cash"
    exposure_log = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "strategy_id": candidate_id,
                "strategy_name": candidate_id,
                "model_type": "random_forest",
                "training_mode": "rolling_5y",
                "target_name": "risk_on_20d",
                "feature_set_name": "qqq_self_cross_market",
                "exposure_mode": "binary_0_100",
                "rebalance_frequency": "weekly",
                "target_exposure": exposure,
                "effective_exposure": exposure,
                "qqq_weight": exposure,
                "cash_weight": 1.0 - exposure,
                "held_assets": held,
                "raw_score": 0.70 if exposure else 0.30,
                "threshold": 0.50,
            },
            {
                "date": date,
                "strategy_id": candidate_id,
                "strategy_name": candidate_id,
                "model_type": "random_forest",
                "training_mode": "rolling_5y",
                "target_name": "risk_on_20d",
                "feature_set_name": "qqq_self_cross_market",
                "exposure_mode": "binary_0_100",
                "rebalance_frequency": "weekly",
                "target_exposure": exposure,
                "effective_exposure": exposure,
                "qqq_weight": exposure,
                "cash_weight": 1.0 - exposure,
                "held_assets": held,
                "raw_score": 0.70 if exposure else 0.30,
                "threshold": 0.50,
            },
        ]
    )
    return QQQOnlySpecOutput(
        result={"strategy_id": candidate_id, "production_ready": False},
        daily_equity=pd.DataFrame(),
        daily_drawdown=pd.DataFrame(),
        exposure_log=exposure_log,
    )
