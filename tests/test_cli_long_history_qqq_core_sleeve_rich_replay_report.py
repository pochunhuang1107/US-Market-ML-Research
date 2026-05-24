from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_long_history_core_sleeve_rich_replay import _inputs

from us_market_ml.cli.create_long_history_qqq_core_sleeve_rich_replay_report import (
    run_long_history_qqq_core_sleeve_rich_replay_report_from_args,
)
from us_market_ml.long_history_core_sleeve_rich_replay import (
    DECISION_MATRIX_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
)


def test_cli_core_sleeve_rich_replay_report_writes_required_files(tmp_path: Path) -> None:
    inputs = _inputs()
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "report"
    input_dir.mkdir()
    paths = {
        "recommendations": input_dir / "recommendations.csv",
        "decision": input_dir / "decision.csv",
        "results": input_dir / "results.csv",
        "daily_equity": input_dir / "daily_equity.csv",
        "daily_drawdown": input_dir / "daily_drawdown.csv",
        "subperiods": input_dir / "subperiods.csv",
        "rolling": input_dir / "rolling.csv",
        "selected_symbols": input_dir / "selected_symbols.csv",
        "symbol_exposure": input_dir / "symbol_exposure.csv",
        "symbol_periods": input_dir / "symbol_periods.csv",
    }
    inputs.decision_recommendations.to_csv(paths["recommendations"], index=False)
    inputs.decision_matrix.to_csv(paths["decision"], index=False)
    inputs.refinement_results.to_csv(paths["results"], index=False)
    inputs.refinement_daily_equity.to_csv(paths["daily_equity"], index=False)
    pd.DataFrame([{"candidate_id": "candidate_0001"}]).to_csv(
        paths["daily_drawdown"],
        index=False,
    )
    inputs.refinement_subperiods.to_csv(paths["subperiods"], index=False)
    inputs.refinement_rolling_windows.to_csv(paths["rolling"], index=False)
    inputs.candidate_selected_symbols.to_csv(paths["selected_symbols"], index=False)
    inputs.candidate_symbol_exposure.to_csv(paths["symbol_exposure"], index=False)
    inputs.candidate_symbol_periods.to_csv(paths["symbol_periods"], index=False)

    exit_code = run_long_history_qqq_core_sleeve_rich_replay_report_from_args(
        [
            "--core-sleeve-decision-recommendations",
            str(paths["recommendations"]),
            "--core-sleeve-decision-matrix",
            str(paths["decision"]),
            "--core-sleeve-refinement-results",
            str(paths["results"]),
            "--core-sleeve-refinement-daily-equity",
            str(paths["daily_equity"]),
            "--core-sleeve-refinement-daily-drawdown",
            str(paths["daily_drawdown"]),
            "--core-sleeve-refinement-subperiods",
            str(paths["subperiods"]),
            "--core-sleeve-refinement-rolling-windows",
            str(paths["rolling"]),
            "--candidate-selected-symbols",
            str(paths["selected_symbols"]),
            "--candidate-symbol-exposure",
            str(paths["symbol_exposure"]),
            "--candidate-symbol-periods",
            str(paths["symbol_periods"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / SUMMARY_NAME).exists()
    assert (output_dir / DECISION_MATRIX_NAME).exists()
    assert (output_dir / REPORT_NAME).exists()
    assert pd.read_csv(output_dir / SUMMARY_NAME)["production_ready"].eq(False).all()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "Production promotion allowed: `no`" in markdown
    assert "broad sweep" in markdown


def test_no_cloud_model_card_or_production_export_paths_in_script() -> None:
    script = Path(
        "scripts/create_long_history_qqq_core_sleeve_rich_replay_report.sh"
    ).read_text(encoding="utf-8")

    assert "s3://" not in script.lower()
    assert "gs://" not in script.lower()
    assert "model_card" not in script
    assert "production signal export" in script
    assert "production-ready marking" in script
