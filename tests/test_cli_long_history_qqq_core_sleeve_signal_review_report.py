from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_long_history_core_sleeve_signal_review import _inputs

from us_market_ml.cli.create_long_history_qqq_core_sleeve_signal_review_report import (
    run_long_history_qqq_core_sleeve_signal_review_report_from_args,
)
from us_market_ml.long_history_core_sleeve_signal_review import (
    DECISION_MATRIX_NAME,
    DEEP_DIVE_PREFIX,
    DRAWDOWN_EVENTS_NAME,
    RECENT_WEAKNESS_NAME,
    REPORT_NAME,
    RICH_REPLAY_PREFIX,
    SUMMARY_NAME,
    SYMBOL_PATH_NAME,
    SYMBOL_REGIME_BREAKDOWN_NAME,
    TRACKING_PREFIX,
)


def test_cli_core_sleeve_signal_review_report_writes_required_files(tmp_path: Path) -> None:
    inputs = _inputs()
    tracking_dir = tmp_path / "tracking"
    rich_replay_dir = tmp_path / "rich_replay"
    deep_dive_dir = tmp_path / "deep_dive"
    output_dir = tmp_path / "signal_review"
    tracking_dir.mkdir()
    rich_replay_dir.mkdir()
    deep_dive_dir.mkdir()

    inputs.tracking_summary.to_csv(tracking_dir / f"{TRACKING_PREFIX}_summary.csv", index=False)
    inputs.tracking_status.to_csv(tracking_dir / f"{TRACKING_PREFIX}_status.csv", index=False)
    inputs.tracking_recent_windows.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_recent_windows.csv",
        index=False,
    )
    inputs.tracking_drawdown_watch.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_drawdown_watch.csv",
        index=False,
    )
    inputs.tracking_symbol_watch.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_symbol_watch.csv",
        index=False,
    )
    inputs.tracking_decision_matrix.to_csv(
        tracking_dir / f"{TRACKING_PREFIX}_decision_matrix.csv",
        index=False,
    )
    inputs.rich_replay_summary.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_summary.csv",
        index=False,
    )
    inputs.rich_replay_selected_symbols.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_selected_symbols.csv",
        index=False,
    )
    inputs.rich_replay_symbol_exposure.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_symbol_exposure.csv",
        index=False,
    )
    inputs.rich_replay_subperiods.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv",
        index=False,
    )
    inputs.rich_replay_rolling_windows.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv",
        index=False,
    )
    inputs.rich_replay_decision_matrix.to_csv(
        rich_replay_dir / f"{RICH_REPLAY_PREFIX}_decision_matrix.csv",
        index=False,
    )
    inputs.candidate_symbol_periods.to_csv(
        deep_dive_dir / f"{DEEP_DIVE_PREFIX}_symbol_periods.csv",
        index=False,
    )
    inputs.candidate_selected_symbols.to_csv(
        deep_dive_dir / f"{DEEP_DIVE_PREFIX}_selected_symbols.csv",
        index=False,
    )

    exit_code = run_long_history_qqq_core_sleeve_signal_review_report_from_args(
        [
            "--core-sleeve-tracking-dir",
            str(tracking_dir),
            "--core-sleeve-rich-replay-dir",
            str(rich_replay_dir),
            "--candidate-deep-dive-dir",
            str(deep_dive_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    for name in [
        SUMMARY_NAME,
        SYMBOL_PATH_NAME,
        SYMBOL_REGIME_BREAKDOWN_NAME,
        DRAWDOWN_EVENTS_NAME,
        RECENT_WEAKNESS_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
    ]:
        assert (output_dir / name).exists()
    assert pd.read_csv(output_dir / SUMMARY_NAME)["production_ready"].eq(False).all()
    assert pd.read_csv(output_dir / DECISION_MATRIX_NAME)["production_ready"].eq(False).all()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "research-only signal-path" in markdown
    assert "broad sweep" in markdown
    assert "production-ready" in markdown
