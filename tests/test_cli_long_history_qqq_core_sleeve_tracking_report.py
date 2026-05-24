from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_long_history_core_sleeve_tracking import _inputs

from us_market_ml.cli.create_long_history_qqq_core_sleeve_tracking_report import (
    run_long_history_qqq_core_sleeve_tracking_report_from_args,
)
from us_market_ml.long_history_core_sleeve_tracking import (
    DECISION_MATRIX_NAME,
    DRAWDOWN_WATCH_NAME,
    RECENT_WINDOWS_NAME,
    REPORT_NAME,
    RICH_REPLAY_PREFIX,
    STATUS_NAME,
    SUMMARY_NAME,
    SYMBOL_WATCH_NAME,
)


def test_cli_core_sleeve_tracking_report_writes_required_files(tmp_path: Path) -> None:
    inputs = _inputs()
    input_dir = tmp_path / "rich_replay"
    output_dir = tmp_path / "tracking"
    input_dir.mkdir()

    inputs.summary.to_csv(input_dir / f"{RICH_REPLAY_PREFIX}_summary.csv", index=False)
    inputs.daily_equity.to_csv(input_dir / f"{RICH_REPLAY_PREFIX}_daily_equity.csv", index=False)
    inputs.daily_drawdown.to_csv(
        input_dir / f"{RICH_REPLAY_PREFIX}_daily_drawdown.csv",
        index=False,
    )
    inputs.subperiods.to_csv(input_dir / f"{RICH_REPLAY_PREFIX}_subperiods.csv", index=False)
    inputs.rolling_windows.to_csv(
        input_dir / f"{RICH_REPLAY_PREFIX}_rolling_windows.csv",
        index=False,
    )
    inputs.symbol_exposure.to_csv(
        input_dir / f"{RICH_REPLAY_PREFIX}_symbol_exposure.csv",
        index=False,
    )
    inputs.selected_symbols.to_csv(
        input_dir / f"{RICH_REPLAY_PREFIX}_selected_symbols.csv",
        index=False,
    )
    inputs.decision_matrix.to_csv(
        input_dir / f"{RICH_REPLAY_PREFIX}_decision_matrix.csv",
        index=False,
    )

    exit_code = run_long_history_qqq_core_sleeve_tracking_report_from_args(
        [
            "--core-sleeve-rich-replay-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    for name in [
        SUMMARY_NAME,
        STATUS_NAME,
        RECENT_WINDOWS_NAME,
        DRAWDOWN_WATCH_NAME,
        SYMBOL_WATCH_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
    ]:
        assert (output_dir / name).exists()
    assert pd.read_csv(output_dir / SUMMARY_NAME)["production_ready"].eq(False).all()
    assert pd.read_csv(output_dir / DECISION_MATRIX_NAME)["production_ready"].eq(False).all()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "research-only tracking" in markdown
    assert "broad sweep" in markdown
    assert "production-ready" in markdown
