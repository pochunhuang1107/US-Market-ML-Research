from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_long_history_core_sleeve_visual_deep_dive import _inputs, _write_input_dirs

from us_market_ml.cli.create_long_history_qqq_core_sleeve_visual_deep_dive import (
    run_long_history_qqq_core_sleeve_visual_deep_dive_from_args,
)
from us_market_ml.long_history_core_sleeve_visual_deep_dive import (
    CHART_NAMES,
    REPORT_NAME,
    SUMMARY_NAME,
)


def test_cli_core_sleeve_visual_deep_dive_writes_required_files(tmp_path: Path) -> None:
    inputs = _inputs()
    rich_dir = tmp_path / "rich_replay"
    signal_dir = tmp_path / "signal_review"
    tracking_dir = tmp_path / "tracking"
    output_dir = tmp_path / "visual"
    _write_input_dirs(inputs, rich_dir, signal_dir, tracking_dir)

    exit_code = run_long_history_qqq_core_sleeve_visual_deep_dive_from_args(
        [
            "--core-sleeve-rich-replay-dir",
            str(rich_dir),
            "--core-sleeve-signal-review-dir",
            str(signal_dir),
            "--core-sleeve-tracking-dir",
            str(tracking_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / SUMMARY_NAME).exists()
    assert (output_dir / REPORT_NAME).exists()
    for name in CHART_NAMES:
        assert (output_dir / name).exists()
    assert pd.read_csv(output_dir / SUMMARY_NAME)["production_ready"].eq(False).all()
    markdown = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "research-only visual report" in markdown
    assert "broad sweep" in markdown
    assert "production-ready" in markdown
