"""Create the long-history QQQ core-sleeve tracking report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.long_history_core_sleeve_tracking import (
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RICH_REPLAY_DIR,
    DRAWDOWN_WATCH_NAME,
    RECENT_WINDOWS_NAME,
    REPORT_NAME,
    STATUS_NAME,
    SUMMARY_NAME,
    SYMBOL_WATCH_NAME,
    build_long_history_core_sleeve_tracking_artifacts,
    read_long_history_core_sleeve_tracking_inputs,
    write_long_history_core_sleeve_tracking_artifacts,
)


def run_long_history_qqq_core_sleeve_tracking_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the tracking report from existing local artifacts."""

    args = _parse_args(argv)
    _ensure_local_only_path(str(args.core_sleeve_rich_replay_dir), "--core-sleeve-rich-replay-dir")
    _ensure_local_only_path(str(args.output_dir), "--output-dir")

    inputs = read_long_history_core_sleeve_tracking_inputs(
        rich_replay_dir=args.core_sleeve_rich_replay_dir,
    )
    artifacts = build_long_history_core_sleeve_tracking_artifacts(inputs)
    paths = write_long_history_core_sleeve_tracking_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print("US Market ML long-history QQQ core-sleeve tracking report")
    print("report/tracking only: true")
    print("local-only: true")
    print("broad sweep: false")
    print("model training: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"rich replay dir: {args.core_sleeve_rich_replay_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"summary path: {paths[SUMMARY_NAME]}")
    print(f"status path: {paths[STATUS_NAME]}")
    print(f"recent windows path: {paths[RECENT_WINDOWS_NAME]}")
    print(f"drawdown watch path: {paths[DRAWDOWN_WATCH_NAME]}")
    print(f"symbol watch path: {paths[SYMBOL_WATCH_NAME]}")
    print(f"decision matrix path: {paths[DECISION_MATRIX_NAME]}")
    print(f"report path: {paths[REPORT_NAME]}")
    print(f"summary rows: {len(artifacts.summary)}")
    print(f"recent-window rows: {len(artifacts.recent_windows)}")
    print(f"production-ready rows: {int(artifacts.summary['production_ready'].sum())}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the long-history QQQ core-sleeve tracking report from completed "
            "rich replay artifacts."
        )
    )
    parser.add_argument(
        "--core-sleeve-rich-replay-dir",
        type=Path,
        default=Path(
            os.environ.get("CORE_SLEEVE_RICH_REPLAY_DIR", str(DEFAULT_RICH_REPLAY_DIR))
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    return parser.parse_args(argv)


def _ensure_local_only_path(value: str, label: str) -> None:
    lowered = value.lower()
    blocked_prefixes = ("s3://", "gs://", "az://", "abfs://", "http://", "https://")
    if lowered.startswith(blocked_prefixes):
        raise ValueError(f"{label} must be a local filesystem path, got: {value}")


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_core_sleeve_tracking_report_from_args())
