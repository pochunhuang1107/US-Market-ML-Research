"""Create the long-history QQQ core-sleeve visual deep-dive report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.long_history_core_sleeve_visual_deep_dive import (
    CHART_NAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RICH_REPLAY_DIR,
    DEFAULT_SIGNAL_REVIEW_DIR,
    DEFAULT_TRACKING_DIR,
    REPORT_NAME,
    SUMMARY_NAME,
    build_long_history_core_sleeve_visual_deep_dive_artifacts,
    read_long_history_core_sleeve_visual_deep_dive_inputs,
    write_long_history_core_sleeve_visual_deep_dive_artifacts,
)


def run_long_history_qqq_core_sleeve_visual_deep_dive_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the visual deep dive from existing local artifacts."""

    args = _parse_args(argv)
    for label, value in [
        ("--core-sleeve-rich-replay-dir", args.core_sleeve_rich_replay_dir),
        ("--core-sleeve-signal-review-dir", args.core_sleeve_signal_review_dir),
        ("--core-sleeve-tracking-dir", args.core_sleeve_tracking_dir),
        ("--output-dir", args.output_dir),
    ]:
        _ensure_local_only_path(str(value), label)

    inputs = read_long_history_core_sleeve_visual_deep_dive_inputs(
        rich_replay_dir=args.core_sleeve_rich_replay_dir,
        signal_review_dir=args.core_sleeve_signal_review_dir,
        tracking_dir=args.core_sleeve_tracking_dir,
    )
    artifacts = build_long_history_core_sleeve_visual_deep_dive_artifacts(inputs)
    paths = write_long_history_core_sleeve_visual_deep_dive_artifacts(
        artifacts=artifacts,
        inputs=inputs,
        output_dir=args.output_dir,
    )

    print("US Market ML long-history QQQ core-sleeve visual deep-dive report")
    print("research-only visual report: true")
    print("local-only: true")
    print("reads existing artifacts only: true")
    print("broad sweep: false")
    print("model training: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"rich replay dir: {args.core_sleeve_rich_replay_dir}")
    print(f"signal review dir: {args.core_sleeve_signal_review_dir}")
    print(f"tracking dir: {args.core_sleeve_tracking_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"summary path: {paths[SUMMARY_NAME]}")
    print(f"report path: {paths[REPORT_NAME]}")
    for name in CHART_NAMES:
        print(f"chart path: {paths[name]}")
    print(f"summary rows: {len(artifacts.summary)}")
    print(f"chart files: {len(CHART_NAMES)}")
    print(f"production-ready rows: {int(artifacts.summary['production_ready'].sum())}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the long-history QQQ core-sleeve visual deep-dive report from "
            "completed tracking, signal-review, and rich-replay artifacts."
        )
    )
    parser.add_argument(
        "--core-sleeve-rich-replay-dir",
        type=Path,
        default=Path(os.environ.get("CORE_SLEEVE_RICH_REPLAY_DIR", str(DEFAULT_RICH_REPLAY_DIR))),
    )
    parser.add_argument(
        "--core-sleeve-signal-review-dir",
        type=Path,
        default=Path(
            os.environ.get("CORE_SLEEVE_SIGNAL_REVIEW_DIR", str(DEFAULT_SIGNAL_REVIEW_DIR))
        ),
    )
    parser.add_argument(
        "--core-sleeve-tracking-dir",
        type=Path,
        default=Path(os.environ.get("CORE_SLEEVE_TRACKING_DIR", str(DEFAULT_TRACKING_DIR))),
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
    raise SystemExit(run_long_history_qqq_core_sleeve_visual_deep_dive_from_args())
