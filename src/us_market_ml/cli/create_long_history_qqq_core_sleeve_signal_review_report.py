"""Create the long-history QQQ core-sleeve signal-path review report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.long_history_core_sleeve_signal_review import (
    DECISION_MATRIX_NAME,
    DEFAULT_CANDIDATE_DEEP_DIVE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RICH_REPLAY_DIR,
    DEFAULT_TRACKING_DIR,
    DRAWDOWN_EVENTS_NAME,
    RECENT_WEAKNESS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    SYMBOL_PATH_NAME,
    SYMBOL_REGIME_BREAKDOWN_NAME,
    build_long_history_core_sleeve_signal_review_artifacts,
    read_long_history_core_sleeve_signal_review_inputs,
    write_long_history_core_sleeve_signal_review_artifacts,
)


def run_long_history_qqq_core_sleeve_signal_review_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the signal-path review from existing local artifacts."""

    args = _parse_args(argv)
    for label, value in [
        ("--core-sleeve-tracking-dir", args.core_sleeve_tracking_dir),
        ("--core-sleeve-rich-replay-dir", args.core_sleeve_rich_replay_dir),
        ("--candidate-deep-dive-dir", args.candidate_deep_dive_dir),
        ("--output-dir", args.output_dir),
    ]:
        _ensure_local_only_path(str(value), label)

    inputs = read_long_history_core_sleeve_signal_review_inputs(
        tracking_dir=args.core_sleeve_tracking_dir,
        rich_replay_dir=args.core_sleeve_rich_replay_dir,
        candidate_deep_dive_dir=args.candidate_deep_dive_dir,
    )
    artifacts = build_long_history_core_sleeve_signal_review_artifacts(inputs)
    paths = write_long_history_core_sleeve_signal_review_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print("US Market ML long-history QQQ core-sleeve signal-path review report")
    print("research-only diagnostic: true")
    print("local-only: true")
    print("reads existing artifacts only: true")
    print("broad sweep: false")
    print("model training: false")
    print("MLP batch: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"tracking dir: {args.core_sleeve_tracking_dir}")
    print(f"rich replay dir: {args.core_sleeve_rich_replay_dir}")
    print(f"candidate deep dive dir: {args.candidate_deep_dive_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"summary path: {paths[SUMMARY_NAME]}")
    print(f"symbol path: {paths[SYMBOL_PATH_NAME]}")
    print(f"symbol regime breakdown path: {paths[SYMBOL_REGIME_BREAKDOWN_NAME]}")
    print(f"drawdown events path: {paths[DRAWDOWN_EVENTS_NAME]}")
    print(f"recent weakness path: {paths[RECENT_WEAKNESS_NAME]}")
    print(f"decision matrix path: {paths[DECISION_MATRIX_NAME]}")
    print(f"report path: {paths[REPORT_NAME]}")
    print(f"summary rows: {len(artifacts.summary)}")
    print(f"symbol regime rows: {len(artifacts.symbol_regime_breakdown)}")
    print(f"production-ready rows: {int(artifacts.decision_matrix['production_ready'].sum())}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the long-history QQQ core-sleeve signal-path review report from "
            "completed tracking, rich replay, and candidate deep-dive artifacts."
        )
    )
    parser.add_argument(
        "--core-sleeve-tracking-dir",
        type=Path,
        default=Path(os.environ.get("CORE_SLEEVE_TRACKING_DIR", str(DEFAULT_TRACKING_DIR))),
    )
    parser.add_argument(
        "--core-sleeve-rich-replay-dir",
        type=Path,
        default=Path(
            os.environ.get("CORE_SLEEVE_RICH_REPLAY_DIR", str(DEFAULT_RICH_REPLAY_DIR))
        ),
    )
    parser.add_argument(
        "--candidate-deep-dive-dir",
        type=Path,
        default=Path(
            os.environ.get("CANDIDATE_DEEP_DIVE_DIR", str(DEFAULT_CANDIDATE_DEEP_DIVE_DIR))
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
    raise SystemExit(run_long_history_qqq_core_sleeve_signal_review_report_from_args())
