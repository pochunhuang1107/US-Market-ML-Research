"""Create the long-history QQQ core-sleeve decision report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.long_history_core_sleeve_decision import (
    DECISION_MATRIX_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINEMENT_REPORT_DIR,
    RECOMMENDATIONS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    build_long_history_core_sleeve_decision_artifacts,
    read_long_history_core_sleeve_decision_inputs,
    write_long_history_core_sleeve_decision_artifacts,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REFINEMENT_SWEEP_DIR,
)


def run_long_history_qqq_core_sleeve_decision_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create or refresh the final decision report from existing artifacts."""

    args = _parse_args(argv)
    _ensure_local_only_path(str(args.refinement_sweep_dir), "--refinement-sweep-dir")
    _ensure_local_only_path(str(args.refinement_report_dir), "--refinement-report-dir")
    _ensure_local_only_path(str(args.output_dir), "--output-dir")

    inputs = read_long_history_core_sleeve_decision_inputs(
        refinement_sweep_dir=args.refinement_sweep_dir,
        refinement_report_dir=args.refinement_report_dir,
    )
    artifacts = build_long_history_core_sleeve_decision_artifacts(inputs)
    paths = write_long_history_core_sleeve_decision_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print("US Market ML long-history QQQ core-sleeve decision report")
    print("report/synthesis only: true")
    print("new sweep: false")
    print("model training: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"refinement sweep dir: {args.refinement_sweep_dir}")
    print(f"refinement report dir: {args.refinement_report_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"summary path: {paths[SUMMARY_NAME]}")
    print(f"decision matrix path: {paths[DECISION_MATRIX_NAME]}")
    print(f"recommendations path: {paths[RECOMMENDATIONS_NAME]}")
    print(f"report path: {paths[REPORT_NAME]}")
    print(f"input configs: {len(inputs.results)}")
    print(f"decision rows: {len(artifacts.decision_matrix)}")
    print(f"recommendation rows: {len(artifacts.recommendations)}")
    return 0


def _ensure_local_only_path(value: str, label: str) -> None:
    lowered = value.lower()
    blocked_prefixes = ("s3://", "gs://", "az://", "abfs://", "http://", "https://")
    if lowered.startswith(blocked_prefixes):
        raise ValueError(f"{label} must be a local filesystem path, got: {value}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the long-history QQQ core-sleeve decision report from completed "
            "refinement artifacts."
        )
    )
    parser.add_argument(
        "--refinement-sweep-dir",
        type=Path,
        default=DEFAULT_REFINEMENT_SWEEP_DIR,
    )
    parser.add_argument(
        "--refinement-report-dir",
        type=Path,
        default=DEFAULT_REFINEMENT_REPORT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_core_sleeve_decision_report_from_args())
