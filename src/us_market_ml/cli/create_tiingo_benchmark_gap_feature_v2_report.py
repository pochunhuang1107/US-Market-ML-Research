"""Create the Tiingo benchmark-gap v2 feature study report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from us_market_ml.cli.create_tiingo_benchmark_gap_feature_report import (
    BenchmarkGapFeatureReportData,
    build_tiingo_benchmark_gap_feature_report,
    render_tiingo_benchmark_gap_feature_markdown,
)
from us_market_ml.cli.run_tiingo_benchmark_gap_feature_study import DEFAULT_DATASET_GLOB

DEFAULT_RESULTS_CSV = Path(
    "artifacts/sweeps/tiingo_benchmark_gap_feature_study_v2/"
    "tiingo_benchmark_gap_feature_study_v2_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tiingo_benchmark_gap_feature_study_v2")

SUMMARY_PATH_NAME = "tiingo_benchmark_gap_feature_v2_summary.csv"
DECISION_MATRIX_PATH_NAME = "tiingo_benchmark_gap_feature_v2_decision_matrix.csv"
REPORT_PATH_NAME = "tiingo_benchmark_gap_feature_v2_report.md"
OUTPUT_PATH_NAMES = (SUMMARY_PATH_NAME, DECISION_MATRIX_PATH_NAME, REPORT_PATH_NAME)


def run_tiingo_benchmark_gap_feature_v2_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Build and write the benchmark-gap v2 feature study report."""

    args = _parse_args(argv)
    report_data = build_tiingo_benchmark_gap_feature_report(
        results_csv=args.results_csv,
        market_dataset_glob=args.market_dataset_glob,
    )
    outputs = write_tiingo_benchmark_gap_feature_v2_report(
        report_data=report_data,
        output_dir=args.output_dir,
    )

    print("US Market ML Tiingo benchmark-gap feature v2 study report")
    print(f"dataset source: {report_data.dataset_source}")
    print(f"results CSV: {args.results_csv}")
    print(f"market dataset: {args.market_dataset_glob}")
    for output_name in OUTPUT_PATH_NAMES:
        print(f"{output_name}: {outputs[output_name]}")
    print()
    print(report_data.decision_matrix.head(20).to_string(index=False))
    return 0


def write_tiingo_benchmark_gap_feature_v2_report(
    *,
    report_data: BenchmarkGapFeatureReportData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write v2 summary CSVs and Markdown, returning output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        SUMMARY_PATH_NAME: output_dir / SUMMARY_PATH_NAME,
        DECISION_MATRIX_PATH_NAME: output_dir / DECISION_MATRIX_PATH_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    report_data.feature_summary.to_csv(outputs[SUMMARY_PATH_NAME], index=False)
    report_data.decision_matrix.to_csv(outputs[DECISION_MATRIX_PATH_NAME], index=False)
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_benchmark_gap_feature_v2_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def render_tiingo_benchmark_gap_feature_v2_markdown(
    report_data: BenchmarkGapFeatureReportData,
) -> str:
    """Render the v2-flavored benchmark-gap feature study Markdown report."""

    return render_tiingo_benchmark_gap_feature_markdown(report_data).replace(
        "# Tiingo Benchmark-Gap Feature Study Report",
        "# Tiingo Benchmark-Gap Feature V2 Study Report",
        1,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Tiingo benchmark-gap v2 feature study report."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    return parser.parse_args(argv)


create_tiingo_benchmark_gap_feature_v2_report_from_args = (
    run_tiingo_benchmark_gap_feature_v2_report_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_benchmark_gap_feature_v2_report_from_args())
