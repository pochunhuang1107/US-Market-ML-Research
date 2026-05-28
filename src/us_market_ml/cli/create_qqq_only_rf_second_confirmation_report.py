"""Create the QQQ-only RF second-confirmation report from sweep artifacts."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.qqq_only_rf_second_confirmation import (
    DEFAULT_REPORT_OUTPUT_DIR,
    OUTPUT_PREFIX,
    REPORT_SECTIONS,
    build_qqq_only_rf_second_confirmation_artifacts,
    report_output_paths,
    write_qqq_only_rf_second_confirmation_report_artifacts,
)

DEFAULT_SWEEP_DIR = Path("artifacts/sweeps/qqq_only_rf_second_confirmation")
DEFAULT_RESULTS = DEFAULT_SWEEP_DIR / "qqq_only_rf_second_confirmation_results.csv"
DEFAULT_DAILY_EQUITY = DEFAULT_SWEEP_DIR / "qqq_only_rf_second_confirmation_daily_equity.csv"
DEFAULT_DAILY_DRAWDOWN = DEFAULT_SWEEP_DIR / "qqq_only_rf_second_confirmation_daily_drawdown.csv"
DEFAULT_EXPOSURE_LOG = DEFAULT_SWEEP_DIR / "qqq_only_rf_second_confirmation_exposure_log.csv"


def create_qqq_only_rf_second_confirmation_report_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create the local RF second-confirmation report from completed sweep artifacts."""

    args = _parse_args(argv)
    for flag_name, path in (
        ("--results", args.results),
        ("--daily-equity", args.daily_equity),
        ("--daily-drawdown", args.daily_drawdown),
        ("--exposure-log", args.exposure_log),
    ):
        _ensure_local_file(path, flag_name)
    _ensure_local_only_path(str(args.output_dir), "--output-dir")

    print("US Market ML QQQ-only RF second-confirmation report")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("reads completed sweep artifacts: true")
    print("new broad model zoo: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"dry_run: {args.dry_run}")
    print(f"output directory: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / f"{OUTPUT_PREFIX}_report_preflight.md"
    preflight_path.write_text(_render_preflight(args), encoding="utf-8")
    print(f"preflight Markdown path: {preflight_path}")
    if args.dry_run:
        return 0

    results = pd.read_csv(args.results)
    daily_equity = pd.read_csv(args.daily_equity)
    daily_drawdown = pd.read_csv(args.daily_drawdown)
    exposure_log = pd.read_csv(args.exposure_log)
    artifacts = build_qqq_only_rf_second_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )
    paths = write_qqq_only_rf_second_confirmation_report_artifacts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        output_dir=args.output_dir,
    )

    print()
    print("QQQ-only RF second-confirmation report artifacts written.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print()
    preview_columns = [
        "candidate_rank",
        "strategy_name",
        "strategy_vs_qqq",
        "cost_20bps_strategy_vs_qqq",
        "positive_vs_qqq_window_rate",
        "decision_label",
        "production_ready",
    ]
    print(artifacts.decision_matrix.loc[:, preview_columns].head(20).to_string(index=False))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the QQQ-only RF second-confirmation report."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("RESULTS", str(DEFAULT_RESULTS))),
    )
    parser.add_argument(
        "--daily-equity",
        type=Path,
        default=Path(os.environ.get("DAILY_EQUITY", str(DEFAULT_DAILY_EQUITY))),
    )
    parser.add_argument(
        "--daily-drawdown",
        type=Path,
        default=Path(os.environ.get("DAILY_DRAWDOWN", str(DEFAULT_DAILY_DRAWDOWN))),
    )
    parser.add_argument(
        "--exposure-log",
        type=Path,
        default=Path(os.environ.get("EXPOSURE_LOG", str(DEFAULT_EXPOSURE_LOG))),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPORT_OUTPUT_DIR", str(DEFAULT_REPORT_OUTPUT_DIR))),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    return parser.parse_args(argv)


def _render_preflight(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# QQQ-Only RF Second-Confirmation Report Preflight",
            "",
            f"Local-only: `{True}`",
            f"Uses S3/cloud: `{False}`",
            f"Reads completed sweep artifacts: `{True}`",
            f"New broad model zoo: `{False}`",
            f"Production signal export: `{False}`",
            f"Model card: `{False}`",
            f"Production-ready marking: `{False}`",
            f"Dry run: `{args.dry_run}`",
            "",
            "## Inputs",
            "",
            f"- RESULTS: `{args.results}`",
            f"- DAILY_EQUITY: `{args.daily_equity}`",
            f"- DAILY_DRAWDOWN: `{args.daily_drawdown}`",
            f"- EXPOSURE_LOG: `{args.exposure_log}`",
            f"- OUTPUT_DIR: `{args.output_dir}`",
            "",
            "## Required Report Sections",
            "",
            *[f"- {section}" for section in REPORT_SECTIONS],
            "",
            "## Output Paths",
            "",
            *[f"- {name}: `{path}`" for name, path in report_output_paths(args.output_dir).items()],
            "",
            "Generated artifacts remain uncommitted.",
            "",
        ]
    )


def _ensure_local_only_path(path: str, flag_name: str) -> None:
    if str(path).lower().startswith(
        ("s3://", "s3:/", "gs://", "gs:/", "http://", "http:/", "https://", "https:/")
    ):
        raise ValueError(f"{flag_name} must be a local path, not {path!r}.")


def _ensure_local_file(path: Path, flag_name: str) -> None:
    _ensure_local_only_path(str(path), flag_name)
    if not path.exists():
        raise FileNotFoundError(f"{flag_name} does not exist: {path}")


if __name__ == "__main__":
    raise SystemExit(create_qqq_only_rf_second_confirmation_report_from_args())
