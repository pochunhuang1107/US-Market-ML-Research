"""Create the focused QQQ-only best-candidate deep-dive report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.qqq_only_best_candidate_deep_dive import (
    DEFAULT_OUTPUT_DIR,
    OUTPUT_PREFIX,
    build_qqq_only_best_candidate_deep_dive_artifacts,
    output_paths,
    write_qqq_only_best_candidate_deep_dive_artifacts,
)

DEFAULT_MODEL_ZOO_RESULTS = Path(
    "artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_results.csv"
)
DEFAULT_MODEL_ZOO_DAILY_EQUITY = Path(
    "artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_daily_equity.csv"
)
DEFAULT_MODEL_ZOO_DAILY_DRAWDOWN = Path(
    "artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_daily_drawdown.csv"
)
DEFAULT_MODEL_ZOO_EXPOSURE_LOG = Path(
    "artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_exposure_log.csv"
)
DEFAULT_MODEL_ZOO_SUMMARY = Path(
    "artifacts/reports/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_summary.csv"
)
DEFAULT_MODEL_ZOO_DECISION_MATRIX = Path(
    "artifacts/reports/qqq_only_cross_market_timing_model_zoo/"
    "qqq_only_cross_market_timing_model_zoo_decision_matrix.csv"
)


def create_qqq_only_best_candidate_deep_dive_from_args(
    argv: list[str] | None = None,
) -> int:
    """Create the local deep-dive report from existing model-zoo artifacts."""

    args = _parse_args(argv)
    for flag_name, path in (
        ("--model-zoo-results", args.model_zoo_results),
        ("--model-zoo-daily-equity", args.model_zoo_daily_equity),
        ("--model-zoo-daily-drawdown", args.model_zoo_daily_drawdown),
        ("--model-zoo-exposure-log", args.model_zoo_exposure_log),
        ("--model-zoo-summary", args.model_zoo_summary),
        ("--model-zoo-decision-matrix", args.model_zoo_decision_matrix),
    ):
        _ensure_local_file(path, flag_name)
    _ensure_local_only_path(str(args.output_dir), "--output-dir")

    print("US Market ML QQQ-only best-candidate deep dive")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("new model training: false")
    print("broad sweep: false")
    print("production signal export: false")
    print("model card: false")
    print("production-ready marking: false")
    print(f"dry_run: {args.dry_run}")
    print(f"output directory: {args.output_dir}")
    print(f"top_n_candidates: {args.top_n_candidates}")

    paths = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / f"{OUTPUT_PREFIX}_preflight.md"
    preflight_path.write_text(_render_preflight(args), encoding="utf-8")
    print(f"preflight Markdown path: {preflight_path}")
    if args.dry_run:
        return 0

    artifacts = build_qqq_only_best_candidate_deep_dive_artifacts(
        results=pd.read_csv(args.model_zoo_results),
        daily_equity=pd.read_csv(args.model_zoo_daily_equity),
        daily_drawdown=pd.read_csv(args.model_zoo_daily_drawdown),
        exposure_log=pd.read_csv(args.model_zoo_exposure_log),
        summary=pd.read_csv(args.model_zoo_summary),
        decision_matrix=pd.read_csv(args.model_zoo_decision_matrix),
        top_n_candidates=args.top_n_candidates,
    )
    write_qqq_only_best_candidate_deep_dive_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print()
    print("QQQ-only best-candidate deep-dive artifacts written.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print()
    preview_columns = [
        "candidate_rank",
        "candidate_role",
        "strategy_name",
        "strategy_vs_qqq",
        "cost_10bps_strategy_vs_qqq",
        "decision",
        "production_ready",
    ]
    print(artifacts.decision_matrix.loc[:, preview_columns].to_string(index=False))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a focused, local-only report for the best QQQ-only timing candidate."
        )
    )
    parser.add_argument(
        "--model-zoo-results",
        type=Path,
        default=Path(os.environ.get("MODEL_ZOO_RESULTS", str(DEFAULT_MODEL_ZOO_RESULTS))),
    )
    parser.add_argument(
        "--model-zoo-daily-equity",
        type=Path,
        default=Path(
            os.environ.get("MODEL_ZOO_DAILY_EQUITY", str(DEFAULT_MODEL_ZOO_DAILY_EQUITY))
        ),
    )
    parser.add_argument(
        "--model-zoo-daily-drawdown",
        type=Path,
        default=Path(
            os.environ.get(
                "MODEL_ZOO_DAILY_DRAWDOWN",
                str(DEFAULT_MODEL_ZOO_DAILY_DRAWDOWN),
            )
        ),
    )
    parser.add_argument(
        "--model-zoo-exposure-log",
        type=Path,
        default=Path(
            os.environ.get("MODEL_ZOO_EXPOSURE_LOG", str(DEFAULT_MODEL_ZOO_EXPOSURE_LOG))
        ),
    )
    parser.add_argument(
        "--model-zoo-summary",
        type=Path,
        default=Path(os.environ.get("MODEL_ZOO_SUMMARY", str(DEFAULT_MODEL_ZOO_SUMMARY))),
    )
    parser.add_argument(
        "--model-zoo-decision-matrix",
        type=Path,
        default=Path(
            os.environ.get(
                "MODEL_ZOO_DECISION_MATRIX",
                str(DEFAULT_MODEL_ZOO_DECISION_MATRIX),
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument(
        "--top-n-candidates",
        type=_parse_positive_int,
        default=_parse_positive_int(os.environ.get("TOP_N_CANDIDATES", "10")),
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
            "# QQQ-Only Best Candidate Deep Dive Preflight",
            "",
            f"Local-only: `{True}`",
            f"Uses S3/cloud: `{False}`",
            f"New model training: `{False}`",
            f"Broad sweep: `{False}`",
            f"Production signal export: `{False}`",
            f"Model card: `{False}`",
            f"Production-ready marking: `{False}`",
            f"Dry run: `{args.dry_run}`",
            "",
            "## Inputs",
            "",
            f"- MODEL_ZOO_RESULTS: `{args.model_zoo_results}`",
            f"- MODEL_ZOO_DAILY_EQUITY: `{args.model_zoo_daily_equity}`",
            f"- MODEL_ZOO_DAILY_DRAWDOWN: `{args.model_zoo_daily_drawdown}`",
            f"- MODEL_ZOO_EXPOSURE_LOG: `{args.model_zoo_exposure_log}`",
            f"- MODEL_ZOO_SUMMARY: `{args.model_zoo_summary}`",
            f"- MODEL_ZOO_DECISION_MATRIX: `{args.model_zoo_decision_matrix}`",
            f"- OUTPUT_DIR: `{args.output_dir}`",
            "",
            "Generated artifacts remain uncommitted.",
            "",
        ]
    )


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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
    raise SystemExit(create_qqq_only_best_candidate_deep_dive_from_args())
