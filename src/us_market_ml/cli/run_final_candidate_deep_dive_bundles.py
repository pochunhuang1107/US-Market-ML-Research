"""Run final research-stage candidate deep-dive diagnostic bundles."""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from us_market_ml.cli.run_candidate_deep_dive_report import (
    run_candidate_deep_dive_report_from_args,
)

DEFAULT_FULL_DATA_PATH = Path("artifacts/cache/universe_full_2018_2026.parquet")
DEFAULT_BROAD_MACRO_DATA_PATH = Path(
    "artifacts/cache/universe_broad_macro_2018_2026.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_candidate_deep_dive")
SUMMARY_FILENAME = "final_candidate_deep_dive_summary.md"
DEEP_DIVE_MODULE = "us_market_ml.cli.run_candidate_deep_dive_report"

COMMON_MODEL_FAMILY = "baseline"
COMMON_FEATURE_SET_VERSION = "default_features_v1"
COMMON_RANDOM_STATE = 42
COMMON_MIN_HOLDING_DAYS = 10
COMMON_SWITCH_MARGIN = 0.002
COMMON_SCORE_TEMPERATURE = 1.0
COMMON_MAX_WEIGHT = 1.0
COMMON_TARGET_COL = "excess_return_vs_spy_20d"
COMMON_LABEL_HORIZON_DAYS = 20
COMMON_BENCHMARK_SYMBOLS = "SPY,QQQ"
COMMON_ALLOCATION_MODE = "equal_weight"
COMMON_TRANSACTION_COST_BPS = 10.0

FINAL_CANDIDATE_FOLDS = (
    "fold_2022:2019-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31",
    "fold_2023:2019-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31",
    "fold_2024:2019-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31",
    "fold_2025_2026:2019-01-01:2023-12-31:2024-01-01:2024-12-31:2025-01-01:2026-05-08",
)


@dataclass(frozen=True)
class FinalCandidateSpec:
    """One final research-stage deep-dive candidate."""

    name: str
    universe: str
    model_type: str
    top_k: int
    allocation_mode: str
    transaction_cost_bps: float
    research_label: str
    role: str

    @property
    def candidate_arg(self) -> str:
        return (
            f"{COMMON_MODEL_FAMILY}:{self.name}:{self.model_type}:{self.top_k}:"
            f"{COMMON_RANDOM_STATE}"
        )


@dataclass(frozen=True)
class FinalCandidateGroup:
    """Candidate group mapped to one input universe and output directory."""

    name: str
    data_path: Path
    output_dir: Path
    report_name: str
    candidates: tuple[FinalCandidateSpec, ...]


FULL_CANDIDATES = (
    FinalCandidateSpec(
        name="fixed_fold_reference_full_linear_top1",
        universe="full",
        model_type="linear_regression",
        top_k=1,
        allocation_mode=COMMON_ALLOCATION_MODE,
        transaction_cost_bps=COMMON_TRANSACTION_COST_BPS,
        research_label="full / linear_regression / top1 / equal_weight / cost10",
        role="Fixed-fold return champion and historical reference.",
    ),
    FinalCandidateSpec(
        name="longer_window_balanced_full_ridge_top5",
        universe="full",
        model_type="ridge",
        top_k=5,
        allocation_mode=COMMON_ALLOCATION_MODE,
        transaction_cost_bps=COMMON_TRANSACTION_COST_BPS,
        research_label="full / ridge / top5 / equal_weight / cost10",
        role="Longer-window balanced candidate.",
    ),
    FinalCandidateSpec(
        name="full_balanced_reference_ridge_top3",
        universe="full",
        model_type="ridge",
        top_k=3,
        allocation_mode=COMMON_ALLOCATION_MODE,
        transaction_cost_bps=COMMON_TRANSACTION_COST_BPS,
        research_label="full / ridge / top3 / equal_weight / cost10",
        role="Full-universe balanced reference.",
    ),
)

BROAD_MACRO_CANDIDATES = (
    FinalCandidateSpec(
        name="benchmark_relative_broadmacro_linear_top1",
        universe="broad_macro",
        model_type="linear_regression",
        top_k=1,
        allocation_mode=COMMON_ALLOCATION_MODE,
        transaction_cost_bps=COMMON_TRANSACTION_COST_BPS,
        research_label="broad_macro / linear_regression / top1 / equal_weight / cost10",
        role="Benchmark-relative sliding-window candidate.",
    ),
    FinalCandidateSpec(
        name="defensive_broadmacro_ridge_top5",
        universe="broad_macro",
        model_type="ridge",
        top_k=5,
        allocation_mode=COMMON_ALLOCATION_MODE,
        transaction_cost_bps=COMMON_TRANSACTION_COST_BPS,
        research_label="broad_macro / ridge / top5 / equal_weight / cost10",
        role="Defensive broad_macro reference.",
    ),
)


def run_final_candidate_deep_dive_bundles_from_args(
    argv: list[str] | None = None,
) -> int:
    """Generate final candidate deep-dive bundles by reusing the deep-dive report CLI."""

    args = _parse_args(argv)
    requested_groups = set(args.group)
    groups = build_final_candidate_groups(
        full_data_path=args.full_data_path,
        broad_macro_data_path=args.broad_macro_data_path,
        output_dir=args.output_dir,
    )
    groups = tuple(group for group in groups if group.name in requested_groups)
    runnable_groups, skipped_groups = resolve_runnable_groups(groups)

    print("US Market ML final candidate deep-dive bundles")
    print("This is a research-stage diagnostic run; no candidate is production-ready.")
    print(f"output root: {args.output_dir}")
    print(f"summary markdown: {args.output_dir / SUMMARY_FILENAME}")
    for group in runnable_groups:
        print(f"{group.name} output dir: {group.output_dir}")
    for _group, warning in skipped_groups:
        print(f"WARNING: {warning}")

    if args.dry_run:
        print()
        print("Dry run: commands that would run")
        for group in runnable_groups:
            command = build_deep_dive_command(group, python_bin=args.python_bin)
            print(shlex.join(command))
        return 0

    if not runnable_groups:
        raise FileNotFoundError("No final candidate deep-dive groups have available data files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_final_candidate_deep_dive_summary(
        output_dir=args.output_dir,
        groups=runnable_groups,
        skipped_groups=skipped_groups,
    )

    for group in runnable_groups:
        print()
        print(f"Running {group.name} deep-dive bundle")
        print(f"data path: {group.data_path}")
        print(f"output dir: {group.output_dir}")
        run_candidate_deep_dive_report_from_args(build_deep_dive_args(group))

    print()
    print(f"final candidate deep-dive summary: {summary_path}")
    return 0


def build_final_candidate_groups(
    *,
    full_data_path: Path = DEFAULT_FULL_DATA_PATH,
    broad_macro_data_path: Path = DEFAULT_BROAD_MACRO_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[FinalCandidateGroup, ...]:
    """Return the current final research-stage candidate groups."""

    return (
        FinalCandidateGroup(
            name="full",
            data_path=full_data_path,
            output_dir=output_dir / "full",
            report_name="final_candidate_deep_dive_full",
            candidates=FULL_CANDIDATES,
        ),
        FinalCandidateGroup(
            name="broad_macro",
            data_path=broad_macro_data_path,
            output_dir=output_dir / "broad_macro",
            report_name="final_candidate_deep_dive_broad_macro",
            candidates=BROAD_MACRO_CANDIDATES,
        ),
    )


def resolve_runnable_groups(
    groups: Sequence[FinalCandidateGroup],
) -> tuple[tuple[FinalCandidateGroup, ...], tuple[tuple[FinalCandidateGroup, str], ...]]:
    """Validate input data paths, failing for full and warning for optional broad_macro."""

    runnable: list[FinalCandidateGroup] = []
    skipped: list[tuple[FinalCandidateGroup, str]] = []
    for group in groups:
        if group.data_path.exists():
            runnable.append(group)
            continue
        if group.name == "broad_macro":
            skipped.append(
                (
                    group,
                    f"broad_macro data file is missing at {group.data_path}; "
                    "skipping broad_macro candidates.",
                )
            )
            continue
        raise FileNotFoundError(
            f"Required full universe data file is missing: {group.data_path}"
        )
    return tuple(runnable), tuple(skipped)


def build_deep_dive_command(
    group: FinalCandidateGroup,
    *,
    python_bin: str = "python",
) -> list[str]:
    """Build the subprocess-style command for logging and shell dry-runs."""

    return [python_bin, "-m", DEEP_DIVE_MODULE, *build_deep_dive_args(group)]


def build_deep_dive_args(group: FinalCandidateGroup) -> list[str]:
    """Build argv for the reusable candidate deep-dive report CLI."""

    argv = [
        "--data-path",
        str(group.data_path),
        "--output-dir",
        str(group.output_dir),
        "--report-name",
        group.report_name,
    ]
    for fold in FINAL_CANDIDATE_FOLDS:
        argv.extend(["--fold", fold])
    for candidate in group.candidates:
        argv.extend(["--candidate", candidate.candidate_arg])
    argv.extend(
        [
            "--benchmark-symbols",
            COMMON_BENCHMARK_SYMBOLS,
            "--target-col",
            COMMON_TARGET_COL,
            "--feature-set-version",
            COMMON_FEATURE_SET_VERSION,
            "--label-horizon-days",
            str(COMMON_LABEL_HORIZON_DAYS),
            "--transaction-cost-bps",
            _format_number(COMMON_TRANSACTION_COST_BPS),
            "--min-holding-days",
            str(COMMON_MIN_HOLDING_DAYS),
            "--switch-margin",
            _format_number(COMMON_SWITCH_MARGIN),
            "--allocation-mode",
            COMMON_ALLOCATION_MODE,
            "--score-temperature",
            _format_number(COMMON_SCORE_TEMPERATURE),
            "--max-weight",
            _format_number(COMMON_MAX_WEIGHT),
        ]
    )
    return argv


def write_final_candidate_deep_dive_summary(
    *,
    output_dir: Path,
    groups: Sequence[FinalCandidateGroup],
    skipped_groups: Sequence[tuple[FinalCandidateGroup, str]] = (),
) -> Path:
    """Write the local README-style summary for the generated bundles."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(
        render_final_candidate_deep_dive_summary(
            groups=groups,
            skipped_groups=skipped_groups,
        ),
        encoding="utf-8",
    )
    return summary_path


def render_final_candidate_deep_dive_summary(
    *,
    groups: Sequence[FinalCandidateGroup],
    skipped_groups: Sequence[tuple[FinalCandidateGroup, str]] = (),
) -> str:
    """Render markdown explaining the final deep-dive diagnostic bundles."""

    lines = [
        "# Final Candidate Deep-Dive Bundles",
        "",
        "This report wrapper generates diagnostic bundles for the current "
        "research-stage final candidate shortlist. It reuses the existing "
        "`run_candidate_deep_dive_report.py` workflow and does not run a broad sweep, "
        "change model training behavior, update daily signal export, or update model cards.",
        "",
        "No candidate is production-ready yet.",
        "",
        "## Evaluation Design",
        "",
        "The bundles use non-overlapping fixed-fold out-of-sample periods:",
    ]
    for fold in FINAL_CANDIDATE_FOLDS:
        name, train_start, train_end, validation_start, validation_end, test_start, test_end = (
            fold.split(":")
        )
        lines.append(
            f"- `{name}`: train {train_start} to {train_end}; validation "
            f"{validation_start} to {validation_end}; test {test_start} to {test_end}"
        )

    lines.extend(
        [
            "",
            "## Current Research-Stage Labels",
            "",
        ]
    )
    for group in groups:
        lines.append(f"### {group.name}")
        lines.append("")
        for candidate in group.candidates:
            lines.append(
                f"- `{candidate.name}`: `{candidate.research_label}`. {candidate.role}"
            )
        lines.append("")

    lines.extend(
        [
            "## Output Guide",
            "",
        ]
    )
    for group in groups:
        group_chart = group.output_dir / f"{group.report_name}_candidate_equity.png"
        lines.append(f"- `{group.name}` group equity chart: `{group_chart}`")
        for candidate in group.candidates:
            candidate_dir = group.output_dir / candidate.name
            lines.append(
                f"- `{candidate.name}` charts and diagnostics: equity "
                f"`{candidate_dir / f'{candidate.name}_equity.png'}`, heatmap "
                f"`{candidate_dir / f'{candidate.name}_monthly_selection_heatmap.png'}`, "
                f"selection frequency "
                f"`{candidate_dir / f'{candidate.name}_selection_frequency.png'}`"
            )

    if skipped_groups:
        lines.extend(["", "## Skipped Groups", ""])
        for _group, warning in skipped_groups:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final research-stage candidate deep-dive bundles using "
            "the existing candidate deep-dive report CLI."
        )
    )
    parser.add_argument("--full-data-path", type=Path, default=DEFAULT_FULL_DATA_PATH)
    parser.add_argument(
        "--broad-macro-data-path",
        type=Path,
        default=DEFAULT_BROAD_MACRO_DATA_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--group",
        action="append",
        choices=("full", "broad_macro"),
        default=None,
        help="Candidate group to run. Repeat to run multiple groups; default runs both.",
    )
    parser.add_argument("--python-bin", default="python")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deep-dive commands without running them.",
    )
    args = parser.parse_args(argv)
    if args.group is None:
        args.group = ["full", "broad_macro"]
    return args


def _format_number(value: float) -> str:
    return f"{value:g}"


if __name__ == "__main__":
    raise SystemExit(run_final_candidate_deep_dive_bundles_from_args())
