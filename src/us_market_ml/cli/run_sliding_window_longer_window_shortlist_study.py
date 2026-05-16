"""Run the local longer-window sliding-window shortlist validation study."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import pandas as pd

from us_market_ml.cli.run_sliding_window_candidate_study import (
    DEFAULT_BROAD_MACRO_UNIVERSE,
    DEFAULT_FULL_UNIVERSE,
    FEATURE_SET_VERSION,
    LABEL_HORIZON_DAYS,
    MIN_HOLDING_DAYS,
    RANDOM_STATE,
    SWITCH_MARGIN,
    TARGET_COL,
)
from us_market_ml.cli.run_sliding_window_champion_study import (
    DEFAULT_DATA_END,
    DEFAULT_DATA_START,
)
from us_market_ml.cli.run_sliding_window_shortlist_benchmark_report import (
    SHORTLIST_SPECS,
)
from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    build_sweep_run_key,
    run_sweep,
    summarize_sweep_results,
)
from us_market_ml.splits import WalkForwardFold, make_sliding_window_folds_for_train_windows

DEFAULT_RUN_NAME = "sliding_window_longer_window_shortlist_study"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps")
TRAIN_WINDOW_YEARS = (2, 3, 5)
VALIDATION_WINDOW_MONTHS = 6
LONGER_WINDOW_TEST_DESIGNS = (
    {"test_window_months": 6, "step_months": 3},
    {"test_window_months": 12, "step_months": 6},
)
INCLUDE_PARTIAL_FINAL_TEST = True
MIN_PARTIAL_TEST_DAYS = 30


@dataclass(frozen=True)
class ShortlistCandidate:
    """One exact research-stage shortlist candidate."""

    universe: str
    model_type: str
    top_k: int
    allocation_mode: str
    transaction_cost_bps: float

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.universe}:{self.model_type}:top{self.top_k}:"
            f"{self.allocation_mode}:cost{self.transaction_cost_bps:g}"
        )

    @property
    def label(self) -> str:
        return (
            f"{self.universe} / {self.model_type} / top{self.top_k} / "
            f"{self.allocation_mode} / cost{self.transaction_cost_bps:g}"
        )


@dataclass(frozen=True)
class ShortlistStudyConfigBlock:
    """A sweep block plus the exact shortlist candidates it represents."""

    candidates: tuple[ShortlistCandidate, ...]
    config: SweepRunConfig


def run_sliding_window_longer_window_shortlist_study_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run the local longer-window shortlist validation study CLI."""

    args = _parse_args(argv)
    folds = build_longer_window_shortlist_folds(
        data_start=args.data_start,
        data_end=args.data_end,
        include_partial_final_test=args.include_partial_final_test,
    )
    if not folds:
        raise SystemExit("No longer-window sliding-window folds were generated.")

    universes = build_longer_window_shortlist_universes(
        full_universe_path=args.full_universe,
        broad_macro_universe_path=args.broad_macro_universe,
    )
    candidates = build_available_shortlist_candidates(universes)
    if not candidates:
        raise SystemExit("No shortlist candidates are available for the local data files.")

    blocks = build_longer_window_shortlist_config_blocks(
        folds=folds,
        universes=universes,
        candidates=candidates,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
    )
    _validate_unique_run_keys(blocks)

    if args.count_only:
        print(count_independent_configs(blocks))
        return 0

    _print_plan(
        blocks=blocks,
        folds=folds,
        candidates=candidates,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_equivalent_study_command(args)
        return 0

    latest_results = pd.DataFrame()
    for index, block in enumerate(blocks):
        candidate_labels = ", ".join(candidate.label for candidate in block.candidates)
        print(f"Starting shortlist block {index + 1}/{len(blocks)}: {candidate_labels}")
        run_config = replace(block.config, resume=args.resume or index > 0)
        latest_results = run_sweep(run_config)

    output_dir = Path(args.output_dir)
    summary = summarize_sweep_results(latest_results)
    print()
    print("US Market ML longer-window sliding-window shortlist study complete")
    print(f"results CSV: {output_dir / f'{args.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{args.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{args.run_name}_failed_keys.txt'}")
    print(f"result rows: {len(latest_results)}")
    print()
    if summary.empty:
        print("No completed test rows to summarize.")
    else:
        print(summary.to_string(index=False))
    return 0


def build_longer_window_shortlist_folds(
    *,
    data_start: str,
    data_end: str,
    include_partial_final_test: bool = INCLUDE_PARTIAL_FINAL_TEST,
) -> list[SweepFoldConfig]:
    """Return deterministic 6-month and 12-month rolling test-window folds."""

    folds: list[SweepFoldConfig] = []
    for design in LONGER_WINDOW_TEST_DESIGNS:
        walk_forward_folds = make_sliding_window_folds_for_train_windows(
            data_start=data_start,
            data_end=data_end,
            train_window_years_values=TRAIN_WINDOW_YEARS,
            validation_window_months=VALIDATION_WINDOW_MONTHS,
            test_window_months=design["test_window_months"],
            step_months=design["step_months"],
            include_partial_final_test=include_partial_final_test,
            min_partial_test_days=MIN_PARTIAL_TEST_DAYS,
        )
        folds.extend(_to_sweep_fold_config(fold) for fold in walk_forward_folds)
    return folds


def build_longer_window_shortlist_universes(
    *,
    full_universe_path: Path | str = DEFAULT_FULL_UNIVERSE.data_path,
    broad_macro_universe_path: Path | str = DEFAULT_BROAD_MACRO_UNIVERSE.data_path,
    warn: Callable[[str], None] | None = None,
) -> list[SweepUniverseConfig]:
    """Build local universes, requiring full data and skipping missing broad_macro."""

    full_path = Path(full_universe_path)
    if not full_path.exists():
        raise FileNotFoundError(
            "Required full universe file not found at "
            f"{full_path}. The full universe is required for shortlist candidates."
        )

    universes = [SweepUniverseConfig(name=DEFAULT_FULL_UNIVERSE.name, data_path=full_path)]
    broad_macro_path = Path(broad_macro_universe_path)
    if broad_macro_path.exists():
        universes.append(
            SweepUniverseConfig(
                name=DEFAULT_BROAD_MACRO_UNIVERSE.name,
                data_path=broad_macro_path,
            )
        )
    else:
        warning = (
            "WARNING: broad_macro universe file not found at "
            f"{broad_macro_path}; skipping broad_macro shortlist candidates."
        )
        if warn is None:
            print(warning, file=sys.stderr)
        else:
            warn(warning)
    return universes


def build_available_shortlist_candidates(
    universes: Sequence[SweepUniverseConfig],
) -> list[ShortlistCandidate]:
    """Return the fixed shortlist filtered to universes available on disk."""

    available_universes = {universe.name for universe in universes}
    candidates: list[ShortlistCandidate] = []
    for spec in SHORTLIST_SPECS:
        candidate = ShortlistCandidate(
            universe=str(spec["universe"]),
            model_type=str(spec["model_type"]),
            top_k=int(spec["top_k"]),
            allocation_mode=str(spec["allocation_mode"]),
            transaction_cost_bps=float(spec["transaction_cost_bps"]),
        )
        if candidate.universe in available_universes:
            candidates.append(candidate)
    return candidates


def build_longer_window_shortlist_config_blocks(
    *,
    folds: list[SweepFoldConfig],
    universes: Sequence[SweepUniverseConfig],
    candidates: Sequence[ShortlistCandidate],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
) -> list[ShortlistStudyConfigBlock]:
    """Build exact shortlist sweep blocks without crossing unrelated candidates."""

    universe_by_name = {universe.name: universe for universe in universes}
    blocks: list[ShortlistStudyConfigBlock] = []
    for candidate_group in _group_compatible_candidates(candidates):
        first = candidate_group[0]
        universe = universe_by_name[first.universe]
        blocks.append(
            ShortlistStudyConfigBlock(
                candidates=tuple(candidate_group),
                config=SweepRunConfig(
                    universes=[universe],
                    folds=folds,
                    models=[_baseline_model(first.model_type)],
                    signals=[
                        _signal(
                            top_k=candidate.top_k,
                            allocation_mode=candidate.allocation_mode,
                        )
                        for candidate in candidate_group
                    ],
                    target_col=TARGET_COL,
                    label_horizon_days=LABEL_HORIZON_DAYS,
                    feature_set_versions=[FEATURE_SET_VERSION],
                    min_holding_days=MIN_HOLDING_DAYS,
                    switch_margin=SWITCH_MARGIN,
                    transaction_cost_bps_values=[first.transaction_cost_bps],
                    output_dir=output_dir,
                    run_name=run_name,
                    resume=resume,
                    max_workers=max_workers,
                ),
            )
        )
    return blocks


def count_independent_configs(blocks: Sequence[ShortlistStudyConfigBlock]) -> int:
    """Count independent universe/fold/model/signal/cost tasks across blocks."""

    return sum(_count_sweep_config_independent_configs(block.config) for block in blocks)


def build_shortlist_run_keys(blocks: Sequence[ShortlistStudyConfigBlock]) -> list[str]:
    """Return every deterministic sweep run key implied by the shortlist blocks."""

    run_keys: list[str] = []
    for block in blocks:
        config = block.config
        for universe, fold, model, signal, feature_set_version, cost in product(
            config.universes,
            config.folds,
            config.models,
            config.signals,
            config.feature_set_versions,
            config.transaction_cost_bps_values,
        ):
            run_keys.append(
                build_sweep_run_key(
                    universe=universe,
                    fold=fold,
                    model=model,
                    signal=signal,
                    label_horizon_days=config.label_horizon_days,
                    min_holding_days=config.min_holding_days,
                    switch_margin=config.switch_margin,
                    transaction_cost_bps=cost,
                    feature_set_version=feature_set_version,
                )
            )
    return run_keys


def format_fold_arg(fold: SweepFoldConfig) -> str:
    """Format a fold as the sweep CLI ``--fold`` argument value."""

    return ":".join(
        [
            fold.name,
            fold.train_start,
            fold.train_end,
            fold.validation_start,
            fold.validation_end,
            fold.test_start,
            fold.test_end,
        ]
    )


def _group_compatible_candidates(
    candidates: Sequence[ShortlistCandidate],
) -> list[list[ShortlistCandidate]]:
    groups: dict[tuple[str, str, str, float], list[ShortlistCandidate]] = {}
    for candidate in candidates:
        key = (
            candidate.universe,
            candidate.model_type,
            candidate.allocation_mode,
            candidate.transaction_cost_bps,
        )
        groups.setdefault(key, []).append(candidate)
    return list(groups.values())


def _baseline_model(model_type: str) -> SweepModelConfig:
    return SweepModelConfig(
        model_family="baseline",
        model_type=model_type,
        random_state=RANDOM_STATE,
    )


def _signal(*, top_k: int, allocation_mode: str) -> SweepSignalConfig:
    return SweepSignalConfig(
        top_k=top_k,
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        allocation_mode=allocation_mode,
        score_temperature=1.0,
        max_weight=1.0,
    )


def _count_sweep_config_independent_configs(config: SweepRunConfig) -> int:
    return (
        len(config.universes)
        * len(config.folds)
        * len(config.models)
        * len(config.signals)
        * len(config.feature_set_versions)
        * len(config.transaction_cost_bps_values)
    )


def _validate_unique_run_keys(blocks: Sequence[ShortlistStudyConfigBlock]) -> None:
    run_keys = build_shortlist_run_keys(blocks)
    if len(run_keys) != len(set(run_keys)):
        raise ValueError(
            "Generated longer-window shortlist study contains duplicate run keys."
        )


def _print_plan(
    *,
    blocks: Sequence[ShortlistStudyConfigBlock],
    folds: Sequence[SweepFoldConfig],
    candidates: Sequence[ShortlistCandidate],
    dry_run: bool,
) -> None:
    first_config = blocks[0].config
    output_dir = Path(first_config.output_dir)
    run_name = first_config.run_name
    print("US Market ML longer-window sliding-window shortlist study")
    print("local-only: true")
    print(f"dry_run: {dry_run}")
    print(f"run_name: {run_name}")
    print(f"output_dir: {output_dir}")
    print(f"results CSV: {output_dir / f'{run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{run_name}_failed_keys.txt'}")
    print(f"terminal log: {output_dir / f'{run_name}_terminal.log'}")
    print(f"fold count: {len(folds)}")
    print(f"candidate count: {len(candidates)}")
    print(f"estimated independent config count: {count_independent_configs(blocks)}")
    print(f"max workers: {first_config.max_workers}")
    print(f"resume: {first_config.resume}")
    print("model_family: baseline")
    print(f"feature_set_version: {FEATURE_SET_VERSION}")
    print(f"target_col: {TARGET_COL}")
    print(f"label_horizon_days: {LABEL_HORIZON_DAYS}")
    print()
    print("Longer-window fold designs:")
    for design in LONGER_WINDOW_TEST_DESIGNS:
        design_folds = [
            fold
            for fold in folds
            if f"_t{design['test_window_months']}m_" in fold.name
        ]
        print(
            "- "
            f"test_window_months={design['test_window_months']}, "
            f"step_months={design['step_months']}, "
            f"folds={len(design_folds)}"
        )
    print()
    print("Universes:")
    printed_universes: set[str] = set()
    for block in blocks:
        universe = block.config.universes[0]
        if universe.name in printed_universes:
            continue
        printed_universes.add(universe.name)
        print(f"- {universe.name}:{universe.data_path}")
    print()
    print("Shortlist candidates:")
    for candidate in candidates:
        print(f"- {candidate.label}")
    print()
    print("Generated folds:")
    for fold in folds:
        print(f"- {format_fold_arg(fold)}")


def _print_equivalent_study_command(args: argparse.Namespace) -> None:
    print()
    print("Equivalent command:")
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_sliding_window_longer_window_shortlist_study",
        "--run-name",
        args.run_name,
        "--output-dir",
        str(args.output_dir),
        "--data-start",
        args.data_start,
        "--data-end",
        args.data_end,
        "--full-universe",
        str(args.full_universe),
        "--broad-macro-universe",
        str(args.broad_macro_universe),
        "--max-workers",
        str(args.max_workers),
    ]
    if not args.resume:
        command.append("--no-resume")
    if not args.include_partial_final_test:
        command.append("--no-include-partial-final-test")
    print(_format_command(command))


def _format_command(command: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in command)


def _to_sweep_fold_config(fold: WalkForwardFold) -> SweepFoldConfig:
    return SweepFoldConfig(
        name=fold.name,
        train_start=_date_string(fold.train_start),
        train_end=_date_string(fold.train_end),
        validation_start=_date_string(fold.validation_start),
        validation_end=_date_string(fold.validation_end),
        test_start=_date_string(fold.test_start),
        test_end=_date_string(fold.test_end),
    )


def _date_string(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only longer-window validation study for the "
            "sliding-window shortlist candidates."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--full-universe", type=Path, default=DEFAULT_FULL_UNIVERSE.data_path)
    parser.add_argument(
        "--broad-macro-universe",
        type=Path,
        default=DEFAULT_BROAD_MACRO_UNIVERSE.data_path,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from local checkpoint files by default.",
    )
    parser.add_argument(
        "--include-partial-final-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a final test window capped at --data-end when it has enough days.",
    )
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", "4")),
        help="Number of local worker processes. Defaults to MAX_WORKERS or 4.",
    )
    args = parser.parse_args(argv)

    _parse_date(args.data_start, "--data-start")
    _parse_date(args.data_end, "--data-end")
    return args


def _parse_date(value: str, parameter_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{parameter_name} must be a valid date.") from exc


def _parse_max_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid worker count: {value!r}") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--max-workers must be greater than or equal to 1.")
    return workers


if __name__ == "__main__":
    raise SystemExit(run_sliding_window_longer_window_shortlist_study_from_args())
