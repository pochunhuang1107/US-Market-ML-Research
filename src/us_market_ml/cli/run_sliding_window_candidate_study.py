"""Run the local overnight sliding-window candidate robustness study."""

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

from us_market_ml.cli.run_sliding_window_champion_study import (
    DEFAULT_DATA_END,
    DEFAULT_DATA_START,
    RETRAIN_STEP_MONTHS,
    TEST_WINDOW_MONTHS,
    TRAIN_WINDOW_YEARS,
    VALIDATION_WINDOW_MONTHS,
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

DEFAULT_RUN_NAME = "sliding_window_overnight_candidate_study"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps")
DEFAULT_FULL_UNIVERSE = SweepUniverseConfig(
    name="full",
    data_path=Path("artifacts/cache/universe_full_2018_2026.parquet"),
)
DEFAULT_BROAD_MACRO_UNIVERSE = SweepUniverseConfig(
    name="broad_macro",
    data_path=Path("artifacts/cache/universe_broad_macro_2018_2026.parquet"),
)
CHAMPION_CANDIDATE_NAME = "linear_smooth10_margin002_v1"
CHAMPION_MODEL_TYPE = "linear_regression"
FEATURE_SET_VERSION = "default_features_v1"
TARGET_COL = "excess_return_vs_spy_20d"
LABEL_HORIZON_DAYS = 20
MIN_HOLDING_DAYS = 10
SWITCH_MARGIN = 0.002
RANDOM_STATE = 42
SUPPORTED_BASELINE_MODEL_TYPES = (
    "ridge",
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "extra_trees",
)
PHASE4_REQUESTED_MODEL_TYPES = (
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)


@dataclass(frozen=True)
class CandidateStudyPhase:
    """One named phase, possibly split into multiple sweep-run configs."""

    name: str
    description: str
    configs: tuple[SweepRunConfig, ...]
    notes: tuple[str, ...] = ()


def run_sliding_window_candidate_study_from_args(argv: list[str] | None = None) -> int:
    """Run the local sliding-window candidate robustness study CLI."""

    args = _parse_args(argv)
    folds = build_sliding_window_candidate_folds(
        data_start=args.data_start,
        data_end=args.data_end,
        include_partial_final_test=args.include_partial_final_test,
    )
    if not folds:
        raise SystemExit("No sliding-window folds were generated for the requested dates.")

    universes = build_candidate_universes(
        include_broad_macro=args.include_broad_macro,
        full_universe_path=args.full_universe,
        broad_macro_universe_path=args.broad_macro_universe,
    )
    phases = build_sliding_window_candidate_phase_configs(
        folds=folds,
        universes=universes,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        include_mlp=args.include_mlp,
    )
    _validate_unique_run_keys(phases)

    if args.count_only:
        print(count_phase_independent_configs(phases))
        return 0

    _print_plan(phases=phases, folds=folds, dry_run=args.dry_run)
    if args.dry_run:
        _print_equivalent_study_command(args)
        return 0

    latest_results = pd.DataFrame()
    config_index = 0
    for phase in phases:
        print(f"Starting {phase.name}: {phase.description}")
        for config in phase.configs:
            run_config = replace(config, resume=args.resume or config_index > 0)
            latest_results = run_sweep(run_config)
            config_index += 1

    output_dir = Path(args.output_dir)
    summary = summarize_sweep_results(latest_results)
    print()
    print("US Market ML sliding-window candidate study complete")
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


def build_sliding_window_candidate_folds(
    *,
    data_start: str,
    data_end: str,
    include_partial_final_test: bool = True,
) -> list[SweepFoldConfig]:
    """Return the same deterministic sliding-window folds used by the champion study."""

    walk_forward_folds = make_sliding_window_folds_for_train_windows(
        data_start=data_start,
        data_end=data_end,
        train_window_years_values=TRAIN_WINDOW_YEARS,
        validation_window_months=VALIDATION_WINDOW_MONTHS,
        test_window_months=TEST_WINDOW_MONTHS,
        step_months=RETRAIN_STEP_MONTHS,
        include_partial_final_test=include_partial_final_test,
    )
    return [_to_sweep_fold_config(fold) for fold in walk_forward_folds]


def build_candidate_universes(
    *,
    include_broad_macro: bool = True,
    full_universe_path: Path | str = DEFAULT_FULL_UNIVERSE.data_path,
    broad_macro_universe_path: Path | str = DEFAULT_BROAD_MACRO_UNIVERSE.data_path,
    warn: Callable[[str], None] | None = None,
) -> list[SweepUniverseConfig]:
    """Build the universe list, skipping missing broad-macro data with a warning."""

    universes = [
        SweepUniverseConfig(name=DEFAULT_FULL_UNIVERSE.name, data_path=Path(full_universe_path))
    ]
    if not include_broad_macro:
        return universes

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
            f"{broad_macro_path}; skipping broad_macro universe."
        )
        if warn is None:
            print(warning, file=sys.stderr)
        else:
            warn(warning)
    return universes


def build_sliding_window_candidate_phase_configs(
    *,
    folds: list[SweepFoldConfig],
    universes: list[SweepUniverseConfig],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    include_mlp: bool = False,
) -> list[CandidateStudyPhase]:
    """Build phase-specific sweep configs without crossing unrelated candidate grids."""

    phases = [
        CandidateStudyPhase(
            name="Phase 1",
            description="current return champion sanity",
            configs=(
                _sweep_config(
                    universes=universes,
                    folds=folds,
                    models=[_baseline_model(CHAMPION_MODEL_TYPE)],
                    signals=[_signal(top_k=1, allocation_mode="equal_weight")],
                    transaction_cost_bps_values=[10.0],
                    output_dir=output_dir,
                    run_name=run_name,
                    resume=resume,
                    max_workers=max_workers,
                ),
            ),
        ),
        CandidateStudyPhase(
            name="Phase 2",
            description="champion-family defensive allocation references",
            configs=(
                _sweep_config(
                    universes=universes,
                    folds=folds,
                    models=[_baseline_model(CHAMPION_MODEL_TYPE)],
                    signals=_defensive_allocation_signals(top_k_values=(2, 3, 5)),
                    transaction_cost_bps_values=[10.0, 20.0],
                    output_dir=output_dir,
                    run_name=run_name,
                    resume=resume,
                    max_workers=max_workers,
                ),
            ),
        ),
        CandidateStudyPhase(
            name="Phase 3",
            description="balanced ridge reference family",
            configs=(
                _sweep_config(
                    universes=universes,
                    folds=folds,
                    models=[_baseline_model("ridge")],
                    signals=_defensive_allocation_signals(top_k_values=(3, 5)),
                    transaction_cost_bps_values=[10.0, 20.0],
                    output_dir=output_dir,
                    run_name=run_name,
                    resume=resume,
                    max_workers=max_workers,
                ),
            ),
        ),
        _phase4_model_family_baseline(
            universes=universes,
            folds=folds,
            output_dir=output_dir,
            run_name=run_name,
            resume=resume,
            max_workers=max_workers,
        ),
    ]
    if include_mlp:
        phases.append(
            CandidateStudyPhase(
                name="Phase 5",
                description="optional MLP robustness probe",
                configs=(
                    _sweep_config(
                        universes=universes,
                        folds=folds,
                        models=[
                            SweepModelConfig(
                                model_family="mlp",
                                model_type="mlp",
                                hidden_units=(64, 32),
                                random_state=RANDOM_STATE,
                                max_epochs=100,
                                patience=10,
                            )
                        ],
                        signals=[
                            _signal(top_k=1, allocation_mode="equal_weight"),
                            _signal(top_k=3, allocation_mode="equal_weight"),
                        ],
                        transaction_cost_bps_values=[10.0],
                        output_dir=output_dir,
                        run_name=run_name,
                        resume=resume,
                        max_workers=max_workers,
                    ),
                ),
                notes=("MLP is opt-in only and is excluded unless --include-mlp is set.",),
            )
        )
    return phases


def count_phase_independent_configs(phases: Sequence[CandidateStudyPhase]) -> int:
    """Count independent universe/fold/model/signal/cost tasks across phases."""

    return sum(
        _count_sweep_config_independent_configs(config)
        for phase in phases
        for config in phase.configs
    )


def build_candidate_run_keys(phases: Sequence[CandidateStudyPhase]) -> list[str]:
    """Return every deterministic sweep run key implied by the study phases."""

    run_keys: list[str] = []
    for phase in phases:
        for config in phase.configs:
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


def _phase4_model_family_baseline(
    *,
    universes: list[SweepUniverseConfig],
    folds: list[SweepFoldConfig],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
) -> CandidateStudyPhase:
    supported = [
        model_type
        for model_type in PHASE4_REQUESTED_MODEL_TYPES
        if model_type in SUPPORTED_BASELINE_MODEL_TYPES
    ]
    unsupported = [
        model_type
        for model_type in PHASE4_REQUESTED_MODEL_TYPES
        if model_type not in SUPPORTED_BASELINE_MODEL_TYPES
    ]
    tree_model_types = [
        model_type
        for model_type in supported
        if model_type in {"random_forest", "extra_trees", "gradient_boosting"}
    ]
    configs = [
        _sweep_config(
            universes=universes,
            folds=folds,
            models=[_baseline_model("ridge")],
            signals=[_signal(top_k=1, allocation_mode="equal_weight")],
            transaction_cost_bps_values=[10.0],
            output_dir=output_dir,
            run_name=run_name,
            resume=resume,
            max_workers=max_workers,
        )
    ]
    if tree_model_types:
        configs.append(
            _sweep_config(
                universes=universes,
                folds=folds,
                models=[_baseline_model(model_type) for model_type in tree_model_types],
                signals=[
                    _signal(top_k=1, allocation_mode="equal_weight"),
                    _signal(top_k=3, allocation_mode="equal_weight"),
                ],
                transaction_cost_bps_values=[10.0],
                output_dir=output_dir,
                run_name=run_name,
                resume=resume,
                max_workers=max_workers,
            )
        )

    notes = ()
    if unsupported:
        notes = (
            "Unsupported baseline model types skipped: " + ", ".join(unsupported),
        )
    return CandidateStudyPhase(
        name="Phase 4",
        description="model-family robustness baseline",
        configs=tuple(configs),
        notes=notes,
    )


def _defensive_allocation_signals(
    *,
    top_k_values: Sequence[int],
) -> list[SweepSignalConfig]:
    signals: list[SweepSignalConfig] = []
    for top_k in top_k_values:
        signals.append(_signal(top_k=top_k, allocation_mode="equal_weight"))
        for temperature in (0.5, 1.0, 2.0):
            signals.append(
                _signal(
                    top_k=top_k,
                    allocation_mode="softmax_score_weighted",
                    score_temperature=temperature,
                )
            )
        for max_weight in (0.4, 0.5, 0.7):
            if top_k * max_weight < 1.0:
                continue
            signals.append(
                _signal(
                    top_k=top_k,
                    allocation_mode="capped_score_weighted",
                    max_weight=max_weight,
                )
            )
    return signals


def _sweep_config(
    *,
    universes: list[SweepUniverseConfig],
    folds: list[SweepFoldConfig],
    models: list[SweepModelConfig],
    signals: list[SweepSignalConfig],
    transaction_cost_bps_values: list[float],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
) -> SweepRunConfig:
    return SweepRunConfig(
        universes=universes,
        folds=folds,
        models=models,
        signals=signals,
        target_col=TARGET_COL,
        label_horizon_days=LABEL_HORIZON_DAYS,
        feature_set_versions=[FEATURE_SET_VERSION],
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        transaction_cost_bps_values=transaction_cost_bps_values,
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
    )


def _baseline_model(model_type: str) -> SweepModelConfig:
    return SweepModelConfig(
        model_family="baseline",
        model_type=model_type,
        random_state=RANDOM_STATE,
    )


def _signal(
    *,
    top_k: int,
    allocation_mode: str,
    score_temperature: float = 1.0,
    max_weight: float = 1.0,
) -> SweepSignalConfig:
    return SweepSignalConfig(
        top_k=top_k,
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        allocation_mode=allocation_mode,
        score_temperature=score_temperature,
        max_weight=max_weight,
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


def _validate_unique_run_keys(phases: Sequence[CandidateStudyPhase]) -> None:
    run_keys = build_candidate_run_keys(phases)
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Generated sliding-window candidate study contains duplicate run keys.")


def _print_plan(
    *,
    phases: Sequence[CandidateStudyPhase],
    folds: Sequence[SweepFoldConfig],
    dry_run: bool,
) -> None:
    first_config = phases[0].configs[0]
    output_dir = Path(first_config.output_dir)
    print("US Market ML sliding-window overnight candidate study")
    print("local-only: true")
    print(f"dry_run: {dry_run}")
    print(f"champion reference: {CHAMPION_CANDIDATE_NAME}")
    print(f"run_name: {first_config.run_name}")
    print(f"output_dir: {output_dir}")
    print(f"results CSV: {output_dir / f'{first_config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{first_config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{first_config.run_name}_failed_keys.txt'}")
    print(f"fold count: {len(folds)}")
    print(f"max workers: {first_config.max_workers}")
    print(f"resume: {first_config.resume}")
    includes_mlp = any(
        config.models[0].model_family == "mlp"
        for phase in phases
        for config in phase.configs
    )
    print(f"include MLP: {includes_mlp}")
    print(f"estimated independent config count: {count_phase_independent_configs(phases)}")
    print()
    print("Universes:")
    for universe in first_config.universes:
        print(f"- {universe.name}:{universe.data_path}")
    print()
    print("Generated folds:")
    print(f"- first: {format_fold_arg(folds[0])}")
    print(f"- last: {format_fold_arg(folds[-1])}")
    print()
    print("Phases:")
    for phase in phases:
        phase_count = sum(
            _count_sweep_config_independent_configs(config) for config in phase.configs
        )
        print(f"- {phase.name}: {phase.description}")
        print(f"  estimated independent configs: {phase_count}")
        print(f"  sweep config blocks: {len(phase.configs)}")
        for note in phase.notes:
            print(f"  note: {note}")


def _print_equivalent_study_command(args: argparse.Namespace) -> None:
    print()
    print("Command that would run:")
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_sliding_window_candidate_study",
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
    if not args.include_broad_macro:
        command.append("--no-include-broad-macro")
    if args.include_mlp:
        command.append("--include-mlp")
    print(_format_command(command))


def _equivalent_sweep_command(config: SweepRunConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_sweep",
        "--run-name",
        config.run_name,
        "--output-dir",
        str(config.output_dir),
    ]
    for universe in config.universes:
        command.extend(["--universe", f"{universe.name}:{universe.data_path}"])
    command.extend(
        [
            "--model-family",
            _model_family_arg(config.models),
            "--baseline-models",
            ",".join(model.model_type for model in config.models if model.model_family == "baseline")
            or "ridge",
            "--feature-set-versions",
            ",".join(config.feature_set_versions),
            "--seeds",
            ",".join(str(model.random_state) for model in config.models),
            "--top-k",
            ",".join(str(signal.top_k) for signal in config.signals),
            "--min-holding-days",
            ",".join(str(signal.min_holding_days or config.min_holding_days) for signal in config.signals),
            "--switch-margins",
            ",".join(f"{signal.switch_margin or config.switch_margin:g}" for signal in config.signals),
            "--allocation-modes",
            ",".join(signal.allocation_mode for signal in config.signals),
            "--score-temperatures",
            ",".join(f"{signal.score_temperature:g}" for signal in config.signals),
            "--max-weights",
            ",".join(f"{signal.max_weight:g}" for signal in config.signals),
            "--transaction-cost-bps",
            ",".join(f"{value:g}" for value in config.transaction_cost_bps_values),
            "--target-col",
            config.target_col,
            "--label-horizon-days",
            str(config.label_horizon_days),
            "--max-workers",
            str(config.max_workers),
        ]
    )
    if not config.resume:
        command.append("--no-resume")
    for fold in config.folds:
        command.extend(["--fold", format_fold_arg(fold)])
    return command


def _model_family_arg(models: Sequence[SweepModelConfig]) -> str:
    model_families = {model.model_family for model in models}
    if model_families == {"baseline"}:
        return "baseline"
    if model_families == {"mlp"}:
        return "mlp"
    return "both"


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
            "Run a local-only overnight sliding-window candidate robustness study "
            "after the benchmark-relative stability report."
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
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument(
        "--include-broad-macro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include broad_macro if the local parquet exists; skip with warning otherwise.",
    )
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
        help="Include a final near-quarterly test window capped at --data-end.",
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
    raise SystemExit(run_sliding_window_candidate_study_from_args())
