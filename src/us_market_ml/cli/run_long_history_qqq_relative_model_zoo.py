"""Run the long-history QQQ-relative model-zoo research batch."""

from __future__ import annotations

import argparse
import glob
import os
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import pandas as pd

from us_market_ml.data import read_local_market_data
from us_market_ml.long_history_training import (
    END_DATE,
    LONG_HISTORY_FEATURE_SET_VERSION,
    START_DATE,
    SWEEP_PROFILES,
    LongHistoryRunSpec,
    SkippedLongHistoryRun,
    build_long_history_dataset,
    build_long_history_run_specs,
    dataset_start_for_required_symbols,
    get_sweep_profile,
    preflight_counts_for_spec,
    run_long_history_model_zoo_spec,
    run_spec_is_structurally_valid,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/long_history_qqq_relative_model_zoo")
DEFAULT_RUN_NAME = "long_history_qqq_relative_model_zoo"
DEFAULT_SWEEP_PROFILE = "smoke"
DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS = 1000
DEFAULT_MAX_WORKERS = 1
DATASET_SOURCE_LABEL = "tiingo_only"
PRICE_MODE = "tiingo_adjusted"

PREFLIGHT_CSV_NAME = "long_history_qqq_relative_model_zoo_preflight.csv"
PREFLIGHT_MD_NAME = "long_history_qqq_relative_model_zoo_preflight.md"
RESULTS_CSV_NAME = "long_history_qqq_relative_model_zoo_results.csv"
COMPLETED_KEYS_NAME = "long_history_qqq_relative_model_zoo_completed_keys.txt"
SKIPPED_KEYS_NAME = "long_history_qqq_relative_model_zoo_skipped_keys.txt"
FAILED_KEYS_NAME = "long_history_qqq_relative_model_zoo_failed_keys.txt"
TERMINAL_LOG_NAME = "long_history_qqq_relative_model_zoo_terminal.log"

PREFLIGHT_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "run_key",
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "recency_decay_profile",
    "base_weight",
    "recent_window_years",
    "finetune_window_years",
    "finetune_lr_scale",
    "historical_training_start",
    "selected_feature_count",
    "train_eligible_row_count",
    "train_symbol_count",
    "prediction_probe_row_count",
    "prediction_probe_symbol_count",
    "min_symbols_per_prediction_probe_date",
    "min_history_rows",
    "min_train_eligible_rows",
    "valid_config",
    "invalid_reason",
]

RESULT_COLUMNS = [
    "run_key",
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "recency_decay_profile",
    "base_weight",
    "recent_window_years",
    "finetune_window_years",
    "finetune_lr_scale",
    "start_date",
    "end_date",
    "historical_training_start",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_like",
    "return_to_drawdown",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "drawdown_improvement_vs_qqq",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "spy_buy_hold_return_to_drawdown",
    "decision",
]


@dataclass(frozen=True)
class LongHistoryModelZooPlan:
    """Executable long-history model-zoo plan."""

    specs: tuple[LongHistoryRunSpec, ...]
    sweep_profile: str
    market_dataset_glob: str
    output_dir: Path
    run_name: str
    start_date: str
    end_date: str
    resume: bool
    max_workers: int
    min_history_rows: int
    min_train_eligible_rows: int


@dataclass(frozen=True)
class LongHistoryModelZooPreflight:
    """Preflight output paths and counts."""

    csv_path: Path
    markdown_path: Path
    results_csv_path: Path
    completed_keys_path: Path
    skipped_keys_path: Path
    failed_keys_path: Path
    terminal_log_path: Path
    total_config_count: int
    valid_config_count: int
    skipped_config_count: int
    completed_checkpoint_count: int
    failed_checkpoint_count: int
    pending_config_count: int


def run_long_history_qqq_relative_model_zoo_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the long-history QQQ-relative model zoo."""

    args = _parse_args(argv)
    _install_interrupt_handlers()
    _ensure_dataset_available(args.market_dataset_glob)
    plan = build_long_history_qqq_relative_model_zoo_plan(
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        run_name=args.run_name,
        sweep_profile=args.sweep_profile,
        start_date=args.start_date,
        end_date=args.end_date,
        resume=args.resume,
        max_workers=args.max_workers,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
    )
    preflight = preflight_long_history_qqq_relative_model_zoo(plan)
    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        run_long_history_qqq_relative_model_zoo(plan, preflight)
    except KeyboardInterrupt:
        print("Interrupt received; long-history QQQ-relative model zoo stopped cleanly.")
        return 130
    return 0


def build_long_history_qqq_relative_model_zoo_plan(
    *,
    market_dataset_glob: str,
    output_dir: Path,
    run_name: str,
    sweep_profile: str = DEFAULT_SWEEP_PROFILE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    resume: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    min_train_eligible_rows: int = DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
) -> LongHistoryModelZooPlan:
    """Build an executable model-zoo plan."""

    if max_workers != 1:
        # The long-history runner is deliberately sequential for first-pass
        # resumability and to avoid oversubscribing sklearn tree models.
        max_workers = 1
    specs = build_long_history_run_specs(sweep_profile=sweep_profile)
    return LongHistoryModelZooPlan(
        specs=tuple(specs),
        sweep_profile=get_sweep_profile(sweep_profile).name,
        market_dataset_glob=market_dataset_glob,
        output_dir=output_dir,
        run_name=run_name,
        start_date=start_date,
        end_date=end_date,
        resume=resume,
        max_workers=max_workers,
        min_history_rows=min_history_rows,
        min_train_eligible_rows=min_train_eligible_rows,
    )


def preflight_long_history_qqq_relative_model_zoo(
    plan: LongHistoryModelZooPlan,
) -> LongHistoryModelZooPreflight:
    """Write preflight CSV/Markdown and seed skipped checkpoint keys."""

    paths = _output_paths(plan.output_dir)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_results_file(paths["results_csv_path"])
    paths["completed_keys_path"].touch(exist_ok=True)
    paths["failed_keys_path"].touch(exist_ok=True)
    paths["terminal_log_path"].touch(exist_ok=True)

    raw_df = read_local_market_data(plan.market_dataset_glob)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    historical_start = dataset_start_for_required_symbols(raw_df)
    dataset = build_long_history_dataset(
        raw_df.loc[
            raw_df["date"].between(historical_start, pd.Timestamp(plan.end_date)),
        ].copy(),
        feature_set_version=LONG_HISTORY_FEATURE_SET_VERSION,
    )

    rows = []
    for spec in plan.specs:
        structural_valid, structural_reason = run_spec_is_structurally_valid(spec)
        if structural_valid:
            counts = preflight_counts_for_spec(
                dataset,
                spec,
                min_history_rows=plan.min_history_rows,
                min_train_eligible_rows=plan.min_train_eligible_rows,
                start_date=plan.start_date,
                end_date=plan.end_date,
            )
        else:
            counts = {
                "selected_feature_count": 0,
                "train_eligible_row_count": 0,
                "train_symbol_count": 0,
                "prediction_probe_row_count": 0,
                "prediction_probe_symbol_count": 0,
                "min_symbols_per_prediction_probe_date": 0,
                "valid_config": False,
                "invalid_reason": structural_reason,
            }
        rows.append(
            {
                "dataset_source": DATASET_SOURCE_LABEL,
                "sweep_profile": plan.sweep_profile,
                "run_key": spec.run_key,
                "model_type": spec.model_type,
                "training_mode": spec.training_mode,
                "target_variant": spec.target_variant,
                "feature_set_version": spec.feature_set_version,
                "allocation_rule": spec.allocation_rule,
                "rebalance_frequency": spec.rebalance_frequency,
                "min_holding_days": spec.min_holding_days,
                "switch_margin": spec.switch_margin,
                "transaction_cost_bps": spec.transaction_cost_bps,
                "recency_decay_profile": spec.recency_decay_profile,
                "base_weight": spec.base_weight,
                "recent_window_years": spec.recent_window_years,
                "finetune_window_years": spec.finetune_window_years,
                "finetune_lr_scale": spec.finetune_lr_scale,
                "historical_training_start": str(historical_start.date()),
                "min_history_rows": plan.min_history_rows,
                "min_train_eligible_rows": plan.min_train_eligible_rows,
                **counts,
            }
        )
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(paths["preflight_csv_path"], index=False)

    invalid_keys = set(frame.loc[~frame["valid_config"], "run_key"].astype(str))
    existing_skipped = (
        _read_key_checkpoint(paths["skipped_keys_path"]) if plan.resume else set()
    )
    skipped_keys = sorted(existing_skipped.union(invalid_keys))
    paths["skipped_keys_path"].write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )

    valid_keys = set(frame.loc[frame["valid_config"], "run_key"].astype(str))
    completed_keys = (
        _read_key_checkpoint(paths["completed_keys_path"]) if plan.resume else set()
    )
    completed_in_plan = valid_keys.intersection(completed_keys)
    failed_count = _non_empty_line_count(paths["failed_keys_path"]) if plan.resume else 0
    pending_count = max(0, len(valid_keys) - len(completed_in_plan))
    preflight = LongHistoryModelZooPreflight(
        csv_path=paths["preflight_csv_path"],
        markdown_path=paths["preflight_md_path"],
        results_csv_path=paths["results_csv_path"],
        completed_keys_path=paths["completed_keys_path"],
        skipped_keys_path=paths["skipped_keys_path"],
        failed_keys_path=paths["failed_keys_path"],
        terminal_log_path=paths["terminal_log_path"],
        total_config_count=len(frame),
        valid_config_count=int(frame["valid_config"].sum()) if not frame.empty else 0,
        skipped_config_count=int((~frame["valid_config"]).sum()) if not frame.empty else 0,
        completed_checkpoint_count=len(completed_in_plan),
        failed_checkpoint_count=failed_count,
        pending_config_count=pending_count,
    )
    paths["preflight_md_path"].write_text(
        _render_preflight_markdown(frame=frame, plan=plan, preflight=preflight),
        encoding="utf-8",
    )
    return preflight


def run_long_history_qqq_relative_model_zoo(
    plan: LongHistoryModelZooPlan,
    preflight: LongHistoryModelZooPreflight,
) -> None:
    """Run pending valid configs sequentially with resumable checkpoints."""

    if not plan.resume:
        for path in (
            preflight.results_csv_path,
            preflight.completed_keys_path,
            preflight.skipped_keys_path,
            preflight.failed_keys_path,
        ):
            path.unlink(missing_ok=True)
        _ensure_results_file(preflight.results_csv_path)
        preflight.completed_keys_path.touch()
        preflight.skipped_keys_path.touch()
        preflight.failed_keys_path.touch()

    preflight_frame = pd.read_csv(preflight.csv_path)
    valid_keys = set(preflight_frame.loc[preflight_frame["valid_config"], "run_key"].astype(str))
    completed_keys = _read_key_checkpoint(preflight.completed_keys_path) if plan.resume else set()
    skipped_keys = _read_key_checkpoint(preflight.skipped_keys_path)
    raw_df = read_local_market_data(plan.market_dataset_glob)

    started_at = time.monotonic()
    last_progress = 0.0
    for index, spec in enumerate(plan.specs, start=1):
        if spec.run_key not in valid_keys:
            continue
        if plan.resume and spec.run_key in completed_keys:
            continue
        if spec.run_key in skipped_keys:
            continue
        try:
            row = run_long_history_model_zoo_spec(
                raw_df,
                spec,
                start_date=plan.start_date,
                end_date=plan.end_date,
                min_history_rows=plan.min_history_rows,
                min_train_eligible_rows=plan.min_train_eligible_rows,
            )
        except SkippedLongHistoryRun as exc:
            _append_line(preflight.skipped_keys_path, spec.run_key)
            skipped_keys.add(spec.run_key)
            _append_log(preflight.terminal_log_path, f"SKIPPED\t{spec.run_key}\t{exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - keep the long batch moving.
            _append_line(
                preflight.failed_keys_path,
                f"{spec.run_key}\t{type(exc).__name__}: {' '.join(str(exc).splitlines())}",
            )
            _append_log(
                preflight.terminal_log_path,
                f"FAILED\t{spec.run_key}\t{type(exc).__name__}: {exc}",
            )
            continue

        _append_results(preflight.results_csv_path, pd.DataFrame([row]))
        _append_line(preflight.completed_keys_path, spec.run_key)
        completed_keys.add(spec.run_key)

        now = time.monotonic()
        if now - last_progress >= 60.0:
            last_progress = now
            elapsed = int(now - started_at)
            print(
                "progress: "
                f"completed={len(completed_keys)} skipped={len(skipped_keys)} "
                f"failed={_non_empty_line_count(preflight.failed_keys_path)} "
                f"seen={index}/{len(plan.specs)} elapsed_seconds={elapsed}",
                flush=True,
            )


def _render_preflight_markdown(
    *,
    frame: pd.DataFrame,
    plan: LongHistoryModelZooPlan,
    preflight: LongHistoryModelZooPreflight,
) -> str:
    profile = get_sweep_profile(plan.sweep_profile)
    valid = frame.loc[frame["valid_config"]]
    invalid = frame.loc[~frame["valid_config"]].head(30)
    model_summary = (
        frame.groupby(["model_type", "training_mode"], dropna=False)
        .agg(
            config_count=("run_key", "count"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    lines = [
        "# Long-History QQQ-Relative Model Zoo Preflight",
        "",
        "Local-only diagnostic before model training/backtesting.",
        "",
        "## Research Question",
        "",
        "Can a long-history ML strategy using a dynamic ETF universe, QQQ-relative "
        "targets, and richer regime/cycle features beat QQQ buy-and-hold from "
        f"{plan.start_date} to {plan.end_date}?",
        "",
        "## Controls",
        "",
        "- Primary benchmark: `QQQ buy-and-hold`",
        "- Secondary references: `SPY buy-and-hold`, `cash`",
        "- Local-only: `true`",
        "- S3/cloud access: `false`",
        "- Production signal export: `false`",
        "- Rich replay/model card/paper tracking: `false`",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Price mode: `{PRICE_MODE}`",
        f"- Dataset path/glob: `{plan.market_dataset_glob}`",
        "",
        "## Sweep Profile",
        "",
        f"- Sweep profile: `{profile.name}`",
        f"- Profile purpose: `{profile.purpose}`",
        f"- Historical training start: `{frame['historical_training_start'].dropna().iloc[0] if not frame.empty else 'unknown'}`",
        f"- Forward start: `{plan.start_date}`",
        f"- Forward end: `{plan.end_date}`",
        f"- Feature set versions: `{sorted(frame['feature_set_version'].dropna().unique().tolist())}`",
        f"- Target variants: `{sorted(frame['target_variant'].dropna().unique().tolist())}`",
        f"- Training modes: `{sorted(frame['training_mode'].dropna().unique().tolist())}`",
        f"- Model types: `{sorted(frame['model_type'].dropna().unique().tolist())}`",
        f"- Allocation rules: `{sorted(frame['allocation_rule'].dropna().unique().tolist())}`",
        f"- Rebalance frequencies: `{sorted(frame['rebalance_frequency'].dropna().unique().tolist())}`",
        f"- Minimum history rows: `{plan.min_history_rows}`",
        f"- Minimum train eligible rows: `{plan.min_train_eligible_rows}`",
        "",
        "## Counts",
        "",
        f"- Total configs/runs: `{preflight.total_config_count}`",
        f"- Valid configs/runs: `{preflight.valid_config_count}`",
        f"- Skipped configs/runs: `{preflight.skipped_config_count}`",
        f"- Already completed if RESUME=1: `{preflight.completed_checkpoint_count}`",
        f"- Failed checkpoint count if RESUME=1: `{preflight.failed_checkpoint_count}`",
        f"- Pending configs/runs: `{preflight.pending_config_count}`",
        "",
        "## Output Paths",
        "",
        f"- Results CSV: `{preflight.results_csv_path}`",
        f"- Preflight CSV: `{preflight.csv_path}`",
        f"- Preflight Markdown: `{preflight.markdown_path}`",
        f"- Completed keys: `{preflight.completed_keys_path}`",
        f"- Skipped keys: `{preflight.skipped_keys_path}`",
        f"- Failed keys: `{preflight.failed_keys_path}`",
        f"- Terminal/progress log: `{preflight.terminal_log_path}`",
        "",
        "## Model/Mode Summary",
        "",
        _markdown_table(
            model_summary,
            ["model_type", "training_mode", "config_count", "valid_config_count", "invalid_config_count"],
        ),
        "",
        "## Invalid/Skipped Preview",
        "",
        _markdown_table(
            invalid,
            [
                "model_type",
                "training_mode",
                "target_variant",
                "allocation_rule",
                "rebalance_frequency",
                "invalid_reason",
            ],
        ),
        "",
        "## Valid Preview",
        "",
        _markdown_table(
            valid.head(20),
            [
                "model_type",
                "training_mode",
                "target_variant",
                "allocation_rule",
                "rebalance_frequency",
                "train_eligible_row_count",
                "prediction_probe_row_count",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _print_plan(
    *,
    plan: LongHistoryModelZooPlan,
    preflight: LongHistoryModelZooPreflight,
    dry_run: bool,
) -> None:
    frame = pd.read_csv(preflight.csv_path)
    print("US Market ML long-history QQQ-relative model zoo")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("production signal export: false")
    print("rich replay: false")
    print("model card: false")
    print(f"dry_run: {dry_run}")
    print(f"sweep_profile: {plan.sweep_profile}")
    print(f"dataset glob: {plan.market_dataset_glob}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"price mode: {PRICE_MODE}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print(f"primary benchmark: QQQ buy-and-hold")
    print(f"secondary references: SPY buy-and-hold, cash")
    print(f"feature set versions: {_unique_values(frame, 'feature_set_version')}")
    print(f"target variants: {_unique_values(frame, 'target_variant')}")
    print(f"training modes: {_unique_values(frame, 'training_mode')}")
    print(f"model types: {_unique_values(frame, 'model_type')}")
    print(f"allocation rules: {_unique_values(frame, 'allocation_rule')}")
    print(f"rebalance frequencies: {_unique_values(frame, 'rebalance_frequency')}")
    print(f"min_holding_days values: {_unique_values(frame, 'min_holding_days')}")
    print(f"switch_margin values: {_unique_values(frame, 'switch_margin')}")
    print(f"transaction_cost_bps values: {_unique_values(frame, 'transaction_cost_bps')}")
    print(f"run_name: {plan.run_name}")
    print(f"output directory: {plan.output_dir}")
    print(f"results CSV path: {preflight.results_csv_path}")
    print(f"completed checkpoint path: {preflight.completed_keys_path}")
    print(f"skipped checkpoint path: {preflight.skipped_keys_path}")
    print(f"failed checkpoint path: {preflight.failed_keys_path}")
    print(f"terminal/progress log path: {preflight.terminal_log_path}")
    print(f"preflight CSV path: {preflight.csv_path}")
    print(f"preflight Markdown path: {preflight.markdown_path}")
    print(f"total configs/runs: {preflight.total_config_count}")
    print(f"valid configs/runs: {preflight.valid_config_count}")
    print(f"skipped configs/runs: {preflight.skipped_config_count}")
    print(f"already completed configs if RESUME=1: {preflight.completed_checkpoint_count}")
    print(f"failed checkpoint count if RESUME=1: {preflight.failed_checkpoint_count}")
    print(f"pending configs/runs: {preflight.pending_config_count}")
    print(f"max workers: {plan.max_workers}")
    print(f"resume: {plan.resume}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local long-history QQQ-relative model-zoo batch."
    )
    parser.add_argument("--market-dataset-glob", default=_default_market_dataset_glob())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", DEFAULT_RUN_NAME))
    parser.add_argument(
        "--sweep-profile",
        default=os.environ.get("SWEEP_PROFILE", DEFAULT_SWEEP_PROFILE),
        choices=tuple(SWEEP_PROFILES),
    )
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", END_DATE))
    parser.add_argument(
        "--max-workers",
        type=_parse_positive_int,
        default=_parse_positive_int(os.environ.get("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("RESUME", "1") != "0",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-history-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(
            os.environ.get("MIN_HISTORY_ROWS", str(DEFAULT_MIN_HISTORY_ROWS))
        ),
    )
    parser.add_argument(
        "--min-train-eligible-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(
            os.environ.get("MIN_TRAIN_ELIGIBLE_ROWS", str(DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS))
        ),
    )
    args = parser.parse_args(argv)
    _parse_date(args.start_date, "--start-date")
    _parse_date(args.end_date, "--end-date")
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    return args


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "preflight_csv_path": output_dir / PREFLIGHT_CSV_NAME,
        "preflight_md_path": output_dir / PREFLIGHT_MD_NAME,
        "results_csv_path": output_dir / RESULTS_CSV_NAME,
        "completed_keys_path": output_dir / COMPLETED_KEYS_NAME,
        "skipped_keys_path": output_dir / SKIPPED_KEYS_NAME,
        "failed_keys_path": output_dir / FAILED_KEYS_NAME,
        "terminal_log_path": output_dir / TERMINAL_LOG_NAME,
    }


def _ensure_results_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(path, index=False)


def _append_results(path: Path, rows: pd.DataFrame) -> None:
    _ensure_results_file(path)
    for column in RESULT_COLUMNS:
        if column not in rows.columns:
            rows[column] = pd.NA
    rows.loc[:, RESULT_COLUMNS].to_csv(path, mode="a", header=False, index=False)


def _read_key_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().split("\t", maxsplit=1)[0] for line in path.read_text().splitlines() if line.strip()}


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _append_log(path: Path, line: str) -> None:
    _append_line(path, f"{pd.Timestamp.now().isoformat()}\t{line}")


def _non_empty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _unique_values(frame: pd.DataFrame, column: str) -> list[object]:
    values = frame[column].dropna().unique().tolist()
    return sorted(values)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("\n", " ") for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _default_market_dataset_glob() -> str:
    return os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB)


def _ensure_dataset_available(pattern: str) -> None:
    if glob.has_magic(pattern):
        if glob.glob(pattern, recursive=True):
            return
        raise FileNotFoundError(f"Default Tiingo-only dataset glob matched no files: {pattern}")
    if not Path(pattern).exists():
        raise FileNotFoundError(f"Default Tiingo-only dataset path does not exist: {pattern}")


def _parse_date(value: str, flag_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be a valid date.") from exc


def _parse_positive_int(value: str | int) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _install_interrupt_handlers() -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    raise SystemExit(run_long_history_qqq_relative_model_zoo_from_args())
