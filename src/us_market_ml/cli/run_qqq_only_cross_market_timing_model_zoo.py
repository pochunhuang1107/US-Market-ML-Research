"""Run the QQQ-only cross-market timing model zoo."""

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
from us_market_ml.qqq_only_cross_market_timing_model_zoo import (
    DAILY_DRAWDOWN_COLUMNS,
    DAILY_EQUITY_COLUMNS,
    EXPOSURE_LOG_COLUMNS,
    FEATURE_SUMMARY_COLUMNS,
    OUTPUT_PREFIX,
    RESULT_COLUMNS,
    QQQOnlyRunSpec,
    ReferenceSeriesDiscovery,
    SkippedQQQOnlyRun,
    build_qqq_only_feature_dataset,
    build_qqq_only_sweep_specs,
    discover_reference_series,
    load_reference_series,
    run_qqq_only_spec,
    write_qqq_only_report_artifacts,
)

DEFAULT_MARKET_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/qqq_only_cross_market_timing_model_zoo")
DEFAULT_REPORT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_cross_market_timing_model_zoo")
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_SWEEP_PROFILE = "core"
DEFAULT_MAX_WORKERS = 1
DEFAULT_MIN_TRAIN_ROWS = 252

RESULTS_CSV_NAME = f"{OUTPUT_PREFIX}_results.csv"
DAILY_EQUITY_CSV_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
DAILY_DRAWDOWN_CSV_NAME = f"{OUTPUT_PREFIX}_daily_drawdown.csv"
EXPOSURE_LOG_CSV_NAME = f"{OUTPUT_PREFIX}_exposure_log.csv"
FEATURE_SUMMARY_CSV_NAME = f"{OUTPUT_PREFIX}_feature_summary.csv"
COMPLETED_KEYS_NAME = f"{OUTPUT_PREFIX}_completed_keys.txt"
FAILED_KEYS_NAME = f"{OUTPUT_PREFIX}_failed_keys.txt"
SKIPPED_KEYS_NAME = f"{OUTPUT_PREFIX}_skipped_keys.txt"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
PREFLIGHT_CSV_NAME = f"{OUTPUT_PREFIX}_preflight.csv"
TERMINAL_LOG_NAME = f"{OUTPUT_PREFIX}_terminal.log"


@dataclass(frozen=True)
class QQQOnlyModelZooPlan:
    """Executable QQQ-only model-zoo plan."""

    specs: tuple[QQQOnlyRunSpec, ...]
    market_dataset_glob: str
    reference_series_glob: str | None
    reference_discovery: ReferenceSeriesDiscovery
    output_dir: Path
    report_output_dir: Path
    start_date: str
    end_date: str
    sweep_profile: str
    max_workers: int
    resume: bool
    min_train_rows: int


@dataclass(frozen=True)
class QQQOnlyPreflight:
    """Preflight output paths and counts."""

    paths: dict[str, Path]
    total_config_count: int
    baseline_config_count: int
    model_config_count: int
    completed_checkpoint_count: int
    skipped_checkpoint_count: int
    failed_checkpoint_count: int
    pending_config_count: int


def run_qqq_only_cross_market_timing_model_zoo_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the QQQ-only timing model zoo."""

    args = _parse_args(argv)
    _install_interrupt_handlers()
    plan = build_qqq_only_model_zoo_plan(
        market_dataset_glob=args.market_dataset_glob,
        reference_series_glob=args.reference_series_glob,
        output_dir=args.output_dir,
        report_output_dir=args.report_output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        sweep_profile=args.sweep_profile,
        max_workers=args.max_workers,
        resume=args.resume,
        min_train_rows=args.min_train_rows,
    )
    preflight = preflight_qqq_only_model_zoo(plan)
    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    _ensure_reference_series_available(plan.reference_discovery)
    try:
        run_qqq_only_model_zoo(plan, preflight)
    except KeyboardInterrupt:
        print("Interrupt received; QQQ-only timing model zoo stopped cleanly.")
        return 130
    return 0


def build_qqq_only_model_zoo_plan(
    *,
    market_dataset_glob: str,
    reference_series_glob: str | None,
    output_dir: Path,
    report_output_dir: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    sweep_profile: str = DEFAULT_SWEEP_PROFILE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    resume: bool = True,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> QQQOnlyModelZooPlan:
    """Build an executable plan."""

    specs = build_qqq_only_sweep_specs(sweep_profile)
    discovery = discover_reference_series(reference_series_glob)
    return QQQOnlyModelZooPlan(
        specs=tuple(specs),
        market_dataset_glob=market_dataset_glob,
        reference_series_glob=reference_series_glob,
        reference_discovery=discovery,
        output_dir=output_dir,
        report_output_dir=report_output_dir,
        start_date=start_date,
        end_date=end_date,
        sweep_profile=sweep_profile,
        max_workers=max_workers,
        resume=resume,
        min_train_rows=min_train_rows,
    )


def preflight_qqq_only_model_zoo(plan: QQQOnlyModelZooPlan) -> QQQOnlyPreflight:
    """Write preflight files and initialize resumable outputs."""

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(plan.output_dir)
    _ensure_csv(paths["results"], RESULT_COLUMNS)
    _ensure_csv(paths["daily_equity"], DAILY_EQUITY_COLUMNS)
    _ensure_csv(paths["daily_drawdown"], DAILY_DRAWDOWN_COLUMNS)
    _ensure_csv(paths["exposure_log"], EXPOSURE_LOG_COLUMNS)
    _ensure_csv(paths["feature_summary"], FEATURE_SUMMARY_COLUMNS)
    paths["completed_keys"].touch(exist_ok=True)
    paths["failed_keys"].touch(exist_ok=True)
    paths["skipped_keys"].touch(exist_ok=True)
    paths["terminal_log"].touch(exist_ok=True)

    rows = []
    for spec in plan.specs:
        rows.append(
            {
                "run_key": spec.run_key,
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.strategy_name,
                "is_baseline": spec.is_baseline,
                "baseline_rule": spec.baseline_rule,
                "model_type": spec.model_type,
                "training_mode": spec.training_mode,
                "target_name": spec.target_name,
                "target_horizon": spec.target_horizon,
                "feature_set_name": spec.feature_set_name,
                "exposure_mode": spec.exposure_mode,
                "rebalance_frequency": spec.rebalance_frequency,
                "transaction_cost_bps": spec.transaction_cost_bps,
                "threshold": spec.threshold,
                "min_holding_days": spec.min_holding_days,
                "switch_margin": spec.switch_margin,
            }
        )
    preflight_frame = pd.DataFrame(rows)
    preflight_frame.to_csv(paths["preflight_csv"], index=False)

    completed = _read_key_checkpoint(paths["completed_keys"]) if plan.resume else set()
    skipped = _read_key_checkpoint(paths["skipped_keys"]) if plan.resume else set()
    failed_count = _non_empty_line_count(paths["failed_keys"]) if plan.resume else 0
    all_keys = {spec.run_key for spec in plan.specs}
    completed_in_plan = all_keys.intersection(completed)
    skipped_in_plan = all_keys.intersection(skipped)
    pending = max(0, len(all_keys) - len(completed_in_plan) - len(skipped_in_plan))
    preflight = QQQOnlyPreflight(
        paths=paths,
        total_config_count=len(plan.specs),
        baseline_config_count=sum(1 for spec in plan.specs if spec.is_baseline),
        model_config_count=sum(1 for spec in plan.specs if not spec.is_baseline),
        completed_checkpoint_count=len(completed_in_plan),
        skipped_checkpoint_count=len(skipped_in_plan),
        failed_checkpoint_count=failed_count,
        pending_config_count=pending,
    )
    paths["preflight_md"].write_text(
        _render_preflight_markdown(plan=plan, preflight=preflight),
        encoding="utf-8",
    )
    return preflight


def run_qqq_only_model_zoo(
    plan: QQQOnlyModelZooPlan,
    preflight: QQQOnlyPreflight,
) -> None:
    """Run pending configs sequentially with resumable checkpoints."""

    if not plan.resume:
        for key in (
            "results",
            "daily_equity",
            "daily_drawdown",
            "exposure_log",
            "completed_keys",
            "failed_keys",
            "skipped_keys",
        ):
            preflight.paths[key].unlink(missing_ok=True)
        _ensure_csv(preflight.paths["results"], RESULT_COLUMNS)
        _ensure_csv(preflight.paths["daily_equity"], DAILY_EQUITY_COLUMNS)
        _ensure_csv(preflight.paths["daily_drawdown"], DAILY_DRAWDOWN_COLUMNS)
        _ensure_csv(preflight.paths["exposure_log"], EXPOSURE_LOG_COLUMNS)
        preflight.paths["completed_keys"].touch()
        preflight.paths["failed_keys"].touch()
        preflight.paths["skipped_keys"].touch()

    _ensure_market_data_available(plan.market_dataset_glob)
    raw_market = read_local_market_data(plan.market_dataset_glob)
    reference = load_reference_series(plan.reference_discovery.selected_glob)
    dataset = build_qqq_only_feature_dataset(
        raw_market,
        reference,
        start_date=plan.start_date,
        end_date=plan.end_date,
    )
    dataset.feature_summary.to_csv(preflight.paths["feature_summary"], index=False)

    completed = _read_key_checkpoint(preflight.paths["completed_keys"]) if plan.resume else set()
    skipped = _read_key_checkpoint(preflight.paths["skipped_keys"])
    prediction_cache: dict[tuple[object, ...], pd.DataFrame] = {}
    started_at = time.monotonic()
    last_progress = 0.0

    for index, spec in enumerate(plan.specs, start=1):
        if plan.resume and spec.run_key in completed:
            continue
        if spec.run_key in skipped:
            continue
        try:
            output = run_qqq_only_spec(
                dataset,
                spec,
                start_date=plan.start_date,
                end_date=plan.end_date,
                prediction_cache=prediction_cache,
                min_train_rows=plan.min_train_rows,
            )
        except SkippedQQQOnlyRun as exc:
            _append_line(preflight.paths["skipped_keys"], spec.run_key)
            skipped.add(spec.run_key)
            _append_log(preflight.paths["terminal_log"], f"SKIPPED\t{spec.run_key}\t{exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - keep long research runs moving.
            message = f"{spec.run_key}\t{type(exc).__name__}: {' '.join(str(exc).splitlines())}"
            _append_line(preflight.paths["failed_keys"], message)
            _append_log(preflight.paths["terminal_log"], f"FAILED\t{message}")
            continue

        _append_frame(preflight.paths["results"], pd.DataFrame([output.result]), RESULT_COLUMNS)
        _append_frame(preflight.paths["daily_equity"], output.daily_equity, DAILY_EQUITY_COLUMNS)
        _append_frame(
            preflight.paths["daily_drawdown"], output.daily_drawdown, DAILY_DRAWDOWN_COLUMNS
        )
        _append_frame(preflight.paths["exposure_log"], output.exposure_log, EXPOSURE_LOG_COLUMNS)
        _append_line(preflight.paths["completed_keys"], spec.run_key)
        completed.add(spec.run_key)

        now = time.monotonic()
        if now - last_progress >= 60.0:
            last_progress = now
            print(
                "progress: "
                f"completed={len(completed)} skipped={len(skipped)} "
                f"failed={_non_empty_line_count(preflight.paths['failed_keys'])} "
                f"seen={index}/{len(plan.specs)} elapsed_seconds={int(now - started_at)}",
                flush=True,
            )

    _write_report_from_outputs(preflight.paths, plan.report_output_dir)


def _write_report_from_outputs(paths: dict[str, Path], report_output_dir: Path) -> None:
    results = pd.read_csv(paths["results"])
    daily_equity = pd.read_csv(paths["daily_equity"])
    daily_drawdown = pd.read_csv(paths["daily_drawdown"])
    exposure_log = pd.read_csv(paths["exposure_log"])
    feature_summary = pd.read_csv(paths["feature_summary"])
    if results.empty:
        raise ValueError("No completed result rows were available for report generation.")
    outputs = write_qqq_only_report_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        feature_summary=feature_summary,
        output_dir=report_output_dir,
    )
    for name, path in outputs.items():
        print(f"report {name}: {path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local QQQ-only cross-market timing model zoo."
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_MARKET_DATASET_GLOB),
    )
    parser.add_argument(
        "--reference-series-glob",
        default=os.environ.get("REFERENCE_SERIES_GLOB") or None,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    )
    parser.add_argument(
        "--report-output-dir",
        type=Path,
        default=Path(os.environ.get("REPORT_OUTPUT_DIR", DEFAULT_REPORT_OUTPUT_DIR)),
    )
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", DEFAULT_END_DATE))
    parser.add_argument(
        "--sweep-profile",
        default=os.environ.get("SWEEP_PROFILE", DEFAULT_SWEEP_PROFILE),
        choices=("smoke", "core", "full"),
    )
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
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    parser.add_argument(
        "--min-train-rows",
        type=_parse_positive_int,
        default=_parse_positive_int(os.environ.get("MIN_TRAIN_ROWS", str(DEFAULT_MIN_TRAIN_ROWS))),
    )
    args = parser.parse_args(argv)
    _parse_date(args.start_date, "--start-date")
    _parse_date(args.end_date, "--end-date")
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    return args


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "results": output_dir / RESULTS_CSV_NAME,
        "daily_equity": output_dir / DAILY_EQUITY_CSV_NAME,
        "daily_drawdown": output_dir / DAILY_DRAWDOWN_CSV_NAME,
        "exposure_log": output_dir / EXPOSURE_LOG_CSV_NAME,
        "feature_summary": output_dir / FEATURE_SUMMARY_CSV_NAME,
        "completed_keys": output_dir / COMPLETED_KEYS_NAME,
        "failed_keys": output_dir / FAILED_KEYS_NAME,
        "skipped_keys": output_dir / SKIPPED_KEYS_NAME,
        "preflight_md": output_dir / PREFLIGHT_MD_NAME,
        "preflight_csv": output_dir / PREFLIGHT_CSV_NAME,
        "terminal_log": output_dir / TERMINAL_LOG_NAME,
    }


def _ensure_csv(path: Path, columns: Sequence[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def _append_frame(path: Path, rows: pd.DataFrame, columns: Sequence[str]) -> None:
    _ensure_csv(path, columns)
    out = rows.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out.loc[:, list(columns)].to_csv(path, mode="a", header=False, index=False)


def _read_key_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().split("\t", maxsplit=1)[0]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _append_log(path: Path, line: str) -> None:
    _append_line(path, f"{pd.Timestamp.now().isoformat()}\t{line}")


def _non_empty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _render_preflight_markdown(
    *,
    plan: QQQOnlyModelZooPlan,
    preflight: QQQOnlyPreflight,
) -> str:
    discovery = plan.reference_discovery
    model_summary = (
        pd.DataFrame([asdict_for_preflight(spec) for spec in plan.specs])
        .groupby(["model_type", "training_mode"], dropna=False)
        .size()
        .reset_index(name="config_count")
    )
    lines = [
        "# QQQ-Only Cross-Market Timing Model Zoo Preflight",
        "",
        (
            "Local-only research preflight. The only tradable risky asset is QQQ; "
            "every strategy state is QQQ and/or cash."
        ),
        "",
        "## Reference Series Discovery",
        "",
        f"- Selected reference glob: `{discovery.selected_glob}`",
        f"- Candidate reference globs: `{list(discovery.candidate_globs)}`",
        f"- Matched reference files: `{len(discovery.matched_paths)}`",
        f"- Expected symbols: `{list(discovery.expected_symbols)}`",
        f"- Found symbols: `{list(discovery.found_symbols)}`",
        f"- Missing symbols: `{list(discovery.missing_symbols)}`",
        f"- Explicitly excluded symbols: `{['DGS10']}`",
        "",
        "## Controls",
        "",
        "- Primary benchmark: `QQQ buy-and-hold`",
        "- Secondary references: `cash`, `SPY buy-and-hold`",
        "- Holdings rule: `QQQ/cash only`",
        "- ETF rotation holdings: `false`",
        "- SPY core sleeve: `false`",
        "- S3/cloud access: `false`",
        "- Production signal export: `false`",
        "- Model card: `false`",
        "- Production-ready marking: `false`",
        f"- Market dataset glob: `{plan.market_dataset_glob}`",
        f"- Start date: `{plan.start_date}`",
        f"- End date: `{plan.end_date}`",
        f"- Sweep profile: `{plan.sweep_profile}`",
        f"- MAX_WORKERS requested: `{plan.max_workers}`",
        "",
        "## Counts",
        "",
        f"- Total configs/runs: `{preflight.total_config_count}`",
        f"- Baseline configs/runs: `{preflight.baseline_config_count}`",
        f"- Model configs/runs: `{preflight.model_config_count}`",
        f"- Already completed if RESUME=1: `{preflight.completed_checkpoint_count}`",
        f"- Already skipped if RESUME=1: `{preflight.skipped_checkpoint_count}`",
        f"- Failed checkpoint count if RESUME=1: `{preflight.failed_checkpoint_count}`",
        f"- Pending configs/runs: `{preflight.pending_config_count}`",
        "",
        "## Output Paths",
        "",
        *[f"- {name}: `{path}`" for name, path in preflight.paths.items()],
        "",
        "## Model/Mode Summary",
        "",
        _markdown_table(model_summary, ["model_type", "training_mode", "config_count"]),
        "",
    ]
    if discovery.missing_symbols:
        lines.extend(
            [
                "## Missing Reference Series",
                "",
                (
                    "DRY_RUN can complete with missing reference files so the paths "
                    "and symbols are visible. A normal run fails until all expected "
                    "reference symbols are present."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def asdict_for_preflight(spec: QQQOnlyRunSpec) -> dict[str, object]:
    return {
        "model_type": spec.model_type,
        "training_mode": spec.training_mode,
        "target_name": spec.target_name,
        "feature_set_name": spec.feature_set_name,
        "exposure_mode": spec.exposure_mode,
        "rebalance_frequency": spec.rebalance_frequency,
    }


def _print_plan(
    *,
    plan: QQQOnlyModelZooPlan,
    preflight: QQQOnlyPreflight,
    dry_run: bool,
) -> None:
    discovery = plan.reference_discovery
    print("US Market ML QQQ-only cross-market timing model zoo")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("holdings rule: QQQ/cash only")
    print("ETF rotation holdings: false")
    print("SPY core sleeve: false")
    print("production signal export: false")
    print("model card: false")
    print("production_ready always false")
    print(f"dry_run: {dry_run}")
    print(f"sweep_profile: {plan.sweep_profile}")
    print(f"market dataset glob: {plan.market_dataset_glob}")
    print(f"reference selected glob: {discovery.selected_glob}")
    print(f"reference candidate globs: {list(discovery.candidate_globs)}")
    print(f"reference matched file count: {len(discovery.matched_paths)}")
    print(f"reference expected symbols: {list(discovery.expected_symbols)}")
    print(f"reference found symbols: {list(discovery.found_symbols)}")
    print(f"reference missing symbols: {list(discovery.missing_symbols)}")
    print(f"reference explicitly excluded symbols: {list(['DGS10'])}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print("primary benchmark: QQQ buy-and-hold")
    print("secondary references: cash, SPY buy-and-hold")
    print(f"output directory: {plan.output_dir}")
    print(f"report output directory: {plan.report_output_dir}")
    print(f"results CSV path: {preflight.paths['results']}")
    print(f"daily equity CSV path: {preflight.paths['daily_equity']}")
    print(f"daily drawdown CSV path: {preflight.paths['daily_drawdown']}")
    print(f"exposure log CSV path: {preflight.paths['exposure_log']}")
    print(f"feature summary CSV path: {preflight.paths['feature_summary']}")
    print(f"completed checkpoint path: {preflight.paths['completed_keys']}")
    print(f"failed checkpoint path: {preflight.paths['failed_keys']}")
    print(f"skipped checkpoint path: {preflight.paths['skipped_keys']}")
    print(f"preflight Markdown path: {preflight.paths['preflight_md']}")
    print(f"terminal/progress log path: {preflight.paths['terminal_log']}")
    print(f"total configs/runs: {preflight.total_config_count}")
    print(f"baseline configs/runs: {preflight.baseline_config_count}")
    print(f"model configs/runs: {preflight.model_config_count}")
    print(f"pending configs/runs: {preflight.pending_config_count}")
    print(f"max workers requested: {plan.max_workers}")
    print(f"resume: {plan.resume}")
    if dry_run and discovery.missing_symbols:
        print("normal run status: will fail until missing reference symbols are available")


def _ensure_market_data_available(pattern: str) -> None:
    if glob.has_magic(pattern):
        if glob.glob(pattern, recursive=True):
            return
        raise FileNotFoundError(f"Market dataset glob matched no files: {pattern}")
    if not Path(pattern).exists():
        raise FileNotFoundError(f"Market dataset path does not exist: {pattern}")


def _ensure_reference_series_available(discovery: ReferenceSeriesDiscovery) -> None:
    if discovery.missing_symbols:
        raise FileNotFoundError(
            "Missing required local reference series. "
            f"selected_glob={discovery.selected_glob!r}; "
            f"candidate_globs={list(discovery.candidate_globs)!r}; "
            f"expected_symbols={list(discovery.expected_symbols)!r}; "
            f"found_symbols={list(discovery.found_symbols)!r}; "
            f"missing_symbols={list(discovery.missing_symbols)!r}"
        )


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


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, list(columns)].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _install_interrupt_handlers() -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    raise SystemExit(run_qqq_only_cross_market_timing_model_zoo_from_args())
