"""Run a local-only extended-history overnight robustness study."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    drop_rows_with_missing_model_values,
    get_feature_columns,
    read_local_market_data,
)
from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    build_sweep_run_key,
    run_sweep,
)
from us_market_ml.splits import (
    PurgeConfig,
    WalkForwardFold,
    make_sliding_window_folds_for_train_windows,
    purge_fold_boundaries,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_extended_tiingo_split/**/*.parquet"
)
DEFAULT_RUN_NAME = "extended_history_overnight_study"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/extended_history_overnight_study")
DEFAULT_DATA_START = "1993-01-29"
DEFAULT_DATA_END = "2026-05-15"
CHAMPION_NAME = "linear_smooth10_margin002_v1"
TARGET_COL = "excess_return_vs_spy_20d"
FEATURE_SET_VERSION = "default_features_v1"
LABEL_HORIZON_DAYS = 20
MIN_HOLDING_DAYS = 10
SWITCH_MARGIN = 0.002
RANDOM_STATE = 42

FOLD_CATALOG_NAME = "extended_history_fold_catalog.csv"
RANKING_NAME = "extended_history_candidate_rank.csv"
FOLD_DETAILS_NAME = "extended_history_fold_details.csv"
REGIME_NAME = "extended_history_regime_summary.csv"
REPORT_NAME = "extended_history_report.md"
INVALID_FOLD_REPORT_NAME = "extended_history_invalid_fold_report.csv"
MIN_USABLE_TRAIN_ROWS = 1
MIN_USABLE_TEST_ROWS = 1
MIN_TRAIN_SYMBOL_COUNT = 1
MIN_TEST_SYMBOL_COUNT = 1
INVALID_FOLD_REPORT_COLUMNS = [
    "key",
    "fold_name",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "model_type",
    "allocation_mode",
    "top_k",
    "transaction_cost_bps",
    "reason",
    "usable_train_rows",
    "usable_test_rows",
    "train_symbol_count",
    "test_symbol_count",
]


@dataclass(frozen=True)
class ExtendedFold:
    """A sweep fold plus report metadata."""

    fold: SweepFoldConfig
    study_design: str
    regime: str


@dataclass(frozen=True)
class ExtendedHistoryPreflight:
    """Pre-dispatch validity outputs for the extended-history sweep."""

    skipped_keys_path: Path
    invalid_fold_report: Path
    skipped_config_count: int
    valid_config_count: int
    total_config_count: int


def run_extended_history_overnight_study_from_args(argv: list[str] | None = None) -> int:
    """Run the extended-history study and write summary report artifacts."""

    args = _parse_args(argv)
    folds = build_extended_history_folds(args.data_start, args.data_end)
    if not folds:
        raise SystemExit("No extended-history folds were generated.")

    config = build_extended_history_sweep_config(
        market_dataset_glob=args.market_dataset_glob,
        folds=[item.fold for item in folds],
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
    )
    preflight = preflight_extended_history_invalid_folds(config=config)
    config = SweepRunConfig(
        universes=config.universes,
        folds=config.folds,
        models=config.models,
        signals=config.signals,
        target_col=config.target_col,
        initial_equity=config.initial_equity,
        prediction_col=config.prediction_col,
        label_horizon_days=config.label_horizon_days,
        feature_set_versions=config.feature_set_versions,
        min_holding_days=config.min_holding_days,
        switch_margin=config.switch_margin,
        transaction_cost_bps_values=config.transaction_cost_bps_values,
        output_dir=config.output_dir,
        run_name=config.run_name,
        resume=config.resume,
        max_workers=config.max_workers,
        skipped_keys_path=preflight.skipped_keys_path,
        skip_no_usable_training_rows=True,
    )
    fold_catalog = _write_fold_catalog(folds, args.output_dir)

    if args.dry_run:
        _print_plan(
            config=config,
            folds=folds,
            fold_catalog=fold_catalog,
            preflight=preflight,
            dry_run=True,
        )
        return 0

    if not args.report_only:
        _print_plan(
            config=config,
            folds=folds,
            fold_catalog=fold_catalog,
            preflight=preflight,
            dry_run=False,
        )
        run_sweep(config)

    outputs = build_extended_history_report(
        results_csv=args.output_dir / f"{args.run_name}_results.csv",
        completed_keys=args.output_dir / f"{args.run_name}_completed_keys.txt",
        failed_keys=args.output_dir / f"{args.run_name}_failed_keys.txt",
        fold_catalog=fold_catalog,
        output_dir=args.output_dir,
    )
    print()
    print("Extended-history report artifacts")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def build_extended_history_folds(data_start: str, data_end: str) -> list[ExtendedFold]:
    """Build fixed-regime plus annual longer-history rolling folds."""

    fixed_specs = [
        (
            "regime_dot_com",
            "dot_com",
            "1993-01-29",
            "1998-12-31",
            "1999-01-01",
            "1999-12-31",
            "2000-01-01",
            "2002-12-31",
        ),
        (
            "regime_gfc",
            "gfc",
            "1999-01-01",
            "2005-12-31",
            "2006-01-01",
            "2006-12-31",
            "2007-01-01",
            "2009-12-31",
        ),
        (
            "regime_post_gfc",
            "post_gfc",
            "2003-01-01",
            "2009-12-31",
            "2010-01-01",
            "2010-12-31",
            "2011-01-01",
            "2014-12-31",
        ),
        (
            "regime_late_cycle",
            "late_cycle",
            "2007-01-01",
            "2013-12-31",
            "2014-01-01",
            "2014-12-31",
            "2015-01-01",
            "2018-12-31",
        ),
        (
            "regime_covid_rebound",
            "covid_2020",
            "2011-01-01",
            "2017-12-31",
            "2018-01-01",
            "2019-12-31",
            "2020-01-01",
            "2021-12-31",
        ),
        (
            "regime_inflation_2022",
            "inflation_2022",
            "2014-01-01",
            "2020-12-31",
            "2021-01-01",
            "2021-12-31",
            "2022-01-01",
            "2023-12-31",
        ),
        (
            "regime_current",
            "current_2024_2026",
            "2016-01-01",
            "2022-12-31",
            "2023-01-01",
            "2023-12-31",
            "2024-01-01",
            data_end,
        ),
    ]

    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)
    folds: list[ExtendedFold] = []
    for (
        name,
        regime,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
    ) in fixed_specs:
        fold = SweepFoldConfig(
            name=name,
            train_start=_bounded_date(train_start, lower=start),
            train_end=_bounded_date(train_end, lower=start),
            validation_start=_bounded_date(val_start, lower=start),
            validation_end=_bounded_date(val_end, upper=end),
            test_start=_bounded_date(test_start, lower=start),
            test_end=_bounded_date(test_end, upper=end),
        )
        if pd.Timestamp(fold.train_start) <= pd.Timestamp(fold.train_end) and pd.Timestamp(
            fold.test_start
        ) <= pd.Timestamp(fold.test_end):
            folds.append(ExtendedFold(fold=fold, study_design="fixed_regime", regime=regime))

    rolling = make_sliding_window_folds_for_train_windows(
        data_start=data_start,
        data_end=data_end,
        train_window_years_values=(5, 10),
        validation_window_months=12,
        test_window_months=12,
        step_months=12,
        include_partial_final_test=True,
        min_partial_test_days=60,
    )
    for fold in rolling:
        prefixed = _to_sweep_fold_config(fold, prefix="rolling")
        folds.append(
            ExtendedFold(
                fold=prefixed,
                study_design="annual_longer_history",
                regime=_regime_for_test_period(prefixed.test_start, prefixed.test_end),
            )
        )

    seen: set[str] = set()
    unique: list[ExtendedFold] = []
    for item in folds:
        if item.fold.name in seen:
            continue
        seen.add(item.fold.name)
        unique.append(item)
    return unique


def build_extended_history_sweep_config(
    *,
    market_dataset_glob: str,
    folds: list[SweepFoldConfig],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
) -> SweepRunConfig:
    """Build the focused champion-family allocation robustness sweep."""

    return SweepRunConfig(
        universes=[SweepUniverseConfig(name="extended_full", data_path=market_dataset_glob)],
        folds=folds,
        models=[
            SweepModelConfig(
                model_family="baseline",
                model_type="linear_regression",
                random_state=RANDOM_STATE,
            )
        ],
        signals=[
            _signal(top_k=1, allocation_mode="equal_weight"),
            _signal(top_k=3, allocation_mode="equal_weight"),
            _signal(top_k=5, allocation_mode="equal_weight"),
            _signal(top_k=3, allocation_mode="score_weighted"),
            _signal(top_k=5, allocation_mode="score_weighted"),
            _signal(top_k=3, allocation_mode="softmax_score_weighted", score_temperature=0.5),
            _signal(top_k=5, allocation_mode="softmax_score_weighted", score_temperature=1.0),
            _signal(top_k=3, allocation_mode="capped_score_weighted", max_weight=0.5),
            _signal(top_k=5, allocation_mode="capped_score_weighted", max_weight=0.4),
        ],
        target_col=TARGET_COL,
        label_horizon_days=LABEL_HORIZON_DAYS,
        feature_set_versions=[FEATURE_SET_VERSION],
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        transaction_cost_bps_values=[10.0, 20.0],
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
        skipped_keys_path=output_dir / f"{run_name}_skipped_keys.txt",
        skip_no_usable_training_rows=True,
    )


def preflight_extended_history_invalid_folds(
    *,
    config: SweepRunConfig,
) -> ExtendedHistoryPreflight:
    """Write skipped extended-history configs that cannot fit a model."""

    output_dir = Path(config.output_dir)
    skipped_keys_path = output_dir / f"{config.run_name}_skipped_keys.txt"
    invalid_fold_report = output_dir / INVALID_FOLD_REPORT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    total_config_count = _independent_config_count(config)
    invalid_rows = _build_invalid_fold_report_rows(config)
    invalid_keys = {str(row["key"]) for row in invalid_rows}
    existing_skipped = _read_key_checkpoint(skipped_keys_path) if config.resume else set()
    skipped_keys = sorted(existing_skipped.union(invalid_keys))

    skipped_keys_path.write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )
    pd.DataFrame(invalid_rows, columns=INVALID_FOLD_REPORT_COLUMNS).to_csv(
        invalid_fold_report,
        index=False,
    )
    return ExtendedHistoryPreflight(
        skipped_keys_path=skipped_keys_path,
        invalid_fold_report=invalid_fold_report,
        skipped_config_count=len(skipped_keys),
        valid_config_count=max(total_config_count - len(skipped_keys), 0),
        total_config_count=total_config_count,
    )


def _build_invalid_fold_report_rows(config: SweepRunConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    validity = _estimate_fold_validity(config)
    for universe in config.universes:
        for fold_config in config.folds:
            for model_config in config.models:
                for signal_config in config.signals:
                    for feature_set_version in config.feature_set_versions:
                        fold_validity = validity[
                            (str(universe.data_path), fold_config.name, feature_set_version)
                        ]
                        for transaction_cost_bps in config.transaction_cost_bps_values:
                            reason = _invalid_fold_reason(
                                usable_train_rows=fold_validity["usable_train_rows"],
                                usable_test_rows=fold_validity["usable_test_rows"],
                                train_symbol_count=fold_validity["train_symbol_count"],
                                test_symbol_count=fold_validity["test_symbol_count"],
                            )
                            if reason is None:
                                continue
                            run_key = build_sweep_run_key(
                                universe=universe,
                                fold=fold_config,
                                model=model_config,
                                signal=signal_config,
                                label_horizon_days=config.label_horizon_days,
                                min_holding_days=config.min_holding_days,
                                switch_margin=config.switch_margin,
                                transaction_cost_bps=transaction_cost_bps,
                                feature_set_version=feature_set_version,
                            )
                            rows.append(
                                {
                                    "key": run_key,
                                    "fold_name": fold_config.name,
                                    "train_start": fold_config.train_start,
                                    "train_end": fold_config.train_end,
                                    "test_start": fold_config.test_start,
                                    "test_end": fold_config.test_end,
                                    "model_type": model_config.model_type,
                                    "allocation_mode": signal_config.allocation_mode,
                                    "top_k": signal_config.top_k,
                                    "transaction_cost_bps": transaction_cost_bps,
                                    "reason": reason,
                                    "usable_train_rows": fold_validity["usable_train_rows"],
                                    "usable_test_rows": fold_validity["usable_test_rows"],
                                    "train_symbol_count": fold_validity["train_symbol_count"],
                                    "test_symbol_count": fold_validity["test_symbol_count"],
                                }
                            )
    return rows


def _estimate_fold_validity(config: SweepRunConfig) -> dict[tuple[str, str, str], dict[str, int]]:
    estimates: dict[tuple[str, str, str], dict[str, int]] = {}
    for universe in config.universes:
        raw_df = read_local_market_data(universe.data_path)
        for feature_set_version in config.feature_set_versions:
            dataset = build_supervised_dataset(
                raw_df,
                SupervisedDatasetConfig(feature_set_version=feature_set_version),
            )
            feature_cols = get_feature_columns(dataset, feature_set_version=feature_set_version)
            if config.target_col not in dataset.columns:
                raise ValueError(
                    "Configured target_col is missing after supervised dataset construction: "
                    f"{config.target_col}"
                )
            for fold_config in config.folds:
                fold = WalkForwardFold(
                    name=fold_config.name,
                    train_start=pd.Timestamp(fold_config.train_start),
                    train_end=pd.Timestamp(fold_config.train_end),
                    validation_start=pd.Timestamp(fold_config.validation_start),
                    validation_end=pd.Timestamp(fold_config.validation_end),
                    test_start=pd.Timestamp(fold_config.test_start),
                    test_end=pd.Timestamp(fold_config.test_end),
                )
                train_df, _, test_df = _split_for_preflight(dataset, fold, config)
                usable_train = drop_rows_with_missing_model_values(
                    train_df,
                    feature_cols,
                    config.target_col,
                )
                usable_test = drop_rows_with_missing_model_values(
                    test_df,
                    feature_cols,
                    config.target_col,
                )
                estimates[(str(universe.data_path), fold_config.name, feature_set_version)] = {
                    "usable_train_rows": int(len(usable_train)),
                    "usable_test_rows": int(len(usable_test)),
                    "train_symbol_count": _symbol_count(usable_train),
                    "test_symbol_count": _symbol_count(usable_test),
                }
    return estimates


def _split_for_preflight(
    dataset: pd.DataFrame,
    fold: WalkForwardFold,
    config: SweepRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.label_horizon_days <= 0:
        return fold.split(dataset)
    return purge_fold_boundaries(
        dataset,
        fold,
        PurgeConfig(label_horizon_days=config.label_horizon_days),
    )


def _invalid_fold_reason(
    *,
    usable_train_rows: int,
    usable_test_rows: int,
    train_symbol_count: int,
    test_symbol_count: int,
) -> str | None:
    if usable_train_rows < MIN_USABLE_TRAIN_ROWS:
        return "skipped_invalid_fold: insufficient_usable_train_rows"
    if usable_test_rows < MIN_USABLE_TEST_ROWS:
        return "skipped_invalid_fold: insufficient_usable_test_rows"
    if train_symbol_count < MIN_TRAIN_SYMBOL_COUNT:
        return "skipped_invalid_fold: insufficient_train_symbol_count"
    if test_symbol_count < MIN_TEST_SYMBOL_COUNT:
        return "skipped_invalid_fold: insufficient_test_symbol_count"
    return None


def _symbol_count(df: pd.DataFrame) -> int:
    if "symbol" not in df.columns or df.empty:
        return 0
    return int(df["symbol"].dropna().nunique())


def _independent_config_count(config: SweepRunConfig) -> int:
    return (
        len(config.universes)
        * len(config.folds)
        * len(config.models)
        * len(config.signals)
        * len(config.feature_set_versions)
        * len(config.transaction_cost_bps_values)
    )


def _read_key_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_extended_history_report(
    *,
    results_csv: Path,
    completed_keys: Path,
    failed_keys: Path,
    fold_catalog: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Aggregate completed extended-history sweep rows into report artifacts."""

    if not results_csv.exists():
        raise FileNotFoundError(f"Missing extended-history results CSV: {results_csv}")
    results = pd.read_csv(results_csv)
    folds = pd.read_csv(fold_catalog)
    test_rows = results.loc[results["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("No split == 'test' rows found in extended-history results.")

    for column in _numeric_columns():
        if column in test_rows:
            test_rows[column] = pd.to_numeric(test_rows[column], errors="coerce")

    group_cols = [
        "model_family",
        "model_type",
        "feature_set_version",
        "top_k",
        "min_holding_days",
        "switch_margin",
        "allocation_mode",
        "score_temperature",
        "max_weight",
        "transaction_cost_bps",
        "target_col",
        "label_horizon_days",
        "random_state",
    ]
    ranking = (
        test_rows.groupby(group_cols, dropna=False)
        .agg(
            total_return=("total_return", "mean"),
            median_total_return=("total_return", "median"),
            worst_fold_total_return=("total_return", "min"),
            sharpe_like=("sharpe_like", "mean"),
            max_drawdown=("max_drawdown", "min"),
            mean_max_drawdown=("max_drawdown", "mean"),
            final_equity=("final_equity", "mean"),
            mean_daily_turnover=("mean_daily_turnover", "mean"),
            total_transaction_cost=("total_transaction_cost", "mean"),
            positive_fold_rate=("total_return", lambda values: float((values > 0).mean())),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    ranking["candidate_spec"] = ranking.apply(_candidate_spec, axis=1)
    ranking["allocation_label"] = ranking.apply(_allocation_label, axis=1)
    ranking["balanced_score"] = _balanced_score(ranking)
    ranking = ranking.sort_values(
        ["balanced_score", "total_return", "sharpe_like"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    fold_details = test_rows.merge(
        folds,
        on="fold",
        how="left",
        validate="many_to_one",
    )
    period_summary = (
        fold_details.groupby(group_cols, dropna=False)
        .agg(
            first_test_start=("test_start", "min"),
            last_test_end=("test_end", "max"),
            fold_periods=("fold", _join_unique),
        )
        .reset_index()
    )
    ranking = ranking.merge(period_summary, on=group_cols, how="left", validate="one_to_one")
    regime_summary = _build_regime_summary(fold_details, group_cols)
    report = _render_report(
        ranking=ranking,
        regime_summary=regime_summary,
        completed_keys=completed_keys,
        failed_keys=failed_keys,
        fold_catalog=fold_catalog,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rank CSV": output_dir / RANKING_NAME,
        "fold detail CSV": output_dir / FOLD_DETAILS_NAME,
        "regime CSV": output_dir / REGIME_NAME,
        "markdown report": output_dir / REPORT_NAME,
    }
    ranking.to_csv(outputs["rank CSV"], index=False)
    fold_details.to_csv(outputs["fold detail CSV"], index=False)
    regime_summary.to_csv(outputs["regime CSV"], index=False)
    outputs["markdown report"].write_text(report, encoding="utf-8")
    return outputs


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


def _write_fold_catalog(folds: Sequence[ExtendedFold], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in folds:
        rows.append(
            {
                "fold": item.fold.name,
                "study_design": item.study_design,
                "regime": item.regime,
                "train_start": item.fold.train_start,
                "train_end": item.fold.train_end,
                "validation_start": item.fold.validation_start,
                "validation_end": item.fold.validation_end,
                "test_start": item.fold.test_start,
                "test_end": item.fold.test_end,
            }
        )
    path = output_dir / FOLD_CATALOG_NAME
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _build_regime_summary(fold_details: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    available_group_cols = [column for column in group_cols if column in fold_details.columns]
    summary = (
        fold_details.groupby([*available_group_cols, "regime"], dropna=False)
        .agg(
            total_return=("total_return", "mean"),
            sharpe_like=("sharpe_like", "mean"),
            max_drawdown=("max_drawdown", "min"),
            final_equity=("final_equity", "mean"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    summary["candidate_spec"] = summary.apply(_candidate_spec, axis=1)
    summary["allocation_label"] = summary.apply(_allocation_label, axis=1)
    return summary.sort_values(
        ["regime", "total_return", "sharpe_like"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _render_report(
    *,
    ranking: pd.DataFrame,
    regime_summary: pd.DataFrame,
    completed_keys: Path,
    failed_keys: Path,
    fold_catalog: Path,
) -> str:
    champion_mask = (
        (ranking["top_k"] == 1)
        & (ranking["allocation_mode"] == "equal_weight")
        & np.isclose(ranking["transaction_cost_bps"], 10.0)
    )
    champion = ranking.loc[champion_mask].iloc[0] if champion_mask.any() else ranking.iloc[0]
    best = ranking.iloc[0]
    allocation_rows = ranking.loc[ranking["top_k"] > 1]
    best_drawdown = (
        allocation_rows.sort_values("max_drawdown", ascending=False).iloc[0]
        if not allocation_rows.empty
        else champion
    )
    failed_count = _count_lines(failed_keys)
    completed_count = _count_lines(completed_keys)
    top_table = _markdown_table(
        ranking.head(8),
        [
            "candidate_spec",
            "allocation_label",
            "transaction_cost_bps",
            "total_return",
            "sharpe_like",
            "max_drawdown",
            "final_equity",
            "mean_daily_turnover",
            "first_test_start",
            "last_test_end",
            "fold_count",
        ],
    )
    regime_table = _markdown_table(
        regime_summary.loc[regime_summary["candidate_spec"] == champion["candidate_spec"]],
        ["regime", "total_return", "sharpe_like", "max_drawdown", "final_equity", "fold_count"],
    )
    decision = (
        "Keep the current champion family as the research champion."
        if best["candidate_spec"] == champion["candidate_spec"]
        else "Promote the top-ranked allocation variant for focused follow-up."
    )
    return "\n".join(
        [
            "# Extended-History Overnight Study Report",
            "",
            f"Champion reference: `{CHAMPION_NAME}`.",
            f"Completed configs: {completed_count}. Failed configs: {failed_count}.",
            f"Fold catalog: `{fold_catalog}`.",
            "",
            "## Top Candidates",
            "",
            top_table,
            "",
            "## Champion Regime Readout",
            "",
            regime_table,
            "",
            "## Research Questions",
            "",
            f"- Champion across regimes: `{champion['candidate_spec']}` averaged "
            f"{_fmt(champion['total_return'])} total_return with worst-fold drawdown "
            f"{_fmt(champion['max_drawdown'])}.",
            f"- Allocation drawdown: best allocation drawdown row is "
            f"`{best_drawdown['candidate_spec']}` with max_drawdown "
            f"{_fmt(best_drawdown['max_drawdown'])}.",
            f"- Overfitting check: champion positive-fold rate is "
            f"{_fmt(champion['positive_fold_rate'])}; compare this with the top ranked "
            "row before promoting anything.",
            f"- Current recommendation: {decision} Focus candidate "
            f"`{best['candidate_spec']}`.",
            "",
            "Selected-symbol concentration is not available in the sweep result CSV; "
            "this report includes turnover/cost columns where the sweep runner captures them.",
            "",
        ]
    )


def _print_plan(
    *,
    config: SweepRunConfig,
    folds: Sequence[ExtendedFold],
    fold_catalog: Path,
    preflight: ExtendedHistoryPreflight,
    dry_run: bool,
) -> None:
    output_dir = Path(config.output_dir)
    independent_configs = preflight.total_config_count
    print("US Market ML extended-history overnight study")
    print("local-only: true")
    print("uses S3: false")
    print("includes GPU/NN work: false")
    print(f"dry_run: {dry_run}")
    print(f"dataset glob: {config.universes[0].data_path}")
    print(f"run_name: {config.run_name}")
    print(f"output_dir: {output_dir}")
    print(f"results CSV: {output_dir / f'{config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{config.run_name}_failed_keys.txt'}")
    print(f"skipped checkpoint: {preflight.skipped_keys_path}")
    print(f"invalid fold report: {preflight.invalid_fold_report}")
    print(f"fold catalog: {fold_catalog}")
    print(f"fold count: {len(folds)}")
    print(f"signal count: {len(config.signals)}")
    print(f"transaction costs: {config.transaction_cost_bps_values}")
    print(f"estimated independent configs: {independent_configs}")
    print(f"estimated valid configs: {preflight.valid_config_count}")
    print(f"estimated skipped invalid configs: {preflight.skipped_config_count}")
    print(f"max workers: {config.max_workers}")
    print(f"resume: {config.resume}")
    print()
    print("Allocation modes:")
    for signal in config.signals:
        print(
            "- "
            f"top_k={signal.top_k}, allocation_mode={signal.allocation_mode}, "
            f"score_temperature={signal.score_temperature:g}, max_weight={signal.max_weight:g}"
        )
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(config)))


def _equivalent_command(config: SweepRunConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_extended_history_overnight_study",
        "--run-name",
        config.run_name,
        "--output-dir",
        str(config.output_dir),
        "--market-dataset-glob",
        str(config.universes[0].data_path),
        "--max-workers",
        str(config.max_workers),
    ]
    if not config.resume:
        command.append("--no-resume")
    return command


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local extended ETF history champion/allocation robustness study."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
        help="Local parquet dataset glob. Defaults to MARKET_DATASET_GLOB or the local dataset.",
    )
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from local checkpoint files by default.",
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
    if pd.Timestamp(args.data_start) > pd.Timestamp(args.data_end):
        parser.error("--data-start must be earlier than or equal to --data-end.")
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


def _bounded_date(
    value: str,
    *,
    lower: pd.Timestamp | None = None,
    upper: pd.Timestamp | None = None,
) -> str:
    timestamp = pd.Timestamp(value)
    if lower is not None:
        timestamp = max(timestamp, lower)
    if upper is not None:
        timestamp = min(timestamp, upper)
    return timestamp.date().isoformat()


def _to_sweep_fold_config(fold: WalkForwardFold, *, prefix: str) -> SweepFoldConfig:
    return SweepFoldConfig(
        name=f"{prefix}_{fold.name}",
        train_start=_date_string(fold.train_start),
        train_end=_date_string(fold.train_end),
        validation_start=_date_string(fold.validation_start),
        validation_end=_date_string(fold.validation_end),
        test_start=_date_string(fold.test_start),
        test_end=_date_string(fold.test_end),
    )


def _date_string(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _regime_for_test_period(test_start: str, test_end: str) -> str:
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    if start <= pd.Timestamp("2002-12-31") and end >= pd.Timestamp("2000-01-01"):
        return "dot_com"
    if start <= pd.Timestamp("2009-12-31") and end >= pd.Timestamp("2007-01-01"):
        return "gfc"
    if start <= pd.Timestamp("2021-12-31") and end >= pd.Timestamp("2020-01-01"):
        return "covid_2020"
    if start <= pd.Timestamp("2023-12-31") and end >= pd.Timestamp("2022-01-01"):
        return "inflation_2022"
    if start >= pd.Timestamp("2024-01-01"):
        return "current_2024_2026"
    return "ordinary"


def _candidate_spec(row: pd.Series) -> str:
    return (
        f"{row['model_family']}:{row['model_type']}:{row['feature_set_version']}:"
        f"top{int(row['top_k'])}:{row['allocation_mode']}:"
        f"temp{float(row['score_temperature']):g}:maxw{float(row['max_weight']):g}:"
        f"cost{float(row['transaction_cost_bps']):g}bps"
    )


def _allocation_label(row: pd.Series) -> str:
    top_k = int(row["top_k"])
    mode = str(row["allocation_mode"])
    if mode == "equal_weight":
        return f"top{top_k} equal weight"
    if mode == "softmax_score_weighted":
        return f"top{top_k} softmax score weighted"
    if mode == "capped_score_weighted":
        return f"top{top_k} capped score weighted"
    return f"top{top_k} {mode.replace('_', ' ')}"


def _balanced_score(frame: pd.DataFrame) -> pd.Series:
    return (
        _zscore(frame["total_return"])
        + _zscore(frame["sharpe_like"])
        + _zscore(frame["max_drawdown"])
        + _zscore(frame["final_equity"])
        - _zscore(frame["mean_daily_turnover"])
    ) / 5.0


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std


def _numeric_columns() -> list[str]:
    return [
        "top_k",
        "min_holding_days",
        "switch_margin",
        "score_temperature",
        "max_weight",
        "transaction_cost_bps",
        "label_horizon_days",
        "random_state",
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
        "mean_daily_turnover",
        "total_transaction_cost",
    ]


def _join_unique(values: pd.Series) -> str:
    return ",".join(sorted(str(value) for value in values.dropna().unique()))


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    formatted = subset.apply(lambda column: column.map(_format_table_value))
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def _format_table_value(value: object) -> str:
    if isinstance(value, float | np.floating):
        return _fmt(float(value))
    return str(value)


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _format_command(command: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(run_extended_history_overnight_study_from_args())
