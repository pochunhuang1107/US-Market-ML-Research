#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_paper_tracking_updates}"
  local signal_path="$OUTPUT_DIR/qqq_only_rf_paper_tracking_update_signal.csv"
  local report_path="$OUTPUT_DIR/qqq_only_rf_paper_tracking_update_report.md"

  local latest_price_date
  latest_price_date="$(
    MARKET_DATASET_GLOB="$MARKET_DATASET_GLOB" python - <<'PY'
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd


def read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError):
        return pd.read_parquet(path)


def max_qqq_date(paths: list[Path]) -> pd.Timestamp | None:
    qqq_partition_paths = [
        path for path in paths if any(part.upper() == "SYMBOL=QQQ" for part in path.parts)
    ]
    frames = []
    if qqq_partition_paths:
        for path in qqq_partition_paths:
            frame = read_parquet_columns(path, ["date"])
            if "date" not in frame:
                raise ValueError(f"QQQ parquet file is missing a date column: {path}")
            frames.append(frame[["date"]])
    else:
        for path in paths:
            frame = read_parquet_columns(path, ["date", "symbol"])
            if "date" not in frame or "symbol" not in frame:
                continue
            qqq = frame.loc[frame["symbol"].astype(str).str.upper().eq("QQQ"), ["date"]]
            if not qqq.empty:
                frames.append(qqq)

    if not frames:
        return None
    dates = pd.to_datetime(pd.concat(frames, ignore_index=True)["date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return None
    return dates.max()


pattern = os.environ["MARKET_DATASET_GLOB"]
matched_paths = [Path(path) for path in glob.glob(pattern, recursive=True)]
if not matched_paths:
    print(f"ERROR: no parquet files matched MARKET_DATASET_GLOB={pattern!r}", file=sys.stderr)
    raise SystemExit(1)

latest = max_qqq_date(matched_paths)
if latest is None:
    print(
        "ERROR: QQQ data is missing from local ETF parquet files matched by "
        f"MARKET_DATASET_GLOB={pattern!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(latest.strftime("%Y-%m-%d"))
PY
  )"

  printf 'QQQ-only RF paper-tracking operator check\n'
  printf 'research-only: true\n'
  printf 'local-only: true\n'
  printf 'S3/cloud: false\n'
  printf 'broad sweep: false\n'
  printf 'production signal export: false\n'
  printf 'model card: false\n'
  printf 'production_ready always false\n'
  printf 'tradable holdings: QQQ/cash only\n'
  printf 'output under artifacts/model_signals: false\n'
  printf 'latest local QQQ price date: %s\n\n' "$latest_price_date"

  update_log="$(mktemp "${TMPDIR:-/tmp}/qqq_only_rf_paper_tracking_update.XXXXXX")"
  trap 'rm -f "${update_log:-}"' EXIT

  printf 'Running scripts/update_qqq_only_rf_paper_tracking.sh with END_DATE=%s...\n\n' \
    "$latest_price_date"
  if ! END_DATE="$latest_price_date" bash scripts/update_qqq_only_rf_paper_tracking.sh \
    >"$update_log" 2>&1; then
    printf 'ERROR: QQQ-only RF paper-tracking update failed. Updater log follows:\n\n' >&2
    cat "$update_log" >&2
    exit 1
  fi

  if [[ ! -f "$signal_path" ]]; then
    printf 'ERROR: update signal CSV is missing after update: %s\n' "$signal_path" >&2
    printf 'Updater log follows:\n\n' >&2
    cat "$update_log" >&2
    exit 1
  fi

  SIGNAL_PATH="$signal_path" REPORT_PATH="$report_path" python - <<'PY'
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd


PRIMARY_CANDIDATE_ID = "primary_rf_threshold_050"
CANDIDATE_ORDER = {
    "primary_rf_threshold_050": 0,
    "secondary_rf_threshold_045_hold10": 1,
    "rejected_rf_threshold_045_hold0": 2,
}
SUMMARY_COLUMNS = [
    "candidate_id",
    "latest_available_price_date",
    "raw_score",
    "threshold",
    "effective_exposure",
    "held_assets",
    "exposure_changed",
    "next_action_interpretation",
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def false_like(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "0.0"}


def numeric(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


def render_table(frame: pd.DataFrame) -> str:
    rows = [[format_value(value) for value in row] for row in frame.to_numpy(dtype=object)]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(frame.columns)
    ]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(frame.columns))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def primary_action(primary: pd.Series | None) -> str:
    if primary is None:
        return "REVIEW REQUIRED"
    if not false_like(primary.get("production_ready", False)):
        return "REVIEW REQUIRED"
    held_assets = str(primary.get("held_assets", "")).strip()
    if held_assets not in {"QQQ", "cash", "QQQ,cash"}:
        return "REVIEW REQUIRED"
    exposure = numeric(primary.get("effective_exposure"))
    if not math.isfinite(exposure):
        return "REVIEW REQUIRED"
    if math.isclose(exposure, 1.0, abs_tol=1e-9):
        return "MAINTAIN 100% QQQ"
    if math.isclose(exposure, 0.0, abs_tol=1e-9):
        return "MOVE TO CASH / RISK-OFF"
    return "REVIEW REQUIRED"


signal_path = Path(os.environ["SIGNAL_PATH"])
report_path = Path(os.environ["REPORT_PATH"])
if not signal_path.exists():
    fail(f"update signal CSV is missing after update: {signal_path}")

signal = pd.read_csv(signal_path)
missing_columns = [column for column in SUMMARY_COLUMNS if column not in signal.columns]
if missing_columns:
    fail("update signal CSV is missing columns: " + ", ".join(missing_columns))

missing_candidates = [
    candidate_id for candidate_id in CANDIDATE_ORDER if candidate_id not in set(signal["candidate_id"])
]
if missing_candidates:
    fail("update signal CSV is missing candidates: " + ", ".join(missing_candidates))

production_values = signal.get("production_ready", pd.Series([False] * len(signal)))
if not production_values.map(false_like).all():
    fail("update signal CSV has production_ready values other than false")

held_assets = set(signal["held_assets"].fillna("").astype(str))
if not held_assets.issubset({"QQQ", "cash", "QQQ,cash"}):
    fail("update signal CSV contains non-QQQ/cash held assets")

summary = signal.loc[signal["candidate_id"].isin(CANDIDATE_ORDER), SUMMARY_COLUMNS].copy()
summary["_order"] = summary["candidate_id"].map(CANDIDATE_ORDER)
summary = summary.sort_values("_order", kind="mergesort").drop(columns=["_order"])
primary_rows = signal.loc[signal["candidate_id"].eq(PRIMARY_CANDIDATE_ID)]
primary = None if primary_rows.empty else primary_rows.iloc[0]

print("Compact final summary:")
print(f"PRIMARY CANDIDATE: {PRIMARY_CANDIDATE_ID}")
print(render_table(summary))
print()
print(f"PRIMARY ACTION: {primary_action(primary)}")
print(f"Report path: {report_path}")
PY
}

main "$@"
