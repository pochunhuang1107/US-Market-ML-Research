# Daily Model Signal Runbook

## Purpose

This runbook describes the local daily operational workflow for producing and
checking model signal artifacts before they are handed to the simulation
platform. It is local-only and deterministic: no S3 publishing, no scheduled
automation, no broker integration, and no order placement.

## Current Model

- `model_version`: `linear_smooth10_margin002_v1`
- `candidate`: `baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002`
- `feature_set_version`: `default_features_v1`
- `label_horizon_days`: `20`

## Local Daily Export

Install local development dependencies first:

```bash
python -m pip install -e ".[dev,nn]"
```

Run the latest daily export from a local curated ETF OHLCV file:

```bash
us-market-ml-daily-signal-export \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/model_signals_daily \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --model-version linear_smooth10_margin002_v1 \
  --feature-set-version default_features_v1 \
  --train-start 2018-01-01 \
  --train-end 2022-12-31 \
  --validation-start 2023-01-01 \
  --validation-end 2023-12-31 \
  --label-horizon-days 20 \
  --format parquet \
  --merge-existing
```

Expected output files:

```text
artifacts/model_signals_daily/model_version=<VERSION>/signals.parquet
artifacts/model_signals_daily/model_version=<VERSION>/model_signal_manifest.json
```

For the current model, `<VERSION>` is `linear_smooth10_margin002_v1`.

## Local Healthcheck

Run the healthcheck against the exported artifact and manifest:

```bash
us-market-ml-daily-signal-healthcheck \
  --artifact-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/signals.parquet \
  --manifest-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/model_signal_manifest.json \
  --model-version linear_smooth10_margin002_v1 \
  --min-symbol-count 2 \
  --output-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/healthcheck.csv \
  --warnings-output-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/healthcheck_warnings.txt
```

The command returns `0` when status is `ready` or `warning`, and `1` when
status is `failed`. Do not pass a failed artifact to the simulation platform.

## Inspect Latest Top Signal

Use pandas to inspect the latest top-ranked rows:

```bash
python - <<'PY'
import pandas as pd

path = "artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/signals.parquet"
signals = pd.read_parquet(path)
signals["date"] = pd.to_datetime(signals["date"])
latest = signals["date"].max()
top = signals[(signals["date"] == latest) & (signals["signal_bucket"] == "top")]
print(top.sort_values(["rank", "score", "symbol"], ascending=[True, False, True]))
PY
```

## What Warnings Mean

Warnings mean the artifact contract passed, but an operator should inspect the
result before using it downstream. Common warnings include:

- latest signal date does not match an expected date;
- symbol count is lower than the configured minimum;
- no row has `signal_bucket == "top"`;
- manifest `artifact_path` differs from the checked path;
- export-time warnings were already recorded in `model_signal_manifest.json`.

Warnings are not automatic failures, but they should be resolved or explicitly
accepted before platform ingestion.

## What Not To Do

- Do not treat output as live trading advice.
- Do not publish to the platform if healthcheck status is `failed`.
- Do not include `future_return`, `forward_return`, `realized_return`,
  `target_return`, or prefixed variants of those columns in runtime artifacts.
- Do not add S3 publishing, Lambda/EventBridge scheduling, broker integration,
  or trading actions to this local runbook.

## Later Phases

- S3 publishing for reviewed artifacts.
- Lambda/EventBridge automation for scheduled export and validation.
- Platform scheduled tracking after the local daily workflow is stable.
