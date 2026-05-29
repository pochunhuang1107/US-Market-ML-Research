"""Research-only update workflow for locked QQQ-only RF paper tracking.

This module reruns the locked paper-tracking candidates against local data and
compares the latest QQQ/cash exposure snapshot with the prior local snapshot.
It is deliberately not a production signal export, does not create a model card,
and never writes under ``artifacts/model_signals``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.qqq_only_best_candidate_deep_dive import (
    confirm_held_assets_are_qqq_cash_only,
)
from us_market_ml.qqq_only_cross_market_timing_model_zoo import QQQOnlySpecOutput
from us_market_ml.qqq_only_rf_paper_tracking import (
    CANDIDATE_IDS,
    CANDIDATE_REGISTRY_JSON_NAME,
    CANDIDATE_REGISTRY_NAME,
    INITIAL_SIGNAL_NAME,
    PRIMARY_CANDIDATE_ID,
    REJECTED_REFERENCE_ID,
    SECONDARY_CANDIDATE_ID,
    build_initial_signal_snapshot,
    build_qqq_only_rf_paper_tracking_candidate_registry,
)

OUTPUT_PREFIX = "qqq_only_rf_paper_tracking_update"
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/qqq_only_rf_paper_tracking_updates")

UPDATE_SIGNAL_NAME = f"{OUTPUT_PREFIX}_signal.csv"
UPDATE_SIGNAL_JSON_NAME = f"{OUTPUT_PREFIX}_signal.json"
UPDATE_SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
UPDATE_HISTORY_NAME = f"{OUTPUT_PREFIX}_history.csv"
UPDATE_DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
UPDATE_REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"
EXPOSURE_SNAPSHOT_CHART_NAME = f"{OUTPUT_PREFIX}_exposure_snapshot.png"
SIGNAL_HISTORY_CHART_NAME = f"{OUTPUT_PREFIX}_signal_history.png"

UPDATE_REPORT_SECTIONS: tuple[str, ...] = (
    "## 1. Executive summary.",
    "## 2. Latest paper-tracking signal.",
    "## 3. Primary candidate status.",
    "## 4. Secondary reference status.",
    "## 5. Rejected reference status.",
    "## 6. Exposure changes versus previous snapshot.",
    "## 7. Monitoring notes.",
    "## 8. Recommendation.",
    "## 9. Caveats:",
)

UPDATE_SIGNAL_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "role",
    "as_of_date",
    "latest_available_price_date",
    "target_exposure",
    "effective_exposure",
    "qqq_weight",
    "cash_weight",
    "held_assets",
    "raw_score",
    "threshold",
    "rebalance_weekday",
    "is_rebalance_day",
    "previous_effective_exposure",
    "exposure_changed",
    "next_action_interpretation",
    "production_ready",
)

UPDATE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "latest_available_price_date",
    "primary_candidate_id",
    "primary_current_exposure",
    "primary_raw_score",
    "primary_threshold",
    "primary_held_assets",
    "primary_exposure_changed",
    "secondary_current_exposure",
    "rejected_reference_current_exposure",
    "recommendation",
    "production_ready",
)

UPDATE_DECISION_MATRIX_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "role",
    "as_of_date",
    "latest_available_price_date",
    "previous_effective_exposure",
    "current_effective_exposure",
    "raw_score",
    "threshold",
    "held_assets",
    "exposure_changed",
    "decision_label",
    "recommendation",
    "production_ready",
)

DECISION_LABELS: frozenset[str] = frozenset(
    {"maintain_tracking", "exposure_change_watch", "data_missing", "guardrail_failed"}
)


@dataclass(frozen=True)
class QQQOnlyRFPaperTrackingUpdateArtifacts:
    """In-memory artifacts for one paper-tracking update run."""

    locked_registry: pd.DataFrame
    update_signal: pd.DataFrame
    update_summary: pd.DataFrame
    update_history: pd.DataFrame
    update_decision_matrix: pd.DataFrame
    report_markdown: str


def read_locked_paper_tracking_registry(package_dir: Path) -> pd.DataFrame:
    """Read and validate the locked package registry from the package output."""

    csv_path = package_dir / CANDIDATE_REGISTRY_NAME
    json_path = package_dir / CANDIDATE_REGISTRY_JSON_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing locked candidate registry: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing locked candidate registry JSON: {json_path}")
    registry = pd.read_csv(csv_path)
    expected = build_qqq_only_rf_paper_tracking_candidate_registry()
    _validate_locked_registry(registry, expected)

    json_records = json.loads(json_path.read_text(encoding="utf-8"))
    json_registry = pd.DataFrame(json_records)
    _validate_locked_registry(json_registry, expected)
    return registry.loc[:, list(expected.columns)].copy()


def build_qqq_only_rf_paper_tracking_update_artifacts(
    *,
    locked_registry: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
    package_dir: Path,
    output_dir: Path,
) -> QQQOnlyRFPaperTrackingUpdateArtifacts:
    """Build latest signal, history, decision matrix, and update report."""

    _ensure_not_model_signals_path(output_dir)
    expected_registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    _validate_locked_registry(locked_registry, expected_registry)
    _validate_replay_outputs(locked_registry, replay_outputs)

    replay_exposure_log = _combine_exposure_logs(replay_outputs)
    confirm_held_assets_are_qqq_cash_only(replay_exposure_log)

    previous = _load_previous_signal(package_dir=package_dir, output_dir=output_dir)
    update_signal = build_update_signal_snapshot(
        locked_registry=locked_registry,
        replay_outputs=replay_outputs,
        previous_signal=previous,
    )
    _validate_signal_guardrails(update_signal)

    update_history = build_update_history(
        existing_history_path=output_dir / UPDATE_HISTORY_NAME,
        update_signal=update_signal,
    )
    update_summary = build_update_summary(update_signal)
    update_decision_matrix = build_update_decision_matrix(update_signal)
    report = render_qqq_only_rf_paper_tracking_update_markdown(
        update_signal=update_signal,
        update_summary=update_summary,
        update_history=update_history,
        update_decision_matrix=update_decision_matrix,
    )
    return QQQOnlyRFPaperTrackingUpdateArtifacts(
        locked_registry=locked_registry.copy(),
        update_signal=update_signal,
        update_summary=update_summary,
        update_history=update_history,
        update_decision_matrix=update_decision_matrix,
        report_markdown=report,
    )


def build_update_signal_snapshot(
    *,
    locked_registry: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
    previous_signal: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return latest research-only signal rows with prior exposure comparison."""

    base_signal = build_initial_signal_snapshot(locked_registry, replay_outputs)
    registry_roles = locked_registry.set_index("candidate_id")["role"].to_dict()
    previous_lookup = _previous_effective_exposure_by_candidate(previous_signal)

    rows: list[dict[str, object]] = []
    for row in base_signal.to_dict("records"):
        candidate_id = str(row["candidate_id"])
        current_exposure = _float(row.get("effective_exposure"))
        previous_exposure = previous_lookup.get(candidate_id, np.nan)
        exposure_changed = bool(
            np.isfinite(previous_exposure)
            and np.isfinite(current_exposure)
            and not np.isclose(previous_exposure, current_exposure)
        )
        row["role"] = registry_roles[candidate_id]
        row["previous_effective_exposure"] = previous_exposure
        row["exposure_changed"] = exposure_changed
        row["production_ready"] = False
        rows.append(row)

    signal = pd.DataFrame(rows)
    signal = signal.loc[:, list(UPDATE_SIGNAL_COLUMNS)]
    _validate_signal_guardrails(signal)
    return signal


def build_update_summary(update_signal: pd.DataFrame) -> pd.DataFrame:
    """Build one-row compact update summary."""

    _validate_signal_guardrails(update_signal)
    lookup = update_signal.set_index("candidate_id").to_dict("index")
    primary = lookup[PRIMARY_CANDIDATE_ID]
    secondary = lookup[SECONDARY_CANDIDATE_ID]
    rejected = lookup[REJECTED_REFERENCE_ID]
    label = _overall_recommendation(update_signal)
    row = {
        "latest_available_price_date": primary["latest_available_price_date"],
        "primary_candidate_id": PRIMARY_CANDIDATE_ID,
        "primary_current_exposure": primary["effective_exposure"],
        "primary_raw_score": primary["raw_score"],
        "primary_threshold": primary["threshold"],
        "primary_held_assets": primary["held_assets"],
        "primary_exposure_changed": primary["exposure_changed"],
        "secondary_current_exposure": secondary["effective_exposure"],
        "rejected_reference_current_exposure": rejected["effective_exposure"],
        "recommendation": label,
        "production_ready": False,
    }
    return pd.DataFrame([row], columns=UPDATE_SUMMARY_COLUMNS)


def build_update_history(
    *,
    existing_history_path: Path,
    update_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Append or replace current as-of rows in the local update history."""

    _validate_signal_guardrails(update_signal)
    if existing_history_path.exists():
        history = pd.read_csv(existing_history_path)
    else:
        history = pd.DataFrame(columns=UPDATE_SIGNAL_COLUMNS)
    for column in UPDATE_SIGNAL_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan
    history = history.loc[:, list(UPDATE_SIGNAL_COLUMNS)].copy()

    keys = update_signal[["candidate_id", "as_of_date"]].astype(str)
    existing_keys = history[["candidate_id", "as_of_date"]].astype(str)
    current_key_set = set(map(tuple, keys.to_numpy()))
    keep_mask = [tuple(values) not in current_key_set for values in existing_keys.to_numpy()]
    kept = history.loc[keep_mask].copy()
    combined = pd.concat([kept, update_signal], ignore_index=True)
    combined["production_ready"] = False
    combined = combined.sort_values(
        ["as_of_date", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_signal_guardrails(combined)
    return combined


def build_update_decision_matrix(update_signal: pd.DataFrame) -> pd.DataFrame:
    """Build candidate-level update decision rows."""

    _validate_signal_guardrails(update_signal)
    rows: list[dict[str, object]] = []
    for row in update_signal.to_dict("records"):
        label = _candidate_decision_label(row)
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "role": row["role"],
                "as_of_date": row["as_of_date"],
                "latest_available_price_date": row["latest_available_price_date"],
                "previous_effective_exposure": row["previous_effective_exposure"],
                "current_effective_exposure": row["effective_exposure"],
                "raw_score": row["raw_score"],
                "threshold": row["threshold"],
                "held_assets": row["held_assets"],
                "exposure_changed": row["exposure_changed"],
                "decision_label": label,
                "recommendation": _recommendation_for_label(label, str(row["role"])),
                "production_ready": False,
            }
        )
    matrix = pd.DataFrame(rows, columns=UPDATE_DECISION_MATRIX_COLUMNS)
    if not set(matrix["decision_label"]).issubset(DECISION_LABELS):
        raise ValueError("Unexpected paper-tracking update decision label.")
    return matrix


def render_qqq_only_rf_paper_tracking_update_markdown(
    *,
    update_signal: pd.DataFrame,
    update_summary: pd.DataFrame,
    update_history: pd.DataFrame,
    update_decision_matrix: pd.DataFrame,
) -> str:
    """Render the short research-only update report."""

    primary = update_signal.loc[update_signal["candidate_id"].eq(PRIMARY_CANDIDATE_ID)]
    secondary = update_signal.loc[update_signal["candidate_id"].eq(SECONDARY_CANDIDATE_ID)]
    rejected = update_signal.loc[update_signal["candidate_id"].eq(REJECTED_REFERENCE_ID)]
    changed = update_signal.loc[update_signal["exposure_changed"].astype(bool)]
    recommendation = str(update_summary.iloc[0]["recommendation"])
    latest_date = str(update_summary.iloc[0]["latest_available_price_date"])
    return "\n".join(
        [
            "# QQQ-Only RF Paper-Tracking Update",
            "",
            "## 1. Executive summary.",
            "",
            "- Package type: `research-only paper-tracking update`",
            f"- Latest available QQQ price date: `{latest_date}`",
            "- Tradable holdings: `QQQ`, `cash`",
            "- Production-ready: `false`",
            "- Production signal export: `false`",
            "- Model card: `false`",
            f"- Recommendation: `{recommendation}`",
            "",
            "## 2. Latest paper-tracking signal.",
            "",
            _markdown_table(update_signal, UPDATE_SIGNAL_COLUMNS),
            "",
            f"![Exposure snapshot]({EXPOSURE_SNAPSHOT_CHART_NAME})",
            "",
            "## 3. Primary candidate status.",
            "",
            _markdown_table(primary, UPDATE_SIGNAL_COLUMNS),
            "",
            "## 4. Secondary reference status.",
            "",
            _markdown_table(secondary, UPDATE_SIGNAL_COLUMNS),
            "",
            "## 5. Rejected reference status.",
            "",
            _markdown_table(rejected, UPDATE_SIGNAL_COLUMNS),
            "",
            "## 6. Exposure changes versus previous snapshot.",
            "",
            _markdown_table(changed, UPDATE_SIGNAL_COLUMNS)
            if not changed.empty
            else "_No exposure changes versus the previous local snapshot._",
            "",
            f"![Signal history]({SIGNAL_HISTORY_CHART_NAME})",
            "",
            "## 7. Monitoring notes.",
            "",
            "- Monitor the primary threshold 0.50 row first.",
            "- Keep the threshold 0.45 with min_holding_days 10 row as a secondary reference.",
            "- Compare QQQ/cash exposure, raw score, and exposure changes against prior snapshots.",
            "- Do not change the locked registry from this update workflow.",
            "",
            "## 8. Recommendation.",
            "",
            _markdown_table(update_summary, UPDATE_SUMMARY_COLUMNS),
            "",
            _markdown_table(update_decision_matrix, UPDATE_DECISION_MATRIX_COLUMNS),
            "",
            "## 9. Caveats:",
            "",
            "- Research-only.",
            "- No production signal export.",
            "- No model card.",
            "- QQQ/cash only.",
            "- QQQ buy-and-hold remains the primary benchmark.",
            "- Other ETFs/reference series are features only.",
            "- `production_ready` is always false.",
            "",
            f"Local history rows tracked: `{len(update_history)}`",
            "",
        ]
    )


def write_qqq_only_rf_paper_tracking_update_artifacts(
    *,
    artifacts: QQQOnlyRFPaperTrackingUpdateArtifacts,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the update CSV, JSON, Markdown, and optional chart artifacts."""

    _ensure_not_model_signals_path(output_dir)
    _validate_signal_guardrails(artifacts.update_signal)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = report_output_paths(output_dir)
    artifacts.update_signal.to_csv(paths[UPDATE_SIGNAL_NAME], index=False)
    _write_json_records(artifacts.update_signal, paths[UPDATE_SIGNAL_JSON_NAME])
    artifacts.update_summary.to_csv(paths[UPDATE_SUMMARY_NAME], index=False)
    artifacts.update_history.to_csv(paths[UPDATE_HISTORY_NAME], index=False)
    artifacts.update_decision_matrix.to_csv(paths[UPDATE_DECISION_MATRIX_NAME], index=False)
    paths[UPDATE_REPORT_NAME].write_text(artifacts.report_markdown, encoding="utf-8")
    _write_update_charts(artifacts=artifacts, paths=paths)
    return paths


def report_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all paper-tracking update output paths."""

    names = (
        UPDATE_SIGNAL_NAME,
        UPDATE_SIGNAL_JSON_NAME,
        UPDATE_SUMMARY_NAME,
        UPDATE_HISTORY_NAME,
        UPDATE_DECISION_MATRIX_NAME,
        UPDATE_REPORT_NAME,
        EXPOSURE_SNAPSHOT_CHART_NAME,
        SIGNAL_HISTORY_CHART_NAME,
    )
    return {name: output_dir / name for name in names}


def _load_previous_signal(*, package_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    history_path = output_dir / UPDATE_HISTORY_NAME
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            return history
    initial_path = package_dir / INITIAL_SIGNAL_NAME
    if initial_path.exists():
        initial = pd.read_csv(initial_path)
        if not initial.empty:
            return initial
    return None


def _previous_effective_exposure_by_candidate(
    previous_signal: pd.DataFrame | None,
) -> dict[str, float]:
    if previous_signal is None or previous_signal.empty:
        return {}
    previous = previous_signal.copy()
    if "candidate_id" not in previous or "effective_exposure" not in previous:
        return {}
    if "as_of_date" in previous:
        previous["as_of_date"] = pd.to_datetime(previous["as_of_date"], errors="coerce")
        previous = previous.sort_values("as_of_date", kind="mergesort")
    out: dict[str, float] = {}
    for candidate_id, group in previous.groupby("candidate_id", sort=False):
        value = group.tail(1).iloc[0].get("effective_exposure")
        out[str(candidate_id)] = _float(value)
    return out


def _validate_locked_registry(registry: pd.DataFrame, expected: pd.DataFrame) -> None:
    missing = set(expected.columns).difference(registry.columns)
    if missing:
        raise ValueError("Locked registry is missing columns: " + ", ".join(sorted(missing)))
    actual = registry.loc[:, list(expected.columns)].copy()
    if tuple(actual["candidate_id"].astype(str)) != CANDIDATE_IDS:
        raise ValueError("Locked registry candidate IDs have changed.")
    if not _false_series(actual["production_ready"]).all():
        raise ValueError("Locked registry production_ready must always be false.")
    if set(actual["tradable_holdings"].astype(str)) != {"QQQ,cash"}:
        raise ValueError("Locked registry tradable holdings must be QQQ,cash only.")

    compare_columns = [
        "candidate_id",
        "role",
        "model_type",
        "training_mode",
        "target_name",
        "feature_set_name",
        "exposure_mode",
        "rebalance_frequency",
        "rebalance_weekday",
        "random_state",
        "n_estimators",
        "min_samples_leaf",
        "threshold",
        "min_holding_days",
        "switch_margin",
        "tradable_holdings",
        "primary_benchmark",
        "other_etfs_role",
    ]
    for column in compare_columns:
        actual_values = [_normal_registry_value(value) for value in actual[column]]
        expected_values = [_normal_registry_value(value) for value in expected[column]]
        if actual_values != expected_values:
            raise ValueError(f"Locked registry column {column!r} differs from source of truth.")


def _validate_replay_outputs(
    registry: pd.DataFrame,
    replay_outputs: Mapping[str, QQQOnlySpecOutput],
) -> None:
    missing = [
        candidate_id
        for candidate_id in registry["candidate_id"]
        if candidate_id not in replay_outputs
    ]
    if missing:
        raise ValueError("Missing replay outputs for: " + ", ".join(missing))


def _combine_exposure_logs(replay_outputs: Mapping[str, QQQOnlySpecOutput]) -> pd.DataFrame:
    frames = []
    for candidate_id in CANDIDATE_IDS:
        frame = replay_outputs[candidate_id].exposure_log.copy()
        frame.insert(0, "candidate_id", candidate_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate_signal_guardrails(signal: pd.DataFrame) -> None:
    missing = set(UPDATE_SIGNAL_COLUMNS).difference(signal.columns)
    if missing:
        raise ValueError("Update signal is missing columns: " + ", ".join(sorted(missing)))
    if not _false_series(signal["production_ready"]).all():
        raise ValueError("Update signal production_ready must always be false.")
    held = set(signal["held_assets"].fillna("").astype(str))
    if not held.issubset({"QQQ", "cash", "QQQ,cash"}):
        raise ValueError("Update signal holdings must be QQQ/cash only.")
    weights = pd.to_numeric(signal["qqq_weight"], errors="coerce") + pd.to_numeric(
        signal["cash_weight"],
        errors="coerce",
    )
    if weights.notna().any() and not np.allclose(weights.dropna(), 1.0):
        raise ValueError("Update signal QQQ/cash weights must sum to 1.0.")


def _candidate_decision_label(row: Mapping[str, object]) -> str:
    if str(row.get("held_assets")) not in {"QQQ", "cash", "QQQ,cash"}:
        return "guardrail_failed"
    if not np.isfinite(_float(row.get("effective_exposure"))) or not np.isfinite(
        _float(row.get("raw_score"))
    ):
        return "data_missing"
    if bool(row.get("exposure_changed")):
        return "exposure_change_watch"
    return "maintain_tracking"


def _overall_recommendation(update_signal: pd.DataFrame) -> str:
    labels = {_candidate_decision_label(row) for row in update_signal.to_dict("records")}
    if "guardrail_failed" in labels:
        return "guardrail_failed"
    if "data_missing" in labels:
        return "data_missing"
    if "exposure_change_watch" in labels:
        return "exposure_change_watch"
    return "maintain_tracking"


def _recommendation_for_label(label: str, role: str) -> str:
    if label == "guardrail_failed":
        return "Stop and inspect guardrails before using this research snapshot."
    if label == "data_missing":
        return "Inspect local data freshness before interpreting this snapshot."
    if label == "exposure_change_watch":
        return f"Track the {role} exposure change in the next paper-tracking review."
    return f"Maintain research paper-tracking for the {role}; do not promote to production."


def _write_update_charts(
    *,
    artifacts: QQQOnlyRFPaperTrackingUpdateArtifacts,
    paths: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal = artifacts.update_signal.copy()
    signal["effective_exposure"] = pd.to_numeric(signal["effective_exposure"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        signal["candidate_id"],
        signal["effective_exposure"],
        color=["#2468a2", "#6d7f2a", "#9b4d4d"],
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Effective QQQ exposure")
    ax.set_title("QQQ-only RF paper-tracking update exposure snapshot")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[EXPOSURE_SNAPSHOT_CHART_NAME], dpi=140)
    plt.close(fig)

    history = artifacts.update_history.copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if history.empty:
        ax.text(0.5, 0.5, "No update history rows", ha="center", va="center")
    else:
        history["as_of_date"] = pd.to_datetime(history["as_of_date"], errors="coerce")
        history["effective_exposure"] = pd.to_numeric(
            history["effective_exposure"],
            errors="coerce",
        )
        for candidate_id, group in history.groupby("candidate_id", sort=False):
            group = group.sort_values("as_of_date", kind="mergesort")
            ax.step(
                group["as_of_date"],
                group["effective_exposure"],
                where="post",
                label=str(candidate_id),
            )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Effective QQQ exposure")
    ax.set_title("QQQ-only RF paper-tracking signal history")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths[SIGNAL_HISTORY_CHART_NAME], dpi=140)
    plt.close(fig)


def _write_json_records(frame: pd.DataFrame, path: Path) -> None:
    cleaned = frame.replace({np.nan: None})
    path.write_text(
        json.dumps(cleaned.to_dict(orient="records"), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _ensure_not_model_signals_path(path: Path) -> None:
    parts = tuple(part.lower() for part in Path(path).parts)
    for index, part in enumerate(parts[:-1]):
        if part == "artifacts" and parts[index + 1] == "model_signals":
            raise ValueError("Paper-tracking update must not write under artifacts/model_signals.")


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
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def _normal_registry_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return ""
        if number.is_integer():
            return int(number)
        return number
    return value


def _false_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"false", "0", "0.0"})


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
