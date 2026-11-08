"""Monitoring utilities for Lab 12 and Lab 13 workflows."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class DatasetValidationError(ValueError):
    """Raised when an incoming batch does not match the Telco data contract."""


def read_telco_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"TotalCharges": "string"})


def validate_telco_frame(
    frame: pd.DataFrame,
    reference_columns: list[str],
    *,
    require_target: bool,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate a Telco frame and return a cleaned copy."""
    data = frame.copy()
    expected = set(reference_columns)
    if not require_target:
        expected.discard("Churn")
    actual = set(data.columns)
    missing = expected - actual
    allowed = expected | ({"Churn"} if not require_target and "Churn" in data else set())
    unexpected = actual - allowed
    if missing or unexpected:
        raise DatasetValidationError(
            f"Column mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if require_target and "Churn" not in data:
        raise DatasetValidationError("Labeled Telco rows require a Churn column")
    if data.empty:
        raise DatasetValidationError("Dataset must contain at least one row")
    if "customerID" not in data or data["customerID"].isna().any():
        raise DatasetValidationError("customerID must be present and non-null")
    if data["customerID"].astype(str).str.strip().eq("").any():
        raise DatasetValidationError("customerID must not be blank")
    if data["customerID"].astype(str).duplicated().any():
        raise DatasetValidationError("Dataset contains duplicate customerID values")

    category_source = reference_frame if reference_frame is not None else frame
    reference_categories = {
        column: set(category_source[column].dropna().astype(str).unique())
        for column in category_source.select_dtypes(include=["object", "string", "category"])
        if column not in {"customerID", "TotalCharges", "Churn"}
    }
    for column, categories in reference_categories.items():
        if data[column].isna().any():
            raise DatasetValidationError(f"{column} contains missing values")
        observed = set(data[column].astype(str).unique())
        invalid = observed - categories
        if invalid:
            raise DatasetValidationError(
                f"{column} contains categories not present in the reference: "
                f"{sorted(invalid)}"
            )

    for column in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        if column not in data:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = numeric.isna()
        if column == "TotalCharges":
            blank = data[column].isna() | data[column].astype("string").str.strip().eq("").fillna(False)
            allowed_blank = blank & pd.to_numeric(data["tenure"], errors="coerce").eq(0)
            invalid = (numeric.isna() & ~allowed_blank)
        finite = np.isfinite(numeric.dropna().to_numpy(dtype=float))
        if (
            invalid.any()
            or not finite.all()
            or (numeric.dropna() < 0).any()
        ):
            raise DatasetValidationError(f"{column} must contain finite non-negative numbers")
        if column in {"SeniorCitizen", "tenure"} and not np.equal(
            numeric.dropna().to_numpy(dtype=float),
            np.floor(numeric.dropna().to_numpy(dtype=float)),
        ).all():
            raise DatasetValidationError(f"{column} must contain integers")
        if column == "SeniorCitizen" and not set(numeric.dropna().unique()) <= {0, 1}:
            raise DatasetValidationError("SeniorCitizen must contain only 0 or 1")
        data[column] = numeric

    if "Churn" in data:
        observed_target = set(data["Churn"].dropna().astype(str).str.strip())
        if data["Churn"].isna().any() or not observed_target <= {"Yes", "No"}:
            raise DatasetValidationError("Churn must contain only Yes/No labels")
    if require_target and "Churn" not in data:
        raise DatasetValidationError("Labeled Telco rows require a Churn column")
    return data


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    epsilon = 1e-6
    expected_rate = np.clip(expected / max(expected.sum(), 1), epsilon, None)
    actual_rate = np.clip(actual / max(actual.sum(), 1), epsilon, None)
    expected_rate /= expected_rate.sum()
    actual_rate /= actual_rate.sum()
    value = np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate))
    return float(value)


def _feature_psi(reference: pd.Series, current: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(reference):
        reference_values = pd.to_numeric(reference, errors="coerce")
        current_values = pd.to_numeric(current, errors="coerce")
        baseline = reference_values.dropna().to_numpy(dtype=float)
        observed = current_values.dropna().to_numpy(dtype=float)
        if not len(baseline) or not len(observed):
            if not reference_values.isna().any() and not current_values.isna().any():
                return 0.0
            reference_counts = np.array(
                [
                    int(reference_values.notna().sum()),
                    int(reference_values.isna().sum()),
                ],
                dtype=float,
            )
            current_counts = np.array(
                [
                    int(current_values.notna().sum()),
                    int(current_values.isna().sum()),
                ],
                dtype=float,
            )
            return _psi(reference_counts, current_counts)
        if np.unique(baseline).size <= 10:
            baseline_values = reference_values.fillna("__MISSING__").astype(str)
            current_values = current_values.fillna("__MISSING__").astype(str)
            categories = sorted(set(baseline_values.unique()) | set(current_values.unique()))
            baseline_counts = np.array(
                [int((baseline_values == item).sum()) for item in categories]
            )
            current_counts = np.array(
                [int((current_values == item).sum()) for item in categories]
            )
            return _psi(baseline_counts.astype(float), current_counts.astype(float))
        quantiles = np.unique(np.quantile(baseline, np.linspace(0, 1, 11)))
        if len(quantiles) < 2:
            center = float(baseline[0])
            epsilon = max(abs(center) * 1e-6, 1e-6)
            edges = np.array([-np.inf, center - epsilon, center + epsilon, np.inf])
        else:
            edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
        baseline_counts, _ = np.histogram(baseline, bins=edges)
        observed_counts, _ = np.histogram(observed, bins=edges)
        baseline_counts = np.append(baseline_counts, int(reference_values.isna().sum()))
        observed_counts = np.append(observed_counts, int(current_values.isna().sum()))
        return _psi(baseline_counts.astype(float), observed_counts.astype(float))

    baseline_values = reference.fillna("__MISSING__").astype(str)
    current_values = current.fillna("__MISSING__").astype(str)
    categories = sorted(set(baseline_values.unique()) | set(current_values.unique()))
    baseline_counts = np.array([int((baseline_values == item).sum()) for item in categories])
    current_counts = np.array([int((current_values == item).sum()) for item in categories])
    return _psi(baseline_counts.astype(float), current_counts.astype(float))


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    threshold: float = 0.2,
) -> dict[str, Any]:
    excluded = {"customerID", "Churn"}
    features = [column for column in reference.columns if column not in excluded]
    missing = set(features) - set(current.columns)
    if missing:
        raise DatasetValidationError(f"Current data is missing features: {sorted(missing)}")
    scores = {
        column: _feature_psi(reference[column], current[column])
        for column in features
    }
    drifted = sorted(column for column, score in scores.items() if score >= threshold)
    return {
        "status": "DRIFT_DETECTED" if drifted else "NO_DRIFT",
        "threshold": float(threshold),
        "drifted_features": drifted,
        "feature_psi": scores,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
            if isinstance(event, dict):
                events.append(event)
    return events


def detect_prediction_drift(
    reference_events: list[dict[str, Any]],
    current_events: list[dict[str, Any]],
    *,
    threshold: float = 0.2,
) -> dict[str, Any]:
    def successful(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [event for event in events if event.get("status") == "success"]

    reference = successful(reference_events)
    current = successful(current_events)
    fields = ("prediction", "churn_probability")
    if not reference or not current:
        return {
            "status": "INSUFFICIENT_DATA",
            "threshold": float(threshold),
            "reference_rows": len(reference),
            "current_rows": len(current),
            "feature_psi": {},
            "drifted_features": [],
        }
    scores = {
        field: _feature_psi(
            pd.Series([event.get(field) for event in reference], dtype="float64"),
            pd.Series([event.get(field) for event in current], dtype="float64"),
        )
        for field in fields
    }
    drifted = sorted(field for field, score in scores.items() if score >= threshold)
    return {
        "status": "DRIFT_DETECTED" if drifted else "NO_DRIFT",
        "threshold": float(threshold),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "feature_psi": scores,
        "drifted_features": drifted,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
    }


def detect_concept_drift(
    reference_labels: list[int] | np.ndarray,
    reference_predictions: list[int] | np.ndarray,
    current_labels: list[int] | np.ndarray,
    current_predictions: list[int] | np.ndarray,
    *,
    metric: str = "f1_score",
    maximum_drop: float = 0.1,
) -> dict[str, Any]:
    reference_metrics = classification_metrics(reference_labels, reference_predictions)
    current_metrics = classification_metrics(current_labels, current_predictions)
    if metric not in reference_metrics:
        raise ValueError(f"Unsupported concept-drift metric: {metric}")
    drop = reference_metrics[metric] - current_metrics[metric]
    detected = drop >= maximum_drop
    return {
        "status": "DRIFT_DETECTED" if detected else "NO_DRIFT",
        "metric": metric,
        "maximum_drop": float(maximum_drop),
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "metric_drop": float(drop),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def summarize_api_performance(
    events: list[dict[str, Any]],
    *,
    maximum_p95_latency_ms: float = 1000.0,
    maximum_error_rate: float = 0.05,
) -> dict[str, Any]:
    if not events:
        return {
            "status": "INSUFFICIENT_DATA",
            "request_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "latency_ms": {},
        }
    errors = sum(event.get("status") != "success" for event in events)
    latencies = [
        float(event["latency_ms"])
        for event in events
        if event.get("latency_ms") is not None
        and math.isfinite(float(event["latency_ms"]))
        and float(event["latency_ms"]) >= 0
    ]
    error_rate = errors / len(events)
    latency = {
        "mean": float(np.mean(latencies)) if latencies else None,
        "p50": float(np.percentile(latencies, 50)) if latencies else None,
        "p95": float(np.percentile(latencies, 95)) if latencies else None,
        "max": float(max(latencies)) if latencies else None,
    }
    alert_reasons = []
    if error_rate > maximum_error_rate:
        alert_reasons.append("error_rate_exceeded")
    if latency["p95"] is not None and latency["p95"] > maximum_p95_latency_ms:
        alert_reasons.append("p95_latency_exceeded")
    return {
        "status": "ALERT" if alert_reasons else "HEALTHY",
        "request_count": len(events),
        "error_count": int(errors),
        "error_rate": float(error_rate),
        "latency_ms": latency,
        "thresholds": {
            "maximum_p95_latency_ms": float(maximum_p95_latency_ms),
            "maximum_error_rate": float(maximum_error_rate),
        },
        "alert_reasons": alert_reasons,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def performance_trend(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        timestamp = event.get("timestamp")
        if not timestamp:
            continue
        try:
            moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid event timestamp: {timestamp}") from error
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        day = moment.astimezone(timezone.utc).date().isoformat()
        groups.setdefault(day, []).append(event)
    return {
        "period": "UTC day",
        "windows": [
            {"date": day, **summarize_api_performance(groups[day])}
            for day in sorted(groups)
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def should_trigger_retraining(
    *, drift_detected: bool, new_data_available: bool
) -> bool:
    return bool(drift_detected and new_data_available)


def write_json_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
