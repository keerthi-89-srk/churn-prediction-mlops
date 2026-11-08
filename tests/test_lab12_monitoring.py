import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.lab12_monitoring import (
    DatasetValidationError,
    detect_concept_drift,
    detect_data_drift,
    detect_prediction_drift,
    performance_trend,
    should_trigger_retraining,
    summarize_api_performance,
    validate_telco_frame,
    write_json_report,
)


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _telco_frame_cached() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "churn.csv", dtype={"TotalCharges": "string"})


def _telco_frame() -> pd.DataFrame:
    return _telco_frame_cached().copy()


def test_baseline_against_itself_has_no_data_drift():
    baseline = _telco_frame()
    report = detect_data_drift(baseline, baseline.copy(), threshold=0.2)
    assert report["status"] == "NO_DRIFT"
    assert report["drifted_features"] == []


def test_shifted_telco_rows_detect_feature_drift_without_persisting_fixture():
    baseline = _telco_frame()
    evolved = baseline.iloc[:100].copy()
    evolved["tenure"] = 0
    evolved["MonthlyCharges"] = evolved["MonthlyCharges"] + 200
    evolved["Contract"] = "Two year"
    report = detect_data_drift(baseline, evolved, threshold=0.2)
    assert report["status"] == "DRIFT_DETECTED"
    assert {"tenure", "MonthlyCharges", "Contract"} <= set(report["drifted_features"])


def test_telco_batch_validation_rejects_bank_churn_schema():
    bank_data = pd.read_csv(ROOT / "data" / "raw" / "churn2.csv")
    reference = _telco_frame()
    with pytest.raises(DatasetValidationError, match="Column mismatch"):
        validate_telco_frame(bank_data, list(reference.columns), require_target=True)


def test_telco_batch_validation_allows_unlabeled_customer_batch():
    reference = _telco_frame()
    evolved = reference.iloc[:5].drop(columns="Churn").copy()
    cleaned = validate_telco_frame(
        evolved,
        list(reference.columns),
        require_target=False,
        reference_frame=reference,
    )
    assert "Churn" not in cleaned
    assert len(cleaned) == 5


def test_telco_batch_validation_rejects_duplicate_customer_ids():
    reference = _telco_frame()
    batch = reference.iloc[:2].copy()
    batch.iloc[1, batch.columns.get_loc("customerID")] = batch.iloc[0]["customerID"]
    with pytest.raises(DatasetValidationError, match="duplicate customerID"):
        validate_telco_frame(batch, list(reference.columns), require_target=True)


def test_telco_batch_validation_rejects_categories_outside_baseline():
    reference = _telco_frame()
    batch = reference.iloc[:2].copy()
    batch["Contract"] = "New contract"
    with pytest.raises(DatasetValidationError, match="categories not present in the reference"):
        validate_telco_frame(
            batch,
            list(reference.columns),
            require_target=True,
            reference_frame=reference,
        )


def test_prediction_drift_detects_changed_prediction_distribution():
    reference = [
        {"status": "success", "prediction": 0, "churn_probability": 0.1}
        for _ in range(100)
    ]
    current = [
        {"status": "success", "prediction": 1, "churn_probability": 0.9}
        for _ in range(100)
    ]
    report = detect_prediction_drift(reference, current, threshold=0.2)
    assert report["status"] == "DRIFT_DETECTED"
    assert set(report["drifted_features"]) == {"prediction", "churn_probability"}


def test_prediction_drift_reports_insufficient_history():
    report = detect_prediction_drift([], [], threshold=0.2)
    assert report["status"] == "INSUFFICIENT_DATA"


def test_concept_drift_detects_f1_drop():
    report = detect_concept_drift(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        maximum_drop=0.1,
    )
    assert report["status"] == "DRIFT_DETECTED"
    assert report["metric_drop"] == pytest.approx(1.0)


def test_api_performance_thresholds_and_trend():
    events = [
        {"status": "success", "latency_ms": 10, "timestamp": "2026-01-01T10:00:00+00:00"},
        {"status": "failure", "latency_ms": 2000, "timestamp": "2026-01-02T10:00:00+00:00"},
    ]
    report = summarize_api_performance(
        events,
        maximum_p95_latency_ms=1000,
        maximum_error_rate=0.05,
    )
    assert report["status"] == "ALERT"
    assert report["request_count"] == 2
    assert report["error_count"] == 1
    assert len(performance_trend(events)["windows"]) == 2


@pytest.mark.parametrize(
    ("drift", "data", "expected"),
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_retraining_trigger_requires_drift_and_new_data(drift, data, expected):
    assert should_trigger_retraining(
        drift_detected=drift,
        new_data_available=data,
    ) is expected


def test_report_write_is_json_and_creates_parent_directory(tmp_path):
    path = write_json_report(tmp_path / "nested" / "report.json", {"status": "OK"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "OK"}

