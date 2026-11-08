# Labs 12–13 — Final Validation Summary

## Result

Lab 12 monitoring and Lab 13 guarded orchestration are implemented. The
existing Production model, locally verified Docker image/container, data, and
Kubernetes configuration were not changed. No actual evolved Telco batch was
available, so positive-drift and retraining validation used temporary
synthetic Telco rows; `churn2.csv` was rejected because it is a bank dataset.

## Actual checks

| Check | Result |
|---|---|
| Python compilation | 41 project Python files compiled without writing bytecode |
| Full test suite | **62 passed, 4 warnings, in 89.14 seconds** |
| Model prediction compatibility | Registered and deployed models produced matching predictions on all 1,409 test records |
| Preprocessing compatibility | Registered and deployed preprocessors produced identical transformations on all 1,409 test records; maximum difference was 0.0 |
| Preprocessor serialization | SHA-256 hashes differ; transformation equivalence does not mean byte-for-byte identity |
| Isolated drift/retraining | Drift detected; candidate trained and MLflow metrics/artifacts logged; no promotion |
| Isolated MLflow lifecycle | Rejection preserved Production; eligible candidate promotion archived prior version |
| Duplicate protection | Exact repeat rejected as `DUPLICATE_BATCH`; duplicate customer IDs rejected |
| Existing API checks | API tests included in full suite; local Docker health, metadata, prediction, and Swagger checks returned HTTP 200 |
| Data drift smoke test | Baseline compared with itself: `NO_DRIFT` |
| Concept drift smoke test | Same baseline used on both sides: `NO_DRIFT`, F1 0.8125 on both |
| Prediction drift smoke test | Same historical window used on both sides: `NO_DRIFT` |
| Model performance smoke test | Baseline F1 0.8125 vs configured minimum 0.50; not an independent holdout |
| API telemetry | 61 errors / 590 events (10.34%) triggered the configured 5% error-rate alert; p95 latency 86.03 ms |
| Docker | Existing `telco-churn-api:lab11` image and `telco-churn-demo` container verified locally and retained |
| Kubernetes | Cluster reachable; existing Telco workload is **0/1 available** with repeated health-probe timeouts; no changes made |
| Actual MLflow registry | Still v1 Production and v2 Archived; no candidate added to the actual registry |
| GitHub Actions | Workflow YAML parsed; actual self-hosted runner registration and workflow execution not verified |

Warnings in the full test suite were from the existing Starlette/AnyIO
deprecation and MLflow stage-transition deprecations. The project already uses
MLflow model stages; the current behavior is retained.

The prediction/preprocessing compatibility check covered the complete 1,409
record test split. Predictions matched for every record, and preprocessor
outputs had a maximum numeric difference of 0.0. The registered and deployed
preprocessor serialized SHA-256 hashes differ; the result establishes
transformation equivalence on this test split, not byte-for-byte identity.

The API telemetry window was historical (2026-10-26 through 2026-10-27), not
a live health failure. Synthetic data was used only for isolated
drift/retraining validation because no compatible evolved Telco batch was
available. These results do not establish drift behavior on real future
Telco data.

Kubernetes was reachable, but the existing workload was not healthy: it was
0/1 available with repeated health-probe timeouts. No successful Kubernetes
deployment is claimed, and no Kubernetes resources were changed. The GitHub
Actions YAML was parsed locally; the self-hosted runner and workflow execution
were not verified.

## Files created or updated for the original Lab 12–13 work

**Created**

- `.github/workflows/lab12_lab13.yml`
- `12_api_performance_monitoring.py`
- `12_concept_drift.py`
- `12_data_drift.py`
- `12_performance_monitoring.py`
- `12_prediction_drift.py`
- `12_retrain_model.py`
- `13_end_to_end.py`
- `src/lab12_monitoring.py`
- `src/lab13_pipeline.py`
- `tests/test_lab12_monitoring.py`
- `tests/test_lab13_pipeline.py`
- `LAB12_COMPLETE_REPORT.md`
- `LAB13_COMPLETE_REPORT.md`
- `FINAL_VALIDATION_SUMMARY.md`

**Updated**

- `config.yaml` — Lab 12 thresholds and conservative Lab 13 defaults.
- `.gitignore` — excludes `data/incoming/`; pre-existing edits in this file
  were preserved.

Pre-existing untracked Lab 11 files shown by Git status were present before
that work and were not recreated or changed. No files in the Big Mart project
were modified.
