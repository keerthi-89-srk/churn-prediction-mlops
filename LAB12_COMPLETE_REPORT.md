# Lab 12 — Advanced Monitoring and Continuous Training

## Implementation

Lab 12 monitoring utilities are in `src/lab12_monitoring.py`. The command-line
entry points are:

- `12_data_drift.py` — per-feature PSI data drift against the configured Telco
  reference dataset.
- `12_prediction_drift.py` — PSI comparisons of logged predictions and churn
  probabilities between two JSONL windows.
- `12_concept_drift.py` — compares Production-model metrics on two labeled
  Telco windows.
- `12_performance_monitoring.py` — evaluates the Production model against
  labeled Telco data and the configured quality-gate floor.
- `12_api_performance_monitoring.py` — request count, errors, latency
  percentiles, UTC-day trend, and threshold alerts from the API JSONL log.
- `12_retrain_model.py` — invokes the guarded Lab 13 retraining path when a
  valid, new, drifted Telco batch is supplied.

Thresholds are in `config.yaml` under `lab12`:

| Monitor | Configured threshold |
|---|---:|
| Data PSI | 0.20 |
| Prediction PSI | 0.20 |
| Maximum F1 drop for concept drift | 0.10 |
| Maximum API p95 latency | 1,000 ms |
| Maximum API error rate | 5% |

Drift/performance reports are emitted as JSON under `reports/lab12/` when the
commands are run. The existing API request logger was reused; it logs status,
latency, prediction metadata, and field count without storing customer
values.

## Validation results

- The final full project suite completed with **62 passed, 4 warnings, in
  89.14 seconds**. The warnings were from the existing Starlette/AnyIO and
  MLflow registry stage deprecations.
- A synthetic evolved Telco window created in memory was detected as drifted.
  Trigger tests covered no drift/no data, no drift/new data, drift/no data,
  and drift/new data. These tests validate trigger behavior but are not
  evidence of drift in a real evolved Telco population.
- Data drift CLI smoke test using `churn.csv` against itself: **NO_DRIFT**,
  0/19 drifted features (7,044 rows on each side).
- Prediction drift smoke test comparing the same historical log window:
  **NO_DRIFT** (529 successful events). This is a path/format smoke test, not
  evidence of stability across independent time windows.
- Concept drift smoke test comparing the same baseline on both sides:
  **NO_DRIFT**, F1 0.8125 on both. This is not an independent holdout.
- Production performance smoke test on the existing baseline CSV: **HEALTHY**
  against the configured F1 floor of 0.50; observed F1 0.8125. Since this is
  the baseline dataset and is not an independent new labeled window, do not
  treat it as a new-production performance estimate.
- Historical API telemetry contained 590 events, of which 61 were non-success
  (10.34%, above the configured 5% limit). p95 latency was 86.03 ms. The
  monitor correctly returned **ALERT** for error rate. The event window was
  2026-10-26 through 2026-10-27; this is historical telemetry, not a live
  health failure.

## Data limitation

There is no compatible evolved Telco dataset in the project.
`data/raw/churn2.csv` is a different bank-customer dataset (10,000 rows, 14
columns), so the new validator rejects it. Drift-positive tests used isolated
synthetic Telco rows derived from the existing reference; neither source CSV
was changed. Monitoring results using synthetic rows do not substitute for
testing on representative new customer data.

## Commands

Run from the project directory with the existing virtual environment active:

```powershell
python .\12_data_drift.py --current .\path\to\new_telco.csv
python .\12_prediction_drift.py --reference-log .\logs\reference.jsonl --current-log .\logs\current.jsonl
python .\12_concept_drift.py --reference .\path\to\labeled_reference.csv --current .\path\to\labeled_current.csv
python .\12_performance_monitoring.py --data .\path\to\labeled_current.csv
python .\12_api_performance_monitoring.py
python .\12_retrain_model.py --incoming .\path\to\new_labeled_telco.csv
```

Candidate registration is opt-in:

```powershell
python .\12_retrain_model.py --incoming .\path\to\new_labeled_telco.csv --register-candidate
```

The incoming labeled batch must match the Telco schema, have unique new
`customerID` values, and contain valid `Yes`/`No` churn labels. A batch that
does not drift is ingested once but does not trigger retraining by default.
