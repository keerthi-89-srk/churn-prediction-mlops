# Lab 13 — End-to-End MLOps Automation

## Implemented workflow

`src/lab13_pipeline.py` and `13_end_to_end.py` implement the guarded workflow:

1. Read and validate a labeled Telco CSV against the baseline schema and
   categories.
2. Detect a repeated file by SHA-256 and repeated customers by `customerID`.
   Valid batches are stored once under `data/incoming/ingested/`; the
   manifest and incoming data directory are ignored by Git.
3. Trigger retraining only when new data and configured drift conditions are
   met (unless `lab13.retrain_only_when_drifted` is changed).
4. Fit a new preprocessing pipeline and Random Forest candidate on the
   baseline plus ingested batches. The Production model uses its existing,
   matching preprocessor for comparison.
5. Evaluate both models on the same held-out split; log candidate parameters,
   metrics, model, preprocessor artifact, and final report to MLflow.
6. Require the candidate to pass the configured quality gate and strictly
   beat Production by the configured improvement margin.
7. Leave rejected candidates unregistered. Registration and promotion are
   opt-in; `--promote` implies registration.
8. For an explicitly requested deployment, preflight Docker and the current
   Kubernetes workload, build a unique candidate image before promotion,
   then update the Telco Deployment only after promotion. Verify rollout,
   health, model version, and a real prediction; rollback the Kubernetes
   revision and registry stage if deployment verification fails.

`config.yaml` defaults `auto_promote` and `auto_deploy` to `false`. Neither
existing Production model files nor Lab 11 deployment files are modified by
the default workflow.

## Validation results

- The final full project suite completed with **62 passed, 4 warnings, in
  89.14 seconds**.
- In an isolated temporary project and MLflow SQLite registry, a synthetic
  400-row Telco batch with changed `Contract`, `tenure`, and `TotalCharges`
  was detected as drifted, ingested once, retrained, compared with Production,
  and logged with metrics/artifacts. The test did not promote the candidate.
- An isolated registry test verified that an ineligible candidate was not
  registered and the existing Production version stayed active; an eligible
  candidate was promoted and the former Production version archived. These
  transitions used only a temporary registry.
- Duplicate batch ingestion was tested: the first batch was accepted and the
  exact repeat returned `DUPLICATE_BATCH`; duplicate customer IDs and the
  incompatible bank dataset were rejected.
- Read-only inspection after tests confirmed the actual registry still has
  `TelcoChurnClassifier` v1 in Production and v2 Archived. The local
  deployment bundle SHA-256 checksums still match its original metadata.
- On all 1,409 test records, model predictions from the registered and
  deployed model paths matched. The registered and deployed preprocessors
  produced identical transformations on all 1,409 records, with a maximum
  difference of 0.0. Their serialized SHA-256 hashes differ, so this confirms
  transformation compatibility on this test set, not byte-for-byte identity.
- Existing Docker image `telco-churn-api:lab11` and container
  `telco-churn-demo` were preserved and verified locally. The container was
  running with host port 8001 mapped to container port 8000. Health, metadata,
  prediction, and Swagger endpoints returned HTTP 200; the observed
  prediction was class 0 with churn probability 0.342777.
- Kubernetes was reachable, but the existing Telco Deployment was **0/1
  available**. The pod had repeated health-probe timeouts and restarts. No
  Kubernetes resource or image was changed; the new deployment preflight
  refuses to deploy while the current workload is unhealthy. No successful
  Kubernetes deployment is claimed.
- The new GitHub Actions workflow is manual-dispatch and uses `runs-on:
  self-hosted`; registration, promotion, and deployment inputs default to
  false. Its YAML parsed locally, but the workflow was not run and actual
  GitHub runner registration/settings could not be verified from this
  workspace. No GitHub settings or secrets were changed.

## Local artifact and reproducibility requirements

The deployment bundle under `deployment/`, the MLflow SQLite database
`mlflow.db`, and local MLflow artifacts under `mlruns/` are excluded from Git.
The local Production model bundle is required for deployment-dependent
workflows. Restore it from an accessible MLflow tracking/artifact store, or
prepare it locally with `11_prepare_deployment.py`, before running API
deployment or guarded deployment preflight. The workflow is not fully
reproducible from a fresh Git checkout without restoring the required
artifacts and supplying compatible local data.

No actual evolved Telco batch was available; synthetic data and isolated
registries were used for positive-drift and candidate lifecycle tests. The
GitHub Actions workflow's execution and self-hosted runner configuration also
remain unverified.

## Commands

Run from the project directory with the Python 3.10 environment and required
local MLflow artifacts/data available:

```powershell
python .\13_end_to_end.py --incoming .\path\to\new_labeled_telco.csv
```

This validates, drift-checks, ingests, retrains, compares, and writes a report
without registering or promoting the candidate.

Register an eligible candidate without promotion:

```powershell
python .\13_end_to_end.py --incoming .\path\to\new_labeled_telco.csv --register-candidate
```

Promote only an eligible candidate:

```powershell
python .\13_end_to_end.py --incoming .\path\to\new_labeled_telco.csv --promote
```

Build and deploy only after successful candidate promotion:

```powershell
python .\13_end_to_end.py --incoming .\path\to\new_labeled_telco.csv --promote --deploy
```

The last command currently stops at preflight because the existing Telco
Kubernetes Deployment is not healthy. It must not be used until that existing
workload is investigated and returns to a ready state.

GitHub Actions: dispatch `.github/workflows/lab12_lab13.yml`. Set
`execute_pipeline` only when the runner can access a compatible incoming file
and the intended MLflow registry. Leave `promote_candidate` and
`deploy_candidate` disabled unless those changes are explicitly intended.
