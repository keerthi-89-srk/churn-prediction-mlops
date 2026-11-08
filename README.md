# Telco Customer Churn Prediction MLOps

A learning project that develops a customer churn classifier and the MLOps
practices around it: data checks, preprocessing, experiment tracking, model
evaluation, API serving, deployment preparation, monitoring, and guarded
retraining.

## Project objectives

- Predict whether a Telco customer is likely to churn.
- Build repeatable data, training, evaluation, and inference workflows.
- Track experiments and manage model versions with MLflow.
- Serve predictions through a validated FastAPI application.
- Practice containerization, Kubernetes deployment preparation, monitoring,
  and conservative retraining workflows.
- Keep retraining, model promotion, and deployment guarded rather than
  promoting or deploying automatically by default.

## Dataset

The project uses the Telco Customer Churn dataset, with customer demographic,
account, subscribed-service, contract, and billing attributes. `Churn` is the
Yes/No target; `customerID` is an identifier and is not a predictive feature.
The source data includes blank `TotalCharges` values for some zero-tenure
customers; validation and API input rules account for that case.

The repository also contains `churn2.csv`, which is a bank-customer dataset,
not a compatible Telco churn dataset. Lab 12/13 data validation rejects it.
No compatible evolved Telco batch was available for the final monitoring and
retraining validation, so drift-positive tests used synthetic Telco rows.

## Technologies

- Python 3.10
- pandas, NumPy, scikit-learn, and joblib
- MLflow Tracking and Model Registry
- FastAPI, Pydantic, and Uvicorn
- pytest and HTTPX-based API tests
- Docker and Kubernetes (`kubectl`)
- YAML configuration and GitHub Actions workflow definitions

## Architecture and workflow

1. Load and validate the raw Telco data and separate the customer identifier
   and churn target from model features.
2. Fit and apply a preprocessing pipeline, prepare a reproducible train/test
   split, and train/evaluate classifiers.
3. Log parameters, metrics, and artifacts to MLflow; apply a model quality
   gate and manage registered model versions.
4. Load a matched Production model and preprocessor in the API, validate
   requests, return predictions, and record privacy-conscious request
   telemetry.
5. Prepare a local deployment bundle, containerize the API, and define
   Kubernetes resources.
6. Monitor data, prediction, concept, and API performance signals. Lab 13
   validates new labeled batches and only retrains when its configured
   conditions are met. Candidate registration, promotion, and deployment are
   opt-in.

The configured MLflow tracking URI is `sqlite:///mlflow.db`, the experiment
is `Telco_Churn_Prediction`, and the registered model is
`TelcoChurnClassifier`.

## Labs 1–13

| Lab | Summary |
|---|---|
| 1 | Load the Telco churn source data and establish the project data inputs. |
| 2 | Validate source data, expected columns, values, and output artifacts. |
| 3 | Preprocess customer features and prepare model-ready datasets. |
| 4 | Track model experiments, parameters, metrics, and artifacts with MLflow. |
| 5 | Run the baseline churn training pipeline. |
| 6 | Evaluate models on a held-out test split and report classification metrics. |
| 7 | Apply a configured quality gate to model evaluation results. |
| 8 | Automate lifecycle operations around training and registered models. |
| 9 | Exercise inference and failure-injection/reliability workflows. |
| 10 | Serve predictions with FastAPI and validate API behavior and performance. |
| 11 | Prepare a matched Production model bundle and Docker/Kubernetes deployment. |
| 12 | Monitor data, prediction, concept, model-performance, and API telemetry; provide guarded retraining entry points. |
| 13 | Validate and deduplicate incoming labeled batches, conditionally retrain, compare candidates, and guard promotion/deployment. |

Labs 7–9 and their existing history are unchanged by this documentation
update.

## Installation

Use Python 3.10. From the project directory, create and activate a virtual
environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The repository is not fully reproducible on a fresh computer from Git alone.
`mlflow.db`, `mlruns/`, generated datasets, and `deployment/` are excluded from
Git. In particular, deployment-dependent workflows require the local
Production model bundle (`deployment/model.pkl`,
`deployment/preprocessor.pkl`, and `deployment/model_metadata.json`) to be
restored or prepared from an accessible MLflow tracking/artifact store first.
The original raw dataset and generated artifacts may also need to be supplied
locally.

### Dependency files

- `requirements.txt` contains the broader pinned environment used by the
  project training, notebook, MLflow, and analysis workflows.
- `requirements-api.txt` contains the lean runtime dependencies for the
  FastAPI service.
- `requirements-api-test.txt` includes the API runtime requirements and the
  API test tools (pytest and HTTP client support).

Install the requirements appropriate to the workflow. For the API and its
tests:

```powershell
python -m pip install -r requirements-api-test.txt
```

For the full project environment, use `requirements.txt` in an environment
where its complete pinned dependencies are supported.

## Run the API and tests

The API expects the local deployment bundle described above. Start the server
from the project root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

Run the API-specific tests:

```powershell
python -m pytest tests/test_api.py tests/test_performance.py -q
```

Run the full project suite:

```powershell
python -m pytest -q
```

The latest recorded full-suite result is **62 passed, 4 warnings in 89.14
seconds**. The warnings include existing Starlette/AnyIO and MLflow stage
deprecations.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Basic service status. |
| `GET` | `/health` | Readiness check; returns HTTP 503 unless both model and preprocessor are loaded. |
| `GET` | `/metadata` | Model name, version, run ID, and readiness information. |
| `POST` | `/predict` | Validate and predict for one customer. |
| `POST` | `/predict/batch` | Validate and predict for 1–20 customers. |

Example request for one customer:

```powershell
$body = @{
  customerID = "example-001"
  gender = "Female"
  SeniorCitizen = 0
  Partner = "Yes"
  Dependents = "No"
  tenure = 12
  PhoneService = "Yes"
  MultipleLines = "No"
  InternetService = "DSL"
  OnlineSecurity = "No"
  OnlineBackup = "Yes"
  DeviceProtection = "No"
  TechSupport = "No"
  StreamingTV = "No"
  StreamingMovies = "No"
  Contract = "Month-to-month"
  PaperlessBilling = "Yes"
  PaymentMethod = "Electronic check"
  MonthlyCharges = 65.5
  TotalCharges = 786.0
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body $body
```

The response contains a binary `prediction` and a `churn_probability`.
Requests with missing/extra fields, invalid categories, invalid numeric
values, or inconsistent blank `TotalCharges` are rejected. Blank
`TotalCharges` is accepted only when `tenure` is zero. Inference telemetry
records request status and prediction metadata, not customer values or IDs.

## MLflow tracking and model registry

Configuration in `config.yaml` points MLflow to the local SQLite store
`mlflow.db` and names the experiment `Telco_Churn_Prediction`. Training runs
record parameters, metrics, and artifacts. The registered model is
`TelcoChurnClassifier`; the deployment preparation workflow requires exactly
one version in the `Production` stage and retrieves its matching
preprocessor from the same MLflow run.

The latest validation inspected the actual local registry as v1 in
`Production` and v2 `Archived`. This local registry and its artifacts are not
included in Git. The evaluation report records held-out metrics of accuracy
0.7807, precision 0.5758, recall 0.6604, F1 0.6152, and ROC AUC 0.8403 on
1,409 test records.

## Lab 12 monitoring

Lab 12 provides configured thresholds for data PSI (0.20), prediction PSI
(0.20), maximum concept-drift F1 drop (0.10), API p95 latency (1,000 ms), and
API error rate (5%). It reports data/prediction drift, compares labeled
windows for concept drift, checks Production-model performance against a
quality floor, and summarizes API request counts, errors, latency percentiles,
and UTC-day trends.

The final telemetry report describes historical data, not live monitoring:
590 API events from 2026-10-26 through 2026-10-27 included 61 non-success
events (10.34%), exceeding the configured error-rate threshold; p95 latency
was 86.03 ms. Synthetic drift tests demonstrate code paths, not measured drift
in a real evolved Telco population.

## Lab 13 guarded retraining

Lab 13 checks the Telco schema and labels, rejects duplicate files and
customer IDs, and retrains only when there is new data and configured drift
conditions are met. Candidate training, evaluation, and MLflow logging are
separate from candidate registration and promotion. `auto_promote` and
`auto_deploy` default to `false`; promotion requires the configured quality
gate and improvement criteria. Deployment has additional Docker/Kubernetes
preflight and post-deployment checks.

Validation used temporary synthetic Telco rows and isolated registries; no
candidate was added to the actual registry. No compatible evolved Telco
dataset was available. The current local Kubernetes workload is unhealthy
(0/1 available), so the deployment path must not be treated as a successful
deployment.

## Docker and Kubernetes

Docker builds include the local `deployment/` bundle. Restore or prepare that
bundle from the accessible MLflow Production tracking/artifact store before
building. Run `11_prepare_deployment.py` only when the local `deployment/`
directory does not exist. If it already exists, inspect and verify
`model.pkl`, `preprocessor.pkl`, and `model_metadata.json` (including model
provenance and artifact checksums) instead of rerunning the script, which
refuses to overwrite an existing directory:

```powershell
if (Test-Path .\deployment -PathType Container) {
    Get-ChildItem .\deployment
    Get-Content .\deployment\model_metadata.json
    Get-FileHash .\deployment\model.pkl, .\deployment\preprocessor.pkl -Algorithm SHA256
} else {
    python .\11_prepare_deployment.py
}
```

Proceed with the build only after confirming the bundle artifacts and metadata
match the intended Production model:

```powershell
docker build -t telco-churn-api:lab11 .
docker run --rm -p 8000:8000 telco-churn-api:lab11
```

The existing `telco-churn-api:lab11` image and `telco-churn-demo` container
were verified locally. Kubernetes was reachable during validation, but the
existing Telco Deployment was **0/1 available because of health-probe
failures**. No successful Kubernetes deployment is claimed.

After confirming Docker Engine is running, configuring and independently
verifying the intended Kubernetes context, restoring the local bundle, and
investigating the unhealthy workload, run the non-mutating preflight:

```powershell
.\deploy_lab11.ps1 -CheckOnly
```

Applying resources is a separate, explicit action:

```powershell
.\deploy_lab11.ps1 -Apply -ExpectedContext "<exact-context-name>"
```

Do not apply to the currently unhealthy cluster without first resolving the
probe failures and explicitly confirming the target context. The manifest is
a single-replica learning configuration with a ClusterIP service, resource
limits, non-root execution, a read-only root filesystem, and ephemeral
inference logs; it is not a production availability design.

## Repository structure

```text
api/                  FastAPI app, schemas, model loading, request logging
data/                 Raw and processed datasets (local/generated; excluded)
deployment/            Local model/preprocessor bundle and metadata (excluded)
k8s/                   Kubernetes Deployment and Service manifests
models/                Local model and preprocessing artifacts (excluded)
notebooks/             Project implementation notebook
pipelines/              Earlier baseline and MLflow pipeline entry points
reports/                Evaluation and monitoring reports
src/mlops_pipeline/     Data, preprocessing, training, lifecycle, inference
src/                   Shared preprocessing, evaluation, Lab 12/13 logic
tests/                  API, performance, monitoring, and pipeline tests
*.py                    Lab 11–13 command-line entry points
config.yaml             Paths, model, MLflow, and monitoring configuration
requirements*.txt       Project, API, and API-test dependencies
Dockerfile              API image definition
deploy_lab11.ps1         Guarded Docker/Kubernetes deployment script
```

## Validation and known limitations

- Latest full test suite: **62 passed, 4 warnings in 89.14 seconds**.
- Model predictions matched on all 1,409 test records.
- Registered and deployed preprocessors produced identical transformations
  on all 1,409 records, with a maximum difference of 0.0. Their serialized
  SHA-256 hashes differ, so this is transformation equivalence, not
  byte-for-byte identity.
- Monitoring/retraining drift-positive validation used synthetic Telco rows;
  performance and no-drift smoke tests used the existing baseline or the same
  historical window rather than independent live production windows.
- `mlflow.db`, `mlruns/`, generated datasets, and `deployment/` are excluded
  from Git. Restore artifacts from an accessible MLflow store before
  deployment-dependent workflows. The project is not fully reproducible from
  a fresh Git clone alone.
- Kubernetes was reachable but its existing Telco workload was 0/1 available
  due to health-probe failures.
- The GitHub Actions workflow YAML parsed locally, but self-hosted runner
  registration/settings and actual workflow execution were not verified.
- Future improvements include adding a representative, consented evolved
  Telco dataset; testing monitoring against independent time windows;
  resolving Kubernetes probes and adding persistent telemetry; and verifying
  CI execution on a configured runner.
