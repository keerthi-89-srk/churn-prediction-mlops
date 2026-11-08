# Lab 11 — Guarded Deployment Automation

## Local preparation and prerequisites

The API image expects a matched Production model and preprocessor in the local
`deployment/` directory. This directory is excluded from Git. Before
deployment-dependent workflows, restore the Production model and its
preprocessing artifact from an accessible MLflow tracking/artifact store.

With the project's Python 3.10 environment active, run the preparation script
from the project root only if the local `deployment/` directory does not
exist. If the directory already exists, inspect and verify `model.pkl`,
`preprocessor.pkl`, and `model_metadata.json`, including Production
provenance and artifact checksums; do not rerun the script:

```powershell
if (Test-Path .\deployment -PathType Container) {
    Get-ChildItem .\deployment
    Get-Content .\deployment\model_metadata.json
    Get-FileHash .\deployment\model.pkl, .\deployment\preprocessor.pkl -Algorithm SHA256
} else {
    python .\11_prepare_deployment.py
}
```

The preparation script reads the tracking URI and model name from
`config.yaml`, requires exactly one `Production` model version, downloads the
preprocessor from that model version's MLflow run, and checks both artifacts
on sample input before creating `deployment/`. It fails rather than
overwriting an existing deployment directory. Compare artifact checksums with
the values in `model_metadata.json` and confirm that the model and
preprocessor belong to the intended Production version before building or
deploying.

The expected local bundle includes `model.pkl`, `preprocessor.pkl`, and
`model_metadata.json`. The deployment script checks model provenance and
SHA-256 checksums before proceeding.

## Current validation status

- The existing `telco-churn-api:lab11` Docker image and
  `telco-churn-demo` container were verified locally and retained. Local
  health, metadata, prediction, and Swagger endpoint checks returned HTTP
  200.
- Kubernetes was reachable during validation, but the existing Telco
  Deployment was **0/1 available because of repeated health-probe failures**.
  No Kubernetes resources were changed. This does not constitute a successful
  Kubernetes deployment.
- The deployment bundle is local and excluded from Git; it must be restored
  or prepared from the accessible Production MLflow artifacts before a
  deployment workflow can run.

## Run the API locally

With the local Production bundle available and API requirements installed:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. Stop the local server
with Ctrl+C.

## Docker and Kubernetes deployment

The image, Kubernetes Deployment, and ClusterIP Service consistently use:

- Image: `telco-churn-api:lab11`
- Kubernetes Deployment: `telco-churn-api`
- Container: `telco-churn-api`

Before any image build or cluster change:

1. Restore/prepare the local Production model bundle from an accessible
   MLflow tracking/artifact store.
2. Start Docker Desktop and verify Docker Engine is available.
3. Configure and independently verify the intended Kubernetes context with
   `kubectl config current-context`.
4. Investigate and resolve the existing workload's health-probe failures.
   The currently observed workload is 0/1 available; do not treat it as a
   successful deployment or apply the manifest to it without resolving the
   issue and explicitly confirming the target context.

The non-mutating preflight does not build an image or change cluster
resources:

```powershell
.\deploy_lab11.ps1 -CheckOnly
```

After preflight and approval for the selected cluster, the separate explicit
invocation builds and applies resources:

```powershell
.\deploy_lab11.ps1 -Apply -ExpectedContext "<exact-context-name>"
```

The script verifies the local bundle, Docker Engine, and Kubernetes
connectivity before it builds and applies `k8s/deployment.yaml`; it then waits
for rollout. No successful Kubernetes deployment is claimed by this report.

The deployment uses a single replica, local image tag, in-cluster ClusterIP
service, resource limits, non-root container, read-only root filesystem, and
temporary inference logs. It is a learning/lab configuration, not a
production availability or persistent-monitoring design.
