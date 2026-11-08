"""Guarded incoming-data, retraining, and promotion workflow for Lab 13."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.lab12_monitoring import (
    DatasetValidationError,
    detect_concept_drift,
    detect_data_drift,
    read_telco_csv,
    should_trigger_retraining,
    validate_telco_frame,
    write_json_report,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_tracking_uri(uri: str, project_root: Path) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    database_path = Path(uri[len(prefix) :])
    if not database_path.is_absolute():
        database_path = (project_root / database_path).resolve()
    return prefix + database_path.as_posix()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "batches": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read ingestion manifest {path}: {error}") from error
    if manifest.get("version") != 1 or not isinstance(manifest.get("batches"), list):
        raise ValueError(f"Unsupported or malformed ingestion manifest: {path}")
    return manifest


def ingest_batch(
    source: str | Path,
    reference: pd.DataFrame,
    *,
    ingest_directory: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate and atomically record a batch once, without editing its source."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Incoming dataset not found: {source_path}")
    digest = sha256_file(source_path)
    data = validate_telco_frame(
        read_telco_csv(source_path),
        list(reference.columns),
        require_target=True,
        reference_frame=reference,
    )
    existing_ids = set(reference["customerID"].astype(str))

    destination_dir = Path(ingest_directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = manifest_file.with_suffix(manifest_file.suffix + ".lock")
    try:
        with lock_file.open("x", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
    except FileExistsError as error:
        raise RuntimeError(
            f"Another ingestion is updating the manifest: {lock_file}"
        ) from error

    try:
        manifest = _read_manifest(manifest_file)
        known_hashes = {item.get("sha256") for item in manifest["batches"]}
        for batch_hash in known_hashes:
            if not isinstance(batch_hash, str) or len(batch_hash) != 64:
                raise ValueError(f"Invalid batch checksum in manifest: {batch_hash!r}")
            stored_batch = destination_dir / f"{batch_hash}.csv"
            if not stored_batch.is_file():
                raise FileNotFoundError(
                    f"Ingested batch listed in manifest is missing: {stored_batch}"
                )
            if sha256_file(stored_batch) != batch_hash:
                raise ValueError(
                    f"Ingested batch checksum does not match manifest: {stored_batch}"
                )
        target = destination_dir / f"{digest}.csv"
        if target.is_file() and sha256_file(target) != digest:
            raise ValueError(f"Existing ingested file checksum mismatch: {target}")
        if digest in known_hashes or target.exists():
            return {
                "status": "DUPLICATE_BATCH",
                "sha256": digest,
                "rows": int(len(data)),
                "stored_path": str(target),
            }

        for stored_batch in destination_dir.glob("*.csv"):
            previous = read_telco_csv(stored_batch)
            overlap = existing_ids.intersection(previous["customerID"].astype(str))
            if overlap:
                raise DatasetValidationError(
                    "Existing ingested dataset overlaps the reference customer IDs"
                )
            previous_ids = set(previous["customerID"].astype(str))
            if previous_ids.intersection(data["customerID"].astype(str)):
                raise DatasetValidationError(
                    "Incoming batch contains customer IDs already ingested"
                )
            existing_ids.update(previous_ids)
        if existing_ids.intersection(data["customerID"].astype(str)):
            raise DatasetValidationError(
                "Incoming batch contains customer IDs already present in prior data"
            )

        temporary_batch = destination_dir / f".{digest}.tmp"
        temporary_manifest = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
        try:
            shutil.copyfile(source_path, temporary_batch)
            if sha256_file(temporary_batch) != digest:
                raise IOError("Incoming batch checksum changed during ingestion")
            record = {
                "sha256": digest,
                "source_name": source_path.name,
                "stored_path": str(target),
                "rows": int(len(data)),
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            manifest["batches"].append(record)
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_batch, target)
            os.replace(temporary_manifest, manifest_file)
        finally:
            temporary_batch.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
        return {"status": "INGESTED", **record}
    finally:
        lock_file.unlink(missing_ok=True)


def candidate_is_eligible(
    *,
    candidate_metric: float,
    production_metric: float,
    quality_gate_minimum: float,
    minimum_improvement: float,
) -> tuple[bool, list[str]]:
    reasons = []
    if candidate_metric < quality_gate_minimum:
        reasons.append("quality_gate_failed")
    if candidate_metric <= production_metric + minimum_improvement:
        reasons.append("production_model_not_beaten")
    return not reasons, reasons


def register_candidate_version(
    client: MlflowClient,
    *,
    model_name: str,
    model_uri: str,
    run_id: str,
    eligible: bool,
) -> Any | None:
    if not eligible:
        return None
    version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description="Lab 13 candidate; passed quality and Production comparison.",
    )
    if str(version.status).upper() != "READY":
        raise RuntimeError(
            f"MLflow candidate version {version.version} is not READY: {version.status}"
        )
    return version


def promote_candidate_version(
    client: MlflowClient,
    *,
    model_name: str,
    candidate_version: str,
    eligible: bool,
) -> None:
    if not eligible:
        raise ValueError("A candidate that failed validation cannot be promoted")
    current_production = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.current_stage == "Production"
    ]
    if len(current_production) != 1:
        raise RuntimeError(
            f"Expected exactly one current Production version for {model_name}; "
            f"found {len(current_production)}"
        )
    if str(current_production[0].version) == str(candidate_version):
        raise ValueError("Candidate is already the Production model")
    client.transition_model_version_stage(
        name=model_name,
        version=candidate_version,
        stage="Production",
        archive_existing_versions=True,
    )


def _production_model(
    config: dict[str, Any],
    client: MlflowClient,
    project_root: Path,
) -> tuple[Any, Any, Any]:
    name = config["registry"]["model_name"]
    versions = [
        version
        for version in client.search_model_versions(f"name='{name}'")
        if version.current_stage == "Production"
    ]
    if len(versions) != 1:
        raise RuntimeError(
            f"Expected exactly one Production model version for {name}; "
            f"found {len(versions)}"
        )
    production_version = versions[0]
    bundle_metadata_path = project_root / "deployment" / "model_metadata.json"
    bundle_metadata = json.loads(bundle_metadata_path.read_text(encoding="utf-8"))
    if str(bundle_metadata.get("model_version")) != str(production_version.version):
        raise RuntimeError(
            "The local deployment bundle does not match the Production model version"
        )
    if bundle_metadata.get("run_id") != production_version.run_id:
        raise RuntimeError(
            "The local deployment bundle does not match the Production MLflow run"
        )
    for artifact_name in ("model.pkl", "preprocessor.pkl"):
        artifact_path = project_root / "deployment" / artifact_name
        expected_hash = bundle_metadata.get("sha256", {}).get(artifact_name)
        if not expected_hash or sha256_file(artifact_path) != expected_hash:
            raise RuntimeError(
                f"Production deployment artifact checksum failed: {artifact_name}"
            )
    model = mlflow.sklearn.load_model(
        f"models:/{name}/{production_version.version}"
    )
    preprocessor = joblib.load(project_root / "deployment" / "preprocessor.pkl")
    return production_version, model, preprocessor


def _evaluate(
    model: Any,
    preprocessor: Any,
    raw_features: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, float]:
    from src.lab12_monitoring import classification_metrics

    transformed = preprocessor.transform(raw_features)
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)[:, 1]
    result = classification_metrics(labels, np.asarray(predictions, dtype=int))
    result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return result


def _fit_candidate_preprocessor(features: pd.DataFrame) -> Any:
    categorical = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def _build_candidate_image(
    *,
    model: Any,
    preprocessor: Any,
    config: dict[str, Any],
    version: Any,
    metrics: dict[str, float],
) -> str:
    image = f"telco-churn-api:lab13-v{version.version}"
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        raise RuntimeError(f"Refusing to overwrite existing image {image}")

    model_id = getattr(version, "model_id", None)
    if not model_id and str(version.source).startswith("models:/m-"):
        model_id = str(version.source).rsplit("/", 1)[-1]
    run_id = version.run_id
    if not model_id:
        raise RuntimeError("Registered candidate version has no MLflow model ID")
    with tempfile.TemporaryDirectory(prefix="telco-lab13-image-") as context_name:
        context = Path(context_name)
        shutil.copytree(ROOT / "api", context / "api")
        shutil.copy2(ROOT / "config.yaml", context / "config.yaml")
        shutil.copy2(ROOT / "requirements-api.txt", context / "requirements-api.txt")
        shutil.copy2(ROOT / "Dockerfile", context / "Dockerfile")
        deployment_dir = context / "deployment"
        deployment_dir.mkdir()
        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        metadata = {
            "model_name": config["registry"]["model_name"],
            "model_version": int(version.version),
            "model_id": model_id,
            "run_id": run_id,
            "preprocessor_run_id": run_id,
            "model_uri": f"models:/{config['registry']['model_name']}/{version.version}",
            "model_source": version.source,
            "model_stage": "Candidate",
            "metrics": metrics,
            "preprocessor_artifact": "preprocessor.pkl",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {
                "model.pkl": sha256_file(model_path),
                "preprocessor.pkl": sha256_file(preprocessor_path),
            },
        }
        (deployment_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        build = subprocess.run(
            ["docker", "build", "--tag", image, "--file", "Dockerfile", "."],
            cwd=context,
            check=False,
        )
        if build.returncode:
            raise RuntimeError(f"Docker image build failed for {image}")
    return image


def _deployment_preflight() -> None:
    docker = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if docker.returncode or not docker.stdout.strip():
        raise RuntimeError("Docker Engine is unavailable for candidate deployment")
    context = subprocess.run(
        ["kubectl", "config", "current-context"],
        capture_output=True,
        text=True,
        check=False,
    )
    if context.returncode or not context.stdout.strip():
        raise RuntimeError("No Kubernetes context is configured for deployment")
    current = subprocess.run(
        ["kubectl", "get", "deployment", "telco-churn-api", "-o=json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if current.returncode:
        raise RuntimeError("Could not inspect the existing Telco Kubernetes deployment")
    deployment = json.loads(current.stdout)
    desired = int(deployment.get("spec", {}).get("replicas", 1))
    available = int(deployment.get("status", {}).get("availableReplicas", 0))
    if available < desired:
        raise RuntimeError(
            "Refusing deployment because the current Telco Deployment is unhealthy "
            f"({available}/{desired} replicas available)"
        )


def _update_deployment_image(image: str, model_version: str) -> None:
    current_deployment = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            "telco-churn-api",
            "-o=json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if current_deployment.returncode:
        raise RuntimeError("Could not inspect the existing Telco Kubernetes deployment")
    deployment = json.loads(current_deployment.stdout)
    desired_replicas = int(deployment.get("spec", {}).get("replicas", 1))
    available_replicas = int(deployment.get("status", {}).get("availableReplicas", 0))
    if available_replicas < desired_replicas:
        raise RuntimeError(
            "Refusing deployment update because the current Telco Deployment is "
            f"unhealthy ({available_replicas}/{desired_replicas} replicas available)"
        )
    update = subprocess.run(
        [
            "kubectl",
            "set",
            "image",
            "deployment/telco-churn-api",
            f"telco-churn-api={image}",
        ],
        check=False,
    )
    if update.returncode:
        raise RuntimeError("Kubernetes image update failed after candidate promotion")
    rollout = subprocess.run(
        ["kubectl", "rollout", "status", "deployment/telco-churn-api", "--timeout=120s"],
        check=False,
    )
    probe_code = (
        "import json, urllib.request; "
        "base='http://127.0.0.1:8000'; "
        "health=json.load(urllib.request.urlopen(base+'/health', timeout=10)); "
        "meta=json.load(urllib.request.urlopen(base+'/metadata', timeout=10)); "
        f"assert str(meta['model_version']) == {model_version!r}, meta; "
        "assert health.get('status') == 'healthy' and meta.get('model_loaded') "
        "and meta.get('preprocessor_loaded'), (health, meta); "
        "payload={'customerID':'LAB13-VERIFY','gender':'Male','SeniorCitizen':0,"
        "'Partner':'Yes','Dependents':'Yes','tenure':10,'PhoneService':'Yes',"
        "'MultipleLines':'No','InternetService':'DSL','OnlineSecurity':'Yes',"
        "'OnlineBackup':'Yes','DeviceProtection':'No','TechSupport':'Yes',"
        "'StreamingTV':'No','StreamingMovies':'No','Contract':'Month-to-month',"
        "'PaperlessBilling':'Yes','PaymentMethod':'Electronic check',"
        "'MonthlyCharges':50.0,'TotalCharges':'500.0'}; "
        "request=urllib.request.Request(base+'/predict', "
        "data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}); "
        "prediction=json.load(urllib.request.urlopen(request, timeout=30)); "
        "assert prediction['prediction'] in (0,1) and "
        "0 <= prediction['churn_probability'] <= 1, prediction"
    )
    verification = subprocess.run(
        [
            "kubectl",
            "exec",
            "deployment/telco-churn-api",
            "--",
            "python",
            "-c",
            probe_code,
        ],
        check=False,
    )
    if rollout.returncode or verification.returncode:
        rollback = subprocess.run(
            ["kubectl", "rollout", "undo", "deployment/telco-churn-api"],
            check=False,
        )
        if rollback.returncode:
            raise RuntimeError(
                "Kubernetes rollout failed and automatic rollback also failed"
            )
        restored = subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                "deployment/telco-churn-api",
                "--timeout=120s",
            ],
            check=False,
        )
        if restored.returncode:
            raise RuntimeError(
                "Deployment verification failed and the prior deployment did not "
                "recover after rollback"
            )
        reason = "rollout" if rollout.returncode else "API verification"
        raise RuntimeError(
            f"Kubernetes {reason} failed; the prior deployment was restored"
        )


def run_end_to_end(
    incoming_path: str | Path,
    *,
    config_path: str | Path = ROOT / "config.yaml",
    ingest_directory: str | Path = ROOT / "data" / "incoming" / "ingested",
    manifest_path: str | Path = ROOT / "data" / "incoming" / "manifest.json",
    report_path: str | Path = ROOT / "reports" / "lab13" / "end_to_end.json",
    register_candidate: bool = False,
    promote_candidate: bool = False,
    deploy_candidate: bool = False,
) -> dict[str, Any]:
    if deploy_candidate and not promote_candidate:
        raise ValueError("Deployment requires explicit candidate promotion")
    if promote_candidate:
        register_candidate = True

    config_file = Path(config_path)
    root = config_file.resolve().parent
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    reference_raw = read_telco_csv(root / config["data"]["raw_path"])
    reference = validate_telco_frame(
        reference_raw,
        list(reference_raw.columns),
        require_target=True,
    )
    incoming_file = Path(incoming_path).resolve()
    current_raw = read_telco_csv(incoming_file)
    current = validate_telco_frame(
        current_raw,
        list(reference.columns),
        require_target=True,
        reference_frame=reference,
    )
    incoming_digest = sha256_file(incoming_file)
    ingestion = ingest_batch(
        incoming_file,
        reference,
        ingest_directory=ingest_directory,
        manifest_path=manifest_path,
    )
    if ingestion["status"] == "DUPLICATE_BATCH":
        result = {
            "status": "DUPLICATE_BATCH",
            "incoming_sha256": incoming_digest,
            "ingestion": ingestion,
            "registered_candidate": None,
            "promoted": False,
            "deployment_image": None,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_report(report_path, result)
        return result

    drift = detect_data_drift(
        reference,
        current,
        threshold=float(config["lab12"]["data_drift_psi_threshold"]),
    )
    drift_detected = drift["status"] == "DRIFT_DETECTED"
    trigger = (
        should_trigger_retraining(
            drift_detected=drift_detected,
            new_data_available=ingestion["status"] == "INGESTED",
        )
        if config["lab13"].get("retrain_only_when_drifted", True)
        else True
    )
    result: dict[str, Any] = {
        "status": "TRIGGERED" if trigger else "NO_RETRAIN_TRIGGER",
        "trigger_conditions": {
            "data_drift": drift["status"] == "DRIFT_DETECTED",
            "new_data_available": ingestion["status"] == "INGESTED",
            "retraining_triggered": trigger,
        },
        "incoming_sha256": incoming_digest,
        "ingestion": ingestion,
        "data_drift": drift,
        "registered_candidate": None,
        "promoted": False,
        "deployment_image": None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not trigger:
        write_json_report(report_path, result)
        return result

    training_frames = [reference]
    for batch_path in sorted(Path(ingest_directory).glob("*.csv")):
        training_frames.append(
            validate_telco_frame(
                read_telco_csv(batch_path),
                list(reference.columns),
                require_target=True,
                reference_frame=reference,
            )
        )
    training_data = pd.concat(training_frames, ignore_index=True)
    raw_features = training_data.drop(columns=["customerID", "Churn"])
    labels = training_data["Churn"].map({"No": 0, "Yes": 1}).to_numpy(dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        raw_features,
        labels,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
        stratify=labels,
    )

    tracking_uri = resolve_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"],
        root,
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    production_version, production_model, preprocessor = _production_model(
        config, client, root
    )
    candidate_preprocessor = _fit_candidate_preprocessor(x_train)
    transformed_train = candidate_preprocessor.fit_transform(x_train)

    parameters = dict(config["model"])
    candidate_model = RandomForestClassifier(**parameters)
    candidate_model.fit(transformed_train, y_train)
    candidate_metrics = _evaluate(
        candidate_model, candidate_preprocessor, x_test, y_test
    )
    production_metrics = _evaluate(production_model, preprocessor, x_test, y_test)
    comparison_metric = config["registry"]["decision_metric"]
    gate_metric = config["quality_gate"]["metric"]
    eligible, rejection_reasons = candidate_is_eligible(
        candidate_metric=float(candidate_metrics[comparison_metric]),
        production_metric=float(production_metrics[comparison_metric]),
        quality_gate_minimum=float(config["quality_gate"]["minimum_value"]),
        minimum_improvement=float(config["lab13"]["minimum_f1_improvement"]),
    )
    result.update(
        {
            "production_version": str(production_version.version),
            "production_metrics": production_metrics,
            "candidate_metrics": candidate_metrics,
            "quality_gate": {
                "metric": gate_metric,
                "minimum": float(config["quality_gate"]["minimum_value"]),
                "actual": float(candidate_metrics[gate_metric]),
                "passed": float(candidate_metrics[gate_metric])
                >= float(config["quality_gate"]["minimum_value"]),
            },
            "candidate_eligible": eligible,
            "rejection_reasons": rejection_reasons,
        }
    )

    with mlflow.start_run(run_name="Lab13-guarded-candidate") as run:
        mlflow.log_params(
            {
                **parameters,
                "lab": 13,
                "incoming_sha256": result["incoming_sha256"],
                "incoming_rows": int(len(current)),
                "training_rows": int(len(training_data)),
                "production_version": str(production_version.version),
            }
        )
        mlflow.log_metrics(
            {
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                **{f"production_{key}": value for key, value in production_metrics.items()},
            }
        )
        with tempfile.TemporaryDirectory(prefix="telco-lab13-artifacts-") as tmp:
            preprocessor_file = Path(tmp) / "preprocessor.pkl"
            joblib.dump(candidate_preprocessor, preprocessor_file)
            mlflow.log_artifact(str(preprocessor_file), artifact_path="preprocessing_pipeline")
            model_info = mlflow.sklearn.log_model(
                sk_model=candidate_model,
                name="candidate_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        result["mlflow_run_id"] = run.info.run_id

    if eligible and deploy_candidate:
        try:
            _deployment_preflight()
        except (RuntimeError, json.JSONDecodeError, OSError) as error:
            result["status"] = "DEPLOYMENT_PREFLIGHT_FAILED"
            result["deployment_error"] = str(error)
            client.log_text(
                result["mlflow_run_id"],
                json.dumps(result, indent=2, allow_nan=False),
                "lab13_candidate_report_final.json",
            )
            write_json_report(report_path, result)
            raise RuntimeError(
                f"Candidate was not registered or promoted: {error}"
            ) from error

    registered_version = None
    if eligible and register_candidate:
        registered_version = register_candidate_version(
            client,
            model_name=config["registry"]["model_name"],
            model_uri=model_info.model_uri,
            run_id=result["mlflow_run_id"],
            eligible=eligible,
        )
        result["registered_candidate"] = str(registered_version.version)
    if promote_candidate:
        if not eligible or registered_version is None:
            result["status"] = "CANDIDATE_REJECTED"
        else:
            image = None
            if deploy_candidate:
                image = _build_candidate_image(
                    model=candidate_model,
                    preprocessor=candidate_preprocessor,
                    config=config,
                    version=registered_version,
                    metrics=candidate_metrics,
                )
                result["deployment_image"] = image
            promote_candidate_version(
                client,
                name=config["registry"]["model_name"],
                candidate_version=registered_version.version,
                eligible=eligible,
            )
            result["promoted"] = True
            result["status"] = "PROMOTED"
            if deploy_candidate and image is not None:
                try:
                    _update_deployment_image(image, str(registered_version.version))
                except Exception as error:
                    try:
                        client.transition_model_version_stage(
                            name=config["registry"]["model_name"],
                            version=registered_version.version,
                            stage="Archived",
                            archive_existing_versions=False,
                        )
                        client.transition_model_version_stage(
                            name=config["registry"]["model_name"],
                            version=production_version.version,
                            stage="Production",
                            archive_existing_versions=False,
                        )
                    except Exception as rollback_error:
                        result["status"] = "DEPLOYMENT_FAILED_REGISTRY_ROLLBACK_FAILED"
                        result["deployment_error"] = str(error)
                        result["registry_rollback_error"] = str(rollback_error)
                        client.log_text(
                            result["mlflow_run_id"],
                            json.dumps(result, indent=2, allow_nan=False),
                            "lab13_candidate_report_final.json",
                        )
                        write_json_report(report_path, result)
                        raise RuntimeError(
                            "Deployment failed and Production registry rollback failed; "
                            f"deployment error: {error}; rollback error: {rollback_error}"
                        ) from rollback_error
                    result["promoted"] = False
                    result["status"] = "DEPLOYMENT_FAILED_ROLLED_BACK"
                    result["deployment_error"] = str(error)
                    client.log_text(
                        result["mlflow_run_id"],
                        json.dumps(result, indent=2, allow_nan=False),
                        "lab13_candidate_report_final.json",
                    )
                    write_json_report(report_path, result)
                    raise RuntimeError(
                        f"Deployment failed; the prior Production model was restored: {error}"
                    ) from error
                result["status"] = "DEPLOYED"
    elif eligible:
        result["status"] = (
            "CANDIDATE_REGISTERED"
            if registered_version is not None
            else "CANDIDATE_ACCEPTED_NOT_REGISTERED"
        )
    else:
        result["status"] = "CANDIDATE_REJECTED"
    client.log_text(
        result["mlflow_run_id"],
        json.dumps(result, indent=2, allow_nan=False),
        "lab13_candidate_report_final.json",
    )
    write_json_report(report_path, result)
    return result
