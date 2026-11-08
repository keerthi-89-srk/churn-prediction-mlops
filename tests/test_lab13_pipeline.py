import json
import hashlib
import shutil
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from src.lab13_pipeline import (
    candidate_is_eligible,
    _deployment_preflight,
    ingest_batch,
    promote_candidate_version,
    register_candidate_version,
    resolve_tracking_uri,
    run_end_to_end,
)


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _baseline_cached() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "churn.csv", dtype={"TotalCharges": "string"})


def _baseline() -> pd.DataFrame:
    return _baseline_cached().copy()


def _incoming_batch(tmp_path: Path, *, rows: int = 100) -> Path:
    reference = _baseline()
    batch = reference.iloc[:rows].copy()
    batch["customerID"] = [f"LAB13-{index:05d}" for index in range(len(batch))]
    path = tmp_path / "incoming.csv"
    batch.to_csv(path, index=False)
    return path


def test_ingestion_records_a_batch_once_and_leaves_source_unchanged(tmp_path):
    source = _incoming_batch(tmp_path)
    original_hash = source.read_bytes()
    manifest = tmp_path / "ingestion" / "manifest.json"
    target_dir = tmp_path / "ingestion" / "batches"
    first = ingest_batch(
        source,
        _baseline(),
        ingest_directory=target_dir,
        manifest_path=manifest,
    )
    second = ingest_batch(
        source,
        _baseline(),
        ingest_directory=target_dir,
        manifest_path=manifest,
    )
    assert first["status"] == "INGESTED"
    assert second["status"] == "DUPLICATE_BATCH"
    assert source.read_bytes() == original_hash
    records = json.loads(manifest.read_text(encoding="utf-8"))["batches"]
    assert len(records) == 1


def test_ingestion_rejects_customer_ids_already_in_reference(tmp_path):
    source = tmp_path / "existing_ids.csv"
    _baseline().iloc[:10].to_csv(source, index=False)
    with pytest.raises(ValueError, match="already present"):
        ingest_batch(
            source,
            _baseline(),
            ingest_directory=tmp_path / "ingested",
            manifest_path=tmp_path / "manifest.json",
        )


def test_candidate_must_pass_gate_and_improve_production():
    assert candidate_is_eligible(
        candidate_metric=0.61,
        production_metric=0.60,
        quality_gate_minimum=0.5,
        minimum_improvement=0.0,
    ) == (True, [])
    eligible, reasons = candidate_is_eligible(
        candidate_metric=0.49,
        production_metric=0.60,
        quality_gate_minimum=0.5,
        minimum_improvement=0.0,
    )
    assert not eligible
    assert set(reasons) == {"quality_gate_failed", "production_model_not_beaten"}


def test_relative_mlflow_database_path_resolves_from_project_root(tmp_path):
    uri = resolve_tracking_uri("sqlite:///mlflow.db", tmp_path)
    assert uri == f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"


def test_deployment_preflight_rejects_unhealthy_current_cluster(monkeypatch):
    results = [
        SimpleNamespace(returncode=0, stdout="29.9.2"),
        SimpleNamespace(returncode=0, stdout="docker-desktop"),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "spec": {"replicas": 1},
                    "status": {"availableReplicas": 0},
                }
            ),
        ),
    ]
    monkeypatch.setattr(
        "src.lab13_pipeline.subprocess.run",
        lambda *args, **kwargs: results.pop(0),
    )
    with pytest.raises(RuntimeError, match="Deployment is unhealthy"):
        _deployment_preflight()


def test_deployment_preflight_passes_only_when_current_workload_is_ready(monkeypatch):
    results = [
        SimpleNamespace(returncode=0, stdout="29.9.2"),
        SimpleNamespace(returncode=0, stdout="docker-desktop"),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "spec": {"replicas": 1},
                    "status": {"availableReplicas": 1},
                }
            ),
        ),
    ]
    monkeypatch.setattr(
        "src.lab13_pipeline.subprocess.run",
        lambda *args, **kwargs: results.pop(0),
    )
    _deployment_preflight()


def test_isolated_mlflow_registry_rejects_bad_candidate_and_promotes_eligible_one(
    tmp_path,
    monkeypatch,
    request,
):
    database = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database.as_posix()}"
    old_tracking_uri = mlflow.get_tracking_uri()
    old_registry_uri = mlflow.get_registry_uri()
    request.addfinalizer(lambda: mlflow.set_tracking_uri(old_tracking_uri))
    request.addfinalizer(lambda: mlflow.set_registry_uri(old_registry_uri))
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "lab13-isolated-registry",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    model = DummyClassifier(strategy="constant", constant=0).fit([[0], [1]], [0, 1])
    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(
            model,
            name="isolated_candidate",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    model_name = "Lab13IsolatedTelcoModel"
    client.create_registered_model(model_name)
    production = client.create_model_version(
        name=model_name,
        source=model_info.model_uri,
        run_id=run_id,
    )
    client.transition_model_version_stage(
        name=model_name,
        version=production.version,
        stage="Production",
    )

    rejected = register_candidate_version(
        client,
        model_name=model_name,
        model_uri=model_info.model_uri,
        run_id=run_id,
        eligible=False,
    )
    assert rejected is None
    assert [
        item.version
        for item in client.search_model_versions(f"name='{model_name}'")
        if item.current_stage == "Production"
    ] == [production.version]

    candidate = register_candidate_version(
        client,
        model_name=model_name,
        model_uri=model_info.model_uri,
        run_id=run_id,
        eligible=True,
    )
    assert candidate is not None
    promote_candidate_version(
        client,
        model_name=model_name,
        candidate_version=candidate.version,
        eligible=True,
    )
    versions = client.search_model_versions(f"name='{model_name}'")
    assert {item.version: item.current_stage for item in versions} == {
        production.version: "Archived",
        candidate.version: "Production",
    }


def test_drifted_batch_retrains_and_tracks_candidate_in_isolated_mlflow(
    tmp_path,
    monkeypatch,
    request,
):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "config.yaml", project / "config.yaml")
    (project / "data" / "raw").mkdir(parents=True)
    shutil.copy2(
        ROOT / "data" / "raw" / "churn.csv",
        project / "data" / "raw" / "churn.csv",
    )
    (project / "deployment").mkdir()
    for name in ("model.pkl", "preprocessor.pkl"):
        shutil.copy2(ROOT / "deployment" / name, project / "deployment" / name)

    tracking_uri = f"sqlite:///{(tmp_path / 'isolated-mlflow.db').as_posix()}"
    old_tracking_uri = mlflow.get_tracking_uri()
    old_registry_uri = mlflow.get_registry_uri()
    request.addfinalizer(lambda: mlflow.set_tracking_uri(old_tracking_uri))
    request.addfinalizer(lambda: mlflow.set_registry_uri(old_registry_uri))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    name = "Lab13IsolatedPipelineModel"
    client.create_registered_model(name)
    artifact_directory = tmp_path / "mlflow-artifacts"
    experiment_id = client.create_experiment(
        "lab13-isolated-pipeline",
        artifact_location=artifact_directory.as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    production_model = __import__("joblib").load(project / "deployment" / "model.pkl")
    with mlflow.start_run(run_name="isolated-production") as run:
        mlflow.log_metric("f1_score", 0.6)
        info = mlflow.sklearn.log_model(
            production_model,
            name="production_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        production_run_id = run.info.run_id
    version = client.create_model_version(
        name=name,
        source=info.model_uri,
        run_id=production_run_id,
    )
    client.transition_model_version_stage(
        name=name,
        version=version.version,
        stage="Production",
    )
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["registry"]["model_name"] = name
    config["mlflow"]["experiment_name"] = "lab13-isolated-pipeline"
    (project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata = {
        "model_name": name,
        "model_version": int(version.version),
        "run_id": production_run_id,
        "sha256": {
            filename: hashlib.sha256(
                (project / "deployment" / filename).read_bytes()
            ).hexdigest()
            for filename in ("model.pkl", "preprocessor.pkl")
        },
    }
    (project / "deployment" / "model_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    baseline = _baseline()
    evolved = baseline.sample(n=400, random_state=23).copy()
    evolved["customerID"] = [f"ISOLATED-{index:05d}" for index in range(len(evolved))]
    evolved["Contract"] = "Two year"
    evolved["tenure"] = 0
    evolved["TotalCharges"] = "0"
    incoming = tmp_path / "drifted-telco.csv"
    evolved.to_csv(incoming, index=False)
    report_path = tmp_path / "lab13-report.json"
    result = run_end_to_end(
        incoming,
        config_path=project / "config.yaml",
        ingest_directory=tmp_path / "ingested",
        manifest_path=tmp_path / "manifest.json",
        report_path=report_path,
    )

    assert result["data_drift"]["status"] == "DRIFT_DETECTED"
    assert result["ingestion"]["status"] == "INGESTED"
    assert result["mlflow_run_id"]
    assert result["candidate_metrics"]["f1_score"] >= 0.0
    assert result["production_version"] == str(version.version)
    assert report_path.exists()
    experiment_run = client.get_run(result["mlflow_run_id"])
    assert experiment_run.info.status == "FINISHED"
    assert experiment_run.data.params["incoming_sha256"]
    assert "candidate_f1_score" in experiment_run.data.metrics
    assert [
        item.version
        for item in client.search_model_versions(f"name='{name}'")
        if item.current_stage == "Production"
    ] == [version.version]

def test_no_drift_ingests_batch_but_does_not_retrain_or_touch_registry(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "config.yaml", project / "config.yaml")
    (project / "data" / "raw").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "raw" / "churn.csv", project / "data" / "raw" / "churn.csv")
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["lab12"]["data_drift_psi_threshold"] = 100.0
    (project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    incoming = _incoming_batch(tmp_path, rows=100)
    report_path = tmp_path / "reports" / "result.json"
    result = run_end_to_end(
        incoming,
        config_path=project / "config.yaml",
        ingest_directory=tmp_path / "ingested",
        manifest_path=tmp_path / "manifest.json",
        report_path=report_path,
    )
    assert result["status"] == "NO_RETRAIN_TRIGGER"
    assert result["ingestion"]["status"] == "INGESTED"
    assert not result["trigger_conditions"]["retraining_triggered"]
    assert report_path.exists()


def test_repeated_pipeline_batch_is_not_retrained_or_registered(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "config.yaml", project / "config.yaml")
    (project / "data" / "raw").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "raw" / "churn.csv", project / "data" / "raw" / "churn.csv")
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["lab12"]["data_drift_psi_threshold"] = 100.0
    (project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    incoming = _incoming_batch(tmp_path, rows=100)
    ingest_batch(
        incoming,
        _baseline(),
        ingest_directory=tmp_path / "ingested",
        manifest_path=tmp_path / "manifest.json",
    )
    result = run_end_to_end(
        incoming,
        config_path=project / "config.yaml",
        ingest_directory=tmp_path / "ingested",
        manifest_path=tmp_path / "manifest.json",
        report_path=tmp_path / "reports" / "result.json",
    )
    assert result["status"] == "DUPLICATE_BATCH"
    assert result["registered_candidate"] is None


def test_incompatible_churn2_is_rejected_before_ingestion(tmp_path):
    with pytest.raises(ValueError, match="Column mismatch"):
        run_end_to_end(
            ROOT / "data" / "raw" / "churn2.csv",
            ingest_directory=tmp_path / "ingested",
            manifest_path=tmp_path / "manifest.json",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "manifest.json").exists()
