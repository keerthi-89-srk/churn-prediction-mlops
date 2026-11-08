import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from api.preprocessing import clean_prediction_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_deployment_bundle() -> Path:
    if DEPLOYMENT_DIR.exists():
        raise FileExistsError(
            f"{DEPLOYMENT_DIR} already exists. Inspect or move it manually; "
            "this script never deletes or overwrites deployment artifacts."
        )

    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get(
        "mlflow", {}
    ).get("tracking_uri")
    model_name = config.get("registry", {}).get("model_name")
    if not tracking_uri or not model_name:
        raise ValueError("config.yaml must define MLflow tracking URI and model name")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    versions = client.search_model_versions(f"name='{model_name}'")
    production_versions = [
        version
        for version in versions
        if version.current_stage == "Production"
    ]
    if len(production_versions) != 1:
        raise RuntimeError(
            f"Expected exactly one Production version for {model_name}; "
            f"found {len(production_versions)}."
        )

    selected = production_versions[0]
    if not selected.run_id:
        raise RuntimeError("Production model version has no source run ID")

    model_id = selected.source.rsplit("/", 1)[-1]
    if not model_id.startswith("m-"):
        raise RuntimeError("Production model source does not identify an MLflow model")

    preprocessor_source = client.download_artifacts(
        selected.run_id,
        "preprocessing_pipeline/preprocessor.pkl",
    )
    preprocessor = joblib.load(preprocessor_source)
    model_uri = f"models:/{model_name}/{selected.version}"
    model = mlflow.sklearn.load_model(model_uri)

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise TypeError("Production model must provide predict() and predict_proba()")
    if not callable(getattr(preprocessor, "transform", None)):
        raise TypeError("Run-linked preprocessor does not provide transform()")

    raw_data_path = PROJECT_ROOT / config["data"]["raw_path"]
    sample = pd.read_csv(raw_data_path, nrows=5)
    sample_features = clean_prediction_data(sample)
    transformed = preprocessor.transform(sample_features)
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    if len(predictions) != len(sample_features):
        raise RuntimeError("Production model returned an unexpected prediction count")
    if probabilities.shape[0] != len(sample_features) or probabilities.shape[1] < 2:
        raise RuntimeError("Production model returned invalid churn probabilities")

    run = client.get_run(selected.run_id)
    metrics = {
        key: float(value)
        for key, value in run.data.metrics.items()
    }

    with tempfile.TemporaryDirectory(
        prefix=".deployment-stage-",
        dir=PROJECT_ROOT,
    ) as staging_path:
        staging_dir = Path(staging_path)
        model_path = staging_dir / "model.pkl"
        preprocessor_path = staging_dir / "preprocessor.pkl"
        metadata_path = staging_dir / "model_metadata.json"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        metadata = {
            "model_name": model_name,
            "model_version": int(selected.version),
            "model_id": model_id,
            "run_id": selected.run_id,
            "preprocessor_run_id": selected.run_id,
            "model_uri": model_uri,
            "model_source": selected.source,
            "model_stage": selected.current_stage,
            "metrics": metrics,
            "preprocessor_artifact": (
                "preprocessing_pipeline/preprocessor.pkl"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {
                "model.pkl": sha256_file(model_path),
                "preprocessor.pkl": sha256_file(preprocessor_path),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        if DEPLOYMENT_DIR.exists():
            raise FileExistsError(
                f"{DEPLOYMENT_DIR} appeared while preparing the bundle; "
                "no existing files were changed."
            )
        Path(staging_path).rename(DEPLOYMENT_DIR)

    print(
        f"[SUCCESS] Prepared {model_name} version {selected.version} "
        f"from MLflow run {selected.run_id}."
    )
    print(f"[SUCCESS] Model and matching preprocessor saved to {DEPLOYMENT_DIR}.")
    print(
        "[INFO] Existing MLflow records and model files were not modified; "
        "no model was retrained."
    )
    return DEPLOYMENT_DIR


if __name__ == "__main__":
    try:
        prepare_deployment_bundle()
    except Exception as error:
        print(
            f"[ERROR] Deployment bundle preparation failed "
            f"({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        sys.exit(1)
