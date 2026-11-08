import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from src.lab12_monitoring import (
    classification_metrics,
    read_telco_csv,
    validate_telco_frame,
    write_json_report,
)
from src.lab13_pipeline import resolve_tracking_uri


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Production model performance on labeled Telco data."
    )
    parser.add_argument("--data", required=True, help="Labeled Telco Customer Churn CSV")
    parser.add_argument("--report", default="reports/lab12/model_performance.json")
    args = parser.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    reference = read_telco_csv(ROOT / config["data"]["raw_path"])
    reference = validate_telco_frame(
        reference,
        list(reference.columns),
        require_target=True,
    )
    current_raw = read_telco_csv(ROOT / args.data)
    current = validate_telco_frame(
        current_raw,
        list(reference.columns),
        require_target=True,
        reference_frame=reference,
    )

    tracking_uri = resolve_tracking_uri(config["mlflow"]["tracking_uri"], ROOT)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = [
        version
        for version in client.search_model_versions(
            f"name='{config['registry']['model_name']}'"
        )
        if version.current_stage == "Production"
    ]
    if len(versions) != 1:
        raise RuntimeError(
            "Expected exactly one Production model version; "
            f"found {len(versions)}"
        )
    production = versions[0]
    model = mlflow.sklearn.load_model(
        f"models:/{config['registry']['model_name']}/{production.version}"
    )
    preprocessor_path = client.download_artifacts(
        production.run_id,
        "preprocessing_pipeline/preprocessor.pkl",
    )
    preprocessor = joblib.load(preprocessor_path)

    features = current.drop(columns=["customerID", "Churn"])
    labels = current["Churn"].map({"No": 0, "Yes": 1}).to_numpy(dtype=int)
    transformed = preprocessor.transform(features)
    predictions = np.asarray(model.predict(transformed), dtype=int)
    probabilities = model.predict_proba(transformed)[:, 1]
    metrics = classification_metrics(labels, predictions)
    metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    metric_name = config["quality_gate"]["metric"]
    minimum = float(config["quality_gate"]["minimum_value"])
    actual = float(metrics[metric_name])
    report = {
        "status": "HEALTHY" if actual >= minimum else "ALERT",
        "model_name": config["registry"]["model_name"],
        "model_version": str(production.version),
        "run_id": production.run_id,
        "evaluated_rows": int(len(current)),
        "metric": metric_name,
        "minimum_value": minimum,
        "actual_value": actual,
        "metrics": metrics,
    }
    write_json_report(ROOT / args.report, report)
    print(
        f"{report['status']}: Production v{production.version} "
        f"{metric_name}={actual:.4f} (minimum {minimum:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
