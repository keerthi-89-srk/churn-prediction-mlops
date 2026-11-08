import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.lab12_monitoring import (
    detect_concept_drift,
    read_telco_csv,
    validate_telco_frame,
    write_json_report,
)
from src.lab13_pipeline import resolve_tracking_uri


ROOT = Path(__file__).resolve().parent


def _evaluate(frame: pd.DataFrame, model, preprocessor) -> tuple[np.ndarray, np.ndarray]:
    features = frame.drop(columns=["customerID", "Churn"])
    labels = frame["Churn"].map({"No": 0, "Yes": 1}).to_numpy(dtype=int)
    transformed = preprocessor.transform(features)
    return labels, np.asarray(model.predict(transformed), dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure labeled-window performance drift.")
    parser.add_argument("--reference", required=True, help="Labeled reference Telco CSV")
    parser.add_argument("--current", required=True, help="Labeled current Telco CSV")
    parser.add_argument("--report", default="reports/lab12/concept_drift.json")
    args = parser.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    reference_raw = read_telco_csv(ROOT / args.reference)
    reference = validate_telco_frame(
        reference_raw, list(reference_raw.columns), require_target=True
    )
    current_raw = read_telco_csv(ROOT / args.current)
    current = validate_telco_frame(
        current_raw,
        list(reference_raw.columns),
        require_target=True,
        reference_frame=reference_raw,
    )
    tracking_uri = resolve_tracking_uri(config["mlflow"]["tracking_uri"], ROOT)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    production_versions = [
        version
        for version in client.search_model_versions(
            f"name='{config['registry']['model_name']}'"
        )
        if version.current_stage == "Production"
    ]
    if len(production_versions) != 1:
        raise RuntimeError(
            "Expected exactly one Production model version; "
            f"found {len(production_versions)}"
        )
    production = production_versions[0]
    model = mlflow.sklearn.load_model(
        f"models:/{config['registry']['model_name']}/{production.version}"
    )
    preprocessor_path = client.download_artifacts(
        production.run_id,
        "preprocessing_pipeline/preprocessor.pkl",
    )
    preprocessor = joblib.load(preprocessor_path)
    reference_y, reference_pred = _evaluate(reference, model, preprocessor)
    current_y, current_pred = _evaluate(current, model, preprocessor)
    report = detect_concept_drift(
        reference_y,
        reference_pred,
        current_y,
        current_pred,
        maximum_drop=float(config["lab12"]["concept_drift_max_f1_drop"]),
    )
    write_json_report(ROOT / args.report, report)
    print(
        f"{report['status']}: reference F1="
        f"{report['reference_metrics']['f1_score']:.4f}, current F1="
        f"{report['current_metrics']['f1_score']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
