import argparse
from pathlib import Path

import yaml

from src.lab12_monitoring import (
    detect_data_drift,
    read_telco_csv,
    validate_telco_frame,
    write_json_report,
)


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Telco feature distributions.")
    parser.add_argument("--current", required=True, help="Incoming Telco CSV")
    parser.add_argument("--report", default="reports/lab12/data_drift.json")
    args = parser.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    reference = read_telco_csv(ROOT / config["data"]["raw_path"])
    current = read_telco_csv(args.current)
    reference = validate_telco_frame(
        reference, list(reference.columns), require_target=True
    )
    current = validate_telco_frame(
        current,
        list(reference.columns),
        require_target=False,
        reference_frame=reference,
    )
    report = detect_data_drift(
        reference,
        current,
        threshold=float(config["lab12"]["data_drift_psi_threshold"]),
    )
    write_json_report(ROOT / args.report, report)
    print(f"{report['status']}: {len(report['drifted_features'])} drifted features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
