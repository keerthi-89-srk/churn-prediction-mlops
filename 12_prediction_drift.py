import argparse
from pathlib import Path

import yaml

from src.lab12_monitoring import (
    detect_prediction_drift,
    read_jsonl,
    write_json_report,
)


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare logged prediction windows.")
    parser.add_argument("--reference-log", required=True)
    parser.add_argument("--current-log", required=True)
    parser.add_argument("--report", default="reports/lab12/prediction_drift.json")
    args = parser.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    report = detect_prediction_drift(
        read_jsonl(ROOT / args.reference_log),
        read_jsonl(ROOT / args.current_log),
        threshold=float(config["lab12"]["prediction_drift_psi_threshold"]),
    )
    write_json_report(ROOT / args.report, report)
    print(f"{report['status']}: {len(report['drifted_features'])} drifted outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
