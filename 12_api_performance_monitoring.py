import argparse
from pathlib import Path

import yaml

from src.lab12_monitoring import (
    performance_trend,
    read_jsonl,
    summarize_api_performance,
    write_json_report,
)


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize API request metrics.")
    parser.add_argument("--log", default="logs/inference/inference_log.jsonl")
    parser.add_argument("--report", default="reports/lab12/api_performance.json")
    args = parser.parse_args()
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    events = read_jsonl(ROOT / args.log)
    limits = config["lab12"]
    report = summarize_api_performance(
        events,
        maximum_p95_latency_ms=float(limits["api_max_p95_latency_ms"]),
        maximum_error_rate=float(limits["api_max_error_rate"]),
    )
    report["trend"] = performance_trend(events)
    write_json_report(ROOT / args.report, report)
    print(
        f"{report['status']}: requests={report['request_count']}, "
        f"errors={report['error_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
