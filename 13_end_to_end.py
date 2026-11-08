import argparse
from pathlib import Path

from src.lab13_pipeline import ROOT, run_end_to_end


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, drift-check, ingest, retrain, and compare a Telco batch."
    )
    parser.add_argument("--incoming", required=True, help="Labeled Telco CSV")
    parser.add_argument("--report", default="reports/lab13/end_to_end.json")
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help="Register an eligible candidate in MLflow; off by default.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote only an eligible candidate; off by default.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Build a unique image and update the existing Telco deployment.",
    )
    args = parser.parse_args()
    report = run_end_to_end(
        ROOT / args.incoming,
        report_path=ROOT / args.report,
        register_candidate=args.register_candidate,
        promote_candidate=args.promote,
        deploy_candidate=args.deploy,
    )
    print(f"Lab 13 status: {report['status']}")
    print(f"Report: {ROOT / args.report}")
    if report["status"] == "CANDIDATE_REJECTED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
