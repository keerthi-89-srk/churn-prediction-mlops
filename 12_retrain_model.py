"""Compatibility entry point for drift-gated Lab 13 candidate retraining."""

from src.lab13_pipeline import ROOT


def main() -> int:
    import argparse
    from pathlib import Path
    import json

    from src.lab13_pipeline import run_end_to_end

    parser = argparse.ArgumentParser(
        description="Train and compare a Lab 12 candidate from a validated Telco batch."
    )
    parser.add_argument("--incoming", required=True)
    parser.add_argument("--report", default="reports/lab12/retraining.json")
    parser.add_argument("--register-candidate", action="store_true")
    args = parser.parse_args()
    result = run_end_to_end(
        ROOT / Path(args.incoming),
        report_path=ROOT / args.report,
        register_candidate=args.register_candidate,
    )
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "CANDIDATE_REJECTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
