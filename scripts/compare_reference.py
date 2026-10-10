"""Compare a completed full reproduction with retained non-timing aggregates."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, path, differences):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Different fields: {path}")
        for key in expected:
            compare(actual[key], expected[key], path+[key], differences)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different list length: {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path+[str(index)], differences)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        tolerance = 1e-6 if "RF" in path and "auroc" in path else 1e-12
        if not math.isfinite(actual) or abs(actual-expected) > tolerance:
            raise ValueError(f"Numerical mismatch at {'/'.join(path)}: {actual} versus {expected}")
        if actual != expected:
            differences.append({"field": "/".join(path), "absolute_difference": abs(actual-expected),
                                "tolerance": tolerance})
    elif actual != expected:
        raise ValueError(f"Different value: {path}")


def check(work_dir):
    record = json.loads((work_dir / "reproduction.json").read_text())
    if record["profile"] != "paper-protocol" or not all(
        any(item["stage"] == stage and item["status"] == "passed" for item in record["stages"])
        for stage in ("initial", "verify")):
        raise ValueError("A completed full initial and verification run is required, not a smoke run")
    reference = ROOT / "reference_results"
    initial = json.loads((work_dir / "results/summary.json").read_text())
    original = json.loads((reference / "initial_summary.json").read_text())
    differences = []
    for name in original:
        for key in ("modes", "raw_minus_pp", "seed_rates", "detectors", "diagnostics"):
            compare(initial[name][key], original[name][key], ["initial", name, key], differences)
        for mode in original[name]["fidelity"]:
            for key in ("mae_full_scores", "mae_public_scores"):
                compare(initial[name]["fidelity"][mode][key], original[name]["fidelity"][mode][key],
                        ["initial", name, "fidelity", mode, key], differences)
    for subdir, filename, key in (
        ("results_technical", "xgboost_verification.json", "independent_aggregates"),
        ("results_reviewer3", "lr_verification.json", "backbone_results"),
    ):
        actual = json.loads((work_dir / subdir / "verification.json").read_text())
        expected = json.loads((reference / filename).read_text())
        if actual["status"] != "passed":
            raise ValueError(f"Verifier did not pass: {subdir}")
        compare(actual[key], expected[key], [subdir, key], differences)
    return {"status": "passed", "scope": "Initial detection/attack/fidelity/diagnostics and structured XGBoost/LR aggregates",
            "timings_compared_to_original": False, "nonzero_differences_within_tolerance": differences,
            "tolerances": {"refitted_rf_auroc": 1e-6, "other_metrics": 1e-12}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = check(args.work_dir)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Reference comparison failed: {error}\n")


if __name__ == "__main__":
    main()
