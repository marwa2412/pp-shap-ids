"""Export saved numerical results and category summaries without document output."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "initial": "results/summary.json",
    "ledger": "results/ledger/summary.json",
    "structured_verification": "results_technical/verification.json",
    "ledger_scaling": "results_technical/ledger/scaling_summary.json",
    "linear_and_latency_verification": "results_reviewer3/verification.json",
}
CATEGORY_COLUMNS = ["dataset", "category", "attack_attempts_across_seeds", "seeds_with_cases",
                    "raw_mean", "pp_mean", "mean_paired_seed_drop"]


def category_summary(outcomes):
    required = {"dataset", "category", "seed", "mode", "noise_rep", "success"}
    if not required.issubset(outcomes.columns):
        raise ValueError(f"Missing attack outcome columns: {sorted(required - set(outcomes.columns))}")
    if outcomes.empty or outcomes[list(required)].isna().any().any():
        raise ValueError("Attack outcomes must be nonempty and have no missing grouping/outcome values")
    if not outcomes["success"].isin([0, 1]).all():
        raise ValueError("Attack success values must be binary")
    rows = []
    for (dataset, category), part in outcomes.groupby(["dataset", "category"]):
        # Preserve the original averaging order: repetitions, then seeds.
        rates = part.groupby(["seed", "mode", "noise_rep"]).success.mean().groupby(["seed", "mode"]).mean().unstack()
        if not {"raw", "pp"}.issubset(rates.columns) or rates[["raw", "pp"]].isna().any().any():
            raise ValueError(f"Missing matched raw/protected seed rates for {dataset}/{category}")
        counts = part[part["mode"] == "raw"].groupby("seed").size()
        rows.append({"dataset": dataset, "category": category,
                     "attack_attempts_across_seeds": int(counts.sum()),
                     "seeds_with_cases": len(counts), "raw_mean": float(rates["raw"].mean()),
                     "pp_mean": float(rates["pp"].mean()),
                     "mean_paired_seed_drop": float((rates["raw"] - rates["pp"]).mean())})
    return pd.DataFrame(rows, columns=CATEGORY_COLUMNS)


def export_results(work_dir=ROOT, output_dir=None):
    work_dir = Path(work_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else work_dir / "numerical_results"
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory; refusing to overwrite {output_dir}")
    results, hashes = {}, {}
    for key, relative in SOURCES.items():
        raw = (work_dir / relative).read_bytes()
        results[key] = json.loads(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    for key in ("structured_verification", "linear_and_latency_verification"):
        if results[key].get("status") != "passed":
            raise ValueError(f"A passed verification report is required: {SOURCES[key]}")
    outcomes_path = work_dir / "results/attack_outcomes.csv"
    hashes["results/attack_outcomes.csv"] = hashlib.sha256(outcomes_path.read_bytes()).hexdigest()
    categories = category_summary(pd.read_csv(outcomes_path))
    payload = {
        "scope": "Export of supplied saved results; not a new experiment or an independent verification",
        "units": "Original JSON fields and units preserved; category ASRs and drops are fractions, not percentages",
        "source_sha256": hashes,
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "numerical_summary.json").write_text(serialized, encoding="utf-8")
    categories.to_csv(output_dir / "category_summary.csv", index=False)
    return {"status": "passed", "category_rows": len(categories),
            "output_files": ["numerical_summary.json", "category_summary.csv"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT, help="Read results from this experiment workspace")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    try:
        print(json.dumps(export_results(args.work_dir, args.output_dir), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Numerical export failed: {error}\n")


if __name__ == "__main__":
    main()
