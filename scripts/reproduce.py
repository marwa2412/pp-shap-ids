"""Run numerical experiments, verification, and JSON/CSV exports in isolation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "initial": [("run_camera_ready.py", [])],
    "ledger": [("run_ledger_validation.py", [])],
    "holdouts": [("run_generalization.py", [])],
    "controls": [("run_random_controls.py", [])],
    "replay-scaling": [("run_scaling_replay.py", [])],
    "linear": [("run_second_backbone.py", [])],
    "latency": [("run_request_latency.py", [])],
    "verify": [("verify_technical.py", []), ("verify_reviewer3.py", [])],
    "results": [("export_numerical_results.py", [])],
}
REQUIRED = {
    "initial": [],
    "ledger": ["results/summary.json"],
    "holdouts": [],
    "controls": ["results_technical/UNSW_NB15/split_audit.json", "results_technical/TON_IOT/split_audit.json"],
    "replay-scaling": ["results/ledger/summary.json"],
    "linear": ["results_technical/UNSW_NB15/split_audit.json", "results_technical/TON_IOT/split_audit.json"],
    "latency": ["results_technical/UNSW_NB15/seed_42/summary.json", "results_reviewer3/backbone_complete.json"],
    "verify": ["results_technical/random_control_protocol.json", "results_technical/ledger/scaling_summary.json",
               "results_reviewer3/latency/complete.json"],
    "results": ["results/summary.json", "results/attack_outcomes.csv", "results/ledger/summary.json",
                "results_technical/verification.json", "results_technical/ledger/scaling_summary.json",
                "results_reviewer3/verification.json"],
}


def frozen_hashes():
    manifest = json.loads((ROOT / "paper_code_manifest.json").read_text())
    expected = {**manifest["sha256"], **manifest["additional_sha256"]}
    for name, value in expected.items():
        actual = hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest()
        if actual != value:
            raise ValueError(f"Paper code was changed: {name}; use a separately documented experimental fork")
    return expected


def create_workspace(path, profile):
    hashes = frozen_hashes()
    marker = path / "reproduction.json"
    if path.exists():
        if not marker.is_file():
            raise FileExistsError("Choose a new work directory; existing unrecognized directories are never modified")
        record = json.loads(marker.read_text())
        if record["profile"] != profile or record["paper_code_sha256"] != hashes:
            raise ValueError("Workspace profile/source differs; choose a new work directory")
        for name, value in hashes.items():
            if hashlib.sha256((path / "experiments" / name).read_bytes()).hexdigest() != value:
                raise ValueError(f"Workspace code changed: {name}")
        return record
    path.mkdir(parents=True)
    (path / "experiments").mkdir()
    for source in (ROOT / "experiments").iterdir():
        if source.suffix in (".py", ".txt"):
            shutil.copyfile(source, path / "experiments" / source.name)
    record = {"profile": profile, "paper_code_sha256": hashes, "python": sys.version,
              "stages": [], "scope": "New reproduction; not the retained manuscript evidence"}
    marker.write_text(json.dumps(record, indent=2)+"\n")
    return record


def stage_commands(stage):
    if stage == "smoke":
        return [("run_camera_ready.py", ["--seeds", "42", "--rows", "160000", "--attack-rows", "50",
                 "--explain-rows", "100", "--calibration-rows", "200"]),
                ("run_ledger_validation.py", ["--trials", "3"])]
    return STAGES[stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=["all", "smoke", *STAGES], default="all")
    parser.add_argument("--plan", action="store_true", help="Print the command order without creating or running anything")
    args = parser.parse_args()
    path = args.work_dir.expanduser().resolve()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if args.plan:
        for stage in stages:
            for script, options in stage_commands(stage):
                print(stage + ": python experiments/" + script + " " + " ".join(options))
        return
    try:
        profile = "smoke-not-paper-results" if args.stage == "smoke" else "paper-protocol"
        record = create_workspace(path, profile)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4",
                   XDG_CACHE_HOME=str(path / ".cache"))
        if any(stage in stages for stage in ("initial", "smoke", "holdouts", "replay-scaling")):
            subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--check"], env=env, check=True)
        for stage in stages:
            if any(item["stage"] == stage for item in record["stages"]):
                raise FileExistsError(f"Stage {stage} was already attempted. Use a new workspace; no automatic overwriting/resume.")
            for relative in REQUIRED.get(stage, []):
                if not (path / relative).is_file():
                    raise FileNotFoundError(f"Missing prerequisite for {stage}: {relative}")
            entry = {"stage": stage, "status": "running"}
            record["stages"].append(entry)
            marker = path / "reproduction.json"
            marker.write_text(json.dumps(record, indent=2)+"\n")
            try:
                for script, options in stage_commands(stage):
                    print(f"Running {stage}: {script}", flush=True)
                    subprocess.run([sys.executable, str(path / "experiments" / script), *options],
                                   cwd=path, env=env, check=True)
                if stage == "verify":
                    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "experiments", "-v"],
                                   cwd=path, env=env, check=True)
                entry["status"] = "passed"
            except Exception:
                entry["status"] = "failed"
                raise
            finally:
                marker.write_text(json.dumps(record, indent=2)+"\n")
        print(f"Completed {args.stage}. Evidence is in {path}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Reproduction stopped: {error}\nOriginal paper evidence was not selected as an output.\n")


if __name__ == "__main__":
    main()
