"""Mutation tests and model replay using artifacts from the corrected experiments."""

import argparse
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ledger import append, canonical, checkpoint, file_digest, verify, verify_artifacts

ROOT = Path(__file__).resolve().parents[1]


def rebuilt(entries, log_id):
    result = []
    for entry in entries:
        append(result, entry["payload"], log_id)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    root = ROOT / args.results
    out = root / "ledger"
    out.mkdir(parents=True, exist_ok=True)
    payloads = []
    for path in sorted(root.glob("*/seed_*/replay_payloads.json")):
        payloads.extend(json.loads(path.read_text()))
    if len(payloads) < 10:
        raise ValueError("Run the detector evaluation before ledger validation")
    log_id = "artiis-camera-ready-validation-v1"
    entries = []
    for payload in payloads:
        append(entries, payload, log_id)
    trusted = checkpoint(entries, log_id)
    # The threat model protects this checkpoint independently of the mutable log.
    (out / "audit_log.jsonl").write_bytes(b"\n".join(canonical(x) for x in entries) + b"\n")
    (out / "trusted_checkpoint.json").write_bytes(canonical(trusted) + b"\n")
    rng = np.random.default_rng(20260929)
    mutations = ("explanation_edit", "actor_role_edit", "policy_version_edit", "timestamp_edit",
                 "interior_delete", "insert", "duplicate", "reorder", "suffix_delete", "rollback", "rewrite_and_rehash")
    rows = []
    for kind in mutations:
        for trial in range(args.trials):
            altered = copy.deepcopy(entries)
            i = int(rng.integers(1, len(entries)-1))
            if kind == "explanation_edit":
                altered[i]["payload"]["sil"]["items"][0]["released_score"] += 1
            elif kind == "actor_role_edit":
                altered[i]["payload"]["role"] = "administrator"
            elif kind == "policy_version_edit":
                altered[i]["payload"]["policy_version"] = "forged-policy"
            elif kind == "timestamp_edit":
                altered[i]["payload"]["timestamp"] = "forged-time"
            elif kind == "interior_delete":
                del altered[i]
            elif kind in ("insert", "duplicate"):
                injected = copy.deepcopy(altered[i])
                if kind == "insert":
                    injected["payload"]["row_id"] = -1
                altered.insert(i, injected)
            elif kind == "reorder":
                altered[i], altered[i-1] = altered[i-1], altered[i]
            elif kind == "suffix_delete":
                altered = altered[:-1]
            elif kind == "rollback":
                altered = altered[:i]
            else:
                altered[i]["payload"]["role"] = "administrator"
                altered = rebuilt(altered, log_id)
            ok, reason = verify(altered, trusted)
            rows.append({"case": kind, "trial": trial, "expected_valid": False,
                         "accepted": ok, "correct": not ok, "reason": reason})

    for trial in range(args.trials):
        prefix = copy.deepcopy(entries[:int(rng.integers(2, len(entries)))])
        prefix_trust = checkpoint(prefix, log_id)
        ok, reason = verify(prefix, prefix_trust)
        rows.append({"case": "valid_snapshot", "trial": trial, "expected_valid": True,
                     "accepted": ok, "correct": ok, "reason": reason})
        extended = copy.deepcopy(prefix)
        append(extended, payloads[0], log_id)
        ok, reason = verify(extended, checkpoint(extended, log_id))
        rows.append({"case": "valid_append_updated_checkpoint", "trial": trial, "expected_valid": True,
                     "accepted": ok, "correct": ok, "reason": reason})

    # Demonstrate, rather than conceal, the consequence of losing the trust anchor.
    altered = copy.deepcopy(entries)
    altered[0]["payload"]["role"] = "administrator"
    altered = rebuilt(altered, log_id)
    compromised_accepted, _ = verify(altered, checkpoint(altered, log_id))

    artifact_rows = []
    with tempfile.TemporaryDirectory(prefix="artiis-ledger-artifacts-") as directory:
        stage = Path(directory)
        for trial in range(args.trials):
            payload = payloads[int(rng.integers(len(payloads)))]
            for kind in ("model", "schema", "policy"):
                for mutation in ("missing", "substituted"):
                    for ref in payload["artifacts"].values():
                        target = stage / ref["path"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(root / ref["path"], target)
                    target = stage / payload["artifacts"][kind]["path"]
                    if mutation == "missing":
                        target.unlink()
                    else:
                        target.write_bytes(target.read_bytes() + b" ")
                    ok, reason = verify_artifacts(payload, stage)
                    artifact_rows.append({"case": f"{mutation}_{kind}", "trial": trial,
                                          "rejected": not ok, "reason": reason})

    checksums = {}
    for payload in payloads:
        for ref in payload["artifacts"].values():
            if ref["path"] not in checksums:
                checksums[ref["path"]] = file_digest(root / ref["path"])
            if checksums[ref["path"]] != ref["sha256"]:
                raise ValueError("Replay artifact integrity verification failed")
    replay_rows = []
    for model_path in sorted({p["artifacts"]["model"]["path"] for p in payloads}):
        batch = [p for p in payloads if p["artifacts"]["model"]["path"] == model_path]
        schema_path = root / batch[0]["artifacts"]["schema"]["path"]
        columns = json.loads(schema_path.read_text())["columns"]
        model = XGBClassifier()
        model.load_model(root / model_path)
        encoded = pd.DataFrame([p["encoded_input"] for p in batch], columns=columns)
        probabilities = model.predict_proba(encoded)[:, 1]
        for payload, probability in zip(batch, probabilities):
            error = abs(float(probability)-payload["probability"])
            replay_rows.append({"dataset": payload["dataset"], "seed": payload["seed"],
                                "row_id": payload["row_id"], "absolute_probability_error": error,
                                "decision_matches": int(probability >= 0.5) == payload["prediction"],
                                "within_tolerance": error <= 1e-7})

    benchmark = []
    for repeat in range(10):
        t0 = time.perf_counter()
        benchmark_entries = []
        for payload in payloads:
            append(benchmark_entries, payload, log_id)
        append_ms = (time.perf_counter()-t0)*1000/len(payloads)
        anchor = checkpoint(benchmark_entries, log_id)
        t0 = time.perf_counter()
        ok, _ = verify(benchmark_entries, anchor)
        verify_ms = (time.perf_counter()-t0)*1000/len(payloads)
        assert ok
        benchmark.append({"repeat": repeat, "append_ms_per_record": append_ms,
                          "verify_ms_per_record": verify_ms})

    trials = pd.DataFrame(rows)
    artifact_trials = pd.DataFrame(artifact_rows)
    replay = pd.DataFrame(replay_rows)
    performance = pd.DataFrame(benchmark)
    trials.to_csv(out / "mutation_trials.csv", index=False)
    artifact_trials.to_csv(out / "artifact_trials.csv", index=False)
    replay.to_csv(out / "replay_results.csv", index=False)
    performance.to_csv(out / "timings.csv", index=False)
    attacks = trials[~trials.expected_valid]
    controls = trials[trials.expected_valid]
    summary = {"records": len(entries), "trials_per_case": args.trials,
               "mutation_cases": len(mutations), "mutation_trials": len(attacks),
               "mutations_detected": int(attacks.correct.sum()),
               "valid_control_trials": len(controls), "valid_control_false_alarms": int((~controls.correct).sum()),
               "artifact_trials": len(artifact_trials), "artifact_failures_detected": int(artifact_trials.rejected.sum()),
               "replay_records": len(replay), "replay_decisions_match": int(replay.decision_matches.sum()),
               "max_replay_probability_error": float(replay.absolute_probability_error.max()),
               "replay_within_tolerance": int(replay.within_tolerance.sum()),
               "checkpoint_compromise_control_accepted": compromised_accepted,
               "append_ms_per_record_mean": float(performance.append_ms_per_record.mean()),
               "verify_ms_per_record_mean": float(performance.verify_ms_per_record.mean()),
               "scope": "in-process snapshot integrity and detector replay from retained encoded vectors",
               "trust_boundary": "attacker can replace log, but not independently retained checkpoint",
               "timestamps": "synthetic sequence labels; not trusted wall-clock evidence"}
    summary["source_sha256"] = {name: file_digest(Path(__file__).parent / name)
                                for name in ("ledger.py", "run_ledger_validation.py")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not (trials.correct.all() and artifact_trials.rejected.all() and replay.within_tolerance.all()
            and replay.decision_matches.all() and compromised_accepted):
        raise SystemExit("Ledger validation failed; inspect saved trials")


if __name__ == "__main__":
    main()
