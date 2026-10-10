"""In-process ledger size sweep and replay from cached source network records."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluation import TrainPreprocessor, load_data, sil_render
from ledger import append, canonical, checkpoint, file_digest, verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_technical" / "ledger"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay():
    rows, source_audit = [], {}
    for name in ("UNSW_NB15", "TON_IOT"):
        raw, labels, _, metadata = load_data(name)
        prior = json.loads((ROOT / "results" / name / "dataset.json").read_text())
        hashes = {Path(p).name: file_digest(p) for p in metadata["source_files"]}
        assert hashes == prior["source_sha256"], "Cached source bytes changed"
        source_audit[name] = hashes
        for directory in sorted((ROOT / "results" / name).glob("seed_*")):
            schema = json.loads((directory / "schema.json").read_text())
            processor = TrainPreprocessor()
            for key in ("columns", "numeric", "categorical", "medians", "frequencies"):
                setattr(processor, key, schema[key])
            payloads = json.loads((directory / "replay_payloads.json").read_text())
            for p in payloads:
                for ref in p["artifacts"].values():
                    assert file_digest(ROOT / "results" / ref["path"]) == ref["sha256"]
            ids = [p["row_id"] for p in payloads]
            encoded = processor.transform(raw.loc[ids])
            saved = np.array([p["encoded_input"] for p in payloads])
            errors = np.abs(encoded.to_numpy()-saved).max(axis=1)
            model = XGBClassifier()
            model.load_model(directory / "model.json")
            probabilities = model.predict_proba(encoded)[:, 1]
            for i, p in enumerate(payloads):
                items = p["sil"]["items"]
                regenerated = sil_render(float(probabilities[i]),
                    [x["group"] for x in items], np.array([x["released_score"] for x in items]))
                rows.append({"dataset": name, "seed": p["seed"], "row_id": p["row_id"],
                    "raw_label": int(labels.loc[p["row_id"]]),
                    "encoded_max_error": float(errors[i]),
                    "probability_error": abs(float(probabilities[i])-p["probability"]),
                    "decision_matches": int(probabilities[i] >= 0.5) == p["prediction"],
                    "sil_matches_stored_release": regenerated == p["sil"]})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "source_replay.csv", index=False)
    assert (results.encoded_max_error <= 1e-12).all()
    assert (results.probability_error <= 1e-7).all()
    assert results.decision_matches.all() and results.sil_matches_stored_release.all()
    summary = {"n": len(results), "decisions_match": int(results.decision_matches.sum()),
               "max_encoded_error": float(results.encoded_max_error.max()),
               "max_probability_error": float(results.probability_error.max()),
               "sil_matches": int(results.sil_matches_stored_release.sum()),
               "source_sha256": source_audit,
               "boundary": "cached feature records through frozen preprocessing and detector; "
                           "renderer uses stored released scores, not regenerated noise; not packet replay"}
    save_json(OUT / "source_replay_summary.json", summary)
    print("Source-record replay:", json.dumps(summary), flush=True)


def size_sweep():
    payloads = []
    for p in sorted((ROOT / "results").glob("*/seed_*/replay_payloads.json")):
        payloads.extend(json.loads(p.read_text()))
    assert len(payloads) == 500
    sizes, repeats = [500, 5000, 50000], 7
    save_json(OUT / "scaling_protocol.json", {"sizes": sizes, "timing_repeats": repeats,
        "payload_pool": 500, "payload_reuse": "cycled benchmark payloads; synthetic load",
        "warmup_records": 500, "randomized_order_seed": 20260930,
        "excluded": ["disk I/O", "network", "concurrency", "checkpoint authentication",
                     "distributed recovery", "production availability"],
        "python": platform.python_version(), "platform": platform.platform(),
        "source_sha256": {n: file_digest(Path(__file__).parent/n)
                          for n in ("ledger.py", "run_scaling_replay.py")}})
    warm = []
    for p in payloads:
        append(warm, p, "warmup")
    assert verify(warm, checkpoint(warm, "warmup"))[0]
    del warm
    rng = np.random.default_rng(20260930)
    jobs = [(n, r) for n in sizes for r in range(repeats)]
    rng.shuffle(jobs)
    rows = []
    for n, repeat in jobs:
        entries, log_id = [], f"size-sweep-{n}-{repeat}"
        start = time.perf_counter_ns()
        for i in range(n):
            append(entries, payloads[i % len(payloads)], log_id)
        append_seconds = (time.perf_counter_ns()-start)/1e9
        anchor = checkpoint(entries, log_id)
        start = time.perf_counter_ns()
        valid, _ = verify(entries, anchor)
        verify_seconds = (time.perf_counter_ns()-start)/1e9
        assert valid
        serialized_bytes = sum(len(canonical(e))+1 for e in entries)
        entries[-1]["payload"]["role"] = "forged"
        mutation_rejected = not verify(entries, anchor)[0]
        assert mutation_rejected
        rows.append({"n": n, "repeat": repeat, "append_seconds": append_seconds,
                     "verify_seconds": verify_seconds, "jsonl_bytes": serialized_bytes,
                     "valid_accepted": valid, "last_record_mutation_rejected": mutation_rejected})
        print("Ledger sweep", n, repeat, round(append_seconds, 4), round(verify_seconds, 4), flush=True)
        del entries
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "scaling_timings.csv", index=False)
    summary = []
    for n, group in frame.groupby("n"):
        row = {"n": int(n), "repeats": len(group), "jsonl_bytes": int(group.jsonl_bytes.median())}
        for op in ("append", "verify"):
            values = group[f"{op}_seconds"]
            row[op] = {"median_seconds": float(values.median()), "min_seconds": float(values.min()),
                       "max_seconds": float(values.max()), "median_us_per_record": float(values.median()*1e6/n)}
        summary.append(row)
    save_json(OUT / "scaling_summary.json", summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "source_replay_summary.json").exists() or (OUT / "scaling_timings.csv").exists():
        raise FileExistsError("Existing evidence must not be overwritten")
    replay()
    size_sweep()


if __name__ == "__main__":
    main()
