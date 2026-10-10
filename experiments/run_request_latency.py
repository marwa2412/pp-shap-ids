"""Local completion-time benchmark, distinct from amortized SHAP cost."""

import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import shap
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from evaluation import DATASETS, DisclosurePolicy, TrainPreprocessor, sil_render
from ledger import append, checkpoint, file_digest, verify
from run_generalization import save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_reviewer3" / "latency"
PROTOCOL = {
    "model_seed_or_evaluation_policy_seed": 42,
    "backbones": ["XGBoost", "LogReg"], "batch_sizes": [1, 32, 256],
    "measured_batches_per_size": 100, "warmup_batches_per_size": 5,
    "job_and_record_selection_seed": 2026093003, "threads": 4,
    "included": ["frozen raw-feature preprocessing", "prediction", "SHAP",
                 "grouping/clipping/noise", "top-three SIL rendering",
                 "payload construction and in-memory hash-linked append"],
    "excluded": ["feature extraction from packets", "request scheduling and batch filling",
                 "network", "durable I/O", "checkpoint transport/authentication",
                 "model loading and training", "full snapshot verification and replay"],
    "input_policy": "explain every sampled record, regardless of predicted class",
    "timing": "perf_counter_ns; total is local batch completion; fresh log per batch",
    "quantiles": "numpy linear p50/p95 across 100 batches; not confidence intervals",
    "assertions": "each output probability checked; every completed log verified outside timed boundary",
}


def restore_processor(schema):
    processor = TrainPreprocessor()
    for key in ("columns", "numeric", "categorical", "medians", "frequencies"):
        setattr(processor, key, schema[key])
    return processor


def restore_policy(snapshot, columns):
    policy = DisclosurePolicy.__new__(DisclosurePolicy)
    policy.columns = list(columns)
    policy.group_names = snapshot["group_names"]
    policy.group_indices = [np.array([columns.index(c) for c in snapshot["groups"][g]])
                            for g in policy.group_names]
    policy.feature_clip = np.array(snapshot["feature_clip"])
    policy.group_clip = np.array(snapshot["group_clip"])
    policy.eta = snapshot["eta"]
    return policy


def timed_batch(raw, context, noise_seed):
    processor, model, explain, policy, artifacts = context
    entries = []
    start = time.perf_counter_ns()
    encoded = processor.transform(raw)
    after_preprocess = time.perf_counter_ns()
    probabilities = model.predict_proba(encoded)[:, 1]
    after_predict = time.perf_counter_ns()
    phi = explain(encoded)
    after_shap = time.perf_counter_ns()
    scores = policy.release(phi, "pp", noise_seed)
    rendered = [sil_render(float(p), policy.group_names, s) for p, s in zip(probabilities, scores)]
    after_render = time.perf_counter_ns()
    for i, row_id in enumerate(encoded.index):
        payload = {"row_id": int(row_id), "role": "tier1", "event": "local-latency-benchmark",
                   "encoded_input": encoded.iloc[i].tolist(), "artifacts": artifacts,
                   "probability": float(probabilities[i]), "prediction": int(probabilities[i] >= 0.5),
                   "sil": rendered[i]}
        append(entries, payload, "latency-batch")
    end = time.perf_counter_ns()
    intervals = [start, after_preprocess, after_predict, after_shap, after_render, end]
    names = ["preprocess", "predict", "shap", "release_render", "ledger_append"]
    timing = {name + "_ms": (right - left) / 1e6
              for name, left, right in zip(names, intervals[:-1], intervals[1:])}
    timing["total_ms"] = (end - start) / 1e6
    assert verify(entries, checkpoint(entries, "latency-batch"))[0]
    return probabilities, timing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Do not overwrite recorded timing evidence")
    save_json(OUT / "protocol.json", {**PROTOCOL, "platform": platform.platform(),
        "source_sha256": {n: file_digest(Path(__file__).parent / n) for n in
                          ("run_request_latency.py", "evaluation.py", "ledger.py")}})
    contexts, raw_records, expected, references = {}, {}, {}, {}
    for name, cfg in DATASETS.items():
        source = ROOT / "results_technical" / name
        raw_records[name] = pd.read_parquet(source / "test_source_records.parquet").drop(
            columns=["label", cfg["category"]] + cfg["drop"])
        for backbone in PROTOCOL["backbones"]:
            folder = source / "seed_42" if backbone == "XGBoost" else ROOT / "results_reviewer3" / name
            policy_folder = folder if backbone == "XGBoost" else folder / "seed_42"
            schema_path, policy_path = folder / "schema.json", policy_folder / "policy.json"
            processor = restore_processor(json.loads(schema_path.read_text()))
            policy = restore_policy(json.loads(policy_path.read_text()), processor.columns)
            if backbone == "XGBoost":
                model_path = folder / "model.json"
                model = XGBClassifier()
                model.load_model(model_path)
                model.set_params(n_jobs=4)
                explainer = shap.TreeExplainer(model)
                explain = lambda x, e=explainer: np.asarray(e.shap_values(x))
            else:
                model_path = folder / "model.joblib"
                model = joblib.load(model_path)
                arrays = np.load(policy_folder / "verification_arrays.npz")
                masker = shap.maskers.Independent(arrays["background_mean"][None, :], max_samples=1)
                explainer = shap.LinearExplainer(model.steps[1][1], masker)
                explain = lambda x, e=explainer, s=model.steps[0][1]: np.asarray(e.shap_values(s.transform(x)))
            artifacts = {key: file_digest(path) for key, path in
                         (("model", model_path), ("schema", schema_path), ("policy", policy_path))}
            contexts[name, backbone] = (processor, model, explain, policy, artifacts)
            expected[name, backbone] = pd.read_csv(folder / "test_predictions.csv").set_index("row_id").probability
            references[f"{name}/{backbone}"] = artifacts
    save_json(OUT / "artifacts.json", references)
    rng = np.random.default_rng(PROTOCOL["job_and_record_selection_seed"])
    jobs = [(name, model, size, rep) for name in DATASETS for model in PROTOCOL["backbones"]
            for size in PROTOCOL["batch_sizes"] for rep in range(100)]
    rng.shuffle(jobs)
    rows = []
    with threadpool_limits(limits=4):
        for key, context in contexts.items():
            for size in PROTOCOL["batch_sizes"]:
                for rep in range(5):
                    timed_batch(raw_records[key[0]].iloc[:size], context, 900000 + rep)
        for job, (name, model, size, rep) in enumerate(jobs):
            positions = rng.choice(len(raw_records[name]), size, replace=False)
            raw = raw_records[name].iloc[positions]
            probabilities, timing = timed_batch(raw, contexts[name, model], 500000 + job)
            error = np.max(np.abs(probabilities - expected[name, model].loc[raw.index].to_numpy()))
            assert error <= 3e-8
            rows.append({"dataset": name, "model": model, "batch_size": size, "repeat": rep,
                         "job": job, **timing, "max_prediction_error": float(error),
                         "log_verified": True, "row_ids": json.dumps(raw.index.tolist())})
            if (job + 1) % 100 == 0:
                print("Latency measured batches", job + 1, "/", len(jobs), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "timings.csv", index=False)
    save_json(OUT / "complete.json", {"measured_batches": len(rows),
        "measured_record_predictions": sum(row["batch_size"] for row in rows),
        "all_prediction_checks_passed": True, "all_log_checks_passed": True})


if __name__ == "__main__":
    main()
