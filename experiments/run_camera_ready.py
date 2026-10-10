"""Offline reproducible camera-ready evaluation; outputs stay in this copy."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluation import (DATASETS, DisclosurePolicy, TrainPreprocessor, attack_order,
                        detection_metrics, load_data, numeric_attack_indices,
                        paired_bootstrap, query_attack, reconstruction_mae, sil_render)
from ledger import file_digest

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def confidence_summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(t.ppf(0.975, len(values)-1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_seed_t": [mean-half, mean+half], "n_seeds": len(values)}


def run_seed(name, seed, X, y, categories, args, out):
    run_dir = out / name / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and args.resume:
        print(f"Reusing completed {name} seed={seed}", flush=True)
        return json.loads(summary_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.25,
                                         stratify=y, random_state=seed)
    train_raw, test_raw = X.iloc[train_idx], X.iloc[test_idx]
    train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
    processor = TrainPreprocessor().fit(train_raw)
    train = processor.transform(train_raw)
    test = processor.transform(test_raw)
    train_hash = set(pd.util.hash_pandas_object(train_raw, index=False).tolist())
    test_hash = set(pd.util.hash_pandas_object(test_raw, index=False).tolist())
    assert not train_hash & test_hash, "Exact predictor duplicates crossed the split"
    np.savez_compressed(run_dir / "split_ids.npz", train=train_raw.index.to_numpy(), test=test_raw.index.to_numpy())
    write_json(run_dir / "schema.json", processor.snapshot())

    detector_rows = []
    model = XGBClassifier(n_estimators=250, max_depth=8, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", n_jobs=4, random_state=seed)
    t0 = time.perf_counter()
    model.fit(train, train_y)
    test_prob = model.predict_proba(test)[:, 1]
    detector_rows.append({"model": "XGBoost", **detection_metrics(test_y, test_prob),
                          "fit_and_predict_seconds": time.perf_counter()-t0})
    model.save_model(run_dir / "model.json")
    print(f"{name} seed={seed}: XGBoost F1={detector_rows[-1]['f1']:.4f}", flush=True)

    for model_name in ("LogReg", "RF"):
        t0 = time.perf_counter()
        if model_name == "LogReg":
            scaler = StandardScaler()
            train_scaled, test_scaled = scaler.fit_transform(train), scaler.transform(test)
            baseline = LogisticRegression(max_iter=10000, random_state=seed)
            baseline.fit(train_scaled, train_y)
            if baseline.n_iter_.max() >= baseline.max_iter:
                raise RuntimeError("Logistic regression did not converge")
            p = baseline.predict_proba(test_scaled)[:, 1]
        else:
            baseline = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                              n_jobs=4, random_state=seed)
            baseline.fit(train, train_y)
            p = baseline.predict_proba(test)[:, 1]
        detector_rows.append({"model": model_name, **detection_metrics(test_y, p),
                              "fit_and_predict_seconds": time.perf_counter()-t0})

    rng = np.random.default_rng(seed)
    calibration_positions = rng.choice(len(train), min(args.calibration_rows, len(train)), replace=False)
    explain_positions = rng.choice(len(test), min(args.explain_rows, len(test)), replace=False)
    eligible = np.where((test_y.to_numpy() == 1) & (test_prob >= 0.5))[0]
    attack_positions = rng.choice(eligible, min(args.attack_rows, len(eligible)), replace=False)
    if len(attack_positions) < 20:
        raise ValueError("Insufficient correctly detected malicious cases for the attack evaluation")
    calibration = train.iloc[calibration_positions]
    explained = test.iloc[explain_positions]
    attacked = test.iloc[attack_positions]
    explainer = shap.TreeExplainer(model)
    calibration_shap = np.asarray(explainer.shap_values(calibration))
    t0 = time.perf_counter()
    explain_shap = np.asarray(explainer.shap_values(explained))
    shap_ms = (time.perf_counter()-t0) * 1000 / len(explained)
    attack_shap = np.asarray(explainer.shap_values(attacked))
    base = float(explainer.expected_value)
    explain_prob = test_prob[explain_positions]
    policy = DisclosurePolicy(test.columns, calibration_shap)
    write_json(run_dir / "policy.json", policy.snapshot())
    mutable = numeric_attack_indices(train, processor)
    train_std = train.std().to_numpy()
    train_min, train_max = train.min().to_numpy(), train.max().to_numpy()
    write_json(run_dir / "attack_config.json", {
        "mutable_features": [train.columns[i] for i in mutable],
        "coordinates": 5, "score_queries": 10, "step_standard_deviations": 0.5,
        "score_access": "probability of malicious class, shared across all conditions",
        "baseline_probability_queries": 1, "explanation_releases": 1,
        "reference_statistics": "training split", "threshold": 0.5,
        "attack_domain": "continuous numeric feature space; not protocol-valid network traffic",
        "attack_original_row_ids": attacked.index.tolist(),
        "calibration_original_row_ids": calibration.index.tolist(),
        "explanation_original_row_ids": explained.index.tolist()})

    outcomes, fidelity_rows, comparisons = [], [], []
    all_success, all_scores = {}, {}
    replay_payloads = []
    modes = ("none", "raw", "clip", "clip_noise", "group", "pp")
    for noise_rep in range(args.noise_repeats):
        noise_seed = 100000 + seed*101 + noise_rep
        for mode in modes:
            if mode in ("none", "raw", "clip", "group") and noise_rep > 0:
                continue
            scores = None if mode == "none" else policy.release(attack_shap, mode, noise_seed)
            orders = attack_order(scores, mode, policy, mutable, len(attacked), seed+17000)
            successes, adversarial_prob, adv = query_attack(model, attacked, orders,
                                                            train_std, train_min, train_max)
            # Budget invariants also guard changes to the evaluation code.
            changed = np.abs(adv-attacked.to_numpy()) > 1e-10
            assert (changed.sum(axis=1) <= 5).all()
            assert np.all(np.abs(adv-attacked.to_numpy()) <= 0.5*train_std + 1e-7)
            assert not changed[:, [i for i in range(test.shape[1]) if i not in mutable]].any()
            all_success[(mode, noise_rep)] = successes
            all_scores[(mode, noise_rep)] = scores
            for i, row_id in enumerate(attacked.index):
                outcomes.append({"dataset": name, "seed": seed, "noise_rep": noise_rep,
                                 "mode": mode, "row_id": int(row_id),
                                 "category": str(categories.loc[row_id]),
                                 "initial_probability": float(test_prob[attack_positions[i]]),
                                 "adversarial_probability": float(adversarial_prob[i]),
                                 "success": int(successes[i]), "queries": 10,
                                 "changed_features": int(changed[i].sum()),
                                 "selected_features": ";".join(test.columns[j] for j in orders[i])})
            if mode != "none":
                t0 = time.perf_counter()
                released = policy.release(explain_shap, mode, noise_seed)
                transform_ms = (time.perf_counter()-t0)*1000/len(explained)
                public = released.copy()
                if mode in ("group", "pp"):
                    top = np.argsort(-public, axis=1, kind="stable")[:, :3]
                    mask = np.zeros_like(public, dtype=bool)
                    np.put_along_axis(mask, top, True, axis=1)
                    public[~mask] = 0.0
                # Full-score reconstruction is an internal diagnostic, not a public release.
                fidelity_rows.append({"mode": mode, "noise_rep": noise_rep,
                                      "mae_full_scores": reconstruction_mae(explain_prob, base, released),
                                      "mae_public_scores": reconstruction_mae(explain_prob, base, public),
                                      "transform_ms_per_sample": transform_ms})
        pp_success = all_success[("pp", noise_rep)]
        raw_success = all_success[("raw", 0)]
        comparisons.append({"noise_rep": noise_rep, "raw_asr": float(raw_success.mean()),
                            "pp_asr": float(pp_success.mean()), "n": len(attacked),
                            "drop": float(raw_success.mean()-pp_success.mean()),
                            "paired_bootstrap_ci95": paired_bootstrap(raw_success, pp_success, noise_seed)})
        if noise_rep == 0:
            scores = all_scores[("pp", 0)]
            for i in range(min(50, len(attacked))):
                score = float(test_prob[attack_positions[i]])
                payload = {"dataset": name, "seed": seed, "row_id": int(attacked.index[i]),
                           "role": "tier1", "timestamp": f"benchmark-seed-{seed}-event-{i:06d}",
                           "model_version": "xgboost-camera-ready-v1", "policy_version": "ppshap-v1",
                           "schema_version": "training-only-v1", "prediction": int(score >= 0.5),
                           "probability": score, "encoded_input": attacked.iloc[i].tolist(),
                           "sil": sil_render(score, policy.group_names, scores[i]),
                           "artifacts": {}}
                for key, filename in (("model", "model.json"), ("schema", "schema.json"), ("policy", "policy.json")):
                    artifact = run_dir / filename
                    payload["artifacts"][key] = {"path": str(artifact.relative_to(out)),
                                                 "sha256": file_digest(artifact)}
                replay_payloads.append(payload)

    # Quantify grouping and clipping to inform dataset-specific interpretation.
    grouped = policy.aggregate(attack_shap)
    group_before = np.argsort(-grouped, axis=1)[:, :3]
    group_after = np.argsort(-all_scores[("pp", 0)], axis=1)[:, :3]
    overlap = np.mean([len(set(a) & set(b))/3 for a, b in zip(group_before, group_after)])
    dominant = pd.Series([policy.group_names[i] for i in group_before[:, 0]]).value_counts().to_dict()
    grouped_abs = np.abs(grouped)
    concentration = grouped_abs.max(axis=1) / np.maximum(grouped_abs.sum(axis=1), 1e-12)
    diagnostics = {"semantic_top3_overlap": float(overlap),
                   "mean_max_group_abs_share": float(concentration.mean()),
                   "group_clipping_fraction": float((np.abs(grouped) > policy.group_clip).mean()),
                   "mean_initial_probability_attacked": float(test_prob[attack_positions].mean()),
                   "dominant_groups": {key: int(value) for key, value in dominant.items()}}
    pd.DataFrame(outcomes).to_csv(run_dir / "attack_outcomes.csv", index=False)
    pd.DataFrame(fidelity_rows).to_csv(run_dir / "fidelity.csv", index=False)
    write_json(run_dir / "replay_payloads.json", replay_payloads)
    summary = {"dataset": name, "seed": seed, "n_train": len(train), "n_test": len(test),
               "n_attack": len(attacked), "n_explain": len(explained),
               "detectors": detector_rows, "comparisons": comparisons,
               "shap_ms_per_sample": shap_ms, "diagnostics": diagnostics,
               "elapsed_seconds": time.perf_counter()-started}
    write_json(summary_path, summary)
    print(f"{name} seed={seed}: raw={comparisons[0]['raw_asr']:.3f}, "
          f"PP={np.mean([r['pp_asr'] for r in comparisons]):.3f}, "
          f"{summary['elapsed_seconds']:.1f}s", flush=True)
    return summary


def aggregate(out, summaries):
    detector_rows, attack_frames, fidelity_frames = [], [], []
    for entry in summaries:
        name, seed = entry["dataset"], entry["seed"]
        for row in entry["detectors"]:
            detector_rows.append({"dataset": name, "seed": seed, **row})
        directory = out / name / f"seed_{seed}"
        attack_frames.append(pd.read_csv(directory / "attack_outcomes.csv"))
        fidelity_frames.append(pd.read_csv(directory / "fidelity.csv").assign(dataset=name, seed=seed))
    detectors = pd.DataFrame(detector_rows)
    outcomes = pd.concat(attack_frames, ignore_index=True)
    fidelity = pd.concat(fidelity_frames, ignore_index=True)
    detectors.to_csv(out / "detection_all_seeds.csv", index=False)
    outcomes.to_csv(out / "attack_outcomes.csv", index=False)
    fidelity.to_csv(out / "fidelity_all_seeds.csv", index=False)
    summary = {}
    for name in sorted(outcomes.dataset.unique()):
        runs = [x for x in summaries if x["dataset"] == name]
        mode_rates = outcomes[outcomes.dataset == name].groupby(["seed", "mode", "noise_rep"]).success.mean()
        seed_rates = mode_rates.groupby(["seed", "mode"]).mean().unstack()
        mode_summary = {mode: confidence_summary(seed_rates[mode]) for mode in seed_rates}
        difference = seed_rates["raw"] - seed_rates["pp"]
        summary[name] = {"modes": mode_summary, "raw_minus_pp": confidence_summary(difference),
                         "seed_rates": seed_rates.reset_index().to_dict(orient="records"),
                         "detectors": {}, "diagnostics": {}, "fidelity": {}}
        for model in detectors.model.unique():
            subset = detectors[(detectors.dataset == name) & (detectors.model == model)]
            summary[name]["detectors"][model] = {metric: confidence_summary(subset[metric])
                for metric in ("f1", "precision", "recall", "fpr", "auroc")}
        for metric in ("semantic_top3_overlap", "mean_max_group_abs_share", "group_clipping_fraction", "mean_initial_probability_attacked"):
            summary[name]["diagnostics"][metric] = confidence_summary([x["diagnostics"][metric] for x in runs])
        for mode in fidelity["mode"].unique():
            subset = fidelity[(fidelity.dataset == name) & (fidelity["mode"] == mode)].groupby("seed").mean(numeric_only=True)
            summary[name]["fidelity"][mode] = {metric: confidence_summary(subset[metric])
                for metric in ("mae_full_scores", "mae_public_scores", "transform_ms_per_sample")}
        summary[name]["shap_ms_per_sample"] = confidence_summary([x["shap_ms_per_sample"] for x in runs])
    write_json(out / "summary.json", summary)
    category_rates = outcomes.groupby(["dataset", "seed", "mode", "noise_rep", "category"]).agg(
        n=("success", "size"), successes=("success", "sum"), asr=("success", "mean")).reset_index()
    category_rates.to_csv(out / "category_rates.csv", index=False)
    print("Aggregate results:", json.dumps({name: x["raw_minus_pp"] for name, x in summary.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--rows", type=int, default=160000)
    parser.add_argument("--attack-rows", type=int, default=500)
    parser.add_argument("--explain-rows", type=int, default=2500)
    parser.add_argument("--calibration-rows", type=int, default=2000)
    parser.add_argument("--output", default="results")
    parser.add_argument("--cache")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key not in ("resume", "output", "cache")}
    config["source_hashes"] = {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")
                              if p.name in ("run_camera_ready.py", "evaluation.py", "ledger.py")}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different configuration; use a new output directory")
    write_json(config_path, config)
    write_json(out / "runtime.json", {"python": platform.python_version(), "platform": platform.system(),
                                     "machine": platform.machine(), "packages": {
        pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "pyarrow")}})
    summaries = []
    for name in args.datasets:
        X, y, categories, metadata = load_data(name, args.rows, args.cache)
        metadata["source_sha256"] = {Path(p).name: file_digest(p) for p in metadata["source_files"]}
        write_json(out / name / "dataset.json", metadata)
        print(f"{name}: {len(X)} distinct rows from {metadata['input_rows']}; attack ratio={y.mean():.4f}", flush=True)
        for seed in args.seeds:
            summaries.append(run_seed(name, seed, X, y, categories, args, out))
    aggregate(out, summaries)


if __name__ == "__main__":
    main()
