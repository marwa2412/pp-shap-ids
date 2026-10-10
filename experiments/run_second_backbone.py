"""Prespecified logistic-regression disclosure tests on the frozen holdouts."""

import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluation import (DATASETS, DisclosurePolicy, TrainPreprocessor, attack_order,
                        detection_metrics, numeric_attack_indices, query_attack)
from ledger import file_digest
from run_generalization import reconstruction_bound, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_reviewer3"
PROTOCOL = {
    "backbone": "StandardScaler followed by LogisticRegression",
    "C": 1.0, "solver": "lbfgs", "tol": 1e-4, "max_iter": 10000,
    "class_weight": None, "training_threads": 4,
    "model_fits_per_dataset": 1,
    "evaluation_seeds": [42, 43, 44, 45, 46],
    "seed_meaning": "background/calibration, fidelity/attack selection, disclosure randomness; not model fits",
    "background_and_calibration_rows": 2000, "fidelity_rows": 2500,
    "attack_rows": 500, "noise_repeats": 3,
    "conditions": ["none", "raw", "pp"], "score_queries": 10,
    "coordinate_budget": 5, "step_training_sd": 0.5,
    "explainer": "shap.LinearExplainer with Independent masker; standardized inputs; log-odds",
    "split": "existing UNSW chronological and ToN unordered-pair holdouts, unchanged",
    "tuning": "none; recorded before fitting; extension motivated by reviewer 3",
    "cross_model_comparison": "model-specific detected-positive samples; not paired across backbones",
}


def prepare(name):
    source = ROOT / "results_technical" / name
    audit = json.loads((source / "split_audit.json").read_text())
    for filename, expected in audit["prepared_sha256"].items():
        assert file_digest(source / filename) == expected
    train_source = pd.read_parquet(source / "train_source_records.parquet")
    test_source = pd.read_parquet(source / "test_source_records.parquet")
    cfg = DATASETS[name]
    excluded = ["label", cfg["category"]] + cfg["drop"]
    raw_train, raw_test = [frame.drop(columns=excluded) for frame in (train_source, test_source)]
    processor = TrainPreprocessor().fit(raw_train)
    return train_source, test_source, processor, processor.transform(raw_train), processor.transform(raw_test)


def evaluate_seed(name, seed, model, train, test, test_source, processor, probability):
    folder = OUT / name / f"seed_{seed}"
    folder.mkdir()
    rng = np.random.default_rng(seed)
    cal = rng.choice(len(train), 2000, replace=False)
    explain = rng.choice(len(test), 2500, replace=False)
    eligible = np.flatnonzero((test_source.label.to_numpy() == 1) & (probability >= 0.5))
    if len(eligible) < 500:
        raise ValueError("Fewer than 500 eligible alerts; do not silently alter protocol")
    attack = rng.choice(eligible, 500, replace=False)
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    background = scaler.transform(train.iloc[cal])
    explainer = shap.LinearExplainer(logistic, shap.maskers.Independent(background, max_samples=2000))
    cal_phi = np.asarray(explainer.shap_values(background))
    phi = np.asarray(explainer.shap_values(scaler.transform(test.iloc[explain])))
    attack_phi = np.asarray(explainer.shap_values(scaler.transform(test.iloc[attack])))
    base = float(explainer.expected_value)
    assert np.allclose(expit(base + phi.sum(axis=1)), probability[explain], atol=1e-10, rtol=0)
    policy = DisclosurePolicy(train.columns, cal_phi)
    save_json(folder / "policy.json", policy.snapshot())
    mutable = numeric_attack_indices(train, processor)
    std, low, high = [v.to_numpy() for v in (train.std(), train.min(), train.max())]
    rows, releases, bounds, arrays = [], [], [], {}
    for mode, rep in [("none", 0), ("raw", 0)] + [("pp", r) for r in range(3)]:
        noise_seed = 100000 + seed * 101 + rep
        scores = None if mode == "none" else policy.release(attack_phi, mode, noise_seed)
        orders = attack_order(scores, mode, policy, mutable, len(attack), seed + 17000)
        success, adversarial_p, adversarial = query_attack(model, test.iloc[attack], orders, std, low, high)
        change = np.abs(adversarial - test.iloc[attack].to_numpy())
        assert (change <= 0.5 * std + 1e-7).all()
        assert ((change > 1e-10).sum(axis=1) <= 5).all()
        assert not (change[:, np.setdiff1d(np.arange(test.shape[1]), mutable)] > 1e-10).any()
        arrays[f"adv_{mode}_{rep}"] = adversarial
        arrays[f"orders_{mode}_{rep}"] = orders
        for i, pos in enumerate(attack):
            rows.append({"mode": mode, "noise_rep": rep, "row_id": int(test.index[pos]),
                         "initial_probability": float(probability[pos]),
                         "adversarial_probability": float(adversarial_p[i]),
                         "success": int(success[i]), "queries": 10})
        if mode == "pp":
            released = policy.release(phi, "pp", noise_seed)
            grouped = policy.aggregate(phi)
            clipped = np.clip(grouped, -policy.group_clip, policy.group_clip)
            bound = reconstruction_bound(base, phi, grouped, clipped, released, probability[explain])
            assert (bound["error"] <= bound["bound"] + 1e-10).all()
            releases.append(released)
            bounds.append(pd.DataFrame(bound).assign(noise_rep=rep, row_id=test.index[explain]))
    pd.DataFrame(rows).to_csv(folder / "attack_outcomes.csv", index=False)
    pd.concat(bounds).to_csv(folder / "bound_checks.csv", index=False)
    np.savez_compressed(folder / "verification_arrays.npz", **arrays,
        calibration_row_ids=train.index[cal].to_numpy(), explain_row_ids=test.index[explain].to_numpy(),
        attack_row_ids=test.index[attack].to_numpy(), explain_phi=phi, attack_phi=attack_phi,
        base=np.array(base), releases=np.stack(releases), background_mean=background.mean(axis=0),
        train_std=std, train_min=low, train_max=high, mutable=mutable,
        original_attack=test.iloc[attack].to_numpy(), explain_probability=probability[explain])
    frame = pd.DataFrame(rows)
    rates = frame.groupby("mode").success.mean().to_dict()
    save_json(folder / "summary.json", {"asr": rates,
        "public_mae": float(pd.concat(bounds).error.mean()),
        "max_raw_reconstruction_error": float(np.abs(expit(base + phi.sum(axis=1)) - probability[explain]).max())})
    print(name, seed, rates, flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "backbone_protocol.json").exists():
        raise FileExistsError("Existing evidence is not overwritten; use a fresh research copy")
    save_json(OUT / "backbone_protocol.json", PROTOCOL)
    save_json(OUT / "runtime.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "shap", "xgboost", "joblib")},
        "source_sha256": {n: file_digest(Path(__file__).parent / n) for n in
                          ("run_second_backbone.py", "evaluation.py", "run_generalization.py", "ledger.py")},
    })
    with threadpool_limits(limits=4):
        for name in DATASETS:
            train_source, test_source, processor, train, test = prepare(name)
            folder = OUT / name
            folder.mkdir()
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=10000, tol=1e-4, class_weight=None))
            model.fit(train, train_source.label)
            assert model.steps[1][1].n_iter_.max() < 10000
            joblib.dump(model, folder / "model.joblib")
            save_json(folder / "schema.json", processor.snapshot())
            probability = model.predict_proba(test)[:, 1]
            pd.DataFrame({"row_id": test.index, "label": test_source.label.to_numpy(),
                          "probability": probability}).to_csv(folder / "test_predictions.csv", index=False)
            metrics = detection_metrics(test_source.label, probability)
            save_json(folder / "detection.json", {**metrics,
                "iterations": int(model.steps[1][1].n_iter_.max()),
                "model_sha256": file_digest(folder / "model.joblib")})
            print(name, "LR detection", metrics, flush=True)
            for seed in PROTOCOL["evaluation_seeds"]:
                evaluate_seed(name, seed, model, train, test, test_source, processor, probability)
    save_json(OUT / "backbone_complete.json", {"datasets": 2, "models": 2, "evaluation_runs": 10})


if __name__ == "__main__":
    main()
