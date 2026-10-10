"""Fixed holdouts and a checked reconstruction bound; never overwrite main results."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import shap
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from evaluation import (DATASETS, DisclosurePolicy, TrainPreprocessor, attack_order,
                        detection_metrics, numeric_attack_indices, query_attack)
from ledger import file_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_technical"
SELECTION_SEED = 20260930
CUTOFF = 1422748800  # 2015-02-01 00:00:00 UTC, between recorded capture periods.
PROTOCOL = {
    "selection_seed": SELECTION_SEED, "model_seeds": [42, 43, 44, 45, 46],
    "unsw_cutoff_unix": CUTOFF, "unsw_train_sample_cap": 160000,
    "unsw_test_sample_cap": 80000,
    "ton_split": "first fold of 4-fold StratifiedGroupKFold; unordered endpoint pair",
    "noise_repeats": 3, "calibration_rows": 2000, "explain_rows": 2500,
    "attack_rows_max": 500, "score_queries": 10, "mutable_coordinates": 5,
    "step_training_sd": 0.5, "decision_threshold": 0.5,
    "model": {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.08,
              "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
              "objective": "binary:logistic", "eval_metric": "logloss", "n_jobs": 4},
    "tuning": "none; split, hyperparameters, and sample sizes fixed before fitting",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hub_source(name):
    repo = {"UNSW_NB15": "datasets--Mouwiya--UNSW-NB15",
            "TON_IOT": "datasets--codymlewis--TON_IoT_network"}[name]
    return Path.home() / ".cache/huggingface/hub" / repo / "snapshots" / DATASETS[name]["revision"]


def read_selected_parquet(paths, selected):
    """Preserve source row IDs without retaining the full decoded mirror in RAM."""
    selected = np.sort(np.asarray(selected))
    frames, offset = [], 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
            left, right = np.searchsorted(selected, [offset, offset + len(batch)])
            ids = selected[left:right]
            if len(ids):
                part = batch.take(pa.array(ids-offset)).to_pandas()
                part.index = ids
                frames.append(part)
            offset += len(batch)
    result = pd.concat(frames)
    assert np.array_equal(result.index.to_numpy(), selected)
    return result


def endpoint_pairs(frame):
    return pd.Series(["|".join(sorted((str(a), str(b))))
                      for a, b in zip(frame.src_ip, frame.dst_ip)], index=frame.index)


def clean_partition_pair(train, test, name):
    cfg = DATASETS[name]
    excluded = ["label", cfg["category"]] + cfg["drop"]
    xtr, xte = train.drop(columns=excluded), test.drop(columns=excluded)
    joined = pd.concat([xtr.assign(_label=train.label), xte.assign(_label=test.label)])
    feature_names = list(xtr.columns)
    hashes = pd.util.hash_pandas_object(joined[feature_names], index=False)
    conflicts = joined.assign(_hash=hashes).groupby("_hash")._label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting labels for sampled identical predictors; do not fit")
    train_keep, test_keep = ~xtr.duplicated(), ~xte.duplicated()
    train, test = train.loc[train_keep], test.loc[test_keep]
    xtr, xte = xtr.loc[train_keep], xte.loc[test_keep]
    train_hash = set(pd.util.hash_pandas_object(xtr, index=False))
    overlap = pd.util.hash_pandas_object(xte, index=False).isin(train_hash)
    stats = {"train_duplicates_removed": int((~train_keep).sum()),
             "test_duplicates_removed": int((~test_keep).sum()),
             "test_train_predictor_overlaps_removed": int(overlap.sum()),
             "conflicting_predictor_label_groups": 0}
    test, xte = test.loc[~overlap], xte.loc[~overlap]
    assert not train_hash.intersection(pd.util.hash_pandas_object(xte, index=False))
    assert set(train.label.unique()) == set(test.label.unique()) == {0, 1}
    return train, test, xtr, xte, stats


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Use a fresh output directory; completed evidence is never silently reused")
    save_json(protocol_path, PROTOCOL)
    save_json(OUT / "runtime.json", {"python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "source_sha256": {n: file_digest(Path(__file__).parent/n)
                          for n in ["run_generalization.py", "evaluation.py", "ledger.py"]}})
    prepared, audit = {}, {}
    for name in DATASETS:
        source = hub_source(name)
        if name == "UNSW_NB15":
            paths = sorted((source / "data").glob("*.parquet"))
            times = pa.concat_tables([pq.read_table(p, columns=["Stime", "Ltime", "label"])
                                     for p in paths]).to_pandas()
            assert times[["Stime", "Ltime"]].notna().all().all()
            assert (times.Ltime >= times.Stime).all()
            rng = np.random.default_rng(SELECTION_SEED)
            earlier = np.flatnonzero(times.Ltime.to_numpy() < CUTOFF)
            later = np.flatnonzero(times.Stime.to_numpy() >= CUTOFF)
            train_ids = rng.choice(earlier, min(len(earlier), 160000), replace=False)
            test_ids = rng.choice(later, min(len(later), 80000), replace=False)
            selected = read_selected_parquet(paths, np.r_[train_ids, test_ids])
            train, test = selected.loc[np.sort(train_ids)], selected.loc[np.sort(test_ids)]
            detail = {"source_rows": len(times), "earlier_population": len(earlier),
                      "later_population": len(later),
                      "boundary_spanning_rows_excluded": int(len(times)-len(earlier)-len(later)),
                      "official_full_corpus_rows": 2540044,
                      "official_full_corpus_equivalence_verified": False,
                      "dates_utc": pd.to_datetime(times.Stime, unit="s", utc=True).dt.normalize()
                                     .value_counts().sort_index().rename_axis("day").astype(int)
                                     .rename(index=str).to_dict()}
        else:
            paths = [source / "train_test_network.csv"]
            frame = pd.read_csv(paths[0])
            assert not any(c.lower() in ("ts", "timestamp", "time") for c in frame)
            pairs = endpoint_pairs(frame)
            tr, te = next(StratifiedGroupKFold(n_splits=4, shuffle=True,
                random_state=SELECTION_SEED).split(frame, frame.label, groups=pairs))
            train, test = frame.iloc[tr], frame.iloc[te]
            assert not set(pairs.iloc[tr]).intersection(pairs.iloc[te])
            arrow_root = Path.home()/".cache/huggingface/datasets"/DATASETS[name]["folder"]
            train_arrow = list(arrow_root.glob(f"**/{DATASETS[name]['revision']}/*train.arrow"))[0]
            test_arrow = list(arrow_root.glob(f"**/{DATASETS[name]['revision']}/*test.arrow"))[0]
            # Arrow metadata can differ even when the actual tables are identical.
            with pa.memory_map(str(train_arrow), "r") as a, pa.memory_map(str(test_arrow), "r") as b:
                same = ipc.open_stream(a).read_all().equals(ipc.open_stream(b).read_all(), check_metadata=False)
            detail = {"source_rows": len(frame), "unordered_endpoint_pairs": int(pairs.nunique()),
                      "timestamp_present": False, "cached_train_test_tables_identical": same,
                      "train_endpoint_pairs_before_cleaning": int(pairs.iloc[tr].nunique()),
                      "test_endpoint_pairs_before_cleaning": int(pairs.iloc[te].nunique()),
                      "category_counts": frame.type.value_counts().astype(int).to_dict(),
                      "official_full_corpus_equivalence_verified": False}
        for side in (train, test):
            assert side.label.notna().all() and set(side.label.unique()) == {0, 1}
        detail["source_files"] = [{"name": p.name, "sha256": file_digest(p),
                                   "size_bytes": p.stat().st_size} for p in paths]
        detail["repository_revision"] = DATASETS[name]["revision"]
        detail["sampled_train_before_cleaning"] = len(train)
        detail["sampled_test_before_cleaning"] = len(test)
        train, test, xtr, xte, cleaning = clean_partition_pair(train, test, name)
        detail.update(cleaning)
        detail.update(n_train=len(train), n_test=len(test),
                      train_attack_count=int(train.label.sum()), test_attack_count=int(test.label.sum()))
        if name == "UNSW_NB15":
            detail.update(train_max_end=int(train.Ltime.max()), test_min_start=int(test.Stime.min()))
            assert detail["train_max_end"] < detail["test_min_start"]
        else:
            a, b = endpoint_pairs(train), endpoint_pairs(test)
            assert not set(a).intersection(b)
            detail.update(train_pairs=int(a.nunique()), test_pairs=int(b.nunique()))
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        train.to_parquet(folder / "train_source_records.parquet", index=True)
        test.to_parquet(folder / "test_source_records.parquet", index=True)
        detail["prepared_sha256"] = {p.name: file_digest(p) for p in folder.glob("*.parquet")}
        save_json(folder / "split_audit.json", detail)
        audit[name] = detail
        prepared[name] = (train, test, xtr, xte)
        print(name, "prepared", len(train), len(test), flush=True)
    save_json(OUT / "dataset_audit.json", audit)
    return prepared


def reconstruction_bound(base, phi, grouped, clipped, released, probability):
    """Pointwise triangle-inequality bound, including numerical SHAP residual."""
    phi, grouped, clipped, released = [np.asarray(v, dtype=float)
                                       for v in (phi, grouped, clipped, released)]
    top = np.argsort(-released, axis=1, kind="stable")[:, :3]
    keep = np.zeros_like(released, dtype=bool)
    np.put_along_axis(keep, top, True, axis=1)
    p_release = expit(base + np.where(keep, released, 0).sum(axis=1))
    residual = np.abs(probability-expit(base+phi.sum(axis=1)))
    clipping = np.abs(grouped-clipped).sum(axis=1)
    noise = np.abs(released-clipped).sum(axis=1)
    omitted = np.where(keep, 0, np.abs(released)).sum(axis=1)
    bound = np.minimum(1.0, residual + (clipping+noise+omitted)/4)
    return {"error": np.abs(probability-p_release), "bound": bound,
            "raw_residual": residual, "clipping_l1": clipping,
            "noise_l1": noise, "omitted_l1": omitted}


def run(name, seed, prepared):
    train_source, test_source, raw_train, raw_test = prepared
    folder = OUT / name / f"seed_{seed}"
    folder.mkdir()
    processor = TrainPreprocessor().fit(raw_train)
    train, test = processor.transform(raw_train), processor.transform(raw_test)
    save_json(folder / "schema.json", processor.snapshot())
    model = XGBClassifier(**PROTOCOL["model"], random_state=seed)
    model.fit(train, train_source.label)
    model.save_model(folder / "model.json")
    probability = model.predict_proba(test)[:, 1]
    metrics = detection_metrics(test_source.label, probability)
    pd.DataFrame({"row_id": test.index, "label": test_source.label.to_numpy(),
                  "probability": probability}).to_csv(folder / "test_predictions.csv", index=False)
    rng = np.random.default_rng(seed)
    cal = rng.choice(len(train), min(len(train), 2000), replace=False)
    explain = rng.choice(len(test), min(len(test), 2500), replace=False)
    eligible = np.flatnonzero((test_source.label.to_numpy() == 1) & (probability >= 0.5))
    attack = rng.choice(eligible, min(500, len(eligible)), replace=False)
    if len(attack) < 20:
        raise ValueError("Too few detected malicious records for the prespecified attack test")
    explainer = shap.TreeExplainer(model)
    cal_phi = np.asarray(explainer.shap_values(train.iloc[cal]), dtype=float)
    phi = np.asarray(explainer.shap_values(test.iloc[explain]), dtype=float)
    attack_phi = np.asarray(explainer.shap_values(test.iloc[attack]), dtype=float)
    base = float(explainer.expected_value)
    policy = DisclosurePolicy(test.columns, cal_phi)
    save_json(folder / "policy.json", policy.snapshot())
    mutable = numeric_attack_indices(train, processor)
    std, low, high = [v.to_numpy() for v in (train.std(), train.min(), train.max())]
    outcomes, fidelity = [], []
    releases, bound_rows, arrays = [], [], {}
    for mode, rep in [("none", 0), ("raw", 0)] + [("pp", r) for r in range(3)]:
        noise_seed = 100000 + seed*101 + rep
        scores = None if mode == "none" else policy.release(attack_phi, mode, noise_seed)
        orders = attack_order(scores, mode, policy, mutable, len(attack), seed+17000)
        successes, adv_probability, adversarial = query_attack(
            model, test.iloc[attack], orders, std, low, high)
        original = test.iloc[attack].to_numpy()
        change = np.abs(adversarial-original)
        assert (change <= 0.5*std+1e-7).all()
        assert ((change > 1e-10).sum(axis=1) <= 5).all()
        assert not (change[:, np.setdiff1d(np.arange(test.shape[1]), mutable)] > 1e-10).any()
        arrays[f"adv_{mode}_{rep}"] = adversarial
        arrays[f"orders_{mode}_{rep}"] = orders
        for i, pos in enumerate(attack):
            outcomes.append({"mode": mode, "noise_rep": rep, "row_id": int(test.index[pos]),
                "initial_probability": float(probability[pos]),
                "adversarial_probability": float(adv_probability[i]), "success": int(successes[i]),
                "queries": 10, "changed_features": int((change[i] > 1e-10).sum())})
        if mode == "pp":
            released = policy.release(phi, "pp", noise_seed)
            grouped = policy.aggregate(phi)
            clipped = np.clip(grouped, -policy.group_clip, policy.group_clip)
            bound = reconstruction_bound(base, phi, grouped, clipped, released, probability[explain])
            assert (bound["error"] <= bound["bound"]+1e-10).all()
            releases.append(released)
            fidelity.append(float(bound["error"].mean()))
            bound_rows.append(pd.DataFrame(bound).assign(noise_rep=rep, row_id=test.index[explain]))
    np.savez_compressed(folder / "verification_arrays.npz", **arrays,
        attack_row_ids=test.index[attack].to_numpy(), explain_row_ids=test.index[explain].to_numpy(),
        calibration_row_ids=train.index[cal].to_numpy(), explain_phi=phi, base=np.array(base),
        attack_phi=attack_phi, releases=np.stack(releases), train_std=std, train_min=low,
        train_max=high, mutable=mutable, original_attack=test.iloc[attack].to_numpy(),
        explain_probability=probability[explain])
    outcomes = pd.DataFrame(outcomes)
    outcomes.to_csv(folder / "attack_outcomes.csv", index=False)
    pd.concat(bound_rows).to_csv(folder / "bound_checks.csv", index=False)
    rates = outcomes.groupby(["mode", "noise_rep"]).success.mean().groupby("mode").mean()
    summary = {"dataset": name, "seed": seed, "detection": metrics, "n_attack": len(attack),
        "asr": rates.to_dict(), "raw_minus_pp": float(rates["raw"]-rates["pp"]),
        "public_mae": float(np.mean(fidelity)), "bound_cases": len(explain)*3}
    save_json(folder / "summary.json", summary)
    print(name, seed, "F1", round(metrics["f1"], 4), "ASR", rates.round(4).to_dict(), flush=True)
    return summary


def main():
    prepared = prepare()
    summaries = []
    for name, data in prepared.items():
        for seed in PROTOCOL["model_seeds"]:
            summaries.append(run(name, seed, data))
    save_json(OUT / "runs.json", summaries)
    summary = {}
    for name in DATASETS:
        rows = [x for x in summaries if x["dataset"] == name]
        def describe(values):
            return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
        summary[name] = {
            "detection": {k: describe([x["detection"][k] for x in rows])
                          for k in ("f1", "precision", "recall", "fpr", "auroc")},
            "asr": {k: describe([x["asr"][k] for x in rows]) for k in ("none", "raw", "pp")},
            "raw_minus_pp": describe([x["raw_minus_pp"] for x in rows]),
            "public_mae": describe([x["public_mae"] for x in rows]),
            "n_attack_per_seed": [x["n_attack"] for x in rows],
            "uncertainty_scope": "training and attack randomness on one fixed holdout, not split variability"}
    save_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
