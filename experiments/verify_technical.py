"""Independent arithmetic and saved-model checks; does not use runner metric helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_technical"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode(raw, schema):
    columns = {}
    for c in schema["columns"]:
        if c in schema["numeric"]:
            columns[c] = pd.to_numeric(raw[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).fillna(schema["medians"][c])
        else:
            columns[c] = raw[c].fillna("__MISSING__").astype(str).map(
                schema["frequencies"][c]).fillna(0.)
    return pd.DataFrame(columns, index=raw.index).astype(float)


def manual_metrics(y, p):
    y, pred = np.asarray(y), np.asarray(p) >= 0.5
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    positive, negative = tp+fn, tn+fp
    ranks = rankdata(p, method="average")
    auc = (ranks[y == 1].sum()-positive*(positive+1)/2)/(positive*negative)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1": 2*tp/(2*tp+fp+fn), "precision": tp/(tp+fp) if tp+fp else 0.,
        "recall": tp/positive, "fpr": fp/negative, "accuracy": (tp+tn)/len(y), "auroc": auc}


def check():
    runtime = json.loads((OUT/"runtime.json").read_text())
    for name, value in runtime["source_sha256"].items():
        assert digest(ROOT/"experiments"/name) == value
    random_protocol = json.loads((OUT/"random_control_protocol.json").read_text())
    assert digest(ROOT/"experiments/run_random_controls.py") == random_protocol["source_sha256"]
    total_predictions, total_attacks, total_bounds, bound_nonvacuous = 0, 0, 0, 0
    maximum_model_error, maximum_raw_residual = 0., 0.
    aggregates = {}
    for dataset in ("UNSW_NB15", "TON_IOT", "UNSW_NB15_RANDOM", "TON_IOT_RANDOM"):
        folder = OUT/dataset
        audit = json.loads((folder/"split_audit.json").read_text())
        for name, value in audit["prepared_sha256"].items():
            assert digest(folder/name) == value
        tr = pd.read_parquet(folder/"train_source_records.parquet")
        te = pd.read_parquet(folder/"test_source_records.parquet")
        assert not set(tr.index).intersection(te.index)
        is_unsw = dataset.startswith("UNSW")
        dropped = (["srcip", "dstip", "Stime", "Ltime", "attack_cat", "label"] if is_unsw
                   else ["src_ip", "dst_ip", "type", "label"])
        xtr, xte = tr.drop(columns=dropped), te.drop(columns=dropped)
        category = "attack_cat" if is_unsw else "type"
        for part in (tr, te):
            assert set(part.label.unique()) == {0, 1}
            labels = part[category].fillna("normal").astype(str).str.strip().str.lower()
            assert (labels[part.label == 0] == "normal").all()
            assert labels[part.label == 1].ne("normal").all()
            assert labels[part.label == 1].ne("").all()
        assert not set(pd.util.hash_pandas_object(xtr, index=False)).intersection(
            pd.util.hash_pandas_object(xte, index=False))
        if dataset == "UNSW_NB15":
            assert tr.Ltime.max() < 1422748800 <= te.Stime.min()
        if dataset == "TON_IOT":
            pairs = lambda d: {tuple(sorted((a, b))) for a, b in zip(d.src_ip, d.dst_ip)}
            assert not pairs(tr).intersection(pairs(te))
        if dataset.endswith("_RANDOM"):
            parent = OUT/dataset.removesuffix("_RANDOM")
            a, b = pd.read_parquet(parent/"train_source_records.parquet"), pd.read_parquet(parent/"test_source_records.parquet")
            assert set(tr.index) | set(te.index) == set(a.index) | set(b.index)
            pd.testing.assert_frame_equal(pd.concat([tr, te]).sort_index(), pd.concat([a, b]).sort_index())
            assert tr.label.value_counts().to_dict() == a.label.value_counts().to_dict()
            assert te.label.value_counts().to_dict() == b.label.value_counts().to_dict()
        runs = []
        for seed in (42, 43, 44, 45, 46):
            run = folder/f"seed_{seed}"
            summary = json.loads((run/"summary.json").read_text())
            schema = json.loads((run/"schema.json").read_text())
            assert schema["columns"] == list(xtr.columns)
            for c in schema["numeric"]:
                series = pd.to_numeric(xtr[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                median = float(series.median()) if series.notna().any() else 0.
                assert median == schema["medians"][c]
            for c in schema["categorical"]:
                freq = xtr[c].fillna("__MISSING__").astype(str).value_counts()/len(xtr)
                assert freq.to_dict() == schema["frequencies"][c]
            test, train = encode(xte, schema), encode(xtr, schema)
            model = XGBClassifier()
            model.load_model(run/"model.json")
            probabilities = model.predict_proba(test)[:, 1]
            stored = pd.read_csv(run/"test_predictions.csv", float_precision="round_trip")
            assert list(stored.row_id) == list(te.index)
            assert np.array_equal(stored.label, te.label)
            error = float(np.max(np.abs(probabilities-stored.probability)))
            assert error < 1e-7
            maximum_model_error = max(maximum_model_error, error)
            actual = manual_metrics(te.label.to_numpy(), probabilities)
            for key, value in actual.items():
                assert abs(value-summary["detection"][key]) < 1e-12, (dataset, seed, key)
            total_predictions += len(test)
            a = np.load(run/"verification_arrays.npz")
            attacked = test.loc[a["attack_row_ids"]].to_numpy()
            assert np.array_equal(attacked, a["original_attack"])
            assert (te.loc[a["attack_row_ids"], "label"] == 1).all()
            assert (model.predict_proba(pd.DataFrame(attacked, columns=test.columns))[:, 1] >= .5).all()
            assert set(a["calibration_row_ids"]).issubset(train.index)
            assert set(a["explain_row_ids"]).issubset(test.index)
            explained = test.loc[a["explain_row_ids"]]
            assert np.array_equal(model.predict_proba(explained)[:, 1], a["explain_probability"])
            assert np.allclose(a["train_std"], train.std().to_numpy(), rtol=0, atol=1e-12)
            outcomes = pd.read_csv(run/"attack_outcomes.csv", float_precision="round_trip")
            rates = {}
            for (mode, rep), part in outcomes.groupby(["mode", "noise_rep"]):
                assert list(part.row_id) == list(a["attack_row_ids"])
                assert (part.queries == 10).all()
                adv, orders = a[f"adv_{mode}_{rep}"], a[f"orders_{mode}_{rep}"]
                assert all(len(set(o)) == 5 and set(o).issubset(a["mutable"]) for o in orders)
                difference = np.abs(adv-attacked)
                assert (difference <= .5*a["train_std"]+1e-7).all()
                assert not (difference[:, np.setdiff1d(np.arange(test.shape[1]), a["mutable"])] > 1e-10).any()
                assert np.array_equal((difference > 1e-10).sum(axis=1), part.changed_features)
                p = model.predict_proba(pd.DataFrame(adv, columns=test.columns))[:, 1]
                assert np.max(np.abs(p-part.adversarial_probability)) < 1e-7
                assert np.array_equal(p < .5, part.success.astype(bool))
                rates.setdefault(mode, []).append(float((p < .5).mean()))
                total_attacks += len(p)
            averaged = {k: float(np.mean(v)) for k, v in rates.items()}
            for mode, rate in averaged.items():
                assert abs(rate-summary["asr"][mode]) < 1e-15
            assert abs(averaged["raw"]-averaged["pp"]-summary["raw_minus_pp"]) < 1e-15
            policy = json.loads((run/"policy.json").read_text())
            group_indices = [[schema["columns"].index(c) for c in policy["groups"][g]]
                             for g in policy["group_names"]]
            assert sorted(sum(group_indices, [])) == list(range(len(schema["columns"])))
            phi = a["explain_phi"]
            explainer = shap.TreeExplainer(model)
            recomputed_phi = np.asarray(explainer.shap_values(explained), dtype=float)
            assert np.allclose(phi, recomputed_phi, rtol=0, atol=1e-12)
            calibration_phi = np.asarray(explainer.shap_values(train.loc[a["calibration_row_ids"]]), dtype=float)
            calibration_groups = np.column_stack([calibration_phi[:, idx].sum(axis=1) for idx in group_indices])
            assert np.allclose(policy["feature_clip"], np.quantile(np.abs(calibration_phi), .9, axis=0), rtol=0, atol=1e-12)
            assert np.allclose(policy["group_clip"], np.quantile(np.abs(calibration_groups), .9, axis=0), rtol=0, atol=1e-12)
            assert float(explainer.expected_value) == float(a["base"])
            grouped = np.column_stack([phi[:, idx].sum(axis=1) for idx in group_indices])
            clipped = np.clip(grouped, -np.asarray(policy["group_clip"]), policy["group_clip"])
            base = float(a["base"])
            sigmoid = lambda z: 1/(1+np.exp(-z))
            residual = np.abs(a["explain_probability"]-sigmoid(base+phi.sum(axis=1)))
            maximum_raw_residual = max(maximum_raw_residual, float(residual.max()))
            maes = []
            for rep, release in enumerate(a["releases"]):
                top = np.argsort(-release, axis=1, kind="stable")[:, :3]
                mask = np.zeros_like(release, dtype=bool)
                np.put_along_axis(mask, top, True, axis=1)
                noise = np.random.default_rng(100000+seed*101+rep).laplace(
                    0, np.asarray(policy["group_clip"])/policy["eta"], release.shape)
                assert np.allclose(release, clipped+noise, rtol=0, atol=1e-12)
                error = np.abs(a["explain_probability"]-sigmoid(base+np.where(mask, release, 0).sum(axis=1)))
                budget = (np.abs(grouped-clipped).sum(axis=1)+np.abs(release-clipped).sum(axis=1)
                          +np.where(mask, 0, np.abs(release)).sum(axis=1))
                bound = np.minimum(1, residual+budget/4)
                assert (error <= bound+1e-10).all()
                total_bounds += len(phi)
                bound_nonvacuous += int((bound < 1-1e-12).sum())
                maes.append(float(error.mean()))
            assert abs(np.mean(maes)-summary["public_mae"]) < 1e-12
            runs.append(summary)
        describe = lambda values: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
        aggregates[dataset] = {"detection": {k: describe([r["detection"][k] for r in runs])
                                          for k in ("f1", "precision", "recall", "fpr", "auroc")},
            "asr": {k: describe([r["asr"][k] for r in runs]) for k in ("none", "raw", "pp")},
            "raw_minus_pp": describe([r["raw_minus_pp"] for r in runs]),
            "public_mae": describe([r["public_mae"] for r in runs])}
        print("Verified", dataset, flush=True)
    performance = pd.read_csv(OUT/"ledger/scaling_timings.csv")
    assert len(performance) == 21 and performance.valid_accepted.all()
    assert performance.last_record_mutation_rejected.all()
    sizes = json.loads((OUT/"ledger/scaling_summary.json").read_text())
    for row in sizes:
        part = performance[performance.n == row["n"]]
        assert len(part) == 7
        for operation in ("append", "verify"):
            assert abs(part[f"{operation}_seconds"].median()-row[operation]["median_seconds"]) < 1e-12
    replay = pd.read_csv(OUT/"ledger/source_replay.csv")
    assert len(replay) == 500 and replay.decision_matches.all() and replay.sil_matches_stored_release.all()
    assert replay.encoded_max_error.max() == replay.probability_error.max() == 0
    report = {"status": "passed", "models": 20, "recomputed_detection_predictions": total_predictions,
        "recomputed_adversarial_predictions": total_attacks, "pointwise_bound_checks": total_bounds,
        "bound_below_one_cases": bound_nonvacuous, "maximum_csv_probability_rounding_error": maximum_model_error,
        "maximum_raw_shap_probability_residual": maximum_raw_residual,
        "shap_and_training_clipping_recomputed": True, "category_label_consistency": True,
        "source_record_replays": len(replay), "ledger_size_trials": len(performance),
        "independent_aggregates": aggregates, "verifier_sha256": digest(__file__)}
    (OUT/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "independent_aggregates"}, indent=2))
    return report


if __name__ == "__main__":
    check()
