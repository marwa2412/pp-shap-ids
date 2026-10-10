"""Verify LR predictions analytically and recompute all new reported summaries."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from verify_technical import digest, encode, manual_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_reviewer3"


def summarize(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def check_backbone():
    runtime = json.loads((OUT / "runtime.json").read_text())
    for name, value in runtime["source_sha256"].items():
        assert digest(ROOT / "experiments" / name) == value
    counts = {"models": 0, "detection_predictions": 0, "adversarial_predictions": 0,
              "bound_checks": 0, "bound_below_one": 0}
    results = {}
    for name in ("UNSW_NB15", "TON_IOT"):
        source, folder = ROOT / "results_technical" / name, OUT / name
        audit = json.loads((source / "split_audit.json").read_text())
        for filename, value in audit["prepared_sha256"].items():
            assert digest(source / filename) == value
        tr = pd.read_parquet(source / "train_source_records.parquet")
        te = pd.read_parquet(source / "test_source_records.parquet")
        assert not set(tr.index).intersection(te.index)
        schema = json.loads((folder / "schema.json").read_text())
        assert schema == json.loads((source / "seed_42/schema.json").read_text())
        train, test = encode(tr, schema), encode(te, schema)
        model = joblib.load(folder / "model.joblib")
        scaler, linear = model.steps[0][1], model.steps[1][1]
        assert list(linear.classes_) == [0, 1]
        assert linear.solver == "lbfgs" and linear.C == 1 and linear.n_iter_.max() < 10000
        np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy(), rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(scaler.var_, train.var(ddof=0).to_numpy(), rtol=1e-12, atol=1e-10)
        beta, intercept = linear.coef_[0], float(linear.intercept_[0])
        scaled = lambda x: (np.asarray(x) - scaler.mean_) / scaler.scale_
        predict = lambda x: expit(scaled(x) @ beta + intercept)
        probabilities = predict(test)
        stored = pd.read_csv(folder / "test_predictions.csv", float_precision="round_trip")
        assert list(stored.row_id) == list(te.index) and np.array_equal(stored.label, te.label)
        np.testing.assert_allclose(probabilities, stored.probability, atol=1e-12, rtol=0)
        np.testing.assert_allclose(probabilities, model.predict_proba(test)[:, 1], atol=1e-12, rtol=0)
        metrics = manual_metrics(te.label, probabilities)
        detected = json.loads((folder / "detection.json").read_text())
        assert digest(folder / "model.joblib") == detected["model_sha256"]
        for key, value in metrics.items():
            assert abs(value - detected[key]) < 1e-12
        counts["models"] += 1
        counts["detection_predictions"] += len(test)
        runs = []
        for seed in (42, 43, 44, 45, 46):
            run = folder / f"seed_{seed}"
            arrays = np.load(run / "verification_arrays.npz")
            cal_ids, ex_ids, attack_ids = [arrays[k] for k in
                ("calibration_row_ids", "explain_row_ids", "attack_row_ids")]
            assert set(cal_ids).issubset(tr.index)
            assert set(ex_ids).issubset(te.index) and set(attack_ids).issubset(te.index)
            assert len(set(attack_ids)) == 500 and (te.loc[attack_ids].label == 1).all()
            assert (predict(test.loc[attack_ids]) >= 0.5).all()
            mean = scaled(train.loc[cal_ids]).mean(axis=0)
            cal_phi = (scaled(train.loc[cal_ids]) - mean) * beta
            phi = (scaled(test.loc[ex_ids]) - mean) * beta
            attack_phi = (scaled(test.loc[attack_ids]) - mean) * beta
            base = intercept + mean @ beta
            for key, value in (("background_mean", mean), ("explain_phi", phi),
                               ("attack_phi", attack_phi), ("base", base)):
                np.testing.assert_allclose(value, arrays[key], atol=1e-10, rtol=1e-12)
            policy = json.loads((run / "policy.json").read_text())
            members = [schema["columns"].index(c) for g in policy["group_names"] for c in policy["groups"][g]]
            assert sorted(members) == list(range(len(schema["columns"])))
            groups = [np.array([schema["columns"].index(c) for c in policy["groups"][g]])
                      for g in policy["group_names"]]
            grouped_cal = np.column_stack([cal_phi[:, group].sum(axis=1) for group in groups])
            thresholds = np.quantile(np.abs(grouped_cal), 0.9, axis=0)
            np.testing.assert_allclose(thresholds, policy["group_clip"], atol=1e-10)
            outcomes = pd.read_csv(run / "attack_outcomes.csv", float_precision="round_trip")
            original = test.loc[attack_ids].to_numpy()
            std, low, high = [v.to_numpy() for v in (train.std(), train.min(), train.max())]
            for key, value in (("train_std", std), ("train_min", low), ("train_max", high), ("original_attack", original)):
                np.testing.assert_allclose(arrays[key], value, atol=1e-10)
            rates = {"none": [], "raw": [], "pp": []}
            for mode, rep in [("none", 0), ("raw", 0)] + [("pp", r) for r in range(3)]:
                part = outcomes[(outcomes["mode"] == mode) & (outcomes.noise_rep == rep)]
                assert list(part.row_id) == list(attack_ids) and (part.queries == 10).all()
                adv, orders = arrays[f"adv_{mode}_{rep}"], arrays[f"orders_{mode}_{rep}"]
                assert orders.shape == (500, 5)
                assert all(len(set(order)) == 5 and set(order).issubset(arrays["mutable"]) for order in orders)
                # Independently replay each two-direction query using the explicit linear predictor.
                replay = original.copy()
                row = np.arange(len(original))
                for coordinates in orders.T:
                    lower = np.minimum(low[coordinates], original[row, coordinates])
                    upper = np.maximum(high[coordinates], original[row, coordinates])
                    minus, plus = replay.copy(), replay.copy()
                    minus[row, coordinates] = np.clip(original[row, coordinates] - 0.5 * std[coordinates], lower, upper)
                    plus[row, coordinates] = np.clip(original[row, coordinates] + 0.5 * std[coordinates], lower, upper)
                    pminus, pplus, pcurrent = predict(minus), predict(plus), predict(replay)
                    choose_minus = (pminus <= pplus) & (pminus < pcurrent)
                    choose_plus = (pplus < pminus) & (pplus < pcurrent)
                    replay[choose_minus], replay[choose_plus] = minus[choose_minus], plus[choose_plus]
                np.testing.assert_allclose(adv, replay, atol=1e-10, rtol=1e-12)
                p = predict(adv)
                np.testing.assert_allclose(p, part.adversarial_probability, atol=1e-12, rtol=0)
                assert np.array_equal(p < 0.5, part.success)
                assert (np.abs(adv-original) <= 0.5*std + 1e-7).all()
                rates[mode].append(float((p < 0.5).mean()))
                counts["adversarial_predictions"] += len(p)
            grouped = np.column_stack([phi[:, group].sum(axis=1) for group in groups])
            clipped = np.clip(grouped, -thresholds, thresholds)
            error_means = []
            saved_bounds = pd.read_csv(run / "bound_checks.csv", float_precision="round_trip")
            for rep in range(3):
                released = clipped + np.random.default_rng(100000 + seed*101 + rep).laplace(0, thresholds, clipped.shape)
                np.testing.assert_allclose(released, arrays["releases"][rep], atol=1e-10)
                keep = np.zeros_like(released, dtype=bool)
                np.put_along_axis(keep, np.argsort(-released, axis=1, kind="stable")[:, :3], True, axis=1)
                error = np.abs(predict(test.loc[ex_ids]) - expit(base + np.where(keep, released, 0).sum(axis=1)))
                residual = np.abs(predict(test.loc[ex_ids]) - expit(base + phi.sum(axis=1)))
                terms = np.abs(grouped-clipped).sum(axis=1) + np.abs(released-clipped).sum(axis=1)
                terms += np.where(keep, 0, np.abs(released)).sum(axis=1)
                bound = np.minimum(1, residual + terms/4)
                assert (error <= bound + 1e-10).all()
                part = saved_bounds[saved_bounds.noise_rep == rep]
                np.testing.assert_allclose(error, part.error, atol=1e-10)
                np.testing.assert_allclose(bound, part.bound, atol=1e-10)
                error_means.append(float(error.mean()))
                counts["bound_checks"] += len(error)
                counts["bound_below_one"] += int((bound < 1).sum())
            run_summary = {mode: float(np.mean(values)) for mode, values in rates.items()}
            run_summary["drop"] = run_summary["raw"] - run_summary["pp"]
            run_summary["public_mae"] = float(np.mean(error_means))
            run_summary["seed"] = seed
            runs.append(run_summary)
        results[name] = {"detection": metrics, "runs": runs,
            "aggregates": {key: summarize([run[key] for run in runs])
                           for key in ("none", "raw", "pp", "drop", "public_mae")}}
        print(name, "LR verified", results[name]["aggregates"], flush=True)
    return counts, results


def check_latency():
    folder = OUT / "latency"
    protocol = json.loads((folder / "protocol.json").read_text())
    for name, value in protocol["source_sha256"].items():
        assert digest(ROOT / "experiments" / name) == value
    rows = pd.read_csv(folder / "timings.csv", float_precision="round_trip")
    assert len(rows) == 1200 and rows.log_verified.all()
    assert (rows.max_prediction_error <= 3e-8).all()
    stages = ["preprocess", "predict", "shap", "release_render", "ledger_append"]
    np.testing.assert_allclose(rows[[s + "_ms" for s in stages]].sum(axis=1), rows.total_ms, atol=1e-9)
    assert (rows[[s + "_ms" for s in stages]] >= 0).all().all()
    summary = []
    for (dataset, model, size), part in rows.groupby(["dataset", "model", "batch_size"]):
        assert len(part) == 100 and set(part["repeat"]) == set(range(100))
        test_ids = set(pd.read_parquet(ROOT / "results_technical" / dataset / "test_source_records.parquet").index)
        assert all(len(json.loads(ids)) == size and set(json.loads(ids)).issubset(test_ids) for ids in part.row_ids)
        row = {"dataset": dataset, "model": model, "batch_size": int(size), "repeats": 100}
        for stage in [*stages, "total"]:
            row[stage + "_p50_ms"] = float(np.quantile(part[stage + "_ms"], 0.5))
            row[stage + "_p95_ms"] = float(np.quantile(part[stage + "_ms"], 0.95))
        row["amortized_p50_ms"] = row["total_p50_ms"] / size
        row["serial_equivalent_records_per_second"] = 1000 * size / row["total_p50_ms"]
        summary.append(row)
    assert len(summary) == 12
    return {"batches": 1200, "record_predictions": int(rows.batch_size.sum()), "summary": summary}


def main():
    counts, results = check_backbone()
    latency = check_latency()
    report = {"status": "passed", **counts, "backbone_results": results, "latency": latency,
              "verifier_sha256": digest(Path(__file__))}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("VERIFIED", counts, "timing batches", latency["batches"], flush=True)


if __name__ == "__main__":
    main()
