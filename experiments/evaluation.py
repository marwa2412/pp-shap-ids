"""Shared, training-only preprocessing and matched feature-space attacks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.special import expit
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

DATASETS = {
    "UNSW_NB15": {"folder": "Mouwiya___unsw-nb15", "revision": "8ad6b19373f143086a66f52b0d6035a6959d83a9",
                  "category": "attack_cat", "drop": ["srcip", "dstip", "Stime", "Ltime"]},
    "TON_IOT": {"folder": "codymlewis___ton_io_t_network", "revision": "b953d9388d2db3bc392ea7150476e4f219a2a1cb",
                "category": "type", "drop": ["src_ip", "dst_ip"]},
}


def load_data(name, max_rows=160000, cache=None):
    cfg = DATASETS[name]
    root = Path(cache) if cache else Path.home() / ".cache/huggingface/datasets"
    paths = sorted((root / cfg["folder"]).glob(f"**/{cfg['revision']}/*train*.arrow"))
    if not paths:
        raise FileNotFoundError(f"Cached Arrow data missing for {name}: {root}")
    batches, count = [], 0
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            for batch in ipc.open_stream(source):
                part = batch.slice(0, min(len(batch), max_rows - count))
                batches.append(part)
                count += len(part)
                if count >= max_rows:
                    break
        if count >= max_rows:
            break
    df = pa.Table.from_batches(batches).to_pandas()
    y = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels")
    categories = df[cfg["category"]].fillna("normal").astype(str).str.strip()
    categories = categories.mask(y == 0, "normal")
    X = df.drop(columns=["label", cfg["category"]] + cfg["drop"], errors="ignore")
    hashes = pd.util.hash_pandas_object(X, index=False)
    conflicts = pd.DataFrame({"hash": hashes, "label": y}).groupby("hash")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting labels for identical predictors require a group split")
    keep = ~X.duplicated()
    metadata = {"dataset": name, "repository_revision": cfg["revision"],
                "input_rows": len(df), "unique_rows": int(keep.sum()),
                "duplicates_removed": int((~keep).sum()),
                "attack_ratio_before": float(y.mean()), "attack_ratio_after": float(y[keep].mean()),
                "source_files": [str(p) for p in paths], "selection": "first_n_repository_training_rows",
                "dropped_columns": ["label", cfg["category"]] + cfg["drop"]}
    return X[keep], y[keep], categories[keep], metadata


class TrainPreprocessor:
    def fit(self, X):
        self.columns = list(X.columns)
        self.numeric, self.categorical, self.medians, self.frequencies = [], [], {}, {}
        for c in self.columns:
            values = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if pd.api.types.is_numeric_dtype(X[c]) or values.isna().mean() < 0.2:
                self.numeric.append(c)
                median = values.median()
                self.medians[c] = 0.0 if pd.isna(median) else float(median)
            else:
                self.categorical.append(c)
                self.frequencies[c] = (X[c].fillna("__MISSING__").astype(str).value_counts() / len(X)).to_dict()
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            if c in self.numeric:
                out[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.medians[c])
            else:
                out[c] = X[c].fillna("__MISSING__").astype(str).map(self.frequencies[c]).fillna(0.0)
        return out.astype(float)

    def snapshot(self):
        return {"columns": self.columns, "numeric": self.numeric, "categorical": self.categorical,
                "medians": self.medians, "frequencies": self.frequencies}


def group_name(column):
    c = column.lower()
    if any(k in c for k in ("byte", "load", "meansz", "body_len")):
        return "traffic_volume"
    if "port" in c:
        return "port_behavior"
    if any(k in c for k in ("dur", "time", "jit", "intpkt", "rtt", "synack", "ackdat")):
        return "temporal_statistics"
    if any(k in c for k in ("pkts", "pkt", "loss")):
        return "packet_statistics"
    if any(k in c for k in ("proto", "state", "service", "ttl")):
        return "protocol_state"
    if any(k in c for k in ("dns", "http", "ssl")):
        return "application_context"
    if any(k in c for k in ("ct_", "ftp", "login", "weird")):
        return "behavioral_context"
    return "other"


class DisclosurePolicy:
    def __init__(self, columns, calibration_shap, eta=1.0):
        self.columns = list(columns)
        self.group_names = sorted(set(map(group_name, columns)))
        self.group_indices = [np.array([i for i, c in enumerate(columns) if group_name(c) == g])
                              for g in self.group_names]
        self.feature_clip = np.quantile(np.abs(calibration_shap), 0.9, axis=0)
        self.group_clip = np.quantile(np.abs(self.aggregate(calibration_shap)), 0.9, axis=0)
        self.eta = eta

    def aggregate(self, values):
        return np.column_stack([values[:, idx].sum(axis=1) for idx in self.group_indices])

    def release(self, values, mode, noise_seed=0):
        if mode == "raw":
            return values.copy()
        grouped = mode in ("group", "pp")
        scores = self.aggregate(values) if grouped else values.copy()
        if mode == "group":
            return scores
        thresholds = self.group_clip if grouped else self.feature_clip
        scores = np.clip(scores, -thresholds, thresholds)
        if mode in ("clip_noise", "pp"):
            rng = np.random.default_rng(noise_seed)
            scores += rng.laplace(0, thresholds / self.eta, scores.shape)
        return scores

    def snapshot(self):
        return {"group_names": self.group_names,
                "groups": {g: [self.columns[i] for i in idx]
                           for g, idx in zip(self.group_names, self.group_indices)},
                "feature_clip": self.feature_clip.tolist(), "group_clip": self.group_clip.tolist(),
                "eta": self.eta, "calibration_quantile": 0.9, "public_groups": 3,
                "formal_differential_privacy_guarantee": False}


def numeric_attack_indices(X_train, preprocessor):
    # Encoded categories, ports and control-state identifiers remain immutable.
    idx = []
    for i, c in enumerate(X_train.columns):
        lower = c.lower()
        excluded = any(k in lower for k in ("port", "ttl", "state", "proto", "service", "tcpb",
                                            "status_code", "qclass", "qtype", "rcode")) or lower.startswith("is_")
        if c in preprocessor.numeric and not excluded and X_train[c].nunique() > 2 and X_train[c].std() > 0:
            idx.append(i)
    if len(idx) < 5:
        raise ValueError("At least five mutable numeric features required")
    return np.array(idx)


def attack_order(scores, mode, policy, mutable, n, seed, k=5):
    rng = np.random.default_rng(seed)
    orders = []
    mutable_set = set(mutable)
    for row in range(n):
        if mode == "none":
            chosen = list(rng.permutation(mutable)[:k])
        elif mode in ("raw", "clip", "clip_noise"):
            chosen = list(mutable[np.argsort(-scores[row, mutable], kind="stable")[:k]])
        else:
            # Only the top-three released groups guide the protected attacker.
            top_groups = np.argsort(-scores[row], kind="stable")[:3]
            pools = [list(rng.permutation([i for i in policy.group_indices[g] if i in mutable_set]))
                     for g in top_groups]
            chosen = []
            while any(pools) and len(chosen) < k:
                for pool in pools:
                    if pool and len(chosen) < k:
                        chosen.append(pool.pop())
            if len(chosen) < k:
                remaining = [i for i in mutable if i not in chosen]
                chosen.extend(rng.permutation(remaining)[:k-len(chosen)])
        orders.append(chosen)
    return np.asarray(orders, dtype=int)


def query_attack(model, X, orders, train_std, train_min, train_max, delta=0.5):
    """Five coordinates, two score queries each; no early stopping or SHAP refresh."""
    original = X.to_numpy(copy=True)
    current = original.copy()
    probabilities = model.predict_proba(X)[:, 1]
    rows = np.arange(len(X))
    for step in range(orders.shape[1]):
        features = orders[:, step]
        candidates = []
        # The original value stays feasible even outside the training range.
        low = np.minimum(train_min[features], original[rows, features])
        high = np.maximum(train_max[features], original[rows, features])
        for direction in (-1, 1):
            candidate = current.copy()
            proposed = original[rows, features] + direction * delta * train_std[features]
            candidate[rows, features] = np.clip(proposed, low, high)
            candidates.append(candidate)
        probs = model.predict_proba(pd.DataFrame(np.vstack(candidates), columns=X.columns))[:, 1].reshape(2, len(X))
        choice = np.argmin(probs, axis=0)
        best_prob = probs[choice, rows]
        better = best_prob < probabilities
        for direction in (0, 1):
            mask = better & (choice == direction)
            current[mask] = candidates[direction][mask]
        probabilities = np.minimum(probabilities, best_prob)
    return probabilities < 0.5, probabilities, current


def detection_metrics(y, probabilities):
    pred = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "fpr": float(fp / (fp + tn)),
            "auroc": float(roc_auc_score(y, probabilities)), "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn)}


def reconstruction_mae(probabilities, base, values):
    return float(np.abs(probabilities - expit(base + values.sum(axis=1))).mean())


def paired_bootstrap(raw, protected, seed=42, repeats=2000):
    difference = np.asarray(raw, dtype=float) - np.asarray(protected, dtype=float)
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference), size=(repeats, len(difference)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def sil_render(probability, group_names, scores):
    top = np.argsort(-np.asarray(scores), kind="stable")[:3]
    items = [{"group": group_names[i], "released_score": float(scores[i])} for i in top]
    names = "; ".join(x["group"].replace("_", " ") for x in items)
    text = (f"Uncalibrated detector score: {probability:.4f}. Released signals: {names}. "
            "Review supporting telemetry before containment; perturbed attribution scores may change ranking.")
    return {"items": items, "text": text, "calibrated": False}
