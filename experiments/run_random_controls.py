"""Post-hoc same-record controls, preserving holdout train/test class counts."""

import json

import numpy as np
import pandas as pd

from evaluation import DATASETS
from ledger import file_digest
from run_generalization import OUT, PROTOCOL, SELECTION_SEED, run, save_json


def main():
    marker = OUT / "random_control_protocol.json"
    if marker.exists():
        raise FileExistsError("Do not overwrite recorded controls")
    save_json(marker, {"purpose": "post-hoc split diagnostic, specified after the first holdout runs",
        "selection_seed": SELECTION_SEED, "model_seeds": PROTOCOL["model_seeds"],
        "sampling": "randomize the same retained record pool; preserve per-class train/test counts",
        "feature_and_model_policy": "unchanged from fixed holdouts; no tuning",
        "source_sha256": file_digest(__file__)})
    results = []
    for name, cfg in DATASETS.items():
        held_train = pd.read_parquet(OUT/name/"train_source_records.parquet")
        held_test = pd.read_parquet(OUT/name/"test_source_records.parquet")
        pool = pd.concat([held_train, held_test]).sort_index()
        assert pool.index.is_unique
        rng = np.random.default_rng(SELECTION_SEED)
        test_ids = np.concatenate([rng.choice(pool.index[pool.label == label],
            int((held_test.label == label).sum()), replace=False) for label in (0, 1)])
        test = pool.loc[np.sort(test_ids)]
        train = pool.loc[~pool.index.isin(test_ids)]
        assert train.label.value_counts().to_dict() == held_train.label.value_counts().to_dict()
        assert test.label.value_counts().to_dict() == held_test.label.value_counts().to_dict()
        excluded = ["label", cfg["category"]] + cfg["drop"]
        xtr, xte = train.drop(columns=excluded), test.drop(columns=excluded)
        assert not set(pd.util.hash_pandas_object(xtr, index=False)).intersection(
            pd.util.hash_pandas_object(xte, index=False))
        folder = OUT / (name+"_RANDOM")
        folder.mkdir()
        train.to_parquet(folder/"train_source_records.parquet", index=True)
        test.to_parquet(folder/"test_source_records.parquet", index=True)
        save_json(folder/"split_audit.json", {"parent_holdout": name, "n_train": len(train),
            "n_test": len(test), "train_attack_count": int(train.label.sum()),
            "test_attack_count": int(test.label.sum()),
            "prepared_sha256": {p.name: file_digest(p) for p in folder.glob("*.parquet")}})
        for seed in PROTOCOL["model_seeds"]:
            results.append(run(name+"_RANDOM", seed, (train, test, xtr, xte)))
    save_json(OUT/"random_control_runs.json", results)
    print("Completed same-record controls", len(results), flush=True)


if __name__ == "__main__":
    main()
