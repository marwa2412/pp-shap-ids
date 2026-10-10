import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (DisclosurePolicy, TrainPreprocessor, attack_order,
                        paired_bootstrap, query_attack, sil_render)
from ledger import append, checkpoint, digest, verify, verify_artifacts, file_digest


class PreprocessingTests(unittest.TestCase):
    def test_train_only_statistics_and_unseen_category(self):
        training = pd.DataFrame({"bytes": [1.0, 3.0, np.nan], "proto": ["tcp", "tcp", "udp"]})
        processor = TrainPreprocessor().fit(training)
        before = copy.deepcopy(processor.snapshot())
        transformed = processor.transform(pd.DataFrame({"bytes": [np.nan, 9999], "proto": ["tcp", "new"]}))
        self.assertEqual(transformed.iloc[0]["bytes"], 2)
        self.assertAlmostEqual(transformed.iloc[0]["proto"], 2/3)
        self.assertEqual(transformed.iloc[1]["proto"], 0)
        self.assertEqual(before, processor.snapshot())

    def test_noise_is_reproducible_with_frozen_calibration(self):
        policy = DisclosurePolicy(["sbytes", "dbytes", "dur", "ct_ftp_cmd"], np.ones((20, 4)))
        before = copy.deepcopy(policy.snapshot())
        values = np.arange(40).reshape(10, 4).astype(float)
        np.testing.assert_array_equal(policy.release(values, "pp", 9), policy.release(values, "pp", 9))
        self.assertFalse(np.array_equal(policy.release(values, "pp", 9), policy.release(values, "pp", 10)))
        self.assertEqual(before, policy.snapshot())


class AttackTests(unittest.TestCase):
    def test_protected_order_uses_released_groups_and_equal_coordinate_count(self):
        columns = ["sbytes", "dbytes", "dur", "rtt", "Spkts", "Dpkts", "sport", "dns_qtype"]
        policy = DisclosurePolicy(columns, np.ones((20, len(columns))))
        mutable = np.arange(6)
        scores = np.zeros((4, len(policy.group_names)))
        orders = attack_order(scores, "pp", policy, mutable, 4, 7)
        self.assertEqual(orders.shape, (4, 5))
        for row in orders:
            self.assertEqual(len(set(row)), 5)
            self.assertTrue(set(row).issubset(set(mutable)))
        np.testing.assert_array_equal(orders, attack_order(scores, "pp", policy, mutable, 4, 7))

    def test_attack_budget_bounds_and_immutable_features(self):
        class ToyModel:
            def __init__(self):
                self.evaluated_rows = 0
            def predict_proba(self, X):
                self.evaluated_rows += len(X)
                x = np.asarray(X)
                p = 1/(1+np.exp(-(x[:, :5].sum(axis=1)-3)))
                return np.column_stack([1-p, p])
        X = pd.DataFrame(np.ones((8, 7)), columns=list("abcdefg"))
        model = ToyModel()
        order = np.tile(np.arange(5), (8, 1))
        success, probabilities, modified = query_attack(model, X, order, np.ones(7), np.zeros(7), np.ones(7)*2)
        self.assertEqual(model.evaluated_rows, 8*11)
        self.assertTrue(success.all())
        np.testing.assert_array_equal(modified[:, 5:], X.to_numpy()[:, 5:])
        self.assertTrue((np.abs(modified-X.to_numpy()) <= 0.5).all())
        self.assertTrue((probabilities < 0.5).all())

    def test_paired_bootstrap_preserves_pairs(self):
        self.assertEqual(paired_bootstrap([0, 1, 0, 1], [0, 1, 0, 1]), [0, 0])

    def test_sil_exposes_three_groups_and_uncalibrated_score(self):
        packet = sil_render(0.9999, ["a", "b", "c", "d"], [4, 1, 3, 2])
        self.assertEqual([x["group"] for x in packet["items"]], ["a", "c", "d"])
        self.assertFalse(packet["calibrated"])
        self.assertIn("Uncalibrated", packet["text"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.entries = []
        for i in range(10):
            append(self.entries, {"value": i, "role": "tier1"}, "test-log")
        self.trusted = checkpoint(self.entries, "test-log")

    def test_intact_and_valid_extension(self):
        self.assertTrue(verify(self.entries, self.trusted)[0])
        extended = copy.deepcopy(self.entries)
        append(extended, {"value": 11}, "test-log")
        self.assertTrue(verify(extended, checkpoint(extended, "test-log"))[0])

    def test_payload_change_and_full_rewrite_detected(self):
        modified = copy.deepcopy(self.entries)
        modified[3]["payload"]["role"] = "admin"
        self.assertFalse(verify(modified, self.trusted)[0])
        rebuilt = []
        for entry in modified:
            append(rebuilt, entry["payload"], "test-log")
        self.assertEqual(verify(rebuilt, self.trusted), (False, "trusted_head"))
        # Compromise of the checkpoint is deliberately outside the threat model.
        self.assertTrue(verify(rebuilt, checkpoint(rebuilt, "test-log"))[0])

    def test_truncation_deletion_insertion_reordering(self):
        cases = [self.entries[:-1], self.entries[:3]+self.entries[4:],
                 self.entries[:3]+[self.entries[3]]+self.entries[3:]]
        swapped = copy.deepcopy(self.entries)
        swapped[2], swapped[3] = swapped[3], swapped[2]
        for entries in cases + [swapped]:
            self.assertFalse(verify(entries, self.trusted)[0])

    def test_payload_detached_and_canonical_keys(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        payload = {"nested": [1]}
        entries = []
        append(entries, payload, "separate")
        payload["nested"].append(2)
        self.assertEqual(entries[0]["payload"]["nested"], [1])

    def test_missing_or_substituted_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.json"
            path.write_text("original")
            payload = {"artifacts": {"model": {"path": "model.json", "sha256": file_digest(path)}}}
            self.assertTrue(verify_artifacts(payload, root)[0])
            path.write_text("changed")
            self.assertFalse(verify_artifacts(payload, root)[0])
            path.unlink()
            self.assertFalse(verify_artifacts(payload, root)[0])


if __name__ == "__main__":
    unittest.main()
