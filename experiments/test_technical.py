"""Focused guards for the added holdout and mathematical checks."""

import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from evaluation import DisclosurePolicy
from run_generalization import clean_partition_pair, endpoint_pairs, reconstruction_bound


class BoundTests(unittest.TestCase):
    def test_random_signed_releases_satisfy_bound(self):
        rng = np.random.default_rng(8)
        phi = rng.normal(size=(1000, 6))
        grouped = phi.reshape(1000, 3, 2).sum(axis=2)
        clipped = np.clip(grouped, -0.3, 0.3)
        release = clipped+rng.laplace(size=grouped.shape)
        probability = expit(-0.2+phi.sum(axis=1))
        result = reconstruction_bound(-0.2, phi, grouped, clipped, release, probability)
        self.assertTrue((result["error"] <= result["bound"]+1e-12).all())

    def test_omitted_negative_group_is_accounted_for(self):
        phi = np.array([[1., 1., 1., -10.]])
        result = reconstruction_bound(0., phi, phi, phi, phi, expit(phi.sum(axis=1)))
        self.assertEqual(result["omitted_l1"][0], 10)
        self.assertGreater(result["error"][0], 0.9)
        self.assertEqual(result["bound"][0], 1.)

    def test_numerical_residual_is_not_ignored(self):
        phi = np.zeros((1, 3))
        result = reconstruction_bound(0., phi, phi, phi, phi, np.array([0.7]))
        self.assertAlmostEqual(result["bound"][0], 0.2)
        self.assertAlmostEqual(result["error"][0], 0.2)

    def test_semantic_groups_form_partition(self):
        columns = ["src_bytes", "src_port", "duration", "proto", "dns_query", "other"]
        policy = DisclosurePolicy(columns, np.zeros((10, len(columns))))
        self.assertEqual(sorted(np.concatenate(policy.group_indices).tolist()), list(range(len(columns))))


class SplitTests(unittest.TestCase):
    def test_pair_is_undirected(self):
        frame = pd.DataFrame({"src_ip": ["a", "b", "a"], "dst_ip": ["b", "a", "c"]})
        pairs = endpoint_pairs(frame)
        self.assertEqual(pairs[0], pairs[1])
        self.assertNotEqual(pairs[0], pairs[2])

    def test_predictor_overlap_is_removed_without_ips(self):
        frame = pd.DataFrame({"src_ip": ["a"]*6, "dst_ip": ["b"]*6,
            "feature": [0., 1., 1., 2., 3., 3.], "label": [0, 1, 1, 0, 1, 1],
            "type": ["normal", "attack", "attack", "normal", "attack", "attack"]})
        _, _, xtr, xte, audit = clean_partition_pair(frame.iloc[:2], frame.iloc[2:], "TON_IOT")
        self.assertEqual(audit["test_train_predictor_overlaps_removed"], 1)
        self.assertEqual(audit["test_duplicates_removed"], 1)
        self.assertEqual(list(xtr.columns), ["feature"])
        self.assertEqual(set(xte.feature), {2., 3.})

    def test_conflicting_labels_stop_fit(self):
        frame = pd.DataFrame({"src_ip": ["a"]*4, "dst_ip": ["b"]*4,
            "feature": [0., 1., 0., 2.], "label": [0, 1, 1, 0],
            "type": ["normal", "attack", "attack", "normal"]})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean_partition_pair(frame.iloc[:2], frame.iloc[2:], "TON_IOT")


if __name__ == "__main__":
    unittest.main()
