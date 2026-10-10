"""Check recorded evidence independently of the experiment runner."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ledger import file_digest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@unittest.skipUnless((RESULTS / "summary.json").exists(), "Full experiment not run yet")
class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcomes = pd.read_csv(RESULTS / "attack_outcomes.csv")
        cls.summary = json.loads((RESULTS / "summary.json").read_text())

    def test_samples_queries_and_budgets_match(self):
        for (dataset, seed), run in self.outcomes.groupby(["dataset", "seed"]):
            expected = set(run[run["mode"] == "raw"].row_id)
            self.assertEqual(len(expected), 500)
            splits = np.load(RESULTS / dataset / f"seed_{seed}" / "split_ids.npz")
            self.assertTrue(expected.issubset(set(splits["test"])))
            self.assertFalse(expected.intersection(splits["train"]))
            for _, condition in run.groupby(["mode", "noise_rep"]):
                self.assertEqual(set(condition.row_id), expected)
                self.assertEqual(len(condition), 500)
            self.assertTrue((run.queries == 10).all())
            self.assertTrue((run.changed_features <= 5).all())
            self.assertTrue((run.initial_probability >= 0.5).all())
            self.assertTrue((run.category != "normal").all())

    def test_aggregates_agree_with_per_example_outcomes(self):
        for dataset, part in self.outcomes.groupby("dataset"):
            seed_rates = part.groupby(["seed", "mode", "noise_rep"]).success.mean().groupby(["seed", "mode"]).mean().unstack()
            for mode in seed_rates:
                self.assertAlmostEqual(float(seed_rates[mode].mean()), self.summary[dataset]["modes"][mode]["mean"])
            self.assertAlmostEqual(float((seed_rates["raw"]-seed_rates["pp"]).mean()),
                                   self.summary[dataset]["raw_minus_pp"]["mean"])

    def test_running_source_matches_saved_provenance(self):
        config = json.loads((RESULTS / "config.json").read_text())
        for name, expected in config["source_hashes"].items():
            self.assertEqual(file_digest(ROOT / "experiments" / name), expected)


if __name__ == "__main__":
    unittest.main()
