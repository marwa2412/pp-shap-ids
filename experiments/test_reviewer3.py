"""Focused tests for the second explanation backbone and timed local path."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import DisclosurePolicy, TrainPreprocessor
from run_request_latency import restore_policy, restore_processor, timed_batch

ROOT = Path(__file__).resolve().parents[1]


class ReviewerThreeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = pd.DataFrame(rng.normal(size=(80, 6)), columns=["sbytes", "dbytes", "dur", "spkts", "dpkts", "loss"])
        self.processor = TrainPreprocessor().fit(self.raw)
        self.encoded = self.processor.transform(self.raw)
        labels = (self.raw.sbytes + 0.6 * self.raw.dur > 0).astype(int)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.model.fit(self.encoded, labels)
        self.background = self.model.steps[0][1].transform(self.encoded.iloc[:30])
        self.explainer = shap.LinearExplainer(self.model.steps[1][1],
            shap.maskers.Independent(self.background, max_samples=30))
        self.explain = lambda x: np.asarray(self.explainer.shap_values(self.model.steps[0][1].transform(x)))

    def test_standardized_linear_shap_identity(self):
        phi = self.explain(self.encoded)
        p = expit(self.explainer.expected_value + phi.sum(axis=1))
        np.testing.assert_allclose(p, self.model.predict_proba(self.encoded)[:, 1], atol=1e-12)

    def test_background_mean_suffices_for_independent_linear_shap(self):
        other = shap.LinearExplainer(self.model.steps[1][1],
            shap.maskers.Independent(self.background.mean(axis=0)[None, :], max_samples=1))
        np.testing.assert_allclose(self.explain(self.encoded),
            other.shap_values(self.model.steps[0][1].transform(self.encoded)), atol=1e-12)

    def test_restored_artifacts_and_local_timing_boundary(self):
        processor = restore_processor(self.processor.snapshot())
        policy = DisclosurePolicy(self.encoded.columns, self.explain(self.encoded.iloc[:30]))
        restored = restore_policy(policy.snapshot(), list(self.encoded.columns))
        np.testing.assert_array_equal(policy.release(self.explain(self.encoded), "pp", 5),
                                      restored.release(self.explain(self.encoded), "pp", 5))
        context = processor, self.model, self.explain, restored, {"model": "test", "policy": "test", "schema": "test"}
        p, times = timed_batch(self.raw.iloc[:4], context, 5)
        np.testing.assert_allclose(p, self.model.predict_proba(self.encoded.iloc[:4])[:, 1], atol=1e-12)
        self.assertAlmostEqual(times["total_ms"], sum(v for k, v in times.items() if k != "total_ms"), places=9)
        self.assertTrue(all(v >= 0 for v in times.values()))

    @unittest.skipUnless((ROOT / "results_reviewer3/verification.json").exists(),
                         "Full LR/latency evidence not present; run the reproduction workflow")
    def test_saved_evidence_keeps_adverse_run_and_all_timing_conditions(self):
        report = json.loads((ROOT / "results_reviewer3/verification.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["models"], 2)
        self.assertEqual(report["adversarial_predictions"], 25000)
        adverse = next(r for r in report["backbone_results"]["TON_IOT"]["runs"] if r["seed"] == 44)
        self.assertLess(adverse["drop"], 0)
        self.assertEqual(report["latency"]["batches"], 1200)
        self.assertEqual(len(report["latency"]["summary"]), 12)


if __name__ == "__main__":
    unittest.main()
