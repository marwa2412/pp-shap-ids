import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import export_numerical_results as exporter
from scripts import reproduce


def sample_outcomes():
    rows = []
    for seed, raw, protected in [(42, [1, 1], [[1, 0], [0, 0]]), (43, [0], [[0], [0]])]:
        for mode, repetitions in [("raw", [raw]), ("pp", protected)]:
            for repetition, values in enumerate(repetitions):
                for value in values:
                    rows.append({"dataset": "synthetic", "category": "fixture", "seed": seed,
                                 "mode": mode, "noise_rep": repetition, "success": value})
    return pd.DataFrame(rows)


def write_fixture(root):
    for key, relative in exporter.SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "passed", "fixture": key, "value": 0.12345678901234567}))
    sample_outcomes().to_csv(root / "results/attack_outcomes.csv", index=False)


class NumericalExportTests(unittest.TestCase):
    def test_repetitions_do_not_inflate_attempts_and_seeds_have_equal_weight(self):
        result = exporter.category_summary(sample_outcomes()).iloc[0]
        self.assertEqual(result.attack_attempts_across_seeds, 3)
        self.assertEqual(result.seeds_with_cases, 2)
        self.assertEqual(result.raw_mean, 0.5)
        self.assertEqual(result.pp_mean, 0.125)
        self.assertEqual(result.mean_paired_seed_drop, 0.375)

    def test_incomplete_or_invalid_outcomes_are_rejected(self):
        original = sample_outcomes()
        invalid_success = original.copy()
        invalid_success.loc[0, "success"] = 2
        cases = [original.iloc[:0], original.drop(columns=["success"]), invalid_success,
                 original[original["mode"] == "raw"],
                 original[~((original["mode"] == "pp") & (original["seed"] == 43))]]
        for frame in cases:
            with self.subTest(rows=len(frame), columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    exporter.category_summary(frame)

    def test_export_preserves_numbers_and_sources_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            report = exporter.export_results(root)
            output = root / "numerical_results"
            self.assertEqual(set(p.name for p in output.iterdir()), set(report["output_files"]))
            payload = json.loads((output / "numerical_summary.json").read_text())
            for key, relative in exporter.SOURCES.items():
                raw = (root / relative).read_bytes()
                self.assertEqual(payload["results"][key], json.loads(raw))
                self.assertEqual(payload["source_sha256"][relative], hashlib.sha256(raw).hexdigest())
            csv = root / "results/attack_outcomes.csv"
            self.assertEqual(payload["source_sha256"]["results/attack_outcomes.csv"],
                             hashlib.sha256(csv.read_bytes()).hexdigest())
            expected = exporter.category_summary(sample_outcomes())
            pd.testing.assert_frame_equal(pd.read_csv(output / "category_summary.csv"), expected)
            with self.assertRaises(FileExistsError):
                exporter.export_results(root)

    def test_failed_or_missing_verification_does_not_create_output(self):
        for status in ("failed", None):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_fixture(root)
                report = root / exporter.SOURCES["structured_verification"]
                if status is None:
                    report.unlink()
                else:
                    report.write_text(json.dumps({"status": status}))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    exporter.export_results(root)
                self.assertFalse((root / "numerical_results").exists())

    def test_nonfinite_results_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            (root / exporter.SOURCES["initial"]).write_text('{"invalid": NaN}')
            with self.assertRaises(ValueError):
                exporter.export_results(root)
            self.assertFalse((root / "numerical_results").exists())

    def test_results_stage_runs_without_manuscript_production(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            reproduce.create_workspace(root, "paper-protocol")
            write_fixture(root)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/reproduce.py"), "--work-dir", str(root), "--stage", "results"],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            record = json.loads((root / "reproduction.json").read_text())
            self.assertEqual(record["stages"], [{"stage": "results", "status": "passed"}])
            self.assertTrue((root / "numerical_results/category_summary.csv").is_file())
            self.assertFalse(any(p.suffix in {".tex", ".pdf", ".png", ".svg"} for p in root.rglob("*")))

    def test_exporter_changes_are_detected_in_existing_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            reproduce.create_workspace(root, "paper-protocol")
            (root / "experiments/export_numerical_results.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Workspace code changed"):
                reproduce.create_workspace(root, "paper-protocol")


if __name__ == "__main__":
    unittest.main()
