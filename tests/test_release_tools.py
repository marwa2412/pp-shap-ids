import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_release, compare_reference, prepare_data, reproduce


class DataPreparationTests(unittest.TestCase):
    def test_atomic_import_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "source", root / "cache/input"
            source.write_bytes(b"pinned fixture")
            entry = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "size_bytes": source.stat().st_size, "path": "input"}
            prepare_data.acquire(entry, target, source)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            prepare_data.acquire(entry, target, source)
            target.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Existing source"):
                prepare_data.acquire(entry, target, source)
            self.assertEqual(target.read_bytes(), b"modified")

    def test_failed_download_never_commits_partial_data(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "input"
            entry = {"url": "https://example.invalid/input", "size_bytes": 3,
                     "sha256": hashlib.sha256(b"abc").hexdigest(), "path": "input"}
            with patch.object(prepare_data.urllib.request, "urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    prepare_data.acquire(entry, target)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_logical_hash_ignores_batching_but_detects_row_order(self):
        with tempfile.TemporaryDirectory() as temp:
            table = pa.Table.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "y"]}), preserve_index=False)
            paths = [Path(temp) / str(i) for i in range(3)]
            for i, size in enumerate([1, 3, 3]):
                selected = table if i < 2 else table.take([2, 1, 0])
                with pa.OSFile(str(paths[i]), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(selected, max_chunksize=size)
            self.assertEqual(prepare_data.cache_summary([paths[0]]), prepare_data.cache_summary([paths[1]]))
            self.assertNotEqual(prepare_data.cache_summary([paths[0]]), prepare_data.cache_summary([paths[2]]))


class WorkspaceTests(unittest.TestCase):
    def test_paper_core_hashes_are_unchanged(self):
        manifest = json.loads((ROOT / "paper_code_manifest.json").read_text())
        self.assertEqual(len(manifest["sha256"]), 11)
        self.assertEqual(len(reproduce.frozen_hashes()), 12)

    def test_unrecognized_existing_directory_is_never_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(FileExistsError):
                reproduce.create_workspace(path, "paper-protocol")
            self.assertEqual(list(path.iterdir()), [])

    def test_profile_and_tampering_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            first = reproduce.create_workspace(path, "paper-protocol")
            self.assertEqual(first, reproduce.create_workspace(path, "paper-protocol"))
            with self.assertRaisesRegex(ValueError, "profile/source"):
                reproduce.create_workspace(path, "smoke-not-paper-results")
            (path / "experiments/ledger.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Workspace code changed"):
                reproduce.create_workspace(path, "paper-protocol")

    def test_smoke_keeps_valid_two_class_input_prefix(self):
        script, options = reproduce.stage_commands("smoke")[0]
        self.assertEqual(script, "run_camera_ready.py")
        self.assertEqual(options[options.index("--rows")+1], "160000")
        self.assertEqual(options[options.index("--seeds")+1], "42")


class PackagingTests(unittest.TestCase):
    def test_release_check_and_pending_license(self):
        files, report = check_release.check()
        self.assertTrue(files)
        self.assertEqual(report["code_files_preserved"], 11)
        self.assertEqual(report["numerical_helpers_verified"], 1)
        if report["license_selected"]:
            check_release.check(require_license=True)
        else:
            with self.assertRaisesRegex(ValueError, "license"):
                check_release.check(require_license=True)

    def test_artifacts_and_unexpected_files_cannot_enter_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("fixture")
            (root / "runs").mkdir()
            (root / "runs/model.pkl").write_bytes(b"not for publication")
            files = check_release.release_files(root)
            self.assertEqual([p.name for p in files], ["README.md"])
            destination = root / "dist/code.zip"
            check_release.package(files, destination, root)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["README.md"])
            with self.assertRaises(FileExistsError):
                check_release.package(files, destination, root)
            (root / "token.env").write_text("private")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                check_release.release_files(root)

    def test_home_paths_are_flagged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("/" + "Users" + "/example/private/file")
            with self.assertRaisesRegex(ValueError, "personal home path"):
                check_release.release_files(root)


    def test_manuscript_generators_cannot_enter_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "experiments").mkdir()
            for name in check_release.MANUSCRIPT_TOOLS:
                with self.subTest(name=name):
                    path = root / "experiments" / name
                    path.write_text("fixture")
                    with self.assertRaisesRegex(ValueError, "Manuscript-production"):
                        check_release.release_files(root)
                    path.unlink()

    def test_workflow_has_only_existing_numerical_stages(self):
        self.assertNotIn("tables", reproduce.STAGES)
        self.assertEqual(reproduce.stage_commands("results"), [("export_numerical_results.py", [])])
        for commands in reproduce.STAGES.values():
            for script, _ in commands:
                self.assertTrue((ROOT / "experiments" / script).is_file())
                self.assertNotIn(script, check_release.MANUSCRIPT_TOOLS)


class ReferenceComparisonTests(unittest.TestCase):
    def test_wrong_result_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "Numerical mismatch"):
            compare_reference.compare({"f1": .8}, {"f1": .9}, ["detectors"], [])

    def test_rf_auc_precision_exception_is_narrow(self):
        differences = []
        compare_reference.compare(.90000003, .9, ["RF", "auroc"], differences)
        self.assertEqual(len(differences), 1)
        with self.assertRaises(ValueError):
            compare_reference.compare(.90000003, .9, ["RF", "f1"], [])


if __name__ == "__main__":
    unittest.main()
