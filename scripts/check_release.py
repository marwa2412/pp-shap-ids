"""Check the code-only release and optionally create a local, allowlisted ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "LICENSE_STATUS.md", "LICENSE", "LICENSE.txt", "CITATION.cff",
             "requirements.txt", ".gitignore", ".gitattributes", ".python-version",
             "paper_code_manifest.json", "data_manifest.json"}
DIRECTORIES = {"experiments", "scripts", "tests", "docs", "reference_results", ".github"}
EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache", "runs", "dist", "numerical_results", ".DS_Store"}
SUFFIXES = {".py", ".json", ".md", ".txt", ".yml", ".yaml"}
MANUSCRIPT_TOOLS = {"build_paper_results.py", "build_technical_results.py", "build_reviewer3_artifacts.py"}
PATTERNS = {
    "personal home path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "AWS access key": re.compile(r"AKIA[A-Z0-9]{16}"),
}


def release_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted in release sources: {relative}")
        if path.name in MANUSCRIPT_TOOLS:
            raise ValueError(f"Manuscript-production tool is not part of this package: {relative}")
        if path.is_dir():
            if relative.parts[0] not in DIRECTORIES:
                raise ValueError(f"Unexpected directory outside the release allowlist: {relative}")
            continue
        allowed = (len(relative.parts) == 1 and relative.name in TOP_FILES) or (
            len(relative.parts) > 1 and relative.parts[0] in DIRECTORIES and path.suffix in SUFFIXES)
        if not allowed or path.stat().st_size > 10_000_000:
            raise ValueError(f"Unexpected or oversized release file: {relative}")
        text = path.read_text(encoding="utf-8")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"Possible {label} in {relative}; inspect before publishing")
        files.append(path)
    return files


def check(root=ROOT, require_license=False):
    files = release_files(root)
    manifest = json.loads((root / "paper_code_manifest.json").read_text())
    expected = manifest["sha256"]
    additional = manifest["additional_sha256"]
    if expected.keys() & additional.keys():
        raise ValueError("Original and additional source manifests overlap")
    for name, value in {**expected, **additional}.items():
        if hashlib.sha256((root / "experiments" / name).read_bytes()).hexdigest() != value:
            raise ValueError(f"Experiment source differs from the recorded paper code: {name}")
    if (root / "requirements.txt").read_bytes() != (root / "experiments/requirements.txt").read_bytes():
        raise ValueError("Root and experiment dependency pins differ")
    licensed = (root / "LICENSE").is_file() or (root / "LICENSE.txt").is_file()
    if require_license and not licensed:
        raise ValueError("No license file is present; --require-license is only for a licensed release")
    return files, {"status": "passed", "code_files_preserved": len(expected), "release_files": len(files),
                   "numerical_helpers_verified": len(additional),
                   "total_bytes": sum(path.stat().st_size for path in files),
                   "license_selected": licensed, "scope": "Local code packaging, not a comprehensive security audit"}


def package(files, destination, root=ROOT):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(root)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-license", action="store_true")
    parser.add_argument("--archive", type=Path, help="Create a NEW local ZIP; never uploads or replaces an existing archive")
    args = parser.parse_args()
    try:
        files, report = check(require_license=args.require_license)
        if args.archive:
            package(files, args.archive)
            report["archive"] = str(args.archive)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    main()
