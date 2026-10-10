# PP-SHAP-IDS

Repository: [marwa2412/pp-shap-ids](https://github.com/marwa2412/pp-shap-ids).

Research code for **Controlled Explanation Disclosure for Intrusion Detection:
Empirical Evaluation and Evidence Replay**, by Marwa Elarfaoui, Kaouthar Chetioui,
and Said Hraoui.

This package contains the PP-SHAP disclosure implementation, matched
feature-space attacks, detector evaluation, reconstruction checks, evidence
ledger/replay, local timing, and numerical JSON/CSV exports. The 11 original
computational scripts in `paper_code_manifest.json` are byte-identical to the
retained experiment code. A separate numerical exporter preserves category
summaries previously calculated inside a presentation script. Its hash is
recorded separately; no experimental algorithm has been changed.

This numerical-only package excludes manuscript-production scripts, LaTeX
generation, figure/diagram drawing, and the manuscript itself. Historical
names such as `run_camera_ready.py` and `verify_reviewer3.py` are retained to
preserve source provenance: they run or verify experiments, not documents.

**No software license is granted.** This package is intentionally unlicensed,
not open source; making it public would not grant general reuse rights.
Publication DOI and proceedings details remain unset until available.
See `CITATION.cff` and
[license status](LICENSE_STATUS.md).

## Setup

The recorded environment is Python 3.14.6 on ARM64 macOS. Linux CI is supplied
for future execution, not claimed as already run. Start in this directory:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s experiments -v
python -m unittest discover -s tests -v
python scripts/check_release.py
```

No RTK, Codex, LaTeX, account token, or personal filesystem path is required.
The experiment suite discovers 25 tests: 21 run without datasets; four
saved-evidence tests explicitly skip until their full outputs exist. The
release-tool suite checks data integrity, safe workspaces, and packaging.
Passing these quick tests does not reproduce the paper's empirical results.

The seven direct numerical dependencies are pinned to the original versions.
The unused plotting dependency has been removed.
Transitive dependencies are resolved by pip, not claimed as a fully locked
cross-platform environment. Full model retraining can vary numerically
across architectures; latency is inherently machine-dependent.

The original experiment implementation completed a full local reproduction
on 2026-10-03. This is a separate numerical-only package; see
[its validation record](docs/VALIDATION.json) for the exact checks performed
on this copy. Re-exporting saved results is not a new model-training run.
The validation record describes local checks before publication. Hosted
GitHub Actions results, when available, are reported separately in the
repository's Actions tab.

## Data Preparation

Datasets and trained models are not bundled. Review the dataset owners' and
mirrors' access/use terms before downloading. Then:

```sh
python scripts/prepare_data.py
python scripts/prepare_data.py --check
```

The helper downloads only the three source files from the pinned revisions
in [data_manifest.json](data_manifest.json), checks their SHA-256 and byte
counts, and prepares the standard `~/.cache/huggingface` layout required by
the unchanged paper scripts. It also checks the ordered schemas, total row
counts, and a logical fingerprint of the initial 160,000 rows. Existing
mismatched files are rejected, never silently replaced. No fallback revision
or substitute dataset is used. Network failure stops preparation; rerunning
reuses fully verified files.

Offline import is available when the pinned source files already exist in
another Hugging Face cache root:

```sh
python scripts/prepare_data.py --from-cache /path/to/huggingface-cache
```

The cache-root argument above is an example supplied by the researcher, not
a required user-specific path. See [dataset notes](docs/DATA.md) for the
distinction between source-byte checks and derived Arrow serialization.

## Reproduce

First inspect the complete command order without creating any output:

```sh
python scripts/reproduce.py --work-dir runs/paper --stage all --plan
```

Run a small end-to-end check, explicitly not paper evidence:

```sh
python scripts/reproduce.py --work-dir runs/smoke --stage smoke
```

Run the full paper protocols in a different fresh directory:

```sh
python scripts/reproduce.py --work-dir runs/paper --stage all
python scripts/compare_reference.py --work-dir runs/paper
```

Each workspace receives unchanged copies of the experiment scripts and a
`reproduction.json` execution record. Existing unrelated directories and
mixed smoke/full profiles are rejected. A stage that has been attempted is
not automatically rerun or resumed. After a failure, choose a new workspace
or inspect the underlying script deliberately; the wrapper never deletes
data to recover. See [reproduction details](docs/REPRODUCTION.md) for stages,
prerequisites, and evidence boundaries.

Full execution covers initial LR/RF/XGBoost detection; XGBoost chronological,
endpoint-pair and random-control protocols; LR disclosure; ledger mutation
and replay; size scaling; latency; independent verification; and numerical
exports. Final outputs are `runs/paper/numerical_results/numerical_summary.json`
and `runs/paper/numerical_results/category_summary.csv`. The original detailed
arrays, JSON reports, and CSVs remain in that run's result directories.
Avoid unrelated heavy computation during latency measurement.

## Contents

| Path | Purpose |
| --- | --- |
| `experiments/` | Numerical implementation, runners, verifiers, exporter, and 25 original tests |
| `scripts/` | Data preparation, isolated reproduction, and local release checks |
| `tests/` | Tests for the new release helpers |
| `reference_results/` | Retained aggregate summaries, not a substitute for per-record evidence |
| `paper_code_manifest.json` | SHA-256 identity of 11 unchanged scripts and the separate numerical exporter |
| `data_manifest.json` | Pinned mirror files, hashes, schemas, row counts, and order checks |
| `runs/` | Newly generated local evidence; ignored by Git and omitted from release ZIPs |

Manuscript drafts, reviewer correspondence, local environments, raw network
records, trained models, and personal audit/build logs are excluded. The
article source archive is a separate deliverable, not this code release.

## Scope

Results apply to disclosed mirror subsets and continuous feature-space
attacks, not executable traffic. A lower mean ASR is not differential
privacy. Grouping alone provides most of the initial observed reduction;
full PP-SHAP trades further reduction for greater reconstruction error and
has an adverse LR run. The reconstruction bound is valid but usually loose.
Human benefit, A3/tenant protection, production readiness, regulatory
compliance, and optimization of the illustrative objective are unvalidated.

## Later Release

No command in this package creates a remote repository, commits, or pushes.
After author review, use the release check and create
a local, code-only archive:

```sh
python scripts/check_release.py
python scripts/check_release.py --archive dist/controlled-explanation-disclosure-numerical-no-license-2026-10-10.zip
```

No license file is added under the current decision. The optional
`--require-license` check intentionally fails for this unlicensed package;
it is only relevant if a licensed release is later approved. The archive
command prepares a local copy, grants no license, and refuses to overwrite
an existing ZIP. Passing the check is not publication approval.
See [the release checklist](docs/RELEASE_CHECKLIST.md).
