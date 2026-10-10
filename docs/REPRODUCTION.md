# Reproduction Details

All commands run from the release root with its Python environment active.
`scripts/reproduce.py` copies the experiment files into a new workspace so
the legacy scripts' relative outputs stay together. It does not modify their
source, seeds, thresholds, attack budgets, aggregation, or reference results.

## Execution Order

| Stage | Scripts | Main output |
| --- | --- | --- |
| `initial` | `run_camera_ready.py` | `results/` |
| `ledger` | `run_ledger_validation.py` | `results/ledger/` |
| `holdouts` | `run_generalization.py` | `results_technical/` |
| `controls` | `run_random_controls.py` | Same-record random-control runs |
| `replay-scaling` | `run_scaling_replay.py` | Source replay and ledger size sweep |
| `linear` | `run_second_backbone.py` | `results_reviewer3/` |
| `latency` | `run_request_latency.py` | Local timing observations |
| `verify` | Both verifier scripts and the experiment test suite | Verification reports |
| `results` | `export_numerical_results.py` | Numerical JSON and category CSV |

`--stage all` runs this order. Individual stages can be run in the same
workspace, for example `--stage initial` followed by `--stage ledger`.
Prerequisites are checked and attempted stages are recorded. A failed stage
is not silently resumed. A partially completed full run is not a complete
verification result.

## Numerical Export

The final `results` stage reads five saved JSON summaries/reports and
`results/attack_outcomes.csv`. Both verification reports must have passed.
It exports `numerical_results/numerical_summary.json`, preserving all input
JSON fields, numbers, and units without display rounding. It also exports
`numerical_results/category_summary.csv`, retaining the original category
aggregation: average within each noise repetition, average repetitions
within each seed, then average seeds equally. Attempt counts come from raw
explanation cases only, so protected repetitions do not inflate the count.
Category ASRs and paired drops are fractions, not percentages.

The JSON includes input SHA-256 hashes. Exporting is not an independent
verification or a claim of new measurements. Timings retain their original
units and measurement conditions. No LaTeX, figures, or document is produced.

To export an existing complete run into a separate new directory:

```sh
python experiments/export_numerical_results.py --work-dir runs/paper --output-dir runs/numerical-export
```

Existing output directories are rejected rather than overwritten. Full
underlying evidence is still needed to recompute predictions or attacks;
the retained aggregate reference files alone are not sufficient.

## Smoke Profile

The smoke profile uses the paper's first 160,000 rows, model seed 42, 200 calibration
records, 100 fidelity records, and up to 50 attacks per dataset. It executes
initial modeling/disclosure and reduced ledger tests. These reduced
checks must not be reported as manuscript experiments. Full-evidence tests
and fixed 500-case assertions apply only to the full profile.
The single-seed SD/interval placeholders in smoke summaries are not
uncertainty estimates and must not be interpreted as such.

The input prefix is not shortened: an early ToN-IoT prefix can contain only
malicious records and is unsuitable for binary classification. No labels
or source ordering are changed to manufacture a small test set.

## Rechecking Retained Paper Artifacts

`reference_results/` contains aggregate outputs only, explicitly labeled as
retained reference material. It does not let a reader independently replay
individual attacks. To recheck the original evidence without retraining,
the researcher needs the corresponding complete `results/`,
`results_technical/`, and `results_reviewer3/` directories and original
source caches. Those artifacts are deliberately not included in Git.

In an isolated working copy containing those artifacts and `experiments/`,
run:

```sh
python experiments/verify_technical.py
python experiments/verify_reviewer3.py
python -m unittest discover -s experiments -v
```

The verifiers write their reports in that copy. Do not point rerun tools at
the retained archival evidence. The 11 original computational script hashes
in `paper_code_manifest.json` agree with that archived experiment snapshot.
The numerical exporter has a separate hash. The three presentation-only
builders are excluded, and the workflow's last stage is now `results`.

## What Verification Means

Detection confusion counts, ASRs, reconstruction MAEs, train-only statistics,
split separation, bounds, and summary arithmetic can be checked from retained
models and arrays. Hardware-level floating-point ties can change unrounded
AUROC slightly. A new retraining run need not be byte-identical across
architectures; do not relabel new outputs as the original observations.

Latency scripts collect new timings. Their verification checks arithmetic,
record predictions, and the stated local boundary, not equality to the
paper's old wall-clock times. No workload, A3, production, or legal compliance
validation is introduced by this packaging work.
