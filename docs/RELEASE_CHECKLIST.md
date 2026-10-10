# Before Publishing

- Obtain all authors' approval for public code release and check applicable
  institutional requirements. The current decision is to grant no license.
- Keep `LICENSE_STATUS.md` consistent with that decision; do not add a
  `LICENSE` file or describe the package as open source. Public visibility
  does not grant general reuse rights.
- Review the citation information. Keep DOI/proceedings metadata absent
  until assigned; add the real repository URL only after it exists.
- Run both test suites and `python scripts/check_release.py`. The optional
  `--require-license` gate is for a future licensed release and is expected
  to fail under the current no-license decision.
- Review exactly what Git will include, not just the working directory.
- Keep raw records, caches, trained models, internal audit logs, reviewer
  correspondence, and manuscript build products out of the code repository.
- Publish separately authorized research artifacts only after checking
  redistribution terms and privacy/provenance implications.
- On another machine, run setup and the data preparation check. Full
  empirical reproduction is a separate task from quick unit tests.

## Changes in This Release Copy

The 11 original algorithm/runner/verifier files are unchanged. Three
manuscript-production builders are excluded; category aggregation is
preserved in a tested numerical exporter. The last workflow stage writes
JSON/CSV, not documents. Numerical dependency versions are unchanged; the
unused plotting dependency is removed. Release-only files provide data
acquisition and identity checks, isolated workspaces, code-only ZIP creation,
documentation, citation metadata, Git exclusions, and a CI definition.
The LR saved-evidence test now explicitly skips when its artifacts are absent.

CI has read-only repository permissions, does not download research datasets,
does not train full models, and does not publish releases. It will run only
after the code is deliberately placed on GitHub and its events are triggered.
Local verification does not mean hosted CI has already passed.

The credential scan checks common token/private-key formats and personal
home paths. It is a useful guard, not a guarantee that every possible secret
or privacy issue has been detected.
