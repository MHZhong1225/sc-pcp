# Experiment runner map

This directory is intentionally flat because frozen result manifests bind every
runner by **relative path, byte count, and SHA-256**. Renaming, moving, merging,
or deleting one of the 41 Python files would break the recorded source contract.
The files are small (about 2.23 MiB in total); generated `__pycache__` content is
not part of the project and may always be removed.

## Where to start

- Current frozen-result audit and paper rendering live in `tools/`; see the
  commands in the repository `README.md`.
- `run_paper_suite.py` is the single entry point for the older
  production/no-γ RQ1/RQ3 suite.
- `run_per_step.py` and `run_per_step_study.py` are the underlying generic
  single-run and study launchers.
- The two `run_controlled_clinical_mimic_cxr_budget_followup*.py` files are the
  current MIMIC-CXR v2 pre-coverage and science runners.

## File groups

| Group | Files | Purpose |
|---|---:|---|
| Historical production suite | 4 | `run_paper_suite.py`, `run_per_step.py`, `run_per_step_study.py`, `plot_per_step.py` |
| Signed-γ and gated clinical lineage | 16 | Native signed-γ preflight/science/repair/replay plus clinical extension, fidelity v3–v6, environment-support v1, and budget-follow-up v2 |
| Formal evidence | 7 | Exact finite MDP, all-six controlled benchmark, copula, horizon/overlap, calibration-size, propensity, and strict-split studies |
| Prefix/mechanism diagnostics | 6 | Controlled prefix benchmark/ablations, fixed-schedule COT, marginal-prefix pilot, sequential-DR, and tail-shift value |
| Retained negative/historical studies | 5 | Conservatism, PAC-grid NO-GO, Phase 0, Phase 0 sanity, and Phase 0c |
| Frozen-result summarizers | 3 | Conservatism, Phase 0, and Phase 0c summarizers |

The development, repair, retry, and NO-GO names are not disposable drafts.
They preserve the decision history and are imported by later protocols or bound
to immutable negative-result artifacts. New experiments should use a new output
root; existing runners and result roots must not be edited in place to improve a
reported result.

