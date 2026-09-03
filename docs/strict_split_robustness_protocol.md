# Strict-split robustness protocol

Status on 2026-08-26: **formal study complete and independently audited**.
The immutable result root is
[`results/work/strict_split_robustness_v1_20260826`](../results/work/strict_split_robustness_v1_20260826),
with the full table and claim boundary in
[`formal_experiments_20260826.md`](formal_experiments_20260826.md).

## Scope

This study is a theory-aligned robustness audit, not a new method and not a
rule for replacing the frozen paper method after observing the results.  The
canonical method remains uncapped committed-prefix importance-weighted
marginal calibration with free stagewise radii.  Its claim remains asymptotic
per-step marginal coverage; neither arm of this audit is described as a
finite-sample, distribution-free, PAC, or data-conditional guarantee.

The audit changes exactly one input to the unchanged selector.  Both arms
freeze the identical candidate family from `D_COT`:

\[
\mathcal G_t=\mathcal G_t(D_{\rm COT}).
\]

The canonical arm estimates candidate coverage and width using

\[
D_{\rm COT}\cup D_{\rm cert},
\]

whereas the strict arm uses only

\[
D_{\rm cert}.
\]

Conditionally on `D_COT`, the strict arm therefore selects over a fixed finite
grid using an independent calibration role.  It also has fewer selection
trajectories, so similar results are informative robustness evidence rather
than an automatic reason to alter the canonical method.

## Frozen settings

| Setting | Seeds | Grid role | Canonical selection | Strict selection | Fresh evaluation |
|---|---:|---|---|---|---:|
| Synthetic main | 1000--1099 (paired legacy reuse) | `D_COT` | `D_COT` + `D_cert` | `D_cert` | 50,000 per arm/seed |
| MIMIC-IV | 0--19 (paired legacy reuse) | `D_COT` | `D_COT` + `D_cert` | `D_cert` | 50,000 per arm/seed |
| Controlled \(\gamma=-2\) | 99000, 99010, ..., 99190 (fresh) | first 1,000 of 3,000 source trajectories | all 3,000 | remaining 2,000 | 20,000 per arm/seed |

Synthetic and MIMIC-IV load their exact main-paper YAML configurations.  The
two arms in each seed share `_paper_seed(seed, 900001)` for fresh target-policy
evaluation.  Their reused consecutive base seeds inherit the frozen legacy RNG
mapping and are not represented as a new independent RNG design.

The controlled setting exactly retains the formal all-six benchmark's
same-kernel mechanism and budget at \(\gamma=-2\): MIMIC-IV context,
single-step target-policy ratio cap 3, source-score q80/q95 response endpoints,
alternative-policy tilt 20, 3,000 source calibration trajectories, and 20,000
matched reference trajectories.  Only the selection subset changes.  Fresh
base seeds are spaced by ten so each task's `seed`, `seed+1` outcome-model
stream, and `seed+2` behavior-model stream cannot collide with the next task.
Calibration and evaluation use `_paper_seed(seed, 1700101)` and
`_paper_seed(seed, 1700401)`.  Bootstrap RNG 99900 is separate.  The runner
enumerates all 101 actual fresh RNG IDs, checks internal uniqueness, and audits
them against prior artifacts, actual source declarations, and coordinated
external reservations.  Reservation declarations are inventoried separately
and are not mistaken for actual use.

The later manuscript convention that displays \(\gamma=-4\) as the controlled
hero stress case does not change this study. Strict-split evidence exists only
for \(\gamma=-2\); it must not be relabeled or extrapolated to \(\gamma=-4\).

## Metrics and uncertainty

The primary coverage metric is always

\[
\operatorname{WSC}=\min_t\frac1S\sum_{s=1}^S C_{s,t},
\]

never `mean_seed(min_t C_seed,t)` and never MeanCov.  For each arm the audit
also reports mean normalized width, selection availability over every
prespecified seed, the minimum selected-prefix ESS fraction, and the minimum
candidate ESS fraction.  Coverage, width, and ESS summaries condition on
successful selection; availability always uses all prespecified seeds.

Paired WSC differences, arithmetic width differences, geometric width ratios,
and selected/candidate ESS differences use the joint-available seed set.
Availability differences use every paired seed.  Ninety-five-percent intervals
come from 10,000 complete-seed-vector bootstrap resamples shared across the two
arms.  The selection-rate interval is Wilson.  No result field is interpreted
as a pass/fail upgrade gate.

## Provenance and execution

The runner validates and binds the parent formal source manifest SHA256
`e6a1bba...c64e24`, archive SHA256 `2116b992...b2ea0b`, and parent source hash
`7665dfbe...16643`.  Because this is post-snapshot work, it separately binds
the active source tree, the complete strict-split YAML, both base YAML files,
the resolved controlled configuration, Python/NumPy/Torch/CUDA and BLAS
metadata, the launch argv, and a stable global task-to-device mapping.
Per-seed directories and root summaries are written atomically.  Fresh runs
reject an existing output root; resume rejects malformed, provenance-mismatched,
unexpected, or partially complete artifacts.

Formal execution, only after the source/config/seed freeze, is:

```bash
conda run -n ucp python scripts/run_strict_split_robustness.py \
  --output-dir results/work/strict_split_robustness_v1_20260826 \
  --devices cuda:0,cuda:1
```

Exact resume appends `--resume`.  The reduced CPU smoke uses engineering seed
13701 and records `science_seed_opened: false`; it is not a scientific result.

## Formal result status

Both variants were available for every prespecified seed. Paired strict-minus-
canonical WSC intervals crossed zero in Synthetic, MIMIC-IV, and controlled
\(\gamma=-2\). Synthetic and MIMIC-IV width-ratio intervals also crossed one;
the controlled strict arm was 0.85% wider, with a 95% interval of 0.14%--1.60%.
The audit therefore supports similar behavior under an independent calibration
role, not equivalence, coverage improvement, or a post-hoc rule for changing the
canonical method.
