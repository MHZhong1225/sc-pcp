# SC-PCP: committed-prefix calibration under policy feedback

SC-PCP calibrates sequential prediction sets when the set radius changes the
deployed action policy and therefore changes later states, actions, and scores.
The method transports each post-action score with the complete observed-action
prefix induced by the radii already committed.

## Current paper contract

- **Method:** one method named `SC-PCP`, implemented in
  [`src/scpcp/marginal_prefix.py`](src/scpcp/marginal_prefix.py) and integrated
  by [`src/scpcp/experiment.py`](src/scpcp/experiment.py).
- **Primary metric:**
  \(\min_t\operatorname{mean}_{s}(C_{s,t})\). `MeanCov` is supplementary and
  must not replace this worst-step marginal coverage metric.
- **Primary experiment:** five datasets—Synthetic, MIMIC-IV, eICU, INSPIRE,
  and MIMIC-CXR—at the prespecified environment setting \(\gamma=-4\).
  The other four signed values are descriptive sensitivity analyses.
- **Comparators:** exactly `Standard CP`, `ACI`, `MFCS`, `SPCI`, `PRC`, and
  `SC-PCP`.
- **Guarantee:** asymptotic per-step marginal coverage under the stated causal,
  overlap, propensity, and uniform-convergence assumptions. It is not a
  finite-sample, distribution-free, PAC, data-conditional, or clinical
  deployment guarantee.

Here \(\gamma\) is a dataset-environment feedback parameter, not an SC-PCP
hyperparameter. Dataset-specific frozen environments and budgets are used; one
parameterization is not mechanically copied across all five datasets.

## Method in one equation

After committing \(q_0,\ldots,q_{t-1}\), candidate radius \(r\) at stage \(t\)
uses

\[
W_{it}(r;q_{<t})=
\prod_{h<t}\frac{\pi_{q_h,h}(A_{ih}\mid S_{ih})}{\mu_h(A_{ih}\mid S_{ih})}
\frac{\pi_{r,t}(A_{it}\mid S_{it})}{\mu_t(A_{it}\mid S_{it})}.
\]

SC-PCP estimates the candidate's target-policy coverage with a Hájek ratio,
keeps candidates whose empirical coverage is at least \(1-\alpha\), selects the
least-wide feasible candidate, and commits it before moving forward. The
cumulative calibration product is uncapped; float64 log stabilization does not
alter the ratio. The target policy itself retains its structural one-step ratio
cap.

The complete definition and proof boundary are in
[`docs/final_method.md`](docs/final_method.md). A concise manuscript description
is in [`docs/method_overview_20260903.md`](docs/method_overview_20260903.md).

## Current results

The frozen current record is
[`docs/main_results_20260903.md`](docs/main_results_20260903.md). At the sole
primary endpoint \(\gamma=-4\), SC-PCP meets the prespecified point criterion in
3/5 datasets; MIMIC-IV and eICU remain explicit near-target failures. MFCS meets
the point criterion in 5/5 but is systematically more conservative. No method
is claimed to dominate coverage and width universally.

Useful outputs:

- LaTeX main table:
  [`docs/table_main_results_20260903.tex`](docs/table_main_results_20260903.tex)
- paper figure/table index:
  [`docs/figure_portfolio_20260903.md`](docs/figure_portfolio_20260903.md)
- machine-readable sources, editable figures, and QA:
  [`results/work/paper_experiment_figures_20260903`](results/work/paper_experiment_figures_20260903)
- PDF-only eight-figure submission bundle:
  [`results/paper_experiment_figures_20260903`](results/paper_experiment_figures_20260903)

The completed MIMIC-CXR v1 near miss and v2 budget-only follow-up are both
retained. The v1 record is historical evidence, not a failed pilot to erase or
a value to mix into the v2 table.

## Validate and render

Use the `ucp` conda environment. GPU experiments use `cuda:0,cuda:1`; do not
silently fall back to CPU.

Audit the frozen five-dataset table without rerunning science:

```bash
conda run -n ucp python tools/interpret_five_dataset_signed_gamma_results.py
```

Re-render the current frozen inputs into new empty directories:

```bash
conda run -n ucp python tools/render_paper_figures.py \
  --main-source results/work/paper_experiment_figures_20260903 \
  --work-output results/work/paper_experiment_figures_rerender \
  --paper-output results/paper_experiment_figures_rerender
```

Run the per-step validation suite:

```bash
conda run -n ucp pytest -q tests/per_step
```

Some sealed protocol tests were written as prelaunch assertions and therefore
expect one-time result roots not to exist. In a completed workspace, do not
delete formal artifacts merely to make those assertions pass; use the stored
manifest validators and report the stateful exclusions.

## Historical production/no-\(\gamma\) robustness suite

The older RQ1/RQ3 runner remains reproducible but is not the current signed-
\(\gamma\) primary experiment. A fresh run must use an empty output root:

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_final
```

Resume the exact same manifest by appending `--resume`. Render it with:

```bash
conda run -n ucp python tools/render_paper_results.py \
  --input results/work/paper_final \
  --output results/paper_final
```

Do not substitute these production/no-\(\gamma\) rows for missing or
underperforming signed-feedback cells.

## Repository map

- [`src/scpcp/`](src/scpcp): method, policies, simulators, adapters, and formal
  study support.
- [`configs/`](configs): frozen experiment configurations.
- [`scripts/`](scripts): science runners; the [runner map](scripts/README.md)
  separates current entry points, formal studies, and retained protocol history.
  Formal roots are immutable and resume fails closed on provenance mismatch.
- [`tools/`](tools): deterministic interpretation and rendering.
- [`tests/per_step/`](tests/per_step): method, protocol, and artifact-contract
  tests; the [test map](tests/README.md) explains why historical and NO-GO
  contracts remain in the active collection.
- [`docs/README.md`](docs/README.md): current documentation map and explicit
  separation between manuscript sources, formal evidence, and history.
- `data/`: local clinical caches; intentionally not source-controlled.
- `results/`: generated evidence and render bundles; intentionally not
  source-controlled.
- `baselines/`: trimmed upstream reference source. The project runs its own
  task-aligned adapters and does not import these repositories; bulky upstream
  example datasets and published-output copies are intentionally omitted. The
  retained checkout revisions are MFCS `c737536d874f`, MultiDimSPCI
  `2b22e47088ed`, and PRC `b11d3964f426`.

Final paper directories contain PDF files only. Editable SVG/TIFF/PNG and
source-data exports belong under the corresponding `results/work/` bundle.
