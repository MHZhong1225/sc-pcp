# SC-PCP: per-step performative conformal calibration

This repository studies sequential prediction sets that change the policy using
them and therefore change the distribution of future states, actions, and
scores.  The paper target is **per-step marginal coverage**:

\[
\Pr_{P_{q_{0:t}}}\!\left\{Y_{t+1}\in C_{q_t}(S_t,A_t)\right\}
\ge 1-\alpha,\qquad t=0,\ldots,T-1.
\]

The sole final method is **SC-PCP**: marginal calibration with an unclipped
cumulative committed-prefix importance product under a structurally
ratio-capped target policy, using a free stagewise radius vector
\(q=(q_0,\ldots,q_{T-1})\). It does not constrain the schedule to a common
scale or a prespecified stage profile.

## Why historical calibration is insufficient

The prediction set affects the behavior-anchored target policy.  A radius
chosen from logged scores therefore changes not only set width, but also the
action distribution and the downstream score law.  Standard CP calibrates the
logged law; SC-PCP instead estimates the score quantile under the policy induced
by the radius being considered.  The calibration is performed one stage at a
time because the distribution at stage \(t\) depends on the radii already
committed at stages \(0{:}t-1\).

All methods share the same frozen heteroscedastic outcome model and normalized
maximum score

\[
R_{it}=\max_j
\frac{|Y_{i,t+1,j}-\hat\mu_j(S_{it},A_{it})|}
{\hat\sigma_j(S_{it},A_{it})},
\]

with rectangular prediction set

\[
C_{q_t}(s,a)=\left\{y:
|y_j-\hat\mu_j(s,a)|\le q_t\hat\sigma_j(s,a),\ \forall j\right\}.
\]

## Final SC-PCP procedure

Patients are split before any fitting.  \(D_{\rm pred}\) fits and freezes the
outcome model and, for clinical data, the behavior-policy nuisance model.
\(D_{\rm COT}\) and \(D_{\rm cert}\) stay patient-disjoint, but the final
calibration sample is

\[
D_{\rm cal}=D_{\rm COT}\cup D_{\rm cert}.
\]

The historical role name \(D_{\rm COT}\) is retained for artifact compatibility;
the final SC-PCP path does not fit a COT model. It uses \(D_{\rm COT}\) to
construct a separate 101-point empirical score grid for each stage. Both
patient-disjoint parts of \(D_{\rm cal}\) then contribute to every calibration
estimate. Because \(D_{\rm COT}\) is reused, the validity argument relies on
uniform convergence over the complete compact radius class, not independence
of the empirical grid.

Suppose \(q_0,\ldots,q_{t-1}\) have been committed.  For candidate radius \(r\)
at stage \(t\), SC-PCP computes the full observed-action prefix weight

\[
W_{it}(r;q_{<t})=
\prod_{h<t}
\frac{\pi_{q_h,h}(A_{ih}\mid S_{ih})}
     {\mu_h(A_{ih}\mid S_{ih})}
\frac{\pi_{r,t}(A_{it}\mid S_{it})}
     {\mu_t(A_{it}\mid S_{it})}.
\]

The cumulative importance products are not clipped or capped. The
implementation keeps raw log weights in float64 and subtracts the
candidate-specific maximum before exponentiation. This common rescaling leaves
the Hájek estimate and effective sample size unchanged. It is distinct from
the per-action target/reference policy-ratio cap that is part of the deployed
policy definition.

The candidate's target-policy marginal coverage estimate is

\[
\widehat C_t(r;q_{<t})=
\frac{\sum_{i\in D_{\rm cal}}W_{it}(r;q_{<t})
\mathbf 1\{R_{it}\le r\}}
{\sum_{i\in D_{\rm cal}}W_{it}(r;q_{<t})}.
\]

SC-PCP retains candidates with \(\widehat C_t\ge 1-\alpha\), selects the one
with the smallest importance-weighted normalized width, commits it, and moves
to stage \(t+1\).  If no candidate is feasible, selection is unavailable for
that seed.  It does not silently replace a failed selection with the largest
set.  The final artifact records candidate coverages, widths, effective sample
sizes, log-weight spans, endpoint selections, and any failure stage.

## Guarantee boundary

The method targets an **asymptotic per-step marginal guarantee**, not a
finite-sample distribution-free, PAC, or data-conditional certificate.  For
fixed \(T\), sequential identification and positivity, exact or uniformly
ratio-consistent fitted logging propensities, a uniform LLN over the compact
prefix-radius class, and selection availability with probability tending to
one imply

\[
\min_{0\le t<T} C_t(\widehat q_{0:t})
\ge 1-\alpha-o_p(1).
\]

The proof uses empirical feasibility plus a uniform bound on the complete
coverage surface; it does not require a unique or stable population selector.
This is the honest boundary of the current implementation: the selected radii
and their induced deployment laws are learned from the same calibration
sample, so no exact finite-sample conformal claim is made. Clinical results are
controlled evaluations in a frozen held-out empirical environment, not claims
about an unobserved real-world intervention.

## Data roles and comparison methods

All roles are patient-disjoint.  Synthetic data use 40% \(D_{\rm pred}\), 20%
\(D_{\rm COT}\), and 40% \(D_{\rm cert}\).  Clinical data use 40%
\(D_{\rm pred}\), 15% \(D_{\rm COT}\), 30% \(D_{\rm cert}\), and 15%
\(D_{\rm env}\).  \(D_{\rm env}\) is used only to construct the frozen clinical
evaluator.

Every complete seed contains exactly these six canonical paper names:

| Method | Information regime |
| --- | --- |
| `Standard CP` | logged calibration data only |
| `ACI` | logged initialization + 2,000 target-policy adaptation trajectories |
| `MFCS` | logged calibration data only |
| `SPCI` | logged initialization + 2,000 target-policy adaptation trajectories |
| `PRC` | logged initialization + 2,000 target-policy adaptation trajectories |
| `SC-PCP` | logged calibration data only |

MFCS, SPCI, and PRC are task-aligned adapters because their upstream interfaces
do not directly accept this project's longitudinal treatment trajectories,
radius-dependent policies, and common rectangular multivariate sets.  Figures
and result records nevertheless use only the canonical names above.  The three
online methods each receive their own 2,000 trajectories over three rounds;
they do not share that budget.  See
[`docs/baselines_and_settings.md`](docs/baselines_and_settings.md) for the exact
algorithms and caveats.

## Final paper suite

The default environment is `ucp`, and the runner uses both GPUs.  RQ1 contains
Synthetic (\(T=12\), 100 seeds), eICU, INSPIRE, and MIMIC-IV (\(T=12\), 20 seeds
each), and MIMIC-CXR (\(T=6\), 20 seeds).  RQ3 adds Synthetic feedback strengths
\(\beta\in\{0,0.5,2\}\), 100 seeds each; \(\beta=1\) reuses RQ1.  RQ2 and RQ4
reuse artifacts declared in the suite manifest.

Start a new complete suite in an empty output root:

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_final
```

Resume the exact same suite after interruption:

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_final \
  --resume
```

Resume validates the stored suite manifest and every completed seed; it is not
an invitation to change datasets, source, or configuration in place.  A fresh
run refuses a nonempty output root, and the top-level `COMPLETE` marker is
written only after all requested settings finish.

Render the completed suite:

```bash
conda run -n ucp python tools/render_paper_results.py \
  --input results/work/paper_final \
  --output results/paper_final
```

The renderer fails closed on missing settings, seeds, canonical method rows, or
completion markers.  Its final output directory contains PDF files only.

The frozen 2026-08-22 result table, interpretation, and limitations are recorded
in [`docs/main_results_20260822.md`](docs/main_results_20260822.md).
The complete artifact map—formal suite, controlled-shift diagnostics, and
explicit NO-GO studies—is maintained in
[`docs/experiment_data_inventory_20260824.md`](docs/experiment_data_inventory_20260824.md).

The three formal studies completed on 2026-08-25—an exact finite-MDP
identification audit, a controlled six-method comparison, and a predeclared
equal-marginal copula gate—are reported in
[`docs/formal_experiments_20260825.md`](docs/formal_experiments_20260825.md).
In the fresh controlled all-six study, SC-PCP improves WSC over Standard CP by
3.46 and 1.86 percentage points at \(\gamma=-4,-2\), but reaches only 0.8983
and 0.8974 rather than uniformly attaining 0.90. The orthogonal copula gate is
a formal NO-GO because its directional effect is statistically clear but too
small under the frozen practical-magnitude thresholds. These are controlled
semi-synthetic diagnostics, not natural clinical treatment-effect evidence or
a universal SOTA claim.

For paper-facing presentation, \(\gamma=-4\) is the default displayed hero
stress case within the controlled semi-synthetic study. This is a presentation
choice only: \(\gamma=-2\) remains the frozen primary cell, \(\gamma=-4\)
remains the prespecified stress endpoint, and the complete five-point signed
curve remains authoritative. No controlled semi-synthetic cell replaces the
separate five-setting production-style suite.

Earlier two-method confirmation and post-confirmatory ablation artifacts remain
documented in [the experimental evidence ledger](docs/experimental_evidence_20260824.md),
but their protocol-specific values must not replace the later all-six results.
The manuscript structure and claim boundary are collected in
[the ICLR paper blueprint](docs/paper_blueprint_20260824.md).

The four theory- and robustness-facing studies completed on 2026-08-26 are
reported in
[`docs/formal_experiments_20260826.md`](docs/formal_experiments_20260826.md).
They quantify the horizon--overlap loss of effective sample size, fixed-grid
coverage-surface convergence as calibration size grows, fitted-propensity
sensitivity under a fixed target law, and an independent-calibration
strict-split variant. These diagnostics leave the canonical SC-PCP selector
unchanged. They strengthen the empirical support for its asymptotic theory but
do not create a finite-sample, distribution-free, PAC, clinical, or universal
SOTA claim.

The completed dataset-native controlled clinical extension is also recorded in
[`docs/formal_experiments_20260826.md`](docs/formal_experiments_20260826.md#10-dataset-native-controlled-clinical-extension-v2).
All four clinical settings pass the support gate, but only MIMIC-IV passes the
logging-mixture K0 fidelity gate (20/20) and the donor-overlap screen. At its
confirmatory \(\gamma=-4\) endpoint, Standard CP has WSC 0.86358 and SC-PCP
0.90089 (+3.73 percentage points) with a 1.204 width ratio; SC-PCP's 95% WSC
interval still crosses 0.90. eICU, INSPIRE, and MIMIC-CXR + IV/ED are formal
K0 NO-GO results (12/20, 13/20, and 10/20), so no scientific coverage rows were
generated for them. These NO-GO panels must not be filled with production-style,
Synthetic \(\beta=2\), or older MIMIC v1 curves.

The corresponding deterministic paper figures are
[`figure_theory_diagnostics.pdf`](results/paper_theorem_robustness_20260826/figure_theory_diagnostics.pdf)
and
[`figure_robustness_audits.pdf`](results/paper_theorem_robustness_20260826/figure_robustness_audits.pdf).
Their editable exports, source-data CSVs, and QA report are in
[`results/work/theorem_robustness_report_20260826`](results/work/theorem_robustness_report_20260826).

The complete submission figure portfolio is indexed in
[`docs/figure_portfolio_20260826.md`](docs/figure_portfolio_20260826.md). Newly
rendered frozen-artifact outputs include the
[`method schematic`](results/paper_method_schematic_20260826/figure_method_schematic.pdf),
[`five-setting Pareto figure`](results/paper_main_suite_figures_20260826/figure_main_pareto.pdf),
[`dataset-native gated controlled-stress grid`](results/paper_five_setting_stage_profiles_20260826/figure_controlled_stress_grid.pdf),
[`all-five production/native stagewise profiles`](results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf),
[`exact-prefix identification heatmap`](results/paper_formal_mechanism_20260826/figure_exact_prefix_identification.pdf),
[`controlled all-six figure`](results/paper_formal_mechanism_20260826/figure_controlled_signed_all_six.pdf),
and the
[`controlled all-six table`](results/paper_formal_mechanism_20260826/table_controlled_signed_all_six.pdf).
The older single-panel
[`controlled stress profile`](results/paper_controlled_stress_stage_profile_20260826/figure_controlled_stress_stage_profile.pdf)
remains a valid 2026-08-25 MIMIC v1 render, but it is protocol-specific and is
not the all-dataset figure.
Every paper directory is PDF-only; editable SVG, 600-dpi TIFF, source CSV,
analysis, QA, and hash manifests remain in the corresponding `results/work`
bundle. These renderers do not rerun scientific seeds.

## Validation and entry points

```bash
conda run -n ucp pytest -q
```

Re-render the frozen theory/robustness figures into fresh empty output roots:

```bash
conda run -n ucp python tools/render_theorem_robustness_results.py \
  --work-output results/work/theorem_robustness_report_rerender \
  --paper-output results/paper_theorem_robustness_rerender
```

Re-render commands for the method, main-suite, and formal-mechanism figures are
listed in [`docs/figure_portfolio_20260826.md`](docs/figure_portfolio_20260826.md).

The main method is implemented in `src/scpcp/marginal_prefix.py`, integrated by
`src/scpcp/experiment.py`, scheduled by `scripts/run_paper_suite.py`, and
rendered by `tools/render_paper_results.py`.
