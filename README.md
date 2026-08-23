# SC-PCP: per-step performative conformal calibration

This repository studies sequential prediction sets that change the policy using
them and therefore change the distribution of future states, actions, and
scores.  The paper target is **per-step marginal coverage**:

\[
\Pr_{P_{q_{0:t}}}\!\left\{Y_{t+1}\in C_{q_t}(S_t,A_t)\right\}
\ge 1-\alpha,\qquad t=0,\ldots,T-1.
\]

The sole final method is **SC-PCP**: uncapped committed-prefix
importance-weighted marginal calibration with a free stagewise radius vector
\(q=(q_0,\ldots,q_{T-1})\).  It does not constrain the schedule to a common
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
the final SC-PCP path does not fit a COT model.  It uses \(D_{\rm COT}\) only to
freeze a separate 101-point empirical score grid for each stage before final
selection.  Both patient-disjoint parts of \(D_{\rm cal}\) then contribute to
every calibration estimate.

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

These importance weights are not clipped or capped.  The implementation keeps
raw log weights in float64 and subtracts the candidate-specific maximum before
exponentiation.  This common rescaling leaves the Hájek estimate and effective
sample size unchanged.  It is distinct from the target/reference policy-ratio
cap used when defining the shared deployment policy.

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
fixed \(T\), sequential identification and positivity, exact or consistently
estimated uncapped prefix ratios, a fixed finite grid (or the corresponding
uniform convergence for convergent empirical grids), and a stable population
selector imply

\[
\min_{0\le t<T} C_t(\widehat q_{0:t})
\ge 1-\alpha-o_p(1).
\]

This is the honest boundary of the current implementation: the selected radii
and their induced deployment laws are learned from the same calibration sample,
so no exact finite-sample conformal claim is made.  Clinical results are
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

## Validation and entry points

```bash
conda run -n ucp pytest -q
```

The main method is implemented in `src/scpcp/marginal_prefix.py`, integrated by
`src/scpcp/experiment.py`, scheduled by `scripts/run_paper_suite.py`, and
rendered by `tools/render_paper_results.py`.
