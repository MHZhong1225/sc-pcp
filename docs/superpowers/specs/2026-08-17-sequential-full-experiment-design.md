# Sequential SC-PCP Full Experiment Design

**Date:** 2026-08-17

**Repository:** `/home/ubuntu/zmh/sc-pcp`

**Implementation branch:** `codex/sequential-full-experiment-20260817`
**Target environment:** `/home/ubuntu/anaconda3/envs/ucp`, two NVIDIA RTX 4090 D GPUs

## 1. Objective

Replace the current profiled family

\[
q_t(s)=s b_t
\]

with a genuinely sequential schedule

\[
\mathbf q=(q_0,\ldots,q_{T_{\max}-1}),
\]

where stage \(t\) is calibrated under the target-policy risk set induced by the already selected prefix \(q_{<t}\). The experiment must determine whether this additional freedom reduces prediction-set width without sacrificing supported-stage coverage. The clinical phase must evaluate variable-length episodes instead of treating every patient as a complete fixed rectangle.

The work is gated. Phase 0 tests structural efficiency with an oracle. Practical COT, variable-horizon clinical reconstruction, and the full paper suite are implemented only if the oracle passes the pre-registered Go criteria.

## 2. Non-goals

- Phase 0 does not claim a practical method or theorem.
- A fixed-\(T\) raw minimum is not a headline coverage guarantee.
- Clinical results are controlled empirical-environment experiments, not evidence of real patient deployment or causal treatment benefit.
- The first practical version does not add a global inflation scale after learning the sequential schedule.
- The first variable-horizon version does not claim coverage for unobserved outcomes without an observation/censoring model.
- Existing paper results and the current main checkout are not overwritten.

## 3. Frozen terminology and data roles

Use the following names in new code and artifacts:

- `prediction`: frozen outcome and behavior nuisance fitting.
- `transport`: sequential occupancy construction.
- `calibration`: frozen-schedule coverage bounds and selection diagnostics.
- `environment`: controlled empirical evaluator.
- `coverage bounds`: statistical lower bounds. New code does not use “certificate” as the main framing.

Existing split proportions remain unchanged during the first migration:

- Synthetic: 40% prediction, 20% transport, 40% calibration.
- Clinical: 40% prediction, 15% transport, 30% calibration, 15% environment.

All splits remain patient-level. A patient cannot appear in multiple roles.

## 4. Sequential estimand

For patient \(i\) and stage \(t\), define before observing the stage outcome:

- \(A_{it}=1\): the patient is still in the episode and at risk for a prediction.
- \(O_{it}=1\): the required outcome is observed in the response window.
- \(E_{it}=A_{it}O_{it}\): the prediction is evaluable.
- \(Z_{it}=1\{Y_{it}\in C_t(H_{it})\}\): the prediction set covers the joint outcome.

The primary stagewise estimand under target policy \(\pi\) is

\[
c_t^\pi
=
P_\pi(Z_t=1\mid A_t=1,O_t=1).
\]

This is explicitly conditional on both being at risk and having an observed outcome. Administrative termination after a stage is neither a hit nor a miss. Outcome non-observation is not silently reclassified as termination.

Every stagewise result must report:

- at-risk count \(n_t^A=\sum_i A_{it}\);
- observed/evaluable count \(n_t^E=\sum_i E_{it}\);
- at-risk rate and observation rate;
- weighted effective sample size when off-policy weights are used;
- a supported/unsupported indicator.

The supported set is frozen without looking at coverage hits:

\[
\mathcal T_{\rm sup}
=
\left\{
t\le T_{\max}:
n_t^E\ge 200,
\frac{n_t^E}{n_0^A}\ge 0.05,
\operatorname{ESS}_t\ge 100
\right\}.
\]

For unweighted fresh rollouts, the ESS condition is omitted. Supported stages must form a prefix; once a stage becomes unsupported, every later stage is reported as unsupported even if a noisy later count crosses a threshold.

## 5. Coverage and width summaries

The primary result is the full supported-stage curve \(\{\widehat c_t\}\), a simultaneous one-sided 95% lower band, and the risk-set/observation curves.

Two scalar summaries are co-reported:

### 5.1 Transition-micro coverage

\[
\widehat C_{\rm micro}
=
\frac{\sum_{i,t}E_{it}Z_{it}}
     {\sum_{i,t}E_{it}}.
\]

Longer observed episodes contribute more transitions.

### 5.2 Patient-weighted coverage

\[
\widehat C_{\rm patient}
=
\frac{1}{N_+}
\sum_{i:L_i^E>0}
\frac{\sum_tE_{it}Z_{it}}{L_i^E},
\qquad
L_i^E=\sum_tE_{it}.
\]

Each evaluable patient receives equal weight. Confidence intervals for micro and patient-weighted summaries use patients as clusters.

The safety guardrail is

\[
G_{\min}=\min_{t\in\mathcal T_{\rm sup}} L_t,
\]

where \(L_t\) is a simultaneous one-sided lower confidence bound. Raw `mean(stage coverage)` and raw `min(stage coverage)` remain diagnostic fields only.

Normalized width uses the identical evaluable mask:

\[
W_{\rm micro}
=
\frac{\sum_{i,t}E_{it}W_{it}}
     {\sum_{i,t}E_{it}}.
\]

A patient-weighted normalized width is computed by first averaging active widths within patient and then averaging patients. Mean log volume, median volume, score mean, and controlled clinical cost also exclude non-evaluable padding.

Because policy-dependent termination changes the survivor population, every method also reports a standardized-width sensitivity analysis on one frozen reference risk-set distribution. This sensitivity result is secondary and cannot replace the policy-specific primary estimand.

## 6. Phase 0: structural oracle gate

### 6.1 Compared methods

1. **Current Profiled Oracle.** For each seed, learn the current profile \(b_t\) only from the existing transport split. Keep the exact existing 101-scale family. On independent oracle-tuning rollouts, evaluate every full schedule \(s_kb_t\) and select the minimum-width schedule whose supported-stage simultaneous lower bounds reach 0.90.
2. **Fully Sequential Oracle.** Starting with an empty prefix, evaluate all 101 active-score quantile candidates from 0.50 to 0.999 at stage \(t\) under the occupancy induced by the frozen prefix. Evaluate all candidates; do not stop at the first failure because performative coverage need not be monotone in radius. Select the minimum-width candidate whose stage lower bound reaches 0.90, append it to the prefix, and continue.
3. **Standard CP.** Reported only as a descriptive reference. It is not the denominator of the Go/No-Go comparison.

The structural comparator is therefore “the current learned profile with oracle scale selection,” not the deployed practical SC-PCP record and not an obsolete shared scalar method.

### 6.2 Monte Carlo separation

For every seed:

- candidate construction/tuning uses 5,000 target-policy rollouts per candidate;
- selected schedules are evaluated with a separate 50,000-rollout batch;
- the two oracle methods use matched random-number streams where the simulator permits it;
- tuning and final evaluation seeds are disjoint and stored in metadata;
- no final-evaluation result is used to alter a schedule.

Phase 0 uses 100 paired synthetic seeds. The existing standard synthetic scenario and a separately labeled `tail_shift` scenario are both run. `tail_shift` retains the observed difficulty state and makes that state control residual tail shape; it does not replace the original simulator.

Phase 0 remains all-active with \(T=12\). Its purpose is structural schedule efficiency, not validation of variable-length handling.

### 6.3 Tail-shift mechanism

Add an observed binary difficulty state \(H_t\) whose transition depends on the current observed state and treatment action. Conditional residuals follow

\[
\epsilon_t\mid H_t=0\sim N(0,1),
\qquad
\epsilon_t\mid H_t=1
\sim0.9N(0,1)+0.1N(0,4^2).
\]

The frozen Gaussian outcome model receives \(H_t\) as an observed covariate but remains misspecified for tail shape. This creates the transparent chain

\[
q\rightarrow A_t\rightarrow H_{t+1}
\rightarrow\text{score-tail shift}_{t+1}
\]

without hiding \(H_t\) or replacing the standard scenario.

### 6.4 Pre-registered decision rule

Go requires all of the following on the `tail_shift` 100-seed paired experiment:

1. every supported stage has a simultaneous one-sided 95% lower bound at least 0.90;
2. the paired geometric mean ratio
   \(W_{\rm micro}^{\rm seq}/W_{\rm micro}^{\rm profiled}\le0.90\);
3. the 95% paired confidence interval for that ratio has upper endpoint below 1.00;
4. the patient-weighted width ratio is at most 0.92;
5. selection succeeds for at least 95 of 100 seeds;
6. on the standard scenario, the sequential oracle retains coverage and its micro width is no more than 2% worse than the profiled oracle.

Any failure is No-Go. Improvements between 0% and 10% are not promoted to the full clinical route. No-Go preserves the current best method and records the oracle result as a negative experiment.

## 7. Practical Sequential SC-PCP

This section is implemented only after Phase 0 Go.

### 7.1 Sequential occupancy transport

Define the active-state subdistribution

\[
\nu_t^\pi(ds)=P_\pi(A_t=1,S_t\in ds),
\qquad
\rho_t(s)=
\frac{d\nu_t^{\hat q_{<t}}}{d\nu_t^\mu}(s).
\]

Given candidate \(q\), let

\[
\lambda_{it}(q)
=
\frac{\pi_q(A_{it}\mid S_{it})}
     {\mu(A_{it}\mid S_{it})}.
\]

Estimate the risk-set-aware coverage curve using

\[
\widehat c_t(q\mid\hat q_{<t})
=
\frac{
\sum_iE_{it}\widehat\rho_t(S_{it})
\lambda_{it}(q)Z_{it}(q)
}{
\sum_iE_{it}\widehat\rho_t(S_{it})
\lambda_{it}(q)
}.
\]

After selecting \(\hat q_t\), fit the next occupancy head on transitions with \(A_{it}=A_{i,t+1}=1\):

\[
\widehat\rho_{t+1}(S_{i,t+1})
\approx
E_\mu\left[
\widehat\rho_t(S_{it})
\lambda_{it}(\hat q_t)
\mid S_{i,t+1},A_{i,t+1}=1
\right].
\]

`SequentialCOT` has one head per stage, does not take the current radius as a network input, and uses MSE because the target is a conditional mean.

### 7.2 Patient cross-fitting

The transport role uses three fixed patient folds. At stage \(t\):

1. train the occupancy head on two folds;
2. estimate the complete candidate curve on the held-out fold;
3. obtain one fold-specific selected log radius and held-out ESS;
4. aggregate fold log radii using an ESS-weighted median;
5. propagate the same aggregate radius to the next stage in all folds.

The weighted median is deterministic: sort log radii, use normalized nonnegative ESS weights, and choose the first value whose cumulative weight is at least 0.5. A fold with ESS below 100 contributes zero weight. If no fold is supported, the stage and suffix are unsupported and the method abstains.

The final schedule is produced entirely from the transport role. The untouched calibration role evaluates only that frozen schedule. It does not search another scale.

### 7.3 Prefix-IW baseline

Sequential Prefix-IW uses the same stage grids, cross-fit folds, selection rule, support gates, and final evaluation. The only difference is the occupancy estimator. This isolates the value of marginal COT from the value of stagewise schedule selection.

## 8. Variable-length data model

Extend `TrajectoryBatch` without breaking existing four-argument callers:

```python
@dataclass(frozen=True)
class TrajectoryBatch:
    states: Tensor
    actions: Tensor
    outcomes: Tensor
    patient_ids: Tensor
    at_risk_mask: Tensor | None = None       # bool [N, T]
    observed_outcome_mask: Tensor | None = None  # bool [N, T]
    done: Tensor | None = None               # bool [N, T]
```

`None` means all true for the first two masks and all false for `done`, preserving fixed-horizon behavior. Masks must be boolean and shape `[N,T]`. The at-risk mask must be prefix-monotone. `done` can be true at most once, must be at risk at the same stage, and makes all later stages not at risk. `evaluation_mask` is the conjunction of at-risk and observed-outcome masks. `lengths` is the at-risk count per patient.

All subset, device-transfer, prefix, concatenation, flattening, training, COT, simulator, metric, width, cost, and artifact paths propagate these fields. Inactive or unobserved padding must not affect fitted models or reported metrics.

## 9. Variable-length clinical reconstruction

Rebuild clinical caches from raw sources; current v17 complete-case caches are not reused for the primary variable-length experiment.

Use these fixed maximum administrative horizons:

| Dataset | Bin width | Maximum bins | Maximum administrative window |
|---|---:|---:|---:|
| MIMIC-IV | 4 h | 48 | 192 h |
| eICU | 4 h | 48 | 192 h |
| MIMIC-CXR | 6 h | 24 | 144 h after index CXR |
| INSPIRE | 10 min | 48 | 480 min |

At-risk status comes from the episode endpoint available before preprocessing:

- MIMIC-IV: ICU outtime, with discharge/death reason reported separately where available.
- eICU: unit discharge offset/status/location.
- MIMIC-CXR: ICU endpoint measured from the selected index-CXR origin.
- INSPIRE: anesthesia end; this is labeled procedure termination, not an absorbing clinical outcome.

Required response-window observation defines `observed_outcome_mask`. Missing measurement does not set `done`. Administrative truncation at the maximum horizon also does not set `done`.

The empirical evaluator uses a separate action/state/stage-conditioned termination model fitted only on the environment role. A sampled terminal transition ends the rollout and marks later stages inactive. Outcome/state donor libraries contain only evaluable active transitions. The result is still labeled a frozen model-based controlled evaluation.

The new cache schema is versioned independently of v17 and stores:

- cohort flow before and after each exclusion;
- at-risk, observed, censored, discharged, died, and administratively truncated counts;
- masks, termination reason, and administrative length;
- raw and supported risk-set counts by stage.

## 10. Simultaneous uncertainty

For a frozen practical schedule, construct patient-cluster bootstrap max-statistic lower bounds jointly over supported stages. Unsupported stages are serialized as `null` and never converted to zero or removed from their original index.

For oracle fresh rollouts, use trajectory-cluster resampling and a max-statistic over supported stages. Across 100 seeds, report paired seed bootstrap intervals for width ratios and seed-level variation. Candidate multiplicity during practical final evaluation is absent because the schedule is already frozen. Candidate curves used during schedule construction are not presented as formal guarantees.

An unbounded or data-selected horizon is outside the first implementation. All claims are for the pre-registered finite \(T_{\max}\). An anytime-valid extension requires a separate design.

## 11. Experiment matrix after Phase 0 Go

### RQ1: Does sequential coupling matter?

Methods:

- Standard CP;
- ACI;
- MFCS;
- MultiDimSPCI;
- PRC;
- Sequential Prefix-IW;
- practical Sequential SC-PCP.

The current profiled method is shown only as a frozen structural ablation in the sequential-radius figure and paired oracle analysis, not as a second active SC-PCP implementation.

### RQ2: Does sequential calibration reduce conservatism?

Plot stage versus radius for Historical CP, current profiled oracle, fully sequential oracle, and practical sequential SC-PCP. Pair it with at-risk and observed-count panels. Unsupported suffixes remain blank.

### RQ3: Does COT recover sequential occupancy?

Compare oracle occupancy, Sequential COT, and Prefix-IW for

\[
T\in\{4,8,12,16,24\}.
\]

Report ESS, population/MC \(L_1\) occupancy error, and coverage-estimation error.

### RQ4: When does performativity matter?

Run

\[
\beta\in\{0,0.5,1,2\}
\]

and report policy shift, occupancy shift, score shift, coverage shift, risk-set shift, and termination shift.

### Clinical main experiments

Run 20 prespecified seeds for MIMIC-IV, MIMIC-CXR, eICU, and INSPIRE. Run 100 seeds for each primary synthetic condition. Run the finite-MDP validation for 200 seeds. All final schedule evaluations use 50,000 fresh rollouts per seed.

## 12. Result schema and figures

Retain legacy fields for backward reading, but add:

- `coverage_estimand = risk_set_and_observation_conditional`;
- `at_risk_by_time`;
- `observed_by_time`;
- `supported_stage_mask`;
- `supported_stage_count`;
- `active_transition_count`;
- `transition_micro_coverage`;
- `patient_weighted_coverage`;
- `transition_micro_normalized_width`;
- `patient_weighted_normalized_width`;
- `minimum_supported_stage_lcb`;
- `termination_source` and termination counts;
- construction and evaluation RNG seeds.

Curves always retain length \(T_{\max}\). Unsupported values are JSON `null`. Renderers preserve stage indices, use NaN-aware aggregation, display the number of contributing seeds, and include a risk-set/observation panel. Main coverage tables use stagewise guardrail, micro coverage, patient-weighted coverage, normalized width, selection rate, and selected-and-covered rate.

## 13. Failure handling and resumability

- Every run writes to a new output root and never overwrites `results/work/paper_final`.
- Each seed writes atomically to its own directory.
- A setting becomes `COMPLETE` only when the exact prespecified seed set and method rows exist.
- A suite-level `COMPLETE` requires every setting to be complete.
- The new runner supports resume by validating and skipping complete seed directories. It never treats a partial seed as complete.
- GPU worker count starts at the current proven values: four per GPU for synthetic, two per GPU for MIMIC-IV/eICU/INSPIRE, and one per GPU for MIMIC-CXR. A one-seed memory/runtime smoke precedes every new dataset family.
- Source-tree hash, configuration hash, git commit, dirty-state digest, CUDA/Torch versions, and renderer hash are stored with the suite.

## 14. Testing strategy

All production changes follow red-green-refactor.

1. Data tests prove mask defaults, prefix invariants, propagation through subset/device/prefix/concatenation, and unchanged legacy behavior when all masks are active.
2. Hand-calculated toy arrays prove stagewise, micro, patient-weighted, supported-stage, pathwise, and width formulas. Extreme values in inactive padding must not change any result.
3. Weighted tests prove inactive scores/weights do not change Hájek estimates, ESS, bootstrap bounds, or selection.
4. Oracle tests use a tiny deterministic triangular environment where the known optimal schedule differs from every profiled schedule.
5. Sequential selection tests evaluate all candidates and cover non-monotone performative curves.
6. COT tests verify MSE conditional-mean fitting, active-transition filtering, fold isolation, deterministic ESS-weighted median, and unsupported suffix behavior.
7. Clinical construction tests retain short episodes, distinguish missing outcomes from termination, and reproduce endpoint masks.
8. Empirical-environment tests prove terminal transitions stop rollouts and later padding is inactive.
9. Artifact/renderer tests prove `[value, null, null]` does not shift stage indices and that old records remain readable.
10. Full-active regression tests reproduce the current fixed-horizon coverage and width formulas exactly.

Before any GPU run, the complete CPU test suite must pass with no failed tests. One-seed GPU smoke must then produce finite schedules, correct masks, disjoint tuning/evaluation seeds, and a complete result record.

## 15. Version retention and cleanup

Until Phase 0 is decided, no existing method is deleted.

After a Go and successful full verification:

- the active main branch contains one practical SC-PCP implementation: Sequential SC-PCP;
- Sequential Prefix-IW and the six task baselines remain explicit comparison methods;
- abandoned internal prototypes, global-inflation variants, and duplicate selector paths are removed;
- the current profiled implementation is preserved by its immutable commit and archived raw results, not as a second active SC-PCP code path;
- failed experiment outputs are moved outside the release result tree with manifests intact;
- only verified final tables/figures enter the release directory.

After a No-Go:

- the current profiled method remains the active method;
- the Phase 0 oracle implementation and complete negative result are retained on the experiment branch/archive for audit;
- no practical sequential or clinical migration code is merged into the active branch.

## 16. Acceptance criteria

The project is complete only when one of these terminal outcomes is reached:

### No-Go completion

- 100 paired seeds finish for both standard and tail-shift Phase 0 conditions;
- the pre-registered gate is evaluated from fresh rollouts;
- the current best method remains untouched;
- a reproducible negative-result report and raw manifest are saved.

### Go completion

- every Phase 0 Go condition passes;
- practical Sequential SC-PCP and Prefix-IW pass all unit/regression tests;
- variable-length clinical caches record cohort flow, masks, and termination sources;
- synthetic, stress, clinical, and finite-MDP prespecified seeds are complete;
- all main results use the new risk-set metric hierarchy;
- figures/tables pass completeness, numerical, and visual checks;
- one active best SC-PCP implementation remains in the release code path.
