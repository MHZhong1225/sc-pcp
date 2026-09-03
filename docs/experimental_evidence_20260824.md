# SC-PCP experimental evidence ledger (2026-08-24)

> **2026-08-25 formal update.** A later fresh controlled study now contains all
> six canonical methods, and a separate exact finite-MDP study plus orthogonal
> copula gate have also completed. Their authoritative results are in
> [formal_experiments_20260825.md](formal_experiments_20260825.md). In that
> all-six artifact, SC-PCP WSC at \(\gamma=-4,-2\) is .898277/.897367—not the
> .9011 values from the earlier two-method artifact below. The earlier study
> remains valid mechanism evidence, but protocols and numbers must not be mixed.

> **2026-08-26 theory/robustness update.** Horizon×overlap, calibration-size
> convergence, propensity robustness, and strict-split studies are complete and
> independently audited. Their authoritative tables and claim boundaries are in
> [formal_experiments_20260826.md](formal_experiments_20260826.md). They do not
> change the canonical selector or retroactively upgrade any historical result.

> **2026-08-26 figure update.** The submission-ready deterministic figures and
> their exact artifact boundaries are indexed in
> [figure_portfolio_20260826.md](figure_portfolio_20260826.md). In particular,
> the current controlled ranking figure reads only the formal all-six artifact;
> the older two-method PDF below is retained as protocol-specific history.

## Bottom line

The evidence now supports two distinct statements.

1. In the frozen production-style synthetic and clinical suite, the natural
   prediction-mediated policy shift is weak. SC-PCP nevertheless returns a
   schedule in every run, has point-estimate marginal WSC at or above 0.90 in
   all five RQ1 settings, lies on the point-estimate coverage-width Pareto
   frontier in all five, and is the narrowest point-eligible method in three.
2. In a separate controlled semi-synthetic stress benchmark, prediction-induced
   treatment changes create a large signed shift in the target score law. On a
   fresh 20-seed all-six bank, Standard CP changes from severe undercoverage to
   overcoverage across the signed mechanism. SC-PCP corrects 3.46 and 1.86
   coverage points at the two negative endpoints and narrows at the positive
   endpoints, while retaining slight negative-endpoint undercoverage
   (.898277/.897367 WSC).

The second result establishes a calibration-relevant performative-treatment
mechanism. It does **not** establish that this mechanism is naturally strong in
MIMIC-IV or another ICU population. The controlled environment uses real
MIMIC-IV covariates and observed standardized residual innovations, but its
signed transition reweighting is deliberately semi-synthetic and
calibration-aligned.

The paper can therefore make a defensible claim about an offline longitudinal
calibration method and a controlled mechanism benchmark. It cannot claim
universal SOTA, finite-sample conformal validity, or a discovered clinical
causal effect.

## Metric and uncertainty conventions

- Coverage target: 0.90.
- Primary coverage metric:

  \[
  \operatorname{WSC}=\min_t\frac{1}{S}\sum_{s=1}^S C_{s,t}.
  \]

  This is never replaced by \(S^{-1}\sum_s\min_t C_{s,t}\) or by MeanCov.
- Paper-suite WSC intervals use complete-seed trajectory-vector bootstrap as
  documented in [evaluation_metrics.md](evaluation_metrics.md).
- Controlled-benchmark same-radius drift, Q90 shift, and width-ratio intervals
  use a paired 10,000-resample complete-seed bootstrap.
- Controlled-benchmark WSC audit bands are centered studentized max-t
  simultaneous bands across the 12 stage means. A lower band below 0.90 is not
  interpreted as evidence of failure; a stage whose simultaneous upper bound is
  below 0.90 is evidence of undercoverage.
- The unit of inference is the complete training/evaluation seed, not the
  20,000 or 50,000 Monte Carlo rollouts within a seed.

## 1. Frozen six-method paper suite

Artifact root:
[paper_marginal_final_20260822](../results/work/paper_marginal_final_20260822).
The suite contains 480 complete seed artifacts, 2,880 method rows, and exactly
the six canonical methods. Selection was observed in 100% of runs.

Values are `WSC / mean normalized width`.

| Setting | Seeds | Standard CP | ACI | MFCS | SPCI | PRC | SC-PCP | Narrowest point-eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Synthetic | 100 | .899322 / 1.83128 | .899602 / 1.83162 | .913848 / 1.90744 | .898322 / 1.83136 | .910592 / 1.89040 | **.901774 / 1.84359** | SC-PCP |
| MIMIC-IV | 20 | .898257 / 2.14564 | .898823 / 2.11133 | .907873 / 2.29127 | .897124 / 2.01482 | .904287 / 2.24315 | **.901176 / 2.18438** | SC-PCP |
| MIMIC-CXR + IV/ED | 20 | .901968 / 4.74854 | **.900052 / 4.64639** | .913168 / 5.05717 | .896268 / 4.61324 | .912319 / 5.03980 | .904014 / 4.78853 | ACI |
| eICU | 20 | .905597 / 2.11699 | **.903720 / 2.03384** | .920717 / 2.31568 | .897371 / 1.91537 | .916715 / 2.27346 | .908098 / 2.15251 | ACI |
| INSPIRE | 20 | .898424 / 2.44173 | .897995 / 2.40384 | .904041 / 2.60431 | .897997 / 2.30514 | .903063 / 2.57270 | **.900954 / 2.49813** | SC-PCP |

SC-PCP WSC bootstrap intervals are:

- Synthetic: [.900560, .902046]
- MIMIC-IV: [.898401, .904067]
- MIMIC-CXR + IV/ED: [.895567, .908900]
- eICU: [.902595, .912727]
- INSPIRE: [.898540, .902812]

Under the declared point-eligibility rule, SC-PCP is the narrowest eligible
method on Synthetic, MIMIC-IV, and INSPIRE. Its paired geometric width
reductions relative to the strongest point-eligible baseline in those settings
are 2.47%, 2.62%, and 2.88%, respectively. On MIMIC-CXR and eICU, ACI is
narrower. Thus the supported statement is “point-estimate Pareto frontier in
5/5 and best target-level efficiency in 3/5,” not “universally SOTA.”

Information budgets also differ: ACI, SPCI, and PRC receive 2,000 additional
target-policy trajectories in their online adapter evaluation. The table is a
comparison of final systems, not an equal-information estimator contest.

## 2. Existing controlled problem-value and Prefix-IW evidence

### Tail-shift problem value

Artifact:
[tail_shift_problem_value_20260821](../results/work/tail_shift_problem_value_20260821).
Across 100 seeds with 50,000 fresh rollouts per seed:

| Method | WSC (95% interval) | Mean width |
|---|---:|---:|
| Standard CP | .898650 [.897534, .899359] | 1.82905 |
| Greedy sequential oracle | .901508 [.900620, .902090] | 1.84077 |
| Profiled scalar oracle | .914016 [.912132, .914854] | 1.90776 |

This establishes that a small but detectable coverage correction is useful in
that frozen synthetic regime, and that an efficient correction exists near the
free stagewise oracle.

### Independent direct Prefix-IW confirmation

Artifact:
[marginal_prefix_iw_tail_shift_confirm100_20260821](../results/work/marginal_prefix_iw_tail_shift_confirm100_20260821).
This study used 100 unseen seeds (1000--1099), 3,000 calibration trajectories,
and 50,000 fresh rollouts per seed.

| Method | WSC | Width |
|---|---:|---:|
| Standard CP | .899203 | 1.83215 |
| Greedy oracle | .901840 | 1.84372 |
| Direct Prefix-IW | .901736 | 1.84354 |

The Prefix-IW simultaneous worst-stage band is [.900330, .903146]. Its paired
geometric width ratio to the greedy oracle is .999899. The minimum selected ESS
is 2704.47, the minimum candidate ESS is 2660.02, and there are no endpoint
selections. This is the strongest prior evidence that direct committed-prefix
IW is the appropriate transport engine. It is a diagnostic study from an older
source snapshot and is not used as a finite-sample validity claim.

## 3. Controlled signed performative-treatment benchmark

### Construction and estimand

The isolated runner is
[run_controlled_prefix_benchmark.py](../scripts/run_controlled_prefix_benchmark.py).
For every fixed signed strength \(\gamma\), source and target trajectories use
the same transition kernel \(K_\gamma\). Source trajectories use the logging
policy \(\mu\); target trajectories use the prediction-radius-dependent policy
\(\pi_q\). Consequently the trajectory density ratio between source and target
contains only the sequential action ratios, not an unmodelled
\(K_\gamma/K_0\) term.

The environment maintains static covariates and rolling-history coherence. It
combines a fitted next-state transition with observed D_env state residuals and
observed standardized outcome residuals. The donor difficulty is the stagewise
empirical rank of the frozen conformity score in D_env, and \(\gamma\) controls
the signed action-by-difficulty donor reweighting. This is deliberately a
calibration-aligned semi-synthetic stress mechanism; it is not an estimate of a
clinical treatment effect.

For Standard CP, the primary mechanism quantity is the same-radius gap

\[
\Delta C_t(q_{\rm std})
=
P_{\pi_{q_{\rm std}},K_\gamma}(R_t\le q_{{\rm std},t})
-
P_{\mu,K_\gamma}(R_t\le q_{{\rm std},t}).
\]

It cannot be explained by simply giving the target method a different radius.

### Frozen development

Artifact:
[controlled_prefix_benchmark_development20_20260824](../results/work/controlled_prefix_benchmark_development20_20260824).
This bank uses 20 development seeds that had previously been used in COT/DR
diagnostics. It selected the unchanged canonical selector and the moderate
primary mechanism endpoint \(\gamma=-2\); it is not confirmation evidence.

| \(\gamma\) | Standard target WSC | SC-PCP target WSC | Late same-radius Standard drift, pp [95%] | Target Q90 shift | SC/Standard width [95%] | Min selected ESS/n |
|---:|---:|---:|---:|---:|---:|---:|
| -4 | .859122 | .898750 | -3.460 [-3.751, -3.171] | +19.73% | 1.2196 [1.2028, 1.2363] | .0314 |
| -2 | .876897 | **.902190** | -1.772 [-1.933, -1.618] | +11.64% | 1.1598 [1.1459, 1.1736] | .1485 |
| 0 | .898015 | .898300 | +.122 [+.059, +.186] | -.96% | 1.0094 [1.0008, 1.0182] | .5472 |
| +2 | .907235 | .901700 | +.897 [+.777, +1.021] | -6.41% | .9616 [.9541, .9684] | .8441 |
| +4 | .907542 | .899607 | +1.069 [+.914, +1.228] | -8.10% | .9548 [.9462, .9630] | .8511 |

### Held-out confirmation

Artifact:
[controlled_prefix_benchmark_confirm20_20260824](../results/work/controlled_prefix_benchmark_confirm20_20260824).
The confirmation used the disjoint, previously unopened seed bank
12400, 12402, ..., 12438. Development and confirmation have the identical
experiment source hash
`23403dc6d0282a4b0c22e8894a5e4dbd7f523737454e049f969080c14f3dee0d`.
No selector, target, gamma value, or data budget was changed between them.

| \(\gamma\) | Standard source WSC | Standard target WSC | SC-PCP target WSC | Late same-radius Standard drift, pp [95%] | Q90 shift [95%] | SC/Standard width [95%] | Min selected ESS/n |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -4 | .898395 | .861487 | **.901087** | -3.443 [-3.738, -3.204] | +19.41% [+17.44, +21.66] | 1.2250 [1.2016, 1.2516] | .0338 |
| -2 | .896982 | .879432 | **.901060** | -1.775 [-1.915, -1.657] | +12.20% [+11.18, +13.40] | 1.1563 [1.1444, 1.1697] | .2154 |
| 0 | .897030 | .898360 | .899990 | +.131 [+.077, +.186] | -.92% [-1.36, -.49] | 1.0064 [1.0022, 1.0105] | .6124 |
| +2 | .898040 | .906357 | **.901310** | +.831 [+.720, +.972] | -6.13% [-7.15, -5.36] | .9671 [.9613, .9720] | .7073 |
| +4 | .898302 | .910190 | **.901297** | +1.155 [+.939, +1.447] | -8.67% [-11.00, -6.98] | .9527 [.9408, .9622] | .5982 |

The confirmatory result has three important features.

1. The performative mechanism is large and signed. At negative strengths,
   Standard CP loses 1.8--3.4 coverage points at the same historical radius;
   at positive strengths it becomes overconservative.
2. The unchanged SC-PCP selector holds WSC at approximately 0.90 over the
   complete signed curve. The value .899990 at \(\gamma=0\) is 0.001 percentage
   point below 0.90 and rounds to .9000; no practical threshold is introduced.
3. SC-PCP is bidirectional rather than an inflation rule. It expands relative
   to invalid Standard CP under negative shift, but is 3.3% and 4.7% narrower
   under \(\gamma=+2,+4\), where Standard CP is conservative.

For \(\gamma=-4\) and \(-2\), the simultaneous upper audit for the worst
Standard stage remains below 0.90, whereas no SC-PCP stage has a simultaneous
upper bound below 0.90. SC-PCP's width is only 2.4% and 2.1% above the
fixed-induced-policy target-Q90 response at \(\gamma=-4,-2\); across the signed
curve this diagnostic overhead is about 1.9--3.4%.

The extreme \(\gamma=-4\) cell is an overlap stress boundary. Its minimum
selected calibration ESS fraction is .0338 (about 101 effective trajectories
out of 3,000). This result must be reported, not hidden or repaired with a
post-hoc margin.

The paper figure, editable source, exact source-data CSV, and analysis JSON are:

- [controlled benchmark PDF](../results/paper_controlled_prefix_benchmark_20260824/figure_controlled_performative_benchmark.pdf)
- [working figure and source data](../results/work/controlled_prefix_report_20260824)
- [renderer](../tools/render_controlled_prefix_benchmark.py)

## 4. Post-confirmatory prefix and policy-coupling ablations

Artifact:
[controlled_prefix_ablations_confirm20_20260824](../results/work/controlled_prefix_ablations_confirm20_20260824).
This is explicitly a post-confirmatory explanatory study on the already opened
confirmation bank; it did not select or modify the canonical method. The runner
is hard-bound to the parent metadata, summary, source tree, and complete seed
bundle. All 500 seed--gamma--method records are finite and selected, and the
canonical rows reproduce the parent artifact exactly.

Values are `WSC / mean normalized width`.

| \(\gamma\) | Full SC-PCP | Without current ratio | Current-only | Frozen-policy Prefix-IW | One-step coupled Prefix-IW |
|---:|---:|---:|---:|---:|---:|
| -4 | **.901087 / 5.1001** | .879435 / 4.6990 | .879740 / 4.6794 | .897450 / 4.5067 | .882515 / 4.6015 |
| -2 | **.901060 / 3.6095** | .887890 / 3.4102 | .888130 / 3.3623 | .897907 / 3.3437 | .890217 / 3.3584 |
| 0 | .899990 / 2.2456 | .899010 / 2.2550 | .900435 / 2.2623 | .897107 / 2.2090 | .897840 / 2.2024 |
| +2 | .901310 / 1.7653 | .902337 / 1.7927 | .904075 / 1.8201 | .899095 / 1.5953 | .884472 / 1.6217 |
| +4 | .901297 / 1.6677 | .903070 / 1.6975 | .905952 / 1.7347 | .898102 / 1.3955 | .867055 / 1.4345 |

At the development-selected moderate endpoint \(\gamma=-2\), omitting the
current action ratio reduces WSC relative to full SC-PCP by 1.317 points
[paired 95% interval: 0.668, 1.517], while retaining only the current ratio and
discarding the committed history reduces it by 1.293 points [0.891, 1.521]. At
\(\gamma=-4\), the losses are 2.165 and 2.135 points. The current-only selector
has minimum selection ESS/n of .756 at \(\gamma=-4\), versus .034 for full
SC-PCP, yet still undercovers. Its failure therefore reflects omitted occupancy
transport rather than greater weight variance.

The fixed-versus-coupled contrast holds the calibration sample, Prefix-IW
radii, and selection statistics fixed. The frozen row deploys the same
\(\pi_{q_{\rm fix}}\) under which its radii were calibrated; the coupled row
instead deploys the policy induced by the calibrated radii. Coupled-minus-
frozen WSC is -1.493, -0.769, +0.073, -1.462, and -3.105 points at
\(\gamma=-4,-2,0,+2,+4\), respectively. The paired intervals exclude zero in
all four nonzero cells and include zero for the \(\gamma=0\) placebo. This is
direct evidence that a fixed-policy off-policy quantile cannot simply be fed
back into a policy-responsive deployment without recalibration.

The positive-shift rows provide an important qualification. Removing a ratio
can overcover rather than undercover, but it also produces wider sets and a
poorer target-Q90 response. The ablation claim is therefore not that every
ratio always increases coverage; it is that the complete current-plus-history
prefix is required for balanced bidirectional calibration. The frozen-policy
row is a Hájek quantile diagnostic, not an exact COPP implementation or a
finite-sample weighted-conformal result.

## 5. COT and DR estimator diagnostics

These studies explain why the final method uses direct Prefix-IW; they are not
additional paper methods.

### COT

Artifact:
[fixed_schedule_cot_probe_replication20_20260824](../results/work/fixed_schedule_cot_probe_replication20_20260824).
Prefix-IW versus COT CDF error at \(\gamma=0,-2,-3,-4\) is
.011555/.011648, .012889/.013944, .014058/.017376, and
.014777/.019134. Corresponding Q90 errors are
.035325/.036334, .049948/.051945, .059165/.064000, and
.063587/.071693. COT becomes less accurate as shift strengthens.

### Sequential DR

Artifact:
[sequential_dr_probe_dev20_20260824](../results/work/sequential_dr_probe_dev20_20260824).
DR-to-Prefix Q90 error ratios at \(\gamma=0,-2,-3,-4\) are
1.0117, 1.0009, 1.0400, and .9976; every 95% upper interval exceeds 1.
The frozen study decision is NO-GO. The evidence supports using direct
Prefix-IW instead of promoting COT or DR.

## 6. Supported and unsupported paper claims

### Supported

- SC-PCP targets the score law induced by a prediction-radius-dependent,
  sequential treatment policy using committed trajectory-prefix action ratios.
- The current action ratio is necessary because \(R_t\) is observed after
  \(A_t\); prior committed ratios transport the state occupancy reaching stage
  \(t\). The post-confirmatory ablations empirically isolate both components.
- A fixed-policy Prefix-IW quantile is not self-consistent after its calibrated
  radius is allowed to induce a different deployment policy; the nonzero
  signed cells and the \(\gamma=0\) placebo isolate this coupling.
- In the held-out controlled signed benchmark, Standard CP experiences a
  large, reproducible, same-radius coverage drift. The later all-six artifact
  shows that SC-PCP substantially corrects negative-shift undercoverage and
  adjusts width bidirectionally, while retaining slight residual undercoverage
  at the two negative endpoints.
- Direct Prefix-IW is empirically more reliable than the investigated COT and
  DR estimators.
- In the frozen five-setting paper suite, SC-PCP is on the point-estimate
  coverage-width frontier in all settings and is the narrowest point-eligible
  method in three.

### Unsupported or unsafe

- Universal SOTA or superiority on every dataset.
- A naturally strong clinical performative-treatment effect in MIMIC-IV,
  eICU, INSPIRE, or MIMIC-CXR.
- A clinical causal treatment-effect interpretation of the controlled donor
  transition mechanism.
- Exact weighted conformal, finite-sample, distribution-free, PAC,
  data-conditional, or action-conditional validity.
- A globally optimal schedule, an exact global fixed point, or an optimization
  equivalence that reduces the same \(K^T\) problem to \(TK\).
- A first method for conformal prediction under feedback, off-policy conformal
  prediction, or policy-coupled conformal prediction broadly construed.

## 7. Recommended paper story

The cleanest story is:

1. **Problem.** A prediction set changes a sequential treatment policy; current
   and past actions therefore change the distribution of the conformity score
   to be calibrated.
2. **Identification.** Under sequential ignorability, consistency, positivity,
   and a shared transition/outcome kernel, the target stagewise score law is
   identified by the committed prefix product

   \[
   W_t(q_{0:t})=\prod_{h=0}^t
   \frac{\pi^{q_h}(A_h\mid S_h)}{\mu(A_h\mid S_h)}.
   \]

3. **Method.** SC-PCP greedily commits one locally width-minimal empirically
   feasible radius per stage using the uncapped cumulative prefix product under
   a structurally ratio-capped target policy.
4. **Guarantee.** A uniform convergence theorem over the complete compact
   prefix-radius class yields asymptotic per-step marginal coverage for the
   data-dependent selected schedule. It is not an exact weighted-conformal rank
   theorem.
5. **Evidence.** Natural production-style shifts are weak; the controlled
   signed benchmark isolates when transport matters and confirms that SC-PCP
   corrects undercoverage and avoids unnecessary overcoverage. COT/DR
   diagnostics motivate direct Prefix-IW.

## 8. Remaining solid-ICLR gates

The held-out confirmation closes the main “does the method work under material
performative shift?” gap. Two associated documentation gaps have also been
closed in [final_method.md](final_method.md): it now gives the actual
ratio-capped policy projection, distinguishes that structural cap from the
unclipped cumulative calibration product, and states the identification plus
uniform selected-schedule validity argument with empirical-grid reuse, fitted
propensity, availability, and positivity assumptions.

The theory/robustness studies requested after the 2026-08-25 formal bundle are
now complete: horizon×overlap, calibration-size convergence, propensity
sensitivity, and strict split are no longer experimental TODOs. The remaining
submission gates are writing/proof gates and should not trigger another round
of selector tuning.

1. Explain the boundary with COPP, feedback covariate shift, CPC, PRC, RAC, and
   policy-coupled conformal prediction. The novelty must be narrowly stated as
   an offline longitudinal prediction-radius/policy coupling with a causal
   committed-prefix construction.
2. Transfer the theorem and assumptions into the manuscript and subject them
   to an independent proof review; the workspace write-up is not itself peer
   validation.
3. The independently frozen equal-marginal copula benchmark has now completed
   with a formal NO-GO: its signed effect is nonzero but fails the predeclared
   practical-magnitude gates. Keep that negative result rather than retuning
   the DGP or thresholds.
4. Disclose both low-overlap diagnostics: the controlled \(\gamma=-4\) endpoint
   and the RQ5 (T=20,\mathrm{TV}=0.15) ESS decline. Avoid deleting seeds,
   clipping cumulative weights, or adding a gamma-specific buffer.
5. Report the RQ6 empirical decay as descriptive rather than a proved
   (n^{-1/2}) rate; report the fixed-target and end-to-end propensity layers as
   distinct estimands; do not promote strict split after seeing its result.
6. Do not alter the selector after this confirmation. Any new one-step,
   effective-rank, or cross-fitted variant requires a new protocol and new
   development/confirmation seed banks.

The controlled all-six baseline extension and exact finite-MDP M0--M3 audit are
also complete; they are no longer submission TODOs. The all-six study shows a
large correction relative to Standard CP under negative shift but slight
residual undercoverage, so it supports a trade-off claim rather than universal
SOTA or uniform finite-sample nominal coverage.

## Reproducibility

The manuscript-facing title, abstract, contribution statements, section
structure, and table placement are collected in
[paper_blueprint_20260824.md](paper_blueprint_20260824.md).

Controlled development:

```bash
conda run -n ucp python scripts/run_controlled_prefix_benchmark.py \
  --study development20 \
  --output-dir results/work/controlled_prefix_benchmark_development20_20260824 \
  --devices cuda:0,cuda:1
```

Held-out confirmation:

```bash
conda run -n ucp python scripts/run_controlled_prefix_benchmark.py \
  --study confirm \
  --output-dir results/work/controlled_prefix_benchmark_confirm20_20260824 \
  --devices cuda:0,cuda:1
```

Post-confirmatory explanatory ablations:

```bash
conda run -n ucp python scripts/run_controlled_prefix_ablations.py \
  --output-dir results/work/controlled_prefix_ablations_confirm20_20260824 \
  --devices cuda:0,cuda:1
```

Paper figure and source data:

```bash
conda run -n ucp python tools/render_controlled_prefix_benchmark.py \
  --development results/work/controlled_prefix_benchmark_development20_20260824 \
  --confirm results/work/controlled_prefix_benchmark_confirm20_20260824 \
  --work-output results/work/controlled_prefix_report_20260824 \
  --paper-output results/paper_controlled_prefix_benchmark_20260824
```

Validation completed after the controlled ablation artifact was frozen:

```text
718 passed in 135.77s
```

Focused renderer, controlled-method, and ablation tests also passed; the
independent prelaunch ablation review reported 16 focused tests and no P0/P1
blocker.

Earlier focused renderer and controlled-method tests:

```text
15 passed in 1.07s
```
