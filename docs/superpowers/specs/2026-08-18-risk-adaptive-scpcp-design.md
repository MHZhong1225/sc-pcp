# Risk-Adaptive SC-PCP SOTA Development Design

**Date:** 2026-08-18

**Status:** Phase 0C approved for implementation; later subprojects remain gated

**Base revision:** `59b7f1e608d59db431982e84864d28f81c309e79`

## 1. Objective and claim boundary

The objective is to develop a deployable sequential conformal method that is materially narrower than every eligible, coverage-valid deployable baseline without weakening stagewise coverage. The target is benchmark SOTA under a frozen literature cutoff, information regime, tuning budget, outcome model, score definition, and evaluation protocol. The work does not promise field-global SOTA in advance.

The fixed-horizon confirmatory success criteria are:

1. every marginal stage and every required prespecified risk-group cell has a simultaneous one-sided coverage lower confidence bound of at least `0.90`;
2. selection succeeds for at least 95 of 100 unopened confirmation seeds;
3. under tail shift, the all-seed fallback-system 100-pair geometric-mean normalized-width ratio relative to every coverage-valid deployable baseline is at most `0.92`;
4. the multiplicity-adjusted one-sided paired confidence-interval upper endpoint is below `1.00` for every eligible deployable baseline;
5. shared-score and end-to-end comparisons are both reported;
6. oracle diagnostics are excluded from SOTA rankings.

The prior seeds `0..99` and all Phase 0A artifacts are exploratory. They cannot be reused as confirmatory evidence. A failed candidate remains an auditable negative result and does not replace the current active method.

## 2. Program decomposition

The work is split into three independently gated subprojects.

### 2.1 Phase 0C: fixed-T joint-search attainability audit

Phase 0C keeps the existing predictor, normalized-max box score, stage grids, target policy, common-random-number tuning, and independent 50,000-rollout evaluation. It changes only schedule search.

The search starts from the profiled, greedy, and conservative upper-endpoint schedules. It performs deterministic forward and reverse cyclic coordinate sweeps. For coordinate `t`, it constructs complete schedules by replacing only `q_t`, replays the full trajectory under a common tuning-noise bundle, and selects the jointly feasible candidate with minimum full-trajectory micro normalized width; exact width ties use the lowest original grid index. An update is committed only when it strictly improves the incumbent. A small adjacent-coordinate block search is allowed only after the coordinate implementation passes exact reduced-problem checks. Continuous or large-grid results are labeled `best_found`, never global optima.

Phase 0C is a falsifier, not the proposed deployable method. Its reported tail ratio is joint-search width divided by the exact-current profiled-oracle width on paired seeds. On a fresh development bank, the scalar-radius route stops if both conditions hold:

- tail-shift width ratio remains above `0.92`; and
- doubling the prespecified search budget improves width by less than `0.5%` relative.

The base budget is the three fixed initializations with two forward-plus-reverse sweep pairs. The doubled budget keeps the same starts and uses four sweep pairs. Let `R_B` and `R_2B` be the paired geometric-mean joint-search/current-profiled tail micro-width ratios over the same 40 development seeds, and let `Delta_B=(R_B-R_2B)/R_B`. A checkpoint is coverage-valid only when its independent fresh evaluation has simultaneous seed-level stage lower bounds at least `0.90` in both scenarios. A decision requires coverage validity and 40/40 jointly feasible pairs; otherwise the scalar route stops as unavailable. If `R_2B > 0.92` and `Delta_B < 0.005`, the scalar route stops. If `R_2B > 0.92` and `Delta_B >= 0.005`, one final capped run with eight sweep pairs is allowed; the route then stops whenever its ratio remains above `0.92`. If the ratio is at most `0.92`, the result is labeled a promising oracle diagnostic and requires a separate practical-method design before any confirmation bank is opened. Phase 0C never reads a confirmation bank.

An improvement of less than one percentage point over greedy is recorded but is not used to justify further scalar-grid tuning. A one-seed measured GPU smoke precedes the development run and records wall time per sweep and peak VRAM. The runner enforces the fixed sweep cap and a preregistered wall-time cap. The implementation may cache the incumbent prefix state and replay only the changed coordinate and suffix, but a deterministic test must prove equivalence to complete CRN replay.

### 2.2 Risk-adaptive oracle screen

If the scalar family fails, test whether patient- and action-adaptive radii have enough attainable headroom. The screen uses only pre-outcome observable state, including the observed synthetic tail-risk state, and computes radii for every candidate action before action sampling. It compares:

- state-blind stage radii;
- prespecified observed-risk-group radii; and
- learned state-action tail-shape radii.

The screen uses oracle rollouts only to decide whether the representation family is worth implementing. It cannot enter a deployable SOTA table. Its comparator is the best coverage-valid scalar oracle found by Phase 0C; its primary quantity is the paired tail micro-width ratio on a separate oracle-development bank, with patient width reported as a diagnostic. Selection uses tuning rollouts and evaluation uses independent fresh rollouts. If the risk-adaptive oracle cannot attain a point ratio at most `0.92` while preserving simultaneous marginal and required-group coverage, the radius-rescaling route stops and the next design must change prediction-region geometry.

### 2.3 RA-SC-PCP: practical fixed-T method

For complete predecision history `X_t` and candidate action `a`, define

\[
q_t(X_t,a)=s\,b_{t,r(X_t)}\,g_{\theta,t}(X_t,a).
\]

Here:

- `g_theta` is a strictly positive learned tail-shape factor based on out-of-fold normalized residuals;
- `r(X_t)` is a prespecified low-dimensional risk stratum measurable using only information available immediately before the current action;
- `b[t,r]` is a stage-risk profile selected on transport/tuning data;
- `s` is a final global safety scale;
- the factorization is normalized so `g_theta` has unit geometric mean within each frozen reference `(stage, risk-stratum)` cell, and the geometric mean of all `b[t,r]` entries is one, making `s`, `b`, and `g_theta` identifiable.

Let `X_t` denote the complete predecision sufficient history used by the policy, risk-stratum rule, behavior propensity, tail model, and occupancy model. It may contain responses observed before stage `t`. The policy receives the complete action-specific radius vector before sampling an action. Its worst-case cost for action `a` uses the corresponding `q_t(X_t,a)`. The prediction region after sampling uses the same precomputed action entry. The current or future outcome, next state, current observation indicator, and current termination event may not enter the radius or action decision.

The selected-action score is

\[
V_t=\max_d\frac{|Y_{td}-\widehat\mu_d(X_t,A_t)|}
{\widehat\sigma_d(X_t,A_t)g_{\theta,t}(X_t,A_t)}.
\]

For a complete candidate `eta=(s,b)` with frozen `g_theta` and risk rule, define

\[
\rho_t^{\eta}(x)=\frac{dP_{\pi_{\eta}}(X_t=x)}{dP_{\mu}(X_t=x)},\qquad
w_t^{\eta}=\rho_t^{\eta}(X_t)
\frac{\pi_{\eta}(A_t\mid X_t)}{\mu(A_t\mid X_t)}.
\]

Sequential COT and Prefix-IW transport scores using these candidate-specific weights. Each complete candidate recomputes its policy and occupancy along the entire induced prefix. A scalar-radius COT head or weights fitted for another candidate cannot be reused. The implementation must use either patient-cross-fitted candidate-conditional COT, a COT explicitly conditioned on the full candidate parameter, or forward occupancy under a transition model frozen independently of candidate search.

First fit a raw positive tail factor `tilde_g_theta` from out-of-fold residuals. Define the pre-action risk index

\[
h_{\theta,t}(X_t)=\sum_a\widehat\mu_{\mathrm{ref}}(a\mid X_t)
\log \widetilde g_{\theta,t}(X_t,a).
\]

Stage-specific cutpoints are the one-third and two-thirds empirical quantiles of this raw index on `D_shape`, producing frozen low, middle, and high strata. Deployment always computes the stratum from raw `tilde_g_theta` and these raw cutpoints. Only after the stratum rule is frozen, compute the reference cell geometric mean `c[t,r]` and define `g_theta=tilde_g_theta/c[t,r]`; normalized `g_theta` is never fed back into the grouping rule. The oracle tail-shift screen additionally reports the simulator's observed binary difficulty group, but that oracle-only grouping never replaces the deployable learned strata.

## 3. Statistical estimands and support

For fixed `T=12`, the primary safety estimand is induced-policy stage-risk-group joint-box coverage:

\[
C_{t,r}^{\pi}=P_{\pi}\{Y_t\in\mathcal C_t(X_t,A_t)\mid r(X_t)=r\}.
\]

Every marginal stage and every required `(t,r)` must satisfy coverage at least `0.90`. Groupwise validity implies marginal validity only when every positive-mass stratum is included, so marginal stages are also gated directly. Transition-micro normalized width is primary efficiency; equal-patient width is co-primary and is retained even though the two coincide in the all-active fixed-horizon experiment.

The complete family of 12 marginal stages and 36 `(t,r)` cells per scenario is frozen before evaluation. Within a scenario, the same frozen RA-derived raw-tail stratifier and cutpoints are applied to RA and every baseline; baselines cannot define their own tertiles. Support is determined without looking at coverage hits. Each required `(t,r)` needs at least 200 raw evaluable rows and weighted effective sample size at least 100 on the relevant calibration/gate role. A required cell that lacks support before `D_eval` makes that candidate abstain for the seed; it is not deleted from the confidence family. Reports include target-policy stratum mass and `(t,r,a)` overlap diagnostics. Any target-positive action with zero logging support fails, and the stochastic behavior-anchored action-ratio cap remains mandatory.

Synthetic fresh-rollout evaluation provides Monte Carlo estimates and finite-Monte-Carlo confidence intervals conditional on the simulator and frozen policy; only finite-MDP enumeration is exact. Practical logged-data inference requires consistency, sufficient-state sequential exchangeability, action positivity, active-state domination, and a stable outcome/transition kernel. Estimated propensity and occupancy ratios make practical clinical guarantees asymptotic/model-dependent; reports must not call them unconditional finite-sample distribution-free guarantees. A practical guarantee additionally assumes that true ratios lie below the declared caps or includes a valid truncation-bias bound. Cap-hit rates above `0.01`, an unbounded sensitivity to the prespecified cap grid, or absent bias control makes the formal claim unavailable even if the sampling LCB passes.

## 4. Data isolation and seed policy

Patient or episode roles are disjoint:

1. `D_model`: outcome-model fitting;
2. `D_shape`: out-of-fold residual construction, tail-shape fitting, and frozen stratum cutpoints;
3. `D_transport`: patient-cross-fitted propensity/occupancy fitting, `b` search, the finite `s` grid, and all non-calibration hyperparameter selection;
4. `D_cal`: after `g`, strata, `b`, the `s` grid, and nuisance-fitting algorithms are frozen, apply the preregistered weighted calibration rule to select `s`;
5. `D_gate`: independently evaluate the one frozen calibrated candidate; failure causes whole-seed fallback and no reselection;
6. `D_eval`: final independent evaluation and confirmatory inference.

Cross-fitting may reuse modeling rows across folds but never places a row's outcome in the model used to score that row. Every propensity/COT weight used during candidate search is produced by a patient-level training fold that excludes that patient; pooled out-of-fold hits, weights, and ESS drive selection. After `eta` is frozen, nuisances are refit on full `D_transport` only for downstream use. Hyperparameter search never reads `D_cal`, `D_gate`, or `D_eval`; `D_eval` never changes the method.

Standard and tail-shift fit separate scenario-specific outcome, tail, profile, and scale parameters; they share architecture and hyperparameter budgets, not `eta`. The `D_cal` rule evaluates every frozen `s` candidate with its own full-prefix policy and candidate-specific out-of-sample weights from the nuisance models refit on `D_transport`. Within each scenario, a single patient-clustered one-sided max-t band with 10,000 deterministic bootstrap replicates covers every candidate times that scenario's 48 marginal/group cells at family alpha `0.025`; the two prespecified scenarios therefore spend at most `0.05` jointly. Among candidates whose complete band clears `0.90` and whose support checks pass, select minimum estimated end-to-end micro width with the lowest grid index as an exact tie-breaker. If none is feasible, abstain. This uniform candidate-by-cell band makes the finite search selection-valid for sampling variation conditional on correct nuisance weights; it does not remove nuisance-model or clipping bias.

`D_gate` evaluates only the single frozen candidate selected for its scenario. Using 10,000 deterministic patient-cluster bootstrap replicates, it constructs a one-sided max-t band over that scenario's 48 cells at alpha `0.025`. Any marginal or required-group LCB below `0.90`, raw count below 200, ESS below 100, cap-hit rate above `0.01`, or overlap failure triggers whole-seed fallback to current SC-PCP. `D_gate` may not inspect width, clinical utility, or another candidate and may not reselect. Once `D_eval` is opened, missing support or an undefined required cell makes the entire confirmation fail as unavailable; evaluation data never trigger fallback or alter deployment.

Seed banks are frozen as follows:

- historical exploratory bank: `0..99`;
- Phase 0C development bank: `10000..10039`;
- risk-adaptive oracle-development bank: `11000..11039`;
- unopened RA fixed-T confirmation bank: `20000..20099`;
- reserved, unopened geometry-route confirmation bank: `30000..30099`.

No route reads another route's opened confirmation outcomes. The RA bank is run once after source, configuration, baseline versions, seed manifest, and experiment hashes are frozen. For each primary registered comparator `j` and scenario `u`, use its 40 development paired log-width ratios, the conservative one-sided Bonferroni level `0.05 / (2B)` for `B` primary comparators across two scenarios, and the normal approximation

\[
\operatorname{Power}_{j,u}=\Phi\!\left(
\sqrt{100}\,|\log(0.92)|/\widehat\sigma_{j,u}-z_{1-0.05/(2B)}
\right).
\]

The minimum power across all comparator-scenario pairs must be at least `0.90`; a comparator absent from either development scenario cannot enter the primary registry. Otherwise confirmation is declared underpowered and does not run. Infrastructure-only reruns require the identical commit and configuration and retain the failed-run audit trail. The geometry bank remains unopened until a separate geometry-specific design, code, and baseline registry are frozen. If RA confirmation is opened and fails, subsequent geometry development is exploratory until that independent preregistration is complete; no bank is reused.

## 5. Baseline protocol

Comparisons are separated by information regime.

Logged-only primary competitors include stagewise split CP, current SC-PCP, full-depth MFCS, estimated Prefix-IW/WCP, and a CPC-style safe-reference adapter when the same bounded loss and policy family can be defined. Sequential WCP with known oracle ratios is reported separately as an unattainable information upper-bound diagnostic and is excluded from deployable rankings.

Online-feedback diagnostics include ACI, SAOCP/SF-OGD, Conformal PID, Online Conformal Prediction via Online Optimization, and PRC. They do not enter the logged-only SOTA gate until a separate protocol freezes episode reset, burn-in, feedback delay, update frequency, label budget, and the mapping from their native long-run coverage target to the required stage-risk-group family. SPCI and MultiDimSPCI remain temporal-dependence comparators and are not described as solving policy-induced shift.

The literature cutoff is 2026-08-18. Before `D_eval` is opened, a machine-readable baseline registry freezes every eligible method, source/commit, adapter status, search space, validation-query count, GPU cap, label budget, and operational fallback. Every primary method uses frozen current SC-PCP as its whole-seed fallback, must select its own method in at least 95 seeds, and therefore has a defined 100/100-pair operational-system comparison. Method-only common-pair comparisons require at least 95 seeds and remain secondary. All deployable competitors receive the same patient split, predictor, base score for the shared-score comparison, coverage target, final evaluator, and tuning budget. Original author implementations and recommended search spaces are preferred. A custom CPC mapping is labeled an adapter and enters the primary registry only if a preregistered checklist establishes the same bounded loss, policy family, and feedback budget.

Three views are mandatory. The shared-score view sets `g_theta=1` and uses the common normalized-max box score. The common-policy view recalibrates and evaluates all region methods under the same frozen reference action policy and common occupancy, isolating set geometry. The end-to-end view evaluates each complete induced decision system and is the confirmatory width table. The approved 8% material gate applies only to this end-to-end table, while the common-policy RA/current adjusted ratio must also have upper endpoint below one. Width normalization always uses method-external outcome scales frozen from `D_model`; it never divides by `g_theta`.

Because end-to-end width can improve by steering toward easier actions, it is paired with hard decision-system guardrails. Reports include action distributions, occupancy shift, ratio/cap diagnostics, and expected clinical cost. Let clinical cost use the existing disease, toxicity, and action-cost weights and standardize its paired difference by the reference-policy cost standard deviation frozen on `D_model`. The one-sided familywise 95% upper bound for RA minus current SC-PCP must not exceed `0.02` standard-deviation units.

End-to-end width, common-policy width, and clinical-cost noninferiority are three required intersection-union gates. Within each gate, use a seed-clustered one-sided max-t procedure with 10,000 replicates and fresh fixed RNG seed `2_718_281`. The end-to-end family contains every primary registry baseline in both scenarios; tail-shift additionally requires the `0.92` point threshold, while standard requires adjusted superiority. The common-policy family contains RA/current ratios in both scenarios. The utility family contains RA-minus-current standardized cost differences in both scenarios. No result may choose which scenario or comparator enters a family after evaluation.

## 6. Confirmatory inference

All paired analyses sort by seed and use identical paired seeds. The operational algorithm falls back to frozen current SC-PCP whenever RA calibration, support, or `D_gate` selection fails, so confirmatory coverage and width remain defined for all 100 seeds; RA-only conditional metrics are secondary. Width ratios are analyzed on the log scale. The report includes point ratios, paired bootstrap intervals, and a one-sided familywise multiplicity adjustment across the complete preregistered baseline family, whether or not a baseline later passes coverage. Every RA-only comparison requires at least 95 non-fallback paired seeds, while the primary material-effect gate always uses the 100-pair all-seed fallback system. The final pass requires the adjusted upper endpoint below one and the point ratio at most `0.92` for every coverage-valid eligible comparator.

The primary coverage family contains both scenarios jointly: 24 marginal-stage means and 72 required stage-stratum means. At family alpha `0.05`, a single seed-clustered one-sided max-t band with 10,000 deterministic bootstrap replicates covers all 96 means. Seeds, not rollouts or stages, are the resampling unit. The method must have at least 95 non-fallback selected seeds, but the primary band is computed for the all-seed fallback system. The coverage-valid status of every preregistered baseline is computed with the same frozen 96-cell rule and cannot alter the multiplicity family.

Coverage reporting keeps four noninterchangeable quantities:

- minimum over stages of the seed-mean stage curve;
- minimum simultaneous stage lower confidence bound;
- mean across seeds of each seed's stage minimum;
- raw minimum over seed and stage.

Only the simultaneous lower-bound quantity enters the coverage gate. Selection denominators, paired denominators, endpoint rates, support counts, ESS, maximum weights, and cap-hit rates are always shown.

## 7. Implementation isolation

The current `59b7f1e` implementation and final Phase 0A results remain immutable. New work is developed on a new branch/worktree and in isolated modules and configurations. Phase 0C may reuse the general schedule evaluator and frozen-schedule evaluator, but existing profiled and greedy paths remain regression-identical.

The first implementation plan covers only Phase 0C. The risk-adaptive oracle and practical method receive separate plans after the Phase 0C stop decision. Variable-length trajectories, clinical cache reconstruction, observation models, and clinical experiments remain out of scope until the practical fixed-T confirmation gate passes.

Every subproject uses test-driven development, deterministic tiny environments, CPU full-suite regression tests, a one-seed GPU smoke, atomic per-seed outputs, resumability, exact source/config/seed hashes, and an independent result/figure review. Production runs use the remote `ucp` environment and explicitly pinned GPUs.

## 8. Decision handling

If a gate fails, the corresponding route is `NO_GO`: retain the current active method and preserve source and complete negative artifacts. Work may continue only under the next representation route's own frozen design and unopened bank; an opened bank is never reassigned. Thresholds are not relaxed after results are observed. If all gates pass, the candidate becomes the best fixed-T benchmark version; this still does not authorize a clinical or field-global SOTA claim until variable-length and clinical confirmation are completed.
