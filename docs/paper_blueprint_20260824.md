# ICLR paper blueprint: Committed-Prefix SC-PCP

This blueprint converts the frozen method, held-out confirmation, and
post-confirmatory ablations into one claim-disciplined paper. It is a writing
plan, not a new experimental protocol.

The latest experiment authority is
[`formal_experiments_20260825.md`](formal_experiments_20260825.md). It adds an
exact finite-MDP audit, a fresh controlled all-six comparison, and a formal
orthogonal-copula NO-GO. The older two-method figure remains a mechanism
visualization; its approximately .9011 SC-PCP values are not the current
all-six ranking values.

The latest theory and robustness authority is
[`formal_experiments_20260826.md`](formal_experiments_20260826.md). It adds
horizon×overlap, calibration-size convergence, fixed-target propensity
sensitivity, and strict-split audits without changing canonical SC-PCP.
The same authority now also records the completed dataset-native controlled
clinical extension v2: MIMIC-IV passes the frozen gates and yields curves,
whereas eICU, INSPIRE, and MIMIC-CXR + IV/ED are K0 fidelity NO-GO results with
no scientific coverage rows. This extension does not overwrite the 2026-08-25
MIMIC v1 all-six signed study.

**Reporting convention.** Within the controlled semi-synthetic study only, we
use \(\gamma=-4\) as the default displayed hero stress case because it makes
the signed mechanism and correction visually explicit. This does not change
the frozen protocol: \(\gamma=-2\) remains the primary cell, \(\gamma=-4\)
remains the prespecified overlap-stress endpoint, the complete five-point
signed curve remains authoritative, and the production-style suite remains a
separate experiment family.

## Recommended title

**When Prediction Sets Change Decisions: Committed-Prefix Calibration for
Sequential Policies**

## One-sentence claim

SC-PCP uses the causal action prefix induced by already committed prediction
radii to calibrate each post-action score under the longitudinal policy that
those radii themselves induce.

## Draft abstract

Prediction sets are often treated as passive outputs, yet in longitudinal
decision systems their radii can alter actions, future states, and the
distribution of the scores being calibrated. Standard calibration under the
logging policy can therefore be misaligned with its own deployment
distribution. We formulate offline per-step marginal calibration under a
prediction-radius-dependent sequential policy and introduce SC-PCP, a
committed-prefix procedure that transports each stage's score law with the
cumulative action likelihood ratio induced by the selected radii. SC-PCP
greedily commits the locally narrowest empirically feasible radius and retains
the current-action ratio because the stage score is observed after treatment.
Under consistency, sequential exchangeability, positivity, uniformly
consistent behavior propensities, and uniform convergence over a compact
prefix-radius class, we establish asymptotic per-step marginal coverage for the
data-dependent selected schedule. Exact finite-MDP experiments show that both
history and current-action ratios are needed when both feedback channels are
active. A paired horizon--overlap study then shows that effective sample size
degrades as sequential depth and policy divergence increase, while a
calibration-size study shows the complete coverage surface becoming more
accurate with more logged trajectories. In a fresh controlled six-method
benchmark, we display \(\gamma=-4\) as the low-overlap hero stress case while
retaining \(\gamma=-2\) as the frozen primary cell and reporting the complete
signed curve. At \(\gamma=-4\), Standard CP reaches 86.37% WSC; SC-PCP raises
WSC to 89.83% (+3.46 percentage points) with a 1.202 width ratio, whereas MFCS
is point-eligible but wider. Across the full curve, SC-PCP makes a large
correction under negative shift and narrows under positive shift, but does not
uniformly attain 90%. Across five frozen production-style settings, SC-PCP lies
on the point-estimate
coverage--width frontier in all five and is the narrowest point-eligible method
in three.

## Three contributions

1. **Prediction-coupled longitudinal formulation and identification.** We
   formulate stagewise calibration when prediction radii determine a
   sequential policy, and identify every post-action score event with the
   complete committed action prefix. The current-action ratio is included
   because \(R_t\) is generated after \(A_t\); earlier ratios transport the
   state occupancy reaching \(t\).
2. **Committed-prefix selection and selected-schedule guarantee.** SC-PCP
   commits the locally narrowest empirically feasible stage radius using an
   unclipped cumulative likelihood product under a structurally ratio-capped
   policy. Uniform convergence over the compact prefix-radius class yields
   asymptotic per-step marginal coverage for the random selected schedule. The
   result does not require monotonic coverage, a unique solution, or an exact
   fixed point.
3. **Structural, convergence, and robustness evidence.** Exact M0--M3 finite-MDP cells
   isolate when current, historical, or full-prefix transport is identified.
   A fresh controlled all-six benchmark quantifies the resulting
   coverage--width trade-off; a separately frozen equal-marginal copula design
   is retained as a substantive-magnitude NO-GO. Paired horizon--overlap and
   calibration-size studies connect the estimator's finite-sample behavior to
   its asymptotic argument; propensity and strict-split audits expose nuisance
   and sample-reuse boundaries. In five production-style
   settings, SC-PCP is point-Pareto-efficient in all five and the narrowest
   point-eligible method in three.

## Paper structure

1. **Introduction.** Begin with the deployment mismatch
   \(q\rightarrow\pi_q\rightarrow P_q(R_t)\), not with importance weighting.
   Explain why a post-action score needs both current and historical action
   ratios, then state the held-out signed result and the asymptotic guarantee.
2. **Related work.** Separate fixed-target off-policy conformal prediction,
   feedback/policy-control conformal methods, and longitudinal sequential
   importance transport. Phrase novelty as a distinction, never as broadly
   first.
3. **Problem formulation.** Define logged trajectories, the frozen normalized
   score, the radius-dependent stochastic policy, stagewise target coverage,
   and the primary estimand

   \[
   \operatorname{MarginalWSC}
   =\min_t\mathbb E_D[C_t(\widehat q_{0:t})].
   \]
4. **Committed-Prefix SC-PCP.** Give prefix identification, Hájek coverage and
   width surfaces, the sequential selector, availability, and the true
   \(O(NTKA)\) candidate-evaluation complexity. Do not claim equivalence to a
   global \(K^T\) optimizer.
5. **Theory.** Prove identification, uniform convergence of the fitted-weight
   Hájek surface over the complete compact prefix class, and empirical
   feasibility at the random selected schedule:

   \[
   C_t(\widehat q_{0:t})
   \ge 1-\alpha-\Delta_n,
   \qquad \Delta_n=o_p(1).
   \]
   State separately why this is not an exact weighted-conformal rank theorem.
6. **Experiments.** Present exact finite-MDP identification, the controlled
   all-six trade-off, the frozen five-setting comparison, horizon×overlap and
   calibration-size diagnostics, and then propensity/strict-split robustness.
   Keep the orthogonal copula NO-GO and current/history/COT/DR diagnostics as
   boundary evidence rather than additional methods.
7. **Limitations and discussion.** Put the semi-synthetic boundary, natural
   weak shift, causal assumptions, low-overlap endpoint, and greedy local
   optimality in the main paper rather than hiding them in an appendix.

## Main figures and tables

The complete file map, captions, source bundles, and rendering commands are in
[`figure_portfolio_20260826.md`](figure_portfolio_20260826.md). The recommended
main-text order is:

1. **Problem and method schematic.** Use
   [`figure_method_schematic.pdf`](../results/paper_method_schematic_20260826/figure_method_schematic.pdf)
   to separate history-mediated occupancy from the current-radius action and
   post-action score, then show complete-prefix transport and stagewise commit.
2. **Exact population identification.** Use
   [`figure_exact_prefix_identification.pdf`](../results/paper_formal_mechanism_20260826/figure_exact_prefix_identification.pdf).
   The M0--M3 heatmap makes the current/history/full-prefix structure visible;
   these rows are transport diagnostics, not baseline methods.
3. **Coverage--width Pareto result.** Use
   [`figure_main_pareto.pdf`](../results/paper_main_suite_figures_20260826/figure_main_pareto.pdf).
   It contains all six canonical methods and supports precisely `5/5 point
   Pareto, 3/5 narrowest eligible`, not universal SOTA.
4. **Dataset-native gated controlled-stress grid.** Use
   [`figure_controlled_stress_grid.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_controlled_stress_grid.pdf).
   Keep Synthetic native \(\beta=2\) as a separate stratum. MIMIC-IV clinical
   v2 \(\gamma=-4\) is the only clinical curves panel: Standard CP reaches
   86.36% WSC and SC-PCP 90.09% (+3.73 pp; width ratio 1.204), but the SC-PCP
   CI crosses 0.90. eICU, INSPIRE, and MIMIC-CXR + IV/ED remain explicit K0
   NO-GO gate cards with no invented method values. This figure must not be
   described as five comparable \(\gamma=-4\) curves.
5. **Controlled signed mechanism and all-six trade-off.** Use
   [`figure_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/figure_controlled_signed_all_six.pdf).
   All panels come from the formal all-six artifact. Interpret the full signed
   curve immediately after the displayed \(\gamma=-4\) slice; retain the
   figure's truthful `primary` label at \(\gamma=-2\) and `stress` label at
   \(\gamma=-4\). The older two-method PDF remains historical mechanism
   evidence and must not supply ranking values.
6. **Theory-facing diagnostics.** Use
   [`figure_theory_diagnostics.pdf`](../results/paper_theorem_robustness_20260826/figure_theory_diagnostics.pdf)
   for horizon×overlap and \(n_{\rm cal}\) behavior. It shows statistical cost
   and surface recovery, not a proved finite-sample rate or exact convergence
   to 0.90.

The propensity/strict-split
[`figure_robustness_audits.pdf`](../results/paper_theorem_robustness_20260826/figure_robustness_audits.pdf)
belongs in the appendix.
The complete production/native
[`figure_stagewise_profiles.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf)
also belongs in the appendix; it preserves all five RQ1 settings and must not
be merged with the controlled \(\gamma=-4\) data-generating mechanism. The
older three-setting compact render remains valid history, not the all-five
figure.

### Table 1: frozen five-setting comparison

Report exactly the six canonical methods with WSC, MeanCov, normalized width,
and Selection. Bold only the narrowest method satisfying point WSC \(\ge.90\)
and Selection \(\ge.95\). The current renderer splits this logical table into
the synthetic and clinical single-page PDFs under
[`paper_marginal_final_20260822`](../results/paper_marginal_final_20260822).
Disclose that ACI, SPCI, and PRC receive 2,000 target-policy trajectories.

### Table 2: controlled signed all-six comparison

Use
[`table_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/table_controlled_signed_all_six.pdf),
which lists all 5 signed strengths × 6 methods with WSC, MeanCov, width,
Selection, and intervals. The message is large correction relative to Standard
CP under negative shift, slight residual undercoverage, and no universal
coverage--width winner. Disclose the 0-versus-2,000 target-adaptation budgets.
If space requires a main-text single-cell slice, title it `Displayed stress
slice (gamma = -4)` and keep the full 5×6 table in the supplement; never relabel
that slice as the primary result.

### Appendix: post-confirmatory structural ablations

The artifact `results/work/controlled_prefix_ablations_confirm20_20260824`
remains useful for showing the consequences of deleting current or historical
ratios. It must stay separate from the exact population heatmap and from the
six-method baseline tables.

## Closest-work boundary

- **[COPP (AISTATS 2023)](https://proceedings.mlr.press/v206/zhang23c.html):** historical-policy data and sequential target-policy
  prediction, but the target policy is given. SC-PCP calibrates stagewise
  post-action scores when the unknown radii themselves parameterize the target
  policy.
- **[Conformal Policy Control (2026)](https://arxiv.org/abs/2603.02196):** already treats calibration-selected
  policies and multi-round feedback with stronger finite-sample policy-control
  guarantees. SC-PCP's remaining distinction is within-patient longitudinal
  treatment dynamics, free stagewise radii, and the causal current-plus-history
  prefix.
- **[PC-RACP (2026)](https://arxiv.org/abs/2607.02206):** already studies prediction-set-induced actions and
  policy-coupled coverage with finite-sample weighted conformal calibration in
  a contextual decision problem. SC-PCP addresses repeated stochastic actions
  and state-occupancy transport, but has a weaker asymptotic guarantee.
- **[Performative Risk Control (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6c71e8beb41e142e463b16818537ed0-Abstract-Conference.html):** iteratively collects shifted data for a
  scalar threshold. SC-PCP is an offline, zero-target-feedback construction
  from logged longitudinal trajectories.

The manuscript must not claim to be the first feedback, policy-coupled,
off-policy, or multi-round conformal method.

## Limitations paragraph

Our guarantee is asymptotic and per-step marginal; it is neither an exact
weighted-conformal rank guarantee nor a finite-sample or data-conditional
certificate. Identification relies on consistency, sequential exchangeability,
positivity, and a correctly specified or uniformly consistent behavior
propensity, assumptions that cannot be verified from observational clinical
data alone. The signed benchmark is deliberately semi-synthetic and
calibration-aligned: it demonstrates a controlled performative-treatment
mechanism but does not estimate a clinical treatment effect or establish that
the mechanism is naturally strong in the source cohorts. Prefix likelihood
ratios may degenerate with horizon or weak overlap, as reflected by the low
effective sample size at the extreme \(\gamma=-4\) endpoint and the paired
\(T=20,\mathrm{TV}=0.15\) diagnostic. Severe propensity misspecification also
reduces overlap and can create stage-specific discrepancies even when WSC
remains nominal. Finally, the
committed-prefix selector is locally width-minimizing and does not solve a
global schedule optimization problem; in weak-shift settings, Standard CP or
another baseline may remain narrower.

## Remaining submission work

1. Subject the theorem in [final_method.md](final_method.md) to an independent
   proof review and transfer the complete assumptions into the manuscript.
2. Add a closest-work comparison table with primary citations and avoid broad
   priority claims.
3. Integrate the completed 2026-08-26 diagnostics without claiming a proved
   finite-sample rate, propensity double robustness, strict-split equivalence,
   or exact nominal convergence.
4. Report the independently frozen orthogonal copula benchmark as a NO-GO and
   keep the full low-overlap and negative results in the supplement; do not
   delete seeds, clip the cumulative weights, or add gamma-specific buffers.
