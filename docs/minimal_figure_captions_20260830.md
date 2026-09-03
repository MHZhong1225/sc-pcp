# Minimal-text figures: manuscript captions (2026-08-30)

These captions carry the interpretation deliberately removed from the figure canvas. The
figures themselves retain only panel letters, axes and ticks, dataset or matrix labels, and
short method legends.

## Main signed-feedback results

### Figure: default signed-feedback stage profiles

Stagewise target-policy coverage and normalized prediction-set width at the prespecified
primary endpoint \(\gamma=-4\). Columns show Native Synthetic, MIMIC-IV, eICU and INSPIRE;
all panels contain the six canonical methods. The upper row reports
\(100(C_t-0.90)\) percentage points and pointwise two-sided Student-\(t\) intervals across
eligible seeds. The lower row reports the protocol-specific normalized width and its
pointwise Student-\(t\) interval. WSC is defined separately as
\(\min_t\operatorname{mean}_{s}(C_{s,t})\), not as the average of seedwise minima. eICU has
19 eligible runs, while availability and Selection Rate retain the denominator of 20
prespecified runs. MIMIC-CXR + IV/ED is absent because its controlled environment did not
pass the frozen pre-coverage fidelity protocol; no coverage or width curve exists for that
setting.

### Figure: complete signed-\(\gamma\) profiles

WSC, MeanCov and mean normalized width across
\(\gamma\in\{-4,-2,0,+2,+4\}\) for the four settings with authorized science rows. Error
bars are complete-seed-vector percentile intervals for WSC and two-sided Student-\(t\)
intervals for MeanCov and mean width. The \(\gamma=-4\) column is the prespecified
confirmatory method-comparison endpoint; the remaining signed points are descriptive
sensitivity analyses and do not create additional ranking claims. Width definitions remain
protocol-specific, so absolute width is not ranked across datasets.

### Table: complete metrics at \(\gamma=-4\)

For each dataset and canonical method, the table reports WSC and its 95% interval, first
zero-based worst stage, MeanCov and its 95% interval, mean normalized width and its 95%
interval, Selection Rate with Wilson interval and gate count, point/interval/eligibility
flags, and the frozen information budget. MIMIC-CXR rows are `NA` because its terminal
pre-coverage gate produced no method-level science values; `NA` is not zero coverage.

### Table: production/no-\(\gamma\) robustness

Complete six-method metrics in the five frozen production/native environments. These
results have no controlled \(\gamma\) and serve only as a robustness supplement. They are
not substituted for the primary signed-feedback experiment and do not repair the terminal
MIMIC-CXR controlled-environment gate.

## Mechanism, robustness and ablation evidence

### Figure: exact finite-MDP identification

Mean maximum absolute population coverage-surface bias over 500 paired finite-MDP
instances, crossing four feedback mechanisms with four transport diagnostics. Outlined
cells are exact to the frozen numerical tolerance. The comparison is an identification
diagnostic rather than a six-method benchmark or a finite-sample coverage guarantee.

### Figure: horizon, overlap and calibration size

Panels a--c summarize the exact-MDP horizon-by-policy-divergence grid: coverage deviation,
minimum selected-prefix ESS fraction and committed-surface sup-norm error. Panels d--f show
calibration-size sensitivity for full-prefix surface error, canonical WSC and normalized
width. Intervals and statistics are copied from the frozen artifacts; no bootstrap was
recomputed for rendering. The experiment quantifies overlap and estimation costs and does
not claim uniform superiority over baselines.

### Figure: propensity and strict-split robustness

Panels a--d compare oracle, correctly specified and reduced-state propensity arms using
propensity error, WSC, minimum ESS and target-law drift. Panels e--f report paired strict
minus canonical changes in WSC and geometric width for Synthetic, MIMIC-IV and the frozen
controlled \(\gamma=-2\) setting. Fixed-target-law and target-law-drift quantities are
distinct estimands. The strict split is a post-freeze audit, not a replacement estimator.

### Figure: committed-prefix ablation

WSC, normalized width, target Q90-to-radius ratio and minimum ESS fraction across signed
\(\gamma\) for the full committed-prefix method and four diagnostic
variants. These variants isolate the current-action factor, history contribution and
deployment-policy coupling. They are explanatory ablations, not additional canonical
baseline rows, and they do not authorize post-hoc changes to SC-PCP.
