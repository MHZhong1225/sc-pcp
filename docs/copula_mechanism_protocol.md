# Equal-marginal copula mechanism protocol

This is an isolated theorem-facing benchmark. It does not call or modify the
canonical `SC-PCP` implementation and does not produce paper-method rows.

## Frozen mechanism

The observed state is \(S_t=(H_t,t/T)\), where \(H_t\in\{0,1\}\) is the
easy/hard regime. The logging policy is

\[
\mu(A_t=1\mid H_t)
=\epsilon+(1-2\epsilon)\sigma(b_0+b_HH_t).
\]

For a radius \(q\), only the target action logit changes:

\[
\pi_q(A_t=1\mid H_t)
=\epsilon+(1-2\epsilon)
\sigma\{b_0+b_HH_t+\kappa\lambda r(q)\}.
\]

The beta-specific transition kernel is shared exactly by source and target:

\[
P_\beta(H_{t+1}=1\mid H_t,A_t)
=\sigma\{c_0+c_HH_t+\beta(2A_t-1)\}.
\]

There is no \(q\) or \(\kappa\) argument in the kernel implementation. Thus the
only path from the radius to the outcome law is

\[
q\rightarrow\pi_q\rightarrow A_t\rightarrow H_{t+1}.
\]

Given \(H_{t+1}\), draw independent \(E_1,E_2\sim N(0,1)\) and set

\[
Z_1=E_1,\qquad
Z_2=\rho(H_{t+1})E_1+
\sqrt{1-\rho(H_{t+1})^2}E_2,
\]

with $\rho(0)=0.90$ and $\rho(1)=0$. Each coordinate is therefore exactly
\(N(0,1)\) conditional on every observed regime/action cell. Only the
cross-outcome correlation changes. The conformity score is

\[
R_t=\max(|Z_{t,1}|,|Z_{t,2}|).
\]

Q90 uses the empirical-left rank `ceil(0.9*n)-1`, without interpolation.

## Frozen factorial design and gate

- \(\kappa\in\{0,0.5,1\}\).
- $\beta\in\{-1,-0.5,0,0.5,1\}$.
- Radii are (1.70,1.90,2.10); the primary radius is 1.90.
- Late stages are zero-based stages 4--11.
- Formal seeds are the untouched even bank 94,000--94,198.
- Each formal seed uses 50,000 paired common-random-number trajectories.

Before any optional six-method study, the mechanism gate requires:

- the $\kappa=0$ and $\beta=0$ placebos to stay within 0.3 percentage points
  of coverage drift and 1% relative Q90 drift;
- signed late-stage relative Q90 shifts of at least 3%;
- signed late-stage same-radius coverage shifts of at least 1.5 percentage
  points;
- seed-paired 95% confidence intervals for all four signed Q90/coverage effects
  to exclude zero;
- every primary late-stage seed/stage cell to have prefix ESS fraction at least
  0.15, maximum incremental \(\pi/\mu\) ratio at most 10, and maximum normalized
  weight share at most 0.02;
- the worst (not average) marginal mean, variance, and correlation audit across
  every seed/factorial cell to pass.

If this original pre-probe v1 fails, its result is a formal NO-GO. The DGP or
gate must not be retuned using the formal result. `gate.json` authorizes the
canonical six methods only after every predeclared check passes.

## Frozen result (2026-08-25)

The 100-seed formal artifact is
[`results/work/copula_mechanism_v1_20260825`](../results/work/copula_mechanism_v1_20260825).
Its `gate.status` is **fail**: the signed effects are statistically nonzero, but
relative Q90 shifts are only 0.80--0.93% and coverage shifts only 0.30--0.35
percentage points, below the frozen 3% and 1.5-point magnitude thresholds.
All placebo, equal-marginal, copula, and overlap checks passed. The optional
six-method stage is therefore unauthorized and was not run. Full values and
hashes are recorded in
[`formal_experiments_20260825.md`](formal_experiments_20260825.md).

## Engineering-contamination disclosure

Before the strong gate was frozen, seeds 1 and 93,000 were used for engineering
probes. Viewed cells included correlations 0.99/0.999, effective beta extremes
of +/-3, policy logit shifts 2.0/2.3/5.0, and radii 1.75/1.80/1.90. Those probes
looked at hard-regime, Q90, coverage, and ESS outputs. None of those parameter
cells or seeds is eligible for the v1 confirmation. The machine-readable copy
of this disclosure is stored in every study manifest.

## Commands

Run the frozen two-GPU mechanism study:

```bash
conda run -n ucp python scripts/run_copula_mechanism.py \
  --config configs/copula_mechanism.yaml \
  --output-dir results/work/copula_mechanism_v1_20260825 \
  --devices cuda:0,cuda:1
```

Resume only the exact same source/config/seed protocol by adding `--resume`.
Malformed, partial, source-mismatched, config-mismatched, or payload-tampered
artifacts fail closed. Seed-to-device assignment is frozen from each seed's
global index in `config.seeds`, stored in the manifest, and checked again in
every seed metadata file, so resume cannot silently move a seed to another GPU.

The formal artifact is bound to source hash `7665dfbe...16643`. The active
tree later changed because of a resume-only maintenance fix in another runner;
bitwise reproduction of this artifact therefore requires the frozen source
snapshot documented in `formal_experiments_20260825.md`, not the current tree.

Before creating the output directory, the runner enumerates every formal base
RNG ID, scans existing result artifacts and declared source/config RNG IDs,
adds the coordinated controlled and finite-MDP reservations, and fails on any
intersection. The counts, set digests, explicit formal mapping, collision list,
and a digest of the full audit are frozen in `manifest.json` and rechecked at
study completion.

Downstream code must call `require_six_method_gate(root)` from
`scripts/run_copula_mechanism.py` before launching any method comparison.
