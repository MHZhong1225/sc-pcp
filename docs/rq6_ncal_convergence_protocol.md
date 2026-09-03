# RQ6 calibration-size convergence: frozen protocol and engineering preflight

Status on 2026-08-26: **formal scientific study complete and independently audited**.
The immutable result root is
[`results/work/rq6_ncal_convergence_v1`](../results/work/rq6_ncal_convergence_v1),
and the authoritative table is in
[`formal_experiments_20260826.md`](formal_experiments_20260826.md). This study is isolated from the paper suite and does
not alter [`src/scpcp/marginal_prefix.py`](../src/scpcp/marginal_prefix.py), the
canonical selector, or any existing result artifact.  The canonical method and claim
boundary remain those in [`final_method.md`](final_method.md).

## 1. Question and two estimands

RQ6 asks whether the finite logged-sample behavior of committed-prefix calibration
converges as the total calibration budget grows.  It separates two estimands that
must not be mixed.

### Track A: fixed-population-grid transport surface

For each fixed MDP problem and logged resample, let

\[
\widehat C_{n,t}(q_{0:t})
=
\frac{\sum_i W_{i,t}^{q_{0:t}}\mathbf 1\{R_{i,t}\le q_t\}}
     {\sum_i W_{i,t}^{q_{0:t}}},
\]

and let \(C_t(q_{0:t})\) be the corresponding exact population probability from
finite-state recursion.  Track A reports

\[
E_n
=
\max_{t=0,\ldots,3}\;
\max_{q_{0:t}\in\mathcal G^{t+1}}
\left|\widehat C_{n,t}(q_{0:t})-C_t(q_{0:t})\right|.
\]

The fixed grid has \(K=7\).  Therefore the implementation evaluates exactly
\(7,49,343,2401\) unique prefixes at stages 0--3.  This is not a shortcut to a
smaller estimand: future radii do not affect a stage-\(t\) event, so these prefix
sets are exactly the union induced by all \(7^4=2401\) complete schedules.  The
focused regression test also compares this vectorized calculation cell-for-cell
with the existing all-schedule Hájek evaluator.

### Track B: canonical empirical grid and committed selector

For every \(n\), `D_COT` alone freezes the stagewise seven-point empirical-quantile
grids at quantiles 0.50--0.999.  `D_COT \cup D_cert` then enters the existing,
unmodified `select_marginal_prefix_schedule`.  If selection succeeds, the frozen
schedule is evaluated by exact population recursion.  The primary outputs are:

- selection availability;
- exact population coverage by stage and
  \(\min_t\operatorname{mean}_{\text{problem, logged}} C_t\), conditional on
  selection being available;
- exact population mean normalized width, conditional on availability;
- selected-schedule ESS, endpoint use, and nominal-target attainment.

Availability is reported separately.  An unavailable selector output is never
silently converted into zero coverage or removed without disclosure.

## 2. Frozen design

| Component | Frozen value |
|---|---:|
| Mechanism | exact finite-MDP `M3_full_feedback` |
| States / actions | 8 / 3 |
| Horizon | 4 |
| Fixed radius grid | 7 points from 1.40 to 3.50 |
| Target | \(1-\alpha=0.90\) |
| Total calibration sizes | 250, 500, 1,000, 2,000, 5,000, 10,000 |
| `D_COT:D_cert` | 1:2, nearest-integer `D_COT` split |
| Fixed MDP problems | 100 |
| Logged resamples per problem | 20 |
| Problem-cluster bootstrap | 10,000 shared resamples |
| Workers | 4 CPU processes by default |

The exact role budgets are `(83,167)`, `(167,333)`, `(333,667)`,
`(667,1333)`, `(1667,3333)`, and `(3333,6667)`.

The formal-config validator locks every scientific field in this table plus the
alpha, radius bounds, empirical-quantile bounds, policy TV, role ratio, all problem
and logged replication counts, every RNG start, bootstrap count, and namespace.
Only `output_dir`, `workers`, and `surface_chunk_size` are runtime overrides.

For a given problem/logged replicate, the runner simulates one independent maximum
`D_COT` pool of 3,333 trajectories and one maximum `D_cert` pool of 6,667
trajectories.  Every smaller budget uses the corresponding prefix of each pool.
Thus all six \(n\) values are paired through nested common random numbers, and the
maximum pool is never regenerated per \(n\).  Tracks A and B consume these same
role-specific prefixes.

## 3. Outcome-blind policy contract

The M3 transition and score mechanisms are retained, but the deployment policy
response is explicitly outcome-blind:

\[
\pi_q(a\mid s)
\propto
\exp\left\{
\log \mu(a\mid s)
+\lambda\,h(q)\,[1,0,-1]_a
\right\},
\qquad
h(q)=\operatorname{clip}\frac{q-1.40}{3.50-1.40}.
\]

The policy reads only the behavior probabilities, current state, and candidate
radius.  It never reads outcome means, predictor errors, scores, or future states.
The scalar \(\lambda\) is solved separately for each MDP so that the unweighted
state-mean TV from \(\mu\) is exactly 0.05 at the prespecified midpoint radius
2.45.  At radius 1.40 the target policy equals the behavior policy.  This
definition and its TV calibration are artifact fields and tested contracts.

## 4. Randomness and collision audit

The formal namespace is `rq6_ncal_convergence_v1:97000`.  The runner enumerates and
stores every one of its 4,101 RNG IDs:

- MDP problems: 97,000--97,099 (100 IDs);
- independent `D_COT`/`D_cert` streams:
  `97,100,000 + 2(20p+r)` and the following integer, for
  \(p=0,\ldots,99\), \(r=0,\ldots,19\) (4,000 IDs, ending at 97,103,999);
- shared problem-cluster bootstrap: 97,900,000 (one ID).

Before a formal launch, the runner inventories RNG declarations in source and RNG
IDs in prior artifacts, publishes the full label-to-ID mapping and hashes, and
fails on any collision.  It explicitly acknowledges the coordinated reservation
for RQ6 and excludes these other namespaces:

- 52,000--52,999: exact finite MDP;
- 91,000--91,999: controlled six-method study;
- 94,000--94,999: orthogonal copula;
- 96,000--96,999: RQ5 horizon/overlap;
- 98,000--98,999: propensity robustness;
- 99,000--99,999: strict-split audit;
- 100,000--100,999: future score robustness.

The current full-workspace audit passes with 4,101 unique formal IDs and zero
collisions.  The audit is rerun after all problem jobs; any change blocks final
publication.  Literal namespace-reservation declarations are excluded from the
actual-use scan and recorded through the coordinated-reservation table instead;
an actual `seed`/`rng` assignment or prior artifact still causes a collision.  A
mutual-reservation regression test checks this distinction.

## 5. Inference and paired reporting

The cluster is the fixed MDP problem, not one of the 2,000 logged resamples.
For each bootstrap draw, the runner resamples the 100 problem indices and retains
all 20 paired logged repetitions within each sampled problem.  The same 10,000 by
100 index matrix is shared across all \(n\) values and both tracks.  Track-B ratio
statistics aggregate numerator and denominator within each sampled problem before
forming coverage, width, or availability; WSC is then the minimum of the bootstrap
stage means.

The summary additionally reports the mean and median within-problem SD over the 20
logged repetitions for Track-A error, Track-B availability, conditional WSC, and
conditional width.  This distinguishes across-problem uncertainty from logged-data
variability.

The six-point log-log slope of mean Track-A error is retained only as
`descriptive_not_a_claimed_rate`; it is not interpreted as an asymptotic exponent.

## 6. Atomic artifacts and provenance

The formal output root is `results/work/rq6_ncal_convergence_v1`.  A fresh run fails
if that path already exists.  Each problem is written to an atomic
`problem_<seed>/` directory containing `result.json`, `metadata.json`, and a final
`COMPLETE` marker.  Every problem artifact binds its exact config hash, active
source-tree hash, problem index/seed, row count, and payload hash.

Resume validates complete payload structure, all 120 replicate-by-\(n\) cells,
role budgets, RNG formulas, policy TV, the `7/49/343/2401` Track-A prefix contract,
and Track-B grid/population fields.  Unknown problem IDs, abandoned temporary
directories, malformed artifacts, config changes, or source changes fail closed.
The root `COMPLETE` marker binds the manifest, engineering preflight, summary, and
a hash/size manifest for all 100 problem artifacts.

RQ6 also validates and binds the parent formal source snapshot before preflight,
launch, resume, and completion:

- manifest:
  `results/work/formal_source_snapshot_7665dfbe_20260825.manifest.json`;
- manifest SHA-256:
  `e6a1bba7f3be47d39357f212824e7720262e7d5212a14628e3b8981088c64e24`;
- archive SHA-256:
  `2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b`;
- parent source SHA-256:
  `7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643`.

This is a lineage binding, not a claim that the new RQ6 files existed in the parent
archive.  The active post-snapshot source tree receives its own separate hash.  The
root manifest records the launch argv, Python executable/version, NumPy version,
Torch/CUDA version, BLAS/LAPACK identity, platform, and thread environment.  Each
problem metadata/`COMPLETE` marker binds hashes of that parent snapshot and runtime
environment; resume requires exact agreement.

## 7. Engineering preflight (not scientific evidence)

The required preflight used excluded engineering RNG IDs 7, 11, and 13; it consumed
no formal ID and wrote no formal study artifact.  It exercised the full
\(T=4,K=7\) geometry at \(n=250\), including all 2,401 schedules, Track A, and the
canonical Track-B selector.

Observed on the current machine:

- full smoke: about 0.016 seconds of measured kernels;
- conservative kernel-only extrapolation: about 16.0 seconds per formal problem;
- four-worker kernel-only wall-time estimate: about 0.11 hours;
- planning estimate with a 3x process/I/O margin: about 0.33 hours;
- final Track-A weight matrix: 192,080,000 bytes;
- analytic peak incremental array memory: 219,520,000 bytes per worker;
- planning peak including the observed Python/Torch process RSS: about
  807 MB per worker;
- reserved launch budget: 1 GiB per worker, 4 GiB total for four workers.

These timings are an engineering extrapolation, not a promised runtime.  At actual
launch the runner repeats the preflight and binds the then-active source and config
hashes.  The transient engineering hash is deliberately not frozen here: it must be
recomputed if any executable source changes.

## 8. Commands and current verification

Engineering preflight only:

```bash
conda run -n ucp python scripts/run_rq6_ncal_convergence.py --preflight-only
```

Formal launch command used after protocol freeze:

```bash
conda run -n ucp python scripts/run_rq6_ncal_convergence.py
```

Fail-closed resume of the exact same source/config contract:

```bash
conda run -n ucp python scripts/run_rq6_ncal_convergence.py --resume
```

Focused verification completed:

```text
27 passed
tests/per_step/test_rq6_ncal_convergence.py
tests/per_step/test_exact_finite_mdp.py
tests/per_step/test_marginal_prefix.py
```

The final repository-wide per-step validation also completed with `786 passed`:

```bash
conda run -n ucp pytest -q tests/per_step
```

## 9. Formal result status

All 100 problems and 12,000 problem×replicate×sample-size cells completed.
Track-A mean surface sup error decreased from 0.05599 at (n=250) to 0.009015
at (n=10{,}000); the six-point descriptive log--log slope was -0.4951.
Track-B availability was 100% at every sample size, rowwise target attainment
rose from 62.75% to 100%, and endpoint selection fell from 51.75% to zero.
Canonical WSC changed from 0.93358 to 0.91979 rather than converging exactly or
monotonically to 0.90. These results support empirical surface convergence, not
a claimed finite-sample rate or exact nominal convergence.

## 10. Claim boundary

RQ6 can support a bounded statement about finite logged-sample convergence of the
full-prefix Hájek surface and the canonical selector in this frozen outcome-blind
M3 benchmark.  It does **not** establish finite-sample distribution-free, PAC, or
data-conditional coverage.  It does not turn one controlled finite-MDP family into
a universal real-data or SOTA claim.  Negative convergence, availability, coverage,
or width results must be retained exactly as produced.
