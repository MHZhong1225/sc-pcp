# Phase 0A Greedy Sequential Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task, and use `superpowers:test-driven-development` for every behavior change. Do not start Phase 0B or any clinical/COT implementation from this plan.

**Goal:** Determine, with a reproducible 100-seed paired GPU experiment, whether freeing the current profiled schedule into a greedy stagewise oracle reduces normalized width by at least 10% while retaining 90% stagewise coverage.

**Architecture:** Add an isolated Phase 0A path beside the existing paper pipeline. It reuses the current predictor, policy, split logic, and exact transport-refined profile; adds a separate observed-tail-shift simulator; evaluates candidate schedules with explicit exogenous-noise common random numbers; freezes schedules from 5,000-rollout point estimates; evaluates them once on independent 50,000-rollout batches; and writes atomic, resumable artifacts. The existing simulator and `run_seed()` path remain regression-identical.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, pandas, PyYAML, pytest, two NVIDIA 4090D GPUs, conda environment `ucp`.

## Global Constraints

- Work only in `/home/ubuntu/zmh/.sc-pcp-worktrees/sequential-full-experiment-20260817` on branch `codex/sequential-full-experiment-20260817`.
- Never edit `/home/ubuntu/zmh/sc-pcp`; it is the user's main checkout and currently contains untracked `tools/`.
- Run Python with `/home/ubuntu/anaconda3/envs/ucp/bin/python` or `conda run -n ucp` and default to GPU for smoke/full experiments.
- Preserve the existing standard simulator's random-number order and fixed-seed outputs exactly.
- Candidate grids come only from `D_COT`; neither 5,000-rollout tuning scores nor 50,000-rollout evaluation scores may alter a grid.
- Oracle tuning uses point coverage `>= 0.90`; no LCB enters schedule selection.
- Final evaluation seeds are disjoint from all construction/tuning seeds. The two methods share the same final exogenous-noise bundle within each `(seed, scenario)`.
- Primary records contain exactly `2 scenarios × 2 methods = 4` rows per seed. Discretization sensitivity and finite-MDP diagnostics live in `surfaces.npz`/diagnostics, not as extra primary method rows.
- No feasible candidate means explicit selection failure; there is no endpoint fallback.
- Greedy Sequential Oracle is never called globally optimal. Beam search reports only a best-found gap.
- Do not delete or replace current results. Phase 0A is a gate; No-Go preserves the current method.

---

### Task 1: Track the reproducibility and regression harness

**Files:**

- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `tests/per_step/test_core.py`
- Create: `tests/per_step/test_logged_metrics.py`
- Create: `tests/per_step/test_online_baselines.py`
- Create: `tests/per_step/test_paper_protocol.py`
- Create: `tests/per_step/test_profiled_ordered_method.py`
- Create: `tests/per_step/test_study_artifacts.py`
- Create: `tests/per_step/test_tabular_oracle_validation.py`
- Create: `tests/per_step/test_tabular_validation_reporting.py`
- Create: `tools/render_paper_results.py`
- Create: `tools/summarize_tabular_validation.py`
- Create: `tools/select_baseline_hyperparameters.py`
- Create: `tools/run_full200_shards.py`

- [ ] **Step 1: Demonstrate the branch has no runnable tracked tests**

Run:

```bash
cd /home/ubuntu/zmh/.sc-pcp-worktrees/sequential-full-experiment-20260817
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
```

Expected: pytest reports no tests, and `source_tree_sha256()` cannot safely support a runner because `pyproject.toml` is absent.

- [ ] **Step 2: Import the known 102-test harness without importing user outputs**

Copy only `.gitignore`, `pyproject.toml`, `tests/per_step/*.py`, and the four test-required modules `tools/render_paper_results.py`, `tools/summarize_tabular_validation.py`, `tools/select_baseline_hyperparameters.py`, and `tools/run_full200_shards.py` from the main checkout. These modules are the complete transitive in-repository dependency/path-load set of the imported regression harness; do not copy any other untracked tool or output. Replace `.gitignore` with a tracked minimal version that continues to ignore generated data/results/caches but does not ignore `.gitignore`, `pyproject.toml`, `docs/`, `tests/`, or `tools/`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
.vscode/
data/
dataset/
manifests/
npz/
results/
archive/
baselines/
figures/
outputs/
reports/
*.ipynb
*.log
```

- [ ] **Step 3: Verify the untouched baseline**

Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
```

Expected: `102 passed`.

- [ ] **Step 4: Commit the harness**

```bash
git add .gitignore pyproject.toml tests/per_step tools/render_paper_results.py tools/summarize_tabular_validation.py tools/select_baseline_hyperparameters.py tools/run_full200_shards.py
git commit -m "chore: track phase0 test harness"
```

---

### Task 2: Add the isolated observed tail-shift scenario

**Files:**

- Modify: `src/scpcp/config.py`
- Modify: `src/scpcp/simulator.py`
- Modify: `src/scpcp/experiment.py`
- Create: `tests/per_step/test_phase0_tail_shift.py`

- [ ] **Step 1: Write failing configuration and simulator tests**

Tests must assert:

```python
def test_standard_rollout_is_bitwise_regression_identical(): ...
def test_tail_shift_state_exposes_binary_difficulty(): ...
def test_treatment_changes_next_difficulty_probability(): ...
def test_difficult_state_has_heavier_residual_tail(): ...
def test_tail_shift_fixed_seed_is_reproducible(): ...
```

The standard fixture is `n=2`, `horizon=2`, `seed=12345`, CPU, default `SyntheticConfig`; store its exact states/actions/outcomes in the test before modifying production code.

- [ ] **Step 2: Run the focused test and confirm red**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_tail_shift.py
```

Expected: failures because the scenario fields and environment do not exist.

- [ ] **Step 3: Add validated scenario configuration**

Extend `SyntheticConfig` with exactly these defaults:

```python
scenario: str = "standard"
difficulty_initial_probability: float = 0.15
difficulty_intercept: float = -2.0
difficulty_state_effect: float = 0.35
difficulty_persistence: float = 2.0
difficulty_treatment_effect: float = 1.25
tail_contamination_probability: float = 0.10
tail_scale: float = 4.0
```

In `ExperimentConfig.validate()` require scenario in `{"standard", "tail_shift"}`, probabilities in `[0,1]`, and positive `tail_scale`. Do not add Phase 0B masks or termination fields.

- [ ] **Step 4: Implement a separate seven-state environment**

Keep `SyntheticTreatmentEnvironment` unchanged. Add:

```python
@dataclass(frozen=True)
class TailShiftTreatmentEnvironment:
    config: SyntheticConfig
    state_dim: int = 7
    outcome_dim: int = 2
    n_actions: int = 3

    def difficulty_probability(self, state: Tensor, action: Tensor) -> Tensor:
        intensity = action.to(state.dtype) / (self.n_actions - 1)
        logit = (
            self.config.difficulty_intercept
            + self.config.difficulty_state_effect * state[:, 0]
            + self.config.difficulty_persistence * state[:, 6]
            - self.config.difficulty_treatment_effect * intensity
        )
        return torch.sigmoid(logit)
```

`H_t=state[:,6]` is observed by the predictor. Conditional on `H_t=1`, apply a contamination multiplier of `4` with probability `0.10` to the transition residual; otherwise use multiplier `1`. Draw `H_{t+1}` after the stage outcome using the action-dependent probability above. Clamp only the six continuous coordinates, never the binary coordinate.

- [ ] **Step 5: Route only the requested scenario**

In `_prepare_task`, select `TailShiftTreatmentEnvironment` only when `config.synthetic.scenario == "tail_shift"`; keep standard default behavior untouched.

- [ ] **Step 6: Verify focused and full tests**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_tail_shift.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add src/scpcp/config.py src/scpcp/simulator.py src/scpcp/experiment.py tests/per_step/test_phase0_tail_shift.py
git commit -m "feat: add observed tail-shift scenario"
```

---

### Task 3: Add explicit exogenous noise and CRN primitives

**Files:**

- Modify: `src/scpcp/simulator.py`
- Create: `tests/per_step/test_phase0_crn.py`

- [ ] **Step 1: Write failing CRN tests**

Cover all invariants:

```python
def test_noise_bundle_is_reproducible_and_has_expected_shapes(): ...
def test_inverse_cdf_action_sampling_uses_shared_uniforms(): ...
def test_step_from_noise_is_pure(): ...
def test_candidate_order_does_not_change_patient_noise(): ...
def test_legacy_rollout_fixture_remains_exact(): ...
```

- [ ] **Step 2: Confirm red**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_crn.py
```

- [ ] **Step 3: Implement the immutable noise bundle**

Add:

```python
@dataclass(frozen=True)
class SyntheticNoiseBundle:
    initial_normal: Tensor          # [N, 6]
    initial_difficulty_uniform: Tensor  # [N]
    action_uniform: Tensor          # [T, N]
    shared_normal: Tensor           # [T, N]
    independent_normal: Tensor      # [T, N]
    innovation_normal: Tensor       # [T, N, 4]
    difficulty_uniform: Tensor      # [T, N]
    contamination_uniform: Tensor   # [T, N]


def make_synthetic_noise_bundle(
    *, n: int, horizon: int, seed: int, device: str | torch.device
) -> SyntheticNoiseBundle: ...


def inverse_cdf_actions(probabilities: Tensor, uniforms: Tensor) -> Tensor:
    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("uniforms must match the policy batch shape")
    return (uniforms[..., None] > probabilities.cumsum(dim=-1)).sum(dim=-1)
```

Generate six initial normal columns in the legacy call order. Generate all noise once, store the construction seed, and never call a random API inside `step_from_noise()`.

- [ ] **Step 4: Add pure environment hooks**

Both synthetic environments implement:

```python
def initial_state_from_noise(self, bundle: SyntheticNoiseBundle) -> Tensor: ...

def step_from_noise(
    self,
    state: Tensor,
    action: Tensor,
    *,
    shared: Tensor,
    independent: Tensor,
    innovations: Tensor,
    difficulty_uniform: Tensor,
    contamination_uniform: Tensor,
) -> tuple[Tensor, Tensor]: ...
```

The standard hook must implement the same equations as `step()` but is an additional Phase 0 path; do not rewrite legacy `step()` around it if that changes random order.

- [ ] **Step 5: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_crn.py tests/per_step/test_phase0_tail_shift.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add src/scpcp/simulator.py tests/per_step/test_phase0_crn.py
git commit -m "feat: add auditable synthetic common random numbers"
```

---

### Task 4: Implement candidate-streamed oracle tuning

**Files:**

- Create: `src/scpcp/phase0_oracle.py`
- Create: `tests/per_step/test_phase0_oracle.py`

- [ ] **Step 1: Write deterministic selection tests first**

Use small fake policy/environment/scorer objects. Assert:

- all candidates are evaluated even when coverage is non-monotone;
- profiled selection requires every stage point coverage `>= target` and minimizes total normalized width among feasible schedules;
- greedy selection freezes the chosen prefix and minimizes current-stage normalized width among current-stage feasible candidates;
- a future radius never changes earlier hits;
- ties resolve by the lowest candidate index;
- an empty feasible set records `selection_available=False` and the exact `failure_stage`;
- endpoint selection is recorded and never silently expanded;
- candidate order and chunk sizes `1`, `16`, and `101` yield identical metrics after undoing the permutation.

- [ ] **Step 2: Define the minimal result types**

```python
@dataclass(frozen=True)
class CandidateMetrics:
    coverage: Tensor          # [K, T] for profiled; [K] at a greedy stage
    normalized_width: Tensor  # matching coverage


@dataclass(frozen=True)
class OracleScheduleResult:
    radii: Tensor | None      # [T]
    selected_indices: tuple[int, ...]
    tuning_coverage: Tensor | None  # [T]
    tuning_width: Tensor | None     # [T]
    selection_available: bool
    failure_stage: int | None
    selected_endpoint: bool
```

- [ ] **Step 3: Implement profiled candidate streaming**

Expose:

```python
@torch.no_grad()
def evaluate_profiled_candidates_crn(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    candidate_schedules: Tensor,  # [K,T]
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    chunk_size: int,
) -> CandidateMetrics: ...
```

For each candidate chunk, maintain `[K_chunk,N,D]` state, flatten only for predictor/policy calls, apply `q[k,t]`, inverse-CDF sample using the same `action_uniform[t]` expanded over candidates, score the outcome immediately, compute normalized width

```python
(2.0 * q[:, None] * scale / outcome_sd).mean(dim=-1)
```

and release the chunk before the next. Never materialize `[101,5000,12,D]` trajectories.

- [ ] **Step 4: Implement greedy streaming**

Expose:

```python
@torch.no_grad()
def greedy_sequential_oracle_schedule(
    environment: object,
    policy: object,
    outcome_model: object,
    *,
    stage_grids: Tensor,       # [T,K], frozen from D_COT
    outcome_sd: Tensor,
    noise: SyntheticNoiseBundle,
    target: float = 0.90,
    chunk_size: int = 16,
) -> OracleScheduleResult: ...
```

At stage `t`, expand only the committed `[N,D]` prefix state to `[K,N,D]`, reuse the one stage's exogenous noise for every candidate, select the feasible candidate with minimum current-stage mean normalized width, and commit only that candidate's actions/next states. Cost must be `K*N*T`, not `K*N*sum(1..T)`.

- [ ] **Step 5: Implement profiled selection**

```python
def select_profiled_oracle_schedule(
    candidate_schedules: Tensor,
    metrics: CandidateMetrics,
    *,
    target: float = 0.90,
) -> OracleScheduleResult: ...
```

Feasibility is `metrics.coverage.ge(target).all(dim=1)`. The objective is `metrics.normalized_width.mean(dim=1)`. Evaluate the exact current focused family first.

- [ ] **Step 6: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_oracle.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add src/scpcp/phase0_oracle.py tests/per_step/test_phase0_oracle.py
git commit -m "feat: add profiled and greedy oracle tuning"
```

---

### Task 5: Add independent frozen-schedule evaluation and coverage bands

**Files:**

- Modify: `src/scpcp/phase0_oracle.py`
- Create: `tests/per_step/test_phase0_evaluation.py`

- [ ] **Step 1: Write failing statistical tests**

Test Wilson values against `statsmodels`-independent hand calculations, monotonicity in hits, alpha allocation `0.05/12`, exact 50,000 batch usage, shared final bundle across methods, and disjoint tuning/evaluation stream IDs.

- [ ] **Step 2: Implement seed-level fresh evaluation**

```python
@dataclass(frozen=True)
class FrozenOracleEvaluation:
    coverage: Tensor                  # [T]
    wilson_lower_bound: Tensor        # [T], MC diagnostic
    normalized_width: Tensor          # [T]
    micro_normalized_width: float
    patient_normalized_width: float
    n_rollouts: int


def bonferroni_wilson_lower_bounds(
    hits: Tensor, *, family_alpha: float = 0.05
) -> Tensor: ...


@torch.no_grad()
def evaluate_frozen_schedules_crn(
    ...,
    schedules: dict[str, Tensor],
    noise: SyntheticNoiseBundle,
    outcome_sd: Tensor,
) -> dict[str, FrozenOracleEvaluation]: ...
```

For fixed-length Phase 0A, `W_micro` is the mean over all patient-stage normalized widths and `W_patient` is the mean of each patient's 12-stage mean; they should agree up to floating-point error. Keep both fields because they separate after Phase 0B.

- [ ] **Step 3: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_evaluation.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add src/scpcp/phase0_oracle.py tests/per_step/test_phase0_evaluation.py
git commit -m "feat: add independent phase0 oracle evaluation"
```

---

### Task 6: Build one complete paired seed result

**Files:**

- Modify: `src/scpcp/experiment.py`
- Modify: `src/scpcp/phase0_oracle.py`
- Create: `tests/per_step/test_phase0_seed.py`

- [ ] **Step 1: Write a monkeypatched end-to-end seed contract test**

Assert one seed returns exactly four primary rows:

```python
{
    ("standard", "Current Profiled Oracle"),
    ("standard", "Greedy Sequential Oracle"),
    ("tail_shift", "Current Profiled Oracle"),
    ("tail_shift", "Greedy Sequential Oracle"),
}
```

Each row contains `q_by_time`, tuning/final coverage, seed-level Wilson LCB, stage width, micro/patient width, selection status, failure stage, endpoint status, and construction/evaluation seed IDs.

- [ ] **Step 2: Expose a narrow shared preparation helper**

Move only the already-existing setup shared by `run_seed()` and Phase 0 into a private helper returning:

```python
@dataclass(frozen=True)
class _OracleContext:
    task: _Task
    outcome_model: object
    region: object
    policy: object
    logging_policy: object
    outcome_sd: Tensor
    cot_scores: Tensor
    schedule_family: _RefinedScheduleFamily
```

Do not change the order or seeds of `run_seed()` setup. Add a regression test that an existing `run_seed` fixture is unchanged.

- [ ] **Step 3: Freeze both grids from D_COT**

For each scenario/seed:

```python
profiled_schedules = candidate_radius_schedules(
    context.schedule_family.scale_grid,
    context.schedule_family.profile,
)
stage_grids = torch.stack([
    fixed_q_grid(
        context.cot_scores[:, t],
        size=config.q_grid_size,
        lower_quantile=config.q_quantile_min,
        upper_quantile=config.q_quantile_max,
    )
    for t in range(config.horizon)
])
```

Also evaluate the profiled family on the common pre-registered quantile-probability vector as a sensitivity and store its metrics under `surfaces["profiled_common_grid_..."]`; it must not replace the exact-current primary comparator.

- [ ] **Step 4: Implement deterministic streams**

Use stable named streams via `_paper_seed`, for example:

```python
tuning_seed = _paper_seed(seed, 1_300_001 + scenario_index)
evaluation_seed = _paper_seed(seed, 1_400_001 + scenario_index)
```

Assert the two integers differ, store both, and use the same scenario-specific tuning bundle for all candidates and the same evaluation bundle for both frozen methods.

- [ ] **Step 5: Implement `run_phase0_seed()`**

Return existing `SeedResult` so `write_seed_result()` remains the sole atomic writer. Failures to select still generate their primary row with JSON arrays empty/`NaN`; the other method/scenario continues and the seed artifact remains auditable.

- [ ] **Step 6: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_seed.py tests/per_step/test_paper_protocol.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add src/scpcp/experiment.py src/scpcp/phase0_oracle.py tests/per_step/test_phase0_seed.py
git commit -m "feat: orchestrate paired phase0 oracle seeds"
```

---

### Task 7: Add atomic resume-safe GPU runner

**Files:**

- Modify: `src/scpcp/phase0_oracle.py`
- Create: `configs/phase0_oracle.yaml`
- Create: `scripts/run_phase0_oracle.py`
- Create: `tests/per_step/test_phase0_runner.py`

- [ ] **Step 1: Write runner contract tests**

Monkeypatch the expensive seed function. Verify:

- default config fixes `T=12`, `K=101`, `5,000` tuning rollouts, `50,000` final rollouts, and seeds `0:100`;
- CLI defaults to one persistent worker per GPU;
- a complete seed is skipped only if `COMPLETE`, `records.csv`, `surfaces.npz`, and `metadata.json` exist and the four exact scenario/method rows are present;
- partial or malformed seed directories stop with an actionable error and are never treated as complete;
- study `COMPLETE` is written only when all 100 requested seeds validate;
- source hash/config mismatch aborts resume;
- no output path from an earlier paper run is reused.

- [ ] **Step 2: Create the frozen config**

Start from `configs/per_step_synthetic.yaml` and change only:

```yaml
data:
  dataset: synthetic
horizon: 12
q_grid_size: 101
q_quantile_min: 0.50
q_quantile_max: 0.999
seeds: [0, 1, ..., 99]
samples:
  logged: 5000
  oracle_surface_rollouts: 5000
  oracle_rollouts: 50000
output_dir: results/work/phase0a_profiled_vs_greedy
```

The runner constructs both scenarios with `dataclasses.replace`; do not duplicate the full config by scenario.

Expose `candidate_chunk_size: int = 16` as a keyword-only `run_phase0_seed` argument, reject non-positive values before work, and pass it to exact-profiled, common-grid-profiled, and greedy tuning. The runner's CLI value must therefore change real execution rather than merely metadata.

- [ ] **Step 3: Implement the runner**

CLI:

```text
--config PATH
--seeds 0:100
--devices cuda:0,cuda:1
--workers-per-device 1
--candidate-chunk-size 16
--output-dir PATH
--resume
```

Use `spawn` multiprocessing, pin each worker to one resolved CUDA device, call `write_study_metadata`, `write_seed_result`, `mark_study_failed`, and `mark_study_complete`, and validate the exact row contract before accepting a seed.

- [ ] **Step 4: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_runner.py tests/per_step/test_study_artifacts.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add configs/phase0_oracle.yaml scripts/run_phase0_oracle.py tests/per_step/test_phase0_runner.py
git commit -m "feat: add resumable phase0 GPU runner"
```

---

### Task 8: Add finite-MDP exact and beam sanity diagnostics

**Files:**

- Create: `src/scpcp/phase0_search.py`
- Create: `scripts/run_phase0_search_sanity.py`
- Create: `tests/per_step/test_phase0_search.py`

- [ ] **Step 1: Write a tiny triangular counterexample**

Construct a two-stage discrete objective where the locally narrow feasible first radius changes occupancy and forces a much wider second radius. Assert greedy is feasible but globally suboptimal.

- [ ] **Step 2: Implement explicit labels**

```python
@dataclass(frozen=True)
class SearchDiagnostic:
    search_type: str  # "exact" or "beam"
    greedy_width: float
    best_found_width: float
    true_optimality_gap: float | None
    best_found_gap: float


def exact_schedule_search(...): ...
def beam_schedule_search(..., *, beam_width: int): ...
```

Only exact enumeration sets `true_optimality_gap`. Beam output must leave it `None` and use the phrase `best_found_gap`.

- [ ] **Step 3: Run one pre-registered finite-grid diagnostic**

`scripts/run_phase0_search_sanity.py` evaluates the existing finite-MDP environment on a reduced `T=4`, `K=5` D_COT-frozen grid. Enumerate all `5^4=625` schedules under one fixed tuning bundle, compare the Greedy Sequential schedule to the exact best feasible grid schedule, and atomically write `finite_mdp_sanity.json`. The JSON must label the result `exact_finite_grid_search`, state that the gap is relative only to the reduced frozen grid, and include coverage, width, chosen indices, and the true finite-grid gap. This diagnostic is run before the 100-seed launch and copied into the Phase 0 study root without entering the Go/No-Go gate.

- [ ] **Step 4: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_search.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add src/scpcp/phase0_search.py scripts/run_phase0_search_sanity.py tests/per_step/test_phase0_search.py
git commit -m "test: add greedy finite-mdp sanity diagnostic"
```

---

### Task 9: Add locked summary and Go/No-Go decision logic

**Files:**

- Create: `scripts/summarize_phase0_oracle.py`
- Create: `tests/per_step/test_phase0_summary.py`

- [ ] **Step 1: Write synthetic pass/fail fixtures**

Test every gate independently, incomplete/malformed artifacts, selection conditioning labels, deterministic paired bootstrap, and the all-or-nothing decision.

- [ ] **Step 2: Implement the pre-registered coverage band**

For each `(scenario, method, stage)`, use selected seeds' fresh `coverage[t]`:

```python
critical = scipy.stats.t.ppf(1.0 - 0.05 / 12, df=n_selected - 1)
lower = mean - critical * sample_sd / math.sqrt(n_selected)
```

Label it `seed_mean_bonferroni_t_lcb`, report `n_selected`, and never substitute the seed-level Wilson diagnostic.

- [ ] **Step 3: Implement paired width inference**

On seeds where both methods select, compute the geometric mean ratio from log ratios. Use exactly 10,000 paired seed bootstrap resamples with fixed seed `2_718_281`; record percentile 95% endpoints. Report endpoint/no-feasible rates and common-grid sensitivity separately.

- [ ] **Step 4: Encode the gate literally**

`tail_shift` Go requires:

```python
all(greedy_stage_lcb >= 0.90)
and geometric_mean_micro_ratio <= 0.90
and bootstrap_micro_ratio_upper < 1.00
and geometric_mean_patient_ratio <= 0.92
and greedy_selection_count >= 95
```

`standard` additionally requires all greedy stage LCBs `>=0.90` and geometric mean micro ratio `<=1.02`. Any false or unavailable condition yields `NO_GO`; never round before comparison.

- [ ] **Step 5: Write durable outputs**

Create atomically in the experiment root:

- `phase0_summary.csv`
- `phase0_decision.json`
- `phase0_radius_and_coverage.pdf`
- `phase0_summary.md`

The report says either `GO` or `NO_GO`, never “SOTA”. It may call a result better only when the frozen gate passes.

- [ ] **Step 6: Verify and commit**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0_summary.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git add scripts/summarize_phase0_oracle.py tests/per_step/test_phase0_summary.py
git commit -m "feat: add preregistered phase0 decision report"
```

---

### Task 10: CPU verification and single-seed GPU smoke

**Files:**

- Modify only files implicated by observed failures.

- [ ] **Step 1: Run the complete CPU suite**

```bash
cd /home/ubuntu/zmh/.sc-pcp-worktrees/sequential-full-experiment-20260817
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass and worktree is clean.

- [ ] **Step 2: Run one real seed on GPU 0**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python scripts/run_phase0_oracle.py \
  --config configs/phase0_oracle.yaml \
  --seeds 0:1 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --candidate-chunk-size 16 \
  --output-dir results/work/phase0a_smoke_seed0
```

- [ ] **Step 3: Audit the smoke before scaling**

Verify:

- four primary rows and all arrays length 12;
- tuning/final seed IDs differ;
- exact-current and greedy final evaluation IDs match within scenario;
- coverage, LCB, and width values are finite where selected;
- no candidate curve uses final-evaluation data;
- peak GPU memory leaves headroom;
- standard simulator regression still passes;
- wall time supports a bounded full-run estimate.

If GPU utilization is low and memory has at least 2× headroom, test `workers-per-device=2` on seeds `1:5`; otherwise retain one worker/GPU. Do not use four workers/GPU without evidence.

- [ ] **Step 4: Request code review and fix only verified issues**

Use `superpowers:requesting-code-review`, rerun focused tests for each fix, then rerun the full suite and one-seed smoke.

---

### Task 11: Launch, monitor, and decide the full experiment

**Files:**

- No source changes after launch. Any source change requires a new output root and a fresh 100-seed run.

- [ ] **Step 1: Freeze provenance**

Record clean Git revision, source-tree SHA-256, config SHA-256, CUDA/PyTorch versions, devices, chunk size, worker count, and all expected seeds in study metadata.

Run the frozen reduced-grid sanity diagnostic and retain its output:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python scripts/run_phase0_search_sanity.py \
  --device cuda:0 \
  --output results/work/phase0a_finite_mdp_sanity.json
```

- [ ] **Step 2: Launch 100 paired seeds**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python scripts/run_phase0_oracle.py \
  --config configs/phase0_oracle.yaml \
  --seeds 0:100 \
  --devices cuda:0,cuda:1 \
  --workers-per-device 1 \
  --candidate-chunk-size 16 \
  --output-dir results/work/phase0a_profiled_vs_greedy
```

Use the smoke-validated worker count if it differs. Capture the job PID/log without modifying the experiment directory contract.

- [ ] **Step 3: Monitor durability, not intermediate performance**

Report completed seed count, failures, GPU utilization/memory, and estimated completion time. Do not inspect partial performance to change grids, thresholds, seeds, or scenarios.

- [ ] **Step 4: Resume only with identical provenance**

If interrupted, rerun the same command with `--resume`. The runner must skip only contract-valid complete seeds and reject source/config hash drift.

- [ ] **Step 5: Validate completeness and summarize**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python scripts/summarize_phase0_oracle.py \
  --input-dir results/work/phase0a_profiled_vs_greedy
```

Require 100 seed directories, 100 `COMPLETE` markers, 400 primary rows, exact scenario/method balance, and matching provenance before computing the decision.

- [ ] **Step 6: Apply the terminal rule**

- If `NO_GO`: keep the current method and all historical results untouched; save the negative report and stop before Phase 0B.
- If `GO`: retain this branch as the best structural version, then create a separate reviewed plan for Phase 0B masks/variable length and practical sequential COT. Do not silently promote Phase 0A oracle numbers as the deployed method or as SOTA.

- [ ] **Step 7: Final verification**

Use `superpowers:verification-before-completion`. Re-run the full tests, artifact validator, and report generation; inspect the PDF visually; report the exact revision, output directory, counts, gate values, and decision.
