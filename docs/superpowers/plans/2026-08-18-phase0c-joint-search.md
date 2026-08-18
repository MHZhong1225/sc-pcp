# Phase 0C Joint-Search Attainability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed Phase 0C audit that tests whether deterministic multi-start joint coordinate search can reduce the fixed-horizon scalar-radius oracle's tail-shift width ratio to at most `0.92` without weakening simultaneous stagewise coverage.

**Architecture:** Add isolated Phase 0C modules rather than changing the frozen Phase 0A profiled/greedy implementation. A pure coordinate-search state machine consumes a CRN suffix evaluator, the per-seed study produces paired B/2B checkpoints on the same tuning and fresh-evaluation streams, and independent runner/summarizer scripts enforce atomic artifacts, provenance, statistical gates, and the conditional 8-sweep extension. Existing Phase 0A paths remain regression-identical.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, SciPy, Matplotlib, pytest, CUDA on two NVIDIA 4090D GPUs, conda environment `/home/ubuntu/anaconda3/envs/ucp`.

**Spec:** `docs/superpowers/specs/2026-08-18-risk-adaptive-scpcp-design.md`

## Global Constraints

- The immutable comparison base is commit `59b7f1e608d59db431982e84864d28f81c309e79`; do not modify Phase 0A code, configuration, outputs, or the current active method.
- This plan implements only Phase 0C. It does not implement risk-adaptive radii, variable-length trajectories, clinical caches, deployable baselines, or a SOTA claim.
- Development seeds are exactly ordered integers `10000..10039`. Historical seeds `0..99` remain exploratory and no confirmation bank may be opened.
- Scenarios are exactly `standard` and `tail_shift`; horizon is `12`; target coverage is `0.90`; stage-grid size is `101`.
- Fixed starts and tie order are exactly `profiled`, `greedy`, `upper_endpoint`; the upper endpoint is `stage_grids[:, -1]`.
- A sweep pair is forward stages `0..11` followed by reverse stages `11..0`. B is two pairs, 2B is four pairs, and the optional extension is eight sweep pairs (`8SP`, not `8B`).
- Checkpoints 2 and 4 must come from one nested maximum-four-pair search using the same tuning CRN. The optional 8SP extension must continue all three start states from pair 4 and requires machine authorization from the initial decision artifact.
- Each coordinate candidate must be jointly feasible at all 12 stages. The objective is full-trajectory mean normalized width. Exact candidate ties use the lowest original grid index; an equal-width candidate is not committed.
- Tuning uses 5,000 rollouts; frozen schedules use one independent shared 50,000-rollout bundle. Tuning and evaluation stream IDs must differ globally.
- Phase 0C coverage validity is based on the 40-seed fresh-evaluation stage means, not per-seed Wilson bounds. For each checkpoint, use a one-sided Bonferroni-t band jointly over two scenarios by twelve stages at family alpha `0.05` (`t_{1-0.05/24,39}=3.0440624276034796`).
- Initial analysis requires B and 2B to have 40/40 usable pairs in both scenarios and both checkpoints to be coverage-valid. No epsilon, imputation, selected-only denominator, or fallback is permitted.
- Tail ratios use sorted paired seeds and fresh micro widths: `R_k = exp(mean(log(W_joint,k / W_current)))`. Define `Delta_B=(R_B-R_2B)/R_B`.
- Exact decision boundaries are: `R_2B <= 0.92` → `PROMISING_ORACLE_DIAGNOSTIC`; `R_2B > 0.92` and `Delta_B < 0.005` → `STOP_SCALAR_SATURATED`; `R_2B > 0.92` and `Delta_B >= 0.005` → `EXTENSION_8SP_REQUIRED`; missing validity/pairs → `STOP_SCALAR_UNAVAILABLE`.
- An authorized 8SP run ends `PROMISING_ORACLE_DIAGNOSTIC` only when its valid 40-pair `R_8SP <= 0.92`; otherwise it ends `STOP_SCALAR_INSUFFICIENT` or `STOP_SCALAR_UNAVAILABLE`.
- Never emit `GO` or `SOTA`. A promising result is an oracle diagnostic that authorizes a separate practical-method design only.
- All implementation changes follow strict RED→GREEN TDD. Tests use hand-derived literals and real behavior; production code is not written before its covering failure is observed.
- Run code and tests in a new remote linked worktree, under `/home/ubuntu/anaconda3/envs/ucp`; production CUDA jobs explicitly pin `cuda:0,cuda:1` with one persistent worker per device.
- One maximum-four-pair GPU smoke on seed `9999` precedes the 40-seed study. The formal per-seed wall cap is read from the signed smoke manifest as `ceil(1.5 * smoke_seconds / 300) * 300` seconds; it is frozen in study metadata before any development seed runs.
- Every seed and study is atomically published, resumable, deeply validated, and bound to source tree, experiment tree, config, ordered seed manifest, stream IDs, chunk size, devices, worker count, algorithm version, start order, sweep budgets, and wall cap hashes.
- User-facing deliverables are copied only to `/Users/bule/Documents/Codex/2026-08-17/api-home-ubuntu-zmh-performativecp-ssh/outputs`; intermediate files remain under `work/` or the remote result root.

## File Structure

- Create `src/scpcp/phase0c_joint_search.py`: deterministic state machine, start/checkpoint/trace dataclasses, schedule-cache construction, and CRN suffix candidate evaluation.
- Create `src/scpcp/phase0c_study.py`: scenario preparation, fixed starts, nested B/2B per-seed orchestration, fresh paired evaluation, records, surfaces, and diagnostics.
- Create `configs/phase0c_joint_search.yaml`: Phase 0A scientific settings with development seed bank `10000..10039` and isolated output root.
- Create `scripts/run_phase0c_joint_search.py`: smoke calibration, two-GPU spawn runner, atomic study/seed publication, resume, provenance, deep schema validation, and conditional 8SP continuation.
- Create `scripts/summarize_phase0c_joint_search.py`: strict input validation, coverage/ratio inference, decision state machine, atomic JSON/CSV/Markdown/figure bundle, and output manifest.
- Create `tests/per_step/test_phase0c_joint_search.py`: pure search and tie/feasibility/deadline behavior.
- Create `tests/per_step/test_phase0c_crn_replay.py`: cached suffix evaluation against explicit full CRN replay.
- Create `tests/per_step/test_phase0c_seed.py`: per-seed stream, start, row, surface, and fresh-evaluation contract.
- Create `tests/per_step/test_phase0c_runner.py`: GPU pinning, smoke cap, fresh/resume, extension authorization, and fail-closed artifact validation.
- Create `tests/per_step/test_phase0c_summary.py`: hand-checked statistics, decisions, mutations, atomic publisher, and figure semantics.
- Do not modify `src/scpcp/phase0_oracle.py`, `src/scpcp/phase0_search.py`, `scripts/run_phase0_oracle.py`, `scripts/summarize_phase0_oracle.py`, or any Phase 0A output.

---

### Task 1: Deterministic Multi-Start Coordinate-Search State Machine

**Files:**
- Create: `src/scpcp/phase0c_joint_search.py`
- Create: `tests/per_step/test_phase0c_joint_search.py`

**Interfaces:**
- Consumes: `scpcp.phase0_oracle.CandidateMetrics` and `OracleScheduleResult` semantics.
- Produces: `SearchStart`, `CoordinateStep`, `SearchState`, `JointSearchCheckpoint`, `JointSearchOutcome`, `coordinate_candidate_schedules(...)`, `cyclic_joint_coordinate_search(...)`, and `resume_cyclic_joint_coordinate_search(...)` for Tasks 2–4.

- [ ] **Step 1: Write literal RED tests for schedule construction, feasibility, objective, and ties**

Define a table-driven fake evaluator that maps `(stage, grid_index, incumbent tuple)` to literal coverage and width tensors. The first tests must assert:

```python
def test_coordinate_candidates_replace_only_requested_stage():
    incumbent = torch.tensor([1.0, 2.0, 3.0])
    grid = torch.tensor([0.5, 1.5, 2.5])
    got = coordinate_candidate_schedules(incumbent, stage=1, stage_grid=grid)
    want = torch.tensor([[1.0, 0.5, 3.0], [1.0, 1.5, 3.0], [1.0, 2.5, 3.0]])
    assert torch.equal(got, want)

def test_coordinate_choice_requires_all_stage_coverage_and_uses_full_mean_width():
    metrics = CandidateMetrics(
        coverage=torch.tensor([[0.91, 0.89], [0.90, 0.90], [0.91, 0.91]]),
        normalized_width=torch.tensor([[0.10, 0.10], [1.00, 3.00], [2.00, 1.00]]),
    )
    choice = choose_coordinate_candidate(metrics, target=0.90)
    assert choice.grid_index == 2
    assert choice.micro_width == 1.5

def test_duplicate_minimum_uses_lowest_original_grid_index():
    metrics = CandidateMetrics(
        coverage=torch.full((3, 2), 0.91),
        normalized_width=torch.tensor([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    assert choose_coordinate_candidate(metrics, target=0.90).grid_index == 1
```

Add independent tests that an equal-width proposal is not committed, a coordinate with zero feasible proposals preserves a feasible incumbent, all infeasible starts return `NO_FEASIBLE_START`, and a start tie resolves in the input order.

- [ ] **Step 2: Run the focused tests and verify feature-missing RED**

Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_joint_search.py
```

Expected: collection fails only because `scpcp.phase0c_joint_search` does not exist, or behavioral assertions fail because the new symbols are absent. Fix fixture/import errors until the failure is feature-specific.

- [ ] **Step 3: Implement the minimal immutable search types and selection primitives**

Use these exact public contracts:

```python
@dataclass(frozen=True)
class SearchStart:
    name: str
    radii: Tensor
    stage_grid_indices: tuple[int | None, ...]
    coverage: Tensor
    normalized_width: Tensor

@dataclass(frozen=True)
class CoordinateStep:
    start_name: str
    sweep_pair: int
    direction: str
    stage: int
    feasible_count: int
    proposed_grid_index: int | None
    before_micro_width: float
    proposed_micro_width: float | None
    committed: bool
    after_micro_width: float

@dataclass(frozen=True)
class SearchState:
    start_name: str
    radii: Tensor
    stage_grid_indices: tuple[int | None, ...]
    coverage: Tensor
    normalized_width: Tensor
    completed_sweep_pairs: int
    converged_at_pair: int | None

@dataclass(frozen=True)
class JointSearchCheckpoint:
    requested_sweep_pairs: int
    executed_sweep_pairs: int
    best: SearchState
    per_start: tuple[SearchState, ...]
    trace: tuple[CoordinateStep, ...]
    schedule_evaluations: int
    committed_updates: int

@dataclass(frozen=True)
class JointSearchOutcome:
    status: str
    checkpoints: dict[int, JointSearchCheckpoint]
    elapsed_seconds: float
```

Implement `coordinate_candidate_schedules`, `choose_coordinate_candidate`, and `cyclic_joint_coordinate_search`. The search accepts an evaluator callable with signature:

```python
Callable[[str, Tensor, int, Tensor], CandidateMetrics]
```

The first argument is the frozen start name, followed by incumbent schedule, stage, and grid. The search iterates each active start, each pair, `forward=range(T)`, then `reverse=range(T - 1, -1, -1)`. It snapshots pairs 2 and 4 from one run, commits only strict objective improvement, and if every active start makes no commit in one complete pair, materializes later requested checkpoints from the converged state without extra evaluation. `resume_cyclic_joint_coordinate_search` accepts every per-start pair-4 `SearchState`, starts at pair 5, and produces only the pair-8 checkpoint; it rejects mixed or non-four completed-pair counts.

- [ ] **Step 4: Add RED→GREEN tests for ordering, nested checkpoints, exact reduced search, and deadline rollback**

Use the returned trace to assert the exact order:

```python
assert [
    (step.start_name, step.sweep_pair, step.direction, step.stage)
    for step in outcome.checkpoints[2].trace[:4]
] == [
    ("profiled", 1, "forward", 0),
    ("profiled", 1, "forward", 1),
    ("profiled", 1, "reverse", 1),
    ("profiled", 1, "reverse", 0),
]
```

Use a literal `T=3, K=3` landscape to enumerate all `27` schedules and verify the returned schedule is feasible and no wider than any fixed start; separately include a coordinate trap where exhaustive search is better so the API remains labeled `best_found`. Assert maximum-four checkpoint 2 equals a maximum-two run, and a fake clock crossing the deadline rolls back the incomplete pair and omits its checkpoint.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_joint_search.py tests/per_step/test_phase0_oracle.py tests/per_step/test_phase0_search.py
```

Expected: all pass; no existing Phase 0 tests change.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/scpcp/phase0c_joint_search.py tests/per_step/test_phase0c_joint_search.py
git commit -m "feat: add deterministic phase0c coordinate search"
```

### Task 2: CRN Suffix Evaluator and Exact Cache Equivalence

**Files:**
- Modify: `src/scpcp/phase0c_joint_search.py`
- Create: `tests/per_step/test_phase0c_crn_replay.py`

**Interfaces:**
- Consumes: `SearchStart` and `CandidateMetrics` from Task 1; `_evaluate_stage` and `evaluate_profiled_candidates_crn` from `scpcp.phase0_oracle` without changing them.
- Produces: `ScheduleCache`, `build_schedule_cache(...)`, `evaluate_coordinate_candidates_crn(...)`, and `CRNCoordinateEvaluator` for Task 3.

- [ ] **Step 1: Capture a deterministic literal full-replay fixture before production edits**

Build a tiny CPU environment with `N=5`, `T=3`, `K=5`, two outcome dimensions, fixed `SyntheticNoiseBundle`, and fixed predictor/policy. Store literal coverage and width arrays produced by `evaluate_profiled_candidates_crn` for stages `0`, `1`, and `2`; do not derive expected arrays using the new cache code.

- [ ] **Step 2: Write RED tests for cached suffix equivalence**

The core test constructs `[K,T]` schedules explicitly, computes the old full-replay result, and compares the new suffix result:

```python
full = evaluate_profiled_candidates_crn(
    environment,
    policy,
    outcome_model,
    candidate_schedules=coordinate_candidate_schedules(
        incumbent, stage=stage, stage_grid=stage_grid
    ),
    outcome_sd=outcome_sd,
    noise=noise,
    chunk_size=2,
)
cached = evaluate_coordinate_candidates_crn(
    environment,
    policy,
    outcome_model,
    cache=build_schedule_cache(
        environment,
        policy,
        outcome_model,
        schedule=incumbent,
        outcome_sd=outcome_sd,
        noise=noise,
    ),
    incumbent_schedule=incumbent,
    stage=stage,
    stage_grid=stage_grid,
    outcome_sd=outcome_sd,
    noise=noise,
    chunk_size=2,
)
assert torch.equal(cached.coverage, full.coverage)
assert torch.equal(cached.normalized_width, full.normalized_width)
```

Parameterize `stage in (0, 1, 2)` and `chunk_size in (1, 2, 5)`. Add candidate-permutation invariance and assert changing coordinate `t` leaves cached coverage/width before `t` identical.

- [ ] **Step 3: Run the new test and verify RED**

Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_crn_replay.py
```

Expected: failure specifically names missing `ScheduleCache`, `build_schedule_cache`, or `evaluate_coordinate_candidates_crn`.

- [ ] **Step 4: Implement cache construction and suffix replay**

Use this storage contract:

```python
@dataclass(frozen=True)
class ScheduleCache:
    states_before: tuple[Tensor, ...]  # length T + 1; each [N, state_dim]
    coverage: Tensor                  # [T]
    normalized_width: Tensor          # [T]
```

`build_schedule_cache` replays one schedule once and clones each committed boundary state. `evaluate_coordinate_candidates_crn` repeats `cache.states_before[stage]` across a candidate chunk, copies the incumbent prefix metrics for columns `< stage`, and replays stages `stage..T-1` using `_evaluate_stage`. It never mutates a cache. After a committed update, the state machine rebuilds the single-schedule cache once.

`CRNCoordinateEvaluator` implements the Task 1 callable signature. It keeps one immutable cache per start name; when the supplied incumbent schedule differs from the cached schedule after a commit, it rebuilds that start's cache exactly once before suffix evaluation. It rejects an unknown start name or a reused name with an incompatible horizon.

- [ ] **Step 5: Verify equivalence, chunk invariance, and memory release**

Add a weak-reference allocation seam test proving candidate `[K,N,D]` states are released before the next coordinate starts. Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_crn_replay.py tests/per_step/test_phase0c_joint_search.py tests/per_step/test_phase0_oracle.py
```

Expected: all pass; the literal fixture remains bitwise identical.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/scpcp/phase0c_joint_search.py tests/per_step/test_phase0c_crn_replay.py
git commit -m "feat: add cached crn suffix evaluation"
```

### Task 3: One-Seed B/2B Study Orchestration and Artifact Schema

**Files:**
- Create: `src/scpcp/phase0c_study.py`
- Create: `tests/per_step/test_phase0c_seed.py`

**Interfaces:**
- Consumes: `_prepare_oracle_context`, `_paper_seed`, `candidate_radius_schedules`, `fixed_q_grid`, existing Phase 0 profiled/greedy selectors/evaluators, and Task 1–2 search APIs.
- Produces: `PHASE0C_METHODS`, `Phase0CScenarioContext`, `prepare_phase0c_scenario_context(...)`, `run_phase0c_seed(...) -> SeedResult`, and `run_phase0c_extension_seed(...) -> SeedResult` for Task 4.

- [ ] **Step 1: Write RED contract tests with a complete real-shaped context double**

Lock these exact method IDs and rows:

```python
PHASE0C_METHODS = (
    "current_profiled",
    "greedy",
    "joint_B",
    "joint_2B",
)
expected_pairs = {
    (scenario, method)
    for scenario in ("standard", "tail_shift")
    for method in PHASE0C_METHODS
}
assert len(result.records) == 8
assert {(row["scenario"], row["method_id"]) for row in result.records} == expected_pairs
```

The fixture must mirror the real context fields. Assert three fixed start names/order, upper endpoint schedule, one nested `(2,4)` search call per scenario, shared tuning noise within a scenario, different tuning/evaluation streams, exactly one fresh evaluation bundle for all four schedules, and 5,000/50,000 rollout sizes.

- [ ] **Step 2: Run RED and confirm missing production module**

Run:

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_seed.py
```

Expected: collection fails only on missing `scpcp.phase0c_study`.

- [ ] **Step 3: Implement scenario preparation without changing Phase 0A**

Use this public context:

```python
@dataclass(frozen=True)
class Phase0CScenarioContext:
    environment: object
    policy: object
    outcome_model: object
    outcome_sd: Tensor
    profile: Tensor
    profiled_scale_grid: Tensor
    profiled_schedules: Tensor
    stage_grids: Tensor
    tuning_noise: SyntheticNoiseBundle
    evaluation_noise: SyntheticNoiseBundle
    starts: tuple[SearchStart, ...]
    profiled_selection: OracleScheduleResult
    greedy_selection: OracleScheduleResult
```

For each scenario, call `_prepare_oracle_context`, reproduce Phase 0A's profile and stage grids exactly, use tuning stream `_paper_seed(seed, 1_300_001 + scenario_index)` and evaluation stream `_paper_seed(seed, 1_400_001 + scenario_index)`, and construct starts after evaluating them on the same tuning bundle. An unavailable start remains recorded but does not enter active search.

- [ ] **Step 4: Implement the exact record/surface/diagnostic contract**

`run_phase0c_seed` has this signature:

```python
def run_phase0c_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str,
    candidate_chunk_size: int = 16,
    sweep_pair_checkpoints: tuple[int, ...] = (2, 4),
    max_seed_wall_seconds: float,
) -> SeedResult:
```

It rejects non-synthetic data, nonpositive chunk/cap, or checkpoints other than `(2,4)` before model work. Each record contains the fixed schema fields listed below and uses empty JSON arrays plus CSV `NaN` for unavailable numeric results:

```text
schema_version, seed, scenario, method_id, analysis_role, budget_id,
sweep_pairs, selection_status, selection_available, tuning_joint_feasible,
failure_reason, chosen_initialization, selected_endpoint_stage_count,
selected_stage_grid_indices_json, q_by_time_json, tuning_coverage_json,
tuning_stage_width_json, tuning_micro_width, final_coverage_json,
final_wilson_lcb_json, final_stage_width_json, micro_normalized_width,
patient_normalized_width, tuning_stream_id, evaluation_stream_id,
n_tuning_rollouts, n_evaluation_rollouts, schedule_evaluations,
committed_updates, converged_at_pair, wall_time_seconds
```

The four initial method rows use these literal labels:

```python
METHOD_METADATA = {
    "current_profiled": ("REFERENCE", 0),
    "greedy": ("REFERENCE", 0),
    "joint_B": ("B", 2),
    "joint_2B": ("2B", 4),
}
```

`analysis_role` is `reference` for current/greedy and `joint_search` for B/2B. Search statuses are restricted to `SELECTED`, `NO_FEASIBLE_START`, and `WALL_TIME_CAP`. Profiled `SearchStart.stage_grid_indices` is `(None,) * 12`, greedy uses its 12 chosen indices, and upper endpoint uses `(100,) * 12`. The current-profiled record separately retains its one-element scale-grid index; every available row has a 12-element `q_by_time_json`.

Implement `run_phase0c_extension_seed` in the same module. It receives the three validated pair-4 `SearchState` objects for each scenario, reconstructs the identical frozen context and streams, calls `resume_cyclic_joint_coordinate_search`, and writes only the two `joint_8SP` rows plus their pair-8 surfaces/trace. It rejects a state whose schedule, metrics, indices, start name, or completed-pair count differs from the parent artifact.

Surfaces save stage grids; current, greedy, B, and 2B schedules; their tuning/final coverage and stage widths; and all three per-start pair-4 continuation states. Diagnostics save every `CoordinateStep` field. JSON serialization uses `allow_nan=False` and converts missing values to `null`.

- [ ] **Step 5: Add fail-closed and legacy-parity tests**

Test vector lengths, off-grid indices encoded as `-1`, deadline rollback, no-feasible start, unique stream IDs across both scenarios, and equality of newly computed current/greedy selections with direct existing Phase 0 calls on a tiny real synthetic context. For extension continuation, assert all three start states resume from completed pair 4, produce exactly two `joint_8SP` rows, reuse the original evaluation stream, and reject one-byte schedule/state-hash mutations. Assert fixed-T identities:

```python
assert record["micro_normalized_width"] == pytest.approx(
    np.mean(json.loads(record["final_stage_width_json"])), abs=1e-7
)
assert record["patient_normalized_width"] == pytest.approx(
    record["micro_normalized_width"], abs=1e-7
)
```

- [ ] **Step 6: Run focused and all Phase 0 seed tests**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_seed.py tests/per_step/test_phase0_seed.py tests/per_step/test_phase0_oracle.py
```

Expected: all pass; Phase 0A artifacts and setup-order characterizations remain unchanged.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/scpcp/phase0c_study.py tests/per_step/test_phase0c_seed.py
git commit -m "feat: orchestrate phase0c paired checkpoints"
```

### Task 4: GPU Runner, Smoke-Derived Wall Cap, Resume, and 8SP Authorization

**Files:**
- Create: `configs/phase0c_joint_search.yaml`
- Create: `scripts/run_phase0c_joint_search.py`
- Create: `tests/per_step/test_phase0c_runner.py`

**Interfaces:**
- Consumes: `run_phase0c_seed` and its exact eight-row artifact contract from Task 3; `scpcp.artifacts` atomic study helpers.
- Produces: `parse_seeds`, `canonical_config_sha256`, `calibrate_wall_cap`, `validate_seed_artifact`, `run_config`, `authorize_extension`, and CLI modes `smoke`, `initial`, `extension-8sp` for Tasks 5–8.

- [ ] **Step 1: Write RED parser, smoke, pinning, and fresh-run tests**

The config is a valid `ExperimentConfig` YAML copied from `configs/phase0_oracle.yaml` with:

```yaml
seeds:
  start: 10000
  stop: 10040
devices: [cuda:0, cuda:1]
output_dir: results/work/phase0c_joint_search
```

Test CLI validation for chunk size, workers, mode, seed bank, and output directory. A fake CUDA event log must be exactly:

```python
assert events == [
    "set_device:0",
    "enter_device:0",
    "run_seed:10000",
    "validate_seed:10000",
    "empty_cache:0",
]
```

Test a fresh inline executor produces one atomic seed directory and root `COMPLETE` only after revalidating all requested seeds.

- [ ] **Step 2: Run RED and confirm missing runner**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_runner.py
```

Expected: missing `run_phase0c_joint_search` module or parser functions.

- [ ] **Step 3: Implement smoke manifest and deterministic wall-cap calibration**

The smoke uses seed `9999`, both scenarios, maximum four pairs, `cuda:0`, chunk `16`, and the scientific config unchanged. Emit strict JSON:

```json
{
  "protocol": "phase0c_smoke_v1",
  "seed": 9999,
  "max_sweep_pairs": 4,
  "elapsed_seconds": 0.0,
  "max_memory_allocated_bytes": 0,
  "max_memory_reserved_bytes": 0,
  "recommended_max_seed_wall_seconds": 300,
  "source_tree_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "experiment_tree_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "config_sha256": "2222222222222222222222222222222222222222222222222222222222222222"
}
```

The numeric zeros above are test-fixture literals; production fills measured values and calculates:

```python
def calibrate_wall_cap(elapsed_seconds: float) -> int:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError("smoke elapsed_seconds must be finite and positive")
    return max(300, math.ceil(1.5 * elapsed_seconds / 300.0) * 300)
```

The initial mode accepts only a smoke manifest whose three hashes match the active source/experiment/config and whose seed/max-pairs are exact. It stores the derived cap in immutable execution metadata.

- [ ] **Step 4: Implement robust two-GPU spawn and deep seed validation**

Adapt the already-tested Phase 0 runner pattern into this new script without importing its four-row constants. Require exact eight row keys, exact columns, integer seed dtype, vector length 12, finite positive widths, `tuning_coverage >= .90` for selected joint rows, equal evaluation stream IDs within a scenario, distinct tuning/evaluation streams, no global collisions, exact NPZ keys/shapes/finiteness/index mapping, diagnostics/records/NPZ agreement, and matching source/config/seed-manifest/execution hashes.

Resume skips only fully materialized, deeply valid seeds. A missing/corrupt file, `.seed_*` temporary directory, provenance mismatch, root `COMPLETE` with a missing seed, or changed chunk/device/worker/cap/checkpoint value fails before any write.

- [ ] **Step 5: Implement extension authorization without running it**

`extension-8sp` requires an initial root with `COMPLETE`, a valid summary manifest, and decision exactly `EXTENSION_8SP_REQUIRED`. Freeze and validate:

```text
parent_study_manifest_sha256
checkpoint_decision_sha256
source_tree_sha256
experiment_tree_sha256
config_sha256
ordered seed manifest
```

For each seed/scenario, load all three pair-4 `SearchState` continuations and continue them through pair 8. The extension writes exactly two rows per seed (`joint_8SP` for both scenarios) under a new root and never mutates the initial root.

- [ ] **Step 6: Add adversarial resume and extension tests**

Cover 40→39 seeds, bool/float seed IDs, zero/negative/NaN widths, wrong vector lengths, stream collision, truncated NPZ member, records/NPZ mismatch, wrong parent hash, extension without authorization, pair-4 state hash mismatch, worker exception propagation, same-slot same-PID reuse, and failure before root `COMPLETE`.

- [ ] **Step 7: Run focused runner and prior runner regressions**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_runner.py tests/per_step/test_phase0_runner.py tests/per_step/test_study_artifacts.py
```

Expected: all pass, including real spawn/pickling coverage.

- [ ] **Step 8: Commit Task 4**

```bash
git add configs/phase0c_joint_search.yaml scripts/run_phase0c_joint_search.py tests/per_step/test_phase0c_runner.py
git commit -m "feat: add fail-closed phase0c gpu runner"
```

### Task 5: Phase 0C Statistics, Decision State Machine, and Atomic Summary

**Files:**
- Create: `scripts/summarize_phase0c_joint_search.py`
- Create: `tests/per_step/test_phase0c_summary.py`

**Interfaces:**
- Consumes: completed initial/extension study roots and validators from Task 4.
- Produces: `coverage_summary`, `paired_ratio_summary`, `decide_initial`, `decide_extension`, `load_validate_analyze`, `publish_summary`, and seven-file summary bundle for Tasks 6–8.

- [ ] **Step 1: Write independent literal RED tests for all numeric primitives**

Lock these values without using production helpers to derive expectations:

```python
current = np.array([1.0, 1.0, 1.0, 1.0])
b = np.array([0.95, 0.96, 0.94, 0.93])
b2 = np.array([0.94, 0.95, 0.93, 0.92])
assert paired_geometric_ratio(b, current) == pytest.approx(0.9449338571562925)
assert paired_geometric_ratio(b2, current) == pytest.approx(0.9349331496314753)
assert relative_budget_gain(0.9449338571562925, 0.9349331496314753) == pytest.approx(
    0.01058350005037767
)
```

For coverage, set one cell across 40 seeds to `x=np.array([.890+.001*i for i in range(40)])` and all other cells to `0.95`. Assert the worst LCB is `0.9038732857531458`; after subtracting `0.004`, assert `0.8998732857531458` and invalidity. Test all four coverage minima as distinct values.

- [ ] **Step 2: Write exact RED tests for the decision boundaries**

Use pure-function cases:

```python
assert decide_initial(valid=True, r_2b=0.92, delta_b=-0.10) == "PROMISING_ORACLE_DIAGNOSTIC"
assert decide_initial(valid=True, r_2b=0.926, delta_b=0.004301075268817208) == "STOP_SCALAR_SATURATED"
assert decide_initial(valid=True, r_2b=0.921, delta_b=0.005) == "EXTENSION_8SP_REQUIRED"
assert decide_initial(valid=False, r_2b=0.80, delta_b=0.20) == "STOP_SCALAR_UNAVAILABLE"
assert decide_extension(valid=True, r_8sp=0.92) == "PROMISING_ORACLE_DIAGNOSTIC"
assert decide_extension(valid=True, r_8sp=0.9200001) == "STOP_SCALAR_INSUFFICIENT"
```

All comparisons use raw doubles, never rendered strings.

- [ ] **Step 3: Run RED and verify missing summary functions**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_summary.py
```

Expected: failure only because the new summarizer/functions are absent.

- [ ] **Step 4: Implement strict inference**

For each checkpoint, stack fresh coverage as `[40,2,12]`. Compute seed means and sample SD (`ddof=1`), then:

```python
lcb = mean - 3.0440624276034796 * sd / math.sqrt(40)
coverage_valid = bool(np.min(lcb) >= 0.90)
```

Report minimum stage seed-mean, minimum simultaneous LCB, mean seedwise stage minimum, and raw seed-stage minimum. Use B and 2B fresh tail micro widths to compute `R_B`, `R_2B`, and `Delta_B`. Descriptive 10,000-replicate paired bootstrap intervals use sorted seeds and one shared index matrix from `np.random.default_rng(2_718_281)`; they do not alter branch decisions.

- [ ] **Step 5: Implement fail-closed study loading and deterministic atomic publication**

Require root `COMPLETE`, exact ordered `10000..10039`, valid seed artifacts, exact provenance, and no unavailable pair. Publish only after all checks pass:

```text
phase0c_summary.csv
phase0c_decision.json
phase0c_summary.md
phase0c_joint_search.pdf
phase0c_joint_search.svg
phase0c_joint_search.png
phase0c_summary_manifest.json
```

The manifest is installed last and lists byte length and SHA-256 for the other six outputs. Publish via staging plus rollback-safe replacement; a fault on any replacement leaves the prior valid bundle byte-identical and its manifest valid.

- [ ] **Step 6: Add 40-seed integration and mutation tests**

Generate a real-schema fixture with 40 seeds × 8 rows and NPZ/metadata. Mutate one fact at a time: 39 pairs, tuning coverage `.899999`, width `0`, `-1`, or `NaN`, wrong scenario, wrong method, wrong stream, duplicate stream across seeds, mismatched schedule, source/config/seed hash, root completion mismatch, extension parent mismatch. Each mutation must fail before outputs appear.

- [ ] **Step 7: Run focused statistics and publisher tests**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_summary.py
```

Expected: all pass and two independent processes produce byte-identical JSON/CSV/Markdown/SVG outputs for the same fixture.

- [ ] **Step 8: Commit Task 5**

```bash
git add scripts/summarize_phase0c_joint_search.py tests/per_step/test_phase0c_summary.py
git commit -m "feat: add phase0c decision analysis"
```

### Task 6: Honest Diagnostic Figure and Human-Readable Analysis Contract

**Files:**
- Modify: `scripts/summarize_phase0c_joint_search.py`
- Modify: `tests/per_step/test_phase0c_summary.py`

**Interfaces:**
- Consumes: validated analysis dictionary and atomic publisher from Task 5.
- Produces: publication-sized PDF/SVG/PNG and concise Markdown with the actual decision, denominators, aggregation definitions, and limits.

- [ ] **Step 1: Write RED semantic and geometry tests before renderer code**

The figure contains four panels:

1. final checkpoint standard/tail stage mean and simultaneous LCB with `n=40` and a visible `0.90` target;
2. `R_B`, `R_2B`, and optional `R_8SP` on a nontruncated log-ratio axis with visible `0.92` and `1.00` references;
3. paired `1-W_2B/W_B` convergence distribution with `0.005` reference;
4. feasibility, winning initialization, endpoint, runtime, and VRAM audit.

Capture the Matplotlib figure and assert every legend bbox is outside data axes, the coverage y-range includes `0.90`, ratio axis includes `0.92` and `1.00`, denominators are present, and title is the literal decision. SVG text must contain `Oracle diagnostic`, `40/40`, `simultaneous LCB`, and must not contain `SOTA`, `GO`, or `significantly superior`.

- [ ] **Step 2: Run targeted RED**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_summary.py -k "figure or markdown"
```

Expected: failures identify the missing four-panel semantics or renderer.

- [ ] **Step 3: Implement the smallest renderer satisfying the contract**

Use fixed `183 mm × 120 mm`, quantitative axes, figure-level legends outside axes, editable SVG text, embedded PDF fonts, and deterministic metadata. The Markdown must explain why per-stage seed means, seedwise minima, raw minima, and simultaneous LCB differ; state that only the simultaneous LCB gates coverage; state that this is fixed-T/all-active oracle search; and state that a promising decision is not deployable or SOTA.

- [ ] **Step 4: Verify raster/vector outputs and unavailable extension behavior**

Test initial decisions without 8SP render two ratio points and label 8SP `not run`; authorized extension renders the third point. Verify PNG magic/dimensions, PDF magic/page size, SVG editable text, all figures closed, and an unavailable checkpoint displays `unavailable` rather than zero.

- [ ] **Step 5: Run focused and full summary tests**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0c_summary.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add scripts/summarize_phase0c_joint_search.py tests/per_step/test_phase0c_summary.py
git commit -m "feat: render honest phase0c diagnostics"
```

### Task 7: Full Regression, Independent Review, and One-Seed GPU Smoke

**Files:**
- Modify only files from Tasks 1–6 when a reviewed defect requires a fix.
- Record ignored execution evidence under `.superpowers/sdd/2026-08-18-phase0c-joint-search/`.

**Interfaces:**
- Consumes: all Phase 0C code/tests and immutable Phase 0A regression suite.
- Produces: reviewed frozen experiment commit and a hash-bound smoke manifest that authorizes Task 8.

- [ ] **Step 1: Run Phase 0-focused and full CPU suites on the remote worktree**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q tests/per_step/test_phase0*.py
/home/ubuntu/anaconda3/envs/ucp/bin/python -m pytest -q
git diff --check 59b7f1e608d59db431982e84864d28f81c309e79..HEAD
```

Expected: every test passes; no whitespace errors; old Phase 0A deterministic tests remain unchanged.

- [ ] **Step 2: Perform independent whole-branch code/statistical review**

The reviewer must check the full diff from `59b7f1e`, the approved spec, this plan, source/config hashing, search nesting, cache equivalence, 24-cell LCB, exact thresholds, fail-closed behavior, extension authorization, and figure semantics. Fix every Critical/Important issue through reviewed TDD before proceeding.

- [ ] **Step 3: Freeze the experiment commit and run the one-seed smoke**

Run on remote GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/anaconda3/envs/ucp/bin/python \
  scripts/run_phase0c_joint_search.py smoke \
  --config configs/phase0c_joint_search.yaml \
  --seed 9999 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --candidate-chunk-size 16 \
  --output-dir results/work/phase0c_smoke_9999
```

Expected: exit 0, seed and root `COMPLETE`, no OOM/traceback, both scenarios complete through pair 4, positive timing, measured peak allocated/reserved VRAM, and a manifest whose recommended cap equals the fixed formula.

- [ ] **Step 4: Verify smoke replay and resource evidence**

Run the smoke validator, compare sampled cached coordinate surfaces with full replay within the tested CUDA tolerance, inspect `nvidia-smi` logs, and ensure the derived wall cap is at least the measured time. Any source/config change after smoke invalidates the manifest and requires a new smoke.

- [ ] **Step 5: Commit only reviewed fixes, never runtime results**

```bash
git status --short
git log -1 --format=%H
```

Expected: code worktree clean; smoke outputs stay outside git.

### Task 8: Formal 40-Seed GPU Study, Conditional 8SP Extension, and Complete Analysis

**Files:**
- Do not modify source or configuration after the Task 7 smoke.
- Create remote runtime artifacts under `results/work/phase0c_joint_search_${SHORT_SHA}/` and, only if authorized, `results/work/phase0c_joint_search_8sp_${SHORT_SHA}/`, where `SHORT_SHA=$(git rev-parse --short=12 HEAD)` is resolved before launch.
- Copy final user-facing analysis to `/Users/bule/Documents/Codex/2026-08-17/api-home-ubuntu-zmh-performativecp-ssh/outputs/phase0c_joint_search_analysis/`.

**Interfaces:**
- Consumes: frozen Task 7 commit/config, signed smoke manifest, GPUs `cuda:0,cuda:1`, and ordered seeds `10000..10039`.
- Produces: complete initial study, optional authorized extension, validated seven-file summary, Chinese analysis handoff, and an evidence-backed next-route decision.

- [ ] **Step 1: Launch the immutable initial study**

From the remote linked worktree, run:

```bash
/usr/bin/time -v /home/ubuntu/anaconda3/envs/ucp/bin/python \
  scripts/run_phase0c_joint_search.py initial \
  --config configs/phase0c_joint_search.yaml \
  --seeds 10000:10040 \
  --devices cuda:0,cuda:1 \
  --workers-per-device 1 \
  --candidate-chunk-size 16 \
  --smoke-manifest results/work/phase0c_smoke_9999/smoke_manifest.json \
  --output-dir results/work/phase0c_joint_search_$(git rev-parse --short=12 HEAD)
```

Do not read partial `records.csv`, NPZ metrics, or interim ratios while the process is live. Monitor only PID, per-seed `COMPLETE`/`FAILED` markers, GPU utilization/memory/temperature, wall time, and error/OOM/traceback logs.

- [ ] **Step 2: Validate completion before analysis**

Require exit 0, root `COMPLETE`, exact seed directories `10000..10039`, 40/40 seed `COMPLETE`, zero `FAILED`, 320 exact initial rows, all deep validators passing, and manifest/provenance hashes matching the frozen commit/config/smoke.

- [ ] **Step 3: Produce the initial summary and obey the machine decision**

```bash
/home/ubuntu/anaconda3/envs/ucp/bin/python \
  scripts/summarize_phase0c_joint_search.py \
  --input-dir results/work/phase0c_joint_search_$(git rev-parse --short=12 HEAD)
```

If the decision is `STOP_SCALAR_UNAVAILABLE`, `STOP_SCALAR_SATURATED`, or `PROMISING_ORACLE_DIAGNOSTIC`, do not run 8SP. If and only if it is `EXTENSION_8SP_REQUIRED`, continue to Step 4.

- [ ] **Step 4: Run the machine-authorized 8SP extension when required**

```bash
/usr/bin/time -v /home/ubuntu/anaconda3/envs/ucp/bin/python \
  scripts/run_phase0c_joint_search.py extension-8sp \
  --config configs/phase0c_joint_search.yaml \
  --parent-dir results/work/phase0c_joint_search_$(git rev-parse --short=12 HEAD) \
  --decision-json results/work/phase0c_joint_search_$(git rev-parse --short=12 HEAD)/checkpoint_analysis/phase0c_decision.json \
  --devices cuda:0,cuda:1 \
  --workers-per-device 1 \
  --candidate-chunk-size 16 \
  --output-dir results/work/phase0c_joint_search_8sp_$(git rev-parse --short=12 HEAD)
```

Validate exact 40 seeds, 80 extension rows, all parent/decision hashes, continuation hashes, and zero failures. Then rerun the summarizer with both `--input-dir` and `--extension-dir` to obtain the final branch decision.

- [ ] **Step 5: Independently recompute and audit every reported statistic**

From raw complete artifacts, independently reproduce the 24-cell means/LCBs for each gated checkpoint, all four coverage minima, B/2B/8SP ratios, `Delta_B`, denominators, endpoints, winning starts, convergence, runtime, and VRAM. Require maximum numeric disagreement with `phase0c_decision.json` at most `1e-12` and all manifest file hashes/lengths exact.

- [ ] **Step 6: Perform final visual QA at original size**

Inspect PNG original resolution and independently rasterized PDF. Require no legend/data overlap, no clipping, coverage axis including `0.90`, ratio axis including `0.92` and `1.00`, visible denominators, honest decision/title, editable SVG text, and embedded PDF fonts. A visual defect changes source, invalidates the smoke/study provenance, and therefore requires a new frozen run rather than reusing old metrics as a formal artifact.

- [ ] **Step 7: Deliver the complete Chinese analysis without overstating the result**

Copy the validated summary bundle and a self-contained Chinese HTML/Markdown explanation into:

```text
/Users/bule/Documents/Codex/2026-08-17/api-home-ubuntu-zmh-performativecp-ssh/outputs/phase0c_joint_search_analysis/
```

Lead with the actual decision and its consequences:

- `PROMISING_ORACLE_DIAGNOSTIC`: joint scalar search reaches the preregistered headroom threshold; retain Phase 0A as active and write a separate practical-method design before any confirmation.
- `STOP_SCALAR_SATURATED`, `STOP_SCALAR_INSUFFICIENT`, or `STOP_SCALAR_UNAVAILABLE`: retain Phase 0A as active and move to the separately designed risk-adaptive oracle representation screen; do not keep tuning the scalar grid.

Always report that this is a development-bank oracle audit, not a deployable method, confirmation result, or SOTA claim.
