# Self-Consistent Performative Conformal Prediction

This project implements **SC-PCP**.

## Installation

SC-PCP requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```


PyTorch is a required dependency. For GPU use, install the PyTorch build that matches the local CUDA environment before installing SC-PCP.

## Quick Start

Run the public two-stage example:

```bash
python -m pytest -q tests/test_public_api.py
```

Run all public tests with:

```bash
python -m pytest -q
```

The public calibration interface is:

```python
from data import TrajectoryBatch
from scpcp import PerStepCalibrationInputs, calibrate_per_step_marginal

inputs = PerStepCalibrationInputs(
    trajectories=trajectories,
    scores=scores,
    stage_grids=stage_grids,
    outcome_sd=outcome_sd,
    target_coverage=0.90,
)

result = calibrate_per_step_marginal(
    inputs,
    target_policy=target_policy,
    logging_policy=logging_policy,
outcome_model=outcome_model,
)
```

## Run Paper Experiments

The public package contains the method and tests. The complete clinical pipeline additionally requires authorized datasets and the private, source-pinned `internal/` research archive.

Activate the project environment once:

```bash
conda activate ucp
```

Then run a dataset from the repository root:

```bash
python run.py --dataset eicu
```

`--dataset` accepts `synthetic`, `mimic_iv`, `eicu`, `inspire`, and
`mimic_cxr`.

Run the complete five-dataset suite with:

```bash
python run.py --dataset all
```

## Datasets

The paper study contains one synthetic system and four controlled clinical
benchmarks:

| Dataset | Sequential response | Horizon | Action space |
|---|---|---|---|
| Synthetic | Two correlated outcomes under signed action--difficulty feedback | 12 stages | Three categorical actions |
| MIMIC-IV | Hypotension and tachycardia burden | `12 x 4 h` | Fluid--vasopressor intensity grid |
| eICU | Hypotension and tachycardia burden | `12 x 4 h` | Fluid--vasopressor intensity grid |
| INSPIRE | Hypotension and hypertension burden | `12 x 10 min` | None, fluid only, or vasopressor-containing treatment |
| MIMIC-CXR | Hypoxemia and tachypnea burden | `6 x 6 h` | No support, non-invasive support, or invasive ventilation |

The clinical tasks are patient-informed controlled deployment benchmarks, not prospective clinical evaluations. The MIMIC-CXR state additionally includes a frozen DenseNet-121 embedding of the index radiograph.

## Data Layout

The public method consumes a `TrajectoryBatch`:

- `states`: `[N,T+1,D]`, with one state before each action and one terminal
  state;
- `actions`: `[N,T]`;
- `outcomes`: `[N,T,Y]`, where entry `t` is the post-action response
  `Y_(t+1)`; and
- `patient_ids`: `[N]`.

Calibration scores have shape `[N,T]`, stage grids have shape `[T,K]`, and `outcome_sd` has shape `[Y]`. The logging policy implements `probabilities(states) -> [N,A]`; the radius-responsive target policy implements `probabilities_for_grid(states, radii) -> [N,K,A]`; and the frozen outcome model returns coordinate-wise means and positive scales.

All clinical roles are split by patient identifier. A patient appearing in more than one role is not allowed.

## Experimental Protocol

The target per-stage coverage is `0.90`. The primary feedback setting is `gamma=-4`; `gamma` in `{-2, 0, 2, 4}` forms the prespecified sensitivity analysis. Here `gamma` belongs to the deployment environment, not to SC-PCP.

Clinical patients are split into:

```text
D_pred / D_fid / D_env = 40% / 20% / 40%
```

`D_pred` fits and freezes the outcome and logging-policy models. `D_fid` fixes the radius-response range and supplies the native SPCI training stream. `D_env` constructs the frozen controlled evaluator and is never used for calibration. The synthetic benchmark instead uses known logging and target propensities and a known transition kernel.

For each dataset, seed, and feedback setting, the fixed-schedule methods use 3,000 fresh logging-policy trajectories for calibration. The first 1,000 fix the 101-point stage grids, and 20,000 independent target-policy trajectories are used only for evaluation. All methods use 20 seeds at `gamma=-4`. At each sensitivity setting, SPCI uses three prespecified seeds because of its higher computational cost; the other methods retain 20 seeds.



## Methods and Metrics

The canonical comparison contains exactly:

- Standard CP
- ACI
- MFCS
- SPCI
- PRC
- SC-PCP (Our)

Continuous Causal CP is evaluated separately as a diagnostic because its native output is an individualized interval, not a fixed pre-deployment stagewise schedule.

The reported metrics are:

- **Marginal worst-step coverage (WSC):**
  `min_t mean_seed(C_seed,t)`, the primary coverage metric;
- **MeanCov:** average coverage across stages and seeds;
- **average normalized coordinate width:** the primary efficiency metric for
  rectangular prediction regions;
- **normalized region area:** a secondary comparison that preserves the native
  box or ellipsoid geometry.



## Data Availability
The source release does not include restricted clinical data, patient-derived caches, paper result bundles, or generated figures. MIMIC-IV, eICU, INSPIRE, and MIMIC-CXR must be obtained from their respective custodians under the applicable data-use terms. Synthetic trajectories are generated by the included simulator.

## Source Layout

- `run.py`: concise dataset-level entry point for the paper experiment.
- `src/marginal_prefix.py`: committed-prefix SC-PCP selector.
- `src/experiments.py`: public calibration wrapper.
- `src/data.py`: trajectory containers and patient-level split utilities.
- `src/real_data.py`: leakage-safe clinical trajectory builders.
- `src/cxr.py`: frozen MIMIC-CXR encoder.
- `tests/`: public API and method tests.
- `results/`: generated artifacts; intentionally not source-controlled.
