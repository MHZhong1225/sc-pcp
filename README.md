# **SC-PCP**

This repository contains a minimal, runnable implementation of SC-PCP. The
public release keeps the proposed committed-prefix marginal calibration method,
its reusable Python API, and the tests needed to validate the implementation.

## **Install**

SC-PCP requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you use conda:

```bash
conda create -n scpcp python=3.11
conda activate scpcp
python -m pip install -e .
```

PyTorch is installed as a package dependency. For GPU use, install the PyTorch
build matching your CUDA environment before installing SC-PCP.

## **Run**

For a quick smoke test:

```bash
python -m pytest -q tests/test_public_api.py
```

This runs a complete two-stage SC-PCP calibration example and checks the
selected stagewise radii.

Run all public tests with:

```bash
python -m pytest -q
```

The tests cover committed-prefix weighting, candidate selection, numerical
stability, and explicit failure when no candidate radius is feasible.

## **Use SC-PCP**

The public entry point is:

```python
from scpcp import PerStepCalibrationInputs, calibrate_per_step_marginal
```

Create `PerStepCalibrationInputs` with:

```text
trajectories    logged TrajectoryBatch
scores          conformity scores with shape [n, stages]
stage_grids     candidate radii with shape [stages, candidates]
outcome_sd      training-outcome scale used to compare normalized widths
target_coverage desired per-step marginal coverage
```

Then run:

```python
result = calibrate_per_step_marginal(
    inputs,
    target_policy=target_policy,
    logging_policy=logging_policy,
    outcome_model=outcome_model,
)
```

If `result.selection_available` is true, `result.radii` contains the selected
stagewise radii. Otherwise, `result.failure_stage` identifies the first stage
without a feasible candidate.

See
[`tests/test_public_api.py`](tests/test_public_api.py) for a
complete runnable example, including the required policy interfaces and tensor
shapes.

The core implementation is in
[`src/marginal_prefix.py`](src/marginal_prefix.py). The public wrapper is
[`src/experiments.py`](src/experiments.py).

Clinical data, patient-derived caches, paper result bundles, and generated
figures are not distributed with this source release.

## Baseline reproduction boundary

The research adapters keep article algorithms separate from this package's data
layout. `MFCS`, `PRC`, and `SPCI` require a caller-supplied checkout of the
specific upstream release; the adapters verify its revision before running.
`SPCI` additionally requires the upstream-pinned
`sklearn-quantile==0.0.21`. A mismatched dependency is reported as unavailable
rather than substituted with another version or a local score-based method.
ACI implements the published sequential update directly, with one binary
update per patient arrival and no clipping or batched replacement.
