"""Public experiment-facing API for SC-PCP.

The package deliberately exposes the calibration operation rather than the
project's paper-specific runners.  Those runners, their frozen settings, and
their result readers are maintained locally under ``internal/``.
"""

from scpcp.experiments.per_step import (
    PerStepCalibrationInputs,
    calibrate_per_step_marginal,
)

__all__ = ["PerStepCalibrationInputs", "calibrate_per_step_marginal"]
