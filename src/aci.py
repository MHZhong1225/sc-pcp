"""Article-faithful Adaptive Conformal Inference (ACI) adapter.

This module implements the update in Gibbs and Cand\u00e8s (2021), Eq. (2),
without batching, clipping, or truncating the score history.  The only
project-specific choice is how a patient-arrival stream is exposed to the
algorithm: each decision stage is one chronological ACI stream, while one
patient is deployed at a time with the resulting vector of stage radii.

The adapter deliberately fails before deployment if the exact ACI update asks
for an empty or unbounded set.  The current treatment-policy interface only
accepts finite, nonnegative box radii; silently clipping ``alpha_t`` would
change the published algorithm.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from baselines import OnlineBaselineResult
from data import TrajectoryBatch
from scores import score_batch
from simulator import rollout


ACI_PAPER = "https://arxiv.org/abs/2106.00170"
ACI_UPDATE = "alpha_{t+1}=alpha_t+gamma*(alpha-error_t)"
ACI_SEED_STRIDE = 17_923


class ACIRollout(Protocol):
    """One-patient target-policy rollout required by the ACI adapter."""

    def __call__(
        self,
        environment: object,
        policy: object,
        *,
        n: int,
        horizon: int,
        seed: int,
        device: str | torch.device,
        q: Tensor,
    ) -> TrajectoryBatch: ...


class ACIInterfaceError(RuntimeError):
    """The exact ACI set cannot be represented by this policy interface."""


def aci_empirical_radius(scores: Tensor, alpha_t: float) -> Tensor:
    """Return ACI's conformal quantile with its explicit ``+infinity`` atom.

    For ``m`` preceding scores, the ordered support is the ``m`` scores plus a
    point at infinity.  This is the conformal quantile convention used by ACI,
    rather than a clipped finite quantile.  A non-positive ``alpha_t`` requests
    the infinite region; ``alpha_t >= 1`` requests an empty set.  Neither has a
    policy meaning in the present finite-radius interface, so both fail closed.
    """

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("ACI scores must be a nonempty one-dimensional tensor")
    if not bool(torch.isfinite(scores).all()) or not bool((scores >= 0).all()):
        raise ValueError("ACI scores must be finite and nonnegative")
    if not 0.0 < alpha_t < 1.0:
        region = "unbounded" if alpha_t <= 0.0 else "empty"
        raise ACIInterfaceError(
            f"exact ACI requested a {region} prediction set at alpha_t={alpha_t:.8g}; "
            "the treatment policy requires an explicit representation for that set"
        )
    rank = math.ceil((scores.numel() + 1) * (1.0 - alpha_t))
    if rank > scores.numel():
        raise ACIInterfaceError(
            "exact ACI selected its +infinity conformal atom; the treatment policy "
            "cannot represent an unbounded box"
        )
    return scores.sort().values[rank - 1]


@torch.no_grad()
def run_aci_panel(
    environment: object,
    policy: object,
    outcome_model: object,
    initial_scores: Tensor,
    *,
    alpha: float,
    gamma: float,
    target_deployments: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    rollout_fn: ACIRollout | None = None,
) -> OnlineBaselineResult:
    """Run Eq. (2) once per arriving patient and per fixed decision stage.

    The state of each panel member is ``(alpha_t, S_{<t})``.  At an arrival we
    form all stage radii from that pre-arrival state, observe exactly one patient
    rollout, append its stage scores, and apply the binary-error update at every
    stage.  Thus no batch-average update, clipping, score-window truncation, or
    substitute controller is involved.
    """

    if initial_scores.ndim != 2 or initial_scores.shape[1] != horizon:
        raise ValueError("initial_scores must have shape [N, horizon]")
    if target_deployments < 1:
        raise ValueError("ACI target_deployments must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("ACI alpha must lie in (0, 1)")
    if gamma <= 0.0:
        raise ValueError("ACI gamma must be positive")

    histories = [initial_scores[:, time].detach().cpu().clone() for time in range(horizon)]
    alpha_by_time = torch.full((horizon,), float(alpha), dtype=torch.float64)
    covered = torch.zeros(horizon, dtype=torch.float64)
    pathwise_hits = 0
    worst_by_arrival: list[float] = []
    rollout_impl = rollout if rollout_fn is None else rollout_fn

    for arrival in range(target_deployments):
        radii = torch.stack(
            [
                aci_empirical_radius(history, float(alpha_by_time[time]))
                for time, history in enumerate(histories)
            ]
        )
        deployed = rollout_impl(
            environment,
            policy,
            n=1,
            horizon=horizon,
            seed=seed + ACI_SEED_STRIDE * arrival,
            device=device,
            q=radii.to(device=device, dtype=initial_scores.dtype),
        )
        observed_scores = score_batch(
            outcome_model,
            deployed.current_states(),
            deployed.actions,
            deployed.outcomes,
        ).detach().cpu()
        if observed_scores.shape != (1, horizon):
            raise RuntimeError("ACI score function must return shape [1, horizon]")
        errors = (observed_scores[0] > radii).to(torch.float64)
        hits = ~errors.to(torch.bool)
        covered += hits.to(torch.float64)
        pathwise_hits += int(hits.all().item())
        worst_by_arrival.append(float(hits.to(torch.float64).min().item()))
        for time in range(horizon):
            histories[time] = torch.cat((histories[time], observed_scores[:, time]))
        alpha_by_time = alpha_by_time + float(gamma) * (float(alpha) - errors)

    final_radii = torch.stack(
        [
            aci_empirical_radius(history, float(alpha_by_time[time]))
            for time, history in enumerate(histories)
        ]
    ).to(dtype=initial_scores.dtype)
    return OnlineBaselineResult(
        radius_by_time=final_radii,
        target_deployments=target_deployments,
        rounds=target_deployments,
        adaptation_per_time_coverage=(covered / target_deployments).to(torch.float32),
        adaptation_round_worst_coverage=tuple(worst_by_arrival),
        adaptation_pathwise_coverage=pathwise_hits / target_deployments,
    )
