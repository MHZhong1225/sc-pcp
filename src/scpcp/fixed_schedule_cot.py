"""Fixed-schedule occupancy transport for controlled recovery diagnostics.

This module is deliberately separate from the paper SC-PCP implementation.
For a *fixed* stagewise schedule it learns the state marginal ratio
``rho_t(S_t)`` by the forward occupancy recursion, then forms the observed
state-action weight ``rho_t(S_t) pi_t(A_t|S_t) / mu_t(A_t|S_t)``.  It is useful
for testing whether marginal transport can recover a score law when full
trajectory importance weights have poor effective sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from scpcp.config import COTConfig
from scpcp.data import TrajectoryBatch


class _OccupancyHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = feature_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(current_dim, width), nn.ReLU()))
            current_dim = width
        output = nn.Linear(current_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, math.log(math.expm1(1.0)))
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return nn.functional.softplus(self.network(features).squeeze(1)) + 1e-6


class FixedScheduleCOT(nn.Module):
    """One state-ratio head for each noninitial stage of a fixed schedule."""

    def __init__(
        self,
        *,
        state_dim: int,
        horizon: int,
        outcome_model: object,
        config: COTConfig,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.outcome_model = outcome_model
        self.config = config
        representation_dim = outcome_model.config.representation_dim
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("normalization_scales", torch.ones(max(0, horizon - 1)))
        self.heads = nn.ModuleList(
            _OccupancyHead(state_dim + representation_dim, config.hidden_dims)
            for _ in range(max(0, horizon - 1))
        )

    @torch.no_grad()
    def rho(self, stage: int, states: Tensor) -> Tensor:
        if stage == 0:
            return torch.ones(len(states), dtype=states.dtype, device=states.device)
        if not 0 < stage < self.horizon:
            raise ValueError("stage is outside the fixed-schedule horizon")
        raw = self.heads[stage - 1](self.features(states))
        return raw / self.normalization_scales[stage - 1].to(raw).clamp_min(1e-8)

    @torch.no_grad()
    def features(self, states: Tensor) -> Tensor:
        normalized_state = ((states - self.state_center) / self.state_scale).clamp(-10.0, 10.0)
        representation = self.outcome_model.representation(states)
        return torch.cat((normalized_state, representation), dim=1)


@dataclass(frozen=True)
class FixedScheduleCOTDiagnostics:
    validation_mse: tuple[float, ...]
    validation_normalization_error: tuple[float, ...]


@dataclass(frozen=True)
class FittedFixedScheduleCOT:
    model: FixedScheduleCOT
    schedule: Tensor
    diagnostics: FixedScheduleCOTDiagnostics


@dataclass(frozen=True)
class FixedScheduleWeightDiagnostics:
    maximum_weight: Tensor
    effective_sample_size: Tensor


def fit_fixed_schedule_cot(
    batch: TrajectoryBatch,
    *,
    schedule: Tensor,
    target_policy: object,
    logging_policy: object,
    outcome_model: object,
    config: COTConfig,
    device: str | torch.device,
    seed: int,
) -> FittedFixedScheduleCOT:
    """Fit the forward occupancy recursion for one previously frozen schedule.

    The regression loss is MSE, so its population target is the conditional
    mean defining the occupancy recursion.  This differs intentionally from
    the historical practical COT's robust Huber objective.
    """

    if schedule.shape != (batch.horizon,):
        raise ValueError("schedule must have shape [T]")
    resolved = torch.device(device)
    source = batch.to(resolved)
    frozen_schedule = schedule.to(device=resolved, dtype=source.states.dtype)
    model = FixedScheduleCOT(
        state_dim=source.state_dim,
        horizon=source.horizon,
        outcome_model=outcome_model,
        config=config,
    ).to(resolved)
    current_states = source.current_states().reshape(-1, source.state_dim)
    model.state_center.copy_(current_states.mean(dim=0))
    model.state_scale.copy_(current_states.std(dim=0).clamp_min(1e-4))

    train_indices, validation_indices = _patient_split(source, seed=seed)
    validation_mse: list[float] = []
    validation_normalization_error: list[float] = []
    generator = torch.Generator(device=resolved).manual_seed(seed)
    for stage in range(source.horizon - 1):
        head = model.heads[stage]
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for _ in range(config.epochs):
            permutation = train_indices[torch.randperm(len(train_indices), generator=generator, device=resolved)]
            for rows in permutation.split(config.batch_size):
                pseudo_target = _pseudo_target(
                    model,
                    source,
                    stage=stage,
                    rows=rows,
                    schedule=frozen_schedule,
                    target_policy=target_policy,
                    logging_policy=logging_policy,
                )
                prediction = head(model.features(source.states[rows, stage + 1]))
                loss = nn.functional.mse_loss(prediction, pseudo_target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
                optimizer.step()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            training_raw = head(model.features(source.states[train_indices, stage + 1]))
            model.normalization_scales[stage].copy_(training_raw.mean().clamp_min(1e-8))
            validation_prediction = model.rho(stage + 1, source.states[validation_indices, stage + 1])
            validation_target = _pseudo_target(
                model,
                source,
                stage=stage,
                rows=validation_indices,
                schedule=frozen_schedule,
                target_policy=target_policy,
                logging_policy=logging_policy,
            )
            validation_mse.append(float(nn.functional.mse_loss(validation_prediction, validation_target).item()))
            validation_normalization_error.append(float(validation_prediction.mean().sub(1.0).abs().item()))
    model.eval()
    return FittedFixedScheduleCOT(
        model=model,
        schedule=frozen_schedule.detach().cpu(),
        diagnostics=FixedScheduleCOTDiagnostics(
            validation_mse=tuple(validation_mse),
            validation_normalization_error=tuple(validation_normalization_error),
        ),
    )


@torch.no_grad()
def fixed_schedule_state_action_weights(
    fitted: FittedFixedScheduleCOT,
    batch: TrajectoryBatch,
    *,
    target_policy: object,
    logging_policy: object,
) -> tuple[Tensor, FixedScheduleWeightDiagnostics]:
    """Return uncapped state-action density-ratio estimates indexed ``[N,T]``."""

    model = fitted.model
    device = next(model.parameters()).device
    source = batch.to(device)
    schedule = fitted.schedule.to(device=device, dtype=source.states.dtype)
    weights: list[Tensor] = []
    for stage in range(source.horizon):
        states = source.states[:, stage]
        actions = source.actions[:, stage]
        rho = model.rho(stage, states)
        numerator = target_policy.probabilities(states, schedule[stage]).gather(1, actions[:, None]).squeeze(1)
        denominator = logging_policy.probabilities(states).gather(1, actions[:, None]).squeeze(1)
        weights.append(rho * numerator / denominator.clamp_min(1e-12))
    stacked = torch.stack(weights, dim=1)
    normalized = stacked / stacked.sum(dim=0, keepdim=True).clamp_min(1e-12)
    effective_size = normalized.sum(dim=0).square() / normalized.square().sum(dim=0).clamp_min(1e-12)
    return stacked, FixedScheduleWeightDiagnostics(
        maximum_weight=stacked.max(dim=0).values,
        effective_sample_size=effective_size,
    )


@torch.no_grad()
def _pseudo_target(
    model: FixedScheduleCOT,
    batch: TrajectoryBatch,
    *,
    stage: int,
    rows: Tensor,
    schedule: Tensor,
    target_policy: object,
    logging_policy: object,
) -> Tensor:
    states = batch.states[rows, stage]
    actions = batch.actions[rows, stage]
    previous_ratio = model.rho(stage, states)
    numerator = target_policy.probabilities(states, schedule[stage]).gather(1, actions[:, None]).squeeze(1)
    denominator = logging_policy.probabilities(states).gather(1, actions[:, None]).squeeze(1)
    return previous_ratio * numerator / denominator.clamp_min(1e-12)


def _patient_split(batch: TrajectoryBatch, *, seed: int) -> tuple[Tensor, Tensor]:
    patients = torch.unique(batch.patient_ids.detach().cpu(), sorted=True)
    if len(patients) == 1:
        rows = torch.arange(batch.n, device=batch.states.device)
        return rows, rows
    generator = torch.Generator().manual_seed(seed)
    ordered = patients[torch.randperm(len(patients), generator=generator)]
    cutoff = max(1, min(len(ordered) - 1, round(0.85 * len(ordered))))
    train_patients, validation_patients = ordered[:cutoff], ordered[cutoff:]
    patient_ids = batch.patient_ids.detach().cpu()
    train_rows = torch.isin(patient_ids, train_patients).nonzero().squeeze(1).to(batch.states.device)
    validation_rows = torch.isin(patient_ids, validation_patients).nonzero().squeeze(1).to(batch.states.device)
    return train_rows, validation_rows
