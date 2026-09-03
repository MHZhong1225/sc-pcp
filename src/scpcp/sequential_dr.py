"""Fixed-schedule sequential doubly robust score-law transport.

This module is deliberately isolated from the canonical SC-PCP selector.  It
estimates a target-policy score CDF for a *previously frozen* stagewise
schedule.  The fitted continuation values are learned on one source sample and
the doubly robust correction is evaluated on an independent source sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scpcp.config import COTConfig
from scpcp.data import TrajectoryBatch


class _ContinuationHead(nn.Module):
    def __init__(self, feature_dim: int, n_targets: int, grid_size: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = feature_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(current_dim, width), nn.ReLU()))
            current_dim = width
        layers.append(nn.Linear(current_dim, n_targets * grid_size))
        self.network = nn.Sequential(*layers)
        self.n_targets = n_targets
        self.grid_size = grid_size

    def forward(self, features: Tensor) -> Tensor:
        values = torch.sigmoid(self.network(features))
        return values.reshape(len(features), self.n_targets, self.grid_size)


class FixedScheduleSequentialDR(nn.Module):
    """Backward continuation models for one fixed target policy schedule."""

    def __init__(
        self,
        *,
        state_dim: int,
        horizon: int,
        n_actions: int,
        outcome_model: object,
        score_grid: Tensor,
        config: COTConfig,
    ) -> None:
        super().__init__()
        if score_grid.ndim != 2 or score_grid.shape[0] != horizon:
            raise ValueError("score_grid must have shape [T, K]")
        self.horizon = horizon
        self.n_actions = n_actions
        self.outcome_model = outcome_model
        self.grid_size = score_grid.shape[1]
        representation_dim = outcome_model.config.representation_dim
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("score_grid", score_grid)
        self.heads = nn.ModuleList(
            _ContinuationHead(
                state_dim + representation_dim + n_actions,
                horizon - stage,
                self.grid_size,
                config.hidden_dims,
            )
            for stage in range(horizon)
        )

    def features(self, states: Tensor, actions: Tensor) -> Tensor:
        normalized_states = ((states - self.state_center) / self.state_scale).clamp(-10.0, 10.0)
        with torch.no_grad():
            representation = self.outcome_model.representation(states)
        action_features = torch.nn.functional.one_hot(actions.to(torch.long), self.n_actions).to(normalized_states)
        return torch.cat((normalized_states, representation, action_features), dim=1)

    def q_values(self, stage: int, states: Tensor, actions: Tensor) -> Tensor:
        if not 0 <= stage < self.horizon:
            raise ValueError("stage is outside the fixed-schedule horizon")
        return self.heads[stage](self.features(states, actions))

    @torch.no_grad()
    def values(self, stage: int, states: Tensor, target_policy: object, schedule: Tensor) -> Tensor:
        """Return ``V_stage`` for all terminal stages from ``stage`` onward."""

        probabilities = target_policy.probabilities(states, schedule[stage])
        repeated_states = states[:, None, :].expand(-1, self.n_actions, -1).reshape(-1, states.shape[1])
        repeated_actions = torch.arange(self.n_actions, device=states.device).repeat(len(states))
        action_values = self.q_values(stage, repeated_states, repeated_actions).reshape(
            len(states), self.n_actions, self.horizon - stage, self.grid_size
        )
        return (probabilities[:, :, None, None] * action_values).sum(dim=1)


@dataclass(frozen=True)
class FittedSequentialDR:
    model: FixedScheduleSequentialDR
    schedule: Tensor
    validation_mse: tuple[float, ...]


def make_score_cdf_grid(scores: Tensor, *, size: int = 201) -> Tensor:
    """Freeze a common score-CDF grid from source-only scores."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [N, T]")
    probabilities = torch.linspace(0.001, 0.999, size, device=scores.device, dtype=scores.dtype)
    return torch.quantile(scores, probabilities, dim=0).T.contiguous()


def fit_fixed_schedule_sequential_dr(
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    schedule: Tensor,
    target_policy: object,
    outcome_model: object,
    config: COTConfig,
    device: str | torch.device,
    seed: int,
    score_grid: Tensor | None = None,
) -> FittedSequentialDR:
    """Fit source-only continuation values by backward dynamic programming.

    Each stage head jointly predicts the score CDF at all later terminal stages
    and all frozen CDF grid points.  Independent evaluation data are required
    for :func:`sequential_dr_score_cdf`.
    """

    if scores.shape != batch.actions.shape:
        raise ValueError("scores must have shape [N, T]")
    resolved = torch.device(device)
    source = batch.to(resolved)
    source_scores = scores.to(resolved)
    frozen_schedule = schedule.to(device=resolved, dtype=source.states.dtype)
    frozen_grid = (make_score_cdf_grid(source_scores) if score_grid is None else score_grid.to(resolved)).to(source.states)
    model = FixedScheduleSequentialDR(
        state_dim=source.state_dim,
        horizon=source.horizon,
        n_actions=int(target_policy.n_actions),
        outcome_model=outcome_model,
        score_grid=frozen_grid,
        config=config,
    ).to(resolved)
    current_states = source.current_states().reshape(-1, source.state_dim)
    model.state_center.copy_(current_states.mean(dim=0))
    model.state_scale.copy_(current_states.std(dim=0).clamp_min(1e-4))

    train_rows, validation_rows = _patient_split(source, seed=seed)
    generator = torch.Generator(device=resolved).manual_seed(seed)
    validation_mse: list[float] = []
    for stage in reversed(range(source.horizon)):
        head = model.heads[stage]
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        for _ in range(config.epochs):
            order = train_rows[torch.randperm(len(train_rows), generator=generator, device=resolved)]
            for rows in order.split(config.batch_size):
                target = _continuation_target(
                    model,
                    source,
                    source_scores,
                    stage=stage,
                    rows=rows,
                    target_policy=target_policy,
                    schedule=frozen_schedule,
                )
                prediction = model.q_values(stage, source.states[rows, stage], source.actions[rows, stage])
                loss = nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
                optimizer.step()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            target = _continuation_target(
                model,
                source,
                source_scores,
                stage=stage,
                rows=validation_rows,
                target_policy=target_policy,
                schedule=frozen_schedule,
            )
            prediction = model.q_values(stage, source.states[validation_rows, stage], source.actions[validation_rows, stage])
            validation_mse.append(float(nn.functional.mse_loss(prediction, target).item()))
    model.eval()
    return FittedSequentialDR(
        model=model,
        schedule=frozen_schedule.detach().cpu(),
        validation_mse=tuple(reversed(validation_mse)),
    )


@torch.no_grad()
def sequential_dr_score_cdf(
    fitted: FittedSequentialDR,
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    target_policy: object,
    logging_policy: object,
) -> Tensor:
    """Estimate ``P_pi(R_t <= r)`` at the fitted source-only CDF grid.

    The returned CDF is clipped to ``[0, 1]`` and monotonically rearranged
    across score-grid points only after applying the ordinary (unnormalized)
    sequential DR estimating equation.
    """

    if scores.shape != batch.actions.shape:
        raise ValueError("scores must have shape [N, T]")
    model = fitted.model
    device = next(model.parameters()).device
    source = batch.to(device)
    source_scores = scores.to(device)
    schedule = fitted.schedule.to(device=device, dtype=source.states.dtype)
    grid = model.score_grid

    estimate = model.values(0, source.states[:, 0], target_policy, schedule).mean(dim=0).to(torch.float64)
    log_weight = torch.zeros(source.n, dtype=torch.float64, device=device)
    for stage in range(source.horizon):
        states = source.states[:, stage]
        actions = source.actions[:, stage]
        target_probability = target_policy.probabilities(states, schedule[stage]).gather(1, actions[:, None]).squeeze(1)
        logging_probability = logging_policy.probabilities(states).gather(1, actions[:, None]).squeeze(1)
        log_weight += target_probability.to(torch.float64).log() - logging_probability.to(torch.float64).log()
        q_value = model.q_values(stage, states, actions).to(torch.float64)
        terminal_reward = source_scores[:, stage, None].le(grid[stage][None, :]).to(torch.float64).unsqueeze(1)
        if stage + 1 < source.horizon:
            continuation = model.values(stage + 1, source.states[:, stage + 1], target_policy, schedule).to(torch.float64)
            target = torch.cat((terminal_reward, continuation), dim=1)
        else:
            target = terminal_reward
        estimate[stage:] += (log_weight.exp()[:, None, None] * (target - q_value)).mean(dim=0)
    return estimate.clamp(0.0, 1.0).cummax(dim=1).values.to(source_scores)


@torch.no_grad()
def dr_quantile(cdf: Tensor, score_grid: Tensor, *, probability: float = 0.90) -> Tensor:
    """Linearly interpolate the first rearranged CDF crossing at ``probability``."""

    if cdf.shape != score_grid.shape:
        raise ValueError("cdf and score_grid must have the same [T, K] shape")
    values: list[Tensor] = []
    for stage in range(cdf.shape[0]):
        index = torch.searchsorted(cdf[stage], cdf.new_tensor(probability), right=False).clamp_max(cdf.shape[1] - 1)
        if int(index) == 0:
            values.append(score_grid[stage, 0])
            continue
        previous = index - 1
        lower_cdf, upper_cdf = cdf[stage, previous], cdf[stage, index]
        lower_score, upper_score = score_grid[stage, previous], score_grid[stage, index]
        fraction = ((probability - lower_cdf) / (upper_cdf - lower_cdf).clamp_min(1e-8)).clamp(0.0, 1.0)
        values.append(lower_score + fraction * (upper_score - lower_score))
    return torch.stack(values)


@torch.no_grad()
def empirical_cdf(scores: Tensor, score_grid: Tensor, *, weights: Tensor | None = None) -> Tensor:
    """Evaluate an empirical (optionally self-normalized weighted) score CDF."""

    if scores.ndim != 2 or score_grid.ndim != 2 or scores.shape[1] != score_grid.shape[0]:
        raise ValueError("scores and score_grid must be [N, T] and [T, K]")
    if weights is None:
        weights = torch.ones_like(scores)
    values = []
    for stage in range(scores.shape[1]):
        indicator = scores[:, stage, None].le(score_grid[stage][None, :]).to(weights)
        values.append((weights[:, stage, None] * indicator).sum(dim=0) / weights[:, stage].sum().clamp_min(1e-12))
    return torch.stack(values)


@torch.no_grad()
def prefix_action_weights(
    batch: TrajectoryBatch,
    *,
    schedule: Tensor,
    target_policy: object,
    logging_policy: object,
) -> Tensor:
    """Return exact prefix action likelihood ratios without self-normalization."""

    source = batch
    log_weight = torch.zeros(source.n, dtype=torch.float64, device=source.states.device)
    weights = []
    for stage in range(source.horizon):
        states, actions = source.states[:, stage], source.actions[:, stage]
        numerator = target_policy.probabilities(states, schedule[stage]).gather(1, actions[:, None]).squeeze(1)
        denominator = logging_policy.probabilities(states).gather(1, actions[:, None]).squeeze(1)
        log_weight += numerator.to(torch.float64).log() - denominator.to(torch.float64).log()
        weights.append(log_weight.exp())
    return torch.stack(weights, dim=1)


@torch.no_grad()
def _continuation_target(
    model: FixedScheduleSequentialDR,
    batch: TrajectoryBatch,
    scores: Tensor,
    *,
    stage: int,
    rows: Tensor,
    target_policy: object,
    schedule: Tensor,
) -> Tensor:
    terminal = scores[rows, stage, None].le(model.score_grid[stage][None, :]).to(scores).unsqueeze(1)
    if stage + 1 == batch.horizon:
        return terminal
    continuation = model.values(stage + 1, batch.states[rows, stage + 1], target_policy, schedule)
    return torch.cat((terminal, continuation), dim=1)


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
