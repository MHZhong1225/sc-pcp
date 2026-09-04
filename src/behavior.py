"""Calibrated propensity model for the historical treatment policy."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from config import ModelConfig, PolicyConfig
from data import TrajectoryBatch


class BehaviorPolicy(nn.Module):
    r"""A propensity model used both as \(\mu_{\rm ref}\) and OPE denominator.

    Clinical deployments substitute the unknown logging propensity with this
    calibrated model.  Synthetic runs can additionally report oracle estimates
    using the known data-generating policy.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        model: ModelConfig,
        policy: PolicyConfig,
        static_indices: tuple[int, ...] = (),
        *,
        horizon: int = 1,
        decision_time_index: int | None = None,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.propensity_floor = policy.propensity_floor
        self.model_config = model
        self.horizon = horizon
        self.decision_time_index = decision_time_index
        if model.architecture == "gru":
            if state_dim % model.history_length:
                raise ValueError("GRU behavior state dimension must be divisible by history_length")
            self.base_state_dim = state_dim // model.history_length
            self.base_static_indices = tuple(sorted({index % self.base_state_dim for index in static_indices}))
            self.state_encoder = nn.GRU(
                input_size=self.base_state_dim,
                hidden_size=model.hidden_dim,
                num_layers=2,
                batch_first=True,
            )
            network_input = model.hidden_dim + len(self.base_static_indices)
        else:
            self.base_state_dim = state_dim
            self.base_static_indices = ()
            self.state_encoder = nn.Identity()
            network_input = state_dim
        self.network = nn.Sequential(
            nn.Linear(network_input, model.hidden_dim),
            nn.ReLU(),
            nn.Linear(model.hidden_dim, model.hidden_dim),
            nn.ReLU(),
            nn.Linear(model.hidden_dim, n_actions),
        )
        self.stage_bias = nn.Parameter(torch.zeros(horizon, n_actions))
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("temperature", torch.ones(()))

    def logits(self, states: Tensor) -> Tensor:
        normalized = ((states - self.state_center) / self.state_scale).clamp(-10.0, 10.0)
        if self.model_config.architecture != "gru":
            features = normalized
        else:
            sequence = normalized.reshape(-1, self.model_config.history_length, self.base_state_dim).clone()
            if self.base_static_indices:
                static = sequence[:, -1, self.base_static_indices]
                sequence[..., self.base_static_indices] = 0.0
            else:
                static = normalized.new_empty((len(normalized), 0))
            _, hidden = self.state_encoder(sequence)
            features = torch.cat((hidden[-1], static), dim=1)
        logits = self.network(features)
        if self.decision_time_index is not None:
            logits = logits + self.stage_bias[self._stage_indices(states)]
        return logits / self.temperature

    def _stage_indices(self, states: Tensor) -> Tensor:
        if self.decision_time_index is None:
            return torch.zeros(len(states), dtype=torch.long, device=states.device)
        if self.model_config.architecture == "gru":
            decision_time = states.reshape(
                -1,
                self.model_config.history_length,
                self.base_state_dim,
            )[:, -1, self.decision_time_index]
        else:
            decision_time = states[:, self.decision_time_index]
        return (decision_time * self.horizon).round().long().clamp(0, self.horizon - 1)

    @torch.no_grad()
    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        fitted = torch.softmax(self.logits(states), dim=1)
        return (1.0 - self.propensity_floor) * fitted + self.propensity_floor / self.n_actions


def fit_behavior_policy(
    batch: TrajectoryBatch,
    *,
    n_actions: int,
    model_config: ModelConfig,
    policy_config: PolicyConfig,
    device: str | torch.device,
    seed: int,
    static_indices: tuple[int, ...] = (),
    decision_time_index: int | None = None,
) -> BehaviorPolicy:
    """Fit the clinical propensity nuisance and calibrate it by decision stage."""

    resolved = torch.device(device)
    states, actions, _ = batch.flat_transitions()
    states, actions = states.to(resolved), actions.to(resolved)
    model = BehaviorPolicy(
        batch.state_dim,
        n_actions,
        model_config,
        policy_config,
        static_indices=static_indices,
        horizon=batch.horizon,
        decision_time_index=decision_time_index,
    ).to(resolved)
    model.state_center.copy_(states.mean(dim=0))
    model.state_scale.copy_(states.std(dim=0).clamp_min(1e-4))
    generator = torch.Generator().manual_seed(seed)
    patient_order = torch.randperm(batch.n, generator=generator)
    split = max(1, int(0.85 * batch.n))
    train_patients, valid_patients = patient_order[:split], patient_order[split:]
    if len(valid_patients) == 0:
        valid_patients = train_patients[:1]
    train_rows = _patient_rows(train_patients, batch.horizon).to(resolved)
    valid_rows = _patient_rows(valid_patients, batch.horizon).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay
    )
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(model_config.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), device=resolved)]
        for rows in order.split(model_config.batch_size):
            loss = nn.functional.cross_entropy(model.logits(states[rows]), actions[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model_config.gradient_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(nn.functional.cross_entropy(model.logits(states[valid_rows]), actions[valid_rows]).item())
        if valid_loss < best_loss:
            best_loss, stale, best_state = valid_loss, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= model_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _fit_propensity_calibration(model, states[valid_rows], actions[valid_rows])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _fit_propensity_calibration(
    model: BehaviorPolicy,
    states: Tensor,
    actions: Tensor,
) -> None:
    """Calibrate global sharpness and stage-specific treatment prevalence."""

    log_temperature = torch.zeros((), device=states.device, requires_grad=True)
    stage_correction = torch.zeros_like(model.stage_bias, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, stage_correction],
        max_iter=50,
        line_search_fn="strong_wolfe",
    )
    base_logits = model.logits(states).detach() * model.temperature
    stage = model._stage_indices(states)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        centered_correction = stage_correction - stage_correction.mean(dim=1, keepdim=True)
        calibrated = (base_logits + centered_correction[stage]) / temperature
        loss = nn.functional.cross_entropy(calibrated, actions)
        loss = loss + 1e-4 * centered_correction.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    correction = stage_correction.detach()
    correction = correction - correction.mean(dim=1, keepdim=True)
    with torch.no_grad():
        model.stage_bias.add_(correction)
        model.temperature.copy_(log_temperature.detach().exp().clamp(0.05, 20.0))


def _patient_rows(patient_indices: Tensor, horizon: int) -> Tensor:
    return (patient_indices[:, None] * horizon + torch.arange(horizon)).reshape(-1)
