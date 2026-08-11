"""Calibrated propensity model for the historical treatment policy."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from scpcp.config import ModelConfig, PolicyConfig
from scpcp.data import TrajectoryBatch


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
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.propensity_floor = policy.propensity_floor
        self.model_config = model
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
        return self.network(features) / self.temperature

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
) -> BehaviorPolicy:
    """Fit propensity on D_beh without class reweighting, then calibrate temperature."""

    resolved = torch.device(device)
    states, actions, _ = batch.flat_transitions()
    states, actions = states.to(resolved), actions.to(resolved)
    model = BehaviorPolicy(
        batch.state_dim,
        n_actions,
        model_config,
        policy_config,
        static_indices=static_indices,
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
    _fit_temperature(model, states[valid_rows], actions[valid_rows])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _fit_temperature(model: BehaviorPolicy, states: Tensor, actions: Tensor) -> None:
    log_temperature = torch.zeros((), device=states.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=30, line_search_fn="strong_wolfe")
    base_logits = model.logits(states).detach() * model.temperature

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(base_logits / temperature, actions)
        loss.backward()
        return loss

    optimizer.step(closure)
    model.temperature.copy_(log_temperature.detach().exp().clamp(0.05, 20.0))


def _patient_rows(patient_indices: Tensor, horizon: int) -> Tensor:
    return (patient_indices[:, None] * horizon + torch.arange(horizon)).reshape(-1)
