"""Frozen heteroscedastic outcome predictor used by every method."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from scpcp.config import ModelConfig
from scpcp.data import TrajectoryBatch


_ALL_ACTION_INFERENCE_CHUNK_SIZE = 4_096


class GaussianOutcomeModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        config: ModelConfig,
        static_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config
        self.static_indices = static_indices
        if config.architecture == "gru":
            if state_dim % config.history_length:
                raise ValueError("GRU state dimension must be divisible by history_length")
            self.base_state_dim = state_dim // config.history_length
            self.base_static_indices = tuple(sorted({index % self.base_state_dim for index in static_indices}))
            self.state_encoder = nn.GRU(
                input_size=self.base_state_dim,
                hidden_size=config.hidden_dim,
                num_layers=2,
                batch_first=True,
            )
            self.representation_layer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.representation_dim), nn.ReLU()
            )
        else:
            self.base_state_dim = state_dim
            self.base_static_indices = ()
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.representation_dim),
                nn.ReLU(),
            )
            self.representation_layer = nn.Identity()
        self.action_embedding = nn.Embedding(n_actions, config.representation_dim // 2)
        head_input = config.representation_dim + len(self.base_static_indices) + config.representation_dim // 2
        self.head = nn.Sequential(
            nn.Linear(head_input, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4),
        )
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))

    def representation(self, states: Tensor) -> Tensor:
        normalized = self._normalized_states(states)
        if self.config.architecture == "gru":
            sequence = normalized.reshape(-1, self.config.history_length, self.base_state_dim).clone()
            if self.base_static_indices:
                sequence[..., self.base_static_indices] = 0.0
            _, hidden = self.state_encoder(sequence)
            return self.representation_layer(hidden[-1])
        return self.state_encoder(normalized)

    def static_features(self, states: Tensor) -> Tensor:
        if self.config.architecture != "gru" or not self.base_static_indices:
            return states.new_empty((len(states), 0))
        normalized = self._normalized_states(states)
        latest = normalized.reshape(-1, self.config.history_length, self.base_state_dim)[:, -1]
        return latest[:, self.base_static_indices]

    def _normalized_states(self, states: Tensor) -> Tensor:
        # Rare/unseen clinical categories can have near-zero D_pred variance.
        # Winsorizing standardized inputs prevents one such coordinate from
        # dominating a frozen predictor while preserving its direction.
        return ((states - self.state_center) / self.state_scale).clamp(-10.0, 10.0)

    def forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.representation(states)
        features = torch.cat((encoded, self.static_features(states), self.action_embedding(actions)), dim=1)
        output = self.head(features)
        mean, log_scale = output[:, :2], output[:, 2:]
        scale = functional.softplus(log_scale) + self.config.min_scale
        return mean, scale

    @torch.no_grad()
    def predict_all_actions(self, states: Tensor) -> tuple[Tensor, Tensor]:
        n = len(states)
        if n == 0:
            shape = (0, self.n_actions, 2)
            return states.new_empty(shape), states.new_empty(shape)
        # Preserve the historical operation order for calibration-sized calls;
        # only large batches need to avoid repeating state encodings by action.
        if n <= _ALL_ACTION_INFERENCE_CHUNK_SIZE:
            repeated_states = (
                states[:, None, :]
                .expand(n, self.n_actions, self.state_dim)
                .reshape(-1, self.state_dim)
            )
            repeated_actions = torch.arange(
                self.n_actions, device=states.device
            ).repeat(n)
            mean, scale = self(repeated_states, repeated_actions)
            return (
                mean.reshape(n, self.n_actions, 2),
                scale.reshape(n, self.n_actions, 2),
            )

        mean = states.new_empty((n, self.n_actions, 2))
        scale = states.new_empty((n, self.n_actions, 2))
        actions = torch.arange(self.n_actions, device=states.device)
        action_features = self.action_embedding(actions)
        for start in range(0, n, _ALL_ACTION_INFERENCE_CHUNK_SIZE):
            stop = min(start + _ALL_ACTION_INFERENCE_CHUNK_SIZE, n)
            state_batch = states[start:stop]
            batch_size = len(state_batch)
            encoded = self.representation(state_batch)
            static = self.static_features(state_batch)
            features = torch.cat(
                (
                    encoded[:, None, :].expand(-1, self.n_actions, -1),
                    static[:, None, :].expand(-1, self.n_actions, -1),
                    action_features[None, :, :].expand(batch_size, -1, -1),
                ),
                dim=2,
            ).reshape(batch_size * self.n_actions, -1)
            output = self.head(features)
            batch_mean, log_scale = output[:, :2], output[:, 2:]
            batch_scale = functional.softplus(log_scale) + self.config.min_scale
            mean[start:stop] = batch_mean.reshape(batch_size, self.n_actions, 2)
            scale[start:stop] = batch_scale.reshape(batch_size, self.n_actions, 2)
        return mean, scale


def fit_outcome_model(
    batch: TrajectoryBatch,
    *,
    n_actions: int,
    config: ModelConfig,
    device: str | torch.device,
    seed: int,
    static_indices: tuple[int, ...] = (),
) -> GaussianOutcomeModel:
    """Fit a Gaussian NLL predictor on D_pred and return it in frozen eval mode."""

    resolved = torch.device(device)
    states, actions, outcomes = batch.flat_transitions()
    model = GaussianOutcomeModel(batch.state_dim, n_actions, config, static_indices=static_indices).to(resolved)
    model.state_center.copy_(states.mean(dim=0).to(resolved))
    model.state_scale.copy_(states.std(dim=0).clamp_min(1e-4).to(resolved))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(batch.n, generator=generator)
    split = max(1, int(0.85 * batch.n))
    train_patients, valid_patients = indices[:split], indices[split:]
    if len(valid_patients) == 0:
        valid_patients = train_patients[:1]
    train_rows = _patient_rows(train_patients, batch.horizon)
    valid_rows = _patient_rows(valid_patients, batch.horizon)
    states, actions, outcomes = states.to(resolved), actions.to(resolved), outcomes.to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(config.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), device=train_rows.device)]
        for rows in order.split(config.batch_size):
            mean, scale = model(states[rows], actions[rows])
            loss = _gaussian_nll(mean, scale, outcomes[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, scale = model(states[valid_rows], actions[valid_rows])
            valid_loss = float(_gaussian_nll(mean, scale, outcomes[valid_rows]).item())
        if valid_loss < best_loss:
            best_loss, stale, best_state = valid_loss, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _patient_rows(patient_indices: Tensor, horizon: int) -> Tensor:
    offsets = patient_indices[:, None] * horizon + torch.arange(horizon)
    return offsets.reshape(-1)


def _gaussian_nll(mean: Tensor, scale: Tensor, outcomes: Tensor) -> Tensor:
    normalized = (outcomes - mean) / scale
    return 0.5 * (normalized.square() + 2.0 * scale.log()).mean()
