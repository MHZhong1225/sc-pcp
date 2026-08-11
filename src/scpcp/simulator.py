"""Synthetic and empirical environments used by the SC-PCP experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from scpcp.config import SyntheticConfig
from scpcp.data import TrajectoryBatch


class StochasticPolicy(Protocol):
    n_actions: int

    def probabilities(
        self, states: Tensor, q: float | Tensor | None = None
    ) -> Tensor: ...


@dataclass(frozen=True)
class SyntheticBehaviorPolicy:
    """Known logging policy for synthetic experiments with uniform positivity."""

    n_actions: int = 3
    exploration: float = 0.12

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        disease, toxicity, modifier = states[:, 0], states[:, 1], states[:, 2]
        logits = torch.stack(
            (
                0.25 * toxicity - 0.15 * disease,
                0.35 * disease - 0.20 * toxicity + 0.10 * modifier,
                0.65 * disease - 0.45 * toxicity - 0.15 * modifier,
            ),
            dim=1,
        )
        softmax = torch.softmax(logits, dim=1)
        return (1.0 - self.exploration) * softmax + self.exploration / self.n_actions


@dataclass(frozen=True)
class SyntheticTreatmentEnvironment:
    """Six-state nonlinear treatment environment with two outcomes.

    All action-dependent transition and outcome-noise terms are multiplied by
    ``feedback_strength``.  Thus beta=0 removes policy-induced changes in the
    score law, which is the relevant reduction for per-step SC-PCP.
    """

    config: SyntheticConfig

    state_dim: int = 6
    outcome_dim: int = 2
    n_actions: int = 3

    def initial_state(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        return torch.stack(
            (
                1.3 + 0.6 * torch.randn(n, generator=generator, device=device),
                0.4 + 0.3 * torch.randn(n, generator=generator, device=device),
                torch.randn(n, generator=generator, device=device),
                torch.randn(n, generator=generator, device=device),
                torch.randn(n, generator=generator, device=device),
                torch.randn(n, generator=generator, device=device),
            ),
            dim=1,
        )

    def step(
        self, state: Tensor, action: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        disease, toxicity, z1, z2, z3, z4 = state.unbind(dim=1)
        beta = self.config.feedback_strength
        intensity = action.to(state.dtype) / max(self.n_actions - 1, 1)
        nonlinear_disease = self.config.nonlinear_strength * (
            0.20 * torch.sin(disease)
            + 0.12 * disease * z1.tanh()
            + 0.08 * toxicity.square()
        )
        nonlinear_toxicity = self.config.nonlinear_strength * (
            0.15 * torch.tanh(disease * toxicity) + 0.10 * z2.square() - 0.08 * z3
        )
        disease_mean = (
            self.config.state_persistence * disease
            + 0.10 * toxicity
            + 0.12 * z1
            + nonlinear_disease
            - beta * self.config.disease_treatment_effect * intensity
        )
        toxicity_mean = (
            0.82 * toxicity
            + 0.06 * disease
            - 0.08 * z1
            + nonlinear_toxicity
            + beta * self.config.toxicity_treatment_effect * intensity
        )
        disease_scale = self.config.disease_noise * (
            1.0 + 0.10 * disease.abs() + 0.22 * z4.abs() + beta * 0.20 * intensity
        )
        toxicity_scale = self.config.toxicity_noise * (
            1.0 + 0.10 * toxicity.abs() + 0.15 * z3.abs() + beta * 0.25 * intensity
        )
        shared = torch.randn(len(state), generator=generator, device=state.device)
        independent = torch.randn(len(state), generator=generator, device=state.device)
        disease_noise = disease_scale * shared
        toxicity_noise = toxicity_scale * (0.30 * shared + 0.954 * independent)
        next_disease = disease_mean + disease_noise
        next_toxicity = toxicity_mean + toxicity_noise
        innovations = torch.randn(
            (len(state), 4), generator=generator, device=state.device
        )
        next_z1 = 0.75 * z1 + 0.25 * innovations[:, 0] + beta * 0.12 * intensity
        next_z2 = 0.70 * z2 + 0.30 * innovations[:, 1] - beta * 0.10 * intensity
        next_z3 = 0.68 * z3 + 0.32 * innovations[:, 2]
        next_z4 = 0.72 * z4 + 0.28 * innovations[:, 3] + beta * 0.10 * intensity
        next_state = torch.stack(
            (next_disease, next_toxicity, next_z1, next_z2, next_z3, next_z4), dim=1
        ).clamp(-self.config.state_clip, self.config.state_clip)
        outcome = torch.stack((next_disease, next_toxicity), dim=1).clamp(
            -self.config.state_clip, self.config.state_clip
        )
        return next_state, outcome


@dataclass(frozen=True)
class TabularBehaviorPolicy:
    """Known positive logging policy for exact occupancy-transport tests."""

    n_states: int = 5
    n_actions: int = 3
    exploration: float = 0.15

    def probabilities(self, states: Tensor, q: float | Tensor | None = None) -> Tensor:
        state = states.argmax(dim=1)
        table = torch.tensor(
            [
                [0.55, 0.32, 0.13],
                [0.42, 0.38, 0.20],
                [0.27, 0.43, 0.30],
                [0.18, 0.40, 0.42],
                [0.12, 0.33, 0.55],
            ],
            dtype=states.dtype,
            device=states.device,
        )[: self.n_states, : self.n_actions]
        selected = table[state]
        return (1.0 - self.exploration) * selected + self.exploration / self.n_actions


@dataclass(frozen=True)
class TabularTreatmentEnvironment:
    """Small finite MDP with exact occupancy ratios for theorem validation."""

    config: SyntheticConfig
    n_states: int = 5
    n_actions: int = 3
    outcome_dim: int = 2

    @property
    def state_dim(self) -> int:
        return self.n_states

    def initial_state(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        index = torch.multinomial(
            torch.full((self.n_states,), 1.0 / self.n_states, device=device),
            n,
            replacement=True,
            generator=generator,
        )
        return torch.nn.functional.one_hot(index, self.n_states).to(torch.float32)

    def transition_probabilities(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        base = torch.tensor(
            [
                [0.62, 0.24, 0.10, 0.03, 0.01],
                [0.18, 0.48, 0.23, 0.08, 0.03],
                [0.07, 0.20, 0.44, 0.20, 0.09],
                [0.03, 0.08, 0.25, 0.43, 0.21],
                [0.01, 0.04, 0.12, 0.29, 0.54],
            ],
            dtype=dtype,
            device=device,
        )
        transition = base[None].repeat(self.n_actions, 1, 1)
        beta = self.config.feedback_strength
        protective = torch.eye(self.n_states, device=device, dtype=dtype).roll(
            -1, dims=1
        )
        harmful = torch.eye(self.n_states, device=device, dtype=dtype).roll(1, dims=1)
        transition[1] = (1.0 - 0.25 * beta) * transition[1] + 0.25 * beta * protective
        transition[2] = (1.0 - 0.45 * beta) * transition[2] + 0.45 * beta * protective
        transition[0] = (1.0 - 0.10 * beta) * transition[0] + 0.10 * beta * harmful
        return transition / transition.sum(dim=2, keepdim=True)

    def step(
        self, state: Tensor, action: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        state_index = state.argmax(dim=1)
        transitions = self.transition_probabilities(state.device, state.dtype)[
            action, state_index
        ]
        next_index = torch.multinomial(transitions, 1, generator=generator).squeeze(1)
        next_state = torch.nn.functional.one_hot(next_index, self.n_states).to(
            state.dtype
        )
        severity = next_index.to(state.dtype) / (self.n_states - 1)
        intensity = action.to(state.dtype) / (self.n_actions - 1)
        beta = self.config.feedback_strength
        noise = 0.08 * torch.randn(
            (len(state), 2), generator=generator, device=state.device
        )
        outcome = (
            torch.stack(
                (
                    severity - 0.15 * beta * intensity,
                    0.20 * severity + 0.18 * beta * intensity,
                ),
                dim=1,
            )
            + noise
        )
        return next_state, outcome

    @torch.no_grad()
    def exact_state_ratios(
        self,
        policy: StochasticPolicy,
        logging_policy: StochasticPolicy,
        q_grid: Tensor,
        horizon: int,
        device: str | torch.device,
    ) -> Tensor:
        resolved = torch.device(device)
        states = torch.eye(self.n_states, device=resolved)
        transition = self.transition_probabilities(resolved, torch.float32)
        initial = torch.full((self.n_states,), 1.0 / self.n_states, device=resolved)
        mu = logging_policy.probabilities(states)
        d_mu = [initial]
        for _ in range(horizon - 1):
            d_mu.append(torch.einsum("s,sa,asr->r", d_mu[-1], mu, transition))
        rows = []
        for radius in q_grid.to(resolved):
            d_q = [initial]
            for _ in range(horizon - 1):
                pi = policy.probabilities(states, radius)
                d_q.append(torch.einsum("s,sa,asr->r", d_q[-1], pi, transition))
            rows.append(
                torch.stack(
                    [
                        target / source.clamp_min(1e-12)
                        for target, source in zip(d_q, d_mu)
                    ]
                )
            )
        return torch.stack(rows)


@torch.no_grad()
def rollout(
    environment: object,
    policy: StochasticPolicy,
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
    q: float | Tensor | None = None,
    patient_offset: int = 0,
) -> TrajectoryBatch:
    """Generate a batch from a fixed policy or one radius-induced policy."""

    resolved = torch.device(device)
    generator = torch.Generator(device=resolved).manual_seed(seed)
    state = environment.initial_state(n, generator, resolved)
    states, actions, outcomes = [state], [], []
    for time in range(horizon):
        step_q = (
            q[time]
            if isinstance(q, Tensor) and q.ndim == 1 and len(q) == horizon
            else q
        )
        probabilities = policy.probabilities(state, step_q)
        action = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        next_state, outcome = environment.step(state, action, generator)
        states.append(next_state)
        actions.append(action)
        outcomes.append(outcome)
        state = next_state
    return TrajectoryBatch(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
        outcomes=torch.stack(outcomes, dim=1),
        patient_ids=torch.arange(patient_offset, patient_offset + n, device=resolved),
    )


class EmpiricalTransitionEnvironment:
    """A frozen kNN empirical MDP built solely from the held-out D_env split."""

    DEFAULT_QUERY_BATCH_SIZE = 2_048

    def __init__(
        self,
        batch: TrajectoryBatch,
        *,
        n_actions: int,
        neighbors: int,
        bandwidth: float,
        embedding_dim: int = 32,
        static_indices: tuple[int, ...] = (),
        history_length: int = 1,
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
    ) -> None:
        self.n_actions = n_actions
        self.state_dim = batch.state_dim
        self.outcome_dim = batch.outcome_dim
        self.neighbors = neighbors
        self.bandwidth = bandwidth
        self.static_indices = static_indices
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.query_batch_size = query_batch_size
        if batch.state_dim % history_length:
            raise ValueError(
                "state_dim must be divisible by empirical environment history_length"
            )
        self.history_length = history_length
        self.base_state_dim = batch.state_dim // history_length
        self.base_static_indices = tuple(
            sorted({index % self.base_state_dim for index in static_indices})
        )
        # For GRU inputs, only the newest base state is a transition-library
        # key.  A rollout shifts its own history and appends the sampled base
        # successor below, rather than splicing a donor patient's old history
        # into the generated episode.
        current = (
            batch.current_states()
            .reshape(-1, batch.state_dim)
            .cpu()[:, -self.base_state_dim :]
        )
        next_states = (
            batch.states[:, 1:]
            .reshape(-1, batch.state_dim)
            .cpu()[:, -self.base_state_dim :]
        )
        outcomes = batch.outcomes.reshape(-1, batch.outcome_dim).cpu()
        actions = batch.actions.reshape(-1).cpu()
        self.center = current.mean(dim=0)
        self.scale = current.std(dim=0).clamp_min(1e-4)
        normalized = ((current - self.center) / self.scale).clamp(-10.0, 10.0)
        # The empirical MDP operates in one frozen, D_env-only 32-dimensional
        # state embedding.  A PCA basis is transparent, deterministic, and
        # avoids fitting a transition/outcome model on data that must remain
        # external to SC-PCP calibration.
        rank = min(embedding_dim, batch.state_dim, max(1, normalized.shape[0]))
        covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        self.embedding = eigenvectors[:, -rank:].contiguous()
        embedded_current = normalized @ self.embedding
        self.initial_states = batch.states[:, 0].cpu()
        self._libraries: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        self._device_libraries: dict[str, dict[int, tuple[Tensor, Tensor, Tensor]]] = {}
        self._device_transforms: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for action in range(n_actions):
            rows = actions == action
            if not rows.any():
                raise ValueError(f"D_env has no transitions for action {action}")
            self._libraries[action] = (
                embedded_current[rows],
                next_states[rows],
                outcomes[rows],
            )

    def initial_state(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        index = torch.randint(
            len(self.initial_states), (n,), generator=generator, device=device
        )
        return self.initial_states[index.cpu()].to(device)

    def step(
        self, state: Tensor, action: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        next_states = torch.empty_like(state)
        outcomes = torch.empty(
            (len(state), self.outcome_dim), dtype=state.dtype, device=state.device
        )
        for action_value in range(self.n_actions):
            rows = (action == action_value).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            candidate_embedding, library_next, library_outcome = (
                self._library_on_device(action_value, state.device)
            )
            latest_state = state[rows][:, -self.base_state_dim :]
            center, scale, embedding = self._transforms_like(state)
            query = ((latest_state - center) / scale).clamp(-10.0, 10.0) @ embedding
            k = min(self.neighbors, candidate_embedding.shape[0])
            nearest_distance, nearest = self._nearest_neighbors(
                query,
                candidate_embedding,
                k,
            )
            logits = -nearest_distance.square() / (2.0 * self.bandwidth**2)
            draw = torch.multinomial(
                torch.softmax(logits, dim=1), 1, generator=generator
            ).squeeze(1)
            sampled = nearest[torch.arange(len(rows), device=nearest.device), draw]
            sampled_next = library_next[sampled].clone()
            if self.base_static_indices:
                sampled_next[:, self.base_static_indices] = latest_state[
                    :, self.base_static_indices
                ]
            if self.history_length == 1:
                next_states[rows] = sampled_next
            else:
                history = state[rows].reshape(
                    len(rows), self.history_length, self.base_state_dim
                )
                advanced = torch.cat((history[:, 1:], sampled_next[:, None, :]), dim=1)
                next_states[rows] = advanced.reshape(len(rows), self.state_dim)
            outcomes[rows] = library_outcome[sampled]
        return next_states, outcomes

    def _nearest_neighbors(
        self,
        query: Tensor,
        candidate_embedding: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        """Return exact top-k neighbors without materializing the full distance matrix."""

        distance_parts = []
        index_parts = []
        for start in range(0, len(query), self.query_batch_size):
            distances = torch.cdist(
                query[start : start + self.query_batch_size],
                candidate_embedding,
            )
            nearest_distance, nearest = distances.topk(k, largest=False)
            del distances
            distance_parts.append(nearest_distance)
            index_parts.append(nearest)
        return torch.cat(distance_parts, dim=0), torch.cat(index_parts, dim=0)

    def _library_on_device(
        self, action: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = str(device)
        if key not in self._device_libraries:
            self._device_libraries[key] = {}
        cache = self._device_libraries[key]
        if action not in cache:
            cache[action] = tuple(value.to(device) for value in self._libraries[action])
        return cache[action]

    def _transforms_like(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = f"{state.device}:{state.dtype}"
        if key not in self._device_transforms:
            self._device_transforms[key] = (
                self.center.to(state),
                self.scale.to(state),
                self.embedding.to(state),
            )
        return self._device_transforms[key]
