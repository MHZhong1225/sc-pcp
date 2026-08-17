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
class EmpiricalRolloutContext:
    """Per-trajectory donor identity carried through an empirical rollout."""

    donor_patient_ids: Tensor


@dataclass(frozen=True)
class SyntheticNoiseBundle:
    """Exogenous draws shared across synthetic policy candidates."""

    initial_normal: Tensor
    initial_difficulty_uniform: Tensor
    action_uniform: Tensor
    shared_normal: Tensor
    independent_normal: Tensor
    innovation_normal: Tensor
    difficulty_uniform: Tensor
    contamination_uniform: Tensor
    seed: int


def make_synthetic_noise_bundle(
    *,
    n: int,
    horizon: int,
    seed: int,
    device: str | torch.device,
) -> SyntheticNoiseBundle:
    """Draw every synthetic rollout noise source once from one seed."""

    resolved = torch.device(device)
    generator = torch.Generator(device=resolved).manual_seed(seed)
    initial_normal = torch.stack(
        [torch.randn(n, generator=generator, device=resolved) for _ in range(6)],
        dim=1,
    )
    return SyntheticNoiseBundle(
        initial_normal=initial_normal,
        initial_difficulty_uniform=torch.rand(
            n,
            generator=generator,
            device=resolved,
        ),
        action_uniform=torch.rand(
            (horizon, n),
            generator=generator,
            device=resolved,
        ),
        shared_normal=torch.randn(
            (horizon, n),
            generator=generator,
            device=resolved,
        ),
        independent_normal=torch.randn(
            (horizon, n),
            generator=generator,
            device=resolved,
        ),
        innovation_normal=torch.randn(
            (horizon, n, 4),
            generator=generator,
            device=resolved,
        ),
        difficulty_uniform=torch.rand(
            (horizon, n),
            generator=generator,
            device=resolved,
        ),
        contamination_uniform=torch.rand(
            (horizon, n),
            generator=generator,
            device=resolved,
        ),
        seed=seed,
    )


def inverse_cdf_actions(probabilities: Tensor, uniforms: Tensor) -> Tensor:
    """Sample categorical actions from caller-supplied uniforms."""

    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("uniforms must match the policy batch shape")
    actions = (uniforms[..., None] > probabilities.cumsum(dim=-1)).sum(dim=-1)
    return actions.clamp_max(probabilities.shape[-1] - 1)


def _initial_continuous_state(initial_normal: Tensor) -> Tensor:
    return torch.stack(
        (
            1.3 + 0.6 * initial_normal[:, 0],
            0.4 + 0.3 * initial_normal[:, 1],
            initial_normal[:, 2],
            initial_normal[:, 3],
            initial_normal[:, 4],
            initial_normal[:, 5],
        ),
        dim=1,
    )


def _synthetic_transition_components_from_noise(
    *,
    config: SyntheticConfig,
    n_actions: int,
    state: Tensor,
    action: Tensor,
    shared: Tensor,
    independent: Tensor,
    innovations: Tensor,
    residual_multiplier: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    disease, toxicity, z1, z2, z3, z4 = state[:, :6].unbind(dim=1)
    beta = config.feedback_strength
    intensity = action.to(state.dtype) / max(n_actions - 1, 1)
    nonlinear_disease = config.nonlinear_strength * (
        0.20 * torch.sin(disease)
        + 0.12 * disease * z1.tanh()
        + 0.08 * toxicity.square()
    )
    nonlinear_toxicity = config.nonlinear_strength * (
        0.15 * torch.tanh(disease * toxicity) + 0.10 * z2.square() - 0.08 * z3
    )
    disease_mean = (
        config.state_persistence * disease
        + 0.10 * toxicity
        + 0.12 * z1
        + nonlinear_disease
        - beta * config.disease_treatment_effect * intensity
    )
    toxicity_mean = (
        0.82 * toxicity
        + 0.06 * disease
        - 0.08 * z1
        + nonlinear_toxicity
        + beta * config.toxicity_treatment_effect * intensity
    )
    disease_scale = config.disease_noise * (
        1.0 + 0.10 * disease.abs() + 0.22 * z4.abs() + beta * 0.20 * intensity
    )
    toxicity_scale = config.toxicity_noise * (
        1.0 + 0.10 * toxicity.abs() + 0.15 * z3.abs() + beta * 0.25 * intensity
    )
    if residual_multiplier is None:
        disease_noise = disease_scale * shared
        toxicity_noise = toxicity_scale * (0.30 * shared + 0.954 * independent)
    else:
        disease_noise = disease_scale * shared * residual_multiplier
        toxicity_noise = (
            toxicity_scale * (0.30 * shared + 0.954 * independent) * residual_multiplier
        )
        innovations = innovations * residual_multiplier[:, None]
    next_disease = disease_mean + disease_noise
    next_toxicity = toxicity_mean + toxicity_noise
    next_z1 = 0.75 * z1 + 0.25 * innovations[:, 0] + beta * 0.12 * intensity
    next_z2 = 0.70 * z2 + 0.30 * innovations[:, 1] - beta * 0.10 * intensity
    next_z3 = 0.68 * z3 + 0.32 * innovations[:, 2]
    next_z4 = 0.72 * z4 + 0.28 * innovations[:, 3] + beta * 0.10 * intensity
    return next_disease, next_toxicity, next_z1, next_z2, next_z3, next_z4


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

    def initial_state_from_noise(self, bundle: SyntheticNoiseBundle) -> Tensor:
        return _initial_continuous_state(bundle.initial_normal)

    def step(
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        shared = torch.randn(len(state), generator=generator, device=state.device)
        independent = torch.randn(len(state), generator=generator, device=state.device)
        innovations = torch.randn(
            (len(state), 4), generator=generator, device=state.device
        )
        unused_uniform = torch.empty_like(shared)
        return self.step_from_noise(
            state,
            action,
            shared=shared,
            independent=independent,
            innovations=innovations,
            difficulty_uniform=unused_uniform,
            contamination_uniform=unused_uniform,
        )

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        shared: Tensor,
        independent: Tensor,
        innovations: Tensor,
        difficulty_uniform: Tensor,
        contamination_uniform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del difficulty_uniform, contamination_uniform
        components = _synthetic_transition_components_from_noise(
            config=self.config,
            n_actions=self.n_actions,
            state=state,
            action=action,
            shared=shared,
            independent=independent,
            innovations=innovations,
        )
        next_disease, next_toxicity = components[:2]
        next_state = torch.stack(
            components,
            dim=1,
        ).clamp(-self.config.state_clip, self.config.state_clip)
        outcome = torch.stack((next_disease, next_toxicity), dim=1).clamp(
            -self.config.state_clip, self.config.state_clip
        )
        return next_state, outcome


@dataclass(frozen=True)
class TailShiftTreatmentEnvironment:
    """Seven-state treatment environment with observed residual-tail difficulty."""

    config: SyntheticConfig

    state_dim: int = 7
    outcome_dim: int = 2
    n_actions: int = 3

    def initial_state(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        continuous = torch.stack(
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
        difficulty = (
            torch.rand(n, generator=generator, device=device)
            < self.config.difficulty_initial_probability
        ).to(continuous.dtype)
        return torch.cat((continuous, difficulty[:, None]), dim=1)

    def initial_state_from_noise(self, bundle: SyntheticNoiseBundle) -> Tensor:
        continuous = _initial_continuous_state(bundle.initial_normal)
        difficulty = (
            bundle.initial_difficulty_uniform
            < self.config.difficulty_initial_probability
        ).to(continuous.dtype)
        return torch.cat((continuous, difficulty[:, None]), dim=1)

    def difficulty_probability(self, state: Tensor, action: Tensor) -> Tensor:
        intensity = action.to(state.dtype) / (self.n_actions - 1)
        logit = (
            self.config.difficulty_intercept
            + self.config.difficulty_state_effect * state[:, 0]
            + self.config.difficulty_persistence * state[:, 6]
            - self.config.difficulty_treatment_effect * intensity
        )
        return torch.sigmoid(logit)

    def step(
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        contamination_uniform = torch.rand(
            len(state),
            generator=generator,
            device=state.device,
        )
        shared = torch.randn(len(state), generator=generator, device=state.device)
        independent = torch.randn(len(state), generator=generator, device=state.device)
        innovations = torch.randn(
            (len(state), 4), generator=generator, device=state.device
        )
        difficulty_uniform = torch.rand(
            len(state),
            generator=generator,
            device=state.device,
        )
        return self.step_from_noise(
            state,
            action,
            shared=shared,
            independent=independent,
            innovations=innovations,
            difficulty_uniform=difficulty_uniform,
            contamination_uniform=contamination_uniform,
        )

    def step_from_noise(
        self,
        state: Tensor,
        action: Tensor,
        *,
        shared: Tensor,
        independent: Tensor,
        innovations: Tensor,
        difficulty_uniform: Tensor,
        contamination_uniform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        difficulty = state[:, 6]
        contaminated = (
            contamination_uniform < self.config.tail_contamination_probability
        ) & (difficulty == 1.0)
        residual_multiplier = torch.where(
            contaminated,
            torch.full_like(difficulty, self.config.tail_scale),
            torch.ones_like(difficulty),
        )
        components = _synthetic_transition_components_from_noise(
            config=self.config,
            n_actions=self.n_actions,
            state=state,
            action=action,
            shared=shared,
            independent=independent,
            innovations=innovations,
            residual_multiplier=residual_multiplier,
        )
        continuous = torch.stack(
            components,
            dim=1,
        ).clamp(-self.config.state_clip, self.config.state_clip)
        outcome = continuous[:, : self.outcome_dim]
        next_difficulty = (
            difficulty_uniform < self.difficulty_probability(state, action)
        ).to(state.dtype)
        next_state = torch.cat((continuous, next_difficulty[:, None]), dim=1)
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
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int | None = None,
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
        candidate_radii: Tensor,
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
        resolved_radii = candidate_radii.to(resolved)
        if resolved_radii.ndim == 1:
            resolved_radii = resolved_radii[:, None].expand(-1, horizon)
        if resolved_radii.ndim != 2 or resolved_radii.shape[1] != horizon:
            raise ValueError("candidate_radii must have shape [K] or [K,T]")
        rows = []
        for radius_by_time in resolved_radii:
            d_q = [initial]
            for time in range(horizon - 1):
                pi = policy.probabilities(states, radius_by_time[time])
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
    if isinstance(q, Tensor) and q.ndim not in {0, 1}:
        raise ValueError("rollout radius must be scalar or have shape [T]")
    if isinstance(q, Tensor) and q.ndim == 1 and q.shape != (horizon,):
        raise ValueError("stagewise rollout radius must have shape [T]")
    generator = torch.Generator(device=resolved).manual_seed(seed)
    initial_state_with_context = getattr(
        environment,
        "initial_state_with_context",
        None,
    )
    if callable(initial_state_with_context):
        state, rollout_context = initial_state_with_context(n, generator, resolved)
    else:
        state = environment.initial_state(n, generator, resolved)
        rollout_context = None
    states, actions, outcomes = [state], [], []
    for time in range(horizon):
        step_q = (
            q[time]
            if isinstance(q, Tensor) and q.ndim == 1 and len(q) == horizon
            else q
        )
        probabilities = policy.probabilities(state, step_q)
        action = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        if rollout_context is None:
            next_state, outcome = environment.step(
                state,
                action,
                generator,
                time=time,
            )
        else:
            step_with_context = getattr(environment, "step_with_context", None)
            if not callable(step_with_context):
                raise TypeError(
                    "an environment with contextual initial states must implement "
                    "step_with_context"
                )
            next_state, outcome, rollout_context = step_with_context(
                state,
                action,
                generator,
                time=time,
                context=rollout_context,
            )
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
        outcome_model: object | None = None,
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
    ) -> None:
        self.n_actions = n_actions
        self.state_dim = batch.state_dim
        self.outcome_dim = batch.outcome_dim
        self.neighbors = neighbors
        self.bandwidth = bandwidth
        self.static_indices = static_indices
        self.outcome_model = outcome_model
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.query_batch_size = query_batch_size
        if batch.state_dim % history_length:
            raise ValueError(
                "state_dim must be divisible by empirical environment history_length"
            )
        self.history_length = history_length
        self.base_state_dim = batch.state_dim // history_length
        # The stacked history is the predictor's Markov state.  Nearest-neighbor
        # matching and donor successors must therefore use that complete state;
        # matching only the newest frame pairs outcomes with histories on which
        # the frozen GRU was never calibrated and causes rapid rollout drift.
        current = batch.current_states().cpu()
        next_states = batch.states[:, 1:].cpu()
        outcome_payloads = self._outcome_payloads(batch)
        actions = batch.actions.cpu()
        flat_current = current.reshape(-1, batch.state_dim)
        self.center = flat_current.mean(dim=0)
        self.scale = flat_current.std(dim=0).clamp_min(1e-4)
        normalized = ((flat_current - self.center) / self.scale).clamp(-10.0, 10.0)
        # The empirical MDP operates in one frozen, D_env-only 32-dimensional
        # state embedding.  A PCA basis is transparent, deterministic, and
        # avoids fitting a transition/outcome model on data that must remain
        # external to SC-PCP calibration.
        rank = min(embedding_dim, batch.state_dim, max(1, normalized.shape[0]))
        # Clinical history stacks contain many nearly collinear coordinates.
        # LAPACK's single-precision symmetric eigensolver can then either fail
        # to converge or silently return NaNs.  Compute only this small frozen
        # PCA fit in double precision and cast its basis back for rollouts.
        normalized_for_pca = normalized.to(torch.float64)
        covariance = (
            normalized_for_pca.T @ normalized_for_pca / max(1, normalized.shape[0] - 1)
        )
        covariance = (covariance + covariance.T) / 2.0
        _, eigenvectors = torch.linalg.eigh(covariance)
        self.embedding = eigenvectors[:, -rank:].to(normalized.dtype).contiguous()
        embedded_current = (normalized @ self.embedding).reshape(
            batch.n,
            batch.horizon,
            rank,
        )
        self.initial_states = batch.states[:, 0].cpu()
        self.initial_patient_ids = batch.patient_ids.cpu()
        self.horizon = batch.horizon
        self._libraries: dict[
            tuple[int, int],
            tuple[Tensor, Tensor, Tensor, Tensor],
        ] = {}
        self._device_libraries: dict[
            str,
            dict[tuple[int, int], tuple[Tensor, Tensor, Tensor, Tensor]],
        ] = {}
        self._device_transforms: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for time in range(batch.horizon):
            for action in range(n_actions):
                rows = actions[:, time] == action
                if not rows.any():
                    raise ValueError(
                        f"D_env has no transitions for action {action} at stage {time}"
                    )
                self._libraries[(time, action)] = (
                    embedded_current[rows, time],
                    next_states[rows, time],
                    outcome_payloads[rows, time],
                    batch.patient_ids.cpu()[rows],
                )

    def initial_state(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        index = self._initial_indices(n, generator, device)
        return self.initial_states[index.cpu()].to(device)

    def initial_state_with_context(
        self,
        n: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[Tensor, EmpiricalRolloutContext]:
        """Sample starts and remember which patient supplied each state."""

        index = self._initial_indices(n, generator, device)
        patient_ids = self.initial_patient_ids[index.cpu()].to(device)
        return (
            self.initial_states[index.cpu()].to(device),
            EmpiricalRolloutContext(donor_patient_ids=patient_ids),
        )

    def _initial_indices(
        self,
        n: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> Tensor:
        return torch.randint(
            len(self.initial_states),
            (n,),
            generator=generator,
            device=device,
        )

    def step(
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        next_state, outcome, _ = self._step(
            state,
            action,
            generator,
            time=time,
            excluded_patient_ids=None,
        )
        return next_state, outcome

    def step_with_context(
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int,
        context: EmpiricalRolloutContext,
    ) -> tuple[Tensor, Tensor, EmpiricalRolloutContext]:
        """Advance while excluding every transition from the current donor."""

        if context.donor_patient_ids.shape != (len(state),):
            raise ValueError("empirical rollout context must have one donor id per row")
        next_state, outcome, donor_patient_ids = self._step(
            state,
            action,
            generator,
            time=time,
            excluded_patient_ids=context.donor_patient_ids,
        )
        return (
            next_state,
            outcome,
            EmpiricalRolloutContext(donor_patient_ids=donor_patient_ids),
        )

    @torch.no_grad()
    def _step(
        self,
        state: Tensor,
        action: Tensor,
        generator: torch.Generator,
        *,
        time: int | None,
        excluded_patient_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        stage = 0 if time is None else time
        if not 0 <= stage < self.horizon:
            raise ValueError("empirical environment stage is out of range")
        next_states = torch.empty_like(state)
        outcomes = torch.empty(
            (len(state), self.outcome_dim), dtype=state.dtype, device=state.device
        )
        next_donor_patient_ids = torch.empty(
            len(state),
            dtype=self.initial_patient_ids.dtype,
            device=state.device,
        )
        for action_value in range(self.n_actions):
            rows = (action == action_value).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            (
                candidate_embedding,
                library_next,
                library_outcome_payload,
                library_patient_ids,
            ) = self._library_on_device(action_value, state.device, time=stage)
            center, scale, embedding = self._transforms_like(state)
            query = ((state[rows] - center) / scale).clamp(-10.0, 10.0) @ embedding
            excluded = (
                None
                if excluded_patient_ids is None
                else excluded_patient_ids[rows].to(
                    device=state.device,
                    dtype=library_patient_ids.dtype,
                )
            )
            k = self._available_neighbor_count(
                library_patient_ids,
                excluded,
            )
            nearest_distance, nearest = self._nearest_neighbors(
                query,
                candidate_embedding,
                k,
                candidate_patient_ids=library_patient_ids,
                excluded_patient_ids=excluded,
            )
            local_scale = nearest_distance.median(dim=1).values.clamp_min(1e-6)
            standardized_distance = nearest_distance / local_scale[:, None]
            logits = -standardized_distance.square() / (2.0 * self.bandwidth**2)
            draw = torch.multinomial(
                torch.softmax(logits, dim=1), 1, generator=generator
            ).squeeze(1)
            sampled = nearest[torch.arange(len(rows), device=nearest.device), draw]
            next_states[rows] = library_next[sampled]
            sampled_payload = library_outcome_payload[sampled]
            if self.outcome_model is None:
                outcomes[rows] = sampled_payload
            else:
                query_actions = torch.full(
                    (len(rows),),
                    action_value,
                    dtype=action.dtype,
                    device=state.device,
                )
                query_mean, query_scale = self.outcome_model(
                    state[rows],
                    query_actions,
                )
                outcomes[rows] = query_mean + query_scale * sampled_payload
            next_donor_patient_ids[rows] = library_patient_ids[sampled]
        return next_states, outcomes, next_donor_patient_ids

    @torch.no_grad()
    def _outcome_payloads(self, batch: TrajectoryBatch) -> Tensor:
        """Return raw outcomes or signed residuals under the frozen predictor."""

        if self.outcome_model is None:
            return batch.outcomes.cpu()
        model_device = _module_device(self.outcome_model)
        states = batch.current_states().reshape(-1, batch.state_dim)
        actions = batch.actions.reshape(-1)
        outcomes = batch.outcomes.reshape(-1, batch.outcome_dim)
        residual_parts = []
        for start in range(0, len(states), self.query_batch_size):
            stop = start + self.query_batch_size
            mean, scale = self.outcome_model(
                states[start:stop].to(model_device),
                actions[start:stop].to(model_device),
            )
            residual_parts.append(
                (
                    (outcomes[start:stop].to(model_device) - mean)
                    / scale.clamp_min(1e-12)
                ).cpu()
            )
        return torch.cat(residual_parts).reshape(
            batch.n,
            batch.horizon,
            batch.outcome_dim,
        )

    def _available_neighbor_count(
        self,
        candidate_patient_ids: Tensor,
        excluded_patient_ids: Tensor | None,
    ) -> int:
        if excluded_patient_ids is None:
            return min(self.neighbors, len(candidate_patient_ids))
        unique_ids, counts = torch.unique(
            candidate_patient_ids,
            sorted=True,
            return_counts=True,
        )
        positions = torch.searchsorted(unique_ids, excluded_patient_ids)
        clamped = positions.clamp_max(len(unique_ids) - 1)
        excluded_counts = torch.where(
            unique_ids[clamped] == excluded_patient_ids,
            counts[clamped],
            torch.zeros_like(clamped),
        )
        available = len(candidate_patient_ids) - excluded_counts
        minimum_available = int(available.min().item())
        if minimum_available < 1:
            raise RuntimeError(
                "no leave-one-patient-out donor remains for this stage/action"
            )
        return min(self.neighbors, minimum_available)

    def _nearest_neighbors(
        self,
        query: Tensor,
        candidate_embedding: Tensor,
        k: int,
        *,
        candidate_patient_ids: Tensor | None = None,
        excluded_patient_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return exact top-k neighbors without materializing the full distance matrix."""

        if excluded_patient_ids is not None and candidate_patient_ids is None:
            raise ValueError(
                "candidate patient ids are required when excluding rollout donors"
            )

        distance_parts = []
        index_parts = []
        for start in range(0, len(query), self.query_batch_size):
            stop = start + self.query_batch_size
            distances = torch.cdist(
                query[start:stop],
                candidate_embedding,
            )
            if excluded_patient_ids is not None and candidate_patient_ids is not None:
                ineligible = (
                    excluded_patient_ids[start:stop, None]
                    == candidate_patient_ids[None, :]
                )
                distances.masked_fill_(ineligible, float("inf"))
            nearest_distance, nearest = distances.topk(k, largest=False)
            del distances
            distance_parts.append(nearest_distance)
            index_parts.append(nearest)
        return torch.cat(distance_parts, dim=0), torch.cat(index_parts, dim=0)

    def _library_on_device(
        self,
        action: int,
        device: torch.device,
        *,
        time: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        key = str(device)
        if key not in self._device_libraries:
            self._device_libraries[key] = {}
        cache = self._device_libraries[key]
        key_by_stage = (time, action)
        if key_by_stage not in cache:
            cache[key_by_stage] = tuple(
                value.to(device) for value in self._libraries[key_by_stage]
            )
        return cache[key_by_stage]

    def _transforms_like(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = f"{state.device}:{state.dtype}"
        if key not in self._device_transforms:
            self._device_transforms[key] = (
                self.center.to(state),
                self.scale.to(state),
                self.embedding.to(state),
            )
        return self._device_transforms[key]


def _module_device(module: object) -> torch.device:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    buffers = getattr(module, "buffers", None)
    if callable(buffers):
        try:
            return next(buffers()).device
        except StopIteration:
            pass
    return torch.device("cpu")
