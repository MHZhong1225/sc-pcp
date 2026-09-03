"""Path-coherent residual dynamics for controlled feedback benchmarks.

The production empirical environment is intentionally not reused here: it
resamples a complete donor successor state.  This module instead predicts one
base-state frame and adds an observed D_env innovation, while retaining the
simulated patient's static features and rolling history.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scpcp.data import TrajectoryBatch
from scpcp.simulator import inverse_cdf_actions


@dataclass(frozen=True)
class ControlledNoiseBundle:
    initial_indices: Tensor
    action_uniforms: Tensor
    donor_uniforms: Tensor


@dataclass(frozen=True)
class ControlledRollout:
    trajectories: TrajectoryBatch
    donor_difficulty: Tensor
    donor_kernel_ess: Tensor
    donor_probability_max: Tensor


@dataclass(frozen=True)
class _StageModel:
    coefficients: Tensor


class ControlledResidualEnvironment:
    """A same-kernel residual bootstrap environment built from D_env only."""

    def __init__(
        self,
        batch: TrajectoryBatch,
        *,
        outcome_model: object,
        n_actions: int,
        difficulty: Tensor,
        history_length: int,
        static_indices: tuple[int, ...] = (),
        state_feature_names: tuple[str, ...] = (),
        neighbors: int = 100,
        bandwidth: float = 2.0,
        ridge: float = 1e-3,
        representation_geometry: str = "raw",
        donor_weighting: str = "gaussian",
        ridge_mode: str = "v2_raw",
        transition_mode: str = "ridge_residual",
        outcome_residual_mode: str = "standardized",
    ) -> None:
        if batch.state_dim % history_length:
            raise ValueError("state_dim must be divisible by history_length")
        if difficulty.shape != batch.actions.shape:
            raise ValueError("difficulty must have shape [N, T]")
        if representation_geometry not in {"raw", "stagewise_zscore"}:
            raise ValueError("unknown controlled representation geometry")
        if donor_weighting not in {"gaussian", "uniform"}:
            raise ValueError("unknown controlled donor weighting")
        if ridge_mode not in {"v2_raw", "sample_normalized_no_intercept"}:
            raise ValueError("unknown controlled ridge mode")
        if transition_mode not in {"ridge_residual", "local_delta"}:
            raise ValueError("unknown controlled transition mode")
        if outcome_residual_mode not in {"standardized", "raw"}:
            raise ValueError("unknown controlled outcome residual mode")
        self.outcome_model = outcome_model
        self.n_actions = n_actions
        self.horizon = batch.horizon
        self.history_length = history_length
        self.base_state_dim = batch.state_dim // history_length
        self.neighbors = neighbors
        self.bandwidth = bandwidth
        self.ridge = ridge
        self.representation_geometry = representation_geometry
        self.donor_weighting = donor_weighting
        self.ridge_mode = ridge_mode
        self.transition_mode = transition_mode
        self.outcome_residual_mode = outcome_residual_mode
        self.static_base_indices = tuple(sorted({index % self.base_state_dim for index in static_indices}))
        self.cumulative_indices = tuple(
            index for index, name in enumerate(state_feature_names) if name.startswith("cumulative_")
        )
        self.decision_time_index = (
            state_feature_names.index("decision_time") if "decision_time" in state_feature_names else None
        )
        self.initial_states = batch.states[:, 0].cpu()
        self.initial_count = batch.n

        current = batch.current_states().cpu()
        next_frames = batch.states[:, 1:].cpu().reshape(batch.n, batch.horizon, history_length, self.base_state_dim)[:, :, -1]
        representation = self._representation(current.reshape(-1, batch.state_dim)).reshape(batch.n, batch.horizon, -1)
        payload = self._outcome_residuals(batch)
        actions = batch.actions.cpu()
        self._models: list[_StageModel] = []
        self._libraries: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
        self._library_patient_codes: dict[tuple[int, int], tuple[Tensor, int]] = {}
        self._metric_transforms: dict[int, tuple[Tensor, Tensor]] = {}
        for time in range(batch.horizon):
            metric_center, metric_scale = self._metric_transform(
                representation[:, time],
                representation_geometry,
            )
            self._metric_transforms[time] = (metric_center, metric_scale)
            features = self._features(representation[:, time], actions[:, time])
            coefficients = self._fit_ridge(
                features,
                next_frames[:, time],
                ridge,
                mode=ridge_mode,
            )
            self._models.append(_StageModel(coefficients))
            predicted = features @ coefficients
            current_frame = current[:, time].reshape(batch.n, history_length, self.base_state_dim)[:, -1]
            state_payload = (
                next_frames[:, time] - predicted
                if transition_mode == "ridge_residual"
                else next_frames[:, time] - current_frame
            )
            for action in range(n_actions):
                rows = actions[:, time].eq(action)
                if not rows.any():
                    raise ValueError(f"D_env has no transitions for action {action} at stage {time}")
                donor_representation = representation[rows, time]
                metric_representation = self._apply_metric_transform(
                    donor_representation,
                    time=time,
                )
                self._libraries[(time, action)] = (
                    metric_representation,
                    state_payload[rows],
                    payload[rows, time],
                    difficulty.cpu()[rows, time],
                    next_frames[rows, time][:, self.cumulative_indices]
                    - current_frame[rows][:, self.cumulative_indices]
                    if self.cumulative_indices
                    else current_frame.new_empty((int(rows.sum()), 0)),
                )
                _, patient_codes = torch.unique(
                    batch.patient_ids.cpu()[rows],
                    sorted=True,
                    return_inverse=True,
                )
                self._library_patient_codes[(time, action)] = (
                    patient_codes,
                    int(patient_codes.max().item()) + 1,
                )

    @torch.no_grad()
    def initial_state(self, indices: Tensor, device: str | torch.device) -> Tensor:
        return self.initial_states[indices.cpu()].to(device)

    @torch.no_grad()
    def step_from_uniform(
        self,
        state: Tensor,
        action: Tensor,
        donor_uniform: Tensor,
        *,
        time: int,
        gamma: float,
        action_coordinate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if donor_uniform.shape != (len(state),):
            raise ValueError("donor_uniform must have one draw per state")
        next_state = torch.empty_like(state)
        outcome = state.new_empty((len(state), 2))
        selected_difficulty = state.new_empty(len(state))
        kernel_ess = state.new_empty(len(state))
        probability_max = state.new_empty(len(state))
        representation = self._representation(state)
        current_frame = state.reshape(len(state), self.history_length, self.base_state_dim)[:, -1]
        for action_value in range(self.n_actions):
            rows = action.eq(action_value).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            library_rep, state_payload, outcome_residual, difficulty, cumulative_increment = self._library(
                time, action_value, state.device, state.dtype
            )
            query_representation = self._metric_query_representation(
                representation[rows],
                time=time,
                action=action_value,
            )
            distance = torch.cdist(query_representation, library_rep)
            count = min(self.neighbors, len(library_rep))
            nearest_distance, nearest = distance.topk(
                count, largest=False, sorted=True
            )
            logits = self._base_donor_logits(nearest_distance)
            logits = logits + gamma * action_coordinate.to(logits)[action_value] * difficulty[nearest]
            probability = torch.softmax(logits, dim=1)
            draw = torch.searchsorted(probability.cumsum(dim=1), donor_uniform[rows, None]).squeeze(1)
            draw = draw.clamp_max(count - 1)
            chosen = nearest[torch.arange(len(rows), device=state.device), draw]
            if self.transition_mode == "ridge_residual":
                features = self._features(representation[rows], action[rows])
                predicted = features @ self._models[time].coefficients.to(features)
                frame = predicted + state_payload[chosen]
            else:
                frame = current_frame[rows] + state_payload[chosen]
            if self.static_base_indices:
                frame[:, self.static_base_indices] = current_frame[rows][:, self.static_base_indices]
            if self.cumulative_indices:
                frame[:, self.cumulative_indices] = (
                    current_frame[rows][:, self.cumulative_indices] + cumulative_increment[chosen]
                )
            if self.decision_time_index is not None:
                frame[:, self.decision_time_index] = (time + 1) / self.horizon
            sequence = state[rows].reshape(len(rows), self.history_length, self.base_state_dim)
            next_state[rows] = torch.cat((sequence[:, 1:], frame[:, None]), dim=1).reshape(len(rows), -1)
            mean, scale = self.outcome_model(state[rows], action[rows])
            if self.outcome_residual_mode == "standardized":
                outcome[rows] = mean + scale * outcome_residual[chosen]
            else:
                outcome[rows] = mean + outcome_residual[chosen]
            selected_difficulty[rows] = difficulty[chosen]
            kernel_ess[rows] = 1.0 / probability.square().sum(dim=1)
            probability_max[rows] = probability.max(dim=1).values
        return next_state, outcome, selected_difficulty, kernel_ess, probability_max

    @torch.no_grad()
    def patient_aggregated_kernel_diagnostics(
        self,
        state: Tensor,
        action: Tensor,
        *,
        time: int,
        gamma: float,
        action_coordinate: Tensor,
        chunk_size: int = 512,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return patient-level donor ESS, maximum mass, and local unique k.

        A clinical patient can contribute more than one eligible episode.  The
        transition kernel remains episode-weighted, but interpretation-quality
        overlap must aggregate the selected neighbor-row probabilities by
        patient before computing concentration diagnostics.
        """

        if state.ndim != 2 or action.shape != (len(state),):
            raise ValueError("state/action shapes do not align")
        patient_ess = state.new_empty(len(state))
        patient_probability_max = state.new_empty(len(state))
        unique_neighbor_count = state.new_empty(len(state))
        representation = self._representation(state)
        for action_value in range(self.n_actions):
            rows = action.eq(action_value).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            library_rep, _, _, difficulty, _ = self._library(
                time, action_value, state.device, state.dtype
            )
            patient_codes_cpu, patient_count = self._library_patient_codes[
                (time, action_value)
            ]
            patient_codes = patient_codes_cpu.to(state.device)
            for row_chunk in rows.split(chunk_size):
                query_representation = self._metric_query_representation(
                    representation[row_chunk],
                    time=time,
                    action=action_value,
                )
                distance = torch.cdist(query_representation, library_rep)
                count = min(self.neighbors, len(library_rep))
                nearest_distance, nearest = distance.topk(
                    count, largest=False, sorted=True
                )
                logits = self._base_donor_logits(nearest_distance)
                logits = (
                    logits
                    + gamma
                    * action_coordinate.to(logits)[action_value]
                    * difficulty[nearest]
                )
                probability = torch.softmax(logits, dim=1)
                local_patient_codes = patient_codes[nearest]
                mass = probability.new_zeros((len(row_chunk), patient_count))
                mass.scatter_add_(1, local_patient_codes, probability)
                patient_ess[row_chunk] = 1.0 / mass.square().sum(dim=1)
                patient_probability_max[row_chunk] = mass.max(dim=1).values
                unique_neighbor_count[row_chunk] = mass.gt(0).sum(dim=1).to(
                    unique_neighbor_count
                )
        return patient_ess, patient_probability_max, unique_neighbor_count

    def _features(self, representation: Tensor, action: Tensor) -> Tensor:
        one_hot = torch.nn.functional.one_hot(action.to(torch.long), self.n_actions).to(representation)
        return torch.cat((torch.ones((len(representation), 1), device=representation.device, dtype=representation.dtype), representation, one_hot), dim=1)

    @staticmethod
    def _fit_ridge(
        features: Tensor,
        target: Tensor,
        ridge: float,
        *,
        mode: str = "v2_raw",
    ) -> Tensor:
        gram = features.T @ features
        right_hand_side = features.T @ target
        penalty = torch.eye(
            len(gram), dtype=features.dtype, device=features.device
        )
        if mode == "sample_normalized_no_intercept":
            gram = gram / len(features)
            right_hand_side = right_hand_side / len(features)
            penalty[0, 0] = 0.0
        return torch.linalg.solve(gram + ridge * penalty, right_hand_side)

    @staticmethod
    def _metric_transform(
        representation: Tensor,
        geometry: str,
    ) -> tuple[Tensor, Tensor]:
        representation64 = representation.to(torch.float64)
        if geometry == "raw":
            return (
                representation64.new_zeros(representation.shape[1]),
                representation64.new_ones(representation.shape[1]),
            )
        return (
            representation64.mean(dim=0),
            representation64.std(dim=0, unbiased=False).clamp_min(1e-4),
        )

    def _apply_metric_transform(
        self,
        representation: Tensor,
        *,
        time: int,
    ) -> Tensor:
        if self.representation_geometry == "raw":
            return representation
        center, scale = self._metric_transforms[time]
        return (
            representation - center.to(representation)
        ) / scale.to(representation)

    def _metric_query_representation(
        self,
        representation: Tensor,
        *,
        time: int,
        action: int,
    ) -> Tensor:
        del action
        return self._apply_metric_transform(representation, time=time)

    def _base_donor_logits(self, nearest_distance: Tensor) -> Tensor:
        if self.donor_weighting == "uniform":
            return torch.zeros_like(nearest_distance)
        local_scale = _lower_median_from_sorted_rows(nearest_distance).clamp_min(
            1e-6
        )
        return -(
            (nearest_distance / local_scale[:, None]).square()
        ) / (2.0 * self.bandwidth**2)

    @torch.no_grad()
    def _representation(self, states: Tensor) -> Tensor:
        device = _module_device(self.outcome_model)
        parts = []
        for rows in states.to(device).split(4_096):
            parts.append(self.outcome_model.representation(rows).cpu())
        return torch.cat(parts).to(states)

    @torch.no_grad()
    def _outcome_residuals(self, batch: TrajectoryBatch) -> Tensor:
        states, actions, outcomes = batch.flat_transitions()
        device = _module_device(self.outcome_model)
        residuals = []
        for state_part, action_part, outcome_part in zip(states.split(4_096), actions.split(4_096), outcomes.split(4_096)):
            mean, scale = self.outcome_model(state_part.to(device), action_part.to(device))
            residual = outcome_part.to(device) - mean
            if self.outcome_residual_mode == "standardized":
                residual = residual / scale.clamp_min(1e-6)
            residuals.append(residual.cpu())
        return torch.cat(residuals).reshape(batch.n, batch.horizon, batch.outcome_dim)

    def _library(self, time: int, action: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return tuple(value.to(device=device, dtype=dtype) for value in self._libraries[(time, action)])  # type: ignore[return-value]


@torch.no_grad()
def make_controlled_noise(*, n: int, horizon: int, initial_count: int, seed: int, device: str | torch.device) -> ControlledNoiseBundle:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ControlledNoiseBundle(
        initial_indices=torch.randint(initial_count, (n,), generator=generator, device=device),
        action_uniforms=torch.rand((horizon, n), generator=generator, device=device),
        donor_uniforms=torch.rand((horizon, n), generator=generator, device=device),
    )


@torch.no_grad()
def rollout_controlled(
    environment: ControlledResidualEnvironment,
    policy: object,
    *,
    noise: ControlledNoiseBundle,
    gamma: float,
    action_coordinate: Tensor,
    radii: Tensor | None = None,
) -> ControlledRollout:
    if noise.action_uniforms.shape != (environment.horizon, len(noise.initial_indices)):
        raise ValueError("noise horizon does not match environment")
    device = noise.initial_indices.device
    state = environment.initial_state(noise.initial_indices, device)
    states, actions, outcomes, difficulties, esses, maxima = [state], [], [], [], [], []
    for time in range(environment.horizon):
        radius = None if radii is None else radii[time]
        probability = policy.probabilities(state, radius)
        action = inverse_cdf_actions(probability, noise.action_uniforms[time])
        state, outcome, difficulty, ess, maximum = environment.step_from_uniform(
            state, action, noise.donor_uniforms[time], time=time, gamma=gamma, action_coordinate=action_coordinate
        )
        states.append(state); actions.append(action); outcomes.append(outcome)
        difficulties.append(difficulty); esses.append(ess); maxima.append(maximum)
    batch = TrajectoryBatch(
        states=torch.stack(states, dim=1), actions=torch.stack(actions, dim=1), outcomes=torch.stack(outcomes, dim=1),
        patient_ids=torch.arange(len(noise.initial_indices), device=device),
    )
    return ControlledRollout(batch, torch.stack(difficulties, dim=1), torch.stack(esses, dim=1), torch.stack(maxima, dim=1))


def _module_device(module: object) -> torch.device:
    return next(module.parameters()).device


def _lower_median_from_sorted_rows(values: Tensor) -> Tensor:
    """Match ``torch.median(dim=1)`` without CUDA's indexed median kernel."""

    return values[:, (values.shape[1] - 1) // 2]
