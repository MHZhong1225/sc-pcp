"""Per-step q-conditional Conformal Occupancy Transport (COT)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from scpcp.config import COTConfig
from scpcp.data import TrajectoryBatch
from scpcp.outcome_model import GaussianOutcomeModel
from scpcp.policy import BehaviorAnchoredPolicy


class _RatioHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: tuple[int, ...], rho_cap: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = feature_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(current, width), nn.ReLU()))
            current = width
        output = nn.Linear(current, 1)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, math.log(math.expm1(1.0 - 1e-6)))
        layers.append(output)
        self.network = nn.Sequential(*layers)
        self.rho_cap = rho_cap

    def forward(self, features: Tensor) -> Tensor:
        return (nn.functional.softplus(self.network(features).squeeze(1)) + 1e-6).clamp_max(self.rho_cap)


class QConditionalCOT(nn.Module):
    """One positive occupancy-ratio head per noninitial time point.

    The feature vector includes the complete observed state and the frozen
    outcome representation.  Keeping the complete state makes the Markov
    identification assumption explicit rather than silently relying on a lossy
    representation.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        horizon: int,
        outcome_model: GaussianOutcomeModel,
        q_grid: Tensor,
        config: COTConfig,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.config = config
        self.outcome_model = outcome_model
        representation_dim = outcome_model.config.representation_dim
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("q_center", q_grid.float().mean())
        self.register_buffer("q_scale", q_grid.float().std(unbiased=False).clamp_min(1e-4))
        self.register_buffer("q_reference", q_grid.float().clone())
        self.register_buffer("normalization_scales", torch.ones(max(0, horizon - 1), len(q_grid)))
        self.heads = nn.ModuleList(
            _RatioHead(state_dim + representation_dim + 1, config.hidden_dims, config.rho_cap)
            for _ in range(max(0, horizon - 1))
        )

    def features(self, states: Tensor, q: Tensor) -> Tensor:
        if q.shape != (len(states),):
            raise ValueError("COT q values must have shape [N]")
        state_features = self._state_features(states)
        normalized_q = ((q - self.q_center) / self.q_scale)[:, None]
        return torch.cat((state_features, normalized_q), dim=1)

    def _state_features(self, states: Tensor) -> Tensor:
        """Encode each state once, independently of the candidate radius."""

        normalized_state = ((states - self.state_center) / self.state_scale).clamp(-10.0, 10.0)
        with torch.no_grad():
            representation = self.outcome_model.representation(states)
        return torch.cat((normalized_state, representation), dim=1)

    @torch.no_grad()
    def rho(self, time: int, states: Tensor, q: Tensor) -> Tensor:
        if time == 0:
            return torch.ones(len(states), device=states.device, dtype=states.dtype)
        if not 0 < time < self.horizon:
            raise ValueError("COT time is out of range")
        raw = self.heads[time - 1](self.features(states, q))
        reference = self.q_reference.to(q)
        nearest = (q[:, None] - reference[None, :]).abs().argmin(dim=1)
        scale = self.normalization_scales[time - 1, nearest].to(raw)
        return (raw / scale.clamp_min(1e-8)).clamp_max(self.config.rho_cap)

    @torch.no_grad()
    def rho_for_grid(self, time: int, states: Tensor, q_grid: Tensor) -> Tensor:
        if time == 0:
            return torch.ones((len(states), len(q_grid)), device=states.device, dtype=states.dtype)
        if not 0 < time < self.horizon:
            raise ValueError("COT time is out of range")
        resolved_grid = q_grid.to(states)
        raw = self._raw_rho_for_grid(time, states, resolved_grid)
        reference = self.q_reference.to(resolved_grid)
        nearest = (resolved_grid[:, None] - reference[None, :]).abs().argmin(dim=1)
        scales = self.normalization_scales[time - 1, nearest].to(raw)
        return (raw / scales[None, :].clamp_min(1e-8)).clamp_max(self.config.rho_cap)

    @torch.no_grad()
    def calibrate_head(self, time: int, states: Tensor) -> None:
        """Calibrate each frozen q head to empirical mean one on D_COT."""

        if not 0 < time < self.horizon:
            raise ValueError("COT calibration time is out of range")
        q_grid = self.q_reference.to(states)
        raw = self._raw_rho_for_grid(time, states, q_grid)
        scales = _mean_one_scales(raw, cap=self.config.rho_cap)
        self.normalization_scales[time - 1].copy_(scales.to(self.normalization_scales))

    def _raw_rho_for_grid(self, time: int, states: Tensor, q_grid: Tensor) -> Tensor:
        """Evaluate a frozen q-family without repeating states through the GRU."""

        normalized_q = ((q_grid - self.q_center) / self.q_scale)[None, :, None]
        pieces = []
        for state_batch in states.split(self.config.batch_size):
            encoded = self._state_features(state_batch)
            expanded_state = encoded[:, None, :].expand(-1, len(q_grid), -1)
            expanded_q = normalized_q.expand(len(state_batch), -1, -1)
            features = torch.cat((expanded_state, expanded_q), dim=2).reshape(
                len(state_batch) * len(q_grid), -1
            )
            pieces.append(
                self.heads[time - 1](features).reshape(len(state_batch), len(q_grid))
            )
        return torch.cat(pieces, dim=0)


@dataclass(frozen=True)
class COTDiagnostics:
    validation_losses: tuple[float, ...]
    normalization_errors: tuple[float, ...]
    pseudo_target_cap_rates: tuple[float, ...]


@dataclass(frozen=True)
class FittedCOT:
    model: QConditionalCOT
    q_grid: Tensor
    diagnostics: COTDiagnostics


@dataclass(frozen=True)
class WeightDiagnostics:
    """Pre-truncation weight diagnostics, indexed [q, time]."""

    raw_variance: Tensor
    cap_hit_rate: Tensor
    raw_maximum: Tensor


def fit_cot(
    batch: TrajectoryBatch,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    outcome_model: GaussianOutcomeModel,
    config: COTConfig,
    device: str | torch.device,
    seed: int,
) -> FittedCOT:
    r"""Fit \(\rho_{t+1}^q(S_{t+1})=E[Z_t^q\mid S_{t+1}]\) stage by stage.

    Previously fitted heads are frozen before their predictions form the next
    stage's pseudo-target.  MSE is the default because it targets the required
    conditional mean; Huber is exposed only as a practical robustness option.
    """

    resolved = torch.device(device)
    q_grid = q_grid.to(resolved)
    batch = batch.to(resolved)
    model = QConditionalCOT(
        state_dim=batch.state_dim,
        horizon=batch.horizon,
        outcome_model=outcome_model,
        q_grid=q_grid,
        config=config,
    ).to(resolved)
    model.state_center.copy_(batch.current_states().reshape(-1, batch.state_dim).mean(dim=0))
    model.state_scale.copy_(batch.current_states().reshape(-1, batch.state_dim).std(dim=0).clamp_min(1e-4))
    generator = torch.Generator().manual_seed(seed)
    # Split internal early-stopping roles by patient ID.  Some clinical
    # sources contain multiple stays for one patient; splitting rows would
    # leak that patient's trajectory into both training and validation.
    patient_ids = batch.patient_ids.detach().cpu()
    unique_patients = torch.unique(patient_ids, sorted=True)
    patient_order = unique_patients[torch.randperm(len(unique_patients), generator=generator)]
    if len(patient_order) == 1:
        train_ids = valid_ids = patient_order
    else:
        split = min(
            len(patient_order) - 1,
            max(1, int((1.0 - config.validation_fraction) * len(patient_order))),
        )
        train_ids, valid_ids = patient_order[:split], patient_order[split:]
    train_patients = torch.isin(patient_ids, train_ids).nonzero().squeeze(1).to(resolved)
    valid_patients = torch.isin(patient_ids, valid_ids).nonzero().squeeze(1).to(resolved)
    validation_losses: list[float] = []
    normalization_errors: list[float] = []
    cap_rates: list[float] = []
    for time in range(batch.horizon - 1):
        head = model.heads[time]
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        best_state, best_loss, stale = None, float("inf"), 0
        target_cap_count, target_count = 0, 0
        for _ in range(config.epochs):
            head.train()
            order = train_patients[torch.randperm(len(train_patients), device=resolved)]
            for patients in order.split(config.batch_size):
                # Use a whole patient mini-batch for each sampled candidate q.
                # This makes the normalization term target E_mu[rho_t^q]=1 for
                # each q, rather than only after averaging together unrelated
                # q values across patients.
                q_count = min(config.q_samples_per_batch, len(q_grid))
                sampled_q = q_grid[torch.randperm(len(q_grid), device=resolved)[:q_count]]
                losses = []
                for scalar_q in sampled_q:
                    q_values = scalar_q.expand(len(patients))
                    pseudo_target, capped = _pseudo_target(
                        model,
                        batch,
                        time,
                        patients,
                        q_values,
                        target_policy,
                        logging_policy,
                        config,
                    )
                    target_cap_count += int(capped.sum().item())
                    target_count += len(capped)
                    prediction = head(model.features(batch.states[patients, time + 1], q_values))
                    loss = _regression_loss(prediction, pseudo_target, config.loss)
                    losses.append(loss + config.normalization_penalty * (prediction.mean() - 1.0).square())
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
                optimizer.step()
            head.eval()
            with torch.no_grad():
                validation_terms = []
                for scalar_q in q_grid:
                    valid_q = scalar_q.expand(len(valid_patients))
                    valid_target, _ = _pseudo_target(
                        model,
                        batch,
                        time,
                        valid_patients,
                        valid_q,
                        target_policy,
                        logging_policy,
                        config,
                    )
                    valid_prediction = head(model.features(batch.states[valid_patients, time + 1], valid_q))
                    validation_terms.append(
                        _regression_loss(valid_prediction, valid_target, config.loss)
                        + config.normalization_penalty * (valid_prediction.mean() - 1.0).square()
                    )
                valid_loss = float(torch.stack(validation_terms).mean().item())
            if valid_loss < best_loss:
                best_loss, stale, best_state = valid_loss, 0, deepcopy(head.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_state is not None:
            head.load_state_dict(best_state)
        head.eval()
        model.calibrate_head(time + 1, batch.states[train_patients, time + 1])
        with torch.no_grad():
            normalization = (
                model.rho_for_grid(time + 1, batch.states[valid_patients, time + 1], q_grid)
                .mean(dim=0)
                .sub(1.0)
                .abs()
                .max()
            )
        validation_losses.append(best_loss)
        normalization_errors.append(float(normalization.item()))
        cap_rates.append(0.0 if target_count == 0 else target_cap_count / target_count)
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    model.eval()
    return FittedCOT(
        model=model,
        q_grid=q_grid.detach().cpu(),
        diagnostics=COTDiagnostics(tuple(validation_losses), tuple(normalization_errors), tuple(cap_rates)),
    )


@torch.no_grad()
def cot_state_action_weights(
    fitted: FittedCOT,
    batch: TrajectoryBatch,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    weight_cap: float,
) -> tuple[Tensor, WeightDiagnostics]:
    r"""Return capped \(\hat\omega_{it}^q\) plus pre-cap diagnostics."""

    model = fitted.model
    device = next(model.parameters()).device
    batch = batch.to(device)
    q_grid = q_grid.to(device)
    weights, variances, cap_rates, maxima = [], [], [], []
    for time in range(batch.horizon):
        states = batch.states[:, time]
        rho = model.rho_for_grid(time, states, q_grid)
        target_probabilities = target_policy.probabilities_for_grid(states, q_grid)
        observed_action = batch.actions[:, time, None, None].expand(-1, len(q_grid), 1)
        numerator = target_probabilities.gather(2, observed_action).squeeze(2)
        denominator_all = logging_policy.probabilities(states).clamp_min(1e-12)
        denominator = denominator_all.gather(1, batch.actions[:, time, None]).expand(-1, len(q_grid))
        raw = rho * numerator / denominator
        variances.append(raw.var(dim=0, unbiased=False))
        cap_rates.append((raw > weight_cap).float().mean(dim=0))
        maxima.append(raw.max(dim=0).values)
        weights.append(raw.clamp_max(weight_cap))
    return torch.stack(weights, dim=1), WeightDiagnostics(
        raw_variance=torch.stack(variances, dim=1),
        cap_hit_rate=torch.stack(cap_rates, dim=1),
        raw_maximum=torch.stack(maxima, dim=1),
    )


@torch.no_grad()
def prefix_importance_weights(
    batch: TrajectoryBatch,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    weight_cap: float,
) -> tuple[Tensor, WeightDiagnostics]:
    """Capped prefix IW plus its untruncated variance/overlap diagnostics."""

    device = next(target_policy.outcome_model.parameters()).device
    batch = batch.to(device)
    q_grid = q_grid.to(device)
    log_weight = torch.zeros((batch.n, len(q_grid)), device=device)
    weights, variances, cap_rates, maxima = [], [], [], []
    for time in range(batch.horizon):
        states = batch.states[:, time]
        target_probabilities = target_policy.probabilities_for_grid(states, q_grid)
        action_index = batch.actions[:, time, None, None].expand(-1, len(q_grid), 1)
        numerator = target_probabilities.gather(2, action_index).squeeze(2)
        denominator = logging_policy.probabilities(states).gather(1, batch.actions[:, time, None]).expand(-1, len(q_grid))
        log_weight += (numerator.clamp_min(1e-12) / denominator.clamp_min(1e-12)).log()
        raw = log_weight.exp()
        variances.append(raw.var(dim=0, unbiased=False))
        cap_rates.append((raw > weight_cap).float().mean(dim=0))
        maxima.append(raw.max(dim=0).values)
        weights.append(raw.clamp_max(weight_cap))
    return torch.stack(weights, dim=1), WeightDiagnostics(
        raw_variance=torch.stack(variances, dim=1),
        cap_hit_rate=torch.stack(cap_rates, dim=1),
        raw_maximum=torch.stack(maxima, dim=1),
    )


@torch.no_grad()
def exact_tabular_state_action_weights(
    environment: object,
    batch: TrajectoryBatch,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
) -> tuple[Tensor, float]:
    """Exact finite-MDP COT weights and a global analytical weight bound.

    This function is deliberately restricted to environments exposing
    ``exact_state_ratios``.  It is the oracle-COT validation path, not a
    continuous-state rollout approximation.
    """

    device = next(target_policy.outcome_model.parameters()).device
    q_grid = q_grid.to(device)
    batch = batch.to(device)
    ratios = environment.exact_state_ratios(
        target_policy, logging_policy, q_grid, batch.horizon, device
    )  # [K,T,S]
    state_index = batch.current_states().argmax(dim=2)
    weights = []
    bound = 0.0
    all_states = torch.eye(environment.n_states, device=device)
    mu_all = logging_policy.probabilities(all_states)
    for time in range(batch.horizon):
        pi_all = target_policy.probabilities_for_grid(all_states, q_grid)  # [S,K,A]
        all_weight = ratios[:, time].transpose(0, 1)[:, :, None] * pi_all / mu_all[:, None, :]
        bound = max(bound, float(all_weight.max().item()))
        pi_observed = target_policy.probabilities_for_grid(batch.states[:, time], q_grid)
        action = batch.actions[:, time, None, None].expand(-1, len(q_grid), 1)
        numerator = pi_observed.gather(2, action).squeeze(2)
        denominator = logging_policy.probabilities(batch.states[:, time]).gather(1, batch.actions[:, time, None]).expand(-1, len(q_grid))
        rho_observed = ratios[:, time, state_index[:, time]].transpose(0, 1)
        weights.append(rho_observed * numerator / denominator)
    return torch.stack(weights, dim=1), bound


@torch.no_grad()
def exact_tabular_cot_l1_error_bound(
    fitted: FittedCOT,
    environment: object,
    *,
    q_grid: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    weight_cap: float,
) -> Tensor:
    """Enumerate the exact tabular L1 error of the *capped* learned COT weights.

    This is restricted to the finite-MDP theorem-validation branch.  At every
    ``(q, t)`` it returns

    ``E_{(S_t,A_t)~P_mu}|clip(hat_omega_t^q, B) - omega_t^q|``.

    The quantity includes any bias induced by the estimator's configured
    weight cap, so it is a valid deterministic transport-error term for the
    bounded-weight certificate in this fully known environment.  It is not a
    learned-model validation loss and cannot be used by continuous or clinical
    runs.
    """

    if weight_cap <= 0.0:
        raise ValueError("weight_cap must be positive")
    if not all(hasattr(environment, name) for name in ("exact_state_ratios", "transition_probabilities", "n_states")):
        raise ValueError("exact tabular L1 validation requires a finite environment with occupancy access")
    model = fitted.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    q_grid = q_grid.to(device=device, dtype=dtype)
    states = torch.eye(environment.n_states, device=device, dtype=dtype)
    transition = environment.transition_probabilities(device, dtype)
    mu = logging_policy.probabilities(states).clamp_min(1e-12)
    d_mu = [torch.full((environment.n_states,), 1.0 / environment.n_states, device=device, dtype=dtype)]
    for _ in range(model.horizon - 1):
        d_mu.append(torch.einsum("s,sa,asr->r", d_mu[-1], mu, transition))
    exact_rho = environment.exact_state_ratios(
        target_policy, logging_policy, q_grid, model.horizon, device
    ).to(dtype)  # [K,T,S]
    errors = []
    for time in range(model.horizon):
        pi = target_policy.probabilities_for_grid(states, q_grid)  # [S,K,A]
        learned_rho = model.rho_for_grid(time, states, q_grid)  # [S,K]
        learned_weight = (learned_rho[:, :, None] * pi / mu[:, None, :]).clamp_max(weight_cap)
        exact_weight = exact_rho[:, time].transpose(0, 1)[:, :, None] * pi / mu[:, None, :]
        joint_logging = d_mu[time][:, None, None] * mu[:, None, :]
        errors.append((joint_logging * (learned_weight - exact_weight).abs()).sum(dim=(0, 2)))
    return torch.stack(errors, dim=1)


@torch.no_grad()
def _pseudo_target(
    model: QConditionalCOT,
    batch: TrajectoryBatch,
    time: int,
    patients: Tensor,
    q_values: Tensor,
    target_policy: BehaviorAnchoredPolicy,
    logging_policy: object,
    config: COTConfig,
) -> tuple[Tensor, Tensor]:
    current_state = batch.states[patients, time]
    previous_ratio = model.rho(time, current_state, q_values)
    target_probabilities = target_policy.probabilities(current_state, q_values)
    observed_actions = batch.actions[patients, time]
    numerator = target_probabilities.gather(1, observed_actions[:, None]).squeeze(1)
    denominator = logging_policy.probabilities(current_state).gather(1, observed_actions[:, None]).squeeze(1)
    raw = previous_ratio * numerator / denominator.clamp_min(1e-12)
    return raw.clamp_max(config.rho_cap), raw > config.rho_cap


def _regression_loss(prediction: Tensor, target: Tensor, loss_name: str) -> Tensor:
    if loss_name == "mse":
        return nn.functional.mse_loss(prediction, target)
    return nn.functional.smooth_l1_loss(prediction, target)


@torch.no_grad()
def _mean_one_scales(raw_ratios: Tensor, *, cap: float) -> Tensor:
    """Find per-column scales whose capped ratios have empirical mean one."""

    if raw_ratios.ndim != 2 or len(raw_ratios) == 0:
        raise ValueError("raw COT ratios must have shape [N,K]")
    if cap < 1.0:
        raise ValueError("a mean-one density ratio requires cap >= 1")
    lower = torch.full_like(raw_ratios[0], 1e-8)
    upper = raw_ratios.max(dim=0).values.clamp_min(1e-8)
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        mean = (raw_ratios / middle[None, :]).clamp_max(cap).mean(dim=0)
        lower = torch.where(mean > 1.0, middle, lower)
        upper = torch.where(mean > 1.0, upper, middle)
    return upper
