"""Configuration for the per-step SC-PCP experiments.

Paper-scale settings live in ``configs/per_step_synthetic.yaml`` and
``configs/per_step_{dataset}.yaml`` so the normal command line only needs a
config path, seed range, and devices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass(frozen=True)
class SyntheticConfig:
    r"""Controlled two-outcome treatment environment.

    ``feedback_strength`` is the \(\beta\) in the method description.  Every
    action-dependent transition and noise term is multiplied by it, making the
    score law invariant to the policy at beta=0 (up to predictor fitting noise).
    """

    feedback_strength: float = 1.0
    state_clip: float = 8.0
    disease_treatment_effect: float = 0.75
    toxicity_treatment_effect: float = 0.55
    disease_noise: float = 0.35
    toxicity_noise: float = 0.25
    state_persistence: float = 0.78
    nonlinear_strength: float = 0.25
    scenario: str = "standard"
    difficulty_initial_probability: float = 0.15
    difficulty_intercept: float = -2.0
    difficulty_state_effect: float = 0.35
    difficulty_persistence: float = 2.0
    difficulty_treatment_effect: float = 1.25
    tail_contamination_probability: float = 0.10
    tail_scale: float = 4.0


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "mlp"
    history_length: int = 4
    hidden_dim: int = 128
    representation_dim: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    patience: int = 10
    min_scale: float = 1e-3
    gradient_clip: float = 5.0


@dataclass(frozen=True)
class PolicyConfig:
    tilt: float = 1.0
    temperature: float = 1.0
    disease_weight: float = 0.5
    toxicity_weight: float = 0.5
    action_costs: tuple[float, ...] = (0.0, 0.05, 0.10)
    propensity_floor: float = 0.01
    policy_ratio_cap: float = 10.0


@dataclass(frozen=True)
class COTConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    patience: int = 10
    gradient_clip: float = 5.0
    # Keep the direct Python defaults internally consistent with the default
    # policy-ratio cap and the global state-action weight cap.
    rho_cap: float = 4.0
    weight_cap: float = 40.0
    normalization_penalty: float = 1.0
    validation_fraction: float = 0.15
    q_samples_per_batch: int = 4
    loss: str = "huber"


@dataclass(frozen=True)
class ProfileConfig:
    """D_COT-only transport refinement for the SC-PCP schedule shape."""

    refinement_folds: int = 3
    refinement_strength: float = 0.5
    maximum_profile_ratio: float = 1.25
    minimum_effective_size: float = 25.0
    maximum_cap_hit_rate: float = 0.01
    grid_focus_fraction: float = 0.80
    grid_focus_radius: float = 0.075


@dataclass(frozen=True)
class CertificationConfig:
    alpha: float = 0.10
    delta: float = 0.05
    ratio_error_bound: float = 0.0
    # ``none`` is deliberately the default: a neural COT fit alone does not
    # supply the simultaneous L1 ratio-error bound needed for a theorem-level
    # certificate.  ``declared`` must be supplied by an external analysis and
    # is labelled assumption-based.  ``oracle`` is an internal-only value used
    # by the exact finite-MDP validation branch; it is deliberately rejected
    # in user-facing YAML so a learned clinical ratio cannot be mislabeled as
    # oracle-certified.
    ratio_bound_source: str = "none"
    ratio_delta: float = 0.0
    # Used only for the explicitly non-theorem practical cluster bootstrap.
    # One thousand draws keeps the marginal patient-cluster quantiles stable
    # enough for the ordered candidate tests without dominating training.
    practical_bootstrap_resamples: int = 1_000


@dataclass(frozen=True)
class SampleConfig:
    logged: int = 5_000
    oracle_rollouts: int = 50_000
    oracle_surface_rollouts: int = 5_000
    # Total target-policy trajectories available to *each* online baseline.
    # Controllers split this fixed budget across their adaptation rounds.
    online_rollouts: int = 2_000


@dataclass(frozen=True)
class BaselineConfig:
    """Prespecified settings for transparent task-aligned baselines."""

    mfcs_depth: int = 3
    aci_gamma: float = 0.01
    multidim_buffer: int = 1_000
    online_rounds: int = 3
    prc_maximum_step: float = 0.35


@dataclass(frozen=True)
class PaperConfig:
    """Controls manuscript-only mechanism diagnostics."""

    save_mechanism_diagonal: bool = False
    mechanism_seed: int = 0


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "synthetic"
    data_root: Path = Path("/home/ubuntu/zmh/dataset")
    cache_dir: Path = Path("data/real_cache")
    max_patients: int | None = None
    cohort_seed: int = 271_828
    empirical_neighbors: int = 100
    empirical_bandwidth: float = 2.0
    empirical_embedding_dim: int = 32
    cxr_encoder: str = "densenet121"
    cxr_embedding_dim: int = 256
    cxr_epochs: int = 3
    cxr_batch_size: int = 32


@dataclass(frozen=True)
class ExperimentConfig:
    synthetic: SyntheticConfig = SyntheticConfig()
    model: ModelConfig = ModelConfig()
    policy: PolicyConfig = PolicyConfig()
    cot: COTConfig = COTConfig()
    profile: ProfileConfig = ProfileConfig()
    certification: CertificationConfig = CertificationConfig()
    samples: SampleConfig = SampleConfig()
    baselines: BaselineConfig = BaselineConfig()
    paper: PaperConfig = PaperConfig()
    data: DataConfig = DataConfig()
    horizon: int = 12
    q_grid_size: int = 101
    q_quantile_min: float = 0.50
    q_quantile_max: float = 0.999
    seeds: tuple[int, ...] = (0,)
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    output_dir: Path = Path("results/per_step")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        config = _dataclass_from_mapping(cls, raw)
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        devices: tuple[str, ...] | None = None,
        output_dir: Path | None = None,
        seeds: tuple[int, ...] | None = None,
    ) -> "ExperimentConfig":
        result = replace(
            self,
            devices=self.devices if devices is None else devices,
            output_dir=self.output_dir if output_dir is None else output_dir,
            seeds=self.seeds if seeds is None else seeds,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.synthetic.scenario not in {"standard", "tail_shift"}:
            raise ValueError("synthetic scenario must be standard or tail_shift")
        if not (
            0.0 <= self.synthetic.difficulty_initial_probability <= 1.0
            and 0.0 <= self.synthetic.tail_contamination_probability <= 1.0
        ):
            raise ValueError("tail-shift probabilities must lie in [0, 1]")
        if self.synthetic.tail_scale <= 0.0:
            raise ValueError("tail_scale must be positive")
        if self.data.dataset not in {"synthetic", "tabular", "mimic_iv", "eicu", "mimic_cxr", "inspire"}:
            raise ValueError("unknown dataset")
        if self.data.cxr_encoder != "densenet121":
            raise ValueError("cxr_encoder must be densenet121")
        if self.data.empirical_neighbors < 1 or self.data.empirical_bandwidth <= 0.0:
            raise ValueError("empirical environment neighbors and bandwidth must be positive")
        if self.data.empirical_embedding_dim < 1:
            raise ValueError("empirical_embedding_dim must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.q_grid_size < 3:
            raise ValueError("q_grid_size must be at least three")
        if not 0.0 <= self.q_quantile_min < self.q_quantile_max <= 1.0:
            raise ValueError("q quantiles must lie in [0, 1] and be ordered")
        if not 0.0 < self.certification.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.certification.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.certification.ratio_error_bound < 0.0:
            raise ValueError("ratio_error_bound must be nonnegative")
        if self.certification.ratio_bound_source not in {"declared", "none"}:
            raise ValueError(
                "ratio_bound_source must be declared or none; oracle is reserved for the internal exact-tabular path"
            )
        if not 0.0 <= self.certification.ratio_delta < self.certification.delta:
            raise ValueError("ratio_delta must lie in [0, delta)")
        if self.certification.ratio_bound_source == "declared" and self.certification.ratio_delta <= 0.0:
            raise ValueError(
                "a declared statistical ratio bound must set positive ratio_delta; "
                "the internal exact-tabular path is the only zero-failure oracle case"
            )
        if self.certification.practical_bootstrap_resamples < 200:
            raise ValueError("practical_bootstrap_resamples must be at least 200")
        if self.model.epochs < 1 or self.cot.epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.model.architecture not in {"mlp", "gru"} or self.model.history_length < 1:
            raise ValueError("model architecture must be mlp or gru with positive history_length")
        if self.model.batch_size < 1 or self.cot.batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if not 0.0 < self.cot.validation_fraction < 0.5:
            raise ValueError("COT validation_fraction must lie in (0, .5)")
        if self.cot.q_samples_per_batch < 1:
            raise ValueError("COT q_samples_per_batch must be positive")
        if self.cot.loss not in {"mse", "huber"}:
            raise ValueError("COT loss must be mse or huber")
        if self.cot.rho_cap < 1.0 or self.cot.weight_cap <= 0.0:
            raise ValueError("COT rho_cap must be at least one and weight_cap must be positive")
        if self.profile.refinement_folds < 2:
            raise ValueError("profile refinement requires at least two patient folds")
        if not 0.0 < self.profile.refinement_strength <= 1.0:
            raise ValueError("profile refinement_strength must lie in (0, 1]")
        if self.profile.maximum_profile_ratio <= 1.0:
            raise ValueError("profile maximum_profile_ratio must exceed one")
        if self.profile.minimum_effective_size <= 0.0:
            raise ValueError("profile minimum_effective_size must be positive")
        if not 0.0 <= self.profile.maximum_cap_hit_rate < 1.0:
            raise ValueError("profile maximum_cap_hit_rate must lie in [0, 1)")
        if not 0.0 < self.profile.grid_focus_fraction < 1.0:
            raise ValueError("profile grid_focus_fraction must lie in (0, 1)")
        if not 0.0 < self.profile.grid_focus_radius < 0.5:
            raise ValueError("profile grid_focus_radius must lie in (0, .5)")
        if 2.0 * self.profile.grid_focus_radius >= (
            self.q_quantile_max - self.q_quantile_min
        ):
            raise ValueError("profile grid focus must fit inside the candidate quantile range")
        if self.policy.temperature <= 0.0:
            raise ValueError("policy temperature must be positive")
        if self.policy.tilt < 0.0:
            raise ValueError("policy tilt must be nonnegative")
        if not 0.0 < self.policy.propensity_floor < 1.0:
            raise ValueError("propensity_floor must lie in (0, 1)")
        if self.policy.policy_ratio_cap < 1.0:
            raise ValueError("policy_ratio_cap must be at least one")
        if self.cot.weight_cap < self.cot.rho_cap * self.policy.policy_ratio_cap:
            raise ValueError(
                "cot.weight_cap must cover rho_cap * policy_ratio_cap; otherwise COT weights are truncated "
                "and any declared L1 bound must be changed explicitly"
            )
        if not self.seeds or not self.devices:
            raise ValueError("at least one seed and device are required")
        if any(value <= 0 for value in asdict(self.samples).values()):
            raise ValueError("sample sizes must be positive")
        if self.baselines.mfcs_depth < 1:
            raise ValueError("MFCS depth must be positive")
        if not 0.0 < self.baselines.aci_gamma <= 1.0:
            raise ValueError("ACI gamma must lie in (0, 1]")
        if self.baselines.multidim_buffer < 1:
            raise ValueError("MultiDimSPCI buffer must be positive")
        if self.baselines.online_rounds < 1:
            raise ValueError("online baseline rounds must be positive")
        if self.baselines.prc_maximum_step <= 0.0:
            raise ValueError("PRC maximum grid step must be positive")
        if self.samples.online_rollouts < self.baselines.online_rounds:
            raise ValueError(
                "online_rollouts must be at least online_rounds so every adaptation round "
                "receives a trajectory"
            )
        if self.paper.mechanism_seed < 0:
            raise ValueError("paper.mechanism_seed must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data"]["data_root"] = str(self.data.data_root)
        result["data"]["cache_dir"] = str(self.data.cache_dir)
        result["output_dir"] = str(self.output_dir)
        return result


T = TypeVar("T")


def _dataclass_from_mapping(cls: type[T], values: dict[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    converted: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        field_type = hints[field.name]
        if hasattr(field_type, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"{field.name} must be a mapping")
            value = _dataclass_from_mapping(field_type, value)
        elif field.name == "seeds":
            if isinstance(value, dict):
                if set(value) != {"start", "stop"}:
                    raise ValueError("seed range must contain exactly start and stop")
                value = tuple(range(int(value["start"]), int(value["stop"])))
            else:
                value = tuple(value)
        elif field.name in {"devices", "action_costs", "hidden_dims"}:
            value = tuple(value)
        elif field.name in {"output_dir", "data_root", "cache_dir"}:
            value = Path(value)
        converted[field.name] = value
    return cls(**converted)
