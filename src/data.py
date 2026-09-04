"""Trajectory containers and patient-level data splitting."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryBatch:
    r"""Aligned sequential treatment observations.

    ``states[i, t]`` is \(S_t\), ``actions[i, t]`` is \(A_t\), and
    ``outcomes[i, t]`` is \(Y_{t+1}\).  The class intentionally has no path
    maximum: the main SC-PCP formulation is strictly per-step.
    """

    states: Tensor
    actions: Tensor
    outcomes: Tensor
    patient_ids: Tensor

    def __post_init__(self) -> None:
        if self.states.ndim != 3 or self.actions.ndim != 2 or self.outcomes.ndim != 3:
            raise ValueError("states, actions, and outcomes must be [N,T+1,D], [N,T], [N,T,Y]")
        n, horizon = self.actions.shape
        if self.states.shape[:2] != (n, horizon + 1):
            raise ValueError("states must have one more time point than actions")
        if self.outcomes.shape[:2] != (n, horizon):
            raise ValueError("outcomes must align with actions")
        if self.patient_ids.shape != (n,):
            raise ValueError("patient_ids must have shape [N]")

    @property
    def n(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    @property
    def state_dim(self) -> int:
        return self.states.shape[-1]

    @property
    def outcome_dim(self) -> int:
        return self.outcomes.shape[-1]

    def current_states(self) -> Tensor:
        return self.states[:, :-1]

    def flat_transitions(self) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.current_states().reshape(-1, self.state_dim),
            self.actions.reshape(-1),
            self.outcomes.reshape(-1, self.outcome_dim),
        )

    def subset(self, indices: Tensor) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{field.name: getattr(self, field.name)[indices] for field in fields(self)}
        )

    def prefix(self, horizon: int) -> "TrajectoryBatch":
        if not 1 <= horizon <= self.horizon:
            raise ValueError("prefix horizon is out of range")
        return TrajectoryBatch(
            states=self.states[:, : horizon + 1],
            actions=self.actions[:, :horizon],
            outcomes=self.outcomes[:, :horizon],
            patient_ids=self.patient_ids,
        )

    def to(self, device: str | torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


@dataclass(frozen=True)
class DataSplits:
    predictor: TrajectoryBatch
    behavior: TrajectoryBatch | None
    cot: TrajectoryBatch
    certification: TrajectoryBatch
    environment: TrajectoryBatch | None = None


def patient_level_splits(
    batch: TrajectoryBatch,
    *,
    seed: int,
    include_environment: bool,
    include_behavior: bool = True,
) -> DataSplits:
    """Create disjoint roles without reserving data for known objects."""

    unique_ids = torch.unique(batch.patient_ids.cpu(), sorted=True)
    generator = torch.Generator().manual_seed(seed)
    shuffled = unique_ids[torch.randperm(len(unique_ids), generator=generator)]
    if include_environment:
        fractions = (
            (0.40, 0.15, 0.15, 0.15, 0.15)
            if include_behavior
            else (0.40, 0.15, 0.30, 0.15)
        )
    elif include_behavior:
        fractions = (0.40, 0.20, 0.20, 0.20)
    else:
        fractions = (0.40, 0.20, 0.40)
    counts = _split_counts(len(shuffled), fractions)
    groups = []
    cursor = 0
    for count in counts:
        ids = shuffled[cursor : cursor + count]
        groups.append(torch.isin(batch.patient_ids.cpu(), ids).nonzero().squeeze(1).to(batch.patient_ids.device))
        cursor += count
    if include_environment:
        if not include_behavior:
            return DataSplits(
                predictor=batch.subset(groups[0]),
                behavior=None,
                cot=batch.subset(groups[1]),
                certification=batch.subset(groups[2]),
                environment=batch.subset(groups[3]),
            )
        return DataSplits(
            predictor=batch.subset(groups[0]),
            behavior=batch.subset(groups[1]),
            cot=batch.subset(groups[2]),
            certification=batch.subset(groups[3]),
            environment=batch.subset(groups[4]),
        )
    if not include_behavior:
        return DataSplits(
            predictor=batch.subset(groups[0]),
            behavior=None,
            cot=batch.subset(groups[1]),
            certification=batch.subset(groups[2]),
        )
    return DataSplits(
        predictor=batch.subset(groups[0]),
        behavior=batch.subset(groups[1]),
        cot=batch.subset(groups[2]),
        certification=batch.subset(groups[3]),
    )


def concatenate_trajectories(*batches: TrajectoryBatch) -> TrajectoryBatch:
    """Combine disjoint trajectory roles for a baseline's calibration budget."""

    if not batches:
        raise ValueError("at least one trajectory batch is required")
    return TrajectoryBatch(
        **{
            field.name: torch.cat([getattr(batch, field.name) for batch in batches], dim=0)
            for field in fields(TrajectoryBatch)
        }
    )


def _split_counts(total: int, fractions: tuple[float, ...]) -> list[int]:
    if total < len(fractions):
        raise ValueError("not enough patients for the requested split")
    counts = [max(1, int(total * fraction)) for fraction in fractions]
    while sum(counts) > total:
        largest = max(range(len(counts)), key=counts.__getitem__)
        if counts[largest] == 1:
            raise ValueError("not enough patients for nonempty split roles")
        counts[largest] -= 1
    counts[-1] += total - sum(counts)
    return counts
