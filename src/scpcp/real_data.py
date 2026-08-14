"""Leakage-safe clinical trajectory builders for the per-step SC-PCP tasks.

Raw source tables are transformed into explicit
``state -> first-half treatment -> second-half response`` windows.  Every
builder returns continuous treatment exposures first; low/high cutpoints are
then learned only from D_pred patients before actions are assigned to all roles.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
import torch

from scpcp.config import ExperimentConfig
from scpcp.cxr import index_cxr_embeddings
from scpcp.data import TrajectoryBatch, patient_level_splits


MIMIC_VITAL_KIND = {
    220050: "sbp",  # arterial systolic blood pressure
    220051: "dbp",  # arterial diastolic blood pressure
    220052: "map",
    220179: "sbp",  # non-invasive systolic blood pressure
    220180: "dbp",  # non-invasive diastolic blood pressure
    220181: "map",
    220045: "hr",
    220210: "rr",
    220277: "spo2",
    223761: "temperature_fahrenheit",
    223762: "temperature",
}
MIMIC_PRESSORS = {221289, 221749, 221906, 222315, 229617, 229630, 229631, 229632}
# Curated IV crystalloid/colloid concepts.  Do not infer fluid exposure from
# free-text order descriptions: those also contain oral intake, gastric meds,
# and tube flushes in MIMIC-IV.
MIMIC_IV_FLUIDS = {
    220862, 220864,  # albumin 25% / 5%
    220953, 220954, 220955, 220956, 220960,  # legacy saline/Ringer solutions
    225158, 225159, 225823, 225825, 225827, 225828,  # NaCl, D5 fluids, LR
}
MIMIC_RESPIRATORY_DEVICE = 226732
MIMIC_URINE_OUTPUTS = {
    226557,  # right ureteral stent
    226558,  # left ureteral stent
    226559,  # Foley
    226560,  # void
    226561,  # condom catheter
    226564,  # right nephrostomy
    226565,  # left nephrostomy
    226627,  # operating-room urine
    226631,  # PACU urine
    226713,  # incontinent/void estimate
}
CXR_LABEL_COLUMNS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)

# These are the clinically interpretable, time-varying covariates made
# available to every real-data task.  A source that does not contain a feature
# still has a well-defined representation: its value is the clinical default
# and its paired missingness flag is one.  This is preferable to silently
# changing the state schema because a local release lacks one source table.
CLINICAL_STATE_FEATURES = (
    "map",
    "hr",
    "rr",
    "spo2",
    "sbp",
    "dbp",
    "temperature",
    "creatinine",
    "lactate",
    "ph",
    "bicarbonate",
    "hemoglobin",
    "platelets",
    "sodium",
    "potassium",
    "urine_output",
    "ventilation",
    "etco2",
    "fio2",
    "peep",
    "tidal_volume",
    "minute_ventilation",
    "blood_loss",
)
CLINICAL_STATE_DEFAULTS = {
    "map": 80.0,
    "hr": 80.0,
    "rr": 18.0,
    "spo2": 96.0,
    "sbp": 120.0,
    "dbp": 70.0,
    "temperature": 37.0,
    "creatinine": 1.0,
    "lactate": 1.5,
    "ph": 7.40,
    "bicarbonate": 24.0,
    "hemoglobin": 11.0,
    "platelets": 220.0,
    "sodium": 140.0,
    "potassium": 4.0,
    "urine_output": 0.0,
    "ventilation": 0.0,
    "etco2": 35.0,
    "fio2": 21.0,
    "peep": 0.0,
    "tidal_volume": 0.0,
    "minute_ventilation": 0.0,
    "blood_loss": 0.0,
}
CLINICAL_VALUE_BOUNDS = {
    "map": (20.0, 200.0),
    "hr": (20.0, 250.0),
    "rr": (4.0, 80.0),
    "spo2": (20.0, 100.0),
    "sbp": (30.0, 300.0),
    "dbp": (10.0, 200.0),
    "temperature": (25.0, 45.0),
    "creatinine": (0.1, 30.0),
    "lactate": (0.1, 30.0),
    "ph": (6.5, 8.0),
    "bicarbonate": (2.0, 60.0),
    "hemoglobin": (2.0, 25.0),
    "platelets": (1.0, 2_000.0),
    "sodium": (90.0, 200.0),
    "potassium": (1.0, 10.0),
    "urine_output": (0.0, 5_000.0),
    "ventilation": (0.0, 3.0),
    "etco2": (0.0, 100.0),
    "fio2": (15.0, 100.0),
    "peep": (0.0, 30.0),
    "tidal_volume": (0.0, 2_000.0),
    "minute_ventilation": (0.0, 50.0),
    "blood_loss": (0.0, 10_000.0),
}
CLINICAL_SUMMARY_STATS = ("last", "mean", "min", "max")


@dataclass(frozen=True)
class _RawClinicalBatch:
    states: torch.Tensor
    outcomes: torch.Tensor
    treatments: torch.Tensor
    patient_ids: torch.Tensor
    episode_ids: torch.Tensor
    static_indices: tuple[int, ...]
    direct_actions: torch.Tensor | None = None
    # Categorical clinical tasks can use a fixed coarsened ontology before
    # patient roles are assigned.  This keeps action semantics identical over
    # random split seeds rather than deciding an ontology independently in
    # every D_pred role.
    direct_action_count: int | None = None
    original_to_direct_action: dict[int, int] | None = None
    cxr_paths: tuple[str, ...] = ()
    cxr_labels: torch.Tensor | None = None
    cxr_train_mask: torch.Tensor | None = None
    state_feature_names: tuple[str, ...] = ()


def _static_context(
    frame: pd.DataFrame,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...] = (),
    defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Encode admission-time context with explicit numeric missingness flags.

    Categorical context is one-hot encoded on the fixed, cached raw cohort.
    It is therefore available before every action and does not depend on a
    role-specific train/test split.  Numeric values use a cohort median only
    as an imputation value; the adjacent ``*_missing`` column preserves the
    fact that the source value was absent.
    """

    defaults = defaults or {}
    pieces: list[pd.DataFrame] = []
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.notna()
        fallback = defaults.get(column)
        if fallback is None:
            fallback = float(values[observed].median()) if observed.any() else 0.0
        pieces.append(
            pd.DataFrame(
                {
                    column: values.fillna(fallback).astype(np.float32),
                    f"{column}_missing": (~observed).astype(np.float32),
                },
                index=frame.index,
            )
        )
    for column in categorical:
        labels = frame[column].astype("string").str.strip().fillna("__missing__")
        labels = labels.mask(labels.eq(""), "__missing__")
        # Prevent a category seen only in a handful of non-training patients
        # from becoming an effectively unscaled one-hot coordinate.  This
        # support rule is outcome/action agnostic and fixed with the raw
        # cohort; missingness remains a separate, stable level.
        minimum_count = max(2, int(np.ceil(0.01 * len(labels))))
        counts = labels.value_counts()
        rare = counts[(counts < minimum_count) & (counts.index != "__missing__")].index
        labels = labels.mask(labels.isin(rare), "__other__")
        pieces.append(pd.get_dummies(labels, prefix=column, dtype=np.float32))
    if not pieces:
        raise ValueError("static context requires at least one source column")
    return pd.concat(pieces, axis=1).astype(np.float32)


def _clinical_lab_kind(names: pd.Series) -> pd.Series:
    """Map source-specific laboratory labels onto the fixed state schema.

    The matching deliberately favors exact/common chemistry and hematology
    names over broad substring rules.  For example, lactate dehydrogenase and
    urine electrolytes are not substituted for blood lactate/electrolytes.
    """

    labels = names.astype("string").str.lower().str.strip()
    normalized = labels.str.replace(r"[^a-z0-9]+", "", regex=True)
    kind = pd.Series(pd.NA, index=names.index, dtype="string")
    kind.loc[normalized.isin(("creatinine", "creatininewholeblood"))] = "creatinine"
    kind.loc[
        normalized.isin(("lactate", "lacate", "bloodlactate", "lactateblood"))
    ] = "lactate"
    kind.loc[normalized.isin(("ph", "arterialph", "venousph", "pharterial", "phvenous"))] = "ph"
    kind.loc[
        normalized.isin(("bicarbonate", "hco3", "totalco2", "calculatedbicarbonatewholeblood"))
    ] = "bicarbonate"
    kind.loc[
        normalized.isin(("hemoglobin", "hgb", "hb", "hemoglobincalculated"))
    ] = "hemoglobin"
    kind.loc[
        normalized.isin(("plateletcount", "platelets", "platelet", "plateletsx1000"))
    ] = "platelets"
    kind.loc[normalized.isin(("sodium", "sodiumwholeblood"))] = "sodium"
    kind.loc[normalized.isin(("potassium", "potassiumwholeblood"))] = "potassium"
    return kind


def _mimic_lab_item_kinds(root: Path) -> dict[int, str]:
    """Read the small MIMIC dictionary instead of hard-coding local item IDs."""

    items = pd.read_csv(root / "hosp/d_labitems.csv.gz", usecols=["itemid", "label", "fluid"])
    blood = items.fluid.astype("string").str.lower().eq("blood")
    items = items.loc[blood].copy()
    items["kind"] = _clinical_lab_kind(items.label)
    items = items.dropna(subset=["kind"])
    return {int(row.itemid): str(row.kind) for row in items.itertuples(index=False)}


def load_clinical_trajectories(
    config: ExperimentConfig, *, seed: int, device: str | torch.device
) -> tuple[
    TrajectoryBatch,
    int,
    tuple[int, ...],
    tuple[float, ...],
    dict[int, int],
    tuple[str, ...],
]:
    """Load one clinical task, fit its action discretization on D_pred, and return trajectories."""

    raw = _load_or_build_raw(config, seed=seed)
    if config.data.dataset == "mimic_cxr":
        raw = _coarsen_cxr_actions(raw)
    predictor_rows = _predictor_rows(raw, seed)
    if raw.cxr_paths:
        if raw.cxr_labels is None:
            raise RuntimeError("CXR task is missing CheXpert supervision labels")
        encoder_rows = predictor_rows
        if raw.cxr_train_mask is not None:
            encoder_rows = predictor_rows[raw.cxr_train_mask[predictor_rows]]
        if len(encoder_rows) == 0:
            raise RuntimeError("no D_pred index CXRs belong to the official MIMIC-CXR training split")
        embeddings = index_cxr_embeddings(
            list(raw.cxr_paths),
            raw.cxr_labels,
            encoder_rows,
            config=config.data,
            device=device,
            seed=seed + 701,
        )
        states = torch.cat(
            (raw.states, embeddings[:, None, :].expand(-1, raw.states.shape[1], -1)), dim=2
        )
        static_indices = raw.static_indices + tuple(range(raw.states.shape[-1], states.shape[-1]))
        raw = _RawClinicalBatch(
            states=states,
            outcomes=raw.outcomes,
            treatments=raw.treatments,
            patient_ids=raw.patient_ids,
            episode_ids=raw.episode_ids,
            static_indices=static_indices,
            direct_actions=raw.direct_actions,
            direct_action_count=raw.direct_action_count,
            original_to_direct_action=raw.original_to_direct_action,
            state_feature_names=raw.state_feature_names
            + tuple(f"cxr_embedding_{index:03d}" for index in range(embeddings.shape[1])),
        )
    actions, active_actions, direct_to_model = _discretize_actions(raw, predictor_rows)
    original_to_direct = raw.original_to_direct_action or {
        action: action for action in range(max(active_actions) + 1)
    }
    action_mapping = {
        original: direct_to_model[direct]
        for original, direct in original_to_direct.items()
    }
    if len(config.policy.action_costs) < max(active_actions) + 1:
        raise ValueError(
            f"{config.data.dataset} action mapping needs cost index {max(active_actions)} but only "
            f"{len(config.policy.action_costs)} costs are configured"
        )
    action_costs = tuple(config.policy.action_costs[index] for index in active_actions)
    # Clinician treatment propensities are strongly stage-dependent even after
    # conditioning on the measured physiology.  Make decision time an explicit
    # pre-action covariate instead of asking a pooled nuisance model to infer it
    # indirectly from padded histories.  This coordinate is deterministic,
    # outcome-free, and is added before role splitting/model fitting.
    states = _append_decision_time(raw.states)
    state_feature_names = raw.state_feature_names + ("decision_time",)
    states, static_indices = (
        _history_stack(states, raw.static_indices, config.model.history_length)
        if config.model.architecture == "gru"
        else (states, raw.static_indices)
    )
    return (
        TrajectoryBatch(states, actions, raw.outcomes, raw.patient_ids),
        len(active_actions),
        static_indices,
        action_costs,
        action_mapping,
        state_feature_names,
    )


def _load_or_build_raw(config: ExperimentConfig, *, seed: int) -> _RawClinicalBatch:
    limit = config.data.max_patients or 60_000
    # Bump the cache schema when changing temporal eligibility/action parsing;
    # old cached tensors could otherwise retain leakage-prone CXR rows or stale
    # treatment mappings.
    cohort_seed = config.data.cohort_seed
    # Schema v12 adds a fixed, missingness-aware dynamic clinical state
    # (SBP/DBP/temperature and laboratory covariates) and richer
    # admission/ICU context.  Reuse of an older tensor would silently omit
    # these covariates, so every clinical cache is deliberately invalidated.
    # v17 summarizes every completed clinical interval by last/mean/min/max,
    # retains response-half outcomes separately, adds cardiorespiratory/output
    # context, and uses explicit cell-level eICU IV-fluid exposure.  Older
    # caches have different state/action semantics and must not be mixed with
    # this representation.
    schema_version = 17
    cache = Path(config.data.cache_dir) / f"per_step_v{schema_version}_{config.data.dataset}_h{config.horizon}_n{limit}_c{cohort_seed}.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=False)
        return _RawClinicalBatch(**stored)
    builders: dict[str, Callable[[ExperimentConfig, int, int], _RawClinicalBatch]] = {
        "mimic_iv": _build_mimic_iv,
        "eicu": _build_eicu,
        "mimic_cxr": _build_mimic_cxr,
        "inspire": _build_inspire,
    }
    if config.data.dataset not in builders:
        raise ValueError(f"no clinical builder for {config.data.dataset}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache.with_suffix(cache.suffix + ".lock")
    # Two GPU workers can reach a cold cache simultaneously.  Serialize the
    # expensive raw-table build, then publish the tensor atomically so later
    # workers either see no cache or one complete cache, never a partial write.
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if cache.exists():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            return _RawClinicalBatch(**stored)
        # Experiment seed controls only patient-role allocation/model fitting.
        # The raw cohort remains fixed across replications via cohort_seed.
        raw = builders[config.data.dataset](config, limit, cohort_seed)
        _atomic_torch_save(raw.__dict__, cache)
        return raw


def _atomic_torch_save(value: object, destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _predictor_rows(raw: _RawClinicalBatch, seed: int) -> torch.Tensor:
    placeholder = TrajectoryBatch(
        raw.states,
        torch.zeros(raw.states.shape[:2][0], raw.states.shape[1] - 1, dtype=torch.long),
        raw.outcomes,
        raw.patient_ids,
    )
    roles = patient_level_splits(placeholder, seed=seed, include_environment=True)
    return torch.isin(raw.patient_ids, roles.predictor.patient_ids).nonzero().squeeze(1)


def _discretize_actions(
    raw: _RawClinicalBatch, predictor_rows: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, ...], dict[int, int]]:
    if raw.direct_actions is not None:
        actions = raw.direct_actions.to(torch.long)
        # CXR and INSPIRE are pre-coarsened once to stable three-level
        # ontologies before any patient role is drawn. The D_pred check below
        # validates support but never changes either ontology seed-by-seed.
        expected_actions = raw.direct_action_count or 4
        return _merge_rare_actions(actions, predictor_rows, expected_actions=expected_actions, direct=True)
    reference = raw.treatments[predictor_rows]
    levels = []
    for component in range(raw.treatments.shape[-1]):
        values = reference[..., component]
        positive = values[values > 0]
        if len(positive) == 0:
            levels.append(torch.zeros_like(raw.treatments[..., component], dtype=torch.long))
            continue
        median = positive.median()
        all_values = raw.treatments[..., component]
        levels.append(torch.where(all_values <= 0, 0, torch.where(all_values <= median, 1, 2)))
    if len(levels) != 2:
        raise ValueError("continuous treatment tasks require fluid and pressor components")
    return _merge_rare_actions(levels[0] * 3 + levels[1], predictor_rows, expected_actions=9, direct=False)


def _merge_rare_actions(
    actions: torch.Tensor, predictor_rows: torch.Tensor, *, expected_actions: int, direct: bool
) -> tuple[torch.Tensor, tuple[int, ...], dict[int, int]]:
    """Merge action cells below 2% prevalence using D_pred only, then relabel."""

    counts = torch.bincount(actions[predictor_rows].reshape(-1), minlength=expected_actions)
    minimum = max(1, int(np.ceil(0.02 * actions[predictor_rows].numel())))
    stable = (counts >= minimum).nonzero().squeeze(1).tolist()
    if 0 not in stable:
        stable.insert(0, 0)
    global_counts = torch.bincount(actions.reshape(-1), minlength=expected_actions)
    if direct:
        # Direct clinical-action ontologies are fixed before D_pred.  If any
        # active nonzero cell has inadequate D_pred support, fail clearly
        # rather than silently changing a respiratory or treatment category
        # for this seed.  INSPIRE's rare raw combined cell is coarsened once,
        # before this function, into its prespecified three-level ontology.
        unsupported = [
            action
            for action in range(1, expected_actions)
            if global_counts[action] > 0 and action not in stable
        ]
        if unsupported:
            raise RuntimeError(
                "D_pred has insufficient support for direct clinical actions "
                f"{unsupported}; increase max_patients or use a prespecified coarser action definition"
            )
    locations = _action_locations(expected_actions, direct=direct)
    remap = {}
    for action in range(expected_actions):
        candidates = stable
        if action != 0 and not direct:
            treated = [candidate for candidate in stable if candidate != 0]
            if not treated:
                raise RuntimeError("D_pred contains no adequately supported nonzero treatment action")
            candidates = treated
        remap[action] = min(
            candidates,
            key=lambda candidate: (
                abs(locations[action][0] - locations[candidate][0]) + abs(locations[action][1] - locations[candidate][1]),
                -int(counts[candidate].item()),
                candidate,
            ),
        )
    remapped = torch.empty_like(actions)
    for new_label, original in enumerate(stable):
        source = torch.tensor([key for key, value in remap.items() if value == original], device=actions.device)
        remapped[torch.isin(actions, source)] = new_label
    mapping = {original: new_label for original, target in remap.items() for new_label, stable_action in enumerate(stable) if target == stable_action}
    return remapped, tuple(stable), mapping


def _coarsen_cxr_actions(raw: _RawClinicalBatch) -> _RawClinicalBatch:
    """Merge conventional oxygen and HFNC/NIV before patient-role splitting."""

    if raw.direct_actions is None:
        raise RuntimeError("MIMIC-CXR is missing respiratory-support actions")
    if raw.direct_action_count == 3:
        return raw
    if raw.direct_action_count != 4:
        raise RuntimeError("MIMIC-CXR requires the raw four-level respiratory ontology")
    actions = raw.direct_actions.clone()
    actions[raw.direct_actions == 2] = 1
    actions[raw.direct_actions == 3] = 2
    return replace(
        raw,
        direct_actions=actions,
        direct_action_count=3,
        original_to_direct_action={0: 0, 1: 1, 2: 1, 3: 2},
    )


def _action_locations(expected_actions: int, *, direct: bool) -> dict[int, tuple[int, int]]:
    if direct:
        return {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    return {action: (action // 3, action % 3) for action in range(expected_actions)}


def _build_mimic_iv(config: ExperimentConfig, limit: int, seed: int) -> _RawClinicalBatch:
    root = Path(config.data.data_root) / "mimic-iv-3.1"
    interval = 4 * 60
    cohort = _mimic_cohort(root, horizon=config.horizon, interval_minutes=interval, limit=limit, seed=seed)
    intime = cohort.set_index("stay_id").intime
    vitals, devices = _mimic_chart_events(root, cohort, intime, include_device=True)
    labs = _mimic_lab_events(root, cohort, max_minutes=config.horizon * interval)
    urine = _mimic_urine_events(
        root,
        cohort,
        intime,
        max_minutes=config.horizon * interval,
    )
    treatments = _mimic_treatment_events(root, cohort, intime)
    static = _static_context(
        cohort.set_index("stay_id"),
        numeric=("age", "sex", "admit_to_icu_hours"),
        categorical=("admission_type", "admission_location", "first_careunit"),
        defaults={"age": 65.0, "sex": 0.0, "admit_to_icu_hours": 0.0},
    )
    return _assemble_raw(
        cohort,
        id_column="stay_id",
        patient_column="subject_id",
        static=static,
        vitals=pd.concat(
            (vitals, labs, urine, _respiratory_device_state_events(devices)),
            ignore_index=True,
        ),
        treatments=treatments,
        horizon=config.horizon,
        interval_minutes=interval,
        action_minutes=2 * 60,
        outcome_kind="hypo_tachy",
    )


def _build_eicu(config: ExperimentConfig, limit: int, seed: int) -> _RawClinicalBatch:
    root = Path(config.data.data_root) / "eICU"
    interval = 4 * 60
    patients = pd.read_csv(
        root / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "uniquepid",
            "age",
            "gender",
            "ethnicity",
            "hospitalid",
            "hospitaladmitoffset",
            "hospitaladmitsource",
            "unitadmitsource",
            "unittype",
            "unitstaytype",
            "admissionheight",
            "admissionweight",
            "unitdischargeoffset",
        ],
    )
    age = pd.to_numeric(patients.age.replace("> 89", 90), errors="coerce")
    cohort = patients[(age >= 18) & (patients.unitdischargeoffset >= config.horizon * interval)].copy()
    cohort["age"] = age.loc[cohort.index].fillna(65.0)
    cohort["sex"] = (cohort.gender.astype(str).str.lower() == "male").astype(float)
    cohort["hospital_pre_icu_hours"] = (
        -pd.to_numeric(cohort.hospitaladmitoffset, errors="coerce").clip(upper=0) / 60.0
    )
    cohort = _sample_cohort(cohort, "patientunitstayid", limit, seed)
    ids = set(cohort.patientunitstayid.astype(int))
    vitals = _eicu_vital_events(root, ids)
    labs = _eicu_lab_events(root, ids, max_minutes=config.horizon * interval)
    treatments, urine = _eicu_treatment_events(root, ids)
    respiratory = _eicu_respiratory_events(root, ids)
    static = _static_context(
        cohort.set_index("patientunitstayid"),
        numeric=(
            "age",
            "sex",
            "admissionheight",
            "admissionweight",
            "hospital_pre_icu_hours",
        ),
        categorical=(
            "ethnicity",
            "hospitaladmitsource",
            "unitadmitsource",
            "unittype",
            "unitstaytype",
        ),
        defaults={
            "age": 65.0,
            "sex": 0.0,
            "admissionheight": 170.0,
            "admissionweight": 75.0,
            "hospital_pre_icu_hours": 0.0,
        },
    )
    cohort = cohort.rename(columns={"patientunitstayid": "episode_id", "uniquepid": "patient_key"})
    cohort["patient_id"] = pd.factorize(cohort.patient_key)[0].astype(np.int64)
    cohort = cohort.rename(columns={"episode_id": "stay_id"})
    vitals = vitals.rename(columns={"patientunitstayid": "stay_id"})
    labs = labs.rename(columns={"patientunitstayid": "stay_id"})
    urine = urine.rename(columns={"patientunitstayid": "stay_id"})
    respiratory = respiratory.rename(columns={"patientunitstayid": "stay_id"})
    treatments = treatments.rename(columns={"patientunitstayid": "stay_id"})
    static.index.name = "stay_id"
    return _assemble_raw(
        cohort,
        id_column="stay_id",
        patient_column="patient_id",
        static=static,
        vitals=pd.concat((vitals, labs, urine, respiratory), ignore_index=True),
        treatments=treatments,
        horizon=config.horizon,
        interval_minutes=interval,
        action_minutes=2 * 60,
        outcome_kind="hypo_tachy",
    )


def _build_mimic_cxr(config: ExperimentConfig, limit: int, seed: int) -> _RawClinicalBatch:
    root = Path(config.data.data_root)
    mimic = root / "mimic-iv-3.1"
    interval = 6 * 60
    cohort = _mimic_cxr_cohort(root, horizon=config.horizon, interval_minutes=interval, limit=limit * 3, seed=seed)
    cohort, paths, labels, official_train = _index_cxr_rows(
        root / "MIMIC-CXR", cohort, minimum_duration_minutes=config.horizon * interval
    )
    cohort = _sample_cohort(cohort, "stay_id", limit, seed)
    paths = paths.loc[cohort.stay_id]
    labels = torch.from_numpy(labels.loc[cohort.stay_id].to_numpy(np.float32))
    official_train = torch.from_numpy(official_train.loc[cohort.stay_id].to_numpy(dtype=bool))
    intime = cohort.set_index("stay_id").intime
    vitals, devices = _mimic_chart_events(mimic, cohort, intime, include_device=True)
    labs = _mimic_lab_events(mimic, cohort, max_minutes=config.horizon * interval)
    urine = _mimic_urine_events(
        mimic,
        cohort,
        intime,
        max_minutes=config.horizon * interval,
    )
    direct_actions = _respiratory_action_grid(devices, cohort.stay_id.to_numpy(), config.horizon, interval, 3 * 60)
    static = _static_context(
        cohort.set_index("stay_id"),
        numeric=(
            "age",
            "sex",
            "ed_temperature",
            "ed_heartrate",
            "ed_resprate",
            "ed_o2sat",
            "ed_sbp",
            "ed_dbp",
            "ed_acuity",
            "admit_to_icu_hours",
        ),
        categorical=("admission_type", "admission_location", "first_careunit"),
        defaults={
            "age": 65.0,
            "sex": 0.0,
            "ed_temperature": 37.0,
            "ed_heartrate": 80.0,
            "ed_resprate": 18.0,
            "ed_o2sat": 96.0,
            "ed_sbp": 120.0,
            "ed_dbp": 70.0,
            "ed_acuity": 3.0,
            "admit_to_icu_hours": 0.0,
        },
    )
    raw = _assemble_raw(
        cohort,
        id_column="stay_id",
        patient_column="subject_id",
        static=static,
        vitals=pd.concat(
            (vitals, labs, urine, _respiratory_device_state_events(devices)),
            ignore_index=True,
        ),
        treatments=pd.DataFrame(columns=["stay_id", "minutes", "component", "value"]),
        horizon=config.horizon,
        interval_minutes=interval,
        action_minutes=3 * 60,
        outcome_kind="hypox_tachyp",
        direct_actions=direct_actions,
    )
    return _RawClinicalBatch(
        states=raw.states,
        outcomes=raw.outcomes,
        treatments=raw.treatments,
        patient_ids=raw.patient_ids,
        episode_ids=raw.episode_ids,
        static_indices=raw.static_indices,
        direct_actions=raw.direct_actions,
        direct_action_count=4,
        original_to_direct_action={0: 0, 1: 1, 2: 2, 3: 3},
        cxr_paths=tuple(paths.iloc[index] for index in _episode_row_indices(raw.episode_ids, cohort.stay_id.to_numpy()).tolist()),
        cxr_labels=labels[_episode_row_indices(raw.episode_ids, cohort.stay_id.to_numpy())],
        cxr_train_mask=official_train[_episode_row_indices(raw.episode_ids, cohort.stay_id.to_numpy())],
        state_feature_names=raw.state_feature_names,
    )


def _build_inspire(config: ExperimentConfig, limit: int, seed: int) -> _RawClinicalBatch:
    root = Path(config.data.data_root) / "INSPIRE/inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2"
    interval = 10
    operations = pd.read_csv(
        root / "operations.csv.gz",
        usecols=[
            "op_id",
            "subject_id",
            "age",
            "sex",
            "weight",
            "height",
            "race",
            "asa",
            "emop",
            "department",
            "antype",
            "anstart_time",
            "anend_time",
            "cpbon_time",
        ],
    )
    operations = operations[
        (operations.age >= 18)
        & (operations.anend_time >= operations.anstart_time + config.horizon * interval)
        & operations.cpbon_time.isna()
    ].copy()
    operations = operations.sort_values("anstart_time").drop_duplicates("subject_id")
    operations["sex"] = (operations.sex.astype(str).str.upper() == "M").astype(float)
    operations["asa"] = pd.to_numeric(operations.asa, errors="coerce").fillna(2.0)
    operations["emop"] = pd.to_numeric(operations.emop, errors="coerce").fillna(0.0)
    operations = _sample_cohort(operations, "op_id", limit, seed)
    ids = set(operations.op_id.astype(int))
    vitals, treatments = _inspire_events(root, operations, ids)
    labs = _inspire_lab_events(root, operations, max_minutes=config.horizon * interval)
    static = _static_context(
        operations.set_index("op_id"),
        numeric=("age", "sex", "weight", "height", "asa", "emop"),
        categorical=("race", "department", "antype"),
        defaults={
            "age": 60.0,
            "sex": 0.0,
            "weight": 70.0,
            "height": 165.0,
            "asa": 2.0,
            "emop": 0.0,
        },
    )
    raw = _assemble_raw(
        operations,
        id_column="op_id",
        patient_column="subject_id",
        static=static,
        vitals=pd.concat((vitals, labs), ignore_index=True).rename(columns={"op_id": "stay_id"}),
        treatments=treatments.rename(columns={"op_id": "stay_id"}),
        horizon=config.horizon,
        interval_minutes=interval,
        action_minutes=5,
        outcome_kind="hypo_hyper",
        canonical_id_column="stay_id",
    )
    # The raw four cells are clinically natural, but pressor+fluid occurs in
    # only about 0.2% of the prespecified action windows in the complete local
    # cohort.  Fix the low-frequency coarsening *before* D_pred is drawn so
    # every seed shares one interpretable ontology:
    #   0 none, 1 fluid-only, 2 vasopressor-containing (pressor or combined).
    raw_four = (raw.treatments[..., 1] > 0).to(torch.long) + 2 * (raw.treatments[..., 0] > 0).to(torch.long)
    direct = torch.where(raw_four == 0, 0, torch.where(raw_four == 2, 1, 2))
    return _RawClinicalBatch(
        states=raw.states,
        outcomes=raw.outcomes,
        treatments=raw.treatments,
        patient_ids=raw.patient_ids,
        episode_ids=raw.episode_ids,
        static_indices=raw.static_indices,
        direct_actions=direct,
        direct_action_count=3,
        original_to_direct_action={0: 0, 1: 2, 2: 1, 3: 2},
        state_feature_names=raw.state_feature_names,
    )


def _mimic_cohort(root: Path, *, horizon: int, interval_minutes: int, limit: int, seed: int) -> pd.DataFrame:
    stays = pd.read_csv(
        root / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    patients = pd.read_csv(root / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(
        root / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "admission_type", "admission_location"],
        parse_dates=["admittime"],
    )
    cohort = (
        stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients, on="subject_id", how="inner")
        .sort_values("intime")
        .drop_duplicates("subject_id")
    )
    cohort["age"] = cohort.anchor_age + cohort.intime.dt.year - cohort.anchor_year
    cohort["sex"] = (cohort.gender == "M").astype(float)
    cohort["admit_to_icu_hours"] = (cohort.intime - cohort.admittime).dt.total_seconds() / 3600.0
    duration = (cohort.outtime - cohort.intime).dt.total_seconds() / 60
    cohort = cohort[(cohort.age >= 18) & (duration >= horizon * interval_minutes)]
    return _sample_cohort(cohort, "stay_id", limit, seed)


def _mimic_cxr_cohort(
    root: Path, *, horizon: int, interval_minutes: int, limit: int, seed: int
) -> pd.DataFrame:
    """ED-to-ICU cohort with ED triage context for the multimodal task."""

    mimic = root / "mimic-iv-3.1"
    ed_root = root / "mimic-iv-ed-2.2" / "ed"
    stays = pd.read_csv(
        mimic / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    patients = pd.read_csv(
        mimic / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
    )
    ed = pd.read_csv(
        ed_root / "edstays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    ).rename(columns={"stay_id": "ed_stay_id", "intime": "ed_intime", "outtime": "ed_outtime"})
    admissions = pd.read_csv(
        mimic / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "admission_type", "admission_location"],
        parse_dates=["admittime"],
    )
    cohort = (
        stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(ed, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients, on="subject_id", how="inner")
    )
    gap_minutes = (cohort.intime - cohort.ed_outtime).dt.total_seconds() / 60.0
    duration_minutes = (cohort.outtime - cohort.intime).dt.total_seconds() / 60.0
    cohort["age"] = cohort.anchor_age + cohort.intime.dt.year - cohort.anchor_year
    cohort["sex"] = (cohort.gender == "M").astype(float)
    cohort["admit_to_icu_hours"] = (cohort.intime - cohort.admittime).dt.total_seconds() / 3600.0
    cohort = cohort[
        (cohort.age >= 18)
        & duration_minutes.ge(horizon * interval_minutes)
        & gap_minutes.between(0.0, 24.0 * 60.0)
    ].copy()
    # Use the most recent ED stay preceding the ICU transfer and one ICU stay
    # per patient.  This keeps all patient-level split roles disjoint.
    cohort = cohort.sort_values(["subject_id", "intime", "ed_outtime"], ascending=[True, True, False]).drop_duplicates("subject_id")
    triage = pd.read_csv(
        ed_root / "triage.csv.gz",
        usecols=["stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"],
    ).rename(columns={"stay_id": "ed_stay_id"})
    triage = triage.rename(columns={column: f"ed_{column}" for column in triage.columns if column != "ed_stay_id"})
    cohort = cohort.merge(triage, on="ed_stay_id", how="left")
    defaults = {
        "ed_temperature": 37.0,
        "ed_heartrate": 80.0,
        "ed_resprate": 18.0,
        "ed_o2sat": 96.0,
        "ed_sbp": 120.0,
        "ed_dbp": 70.0,
        "ed_acuity": 3.0,
    }
    for column, default in defaults.items():
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").fillna(default)
    return _sample_cohort(cohort, "stay_id", limit, seed)


def _mimic_chart_events(
    root: Path, cohort: pd.DataFrame, intime: pd.Series, *, include_device: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = set(cohort.stay_id.astype(int))
    wanted = set(MIMIC_VITAL_KIND) | ({MIMIC_RESPIRATORY_DEVICE} if include_device else set())
    vital_parts, device_parts = [], []
    columns = ["stay_id", "charttime", "itemid", "valuenum", "value"]
    for chunk in pd.read_csv(root / "icu/chartevents.csv.gz", usecols=columns, chunksize=1_000_000, low_memory=False):
        selected = chunk[chunk.stay_id.isin(ids) & chunk.itemid.isin(wanted)].copy()
        if selected.empty:
            continue
        selected["minutes"] = (
            pd.to_datetime(selected.charttime) - pd.to_datetime(selected.stay_id.map(intime))
        ).dt.total_seconds() / 60
        numeric = selected[selected.itemid.isin(MIMIC_VITAL_KIND) & selected.valuenum.notna()].copy()
        if not numeric.empty:
            numeric["kind"] = numeric.itemid.map(MIMIC_VITAL_KIND)
            fahrenheit = numeric.kind.eq("temperature_fahrenheit")
            numeric.loc[fahrenheit, "valuenum"] = (
                (numeric.loc[fahrenheit, "valuenum"] - 32.0) * 5.0 / 9.0
            )
            numeric.loc[fahrenheit, "kind"] = "temperature"
            vital_parts.append(numeric[["stay_id", "minutes", "kind", "valuenum"]].rename(columns={"valuenum": "value"}))
        device = selected[selected.itemid == MIMIC_RESPIRATORY_DEVICE].copy()
        if not device.empty:
            device_parts.append(device[["stay_id", "minutes", "value"]])
    if not vital_parts:
        raise RuntimeError("no selected MIMIC vital events were found")
    return pd.concat(vital_parts, ignore_index=True), (
        pd.concat(device_parts, ignore_index=True) if device_parts else pd.DataFrame(columns=["stay_id", "minutes", "value"])
    )


def _mimic_lab_events(root: Path, cohort: pd.DataFrame, *, max_minutes: int) -> pd.DataFrame:
    """Return selected blood laboratories after the ICU/CXR decision origin.

    MIMIC hospital labs are keyed by admission rather than ICU stay.  We map
    each selected admission back to its one retained stay, then retain only
    timestamps in the experimental horizon.  Consequently a lab can enter
    only the response half-window that precedes the *next* decision state.
    """

    events_path = root / "hosp/labevents.csv.gz"
    dictionary_path = root / "hosp/d_labitems.csv.gz"
    if not events_path.is_file() or not dictionary_path.is_file():
        return pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    item_kinds = _mimic_lab_item_kinds(root)
    if not item_kinds:
        return pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    retained = cohort.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id").set_index("hadm_id")
    hadm_ids = set(retained.index.astype(int))
    stay_for_hadm = retained.stay_id
    origin_for_hadm = retained.intime
    parts = []
    for chunk in pd.read_csv(
        events_path,
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[chunk.hadm_id.isin(hadm_ids) & chunk.itemid.isin(item_kinds)].copy()
        if selected.empty:
            continue
        selected["minutes"] = (
            pd.to_datetime(selected.charttime) - pd.to_datetime(selected.hadm_id.map(origin_for_hadm))
        ).dt.total_seconds() / 60.0
        selected = selected[selected.minutes.between(0.0, float(max_minutes), inclusive="left")]
        selected = selected[selected.valuenum.notna()].copy()
        if selected.empty:
            continue
        selected["stay_id"] = selected.hadm_id.map(stay_for_hadm)
        selected["kind"] = selected.itemid.map(item_kinds)
        parts.append(selected[["stay_id", "minutes", "kind", "valuenum"]].rename(columns={"valuenum": "value"}))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    )


def _mimic_urine_events(
    root: Path,
    cohort: pd.DataFrame,
    intime: pd.Series,
    *,
    max_minutes: int,
) -> pd.DataFrame:
    """Read explicit urine-output channels, excluding GU irrigation output."""

    path = root / "icu/outputevents.csv.gz"
    if not path.is_file():
        return pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    ids = set(cohort.stay_id.astype(int))
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=["stay_id", "charttime", "itemid", "value"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[
            chunk.stay_id.isin(ids) & chunk.itemid.isin(MIMIC_URINE_OUTPUTS)
        ].copy()
        if selected.empty:
            continue
        selected["minutes"] = (
            pd.to_datetime(selected.charttime)
            - pd.to_datetime(selected.stay_id.map(intime))
        ).dt.total_seconds() / 60.0
        selected["value"] = pd.to_numeric(selected.value, errors="coerce")
        selected = selected[
            selected.minutes.between(0.0, float(max_minutes), inclusive="left")
            & selected.value.notna()
            & selected.value.ge(0.0)
        ].copy()
        if selected.empty:
            continue
        selected["kind"] = "urine_output"
        parts.append(selected[["stay_id", "minutes", "kind", "value"]])
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    )


def _respiratory_support_levels(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.lower()
    return np.select(
        [
            text.str.contains("endotracheal|ventilator|trach", regex=True),
            text.str.contains("high flow|hfnc|bipap|cpap|non-invasive|niv", regex=True),
            text.str.contains("nasal|mask|face|non-rebreather|aerosol", regex=True),
        ],
        [3, 2, 1],
        default=0,
    )


def _respiratory_device_state_events(devices: pd.DataFrame) -> pd.DataFrame:
    """Represent charted respiratory support as a 0--3 state covariate."""

    if devices.empty:
        return pd.DataFrame(columns=["stay_id", "minutes", "kind", "value"])
    events = devices[["stay_id", "minutes", "value"]].copy()
    events["value"] = _respiratory_support_levels(events.value).astype(np.float32)
    events["kind"] = "ventilation"
    return events[["stay_id", "minutes", "kind", "value"]]


def _mimic_treatment_events(root: Path, cohort: pd.DataFrame, intime: pd.Series) -> pd.DataFrame:
    ids = set(cohort.stay_id.astype(int))
    parts = []
    columns = ["stay_id", "starttime", "endtime", "itemid", "amount", "rate"]
    for chunk in pd.read_csv(root / "icu/inputevents.csv.gz", usecols=columns, chunksize=1_000_000, low_memory=False):
        selected = chunk[chunk.stay_id.isin(ids)].copy()
        if selected.empty:
            continue
        selected["minutes"] = (
            pd.to_datetime(selected.starttime) - pd.to_datetime(selected.stay_id.map(intime))
        ).dt.total_seconds() / 60
        selected["end_minutes"] = (
            pd.to_datetime(selected.endtime) - pd.to_datetime(selected.stay_id.map(intime))
        ).dt.total_seconds() / 60
        selected["end_minutes"] = selected.end_minutes.where(
            selected.end_minutes > selected.minutes, selected.minutes
        )
        pressor = selected[selected.itemid.isin(MIMIC_PRESSORS)].copy()
        if not pressor.empty:
            # Rate units differ materially across vasoactive agents.  Use
            # duration-weighted active-infusion exposure rather than summing
            # incomparable mcg/kg/min and units/hour rates.
            pressor["value"] = 1.0
            pressor["component"] = "pressor"
            pressor["interval_kind"] = "active_duration"
            parts.append(pressor[["stay_id", "minutes", "end_minutes", "component", "value", "interval_kind"]])
        fluid = selected[selected.itemid.isin(MIMIC_IV_FLUIDS)].copy()
        if not fluid.empty:
            fluid["value"] = pd.to_numeric(fluid.amount, errors="coerce").abs().fillna(0.0)
            fluid["component"] = "fluid"
            fluid["interval_kind"] = "amount"
            parts.append(fluid[["stay_id", "minutes", "end_minutes", "component", "value", "interval_kind"]])
    if not parts:
        return pd.DataFrame(columns=["stay_id", "minutes", "component", "value"])
    return pd.concat(parts, ignore_index=True)


def _eicu_vital_events(root: Path, ids: set[int]) -> pd.DataFrame:
    parts = []
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
    ]
    mapping = {
        "temperature": "temperature",
        "sao2": "spo2",
        "heartrate": "hr",
        "respiration": "rr",
        "systemicsystolic": "sbp",
        "systemicdiastolic": "dbp",
        "systemicmean": "map",
    }
    for chunk in pd.read_csv(root / "vitalPeriodic.csv.gz", usecols=usecols, chunksize=1_000_000, low_memory=False):
        selected = chunk[chunk.patientunitstayid.isin(ids)]
        if selected.empty:
            continue
        long = selected.melt(
            id_vars=["patientunitstayid", "observationoffset"], value_vars=list(mapping), var_name="kind", value_name="value"
        ).dropna(subset=["value"])
        long["kind"] = long.kind.map(mapping)
        fahrenheit = long.kind.eq("temperature") & pd.to_numeric(
            long.value, errors="coerce"
        ).gt(60.0)
        long.loc[fahrenheit, "value"] = (
            pd.to_numeric(long.loc[fahrenheit, "value"], errors="coerce") - 32.0
        ) * 5.0 / 9.0
        long = long.rename(columns={"observationoffset": "minutes"})
        parts.append(long[["patientunitstayid", "minutes", "kind", "value"]])
    # Periodic arterial/systemic pressure is not recorded for every eICU stay.
    # Use non-invasive aperiodic SBP/DBP as a second source, still timestamped
    # before the response-to-next-state alignment in ``_assemble_raw``.
    aperiodic_path = root / "vitalAperiodic.csv.gz"
    if aperiodic_path.is_file():
        aperiodic_columns = [
            "patientunitstayid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
        ]
        aperiodic_mapping = {"noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp"}
        for chunk in pd.read_csv(
            aperiodic_path,
            usecols=aperiodic_columns,
            chunksize=1_000_000,
            low_memory=False,
        ):
            selected = chunk[chunk.patientunitstayid.isin(ids)]
            if selected.empty:
                continue
            long = selected.melt(
                id_vars=["patientunitstayid", "observationoffset"],
                value_vars=list(aperiodic_mapping),
                var_name="kind",
                value_name="value",
            ).dropna(subset=["value"])
            long["kind"] = long.kind.map(aperiodic_mapping)
            long = long.rename(columns={"observationoffset": "minutes"})
            parts.append(long[["patientunitstayid", "minutes", "kind", "value"]])
    if not parts:
        raise RuntimeError("no selected eICU vital events were found")
    return pd.concat(parts, ignore_index=True)


def _eicu_lab_events(root: Path, ids: set[int], *, max_minutes: int) -> pd.DataFrame:
    """Read available eICU blood-laboratory measurements for the task horizon."""

    path = root / "lab.csv.gz"
    if not path.is_file():
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[chunk.patientunitstayid.isin(ids)].copy()
        if selected.empty:
            continue
        selected["kind"] = _clinical_lab_kind(selected.labname)
        selected["value"] = pd.to_numeric(selected.labresult, errors="coerce")
        selected = selected[
            selected.kind.notna()
            & selected.value.notna()
            & pd.to_numeric(selected.labresultoffset, errors="coerce").between(0.0, float(max_minutes), inclusive="left")
        ].copy()
        if selected.empty:
            continue
        selected["minutes"] = pd.to_numeric(selected.labresultoffset, errors="coerce")
        parts.append(selected[["patientunitstayid", "minutes", "kind", "value"]])
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    )


def _eicu_treatment_events(
    root: Path, ids: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressor_parts = []
    pressor_pattern = "norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine|dobutamine"
    for chunk in pd.read_csv(
        root / "infusionDrug.csv.gz",
        usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate", "drugamount"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[chunk.patientunitstayid.isin(ids)]
        selected = selected[selected.drugname.astype(str).str.contains(pressor_pattern, case=False, regex=True)].copy()
        if selected.empty:
            continue
        # eICU stores vasoactive rates in heterogeneous drug-specific units.
        # Count active infusion chart events rather than summing incompatible
        # raw rates; low/high discretization is then learned on D_pred only.
        selected["value"] = 1.0
        selected["component"] = "pressor"
        pressor_parts.append(selected.rename(columns={"infusionoffset": "minutes"})[["patientunitstayid", "minutes", "component", "value"]])
    fluid_parts, urine_parts = [], []
    for chunk in pd.read_csv(
        root / "intakeOutput.csv.gz",
        usecols=[
            "patientunitstayid",
            "intakeoutputoffset",
            "cellpath",
            "celllabel",
            "cellvaluenumeric",
        ],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = _eicu_fluid_rows(chunk, ids)
        if not selected.empty:
            fluid_parts.append(selected)
        urine = _eicu_urine_rows(chunk, ids)
        if not urine.empty:
            urine_parts.append(urine)
    treatments = (
        pd.concat(pressor_parts + fluid_parts, ignore_index=True)
        if (pressor_parts or fluid_parts)
        else pd.DataFrame(columns=["patientunitstayid", "minutes", "component", "value"])
    )
    urine = (
        pd.concat(urine_parts, ignore_index=True)
        if urine_parts
        else pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    )
    return treatments, urine


def _eicu_fluid_rows(chunk: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    """Select explicit IV crystalloid/colloid volumes from eICU I/O rows.

    ``intaketotal`` mixes oral intake, enteral feeds, medication carriers and
    IV fluid and is therefore not a treatment variable.  The cell-level path
    and amount provide a narrower, auditable definition.  Medication IVPBs
    and line flushes are excluded even when eICU stores them under the broad
    ``Crystalloids`` branch.
    """

    selected = chunk[chunk.patientunitstayid.isin(ids)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "component", "value"])
    path = selected.cellpath.astype("string").str.lower()
    label = selected.celllabel.astype("string").str.lower()
    in_iv_branch = path.str.contains(
        r"\|intake \(ml\)\|(?:crystalloids|colloids) \(ml\)\|",
        regex=True,
        na=False,
    )
    fluid_name = label.str.contains(
        r"normal saline|sodium chloride|lactated ring|ringer|plasma[- ]?lyte|"
        r"dextrose (?:5|10) ?%|albumin|dextran|hetastarch",
        regex=True,
        na=False,
    )
    excluded = label.str.contains(
        r"flush|ivpb|with kcl|potassium|magnesium|calcium|bicarbonate|"
        r"anticoagulant|citrate|mannitol|propofol",
        regex=True,
        na=False,
    )
    selected["value"] = pd.to_numeric(selected.cellvaluenumeric, errors="coerce")
    selected = selected[in_iv_branch & fluid_name & ~excluded & selected.value.gt(0.0)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "component", "value"])
    selected["component"] = "fluid"
    return selected.rename(columns={"intakeoutputoffset": "minutes"})[
        ["patientunitstayid", "minutes", "component", "value"]
    ]


def _eicu_urine_rows(chunk: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    """Select explicit urine-output cells for next-state context."""

    selected = chunk[chunk.patientunitstayid.isin(ids)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    path = selected.cellpath.astype("string").str.lower()
    label = selected.celllabel.astype("string").str.lower().str.strip()
    selected["value"] = pd.to_numeric(selected.cellvaluenumeric, errors="coerce")
    selected = selected[
        path.str.contains(r"\|output \(ml\)\|urine(?:\||$)", regex=True, na=False)
        & label.eq("urine")
        & selected.value.ge(0.0)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    selected["kind"] = "urine_output"
    return selected.rename(columns={"intakeoutputoffset": "minutes"})[
        ["patientunitstayid", "minutes", "kind", "value"]
    ]


def _eicu_respiratory_events(root: Path, ids: set[int]) -> pd.DataFrame:
    """Read charted support and ventilator settings as dynamic state events."""

    path = root / "respiratoryCharting.csv.gz"
    if not path.is_file():
        return pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    label_to_kind = {
        "fio2": "fio2",
        "peep": "peep",
        "peep/cpap": "peep",
        "tidal volume (set)": "tidal_volume",
        "tidal volume observed (vt)": "tidal_volume",
    }
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=[
            "patientunitstayid",
            "respchartoffset",
            "respchartvaluelabel",
            "respchartvalue",
        ],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[chunk.patientunitstayid.isin(ids)].copy()
        if selected.empty:
            continue
        labels = selected.respchartvaluelabel.astype("string").str.lower().str.strip()
        numeric = selected[labels.isin(label_to_kind)].copy()
        if not numeric.empty:
            numeric_labels = numeric.respchartvaluelabel.astype("string").str.lower().str.strip()
            numeric["kind"] = numeric_labels.map(label_to_kind)
            numeric["value"] = pd.to_numeric(numeric.respchartvalue, errors="coerce")
            numeric = numeric.dropna(subset=["value"])
            parts.append(
                numeric.rename(columns={"respchartoffset": "minutes"})[
                    ["patientunitstayid", "minutes", "kind", "value"]
                ]
            )
        support = selected[labels.eq("rt vent on/off")].copy()
        if not support.empty:
            status = support.respchartvalue.astype("string").str.lower().str.strip()
            support["value"] = (~status.str.contains("off|discontinu|stop", regex=True, na=False)).astype(np.float32)
            support["kind"] = "ventilation"
            parts.append(
                support.rename(columns={"respchartoffset": "minutes"})[
                    ["patientunitstayid", "minutes", "kind", "value"]
                ]
            )
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["patientunitstayid", "minutes", "kind", "value"])
    )


def _inspire_events(root: Path, operations: pd.DataFrame, ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = operations.set_index("op_id").anstart_time
    vital_parts, treatment_parts = [], []
    vital_names = {
        "art_mbp": "map",
        "nibp_mbp": "map",
        "art_sbp": "sbp",
        "nibp_sbp": "sbp",
        "art_dbp": "dbp",
        "nibp_dbp": "dbp",
        "bt": "temperature",
        "hr": "hr",
        "rr": "rr",
        "spo2": "spo2",
        "etco2": "etco2",
        "fio2": "fio2",
        "peep": "peep",
        "vt": "tidal_volume",
        "minvol": "minute_ventilation",
        "uo": "urine_output",
        "ebl": "blood_loss",
    }
    # INSPIRE uses compact medication codes as well as expanded labels.  The
    # short forms below cover the common norepinephrine/epinephrine/
    # phenylephrine/vasopressin records without relabelling unrelated drugs.
    pressor_names = {
        "nepi", "norepi", "norepinephrine",
        "epii", "epi", "epinephrine",
        "phe", "phenylephrine", "eph", "ephedrine",
        "vaso", "vasopressin",
        "dopai", "dopamine",
        "dobui", "dobutamine",
    }
    fluid_names = {
        "ns", "hs", "hes", "alb20", "alb5", "d5w", "d10w",
        "rbc", "ffp", "cryo", "pc",
    }
    for chunk in pd.read_csv(root / "vitals.csv.gz", usecols=["op_id", "chart_time", "item_name", "value"], chunksize=1_000_000, low_memory=False):
        selected = chunk[chunk.op_id.isin(ids)].copy()
        if selected.empty:
            continue
        selected["minutes"] = pd.to_numeric(selected.chart_time, errors="coerce") - selected.op_id.map(start)
        selected["item_name"] = selected.item_name.astype(str).str.lower()
        selected["value"] = pd.to_numeric(selected.value, errors="coerce")
        vital = selected[selected.item_name.isin(vital_names) & selected.value.notna()].copy()
        if not vital.empty:
            vital["kind"] = vital.item_name.map(vital_names)
            vital_parts.append(vital[["op_id", "minutes", "kind", "value"]])
        pressor = selected[selected.item_name.isin(pressor_names)].copy()
        if not pressor.empty:
            pressor["component"] = "pressor"
            pressor["value"] = pressor.value.abs().fillna(1.0)
            treatment_parts.append(pressor[["op_id", "minutes", "component", "value"]])
        fluid = selected[selected.item_name.isin(fluid_names)].copy()
        if not fluid.empty:
            fluid["component"] = "fluid"
            fluid["value"] = fluid.value.abs().fillna(1.0)
            treatment_parts.append(fluid[["op_id", "minutes", "component", "value"]])
    if not vital_parts:
        raise RuntimeError("no selected INSPIRE vital events were found")
    return pd.concat(vital_parts, ignore_index=True), (
        pd.concat(treatment_parts, ignore_index=True) if treatment_parts else pd.DataFrame(columns=["op_id", "minutes", "component", "value"])
    )


def _inspire_lab_events(root: Path, operations: pd.DataFrame, *, max_minutes: int) -> pd.DataFrame:
    """Map perioperative laboratory timestamps onto the retained operation IDs."""

    path = root / "labs.csv.gz"
    if not path.is_file():
        return pd.DataFrame(columns=["op_id", "minutes", "kind", "value"])
    retained = operations.drop_duplicates("subject_id").set_index("subject_id")
    subject_ids = set(retained.index.astype(int))
    operation_for_subject = retained.op_id
    start_for_subject = retained.anstart_time
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "chart_time", "item_name", "value"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        selected = chunk[chunk.subject_id.isin(subject_ids)].copy()
        if selected.empty:
            continue
        selected["kind"] = _clinical_lab_kind(selected.item_name)
        selected["value"] = pd.to_numeric(selected.value, errors="coerce")
        selected["minutes"] = (
            pd.to_numeric(selected.chart_time, errors="coerce")
            - pd.to_numeric(selected.subject_id.map(start_for_subject), errors="coerce")
        )
        selected = selected[
            selected.kind.notna()
            & selected.value.notna()
            & selected.minutes.between(0.0, float(max_minutes), inclusive="left")
        ].copy()
        if selected.empty:
            continue
        selected["op_id"] = selected.subject_id.map(operation_for_subject)
        parts.append(selected[["op_id", "minutes", "kind", "value"]])
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["op_id", "minutes", "kind", "value"])
    )


def _index_cxr_rows(
    cxr_root: Path, cohort: pd.DataFrame, *, minimum_duration_minutes: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    metadata = pd.read_csv(cxr_root / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime"])
    metadata = metadata[metadata.subject_id.isin(cohort.subject_id)].copy()
    clock = metadata.StudyTime.fillna(0).astype(float)
    metadata["study_time"] = (
        pd.to_datetime(metadata.StudyDate.astype(str), format="%Y%m%d")
        + pd.to_timedelta((clock // 10000).astype(int), unit="h")
        + pd.to_timedelta(((clock % 10000) // 100).astype(int), unit="m")
        + pd.to_timedelta((clock % 100).astype(int), unit="s")
    )
    candidates = cohort.merge(metadata, on="subject_id", how="inner")
    candidates["offset_hours"] = (candidates.study_time - candidates.intime).dt.total_seconds() / 3600
    # A study may occur up to six hours after ICU admission, but its timestamp
    # then becomes the episode's decision origin below.  Thus every action and
    # response is observed *after* the image is available, without discarding
    # the clinically useful [-2 h, +6 h] index-CXR window.
    candidates = candidates[candidates.offset_hours.between(-2, 6)].copy()
    candidates["distance"] = candidates.offset_hours.abs()
    selected = candidates.sort_values("distance").drop_duplicates("stay_id")
    labels = pd.read_csv(cxr_root / "mimic-cxr-2.0.0-chexpert.csv.gz", usecols=["study_id", *CXR_LABEL_COLUMNS]).set_index("study_id")
    selected = selected.join(labels, on="study_id", how="inner")
    selected = selected.dropna(subset=list(CXR_LABEL_COLUMNS), how="all")
    splits = pd.read_csv(
        cxr_root / "mimic-cxr-2.0.0-split.csv.gz", usecols=["dicom_id", "split"]
    ).set_index("dicom_id")
    selected = selected.join(splits, on="dicom_id", how="inner")
    selected["intime"] = selected[["intime", "study_time"]].max(axis=1)
    remaining_minutes = (selected.outtime - selected.intime).dt.total_seconds() / 60.0
    selected = selected[remaining_minutes >= minimum_duration_minutes].copy()
    paths = selected.set_index("stay_id").apply(
        lambda row: str(cxr_root / "files" / f"p{str(int(row.subject_id))[:2]}" / f"p{int(row.subject_id)}" / f"s{int(row.study_id)}" / f"{row.dicom_id}.jpg"),
        axis=1,
    )
    # Metadata sometimes references an image absent from the local JPG mirror.
    # Filter it here rather than failing later during encoder training, while
    # preserving one-to-one alignment among stays, paths, and CheXpert labels.
    available = paths.map(lambda path: Path(path).is_file())
    paths = paths[available]
    selected = selected[selected.stay_id.isin(paths.index)].copy()
    selected_labels = selected.set_index("stay_id")[list(CXR_LABEL_COLUMNS)].replace(-1.0, 0.5).fillna(0.0)
    official_train = selected.set_index("stay_id").split.eq("train")
    selected_cohort = selected[cohort.columns].copy()
    return selected_cohort, paths, selected_labels, official_train


def _respiratory_action_grid(devices: pd.DataFrame, ids: np.ndarray, horizon: int, interval: int, action_minutes: int) -> torch.Tensor:
    if devices.empty:
        return torch.zeros((len(ids), horizon), dtype=torch.long)
    level = _respiratory_support_levels(devices.value)
    frame = devices.assign(level=level)[["stay_id", "minutes", "level"]].copy()
    frame["minutes"] = pd.to_numeric(frame.minutes, errors="coerce")
    frame = frame.dropna(subset=["minutes"]).sort_values(["stay_id", "minutes"])
    # Treat respiratory-device charting as a stateful clinical setting rather
    # than a one-off event.  At each action-window end, carry forward the most
    # recent charted support level (after resolving simultaneous charting by
    # its highest level).  This avoids coding a continuing ventilator/HFNC
    # episode as "none" merely because no new chart was entered in that bin.
    frame = frame.groupby(["stay_id", "minutes"], as_index=False)["level"].max()
    grid = pd.DataFrame(
        {
            "stay_id": np.repeat(ids, horizon),
            "bin": np.tile(np.arange(horizon), len(ids)),
        }
    )
    grid["decision_minutes"] = (grid.bin * interval + action_minutes).astype(float)
    resolved = []
    for stay_id, query in grid.groupby("stay_id", sort=False):
        history = frame[frame.stay_id == stay_id]
        if history.empty:
            query = query.assign(level=0)
        else:
            query = pd.merge_asof(
                query.sort_values("decision_minutes"),
                history.sort_values("minutes"),
                left_on="decision_minutes",
                right_on="minutes",
                direction="backward",
            ).drop(columns=["stay_id_y", "minutes"], errors="ignore").rename(columns={"stay_id_x": "stay_id"})
            query["level"] = query.level.fillna(0).astype(np.int64)
        resolved.append(query[["stay_id", "bin", "level"]])
    actions = pd.concat(resolved, ignore_index=True).sort_values(["stay_id", "bin"]).level.to_numpy(np.int64)
    return torch.from_numpy(actions.reshape(len(ids), horizon))


def _assemble_raw(
    cohort: pd.DataFrame,
    *,
    id_column: str,
    patient_column: str,
    static: pd.DataFrame,
    vitals: pd.DataFrame,
    treatments: pd.DataFrame,
    horizon: int,
    interval_minutes: int,
    action_minutes: int,
    outcome_kind: str,
    direct_actions: torch.Tensor | None = None,
    canonical_id_column: str | None = None,
) -> _RawClinicalBatch:
    canonical = canonical_id_column or id_column
    cohort = cohort.copy()
    if canonical != id_column:
        cohort = cohort.rename(columns={id_column: canonical})
        id_column = canonical
    ids = cohort[id_column].to_numpy()
    full = pd.MultiIndex.from_product([ids, np.arange(horizon)], names=[id_column, "bin"])
    events = vitals.copy()
    events["minutes"] = pd.to_numeric(events.minutes, errors="coerce")
    events["value"] = pd.to_numeric(events.value, errors="coerce")
    events = events.dropna(subset=["minutes", "value", "kind"])
    lower = events.kind.map({name: bounds[0] for name, bounds in CLINICAL_VALUE_BOUNDS.items()})
    upper = events.kind.map({name: bounds[1] for name, bounds in CLINICAL_VALUE_BOUNDS.items()})
    known = lower.notna() & upper.notna()
    events = events[~known | (events.value.ge(lower) & events.value.le(upper))].copy()
    events["bin"] = (events.minutes // interval_minutes).astype(int)
    events["within"] = events.minutes - events.bin * interval_minutes
    events = events[
        events[id_column].isin(ids)
        & events.bin.between(0, horizon - 1)
        & events.within.between(0.0, interval_minutes, inclusive="left")
    ].sort_values([id_column, "bin", "kind", "minutes"])

    # Outcomes are formed only from the post-action half of each interval.
    # This frame remains separate from the state summaries below so first-half
    # measurements can never leak into Y_(t+1).
    response_events = events[
        events.within.between(action_minutes, interval_minutes, inclusive="left")
    ]
    response = (
        response_events.groupby([id_column, "bin", "kind"]).value.mean().unstack("kind").reindex(full)
    )
    for column in CLINICAL_STATE_FEATURES:
        if column not in response:
            response[column] = np.nan
    response = response.reindex(columns=CLINICAL_STATE_FEATURES)
    required = ("map", "hr") if outcome_kind in {"hypo_tachy", "hypo_hyper"} else ("spo2", "rr")
    complete = response[list(required)].notna().all(axis=1).groupby(level=0).all()
    valid_ids = complete[complete].index.to_numpy()
    if len(valid_ids) == 0:
        raise RuntimeError("no complete trajectories after enforcing observed response windows")
    response = response.loc[valid_ids]

    # At the next decision, the whole completed interval is historical.  Use
    # the prespecified last/mean/min/max summaries for every dynamic covariate;
    # these summaries are written into S_(t+1), never S_t.  Sorting above makes
    # ``last`` a temporal rather than input-row-order statistic.
    aggregated = (
        events.groupby([id_column, "bin", "kind"], sort=False).value
        .agg(list(CLINICAL_SUMMARY_STATS))
        .unstack("kind")
        .reindex(full)
    )
    summary_columns = tuple(
        f"{feature}_{stat}"
        for feature in CLINICAL_STATE_FEATURES
        for stat in CLINICAL_SUMMARY_STATS
    )
    summary = pd.DataFrame(index=full, columns=summary_columns, dtype=np.float32)
    for feature in CLINICAL_STATE_FEATURES:
        for stat in CLINICAL_SUMMARY_STATS:
            key = (stat, feature)
            if key in aggregated.columns:
                summary[f"{feature}_{stat}"] = aggregated[key]
    summary = summary.loc[valid_ids]
    cohort = cohort.set_index(id_column).loc[valid_ids].reset_index()
    static_feature_names = tuple(str(column) for column in static.columns)
    static = static.loc[valid_ids].to_numpy(np.float32)
    treatment_grid = _treatment_grid(treatments, valid_ids, id_column, horizon, interval_minutes, action_minutes)
    summary_values = summary.to_numpy(np.float32).reshape(
        len(valid_ids), horizon, len(summary_columns)
    )
    missing = np.stack(
        [
            ~np.isfinite(summary_values[..., feature_index * len(CLINICAL_SUMMARY_STATS)])
            for feature_index in range(len(CLINICAL_STATE_FEATURES))
        ],
        axis=-1,
    )
    defaults = np.array(
        [
            CLINICAL_STATE_DEFAULTS[feature]
            for feature in CLINICAL_STATE_FEATURES
            for _ in CLINICAL_SUMMARY_STATS
        ],
        dtype=np.float32,
    )
    filled = np.where(np.isfinite(summary_values), summary_values, defaults)
    static_width = static.shape[1]
    dynamic_width = len(summary_columns)
    missing_width = len(CLINICAL_STATE_FEATURES)
    states = np.empty(
        (len(valid_ids), horizon + 1, static_width + dynamic_width + missing_width + 2),
        dtype=np.float32,
    )
    states[:, :, :static_width] = static[:, None, :]
    # Only events observed in the post-action response half-window of bin t
    # are written into S_(t+1).  S_0 therefore uses source defaults together
    # with missingness flags rather than borrowing information from A_0's
    # action window or from any future response.
    states[:, 0, static_width : static_width + dynamic_width] = defaults
    states[:, 0, static_width + dynamic_width : static_width + dynamic_width + missing_width] = 1.0
    states[:, 0, -2:] = 0.0
    cumulative_treatment = np.cumsum(np.log1p(treatment_grid), axis=1)
    states[:, 1:, static_width : static_width + dynamic_width] = filled
    states[:, 1:, static_width + dynamic_width : static_width + dynamic_width + missing_width] = missing.astype(np.float32)
    states[:, 1:, -2:] = cumulative_treatment
    outcomes = _burden_outcomes_from_events(
        response_events,
        full_index=full,
        valid_ids=valid_ids,
        id_column=id_column,
        horizon=horizon,
        kind=outcome_kind,
    )
    episode_ids = torch.from_numpy(valid_ids.astype(np.int64))
    patient_ids = torch.from_numpy(cohort[patient_column].to_numpy(np.int64))
    direct = None
    # The direct-action grid is already ordered by the pre-filter cohort IDs; reorder explicitly below when supplied.
    if direct_actions is not None:
        source_ids = ids
        lookup = {int(value): index for index, value in enumerate(source_ids)}
        direct = direct_actions[torch.tensor([lookup[int(value)] for value in valid_ids], dtype=torch.long)]
    return _RawClinicalBatch(
        states=torch.from_numpy(states),
        outcomes=torch.from_numpy(outcomes),
        treatments=torch.from_numpy(treatment_grid),
        patient_ids=patient_ids,
        episode_ids=episode_ids,
        static_indices=tuple(range(static_width)),
        direct_actions=direct,
        state_feature_names=(
            static_feature_names
            + summary_columns
            + tuple(f"{name}_missing" for name in CLINICAL_STATE_FEATURES)
            + ("cumulative_log_fluid", "cumulative_log_pressor")
        ),
    )


def _treatment_grid(
    treatments: pd.DataFrame,
    ids: np.ndarray,
    id_column: str,
    horizon: int,
    interval: int,
    action_minutes: int,
) -> np.ndarray:
    index = pd.MultiIndex.from_product([ids, np.arange(horizon)], names=[id_column, "bin"])
    if treatments.empty:
        return np.zeros((len(ids), horizon, 2), dtype=np.float32)
    frame = treatments.copy()
    if "end_minutes" in frame:
        return _interval_treatment_grid(frame, ids, id_column, horizon, interval, action_minutes)
    frame["bin"] = (frame.minutes // interval).astype(int)
    frame["within"] = frame.minutes - frame.bin * interval
    frame = frame[
        frame[id_column].isin(ids)
        & frame.bin.between(0, horizon - 1)
        & frame.within.between(0, action_minutes, inclusive="left")
    ]
    grouped = frame.groupby([id_column, "bin", "component"]).value.sum().unstack("component")
    grouped = grouped.reindex(index).reindex(columns=["fluid", "pressor"], fill_value=0.0).fillna(0.0)
    return grouped.to_numpy(np.float32).reshape(len(ids), horizon, 2)


def _interval_treatment_grid(
    frame: pd.DataFrame,
    ids: np.ndarray,
    id_column: str,
    horizon: int,
    interval: int,
    action_minutes: int,
) -> np.ndarray:
    """Aggregate interval-overlap treatment exposure into action half-windows."""

    frame = frame[frame[id_column].isin(ids)].copy()
    if frame.empty:
        return np.zeros((len(ids), horizon, 2), dtype=np.float32)
    frame["end_minutes"] = pd.to_numeric(frame.end_minutes, errors="coerce").fillna(frame.minutes)
    frame["minutes"] = pd.to_numeric(frame.minutes, errors="coerce")
    frame = frame.dropna(subset=["minutes", "end_minutes"])
    frame["end_minutes"] = np.maximum(frame.end_minutes, frame.minutes)
    pieces = []
    for bin_index in range(horizon):
        window_start = bin_index * interval
        window_end = window_start + action_minutes
        overlap_start = frame.minutes.clip(lower=window_start)
        overlap_end = frame.end_minutes.clip(upper=window_end)
        overlap = (overlap_end - overlap_start).clip(lower=0.0)
        # A bolus can have an instantaneous start/end timestamp.  Attribute its
        # full amount to the action window containing that timestamp.
        instantaneous = (frame.end_minutes <= frame.minutes) & frame.minutes.between(window_start, window_end, inclusive="left")
        active = (overlap > 0.0) | instantaneous
        if not active.any():
            continue
        selected = frame.loc[active, [id_column, "component", "value"]].copy()
        selected["bin"] = bin_index
        selected_overlap = overlap.loc[active].to_numpy()
        duration = (frame.loc[active, "end_minutes"] - frame.loc[active, "minutes"]).clip(lower=1e-6).to_numpy()
        kind = frame.loc[active, "interval_kind"].to_numpy() if "interval_kind" in frame else np.repeat("amount", len(selected))
        values = selected.value.to_numpy(np.float64)
        selected["exposure"] = np.where(
            kind == "active_duration",
            selected_overlap,
            np.where(instantaneous.loc[active].to_numpy(), values, values * selected_overlap / duration),
        )
        pieces.append(selected[[id_column, "bin", "component", "exposure"]])
    full = pd.MultiIndex.from_product([ids, np.arange(horizon)], names=[id_column, "bin"])
    if not pieces:
        return np.zeros((len(ids), horizon, 2), dtype=np.float32)
    grouped = pd.concat(pieces, ignore_index=True).groupby([id_column, "bin", "component"]).exposure.sum().unstack("component")
    grouped = grouped.reindex(full).reindex(columns=["fluid", "pressor"], fill_value=0.0).fillna(0.0)
    return grouped.to_numpy(np.float32).reshape(len(ids), horizon, 2)


def _burden_outcomes_from_events(
    response_events: pd.DataFrame,
    *,
    full_index: pd.MultiIndex,
    valid_ids: np.ndarray,
    id_column: str,
    horizon: int,
    kind: str,
) -> np.ndarray:
    """Average the prespecified pointwise clinical burden in each response half.

    Applying the positive-part transform before averaging is important.  For
    example, ``mean((65 - MAP)_+)`` measures time spent below the threshold,
    whereas ``(65 - mean(MAP))_+`` can erase alternating hypotension and
    normotension.
    """

    if kind == "hypo_tachy":
        components = (
            ("map", lambda value: np.maximum(65.0 - value, 0.0) / 15.0),
            ("hr", lambda value: np.maximum(value - 100.0, 0.0) / 40.0),
        )
    elif kind == "hypo_hyper":
        components = (
            ("map", lambda value: np.maximum(65.0 - value, 0.0) / 15.0),
            ("map", lambda value: np.maximum(value - 100.0, 0.0) / 20.0),
        )
    elif kind == "hypox_tachyp":
        components = (
            ("spo2", lambda value: np.maximum(92.0 - value, 0.0) / 10.0),
            ("rr", lambda value: np.maximum(value - 22.0, 0.0) / 15.0),
        )
    else:
        raise ValueError(f"unknown outcome kind: {kind}")

    outcomes = []
    for feature, transform in components:
        observed = response_events[response_events.kind.eq(feature)].copy()
        observed["burden"] = transform(observed.value.to_numpy(dtype=np.float64))
        burden = (
            observed.groupby([id_column, "bin"]).burden.mean().reindex(full_index).loc[valid_ids]
        )
        outcomes.append(burden.to_numpy(np.float32).reshape(len(valid_ids), horizon))
    return np.stack(outcomes, axis=2).astype(np.float32)


def _sample_cohort(frame: pd.DataFrame, id_column: str, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    chosen = np.random.default_rng(seed).choice(frame[id_column].to_numpy(), size=limit, replace=False)
    return frame[frame[id_column].isin(chosen)].copy()


def _episode_row_indices(values: torch.Tensor | np.ndarray, ordered_ids: np.ndarray) -> torch.Tensor:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    lookup = {int(value): index for index, value in enumerate(ordered_ids)}
    return torch.tensor([lookup[int(value)] for value in array], dtype=torch.long)


def _append_decision_time(states: torch.Tensor) -> torch.Tensor:
    """Append normalized pre-action decision time to every base state."""

    if states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("clinical states must have shape [N,T+1,D] with T >= 1")
    decision_time = torch.linspace(
        0.0,
        1.0,
        states.shape[1],
        dtype=states.dtype,
        device=states.device,
    )
    return torch.cat(
        (states, decision_time[None, :, None].expand(states.shape[0], -1, 1)),
        dim=2,
    )


def _history_stack(states: torch.Tensor, static_indices: tuple[int, ...], length: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Expose the last L decision states to the clinical two-layer GRU."""

    if length == 1:
        return states, static_indices
    n, time, width = states.shape
    pieces = []
    for current in range(time):
        start = max(0, current - length + 1)
        history = states[:, start : current + 1]
        if history.shape[1] < length:
            padding = states[:, :1].expand(-1, length - history.shape[1], -1)
            history = torch.cat((padding, history), dim=1)
        pieces.append(history.reshape(n, length * width))
    stacked = torch.stack(pieces, dim=1)
    repeated_static = tuple(offset * width + index for offset in range(length) for index in static_indices)
    return stacked, repeated_static
