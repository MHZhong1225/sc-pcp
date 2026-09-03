from __future__ import annotations

import pytest
import torch
from torch import Tensor

from scpcp.config import ModelConfig
from scpcp.outcome_model import (
    _ALL_ACTION_INFERENCE_CHUNK_SIZE,
    GaussianOutcomeModel,
)
from scpcp.scores import (
    _SCORE_INFERENCE_CHUNK_SIZE,
    ConformalRegion,
    normalized_max_score,
    predict_observed_actions,
    score_batch,
)


def _model(architecture: str) -> GaussianOutcomeModel:
    torch.manual_seed(17)
    if architecture == "gru":
        return GaussianOutcomeModel(
            state_dim=12,
            n_actions=3,
            config=ModelConfig(
                architecture="gru",
                history_length=3,
                hidden_dim=8,
                representation_dim=6,
            ),
            static_indices=(1, 5),
        ).eval()
    return GaussianOutcomeModel(
        state_dim=5,
        n_actions=3,
        config=ModelConfig(
            architecture="mlp",
            hidden_dim=8,
            representation_dim=6,
        ),
    ).eval()


@torch.no_grad()
def _repeated_forward_reference(
    model: GaussianOutcomeModel,
    states: Tensor,
) -> tuple[Tensor, Tensor]:
    n = len(states)
    repeated_states = (
        states[:, None, :]
        .expand(n, model.n_actions, model.state_dim)
        .reshape(-1, model.state_dim)
    )
    repeated_actions = torch.arange(model.n_actions).repeat(n)
    mean, scale = model(repeated_states, repeated_actions)
    return (
        mean.reshape(n, model.n_actions, 2),
        scale.reshape(n, model.n_actions, 2),
    )


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_small_all_action_inference_preserves_repeated_forward_exactly(
    architecture: str,
) -> None:
    model = _model(architecture)
    states = torch.randn(19, model.state_dim)

    observed = model.predict_all_actions(states)
    expected = _repeated_forward_reference(model, states)

    assert torch.equal(observed[0], expected[0])
    assert torch.equal(observed[1], expected[1])


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_large_all_action_inference_chunks_state_encoding_and_preserves_order(
    architecture: str,
) -> None:
    model = _model(architecture)
    n = _ALL_ACTION_INFERENCE_CHUNK_SIZE + 3
    states = torch.randn(n, model.state_dim)
    encoder_batch_sizes: list[int] = []
    hook = model.state_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoder_batch_sizes.append(len(inputs[0]))
    )

    observed = model.predict_all_actions(states)
    hook.remove()
    expected = _repeated_forward_reference(model, states)

    assert observed[0].shape == (n, model.n_actions, 2)
    assert observed[1].shape == (n, model.n_actions, 2)
    assert encoder_batch_sizes == [_ALL_ACTION_INFERENCE_CHUNK_SIZE, 3]
    torch.testing.assert_close(observed[0], expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(observed[1], expected[1], rtol=1e-5, atol=1e-6)
    assert not observed[0].requires_grad
    assert not observed[1].requires_grad


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_all_action_inference_accepts_empty_state_batch(architecture: str) -> None:
    model = _model(architecture)
    states = torch.empty(0, model.state_dim)

    mean, scale = model.predict_all_actions(states)

    assert mean.shape == (0, model.n_actions, 2)
    assert scale.shape == (0, model.n_actions, 2)
    assert mean.dtype == states.dtype
    assert scale.dtype == states.dtype


def _score_inputs(
    model: GaussianOutcomeModel,
    *,
    n: int,
    horizon: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.randn(n, horizon, model.state_dim),
        torch.randint(model.n_actions, (n, horizon)),
        torch.randn(n, horizon, 2),
    )


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
@pytest.mark.parametrize("use_region", [False, True])
def test_small_score_batch_preserves_all_at_once_result_exactly(
    architecture: str,
    use_region: bool,
) -> None:
    model = _model(architecture)
    states, actions, outcomes = _score_inputs(model, n=7, horizon=3)
    scorer = ConformalRegion(model) if use_region else model

    observed = score_batch(scorer, states, actions, outcomes)
    flat_states = states.reshape(-1, model.state_dim)
    flat_actions = actions.reshape(-1)
    flat_outcomes = outcomes.reshape(-1, 2)
    expected = (
        scorer.score(flat_states, flat_actions, flat_outcomes)
        if use_region
        else normalized_max_score(
            model,
            flat_states,
            flat_actions,
            flat_outcomes,
        )
    ).reshape(7, 3)

    assert torch.equal(observed, expected)


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_large_score_batch_chunks_nondivisible_transitions_and_preserves_shape(
    architecture: str,
) -> None:
    model = _model(architecture)
    n, horizon = _SCORE_INFERENCE_CHUNK_SIZE // 2 + 2, 2
    states, actions, outcomes = _score_inputs(model, n=n, horizon=horizon)
    region = ConformalRegion(model)
    encoder_batch_sizes: list[int] = []
    hook = model.state_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoder_batch_sizes.append(len(inputs[0]))
    )

    observed = score_batch(region, states, actions, outcomes)
    hook.remove()
    expected = region.score(
        states.reshape(-1, model.state_dim),
        actions.reshape(-1),
        outcomes.reshape(-1, 2),
    ).reshape(n, horizon)

    assert observed.shape == (n, horizon)
    assert encoder_batch_sizes == [_SCORE_INFERENCE_CHUNK_SIZE, 4]
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    assert not observed.requires_grad


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_score_batch_accepts_empty_trajectory_batch(architecture: str) -> None:
    model = _model(architecture)
    states, actions, outcomes = _score_inputs(model, n=0, horizon=3)

    scores = score_batch(ConformalRegion(model), states, actions, outcomes)

    assert scores.shape == (0, 3)
    assert scores.dtype == states.dtype
    assert not scores.requires_grad


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_small_observed_action_prediction_preserves_forward_exactly(
    architecture: str,
) -> None:
    model = _model(architecture)
    states = torch.randn(19, model.state_dim)
    actions = torch.randint(model.n_actions, (19,))

    observed = predict_observed_actions(model, states, actions)
    expected = model(states, actions)

    assert torch.equal(observed[0], expected[0])
    assert torch.equal(observed[1], expected[1])


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_observed_action_prediction_chunks_large_fresh_batches(
    architecture: str,
) -> None:
    model = _model(architecture)
    n = _SCORE_INFERENCE_CHUNK_SIZE + 3
    states = torch.randn(n, model.state_dim)
    actions = torch.randint(model.n_actions, (n,))
    encoder_batch_sizes: list[int] = []
    hook = model.state_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoder_batch_sizes.append(len(inputs[0]))
    )

    observed = predict_observed_actions(model, states, actions)
    hook.remove()
    expected = model(states, actions)

    assert encoder_batch_sizes == [_SCORE_INFERENCE_CHUNK_SIZE, 3]
    torch.testing.assert_close(observed[0], expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(observed[1], expected[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_observed_action_prediction_accepts_empty_batch(architecture: str) -> None:
    model = _model(architecture)
    states = torch.empty(0, model.state_dim)
    actions = torch.empty(0, dtype=torch.long)

    mean, scale = predict_observed_actions(model, states, actions)

    assert mean.shape == (0, 2)
    assert scale.shape == (0, 2)
