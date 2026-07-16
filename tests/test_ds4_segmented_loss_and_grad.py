from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MUTATION_SOURCE = pathlib.Path(os.environ["DS4_MUTATION_OVERLAY"]) if os.environ.get("DS4_MUTATION_OVERLAY") else PROJECT_ROOT / "python-envs/mlx/src"
if str(_MUTATION_SOURCE) not in sys.path:
    sys.path.insert(0, str(_MUTATION_SOURCE))
if os.environ.get("DS4_MUTATION_OVERLAY"):
    for _name in list(sys.modules):
        if _name == "ds4_ft_mlx" or _name.startswith("ds4_ft_mlx."):
            del sys.modules[_name]

import ds4_ft_mlx.segmented_loss_and_grad as provider_module
from ds4_ft_mlx.segmented_loss_and_grad import make_ds4_segmented_loss_and_grad


class _ToyLayer(nn.Module):
    def __init__(self, shared: mx.array, hidden: int, index: int):
        super().__init__()
        self.shared = shared
        self.index = index
        self.local = mx.eye(hidden) * (0.04 + index * 0.01)
        self.bias = mx.ones((hidden,)) * (0.01 + index * 0.002)

    def __call__(self, hidden_states, input_ids=None, cache=None):
        del input_ids, cache
        if LAYER_EVENTS is not None:
            LAYER_EVENTS.append(self.index)
        return hidden_states + hidden_states @ self.shared.T + hidden_states @ self.local.T + self.bias


LAYER_EVENTS = None
RANDOM_SAMPLES = None
RANDOM_REPLAY_CACHE = {}


class _RandomLayer(_ToyLayer):
    def __call__(self, hidden_states, input_ids=None, cache=None):
        state_key = tuple(tuple(value.tolist()) for value in mx.random.state)
        noise = mx.random.uniform(shape=hidden_states.shape, dtype=hidden_states.dtype) * 0.01
        mx.eval(noise)
        key = (self.index, state_key, tuple(hidden_states.shape), hidden_states.dtype)
        if key not in RANDOM_REPLAY_CACHE:
            RANDOM_REPLAY_CACHE[key] = mx.array(noise)
        noise = RANDOM_REPLAY_CACHE[key]
        if RANDOM_SAMPLES is not None:
            RANDOM_SAMPLES.append(mx.array(noise))
        return super().__call__(hidden_states, input_ids, cache) + noise


class _ToyHead(nn.Module):
    def __init__(self, shared: mx.array):
        super().__init__()
        self.shared = shared

    def __call__(self, hidden_states):
        return hidden_states[..., 0, :] @ self.shared.T


class _ToyBackbone(nn.Module):
    def __init__(self, depth: int, hidden: int = 3, vocab: int = 7):
        super().__init__()
        shared = mx.eye(hidden) * 0.08
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.pipeline_layers = [_ToyLayer(shared, hidden, i) for i in range(depth)]
        self.hc_head = _ToyHead(shared)
        self.norm = nn.Identity()
        self.hc_mult = 1
        self.hidden_size = hidden


class _ToyModel(nn.Module):
    model_type = "deepseek_v4_nn"

    def __init__(self, depth: int = 3, hidden: int = 3, vocab: int = 7):
        super().__init__()
        self.model = _ToyBackbone(depth, hidden, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.unused = mx.ones((2, 2), dtype=mx.float32)

    def __call__(self, inputs):
        hidden = self.model.embed_tokens(inputs)
        hidden = mx.broadcast_to(
            mx.expand_dims(hidden, -2),
            (*hidden.shape[:-1], self.model.hc_mult, self.model.hidden_size),
        )
        for layer in self.model.pipeline_layers:
            hidden = layer(hidden, input_ids=inputs, cache=None)
        return self.lm_head(self.model.norm(self.model.hc_head(hidden)))


def _batch():
    return mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32), mx.array([[1, 4]], dtype=mx.int32)


def _default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    token_count = mask.sum()
    loss = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / token_count
    return loss, token_count


def _assert_tree_close(actual, expected, atol=1e-5, rtol=1e-5):
    actual_flat = tree_flatten(actual)
    expected_flat = tree_flatten(expected)
    assert [path for path, _ in actual_flat] == [path for path, _ in expected_flat]
    for (path, left), (_, right) in zip(actual_flat, expected_flat):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert mx.allclose(left, right, atol=atol, rtol=rtol).item(), path


def _sum_shared_oracle(parameters, gradients):
    sums = {}

    def collect(param, grad):
        if isinstance(param, mx.array):
            key = id(param)
            sums[key] = grad if key not in sums else sums[key] + grad
        elif type(param) is dict:
            for name in param:
                collect(param[name], grad[name])
        elif type(param) in (list, tuple):
            for left, right in zip(param, grad):
                collect(left, right)

    def expand(param):
        if isinstance(param, mx.array):
            return sums[id(param)]
        if type(param) is dict:
            return {name: expand(child) for name, child in param.items()}
        if type(param) is list:
            return [expand(child) for child in param]
        return tuple(expand(child) for child in param)

    collect(parameters, gradients)
    return expand(parameters)


def _monolithic(model, batch, lengths):
    parameters = model.trainable_parameters()
    value_grad = nn.value_and_grad(model, lambda current: _default_loss(current, batch, lengths)[0])
    loss, gradients = value_grad(model)
    token_count = _default_loss(model, batch, lengths)[1]
    gradients = _sum_shared_oracle(parameters, gradients)
    mx.eval(loss, token_count, gradients)
    return (loss, token_count), gradients


def _monolithic_once(model, batch, lengths):
    parameters = model.trainable_parameters()
    targets = batch[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    value_grad = nn.value_and_grad(
        model,
        lambda current: _default_loss(current, batch, lengths)[0],
    )
    loss, gradients = value_grad(model)
    gradients = _sum_shared_oracle(parameters, gradients)
    mx.eval(loss, gradients)
    return (loss, mask.sum()), gradients


def test_factory_validation_and_scaffold_red():
    with pytest.raises(TypeError, match=r"^segment_size must be an int in \[1, 4\]$"):
        make_ds4_segmented_loss_and_grad(segment_size=True)
    with pytest.raises(ValueError, match=r"^segment_size must be in \[1, 4\]$"):
        make_ds4_segmented_loss_and_grad(segment_size=5)


def test_depth_segment_equivalence_matrix():
    batch, lengths = _batch()
    for depth in (1, 2, 3, 5, 8):
        for segment_size in (1, 2, 3, 4):
            mx.random.seed(1000 + depth * 10 + segment_size)
            oracle_model = _ToyModel(depth=depth)
            mx.random.seed(1000 + depth * 10 + segment_size)
            provider_model = _ToyModel(depth=depth)
            oracle_result = _monolithic(oracle_model, batch, lengths)
            provider_result = make_ds4_segmented_loss_and_grad(segment_size=segment_size)(provider_model, batch, lengths)
            mx.eval(provider_result[0][0], provider_result[0][1], provider_result[1])
            assert provider_result[0][1].dtype == mx.int32
            assert provider_result[0][1].item() == oracle_result[0][1].item()
            assert mx.allclose(provider_result[0][0], oracle_result[0][0], atol=1e-5, rtol=1e-5).item()
            _assert_tree_close(provider_result[1], oracle_result[1])


def test_tied_shared_and_repeated_parameters():
    batch, lengths = _batch()
    for depth, segment_size in ((5, 2), (8, 3), (3, 1), (3, 2)):
        mx.random.seed(2025 + depth + segment_size)
        oracle_model = _ToyModel(depth=depth)
        mx.random.seed(2025 + depth + segment_size)
        provider_model = _ToyModel(depth=depth)
        oracle = _monolithic(oracle_model, batch, lengths)
        result = make_ds4_segmented_loss_and_grad(segment_size=segment_size)(
            provider_model, batch, lengths
        )
        _assert_tree_close(result[1], oracle[1])
        unused = result[1]["unused"]
        mx.eval(unused)
        assert mx.array_equal(unused, mx.zeros_like(unused)).item()


def test_exact_output_schema_and_dtypes():
    batch, lengths = _batch()
    model = _ToyModel(depth=3)
    result = make_ds4_segmented_loss_and_grad(segment_size=2)(model, batch, lengths)
    assert type(result) is tuple and len(result) == 2
    assert type(result[0]) is tuple and len(result[0]) == 2
    assert result[0][0].shape == () and result[0][0].dtype == mx.float32
    assert result[0][1].shape == () and result[0][1].dtype == mx.int32
    assert type(result[1]) is dict
    assert [path for path, _ in tree_flatten(result[1])] == [path for path, _ in tree_flatten(model.trainable_parameters())]


def test_default_masked_loss_contract():
    batch = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    lengths = mx.array([[2, 2]], dtype=mx.int32)
    model = _ToyModel(depth=1)
    result = make_ds4_segmented_loss_and_grad()(model, batch, lengths)
    oracle = _default_loss(model, batch, lengths)
    mx.eval(result[0][0], result[0][1], oracle[0], oracle[1])
    assert result[0][1].item() == 1
    assert result[0][1].item() == oracle[1].item()
    assert mx.allclose(result[0][0], oracle[0], atol=1e-5, rtol=1e-5).item()


def test_setup_validation_precedence_and_messages():
    batch, lengths = _batch()
    provider = make_ds4_segmented_loss_and_grad()
    with pytest.raises(TypeError) as wrong_model:
        provider(object(), batch, lengths)
    assert str(wrong_model.value) == (
        "DS4 segmented loss_and_grad requires model_type='deepseek_v4_nn'; got None"
    )

    class Missing:
        model_type = "deepseek_v4_nn"

    with pytest.raises(TypeError) as missing:
        provider(Missing(), batch, lengths)
    assert str(missing.value) == (
        "DS4 segmented loss_and_grad model topology missing: "
        "model.model.embed_tokens,model.model.pipeline_layers,model.model.hc_head,"
        "model.model.norm,model.lm_head"
    )

    model = _ToyModel(depth=1)
    cases = [
        (
            "batch class",
            ([], lengths),
            TypeError,
            "DS4 segmented loss_and_grad batch must be an mx.array; got list",
        ),
        (
            "batch dtype",
            (mx.zeros((1, 2), dtype=mx.float32), lengths),
            TypeError,
            "DS4 segmented loss_and_grad batch must have dtype mlx.core.int32; got mlx.core.float32",
        ),
        (
            "batch shape",
            (mx.zeros((1, 1), dtype=mx.int32), lengths),
            ValueError,
            "DS4 segmented loss_and_grad batch must have shape [B, L] with B >= 1 and L >= 2; got (1, 1)",
        ),
        (
            "lengths class",
            (batch, []),
            TypeError,
            "DS4 segmented loss_and_grad lengths must be an mx.array; got list",
        ),
        (
            "lengths dtype",
            (batch, mx.zeros((1, 2), dtype=mx.float32)),
            TypeError,
            "DS4 segmented loss_and_grad lengths must have dtype mlx.core.int32; got mlx.core.float32",
        ),
        (
            "lengths shape",
            (batch, mx.zeros((1, 3), dtype=mx.int32)),
            ValueError,
            "DS4 segmented loss_and_grad lengths must have shape [B, 2] matching batch B=1; got (1, 3)",
        ),
    ]
    for _, args, error_type, message in cases:
        with pytest.raises(error_type) as failure:
            provider(model, *args)
        assert str(failure.value) == message

    class EmptyDepth(_ToyModel):
        def __init__(self):
            super().__init__(depth=1)
            self.model.pipeline_layers = []

    with pytest.raises(ValueError) as empty_depth:
        provider(EmptyDepth(), batch, lengths)
    assert str(empty_depth.value) == "DS4 segmented loss_and_grad requires at least one decoder layer"

    class EmptyTree(_ToyModel):
        def trainable_parameters(self):
            return {}

    with pytest.raises(ValueError) as empty_tree:
        provider(EmptyTree(), batch, lengths)
    assert str(empty_tree.value) == "DS4 segmented loss_and_grad requires at least one trainable parameter"

    class UnsupportedRoot(_ToyModel):
        def trainable_parameters(self):
            return "root"

    with pytest.raises(TypeError) as unsupported_root:
        provider(UnsupportedRoot(depth=1), batch, lengths)
    assert str(unsupported_root.value) == (
        "DS4 segmented loss_and_grad trainable structure at '<root>' "
        "must be dict, list, tuple, or mx.array; got str"
    )

    class BadLeaf(_ToyModel):
        def trainable_parameters(self):
            return {"nested": [mx.ones((1,), dtype=mx.float32), ("bad",)]}

    with pytest.raises(TypeError) as bad_leaf:
        provider(BadLeaf(depth=1), batch, lengths)
    assert str(bad_leaf.value) == (
        "DS4 segmented loss_and_grad trainable parameter 'nested.1.0' "
        "must be an mx.array; got str"
    )

    class BadDtype(_ToyModel):
        def trainable_parameters(self):
            return {"bad": mx.ones((1,), dtype=mx.int32)}

    with pytest.raises(TypeError) as bad_dtype:
        provider(BadDtype(depth=1), batch, lengths)
    assert str(bad_dtype.value) == (
        "DS4 segmented loss_and_grad requires float16, bfloat16, or float32 "
        "trainable parameter 'bad'; got mlx.core.int32"
    )


def test_stochastic_replay_and_rollback():
    batch, lengths = _batch()
    model = _ToyModel(depth=2)
    provider = make_ds4_segmented_loss_and_grad(segment_size=1)
    mx.random.seed(101)
    entry_state = _state_values()
    first = provider(model, batch, lengths)
    mx.eval(first[0][0], first[1])
    post_state = _state_values()
    second_entry_state = _state_values()
    second = provider(model, batch, lengths)
    mx.eval(second[0][0], second[1])
    second_post_state = _state_values()
    assert mx.allclose(first[0][0], second[0][0]).item()
    _assert_tree_close(first[1], second[1])
    assert all(mx.array_equal(left, right).item() for left, right in zip(entry_state, post_state))
    assert all(mx.array_equal(left, right).item() for left, right in zip(second_entry_state, second_post_state))


def test_stochastic_one_forward_cadence_and_replay():
    batch, lengths = _batch()
    for seed in (123, 456):
        mx.random.seed(77)
        model = _ToyModel(depth=4)
        model.model.pipeline_layers = [_RandomLayer(mx.eye(3) * 0.08, 3, index) for index in range(4)]
        provider = make_ds4_segmented_loss_and_grad(segment_size=1)
        mx.random.seed(seed)
        first = provider(model, batch, lengths)
        first_state = [mx.array(x) for x in mx.random.state]
        mx.eval(first[0][0], first[1], *first_state)
        mx.random.seed(seed)
        second = provider(model, batch, lengths)
        second_state = [mx.array(x) for x in mx.random.state]
        mx.eval(second[0][0], second[1], *second_state)
        assert mx.allclose(first[0][0], second[0][0], atol=1e-5, rtol=1e-5).item()
        _assert_tree_close(first[1], second[1])
        for left, right in zip(first_state, second_state):
            assert mx.array_equal(left, right).item()


def test_stochastic_depth_four_segment_matrix():
    batch, lengths = _batch()
    original_tail = provider_module._vjp_tail
    original_segment = provider_module._vjp_segment
    original_embedding = provider_module._vjp_embedding

    def random_model():
        model = _ToyModel(depth=4)
        model.model.pipeline_layers = [
            (_RandomLayer if index in (0, 2) else _ToyLayer)(
                mx.eye(3) * 0.08, 3, index
            )
            for index in range(4)
        ]
        return model

    def forward_oracle(model):
        inputs = batch[:, :-1]
        boundary = model.model.embed_tokens(inputs)
        boundary = mx.broadcast_to(
            mx.expand_dims(boundary, -2),
            (*boundary.shape[:-1], model.model.hc_mult, model.model.hidden_size),
        )
        mx.eval(boundary)
        boundaries = [mx.stop_gradient(boundary)]
        pre_states = []
        post_states = []
        for layer in model.model.pipeline_layers:
            pre_states.append([mx.array(value) for value in mx.random.state])
            boundary = layer(boundary, input_ids=inputs, cache=None)
            mx.eval(boundary)
            boundary = mx.stop_gradient(boundary)
            mx.eval(boundary)
            boundaries.append(boundary)
            post_states.append([mx.array(value) for value in mx.random.state])
        tail_state = [mx.array(value) for value in mx.random.state]
        logits = model.lm_head(model.model.norm(model.model.hc_head(boundary)))
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        loss, tokens = _loss_from_logits_for_test(logits, targets, mask), mask.sum()
        mx.eval(loss, tokens)
        return boundaries, pre_states, post_states, tail_state, [mx.array(value) for value in mx.random.state]

    def suffix_vjp(model, boundary, start, random_state):
        mx.random.state = [mx.array(value) for value in random_state]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        def suffix(value):
            for index in range(start, 4):
                value = model.model.pipeline_layers[index](value, input_ids=inputs, cache=None)
            logits = model.lm_head(model.model.norm(model.model.hc_head(value)))
            return _loss_from_logits_for_test(logits, targets, mask)

        _, gradients = mx.vjp(suffix, [boundary], [mx.ones((), dtype=mx.float32)])
        mx.eval(gradients[0])
        return gradients[0]

    try:
        for seed in (123, 456):
            for segment_size in (1, 2):
                mx.random.seed(901)
                oracle_model = random_model()
                model = random_model()
                forward_model = random_model()
                model.update(oracle_model.parameters())
                forward_model.update(oracle_model.parameters())
                model_before = {path: mx.array(value) for path, value in tree_flatten(model.parameters())}
                for (_, left), (_, right) in zip(
                    tree_flatten(model.parameters()), tree_flatten(oracle_model.parameters())
                ):
                    mx.eval(left, right)
                    assert mx.allclose(left, right).item()
                global RANDOM_SAMPLES
                mx.random.seed(seed)
                boundaries, pre_states, post_states, tail_state, forward_post = forward_oracle(forward_model)

                observed = {"tail": None, "segments": [], "embedding": None}

                def tail(*args, **kwargs):
                    result = original_tail(*args, **kwargs)
                    observed["tail"] = (args[2], result[0])
                    return result

                def segment(*args, **kwargs):
                    result = original_segment(*args, **kwargs)
                    observed["segments"].append((args[2], args[4], args[5], args[6], args[7], result[0]))
                    return result

                def embedding(*args, **kwargs):
                    result = original_embedding(*args, **kwargs)
                    observed["embedding"] = (args[3], result)
                    return result

                provider_module._vjp_tail = tail
                provider_module._vjp_segment = segment
                provider_module._vjp_embedding = embedding
                try:
                    RANDOM_SAMPLES = []
                    mx.random.seed(seed)
                    result = make_ds4_segmented_loss_and_grad(segment_size=segment_size)(model, batch, lengths)
                    provider_samples = [mx.array(value) for value in RANDOM_SAMPLES]
                finally:
                    provider_module._vjp_tail = original_tail
                    provider_module._vjp_segment = original_segment
                    provider_module._vjp_embedding = original_embedding

                mx.eval(result[0][0], result[0][1], result[1])
                actual_post = _state_values()
                for path, value in tree_flatten(model.parameters()):
                    assert mx.array_equal(value, model_before[path]).item(), ('restored', path)
                RANDOM_SAMPLES = []
                mx.random.seed(seed)
                oracle_result = _monolithic_once(model, batch, lengths)
                oracle_samples = [mx.array(value) for value in RANDOM_SAMPLES]
                oracle_post = _state_values()
                assert len(provider_samples) >= 4
                for index in range(2):
                    assert mx.array_equal(provider_samples[index], oracle_samples[index]).item()
                assert all(mx.array_equal(a, b).item() for a, b in zip(actual_post, oracle_post))
                assert result[0][1].item() == oracle_result[0][1].item()
                assert mx.allclose(result[0][0], oracle_result[0][0], atol=1e-5, rtol=1e-5).item()
                _assert_tree_close(result[1], oracle_result[1])

                expected_ranges = [(start, min(start + segment_size, 4)) for start in range(0, 4, segment_size)]
                assert [item[1:3] for item in observed["segments"]] == list(reversed(expected_ranges))
                tail_boundary, tail_adjoint = observed["tail"]
                expected_tail = suffix_vjp(oracle_model, boundaries[-1], 4, tail_state)
                mx.eval(tail_boundary, tail_adjoint, expected_tail)
                assert mx.allclose(tail_boundary, boundaries[-1], atol=1e-5, rtol=1e-5).item()
                assert mx.allclose(tail_adjoint, expected_tail, atol=1e-5, rtol=1e-5).item()
                for boundary, start, end, incoming, replay_state, input_adjoint in observed["segments"]:
                    assert all(mx.array_equal(a, b).item() for a, b in zip(replay_state, pre_states[start]))
                    expected_input = suffix_vjp(oracle_model, boundaries[start], start, pre_states[start])
                    expected_output = suffix_vjp(oracle_model, boundaries[end], end, post_states[end - 1])
                    mx.eval(boundary, incoming, input_adjoint, expected_input, expected_output)
                    assert mx.allclose(boundary, boundaries[start], atol=1e-5, rtol=1e-5).item()
                    assert mx.allclose(incoming, expected_output, atol=1e-5, rtol=1e-5).item()
                    assert mx.allclose(input_adjoint, expected_input, atol=1e-5, rtol=1e-5).item()
                embedding_adjoint, _ = observed["embedding"]
                expected_embedding = suffix_vjp(oracle_model, boundaries[0], 0, pre_states[0])
                mx.eval(embedding_adjoint, expected_embedding)
                assert mx.allclose(embedding_adjoint, expected_embedding, atol=1e-5, rtol=1e-5).item()
    finally:
        provider_module._vjp_tail = original_tail
        provider_module._vjp_segment = original_segment
        provider_module._vjp_embedding = original_embedding


def test_failure_restores_model_and_random_state():
    batch, lengths = _batch()
    model = _ToyModel(depth=2)
    provider = make_ds4_segmented_loss_and_grad()
    original = {path: mx.array(value) for path, value in tree_flatten(model.parameters())}
    original_state = {path: mx.array(value) for path, value in tree_flatten(model.state)}
    entry_random = [mx.array(x) for x in mx.random.state]
    original_layer = model.model.pipeline_layers[1]

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic reverse failure")

    model.model.pipeline_layers[1] = fail
    with pytest.raises(RuntimeError, match="synthetic reverse failure"):
        provider(model, batch, lengths)
    model.model.pipeline_layers[1] = original_layer
    for path, value in tree_flatten(model.parameters()):
        assert mx.array_equal(value, original[path]).item(), "parameter rollback mismatch"
    for path, value in tree_flatten(model.state):
        assert mx.array_equal(value, original_state[path]).item(), "model-state rollback mismatch"
    assert [x.tolist() for x in mx.random.state] == [x.tolist() for x in entry_random]


@pytest.mark.parametrize("phase", ("setup", "forward", "tail", "reverse", "assembly", "materialization"))
def test_failure_phase_matrix_restores_entry_state(phase, monkeypatch):
    batch, lengths = _batch()
    mx.random.seed(700 + hash(phase) % 100)
    model = _stochastic_model()
    provider = make_ds4_segmented_loss_and_grad(segment_size=1)
    before_parameters = {path: mx.array(value) for path, value in tree_flatten(model.parameters())}
    before_state = {path: mx.array(value) for path, value in tree_flatten(model.state)}
    before_random = [mx.array(value) for value in mx.random.state]
    expected_message = f"synthetic {phase} failure"

    if phase == "setup":
        class SetupFailure(type(model)):
            def trainable_parameters(self):
                noise = mx.random.uniform(shape=(1,), dtype=mx.float32)
                mx.eval(noise)
                raise TypeError(expected_message)

        model = SetupFailure(depth=2)
        before_parameters = {path: mx.array(value) for path, value in tree_flatten(model.parameters())}
        before_state = {path: mx.array(value) for path, value in tree_flatten(model.state)}
        before_random = [mx.array(value) for value in mx.random.state]
        call = lambda: provider(model, batch, lengths)
    elif phase == "forward":
        class FailingLayer(nn.Module):
            def __init__(self, original_layer):
                super().__init__()
                self.shared = original_layer.shared
                self.index = original_layer.index
                self.local = original_layer.local
                self.bias = original_layer.bias

            def __call__(self, hidden_states, *args, **kwargs):
                noise = mx.random.uniform(shape=hidden_states.shape, dtype=hidden_states.dtype)
                mx.eval(noise)
                raise RuntimeError(expected_message)

        failing_layer = FailingLayer(model.model.pipeline_layers[1])
        monkeypatch.setattr(
            model.model,
            "pipeline_layers",
            [model.model.pipeline_layers[0], failing_layer],
        )
        call = lambda: provider(model, batch, lengths)
    elif phase == "tail":
        def fail_loss(*args, **kwargs):
            raise RuntimeError(expected_message)
        monkeypatch.setattr(provider_module, "_loss_from_logits", fail_loss)
        call = lambda: provider(model, batch, lengths)
    elif phase == "reverse":
        def fail_reverse(*args, **kwargs):
            raise RuntimeError(expected_message)
        monkeypatch.setattr(provider_module, "_vjp_segment", fail_reverse)
        call = lambda: provider(model, batch, lengths)
    elif phase == "assembly":
        def fail_assembly(*args, **kwargs):
            raise RuntimeError(expected_message)
        monkeypatch.setattr(provider_module, "_accumulate", fail_assembly)
        call = lambda: provider(model, batch, lengths)
    else:
        original_materialize = provider_module._materialize
        materialize_calls = [0]

        def fail_materialization(*args, **kwargs):
            materialize_calls[0] += 1
            if materialize_calls[0] == 8:
                raise RuntimeError(expected_message)
            return original_materialize(*args, **kwargs)

        monkeypatch.setattr(provider_module, "_materialize", fail_materialization)
        call = lambda: provider(model, batch, lengths)

    with pytest.raises((TypeError, RuntimeError)) as failure:
        call()
    if phase == "setup":
        assert str(failure.value) == expected_message
    else:
        assert str(failure.value) == expected_message
    for path, value in tree_flatten(model.parameters()):
        assert mx.array_equal(value, before_parameters[path]).item(), f"{phase} parameter rollback"
    for path, value in tree_flatten(model.state):
        assert mx.array_equal(value, before_state[path]).item(), f"{phase} model-state rollback"
    assert all(
        mx.array_equal(left, right).item() for left, right in zip(mx.random.state, before_random)
    ), f"{phase} random rollback"


def test_structural_lifetime_oracle(monkeypatch):
    batch, lengths = _batch()
    model = _ToyModel(depth=5)
    observed = []
    original = provider_module._vjp_segment

    def wrapped(*args, **kwargs):
        observed.append((args[4], args[5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(provider_module, "_vjp_segment", wrapped)
    make_ds4_segmented_loss_and_grad(segment_size=2)(model, batch, lengths)
    assert observed == [(4, 5), (2, 4), (0, 2)]
    assert max(end - start for start, end in observed) <= 2


def test_reverse_contributions_released_before_preceding_segment(monkeypatch):
    import gc
    import weakref

    batch, lengths = _batch()
    model = _ToyModel(depth=5)
    original = provider_module._vjp_segment
    previous_refs = []
    release_checks = []

    def wrapped(*args, **kwargs):
        if previous_refs:
            gc.collect()
            release_checks.append(all(reference() is None for reference in previous_refs))
        result = original(*args, **kwargs)
        values = [result[0], *result[1]]
        previous_refs[:] = [weakref.ref(value) for value in values]
        return result

    monkeypatch.setattr(provider_module, "_vjp_segment", wrapped)
    make_ds4_segmented_loss_and_grad(segment_size=2)(model, batch, lengths)
    assert release_checks == [True, True], "completed reverse graph references must be released"



def test_transform_parameter_references_restore_and_release(monkeypatch):
    import gc
    import weakref

    batch, lengths = _batch()
    model = _ToyModel(depth=5)
    entry_ids = tuple(id(value) for _, value in tree_flatten(model.parameters()))
    installed_refs = []
    checks = []
    original_assign = provider_module._assign_parameters

    def assign(model_arg, plan, values):
        original_assign(model_arg, plan, values)
        for _, value in tree_flatten(model_arg.parameters()):
            if id(value) not in entry_ids:
                installed_refs.append(weakref.ref(value))

    def assert_restored(stage):
        gc.collect()
        assert installed_refs, f"{stage} did not observe transform-supplied parameters"
        current_ids = tuple(id(value) for _, value in tree_flatten(model.parameters()))
        checks.append((stage, current_ids == entry_ids, all(reference() is None for reference in installed_refs)))
        assert current_ids == entry_ids, f"{stage} entry parameter identity not restored"
        assert all(reference() is None for reference in installed_refs), (
            f"{stage} transform-supplied parameter references remain model-held"
        )
        installed_refs.clear()

    monkeypatch.setattr(provider_module, "_assign_parameters", assign)
    original_tail = provider_module._vjp_tail
    original_segment = provider_module._vjp_segment
    original_embedding = provider_module._vjp_embedding

    def tail(*args, **kwargs):
        result = original_tail(*args, **kwargs)
        assert_restored("tail")
        return result

    def segment(*args, **kwargs):
        result = original_segment(*args, **kwargs)
        assert_restored(f"segment-{args[4]}-{args[5]}")
        return result

    def embedding(*args, **kwargs):
        result = original_embedding(*args, **kwargs)
        assert_restored("embedding")
        return result

    monkeypatch.setattr(provider_module, "_vjp_tail", tail)
    monkeypatch.setattr(provider_module, "_vjp_segment", segment)
    monkeypatch.setattr(provider_module, "_vjp_embedding", embedding)
    make_ds4_segmented_loss_and_grad(segment_size=2)(model, batch, lengths)
    assert [stage for stage, restored, released in checks] == [
        "tail", "segment-4-5", "segment-2-4", "segment-0-2", "embedding"
    ]
    assert all(restored and released for _, restored, released in checks)


def test_transform_parameter_references_restore_on_transform_failure(monkeypatch):
    import gc
    import weakref

    batch, _ = _batch()
    model = _ToyModel(depth=2)
    plan, _depth, _parameters = provider_module._validate_call_contract(model, batch, _batch()[1])
    inputs = batch[:, :-1]
    boundary = model.model.embed_tokens(inputs)
    boundary = mx.broadcast_to(
        mx.expand_dims(boundary, -2),
        (*boundary.shape[:-1], model.model.hc_mult, model.model.hidden_size),
    )
    mx.eval(boundary)
    entry_ids = tuple(id(value) for _, value in tree_flatten(model.parameters()))
    installed_refs = []
    original_assign = provider_module._assign_parameters
    original_vjp = provider_module.mx.vjp

    def assign(model_arg, plan_arg, values):
        original_assign(model_arg, plan_arg, values)
        for _, value in tree_flatten(model_arg.parameters()):
            if id(value) not in entry_ids:
                installed_refs.append(weakref.ref(value))

    def fail_vjp(*args, **kwargs):
        original_vjp(*args, **kwargs)
        raise RuntimeError("synthetic transform failure")

    monkeypatch.setattr(provider_module, "_assign_parameters", assign)
    monkeypatch.setattr(provider_module.mx, "vjp", fail_vjp)
    with pytest.raises(RuntimeError, match="synthetic transform failure"):
        provider_module._vjp_segment(
            model,
            plan,
            boundary,
            inputs,
            0,
            1,
            mx.ones_like(boundary),
            [mx.array(value) for value in mx.random.state],
            mx.random.state,
        )
    gc.collect()
    assert installed_refs, "failed transform did not install transform-supplied parameters"
    assert tuple(id(value) for _, value in tree_flatten(model.parameters())) == entry_ids
    assert all(reference() is None for reference in installed_refs), (
        "failed transform left transform-supplied parameter references model-held"
    )



def test_nonfinite_passthrough_to_trainer_gate():
    batch, lengths = _batch()
    model = _ToyModel(depth=1)
    model.lm_head.weight = mx.full(model.lm_head.weight.shape, mx.inf)
    result = make_ds4_segmented_loss_and_grad()(model, batch, lengths)
    mx.eval(result[0][0], result[1])
    assert not mx.isfinite(result[0][0]).item()



def test_active_memory_matrix():
    child = textwrap.dedent(
        r'''
        import gc
        import json
        import weakref
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from tests.test_ds4_segmented_loss_and_grad import _ToyModel, _batch
        import tests.test_ds4_segmented_loss_and_grad as test_module
        import ds4_ft_mlx.segmented_loss_and_grad as provider_module
        from ds4_ft_mlx.segmented_loss_and_grad import make_ds4_segmented_loss_and_grad

        depth = int(__import__('os').environ['DS4_MEMORY_DEPTH'])
        size = int(__import__('os').environ['DS4_MEMORY_SIZE'])
        mx.disable_compile()
        batch, lengths = _batch()
        model = _ToyModel(depth=depth)
        mx.eval(model.parameters(), batch, lengths)
        mx.clear_cache()
        baseline = int(mx.get_active_memory())
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        entry_state = [mx.array(value) for value in mx.random.state]
        embedded = model.model.embed_tokens(inputs)
        boundary = mx.broadcast_to(mx.expand_dims(embedded, -2), (*embedded.shape[:-1], model.model.hc_mult, model.model.hidden_size))
        mx.eval(boundary)
        retained_boundaries = [boundary]
        retained_states = [entry_state]
        ranges = [(start, min(start + size, depth)) for start in range(0, depth, size)]
        for start, end in ranges:
            retained_states.append([mx.array(value) for value in mx.random.state])
            for index in range(start, end):
                boundary = model.model.pipeline_layers[index](boundary, input_ids=inputs, cache=None)
            mx.eval(boundary)
            boundary = mx.stop_gradient(boundary)
            mx.eval(boundary)
            retained_boundaries.append(boundary)
        retained_states.append([mx.array(value) for value in mx.random.state])
        logits = model.lm_head(model.model.norm(model.model.hc_head(boundary)))
        loss = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / mask.sum()
        mx.eval(loss)
        retained_states.append([mx.array(value) for value in mx.random.state])
        resident_arrays = list(retained_boundaries)
        resident_arrays.extend(value for state in retained_states for value in state)
        resident_bytes = sum(int(value.nbytes) for value in resident_arrays)
        assert len(retained_boundaries) == len(ranges) + 1
        del retained_boundaries, retained_states, resident_arrays, boundary, embedded, logits, loss
        gc.collect()
        mx.clear_cache()

        calibration_model = _ToyModel(depth=depth)
        mx.eval(calibration_model.parameters(), batch, lengths)
        calibration_plan = provider_module._capture_parameter_plan(calibration_model.trainable_parameters())
        calibration_boundary = calibration_model.model.embed_tokens(inputs)
        calibration_boundary = mx.broadcast_to(mx.expand_dims(calibration_boundary, -2), (*calibration_boundary.shape[:-1], calibration_model.model.hc_mult, calibration_model.model.hidden_size))
        mx.eval(calibration_boundary)
        calibration_end = min(size, depth)
        for index in range(calibration_end):
            calibration_boundary = calibration_model.model.pipeline_layers[index](calibration_boundary, input_ids=inputs, cache=None)
        mx.eval(calibration_boundary)
        calibration_before = int(mx.get_active_memory())
        mx.reset_peak_memory()
        calibration_result = provider_module._vjp_segment(
            calibration_model, calibration_plan, calibration_boundary, inputs, 0, calibration_end,
            mx.ones_like(calibration_boundary), [mx.array(value) for value in mx.random.state],
            mx.random.state,
        )
        mx.eval(calibration_result[0], calibration_result[1])
        isolated_delta = max(0, int(mx.get_peak_memory()) - calibration_before)
        del calibration_result, calibration_model, calibration_boundary
        gc.collect()
        mx.clear_cache()

        model = _ToyModel(depth=depth)
        mx.eval(model.parameters(), batch, lengths)
        test_module.LAYER_EVENTS = []
        events = []
        spans = []
        pending_raw_result_counts = []
        max_pending = [0]
        release_checks = []
        raw_reverse_refs = []
        original_vjp = provider_module.mx.vjp
        original_materialize = provider_module._materialize
        original_segment = provider_module._vjp_segment
        original_embedding = provider_module._vjp_embedding
        def array_ids(value):
            if isinstance(value, mx.array):
                return {id(value)}
            if isinstance(value, dict):
                result = set()
                for child in value.values():
                    result.update(array_ids(child))
                return result
            if isinstance(value, (list, tuple)):
                result = set()
                for child in value:
                    result.update(array_ids(child))
                return result
            return set()
        materialized_ids = []
        def materialize(*values):
            materialized_ids.append(set().union(*(array_ids(value) for value in values)))
            events.append(('eval', len(values)))
            return original_materialize(*values)
        def capture_vjp(*args, **kwargs):
            outputs, gradients = original_vjp(*args, **kwargs)
            raw_reverse_refs[:] = [
                weakref.ref(value) for value in [*outputs, *gradients]
                if isinstance(value, mx.array)
            ]
            pending = int(bool(raw_reverse_refs))
            pending_raw_result_counts.append(pending)
            max_pending[0] = max(max_pending[0], pending)
            return outputs, gradients
        def check_release():
            if raw_reverse_refs:
                gc.collect()
                mx.clear_cache()
                live = any(ref() is not None for ref in raw_reverse_refs)
                pending_raw_result_counts.append(int(live))
                release_checks.append(not live)
                raw_reverse_refs.clear()
        def segment(*args, **kwargs):
            check_release()
            spans.append((args[4], args[5]))
            result = original_segment(*args, **kwargs)
            events.append(('reverse-materialized', args[4], args[5]))
            return result
        def embedding(*args, **kwargs):
            check_release()
            result = original_embedding(*args, **kwargs)
            events.append(('embedding-reverse-materialized',))
            return result
        provider_module.mx.vjp = capture_vjp
        provider_module._materialize = materialize
        provider_module._vjp_segment = segment
        provider_module._vjp_embedding = embedding
        mx.reset_peak_memory()
        result = make_ds4_segmented_loss_and_grad(segment_size=size)(model, batch, lengths)
        final_tree_ids = {id(value) for _, value in tree_flatten(result[1])}
        final_tree_materialized = any(final_tree_ids <= ids for ids in materialized_ids)
        check_release()
        mx.eval(result[0][0], result[0][1], result[1])
        peak = int(mx.get_peak_memory())
        graph_excess = peak - baseline - resident_bytes
        bound = resident_bytes + 2 * isolated_delta + 8_388_608
        payload = {
            'D': depth, 'S': size, 'baseline_active': baseline,
            'resident_boundary_bytes': resident_bytes,
            'maximum_single_segment_graph_delta': isolated_delta,
            'provider_peak_active': peak, 'provider_graph_excess': graph_excess,
            'bound': bound, 'reverse_spans': spans,
            'forward_layer_calls': list(test_module.LAYER_EVENTS),
            'eval_events': len(events), 'max_pending_reverse_results': max_pending[0],
            'pending_raw_result_counts': pending_raw_result_counts,
            'weakref_release_checks': release_checks,
            'final_tree_materialized': final_tree_materialized,
        }
        expected_spans = [(start, min(start + size, depth)) for start in range(0, depth, size)]
        assert all(end - start <= size for start, end in spans), 'differentiated decoder span exceeds segment_size'
        assert spans == list(reversed(expected_spans)), 'reverse segment order mismatch'
        assert max_pending[0] <= 1, 'completed reverse graph references must be released'
        assert all(count <= 1 for count in pending_raw_result_counts), 'pending raw reverse-result count exceeded one'
        assert events and any(event[0] == 'eval' for event in events), 'reverse segment materialization must precede next transform'
        assert list(test_module.LAYER_EVENTS[:depth]) == list(range(depth)), 'forward layer calls missing from lifetime oracle'
        assert release_checks and all(release_checks), 'completed reverse graph references must be released'
        assert final_tree_materialized
        assert peak - baseline <= bound, 'active-memory pinned inequality mismatch'
        print(json.dumps(payload, sort_keys=True))
        ''')
    root = PROJECT_ROOT
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'python-envs/mlx/src'}:{root / 'vendor/mlx-lm'}"
    rows = []
    for depth in (2, 4, 8, 16):
        for size in (1, 2):
            env["DS4_MEMORY_DEPTH"] = str(depth)
            env["DS4_MEMORY_SIZE"] = str(size)
            completed = subprocess.run([sys.executable, "-c", child], cwd=root, env=env, check=False, capture_output=True, text=True)
            if completed.returncode:
                raise AssertionError(completed.stderr or completed.stdout)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            assert payload["D"] == depth and payload["S"] == size
            assert payload["resident_boundary_bytes"] > 48
            assert payload["maximum_single_segment_graph_delta"] >= 0
            assert payload["provider_peak_active"] - payload["baseline_active"] <= payload["bound"]
            rows.append(payload)
    assert len(rows) == 8
    (PROJECT_ROOT / "agent-output/cmux-14-2/memory-r4.log").write_text(
        "ACTIVE_MEMORY_MATRIX PASS 8/8\n" + "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )


def _suffix_boundary_oracle(model, boundary, start, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    def suffix(boundary_value):
        result = boundary_value
        for index in range(start, len(model.model.pipeline_layers)):
            result = model.model.pipeline_layers[index](result, input_ids=inputs, cache=None)
        logits = model.lm_head(model.model.norm(model.model.hc_head(result)))
        return _loss_from_logits_for_test(logits, targets, mask)

    _, gradients = mx.vjp(suffix, [boundary], [mx.ones((), dtype=mx.float32)])
    mx.eval(gradients[0])
    return gradients[0]


def _loss_from_logits_for_test(logits, targets, mask):
    return (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / mask.sum()


def test_boundary_adjoint_matrix():
    batch, lengths = _batch()
    original_tail = provider_module._vjp_tail
    original_segment = provider_module._vjp_segment
    original_embedding = provider_module._vjp_embedding
    try:
        for depth in (1, 2, 3, 5, 8):
            for segment_size in (1, 2, 3, 4):
                mx.random.seed(221 + depth * 10 + segment_size)
                model = _ToyModel(depth=depth)
                inputs = batch[:, :-1]
                embedded = model.model.embed_tokens(inputs)
                initial_boundary = mx.broadcast_to(
                    mx.expand_dims(embedded, -2),
                    (*embedded.shape[:-1], model.model.hc_mult, model.model.hidden_size),
                )
                observed = {"tail": None, "segments": [], "embedding": None}

                def tail(*args, **kwargs):
                    result = original_tail(*args, **kwargs)
                    observed["tail"] = (args[2], result[0])
                    return result

                def segment(*args, **kwargs):
                    result = original_segment(*args, **kwargs)
                    observed["segments"].append((args[2], args[4], args[5], args[6], result[0]))
                    return result

                def embedding(*args, **kwargs):
                    observed["embedding"] = args[3]
                    return original_embedding(*args, **kwargs)

                provider_module._vjp_tail = tail
                provider_module._vjp_segment = segment
                provider_module._vjp_embedding = embedding
                try:
                    make_ds4_segmented_loss_and_grad(segment_size=segment_size)(
                        model, batch, lengths
                    )
                finally:
                    provider_module._vjp_tail = original_tail
                    provider_module._vjp_segment = original_segment
                    provider_module._vjp_embedding = original_embedding

                expected_ranges = [
                    (start, min(start + segment_size, depth))
                    for start in range(0, depth, segment_size)
                ]
                assert [item[1:3] for item in observed["segments"]] == list(reversed(expected_ranges))
                for boundary, start, end, incoming, input_adjoint in observed["segments"]:
                    expected_input = _suffix_boundary_oracle(model, boundary, start, batch, lengths)
                    layer_boundary = boundary
                    for index in range(start, end):
                        layer_boundary = model.model.pipeline_layers[index](
                            layer_boundary, input_ids=inputs, cache=None
                        )
                    expected_output = _suffix_boundary_oracle(
                        model, layer_boundary, end, batch, lengths
                    )
                    mx.eval(expected_input, expected_output, incoming, input_adjoint)
                    assert mx.allclose(
                        incoming, expected_output, atol=1e-5, rtol=1e-5
                    ).item(), "boundary adjoint mismatch"
                    assert mx.allclose(
                        input_adjoint, expected_input, atol=1e-5, rtol=1e-5
                    ).item(), "boundary adjoint mismatch"
                tail_boundary, tail_adjoint = observed["tail"]
                assert mx.allclose(
                    tail_adjoint,
                    _suffix_boundary_oracle(model, tail_boundary, depth, batch, lengths),
                    atol=1e-5,
                    rtol=1e-5,
                ).item(), "boundary adjoint mismatch"
                assert mx.allclose(
                    observed["embedding"],
                    _suffix_boundary_oracle(model, initial_boundary, 0, batch, lengths),
                    atol=1e-5,
                    rtol=1e-5,
                ).item(), "embedded-input adjoint mismatch"
    finally:
        provider_module._vjp_tail = original_tail
        provider_module._vjp_segment = original_segment
        provider_module._vjp_embedding = original_embedding


def test_finite_difference_directional_derivatives():
    batch, lengths = _batch()
    mx.random.seed(222)
    model = _ToyModel(depth=2)
    observed = {}
    original_embedding = provider_module._vjp_embedding

    def embedding(*args, **kwargs):
        observed["input_adjoint"] = args[3]
        return original_embedding(*args, **kwargs)

    provider_module._vjp_embedding = embedding
    try:
        result = make_ds4_segmented_loss_and_grad(segment_size=1)(model, batch, lengths)
    finally:
        provider_module._vjp_embedding = original_embedding
    mx.eval(result[0][0], result[1], observed["input_adjoint"])
    epsilon = 1e-2

    def check_direction(analytic, plus, minus, message):
        finite_difference = (plus - minus) / (2 * epsilon)
        directional = (analytic * direction).sum()
        mx.eval(finite_difference, directional)
        assert abs(float((finite_difference - directional).item())) <= 1e-3, message
        assert abs(float((finite_difference / directional - 1).item())) <= 1e-3, message

    # Independent output-head parameter direction.
    original = mx.array(model.lm_head.weight)
    direction = mx.arange(original.size, dtype=mx.float32).reshape(original.shape) / original.size
    analytic = result[1]["lm_head"]["weight"]
    model.lm_head.weight = original + epsilon * direction
    plus = _default_loss(model, batch, lengths)[0]
    model.lm_head.weight = original - epsilon * direction
    minus = _default_loss(model, batch, lengths)[0]
    model.lm_head.weight = original
    check_direction(analytic, plus, minus, "output-head finite difference mismatch")

    # One shared object is used by every decoder layer and the head; all aliases
    # must receive the same summed gradient, not the last segment contribution.
    shared = model.model.pipeline_layers[0].shared
    shared_original = mx.array(shared)
    direction = mx.arange(shared_original.size, dtype=mx.float32).reshape(shared_original.shape) / shared_original.size
    shared_gradient = next(
        gradient
        for (_, parameter), (_, gradient) in zip(
            tree_flatten(model.trainable_parameters()), tree_flatten(result[1])
        )
        if parameter is shared
    )

    def set_shared(value):
        for layer in model.model.pipeline_layers:
            layer.shared = value
        model.model.hc_head.shared = value

    set_shared(shared_original + epsilon * direction)
    plus = _default_loss(model, batch, lengths)[0]
    set_shared(shared_original - epsilon * direction)
    minus = _default_loss(model, batch, lengths)[0]
    set_shared(shared_original)
    check_direction(shared_gradient, plus, minus, "shared-parameter finite difference mismatch")

    # A smooth perturbation at the embedded boundary independently checks the
    # provider's hidden input adjoint, without treating it as public output.
    inputs = batch[:, :-1]
    embedded = model.model.embed_tokens(inputs)
    initial_boundary = mx.broadcast_to(
        mx.expand_dims(embedded, -2),
        (*embedded.shape[:-1], model.model.hc_mult, model.model.hidden_size),
    )
    direction = mx.arange(initial_boundary.size, dtype=mx.float32).reshape(initial_boundary.shape) / initial_boundary.size

    def boundary_loss(value):
        hidden = value
        for layer in model.model.pipeline_layers:
            hidden = layer(hidden, input_ids=inputs, cache=None)
        logits = model.lm_head(model.model.norm(model.model.hc_head(hidden)))
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        return _loss_from_logits_for_test(logits, targets, mask)

    plus = boundary_loss(initial_boundary + epsilon * direction)
    minus = boundary_loss(initial_boundary - epsilon * direction)
    check_direction(observed["input_adjoint"], plus, minus, "embedded-input finite difference mismatch")


def test_dtype_matrix():
    batch, lengths = _batch()
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        mx.random.seed(223)
        oracle_model = _ToyModel(depth=2)
        mx.random.seed(223)
        provider_model = _ToyModel(depth=2)
        oracle_model.lm_head.weight = oracle_model.lm_head.weight.astype(dtype)
        provider_model.lm_head.weight = provider_model.lm_head.weight.astype(dtype)
        oracle = _monolithic(oracle_model, batch, lengths)
        result = make_ds4_segmented_loss_and_grad(segment_size=2)(provider_model, batch, lengths)
        mx.eval(oracle[0][0], oracle[1], result[0][0], result[1])
        assert result[1]["lm_head"]["weight"].dtype == dtype
        _assert_tree_close(result[1], oracle[1], atol=5e-3, rtol=5e-3)


def _structured_model(depth):
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    args = ModelArgs(
        vocab_size=7,
        hidden_size=4,
        num_hidden_layers=depth,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        qk_rope_head_dim=2,
        index_head_dim=2,
        index_n_heads=1,
        n_routed_experts=1,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=4,
        num_hash_layers=0,
        hc_mult=1,
        layer_types=["sliding_attention"] * depth,
        mlp_layer_types=["moe"] * depth,
        compression_ratio=0,
        expert_dtype="fp4",
        o_groups=1,
    )
    model = Model(args)
    shared = mx.eye(4, dtype=mx.float32) * 0.08
    model.model.layers = [_ToyLayer(shared, 4, index) for index in range(depth)]
    return model


def test_public_deepseek_v4_structured_fixture():
    batch = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
    lengths = mx.array([[1, 4]], dtype=mx.int32)
    original_tail = provider_module._vjp_tail
    original_segment = provider_module._vjp_segment
    original_embedding = provider_module._vjp_embedding
    try:
        for depth, segment_size in ((1, 1), (3, 1), (3, 2), (5, 2)):
            mx.random.seed(224 + depth + segment_size)
            oracle_model = _structured_model(depth)
            mx.random.seed(224 + depth + segment_size)
            provider_model = _structured_model(depth)
            inputs = batch[:, :-1]
            embedded = provider_model.model.embed_tokens(inputs)
            initial_boundary = mx.broadcast_to(
                mx.expand_dims(embedded, -2),
                (*embedded.shape[:-1], provider_model.model.hc_mult, provider_model.model.hidden_size),
            )
            observed = {"tail": None, "segments": [], "embedding": None}

            def tail(*args, **kwargs):
                result = original_tail(*args, **kwargs)
                observed["tail"] = (args[2], result[0])
                return result

            def segment(*args, **kwargs):
                result = original_segment(*args, **kwargs)
                observed["segments"].append((args[2], args[4], args[5], args[6], result[0]))
                return result

            def embedding(*args, **kwargs):
                observed["embedding"] = args[3]
                return original_embedding(*args, **kwargs)

            provider_module._vjp_tail = tail
            provider_module._vjp_segment = segment
            provider_module._vjp_embedding = embedding
            try:
                oracle = _monolithic(oracle_model, batch, lengths)
                result = make_ds4_segmented_loss_and_grad(segment_size=segment_size)(
                    provider_model, batch, lengths
                )
            finally:
                provider_module._vjp_tail = original_tail
                provider_module._vjp_segment = original_segment
                provider_module._vjp_embedding = original_embedding
            mx.eval(oracle[0][0], oracle[1], result[0][0], result[1])
            assert provider_model.model_type == "deepseek_v4_nn"
            assert len(provider_model.model.pipeline_layers) == depth
            assert mx.allclose(result[0][0], oracle[0][0], atol=1e-5, rtol=1e-5).item()
            _assert_tree_close(result[1], oracle[1], atol=1e-5, rtol=1e-5)

            expected_ranges = [
                (start, min(start + segment_size, depth))
                for start in range(0, depth, segment_size)
            ]
            assert [item[1:3] for item in observed["segments"]] == list(reversed(expected_ranges))
            for boundary, start, end, incoming, input_adjoint in observed["segments"]:
                layer_boundary = boundary
                for index in range(start, end):
                    layer_boundary = provider_model.model.pipeline_layers[index](
                        layer_boundary, input_ids=inputs, cache=None
                    )
                expected_input = _suffix_boundary_oracle(
                    provider_model, boundary, start, batch, lengths
                )
                expected_output = _suffix_boundary_oracle(
                    provider_model, layer_boundary, end, batch, lengths
                )
                mx.eval(expected_input, expected_output, incoming, input_adjoint)
                assert mx.allclose(incoming, expected_output, atol=1e-5, rtol=1e-5).item(), "structured boundary adjoint mismatch"
                assert mx.allclose(input_adjoint, expected_input, atol=1e-5, rtol=1e-5).item(), "structured boundary adjoint mismatch"
            tail_boundary, tail_adjoint = observed["tail"]
            assert mx.allclose(
                tail_adjoint,
                _suffix_boundary_oracle(provider_model, tail_boundary, depth, batch, lengths),
                atol=1e-5,
                rtol=1e-5,
            ).item(), "structured boundary adjoint mismatch"
            assert mx.allclose(
                observed["embedding"],
                _suffix_boundary_oracle(provider_model, initial_boundary, 0, batch, lengths),
                atol=1e-5,
                rtol=1e-5,
            ).item(), "structured embedded-input adjoint mismatch"
    finally:
        provider_module._vjp_tail = original_tail
        provider_module._vjp_segment = original_segment
        provider_module._vjp_embedding = original_embedding


def test_nested_dict_list_tuple_parameter_plan():
    nested = {"z": [mx.ones((1,), dtype=mx.float32), (mx.zeros((2,), dtype=mx.float32),)]}
    plan = provider_module._capture_parameter_plan(nested)
    rebuilt = provider_module._rebuild_parameter_tree(plan.template, plan.canonical)
    assert type(rebuilt) is dict and type(rebuilt["z"]) is list and type(rebuilt["z"][1]) is tuple
    assert [path for path, _ in tree_flatten(rebuilt)] == ["z.0", "z.1.0"]


def test_real_provider_accepts_tuple_root_and_restores_structure():
    class TupleRootModel(_ToyModel):
        def trainable_parameters(self):
            parameters = super().trainable_parameters()
            return (parameters["model"], [parameters["lm_head"], parameters["unused"]])

    batch, lengths = _batch()
    mx.random.seed(2301)
    oracle_model = _ToyModel(depth=2)
    mx.random.seed(2301)
    tuple_model = TupleRootModel(depth=2)
    before = tuple_model.trainable_parameters()
    result = make_ds4_segmented_loss_and_grad(segment_size=1)(tuple_model, batch, lengths)
    oracle = _monolithic(oracle_model, batch, lengths)
    mx.eval(result[0][0], result[0][1], result[1], oracle[0][0], oracle[1])
    assert type(result[1]) is tuple
    assert type(result[1][0]) is dict and type(result[1][1]) is list
    assert result[1][1][1].shape == before[1][1].shape
    assert result[0][1].item() == oracle[0][1].item()
    assert mx.allclose(result[0][0], oracle[0][0], atol=1e-5, rtol=1e-5).item()
    for (_, actual), (_, expected) in zip(tree_flatten(result[1]), tree_flatten(oracle[1])):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    assert [path for path, _ in tree_flatten(tuple_model.trainable_parameters())] == [
        path for path, _ in tree_flatten(before)
    ]


def test_path_a_manifest_reproducible_from_index():
    import hashlib
    paths = subprocess.check_output(["git", "ls-files", "agent-output/cmux-13-3b"], cwd=PROJECT_ROOT, text=True).splitlines()
    paths = sorted(paths)
    rows = [f"{hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()}  {path}" for path in paths]
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()
    assert len(rows) == 365
    assert digest == "7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af"


class _SilentTrainerUI:
    instances = []

    def __init__(self, *args, **kwargs):
        self.events = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def advance(self):
        self.events.append(("advance",))

    def report_train(self, *args):
        self.events.append(("train", args))

    def report_val(self, *args):
        self.events.append(("val", args))

    def report_save(self, *args):
        self.events.append(("save", args))

    def val_task(self, *args):
        self.events.append(("val_task", args))
        return self

    def __call__(self, *args, **kwargs):
        self.events.append(("callback", args))


class _SingleProcessGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def _run_real_train(accumulation, *, fail_at=None, with_validation=True):
    trainer_path = PROJECT_ROOT / "vendor/mlx-lm/mlx_lm/tuner/trainer.py"
    assert hashlib.sha256(trainer_path.read_bytes()).hexdigest() == (
        "42e5ee2d13aad0ae31d6ebf63300ed260f186291ef80bf416d3395e40503468f"
    )
    from mlx_lm.tuner import trainer as trainer_module
    from mlx_lm.tuner.trainer import TrainingArgs, train

    batch, lengths = _batch()
    model = _ToyModel(depth=1)
    optimizer = optim.SGD(learning_rate=0.01)
    initial_model = {path: mx.array(value) for path, value in tree_flatten(model.parameters())}
    initial_optimizer = {path: mx.array(value) for path, value in tree_flatten(optimizer.state)}
    train_dataset = ["train"]
    val_dataset = ["val"] if with_validation else None
    calls = []
    callback_events = []
    factory_provider = make_ds4_segmented_loss_and_grad()
    if fail_at is None:
        # Pass the factory-returned callable itself through the real staged
        # trainer; no wrapper may hide provider ownership or invocation.
        provider = factory_provider
    else:
        def provider(current_model, current_batch, current_lengths):
            calls.append((current_batch, current_lengths))
            result = factory_provider(current_model, current_batch, current_lengths)
            if len(calls) == fail_at:
                raise RuntimeError("r4 provider failure")
            return result

    def batches(dataset, **kwargs):
        del kwargs
        while True:
            yield batch, lengths

    class Callback:
        def on_val_loss_report(self, report):
            callback_events.append(("val", dict(report)))

        def on_train_loss_report(self, report):
            callback_events.append(("train", dict(report)))

    args = TrainingArgs(
        batch_size=1,
        iters=2,
        val_batches=1,
        steps_per_report=100 if fail_at is not None else 1,
        steps_per_eval=1,
        steps_per_save=100 if fail_at is not None else 1,
        max_seq_length=8,
        adapter_file="/tmp/story-14-2-r4-adapter.safetensors",
        grad_accumulation_steps=accumulation,
    )
    save_mock = mock.Mock()
    average_mock = mock.Mock(wraps=trainer_module.average_gradients)
    update_mock = mock.Mock(wraps=optimizer.update)
    optimizer.update = update_mock
    accumulator_observations = []
    original_eval = trainer_module.mx.eval

    def eval_spy(*values):
        def collect(value):
            if type(value) is dict and "unused" in value:
                accumulator_observations.append(value)
            elif type(value) in (list, tuple):
                for child in value:
                    collect(child)
        for value in values:
            collect(value)
        return original_eval(*values)

    save_patch = mock.patch.object(trainer_module.mx, "save_safetensors", save_mock)
    average_patch = mock.patch.object(trainer_module, "average_gradients", average_mock)
    eval_patch = mock.patch.object(trainer_module.mx, "eval", eval_spy)
    patches = [
        mock.patch.object(trainer_module, "TrainUI", _SilentTrainerUI),
        mock.patch.object(trainer_module.mx.distributed, "init", return_value=_SingleProcessGroup()),
        save_patch,
        average_patch,
        eval_patch,
    ]
    for patch in patches:
        patch.start()
    caught = None
    try:
        try:
            train(model, optimizer, train_dataset, val_dataset, args, iterate_batches=batches, training_callback=Callback(), loss_and_grad=provider)
        except RuntimeError as exc:
            if fail_at is None:
                raise
            caught = exc
    finally:
        for patch in reversed(patches):
            patch.stop()
    if fail_at is not None:
        assert str(caught) == "r4 provider failure"
    ui = _SilentTrainerUI.instances[-1]
    return (
        model,
        optimizer,
        calls,
        callback_events,
        initial_model,
        initial_optimizer,
        save_mock,
        average_mock,
        update_mock,
        accumulator_observations,
        ui.events,
    )


def test_real_trainer_integration():
    for accumulation in (1, 2):
        (
            model,
            optimizer,
            calls,
            callback_events,
            initial_model,
            _,
            save_patch,
            average_patch,
            update_mock,
            accumulator_observations,
            ui_events,
        ) = _run_real_train(accumulation)
        assert len(calls) == 0, "factory-returned provider was passed unchanged"
        assert save_patch.call_count == 5
        expected_save_paths = [
            "/tmp/story-14-2-r4-adapter.safetensors",
            "/tmp/0000001_adapters.safetensors",
            "/tmp/story-14-2-r4-adapter.safetensors",
            "/tmp/0000002_adapters.safetensors",
            "/tmp/story-14-2-r4-adapter.safetensors",
        ]
        assert [str(call.args[0]) for call in save_patch.call_args_list] == expected_save_paths
        assert all(len(call.args) == 2 and isinstance(call.args[1], dict) for call in save_patch.call_args_list)
        assert average_patch.call_count == (2 if accumulation == 1 else 1)
        assert update_mock.call_count == (2 if accumulation == 1 else 1)
        assert len(accumulator_observations) >= 1
        assert all(isinstance(value, dict) for value in accumulator_observations)
        assert any(not mx.array_equal(value, initial_model[path]).item() for path, value in tree_flatten(model.parameters()))
        callback_kinds = [kind for kind, _ in callback_events]
        assert callback_kinds.count("val") == 2
        assert callback_kinds.count("train") == 2
        for kind, report in callback_events:
            if kind == "val":
                assert set(report) == {"iteration", "val_loss", "val_time"}
                assert report["iteration"] in (0, 1)
            else:
                assert set(report) == {
                    "iteration", "train_loss", "learning_rate", "iterations_per_second",
                    "tokens_per_second", "trained_tokens", "peak_memory",
                }
                assert report["iteration"] in (1, 2)
        assert [event[0] for event in ui_events].count("val") == 2
        assert [event[0] for event in ui_events].count("train") == 2
        assert [event[0] for event in ui_events].count("save") == 2
        assert optimizer.state is not None
        assert model.training is True


def test_real_trainer_failure_after_prior_accumulation():
    (
        model,
        optimizer,
        calls,
        callback_events,
        before_parameters,
        before_optimizer_state,
        save_patch,
        average_patch,
        update_mock,
        accumulator_observations,
        ui_events,
    ) = _run_real_train(2, fail_at=2, with_validation=False)
    assert len(calls) == 2
    for path, value in tree_flatten(model.parameters()):
        assert mx.array_equal(value, before_parameters[path]).item(), "model changed before accumulated update"
    for path, value in tree_flatten(optimizer.state):
        assert mx.array_equal(value, before_optimizer_state[path]).item(), "optimizer changed after failed microbatch"
    assert callback_events == []
    assert ui_events == [("advance",)]
    assert save_patch.call_count == 0
    assert average_patch.call_count == 0
    assert update_mock.call_count == 0
    assert len(accumulator_observations) == 1
    assert isinstance(accumulator_observations[0], dict)


_MUTATION_NODES = {
    "monolithic": "test_mutation_monolithic_reverse",
    "detach-zero": "test_mutation_detached_boundary",
    "local-loss": "test_mutation_end_to_end_gradient",
    "reverse-forward": "test_mutation_reverse_order",
    "skip-duplicate": "test_mutation_decoder_call_order",
    "overwrite-shared": "test_mutation_shared_contribution",
    "omit-unused": "test_mutation_unused_gradient_schema",
    "delayed-materialization": "test_mutation_reverse_materialization",
    "retain-graph": "test_mutation_graph_release",
    "random-resample": "test_mutation_stochastic_replay",
    "random-advance": "test_mutation_random_cadence",
    "cast-reorder": "test_mutation_gradient_schema",
    "token-mask": "test_mutation_masked_loss",
    "sanitize": "test_mutation_nonfinite",
    "extra-result": "test_mutation_public_result",
    "trainer-ownership": "test_mutation_trainer_ownership",
    "per-transform-restore": "test_transform_parameter_references_restore_and_release",
}


def _assert_equivalent(result, model, batch, lengths, message):
    expected = _monolithic(model, batch, lengths)
    mx.eval(result[0][0], result[0][1], result[1], expected[0][0], expected[0][1], expected[1])
    assert result[0][1].item() == expected[0][1].item(), message
    assert mx.allclose(result[0][0], expected[0][0], atol=1e-5, rtol=1e-5).item(), message
    for (_, actual), (_, oracle) in zip(tree_flatten(result[1]), tree_flatten(expected[1])):
        assert actual.shape == oracle.shape and actual.dtype == oracle.dtype, message
        assert mx.allclose(actual, oracle, atol=1e-5, rtol=1e-5).item(), message


def test_mutation_monolithic_reverse():
    model = _ToyModel(depth=5)
    observed = []
    original = provider_module._vjp_segment

    def wrapped(*args, **kwargs):
        observed.append((args[4], args[5]))
        return original(*args, **kwargs)

    provider_module._vjp_segment = wrapped
    try:
        make_ds4_segmented_loss_and_grad(segment_size=2)(model, *_batch())
    finally:
        provider_module._vjp_segment = original
    assert max(end - start for start, end in observed) <= 2, "differentiated decoder span exceeds segment_size"


def test_mutation_detached_boundary():
    batch, lengths = _batch()
    mx.random.seed(3103)
    oracle_model = _ToyModel(depth=3)
    mx.random.seed(3103)
    provider_model = _ToyModel(depth=3)
    result = make_ds4_segmented_loss_and_grad(segment_size=1)(provider_model, batch, lengths)
    _assert_equivalent(result, oracle_model, batch, lengths, "boundary adjoint mismatch")


def test_mutation_end_to_end_gradient():
    batch, lengths = _batch()
    mx.random.seed(3103)
    oracle_model = _ToyModel(depth=3)
    mx.random.seed(3103)
    provider_model = _ToyModel(depth=3)
    result = make_ds4_segmented_loss_and_grad(segment_size=1)(provider_model, batch, lengths)
    _assert_equivalent(result, oracle_model, batch, lengths, "end-to-end gradient mismatch")


def test_mutation_reverse_order():
    model = _ToyModel(depth=5)
    observed = []
    original = provider_module._vjp_segment

    def wrapped(*args, **kwargs):
        observed.append((args[4], args[5]))
        return original(*args, **kwargs)

    provider_module._vjp_segment = wrapped
    try:
        make_ds4_segmented_loss_and_grad(segment_size=2)(model, *_batch())
    finally:
        provider_module._vjp_segment = original
    assert observed == [(4, 5), (2, 4), (0, 2)], "reverse segment order mismatch"


def test_mutation_decoder_call_order():
    batch, lengths = _batch()
    mx.random.seed(3104)
    oracle_model = _ToyModel(depth=4)
    mx.random.seed(3104)
    provider_model = _ToyModel(depth=4)
    result = make_ds4_segmented_loss_and_grad(segment_size=2)(provider_model, batch, lengths)
    _assert_equivalent(result, oracle_model, batch, lengths, "decoder call order mismatch")


def test_mutation_shared_contribution():
    batch, lengths = _batch()
    mx.random.seed(3105)
    oracle_model = _ToyModel(depth=5)
    mx.random.seed(3105)
    provider_model = _ToyModel(depth=5)
    result = make_ds4_segmented_loss_and_grad(segment_size=2)(provider_model, batch, lengths)
    _assert_equivalent(result, oracle_model, batch, lengths, "shared contribution mismatch")


def test_mutation_unused_gradient_schema():
    result = make_ds4_segmented_loss_and_grad()( _ToyModel(depth=2), *_batch())
    unused = result[1].get("unused")
    assert isinstance(unused, mx.array) and mx.array_equal(unused, mx.zeros_like(unused)).item(), "unused gradient leaf mismatch"


def test_mutation_reverse_materialization():
    events = []
    original_materialize = provider_module._materialize
    original_segment = provider_module._vjp_segment

    def materialize(*values):
        events.append("materialize")
        return original_materialize(*values)

    def segment(*args, **kwargs):
        events.append("segment")
        result = original_segment(*args, **kwargs)
        events.append("segment-return")
        return result

    provider_module._materialize = materialize
    provider_module._vjp_segment = segment
    try:
        make_ds4_segmented_loss_and_grad(segment_size=1)(_ToyModel(depth=3), *_batch())
    finally:
        provider_module._materialize = original_materialize
        provider_module._vjp_segment = original_segment
    returns = [index for index, event in enumerate(events) if event == "segment-return"]
    starts = [index for index, event in enumerate(events) if event == "segment"]
    for previous, following in zip(returns, starts[1:]):
        assert "materialize" in events[previous + 1:following], "reverse segment materialization must precede next transform"


def test_mutation_graph_release():
    make_ds4_segmented_loss_and_grad(segment_size=1)(_ToyModel(depth=3), *_batch())
    assert not getattr(provider_module, "_MUTATION_RETAINED", []), "completed reverse graph references must be released"


def _stochastic_model():
    model = _ToyModel(depth=2)
    shared = mx.eye(3) * 0.08
    model.model.pipeline_layers = [_RandomLayer(shared, 3, 0), _RandomLayer(shared, 3, 1)]
    return model


def _state_values():
    return [mx.array(value) for value in mx.random.state]


def _assert_stochastic_cadence(message):
    batch, lengths = _batch()
    mx.random.seed(456)
    expected_model = _stochastic_model()
    mx.random.seed(456)
    model = _stochastic_model()
    mx.random.seed(123)
    expected_forward = expected_model(batch[:, :-1])
    mx.eval(expected_forward)
    expected_state = _state_values()
    mx.random.seed(123)
    expected = _monolithic(expected_model, batch, lengths)
    mx.eval(expected[0][0], expected[1])
    mx.random.seed(123)
    global RANDOM_SAMPLES
    RANDOM_SAMPLES = []
    result = make_ds4_segmented_loss_and_grad(segment_size=1)(model, batch, lengths)
    mx.eval(result[0][0], result[1], *RANDOM_SAMPLES)
    assert len(RANDOM_SAMPLES) >= 4, message
    assert mx.allclose(result[0][0], expected[0][0], atol=1e-5, rtol=1e-5).item(), message
    for (_, actual), (_, oracle) in zip(tree_flatten(result[1]), tree_flatten(expected[1])):
        assert mx.allclose(actual, oracle, atol=1e-5, rtol=1e-5).item(), message
    assert all(mx.array_equal(left, right).item() for left, right in zip(_state_values(), expected_state)), message
    RANDOM_SAMPLES = None


def test_mutation_stochastic_replay():
    _assert_stochastic_cadence("stochastic replay sample mismatch")


def test_mutation_random_cadence():
    _assert_stochastic_cadence("successful random cadence mismatch")


def test_mutation_gradient_schema():
    result = make_ds4_segmented_loss_and_grad(segment_size=2)(_ToyModel(depth=3), *_batch())
    assert [path for path, _ in tree_flatten(result[1])] == [path for path, _ in tree_flatten(_ToyModel(depth=3).trainable_parameters())], "gradient schema mismatch"


def test_mutation_masked_loss():
    batch = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
    lengths = mx.array([[1, 4]], dtype=mx.int32)
    result = make_ds4_segmented_loss_and_grad()(_ToyModel(depth=1), batch, lengths)
    mx.eval(result[0][1])
    assert result[0][1].item() == 4, "masked loss or token mismatch"


def test_mutation_nonfinite():
    model = _ToyModel(depth=1)
    model.lm_head.weight = mx.full(model.lm_head.weight.shape, mx.inf)
    result = make_ds4_segmented_loss_and_grad()(model, *_batch())
    mx.eval(result[0][0])
    assert not mx.isfinite(result[0][0]).item(), "provider must return raw nonfinite values"


def test_mutation_public_result():
    result = make_ds4_segmented_loss_and_grad()(_ToyModel(depth=1), *_batch())
    assert type(result) is tuple and len(result) == 2 and type(result[0]) is tuple and len(result[0]) == 2, "provider result must remain exact two-tuple"


class _MutationSpy:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


def test_mutation_trainer_ownership():
    model = _ToyModel(depth=1)
    spy = _MutationSpy()
    model.optimizer = spy
    make_ds4_segmented_loss_and_grad()(model, *_batch())
    assert spy.calls == 0, "provider invoked trainer-owned collaborator"


def _mutation_replace(name: str, source: str) -> str:
    if name == "per-transform-restore":
        old = "    finally:\n        _restore_entry_parameter_refs(model, plan)\n"
        assert source.count(old) == 3
        return source.replace(old, "    except Exception:\n        raise\n")
    replacements = {
        "monolithic": (
            "            for segment_index in range(len(segment_ranges) - 1, -1, -1):\n",
            "            segment_ranges = [(0, depth)]\n            segment_states = [segment_states[0]]\n            for segment_index in range(len(segment_ranges) - 1, -1, -1):\n",
        ),
        "detach-zero": (
            "                next_incoming = mx.stop_gradient(segment_incoming)\n",
            "                next_incoming = mx.zeros_like(segment_incoming)\n",
        ),
        "local-loss": (
            "                    incoming,\n                    segment_states[segment_index],\n",
            "                    mx.ones_like(incoming),\n                    segment_states[segment_index],\n",
        ),
        "reverse-forward": (
            "            for segment_index in range(len(segment_ranges) - 1, -1, -1):\n",
            "            for segment_index in range(len(segment_ranges)):\n",
        ),
        "skip-duplicate": (
            "        for index in range(start, end):\n            result = layers[index](result, input_ids=inputs, cache=None)\n",
            "        for index in range(start, max(start, end - 1)):\n            result = layers[index](result, input_ids=inputs, cache=None)\n",
        ),
        "overwrite-shared": (
            "        accumulators[index] = mx.stop_gradient(accumulators[index] + contribution)\n",
            "        accumulators[index] = mx.stop_gradient(contribution)\n",
        ),
        "omit-unused": (
            "    if kind == \"leaf\":\n        return accumulators[template[1]]\n",
            "    if kind == \"leaf\":\n        if template[1] == len(accumulators) - 1:\n            return None\n        return accumulators[template[1]]\n",
        ),
        "random-resample": (
            "        _restore_random_state(random_handle, replay_state)\n        outputs, grads = mx.vjp(segment, [boundary, *plan.canonical], [incoming])\n",
            "        _restore_random_state(random_handle, replay_state)\n        boundary = boundary + mx.random.uniform(shape=boundary.shape, dtype=boundary.dtype) * 0.001\n        outputs, grads = mx.vjp(segment, [boundary, *plan.canonical], [incoming])\n",
        ),
        "random-advance": (
            "            _restore_random_state(random_handle, forward_random)\n            return ((loss, token_count), gradients)\n",
            "            return ((loss, token_count), gradients)\n",
        ),
        "cast-reorder": (
            "        return {key: _expand_gradient_tree(child, accumulators) for key, child in template[1]}\n",
            "        return {key: _expand_gradient_tree(child, accumulators) for key, child in sorted(template[1])}\n",
        ),
        "token-mask": (
            "    token_count = mask.sum()\n",
            "    token_count = mx.array(1, dtype=mx.int32)\n",
        ),
        "sanitize": (
            "    loss = ce.astype(mx.float32).sum() / token_count\n",
            "    loss = mx.where(mx.isfinite(ce.astype(mx.float32).sum()), ce.astype(mx.float32).sum(), mx.zeros((), dtype=mx.float32)) / token_count\n",
        ),
        "extra-result": (
            "            return ((loss, token_count), gradients)\n",
            "            return ((loss, token_count), gradients, boundaries)\n",
        ),
        "trainer-ownership": (
            "            return ((loss, token_count), gradients)\n",
            "            getattr(getattr(model, \"optimizer\", None), \"step\", lambda: None)()\n            return ((loss, token_count), gradients)\n",
        ),
    }
    if name == "delayed-materialization":
        start = source.index("def _vjp_segment(")
        end = source.index("def _vjp_embedding(", start)
        segment = source[start:end]
        old = "        _materialize(outputs, grads)\n"
        assert segment.count(old) == 1
        mutated = segment.replace(old, "", 1)
        old_accumulate = "    _materialize(accumulators)\n"
        assert source.count(old_accumulate) == 1
        mutated_source = source[:start] + mutated + source[end:]
        mutated_source = mutated_source.replace(old_accumulate, "", 1)
        old_incoming = "                _materialize(next_incoming)\n"
        assert mutated_source.count(old_incoming) == 1
        cut = mutated_source.rfind(old_incoming)
        return mutated_source[:cut] + mutated_source[cut + len(old_incoming):]
    if name == "retain-graph":
        needle = "import mlx.nn as nn\n"
        assert needle in source
        source = source.replace(needle, needle + "\n_MUTATION_RETAINED = []\n", 1)
        old = "        _materialize(outputs, grads)\n        return grads[0], list(grads[1:])\n"
        new = "        _materialize(outputs, grads)\n        _MUTATION_RETAINED.append((outputs, grads))\n        return grads[0], list(grads[1:])\n"
        assert source.count(old) == 2
        return source.replace(old, new, 1)
    old, new = replacements[name]
    assert source.count(old) == 1, (name, source.count(old))
    return source.replace(old, new, 1)


def _mutation_cli(argv):
    if argv and argv[0] == "--write-mutation":
        name, path = argv[1], argv[2]
        target = pathlib.Path(path)
        base = (PROJECT_ROOT / "python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py").read_text()
        target.write_text(_mutation_replace(name, base))
        return 0
    if argv and argv[0] == "--mutation-node":
        print(_MUTATION_NODES[argv[1]])
        return 0
    if argv and argv[0] == "--verify-mutation-log":
        name, path = argv[1], argv[2]
        markers = {
            "monolithic": "differentiated decoder span exceeds segment_size",
            "detach-zero": "boundary adjoint mismatch",
            "local-loss": "end-to-end gradient mismatch",
            "reverse-forward": "reverse segment order mismatch",
            "skip-duplicate": "decoder call order mismatch",
            "overwrite-shared": "shared contribution mismatch",
            "omit-unused": "unused gradient leaf mismatch",
            "delayed-materialization": "reverse segment materialization must precede next transform",
            "retain-graph": "completed reverse graph references must be released",
            "random-resample": "stochastic replay sample mismatch",
            "random-advance": "successful random cadence mismatch",
            "cast-reorder": "gradient schema mismatch",
            "token-mask": "masked loss or token mismatch",
            "sanitize": "provider must return raw nonfinite values",
            "extra-result": "provider result must remain exact two-tuple",
            "trainer-ownership": "provider invoked trainer-owned collaborator",
            "per-transform-restore": "tail entry parameter identity not restored",
        }
        text = pathlib.Path(path).read_text()
        assert markers[name] in text, (name, markers[name])
        assert not any(marker in text for marker in ("ImportError", "ModuleNotFoundError", "ERROR collecting", "not found"))
        return 0
    raise SystemExit("usage: --write-mutation NAME PATH | --mutation-node NAME | --verify-mutation-log NAME LOG")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        raise SystemExit(_mutation_cli(sys.argv[1:]))
    raise SystemExit(pytest.main([__file__, "-q"]))
