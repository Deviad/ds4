"""Synthetic DS4 segmented reverse-mode loss provider.

The provider owns only the bounded forward/reverse transform.  The trainer
continues to own validation, accumulation, optimization, reporting, and saves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import mlx.core as mx
from mlx.utils import tree_map
import mlx.nn as nn


_ALLOWED_SEGMENT_SIZES = (1, 2, 3, 4)


@dataclass(frozen=True)
class _ParameterPlan:
    template: Any
    canonical: tuple[mx.array, ...]
    indices: dict[int, int]
    paths: tuple[str, ...]
    destination: Any = None


def _provider_path(path: str, part: Any) -> str:
    if path == "<root>":
        return str(part)
    return f"{path}.{part}"


def _validate_segment_size(segment_size: int) -> int:
    if type(segment_size) is not int:
        raise TypeError("segment_size must be an int in [1, 4]")
    if segment_size not in _ALLOWED_SEGMENT_SIZES:
        raise ValueError("segment_size must be in [1, 4]")
    return segment_size


def _is_array(value: Any) -> bool:
    return isinstance(value, mx.array)


def _capture_parameter_plan(value: Any, path: str = "<root>") -> _ParameterPlan:
    canonical: list[mx.array] = []
    indices: dict[int, int] = {}
    paths: list[str] = []

    def visit(node: Any, node_path: str) -> Any:
        if _is_array(node):
            if node.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise TypeError(
                    "DS4 segmented loss_and_grad requires float16, bfloat16, or "
                    f"float32 trainable parameter '{node_path}'; got {node.dtype}"
                )
            identity = id(node)
            if identity not in indices:
                indices[identity] = len(canonical)
                canonical.append(node)
                paths.append(node_path)
            return ("leaf", indices[identity])
        if type(node) is dict:
            return (
                "dict",
                tuple((key, visit(child, _provider_path(node_path, key))) for key, child in node.items()),
            )
        if type(node) is list:
            return ("list", tuple(visit(child, _provider_path(node_path, index)) for index, child in enumerate(node)))
        if type(node) is tuple:
            return ("tuple", tuple(visit(child, _provider_path(node_path, index)) for index, child in enumerate(node)))
        if node_path != "<root>":
            raise TypeError(
                f"DS4 segmented loss_and_grad trainable parameter '{node_path}' "
                f"must be an mx.array; got {type(node).__name__}"
            )
        raise TypeError(
            f"DS4 segmented loss_and_grad trainable structure at '{node_path}' "
            f"must be dict, list, tuple, or mx.array; got {type(node).__name__}"
        )

    template = visit(value, path)
    if not canonical:
        raise ValueError("DS4 segmented loss_and_grad requires at least one trainable parameter")
    return _ParameterPlan(template, tuple(canonical), indices, tuple(paths))


def _rebuild_parameter_tree(template: Any, values: tuple[mx.array, ...]) -> Any:
    kind = template[0]
    if kind == "leaf":
        return values[template[1]]
    if kind == "dict":
        return {key: _rebuild_parameter_tree(child, values) for key, child in template[1]}
    if kind == "list":
        return [_rebuild_parameter_tree(child, values) for child in template[1]]
    return tuple(_rebuild_parameter_tree(child, values) for child in template[1])


def _copy_tree(value: Any) -> Any:
    if _is_array(value):
        return mx.array(value)
    if type(value) is dict:
        return {key: _copy_tree(child) for key, child in value.items()}
    if type(value) is list:
        return [_copy_tree(child) for child in value]
    if type(value) is tuple:
        return tuple(_copy_tree(child) for child in value)
    return value


def _snapshot_model_state(model: Any) -> Any:
    return _copy_tree(model.state)


def _assign_parameter_tree(
    model: Any,
    plan: _ParameterPlan,
    values: tuple[mx.array, ...],
) -> None:
    """Assign aliased leaves without handing tuple roots to ``Module.update``."""
    current = model.parameters()
    if plan.destination is None:
        replacements = {id(parameter): values[index] for index, parameter in enumerate(plan.canonical)}
        destination = _replace_parameter_values(current, replacements)
    else:
        destination = _rebuild_destination(plan.destination, current, values)
    if type(destination) is dict:
        model.update(destination)
        return
    if type(destination) is list:
        model.update(destination)
        return
    # MLX Module.update rejects tuple roots.  A tuple can still be a public
    # trainable-tree root when its leaves map onto the module's normal dict
    # parameter view; use that view for the actual assignment.
    model.update(dict(destination))


def _replace_parameter_values(node: Any, replacements: dict[int, mx.array]) -> Any:
    if _is_array(node):
        return replacements.get(id(node), node)
    if type(node) is dict:
        return {key: _replace_parameter_values(child, replacements) for key, child in node.items()}
    if type(node) is list:
        return [_replace_parameter_values(child, replacements) for child in node]
    if type(node) is tuple:
        return tuple(_replace_parameter_values(child, replacements) for child in node)
    return node


def _restore_model_state(
    model: Any,
    state: Any,
    parameters: Any,
    plan: _ParameterPlan | None = None,
) -> None:
    # Module.update accepts the flattened state dictionary used by MLX.  The
    # parameter update follows it because state snapshots can omit frozen refs.
    if type(state) is dict:
        model.update(state)
    if plan is None:
        model.update(parameters)
    else:
        _assign_parameter_tree(model, plan, tuple(plan.canonical))


def _capture_destination_template(node: Any, indices: dict[int, int]) -> Any:
    if _is_array(node):
        index = indices.get(id(node))
        return ("leaf", index) if index is not None else ("keep",)
    if type(node) is dict:
        return ("dict", tuple((key, _capture_destination_template(child, indices)) for key, child in node.items()))
    if type(node) is list:
        return ("list", tuple(_capture_destination_template(child, indices) for child in node))
    if type(node) is tuple:
        return ("tuple", tuple(_capture_destination_template(child, indices) for child in node))
    return ("keep",)


def _rebuild_destination(template: Any, current: Any, values: tuple[mx.array, ...]) -> Any:
    kind = template[0]
    if kind == "leaf":
        return values[template[1]]
    if kind == "keep":
        return current
    if kind == "dict":
        return {
            key: _rebuild_destination(child, current[key], values)
            for key, child in template[1]
        }
    if kind == "list":
        return [
            _rebuild_destination(child, current[index], values)
            for index, child in enumerate(template[1])
        ]
    return tuple(
        _rebuild_destination(child, current[index], values)
        for index, child in enumerate(template[1])
    )


def _snapshot_random_state(handle: list[mx.array]) -> list[mx.array]:
    snapshot = [mx.array(leaf) for leaf in handle]
    if snapshot:
        mx.eval(*snapshot)
    return snapshot


def _sync_exposed_random_state(handle: list[mx.array], snapshot: list[mx.array]) -> None:
    handle[:] = [mx.array(leaf) for leaf in snapshot]
    exposed = mx.random.state
    if exposed is not handle:
        exposed[:] = [mx.array(leaf) for leaf in snapshot]


def _restore_random_state(handle: list[mx.array], snapshot: list[mx.array]) -> None:
    _sync_exposed_random_state(handle, snapshot)


def _materialize(*values: Any) -> None:
    leaves: list[mx.array] = []

    def collect(value: Any) -> None:
        if _is_array(value):
            leaves.append(value)
        elif type(value) is dict:
            for child in value.values():
                collect(child)
        elif type(value) in (list, tuple):
            for child in value:
                collect(child)

    for value in values:
        collect(value)
    if leaves:
        mx.eval(*leaves)


def _loss_from_logits(logits: mx.array, targets: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    ce = nn.losses.cross_entropy(logits, targets) * mask
    token_count = mask.sum()
    loss = ce.astype(mx.float32).sum() / token_count
    return loss, token_count


def _assign_parameters(model: Any, plan: _ParameterPlan, values: tuple[mx.array, ...]) -> None:
    _assign_parameter_tree(model, plan, values)


def _restore_entry_parameter_refs(model: Any, plan: _ParameterPlan) -> None:
    """Drop transform tracers from the model before the next reverse stage."""
    _assign_parameter_tree(model, plan, tuple(plan.canonical))


def _vjp_tail(
    model: Any,
    plan: _ParameterPlan,
    boundary: mx.array,
    inputs: mx.array,
    targets: mx.array,
    mask: mx.array,
    replay_state: list[mx.array],
    random_handle: list[mx.array],
) -> tuple[mx.array, list[mx.array]]:
    def tail(boundary_value: mx.array, *values: mx.array) -> mx.array:
        _assign_parameters(model, plan, tuple(values))
        output = model.model.hc_head(boundary_value)
        output = model.model.norm(output)
        logits = model.lm_head(output)
        return _loss_from_logits(logits, targets, mask)[0]

    try:
        _restore_random_state(random_handle, replay_state)
        outputs, grads = mx.vjp(
            tail,
            [boundary, *plan.canonical],
            [mx.ones((), dtype=mx.float32)],
        )
        _materialize(outputs, grads)
        return grads[0], list(grads[1:])
    finally:
        _restore_entry_parameter_refs(model, plan)


def _vjp_segment(
    model: Any,
    plan: _ParameterPlan,
    boundary: mx.array,
    inputs: mx.array,
    start: int,
    end: int,
    incoming: mx.array,
    replay_state: list[mx.array],
    random_handle: list[mx.array],
) -> tuple[mx.array, list[mx.array]]:
    layers = model.model.pipeline_layers

    def segment(boundary_value: mx.array, *values: mx.array) -> mx.array:
        _assign_parameters(model, plan, tuple(values))
        result = boundary_value
        for index in range(start, end):
            result = layers[index](result, input_ids=inputs, cache=None)
        return result

    try:
        _restore_random_state(random_handle, replay_state)
        outputs, grads = mx.vjp(segment, [boundary, *plan.canonical], [incoming])
        _materialize(outputs, grads)
        return grads[0], list(grads[1:])
    finally:
        _restore_entry_parameter_refs(model, plan)


def _vjp_embedding(
    model: Any,
    plan: _ParameterPlan,
    inputs: mx.array,
    incoming: mx.array,
    random_handle: list[mx.array],
    entry_random: list[mx.array],
) -> list[mx.array]:
    def embedding(*values: mx.array) -> mx.array:
        _assign_parameters(model, plan, tuple(values))
        embedded = model.model.embed_tokens(inputs)
        return mx.broadcast_to(
            mx.expand_dims(embedded, -2),
            (*embedded.shape[:-1], model.model.hc_mult, model.model.hidden_size),
        )

    try:
        _restore_random_state(random_handle, entry_random)
        _outputs, grads = mx.vjp(embedding, list(plan.canonical), [incoming])
        _materialize(_outputs, grads)
        return list(grads)
    finally:
        _restore_entry_parameter_refs(model, plan)


def _accumulate(accumulators: list[mx.array], contributions: list[mx.array]) -> None:
    for index, contribution in enumerate(contributions):
        accumulators[index] = mx.stop_gradient(accumulators[index] + contribution)
    _materialize(accumulators)


def _expand_gradient_tree(template: Any, accumulators: list[mx.array]) -> Any:
    kind = template[0]
    if kind == "leaf":
        return accumulators[template[1]]
    if kind == "dict":
        return {key: _expand_gradient_tree(child, accumulators) for key, child in template[1]}
    if kind == "list":
        return [_expand_gradient_tree(child, accumulators) for child in template[1]]
    return tuple(_expand_gradient_tree(child, accumulators) for child in template[1])


def _missing_topology(model: Any) -> list[str]:
    paths = (
        "model.model.embed_tokens",
        "model.model.pipeline_layers",
        "model.model.hc_head",
        "model.model.norm",
        "model.lm_head",
    )
    missing: list[str] = []
    for path in paths:
        current = model
        try:
            for part in path.split(".")[1:]:
                current = getattr(current, part)
        except (AttributeError, TypeError):
            missing.append(path)
            continue
        if current is None:
            missing.append(path)
    return missing


def _validate_call_contract(model: Any, batch: Any, lengths: Any) -> tuple[_ParameterPlan, int, Any]:
    actual = getattr(model, "model_type", None)
    if actual != "deepseek_v4_nn":
        raise TypeError(
            "DS4 segmented loss_and_grad requires model_type='deepseek_v4_nn'; "
            f"got {actual!r}"
        )

    missing = _missing_topology(model)
    if missing:
        raise TypeError(
            "DS4 segmented loss_and_grad model topology missing: " + ",".join(missing)
        )

    if not _is_array(batch):
        raise TypeError(
            "DS4 segmented loss_and_grad batch must be an mx.array; "
            f"got {type(batch).__name__}"
        )
    if batch.dtype != mx.int32:
        raise TypeError(
            "DS4 segmented loss_and_grad batch must have dtype mlx.core.int32; "
            f"got {batch.dtype}"
        )
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] < 2:
        raise ValueError(
            "DS4 segmented loss_and_grad batch must have shape [B, L] with B >= 1 "
            f"and L >= 2; got {tuple(batch.shape)}"
        )

    if not _is_array(lengths):
        raise TypeError(
            "DS4 segmented loss_and_grad lengths must be an mx.array; "
            f"got {type(lengths).__name__}"
        )
    if lengths.dtype != mx.int32:
        raise TypeError(
            "DS4 segmented loss_and_grad lengths must have dtype mlx.core.int32; "
            f"got {lengths.dtype}"
        )
    if lengths.ndim != 2 or lengths.shape[0] != batch.shape[0] or lengths.shape[1] != 2:
        raise ValueError(
            "DS4 segmented loss_and_grad lengths must have shape [B, 2] matching "
            f"batch B={batch.shape[0]}; got {tuple(lengths.shape)}"
        )

    layers = model.model.pipeline_layers
    depth = len(layers)
    if depth == 0:
        raise ValueError("DS4 segmented loss_and_grad requires at least one decoder layer")

    trainable_parameters = model.trainable_parameters()
    plan = _capture_parameter_plan(trainable_parameters)
    # MLX's Module.update accepts its own dict-shaped parameter view, while
    # the provider contract deliberately permits tuple roots and aliases.
    plan = replace(plan, destination=_capture_destination_template(model.parameters(), plan.indices))
    return plan, depth, trainable_parameters


def make_ds4_segmented_loss_and_grad(*, segment_size: int = 1) -> Callable[..., Any]:
    """Build a DS4-specific segmented ``loss_and_grad`` callable."""
    size = _validate_segment_size(segment_size)
    random_handle = mx.random.state

    def provider(model: Any, batch: mx.array, lengths: mx.array):
        entry_random = _snapshot_random_state(random_handle)
        entry_state = _snapshot_model_state(model) if hasattr(model, "state") else None
        training = getattr(model, "training", None)
        plan = None
        entry_parameters = None
        try:
            plan, depth, entry_parameters = _validate_call_contract(model, batch, lengths)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

            boundary = model.model.embed_tokens(inputs)
            boundary = mx.broadcast_to(
                mx.expand_dims(boundary, -2),
                (*boundary.shape[:-1], model.model.hc_mult, model.model.hidden_size),
            )
            _materialize(boundary)
            boundaries = [mx.stop_gradient(boundary)]
            _materialize(boundaries[-1])

            segment_states: list[list[mx.array]] = []
            segment_ranges: list[tuple[int, int]] = []
            for start in range(0, depth, size):
                end = min(start + size, depth)
                segment_ranges.append((start, end))
                segment_states.append(_snapshot_random_state(random_handle))
                for index in range(start, end):
                    boundary = model.model.pipeline_layers[index](
                        boundary, input_ids=inputs, cache=None
                    )
                _materialize(boundary)
                boundary = mx.stop_gradient(boundary)
                _materialize(boundary)
                boundaries.append(boundary)

            tail_state = _snapshot_random_state(random_handle)
            output = model.model.hc_head(boundary)
            output = model.model.norm(output)
            logits = model.lm_head(output)
            loss, token_count = _loss_from_logits(logits, targets, mask)
            _materialize(loss, token_count)
            forward_random = _snapshot_random_state(random_handle)

            accumulators = [mx.zeros_like(parameter) for parameter in plan.canonical]
            incoming, contributions = _vjp_tail(
                model,
                plan,
                boundaries[-1],
                inputs,
                targets,
                mask,
                tail_state,
                random_handle,
            )
            _accumulate(accumulators, contributions)
            next_incoming = mx.stop_gradient(incoming)
            _materialize(next_incoming)
            # The raw reverse result owns the completed segment graph.  Detach
            # the cotangent, then drop every raw contribution before building
            # the preceding segment so graph lifetime stays segment-bounded.
            del contributions, incoming
            incoming = next_incoming

            for segment_index in range(len(segment_ranges) - 1, -1, -1):
                start, end = segment_ranges[segment_index]
                segment_incoming, contributions = _vjp_segment(
                    model,
                    plan,
                    boundaries[segment_index],
                    inputs,
                    start,
                    end,
                    incoming,
                    segment_states[segment_index],
                    random_handle,
                )
                _accumulate(accumulators, contributions)
                next_incoming = mx.stop_gradient(segment_incoming)
                _materialize(next_incoming)
                del contributions, segment_incoming, incoming
                incoming = next_incoming

            embedding_contributions = _vjp_embedding(
                model, plan, inputs, incoming, random_handle, entry_random
            )
            _accumulate(accumulators, embedding_contributions)
            del embedding_contributions, incoming
            gradients = _expand_gradient_tree(plan.template, accumulators)
            _materialize(gradients)
            _restore_model_state(model, entry_state, entry_parameters, plan)
            if training is not None:
                (model.train() if training else model.eval())
            _restore_random_state(random_handle, forward_random)
            return ((loss, token_count), gradients)
        except Exception:
            try:
                if entry_state is not None and entry_parameters is not None:
                    _restore_model_state(model, entry_state, entry_parameters, plan)
                if training is not None:
                    (model.train() if training else model.eval())
            finally:
                _restore_random_state(random_handle, entry_random)
            raise

    return provider


__all__ = ["make_ds4_segmented_loss_and_grad"]
