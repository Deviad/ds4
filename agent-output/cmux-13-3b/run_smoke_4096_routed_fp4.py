#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import mlx_lm.lora as lora
import mlx_lm.tuner.trainer as trainer
from mlx_lm.tuner.callbacks import TrainingCallback

GRAPH_LIMIT = 400_000_000_000
STOP_PEAK = 340_000_000_000
EXPECTED_GRADIENT_KEYS = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_proj",
)


def telemetry(event, **extra):
    record = {
        "event": event,
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
        **extra,
    }
    print("DS4_SMOKE " + json.dumps(record, sort_keys=True), flush=True)
    return record


def arrays(tree):
    return [(name, value) for name, value in tree_flatten(tree) if hasattr(value, "shape")]


class SmokeMonitor(TrainingCallback):
    def __init__(self):
        self.model = None
        self.before = None
        self.first_train_report = True

    def snapshot_trainable(self):
        flat = arrays(self.model.trainable_parameters())
        copies = {name: value + mx.zeros_like(value) for name, value in flat}
        mx.eval(*copies.values())
        self.before = copies
        telemetry("trainable_snapshot", tensor_count=len(copies))

    def on_val_loss_report(self, info):
        loss = float(info["val_loss"])
        state = telemetry(
            "validation",
            iteration=int(info["iteration"]),
            val_loss=loss,
            val_time=float(info["val_time"]),
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"STOP: non-finite validation loss {loss}")
        if state["peak_bytes"] >= STOP_PEAK:
            raise RuntimeError(
                f"STOP: validation peak {state['peak_bytes']} >= {STOP_PEAK} before backward"
            )
        if self.before is None:
            self.snapshot_trainable()

    def on_train_loss_report(self, info):
        loss = float(info["train_loss"])
        state = telemetry(
            "train_step",
            iteration=int(info["iteration"]),
            train_loss=loss,
            reported_peak_gb=float(info["peak_memory"]),
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"STOP: non-finite training loss {loss}")
        if state["peak_bytes"] >= STOP_PEAK:
            raise RuntimeError(
                f"STOP: post-step peak {state['peak_bytes']} >= {STOP_PEAK}"
            )
        if self.first_train_report:
            current = dict(arrays(self.model.trainable_parameters()))
            common = sorted(set(current).intersection(self.before or {}))
            changed = []
            for name in common:
                flag = mx.any(current[name] != self.before[name])
                mx.eval(flag)
                if bool(flag.item()):
                    changed.append(name)
            expected_changed = [
                name for name in changed if any(key in name for key in EXPECTED_GRADIENT_KEYS)
            ]
            telemetry(
                "first_update",
                compared_tensors=len(common),
                changed_tensors=len(changed),
                expected_changed_tensors=len(expected_changed),
            )
            if not expected_changed:
                raise RuntimeError("STOP: first optimizer step changed no expected LoRA tensor")
            self.first_train_report = False


monitor = SmokeMonitor()
original_value_and_grad = trainer.nn.value_and_grad


def monitored_value_and_grad(model, loss):
    value_and_grad = original_value_and_grad(model, loss)
    first = True

    def wrapped(*args, **kwargs):
        nonlocal first
        value, gradients = value_and_grad(*args, **kwargs)
        if first:
            flat = arrays(gradients)
            values = [value for _, value in flat]
            if values:
                mx.eval(*values)
            finite = []
            nonzero = []
            expected = []
            expected_nonzero = []
            for name, gradient in flat:
                finite_flag = mx.all(mx.isfinite(gradient))
                nonzero_flag = mx.any(gradient != 0)
                mx.eval(finite_flag, nonzero_flag)
                is_finite = bool(finite_flag.item())
                is_nonzero = bool(nonzero_flag.item())
                finite.append(is_finite)
                nonzero.append(is_nonzero)
                if any(key in name for key in EXPECTED_GRADIENT_KEYS):
                    expected.append(name)
                    if is_nonzero:
                        expected_nonzero.append(name)
            state = telemetry(
                "first_backward",
                gradient_tensors=len(flat),
                finite_gradient_tensors=sum(finite),
                nonzero_gradient_tensors=sum(nonzero),
                expected_gradient_tensors=len(expected),
                expected_nonzero_gradient_tensors=len(expected_nonzero),
            )
            if not flat or not all(finite):
                raise RuntimeError("STOP: first backward produced empty or non-finite gradients")
            if not expected or not expected_nonzero:
                raise RuntimeError("STOP: first backward produced no non-zero expected LoRA gradient")
            if state["peak_bytes"] >= STOP_PEAK:
                raise RuntimeError(
                    f"STOP: first-backward peak {state['peak_bytes']} >= {STOP_PEAK}"
                )
            first = False
        return value, gradients

    return wrapped


original_train_model = lora.train_model


def monitored_train_model(args, model, train_set, valid_set, training_callback=None):
    monitor.model = model
    return original_train_model(
        args,
        model,
        train_set,
        valid_set,
        training_callback=monitor,
    )


os.environ["TOKENIZERS_PARALLELISM"] = "true"
mx.disable_compile()
mx.set_memory_limit(GRAPH_LIMIT)
trainer.nn.value_and_grad = monitored_value_and_grad
lora.train_model = monitored_train_model
lora.get_reporting_callbacks = lambda *args, **kwargs: monitor
telemetry(
    "startup",
    pid=os.getpid(),
    graph_limit_bytes=GRAPH_LIMIT,
    stop_peak_bytes=STOP_PEAK,
    argv=sys.argv[1:],
)
lora.main()

adapter_path = Path(sys.argv[sys.argv.index("--adapter-path") + 1])
adapter_file = adapter_path / "adapters.safetensors"
if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
    raise RuntimeError(f"STOP: missing final adapter {adapter_file}")
weights = mx.load(str(adapter_file))
flat_weights = list(weights.items())
if not flat_weights:
    raise RuntimeError("STOP: saved adapter is empty")
finite = []
nonzero = []
for _, value in flat_weights:
    finite_flag = mx.all(mx.isfinite(value))
    nonzero_flag = mx.any(value != 0)
    mx.eval(finite_flag, nonzero_flag)
    finite.append(bool(finite_flag.item()))
    nonzero.append(bool(nonzero_flag.item()))
if not all(finite) or not any(nonzero):
    raise RuntimeError("STOP: saved adapter contains non-finite values or no non-zero tensor")
telemetry(
    "adapter_validated",
    adapter_file=str(adapter_file),
    tensor_count=len(flat_weights),
    nonzero_tensor_count=sum(nonzero),
    adapter_bytes=adapter_file.stat().st_size,
)
print("DS4_SMOKE_DONE", flush=True)
