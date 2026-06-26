"""Story 13.3a-3 backward AC for the trainable DeepSeek V4 nn.Module port."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten


def _tiny_quantizable_config() -> dict[str, object]:
    # Same 2-layer hash_moe+moe topology as architecture §13; dimensions are
    # rounded to one 32-wide FP4/quantization block because nn.quantize uses
    # group_size=32 in the trainer contract.
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        expert_dtype="fp4",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _fill_fp4(experts, nibble_byte: int, scale: float) -> None:
    experts.w1_weight = mx.full(experts.w1_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w2_weight = mx.full(experts.w2_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w3_weight = mx.full(experts.w3_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w1_scale = mx.full(experts.w1_scale.shape, scale, dtype=mx.bfloat16)
    experts.w2_scale = mx.full(experts.w2_scale.shape, scale, dtype=mx.bfloat16)
    experts.w3_scale = mx.full(experts.w3_scale.shape, scale, dtype=mx.bfloat16)


def _zero_shared(block) -> None:
    block.shared_experts.gate_proj.weight = mx.zeros_like(block.shared_experts.gate_proj.weight)
    block.shared_experts.up_proj.weight = mx.zeros_like(block.shared_experts.up_proj.weight)
    block.shared_experts.down_proj.weight = mx.zeros_like(block.shared_experts.down_proj.weight)


def _synthesize_fp4_and_hash_tables(model) -> None:
    for idx, layer in enumerate(model.model.layers):
        _fill_fp4(layer.mlp.experts, nibble_byte=0x11 + idx, scale=0.04 + idx * 0.01)
        _zero_shared(layer.mlp)
        if layer.mlp.is_hash:
            table = [[token % layer.mlp.n_routed_experts, (token + 1) % layer.mlp.n_routed_experts] for token in range(model.args.vocab_size)]
            layer.mlp.tid2eid = mx.array(table, dtype=mx.int32)


def _attach_trainer_lora(model) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    nn.quantize(model, group_size=32, bits=4)
    # mlx_lm.lora.train_model freezes the loaded base before converting the
    # selected Linear/QuantizedLinear leaves to LoRA; mirror that contract.
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={
            "rank": 2,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": {"self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"},
        },
    )


def _expected_lora_names() -> set[str]:
    names: set[str] = set()
    for layer_idx in range(2):
        for proj in ("q_a_proj", "q_b_proj", "kv_proj"):
            for leaf in ("lora_a", "lora_b"):
                names.add(f"model.layers.{layer_idx}.self_attn.{proj}.{leaf}")
    return names


def _loss_and_grads(model):
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4]], dtype=mx.int32)

    def loss_for_params(params):
        model.update(params)
        logits = model(input_ids)
        assert logits.shape == (1, 3, model.args.vocab_size)
        return nn.losses.cross_entropy(logits, targets).mean()

    loss, grads = mx.value_and_grad(loss_for_params)(model.trainable_parameters())
    grad_flat = dict(tree_flatten(grads))
    mx.eval(loss, *grad_flat.values())
    return loss, grad_flat


def _nonzero(name: str, grad_flat: dict[str, mx.array]) -> bool:
    grad = grad_flat[name]
    return bool(mx.any(mx.abs(grad.astype(mx.float32)) > 0).item())


def test_deepseek_v4_nn_backward_ac_and_frozen_trainer_contract():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import DeepseekV4FP4Experts, Model, ModelArgs

    model = Model(ModelArgs.from_dict(_tiny_quantizable_config()))
    _synthesize_fp4_and_hash_tables(model)

    routed = model.model.layers[0].mlp.routing_indices(
        mx.ones((1, 3, model.args.hidden_size), dtype=mx.float32),
        input_ids=mx.array([[1, 2, 3]], dtype=mx.int32),
    )
    mx.eval(routed)
    assert routed.tolist() == [[[1, 2], [2, 3], [3, 0]]]

    _attach_trainer_lora(model)

    trainable_names = {name for name, _ in tree_flatten(model.trainable_parameters())}
    assert trainable_names == _expected_lora_names()
    assert all("experts" not in name for name in trainable_names)
    assert all("gate_weight" not in name for name in trainable_names)
    assert all("shared_experts" not in name for name in trainable_names)
    assert all("lm_head" not in name for name in trainable_names)
    for layer in model.model.layers:
        assert isinstance(layer.mlp.experts, DeepseekV4FP4Experts)
        assert tree_flatten(layer.mlp.experts.trainable_parameters()) == []
        assert "gate_weight" not in dict(tree_flatten(layer.mlp.trainable_parameters()))

    loss, grad_flat = _loss_and_grads(model)

    assert loss.shape == ()
    assert bool(mx.isfinite(loss).item())
    assert set(grad_flat) == _expected_lora_names()
    assert grad_flat
    for name, grad in grad_flat.items():
        assert bool(mx.all(mx.isfinite(grad)).item()), name
    assert any(_nonzero(name, grad_flat) for name in grad_flat)

    layer0_names = [name for name in _expected_lora_names() if name.startswith("model.layers.0.")]
    assert layer0_names
    assert any(_nonzero(name, grad_flat) for name in layer0_names)
