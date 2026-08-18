# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_gguf_plugin.weights_adapter import (
    Mellum2GGUFAdapter,
    get_weights_adapter,
)
from vllm_gguf_plugin.weights_adapter.olmoe import (
    build_olmoe_mapper,
    split_olmoe_expert_weights,
)


def test_mellum2_adapter_is_selected():
    config = SimpleNamespace(model_type="mellum")

    adapter = get_weights_adapter(config)

    assert isinstance(adapter, Mellum2GGUFAdapter)


def test_mellum2_gguf_names_map_to_vllm_checkpoint_names():
    mapper = build_olmoe_mapper()

    assert mapper._map_name("token_embd.weight") == "model.embed_tokens.weight"
    assert mapper._map_name("blk.3.attn_q.weight") == (
        "model.layers.3.self_attn.q_proj.weight"
    )
    assert mapper._map_name("blk.3.attn_q_norm.weight") == (
        "model.layers.3.self_attn.q_norm.weight"
    )
    assert mapper._map_name("blk.3.ffn_gate_inp.weight") == (
        "model.layers.3.mlp.gate.weight"
    )
    assert mapper._map_name("blk.3.ffn_down_exps.qweight") == (
        "model.layers.3.mlp.experts.0.down_proj.qweight"
    )
    assert mapper._map_name("output_norm.weight") == "model.norm.weight"
    assert mapper._map_name("output.weight") == "lm_head.weight"


def test_mellum2_stacked_experts_are_split_for_vllm():
    weight = torch.arange(24).reshape(3, 2, 4)

    split_weights = list(
        split_olmoe_expert_weights(
            [("model.layers.0.mlp.experts.0.gate_proj.qweight", weight)]
        )
    )

    assert [name for name, _ in split_weights] == [
        "model.layers.0.mlp.experts.0.gate_proj.qweight",
        "model.layers.0.mlp.experts.1.gate_proj.qweight",
        "model.layers.0.mlp.experts.2.gate_proj.qweight",
    ]
    for expert_id, (_, expert_weight) in enumerate(split_weights):
        torch.testing.assert_close(expert_weight, weight[expert_id])
