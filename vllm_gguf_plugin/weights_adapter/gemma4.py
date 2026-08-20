# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ..gguf_utils import detect_gguf_multimodal, maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


class Gemma4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Gemma 4 text and multimodal GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "gemma4"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    @staticmethod
    def _map_text_name(name: str) -> str | None:
        top_level = {
            "token_embd": "model.language_model.embed_tokens",
            "output_norm": "model.language_model.norm",
            "output": "lm_head",
        }
        for gguf_prefix, hf_prefix in top_level.items():
            if name == gguf_prefix or name.startswith(f"{gguf_prefix}."):
                return hf_prefix + name.removeprefix(gguf_prefix)

        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if match is None:
            return None
        layer_idx, suffix = match.groups()
        layer = f"model.language_model.layers.{layer_idx}"
        suffix_map = {
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "attn_q_norm": "self_attn.q_norm",
            "attn_k_norm": "self_attn.k_norm",
            "attn_norm": "input_layernorm",
            "post_attention_norm": "post_attention_layernorm",
            "ffn_norm": "pre_feedforward_layernorm",
            "post_ffw_norm": "post_feedforward_layernorm",
            "post_ffw_norm_1": "post_feedforward_layernorm_1",
            "post_ffw_norm_2": "post_feedforward_layernorm_2",
            "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate_inp": "router.proj",
            "ffn_gate_inp.scale": "router.scale",
            "ffn_down_exps.scale": "router.per_expert_scale",
            "layer_output_scale": "layer_scalar",
            "layer_output_scale.weight": "layer_scalar",
            "ffn_gate_up_exps": "experts.0.gate_up_proj",
            "ffn_down_exps": "experts.0.down_proj",
        }
        for gguf_suffix, hf_suffix in sorted(
            suffix_map.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if suffix == gguf_suffix or suffix.startswith(f"{gguf_suffix}."):
                return layer + "." + hf_suffix + suffix.removeprefix(gguf_suffix)
        return None

    @staticmethod
    def _map_vision_name(name: str) -> str | None:
        if name == "v.position_embd.weight":
            return "model.vision_tower.patch_embedder.position_embedding_table"
        direct = {
            "v.std_bias": "model.vision_tower.std_bias",
            "v.std_scale": "model.vision_tower.std_scale",
            "v.patch_embd": "model.vision_tower.patch_embedder.input_proj",
            "mm.input_projection": "model.embed_vision.embedding_projection",
        }
        for gguf_prefix, hf_prefix in direct.items():
            if name == gguf_prefix or name.startswith(f"{gguf_prefix}."):
                return hf_prefix + name.removeprefix(gguf_prefix)

        match = re.fullmatch(r"v\.blk\.(\d+)\.(.+)", name)
        if match is None:
            return None
        layer_idx, suffix = match.groups()
        layer = f"model.vision_tower.encoder.layers.{layer_idx}"
        suffix_map = {
            "attn_q": "self_attn.q_proj.linear",
            "attn_k": "self_attn.k_proj.linear",
            "attn_v": "self_attn.v_proj.linear",
            "attn_out": "self_attn.o_proj.linear",
            "attn_q_norm": "self_attn.q_norm",
            "attn_k_norm": "self_attn.k_norm",
            "ffn_gate": "mlp.gate_proj.linear",
            "ffn_up": "mlp.up_proj.linear",
            "ffn_down": "mlp.down_proj.linear",
            "ln1": "input_layernorm",
            "attn_post_norm": "post_attention_layernorm",
            "ln2": "pre_feedforward_layernorm",
            "ffn_post_norm": "post_feedforward_layernorm",
        }
        for gguf_suffix, hf_suffix in suffix_map.items():
            if suffix == gguf_suffix or suffix.startswith(f"{gguf_suffix}."):
                return layer + "." + hf_suffix + suffix.removeprefix(gguf_suffix)
        return None

    @classmethod
    def map_name(cls, name: str) -> str | None:
        return cls._map_vision_name(name) or cls._map_text_name(name)

    @staticmethod
    def _weight_files(model_path: str) -> list[str]:
        files = [model_path]
        if mm_proj_path := detect_gguf_multimodal(model_path):
            files.append(str(mm_proj_path))
        return files

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        weight_files = self._weight_files(model_path)
        unquantized_modules = {
            mapped.rsplit(".", 1)[0] if mapped.endswith(".weight") else mapped
            for name in get_gguf_unquantized_params(weight_files)
            if (mapped := self.map_name(name)) is not None
        }
        self.load_spec = GGUFLoadSpec(
            weights_source=weight_files,
            unquantized_modules=list(unquantized_modules),
        )
        return self.load_spec

    @staticmethod
    def _split_expert_weights(
        name: str, weight: torch.Tensor
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if ".experts.0.gate_up_proj." in name:
            gate_name = name.replace("gate_up_proj", "gate_proj")
            up_name = name.replace("gate_up_proj", "up_proj")
            if weight.ndim == 0:
                yield gate_name, weight
                yield up_name, weight
                return
            for expert_id, expert_weight in enumerate(weight.unbind()):
                gate_weight, up_weight = expert_weight.chunk(2, dim=0)
                expert = f".experts.{expert_id}."
                yield gate_name.replace(".experts.0.", expert), gate_weight
                yield up_name.replace(".experts.0.", expert), up_weight
            return

        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                yield (
                    name.replace(".experts.0.", f".experts.{expert_id}."),
                    expert_weight,
                )
            return
        yield name, weight

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(self.load_spec.weights_source)
        for raw_name, weight in weights:
            name = self.map_name(raw_name)
            if name is None:
                continue
            if name == "model.vision_tower.patch_embedder.input_proj.weight":
                weight = weight.flatten(1)
            yield from self._split_expert_weights(name, weight)
