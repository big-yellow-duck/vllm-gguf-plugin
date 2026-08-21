# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class Gemma4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Gemma 4 text and multimodal GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "gemma4"

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        return maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )

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

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            if mapped := self.map_name(name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d Gemma 4 GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

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

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        del model_config
        for name, weight in weights:
            if name == "model.vision_tower.patch_embedder.input_proj.weight":
                weight = weight.flatten(1)
            yield from self._split_expert_weights(name, weight)
