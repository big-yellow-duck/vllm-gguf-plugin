# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from .olmoe import OLMoEGGUFAdapter


class Mellum2GGUFAdapter(OLMoEGGUFAdapter):
    """Adapter for Mellum 2 GGUF models.

    Mellum 2 uses the standard GGUF MoE tensor layout also used by OLMoE:
    router weights are stored as ``ffn_gate_inp`` and expert projections are
    stacked in ``ffn_{gate,up,down}_exps`` tensors.  vLLM's Mellum model uses
    Qwen3-MoE-style per-expert checkpoint names, so the inherited adapter maps
    the raw names and splits the stacked tensors before loading.
    """

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "mellum"
