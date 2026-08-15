"""Narrow compatibility layer for MaxText's MLA config surface.

The hybrid decoder remains local because upstream MaxText has no Mamba-3 block.
MLA field names and projection names intentionally follow MaxText, making a later
kernel swap mechanical instead of a checkpoint conversion.
"""

from __future__ import annotations

from singularity.config import HybridConfig


def maxtext_mla_overrides(config: HybridConfig) -> dict[str, object]:
    mla = config.mla
    return {
        "attention_type": "mla",
        "base_emb_dim": config.hidden_size,
        "base_num_query_heads": mla.num_heads,
        "base_num_kv_heads": 1,
        "head_dim": mla.qk_nope_head_dim + mla.qk_rope_head_dim,
        "q_lora_rank": mla.q_lora_rank,
        "kv_lora_rank": mla.kv_lora_rank,
        "qk_nope_head_dim": mla.qk_nope_head_dim,
        "qk_rope_head_dim": mla.qk_rope_head_dim,
        "v_head_dim": mla.v_head_dim,
        "mla_qk_head_chunk_size": mla.qk_head_chunk_size,
        "mla_naive_kvcache": False,
        "weight_dtype": config.param_dtype,
        "remat_policy": config.remat_policy,
        "scan_layers": config.scan_layers,
        "rope_max_timescale": int(mla.rope_theta),
    }
