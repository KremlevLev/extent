from __future__ import annotations

from extent.config import HybridConfig


def parameter_count(config: HybridConfig) -> dict[str, int]:
    h, i, vocab = config.hidden_size, config.intermediate_size, config.vocab_size
    m, a = config.mamba, config.mla
    inner, heads, rank = int(h * m.expand), int(h * m.expand) // m.head_dim, m.mimo_rank
    common_layer = 2 * h + 3 * h * i
    mamba_projection = (
        2 * inner
        + 2 * rank * m.groups * m.d_state
        + 3 * heads
        + int(m.d_state * m.rope_fraction) // 2
    )
    mamba_layer = common_layer + h * mamba_projection + inner * h
    mamba_layer += (
        2 * m.d_state
        + 2 * heads * rank * m.d_state
        + 3 * rank * heads * m.head_dim
        + 2 * heads
    )
    q_width = a.num_heads * (a.qk_nope_head_dim + a.qk_rope_head_dim)
    kv_width = a.num_kv_heads * (a.qk_nope_head_dim + a.v_head_dim)
    q_params = (
        h * a.q_lora_rank + a.q_lora_rank + a.q_lora_rank * q_width
        if a.q_lora_rank
        else h * q_width
    )
    rope_width = a.num_key_rope_heads * a.qk_rope_head_dim
    mla_layer = common_layer + q_params + h * (a.kv_lora_rank + rope_width)
    mla_layer += a.kv_lora_rank * kv_width + a.num_heads * a.v_head_dim * h
    if a.use_kv_latent_norm:
        mla_layer += a.kv_lora_rank
    if a.use_qk_norm:
        mla_layer += 2 * (a.qk_nope_head_dim + a.qk_rope_head_dim)
    embeddings = vocab * h + h + (0 if config.tie_word_embeddings else h * vocab)
    mamba_layers = len(config.mamba_layer_indices) * mamba_layer
    mla_layers = len(config.attention_layer_indices) * mla_layer
    total = embeddings + mamba_layers + mla_layers
    return {
        "embeddings_and_head": embeddings,
        "mamba_layers": mamba_layers,
        "mla_layers": mla_layers,
        "total": total,
    }


def training_state_gib(config: HybridConfig) -> float:
    """BF16 weights + BF16 grads + one BF16 Lion moment, excluding activations."""
    return parameter_count(config)["total"] * 6 / 2**30
