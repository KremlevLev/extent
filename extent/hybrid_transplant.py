"""Auditable full-model Qwen3 -> Extent hybrid transplant planning.

This module deliberately plans the conversion without allocating checkpoint or
model arrays.  A plan is not an importer: it is the guardrail that prevents a
40-layer run from silently combining an experimentally selected mixer with an
incompatible production module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from extent.config import HybridConfig
from extent.qwen_source import QWEN3_14B, QwenSourceSpec
from extent.weight_mapping import direct_qwen_mappings, teacher_qwen_mappings


MAMBA_TRANSPLANT_METHOD = "INIT-K-balanced-qkvo-lift"
MLA_TRANSPLANT_METHOD = "qwen3_rorope_fold1_bkv_activation_pca"
MLA_EVIDENCE_EXPERIMENT = "EXP-019"
MLA_LATENT_RANK = 448
MLA_ROPE_CACHE_WIDTH = 128


@dataclass(frozen=True)
class MixerTransplantAction:
    layer_index: int
    target_mixer: str
    method: str
    source_tensors: tuple[str, ...]
    requires_calibration: bool


@dataclass(frozen=True)
class HybridTransplantPlan:
    source: str
    num_layers: int
    attention_layer_indices: tuple[int, ...]
    mamba_layer_indices: tuple[int, ...]
    attention_fraction: float
    direct_tensor_count: int
    mixer_tensor_count: int
    source_tensor_count: int
    source_coverage_fraction: float
    actions: tuple[MixerTransplantAction, ...]
    mamba_method: str
    mla_method: str
    mla_evidence_experiment: str
    source_gqa_cache_elements_per_token_per_layer: int
    target_mla_cache_elements_per_token_per_layer: int
    retained_attention_cache_reduction_fraction: float
    source_retained_attention_cache_elements_per_token: int
    target_retained_attention_cache_elements_per_token: int
    architecture_ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mixer_sources(layer_index: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}.self_attn"
    return tuple(
        f"{prefix}.{name}"
        for name in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "q_norm.weight",
            "k_norm.weight",
        )
    )


def _mla_compatibility_blockers(
    config: HybridConfig,
    source: QwenSourceSpec,
) -> list[str]:
    """Compare the production attention contract with the frozen EXP-019 one."""
    actual = config.mla
    blockers = []
    expected = {
        "implementation": "rorope_bkv",
        "num_kv_heads": source.num_key_value_heads,
        "kv_lora_rank": MLA_LATENT_RANK,
        "qk_rope_head_dim": source.head_dim,
    }
    for field, value in expected.items():
        observed = getattr(actual, field)
        if observed != value:
            blockers.append(
                f"production MLA {field}={observed}, but {MLA_EVIDENCE_EXPERIMENT} "
                f"requires {field}={value}"
            )
    return blockers


def build_hybrid_transplant_plan(
    config: HybridConfig,
    source: QwenSourceSpec = QWEN3_14B,
) -> HybridTransplantPlan:
    """Build and validate a complete source-tensor ownership plan."""
    if config.num_layers != source.num_hidden_layers:
        raise ValueError(
            f"layer-count mismatch: student={config.num_layers}, source={source.num_hidden_layers}"
        )
    if config.hidden_size != source.hidden_size:
        raise ValueError(
            f"hidden-size mismatch: student={config.hidden_size}, source={source.hidden_size}"
        )
    attention = tuple(config.attention_layer_indices)
    mamba = tuple(config.mamba_layer_indices)
    if len(attention) != 6 or len(mamba) != 34:
        raise ValueError(
            "Extent-14B production plan requires exactly 6 attention and 34 Mamba layers"
        )
    attention_fraction = len(attention) / config.num_layers
    if attention_fraction != 0.15:
        raise ValueError(f"attention fraction must be exactly 0.15, got {attention_fraction}")

    actions = []
    attention_set = set(attention)
    for layer_index in range(config.num_layers):
        is_attention = layer_index in attention_set
        actions.append(
            MixerTransplantAction(
                layer_index=layer_index,
                target_mixer="rorope_bkv_mla" if is_attention else "mamba3_mimo",
                method=MLA_TRANSPLANT_METHOD if is_attention else MAMBA_TRANSPLANT_METHOD,
                source_tensors=_mixer_sources(layer_index),
                requires_calibration=is_attention,
            )
        )

    direct_sources = {entry.source for entry in direct_qwen_mappings(config)}
    mixer_sources = {name for action in actions for name in action.source_tensors}
    all_sources = direct_sources | mixer_sources
    expected_sources = {entry.source for entry in teacher_qwen_mappings(config)}
    if direct_sources & mixer_sources:
        raise AssertionError("direct and mixer mappings claim the same source tensor")
    if all_sources != expected_sources:
        missing = sorted(expected_sources - all_sources)
        extra = sorted(all_sources - expected_sources)
        raise AssertionError(
            f"incomplete source ownership: missing={missing[:3]}, extra={extra[:3]}"
        )

    source_cache = 2 * source.num_key_value_heads * source.head_dim
    target_cache = MLA_LATENT_RANK + MLA_ROPE_CACHE_WIDTH
    blockers = _mla_compatibility_blockers(config, source)
    return HybridTransplantPlan(
        source=f"{source.repo_id}@{source.revision}",
        num_layers=config.num_layers,
        attention_layer_indices=attention,
        mamba_layer_indices=mamba,
        attention_fraction=attention_fraction,
        direct_tensor_count=len(direct_sources),
        mixer_tensor_count=len(mixer_sources),
        source_tensor_count=len(all_sources),
        source_coverage_fraction=len(all_sources) / source.tensor_count,
        actions=tuple(actions),
        mamba_method=MAMBA_TRANSPLANT_METHOD,
        mla_method=MLA_TRANSPLANT_METHOD,
        mla_evidence_experiment=MLA_EVIDENCE_EXPERIMENT,
        source_gqa_cache_elements_per_token_per_layer=source_cache,
        target_mla_cache_elements_per_token_per_layer=target_cache,
        retained_attention_cache_reduction_fraction=1.0 - target_cache / source_cache,
        source_retained_attention_cache_elements_per_token=source_cache * len(attention),
        target_retained_attention_cache_elements_per_token=target_cache * len(attention),
        architecture_ready=not blockers,
        blockers=tuple(blockers),
    )
