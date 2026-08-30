import jax
import jax.numpy as jnp

from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.weight_mapping import direct_qwen_mappings, teacher_qwen_mappings
from scripts.qwen17_transplant_atlas_campaign import (
    ARMS,
    CHECKPOINTS,
    SEEDS,
    aggregate_atlas,
)
from scripts.qwen_bridge_ablation import homotopy_alpha


def _layer_result():
    scales = {
        "CONTROL-RANDOM": 1.00,
        "CONTROL-FLAT-QKVO": 0.95,
        "BALANCED-RANK-LIFT": 0.80,
        "M3Q-EXACT-LINEAR": 0.75,
        "M3Q-EXACT-COSINE": 0.70,
        "M3Q-EXACT-DELAYED-COSINE": 0.65,
    }
    seeds = {}
    for seed in SEEDS:
        arms = {}
        for arm in ARMS:
            scale = scales[arm]
            evaluations = {
                str(step): {
                    "decoder_output": {"relative_l2": scale * (1.0 - step / 20_000)}
                }
                for step in CHECKPOINTS
            }
            arms[arm] = {"recovery": {"complete": True, "evaluations": evaluations}}
        seeds[str(seed)] = {"arms": arms}
    return {"complete": True, "passed": True, "seeds": seeds}


def test_qwen17_base_mapping_omits_tied_lm_head():
    source = teacher_config_from_spec(QWEN3_1_7B_BASE)
    direct_sources = {entry.source for entry in direct_qwen_mappings(source)}
    teacher_entries = teacher_qwen_mappings(source)

    assert source.tie_word_embeddings is True
    assert "lm_head.weight" not in direct_sources
    assert len(teacher_entries) == QWEN3_1_7B_BASE.tensor_count == 310


def test_tied_teacher_reuses_embedding_without_lm_head_parameter():
    source = teacher_config_from_spec(
        QWEN3_1_7B_BASE,
        param_dtype="float32",
        compute_dtype="float32",
        remat_policy="none",
    )
    tiny = source.__class__(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        param_dtype="float32",
        compute_dtype="float32",
        remat_policy="none",
    )
    params = Qwen3ForCausalLM(tiny).init(
        jax.random.key(0), jnp.zeros((1, 4), jnp.int32)
    )["params"]

    assert "lm_head" not in params
    assert Qwen3ForCausalLM(tiny).apply(
        {"params": params}, jnp.zeros((1, 4), jnp.int32)
    ).shape == (1, 4, 128)


def test_atlas_aggregation_measures_final_and_curve_advantage():
    aggregate = aggregate_atlas({"0": _layer_result()})
    layer = aggregate["layers"]["0"]

    assert aggregate["completed_layers"] == 1
    assert aggregate["layer_wins"] == 1
    assert aggregate["selected_homotopy_schedule"] == "M3Q-EXACT-DELAYED-COSINE"
    assert aggregate["exploratory_advancement_gate_passed"] is False
    assert layer["final_selected_seed_wins_over_exact"] == 3
    assert layer["final_selected_relative_improvement_over_exact"] > 0
    assert layer["selected_homotopy_auc_improvement_over_exact"] > 0


def test_homotopy_schedules_are_bounded_monotonic_and_finish_deployable():
    for schedule in ("linear", "cosine", "delayed-cosine"):
        values = [homotopy_alpha(step, 100, schedule) for step in range(100)]
        assert all(0.0 <= value <= 1.0 for value in values)
        assert all(left <= right for left, right in zip(values, values[1:]))
        assert values[-1] == 1.0

    assert homotopy_alpha(0, 100, "delayed-cosine") == 0.0
    assert homotopy_alpha(14, 100, "delayed-cosine") == 0.0
    assert homotopy_alpha(69, 100, "delayed-cosine") == 1.0
