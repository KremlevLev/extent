from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent import HybridForCausalLM
from extent.config import tiny_config, MLAConfig
from extent.qwen3_teacher import Qwen3ForCausalLM, Qwen3TeacherConfig
from extent.initialization import abstract_parameter_tree
from extent.sharding import create_v5e_mesh, named_sharding_tree
from extent.layers.mamba3 import Mamba3MIMO
from extent.composition_diagnostics import compose_parameters, make_full_probe, make_input_shift_probe, json_scalars, gradient_summary


def fixture_models(compute_dtype="float32"):
    cfg = replace(tiny_config(), num_layers=2, attention_layer_indices=(0, 1), compute_dtype=compute_dtype,
        mla=MLAConfig(implementation="qwen3_gqa", num_heads=4, num_kv_heads=2,
                      qk_rope_head_dim=16, rope_original_head_dim=16, v_head_dim=16))
    source = Qwen3TeacherConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_layers=cfg.num_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        tie_word_embeddings=cfg.tie_word_embeddings, rope_theta=cfg.mla.rope_theta,
        rms_norm_eps=cfg.rms_norm_eps, param_dtype=cfg.param_dtype, compute_dtype=cfg.compute_dtype,
        logits_dtype=cfg.logits_dtype, remat_policy=cfg.remat_policy)
    teacher = Qwen3ForCausalLM(source)
    tokens = jnp.arange(4, dtype=jnp.int32)[None]
    params = teacher.init(jax.random.key(1), tokens)["params"]
    return cfg, source, teacher, params, tokens


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_all_gqa_composition_is_exact_and_diagnostics_do_not_mutate_parameters(dtype):
    cfg, source, teacher, p, tokens = fixture_models(dtype)
    model = HybridForCausalLM(cfg)
    abstract = abstract_parameter_tree(model)
    composed = compose_parameters(p, {}, (), abstract, named_sharding_tree(abstract, create_v5e_mesh()))
    expected = teacher.apply({"params": p}, tokens)
    metrics, leaves, states = jax.jit(make_full_probe(model))(composed, tokens, expected)
    assert float(metrics["prediction_kl"]) == pytest.approx(0, abs=1e-5)
    assert float(metrics["student_nll"]) == pytest.approx(float(metrics["teacher_nll"]), abs=0.002 if dtype == "bfloat16" else 1e-5)
    assert len(states) == cfg.num_layers
    assert bool(metrics["grads_finite"])
    record = json_scalars(leaves)
    summary = gradient_summary(record, float(metrics["grad_norm"]))
    assert sum(summary["group_squared_norm_shares"].values()) == pytest.approx(1, rel=1e-5)
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(composed)):
        np.testing.assert_array_equal(a, b)
    json.dumps(summary, allow_nan=False)


def test_same_input_has_same_local_error_and_composition_keeps_teacher_intact():
    cfg, source, teacher, p, tokens = fixture_models()
    cfg = replace(cfg, attention_layer_indices=(1,))
    model = HybridForCausalLM(cfg)
    inputs = jnp.take(p["embed_tokens"]["embedding"], tokens, axis=0)
    dtype = p["embed_tokens"]["embedding"].dtype
    mamba = Mamba3MIMO(cfg.hidden_size, cfg.mamba, dtype=dtype, param_dtype=dtype)
    mp = mamba.init(jax.random.key(4), inputs)["params"]
    abstract = abstract_parameter_tree(model)
    composed = compose_parameters(p, {0: mp}, (0,), abstract, named_sharding_tree(abstract, create_v5e_mesh()))
    assert "self_attn" in p["layers_0"] and "mamba" not in p["layers_0"]
    assert "self_attn" not in composed["layers_0"] and "mamba" in composed["layers_0"]
    tl, ts = teacher.apply({"params": p}, tokens, return_hidden_states=True)
    metrics, leaves, ss = jax.jit(make_full_probe(model))(composed, tokens, tl)
    assert bool(metrics["grads_finite"])
    projection = leaves["layers_0/mamba/in_proj/kernel"]
    assert set(projection["component_norms"]) == {"gate", "value", "B", "C", "dt", "decay", "trapezoid", "angle"}
    squared = sum(float(v) ** 2 for v in projection["component_norms"].values())
    assert squared ** 0.5 == pytest.approx(float(projection["norm"]), rel=1e-5)
    probe = jax.jit(make_input_shift_probe(cfg, source, 0))
    stats = probe(composed["layers_0"], p["layers_0"], inputs, inputs, ts[0], ss[0])
    assert float(stats["input_drift"]["relative_l2"]) == 0
    assert float(stats["teacher_input_error"]["relative_l2"]) == pytest.approx(float(stats["hybrid_input_error"]["relative_l2"]), rel=1e-5)
    json.dumps(json_scalars(stats), allow_nan=False)
    with pytest.raises(ValueError, match="names"):
        compose_parameters(p, {}, (), abstract, named_sharding_tree(abstract, create_v5e_mesh()))


def test_json_keeps_nonfinite_measurements_and_zero_gradient_ranking_safe():
    value = json_scalars({"bad": jnp.inf, "nan": jnp.nan, "finite": jnp.asarray(False)})
    assert value == {"bad": "inf", "nan": "nan", "finite": False}
    json.dumps(value, allow_nan=False)
    assert gradient_summary({"norm/scale": {"norm": 0., "finite": True}}, 0.)["top_parameters"][0]["squared_norm_share"] is None


def test_campaign_runs_from_prepared_checkpoint_saves_and_resumes(tmp_path, monkeypatch):
    from scripts import m3q_input_shift_campaign as campaign
    from extent.campaign_checkpoint import CampaignCheckpointStore
    cfg, source, teacher, p, tokens = fixture_models("bfloat16")
    monkeypatch.setattr(campaign, "SEEDS", (123,))
    monkeypatch.setattr(campaign, "COUNTS", (0, 1))
    monkeypatch.setattr(campaign, "LENGTHS", (4,))
    monkeypatch.setattr(campaign, "WINDOWS", 1)
    monkeypatch.setattr(campaign, "PLACEMENTS", {"UNIFORM": (1,)})
    monkeypatch.setattr(campaign, "load_config", lambda *a: (cfg, None))
    monkeypatch.setattr(campaign, "contract_for", lambda *a: {"test": True})
    monkeypatch.setattr(campaign, "teacher_config_from_spec", lambda *a, **k: source)
    monkeypatch.setattr(campaign, "require_tpu_mesh", lambda: (create_v5e_mesh(), jax.devices()))
    monkeypatch.setattr(campaign, "_ensure_checkpoint", lambda *a: None)
    monkeypatch.setattr(campaign, "stream_teacher_qwen_weights", lambda p, *a: (p, None))
    monkeypatch.setattr(campaign, "load_wikitext2_tokens", lambda count, *a, **k: np.arange(count, dtype=np.int32) % cfg.vocab_size)
    monkeypatch.setattr(campaign, "artifact_config_from_env", lambda: object())
    monkeypatch.setattr(campaign, "restore_artifact", lambda *a: False)
    uploads, notifications = [], []
    monkeypatch.setattr(campaign, "upload_artifact", lambda *a, **k: uploads.append(a[1]))
    monkeypatch.setattr(campaign, "_safe_notify", lambda *a: notifications.append(a[1]))
    store = CampaignCheckpointStore(tmp_path / "state", "test")
    mamba = Mamba3MIMO(cfg.hidden_size, cfg.mamba, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    mp = mamba.init(jax.random.key(7), jnp.ones((1, 4, cfg.hidden_size), jnp.bfloat16))["params"]
    store.save("prep/seed-123/layer-0", {"params": mp},
               contract={"test": True, "seed": 123, "layer": 0, "kind": "prepared_mamba"}, step=2048, metrics={})
    monkeypatch.setattr(campaign, "CampaignCheckpointStore", lambda *a: store)
    args = ["--output-dir", str(tmp_path / "output"), "--state-dir", str(tmp_path / "state"), "--no-telegram"]
    result = campaign.main(args)
    assert result["status"] == "completed" and len(result["cases"]) == 2
    assert uploads and "completed" in notifications[-1]
    stored = json.loads((tmp_path / "output/extent-m3q-input-shift-campaign.json").read_text())
    assert stored["cases"] == result["cases"]
    monkeypatch.setattr(campaign, "require_tpu_mesh", lambda: pytest.fail("completed run used TPU"))
    assert campaign.main(args)["cases"] == result["cases"]
    assert store.metadata("prep/seed-123/layer-0", {"test": True, "seed": 123, "layer": 0, "kind": "prepared_mamba"})["step"] == 2048
