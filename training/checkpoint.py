# checkpoint.py
import os
from safetensors import safe_open
import jax
import jax.numpy as jnp
import gc
from config.model_config import QwenConfig

def load_and_shard_weights(
    model_dir,
    weight_map,
    config: QwenConfig,
    sharding_repl_1d,
    sharding_repl_2d,
    sharding_col,
    sharding_row,
    has_qk_norm,
):
    flax_params = {
        "embed_tokens": {},
        "norm": {},
        "lm_head": {},
        "model": {} # Сюда складываем слои декодера
    }

    # =====================================================================
    # 1. ЗАГРУЗКА ГЛОБАЛЬНЫХ ПАРАМЕТРОВ
    # =====================================================================
    
    # Эмбеддинги
    embed_key = "model.embed_tokens.weight"
    embed_file = os.path.join(model_dir, weight_map[embed_key])
    with safe_open(embed_file, framework="np", device="cpu") as f:
        raw_embed = f.get_tensor(embed_key)
    flax_params["embed_tokens"]["embedding"] = jax.device_put(raw_embed, sharding_repl_2d)

    # Финальный RMSNorm
    final_norm_key = "model.norm.weight"
    final_norm_file = os.path.join(model_dir, weight_map[final_norm_key])
    with safe_open(final_norm_file, framework="np", device="cpu") as f:
        raw_final_norm = f.get_tensor(final_norm_key)
    flax_params["norm"]["weight"] = jax.device_put(raw_final_norm, sharding_repl_1d)

    # Языковая голова (LM Head)
    lm_head_key = "lm_head.weight"
    lm_head_file = os.path.join(model_dir, weight_map[lm_head_key])
    with safe_open(lm_head_file, framework="np", device="cpu") as f:
        raw_lm_head = f.get_tensor(lm_head_key)
    flax_params["lm_head"]["kernel"] = jax.device_put(raw_lm_head.T, sharding_col)

    # =====================================================================
    # 2. ПОСЛОЙНАЯ ЗАГРУЗКА ДЕКОДЕРА
    # =====================================================================
    for i in range(config.num_hidden_layers):
        layer_key = f"layers_{i}"
        
        # Обратите внимание: инициализируем структуру ВНУТРИ "model"
        flax_params["model"][layer_key] = {
            "input_layernorm": {},
            "post_attention_layernorm": {},
            "self_attn": {
                "q_proj": {},
                "k_proj": {},
                "v_proj": {},
                "o_proj": {},
            },
            "mlp": {},
        }
        
        if has_qk_norm:
            flax_params["model"][layer_key]["self_attn"]["q_norm"] = {}
            flax_params["model"][layer_key]["self_attn"]["k_norm"] = {}

        # --- 2.1 Нормализации ---
        # Входная нормализация (перед Attention)
        in_norm_key = f"model.layers.{i}.input_layernorm.weight"
        in_norm_file = os.path.join(model_dir, weight_map[in_norm_key])
        with safe_open(in_norm_file, framework="np", device="cpu") as f:
            raw_in_norm = f.get_tensor(in_norm_key)
        flax_params["model"][layer_key]["input_layernorm"]["weight"] = jax.device_put(raw_in_norm, sharding_repl_1d)

        # Пост-внимания нормализация (перед MLP)
        post_norm_key = f"model.layers.{i}.post_attention_layernorm.weight"
        post_norm_file = os.path.join(model_dir, weight_map[post_norm_key])
        with safe_open(post_norm_file, framework="np", device="cpu") as f:
            raw_post_norm = f.get_tensor(post_norm_key)
        flax_params["model"][layer_key]["post_attention_layernorm"]["weight"] = jax.device_put(raw_post_norm, sharding_repl_1d)

        # --- 2.2 Attention проекции ---
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            hf_key = f"model.layers.{i}.self_attn.{proj_name}.weight"
            hf_file = os.path.join(model_dir, weight_map[hf_key])
            with safe_open(hf_file, framework="np", device="cpu") as f:
                raw_weight = f.get_tensor(hf_key)

            if proj_name == "o_proj":
                flax_params["model"][layer_key]["self_attn"][proj_name] = {"kernel": jax.device_put(raw_weight.T, sharding_row)}
            else:
                flax_params["model"][layer_key]["self_attn"][proj_name] = {"kernel": jax.device_put(raw_weight.T, sharding_col)}

        # --- 2.3 QK-нормализация (опционально) ---
        if has_qk_norm:
            q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
            q_norm_file = os.path.join(model_dir, weight_map[q_norm_key])
            with safe_open(q_norm_file, framework="np", device="cpu") as f:
                raw_q_norm = f.get_tensor(q_norm_key)
            flax_params["model"][layer_key]["self_attn"]["q_norm"]["weight"] = jax.device_put(raw_q_norm, sharding_repl_1d)

            k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"
            k_norm_file = os.path.join(model_dir, weight_map[k_norm_key])
            with safe_open(k_norm_file, framework="np", device="cpu") as f:
                raw_k_norm = f.get_tensor(k_norm_key)
            flax_params["model"][layer_key]["self_attn"]["k_norm"]["weight"] = jax.device_put(raw_k_norm, sharding_repl_1d)

        # --- 2.4 MLP проекции ---
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        gate_file = os.path.join(model_dir, weight_map[gate_key])
        with safe_open(gate_file, framework="np", device="cpu") as f:
            raw_gate = f.get_tensor(gate_key)
        flax_params["model"][layer_key]["mlp"]["gate_proj"] = {"kernel": jax.device_put(raw_gate.T, sharding_col)}
        
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        up_file = os.path.join(model_dir, weight_map[up_key])
        with safe_open(up_file, framework="np", device="cpu") as f:
            raw_up = f.get_tensor(up_key)
        flax_params["model"][layer_key]["mlp"]["up_proj"] = {"kernel": jax.device_put(raw_up.T, sharding_col)}
        
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        flax_params["model"][layer_key]["mlp"]["down_proj"] = {"kernel": jax.device_put(raw_down.T, sharding_row)}
        
        gc.collect()      

    return {"params": flax_params}