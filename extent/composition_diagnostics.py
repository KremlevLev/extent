"""Read-only probes: full-model gradients and matched-input decoder errors."""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from flax.core import unfreeze

from extent.full_model_distillation import causal_cross_entropy, forward_kl, full_model_eval_metrics
from extent.optimizer import global_norm_fp32, gradient_health
from extent.model import HybridDecoderLayer
from extent.qwen3_teacher import Qwen3DecoderLayer


def amplitude(x):
    return global_norm_fp32({"x": x}) / math.sqrt(x.size)


def error_stats(actual, target):
    error = actual.astype(jnp.float32) - target.astype(jnp.float32)
    return {"error_rms": amplitude(error), "target_rms": amplitude(target),
            "relative_l2": global_norm_fp32({"x": error}) / jnp.maximum(global_norm_fp32({"x": target}), 1e-12)}


def compose_parameters(teacher_params, prepared, replaced, abstract, layout):
    """Use Qwen weights verbatim except specified mixer subtrees; no random init."""
    params = unfreeze(teacher_params)
    for layer in replaced:
        key = f"layers_{layer}"
        params[key].pop("self_attn")
        params[key]["mamba"] = prepared[layer]
    flat = traverse_util.flatten_dict(params)
    shapes = traverse_util.flatten_dict(abstract)
    if flat.keys() != shapes.keys():
        raise ValueError("composed parameter names do not match production model")
    for path, value in flat.items():
        if value.shape != shapes[path].shape or value.dtype != shapes[path].dtype:
            raise ValueError(f"composed shape/dtype mismatch: {'/'.join(path)}")
    return jax.tree.map(jax.device_put, params, layout)


def make_full_probe(model):
    def probe(params, tokens, teacher_logits):
        def objective(p):
            logits, states = model.apply({"params": p}, tokens, return_hidden_states=True)
            loss = forward_kl(logits, teacher_logits, temperature=2.0) + 0.1 * causal_cross_entropy(logits, tokens)
            return loss, (states, full_model_eval_metrics(logits, teacher_logits, tokens, temperature=2.0))
        (loss, (states, metrics)), grads = jax.value_and_grad(objective, has_aux=True)(params)
        metrics = dict(metrics, loss=loss, **gradient_health(grads))
        leaves = {"/".join(path): {"norm": global_norm_fp32({"x": value}),
                  "max_abs": jnp.max(jnp.abs(value.astype(jnp.float32))),
                  "finite": jnp.all(jnp.isfinite(value))}
                  for path, value in traverse_util.flatten_dict(grads).items()}
        cfg = model.config.mamba
        inner = int(model.config.hidden_size * cfg.expand)
        heads = inner // cfg.head_dim
        sizes = (inner, inner, cfg.mimo_rank * cfg.groups * cfg.d_state,
                 cfg.mimo_rank * cfg.groups * cfg.d_state, heads, heads, heads,
                 int(cfg.d_state * cfg.rope_fraction) // 2)
        for path, value in traverse_util.flatten_dict(grads).items():
            if path[-3:] == ("mamba", "in_proj", "kernel"):
                if sum(sizes) != value.shape[-1]:
                    raise ValueError("Mamba packed projection schema changed")
                parts, offset = {}, 0
                for name, size in zip(("gate", "value", "B", "C", "dt", "decay", "trapezoid", "angle"), sizes):
                    parts[name] = global_norm_fp32({"x": value[..., offset:offset+size]})
                    offset += size
                leaves["/".join(path)]["component_norms"] = parts
        return metrics, leaves, states
    return probe


def make_input_shift_probe(config, source, mamba_layer_index):
    student = HybridDecoderLayer(config, mamba_layer_index)
    teacher = Qwen3DecoderLayer(source)
    def probe(student_params, teacher_params, teacher_input, hybrid_input, teacher_output, hybrid_output):
        positions = jnp.arange(teacher_input.shape[1], dtype=jnp.int32)[None]
        local_student = student.apply({"params": student_params}, teacher_input, positions, None)
        conditional_student = student.apply({"params": student_params}, hybrid_input, positions, None)
        local_teacher = teacher.apply({"params": teacher_params}, teacher_input, positions, None)
        conditional_teacher = teacher.apply({"params": teacher_params}, hybrid_input, positions, None)
        # Subtract the residual input from BOTH outputs. Otherwise a large shared
        # identity path can hide the error of the decoder's learned contribution.
        teacher_contribution = local_teacher.astype(jnp.float32) - teacher_input.astype(jnp.float32)
        conditional_contribution = conditional_teacher.astype(jnp.float32) - hybrid_input.astype(jnp.float32)
        return {
            "teacher_input_error": error_stats(local_student.astype(jnp.float32) - teacher_input, teacher_contribution),
            "hybrid_input_error": error_stats(conditional_student.astype(jnp.float32) - hybrid_input, conditional_contribution),
            "teacher_execution_error": error_stats(local_teacher, teacher_output),
            "student_execution_error": error_stats(conditional_student, hybrid_output),
            "input_drift": error_stats(hybrid_input, teacher_input),
            "teacher_input_rms": amplitude(teacher_input), "hybrid_input_rms": amplitude(hybrid_input),
            "teacher_output_rms": amplitude(teacher_output), "hybrid_output_rms": amplitude(hybrid_output),
        }
    return probe


def json_scalars(tree):
    """Keep genuine nonfinite diagnostics visible without invalid JSON numbers."""
    if isinstance(tree, dict):
        return {k: json_scalars(v) for k, v in tree.items()}
    value = np.asarray(tree).item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def gradient_summary(leaves, total_norm):
    ranking = []
    groups = {}
    for path, row in leaves.items():
        norm = row["norm"]
        share = (norm / total_norm) ** 2 if isinstance(norm, (int, float)) and isinstance(total_norm, (int, float)) and total_norm > 0 else None
        group = "/".join(path.split("/")[:2]) if path.startswith("layers_") else path.split("/")[0]
        if share is not None:
            groups[group] = groups.get(group, 0.0) + share
        ranking.append(dict(path=path, **row, squared_norm_share=share))
    ranking.sort(key=lambda r: r["squared_norm_share"] if r["squared_norm_share"] is not None else -1, reverse=True)
    return {"top_parameters": ranking[:20], "group_squared_norm_shares": groups}
