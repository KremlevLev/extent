"""Weighted full-student CE/KL and diagnostics of the actual Adam direction."""
import math


WEIGHTS = {"ce": (1., 0.), "kl": (0., 1.), "mixed": (.5, .5)}


def make_objective(config, teacher_config):
    import jax
    import jax.numpy as jnp
    from extent.end_to_end_173 import hidden
    from extent.end_to_end_174 import separate_head_loss, student_hidden

    def metrics(parameters, donor, tokens):
        layers, last, embedding, norm = donor
        teacher = jax.lax.stop_gradient(hidden(
            parameters["mamba"], layers, last, embedding, norm, tokens,
            config, teacher_config, original=True,
        ))
        student = student_hidden(parameters, donor, tokens, config, teacher_config)
        return separate_head_loss(
            student[:, :-1], teacher[:, :-1], parameters["qwen"][2],
            embedding, tokens[:, 1:],
        )[1]

    def objective(parameters, donor, tokens, weights):
        values = metrics(parameters, donor, tokens)
        return jnp.vdot(values, weights), values

    return objective


def parameter_groups(parameters):
    layers, last, embedding, norm = parameters["qwen"]
    return dict(
        mamba=parameters["mamba"], qwen_layers=layers, last_attention_block=last,
        embedding_head=embedding, final_norm=norm,
    )


def norm_and_cosine(left, right):
    """Rescale before products to keep large-gradient diagnostics finite."""
    import jax
    import jax.numpy as jnp
    from extent.stable_gradient_clip import scaled_norm_parts

    lm, ls, ll = scaled_norm_parts(left)
    rm, rs, rl = scaled_norm_parts(right)
    ld = jnp.where(lm > 0, lm, 1.)
    rd = jnp.where(rm > 0, rm, 1.)
    dot = sum(jnp.sum((a.astype(jnp.float32) / ld) * (b.astype(jnp.float32) / rd))
              for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    cosine = dot / jnp.maximum(ls * rs, 1e-30)
    return dict(left_log10_norm=ll, right_log10_norm=rl, cosine=jnp.clip(cosine, -1., 1.),
                cosine_defined=(lm > 0) & (rm > 0))


def gradient_diagnostics(ce, kl):
    result = {"all": norm_and_cosine(ce, kl)}
    cg, kg = parameter_groups(ce), parameter_groups(kl)
    result.update({name: norm_and_cosine(cg[name], kg[name]) for name in cg})
    return result


def update_diagnostics(before, after, gradient):
    import jax
    import jax.numpy as jnp

    delta = jax.tree.map(lambda a, b: b - a, before, after)
    result = {"all": norm_and_cosine(gradient, delta)}
    # Casting changes are only a proxy. Norms and dynamics can use FP32 too.
    for name, old in parameter_groups(before).items():
        new = parameter_groups(after)[name]
        count = sum(x.size for x in jax.tree.leaves(old))
        fp32_changed = sum(jnp.sum(a != b) for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)))
        bf16_changed = sum(jnp.sum(a.astype(jnp.bfloat16) != b.astype(jnp.bfloat16))
                           for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)))
        result[name] = dict(
            fp32_changed_fraction=fp32_changed / count,
            bf16_changed_fraction=bf16_changed / count,
            **norm_and_cosine(parameter_groups(gradient)[name], parameter_groups(delta)[name]),
        )
    return result


def plan_matched_steps(remaining_seconds, seconds_per_update, reserve_minutes=18):
    """Choose once, before pilots. Leave time for three starts/saves and continuation."""
    available = remaining_seconds - reserve_minutes * 60 - 3 * 180 - 600
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        raise ValueError("Invalid measured update duration")
    affordable = int(available / (3 * seconds_per_update * 1.15))
    return min(256, max(0, affordable // 32 * 32))


def select_winner(rows, steps):
    complete = [
        row for row in rows if row["returncode"] == 0
        and row["report"].get("status") == "pilot_completed"
        and row["report"].get("step") == steps
        and row["report"].get("final_validation_nll") is not None
        and math.isfinite(row["report"]["final_validation_nll"])
    ]
    return min(complete, key=lambda row: row["report"]["final_validation_nll"], default=None)
