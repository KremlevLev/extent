"""Two student trainable scopes with one immutable Qwen teacher."""
import jax
import jax.numpy as jnp

from extent.end_to_end_173 import hidden, make_grad as frozen_gradient


def separate_head_loss(student, teacher, student_head, teacher_head, labels, tile=32):
    """Exact CE and forward KL with independent student and teacher heads."""
    width = student.shape[-1]
    count = labels.size
    pad = (-count) % tile

    def pack(features):
        return jnp.pad(features.reshape(-1, width), ((0, pad), (0, 0))).reshape(-1, tile, width)

    targets = jnp.pad(labels.reshape(-1), (0, pad)).reshape(-1, tile)
    valid = (jnp.arange(count + pad) < count).reshape(-1, tile)
    student_head = student_head.astype(student.dtype)
    teacher_head = teacher_head.astype(teacher.dtype)

    def body(totals, inputs):
        student_features, teacher_features, target, mask = inputs
        student_logp = jax.nn.log_softmax((student_features @ student_head.T).astype(jnp.float32))
        teacher_logp = jax.lax.stop_gradient(
            jax.nn.log_softmax((teacher_features @ teacher_head.T).astype(jnp.float32))
        )
        ce = -jnp.take_along_axis(student_logp, target[:, None], axis=-1)[:, 0]
        kl = jnp.sum(jnp.exp(teacher_logp) * (teacher_logp - student_logp), axis=-1)
        return totals + jnp.stack((jnp.sum(ce * mask), jnp.sum(kl * mask))), None

    totals, _ = jax.lax.scan(
        jax.checkpoint(body), jnp.zeros(2, jnp.float32),
        (pack(student), pack(jax.lax.stop_gradient(teacher)), targets, valid),
    )
    metrics = totals / count
    return 0.5 * jnp.sum(metrics), metrics


def student_hidden(parameters, donor, tokens, config, teacher_config):
    layers, last, embedding, norm = parameters.get("qwen", donor)
    return hidden(parameters["mamba"], layers, last, embedding, norm, tokens, config, teacher_config)


def make_gradient(config, teacher_config, full):
    if not full:
        gradient = frozen_gradient(config, teacher_config)

        def frozen(parameters, donor, tokens):
            layers, last, embedding, norm = donor
            loss, grad, metrics = gradient(parameters["mamba"], layers, embedding, last, norm, tokens)
            return loss, {"mamba": grad}, metrics

        return frozen

    def objective(parameters, donor, tokens):
        layers, last, embedding, norm = donor
        teacher = jax.lax.stop_gradient(hidden(
            parameters["mamba"], layers, last, embedding, norm, tokens, config, teacher_config, original=True,
        ))
        student = student_hidden(parameters, donor, tokens, config, teacher_config)
        return separate_head_loss(student[:, :-1], teacher[:, :-1], parameters["qwen"][2], embedding, tokens[:, 1:])

    def full_gradient(parameters, donor, tokens):
        (loss, metrics), grad = jax.value_and_grad(objective, has_aux=True)(parameters, donor, tokens)
        return loss, grad, metrics

    return full_gradient
