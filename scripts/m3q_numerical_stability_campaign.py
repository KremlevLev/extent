"""EXP-097: same-gradient clipping diagnosis plus paired long recovery."""
from pathlib import Path
import hashlib
import jax
import jax.numpy as jnp
import optax
from extent.full_model_distillation import causal_cross_entropy
from extent.recovery_subspace import apply_corrections, coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts, stable_clip_by_global_norm
from scripts.m3q_subspace_engine import Arm, Campaign, run_campaign, HORIZONS

SPEC = Campaign(97, "numerical-stability", (
    Arm("NAIVE-R8", "INOUT-LORA", 8),
    Arm("SAFE-R8", "INOUT-LORA", 8),
    Arm("SAFE-R8-LOWLR", "INOUT-LORA", 8),
    Arm("NAIVE-PROTECTED-R32", "INOUT-LORA", 32, protected=True),
    Arm("SAFE-PROTECTED-R32", "INOUT-LORA", 32, protected=True),
    Arm("SAFE-PROTECTED-R32-LOWLR", "INOUT-LORA", 32, protected=True),
), "SAFE-R8", "NAIVE-R8")

def make_step(model, teacher, unused_tx, arm, order, protected_columns, head_dim):
    lr = 3e-5 if arm.name.endswith("LOWLR") else 3e-4
    naive_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    safe_tx = optax.chain(stable_clip_by_global_norm(1.0), optax.adam(lr))

    @jax.jit
    def step(base, coordinates, state, teacher_params, tokens, number):
        def objective(c):
            candidate = apply_corrections(base, c, head_dim=head_dim,
                protected_input_columns=protected_columns if arm.protected else None)
            logits, states = model.apply({"params": candidate}, tokens, return_hidden_states=True)
            forward_health = {"logits_finite": jnp.all(jnp.isfinite(logits)),
                "activations": {str(i): {"finite": jnp.all(jnp.isfinite(h)),
                    "max_abs": jnp.max(jnp.abs(h.astype(jnp.float32)))} for i, h in enumerate(states)}}
            return causal_cross_entropy(logits, tokens), forward_health
        (loss, forward_health), grad = jax.value_and_grad(objective, has_aux=True)(coordinates)
        raw_norm = optax.global_norm(grad)
        maximum, scaled, logarithm = scaled_norm_parts(grad)
        naive_updates, naive_state = naive_tx.update(grad, state, coordinates)
        safe_updates, safe_state = safe_tx.update(grad, state, coordinates)
        naive = optax.apply_updates(coordinates, naive_updates)
        safe = optax.apply_updates(coordinates, safe_updates)
        naive_finite = coordinates_finite(naive) & coordinates_finite(naive_state)
        safe_finite = coordinates_finite(safe) & coordinates_finite(safe_state)
        loss_finite, grad_finite = jnp.isfinite(loss), coordinates_finite(grad)
        forward_finite = forward_health["logits_finite"] & jnp.all(jnp.stack([
            h["finite"] for h in forward_health["activations"].values()]))
        norm_only = forward_finite & loss_finite & grad_finite & ~jnp.isfinite(raw_norm) & safe_finite
        health = {"loss": loss, "loss_finite": loss_finite, "gradients_finite": grad_finite,
            "current_coordinates_finite": coordinates_finite(coordinates),
            "current_optimizer_finite": coordinates_finite(state),
            "naive_coordinates_finite": coordinates_finite(naive),
            "naive_optimizer_finite": coordinates_finite(naive_state),
            "safe_coordinates_finite": coordinates_finite(safe),
            "safe_optimizer_finite": coordinates_finite(safe_state),
            "naive_norm_finite": jnp.isfinite(raw_norm), "naive_norm": raw_norm,
            "grad_max_abs": maximum, "scaled_norm": scaled, "log10_grad_norm": logarithm,
            "naive_proposal_finite": naive_finite, "safe_proposal_finite": safe_finite,
            "norm_only_overflow": norm_only,
            "gradient_leaves": jax.tree.map(lambda g: {"finite": jnp.all(jnp.isfinite(g)),
                "max_abs": jnp.max(jnp.abs(g))}, grad)}
        health.update(forward_health)
        health["forward_finite"] = forward_finite
        if arm.name.startswith("NAIVE"):
            new, new_state = naive, naive_state
            finite = forward_finite & loss_finite & grad_finite & jnp.isfinite(raw_norm) & naive_finite
            display_norm = raw_norm
        else:
            new, new_state = safe, safe_state
            finite = forward_finite & loss_finite & grad_finite & jnp.isfinite(logarithm) & safe_finite
            display_norm = jnp.where(maximum == 0, 0.0, jnp.power(10.0, jnp.minimum(logarithm, 38.0)))
        return new, new_state, loss, display_norm, finite, health
    return step

def aggregate(result):
    rows = [result.get("branches", {}).get(str(seed), {}).get(a.name, {})
            for seed in (123, 456) for a in SPEC.arms]
    terminal = all(r.get("complete") or r.get("failed") for r in rows)
    proof = any(r.get("first_norm_only_overflow") for r in rows)
    recovered = {}
    for seed in (123, 456):
        row = result.get("branches", {}).get(str(seed), {}).get("SAFE-R8", {})
        recovered[str(seed)] = bool(row.get("complete") and row["locked_test_nll"] < row["start_test_nll"])
    return {"complete": bool(terminal), "norm_only_overflow_observed": bool(proof),
            "safe_r8_seed_recovery": recovered,
            "scientific_gate_passed": bool(terminal and proof and all(recovered.values())),
            "gate_scope": "numerical-overflow diagnosis + safe rank8 recovery, not initializer superiority"}

def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    return run_campaign(SPEC, argv, step_factory=make_step, aggregate_factory=aggregate,
        contract_extra={
            "evaluate_teacher_baseline": True,
            "gate": "all branches terminal; same-gradient norm-only overflow observed; SAFE-R8 completes and improves start in both seeds",
            "optimizer": {"name": "adam", "lr": "3e-4; LOWLR arms3e-5", "clip": "NAIVE optax global norm / SAFE scaled FP32 norm", "moments": "FP32"},
            "diagnostic_rule": "no nonfinite-gradient replacement, no skipped batches, no Adam reset; retain last finite checkpoint",
            "additional_implementation_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                for p in ("scripts/m3q_numerical_stability_campaign.py", "extent/stable_gradient_clip.py")},
            "registered_horizons": list(HORIZONS),
        })

if __name__ == "__main__":
    main()
