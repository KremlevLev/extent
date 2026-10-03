"""Shared stable CE/weak-anchor updates for EXP-098--100; no global overrides."""
from pathlib import Path
import hashlib
import math

import jax
import jax.numpy as jnp
import optax

from extent.full_model_distillation import causal_cross_entropy, forward_kl
from extent.recovery_subspace import apply_corrections, coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts, stable_clip_by_global_norm
from scripts.m3q_subspace_engine import SEEDS, run_campaign, terminal_result


def make_step(model, teacher, unused_tx, arm, order, protected_columns, head_dim):
    if arm.objective not in ("ce", "weak_teacher", "self"):
        raise ValueError(f"unregistered safe objective: {arm.objective}")
    tx = optax.chain(stable_clip_by_global_norm(1.0), optax.adam(3e-4))

    @jax.jit
    def step(base, coordinates, state, teacher_params, tokens, number):
        target = None
        if arm.objective == "weak_teacher":
            target = jax.lax.stop_gradient(teacher.apply({"params": teacher_params}, tokens))
        elif arm.objective == "self":
            target = jax.lax.stop_gradient(model.apply({"params": base}, tokens))

        def objective(c):
            candidate = apply_corrections(base, c, head_dim=head_dim,
                protected_input_columns=protected_columns if arm.protected else None)
            logits, states = model.apply({"params": candidate}, tokens, return_hidden_states=True)
            ce = causal_cross_entropy(logits, tokens)
            kl = jnp.array(0.0, jnp.float32) if target is None else forward_kl(logits, target, temperature=2.0)
            health = {"logits_finite": jnp.all(jnp.isfinite(logits)), "ce": ce, "anchor_kl": kl,
                "target_finite": jnp.array(True) if target is None else jnp.all(jnp.isfinite(target)),
                "activations": {str(i): {"finite": jnp.all(jnp.isfinite(h)),
                    "max_abs": jnp.max(jnp.abs(h.astype(jnp.float32)))} for i, h in enumerate(states)}}
            return ce + 0.1 * kl, health

        (loss, health), grad = jax.value_and_grad(objective, has_aux=True)(coordinates)
        raw_norm = optax.global_norm(grad)
        maximum, scaled, logarithm = scaled_norm_parts(grad)
        updates, new_state = tx.update(grad, state, coordinates)
        new = optax.apply_updates(coordinates, updates)
        forward = health["logits_finite"] & health["target_finite"] & jnp.all(jnp.stack([
            h["finite"] for h in health["activations"].values()]))
        grad_finite, loss_finite = coordinates_finite(grad), jnp.isfinite(loss)
        current_c, current_s = coordinates_finite(coordinates), coordinates_finite(state)
        proposal_c, proposal_s = coordinates_finite(new), coordinates_finite(new_state)
        proposal_finite = proposal_c & proposal_s
        health.update(loss=loss, loss_finite=loss_finite, forward_finite=forward,
            gradients_finite=grad_finite, current_coordinates_finite=current_c,
            current_optimizer_finite=current_s, safe_coordinates_finite=proposal_c,
            safe_optimizer_finite=proposal_s, safe_proposal_finite=proposal_finite,
            naive_norm=raw_norm, naive_norm_finite=jnp.isfinite(raw_norm),
            grad_max_abs=maximum, scaled_norm=scaled, log10_grad_norm=logarithm,
            norm_only_overflow=forward & loss_finite & grad_finite & current_c & current_s
                & ~jnp.isfinite(raw_norm) & proposal_finite,
            gradient_leaves=jax.tree.map(lambda g: {"finite": jnp.all(jnp.isfinite(g)),
                "max_abs": jnp.max(jnp.abs(g))}, grad))
        finite = forward & loss_finite & grad_finite & current_c & current_s & jnp.isfinite(logarithm) & proposal_finite
        display = jnp.where(maximum == 0, 0.0, jnp.power(10.0, jnp.minimum(logarithm, 38.0)))
        return new, new_state, loss, display, finite, health
    return step


def aggregate(spec, result):
    def metric(row, name):
        value = row.get(name)
        return isinstance(value, (int, float)) and math.isfinite(value)
    comparisons, cross_domain = {}, {}
    for seed in SEEDS:
        rows = result.get("branches", {}).get(str(seed), {})
        a, b = rows.get(spec.primary, {}), rows.get(spec.control, {})
        ready = all(r.get("complete") and metric(r, "locked_test_nll") and metric(r, "start_test_nll")
                    and metric(r, "pg19_test_nll") and metric(r, "start_pg19_nll") for r in (a, b))
        if ready:
            comparisons[str(seed)] = bool(a["locked_test_nll"] <= b["locked_test_nll"] - 0.1
                and a["locked_test_nll"] < a["start_test_nll"]
                and (spec.number != 100 or b["locked_test_nll"] < b["start_test_nll"]))
            cross_domain[str(seed)] = bool(a["pg19_test_nll"] <= b["pg19_test_nll"] - 0.1
                and a["pg19_test_nll"] < a["start_pg19_nll"])
    complete = all(result.get("branches", {}).get(str(s), {}).get(a.name, {}).get("complete", False)
                   for s in SEEDS for a in spec.arms)
    passed = complete and len(comparisons) == len(SEEDS) and all(comparisons.values())
    return {"complete": bool(complete), "terminal": terminal_result(spec, result),
        "seed_wins": comparisons, "pg19_seed_wins": cross_domain,
        "scientific_gate_passed": bool(passed),
        "cross_domain_gate_passed": bool(passed and len(cross_domain) == len(SEEDS) and all(cross_domain.values())),
        "gate_scope": "fresh-range recovery comparison; not random-init superiority or full recovery"}


def run(spec, schedule, argv=None):
    root = Path(__file__).resolve().parents[1]
    files = ("scripts/m3q_safe_recovery_campaign.py", f"scripts/m3q_{spec.name.replace('-', '_')}_campaign.py",
             "extent/stable_gradient_clip.py", "extent/calibration_data.py",
             "extent/campaign_checkpoint.py", "extent/hf_artifact_sync.py", "extent/hf_checkpoint_sync.py")
    # Actual pinned tokenizer/corpus capacity and byte hashes verified locally,
    # without loading model weights. Check BEFORE model allocation on Kaggle.
    hashes = {
        "train": {98: "e80a8699d40cd68f219153129e61d499d62d06d3ce31a0655981b2803234767f",
                  99: "bf35f411b3e1db67ea39b6cea62dcc33217dc14309c1e248ec50ba92067a0d19",
                  100: "ea9f2275a5bc90c70b96803d84638728ceda7af219d5375a96bb31f968edc01a"}[spec.number],
        "validation": "f4383da5e6dc14f231b826e79d0aacb8806db5a0c6ddbd531112475d7a288977",
        "locked_test": "33306342dcbb774423725924893e008989503acc837c3fb07bca964f910f237c",
        "pg19_test": "147d5fbe6fb0d8599ce51107dc27c1cac3c5a84d980ad3840d777c2729bfb134",
    }
    return run_campaign(spec, argv, step_factory=make_step, schedule=schedule,
        aggregate_factory=lambda r: aggregate(spec, r), contract_extra={
            "evaluate_teacher_baseline": True,
            "expected_data_sha256": hashes,
            "optimizer": {"name": "Adam", "lr": 3e-4, "clip": "max-scaled FP32 global norm 1.0",
                          "base_weights": "frozen BF16", "coordinates_and_moments": "FP32"},
            "objectives": {"ce": "CE", "weak_teacher": "CE +0.1*KL(Qwen||student,T=2)*T^2",
                           "self": "CE +0.1*KL(unchanged ONPOLICY||student,T=2)*T^2"},
            "diagnostic_rule": "no nonfinite repair, skipped batches or optimizer reset",
            "gate": "all arms complete; primary beats control >=0.1 Wiki test NLL and improves own start at BOTH seeds"
                    + ("; CE control also improves own start" if spec.number == 100 else ""),
            "secondary_gate": "primary also beats control >=0.1 PG19 NLL and own PG19 start at BOTH seeds",
            "evaluation_freshness": "new packed ranges relative to registered project ranges; same source seeds, not new initialization replication; no universal contamination claim",
            "additional_implementation_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
        })
