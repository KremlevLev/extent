"""Second-stage stable updates for EXP-101--103; fixed EXP-098 warm starts."""
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


def make_step(model, teacher, unused_tx, arm, order, protected_columns, head_dim, *, learning_rate=3e-5, anchor_weight=0.0):
    if arm.objective not in ("ce", "weak_teacher", "self"):
        raise ValueError(f"unregistered safe objective: {arm.objective}")
    tx = optax.chain(stable_clip_by_global_norm(1.0), optax.adam(learning_rate))

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
            return ce + anchor_weight * kl, health

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
    from scripts.m3q_safe_recovery_campaign import aggregate as recovery_aggregate
    answer = recovery_aggregate(spec, result)
    answer["gate_scope"] = "stage2 primary beats control >=0.1 and recovered start at both seeds; no independent replication"
    return answer


def source_initializer(manifest, store):
    def initialize(seed, arm, template, source_hashes):
        entry = manifest["branches"][str(seed)]
        expected = entry["metadata"]
        if expected["contract"]["source_sha256"] != source_hashes:
            raise ValueError("stage2 assembled base differs from pinned EXP098 source")
        restored = store.restore(entry["slot"], expected["contract"], template)
        if restored is None:
            raise FileNotFoundError("missing pinned EXP098 recovery checkpoint")
        payload, meta = restored
        if meta["checkpoint_sha256"] != expected["checkpoint_sha256"] or meta["step"] != 16384:
            raise ValueError("stage2 recovery checkpoint SHA/step mismatch")
        if not bool(coordinates_finite(payload["coordinates"])):
            raise FloatingPointError("nonfinite stage2 source coordinates")
        # Releasing the mask must leave the initial model exactly unchanged.
        config = expected["contract"]["model"]
        m = config["mamba"]
        active = 2 * int(config["hidden_size"] * m["expand"]) + 2 * m["mimo_rank"] * m["groups"] * m["d_state"]
        for name, factors in payload["coordinates"].items():
            if "/in_proj/" in name and not bool(jnp.all(factors["b"][:, active:] == 0)):
                raise ValueError("source has nonzero protected columns; released arm would change its start")
        return payload["coordinates"]
    return initialize


def run(spec, schedule, settings, argv=None):
    import json
    from dataclasses import replace
    from extent.campaign_checkpoint import CampaignCheckpointStore
    from extent.hf_artifact_sync import artifact_config_from_env
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results/EXP-101-103-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = json.loads((root / "results/EXP-101-103-data-preflight.json").read_text(encoding="utf-8"))
    hashes = {name: row["sha256"] for name, row in data.items()}
    files = ["scripts/m3q_plateau_campaign.py", "scripts/m3q_subspace_engine.py",
             "scripts/m3q_safe_recovery_campaign.py", "extent/recovery_subspace.py", "extent/full_model_distillation.py",
             f"scripts/m3q_{spec.name.replace('-', '_')}_campaign.py",
             "results/EXP-101-103-source-manifest.json", "extent/stable_gradient_clip.py",
             "extent/calibration_data.py", "extent/campaign_checkpoint.py",
             "extent/hf_checkpoint_sync.py", "extent/hf_artifact_sync.py"]
    def factory(model, teacher, tx, arm, order, protected_columns, head_dim):
        lr, weight = settings[arm.name]
        return make_step(model, teacher, tx, arm, order, protected_columns, head_dim,
                         learning_rate=lr, anchor_weight=weight)
    # Instantiate read-only pinned source store lazily; plan-only needs no secrets.
    sources = []
    def initialize(seed, arm, template, source_hashes):
        if not sources:
            hub = artifact_config_from_env()
            if hub is None:
                raise ValueError("HF secrets required for stage2 source")
            hub = replace(hub, repo_id=manifest["repo_id"], repo_type="dataset", revision=manifest["revision"])
            # Separate immutable local cache; mark a verified downloaded checkpoint
            # as synced, preventing accidental writes to the pinned revision.
            import tempfile
            class ReadOnlySourceStore(CampaignCheckpointStore):
                def sync(self, slot):
                    return False  # Source revision must never receive writes.
            store = ReadOnlySourceStore(Path(tempfile.gettempdir()) / "extent-stage2-source" / manifest["revision"],
                                             manifest["prefix"], hub)
            sources.append(source_initializer(manifest, store))
        return sources[0](seed, arm, template, source_hashes)
    return run_campaign(spec, argv, step_factory=factory, schedule=schedule,
        initialization_factory=initialize, aggregate_factory=lambda r: aggregate(spec, r),
        contract_extra={
            "evaluate_teacher_baseline": True, "expected_data_sha256": hashes,
            "stage2_source": {"experiment": 98, "arm": "SAFE-PROTECTED-R32", "step": 16384,
                "revision": manifest["revision"], "sha256": {s:e["metadata"]["checkpoint_sha256"] for s,e in manifest["branches"].items()}},
            "stage_transition": "reuse FP32 coordinates on identical frozen ONPOLICY base; fresh Adam in EVERY arm; step counter resets; no fold into BF16",
            "optimizer": {"name": "Adam", "clip": "stable max-scaled global norm 1", "arm_lr_kl_weight": settings},
            "objectives": {"ce": "CE", "weak_teacher": "CE + registered weight * KL(Qwen||student,T=2)*T^2"},
            "selection": "fixed EXP098 final checkpoints; no best checkpoint or seed selection; new start/final locked tests",
            "gate": "all arms complete; primary >=0.1 better than control Wiki NLL and better than recovered start at BOTH seeds",
            "secondary_gate": "primary additionally >=0.1 better than control PG19 NLL and recovered PG19 start at BOTH seeds",
            "additional_implementation_sha256": {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        })
