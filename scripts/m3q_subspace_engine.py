"""Resumable, paired full-depth low-rank recovery campaigns (094--096).

Base model weights are frozen. Only FP32 correction coordinates and Adam states
are checkpointed. Shared controls are deterministic consistency checks, not
additional independent seeds. These are recovery tests, not MLA experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import causal_cross_entropy, forward_kl, hidden_delta_relative_mse
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifacts_together
from extent.initialization import initialize_sharded_parameters
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.recovery_subspace import initialize_corrections, apply_corrections, coordinate_count, coordinates_finite
from extent.sharding import batch_sharding, create_v5e_mesh
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072, configured_contract
from scripts import m3q_sequential_recovery_campaign as sequential
from scripts import m3q_onpolicy_anchor_campaign as exp091
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint
from scripts.qwen_extended_horizon_campaign import _safe_notify

SEEDS = (123, 456)
HORIZONS = (2048, 4096, 8192)
LENGTH = 256
TRAIN_WINDOWS = 8192
DATA = (("train", "train", 8_388_608, TRAIN_WINDOWS),
        ("validation", "validation", 49_152, 32),
        ("locked_test", "test", 32_768, 64))
LR = 3e-4

@dataclass(frozen=True)
class Arm:
    name: str
    subspace: str
    rank: int
    objective: str = "ce"
    protected: bool = False

@dataclass(frozen=True)
class Campaign:
    number: int
    name: str
    arms: tuple[Arm, ...]
    primary: str
    control: str

    @property
    def prefix(self):
        return f"experiments/exp{self.number:03d}-{self.name}"

    @property
    def stem(self):
        return f"extent-m3q-{self.name}"

def contract_for(spec, config):
    root = Path(__file__).resolve().parents[1]
    return {
        "protocol": f"exp{spec.number:03d}-{spec.name}-v1",
        "campaign": asdict(spec), "model": json.loads(json.dumps(asdict(config))),
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_protocol": EXP072["PROTOCOL"],
        "replacement_order": list(configured_contract(config)["replacement_order"]),
        "seeds": list(SEEDS), "horizons": list(HORIZONS), "sequence_length": LENGTH,
        "data": [list(row) for row in DATA],
        "student_training_tokens": len(SEEDS) * len(spec.arms) * HORIZONS[-1] * LENGTH,
        "optimizer": {"name": "adam", "learning_rate": LR, "clip_global_norm": 1.0,
                      "coordinates_and_moments": "float32", "base_weights": "frozen BF16"},
        "objectives": {"ce": "true-token CE", "teacher": "KL(T=2)+0.1CE",
                       "self": "CE+0.1KL(T=2) to unchanged assembled hybrid",
                       "delta_then_teacher": "steps 1..2048: decoder-delta relative MSE +0.1CE; later KL(T=2)+0.1CE"},
        "delta_layers": "replaced layers >0 only; teacher/student see their own upstream streams",
        "protected": "in_proj dt/raw_a/trap/angle projection columns unchanged; B/C/x/z may adapt",
        "selection": "no early stopping/model selection; validation diagnostic; locked test only at8192",
        "gate": "registered primary beats control by >=0.1 test NLL and unchanged start at BOTH seeds",
        "comparison": "equal student tokens, NOT equal parameters/FLOPs; repeated rank8 CE controls are not independent evidence",
        "implementation_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in ("scripts/m3q_subspace_engine.py", "extent/recovery_subspace.py",
                      "extent/full_model_distillation.py")},
    }

def coordinate_plan(config, arm, layer_count):
    inner = int(config.hidden_size * config.mamba.expand)
    heads = inner // config.mamba.head_dim
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    packed = active + 3 * heads + int(config.mamba.d_state * config.mamba.rope_fraction) // 2
    if arm.subspace == "HEAD-GAIN":
        count = layer_count * heads
    elif arm.subspace == "MLP-READOUT-LORA":
        count = layer_count * arm.rank * (config.intermediate_size + config.hidden_size)
    else:
        count = layer_count * arm.rank * (inner + config.hidden_size)
        if arm.subspace == "INOUT-LORA":
            count += layer_count * arm.rank * (config.hidden_size + packed)
    effective = count - (layer_count * arm.rank * (packed - active) if arm.protected else 0)
    return {"allocated_coordinates": count, "effective_coordinates": effective,
            "replicated_adapter_adam_bytes_per_device": count * 4 * 3,
            "adapter_gradient_bytes_per_device": count * 4,
            "excludes": "frozen model weights, activations and compiler temporaries"}

def needs_work(row, target):
    return not row.get("failed") and (row.get("step", 0) < target
            or target == HORIZONS[-1] and not row.get("complete"))

def aggregate(spec, result):
    branches = result.get("branches", {})
    complete = all(branches.get(str(seed), {}).get(arm.name, {}).get("complete", False)
                   for seed in SEEDS for arm in spec.arms)
    wins = {}
    for seed in SEEDS:
        rows = branches.get(str(seed), {})
        a, b = rows.get(spec.primary, {}), rows.get(spec.control, {})
        if a.get("complete") and b.get("complete"):
            wins[str(seed)] = bool(a["locked_test_nll"] <= b["locked_test_nll"] - 0.1
                                   and a["locked_test_nll"] < a["start_test_nll"])
    return {"complete": complete, "seed_wins": wins,
            "scientific_gate_passed": bool(complete and len(wins) == len(SEEDS) and all(wins.values()))}

def render_summary(spec, result):
    lines = [f"# EXP-{spec.number:03d}: {spec.name}", "",
             f"Status: {result['status']}; gate: {result['aggregate']['scientific_gate_passed']}",
             f"Cumulative invocation time: {result.get('cumulative_hours', 0):.3f} h; pending HF: {result.get('remote_sync_pending', False)}", "",
             "| Seed | Method | Step | Coordinates | Start val NLL | Latest val NLL | Start test NLL | Final test NLL | State |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.6f}"
    for seed, rows in result.get("branches", {}).items():
        for name, row in rows.items():
            validations = row.get("validation", {})
            last_val = validations[max(validations, key=int)] if validations else None
            state = "failed" if row.get("failed") else "complete" if row.get("complete") else "partial"
            lines.append(f"| {seed} | {name} | {row.get('step', 0)} | {row.get('coordinates', 0):,} | "
                         f"{fmt(row.get('start_validation_nll'))} | {fmt(last_val)} | "
                         f"{fmt(row.get('start_test_nll'))} | {fmt(row.get('locked_test_nll'))} | {state} |")
    lines += ["", f"Primary: {spec.primary} vs {spec.control}. Final endpoint only; two paired source seeds.",
              "Frozen Qwen3-1.7B hybrid, 24 Mamba + 4 GQA. No claim of full recovery or MLA quality."]
    return "\n".join(lines) + "\n"

def make_train_step(model, teacher, tx, arm, order, protected_columns, head_dim):
    def corrected(base, coordinates):
        return apply_corrections(base, coordinates, head_dim=head_dim,
                                 protected_input_columns=protected_columns if arm.protected else None)

    @jax.jit
    def step(base, coordinates, state, teacher_params, tokens, step_number):
        # Teacher/anchor forward passes are outside the differentiated closure.
        teacher_output = None
        if arm.objective in ("teacher", "delta_then_teacher"):
            teacher_output = jax.tree.map(jax.lax.stop_gradient, teacher.apply(
                {"params": teacher_params}, tokens,
                return_hidden_states=arm.objective == "delta_then_teacher"))
        elif arm.objective == "self":
            teacher_output = jax.lax.stop_gradient(model.apply({"params": base}, tokens))

        def objective(c):
            candidate = corrected(base, c)
            if arm.objective == "delta_then_teacher":
                logits, states = model.apply({"params": candidate}, tokens, return_hidden_states=True)
                teacher_logits, teacher_states = teacher_output
                alignment = jax.lax.cond(step_number <= HORIZONS[0],
                    lambda: hidden_delta_relative_mse(states, teacher_states, tuple(i for i in order if i > 0)),
                    lambda: forward_kl(logits, teacher_logits, temperature=2.0))
                return alignment + 0.1 * causal_cross_entropy(logits, tokens)
            logits = model.apply({"params": candidate}, tokens)
            ce = causal_cross_entropy(logits, tokens)
            if arm.objective == "teacher":
                return forward_kl(logits, teacher_output, temperature=2.0) + 0.1 * ce
            if arm.objective == "self":
                return ce + 0.1 * forward_kl(logits, teacher_output, temperature=2.0)
            return ce
        loss, grad = jax.value_and_grad(objective)(coordinates)
        updates, new_state = tx.update(grad, state, coordinates)
        new = optax.apply_updates(coordinates, updates)
        norm = optax.global_norm(grad)
        finite = (jnp.isfinite(loss) & jnp.isfinite(norm) & coordinates_finite(grad)
                  & coordinates_finite(new) & coordinates_finite(new_state))
        return new, new_state, loss, norm, finite
    return step

def run_campaign(spec, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default=f"/dev/shm/extent-exp{spec.number}-state")
    parser.add_argument("--qwen-cache-dir", default=f"/dev/shm/qwen3-exp{spec.number}-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall_hours <= 8.5:
        raise ValueError("max-wall-hours must be in [1, 8.5]")
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = json.loads(json.dumps(contract_for(spec, config)))
    if args.plan_only:
        plan = {"contract": contract, "memory": {a.name: coordinate_plan(config, a,
                    len(contract["replacement_order"])) for a in spec.arms}}
        print(json.dumps(plan, indent=2))
        return plan
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 30 * 60
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path, summary_path = output / f"{spec.stem}.json", output / f"{spec.stem}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO required before TPU initialization")
    if not result_path.exists():
        restore_artifact(result_path, f"{spec.prefix}/latest.json", hub)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "contract": contract, "status": "running", "branches": {}, "sessions": [], "pending_slots": []}
    if result["contract"] != contract:
        raise ValueError("resume protocol mismatch; refusing to mix experiments")
    if result.get("status") == "failed":
        result.setdefault("previous_failures", []).append({key: result[key] for key in
            ("stage", "error_type", "error") if key in result})
        for key in ("stage", "error_type", "error", "traceback"):
            result.pop(key, None)
    store = CampaignCheckpointStore(Path(args.state_dir) / "adapters", f"{spec.prefix}/checkpoints")
    remote_store = CampaignCheckpointStore(store.root, store.prefix, hub, retry_delays=())
    next_sync = 0.0
    result["sessions"].append({"started_at_utc": datetime.now(timezone.utc).isoformat(), "hours": 0.0})
    session = result["sessions"][-1]

    def persist(sync=False):
        nonlocal next_sync
        session["hours"] = (time.monotonic() - started) / 3600
        result["cumulative_hours"] = sum(s["hours"] for s in result["sessions"])
        result["aggregate"] = aggregate(spec, result)
        result["remote_sync_pending"] = bool(result["pending_slots"])
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(spec, result), encoding="utf-8")
        if not sync or time.monotonic() < next_sync:
            return
        files = [(result_path, f"{spec.prefix}/latest.json"),
                 (summary_path, f"{spec.prefix}/latest-summary.md")]
        pending = list(result["pending_slots"])
        for slot in pending:
            directory = store.root / slot
            meta = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
            remote_meta = directory / "remote-checkpoint.json"
            write_json_atomic(remote_meta, dict(meta, checkpoint_file="state.msgpack"))
            files += [(directory / meta["checkpoint_file"], f"{store.prefix}/{slot}/state.msgpack"),
                      (remote_meta, f"{store.prefix}/{slot}/checkpoint.json")]
        # Clear pending fields in the same atomic Hub commit as their payloads.
        result["pending_slots"], result["remote_sync_pending"] = [], False
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(spec, result), encoding="utf-8")
        try:
            upload_artifacts_together(files, hub, commit_message=f"EXP-{spec.number} bundled recovery progress")
        except Exception as exc:
            result["pending_slots"], result["remote_sync_pending"] = pending, True
            result["remote_sync_error_type"] = type(exc).__name__
            write_json_atomic(result_path, result)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise
            next_sync = time.monotonic() + (3600 if status == 429 else 120)
            print(f"HF-DEFERRED type={type(exc).__name__}; local atomic states retained", flush=True)
        else:
            next_sync = time.monotonic() + 10 * 60
            result.pop("remote_sync_error_type", None)
            write_json_atomic(result_path, result)

    if args.sync_only:
        persist(sync=True)
        return result
    if aggregate(spec, result)["complete"]:
        persist(sync=True)
        print("Already completed; no TPU model allocation.", flush=True)
        return result
    stage = "preflight"
    _safe_notify(args.telegram, f"Extent EXP-{spec.number} started; budget={args.max_wall_hours}h")
    try:
        result["git_revision"] = subprocess.run(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout.strip()
        if not result["branches"]:
            exp091.preflight_source_endpoints()
        result["status"] = "running"
        persist(sync=True)  # Verify write authorization before expensive model allocation.
        devices = list(jax.devices())
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise ValueError("requires a single-host TPU v5e-8")
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        token_sets = {name: load_wikitext2_tokens(windows * LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=offset, dataset_split=split, dataset_config="wikitext-103-raw-v1"
        ).reshape(windows, LENGTH) for name, split, offset, windows in DATA}
        hashes = {name: hashlib.sha256(x.tobytes()).hexdigest() for name, x in token_sets.items()}
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("resume token SHA mismatch")
        result["data_sha256"] = hashes
        stage = "load-model"
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16",
                                          compute_dtype="bfloat16", remat_policy="full")
        teacher = Qwen3ForCausalLM(source)
        initial = initialize_sharded_parameters(teacher, jax.random.key(940), init_tokens, mesh)
        teacher_params, _ = stream_teacher_qwen_weights(initial.params, _ensure_checkpoint(Path(args.qwen_cache_dir)), source)
        del initial
        model = HybridForCausalLM(config)
        initial = initialize_sharded_parameters(model, jax.random.key(941), init_tokens, mesh)
        abstract, layout = initial.abstract_params, initial.layout
        del initial
        seq_contract = configured_contract(config)
        order = tuple(seq_contract["replacement_order"])
        inner = int(config.hidden_size * config.mamba.expand)
        protected_columns = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
        source_stores = (
            CampaignCheckpointStore(Path(args.state_dir) / "exp069", f"{EXP069_PREFIX}/checkpoints", hub),
            CampaignCheckpointStore(Path(args.state_dir) / "exp072", f"{EXP072['HF_PREFIX']}/checkpoints", hub))
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
        steps = {a.name: make_train_step(model, teacher, tx, a, order, protected_columns, config.mamba.head_dim)
                 for a in spec.arms}
        evals = {a.name: jax.jit(lambda base, c, t, a=a: causal_cross_entropy(model.apply({"params":
            apply_corrections(base, c, head_dim=config.mamba.head_dim,
                protected_input_columns=protected_columns if a.protected else None)}, t), t)) for a in spec.arms}

        def evaluate(base, c, arm, name):
            return float(np.mean([float(jax.block_until_ready(evals[arm.name](base, c,
                jax.device_put(w[None, :], batch_layout)))) for w in token_sets[name]]))

        def compose(seed):
            endpoints, source_hashes = {}, {}
            for depth, layer in enumerate(order, 1):
                slot = f"prep/seed-{seed}/layer-{layer}"
                meta = source_stores[0].metadata(slot, dict(seq_contract["source_contract"],
                    seed=seed, layer=layer, kind="prepared_mamba"))
                if meta is None:
                    raise FileNotFoundError(f"missing EXP069 {slot}")
                c = sequential.endpoint_contract(seq_contract, seed, "ONPOLICY", layer, depth, meta["checkpoint_sha256"])
                restored = source_stores[1].restore(f"seed-{seed}/ONPOLICY/layer-{layer}", c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP072 seed={seed} layer={layer}")
                payload, metadata = restored
                endpoints[layer], source_hashes[str(layer)] = payload["params"], metadata["checkpoint_sha256"]
            return compose_parameters(teacher_params, endpoints, order, abstract, layout), source_hashes

        result["status"] = "running"
        # Rotate every method/seed through each horizon before extending it.
        for target in HORIZONS:
            for seed in SEEDS:
                rows = result["branches"].setdefault(str(seed), {})
                if not any(needs_work(rows.get(a.name, {}), target) for a in spec.arms):
                    continue
                if time.monotonic() >= deadline:
                    break
                stage = f"compose-seed-{seed}"
                base, source_hashes = compose(seed)
                for arm in spec.arms:
                    row = rows.setdefault(arm.name, {"step": 0})
                    if not needs_work(row, target):
                        continue
                    stage = f"seed-{seed}-{arm.name}"
                    slot = f"seed-{seed}/{arm.name.lower()}"
                    checkpoint_contract = dict(contract, seed=seed, arm=asdict(arm),
                                               source_sha256=source_hashes, data_sha256=hashes)
                    coords = initialize_corrections(base, order, arm.subspace, seed=seed,
                                                   rank=arm.rank, head_dim=config.mamba.head_dim)
                    state = tx.init(coords)
                    template = {"coordinates": coords, "optimizer": state}
                    restored = store.restore(slot, checkpoint_contract, template)
                    if restored is None:
                        restored = remote_store.restore(slot, checkpoint_contract, template)
                    if restored:
                        payload, meta = restored
                        coords, state, row["step"] = payload["coordinates"], payload["optimizer"], meta["step"]
                    elif row["step"]:
                        raise FileNotFoundError(f"missing recovery state for {slot}; refusing restart")
                    row["coordinates"] = coordinate_count(coords)
                    row["effective_coordinates"] = row["coordinates"] - (len(order) * arm.rank *
                        (base[f"layers_{order[0]}"]["mamba"]["in_proj"]["kernel"].shape[1] - protected_columns)
                        if arm.protected else 0)
                    if row["coordinates"] != coordinate_plan(config, arm, len(order))["allocated_coordinates"]:
                        raise ValueError("runtime adapter shapes differ from registered memory plan")
                    if "start_validation_nll" not in row:
                        row["start_validation_nll"] = evaluate(base, coords, arm, "validation")

                    def save():
                        meta = store.save(slot, {"coordinates": coords, "optimizer": state},
                            contract=checkpoint_contract, step=row["step"], metrics={"finite": not row.get("failed", False)})
                        row["checkpoint_sha256"], row["checkpoint_bytes"] = meta["checkpoint_sha256"], meta["checkpoint_bytes"]
                        if slot not in result["pending_slots"]:
                            result["pending_slots"].append(slot)
                        persist(sync=True)

                    if restored is None:
                        save()  # Durable step zero also permits recovery from an early compile failure.
                    for number in range(row["step"] + 1, target + 1):
                        if time.monotonic() >= deadline:
                            save()
                            break
                        tokens = jax.device_put(token_sets["train"][(number - 1) % TRAIN_WINDOWS][None, :], batch_layout)
                        new, new_state, loss, norm, finite = steps[arm.name](base, coords, state, teacher_params, tokens, jnp.asarray(number))
                        jax.block_until_ready(loss)
                        if not bool(finite):
                            row.update(failed=True, error=f"nonfinite step {number}; last finite state retained")
                            save()
                            break
                        coords, state = new, new_state
                        row.update(step=number, latest_loss=float(loss), latest_grad_norm=float(norm))
                        if number % 256 == 0 or number == target:
                            if number % 1024 == 0:
                                row.setdefault("validation", {})[str(number)] = evaluate(base, coords, arm, "validation")
                            save()
                            print(f"EXP{spec.number} seed={seed} arm={arm.name} step={number} loss={float(loss):.6f}", flush=True)
                    if row["step"] == HORIZONS[-1] and not row.get("failed") and not row.get("complete"):
                        row["locked_test_nll"] = evaluate(base, coords, arm, "locked_test")
                        zero = initialize_corrections(base, order, arm.subspace, seed=seed, rank=arm.rank, head_dim=config.mamba.head_dim)
                        row["start_test_nll"] = evaluate(base, zero, arm, "locked_test")
                        row["complete"] = True
                        save()
                    del coords, state, template, restored
                    gc.collect()
                    if time.monotonic() >= deadline:
                        break
                del base
                gc.collect()
            if time.monotonic() >= deadline:
                break
        done = aggregate(spec, result)["complete"]
        any_failed = any(r.get("failed") for rows in result["branches"].values() for r in rows.values())
        result["status"] = "completed" if done else "branch_failure" if any_failed else "deadline_partial"
        next_sync = 0.0
        persist(sync=True)
        _safe_notify(args.telegram, f"Extent EXP-{spec.number} {result['status']} gate={result['aggregate']['scientific_gate_passed']} remote_synced={not result['remote_sync_pending']}")
        print(f"result_json={result_path}\nsummary={summary_path}", flush=True)
        return result
    except BaseException as exc:
        result.update(status="failed", stage=stage, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        try:
            next_sync = 0.0  # A caught failure must attempt to publish recovery states now.
            persist(sync=True)
        except Exception:
            write_json_atomic(result_path, result)
        _safe_notify(args.telegram, f"Extent EXP-{spec.number} failed stage={stage} type={type(exc).__name__}")
        raise
