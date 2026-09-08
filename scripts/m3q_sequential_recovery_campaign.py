"""EXP-071: matched sequential residual-distribution recovery on Qwen3-1.7B."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters, json_scalars
from extent.config import load_config
from extent.full_model_distillation import full_model_eval_metrics
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import abstract_parameter_tree, initialize_sharded_parameters
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sequential_recovery import make_conditional_recovery_step, recovery_examples
from extent.model import HybridDecoderLayer
from extent.sharding import batch_sharding, create_v5e_mesh, named_sharding_tree, validate_partition_specs
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import contract_for, HF_PREFIX as EXP069_PREFIX, PLACEMENTS
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp071-sequential-input-recovery-v1"
HF_PREFIX = "experiments/exp071-sequential-recovery"
ARMS = ("TEACHER", "ONPOLICY", "MIXED")
SEEDS = (123, 456)
MILESTONES = (1, 4, 8, 12)
STEPS_PER_LAYER = 1024
TRAIN_WINDOWS = 128
TRAIN_LENGTH = 64
EVAL_WINDOWS = 8
EVAL_LENGTH = 256
TRAIN_OFFSET = 2_359_296
EVAL_OFFSET = 32_768


def replacement_order(config):
    return tuple(i for i in range(config.num_layers) if i not in PLACEMENTS["UNIFORM"])


def experiment_contract(config):
    return {
        "protocol": PROTOCOL,
        "source_contract": contract_for(config),
        "arms": list(ARMS), "seeds": list(SEEDS),
        "replacement_order": list(replacement_order(config)[: MILESTONES[-1]]),
        "steps_per_layer": STEPS_PER_LAYER, "train_windows": TRAIN_WINDOWS,
        "train_length": TRAIN_LENGTH, "train_split": "train", "train_offset": TRAIN_OFFSET,
        "eval_windows": EVAL_WINDOWS, "eval_length": EVAL_LENGTH,
        "eval_split": "test", "eval_offset": EVAL_OFFSET,
        "lr": 3e-5, "warmup": 64, "weight_decay": 0.0, "clip_norm": 1.0,
        "trainable": "current_mamba_only",
        "teacher_target": "same frozen Qwen decoder block conditioned on each arm input",
        "objective": "relative MSE of decoder contribution after subtracting shared residual identity",
        "numerics": "BF16 parameters/compute/gradients; FP32 relative-MSE reduction",
    }


def endpoint_contract(contract, seed, arm, layer, depth, base_sha):
    return dict(contract, kind="sequential_mamba_endpoint", seed=int(seed), arm=arm,
                layer=int(layer), depth=int(depth), base_checkpoint_sha256=base_sha)


def aggregate(result):
    paired = []
    for seed in SEEDS:
        branches = result.get("branches", {}).get(str(seed), {})
        if not all(branches.get(arm, {}).get("evaluations", {}).get(str(MILESTONES[-1])) for arm in ARMS):
            continue
        teacher = branches["TEACHER"]["evaluations"]
        for arm in ("ONPOLICY", "MIXED"):
            other = branches[arm]["evaluations"]
            delta = np.asarray([other[str(d)]["student_nll"] - teacher[str(d)]["student_nll"] for d in MILESTONES])
            paired.append({"seed": seed, "arm": arm, "final_nll_delta_vs_teacher": float(delta[-1]),
                           "milestone_nll_delta_vs_teacher": delta.tolist(),
                           "nll_delta_auc": float(np.trapezoid(delta, np.asarray(MILESTONES) / MILESTONES[-1]))})
    mixed = [p for p in paired if p["arm"] == "MIXED"]
    complete = len(mixed) == len(SEEDS)
    return {
        "completed_primary_pairs": len(mixed), "paired_results": paired,
        "mean_mixed_final_nll_delta": float(np.mean([p["final_nll_delta_vs_teacher"] for p in mixed])) if mixed else None,
        "scientific_gate_passed": bool(complete and all(p["final_nll_delta_vs_teacher"] < 0 and p["nll_delta_auc"] < 0 for p in mixed)),
        "gate_definition": "At both seeds, MIXED must beat equal-update TEACHER recovery in final held-out NLL and milestone NLL-delta AUC.",
        "scope": "Causal intervention on recovery input distribution over the first 12 sequential replacements; not a full 85% hybrid result.",
    }


def render_summary(result):
    lines = ["# EXP-071 sequential input-distribution recovery", "",
             f"- Status: `{result['status']}`", f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
             f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`", "",
             "| Seed | Arm | Depth | Student NLL | Excess NLL | KL |", "|---:|---|---:|---:|---:|---:|"]
    for seed, branches in result.get("branches", {}).items():
        for arm, row in branches.items():
            for depth, metrics in row.get("evaluations", {}).items():
                lines.append(f"| {seed} | {arm} | {depth} | {metrics['student_nll']:.6f} | {metrics['excess_nll']:.6f} | {metrics['prediction_kl']:.6f} |")
    lines += ["", "Every current layer starts from the byte-identified EXP-069 endpoint; only its Mamba subtree is updated.",
              "TEACHER, ONPOLICY, and MIXED receive equal optimizer updates and the same underlying text.",
              "The campaign checkpoints every completed layer to the configured private HF dataset and resumes only on an exact contract/hash match."]
    return "\n".join(lines) + "\n"


def require_v5e8():
    devices = list(jax.devices())
    if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
        raise ValueError("EXP-071 requires one TPU v5e-8")
    return create_v5e_mesh(devices), devices


def local_layer_params(teacher_layer_params, mamba_params):
    params = unfreeze(teacher_layer_params)
    params.pop("self_attn")
    params["mamba"] = mamba_params
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp071-state")
    parser.add_argument("--base-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp071-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.0)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.0 <= args.max_wall_hours <= 8.25:
        raise ValueError("wall budget must be 1.0–8.25 hours")
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / "exp071"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "extent-m3q-sequential-recovery-campaign.json"
    summary_path = output / "extent-m3q-sequential-recovery-campaign-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = experiment_contract(config)
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "contract": contract, "status": "running", "branches": {}, "baselines": {}, "checkpoint_events": []}
    if result["contract"] != contract:
        raise ValueError("EXP-071 resume contract mismatch")
    for key in ("error", "error_type", "traceback"):
        result.pop(key, None)
    result.update(status="running", git_revision=subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout.strip())
    store = CampaignCheckpointStore(Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub)
    base_store = CampaignCheckpointStore(Path(args.base_state_dir), f"{EXP069_PREFIX}/checkpoints", hub)
    stage = "startup"

    def persist(*, upload=False):
        result.update(duration_hours=(time.monotonic() - started) / 3600,
                      checkpoint_events=store.events + base_store.events, aggregate=aggregate(result))
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload:
            for local, remote in ((result_path, "latest.json"), (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(local, f"{HF_PREFIX}/{remote}", hub, commit_message="EXP-071 sequential recovery progress")
                except Exception as exc:
                    print(f"hf_summary=FAILED file={remote} type={type(exc).__name__}; local copy retained", flush=True)

    _safe_notify(args.telegram, "Extent EXP-071 started: matched sequential recovery")
    try:
        mesh, devices = require_v5e8()
        result["devices"] = [str(d) for d in devices]
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="full")
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        init = initialize_sharded_parameters(teacher, jax.random.key(710), jax.device_put(np.zeros((1, 1), np.int32), batch_layout), mesh)
        teacher_params, _ = stream_teacher_qwen_weights(init.params, reader, source)
        del init
        order = tuple(contract["replacement_order"])
        train_tokens = load_wikitext2_tokens(TRAIN_WINDOWS * TRAIN_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=TRAIN_OFFSET, dataset_split="train").reshape(TRAIN_WINDOWS, TRAIN_LENGTH)
        eval_tokens = load_wikitext2_tokens(EVAL_WINDOWS * EVAL_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=EVAL_OFFSET, dataset_split="test").reshape(EVAL_WINDOWS, EVAL_LENGTH)
        hashes = {"train": hashlib.sha256(train_tokens.tobytes()).hexdigest(), "eval": hashlib.sha256(eval_tokens.tobytes()).hexdigest()}
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-071 token content changed")
        result["data_sha256"] = hashes

        # Natural Qwen inputs/targets are computed once, then held fixed for all matched arms.
        stage = "teacher-cache"
        teacher_forward = jax.jit(lambda p, t: teacher.apply({"params": p}, t, return_hidden_states=True))
        natural_inputs = {layer: [] for layer in order}
        eval_teacher_logits = []
        embedding_table = teacher_params["embed_tokens"]["embedding"]
        for start in range(0, TRAIN_WINDOWS, 4):
            tokens = jax.device_put(train_tokens[start:start + 4], batch_layout)
            _, states = teacher_forward(teacher_params, tokens)
            embedding = jnp.take(embedding_table, tokens, axis=0)
            for layer in order:
                natural_inputs[layer].append(np.asarray(jax.device_get(embedding if layer == 0 else states[layer - 1]), np.float32))
        natural_inputs = {k: np.concatenate(v) for k, v in natural_inputs.items()}
        for window in eval_tokens:
            logits, _ = teacher_forward(teacher_params, jax.device_put(window[None], batch_layout))
            # Keep evaluation logits in FP32 host RAM. Quantizing the reference
            # distribution would add avoidable noise to small paired NLL deltas.
            eval_teacher_logits.append(np.asarray(jax.device_get(logits), np.float32))
        del teacher_forward
        jax.clear_caches()

        # Load and hash the exact EXP-069 starting endpoints. Missing state is fatal: never silently retrain it.
        base = {}
        base_hashes = {}
        source_contract = contract["source_contract"]
        for seed in SEEDS:
            base[seed], base_hashes[seed] = {}, {}
            for layer in order:
                slot = f"prep/seed-{seed}/layer-{layer}"
                restored = base_store.restore(slot, dict(source_contract, seed=seed, layer=layer, kind="prepared_mamba"))
                if restored is None:
                    raise FileNotFoundError(f"Missing required EXP-069 endpoint: {slot}")
                payload, meta = restored
                base[seed][layer], base_hashes[seed][layer] = payload["params"], meta["checkpoint_sha256"]

        # One generic local block and optimizer shape lets XLA reuse the compiled training program.
        generic_cfg = replace(config, attention_layer_indices=tuple(i for i in range(config.num_layers) if i != 0))
        student_layer = HybridDecoderLayer(generic_cfg, 0)
        teacher_layer = Qwen3DecoderLayer(source)
        tx = create_lion(learning_rate=3e-5, warmup_steps=64, total_steps=STEPS_PER_LAYER,
                         weight_decay=0.0, max_grad_norm=1.0)
        train_step = jax.jit(make_conditional_recovery_step(student_layer, tx, bf16_gradients=True))
        conditional_teacher = jax.jit(lambda p, x: teacher_layer.apply(
            {"params": p}, x, jnp.arange(x.shape[1], dtype=jnp.int32)[None], None))

        def condition_targets(layer, inputs):
            # Use the identical standalone Qwen block and batch shape for both
            # natural and on-policy inputs. This prevents compiler/execution
            # differences from masquerading as an input-distribution effect.
            pieces = []
            for start in range(0, len(inputs), 4):
                value = conditional_teacher(
                    teacher_params[f"layers_{layer}"],
                    jax.device_put(inputs[start:start + 4], batch_layout),
                )
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            return np.concatenate(pieces)

        natural_targets = {
            layer: condition_targets(layer, natural_inputs[layer]) for layer in order
        }

        model_cache = {}
        branch_forward_cache = {}
        evaluation_cache = {}

        def model_parts(replaced):
            replaced = tuple(replaced)
            if replaced in model_cache:
                return model_cache[replaced]
            cfg = replace(config, attention_layer_indices=tuple(i for i in range(config.num_layers) if i not in replaced))
            model = HybridForCausalLM(cfg)
            abstract = abstract_parameter_tree(model)
            validate_partition_specs(abstract, mesh)
            model_cache[replaced] = (model, abstract, named_sharding_tree(abstract, mesh))
            return model_cache[replaced]

        def assemble(replaced, endpoints):
            model, abstract, layout = model_parts(replaced)
            return model, compose_parameters(teacher_params, endpoints, replaced, abstract, layout)

        def branch_inputs(previous, endpoints, layer):
            if layer == 0:
                return np.asarray(jax.device_get(jnp.take(embedding_table, jnp.asarray(train_tokens), axis=0)), np.float32)
            model, params = assemble(previous, endpoints)
            cache_key = (tuple(previous), layer)
            if cache_key not in branch_forward_cache:
                branch_forward_cache[cache_key] = jax.jit(
                    lambda p, t, model=model, layer=layer:
                    model.apply({"params": p}, t, return_hidden_states=True)[1][layer - 1]
                )
            forward = branch_forward_cache[cache_key]
            pieces = []
            for start in range(0, TRAIN_WINDOWS, 4):
                value = forward(params, jax.device_put(train_tokens[start:start + 4], batch_layout))
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            del params
            return np.concatenate(pieces)

        def evaluate(replaced, endpoints):
            model, params = assemble(replaced, endpoints)
            cache_key = tuple(replaced)
            if cache_key not in evaluation_cache:
                evaluation_cache[cache_key] = jax.jit(
                    lambda p, t, tl, model=model: full_model_eval_metrics(
                        model.apply({"params": p}, t), tl, t, temperature=2.0)
                )
            evaluate_batch = evaluation_cache[cache_key]
            records = []
            for index, window in enumerate(eval_tokens):
                metrics = evaluate_batch(params, jax.device_put(window[None], batch_layout), jnp.asarray(eval_teacher_logits[index]))
                jax.block_until_ready(metrics)
                records.append(_metric_record(metrics))
            del params
            summary = {name: bool(all(r[name] for r in records)) if name == "finite"
                       else float(np.mean([r[name] for r in records])) for name in records[0]}
            summary["window_nll"] = [r["student_nll"] for r in records]
            return summary

        endpoints = {seed: {arm: {} for arm in ARMS} for seed in SEEDS}
        for depth, layer in enumerate(order, 1):
            previous = order[:depth - 1]
            current = order[:depth]
            for seed in SEEDS:
                # Counterbalance arm order without changing any examples or updates.
                arm_order = ARMS if (seed + depth) % 2 else tuple(reversed(ARMS))
                for arm in arm_order:
                    row = result["branches"].setdefault(str(seed), {}).setdefault(arm, {"evaluations": {}, "layers": {}})
                    slot = f"seed-{seed}/{arm}/layer-{layer}"
                    c = endpoint_contract(contract, seed, arm, layer, depth, base_hashes[seed][layer])
                    restored = store.restore(slot, c)
                    if restored is not None:
                        payload, meta = restored
                        endpoints[seed][arm][layer] = payload["params"]
                        row["layers"][str(layer)] = meta["metrics"]
                    else:
                        if time.monotonic() + 20 * 60 >= deadline:
                            result["status"] = "deadline_partial"
                            return result
                        stage = f"depth-{depth}-seed-{seed}-{arm}"
                        print(f"exp071 depth={depth}/12 layer={layer} seed={seed} arm={arm} START", flush=True)
                        # TEACHER is the matched extra-update control and does not
                        # need an expensive hybrid-prefix cache. The other two
                        # branches receive their own on-policy residual stream.
                        h_inputs = (natural_inputs[layer] if arm == "TEACHER" else
                                    branch_inputs(previous, endpoints[seed][arm], layer))
                        h_targets = (natural_targets[layer] if arm == "TEACHER" else
                                     condition_targets(layer, h_inputs))
                        x, y = recovery_examples(arm, jnp.asarray(natural_inputs[layer], jnp.bfloat16),
                                                 jnp.asarray(natural_targets[layer], jnp.bfloat16),
                                                 jnp.asarray(h_inputs, jnp.bfloat16), jnp.asarray(h_targets, jnp.bfloat16))
                        candidate = jax.tree.map(jnp.asarray, base[seed][layer])
                        frozen = local_layer_params(teacher_params[f"layers_{layer}"], candidate)
                        opt_state = tx.init(candidate)
                        first_loss, last_metrics = None, None
                        for step in range(STEPS_PER_LAYER):
                            index = deterministic_batch_indices(step, 1, x.shape[0], 71000 + seed + depth)[0]
                            candidate, opt_state, metrics = train_step(candidate, opt_state, frozen, x[index:index + 1], y[index:index + 1])
                            if step in (0, STEPS_PER_LAYER - 1):
                                jax.block_until_ready(metrics)
                                record = _metric_record(metrics)
                                first_loss = record["loss"] if first_loss is None else first_loss
                                last_metrics = record
                                print(f"exp071 layer={layer} seed={seed} arm={arm} step={step} loss={record['loss']:.6g} finite={record['grads_finite']}", flush=True)
                                if not record["grads_finite"]:
                                    raise FloatingPointError(f"non-finite sequential recovery: {stage} step={step}")
                        metrics = {"complete": True, "steps": STEPS_PER_LAYER, "first_loss": first_loss,
                                   "final_loss": last_metrics["loss"], "final_grad_norm": last_metrics["grad_norm"],
                                   "base_checkpoint_sha256": base_hashes[seed][layer], "input_distribution": arm}
                        meta = store.save(slot, {"params": candidate}, contract=c, step=STEPS_PER_LAYER, metrics=metrics)
                        synced = any(event.get("operation") == "upload" and event.get("slot") == slot
                                     and event.get("step") == STEPS_PER_LAYER and event.get("passed")
                                     for event in reversed(store.events))
                        if not synced:
                            raise IOError(f"HF checkpoint upload failed after retries: {slot}; stopping before dependent work")
                        endpoints[seed][arm][layer] = jax.device_get(candidate)
                        row["layers"][str(layer)] = dict(metrics, checkpoint_sha256=meta["checkpoint_sha256"])
                        del h_inputs, h_targets, x, y, candidate, opt_state, frozen
                    if depth in MILESTONES and str(depth) not in row["evaluations"]:
                        row["evaluations"][str(depth)] = evaluate(current, endpoints[seed][arm])
                    persist()
                    gc.collect()
            if depth in MILESTONES:
                for seed in SEEDS:
                    key = f"seed-{seed}/depth-{depth}"
                    if key not in result["baselines"]:
                        result["baselines"][key] = evaluate(current, {i: base[seed][i] for i in current})
                        persist()
                # Endpoints remain durable after every layer. Compact campaign
                # summaries only need remote publication at registered depths;
                # uploading twice after every arm caused avoidable Hub bursts.
                persist(upload=True)
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result.update(status="failed", stage=stage, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        persist(upload=True)
        _safe_notify(args.telegram, f"Extent EXP-071 {result['status']} stage={stage} pairs={result['aggregate']['completed_primary_pairs']}/2")


if __name__ == "__main__":
    main()
