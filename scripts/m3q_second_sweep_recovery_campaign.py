"""EXP-080: second on-policy coordinate sweep over all 24 Mamba layers."""
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

import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import full_model_eval_metrics
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import abstract_parameter_tree, initialize_sharded_parameters
from extent.model import HybridDecoderLayer
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sequential_recovery import make_conditional_recovery_step
from extent.sharding import batch_sharding, named_sharding_tree, validate_partition_specs
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_allocation_campaign import contract_for
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072
from scripts.m3q_full_depth_sequential_confirmation import configured_contract
from scripts.m3q_sequential_recovery_campaign import endpoint_contract as sequential_endpoint_contract
from scripts.m3q_sequential_recovery_campaign import local_layer_params, require_v5e8
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp080-second-onpolicy-coordinate-sweep-v1"
HF_PREFIX = "experiments/exp080-second-sweep"
RESULT_STEM = "extent-m3q-second-sweep-recovery"
ARMS = ("FORWARD-SWEEP", "REVERSE-SWEEP")
SEEDS = (123, 456)
STEPS_PER_LAYER = 2048
MILESTONES = (6, 12, 18, 24)
TRAIN_WINDOWS = 512
TRAIN_LENGTH = 128
TRAIN_OFFSET = 1_310_720
EVAL_WINDOWS = 32
EVAL_LENGTH = 256
EVAL_OFFSET = 131_072
DATA_SEED = 80_000


def experiment_contract(config):
    order = tuple(config.mamba_layer_indices)
    return {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_sequential_contract": configured_contract(config),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "forward_order": list(order),
        "reverse_order": list(reversed(order)),
        "steps_per_layer": STEPS_PER_LAYER,
        "milestones": list(MILESTONES),
        "train_windows": TRAIN_WINDOWS,
        "train_length": TRAIN_LENGTH,
        "train_split": "train",
        "train_offset": TRAIN_OFFSET,
        "eval_windows": EVAL_WINDOWS,
        "eval_length": EVAL_LENGTH,
        "eval_split": "test",
        "eval_offset": EVAL_OFFSET,
        "data_seed": DATA_SEED,
        "objective": (
            "relative MSE of the current decoder contribution against the "
            "same frozen Qwen block conditioned on the hybrid's current input"
        ),
        "optimizer": (
            "per-layer BF16 Lion; lr=3e-5; warmup=64; cosine decay; "
            "no weight decay; clip=1.0"
        ),
        "trainable": "one current Mamba subtree; all other parameters frozen",
        "registered_primary": "FORWARD-SWEEP versus matched REVERSE-SWEEP",
    }


def sweep_endpoint_contract(contract, seed, arm, layer, position, source_sha):
    return dict(
        contract,
        kind="second_sweep_mamba_endpoint",
        seed=int(seed),
        arm=arm,
        layer=int(layer),
        sweep_position=int(position),
        source_exp072_checkpoint_sha256=source_sha,
    )


def aggregate(result):
    rows = []
    paired = []
    for seed in SEEDS:
        branches = result.get("branches", {}).get(str(seed), {})
        for arm in ARMS:
            row = branches.get(arm, {})
            if not row.get("complete"):
                continue
            curve = np.asarray([
                row["evaluations"][str(step)]["student_nll"]
                for step in (0, *MILESTONES)
            ], np.float64)
            rows.append({
                "seed": seed,
                "arm": arm,
                "initial_nll": float(curve[0]),
                "final_nll": float(curve[-1]),
                "final_delta": float(curve[-1] - curve[0]),
                "maximum_nll_ratio": float(np.max(curve) / curve[0]),
                "best_nll": float(np.min(curve)),
            })
        if all(branches.get(arm, {}).get("complete") for arm in ARMS):
            forward = branches["FORWARD-SWEEP"]["evaluations"]
            reverse = branches["REVERSE-SWEEP"]["evaluations"]
            paired.append({
                "seed": seed,
                "final_forward_minus_reverse": float(
                    forward["24"]["student_nll"] - reverse["24"]["student_nll"]
                ),
            })
    primary = [row for row in rows if row["arm"] == "FORWARD-SWEEP"]
    passed = bool(
        len(primary) == len(SEEDS)
        and len(paired) == len(SEEDS)
        and all(
            row["final_delta"] < 0 and row["maximum_nll_ratio"] <= 1.25
            for row in primary
        )
        and all(row["final_forward_minus_reverse"] < 0 for row in paired)
    )
    return {
        "completed_primary_seeds": len(primary),
        "completed_pairs": len(paired),
        "sweep_results": rows,
        "paired_results": paired,
        "scientific_gate_passed": passed,
        "gate_definition": (
            "At both seeds FORWARD-SWEEP must improve its own final NLL, stay "
            "within 1.25x start, and beat matched REVERSE-SWEEP at depth 24."
        ),
    }


def render_summary(result):
    lines = [
        "# EXP-080 second on-policy coordinate sweep", "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`",
        "", "| Seed | Arm | Updated layers | Student NLL | Excess NLL | KL |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, branches in result.get("branches", {}).items():
        for arm, row in branches.items():
            for position, metrics in row.get("evaluations", {}).items():
                lines.append(
                    f"| {seed} | {arm} | {position} | {metrics['student_nll']:.6f} | "
                    f"{metrics['excess_nll']:.6f} | {metrics['prediction_kl']:.6f} |"
                )
    lines += [
        "", "Both arms start from the same complete EXP-072 ONPOLICY model.",
        "Each update target is recomputed from the current hybrid prefix; only sweep order differs.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp080-state")
    parser.add_argument("--exp069-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--exp072-state-dir", default="/dev/shm/extent-exp072-v2-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp080-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.5)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.0 <= args.max_wall_hours <= 8.25:
        raise ValueError("EXP-080 wall budget must be 1-8.25 hours")

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / "exp080"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{RESULT_STEM}.json"
    summary_path = output / f"{RESULT_STEM}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    config, _ = load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )
    contract = experiment_contract(config)
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {"contract": contract, "status": "running", "branches": {}, "checkpoint_events": []}
    )
    if result["contract"] != contract:
        raise ValueError("EXP-080 resume contract mismatch")
    for key in ("error", "error_type", "traceback"):
        result.pop(key, None)
    result.update(
        status="running",
        git_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    )
    store = CampaignCheckpointStore(Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub)
    seq_store = CampaignCheckpointStore(
        Path(args.exp072_state_dir), f"{EXP072['HF_PREFIX']}/checkpoints", hub
    )
    base_store = CampaignCheckpointStore(
        Path(args.exp069_state_dir), f"{EXP069_PREFIX}/checkpoints", hub
    )
    stage = "startup"

    def persist(upload=False):
        result.update(
            duration_hours=(time.monotonic() - started) / 3600,
            checkpoint_events=store.events + seq_store.events + base_store.events,
            aggregate=aggregate(result),
        )
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload:
            for local, remote in ((result_path, "latest.json"), (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(
                        local, f"{HF_PREFIX}/{remote}", hub,
                        commit_message=f"{PROTOCOL} progress",
                    )
                except Exception as exc:
                    print(f"hf_summary=FAILED type={type(exc).__name__}; local retained", flush=True)

    _safe_notify(args.telegram, f"Extent {PROTOCOL} started")
    try:
        mesh, devices = require_v5e8()
        result["devices"] = [str(device) for device in devices]
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16",
            remat_policy="full",
        )
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        initialized = initialize_sharded_parameters(
            teacher, jax.random.key(800), init_tokens, mesh
        )
        teacher_params, _ = stream_teacher_qwen_weights(
            initialized.params, reader, source
        )
        del initialized

        train_tokens = load_wikitext2_tokens(
            TRAIN_WINDOWS * TRAIN_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=TRAIN_OFFSET, dataset_split="train",
        ).reshape(TRAIN_WINDOWS, TRAIN_LENGTH)
        eval_tokens = load_wikitext2_tokens(
            EVAL_WINDOWS * EVAL_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=EVAL_OFFSET, dataset_split="test",
        ).reshape(EVAL_WINDOWS, EVAL_LENGTH)
        hashes = {
            "train": hashlib.sha256(train_tokens.tobytes()).hexdigest(),
            "eval": hashlib.sha256(eval_tokens.tobytes()).hexdigest(),
        }
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-080 token content changed")
        result["data_sha256"] = hashes

        order = tuple(config.mamba_layer_indices)
        seq_contract = contract["source_sequential_contract"]
        source_contract = contract_for(config)
        base, base_hashes = {}, {}
        for seed in SEEDS:
            base[seed], base_hashes[seed] = {}, {}
            for depth, layer in enumerate(order, 1):
                base_slot = f"prep/seed-{seed}/layer-{layer}"
                base_meta = base_store.metadata(
                    base_slot,
                    dict(source_contract, seed=seed, layer=layer, kind="prepared_mamba"),
                )
                if base_meta is None:
                    raise FileNotFoundError(f"missing EXP-069 endpoint: {base_slot}")
                slot = f"seed-{seed}/ONPOLICY/layer-{layer}"
                source_c = sequential_endpoint_contract(
                    seq_contract, seed, "ONPOLICY", layer, depth,
                    base_meta["checkpoint_sha256"],
                )
                restored = seq_store.restore(slot, source_c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP-072 endpoint: {slot}")
                payload, meta = restored
                base[seed][layer] = payload["params"]
                base_hashes[seed][layer] = meta["checkpoint_sha256"]

        teacher_forward = jax.jit(
            lambda p, tokens: teacher.apply({"params": p}, tokens)
        )
        eval_teacher_logits = []
        for window in eval_tokens:
            logits = teacher_forward(
                teacher_params, jax.device_put(window[None], batch_layout)
            )
            eval_teacher_logits.append(np.asarray(jax.device_get(logits), np.float32))

        generic_cfg = replace(
            config,
            attention_layer_indices=tuple(
                index for index in range(config.num_layers) if index != 0
            ),
        )
        student_layer = HybridDecoderLayer(generic_cfg, 0)
        teacher_layer = Qwen3DecoderLayer(source)
        tx = create_lion(
            learning_rate=3e-5, warmup_steps=64,
            total_steps=STEPS_PER_LAYER, weight_decay=0.0, max_grad_norm=1.0,
        )
        train_step = jax.jit(
            make_conditional_recovery_step(student_layer, tx, bf16_gradients=True)
        )
        conditional_teacher = jax.jit(
            lambda p, x: teacher_layer.apply(
                {"params": p}, x,
                jnp.arange(x.shape[1], dtype=jnp.int32)[None], None,
            )
        )
        embedding_table = teacher_params["embed_tokens"]["embedding"]
        model_cache = {}
        prefix_forward_cache = {}
        evaluation_cache = {}

        def model_parts(replaced):
            replaced = tuple(replaced)
            if replaced not in model_cache:
                cfg = replace(
                    config,
                    attention_layer_indices=tuple(
                        index for index in range(config.num_layers)
                        if index not in replaced
                    ),
                )
                model = HybridForCausalLM(cfg)
                abstract = abstract_parameter_tree(model)
                validate_partition_specs(abstract, mesh)
                model_cache[replaced] = (
                    model, abstract, named_sharding_tree(abstract, mesh)
                )
            return model_cache[replaced]

        def assemble(replaced, endpoints):
            model, abstract, layout = model_parts(replaced)
            params = compose_parameters(
                teacher_params, endpoints, replaced, abstract, layout
            )
            return model, params

        def branch_inputs(endpoints, layer):
            if layer == 0:
                return np.asarray(
                    jax.device_get(jnp.take(embedding_table, jnp.asarray(train_tokens), axis=0)),
                    np.float32,
                )
            prefix = tuple(index for index in order if index < layer)
            model, params = assemble(prefix, endpoints)
            if layer not in prefix_forward_cache:
                prefix_forward_cache[layer] = jax.jit(
                    lambda p, tokens, model=model, layer=layer:
                    model.apply(
                        {"params": p}, tokens, return_hidden_states=True
                    )[1][layer - 1]
                )
            pieces = []
            for start in range(0, TRAIN_WINDOWS, 4):
                value = prefix_forward_cache[layer](
                    params, jax.device_put(train_tokens[start:start + 4], batch_layout)
                )
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            del params
            return np.concatenate(pieces)

        def condition_targets(layer, inputs):
            pieces = []
            for start in range(0, len(inputs), 4):
                value = conditional_teacher(
                    teacher_params[f"layers_{layer}"],
                    jax.device_put(inputs[start:start + 4], batch_layout),
                )
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            return np.concatenate(pieces)

        def evaluate(endpoints):
            model, params = assemble(order, endpoints)
            if "full" not in evaluation_cache:
                evaluation_cache["full"] = jax.jit(
                    lambda p, tokens, target, model=model: full_model_eval_metrics(
                        model.apply({"params": p}, tokens), target, tokens,
                        temperature=2.0,
                    )
                )
            records = []
            for index, window in enumerate(eval_tokens):
                metrics = evaluation_cache["full"](
                    params, jax.device_put(window[None], batch_layout),
                    jnp.asarray(eval_teacher_logits[index]),
                )
                jax.block_until_ready(metrics)
                records.append(_metric_record(metrics))
            del params
            summary = {
                name: bool(all(row[name] for row in records))
                if name == "finite"
                else float(np.mean([row[name] for row in records]))
                for name in records[0]
            }
            summary["window_nll"] = [row["student_nll"] for row in records]
            return summary

        for seed in SEEDS:
            arm_order = ARMS if seed == SEEDS[0] else tuple(reversed(ARMS))
            for arm in arm_order:
                endpoints = dict(base[seed])
                row = result["branches"].setdefault(str(seed), {}).setdefault(
                    arm, {"complete": False, "evaluations": {}, "layers": {}}
                )
                if "0" not in row["evaluations"]:
                    row["evaluations"]["0"] = evaluate(endpoints)
                    persist(upload=True)
                sweep_order = order if arm == "FORWARD-SWEEP" else tuple(reversed(order))
                for position, layer in enumerate(sweep_order, 1):
                    slot = f"seed-{seed}/{arm}/position-{position}-layer-{layer}"
                    endpoint_c = sweep_endpoint_contract(
                        contract, seed, arm, layer, position,
                        base_hashes[seed][layer],
                    )
                    restored = store.restore(slot, endpoint_c)
                    if restored is not None:
                        payload, meta = restored
                        endpoints[layer] = payload["params"]
                        row["layers"][str(position)] = meta["metrics"]
                    else:
                        if time.monotonic() + 20 * 60 >= deadline:
                            result["status"] = "deadline_partial"
                            return result
                        stage = f"seed-{seed}-{arm}-position-{position}-layer-{layer}"
                        print(f"{stage} START", flush=True)
                        inputs = branch_inputs(endpoints, layer)
                        targets = condition_targets(layer, inputs)
                        candidate = jax.tree.map(jnp.asarray, endpoints[layer])
                        frozen = local_layer_params(
                            teacher_params[f"layers_{layer}"], candidate
                        )
                        opt_state = tx.init(candidate)
                        first, last = None, None
                        for zero_step in range(STEPS_PER_LAYER):
                            index = deterministic_batch_indices(
                                zero_step, 1, TRAIN_WINDOWS,
                                DATA_SEED + seed + position,
                            )[0]
                            candidate, opt_state, metrics = train_step(
                                candidate, opt_state, frozen,
                                jnp.asarray(inputs[index:index + 1], jnp.bfloat16),
                                jnp.asarray(targets[index:index + 1], jnp.bfloat16),
                            )
                            if zero_step in (0, STEPS_PER_LAYER - 1):
                                jax.block_until_ready(metrics)
                                record = _metric_record(metrics)
                                first = record if first is None else first
                                last = record
                                if not record["grads_finite"]:
                                    raise FloatingPointError(
                                        f"non-finite coordinate update: {stage}"
                                    )
                        metrics = {
                            "complete": True,
                            "steps": STEPS_PER_LAYER,
                            "first_loss": first["loss"],
                            "final_loss": last["loss"],
                            "final_grad_norm": last["grad_norm"],
                            "source_exp072_checkpoint_sha256": base_hashes[seed][layer],
                        }
                        meta = store.save(
                            slot, {"params": candidate}, contract=endpoint_c,
                            step=STEPS_PER_LAYER, metrics=metrics,
                        )
                        synced = any(
                            event.get("operation") == "upload"
                            and event.get("slot") == slot
                            and event.get("step") == STEPS_PER_LAYER
                            and event.get("passed")
                            for event in reversed(store.events)
                        )
                        if not synced:
                            # The local endpoint is valid and the next coordinate
                            # does not depend on Hub availability.  Treat a long
                            # transient Hub outage as reduced resumability instead
                            # of throwing away the active TPU campaign.
                            warning = {
                                "slot": slot,
                                "position": position,
                                "layer": layer,
                                "reason": "hf_checkpoint_upload_failed_after_retries",
                            }
                            result.setdefault("durability_warnings", []).append(warning)
                            print(
                                f"EXP-080 durability warning: {slot}; "
                                "continuing from the verified local endpoint",
                                flush=True,
                            )
                        endpoints[layer] = jax.device_get(candidate)
                        row["layers"][str(position)] = dict(
                            metrics, checkpoint_sha256=meta["checkpoint_sha256"]
                        )
                        del inputs, targets, candidate, opt_state, frozen
                    if position in MILESTONES and str(position) not in row["evaluations"]:
                        row["evaluations"][str(position)] = evaluate(endpoints)
                        persist(upload=True)
                    else:
                        persist()
                    gc.collect()
                row["complete"] = True
                persist(upload=True)
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result.update(
            status="failed", stage=stage, error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(),
        )
        raise
    finally:
        persist(upload=True)
        _safe_notify(
            args.telegram,
            f"Extent {PROTOCOL} {result['status']} "
            f"gate={result['aggregate']['scientific_gate_passed']}",
        )


if __name__ == "__main__":
    main()
