"""EXP-081: assembled-model trust-region coordinate recovery."""
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
from extent.calibration_data import load_pg19_tokens, load_wikitext2_tokens
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


PROTOCOL = "exp081-assembled-trust-region-sweep-v1"
HF_PREFIX = "experiments/exp081-trust-region-sweep"
RESULT_STEM = "extent-m3q-trust-region-sweep"
OUTPUT_SUBDIR = "exp081"
ARMS = ("HARD-ACCEPT", "TRUST-LINE")
PRIMARY_ARM = "TRUST-LINE"
SEEDS = (123, 456)
ALPHAS = {
    "HARD-ACCEPT": (0.0, 1.0),
    "TRUST-LINE": (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
}
MIN_RELATIVE_GAIN = 0.001
STEPS_PER_LAYER = 2048
MILESTONES = (6, 12, 18, 24)
TRAIN_WINDOWS, TRAIN_LENGTH, TRAIN_OFFSET = 512, 128, 1_572_864
CALIBRATION_WINDOWS, CALIBRATION_LENGTH, CALIBRATION_OFFSET = 32, 128, 1_638_400
EVAL_WINDOWS, EVAL_LENGTH, EVAL_OFFSET = 32, 256, 196_608
EVAL_SPLIT = "test"
EVAL_DATASET_CONFIG = "wikitext-2-raw-v1"
DATA_SEED = 81_000
CHECKPOINT_RETRY_DELAYS = None


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def choose_trust_alpha(scores: dict[str, float], minimum_relative_gain=MIN_RELATIVE_GAIN):
    """Choose the lowest-KL alpha, retaining alpha zero without enough gain."""
    baseline = float(scores[alpha_key(0.0)])
    best_key = min(scores, key=lambda key: (float(scores[key]), float(key)))
    best = float(scores[best_key])
    gain = (baseline - best) / max(abs(baseline), 1e-12)
    if float(best_key) == 0.0 or gain < minimum_relative_gain:
        return 0.0, float(max(gain, 0.0))
    return float(best_key), float(gain)


def blend_parameters(current, proposal, alpha: float):
    def blend(old, new):
        value = old.astype(jnp.float32) + alpha * (
            new.astype(jnp.float32) - old.astype(jnp.float32)
        )
        return value.astype(old.dtype)
    return jax.tree.map(blend, current, proposal)


def experiment_contract(config):
    return {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_sequential_contract": configured_contract(config),
        "arms": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "seeds": list(SEEDS),
        "forward_order": list(config.mamba_layer_indices),
        "alphas": {key: list(value) for key, value in ALPHAS.items()},
        "minimum_relative_calibration_kl_gain": MIN_RELATIVE_GAIN,
        "steps_per_layer": STEPS_PER_LAYER,
        "milestones": list(MILESTONES),
        "train": ["train", TRAIN_OFFSET, TRAIN_WINDOWS, TRAIN_LENGTH],
        "calibration": [
            "train", CALIBRATION_OFFSET, CALIBRATION_WINDOWS, CALIBRATION_LENGTH
        ],
        "locked_evaluation": (
            [EVAL_SPLIT, EVAL_OFFSET, EVAL_WINDOWS, EVAL_LENGTH]
            if EVAL_DATASET_CONFIG == "wikitext-2-raw-v1"
            else [EVAL_DATASET_CONFIG, EVAL_SPLIT, EVAL_OFFSET, EVAL_WINDOWS, EVAL_LENGTH]
        ),
        "data_seed": DATA_SEED,
        "proposal_objective": "conditional decoder-contribution relative MSE",
        "acceptance_objective": "assembled-model prediction KL against frozen Qwen",
        "optimizer": "BF16 Lion lr=3e-5 warmup=64 cosine no-decay clip=1.0",
        "trainable": "one proposed Mamba subtree; accepted interpolation only",
    }


def checkpoint_contract(contract, seed, arm, position, source_hashes):
    return dict(
        contract,
        kind="assembled_trust_region_milestone",
        seed=int(seed), arm=arm, position=int(position),
        source_exp072_checkpoint_sha256={str(k): v for k, v in source_hashes.items()},
    )


def aggregate(result):
    rows, pairs = [], []
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
                "seed": seed, "arm": arm,
                "initial_nll": float(curve[0]), "final_nll": float(curve[-1]),
                "final_delta": float(curve[-1] - curve[0]),
                "maximum_nll_ratio": float(np.max(curve) / curve[0]),
                "accepted_coordinates": int(sum(
                    decision["selected_alpha"] > 0
                    for decision in row.get("decisions", {}).values()
                )),
            })
        if all(branches.get(arm, {}).get("complete") for arm in ARMS):
            pairs.append({
                "seed": seed,
                "trust_minus_hard_final_nll": float(
                    branches[PRIMARY_ARM]["evaluations"]["24"]["student_nll"]
                    - branches["HARD-ACCEPT"]["evaluations"]["24"]["student_nll"]
                ),
            })
    primary = [row for row in rows if row["arm"] == PRIMARY_ARM]
    passed = bool(
        len(primary) == len(SEEDS) and len(pairs) == len(SEEDS)
        and all(row["final_delta"] < 0 for row in primary)
        and all(row["maximum_nll_ratio"] <= 1.25 for row in primary)
        and all(row["accepted_coordinates"] > 0 for row in primary)
        and all(row["trust_minus_hard_final_nll"] < 0 for row in pairs)
    )
    return {
        "completed_primary_seeds": len(primary), "completed_pairs": len(pairs),
        "results": rows, "paired_results": pairs,
        "scientific_gate_passed": passed,
        "gate_definition": (
            "At both seeds TRUST-LINE improves locked final NLL, stays within "
            "1.25x start, accepts a nonzero coordinate, and beats HARD-ACCEPT."
        ),
    }


def render_summary(result):
    lines = [
        "# EXP-081 assembled-model trust-region sweep", "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`",
        "", "| Seed | Arm | Position | Test NLL | KL | Accepted |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, branches in result.get("branches", {}).items():
        for arm, row in branches.items():
            accepted = sum(
                value["selected_alpha"] > 0
                for value in row.get("decisions", {}).values()
            )
            for position, metrics in row.get("evaluations", {}).items():
                lines.append(
                    f"| {seed} | {arm} | {position} | {metrics['student_nll']:.6f} "
                    f"| {metrics['prediction_kl']:.6f} | {accepted} |"
                )
    lines += [
        "", "Alpha selection uses only the disjoint calibration split.",
        "Locked test metrics are observed only at registered milestones.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp081-state")
    parser.add_argument("--exp069-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--exp072-state-dir", default="/dev/shm/extent-exp072-v2-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp081-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.5)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.0 <= args.max_wall_hours <= 8.25:
        raise ValueError("EXP-081 wall budget must be 1-8.25 hours")

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / OUTPUT_SUBDIR
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
        else {"contract": contract, "status": "running", "branches": {},
              "checkpoint_events": [], "durability_warnings": []}
    )
    if result["contract"] != contract:
        raise ValueError("EXP-081 resume contract mismatch")
    if result.get("status") == "completed":
        print("EXP-081 already completed; restored result is unchanged", flush=True)
        return result
    for key in ("error", "error_type", "traceback"):
        result.pop(key, None)
    result.update(
        status="running",
        git_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    )
    store = CampaignCheckpointStore(
        Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub,
        retry_delays=CHECKPOINT_RETRY_DELAYS,
    )
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
            for local, remote in ((result_path, "latest.json"),
                                  (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(
                        local, f"{HF_PREFIX}/{remote}", hub,
                        commit_message=f"{PROTOCOL} progress",
                    )
                except Exception as exc:
                    warning = {"artifact": remote, "error_type": type(exc).__name__}
                    result.setdefault("durability_warnings", []).append(warning)
                    print(f"hf_summary=FAILED {warning}; local retained", flush=True)

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
        initialized = initialize_sharded_parameters(
            teacher, jax.random.key(810),
            jax.device_put(np.zeros((1, 1), np.int32), batch_layout), mesh,
        )
        teacher_params, _ = stream_teacher_qwen_weights(initialized.params, reader, source)
        del initialized

        def tokens(count, length, offset, split, dataset_config="wikitext-2-raw-v1"):
            return load_wikitext2_tokens(
                count * length, args.dataset_cache_dir,
                tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision,
                token_offset=offset, dataset_split=split,
                dataset_config=dataset_config,
            ).reshape(count, length)

        train_tokens = tokens(TRAIN_WINDOWS, TRAIN_LENGTH, TRAIN_OFFSET, "train")
        calibration_tokens = tokens(
            CALIBRATION_WINDOWS, CALIBRATION_LENGTH, CALIBRATION_OFFSET, "train"
        )
        if EVAL_DATASET_CONFIG == "pg19-pinned-manifest":
            eval_tokens = load_pg19_tokens(
                EVAL_WINDOWS * EVAL_LENGTH, args.dataset_cache_dir,
                tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision,
                token_offset=EVAL_OFFSET, dataset_split=EVAL_SPLIT,
            ).reshape(EVAL_WINDOWS, EVAL_LENGTH)
        else:
            eval_tokens = tokens(
                EVAL_WINDOWS, EVAL_LENGTH, EVAL_OFFSET, EVAL_SPLIT,
                EVAL_DATASET_CONFIG,
            )
        hashes = {
            "train": hashlib.sha256(train_tokens.tobytes()).hexdigest(),
            "calibration": hashlib.sha256(calibration_tokens.tobytes()).hexdigest(),
            "eval": hashlib.sha256(eval_tokens.tobytes()).hexdigest(),
        }
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-081 token content changed")
        result["data_sha256"] = hashes

        order = tuple(config.mamba_layer_indices)
        source_contract = contract_for(config)
        seq_contract = contract["source_sequential_contract"]
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
                source_slot = f"seed-{seed}/ONPOLICY/layer-{layer}"
                source_c = sequential_endpoint_contract(
                    seq_contract, seed, "ONPOLICY", layer, depth,
                    base_meta["checkpoint_sha256"],
                )
                restored = seq_store.restore(source_slot, source_c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP-072 endpoint: {source_slot}")
                payload, meta = restored
                base[seed][layer] = payload["params"]
                base_hashes[seed][layer] = meta["checkpoint_sha256"]

        teacher_forward = jax.jit(lambda p, value: teacher.apply({"params": p}, value))

        def teacher_logits(windows):
            values = []
            for window in windows:
                value = teacher_forward(
                    teacher_params, jax.device_put(window[None], batch_layout)
                )
                values.append(np.asarray(jax.device_get(value), np.float32))
            return values

        calibration_teacher_logits = teacher_logits(calibration_tokens)
        eval_teacher_logits = teacher_logits(eval_tokens)

        generic_cfg = replace(
            config,
            attention_layer_indices=tuple(
                index for index in range(config.num_layers) if index != 0
            ),
        )
        student_layer = HybridDecoderLayer(generic_cfg, 0)
        teacher_layer = Qwen3DecoderLayer(source)
        tx = create_lion(
            learning_rate=3e-5, warmup_steps=64, total_steps=STEPS_PER_LAYER,
            weight_decay=0.0, max_grad_norm=1.0,
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
        model_cache, prefix_forward_cache = {}, {}

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
            return model, compose_parameters(
                teacher_params, endpoints, replaced, abstract, layout
            )

        full_model, _, _ = model_parts(order)
        eval_fn = jax.jit(
            lambda p, value, target: full_model_eval_metrics(
                full_model.apply({"params": p}, value), target, value,
                temperature=2.0,
            )
        )

        def evaluate(endpoints, windows, targets):
            _, params = assemble(order, endpoints)
            records = []
            for index, window in enumerate(windows):
                metrics = eval_fn(
                    params, jax.device_put(window[None], batch_layout),
                    jnp.asarray(targets[index]),
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

        def branch_inputs(endpoints, layer):
            if layer == 0:
                return np.asarray(jax.device_get(
                    jnp.take(embedding_table, jnp.asarray(train_tokens), axis=0)
                ), np.float32)
            prefix = tuple(index for index in order if index < layer)
            model, params = assemble(prefix, endpoints)
            if layer not in prefix_forward_cache:
                prefix_forward_cache[layer] = jax.jit(
                    lambda p, value, model=model, layer=layer:
                    model.apply(
                        {"params": p}, value, return_hidden_states=True
                    )[1][layer - 1]
                )
            pieces = []
            for start in range(0, TRAIN_WINDOWS, 4):
                value = prefix_forward_cache[layer](
                    params,
                    jax.device_put(train_tokens[start:start + 4], batch_layout),
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

        for seed in SEEDS:
            arm_order = ARMS if seed == SEEDS[0] else tuple(reversed(ARMS))
            for arm in arm_order:
                endpoints = dict(base[seed])
                row = {"complete": False, "evaluations": {}, "decisions": {}}
                start_position = 0
                for milestone in reversed(MILESTONES):
                    slot = f"seed-{seed}/{arm}/position-{milestone}"
                    restored = store.restore(
                        slot, checkpoint_contract(
                            contract, seed, arm, milestone, base_hashes[seed]
                        )
                    )
                    if restored is not None:
                        payload, _ = restored
                        endpoints = {int(key): value for key, value in payload["endpoints"].items()}
                        row = payload["row"]
                        start_position = milestone
                        print(f"EXP-081 restore=PASS slot={slot}", flush=True)
                        break
                result["branches"].setdefault(str(seed), {})[arm] = row
                if "0" not in row["evaluations"]:
                    row["evaluations"]["0"] = evaluate(
                        endpoints, eval_tokens, eval_teacher_logits
                    )
                    persist(upload=True)
                for position, layer in enumerate(order, 1):
                    if position <= start_position:
                        continue
                    if time.monotonic() + 20 * 60 >= deadline:
                        result["status"] = "deadline_partial"
                        return result
                    stage = f"seed-{seed}-{arm}-position-{position}-layer-{layer}"
                    print(f"{stage} START", flush=True)
                    inputs = branch_inputs(endpoints, layer)
                    targets = condition_targets(layer, inputs)
                    current = jax.tree.map(jnp.asarray, endpoints[layer])
                    proposal = current
                    frozen = local_layer_params(
                        teacher_params[f"layers_{layer}"], proposal
                    )
                    opt_state = tx.init(proposal)
                    first = last = None
                    for zero_step in range(STEPS_PER_LAYER):
                        index = deterministic_batch_indices(
                            zero_step, 1, TRAIN_WINDOWS,
                            DATA_SEED + seed + position,
                        )[0]
                        proposal, opt_state, metrics = train_step(
                            proposal, opt_state, frozen,
                            jnp.asarray(inputs[index:index + 1], jnp.bfloat16),
                            jnp.asarray(targets[index:index + 1], jnp.bfloat16),
                        )
                        if zero_step in (0, STEPS_PER_LAYER - 1):
                            jax.block_until_ready(metrics)
                            record = _metric_record(metrics)
                            first = record if first is None else first
                            last = record
                            if not record["grads_finite"]:
                                raise FloatingPointError(f"non-finite proposal: {stage}")

                    scores, trial_metrics = {}, {}
                    trial_endpoints = None
                    for alpha in ALPHAS[arm]:
                        trial_endpoints = dict(endpoints)
                        trial_endpoints[layer] = blend_parameters(current, proposal, alpha)
                        metrics = evaluate(
                            trial_endpoints, calibration_tokens,
                            calibration_teacher_logits,
                        )
                        key = alpha_key(alpha)
                        scores[key] = metrics["prediction_kl"]
                        trial_metrics[key] = metrics
                    selected_alpha, relative_gain = choose_trust_alpha(scores)
                    if selected_alpha > 0:
                        endpoints[layer] = jax.device_get(
                            blend_parameters(current, proposal, selected_alpha)
                        )
                    row["decisions"][str(position)] = {
                        "layer": int(layer), "selected_alpha": selected_alpha,
                        "relative_calibration_kl_gain": relative_gain,
                        "calibration": trial_metrics,
                        "proposal_first_loss": first["loss"],
                        "proposal_final_loss": last["loss"],
                        "proposal_final_grad_norm": last["grad_norm"],
                    }
                    print(
                        f"{stage} selected_alpha={selected_alpha:g} "
                        f"relative_kl_gain={relative_gain:.6g}", flush=True,
                    )
                    del inputs, targets, current, proposal, opt_state, frozen, trial_endpoints
                    if position in MILESTONES:
                        row["evaluations"][str(position)] = evaluate(
                            endpoints, eval_tokens, eval_teacher_logits
                        )
                        slot = f"seed-{seed}/{arm}/position-{position}"
                        meta = store.save(
                            slot,
                            {"endpoints": {str(k): v for k, v in endpoints.items()},
                             "row": row},
                            contract=checkpoint_contract(
                                contract, seed, arm, position, base_hashes[seed]
                            ),
                            step=position,
                            metrics={
                                "student_nll": row["evaluations"][str(position)]["student_nll"],
                                "accepted_coordinates": sum(
                                    value["selected_alpha"] > 0
                                    for value in row["decisions"].values()
                                ),
                            },
                        )
                        synced = any(
                            event.get("operation") == "upload"
                            and event.get("slot") == slot and event.get("passed")
                            for event in reversed(store.events)
                        )
                        if not synced:
                            result.setdefault("durability_warnings", []).append({
                                "slot": slot,
                                "checkpoint_sha256": meta["checkpoint_sha256"],
                                "reason": "hf_checkpoint_upload_failed_after_retries",
                            })
                            print(
                                f"EXP-081 durability warning: {slot}; continuing",
                                flush=True,
                            )
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
