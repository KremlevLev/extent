"""EXP-073: joint full-model recovery from completed EXP-072 endpoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import jax
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import full_model_eval_metrics, make_prediction_distill_step
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import initialize_sharded_optimizer_state, initialize_sharded_parameters
from extent.optimizer import create_lion
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sharding import batch_sharding, create_v5e_mesh, replicated_sharding
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072, configured_contract
from scripts import m3q_sequential_recovery_campaign as sequential
from scripts.m3q_allocation_campaign import run_training_segment
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp073-sequential-warmstart-joint-recovery-v1"
HF_PREFIX = "experiments/exp073-sequential-joint-recovery"
OUTPUT_SUBDIR = "exp073"
RESULT_STEM = "extent-m3q-sequential-joint-recovery"
SUMMARY_TITLE = "EXP-073 sequential warm-start joint recovery"
ARMS = ("TEACHER", "ONPOLICY")
SOURCE_ARM_BY_ARM = {"TEACHER": "TEACHER", "ONPOLICY": "ONPOLICY"}
LR_BY_ARM = {"TEACHER": 3e-5, "ONPOLICY": 3e-5}
WARMUP_STEPS = 128
CLIP_NORM = 1.0
OPTIMIZER_DESCRIPTION = "BF16 Lion; lr=3e-5 warmup=128 no decay clip=1.0"
PRIMARY_ARM = "ONPOLICY"
CONTROL_ARM = "TEACHER"
AGGREGATE_MODE = "paired_advantage"
CONTRACT_EXTRA = {}
SEEDS = (123, 456)
TOTAL_STEPS = 8192
CHECKPOINTS = (0, 1024, 2048, 4096, 6144, 8192)
SEQUENCE_LENGTH = 256
TRAIN_WINDOWS = 4096
TRAIN_OFFSET = 1_310_720
EVAL_WINDOWS = 32
EVAL_OFFSET = 131_072


def detach_donated_tree(tree, layout):
    """Give the trainable student buffers distinct ownership from the teacher.

    Composing from Qwen reuses immutable teacher arrays. A donated train step
    cannot receive the same device buffer both as student and teacher input.
    ``may_alias=False`` preserves values/sharding while forcing new storage.
    """
    detached = jax.tree.map(
        lambda value, sharding: jax.device_put(
            value, sharding, donate=False, may_alias=False
        ),
        tree,
        layout,
    )
    jax.block_until_ready(detached)
    return detached


def experiment_contract(config):
    contract = {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_sequential_contract": configured_contract(config),
        "arms": list(ARMS), "seeds": list(SEEDS),
        "total_steps": TOTAL_STEPS, "checkpoints": list(CHECKPOINTS),
        "sequence_length": SEQUENCE_LENGTH, "train_windows": TRAIN_WINDOWS,
        "train_split": "train", "train_offset": TRAIN_OFFSET,
        "eval_windows": EVAL_WINDOWS, "eval_split": "test", "eval_offset": EVAL_OFFSET,
        "data_seed": 20_260_906,
        "objective": "teacher forward KL at temperature 2 plus 0.1 causal cross entropy",
        "optimizer": OPTIMIZER_DESCRIPTION,
        "trainable": "all student parameters",
        "registered_primary": "ONPOLICY warm start versus equal-update TEACHER warm start",
    }
    contract.update(CONTRACT_EXTRA)
    return contract


def aggregate(result):
    if AGGREGATE_MODE == "stability":
        rows = []
        for seed in SEEDS:
            for arm in ARMS:
                row = result.get("arms", {}).get(str(seed), {}).get(arm, {})
                if not row.get("complete"):
                    continue
                curve = np.asarray([
                    row["evaluations"][str(step)]["student_nll"] for step in CHECKPOINTS
                ], np.float64)
                rows.append({
                    "seed": seed, "arm": arm,
                    "initial_nll": float(curve[0]), "final_nll": float(curve[-1]),
                    "final_delta": float(curve[-1] - curve[0]),
                    "maximum_nll_ratio": float(np.max(curve) / curve[0]),
                    "best_nll": float(np.min(curve)),
                })
        primary = [row for row in rows if row["arm"] == PRIMARY_ARM]
        return {
            "primary_arm": PRIMARY_ARM, "completed_primary_seeds": len(primary),
            "stability_results": rows,
            "scientific_gate_passed": bool(len(primary) == len(SEEDS) and all(
                row["final_delta"] < 0 and row["maximum_nll_ratio"] <= 1.25
                for row in primary
            )),
            "gate_definition": (
                f"At both seeds, {PRIMARY_ARM} final NLL must beat its own step-0 "
                "NLL and no registered checkpoint may exceed 1.25x step-0 NLL."
            ),
        }
    pairs = []
    for seed in SEEDS:
        rows = result.get("arms", {}).get(str(seed), {})
        if not all(rows.get(arm, {}).get("complete") for arm in ARMS):
            continue
        teacher, onpolicy = rows[CONTROL_ARM], rows[PRIMARY_ARM]
        delta = np.asarray([
            onpolicy["evaluations"][str(step)]["student_nll"]
            - teacher["evaluations"][str(step)]["student_nll"]
            for step in CHECKPOINTS
        ], np.float64)
        x = np.asarray(CHECKPOINTS, np.float64) / TOTAL_STEPS
        pairs.append({
            "seed": seed,
            "final_nll_delta_onpolicy_minus_teacher": float(delta[-1]),
            "nll_delta_auc": float(np.trapezoid(delta, x)),
            "checkpoint_nll_deltas": delta.tolist(),
        })
    return {
        "completed_pairs": len(pairs), "paired_results": pairs,
        "mean_final_nll_delta": float(np.mean([
            row["final_nll_delta_onpolicy_minus_teacher"] for row in pairs
        ])) if pairs else None,
        "scientific_gate_passed": bool(len(pairs) == len(SEEDS) and all(
            row["final_nll_delta_onpolicy_minus_teacher"] < 0
            and row["nll_delta_auc"] < 0 for row in pairs
        )),
        "gate_definition": (
            "At both seeds, the EXP-072 ONPOLICY warm start must beat the matched "
            "TEACHER warm start in final held-out NLL and NLL-delta AUC."
        ),
    }


def render_summary(result):
    lines = [
        f"# {SUMMARY_TITLE}", "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Completed primary units: `{result['aggregate'].get('completed_pairs', result['aggregate'].get('completed_primary_seeds', 0))}/2`",
        f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`", "",
        "| Seed | Warm start | Step | Student NLL | Excess NLL | KL |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, arms in result.get("arms", {}).items():
        for arm, row in arms.items():
            for step, metrics in row.get("evaluations", {}).items():
                lines.append(
                    f"| {seed} | {arm} | {step} | {metrics['student_nll']:.6f} | "
                    f"{metrics['excess_nll']:.6f} | {metrics['prediction_kl']:.6f} |"
                )
    lines += ["", "All comparisons use byte-identified EXP-072 endpoints and fixed data; see the recorded contract for the sole arm difference."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp073-state")
    parser.add_argument("--exp069-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--exp072-state-dir", default="/dev/shm/extent-exp072-v2-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp072-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.0)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall_hours <= 8:
        raise ValueError("EXP-073 wall budget must be 1-8 hours")
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / OUTPUT_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{RESULT_STEM}.json"
    summary_path = output / f"{RESULT_STEM}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = experiment_contract(config)
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "contract": contract, "status": "running", "arms": {}, "checkpoint_events": []}
    if result["contract"] != contract:
        raise ValueError("EXP-073 resume contract mismatch")
    for key in ("error", "error_type", "traceback"):
        result.pop(key, None)
    result.update(status="running", git_revision=subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True).stdout.strip())
    store = CampaignCheckpointStore(Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub)
    seq_store = CampaignCheckpointStore(Path(args.exp072_state_dir), f"{EXP072['HF_PREFIX']}/checkpoints", hub)
    base_store = CampaignCheckpointStore(Path(args.exp069_state_dir), f"{EXP069_PREFIX}/checkpoints", hub)
    stage = "startup"

    def persist(upload=False):
        result.update(duration_hours=(time.monotonic() - started) / 3600,
                      checkpoint_events=store.events + seq_store.events + base_store.events,
                      aggregate=aggregate(result))
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload:
            for local, remote in ((result_path, "latest.json"), (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(local, f"{HF_PREFIX}/{remote}", hub, commit_message=f"{PROTOCOL} progress")
                except Exception as exc:
                    print(f"hf_summary=FAILED type={type(exc).__name__}; local retained", flush=True)

    _safe_notify(args.telegram, f"Extent {PROTOCOL} started")
    try:
        devices = list(jax.devices())
        if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
            raise ValueError("EXP-073 requires one TPU v5e-8")
        mesh, batch_layout = create_v5e_mesh(devices), None
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="full")
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        teacher_init = initialize_sharded_parameters(teacher, jax.random.key(731), init_tokens, mesh)
        teacher_params, _ = stream_teacher_qwen_weights(teacher_init.params, reader, source)
        teacher_layout = teacher_init.layout
        del teacher_init
        train = load_wikitext2_tokens(
            TRAIN_WINDOWS * SEQUENCE_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=TRAIN_OFFSET, dataset_split="train").reshape(TRAIN_WINDOWS, SEQUENCE_LENGTH)
        evaluation = load_wikitext2_tokens(
            EVAL_WINDOWS * SEQUENCE_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=EVAL_OFFSET, dataset_split="test").reshape(EVAL_WINDOWS, SEQUENCE_LENGTH)
        hashes = {"train": hashlib.sha256(train.tobytes()).hexdigest(),
                  "eval": hashlib.sha256(evaluation.tobytes()).hexdigest()}
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-073 token content changed")
        result["data_sha256"] = hashes
        seq_contract = contract["source_sequential_contract"]
        order = tuple(seq_contract["replacement_order"])
        model = HybridForCausalLM(config)

        def teacher_apply(p, tokens, return_hidden):
            return teacher.apply({"params": p}, tokens), ()

        for seed in SEEDS:
            arm_order = ARMS if seed == SEEDS[0] else tuple(reversed(ARMS))
            for arm in arm_order:
                if time.monotonic() + 20 * 60 >= deadline:
                    result["status"] = "deadline_partial"
                    return result
                stage = f"seed-{seed}-{arm}"
                source_arm = SOURCE_ARM_BY_ARM[arm]
                initialized = initialize_sharded_parameters(model, jax.random.key(seed), init_tokens, mesh)
                endpoints, endpoint_hashes = {}, {}
                for depth, layer in enumerate(order, 1):
                    base_slot = f"prep/seed-{seed}/layer-{layer}"
                    base_contract = dict(seq_contract["source_contract"], seed=seed, layer=layer, kind="prepared_mamba")
                    base_meta = base_store.metadata(base_slot, base_contract)
                    if base_meta is None:
                        raise FileNotFoundError(f"missing EXP-069 endpoint: {base_slot}")
                    seq_slot = f"seed-{seed}/{source_arm}/layer-{layer}"
                    endpoint_c = sequential.endpoint_contract(
                        seq_contract, seed, source_arm, layer, depth, base_meta["checkpoint_sha256"])
                    restored = seq_store.restore(seq_slot, endpoint_c)
                    if restored is None:
                        raise FileNotFoundError(f"EXP-072 must complete before EXP-073: {seq_slot}")
                    payload, meta = restored
                    endpoints[layer] = payload["params"]
                    endpoint_hashes[str(layer)] = meta["checkpoint_sha256"]
                params = compose_parameters(teacher_params, endpoints, order, initialized.abstract_params, initialized.layout)
                params = detach_donated_tree(params, initialized.layout)
                del endpoints
                arm_contract = dict(contract, seed=seed, arm=arm, kind="full_model",
                                    sequential_endpoint_hashes=endpoint_hashes,
                                    data_sha256=result["data_sha256"])
                tx = create_lion(learning_rate=LR_BY_ARM[arm], warmup_steps=WARMUP_STEPS,
                                 total_steps=TOTAL_STEPS, weight_decay=0.0,
                                 max_grad_norm=CLIP_NORM)
                optimizer = initialize_sharded_optimizer_state(
                    tx, params, initialized.abstract_params, initialized.layout, mesh)
                opt_state, opt_layout = optimizer.opt_state, optimizer.layout
                del optimizer
                slot = f"seed-{seed}/{arm}"
                meta = store.metadata(slot, arm_contract)
                row = {"complete": False, "finite": True, "completed_steps": 0,
                       "evaluations": {}, "training_metrics": {},
                       "sequential_endpoint_hashes": endpoint_hashes,
                       "source_arm": source_arm, "learning_rate": LR_BY_ARM[arm]}
                if meta is not None:
                    host, meta = store.restore(slot, arm_contract, {"params": params, "opt_state": opt_state})
                    params = jax.tree.map(jax.device_put, host["params"], initialized.layout)
                    opt_state = jax.tree.map(jax.device_put, host["opt_state"], opt_layout)
                    row = meta["metrics"]
                    del host
                result["arms"].setdefault(str(seed), {})[arm] = row
                if row.get("complete"):
                    del params, opt_state, initialized
                    continue

                def student_apply(p, tokens, return_hidden):
                    return model.apply({"params": p}, tokens), ()

                metric_layout = {name: replicated_sharding(mesh) for name in (
                    "loss", "prediction_kl", "cross_entropy", "hidden_loss", "grad_norm",
                    "grads_finite", "nonfinite_grad_leaves", "max_abs_grad")}
                train_step = jax.jit(make_prediction_distill_step(
                    student_apply, teacher_apply, tx, temperature=2.0,
                    cross_entropy_weight=0.1, bf16_gradients=True),
                    in_shardings=(initialized.layout, opt_layout, teacher_layout, batch_layout),
                    out_shardings=(initialized.layout, opt_layout, metric_layout), donate_argnums=(0, 1))

                @jax.jit
                def evaluate_batch(p, tp, tokens):
                    return full_model_eval_metrics(model.apply({"params": p}, tokens),
                        teacher.apply({"params": tp}, tokens), tokens, temperature=2.0)

                def evaluate(p):
                    records = []
                    for window in evaluation:
                        metrics = evaluate_batch(p, teacher_params, jax.device_put(window[None], batch_layout))
                        jax.block_until_ready(metrics)
                        records.append(_metric_record(metrics))
                    summary = {name: bool(all(r[name] for r in records)) if name == "finite"
                               else float(np.mean([r[name] for r in records])) for name in records[0]}
                    summary["window_nll"] = [r["student_nll"] for r in records]
                    return summary

                def save(p, opt, current):
                    store.save(slot, {"params": p, "opt_state": opt}, contract=arm_contract,
                               step=current["completed_steps"], metrics=current)
                    if not any(e.get("operation") == "upload" and e.get("slot") == slot
                               and e.get("step") == current["completed_steps"] and e.get("passed")
                               for e in reversed(store.events)):
                        raise IOError(f"EXP-073 checkpoint upload failed: {slot}")
                    persist(upload=current["completed_steps"] in CHECKPOINTS)

                print(f"exp073 seed={seed} arm={arm} resume_step={row['completed_steps']}", flush=True)
                run_training_segment(params, opt_state, row,
                    step_fn=lambda p, o, b: train_step(p, o, teacher_params, b),
                    evaluate=evaluate, train_tokens=train, batch_layout=batch_layout,
                    deadline=deadline, save=save, total_steps=TOTAL_STEPS, checkpoints=CHECKPOINTS)
                del params, opt_state, initialized, train_step, evaluate_batch
                jax.clear_caches()
                gc.collect()
                if not row["complete"]:
                    result["status"] = "deadline_partial"
                    return result
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result.update(status="failed", stage=stage, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        persist(upload=True)
        _safe_notify(args.telegram, f"Extent {PROTOCOL} {result['status']} gate={result['aggregate']['scientific_gate_passed']}")


if __name__ == "__main__":
    main()
