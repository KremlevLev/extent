"""EXP-069: paired whole-model comparison of two retained-GQA placements."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.config import load_config
from extent.decoder_replacement_eval import Qwen3DecoderTail, qwen3_decoder_tail_params
from extent.full_hybrid_materialization import (
    _host_tree_into_target_shards, _replace_layer_mixer, materialize_qwen_gqa_layer,
)
from extent.full_model_distillation import full_model_eval_metrics, make_prediction_distill_step
from extent.hf_artifact_sync import artifact_config_from_env, upload_artifact
from extent.initialization import initialize_sharded_parameters, initialize_sharded_optimizer_state
from extent.layers.mamba3 import Mamba3MIMO
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_parity import load_mixer_arrays
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sharding import batch_sharding, create_v5e_mesh, replicated_sharding
from extent.teacher_activation_cache import load_activation_cache, validate_external_evaluation_cache
from extent.weight_mapping import stream_direct_qwen_weights, stream_teacher_qwen_weights
from scripts.offline_mamba_distill import _slice
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen17_transplant_atlas_campaign import _prune_cache_arrays, _valid_cache
from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_bridge_ablation import _calibrate_readout, _train_recovery
from scripts.qwen_bridge_ablation_campaign import _cache_arguments
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mimo_lift import _train_exact_dual_bridge


PROTOCOL = "exp069-qwen17-dual-allocation-full-model-v1"
HF_PREFIX = "experiments/exp069-allocation"
PLACEMENTS = {"UNIFORM": (6, 13, 20, 27), "ATLAS": (0, 1, 26, 27)}
SEEDS = (123, 456)
TOTAL_STEPS = 3072
CHECKPOINTS = (0, 256, 1024, 2048, 3072)
PREP_STEPS = 2048
DUAL_STEPS = 1024
SEQUENCE_LENGTH = 256
EVAL_WINDOWS = 64
DATA_SEED = 20260906


def contract_for(config) -> dict:
    return {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "base_config": json.loads(json.dumps(config.to_dict())),
        "placements": {k: list(v) for k, v in PLACEMENTS.items()},
        "seeds": list(SEEDS), "total_steps": TOTAL_STEPS,
        "checkpoints": list(CHECKPOINTS), "sequence_length": SEQUENCE_LENGTH,
        "prep_steps": PREP_STEPS, "dual_steps": DUAL_STEPS,
        "prep_train_offset": 1179648, "prep_validation_offset": 196608,
        "train_offset": 0, "evaluation_split": "test", "evaluation_offset": 0,
        "evaluation_windows": EVAL_WINDOWS, "data_seed": DATA_SEED,
        "lr": 3e-5, "warmup": 128, "weight_decay": 0.0,
        "clip_norm": 1.0, "temperature": 2.0, "ce_weight": 0.1,
        "trainable": "all_student_parameters",
    }


def required_layers() -> tuple[int, ...]:
    return tuple(i for i in range(28) if any(i not in p for p in PLACEMENTS.values()))


def aggregate(result: dict) -> dict:
    pairs = []
    for seed in SEEDS:
        rows = result.get("arms", {}).get(str(seed), {})
        if not all(rows.get(arm, {}).get("complete") for arm in PLACEMENTS):
            continue
        uniform, atlas = rows["UNIFORM"], rows["ATLAS"]
        if not uniform["finite"] or not atlas["finite"]:
            continue
        differences = np.asarray([
            atlas["evaluations"][str(s)]["student_nll"]
            - uniform["evaluations"][str(s)]["student_nll"] for s in CHECKPOINTS
        ])
        window_delta = np.asarray(atlas["evaluations"][str(TOTAL_STEPS)]["window_nll"]) - np.asarray(
            uniform["evaluations"][str(TOTAL_STEPS)]["window_nll"]
        )
        if not np.all(np.isfinite(window_delta)):
            continue
        pairs.append({"seed": seed, "final_nll_delta_atlas_minus_uniform": float(differences[-1]),
                      "nll_auc_delta": float(np.trapezoid(differences, np.asarray(CHECKPOINTS) / TOTAL_STEPS)),
                      "window_deltas": window_delta.tolist()})
    complete = len(pairs) == len(SEEDS)
    return {
        "completed_pairs": len(pairs), "paired_results": pairs,
        "mean_final_nll_delta": float(np.mean([p["final_nll_delta_atlas_minus_uniform"] for p in pairs])) if pairs else None,
        "scientific_gate_passed": bool(complete and all(
            p["final_nll_delta_atlas_minus_uniform"] < 0 and p["nll_auc_delta"] < 0 for p in pairs
        )),
        "gate_definition": "Both paired seeds must favor ATLAS in final whole-model NLL and NLL AUC.",
        "scope": "Two-seed placement experiment; no claim of universal layer importance or MLA quality.",
    }


def render_summary(result: dict) -> str:
    lines = ["# EXP-069 whole-model attention allocation", "",
             f"- Status: {result['status']}", f"- Session hours: {result['duration_hours']:.3f}",
             f"- Completed pairs: {result['aggregate']['completed_pairs']}/2",
             f"- Scientific gate: {result['aggregate']['scientific_gate_passed']}", "",
             "| Seed | Placement | Step | Student NLL | Excess NLL |", "|---|---|---:|---:|---:|"]
    for seed, arms in result["arms"].items():
        for arm, row in arms.items():
            if not row.get("evaluations"):
                continue
            step = max(map(int, row["evaluations"]))
            metrics = row["evaluations"][str(step)]
            lines.append(f"| {seed} | {arm} | {step} | {metrics['student_nll']:.6f} | {metrics['excess_nll']:.6f} |")
    lines += ["", "Shared Mamba positions use identical prepared checkpoint hashes within each seed.",
              "Preparation: 1024 dual + 2048 local recovery steps per retained Mamba; 24 per model.",
              "HF upload failures, if any, are listed in checkpoint_events; local state remains available."]
    return "\n".join(lines) + "\n"


def prepare_layer(config, source, reader, arrays, evaluation, train_manifest, eval_manifest, layer, seed, deadline):
    """One selected method only; no repeated random-control preparation."""
    training_slice = _slice(train_manifest["window_layout"], "training")
    calibration_slice = _slice(train_manifest["window_layout"], "calibration")
    eval_slice = validate_external_evaluation_cache(
        train_manifest, eval_manifest, allow_cross_split=True, required_dataset_split="validation",
    )
    dtype = jnp.bfloat16
    inputs = np.asarray(arrays["normalized_input"][calibration_slice])
    targets = np.asarray(arrays["attention_target"][calibration_slice], np.float32)
    mamba = Mamba3MIMO(config.hidden_size, config.mamba, dtype=dtype, param_dtype=dtype)
    dual = Mamba3MIMO(config.hidden_size, config.mamba, dtype=dtype, param_dtype=dtype, execution_mode="dual")
    params = mamba.init(jax.random.fold_in(jax.random.key(seed), layer), jnp.asarray(inputs[:1], dtype))["params"]
    params, bridge, done = _train_exact_dual_bridge(
        params=params, recurrent=mamba, dual=dual, config=config.mamba,
        training_arrays=arrays, training_slice=training_slice, compute_dtype=dtype,
        steps=DUAL_STEPS, learning_rate=3e-5, data_seed=DATA_SEED + seed + 30000,
        freeze_complex=False, parity_inputs=inputs[:1], deadline_monotonic=deadline,
    )
    if not bridge["finite"]:
        raise FloatingPointError(f"non-finite dual preparation: layer {layer}")
    if not done:
        return None
    params, readout = _calibrate_readout(mamba, params, inputs, targets, dtype, 1e-2)
    tail = Qwen3DecoderTail(config.hidden_size, source.intermediate_size, source.rms_norm_eps, dtype, jnp.float32)
    params, recovery, done = _train_recovery(
        params=params, mamba=mamba, tail=tail, tail_params=qwen3_decoder_tail_params(reader, layer),
        training_arrays=arrays, training_slice=training_slice,
        evaluation_arrays=evaluation, evaluation_slice=eval_slice, compute_dtype=dtype,
        total_steps=PREP_STEPS, checkpoints=(0, PREP_STEPS), learning_rate=3e-5,
        decoder_loss_weight=1.0, data_seed=DATA_SEED + seed, evaluation_batch_windows=4,
        deadline_monotonic=deadline, homotopy_schedule="none",
    )
    if not recovery["finite"]:
        raise FloatingPointError(f"non-finite local recovery: layer {layer}")
    if not done:
        return None
    return params, {"bridge": bridge, "readout": readout, "recovery": recovery}


def run_training_segment(params, opt_state, row, *, step_fn, evaluate, train_tokens,
                         batch_layout, deadline, save, total_steps=TOTAL_STEPS,
                         checkpoints=CHECKPOINTS):
    """Resume cursor/optimizer together; never persist a corrupted update."""
    last_saved = row["completed_steps"]
    if not row["evaluations"]:
        initial_metrics = evaluate(params)
        if not initial_metrics["finite"]:
            raise FloatingPointError("non-finite initial evaluation")
        row["evaluations"]["0"] = initial_metrics
        save(params, opt_state, row)
    for zero_step in range(row["completed_steps"], total_steps):
        if time.monotonic() >= deadline:
            break
        index = int(deterministic_batch_indices(zero_step, 1, len(train_tokens), DATA_SEED)[0])
        batch = jax.device_put(train_tokens[index:index + 1], batch_layout)
        params, opt_state, metrics = step_fn(params, opt_state, batch)
        jax.block_until_ready(metrics)
        if not bool(metrics["grads_finite"]) or not all(np.isfinite(float(v)) for v in metrics.values()):
            raise FloatingPointError("non-finite update; last durable checkpoint retained")
        step = zero_step + 1
        row["completed_steps"] = step
        if step in checkpoints:
            evaluated = evaluate(params)
            if not evaluated["finite"]:
                raise FloatingPointError("non-finite evaluation; last durable checkpoint retained")
            row["evaluations"][str(step)] = evaluated
            row["training_metrics"][str(step)] = _metric_record(metrics)
            print(f"full_model step={step} nll={evaluated['student_nll']:.6f}")
        row["complete"] = step == total_steps
        if step % 1024 == 0 or step == total_steps:
            save(params, opt_state, row)
            last_saved = step
    step = row["completed_steps"]
    if str(step) not in row["evaluations"]:
        final_metrics = evaluate(params)
        if not final_metrics["finite"]:
            raise FloatingPointError("non-finite partial evaluation")
        row["evaluations"][str(step)] = final_metrics
    row["complete"] = step == total_steps
    if step != last_saved:
        save(params, opt_state, row)
    return row


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp067-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.0)
    parser.add_argument("--hf-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 0.75 <= args.max_wall_hours <= 7.5:
        raise ValueError("wall budget must be between 0.75 and 7.5 hours")
    started = time.monotonic()
    # Soft boundary: cache builds/XLA compilations and HF calls cannot be preempted.
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output, state_root = Path(args.output_dir), Path(args.state_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    existing_bytes = sum(p.stat().st_size for p in state_root.rglob("*") if p.is_file())
    if shutil.disk_usage(state_root).free + existing_bytes < 40 * 1024**3:
        raise ValueError("state-dir needs a 40 GiB budget including existing checkpoints; use a larger RAM filesystem")
    hub = artifact_config_from_env() if args.hf_sync else None
    if args.hf_sync and hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required for cross-session resume")
    store = CampaignCheckpointStore(state_root, f"{HF_PREFIX}/checkpoints", hub)
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = contract_for(config)
    result = {"protocol": PROTOCOL, "contract": contract, "arms": {}, "preparation": {}, "status": "running"}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              text=True, capture_output=True, check=False)
    result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    result["software"] = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "huggingface_hub")}
    result["software"]["python"] = platform.python_version()
    stage = "startup"

    def persist():
        result.update(duration_hours=(time.monotonic() - started) / 3600,
                      checkpoint_events=store.events, aggregate=aggregate(result))
        path = output / "extent-m3q-allocation-campaign.json"
        write_json_atomic(path, result)
        summary = output / "extent-m3q-allocation-campaign-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        if hub:
            for local, remote in ((path, "latest.json"), (summary, "latest-summary.md")):
                try:
                    upload_artifact(local, f"{HF_PREFIX}/{remote}", hub, commit_message="EXP-069 progress")
                except Exception as exc:
                    print(f"hf_summary=FAILED type={type(exc).__name__}; local result retained")

    _safe_notify(args.telegram, "Extent EXP-069 started: whole-model attention allocation")
    try:
        devices = list(jax.devices())
        result["devices"] = [str(d) for d in devices]
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise ValueError("EXP-069 requires eight TPU devices")
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="full")
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        # Prepare a single shared bank. Both allocations consume byte-identical
        # endpoints at all common Mamba positions, 24 preparations per model.
        for layer in required_layers():
            missing = []
            for seed in SEEDS:
                slot = f"prep/seed-{seed}/layer-{layer}"
                c = dict(contract, seed=seed, layer=layer, kind="prepared_mamba")
                meta = store.metadata(slot, c)
                if meta:
                    result["preparation"][slot] = {"sha256": meta["checkpoint_sha256"], "restored": True}
                else:
                    missing.append((seed, slot, c))
            if not missing:
                continue
            if time.monotonic() + 20 * 60 >= deadline:
                result["status"] = "deadline_partial"
                break
            stage = f"prepare-layer-{layer}"
            common = dict(layer=layer, qwen_cache_dir=args.qwen_cache_dir, qwen_storage="ram",
                          dataset_cache_dir=args.dataset_cache_dir, output_dir=output,
                          compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4,
                          artifact_prefix="exp069", recovery_steps=PREP_STEPS, validation_windows=32,
                          sequence_length=64, training_token_offset=1179648,
                          validation_token_offset=196608, source_model="1.7b-base")
            specs = [_cache_arguments(evaluation_only=e, **common) for e in (False, True)]
            for cache_args, manifest, directory in specs:
                if not _valid_cache(manifest, directory, layer):
                    build_cache(cache_args)
                if time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                result["status"] = "deadline_partial"
                break
            tm, ta, _ = load_activation_cache(specs[0][1], artifact_dir=specs[0][2], verify_hashes=True)
            em, ea, _ = load_activation_cache(specs[1][1], artifact_dir=specs[1][2], verify_hashes=True)
            for seed, slot, c in missing:
                print(f"exp069 preparation layer={layer} seed={seed}")
                prepared = prepare_layer(config, source, reader, ta, ea, tm, em, layer, seed, deadline)
                if prepared is None:
                    result["status"] = "deadline_partial"
                    break
                params, metrics = prepared
                meta = store.save(slot, {"params": params}, contract=c, step=PREP_STEPS, metrics=metrics)
                result["preparation"][slot] = {"sha256": meta["checkpoint_sha256"], "restored": False}
                del params, prepared
                persist()
            del ta, ea
            gc.collect()
            for _, _, directory in specs:
                _prune_cache_arrays(directory, output)
            jax.clear_caches()
            if result["status"] == "deadline_partial":
                break

        if result["status"] != "deadline_partial":
            stage = "full-model"
            run_full_models(config, source, reader, mesh, batch_layout, store, contract, result,
                            args.dataset_cache_dir, deadline, persist)
            result["status"] = "completed" if aggregate(result)["completed_pairs"] == len(SEEDS) else "deadline_partial"
        persist()
        failed_sync = any(not e["passed"] for e in store.events if e["operation"] == "upload")
        _safe_notify(args.telegram, f"Extent EXP-069 {result['status']} pairs={result['aggregate']['completed_pairs']}/2 HF_upload_errors={failed_sync}")
        print(f"EXP069-{result['status']} summary={output / 'extent-m3q-allocation-campaign-summary.md'}")
        return result
    except BaseException as exc:
        result.update(status="failed", stage=stage, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        persist()
        _safe_notify(args.telegram, f"Extent EXP-069 failed stage={stage} error={type(exc).__name__}")
        raise


def run_full_models(config, source, reader, mesh, batch_layout, store, contract, result,
                    dataset_cache_dir, deadline, persist):
    if time.monotonic() + 10 * 60 >= deadline:
        return
    teacher_model = Qwen3ForCausalLM(source)
    init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
    teacher_init = initialize_sharded_parameters(teacher_model, jax.random.key(641), init_tokens, mesh)
    teacher_params, teacher_report = stream_teacher_qwen_weights(teacher_init.params, reader, source)
    result["teacher_tensor_count"] = getattr(teacher_report, "tensor_count", None)
    teacher_layout = teacher_init.layout
    del teacher_init
    train = load_wikitext2_tokens(TOTAL_STEPS * SEQUENCE_LENGTH, dataset_cache_dir,
                                 tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
                                 token_offset=0, dataset_split="train").reshape(TOTAL_STEPS, SEQUENCE_LENGTH)
    validation = load_wikitext2_tokens(EVAL_WINDOWS * SEQUENCE_LENGTH, dataset_cache_dir,
                                      tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
                                      token_offset=0, dataset_split="test").reshape(EVAL_WINDOWS, SEQUENCE_LENGTH)
    result["full_model_data_sha256"] = {"train": hashlib.sha256(train.tobytes()).hexdigest(),
                                       "test": hashlib.sha256(validation.tobytes()).hexdigest()}
    def teacher_apply(params, tokens, return_hidden):
        return teacher_model.apply({"params": params}, tokens), ()

    for seed in SEEDS:
        # Counterbalance execution order, preserving identical data order.
        order = tuple(PLACEMENTS) if seed == SEEDS[0] else tuple(reversed(PLACEMENTS))
        for arm in order:
            slot = f"full/seed-{seed}/{arm}"
            c = dict(contract, seed=seed, arm=arm, kind="full_model")
            meta = store.metadata(slot, c)
            if meta:
                result["arms"].setdefault(str(seed), {})[arm] = meta["metrics"]
                if meta["metrics"]["complete"]:
                    continue
            if time.monotonic() + 10 * 60 >= deadline:
                return
            cfg = replace(config, attention_layer_indices=PLACEMENTS[arm])
            model = HybridForCausalLM(cfg)
            initialized = initialize_sharded_parameters(model, jax.random.key(seed), init_tokens, mesh)
            params = initialized.params
            layout, abstract = initialized.layout, initialized.abstract_params
            del initialized
            prep_hashes = {}
            direct_report = None
            if meta is None:
                params, direct_report = stream_direct_qwen_weights(params, reader, cfg)
                for layer in range(cfg.num_layers):
                    if layer in cfg.attention_layer_indices:
                        params, _ = materialize_qwen_gqa_layer(params, load_mixer_arrays(reader, source, layer), source, layer)
                    else:
                        prep_slot = f"prep/seed-{seed}/layer-{layer}"
                        prepared, pm = store.restore(prep_slot, dict(contract, seed=seed, layer=layer, kind="prepared_mamba"))
                        target = params[f"layers_{layer}"]["mamba"]
                        mixed = _host_tree_into_target_shards(prepared["params"], target)
                        params = _replace_layer_mixer(params, layer, "mamba", mixed)
                        prep_hashes[str(layer)] = pm["checkpoint_sha256"]
                        del prepared, mixed, target
            tx = create_lion(learning_rate=3e-5, warmup_steps=128, total_steps=TOTAL_STEPS,
                             weight_decay=0.0, max_grad_norm=1.0)
            optimizer = initialize_sharded_optimizer_state(tx, params, abstract, layout, mesh)
            opt_state, opt_layout = optimizer.opt_state, optimizer.layout
            del optimizer
            row = {"complete": False, "finite": True, "completed_steps": 0, "evaluations": {},
                   "training_metrics": {}, "attention_layers": list(PLACEMENTS[arm]),
                   "prepared_checkpoint_hashes": prep_hashes,
                   "direct_tensor_count": getattr(direct_report, "tensor_count", None),
                   "parameter_count": sum(int(x.size) for x in jax.tree.leaves(params))}
            if meta:
                host, meta = store.restore(slot, c, {"params": params, "opt_state": opt_state})
                params = jax.tree.map(jax.device_put, host["params"], layout)
                opt_state = jax.tree.map(jax.device_put, host["opt_state"], opt_layout)
                row = meta["metrics"]
                del host
            result["arms"].setdefault(str(seed), {})[arm] = row

            def student_apply(p, tokens, return_hidden):
                return model.apply({"params": p}, tokens), ()
            metric_layout = {name: replicated_sharding(mesh) for name in (
                "loss", "prediction_kl", "cross_entropy", "hidden_loss", "grad_norm",
                "grads_finite", "nonfinite_grad_leaves", "max_abs_grad")}
            train_step = jax.jit(make_prediction_distill_step(
                student_apply, teacher_apply, tx, temperature=2.0, cross_entropy_weight=0.1, bf16_gradients=True),
                in_shardings=(layout, opt_layout, teacher_layout, batch_layout),
                out_shardings=(layout, opt_layout, metric_layout), donate_argnums=(0, 1))

            @jax.jit
            def evaluate_batch(p, t, tokens):
                return full_model_eval_metrics(model.apply({"params": p}, tokens),
                                               teacher_model.apply({"params": t}, tokens), tokens, temperature=2.0)

            def evaluate(p):
                records = []
                for window in validation:
                    metrics = evaluate_batch(p, teacher_params, jax.device_put(window[None], batch_layout))
                    jax.block_until_ready(metrics)
                    records.append(_metric_record(metrics))
                summary = {name: bool(all(r[name] for r in records)) if name == "finite"
                           else float(np.mean([r[name] for r in records])) for name in records[0]}
                summary["window_nll"] = [r["student_nll"] for r in records]
                return summary

            def save(p, opt, current_row):
                store.save(slot, {"params": p, "opt_state": opt}, contract=c,
                           step=current_row["completed_steps"], metrics=current_row)
                persist()

            print(f"exp069 full-model seed={seed} arm={arm} resume_step={row['completed_steps']}")
            run_training_segment(params, opt_state, row,
                                 step_fn=lambda p, o, b: train_step(p, o, teacher_params, b),
                                 evaluate=evaluate, train_tokens=train, batch_layout=batch_layout,
                                 deadline=deadline, save=save, total_steps=TOTAL_STEPS, checkpoints=CHECKPOINTS)
            del params, opt_state, train_step, evaluate_batch, evaluate, save
            jax.clear_caches()
            gc.collect()
            if not row["complete"]:
                return


if __name__ == "__main__":
    main()
