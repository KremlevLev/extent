"""EXP-070: frozen composition/input-shift diagnostics; no optimizer or training."""
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
from extent.composition_diagnostics import compose_parameters, make_full_probe, make_input_shift_probe, make_fp32_baseline_probe, json_scalars, gradient_summary
from extent.config import load_config
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import abstract_parameter_tree, initialize_sharded_parameters
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sharding import create_v5e_mesh, batch_sharding, named_sharding_tree, validate_partition_specs
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import contract_for, HF_PREFIX, PLACEMENTS
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint
from scripts.qwen_extended_horizon_campaign import _safe_notify

PROTOCOL = "exp070-frozen-input-shift-v3"
PREFIX = "experiments/exp070-input-shift/v3"
SEEDS = (123, 456)
COUNTS = (0, 1, 4, 12, 24)
LENGTHS = (64, 256)
WINDOWS = 2


def expected_cases():
    return len(SEEDS) * len(COUNTS) * len(LENGTHS) * WINDOWS


def require_tpu_mesh():
    devices = list(jax.devices())
    if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
        raise ValueError("EXP-070 requires eight TPU devices")
    return create_v5e_mesh(devices), devices


def case_key(seed, count, length, window):
    return f"seed-{seed}/replaced-{count}/length-{length}/window-{window}"


def render_summary(result):
    lines = ["# EXP-070 frozen input-shift diagnostics", "", f"Status: {result['status']}",
             f"Accepted probes: {sum(bool(r.get('accepted')) for r in result['cases'].values())}/{expected_cases()}", "No training or optimizer updates.", "",
             "| Case | NLL | Grad norm | Largest gradient parameter |", "|---|---:|---:|---|"]
    for key, row in result["cases"].items():
        top = row["gradient_summary"]["top_parameters"]
        lines.append(f"| {key} | {row['metrics']['student_nll']} | {row['metrics']['grad_norm']} | {top[0]['path'] if top else '-'} |")
    lines += ["", "## All-GQA numerical controls", ""]
    for key, row in result["cases"].items():
        if "fp32_control" in row:
            lines.append(f"{key}: FP32=" + json.dumps(row["fp32_control"], allow_nan=False)
                         + f"; BF16 excess_NLL={row['metrics']['excess_nll']}, KL={row['metrics']['prediction_kl']}")
    lines += ["", "## Matched-input layer errors", "",
              "Relative L2 is normalized by the teacher decoder contribution, excluding residual identity.",
              "At most three layers per probe, ranked by increase in absolute error RMS; full layer data are in JSON.",
              "| Case | Layer | Teacher-input error | Hybrid-input error | Input drift |", "|---|---:|---:|---:|---:|"]
    for key, row in result["cases"].items():
        def increase(item):
            stats = item[1]
            a, b = stats["hybrid_input_error"]["error_rms"], stats["teacher_input_error"]["error_rms"]
            return a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else float("inf")
        for layer, stats in sorted(row["input_shift"].items(), key=increase, reverse=True)[:3]:
            lines.append(f"| {key} | {layer} | {stats['teacher_input_error']['relative_l2']} | {stats['hybrid_input_error']['relative_l2']} | {stats['input_drift']['relative_l2']} |")
    lines += ["", "Exploratory mechanism probe, not proof that sequential calibration improves recovery."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp067-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.0)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 0.5 <= args.max_wall_hours <= 7.5:
        raise ValueError("wall budget must be 0.5–7.5 hours")
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 1200
    output = Path(args.output_dir) / "exp070-v3"
    output.mkdir(parents=True, exist_ok=True)
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF secrets are required to reuse EXP-069 preparations")
    cfg, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    source_contract = contract_for(cfg)
    contract = {"protocol": PROTOCOL, "source_contract": source_contract,
                "counts": list(COUNTS), "seeds": list(SEEDS), "lengths": list(LENGTHS), "windows": WINDOWS,
                "replacement_order": [i for i in range(cfg.num_layers) if i not in PLACEMENTS['UNIFORM']],
                "dataset_split": "validation", "token_offset": 65536,
                "numerics": "fp32-scaled-overflow-fallback-v1"}
    path = output / "extent-m3q-input-shift-campaign.json"
    summary = output / "extent-m3q-input-shift-campaign-summary.md"
    if not path.exists():
        restore_artifact(path, f"{PREFIX}/latest.json", hub)
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"contract": contract, "cases": {}, "checkpoint_events": []}
    if result["contract"] != contract:
        raise ValueError("diagnostic resume contract mismatch")
    for field in ("error", "error_type", "traceback"):
        result.pop(field, None)
    result.update(status="running", git_revision=subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout.strip())
    store = CampaignCheckpointStore(Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub)

    def persist():
        result.update(session_hours=(time.monotonic() - started) / 3600, checkpoint_events=store.events)
        write_json_atomic(path, result)
        summary.write_text(render_summary(result), encoding="utf-8")
        for local, remote in ((path, "latest.json"), (summary, "latest-summary.md")):
            try:
                upload_artifact(local, f"{PREFIX}/{remote}", hub, commit_message="EXP-070 diagnostic progress")
            except Exception as exc:
                result.setdefault("artifact_upload_errors", []).append({"file": remote, "error_type": type(exc).__name__})
                print(f"hf_diagnostic=FAILED type={type(exc).__name__}; local output retained", flush=True)
        write_json_atomic(path, result)

    _safe_notify(args.telegram, "Extent EXP-070 started: frozen input-shift diagnostics")
    try:
        def completed(key):
            return key in result["cases"] and result["cases"][key].get("accepted", False)
        if len(result["cases"]) == expected_cases() and all(completed(k) for k in result["cases"]):
            result["status"] = "completed"
            return result
        mesh, devices = require_tpu_mesh()
        result["devices"] = [str(d) for d in devices]
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="full")
        teacher = Qwen3ForCausalLM(source)
        initialized = initialize_sharded_parameters(teacher, jax.random.key(641), jax.device_put(np.zeros((1, 1), np.int32), batch_layout), mesh)
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher_params, _ = stream_teacher_qwen_weights(initialized.params, reader, source)
        del initialized
        tokens = load_wikitext2_tokens(WINDOWS * 256, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=65536, dataset_split="validation").reshape(WINDOWS, 256)
        data_hash = hashlib.sha256(tokens.tobytes()).hexdigest()
        if result.get("data_sha256", data_hash) != data_hash:
            raise ValueError("diagnostic token content changed")
        result["data_sha256"] = data_hash
        teacher_forward = jax.jit(lambda p, t: teacher.apply({"params": p}, t, return_hidden_states=True))
        for seed in SEEDS:
            prepared, hashes = {}, {}
            for count in COUNTS:
                keys = [case_key(seed, count, length, w) for length in LENGTHS for w in range(WINDOWS)]
                if all(completed(key) for key in keys):
                    continue
                if time.monotonic() + 600 >= deadline:
                    result["status"] = "deadline_partial"
                    return result
                replaced = tuple(contract["replacement_order"][:count])
                for layer in replaced:
                    if layer not in prepared:
                        slot = f"prep/seed-{seed}/layer-{layer}"
                        restored = store.restore(slot, dict(source_contract, seed=seed, layer=layer, kind="prepared_mamba"))
                        if restored is None:
                            raise FileNotFoundError(f"Missing prepared checkpoint {slot}; no silent retraining")
                        state, meta = restored
                        known = result.setdefault("prepared_checkpoint_hashes", {})
                        if known.get(slot, meta["checkpoint_sha256"]) != meta["checkpoint_sha256"]:
                            raise ValueError(f"prepared checkpoint changed across probes: {slot}")
                        known[slot] = meta["checkpoint_sha256"]
                        prepared[layer], hashes[layer] = state["params"], meta["checkpoint_sha256"]
                case_cfg = replace(cfg, attention_layer_indices=tuple(i for i in range(cfg.num_layers) if i not in replaced))
                model = HybridForCausalLM(case_cfg)
                abstract = abstract_parameter_tree(model)
                validate_partition_specs(abstract, mesh)
                layout = named_sharding_tree(abstract, mesh)
                params = compose_parameters(teacher_params, prepared, replaced, abstract, layout)
                full_probe = jax.jit(make_full_probe(model))
                baseline_probe = jax.jit(make_fp32_baseline_probe(case_cfg, source)) if count == 0 else None
                local_probe = jax.jit(make_input_shift_probe(case_cfg, source, replaced[0])) if replaced else None
                for length in LENGTHS:
                    for window in range(WINDOWS):
                        key = case_key(seed, count, length, window)
                        if completed(key):
                            continue
                        if time.monotonic() >= deadline:
                            result["status"] = "deadline_partial"
                            return result
                        print(f"exp070 probe={key} START", flush=True)
                        batch = jax.device_put(tokens[window:window+1, :length], batch_layout)
                        teacher_logits, teacher_states = teacher_forward(teacher_params, batch)
                        metrics, gradient_leaves, hybrid_states = full_probe(params, batch, teacher_logits)
                        jax.block_until_ready(metrics)
                        shift = {}
                        embedding = jnp.take(teacher_params["embed_tokens"]["embedding"], batch, axis=0)
                        for layer in replaced:
                            ti = teacher_states[layer-1] if layer else embedding
                            si = hybrid_states[layer-1] if layer else embedding
                            shift[str(layer)] = local_probe(params[f"layers_{layer}"], teacher_params[f"layers_{layer}"],
                                ti, si, teacher_states[layer], hybrid_states[layer])
                        metrics, leaves, shift = json_scalars(metrics), json_scalars(gradient_leaves), json_scalars(shift)
                        result["cases"][key] = {"metrics": metrics, "gradient_leaves": leaves,
                            "gradient_summary": gradient_summary(leaves, metrics["grad_norm"]),
                            "input_shift": shift, "prepared_hashes": {str(i): hashes[i] for i in replaced},
                            "replaced_layers": list(replaced), "accepted": count != 0}
                        # Persist the observed BF16 failure BEFORE any extra
                        # control/guard can abort or run out of memory.
                        persist()
                        if baseline_probe is not None:
                            control = json_scalars(baseline_probe(params, teacher_params, batch))
                            row = result["cases"][key]
                            row["fp32_control"] = control
                            row["accepted"] = bool(metrics["finite"] and metrics["grads_finite"] and control["passed"])
                            persist()
                            if not row["accepted"]:
                                raise ValueError(f"all-GQA FP32 parity or BF16 finiteness failed: {key}; raw controls saved")
                        del teacher_logits, teacher_states, hybrid_states, gradient_leaves, embedding
                del params, full_probe, local_probe, baseline_probe
                jax.clear_caches()
                gc.collect()
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        persist()
        _safe_notify(args.telegram, f"Extent EXP-070 {result['status']} probes={len(result['cases'])}/{expected_cases()}")


if __name__ == "__main__":
    main()
