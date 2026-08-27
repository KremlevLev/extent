from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import time
import traceback

import jax
import numpy as np

from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_bridge_ablation_campaign import _cache_arguments, _prune_cache_arrays, _valid_cache
from scripts.qwen_depth_objective_run import resolve_qwen_cache_dir
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import main as run_layer
from extent.experiment_stage import update_stage_manifest


PROTOCOL = "exp057-locked-fresh-text-exact-lift-confirmation"
LAYERS = (18, 6, 29, 0)
SEEDS = (123, 456, 789)
ARMS = ("CONTROL-RANDOM", "BALANCED-RANK-LIFT")
STEPS = 4096
SEQUENCE_LENGTH = 64
TRAIN_OFFSET = 131072
VALIDATION_OFFSET = 8192


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def aggregate_confirmation(results: dict[str, dict]) -> dict:
    layers = {}
    gate = True
    for layer, result in sorted(results.items(), key=lambda item: int(item[0])):
        random_values, lift_values = [], []
        for seed in map(str, SEEDS):
            arms = result.get("seeds", {}).get(seed, {}).get("arms", {})
            random = arms.get("CONTROL-RANDOM", {}).get("recovery", {})
            lift = arms.get("BALANCED-RANK-LIFT", {}).get("recovery", {})
            if random.get("complete") and lift.get("complete"):
                random_values.append(float(random["evaluations"][str(STEPS)]["decoder_output"]["relative_l2"]))
                lift_values.append(float(lift["evaluations"][str(STEPS)]["decoder_output"]["relative_l2"]))
        if random_values:
            delta = np.asarray(lift_values) - np.asarray(random_values)
            record = {
                "random_mean": float(np.mean(random_values)), "balanced_lift_mean": float(np.mean(lift_values)),
                "lift_minus_random_mean": float(np.mean(delta)), "relative_improvement": float(1 - np.mean(lift_values) / np.mean(random_values)),
                "wins": int(np.sum(delta < 0)), "paired_seeds": len(delta),
            }
        else:
            record = {"paired_seeds": 0}
        layers[layer] = record
        gate = gate and record.get("paired_seeds") == 3 and record.get("wins", 0) >= 2 and record.get("lift_minus_random_mean", 1) < 0
    return {
        "layers": layers, "confirmation_gate_passed": bool(gate and len(layers) == len(LAYERS)),
        "gate_definition": "Balanced exact lift beats random in mean and at least 2/3 paired seeds at every locked fresh-text layer.",
    }


def render_summary(result: dict) -> str:
    lines = ["# EXP-057 locked fresh-text exact-lift confirmation", "", f"- Status: `{result['status']}`", f"- Duration: `{result['duration_hours']:.3f}` hours", f"- Complete: `{result['complete']}`", f"- Confirmation gate: `{result['aggregate']['confirmation_gate_passed']}`", "", "| Layer | Random | Balanced lift | Δ | Improvement | Wins |", "|---:|---:|---:|---:|---:|---:|"]
    for layer, row in result["aggregate"]["layers"].items():
        if not row.get("paired_seeds"):
            continue
        lines.append(f"| {layer} | {row['random_mean']:.8f} | {row['balanced_lift_mean']:.8f} | {row['lift_minus_random_mean']:+.8f} | {100*row['relative_improvement']:.3f}% | {row['wins']}/{row['paired_seeds']} |")
    lines += ["", "This locked confirmation uses sequence length 64 and disjoint train/validation offsets; it is still layer-local, not composed end-to-end NLL."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run EXP-057 within a two-hour TPU allocation.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/kaggle/working/qwen3-exp057-weights")
    parser.add_argument("--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=1.85)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-057 requires one TPU v5e-8")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic(); deadline = started_clock + args.max_wall_hours * 3600; started = _now()
    qwen_dir, storage = resolve_qwen_cache_dir(args.qwen_cache_dir, args.qwen_cache_storage)
    stage_path = output / "exp057-stage-manifest.json"
    notice = _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=started\nexperiment=EXP-057 exact-lift confirmation\nhost={socket.gethostname()}\nhard_budget_hours={args.max_wall_hours}")
    results = {}; removed = []; stage = "startup"
    update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="running", details={"started_at_utc": started})
    try:
        for layer in LAYERS:
            if time.monotonic() >= deadline:
                break
            stage = f"layer{layer}"
            train_args, train_manifest, train_dir = _cache_arguments(layer=layer, evaluation_only=False, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix="exp057", recovery_steps=STEPS, validation_windows=64, sequence_length=SEQUENCE_LENGTH, training_token_offset=TRAIN_OFFSET, validation_token_offset=VALIDATION_OFFSET)
            eval_args, eval_manifest, eval_dir = _cache_arguments(layer=layer, evaluation_only=True, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix="exp057", recovery_steps=STEPS, validation_windows=64, sequence_length=SEQUENCE_LENGTH, training_token_offset=TRAIN_OFFSET, validation_token_offset=VALIDATION_OFFSET)
            if not _valid_cache(train_manifest, train_dir, layer): build_cache(train_args)
            if time.monotonic() >= deadline: break
            if not _valid_cache(eval_manifest, eval_dir, layer): build_cache(eval_args)
            path = output / f"exp057-layer{layer}-result.json"
            result = run_layer(["--activation-cache-manifest", str(train_manifest), "--activation-cache-dir", str(train_dir), "--evaluation-cache-manifest", str(eval_manifest), "--evaluation-cache-dir", str(eval_dir), "--qwen-cache-dir", qwen_dir, "--total-steps", str(STEPS), "--checkpoints", "0,256,1024,2048,4096", "--seeds", ",".join(map(str, SEEDS)), "--arms", ",".join(ARMS), "--compute-dtype", "bfloat16", "--experiment-protocol", PROTOCOL, "--result-json", str(path), "--output-dir", str(output)], deadline_monotonic=deadline)
            results[str(layer)] = result
            update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="completed", details={"complete": result["complete"]})
            removed += _prune_cache_arrays(train_dir, output) + _prune_cache_arrays(eval_dir, output)
            jax.clear_caches(); gc.collect()
            if not result["complete"]: break
        complete = set(results) == {str(x) for x in LAYERS} and all(x.get("complete") for x in results.values())
        aggregate = aggregate_confirmation(results)
        result = {"protocol": PROTOCOL, "status": "completed" if complete else "deadline_partial", "started_at_utc": started, "completed_at_utc": _now(), "duration_hours": (time.monotonic()-started_clock)/3600, "max_wall_hours": args.max_wall_hours, "sequence_length": SEQUENCE_LENGTH, "training_token_offset": TRAIN_OFFSET, "validation_token_offset": VALIDATION_OFFSET, "layers": list(LAYERS), "seeds": list(SEEDS), "arms": list(ARMS), "layer_results": results, "aggregate": aggregate, "removed_regenerable_arrays": removed, "start_notification": notice, "complete": complete, "passed": bool(results) and all(x.get("passed") for x in results.values())}
        result_path = output / "extent-exact-lift-confirmation.json"; _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-exact-lift-confirmation-summary.md"; summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(stage_path, experiment=PROTOCOL, stage="final", status="completed", details={"complete": complete, "summary": str(summary)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus={'completed' if complete else 'deadline_partial'}\nexperiment=EXP-057 exact-lift confirmation\ngate={aggregate['confirmation_gate_passed']}")
        print(f"EXP057-COMPLETE complete={complete} gate={aggregate['confirmation_gate_passed']}\nsummary={summary.resolve()}")
        return result
    except BaseException as exc:
        failure = {"protocol": PROTOCOL, "status": "failed", "stage": stage, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "completed_layer_results": results}
        _write_json_with_output_mirror(output / "exp057-failure.json", failure, str(output))
        update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="failed", details={"error": str(exc)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-057\nstage={stage}\nerror={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__": main()
