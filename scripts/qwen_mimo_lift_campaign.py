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

from scripts.qwen_activation_cache import main as build_activation_cache
from scripts.qwen_bridge_ablation_campaign import _cache_arguments, _prune_cache_arrays, _valid_cache
from scripts.qwen_depth_objective_run import resolve_qwen_cache_dir
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import ARM_ORDER, PROTOCOL, main as run_layer_screen
from extent.experiment_stage import update_stage_manifest


CAMPAIGN_PROTOCOL = "exp056-operator-preserving-mimo-lift-campaign"
TARGET_LAYERS = (18, 6, 29, 0)
SEEDS = (123, 456, 789)
RECOVERY_STEPS = 4096


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_result(path: Path, layer: int) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if payload.get("protocol") != PROTOCOL or payload.get("target_layer") != layer:
        return None
    return payload


def aggregate_mimo_lift(layer_results: dict[str, dict]) -> dict:
    layers = {}
    internal_gate = True
    for layer, result in sorted(layer_results.items(), key=lambda item: int(item[0])):
        checkpoints = result.get("training", {}).get("checkpoints", [])
        comparisons = {}
        for arm in ARM_ORDER[1:]:
            trajectory = []
            for checkpoint in checkpoints:
                random_values, arm_values = [], []
                for seed_result in result.get("seeds", {}).values():
                    random_recovery = seed_result.get("arms", {}).get("CONTROL-RANDOM", {}).get("recovery", {})
                    arm_recovery = seed_result.get("arms", {}).get(arm, {}).get("recovery", {})
                    random_eval = random_recovery.get("evaluations", {}).get(str(checkpoint))
                    arm_eval = arm_recovery.get("evaluations", {}).get(str(checkpoint))
                    if random_eval and arm_eval:
                        random_values.append(float(random_eval["decoder_output"]["relative_l2"]))
                        arm_values.append(float(arm_eval["decoder_output"]["relative_l2"]))
                if random_values:
                    delta = np.asarray(arm_values) - np.asarray(random_values)
                    trajectory.append({
                        "step": checkpoint, "arm_minus_random_mean": float(delta.mean()),
                        "wins_over_random": int(np.sum(delta < 0)), "paired_seeds": len(delta),
                        "arm_mean": float(np.mean(arm_values)), "random_mean": float(np.mean(random_values)),
                    })
            crossover = next((point["step"] for point in trajectory if point["arm_minus_random_mean"] < 0), None)
            comparisons[arm] = {"trajectory": trajectory, "earliest_mean_crossover_step": crossover}
        layers[layer] = comparisons
        if int(layer) != 0:
            balanced = comparisons.get("BALANCED-RANK-LIFT", {})
            flat = comparisons.get("CONTROL-FLAT-QKVO", {})
            final = balanced.get("trajectory", [])[-1:] or [{}]
            balanced_cross = balanced.get("earliest_mean_crossover_step")
            flat_cross = flat.get("earliest_mean_crossover_step")
            internal_gate = internal_gate and bool(
                final[0].get("paired_seeds") == len(SEEDS)
                and final[0].get("wins_over_random", 0) >= 2
                and final[0].get("arm_minus_random_mean", 1) < 0
                and balanced_cross is not None
                and (flat_cross is None or balanced_cross <= flat_cross)
            )
    return {
        "layers": layers,
        "screening_gate_passed": bool(internal_gate and {"6", "18", "29"}.issubset(layers)),
        "gate_definition": "At each internal layer, balanced lift beats random in mean and >=2/3 seeds at step 4096, and crosses random no later than flat QKVO. Layer 0 is diagnostic only.",
    }


def render_summary(result: dict) -> str:
    lines = [
        "# EXP-056 operator-preserving MIMO lift campaign", "",
        f"- Status: `{result['status']}`", f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Complete: `{result['complete']}`", f"- Numerical pass: `{result['passed']}`",
        f"- Screening gate: `{result['aggregate']['screening_gate_passed']}`", "",
        "| Layer | Arm | Final Δ vs random | Wins | Earliest crossover |", "|---:|---|---:|---:|---:|",
    ]
    for layer, arms in result["aggregate"]["layers"].items():
        for arm in ARM_ORDER[1:]:
            record = arms.get(arm, {})
            trajectory = record.get("trajectory", [])
            if not trajectory:
                continue
            final = trajectory[-1]
            crossover = record.get("earliest_mean_crossover_step")
            lines.append(
                f"| {layer} | {arm} | {final['arm_minus_random_mean']:+.8f} | "
                f"{final['wins_over_random']}/{final['paired_seeds']} | {crossover if crossover is not None else 'none'} |"
            )
    lines += ["", "Negative delta is better. Layer 0 is a boundary diagnostic and cannot veto the internal-layer gate."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run resilient EXP-056 on one TPU v5e-8 session.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--result-json", default="/kaggle/working/output/extent-mimo-lift-campaign.json")
    parser.add_argument("--qwen-cache-dir", default="/kaggle/working/qwen3-exp056-weights")
    parser.add_argument("--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--storage-dtype", default="float16")
    parser.add_argument("--per-device-windows", type=int, default=4)
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--max-wall-hours", type=float, default=7.5)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-056 requires exactly eight TPU devices")
    if args.max_wall_hours <= 0:
        raise ValueError("max-wall-hours must be positive")

    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    started = _utc_now()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    stage_path = output / "exp056-campaign-stage-manifest.json"
    qwen_dir, qwen_storage = resolve_qwen_cache_dir(args.qwen_cache_dir, args.qwen_cache_storage)
    start_notice = _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=started\nexperiment=EXP-056 exact MIMO lift\nhost={socket.gethostname()}\nhard_budget_hours={args.max_wall_hours}")
    layer_results: dict[str, dict] = {}
    removed = []
    stage = "startup"
    update_stage_manifest(stage_path, experiment=CAMPAIGN_PROTOCOL, stage=stage, status="running", details={"started_at_utc": started})
    try:
        for layer in TARGET_LAYERS:
            if time.monotonic() >= deadline:
                break
            stage = f"layer{layer}"
            layer_path = output / f"exp056-layer{layer}-result.json"
            cached = _valid_result(layer_path, layer) if args.resume else None
            if cached and cached.get("complete"):
                layer_results[str(layer)] = cached
                continue
            train_args, train_manifest, train_dir = _cache_arguments(
                layer=layer, evaluation_only=False, qwen_cache_dir=qwen_dir, qwen_storage=qwen_storage,
                dataset_cache_dir=args.dataset_cache_dir, output_dir=output,
                compute_dtype=args.compute_dtype, storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows, artifact_prefix="exp056",
                recovery_steps=RECOVERY_STEPS, validation_windows=64,
            )
            eval_args, eval_manifest, eval_dir = _cache_arguments(
                layer=layer, evaluation_only=True, qwen_cache_dir=qwen_dir, qwen_storage=qwen_storage,
                dataset_cache_dir=args.dataset_cache_dir, output_dir=output,
                compute_dtype=args.compute_dtype, storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows, artifact_prefix="exp056",
                recovery_steps=RECOVERY_STEPS, validation_windows=64,
            )
            if not (args.resume and _valid_cache(train_manifest, train_dir, layer)):
                build_activation_cache(train_args)
            if time.monotonic() >= deadline:
                break
            if not (args.resume and _valid_cache(eval_manifest, eval_dir, layer)):
                build_activation_cache(eval_args)
            layer_result = run_layer_screen([
                "--activation-cache-manifest", str(train_manifest), "--activation-cache-dir", str(train_dir),
                "--evaluation-cache-manifest", str(eval_manifest), "--evaluation-cache-dir", str(eval_dir),
                "--qwen-cache-dir", qwen_dir, "--total-steps", str(RECOVERY_STEPS),
                "--checkpoints", "0,256,1024,2048,4096", "--seeds", ",".join(map(str, SEEDS)),
                "--evaluation-batch-windows", str(args.evaluation_batch_windows),
                "--compute-dtype", args.compute_dtype, "--result-json", str(layer_path), "--output-dir", str(output),
            ], deadline_monotonic=deadline)
            layer_results[str(layer)] = layer_result
            update_stage_manifest(stage_path, experiment=CAMPAIGN_PROTOCOL, stage=stage, status="completed", details={"complete": layer_result["complete"]})
            removed += _prune_cache_arrays(train_dir, output) + _prune_cache_arrays(eval_dir, output)
            jax.clear_caches(); gc.collect()
            if not layer_result["complete"]:
                break

        complete = set(layer_results) == {str(x) for x in TARGET_LAYERS} and all(x.get("complete") for x in layer_results.values())
        passed = bool(layer_results) and all(x.get("passed") for x in layer_results.values())
        aggregate = aggregate_mimo_lift(layer_results)
        result = {
            "protocol": CAMPAIGN_PROTOCOL, "status": "completed" if complete else "deadline_partial",
            "started_at_utc": started, "completed_at_utc": _utc_now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
            "max_wall_hours": args.max_wall_hours, "target_layers": list(TARGET_LAYERS),
            "seeds": list(SEEDS), "arm_order": list(ARM_ORDER), "recovery_steps": RECOVERY_STEPS,
            "layer_results": layer_results, "aggregate": aggregate,
            "removed_regenerable_arrays": removed, "start_notification": start_notice,
            "complete": complete, "passed": passed,
        }
        _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-mimo-lift-campaign-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(stage_path, experiment=CAMPAIGN_PROTOCOL, stage="final", status="completed", details={"complete": complete, "summary": str(summary)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus={'completed' if complete else 'deadline_partial'}\nexperiment=EXP-056 exact MIMO lift\nlayers={','.join(layer_results)}\nscreening_gate={aggregate['screening_gate_passed']}")
        print(f"EXP056-COMPLETE complete={complete} gate={aggregate['screening_gate_passed']}")
        print(f"result_json={result_path.resolve()}\ncompact_summary={summary.resolve()}")
        return result
    except BaseException as exc:
        failure = {"protocol": CAMPAIGN_PROTOCOL, "status": "failed", "failed_stage": stage, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "completed_layer_results": layer_results}
        _write_json_with_output_mirror(output / "exp056-campaign-failure.json", failure, str(output))
        update_stage_manifest(stage_path, experiment=CAMPAIGN_PROTOCOL, stage=stage, status="failed", details={"error": str(exc)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-056 exact MIMO lift\nstage={stage}\nerror={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
