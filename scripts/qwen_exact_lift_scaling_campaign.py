from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import socket
import time
import traceback

import jax

from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_bridge_ablation_campaign import _cache_arguments, _prune_cache_arrays, _valid_cache
from scripts.qwen_depth_objective_run import resolve_qwen_cache_dir
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import main as train_layer
from scripts.qwen_progressive_composition_eval import main as evaluate_composition
from extent.endpoint_checkpoint import save_endpoint_checkpoint
from extent.experiment_stage import update_stage_manifest
from extent.qwen_source import QWEN3_14B


PROTOCOL = "exp060-exact-lift-progressive-scaling"
SEEDS = (123, 456, 789)
ENDPOINT_IDS = SEEDS + tuple(seed + 1000 for seed in SEEDS)
LAYER_SETS = {2: (0, 18), 4: (0, 12, 18, 29), 8: (0, 6, 12, 18, 23, 29, 34, 39)}
TARGET_LAYERS = tuple(sorted({layer for layers in LAYER_SETS.values() for layer in layers}))
LAYER_BUDGETS = {layer: (8192 if layer == 0 else 4096) for layer in TARGET_LAYERS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_endpoint(output: Path, layer: int) -> bool:
    result_path = output / f"exp060-layer{layer}-training.json"
    metadata_path = output / f"exp060-layer{layer}-endpoints" / "checkpoint.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = metadata_path.parent / metadata["checkpoint_file"]
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError):
        return False
    compatibility = metadata.get("compatibility", {})
    return bool(
        result.get("complete") and result.get("passed")
        and compatibility.get("protocol") == PROTOCOL
        and compatibility.get("target_layer") == layer
        and compatibility.get("seeds") == list(ENDPOINT_IDS)
        and compatibility.get("total_steps") == LAYER_BUDGETS[layer]
        and payload.exists() and payload.stat().st_size == metadata.get("checkpoint_bytes")
    )


def _endpoint_spec(output: Path, layer: int) -> dict:
    directory = output / f"exp060-layer{layer}-endpoints"
    metadata = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    return {"steps": LAYER_BUDGETS[layer], "endpoint_dir": str(directory), "endpoint_checkpoint_sha256": metadata["checkpoint_sha256"], "standalone_joint_metrics": {}}


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = ["# EXP-060 exact-lift progressive scaling", "", f"- Duration: `{result['duration_hours']:.3f}` hours", f"- Numerical pass: `{result['passed']}`", f"- Scientific gate: `{aggregate['scientific_gate_passed']}`", "", "| Replacements | Layers | Mean exact-random NLL | Wins | 95% CI | Gate |", "|---:|---|---:|---:|---|---:|"]
    for stage in aggregate["stages"]:
        lines.append(f"| {stage['replacement_count']} | {','.join(map(str, stage['layers']))} | {stage['mean_exact_minus_random_nll']:+.8f} | {stage['wins']}/3 | [{stage['bootstrap_95_ci'][0]:+.8f}, {stage['bootstrap_95_ci'][1]:+.8f}] | {stage['stage_gate_passed']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run resilient EXP-060 exact-lift 2/4/8-layer scaling.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/kaggle/working/qwen3-exp060-weights")
    parser.add_argument("--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-060 requires one TPU v5e-8")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic(); started = _now(); stage = "startup"
    qwen_dir, storage = resolve_qwen_cache_dir(args.qwen_cache_dir, args.qwen_cache_storage)
    stage_path = output / "exp060-stage-manifest.json"
    notice = _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=started\nexperiment=EXP-060 exact-lift progressive scaling\nhost={socket.gethostname()}")
    update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="running", details={"started_at_utc": started})
    layer_results = {}; removed = []
    try:
        # Interior layers first create inexpensive durable restart boundaries; layer 0 is last and retains evaluation data.
        for layer in tuple(x for x in TARGET_LAYERS if x != 0) + (0,):
            stage = f"train-layer{layer}"
            if args.resume and _valid_endpoint(output, layer):
                print(f"exp060_layer={layer} RESUME-PASS")
                layer_results[str(layer)] = json.loads((output / f"exp060-layer{layer}-training.json").read_text(encoding="utf-8"))
                continue
            steps = LAYER_BUDGETS[layer]
            train_args, train_manifest, train_dir = _cache_arguments(layer=layer, evaluation_only=False, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix="exp060", recovery_steps=steps, validation_windows=256, sequence_length=32, training_token_offset=262144, validation_token_offset=24576)
            eval_args, eval_manifest, eval_dir = _cache_arguments(layer=layer, evaluation_only=True, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix="exp060", recovery_steps=steps, validation_windows=256, sequence_length=32, training_token_offset=262144, validation_token_offset=24576)
            if not _valid_cache(train_manifest, train_dir, layer): build_cache(train_args)
            if not _valid_cache(eval_manifest, eval_dir, layer): build_cache(eval_args)
            checkpoints = "0,1024,2048,4096,8192" if steps == 8192 else "0,1024,2048,4096"
            result_path = output / f"exp060-layer{layer}-training.json"
            training, trained = train_layer(["--activation-cache-manifest", str(train_manifest), "--activation-cache-dir", str(train_dir), "--evaluation-cache-manifest", str(eval_manifest), "--evaluation-cache-dir", str(eval_dir), "--qwen-cache-dir", qwen_dir, "--total-steps", str(steps), "--checkpoints", checkpoints, "--seeds", ",".join(map(str, SEEDS)), "--arms", "CONTROL-RANDOM,BALANCED-RANK-LIFT", "--experiment-protocol", PROTOCOL, "--compute-dtype", "bfloat16", "--result-json", str(result_path), "--output-dir", str(output)], return_endpoint_params=True)
            endpoints = {}
            for seed in SEEDS:
                endpoints[str(seed)] = {"JOINT-MIXER-DECODER": trained[str(seed)]["CONTROL-RANDOM"]}
                endpoints[str(seed + 1000)] = {"JOINT-MIXER-DECODER": trained[str(seed)]["BALANCED-RANK-LIFT"]}
            endpoint_dir = output / f"exp060-layer{layer}-endpoints"
            metadata = save_endpoint_checkpoint(endpoint_dir, endpoints, compatibility={"source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}", "target_layer": layer, "seeds": list(ENDPOINT_IDS), "total_steps": steps, "protocol": PROTOCOL})
            layer_results[str(layer)] = training
            update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="completed", details={"checkpoint_sha256": metadata["checkpoint_sha256"], "disk_free_gib": shutil.disk_usage(output).free / 1024**3})
            removed += _prune_cache_arrays(train_dir, output)
            if layer != 0: removed += _prune_cache_arrays(eval_dir, output)
            jax.clear_caches(); gc.collect()

        stage = "progressive-streamed-evaluation"
        endpoint_specs = {str(layer): _endpoint_spec(output, layer) for layer in TARGET_LAYERS}
        evaluation_dir = output / "exp060-layer0-validation-cache"
        index = {"source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}", "protocol": PROTOCOL, "method": "paired_exact_operator_lift_progressive_composition", "analysis_mode": "exact_lift_scaling", "evaluation_protocol": PROTOCOL, "seeds": list(ENDPOINT_IDS), "target_layers": list(TARGET_LAYERS), "layer_sets": {str(count): list(layers) for count, layers in LAYER_SETS.items()}, "layer_budgets": {str(layer): steps for layer, steps in LAYER_BUDGETS.items()}, "evaluation_cache": {"manifest": str(evaluation_dir / "exp060-layer0-validation-manifest.json"), "artifact_dir": str(evaluation_dir)}, "layers": endpoint_specs, "notes": ["Endpoint ID seed+1000 is the exact-lift arm paired with random endpoint ID seed.", "EXP-060 tests 2/4/8 nested replacement scaling with matched recovery per layer."]}
        index_path = output / "exp060-endpoint-index.json"; _write_json(index_path, index)
        composition = evaluate_composition(["--endpoint-index", str(index_path), "--qwen-cache-dir", qwen_dir, "--compute-dtype", "bfloat16", "--evaluation-batch-windows", "4", "--per-device-windows", "4", "--bootstrap-samples", "2000", "--data-parallel", "--prune-consumed-shards", "--result-json", str(output / "exp060-composition.json"), "--output-dir", str(output)])
        result = {**composition, "duration_hours": (time.monotonic()-started_clock)/3600, "started_at_utc": started, "completed_at_utc": _now(), "layer_training_results": layer_results, "start_notification": notice, "removed_regenerable_arrays": removed}
        result_path = output / "extent-exact-lift-scaling-campaign.json"; _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-exact-lift-scaling-campaign-summary.md"; summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(stage_path, experiment=PROTOCOL, stage="final", status="completed", details={"scientific_gate": result["aggregate"]["scientific_gate_passed"]})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=completed\nexperiment=EXP-060 scaling\ngate={result['aggregate']['scientific_gate_passed']}\nduration_hours={result['duration_hours']:.3f}")
        print(f"EXP060-COMPLETE gate={result['aggregate']['scientific_gate_passed']}\nsummary={summary.resolve()}")
        return result
    except BaseException as exc:
        failure = {"protocol": PROTOCOL, "status": "failed", "stage": stage, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "completed_layers": sorted(layer_results)}
        _write_json_with_output_mirror(output / "exp060-failure.json", failure, str(output))
        update_stage_manifest(stage_path, experiment=PROTOCOL, stage=stage, status="failed", details={"error": str(exc)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-060\nstage={stage}\nerror={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
