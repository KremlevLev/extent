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
from extent.teacher_activation_cache import file_sha256


PROTOCOL = "exp058-one-hour-exact-lift-composition-pilot"
LAYERS = (0, 18)
TRAIN_SEED = 123
PSEUDO_SEEDS = (123, 456)  # random, balanced exact lift
STEPS = 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary(result: dict) -> str:
    aggregate = result["aggregate"]
    if "paired_seeds" in aggregate:
        lines = [
            "# EXP-059 multiseed exact-lift composition confirmation", "",
            f"- Duration: `{result['duration_hours']:.3f}` hours",
            f"- Numerical pass: `{result['passed']}`",
            f"- Confirmation gate: `{aggregate['scientific_gate_passed']}`", "",
            "| Seed | Random NLL | Exact-lift NLL | Exact - random |", "|---:|---:|---:|---:|",
        ]
        for row in aggregate["paired_seeds"]:
            lines.append(f"| {row['seed']} | {row['random_mean_nll']:.8f} | {row['exact_mean_nll']:.8f} | {row['exact_minus_random_nll']:+.8f} |")
        lines += ["", f"Mean exact minus random NLL: `{aggregate['mean_exact_minus_random_nll']:+.8f}`.", f"Paired hierarchical-bootstrap 95% CI: `[{aggregate['bootstrap_95_ci'][0]:+.8f}, {aggregate['bootstrap_95_ci'][1]:+.8f}]`.", ""]
        return "\n".join(lines)
    return "\n".join([
        "# EXP-058 exact-lift two-layer composition pilot", "",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Numerical pass: `{result['passed']}`",
        f"- Pilot gate: `{aggregate['pilot_gate_passed']}`", "",
        "| Branch | Mean NLL | Excess vs original |", "|---|---:|---:|",
        f"| Original Qwen | {aggregate['original_mean_nll']:.8f} | +0.00000000 |",
        f"| Random Mamba layers 0+18 | {aggregate['random_composition_mean_nll']:.8f} | {aggregate['random_minus_original_nll']:+.8f} |",
        f"| Exact-lift Mamba layers 0+18 | {aggregate['exact_lift_composition_mean_nll']:.8f} | {aggregate['exact_minus_original_nll']:+.8f} |",
        "", f"Exact lift minus random NLL: `{aggregate['exact_minus_random_nll']:+.8f}`.",
        "", "This is a one-seed, 1,024-step pilot. It ranks initialization under composition but is not publication-level confirmation.", "",
    ])


def main(
    argv: list[str] | None = None,
    *,
    protocol: str = PROTOCOL,
    train_seeds: tuple[int, ...] = (TRAIN_SEED,),
    analysis_mode: str = "exact_lift_pilot",
    artifact_prefix: str = "exp058",
    final_stem: str = "extent-exact-lift-composition-pilot",
    experiment_label: str = "EXP-058 exact-lift composition pilot",
) -> dict:
    parser = argparse.ArgumentParser(description="Run the one-hour EXP-058 composition pilot.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default=f"/kaggle/working/qwen3-{artifact_prefix}-weights")
    parser.add_argument("--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-058 requires one TPU v5e-8")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic(); started = _now(); stage = "startup"
    qwen_dir, storage = resolve_qwen_cache_dir(args.qwen_cache_dir, args.qwen_cache_storage)
    stage_path = output / f"{artifact_prefix}-stage-manifest.json"
    notice = _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=started\nexperiment={experiment_label}\nhost={socket.gethostname()}")
    update_stage_manifest(stage_path, experiment=protocol, stage=stage, status="running", details={"started_at_utc": started})
    layer_results = {}; endpoint_specs = {}; removed = []
    try:
        for layer in LAYERS:
            stage = f"train-layer{layer}"
            train_args, train_manifest, train_dir = _cache_arguments(layer=layer, evaluation_only=False, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix=artifact_prefix, recovery_steps=STEPS, validation_windows=256, sequence_length=32, training_token_offset=196608, validation_token_offset=16384)
            eval_args, eval_manifest, eval_dir = _cache_arguments(layer=layer, evaluation_only=True, qwen_cache_dir=qwen_dir, qwen_storage=storage, dataset_cache_dir=args.dataset_cache_dir, output_dir=output, compute_dtype="bfloat16", storage_dtype="float16", per_device_windows=4, artifact_prefix=artifact_prefix, recovery_steps=STEPS, validation_windows=256, sequence_length=32, training_token_offset=196608, validation_token_offset=16384)
            if not _valid_cache(train_manifest, train_dir, layer): build_cache(train_args)
            if not _valid_cache(eval_manifest, eval_dir, layer): build_cache(eval_args)
            layer_path = output / f"{artifact_prefix}-layer{layer}-training.json"
            training, trained = train_layer(["--activation-cache-manifest", str(train_manifest), "--activation-cache-dir", str(train_dir), "--evaluation-cache-manifest", str(eval_manifest), "--evaluation-cache-dir", str(eval_dir), "--qwen-cache-dir", qwen_dir, "--total-steps", str(STEPS), "--checkpoints", "0,256,1024", "--seeds", ",".join(map(str, train_seeds)), "--arms", "CONTROL-RANDOM,BALANCED-RANK-LIFT", "--experiment-protocol", protocol, "--compute-dtype", "bfloat16", "--result-json", str(layer_path), "--output-dir", str(output)], return_endpoint_params=True)
            layer_results[str(layer)] = training
            endpoint_dir = output / f"{artifact_prefix}-layer{layer}-endpoints"
            endpoints = {}
            for seed in train_seeds:
                endpoints[str(seed)] = {"JOINT-MIXER-DECODER": trained[str(seed)]["CONTROL-RANDOM"]}
                endpoints[str(seed + 1000)] = {"JOINT-MIXER-DECODER": trained[str(seed)]["BALANCED-RANK-LIFT"]}
            endpoint_ids = tuple(train_seeds) + tuple(seed + 1000 for seed in train_seeds)
            compatibility = {"source": f"{QW3.repo_id}@{QW3.revision}", "target_layer": layer, "seeds": list(endpoint_ids), "total_steps": STEPS, "protocol": protocol}
            metadata = save_endpoint_checkpoint(endpoint_dir, endpoints, compatibility=compatibility)
            endpoint_specs[str(layer)] = {"steps": STEPS, "endpoint_dir": str(endpoint_dir), "endpoint_checkpoint_sha256": metadata["checkpoint_sha256"], "standalone_joint_metrics": {}}
            update_stage_manifest(stage_path, experiment=protocol, stage=stage, status="completed", details={"checkpoint": metadata["checkpoint_sha256"]})
            removed += _prune_cache_arrays(train_dir, output)
            if layer != 0: removed += _prune_cache_arrays(eval_dir, output)
            jax.clear_caches(); gc.collect()

        stage = "streamed-composition"
        endpoint_ids = tuple(train_seeds) + tuple(seed + 1000 for seed in train_seeds)
        evaluation_dir = output / f"{artifact_prefix}-layer0-validation-cache"
        index = {
            "source": f"{QW3.repo_id}@{QW3.revision}", "protocol": protocol,
            "method": "paired_random_vs_exact_operator_lift_two_layer_composition",
            "analysis_mode": analysis_mode, "evaluation_protocol": protocol,
            "seeds": list(endpoint_ids), "target_layers": list(LAYERS),
            "layer_sets": {"2": list(LAYERS)}, "layer_budgets": {str(x): STEPS for x in LAYERS},
            "evaluation_cache": {"manifest": str(evaluation_dir / f"{artifact_prefix}-layer0-validation-manifest.json"), "artifact_dir": str(evaluation_dir)},
            "layers": endpoint_specs,
            "notes": [
                "Endpoint IDs equal the model seed for random and model seed + 1000 for its paired balanced exact-lift arm.",
                f"Paired training seeds: {list(train_seeds)}.",
            ],
        }
        index_path = output / f"{artifact_prefix}-endpoint-index.json"; _write_json(index_path, index)
        composition = evaluate_composition(["--endpoint-index", str(index_path), "--qwen-cache-dir", qwen_dir, "--compute-dtype", "bfloat16", "--evaluation-batch-windows", "4", "--per-device-windows", "4", "--bootstrap-samples", "2000", "--data-parallel", "--prune-consumed-shards", "--result-json", str(output / f"{artifact_prefix}-composition.json"), "--output-dir", str(output)])
        result = {**composition, "duration_hours": (time.monotonic()-started_clock)/3600, "started_at_utc": started, "completed_at_utc": _now(), "training_seeds": list(train_seeds), "layer_training_results": layer_results, "start_notification": notice, "removed_regenerable_arrays": removed}
        result_path = output / f"{final_stem}.json"; _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / f"{final_stem}-summary.md"; summary.write_text(_summary(result), encoding="utf-8")
        gate = result["aggregate"]["scientific_gate_passed"]
        update_stage_manifest(stage_path, experiment=protocol, stage="final", status="completed", details={"scientific_gate": gate})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=completed\nexperiment={experiment_label}\ngate={gate}")
        print(f"{artifact_prefix.upper()}-COMPLETE gate={gate}\nsummary={summary.resolve()}")
        return result
    except BaseException as exc:
        failure = {"protocol": protocol, "status": "failed", "stage": stage, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "completed_layer_results": layer_results}
        _write_json_with_output_mirror(output / f"{artifact_prefix}-failure.json", failure, str(output))
        update_stage_manifest(stage_path, experiment=protocol, stage=stage, status="failed", details={"error": str(exc)})
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=failed\nexperiment={experiment_label}\nstage={stage}\nerror={type(exc).__name__}: {exc}")
        raise


QW3 = QWEN3_14B

if __name__ == "__main__": main()
