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
from scripts.qwen_bridge_ablation_campaign import (
    _cache_arguments,
    _prune_cache_arrays,
    _valid_cache,
)
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import main as run_layer
from extent.experiment_stage import update_stage_manifest
from extent.qwen_source import QWEN3_1_7B_BASE


PROTOCOL = "exp063-qwen3-1.7b-transplant-depth-atlas"
SOURCE_MODEL = "1.7b-base"
SEEDS = (123, 456, 789)
ARMS = ("CONTROL-RANDOM", "BALANCED-RANK-LIFT")
STEPS = 8192
CHECKPOINTS = (0, 256, 1024, 2048, 4096, 8192)
SEQUENCE_LENGTH = 64
VALIDATION_WINDOWS = 128
TRAIN_OFFSET = 393_216
VALIDATION_OFFSET = 0
# Early/middle/late evidence is produced first; remaining layers fill the atlas.
LAYER_ORDER = (0, 13, 27, 6, 20, 3, 10, 17, 24, 1, 4, 7, 11, 14,
               18, 21, 25, 2, 5, 8, 9, 12, 15, 16, 19, 22, 23, 26)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arm_error(result: dict, seed: int, arm: str, step: int) -> float:
    return float(
        result["seeds"][str(seed)]["arms"][arm]["recovery"]
        ["evaluations"][str(step)]["decoder_output"]["relative_l2"]
    )


def aggregate_atlas(layer_results: dict[str, dict]) -> dict:
    layers: dict[str, dict] = {}
    final_deltas = []
    auc_deltas = []
    for layer, result in sorted(layer_results.items(), key=lambda item: int(item[0])):
        if not result.get("complete"):
            layers[layer] = {"complete": False}
            continue
        curves = {}
        for arm in ARMS:
            curves[arm] = np.asarray([
                np.mean([_arm_error(result, seed, arm, step) for seed in SEEDS])
                for step in CHECKPOINTS
            ], dtype=np.float64)
        random_curve = curves["CONTROL-RANDOM"]
        lift_curve = curves["BALANCED-RANK-LIFT"]
        x = np.asarray(CHECKPOINTS, dtype=np.float64) / STEPS
        random_auc = float(np.trapezoid(random_curve, x=x))
        lift_auc = float(np.trapezoid(lift_curve, x=x))
        paired_final = np.asarray([
            _arm_error(result, seed, "BALANCED-RANK-LIFT", STEPS)
            - _arm_error(result, seed, "CONTROL-RANDOM", STEPS)
            for seed in SEEDS
        ])
        record = {
            "complete": True,
            "random_curve": dict(zip(map(str, CHECKPOINTS), map(float, random_curve))),
            "exact_lift_curve": dict(zip(map(str, CHECKPOINTS), map(float, lift_curve))),
            "random_normalized_auc": random_auc,
            "exact_lift_normalized_auc": lift_auc,
            "auc_relative_improvement": float(1.0 - lift_auc / random_auc),
            "final_random_mean": float(random_curve[-1]),
            "final_exact_lift_mean": float(lift_curve[-1]),
            "final_exact_minus_random": float(np.mean(paired_final)),
            "final_relative_improvement": float(1.0 - lift_curve[-1] / random_curve[-1]),
            "final_seed_wins": int(np.sum(paired_final < 0)),
            "layer_gate_passed": bool(np.mean(paired_final) < 0 and np.sum(paired_final < 0) >= 2),
        }
        layers[layer] = record
        final_deltas.append(record["final_exact_minus_random"])
        auc_deltas.append(lift_auc - random_auc)
    complete_rows = [row for row in layers.values() if row.get("complete")]
    wins = sum(bool(row["layer_gate_passed"]) for row in complete_rows)
    complete = len(complete_rows) == len(LAYER_ORDER)
    return {
        "layers": layers,
        "completed_layers": len(complete_rows),
        "layer_wins": wins,
        "mean_final_exact_minus_random": float(np.mean(final_deltas)) if final_deltas else None,
        "mean_auc_exact_minus_random": float(np.mean(auc_deltas)) if auc_deltas else None,
        "scientific_gate_passed": bool(
            complete
            and wins >= 21
            and np.mean(final_deltas) < 0
            and np.mean(auc_deltas) < 0
        ) if complete_rows else False,
        "gate_definition": (
            "All 28 layers complete; exact lift wins at least 21/28 layer gates "
            "and has lower across-layer mean final error and normalized recovery AUC."
        ),
    }


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# EXP-063 Qwen3-1.7B transplant depth atlas",
        "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Completed layers: `{aggregate['completed_layers']}/28`",
        f"- Exact-lift layer wins: `{aggregate['layer_wins']}/{aggregate['completed_layers']}`",
        f"- Scientific gate: `{aggregate['scientific_gate_passed']}`",
        "",
        "| Layer | Random @8192 | Exact @8192 | Improvement | AUC improvement | Seed wins | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, row in aggregate["layers"].items():
        if not row.get("complete"):
            continue
        lines.append(
            f"| {layer} | {row['final_random_mean']:.7f} | "
            f"{row['final_exact_lift_mean']:.7f} | "
            f"{100 * row['final_relative_improvement']:+.2f}% | "
            f"{100 * row['auc_relative_improvement']:+.2f}% | "
            f"{row['final_seed_wins']}/3 | {row['layer_gate_passed']} |"
        )
    lines += [
        "",
        "AUC integrates held-out decoder-output relative L2 over the registered "
        "0/256/1024/2048/4096/8192 recovery checkpoints. Lower is better.",
    ]
    return "\n".join(lines) + "\n"


def _valid_layer_result(path: Path, layer: int) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("protocol") != PROTOCOL
        or int(payload.get("target_layer", -1)) != layer
        or payload.get("source")
        != f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}"
        or not payload.get("complete")
        or not payload.get("passed")
    ):
        return None
    return payload


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run the Qwen3-1.7B exact-lift depth atlas.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp063-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-063 requires one TPU v5e-8")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    started = _now()
    stage = "startup"
    stage_path = output / "exp063-stage-manifest.json"
    start_notice = _safe_notify(
        args.telegram,
        f"Extent TPU campaign\nstatus=started\nexperiment=EXP-063 Qwen3-1.7B atlas"
        f"\nhost={socket.gethostname()}\nhard_budget_hours={args.max_wall_hours}",
    )
    update_stage_manifest(
        stage_path, experiment=PROTOCOL, stage=stage, status="running",
        details={"started_at_utc": started, "layer_order": list(LAYER_ORDER)},
    )
    results: dict[str, dict] = {}
    removed: list[str] = []
    try:
        for layer in LAYER_ORDER:
            if time.monotonic() >= deadline:
                break
            stage = f"layer-{layer}"
            result_path = output / f"exp063-layer{layer}-training.json"
            existing = _valid_layer_result(result_path, layer) if args.resume else None
            if existing is not None:
                print(f"exp063_layer={layer} RESUME-PASS")
                results[str(layer)] = existing
                continue
            common = dict(
                layer=layer,
                qwen_cache_dir=args.qwen_cache_dir,
                qwen_storage="ram",
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output,
                compute_dtype="bfloat16",
                storage_dtype="float16",
                per_device_windows=4,
                artifact_prefix="exp063",
                recovery_steps=STEPS,
                validation_windows=VALIDATION_WINDOWS,
                sequence_length=SEQUENCE_LENGTH,
                training_token_offset=TRAIN_OFFSET,
                validation_token_offset=VALIDATION_OFFSET,
                source_model=SOURCE_MODEL,
            )
            train_args, train_manifest, train_dir = _cache_arguments(
                evaluation_only=False, **common
            )
            eval_args, eval_manifest, eval_dir = _cache_arguments(
                evaluation_only=True, **common
            )
            if not _valid_cache(train_manifest, train_dir, layer):
                build_cache(train_args)
            if time.monotonic() >= deadline:
                break
            if not _valid_cache(eval_manifest, eval_dir, layer):
                build_cache(eval_args)
            layer_result = run_layer(
                [
                    "--activation-cache-manifest", str(train_manifest),
                    "--activation-cache-dir", str(train_dir),
                    "--evaluation-cache-manifest", str(eval_manifest),
                    "--evaluation-cache-dir", str(eval_dir),
                    "--qwen-cache-dir", args.qwen_cache_dir,
                    "--source-model", SOURCE_MODEL,
                    "--total-steps", str(STEPS),
                    "--checkpoints", ",".join(map(str, CHECKPOINTS)),
                    "--seeds", ",".join(map(str, SEEDS)),
                    "--arms", ",".join(ARMS),
                    "--compute-dtype", "bfloat16",
                    "--experiment-protocol", PROTOCOL,
                    "--result-json", str(result_path),
                    "--output-dir", str(output),
                ],
                deadline_monotonic=deadline,
            )
            results[str(layer)] = layer_result
            update_stage_manifest(
                stage_path, experiment=PROTOCOL, stage=stage,
                status="completed" if layer_result.get("complete") else "deadline_partial",
                details={"complete": bool(layer_result.get("complete"))},
            )
            removed += _prune_cache_arrays(train_dir, output)
            removed += _prune_cache_arrays(eval_dir, output)
            jax.clear_caches()
            gc.collect()
            if not layer_result.get("complete"):
                break

        aggregate = aggregate_atlas(results)
        complete = aggregate["completed_layers"] == len(LAYER_ORDER)
        result = {
            "protocol": PROTOCOL,
            "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
            "status": "completed" if complete else "deadline_partial",
            "complete": complete,
            "passed": bool(results) and all(item.get("passed") for item in results.values()),
            "started_at_utc": started,
            "completed_at_utc": _now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
            "max_wall_hours": args.max_wall_hours,
            "layer_order": list(LAYER_ORDER),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "recovery_steps": STEPS,
            "checkpoints": list(CHECKPOINTS),
            "sequence_length": SEQUENCE_LENGTH,
            "training_token_offset": TRAIN_OFFSET,
            "validation_token_offset": VALIDATION_OFFSET,
            "layer_results": results,
            "aggregate": aggregate,
            "removed_regenerable_arrays": removed,
            "start_notification": start_notice,
        }
        result_path = output / "extent-qwen17-transplant-atlas.json"
        _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-qwen17-transplant-atlas-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(
            stage_path, experiment=PROTOCOL, stage="final", status=result["status"],
            details={"completed_layers": aggregate["completed_layers"], "summary": str(summary)},
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment=EXP-063 Qwen3-1.7B atlas"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\nlayers={aggregate['completed_layers']}/28\nwins={aggregate['layer_wins']}",
        )
        print(
            f"EXP063-{result['status'].upper()} layers={aggregate['completed_layers']}/28 "
            f"wins={aggregate['layer_wins']}\nsummary={summary.resolve()}"
        )
        return result
    except BaseException as exc:
        failure = {
            "protocol": PROTOCOL,
            "status": "failed",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_layer_results": results,
            "completed_at_utc": _now(),
        }
        _write_json_with_output_mirror(output / "exp063-failure.json", failure, str(output))
        update_stage_manifest(
            stage_path, experiment=PROTOCOL, stage=stage, status="failed",
            details={"error": str(exc)},
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-063"
            f"\nstage={stage}\nerror={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
