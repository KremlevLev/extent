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

from scripts.qwen_activation_cache import main as build_activation_cache
from scripts.qwen_bridge_ablation import (
    ARM_ORDER,
    PROTOCOL,
    aggregate_bridge_screen,
    main as run_layer_ablation,
)
from scripts.qwen_depth_objective_run import resolve_qwen_cache_dir
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.experiment_stage import update_stage_manifest
from extent.teacher_activation_cache import load_activation_cache


CAMPAIGN_PROTOCOL = "exp054-bridge-initialization-campaign"
TARGET_LAYERS = (18, 0)
SEEDS = (123, 456, 789)
SEQUENCE_LENGTH = 32
RECOVERY_STEPS = 1024
VALIDATION_WINDOWS = 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_arguments(
    *,
    layer: int,
    evaluation_only: bool,
    qwen_cache_dir: str,
    qwen_storage: str,
    dataset_cache_dir: str,
    output_dir: Path,
    compute_dtype: str,
    storage_dtype: str,
    per_device_windows: int,
    artifact_prefix: str = "exp054",
    recovery_steps: int = RECOVERY_STEPS,
    validation_windows: int = VALIDATION_WINDOWS,
) -> tuple[list[str], Path, Path]:
    role = "validation" if evaluation_only else "train"
    artifact_dir = output_dir / f"{artifact_prefix}-layer{layer}-{role}-cache"
    manifest = artifact_dir / f"{artifact_prefix}-layer{layer}-{role}-manifest.json"
    arguments = [
        "--cache-dir", qwen_cache_dir,
        "--dataset-cache-dir", dataset_cache_dir,
        "--target-layer", str(layer),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--microbatch-windows", "4",
        "--data-parallel",
        "--per-device-windows", str(per_device_windows),
        "--compute-dtype", compute_dtype,
        "--storage-dtype", storage_dtype,
        "--output-dir", str(artifact_dir),
        "--result-json", str(manifest),
    ]
    if qwen_storage == "disk":
        arguments.append("--prune-consumed-shards")
    if evaluation_only:
        arguments.extend(
            [
                "--dataset-split", "validation",
                "--evaluation-only",
                "--calibration-windows", "0",
                "--training-windows", "0",
                "--evaluation-windows", str(validation_windows),
                "--token-offset", "0",
            ]
        )
    else:
        arguments.extend(
            [
                "--dataset-split", "train",
                "--calibration-windows", "8",
                "--training-windows", str(recovery_steps),
                "--evaluation-windows", "4",
                "--token-offset", "0",
            ]
        )
    return arguments, manifest, artifact_dir


def _valid_cache(manifest: Path, directory: Path, layer: int) -> bool:
    try:
        payload, _, _ = load_activation_cache(
            manifest, artifact_dir=directory, verify_hashes=True
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return int(payload.get("target_layer", -1)) == layer and bool(payload.get("passed"))


def _valid_layer_result(
    path: Path, layer: int, *, layer_protocol: str = PROTOCOL
) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("protocol") != layer_protocol or int(payload.get("target_layer", -1)) != layer:
        return None
    return payload


def _prune_cache_arrays(directory: Path, output_dir: Path) -> list[str]:
    root = output_dir.resolve()
    candidate = directory.resolve()
    if root not in candidate.parents:
        raise ValueError("refusing to prune outside bridge-campaign output")
    removed = []
    if candidate.exists():
        for path in candidate.glob("*.npy"):
            path.unlink()
            removed.append(str(path))
    return removed


def render_summary(result: dict) -> str:
    lines = [
        f"# {result.get('experiment_name', 'EXP-054 attention bridge initialization screen')}",
        "",
        f"- Status: `{result['status']}`",
        f"- Numerical pass: `{result['passed']}`",
        f"- Complete: `{result['complete']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Screening gate: `{result['aggregate']['screening_gate_passed']}`",
        "",
        "| Layer | Arm | Mean decoder relative L2 | Arm - random | Wins |",
        "|---:|---|---:|---:|---:|",
    ]
    for layer, comparisons in result["aggregate"]["layers"].items():
        for arm in ARM_ORDER[1:]:
            comparison = comparisons.get(arm)
            if comparison is None:
                continue
            lines.append(
                f"| {layer} | {arm} | "
                f"{comparison['decoder_relative_l2_mean']:.8f} | "
                f"{comparison['arm_minus_random_mean']:+.8f} | "
                f"{comparison['wins_over_random']}/{comparison['paired_seeds']} |"
            )
    lines.extend(
        [
            "",
            "Negative `Arm - random` is better. This is a two-layer initialization screen, not full-model recovery.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run a time-bounded resilient bridge TPU campaign."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/extent-bridge-ablation-campaign.json",
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp054-weights"
    )
    parser.add_argument(
        "--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto"
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache"
    )
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--storage-dtype", default="float16")
    parser.add_argument("--per-device-windows", type=int, default=4)
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--artifact-prefix", default="exp054")
    parser.add_argument("--campaign-protocol", default=CAMPAIGN_PROTOCOL)
    parser.add_argument("--layer-protocol", default=PROTOCOL)
    parser.add_argument("--experiment-name", default="EXP-054 bridge initialization")
    parser.add_argument(
        "--summary-filename",
        default="extent-bridge-ablation-campaign-summary.md",
    )
    parser.add_argument("--recovery-steps", type=int, default=RECOVERY_STEPS)
    parser.add_argument("--checkpoints", default="0,256,512,1024")
    parser.add_argument("--bridge-steps", type=int, default=128)
    parser.add_argument("--bridge-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bridge-matrix-loss-weight", type=float, default=0.1)
    parser.add_argument("--bridge-rope-fraction", type=float, default=1.0)
    parser.add_argument("--orientation-steps", type=int, default=128)
    parser.add_argument(
        "--primary-arm", default="BRIDGE-PLUS-ORIENTATION", choices=ARM_ORDER[1:]
    )
    parser.add_argument("--max-wall-hours", type=float, default=3.5)
    parser.add_argument(
        "--telegram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    log_prefix = args.artifact_prefix
    if min(
        args.per_device_windows,
        args.evaluation_batch_windows,
        args.max_wall_hours,
        args.recovery_steps,
        args.bridge_steps,
        args.orientation_steps,
    ) <= 0:
        raise ValueError("batch settings and max-wall-hours must be positive")
    checkpoints = tuple(
        sorted({int(value.strip()) for value in args.checkpoints.split(",")})
    )
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] != args.recovery_steps:
        raise ValueError("checkpoints must include zero and recovery-steps")
    if not args.artifact_prefix or not args.campaign_protocol or not args.layer_protocol:
        raise ValueError("artifact and protocol names must be non-empty")
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("bridge campaign requires exactly eight TPU devices")

    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.max_wall_hours * 3600
    started = _utc_now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    stage_manifest = output_dir / f"{args.artifact_prefix}-campaign-stage-manifest.json"
    qwen_cache_dir, qwen_storage = resolve_qwen_cache_dir(
        args.qwen_cache_dir, args.qwen_cache_storage
    )
    start_notification = _safe_notify(
        args.telegram,
        "Extent TPU campaign\n"
        "status=started\n"
        f"host={socket.gethostname()}\n"
        f"experiment={args.experiment_name}\n"
        f"hard_budget_hours={args.max_wall_hours}",
    )
    update_stage_manifest(
        stage_manifest,
        experiment=args.campaign_protocol,
        stage="campaign",
        status="running",
        details={
            "started_at_utc": started,
            "deadline_monotonic": deadline,
            "qwen_cache_storage": qwen_storage,
        },
    )
    layer_results = {}
    removed_arrays = []
    current_stage = "startup"
    try:
        for layer in TARGET_LAYERS:
            if time.monotonic() >= deadline:
                break
            current_stage = f"layer{layer}"
            layer_result_path = (
                output_dir / f"{args.artifact_prefix}-layer{layer}-result.json"
            )
            completed = (
                _valid_layer_result(
                    layer_result_path,
                    layer,
                    layer_protocol=args.layer_protocol,
                )
                if args.resume
                else None
            )
            if completed is not None and completed.get("complete"):
                print(f"{log_prefix}_layer={layer} RESUME-PASS")
                layer_results[str(layer)] = completed
                continue
            train_args, train_manifest, train_dir = _cache_arguments(
                layer=layer,
                evaluation_only=False,
                qwen_cache_dir=qwen_cache_dir,
                qwen_storage=qwen_storage,
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output_dir,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows,
                artifact_prefix=args.artifact_prefix,
                recovery_steps=args.recovery_steps,
            )
            eval_args, eval_manifest, eval_dir = _cache_arguments(
                layer=layer,
                evaluation_only=True,
                qwen_cache_dir=qwen_cache_dir,
                qwen_storage=qwen_storage,
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output_dir,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows,
                artifact_prefix=args.artifact_prefix,
                recovery_steps=args.recovery_steps,
            )
            if not (args.resume and _valid_cache(train_manifest, train_dir, layer)):
                print(f"{log_prefix}_layer={layer} train_cache=START")
                build_activation_cache(train_args)
            else:
                print(f"{log_prefix}_layer={layer} train_cache=RESUME-PASS")
            if time.monotonic() >= deadline:
                break
            if not (args.resume and _valid_cache(eval_manifest, eval_dir, layer)):
                print(f"{log_prefix}_layer={layer} validation_cache=START")
                build_activation_cache(eval_args)
            else:
                print(f"{log_prefix}_layer={layer} validation_cache=RESUME-PASS")
            layer_result = run_layer_ablation(
                [
                    "--activation-cache-manifest", str(train_manifest),
                    "--activation-cache-dir", str(train_dir),
                    "--evaluation-cache-manifest", str(eval_manifest),
                    "--evaluation-cache-dir", str(eval_dir),
                    "--qwen-cache-dir", qwen_cache_dir,
                    "--total-steps", str(args.recovery_steps),
                    "--checkpoints", args.checkpoints,
                    "--bridge-steps", str(args.bridge_steps),
                    "--orientation-steps", str(args.orientation_steps),
                    "--bridge-learning-rate", str(args.bridge_learning_rate),
                    "--bridge-matrix-loss-weight", str(args.bridge_matrix_loss_weight),
                    "--bridge-rope-fraction", str(args.bridge_rope_fraction),
                    "--experiment-protocol", args.layer_protocol,
                    "--seeds", ",".join(map(str, SEEDS)),
                    "--evaluation-batch-windows", str(args.evaluation_batch_windows),
                    "--compute-dtype", args.compute_dtype,
                    "--result-json", str(layer_result_path),
                    "--output-dir", str(output_dir),
                ],
                deadline_monotonic=deadline,
            )
            layer_results[str(layer)] = layer_result
            update_stage_manifest(
                stage_manifest,
                experiment=args.campaign_protocol,
                stage=current_stage,
                status="completed",
                details={
                    "result_json": str(layer_result_path.resolve()),
                    "complete": layer_result["complete"],
                },
            )
            removed_arrays.extend(_prune_cache_arrays(train_dir, output_dir))
            removed_arrays.extend(_prune_cache_arrays(eval_dir, output_dir))
            jax.clear_caches()
            gc.collect()
            if not layer_result["complete"]:
                break

        complete = set(layer_results) == {str(layer) for layer in TARGET_LAYERS} and all(
            result.get("complete") for result in layer_results.values()
        )
        numerical_pass = bool(layer_results) and all(
            result.get("passed") for result in layer_results.values()
        )
        aggregate = aggregate_bridge_screen(
            layer_results, primary_arm=args.primary_arm
        )
        completed_at = _utc_now()
        result = {
            "protocol": args.campaign_protocol,
            "experiment_name": args.experiment_name,
            "status": "completed" if complete else "deadline_partial",
            "started_at_utc": started,
            "completed_at_utc": completed_at,
            "duration_hours": (time.monotonic() - started_monotonic) / 3600,
            "max_wall_hours": args.max_wall_hours,
            "target_layers": list(TARGET_LAYERS),
            "seeds": list(SEEDS),
            "arm_order": list(ARM_ORDER),
            "recovery_steps": args.recovery_steps,
            "recovery_checkpoints": list(checkpoints),
            "bridge_recipe": {
                "steps": args.bridge_steps,
                "learning_rate": args.bridge_learning_rate,
                "matrix_loss_weight": args.bridge_matrix_loss_weight,
                "rope_fraction": args.bridge_rope_fraction,
            },
            "orientation_steps": args.orientation_steps,
            "primary_arm": args.primary_arm,
            "qwen_cache_storage": qwen_storage,
            "layer_results": layer_results,
            "aggregate": aggregate,
            "removed_regenerable_arrays": removed_arrays,
            "start_notification": start_notification,
            "complete": complete,
            "passed": numerical_pass,
            "notes": [
                "The campaign stops cleanly at the wall-clock deadline and retains every completed arm JSON.",
                "Layer 18 runs first because it is the representative interior layer; layer 0 follows if time remains.",
                "Large activation arrays are deleted only after their layer result has been written.",
                "screening_gate_passed is exploratory and does not replace a later locked confirmation.",
            ],
        }
        mirror = _write_json_with_output_mirror(result_path, result, str(output_dir))
        summary_path = output_dir / args.summary_filename
        summary_path.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(
            stage_manifest,
            experiment=args.campaign_protocol,
            stage="final",
            status="completed",
            details={
                "result_json": str(result_path.resolve()),
                "summary_markdown": str(summary_path.resolve()),
                "complete": complete,
                "screening_gate": aggregate["screening_gate_passed"],
            },
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            f"status={'completed' if complete else 'deadline_partial'}\n"
            f"experiment={args.experiment_name}\n"
            f"layers={','.join(sorted(layer_results))}\n"
            f"numerical_pass={numerical_pass}\n"
            f"screening_gate={aggregate['screening_gate_passed']}",
        )
        print(
            f"{log_prefix.upper()}-COMPLETE "
            f"complete={complete} numerical_pass={numerical_pass} "
            f"screening_gate={aggregate['screening_gate_passed']}"
        )
        print(f"result_json={result_path.resolve()}")
        print(f"compact_summary={summary_path.resolve()}")
        if mirror:
            print(f"output_json={mirror.resolve()}")
        return result
    except BaseException as exc:
        failure = {
            "protocol": args.campaign_protocol,
            "status": "failed",
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
            "completed_layer_results": layer_results,
        }
        failure_path = output_dir / f"{args.artifact_prefix}-campaign-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output_dir))
        update_stage_manifest(
            stage_manifest,
            experiment=args.campaign_protocol,
            stage=current_stage,
            status="failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=failed\n"
            f"experiment={args.experiment_name}\n"
            f"stage={current_stage}\n"
            f"error={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
