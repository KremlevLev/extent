from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import traceback

import jax

from scripts.qwen_activation_cache import main as build_activation_cache
from scripts.qwen_depth_objective_run import (
    main as run_depth_objective,
    resolve_qwen_cache_dir,
)
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_streamed_multiseed_end_to_end import main as run_context_cell
from extent.experiment_stage import update_stage_manifest
from extent.notifications import TelegramNotifierError, send_telegram_message
from extent.qwen_source import QWEN3_14B
from extent.teacher_activation_cache import load_activation_cache


DEPTH_EXPERIMENTS = {
    "exp045": (0, 18),
    "exp046": (9, 29),
}
CONTEXT_LAYERS = (0, 18)
CONTEXT_LENGTHS = (64, 128, 256)
VALIDATION_TOKENS = 8192


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_valid_result(path: Path, protocol_prefix: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not payload.get("passed") or not str(payload.get("protocol", "")).startswith(
        protocol_prefix
    ):
        return None
    return payload


def _safe_notify(enabled: bool, text: str) -> dict:
    if not enabled:
        return {"enabled": False, "sent": False, "reason": "disabled"}
    try:
        message_id = send_telegram_message(text)
        print(f"telegram_notification=PASS message_id={message_id}")
        return {"enabled": True, "sent": True, "message_id": message_id}
    except TelegramNotifierError as exc:
        print(f"telegram_notification=FAILED reason={exc}")
        return {"enabled": True, "sent": False, "reason": str(exc)}


def _context_cache_arguments(
    *,
    layer: int,
    sequence_length: int,
    qwen_cache_dir: str,
    dataset_cache_dir: str,
    output_dir: Path,
    compute_dtype: str,
    storage_dtype: str,
    per_device_windows: int,
) -> tuple[list[str], Path, Path]:
    if VALIDATION_TOKENS % sequence_length:
        raise ValueError("context length must divide the frozen validation range")
    windows = VALIDATION_TOKENS // sequence_length
    artifact_dir = output_dir / f"exp047-layer{layer}-seq{sequence_length}-cache"
    manifest = artifact_dir / f"exp047-layer{layer}-seq{sequence_length}-manifest.json"
    arguments = [
        "--cache-dir", qwen_cache_dir,
        "--dataset-cache-dir", dataset_cache_dir,
        "--target-layer", str(layer),
        "--sequence-length", str(sequence_length),
        "--dataset-split", "validation",
        "--evaluation-only",
        "--calibration-windows", "0",
        "--training-windows", "0",
        "--evaluation-windows", str(windows),
        "--token-offset", "0",
        "--microbatch-windows", "1",
        "--data-parallel",
        "--per-device-windows", str(per_device_windows),
        "--compute-dtype", compute_dtype,
        "--storage-dtype", storage_dtype,
        "--prune-consumed-shards",
        "--output-dir", str(artifact_dir),
        "--result-json", str(manifest),
    ]
    return arguments, manifest, artifact_dir


def aggregate_context_transfer(cell_results: dict[str, dict]) -> dict:
    cells = []
    all_finite = True
    for key, result in sorted(cell_results.items()):
        aggregate = result["aggregate"]
        delta_mixer = float(aggregate["mean_contribution_minus_mixer_nll"])
        delta_joint = float(aggregate["mean_contribution_minus_joint_nll"])
        regret = max(delta_mixer, delta_joint)
        best = delta_mixer <= 0 and delta_joint <= 0
        cells.append(
            {
                "cell": key,
                "target_layer": int(result["target_layer"]),
                "sequence_length": int(result["sequence_length"]),
                "contribution_minus_mixer_nll": delta_mixer,
                "contribution_minus_joint_nll": delta_joint,
                "contribution_regret_to_best_control": regret,
                "contribution_is_best": best,
                "finite": bool(result["passed"]),
            }
        )
        all_finite = all_finite and bool(result["passed"])
    best_count = sum(cell["contribution_is_best"] for cell in cells)
    required_best = 5
    maximum_regret = max(
        (cell["contribution_regret_to_best_control"] for cell in cells),
        default=float("inf"),
    )
    scientific_gate_passed = bool(
        all_finite
        and len(cells) == len(CONTEXT_LAYERS) * len(CONTEXT_LENGTHS)
        and best_count >= required_best
        and maximum_regret <= 0.02
    )
    return {
        "cells": cells,
        "contribution_best_cells": best_count,
        "required_contribution_best_cells": required_best,
        "maximum_contribution_regret_nll": maximum_regret,
        "maximum_allowed_contribution_regret_nll": 0.02,
        "all_finite": all_finite,
        "scientific_gate_passed": scientific_gate_passed,
    }


def _prune_completed_depth_arrays(output_dir: Path) -> list[str]:
    """Delete only regenerable NPY caches after verified depth results exist."""
    removed = []
    resolved_root = output_dir.resolve()
    for experiment, layers in DEPTH_EXPERIMENTS.items():
        result_path = output_dir / f"{experiment}-depth-objective.json"
        if _read_valid_result(result_path, experiment) is None:
            continue
        for layer in layers:
            for role in ("train", "validation"):
                cache_dir = (
                    output_dir / f"{experiment}-layer{layer}-{role}-cache"
                ).resolve()
                if resolved_root not in cache_dir.parents:
                    raise ValueError("refusing to prune outside campaign output")
                if not cache_dir.exists():
                    continue
                for path in cache_dir.glob("*.npy"):
                    path.unlink()
                    removed.append(str(path))
    return removed


def _prune_context_arrays(cache_dir: Path, output_dir: Path) -> list[str]:
    resolved_cache = cache_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output not in resolved_cache.parents:
        raise ValueError("refusing to prune context cache outside campaign output")
    removed = []
    if resolved_cache.exists():
        for path in resolved_cache.glob("*.npy"):
            path.unlink()
            removed.append(str(path))
    return removed


def _prune_completed_endpoint_payloads(output_dir: Path) -> list[str]:
    removed = []
    resolved_root = output_dir.resolve()
    for experiment, layers in DEPTH_EXPERIMENTS.items():
        if _read_valid_result(
            output_dir / f"{experiment}-depth-objective.json", experiment
        ) is None:
            continue
        for layer in layers:
            path = (
                output_dir
                / f"{experiment}-layer{layer}-endpoints"
                / "endpoint_params.msgpack"
            ).resolve()
            if resolved_root not in path.parents:
                raise ValueError("refusing to prune endpoint outside campaign output")
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run the resilient EXP-045/046/047 TPU research campaign."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/extent-tpu-campaign.json",
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-campaign-weights"
    )
    parser.add_argument(
        "--qwen-cache-storage",
        choices=("auto", "disk", "ram"),
        default="auto",
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache"
    )
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--storage-dtype", default="float16")
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument(
        "--telegram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("the cloud campaign requires exactly eight TPU devices")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_stage = output_dir / "extent-campaign-stage-manifest.json"
    qwen_cache_dir, qwen_storage = resolve_qwen_cache_dir(
        args.qwen_cache_dir, args.qwen_cache_storage
    )
    started = _utc_now()
    start_notification = _safe_notify(
        args.telegram,
        "Extent TPU campaign\n"
        "status=started\n"
        f"host={socket.gethostname()}\n"
        f"devices={len(devices)}\n"
        "experiments=EXP-045, EXP-046, EXP-047",
    )
    update_stage_manifest(
        campaign_stage,
        experiment="extent-tpu-campaign",
        stage="campaign",
        status="running",
        details={"started_at_utc": started, "qwen_cache_storage": qwen_storage},
    )

    depth_results: dict[str, dict] = {}
    context_results: dict[str, dict] = {}
    current_stage = "startup"
    try:
        for experiment, layers in DEPTH_EXPERIMENTS.items():
            current_stage = experiment
            result_path = output_dir / f"{experiment}-depth-objective.json"
            completed = (
                _read_valid_result(result_path, experiment)
                if args.resume
                else None
            )
            if completed is not None:
                print(f"campaign_stage={experiment} RESUME-PASS")
                depth_results[experiment] = completed
                continue
            depth_args = [
                "--experiment", experiment,
                "--target-layers", ",".join(map(str, layers)),
                "--qwen-cache-dir", args.qwen_cache_dir,
                "--qwen-cache-storage", args.qwen_cache_storage,
                "--dataset-cache-dir", args.dataset_cache_dir,
                "--output-dir", str(output_dir),
                "--result-json", str(result_path),
                "--compute-dtype", args.compute_dtype,
                "--storage-dtype", args.storage_dtype,
                "--per-device-windows", str(args.per_device_windows),
            ]
            if args.resume:
                depth_args.append("--resume")
            else:
                depth_args.append("--no-resume")
            depth_results[experiment] = run_depth_objective(depth_args)
            jax.clear_caches()
            gc.collect()

        removed_depth_arrays = _prune_completed_depth_arrays(output_dir)
        print(f"pruned_completed_depth_arrays={len(removed_depth_arrays)}")

        for sequence_length in CONTEXT_LENGTHS:
            for layer in CONTEXT_LAYERS:
                key = f"layer{layer}-seq{sequence_length}"
                current_stage = f"exp047-{key}"
                result_path = output_dir / f"exp047-{key}-result.json"
                completed = (
                    _read_valid_result(result_path, "exp047")
                    if args.resume
                    else None
                )
                if completed is not None:
                    print(f"campaign_stage={current_stage} RESUME-PASS")
                    context_results[key] = completed
                    _prune_context_arrays(
                        output_dir / f"exp047-layer{layer}-seq{sequence_length}-cache",
                        output_dir,
                    )
                    continue
                cache_args, context_manifest, context_dir = (
                    _context_cache_arguments(
                        layer=layer,
                        sequence_length=sequence_length,
                        qwen_cache_dir=qwen_cache_dir,
                        dataset_cache_dir=args.dataset_cache_dir,
                        output_dir=output_dir,
                        compute_dtype=args.compute_dtype,
                        storage_dtype=args.storage_dtype,
                        per_device_windows=args.per_device_windows,
                    )
                )
                cache_reused = False
                if args.resume and context_manifest.exists():
                    try:
                        load_activation_cache(
                            context_manifest,
                            artifact_dir=context_dir,
                            verify_hashes=True,
                        )
                        cache_reused = True
                    except (FileNotFoundError, ValueError):
                        cache_reused = False
                if not cache_reused:
                    build_activation_cache(cache_args)
                training_manifest = (
                    output_dir
                    / f"exp045-layer{layer}-train-cache"
                    / f"exp045-layer{layer}-train-manifest.json"
                )
                base_evaluation_manifest = (
                    output_dir
                    / f"exp045-layer{layer}-validation-cache"
                    / f"exp045-layer{layer}-validation-manifest.json"
                )
                training_result = (
                    output_dir
                    / f"exp045-layer{layer}"
                    / f"exp045-layer{layer}-training-confirmation.json"
                )
                endpoint_dir = output_dir / f"exp045-layer{layer}-endpoints"
                context_args = [
                    "--activation-cache-manifest", str(training_manifest),
                    "--evaluation-cache-manifest", str(context_manifest),
                    "--evaluation-cache-dir", str(context_dir),
                    "--qwen-cache-dir", qwen_cache_dir,
                    "--target-layer", str(layer),
                    "--protocol", "exp047-context-transfer",
                    "--training-result-json", str(training_result),
                    "--endpoint-checkpoint-dir", str(endpoint_dir),
                    "--endpoint-source-evaluation-manifest", str(
                        base_evaluation_manifest
                    ),
                    "--endpoint-source-experiment", f"exp045-layer{layer}",
                    "--compute-dtype", args.compute_dtype,
                    "--data-parallel",
                    "--prune-consumed-shards",
                    "--per-device-windows", str(args.per_device_windows),
                    "--result-json", str(result_path),
                    "--output-dir", str(output_dir / f"exp047-{key}"),
                    "--stage-manifest", str(
                        output_dir / "exp047-stage-manifest.json"
                    ),
                ]
                context_results[key] = run_context_cell(context_args)
                _prune_context_arrays(context_dir, output_dir)
                jax.clear_caches()
                gc.collect()

        context_aggregate = aggregate_context_transfer(context_results)
        all_finite = bool(
            all(result["passed"] for result in depth_results.values())
            and context_aggregate["all_finite"]
        )
        removed_endpoint_payloads = _prune_completed_endpoint_payloads(
            output_dir
        )
        result = {
            "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
            "method": "resilient_multi_experiment_TPU_campaign",
            "protocol": "exp045-exp046-exp047",
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "qwen_cache_storage": qwen_storage,
            "depth_results": depth_results,
            "context_results": context_results,
            "context_aggregate": context_aggregate,
            "removed_depth_cache_arrays": removed_depth_arrays,
            "removed_endpoint_payloads": removed_endpoint_payloads,
            "start_notification": start_notification,
            "scientific_gate_passed": bool(
                all(result["scientific_gate_passed"] for result in depth_results.values())
                and context_aggregate["scientific_gate_passed"]
            ),
            "passed": all_finite,
        }
        mirror = _write_json_with_output_mirror(
            Path(args.result_json), result, str(output_dir)
        )
        update_stage_manifest(
            campaign_stage,
            experiment="extent-tpu-campaign",
            stage="final",
            status="completed",
            details={
                "result_json": str(Path(args.result_json).resolve()),
                "scientific_gate_passed": result["scientific_gate_passed"],
            },
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=completed\n"
            f"numerical_pass={result['passed']}\n"
            f"scientific_gate_passed={result['scientific_gate_passed']}\n"
            f"result={Path(args.result_json).name}",
        )
        print(f"campaign_result_json={Path(args.result_json).resolve()}")
        if mirror:
            print(f"campaign_output_json={mirror.resolve()}")
        return result
    except BaseException as exc:
        failure = {
            "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
            "protocol": "exp045-exp046-exp047",
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc(limit=30),
            "passed": False,
        }
        failure_path = output_dir / "extent-tpu-campaign-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output_dir))
        update_stage_manifest(
            campaign_stage,
            experiment="extent-tpu-campaign",
            stage=current_stage,
            status="failed",
            details={
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "failure_json": str(failure_path.resolve()),
            },
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=failed\n"
            f"stage={current_stage}\n"
            f"error={type(exc).__name__}: {str(exc)[:500]}",
        )
        raise


if __name__ == "__main__":
    main()
