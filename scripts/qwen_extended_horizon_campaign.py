from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import socket
import traceback

import jax

from scripts.qwen_depth_objective_run import main as run_depth_objective
from scripts.qwen_long_horizon_campaign import (
    ARM_BRANCHES,
    SEEDS,
    TARGET_LAYERS,
    aggregate_long_horizon,
)
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.artifact_summary import write_compact_summary
from extent.experiment_stage import update_stage_manifest
from extent.notifications import TelegramNotifierError, send_telegram_message
from extent.qwen_source import QWEN3_14B


BUDGETS = {
    "short": {
        "experiment": "exp049-short",
        "steps": 2048,
        "checkpoints": "0,1024,2048",
        "minimum_free_gib": 6,
    },
    "long": {
        "experiment": "exp049-long",
        "steps": 8192,
        "checkpoints": "0,2048,4096,8192",
        "minimum_free_gib": 11,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_completed_layer(
    path: Path,
    *,
    experiment: str,
    layer: int,
    steps: int,
) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not result.get("passed")
        or result.get("protocol") != f"{experiment}-layer{layer}"
        or int(result.get("total_steps_per_arm", -1)) != steps
        or list(result.get("target_layers", [])) != [layer]
    ):
        return None
    return result


def _require_disk_headroom(output_dir: Path, minimum_free_gib: int) -> None:
    free = shutil.disk_usage(output_dir).free
    print(f"disk_free_gib={free / 1024**3:.3f} required_gib={minimum_free_gib}")
    if free < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"EXP-049 requires at least {minimum_free_gib} GiB free before layer"
        )


def _prune_completed_layer(
    output_dir: Path,
    *,
    experiment: str,
    layer: int,
) -> list[str]:
    removed = []
    resolved_root = output_dir.resolve()
    for role in ("train", "validation"):
        cache_dir = (
            output_dir / f"{experiment}-layer{layer}-{role}-cache"
        ).resolve()
        if resolved_root not in cache_dir.parents:
            raise ValueError("refusing to prune outside campaign output")
        if cache_dir.exists():
            for path in cache_dir.glob("*.npy"):
                path.unlink()
                removed.append(str(path))
    endpoint = (
        output_dir
        / f"{experiment}-layer{layer}-endpoints"
        / "endpoint_params.msgpack"
    ).resolve()
    if resolved_root not in endpoint.parents:
        raise ValueError("refusing to prune endpoint outside campaign output")
    if endpoint.exists():
        endpoint.unlink()
        removed.append(str(endpoint))
    return removed


def _combine_budget_layers(
    budget: str,
    layer_results: dict[str, dict],
) -> dict:
    protocol = BUDGETS[budget]
    layers = {
        layer: result["layer_results"][layer]
        for layer, result in layer_results.items()
    }
    passed = all(result.get("passed") for result in layer_results.values())
    return {
        "protocol": f"{protocol['experiment']}-layer0-layer18",
        "total_steps_per_arm": protocol["steps"],
        "target_layers": list(TARGET_LAYERS),
        "layer_results": layers,
        "scientific_gate_passed": bool(
            passed
            and all(
                result.get("scientific_gate_passed")
                for result in layer_results.values()
            )
        ),
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run EXP-049 at 2,048 and 8,192 one-pass recovery steps."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/extent-extended-horizon-campaign.json",
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp049-weights"
    )
    parser.add_argument(
        "--qwen-cache-storage", choices=("auto", "disk", "ram"), default="auto"
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache"
    )
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--storage-dtype", default="float16")
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument(
        "--telegram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-049 requires exactly eight TPU devices")
    if min(args.per_device_windows, args.bootstrap_samples) < 1:
        raise ValueError("window and bootstrap settings must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = output_dir / "exp049-campaign-stage-manifest.json"
    started = _utc_now()
    start_notification = _safe_notify(
        args.telegram,
        "Extent TPU campaign\n"
        "status=started\n"
        f"host={socket.gethostname()}\n"
        f"devices={len(devices)}\n"
        "experiment=EXP-049 extended horizon (2048 vs 8192 steps)",
    )
    update_stage_manifest(
        stage_manifest,
        experiment="exp049-extended-horizon",
        stage="campaign",
        status="running",
        details={"started_at_utc": started},
    )

    budget_results = {}
    removed_payloads = []
    current_stage = "startup"
    try:
        for budget, protocol in BUDGETS.items():
            per_layer = {}
            for layer in sorted(TARGET_LAYERS, reverse=True):
                current_stage = f"exp049-{budget}-layer{layer}"
                result_path = (
                    output_dir / f"exp049-{budget}-layer{layer}-depth-result.json"
                )
                completed = (
                    _read_completed_layer(
                        result_path,
                        experiment=protocol["experiment"],
                        layer=layer,
                        steps=protocol["steps"],
                    )
                    if args.resume
                    else None
                )
                if completed is None:
                    _require_disk_headroom(
                        output_dir, protocol["minimum_free_gib"]
                    )
                    depth_args = [
                        "--experiment", protocol["experiment"],
                        "--target-layers", str(layer),
                        "--total-steps", str(protocol["steps"]),
                        "--training-checkpoints", protocol["checkpoints"],
                        "--qwen-cache-dir", args.qwen_cache_dir,
                        "--qwen-cache-storage", args.qwen_cache_storage,
                        "--dataset-cache-dir", args.dataset_cache_dir,
                        "--output-dir", str(output_dir),
                        "--result-json", str(result_path),
                        "--compute-dtype", args.compute_dtype,
                        "--storage-dtype", args.storage_dtype,
                        "--per-device-windows", str(args.per_device_windows),
                        "--bootstrap-samples", str(args.bootstrap_samples),
                        "--bootstrap-seed", str(args.bootstrap_seed),
                    ]
                    depth_args.append(
                        "--resume" if args.resume else "--no-resume"
                    )
                    completed = run_depth_objective(depth_args)
                else:
                    print(f"campaign_stage={current_stage} RESUME-PASS")
                per_layer[str(layer)] = completed
                removed_payloads.extend(
                    _prune_completed_layer(
                        output_dir,
                        experiment=protocol["experiment"],
                        layer=layer,
                    )
                )
                update_stage_manifest(
                    stage_manifest,
                    experiment="exp049-extended-horizon",
                    stage=f"budget-{budget}-layer{layer}",
                    status="completed",
                    details={"result_json": str(result_path.resolve())},
                )
                jax.clear_caches()
                gc.collect()
            budget_results[budget] = _combine_budget_layers(budget, per_layer)

        aggregate = aggregate_long_horizon(
            budget_results["short"],
            budget_results["long"],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            short_step=BUDGETS["short"]["steps"],
            long_step=BUDGETS["long"]["steps"],
        )
        result = {
            "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
            "method": "paired_multiseed_extended_horizon_transplant_scaling",
            "protocol": "exp049-short2048-vs-long8192-layer0-layer18",
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "target_layers": list(TARGET_LAYERS),
            "seeds": list(SEEDS),
            "short_unique_tokens_per_arm": 2048 * 32,
            "long_unique_tokens_per_arm": 8192 * 32,
            "optimizer_visible_tokens_total": (2048 + 8192)
            * 32
            * len(SEEDS)
            * len(ARM_BRANCHES)
            * len(TARGET_LAYERS),
            "budget_results": budget_results,
            "aggregate": aggregate,
            "removed_regenerable_payloads": removed_payloads,
            "start_notification": start_notification,
            "scientific_gate_passed": aggregate["scientific_gate_passed"],
            "passed": aggregate["all_finite"],
            "notes": [
                "The 2,048-step and 8,192-step budgets are fresh paired three-seed runs in one TPU allocation.",
                "Each layer is cached, trained, evaluated, and pruned before the next layer to bound Kaggle disk usage.",
                "JOINT remains the pre-registered primary arm; objective crossover is secondary.",
                "The budget treatment includes its total-step Lion schedule and nested one-pass data prefix.",
            ],
        }
        mirror = _write_json_with_output_mirror(
            Path(args.result_json), result, str(output_dir)
        )
        summary_artifacts = write_compact_summary(
            result,
            artifact_path=args.result_json,
            output_dir=output_dir,
        )
        update_stage_manifest(
            stage_manifest,
            experiment="exp049-extended-horizon",
            stage="final",
            status="completed",
            details={
                "result_json": str(Path(args.result_json).resolve()),
                "summary_json": summary_artifacts["summary_json"],
                "summary_markdown": summary_artifacts["summary_markdown"],
                "scientific_gate_passed": result["scientific_gate_passed"],
            },
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=completed\n"
            "experiment=EXP-049\n"
            f"numerical_pass={result['passed']}\n"
            f"scientific_gate={result['scientific_gate_passed']}",
        )
        print(json.dumps(result, indent=2))
        print(f"result_json={Path(args.result_json).resolve()}")
        print(f"compact_summary={json.dumps(summary_artifacts, sort_keys=True)}")
        if mirror:
            print(f"output_json={mirror.resolve()}")
        return result
    except BaseException as exc:
        failure = {
            "experiment": "exp049-extended-horizon",
            "status": "failed",
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
        }
        failure_path = output_dir / "extent-extended-horizon-campaign-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output_dir))
        update_stage_manifest(
            stage_manifest,
            experiment="exp049-extended-horizon",
            stage=current_stage,
            status="failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=failed\n"
            f"stage={current_stage}\n"
            f"error={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
