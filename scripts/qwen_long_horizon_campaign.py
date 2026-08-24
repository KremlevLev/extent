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
import numpy as np

from scripts.qwen_depth_objective_run import main as run_depth_objective
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.experiment_stage import update_stage_manifest
from extent.notifications import TelegramNotifierError, send_telegram_message
from extent.qwen_source import QWEN3_14B


TARGET_LAYERS = (0, 18)
SEEDS = (123, 456, 789)
ARM_BRANCHES = {
    "mixer_only": "MIXER-ONLY",
    "joint": "JOINT",
    "contribution": "CONTRIBUTION",
}
BUDGETS = {
    "short": {
        "experiment": "exp048-short",
        "steps": 1024,
        "checkpoints": "0,512,1024",
        "minimum_free_gib": 5,
    },
    "long": {
        "experiment": "exp048-long",
        "steps": 4096,
        "checkpoints": "0,1024,2048,4096",
        "minimum_free_gib": 12,
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


def _read_completed_depth_result(path: Path, budget: str) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = BUDGETS[budget]
    if (
        not result.get("passed")
        or result.get("protocol")
        != f"{expected['experiment']}-layer0-layer18"
        or int(result.get("total_steps_per_arm", -1)) != expected["steps"]
    ):
        return None
    return result


def _bootstrap_scale_delta(
    short_window_nll: dict[str, list[float]],
    long_window_nll: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict:
    ordered_seeds = tuple(sorted(short_window_nll, key=int))
    if ordered_seeds != tuple(sorted(long_window_nll, key=int)):
        raise ValueError("short and long scale comparisons require identical seeds")
    deltas = []
    for seed_key in ordered_seeds:
        short = np.asarray(short_window_nll[seed_key], dtype=np.float64)
        long = np.asarray(long_window_nll[seed_key], dtype=np.float64)
        if short.shape != long.shape or short.ndim != 1 or short.size < 2:
            raise ValueError("paired scale windows must have equal non-trivial shape")
        deltas.append(long - short)
    paired = np.stack(deltas)
    if not np.all(np.isfinite(paired)):
        raise ValueError("paired scale deltas contain non-finite values")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        window_indices = rng.integers(0, paired.shape[1], paired.shape[1])
        draws[index] = float(np.mean(paired[:, window_indices]))
    return {
        "samples": samples,
        "seed": seed,
        "mean_long_minus_short_nll": float(np.mean(paired)),
        "long_minus_short_95ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def aggregate_long_horizon(
    short_result: dict,
    long_result: dict,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    layers = []
    all_finite = bool(short_result.get("passed") and long_result.get("passed"))
    for layer in TARGET_LAYERS:
        short_layer = short_result["layer_results"][str(layer)]
        long_layer = long_result["layer_results"][str(layer)]
        arm_results = {}
        for arm_index, (arm, branch) in enumerate(ARM_BRANCHES.items()):
            per_seed = []
            short_windows = {}
            long_windows = {}
            for seed in SEEDS:
                short_metrics = short_layer["lm_metrics"][
                    f"SEED-{seed}-{branch}-STEP1024"
                ]
                long_metrics = long_layer["lm_metrics"][
                    f"SEED-{seed}-{branch}-STEP4096"
                ]
                short_nll = float(short_metrics["mean_nll"])
                long_nll = float(long_metrics["mean_nll"])
                per_seed.append(
                    {
                        "seed": seed,
                        "short_nll": short_nll,
                        "long_nll": long_nll,
                        "long_minus_short_nll": long_nll - short_nll,
                        "long_beats_short": long_nll < short_nll,
                    }
                )
                short_windows[str(seed)] = short_metrics["window_mean_nll"]
                long_windows[str(seed)] = long_metrics["window_mean_nll"]
                all_finite = all_finite and bool(
                    short_metrics["finite"] and long_metrics["finite"]
                )
            uncertainty = _bootstrap_scale_delta(
                short_windows,
                long_windows,
                samples=bootstrap_samples,
                seed=bootstrap_seed + layer * 10 + arm_index,
            )
            arm_results[arm] = {
                "per_seed": per_seed,
                "long_wins": sum(item["long_beats_short"] for item in per_seed),
                **uncertainty,
            }
        primary = arm_results["joint"]
        layer_gate = bool(
            primary["mean_long_minus_short_nll"] < 0
            and primary["long_wins"] >= 2
            and primary["long_minus_short_95ci"][1] < 0
        )
        layers.append(
            {
                "target_layer": layer,
                "arms": arm_results,
                "primary_joint_gate_passed": layer_gate,
            }
        )
    scientific_gate_passed = bool(
        all_finite
        and len(layers) == len(TARGET_LAYERS)
        and all(layer["primary_joint_gate_passed"] for layer in layers)
    )
    return {
        "layers": layers,
        "primary_arm": "joint",
        "required_long_wins_per_layer": 2,
        "required_mean_delta_sign": "negative",
        "required_bootstrap_upper_sign": "negative",
        "all_finite": all_finite,
        "scientific_gate_passed": scientific_gate_passed,
    }


def _prune_completed_payloads(output_dir: Path, experiment: str) -> list[str]:
    removed = []
    resolved_root = output_dir.resolve()
    for layer in TARGET_LAYERS:
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


def _require_disk_headroom(output_dir: Path, minimum_free_gib: int) -> None:
    free = shutil.disk_usage(output_dir).free
    required = minimum_free_gib * 1024**3
    print(f"disk_free_gib={free / 1024**3:.3f} required_gib={minimum_free_gib}")
    if free < required:
        raise RuntimeError(
            f"EXP-048 requires at least {minimum_free_gib} GiB free before stage"
        )


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run the resilient EXP-048 long-horizon scaling campaign."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/extent-long-horizon-campaign.json",
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp048-weights"
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument(
        "--telegram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-048 requires exactly eight TPU devices")
    if min(args.per_device_windows, args.bootstrap_samples) < 1:
        raise ValueError("window and bootstrap settings must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = output_dir / "exp048-campaign-stage-manifest.json"
    started = _utc_now()
    start_notification = _safe_notify(
        args.telegram,
        "Extent TPU campaign\n"
        "status=started\n"
        f"host={socket.gethostname()}\n"
        f"devices={len(devices)}\n"
        "experiment=EXP-048 long horizon (1024 vs 4096 steps)",
    )
    update_stage_manifest(
        stage_manifest,
        experiment="exp048-long-horizon",
        stage="campaign",
        status="running",
        details={"started_at_utc": started},
    )

    budget_results = {}
    removed_payloads = []
    current_stage = "startup"
    try:
        for budget, protocol in BUDGETS.items():
            current_stage = f"exp048-{budget}"
            result_path = output_dir / f"exp048-{budget}-depth-result.json"
            completed = (
                _read_completed_depth_result(result_path, budget)
                if args.resume
                else None
            )
            if completed is None:
                _require_disk_headroom(output_dir, protocol["minimum_free_gib"])
                depth_args = [
                    "--experiment", protocol["experiment"],
                    "--target-layers", "0,18",
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
                depth_args.append("--resume" if args.resume else "--no-resume")
                completed = run_depth_objective(depth_args)
            else:
                print(f"campaign_stage=exp048-{budget} RESUME-PASS")
            budget_results[budget] = completed
            removed_payloads.extend(
                _prune_completed_payloads(output_dir, protocol["experiment"])
            )
            update_stage_manifest(
                stage_manifest,
                experiment="exp048-long-horizon",
                stage=f"budget-{budget}",
                status="completed",
                details={"result_json": str(result_path.resolve())},
            )
            jax.clear_caches()
            gc.collect()

        aggregate = aggregate_long_horizon(
            budget_results["short"],
            budget_results["long"],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        result = {
            "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
            "method": "paired_multiseed_long_horizon_transplant_scaling",
            "protocol": "exp048-short1024-vs-long4096-layer0-layer18",
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "target_layers": list(TARGET_LAYERS),
            "seeds": list(SEEDS),
            "short_unique_tokens_per_arm": 1024 * 32,
            "long_unique_tokens_per_arm": 4096 * 32,
            "optimizer_visible_tokens_total": (1024 + 4096)
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
                "Both budgets are fresh, paired three-seed runs in the same TPU campaign.",
                "Training sets are nested WikiText prefixes and every arm makes exactly one pass over its unique windows.",
                "The primary arm is JOINT, selected before EXP-048 from EXP-045; other objective crossovers are secondary.",
                "The primary gate requires long JOINT to beat short JOINT at both layers by mean, at least two seeds, and paired-bootstrap upper bound.",
                "Different total-step Lion schedules are part of the frozen budget treatment, so the estimand is recovery budget rather than tokens alone.",
            ],
        }
        mirror = _write_json_with_output_mirror(
            Path(args.result_json), result, str(output_dir)
        )
        update_stage_manifest(
            stage_manifest,
            experiment="exp048-long-horizon",
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
            "experiment=EXP-048\n"
            f"numerical_pass={result['passed']}\n"
            f"scientific_gate={result['scientific_gate_passed']}",
        )
        print(json.dumps(result, indent=2))
        print(f"result_json={Path(args.result_json).resolve()}")
        if mirror:
            print(f"output_json={mirror.resolve()}")
        return result
    except BaseException as exc:
        failure = {
            "experiment": "exp048-long-horizon",
            "status": "failed",
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
        }
        failure_path = output_dir / "extent-long-horizon-campaign-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output_dir))
        update_stage_manifest(
            stage_manifest,
            experiment="exp048-long-horizon",
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
