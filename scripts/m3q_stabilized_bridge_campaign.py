"""EXP-066: scale-stabilized exact-dual bridge screen on Qwen3-1.7B."""

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

from extent.experiment_stage import update_stage_manifest
from extent.qwen_source import QWEN3_1_7B_BASE
from scripts.m3q_complex_bridge_campaign import _endpoint
from scripts.qwen17_transplant_atlas_campaign import _prune_cache_arrays, _valid_cache
from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_bridge_ablation_campaign import _cache_arguments
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import main as run_layer


PROTOCOL = "exp066-m3q-scale-stabilized-exact-dual-bridge"
SOURCE_MODEL = "1.7b-base"
LAYER_ORDER = (27, 20, 13, 6, 0)
SEEDS = (123, 456, 789)
ARMS = (
    "CONTROL-RANDOM",
    "BALANCED-RANK-LIFT",
    "M3Q-DUAL-RANDOM",
    "M3Q-DUAL-RANDOM-PRECAL-WHITENED-NO-COMPLEX",
    "M3Q-DUAL-RANDOM-PRECAL-WHITENED",
)
PRIMARY_ARM = "M3Q-DUAL-RANDOM-PRECAL-WHITENED"
STEPS = 8_192
DUAL_STEPS = 2_048
CHECKPOINTS = (0, 256, 1024, 2048, 4096, 8192)
SEQUENCE_LENGTH = 96
VALIDATION_WINDOWS = 64
TRAIN_OFFSET = 393_216
VALIDATION_OFFSET = 131_072
CACHE_START_RESERVE_SECONDS = 35 * 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def aggregate_results(layers: dict[str, dict]) -> dict:
    rows = {}
    counts = {
        "primary_beats_random": 0,
        "primary_beats_exact_lift": 0,
        "primary_beats_legacy_dual": 0,
        "complex_beats_no_complex": 0,
    }
    comparison_arms = {
        "random": "CONTROL-RANDOM",
        "exact_lift": "BALANCED-RANK-LIFT",
        "legacy_dual": "M3Q-DUAL-RANDOM",
        "no_complex": "M3Q-DUAL-RANDOM-PRECAL-WHITENED-NO-COMPLEX",
    }
    for layer, result in sorted(layers.items(), key=lambda item: int(item[0])):
        per_arm = {arm: [] for arm in ARMS}
        for seed in map(str, SEEDS):
            arms = result.get("seeds", {}).get(seed, {}).get("arms", {})
            for arm in ARMS:
                record = arms.get(arm)
                if record and record.get("recovery", {}).get("complete"):
                    per_arm[arm].append(_endpoint(record))
        layer_row = {}
        for arm, values in per_arm.items():
            if len(values) == len(SEEDS):
                array = np.asarray(values, np.float64)
                layer_row[arm] = {
                    "mean_final_decoder_relative_l2": float(array[:, 0].mean()),
                    "mean_normalized_auc": float(array[:, 1].mean()),
                    "seed_finals": array[:, 0].tolist(),
                }
        primary = layer_row.get(PRIMARY_ARM)
        comparisons = {}
        if primary:
            for label, baseline_name in comparison_arms.items():
                baseline = layer_row.get(baseline_name)
                if not baseline:
                    continue
                final_gain = 1.0 - (
                    primary["mean_final_decoder_relative_l2"]
                    / baseline["mean_final_decoder_relative_l2"]
                )
                auc_gain = 1.0 - (
                    primary["mean_normalized_auc"] / baseline["mean_normalized_auc"]
                )
                wins = int(
                    np.sum(
                        np.asarray(primary["seed_finals"])
                        < np.asarray(baseline["seed_finals"])
                    )
                )
                passed = bool(final_gain > 0 and auc_gain > 0 and wins >= 2)
                comparisons[label] = {
                    "final_gain": float(final_gain),
                    "auc_gain": float(auc_gain),
                    "paired_seed_wins": wins,
                    "layer_gate_passed": passed,
                }
                key = {
                    "random": "primary_beats_random",
                    "exact_lift": "primary_beats_exact_lift",
                    "legacy_dual": "primary_beats_legacy_dual",
                    "no_complex": "complex_beats_no_complex",
                }[label]
                counts[key] += int(passed)
        layer_row["comparisons"] = comparisons
        rows[layer] = layer_row
    completed_layers = sum(all(arm in row for arm in ARMS) for row in rows.values())
    primary_pass = bool(
        completed_layers == len(LAYER_ORDER)
        and counts["primary_beats_random"] >= 4
        and counts["primary_beats_exact_lift"] >= 4
        and counts["primary_beats_legacy_dual"] >= 4
    )
    mechanism_pass = counts["complex_beats_no_complex"] >= 3
    return {
        "layers": rows,
        "completed_layers": completed_layers,
        "layer_gate_counts": counts,
        "primary_screen_passed": primary_pass,
        "complex_mechanism_supported": mechanism_pass,
        "scientific_gate_passed": bool(primary_pass and mechanism_pass),
        "gate_definition": (
            "The stabilized complex arm beats random, balanced lift, and legacy "
            "dual-random in final error, AUC and >=2/3 seeds at >=4/5 layers; "
            "it also beats stabilized no-complex at >=3/5 layers."
        ),
    }


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# EXP-066 scale-stabilized Mamba-3 exact-dual bridge",
        "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Completed layers: `{aggregate['completed_layers']}/5`",
        f"- Numerical pass: `{result['passed']}`",
        f"- Scientific gate: `{aggregate['scientific_gate_passed']}`",
        "",
        "| Layer | Arm | Final decoder L2 | Normalized AUC |",
        "|---:|---|---:|---:|",
    ]
    for layer, row in aggregate["layers"].items():
        for arm in ARMS:
            arm_row = row.get(arm)
            if arm_row:
                lines.append(
                    f"| {layer} | {arm} | "
                    f"{arm_row['mean_final_decoder_relative_l2']:.8f} | "
                    f"{arm_row['mean_normalized_auc']:.8f} |"
                )
    lines.extend(["", "Layer gate counts:"])
    for name, count in aggregate["layer_gate_counts"].items():
        lines.append(f"- `{name}`: `{count}/5`")
    return "\n".join(lines) + "\n"


def _valid_layer_result(path: Path, layer: int) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("protocol") != PROTOCOL
        or int(payload.get("target_layer", -1)) != layer
        or not payload.get("complete")
        or not payload.get("passed")
    ):
        return None
    return payload


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run EXP-066 stabilized dual bridge.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp066-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-066 requires one TPU v5e-8")
    if not 0 < args.max_wall_hours < 8:
        raise ValueError("EXP-066 wall budget must be positive and below eight hours")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    started = _now()
    stage = "startup"
    stage_path = output / "exp066-stage-manifest.json"
    start_notice = _safe_notify(
        args.telegram,
        f"Extent TPU campaign\nstatus=started\nexperiment=EXP-066 stabilized bridge"
        f"\nhost={socket.gethostname()}\nbudget_hours={args.max_wall_hours}",
    )
    update_stage_manifest(
        stage_path,
        experiment=PROTOCOL,
        stage=stage,
        status="running",
        details={"started_at_utc": started, "layer_order": list(LAYER_ORDER)},
    )
    results: dict[str, dict] = {}
    removed: list[str] = []
    try:
        for layer in LAYER_ORDER:
            if time.monotonic() + CACHE_START_RESERVE_SECONDS >= deadline:
                print(f"exp066_layer={layer} SKIP-NO-CACHE-HEADROOM")
                break
            stage = f"layer-{layer}"
            layer_path = output / f"exp066-layer{layer}-training.json"
            existing = _valid_layer_result(layer_path, layer) if args.resume else None
            if existing:
                results[str(layer)] = existing
                print(f"exp066_layer={layer} RESUME-PASS")
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
                artifact_prefix="exp066",
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
            if time.monotonic() >= deadline:
                break
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
                    "--dual-bridge-steps", str(DUAL_STEPS),
                    "--dual-bridge-learning-rate", "3e-5",
                    "--seeds", ",".join(map(str, SEEDS)),
                    "--arms", ",".join(ARMS),
                    "--data-seed", "20260831",
                    "--compute-dtype", "bfloat16",
                    "--experiment-protocol", PROTOCOL,
                    "--result-json", str(layer_path),
                    "--output-dir", str(output),
                ],
                deadline_monotonic=deadline,
            )
            results[str(layer)] = layer_result
            update_stage_manifest(
                stage_path,
                experiment=PROTOCOL,
                stage=stage,
                status="completed" if layer_result.get("complete") else "deadline_partial",
                details={"complete": bool(layer_result.get("complete"))},
            )
            removed += _prune_cache_arrays(train_dir, output)
            removed += _prune_cache_arrays(eval_dir, output)
            jax.clear_caches()
            gc.collect()
            if not layer_result.get("complete"):
                break

        aggregate = aggregate_results(results)
        complete = aggregate["completed_layers"] == len(LAYER_ORDER)
        result = {
            "protocol": PROTOCOL,
            "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
            "status": "completed" if complete else "deadline_partial",
            "complete": complete,
            "passed": bool(results) and all(row.get("passed") for row in results.values()),
            "started_at_utc": started,
            "completed_at_utc": _now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
            "max_wall_hours": args.max_wall_hours,
            "cache_start_reserve_seconds": CACHE_START_RESERVE_SECONDS,
            "layer_order": list(LAYER_ORDER),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "recovery_steps": STEPS,
            "dual_bridge_steps": DUAL_STEPS,
            "checkpoints": list(CHECKPOINTS),
            "sequence_length": SEQUENCE_LENGTH,
            "training_token_offset": TRAIN_OFFSET,
            "validation_token_offset": VALIDATION_OFFSET,
            "layer_results": results,
            "aggregate": aggregate,
            "removed_regenerable_arrays": removed,
            "start_notification": start_notice,
        }
        result_path = output / "extent-m3q-stabilized-bridge-campaign.json"
        _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-m3q-stabilized-bridge-campaign-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(
            stage_path,
            experiment=PROTOCOL,
            stage="final",
            status=result["status"],
            details={"summary": str(summary), "aggregate": aggregate},
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment=EXP-066"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\nlayers={aggregate['completed_layers']}/5"
            f"\ngate={aggregate['scientific_gate_passed']}",
        )
        print(
            f"EXP066-{result['status'].upper()} layers={aggregate['completed_layers']}/5 "
            f"gate={aggregate['scientific_gate_passed']}\nsummary={summary.resolve()}"
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
            "started_at_utc": started,
            "failed_at_utc": _now(),
            "completed_layer_results": results,
        }
        _write_json_with_output_mirror(
            output / "exp066-failure.json", failure, str(output)
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-066"
            f"\nstage={stage}\nerror={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
