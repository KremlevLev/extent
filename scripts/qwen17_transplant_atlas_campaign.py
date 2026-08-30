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


PROTOCOL = "exp063-mamba3-in-the-qwen-homotopy-screen"
SOURCE_MODEL = "1.7b-base"
SEEDS = (123, 456, 789)
ARMS = (
    "CONTROL-RANDOM",
    "CONTROL-FLAT-QKVO",
    "BALANCED-RANK-LIFT",
    "M3Q-EXACT-LINEAR",
    "M3Q-EXACT-COSINE",
    "M3Q-EXACT-DELAYED-COSINE",
)
STEPS = 8192
CHECKPOINTS = (0, 256, 1024, 2048, 4096, 8192)
SEQUENCE_LENGTH = 64
VALIDATION_WINDOWS = 128
TRAIN_OFFSET = 393_216
VALIDATION_OFFSET = 0
# Uniform boundary/interior coverage. Compute is spent on recovery-method
# discovery rather than repeating the already-selected initializer at 28 depths.
LAYER_ORDER = (0, 6, 13, 20, 27)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arm_error(result: dict, seed: int, arm: str, step: int) -> float:
    return float(
        result["seeds"][str(seed)]["arms"][arm]["recovery"]
        ["evaluations"][str(step)]["decoder_output"]["relative_l2"]
    )


def aggregate_atlas(layer_results: dict[str, dict]) -> dict:
    prepared: dict[str, dict] = {}
    for layer, result in sorted(layer_results.items(), key=lambda item: int(item[0])):
        if not result.get("complete"):
            prepared[layer] = {"complete": False}
            continue
        curves: dict[str, np.ndarray] = {}
        for arm in ARMS:
            curves[arm] = np.asarray([
                np.mean([_arm_error(result, seed, arm, step) for seed in SEEDS])
                for step in CHECKPOINTS
            ], dtype=np.float64)
        random_curve = curves["CONTROL-RANDOM"]
        flat_curve = curves["CONTROL-FLAT-QKVO"]
        lift_curve = curves["BALANCED-RANK-LIFT"]
        x = np.asarray(CHECKPOINTS, dtype=np.float64) / STEPS
        auc = {
            arm: float(np.sum(0.5 * (curve[:-1] + curve[1:]) * np.diff(x)))
            for arm, curve in curves.items()
        }
        prepared[layer] = {
            "complete": True,
            "result": result,
            "curves": curves,
            "auc": auc,
            "random_curve": random_curve,
            "flat_curve": flat_curve,
            "lift_curve": lift_curve,
        }

    complete_prepared = [row for row in prepared.values() if row.get("complete")]
    homotopy_arms = ARMS[3:]
    mean_schedule_auc = {
        arm: float(np.mean([row["auc"][arm] for row in complete_prepared]))
        for arm in homotopy_arms
    }
    selected_arm = (
        min(homotopy_arms, key=lambda arm: mean_schedule_auc[arm])
        if complete_prepared else None
    )

    layers: dict[str, dict] = {}
    selected_final_deltas = []
    selected_auc_deltas = []
    for layer, prepared_row in prepared.items():
        if not prepared_row.get("complete") or selected_arm is None:
            layers[layer] = {"complete": False}
            continue
        result = prepared_row["result"]
        curves = prepared_row["curves"]
        auc = prepared_row["auc"]
        random_curve = prepared_row["random_curve"]
        flat_curve = prepared_row["flat_curve"]
        lift_curve = prepared_row["lift_curve"]
        paired_vs_exact = np.asarray([
            _arm_error(result, seed, selected_arm, STEPS)
            - _arm_error(result, seed, "BALANCED-RANK-LIFT", STEPS)
            for seed in SEEDS
        ])
        record = {
            "complete": True,
            "curves": {
                arm: dict(zip(map(str, CHECKPOINTS), map(float, curve)))
                for arm, curve in curves.items()
            },
            "normalized_auc": auc,
            "selected_homotopy_arm": selected_arm,
            "selected_homotopy_auc_improvement_over_exact": float(
                1.0 - auc[selected_arm] / auc["BALANCED-RANK-LIFT"]
            ),
            "final_random_mean": float(random_curve[-1]),
            "final_flat_qkvo_mean": float(flat_curve[-1]),
            "final_exact_lift_mean": float(lift_curve[-1]),
            "final_selected_homotopy_mean": float(curves[selected_arm][-1]),
            "final_selected_minus_exact": float(np.mean(paired_vs_exact)),
            "final_selected_relative_improvement_over_exact": float(
                1.0 - curves[selected_arm][-1] / lift_curve[-1]
            ),
            "final_selected_seed_wins_over_exact": int(np.sum(paired_vs_exact < 0)),
            "layer_gate_passed": bool(
                np.mean(paired_vs_exact) < 0
                and np.sum(paired_vs_exact < 0) >= 2
                and auc[selected_arm] < auc["BALANCED-RANK-LIFT"]
            ),
        }
        layers[layer] = record
        selected_final_deltas.append(record["final_selected_minus_exact"])
        selected_auc_deltas.append(auc[selected_arm] - auc["BALANCED-RANK-LIFT"])
    complete_rows = [row for row in layers.values() if row.get("complete")]
    wins = sum(bool(row["layer_gate_passed"]) for row in complete_rows)
    complete = len(complete_rows) == len(LAYER_ORDER)
    return {
        "layers": layers,
        "completed_layers": len(complete_rows),
        "layer_wins": wins,
        "mean_normalized_auc_by_homotopy_schedule": mean_schedule_auc,
        "selected_homotopy_schedule": selected_arm,
        "mean_selected_homotopy_minus_exact_final": float(np.mean(selected_final_deltas)) if selected_final_deltas else None,
        "mean_selected_homotopy_minus_exact_auc": float(np.mean(selected_auc_deltas)) if selected_auc_deltas else None,
        "exploratory_advancement_gate_passed": bool(
            complete
            and wins >= 4
            and np.mean(selected_final_deltas) < 0
            and np.mean(selected_auc_deltas) < 0
        ) if complete_rows else False,
        "gate_definition": (
            "Select one global schedule by lowest mean AUC across the five-layer "
            "screen; it must then beat standard exact lift at at least 4/5 layer "
            "gates and in across-layer mean final error and AUC. A pass advances "
            "the recipe to a fresh-data locked confirmation."
        ),
    }


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# EXP-063 Mamba-3 in the Qwen homotopy screen",
        "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Completed layers: `{aggregate['completed_layers']}/5`",
        f"- M3Q layer wins over exact lift: `{aggregate['layer_wins']}/{aggregate['completed_layers']}`",
        f"- Selected global schedule: `{aggregate['selected_homotopy_schedule']}`",
        f"- Exploratory advancement gate: `{aggregate['exploratory_advancement_gate_passed']}`",
        "",
        "| Layer | Random | Flat QKVO | Exact | Selected M3Q | Global schedule | Final gain | AUC gain | Wins | Gate |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for layer, row in aggregate["layers"].items():
        if not row.get("complete"):
            continue
        lines.append(
            f"| {layer} | {row['final_random_mean']:.7f} | "
            f"{row['final_flat_qkvo_mean']:.7f} | {row['final_exact_lift_mean']:.7f} | "
            f"{row['final_selected_homotopy_mean']:.7f} | {row['selected_homotopy_arm']} | "
            f"{100 * row['final_selected_relative_improvement_over_exact']:+.2f}% | "
            f"{100 * row['selected_homotopy_auc_improvement_over_exact']:+.2f}% | "
            f"{row['final_selected_seed_wins_over_exact']}/3 | {row['layer_gate_passed']} |"
        )
    lines += [
        "",
        "AUC integrates held-out deployable (alpha=1) decoder-output relative L2 "
        "over the registered checkpoints. Homotopy is never used during evaluation.",
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
    parser = argparse.ArgumentParser(
        description="Screen M3Q attention-to-Mamba homotopy on Qwen3-1.7B."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp063-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-063 M3Q requires one TPU v5e-8")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    started = _now()
    stage = "startup"
    stage_path = output / "exp063-stage-manifest.json"
    start_notice = _safe_notify(
        args.telegram,
        f"Extent TPU campaign\nstatus=started\nexperiment=EXP-063 Mamba-3 in the Qwen"
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
        result_path = output / "extent-m3q-homotopy-screen.json"
        _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / "extent-m3q-homotopy-screen-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        update_stage_manifest(
            stage_path, experiment=PROTOCOL, stage="final", status=result["status"],
            details={"completed_layers": aggregate["completed_layers"], "summary": str(summary)},
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment=EXP-063 M3Q homotopy"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\nlayers={aggregate['completed_layers']}/5\nwins={aggregate['layer_wins']}",
        )
        print(
            f"EXP063-{result['status'].upper()} layers={aggregate['completed_layers']}/5 "
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
