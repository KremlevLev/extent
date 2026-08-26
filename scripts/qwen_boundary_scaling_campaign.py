from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import traceback

import jax

from scripts.qwen_depth_objective_run import main as run_depth_objective
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_progressive_composition_campaign import (
    _atomic_json,
    _checkpoint_schedule,
    _trim_metric,
)
from scripts.qwen_progressive_composition_eval import main as run_composition
from extent.artifact_summary import write_compact_summary
from extent.endpoint_checkpoint import compact_endpoint_checkpoint
from extent.experiment_stage import update_stage_manifest
from extent.qwen_source import QWEN3_14B


SEEDS = (123, 456, 789)
LAYER_SETS = {
    8: (0, 6, 12, 18, 23, 29, 34, 39),
    16: (0, 1, 3, 6, 9, 12, 15, 18, 21, 23, 26, 29, 32, 34, 36, 39),
}
TARGET_LAYERS = LAYER_SETS[16]
LAYER_BUDGETS = {
    layer: (8192 if layer == 0 else 2048) for layer in TARGET_LAYERS
}
BRANCH_SETS = {
    "LAYER0-ONLY": (0,),
    "INTERNAL-7": tuple(layer for layer in LAYER_SETS[8] if layer != 0),
    "COMPOSED-8": LAYER_SETS[8],
    "INTERNAL-15": tuple(layer for layer in LAYER_SETS[16] if layer != 0),
    "COMPOSED-16": LAYER_SETS[16],
}
PROTOCOL = "exp052-8-to-16-layer-boundary-scaling"
EVALUATION_PROTOCOL = "exp052-boundary-scaling"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiment_for_layer(layer: int) -> str:
    return "exp052-long" if layer == 0 else "exp052-short"


def _endpoint_dir(output_dir: Path, layer: int) -> Path:
    return output_dir / f"{_experiment_for_layer(layer)}-layer{layer}-endpoints"


def _valid_completed_cell(
    path: Path, output_dir: Path, layer: int
) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        streamed = result["layer_results"][str(layer)]
        directory = _endpoint_dir(output_dir, layer)
        metadata = json.loads(
            (directory / "checkpoint.json").read_text(encoding="utf-8")
        )
        payload = directory / metadata["checkpoint_file"]
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    compatibility = metadata.get("compatibility", {})
    expected_source = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if (
        result.get("protocol") != f"{_experiment_for_layer(layer)}-layer{layer}"
        or result.get("target_layers") != [layer]
        or int(result.get("total_steps_per_arm", -1)) != LAYER_BUDGETS[layer]
        or not result.get("passed")
        or streamed.get("protocol") != EVALUATION_PROTOCOL
        or not streamed.get("passed")
        or compatibility.get("source") != expected_source
        or int(compatibility.get("target_layer", -1)) != layer
        or compatibility.get("seeds") != list(SEEDS)
        or int(compatibility.get("total_steps", -1)) != LAYER_BUDGETS[layer]
        or not payload.exists()
        or int(payload.stat().st_size) != int(metadata.get("checkpoint_bytes", -1))
    ):
        return None
    return result


def extract_standalone_joint_metrics(
    cell: dict, *, layer: int, steps: int
) -> dict:
    metrics = cell["layer_results"][str(layer)]["lm_metrics"]
    return {
        "original": _trim_metric(metrics["ORIGINAL-CACHED-QWEN"]),
        "seeds": {
            str(seed): _trim_metric(
                metrics[f"SEED-{seed}-JOINT-STEP{steps}"]
            )
            for seed in SEEDS
        },
    }


def build_endpoint_index(output_dir: Path, cells: dict[str, dict]) -> dict:
    layers = {}
    for layer in TARGET_LAYERS:
        steps = LAYER_BUDGETS[layer]
        checkpoint_dir = _endpoint_dir(output_dir, layer)
        metadata = json.loads(
            (checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8")
        )
        layers[str(layer)] = {
            "steps": steps,
            "experiment": _experiment_for_layer(layer),
            "endpoint_dir": checkpoint_dir.name,
            "endpoint_checkpoint_sha256": metadata["checkpoint_sha256"],
            "endpoint_checkpoint_bytes": metadata["checkpoint_bytes"],
            "endpoint_retained_arms": metadata.get("retained_arms"),
            "standalone_joint_metrics": extract_standalone_joint_metrics(
                cells[str(layer)], layer=layer, steps=steps
            ),
        }
    evaluation_dir = output_dir / "exp052-long-layer0-validation-cache"
    return {
        "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
        "protocol": PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "analysis_mode": "boundary_scaling",
        "method": "asymmetric_budget_8_to_16_layer_boundary_scaling",
        "seeds": list(SEEDS),
        "target_layers": list(TARGET_LAYERS),
        "layer_sets": {
            str(count): list(values) for count, values in LAYER_SETS.items()
        },
        "branch_sets": {
            label: list(values) for label, values in BRANCH_SETS.items()
        },
        "layer_budgets": {
            str(layer): steps for layer, steps in LAYER_BUDGETS.items()
        },
        "primary_objective": "JOINT-MIXER-DECODER",
        "evaluation_cache": {
            "manifest": str(
                evaluation_dir
                / "exp052-long-layer0-validation-manifest.json"
            ),
            "artifact_dir": str(evaluation_dir),
        },
        "layers": layers,
        "notes": [
            "The frozen 8-layer set is extended to a nested 16-layer set.",
            "Layer 0 uses 8,192 steps; all internal layers use 2,048 steps.",
            "Layer-0-only and internal-only branches isolate boundary dominance.",
            "All branches share the same 256 paired validation windows.",
            "Composition scaling and boundary mechanism use separate gates.",
        ],
    }


def _require_disk_headroom(output_dir: Path, layer: int) -> None:
    required_gib = 12 if layer == 0 else 7
    free = shutil.disk_usage(output_dir).free
    print(
        f"disk_free_gib={free / 1024**3:.3f} "
        f"required_gib={required_gib} layer={layer}"
    )
    if free < required_gib * 1024**3:
        raise RuntimeError(
            f"EXP-052 requires {required_gib} GiB free before layer {layer}"
        )


def _prune_cell_caches(
    output_dir: Path, layer: int, *, keep_layer0_validation: bool
) -> list[str]:
    removed = []
    root = output_dir.resolve()
    experiment = _experiment_for_layer(layer)
    for role in ("train", "validation"):
        if layer == 0 and role == "validation" and keep_layer0_validation:
            continue
        directory = (
            output_dir / f"{experiment}-layer{layer}-{role}-cache"
        ).resolve()
        if root not in directory.parents:
            raise ValueError("refusing to prune outside EXP-052 output")
        if directory.exists():
            for path in directory.glob("*.npy"):
                path.unlink()
                removed.append(str(path))
    return removed


def _prune_final_payloads(output_dir: Path) -> list[str]:
    removed = []
    root = output_dir.resolve()
    for layer in TARGET_LAYERS:
        checkpoint_dir = _endpoint_dir(output_dir, layer)
        metadata_path = checkpoint_dir / "checkpoint.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        endpoint = (checkpoint_dir / metadata["checkpoint_file"]).resolve()
        if root not in endpoint.parents:
            raise ValueError("refusing to prune outside EXP-052 output")
        if endpoint.exists():
            endpoint.unlink()
            removed.append(str(endpoint))
        removed.extend(
            _prune_cell_caches(
                output_dir, layer, keep_layer0_validation=False
            )
        )
    return removed


def _valid_final(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("protocol") != PROTOCOL or not result.get("passed"):
        return None
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run resilient EXP-052 8-to-16-layer boundary scaling."
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/extent-boundary-scaling-campaign.json",
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp052-weights"
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
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument(
        "--telegram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-052 requires exactly eight TPU devices")
    if min(
        args.per_device_windows,
        args.evaluation_batch_windows,
        args.bootstrap_samples,
    ) < 1:
        raise ValueError("window and bootstrap settings must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    stage_manifest = output_dir / "exp052-campaign-stage-manifest.json"
    completed = _valid_final(result_path) if args.resume else None
    if completed is not None:
        summaries = write_compact_summary(
            completed, artifact_path=result_path, output_dir=output_dir
        )
        removed = _prune_final_payloads(output_dir)
        print(
            "exp052_campaign=RESUME-COMPLETE "
            f"summary={summaries['summary_markdown']} pruned={len(removed)}"
        )
        return completed

    started = _utc_now()
    start_notification = _safe_notify(
        args.telegram,
        "Extent TPU campaign\n"
        "status=started\n"
        f"host={socket.gethostname()}\n"
        "experiment=EXP-052 8-to-16-layer boundary scaling",
    )
    update_stage_manifest(
        stage_manifest,
        experiment=EVALUATION_PROTOCOL,
        stage="campaign",
        status="running",
        details={"started_at_utc": started},
    )

    cells: dict[str, dict] = {}
    removed_intermediate: list[str] = []
    current_stage = "startup"
    try:
        for layer in sorted(TARGET_LAYERS, reverse=True):
            current_stage = f"standalone-layer{layer}"
            steps = LAYER_BUDGETS[layer]
            experiment = _experiment_for_layer(layer)
            cell_path = output_dir / f"exp052-cell-layer{layer}.json"
            cell = (
                _valid_completed_cell(cell_path, output_dir, layer)
                if args.resume
                else None
            )
            if cell is None:
                _require_disk_headroom(output_dir, layer)
                arguments = [
                    "--experiment", experiment,
                    "--target-layers", str(layer),
                    "--total-steps", str(steps),
                    "--training-checkpoints", _checkpoint_schedule(steps),
                    "--qwen-cache-dir", args.qwen_cache_dir,
                    "--qwen-cache-storage", args.qwen_cache_storage,
                    "--dataset-cache-dir", args.dataset_cache_dir,
                    "--output-dir", str(output_dir),
                    "--result-json", str(cell_path),
                    "--compute-dtype", args.compute_dtype,
                    "--storage-dtype", args.storage_dtype,
                    "--evaluation-batch-windows", str(args.evaluation_batch_windows),
                    "--per-device-windows", str(args.per_device_windows),
                    "--bootstrap-samples", str(args.bootstrap_samples),
                    "--bootstrap-seed", str(args.bootstrap_seed + layer),
                    "--resume" if args.resume else "--no-resume",
                ]
                print(f"exp052_standalone_layer={layer} START steps={steps}")
                cell = run_depth_objective(arguments)
            else:
                print(f"exp052_standalone_layer={layer} RESUME-PASS")
            cells[str(layer)] = cell
            metadata_path = _endpoint_dir(output_dir, layer) / "checkpoint.json"
            endpoint_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            compact_metadata = compact_endpoint_checkpoint(
                _endpoint_dir(output_dir, layer),
                retained_arms={"JOINT-MIXER-DECODER"},
                expected_compatibility=endpoint_metadata["compatibility"],
            )
            print(
                f"exp052_endpoint_layer={layer} COMPACT-PASS "
                f"bytes={compact_metadata['checkpoint_bytes']}"
            )
            if set(cells) == {str(value) for value in TARGET_LAYERS}:
                _atomic_json(
                    output_dir / "exp052-endpoint-index.json",
                    build_endpoint_index(output_dir, cells),
                )
            removed_intermediate.extend(
                _prune_cell_caches(
                    output_dir, layer, keep_layer0_validation=True
                )
            )
            update_stage_manifest(
                stage_manifest,
                experiment=EVALUATION_PROTOCOL,
                stage=current_stage,
                status="completed",
                details={
                    "steps": steps,
                    "cell_result": str(cell_path.resolve()),
                    "endpoint_checkpoint": str(
                        (_endpoint_dir(output_dir, layer) / "checkpoint.json").resolve()
                    ),
                },
            )
            jax.clear_caches()
            gc.collect()

        endpoint_index = build_endpoint_index(output_dir, cells)
        endpoint_index_path = output_dir / "exp052-endpoint-index.json"
        _atomic_json(endpoint_index_path, endpoint_index)
        qwen_cache_dirs = {cell["qwen_cache_dir"] for cell in cells.values()}
        if len(qwen_cache_dirs) != 1:
            raise ValueError("standalone cells resolved different Qwen cache roots")
        current_stage = "boundary-streamed-evaluation"
        evaluation_path = output_dir / "exp052-boundary-scaling-evaluation.json"
        composition = run_composition(
            [
                "--endpoint-index", str(endpoint_index_path),
                "--qwen-cache-dir", qwen_cache_dirs.pop(),
                "--compute-dtype", args.compute_dtype,
                "--evaluation-batch-windows", str(args.evaluation_batch_windows),
                "--per-device-windows", str(args.per_device_windows),
                "--bootstrap-samples", str(args.bootstrap_samples),
                "--bootstrap-seed", str(args.bootstrap_seed),
                "--data-parallel",
                "--prune-consumed-shards",
                "--result-json", str(evaluation_path),
                "--output-dir", str(output_dir),
            ]
        )
        completed_at = _utc_now()
        final = {
            **composition,
            "started_at_utc": started,
            "completed_at_utc": completed_at,
            "optimizer_visible_tokens_total": sum(LAYER_BUDGETS.values())
            * 32
            * len(SEEDS)
            * 3,
            "standalone_cell_results": {
                str(layer): {
                    "steps": LAYER_BUDGETS[layer],
                    "result_json": str(
                        (output_dir / f"exp052-cell-layer{layer}.json").resolve()
                    ),
                    "scientific_gate_passed": cells[str(layer)][
                        "scientific_gate_passed"
                    ],
                    "passed": cells[str(layer)]["passed"],
                }
                for layer in TARGET_LAYERS
            },
            "start_notification": start_notification,
            "removed_intermediate_payloads": removed_intermediate,
        }
        mirror = _write_json_with_output_mirror(
            result_path, final, str(output_dir)
        )
        summaries = write_compact_summary(
            final, artifact_path=result_path, output_dir=output_dir
        )
        removed_final = _prune_final_payloads(output_dir)
        update_stage_manifest(
            stage_manifest,
            experiment=EVALUATION_PROTOCOL,
            stage="final",
            status="completed",
            details={
                "result_json": str(result_path.resolve()),
                "summary_markdown": summaries["summary_markdown"],
                "composition_scaling_gate_passed": final[
                    "aggregate"
                ]["composition_scaling_gate_passed"],
                "boundary_mechanism_gate_passed": final[
                    "aggregate"
                ]["boundary_mechanism_gate_passed"],
                "pruned_final_payloads": len(removed_final),
            },
        )
        _safe_notify(
            args.telegram,
            "Extent TPU campaign\n"
            "status=completed\n"
            "experiment=EXP-052 boundary scaling\n"
            f"numerical_pass={final['passed']}\n"
            "composition_gate="
            f"{final['aggregate']['composition_scaling_gate_passed']}\n"
            "boundary_gate="
            f"{final['aggregate']['boundary_mechanism_gate_passed']}",
        )
        print(
            "EXP-052-COMPLETE "
            f"numerical_pass={final['passed']} "
            "composition_gate="
            f"{final['aggregate']['composition_scaling_gate_passed']} "
            "boundary_gate="
            f"{final['aggregate']['boundary_mechanism_gate_passed']}"
        )
        print(f"result_json={result_path.resolve()}")
        print(f"compact_summary={summaries['summary_markdown']}")
        if mirror:
            print(f"output_json={mirror.resolve()}")
        return final
    except BaseException as exc:
        failure = {
            "experiment": EVALUATION_PROTOCOL,
            "status": "failed",
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "failed_at_utc": _utc_now(),
        }
        failure_path = output_dir / "extent-boundary-scaling-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output_dir))
        update_stage_manifest(
            stage_manifest,
            experiment=EVALUATION_PROTOCOL,
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
