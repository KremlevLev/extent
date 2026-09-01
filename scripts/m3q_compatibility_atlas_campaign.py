"""EXP-067: resumable 28-layer Transformer-to-Mamba compatibility atlas."""

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
from extent.hf_artifact_sync import (
    artifact_config_from_env,
    restore_artifact,
    upload_artifact,
)
from extent.qwen_source import QWEN3_1_7B_BASE
from scripts.m3q_complex_bridge_campaign import _endpoint
from scripts.qwen17_transplant_atlas_campaign import _prune_cache_arrays, _valid_cache
from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_bridge_ablation_campaign import _cache_arguments
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_mimo_lift import main as run_layer


PROTOCOL = "exp067-m3q-full-depth-compatibility-atlas"
CONFIRMATION_PROTOCOL = "exp068-m3q-long-horizon-compatibility-confirmation"
SOURCE_MODEL = "1.7b-base"
# Broad depth coverage appears early if a session ends before all 28 layers.
LAYER_ORDER = (
    27, 0, 20, 6, 13, 24, 3, 17, 10, 26, 1, 22, 5, 15,
    8, 19, 12, 25, 2, 23, 4, 21, 7, 18, 9, 16, 11, 14,
)
SEEDS = (123, 456)
ARMS = ("CONTROL-RANDOM", "M3Q-DUAL-RANDOM")
STEPS = 2_048
DUAL_STEPS = 1_024
CHECKPOINTS = (0, 256, 1024, 2048)
SEQUENCE_LENGTH = 64
VALIDATION_WINDOWS = 32
TRAIN_OFFSET = 1_179_648
VALIDATION_OFFSET = 196_608
CACHE_START_RESERVE_SECONDS = 20 * 60
HF_PREFIX = "experiments/exp067-compatibility-atlas"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    lrank, rrank = _rankdata(np.asarray(left)), _rankdata(np.asarray(right))
    if np.std(lrank) == 0 or np.std(rrank) == 0:
        return None
    return float(np.corrcoef(lrank, rrank)[0, 1])


def _mean_checkpoint(seed_records: list[dict], step: int) -> float:
    values = []
    for record in seed_records:
        recovery = record["recovery"]
        values.append(
            float(
                recovery["evaluations"][str(step)]["decoder_output"]["relative_l2"]
            )
        )
    return float(np.mean(values))


def aggregate_atlas(
    layers: dict[str, dict], *, seeds: tuple[int, ...] = SEEDS, final_step: int = STEPS
) -> dict:
    rows = {}
    for layer, result in sorted(layers.items(), key=lambda item: int(item[0])):
        arm_records = {arm: [] for arm in ARMS}
        for seed in map(str, seeds):
            arms = result.get("seeds", {}).get(seed, {}).get("arms", {})
            for arm in ARMS:
                record = arms.get(arm)
                if record and record.get("recovery", {}).get("complete"):
                    arm_records[arm].append(record)
        if any(len(records) != len(seeds) for records in arm_records.values()):
            continue
        random_endpoints = np.asarray(
            [_endpoint(record) for record in arm_records["CONTROL-RANDOM"]]
        )
        dual_endpoints = np.asarray(
            [_endpoint(record) for record in arm_records["M3Q-DUAL-RANDOM"]]
        )
        dual_final = float(dual_endpoints[:, 0].mean())
        random_final = float(random_endpoints[:, 0].mean())
        rows[layer] = {
            "random_final_decoder_relative_l2": random_final,
            "dual_final_decoder_relative_l2": dual_final,
            "dual_final_gain_over_random": float(1.0 - dual_final / random_final),
            "random_normalized_auc": float(random_endpoints[:, 1].mean()),
            "dual_normalized_auc": float(dual_endpoints[:, 1].mean()),
            "dual_auc_gain_over_random": float(
                1.0 - dual_endpoints[:, 1].mean() / random_endpoints[:, 1].mean()
            ),
            "dual_step256_decoder_relative_l2": _mean_checkpoint(
                arm_records["M3Q-DUAL-RANDOM"], 256
            ),
            "dual_step1024_decoder_relative_l2": _mean_checkpoint(
                arm_records["M3Q-DUAL-RANDOM"], 1024
            ),
            "paired_seed_final_wins": int(
                np.sum(dual_endpoints[:, 0] < random_endpoints[:, 0])
            ),
        }
    ordered = sorted(rows, key=lambda value: int(value))
    final = [rows[layer]["dual_final_decoder_relative_l2"] for layer in ordered]
    early256 = [rows[layer]["dual_step256_decoder_relative_l2"] for layer in ordered]
    early1024 = [rows[layer]["dual_step1024_decoder_relative_l2"] for layer in ordered]
    hardest = sorted(
        rows,
        key=lambda layer: rows[layer]["dual_final_decoder_relative_l2"],
        reverse=True,
    )
    retained = [int(layer) for layer in hardest[:4]] if len(rows) == 28 else []
    correlation256 = _spearman(early256, final)
    correlation1024 = _spearman(early1024, final)
    complete = len(rows) == 28
    predictor_pass = bool(
        complete
        and correlation1024 is not None
        and correlation1024 >= 0.80
    )
    return {
        "layers": rows,
        "completed_layers": len(rows),
        "retained_attention_count": 4,
        "retained_attention_fraction": 4 / 28,
        "provisional_retained_attention_layers": retained,
        "ranking_rule": (
            f"four largest mean step-{final_step} dual decoder relative-L2 values"
        ),
        "spearman_step256_to_final": correlation256,
        "spearman_step1024_to_final": correlation1024,
        "early_predictor_gate_passed": predictor_pass,
        "scientific_gate_passed": bool(complete and predictor_pass),
        "gate_definition": (
            f"all 28 layers complete with {len(seeds)} seed(s) and step-1024 ranking "
            "correlates with final compatibility ranking at Spearman >= 0.80"
        ),
    }


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    title = (
        "# EXP-068 Qwen3-1.7B long-horizon compatibility confirmation"
        if result["protocol"] == CONFIRMATION_PROTOCOL
        else "# EXP-067 Qwen3-1.7B full-depth Mamba compatibility atlas"
    )
    lines = [
        title,
        "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Completed layers: `{aggregate['completed_layers']}/28`",
        f"- Numerical pass: `{result['passed']}`",
        f"- Predictor/scientific gate: `{aggregate['scientific_gate_passed']}`",
        f"- Step-1024/final Spearman: `{aggregate['spearman_step1024_to_final']}`",
        f"- Provisional retained attention layers: `{aggregate['provisional_retained_attention_layers']}`",
        "",
        "| Layer | Random final | Dual final | Final gain | Dual AUC gain | Step 1024 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, row in sorted(
        aggregate["layers"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            f"| {layer} | {row['random_final_decoder_relative_l2']:.8f} | "
            f"{row['dual_final_decoder_relative_l2']:.8f} | "
            f"{100 * row['dual_final_gain_over_random']:+.2f}% | "
            f"{100 * row['dual_auc_gain_over_random']:+.2f}% | "
            f"{row['dual_step1024_decoder_relative_l2']:.8f} |"
        )
    comparison = result.get("exp067_comparison")
    if comparison:
        overlap = comparison["retained_attention_overlap"]
        overlap_text = "pending (atlas incomplete)" if overlap is None else f"{overlap}/4"
        lines += [
            "",
            "## Independent long-horizon confirmation",
            "",
            f"- EXP-067/068 rank Spearman: `{comparison['exp067_to_exp068_spearman']}`",
            f"- Retained-layer overlap: `{overlap_text}`",
            f"- EXP-067 layers: `{comparison['exp067_retained_attention_layers']}`",
            f"- EXP-068 layers: `{comparison['exp068_retained_attention_layers']}`",
            f"- Confirmation gate: `{comparison['confirmation_gate_passed']}`",
        ]
    return "\n".join(lines) + "\n"


def _valid_layer(path: Path, layer: int, protocol: str = PROTOCOL) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("protocol") != protocol
        or int(payload.get("target_layer", -1)) != layer
        or not payload.get("complete")
        or not payload.get("passed")
    ):
        return None
    return payload


def _try_restore(path: Path, remote: str, hub) -> bool:
    if hub is None:
        return False
    try:
        return restore_artifact(path, remote, hub)
    except Exception as exc:
        print(f"hf_artifact_restore=SKIP error={type(exc).__name__}: {exc}")
        return False


def _try_upload(path: Path, remote: str, hub, message: str) -> bool:
    if hub is None:
        return False
    try:
        upload_artifact(path, remote, hub, commit_message=message)
        print(f"hf_artifact_upload=PASS path={remote}")
        return True
    except Exception as exc:
        print(f"hf_artifact_upload=SKIP error={type(exc).__name__}: {exc}")
        return False


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run EXP-067 compatibility atlas.")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp067-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=6.75)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--long-horizon-confirmation", action="store_true")
    args = parser.parse_args(argv)
    confirmation = args.long_horizon_confirmation
    protocol = CONFIRMATION_PROTOCOL if confirmation else PROTOCOL
    seeds = (789,) if confirmation else SEEDS
    steps = 8_192 if confirmation else STEPS
    dual_steps = 2_048 if confirmation else DUAL_STEPS
    checkpoints = (0, 256, 1024, 4096, 8192) if confirmation else CHECKPOINTS
    train_offset = 1_441_792 if confirmation else TRAIN_OFFSET
    validation_offset = 229_376 if confirmation else VALIDATION_OFFSET
    artifact_prefix = "exp068" if confirmation else "exp067"
    hf_prefix = (
        "experiments/exp068-long-horizon-compatibility-confirmation"
        if confirmation else HF_PREFIX
    )
    if confirmation and args.max_wall_hours == 6.75:
        args.max_wall_hours = 4.5
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-067 requires one TPU v5e-8")
    if not 0 < args.max_wall_hours < 8:
        raise ValueError("EXP-067 wall budget must remain below eight hours")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hub = artifact_config_from_env() if args.hf_sync else None
    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    started = _now()
    stage = "startup"
    stage_path = output / f"{artifact_prefix}-stage-manifest.json"
    start_notice = _safe_notify(
        args.telegram,
        f"Extent TPU campaign\nstatus=started\nexperiment={artifact_prefix.upper()} compatibility atlas"
        f"\nhost={socket.gethostname()}\nhf_sync={hub is not None}",
    )
    results = {}
    removed = []
    restored_layers = []
    uploaded_layers = []
    try:
        for layer in LAYER_ORDER:
            stage = f"layer-{layer}"
            layer_path = output / f"{artifact_prefix}-layer{layer}-training.json"
            existing = _valid_layer(layer_path, layer, protocol)
            remote = f"{hf_prefix}/layers/layer-{layer}.json"
            if existing is None and _try_restore(layer_path, remote, hub):
                existing = _valid_layer(layer_path, layer, protocol)
                if existing:
                    restored_layers.append(layer)
            if existing:
                results[str(layer)] = existing
                print(f"exp067_layer={layer} RESUME-PASS")
                continue
            if time.monotonic() + CACHE_START_RESERVE_SECONDS >= deadline:
                print(f"exp067_layer={layer} SKIP-NO-CACHE-HEADROOM")
                break
            common = dict(
                layer=layer,
                qwen_cache_dir=args.qwen_cache_dir,
                qwen_storage="ram",
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output,
                compute_dtype="bfloat16",
                storage_dtype="float16",
                per_device_windows=4,
                artifact_prefix=artifact_prefix,
                recovery_steps=steps,
                validation_windows=VALIDATION_WINDOWS,
                sequence_length=SEQUENCE_LENGTH,
                training_token_offset=train_offset,
                validation_token_offset=validation_offset,
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
                    "--total-steps", str(steps),
                    "--checkpoints", ",".join(map(str, checkpoints)),
                    "--dual-bridge-steps", str(dual_steps),
                    "--seeds", ",".join(map(str, seeds)),
                    "--arms", ",".join(ARMS),
                    "--data-seed", "20260901",
                    "--compute-dtype", "bfloat16",
                    "--experiment-protocol", protocol,
                    "--result-json", str(layer_path),
                    "--output-dir", str(output),
                ],
                deadline_monotonic=deadline,
            )
            results[str(layer)] = layer_result
            if layer_result.get("complete") and _try_upload(
                layer_path, remote, hub, f"EXP-067 layer {layer} complete"
            ):
                uploaded_layers.append(layer)
            update_stage_manifest(
                stage_path,
                experiment=protocol,
                stage=stage,
                status="completed" if layer_result.get("complete") else "deadline_partial",
                details={"complete": bool(layer_result.get("complete")), "hf_remote": remote},
            )
            removed += _prune_cache_arrays(train_dir, output)
            removed += _prune_cache_arrays(eval_dir, output)
            jax.clear_caches()
            gc.collect()
            if not layer_result.get("complete"):
                break

        aggregate = aggregate_atlas(results, seeds=seeds, final_step=steps)
        baseline_comparison = None
        if confirmation and hub is not None:
            baseline_path = output / "exp067-reference.json"
            if _try_restore(baseline_path, f"{HF_PREFIX}/latest.json", hub):
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                old_rows = baseline.get("aggregate", {}).get("layers", {})
                new_rows = aggregate["layers"]
                shared = sorted(set(old_rows) & set(new_rows), key=int)
                old_values = [
                    old_rows[layer]["dual_final_decoder_relative_l2"]
                    for layer in shared
                ]
                new_values = [
                    new_rows[layer]["dual_final_decoder_relative_l2"]
                    for layer in shared
                ]
                rank_correlation = _spearman(old_values, new_values)
                old_selected = set(
                    baseline.get("aggregate", {}).get(
                        "provisional_retained_attention_layers", []
                    )
                )
                new_selected = set(aggregate["provisional_retained_attention_layers"])
                selection_available = aggregate["completed_layers"] == 28
                overlap = (
                    len(old_selected & new_selected) if selection_available else None
                )
                baseline_comparison = {
                    "shared_layers": len(shared),
                    "exp067_to_exp068_spearman": rank_correlation,
                    "retained_attention_overlap": overlap,
                    "selection_available": selection_available,
                    "exp067_retained_attention_layers": sorted(old_selected),
                    "exp068_retained_attention_layers": sorted(new_selected),
                    "confirmation_gate_passed": bool(
                        len(shared) == 28
                        and selection_available
                        and rank_correlation is not None
                        and rank_correlation >= 0.80
                        and overlap is not None
                        and overlap >= 3
                    ),
                }
        complete = aggregate["completed_layers"] == 28
        result = {
            "protocol": protocol,
            "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
            "status": "completed" if complete else "deadline_partial",
            "complete": complete,
            "passed": bool(results) and all(row.get("passed") for row in results.values()),
            "started_at_utc": started,
            "completed_at_utc": _now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
            "layer_order": list(LAYER_ORDER),
            "seeds": list(seeds),
            "arms": list(ARMS),
            "recovery_steps": steps,
            "dual_bridge_steps": dual_steps,
            "sequence_length": SEQUENCE_LENGTH,
            "layer_results": results,
            "aggregate": aggregate,
            "hf_sync_enabled": hub is not None,
            "hf_prefix": hf_prefix if hub else None,
            "hf_restored_layers": restored_layers,
            "hf_uploaded_layers": uploaded_layers,
            "removed_regenerable_arrays": removed,
            "start_notification": start_notice,
            "exp067_comparison": baseline_comparison,
        }
        stem = (
            "extent-m3q-long-horizon-compatibility-confirmation"
            if confirmation else "extent-m3q-compatibility-atlas"
        )
        result_path = output / f"{stem}.json"
        _write_json_with_output_mirror(result_path, result, str(output))
        summary = output / f"{stem}-summary.md"
        summary.write_text(render_summary(result), encoding="utf-8")
        _try_upload(result_path, f"{hf_prefix}/latest.json", hub, f"{artifact_prefix.upper()} atlas snapshot")
        _try_upload(summary, f"{hf_prefix}/latest-summary.md", hub, f"{artifact_prefix.upper()} atlas summary")
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment={artifact_prefix.upper()}"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\nlayers={aggregate['completed_layers']}/28"
            f"\nhf_sync={hub is not None}",
        )
        print(
            f"{artifact_prefix.upper()}-{result['status'].upper()} layers={aggregate['completed_layers']}/28 "
            f"gate={aggregate['scientific_gate_passed']}\nsummary={summary.resolve()}"
        )
        return result
    except BaseException as exc:
        failure = {
            "protocol": protocol,
            "status": "failed",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "failed_at_utc": _now(),
            "completed_layer_results": results,
        }
        failure_path = output / f"{artifact_prefix}-failure.json"
        _write_json_with_output_mirror(failure_path, failure, str(output))
        _try_upload(failure_path, f"{hf_prefix}/failure-latest.json", hub, f"{artifact_prefix.upper()} failure")
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus=failed\nexperiment={artifact_prefix.upper()}"
            f"\nstage={stage}\nerror={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
