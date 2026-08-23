from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil

import jax

from scripts.qwen_activation_cache import main as build_activation_cache
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_streamed_multiseed_end_to_end import main as run_layer
from extent.experiment_stage import update_stage_manifest
from extent.qwen_source import QWEN3_14B
from extent.teacher_activation_cache import load_activation_cache


TARGET_LAYERS = (0, 18)
RAM_CACHE_MINIMUM_FREE_BYTES = 36 * 1024**3


def resolve_qwen_cache_dir(
    requested: str,
    storage: str,
    *,
    ram_root: Path = Path("/dev/shm"),
) -> tuple[str, str]:
    """Choose RAM-backed storage only when the mounted filesystem can hold Qwen."""
    if storage not in {"auto", "disk", "ram"}:
        raise ValueError("qwen cache storage must be auto, disk, or ram")
    ram_available = False
    if ram_root.exists():
        try:
            ram_available = (
                shutil.disk_usage(ram_root).free >= RAM_CACHE_MINIMUM_FREE_BYTES
            )
        except OSError:
            ram_available = False
    if storage == "ram" and not ram_available:
        raise ValueError(
            "RAM Qwen cache requested but /dev/shm has less than 36 GiB free"
        )
    if storage == "ram" or (storage == "auto" and ram_available):
        return str(ram_root / "extent-qwen3-exp045-weights"), "ram"
    return requested, "disk"


def stage_saved_output(source: str | Path, target: str | Path) -> dict:
    """Reuse immutable Kaggle Notebook Output without copying large artifacts."""
    source_root = Path(source).resolve()
    target_root = Path(target).resolve()
    if not (source_root / "exp045-stage-manifest.json").exists():
        raise ValueError("resume source does not contain EXP-045 output")
    if source_root == target_root:
        return {"source": str(source_root), "linked": [], "copied_json": []}
    target_root.mkdir(parents=True, exist_ok=True)
    linked = []
    immutable_names = [
        f"exp045-layer{layer}-{role}-cache"
        for layer in TARGET_LAYERS
        for role in ("train", "validation")
    ] + [f"exp045-layer{layer}-endpoints" for layer in TARGET_LAYERS]
    for name in immutable_names:
        source_path = source_root / name
        target_path = target_root / name
        if not source_path.exists() or target_path.exists():
            continue
        os.symlink(source_path, target_path, target_is_directory=True)
        linked.append(name)
    copied_json = []
    immutable_roots = {source_root / name for name in immutable_names}
    for source_path in source_root.rglob("*.json"):
        if any(root in source_path.parents for root in immutable_roots):
            continue
        relative = source_path.relative_to(source_root)
        target_path = target_root / relative
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_json.append(str(relative))
    return {
        "source": str(source_root),
        "linked": linked,
        "copied_json": copied_json,
    }


def _cache_arguments(
    *,
    layer: int,
    evaluation_only: bool,
    qwen_cache_dir: str,
    dataset_cache_dir: str,
    output_dir: Path,
    compute_dtype: str,
    storage_dtype: str,
    per_device_windows: int,
) -> tuple[list[str], Path, Path]:
    role = "validation" if evaluation_only else "train"
    artifact_dir = output_dir / f"exp045-layer{layer}-{role}-cache"
    manifest = artifact_dir / f"exp045-layer{layer}-{role}-manifest.json"
    arguments = [
        "--cache-dir", qwen_cache_dir,
        "--dataset-cache-dir", dataset_cache_dir,
        "--target-layer", str(layer),
        "--sequence-length", "32",
        "--microbatch-windows", "4",
        "--data-parallel",
        "--per-device-windows", str(per_device_windows),
        "--compute-dtype", compute_dtype,
        "--storage-dtype", storage_dtype,
        "--prune-consumed-shards",
        "--output-dir", str(artifact_dir),
        "--result-json", str(manifest),
    ]
    if evaluation_only:
        arguments.extend(
            [
                "--dataset-split", "validation",
                "--evaluation-only",
                "--calibration-windows", "0",
                "--training-windows", "0",
                "--evaluation-windows", "256",
                "--token-offset", "0",
            ]
        )
    else:
        arguments.extend(
            [
                "--dataset-split", "train",
                "--calibration-windows", "8",
                "--training-windows", "1024",
                "--evaluation-windows", "4",
                "--token-offset", "0",
            ]
        )
    return arguments, manifest, artifact_dir


def _valid_completed_result(path: Path, layer: int) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_source = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if (
        result.get("source") != expected_source
        or result.get("protocol") != "exp045-depth-objective"
        or int(result.get("target_layer", -1)) != layer
        or not result.get("passed")
    ):
        return None
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run the frozen EXP-045 depth-aware objective comparison."
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp045-weights"
    )
    parser.add_argument(
        "--qwen-cache-storage",
        choices=("auto", "disk", "ram"),
        default="auto",
        help="use /dev/shm when it exposes at least 36 GiB free (default: auto)",
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache"
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--resume-source-output-dir",
        help="read-only output directory attached from a previous Kaggle version",
    )
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/exp045-depth-objective.json",
    )
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--storage-dtype", default="float16")
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if len(jax.devices()) != 8:
        raise ValueError("EXP-045 one-shot runner requires exactly eight TPU devices")
    if min(args.per_device_windows, args.bootstrap_samples) < 1:
        raise ValueError("window and bootstrap counts must be positive")

    qwen_cache_dir, qwen_cache_storage = resolve_qwen_cache_dir(
        args.qwen_cache_dir, args.qwen_cache_storage
    )
    print(
        f"qwen_cache_storage={qwen_cache_storage} "
        f"qwen_cache_dir={qwen_cache_dir}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_source_output_dir:
        staged = stage_saved_output(args.resume_source_output_dir, output_dir)
        print(f"saved_output_stage={json.dumps(staged, sort_keys=True)}")
    stage_manifest = output_dir / "exp045-stage-manifest.json"
    manifests: dict[tuple[int, str], Path] = {}
    artifact_dirs: dict[tuple[int, str], Path] = {}
    for evaluation_only in (False, True):
        for layer in TARGET_LAYERS:
            cache_args, manifest, artifact_dir = _cache_arguments(
                layer=layer,
                evaluation_only=evaluation_only,
                qwen_cache_dir=qwen_cache_dir,
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output_dir,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows,
            )
            role = "validation" if evaluation_only else "train"
            reused = False
            if args.resume and manifest.exists():
                try:
                    load_activation_cache(
                        manifest,
                        artifact_dir=artifact_dir,
                        verify_hashes=not args.skip_hash_verification,
                    )
                    reused = True
                    print(f"activation_cache=layer{layer}-{role} RESUME-PASS")
                except (FileNotFoundError, ValueError) as exc:
                    print(f"activation_cache=layer{layer}-{role} REBUILD reason={exc}")
            if not reused:
                print(f"activation_cache=layer{layer}-{role} START")
                build_activation_cache(cache_args)
            manifests[(layer, role)] = manifest
            artifact_dirs[(layer, role)] = artifact_dir
            update_stage_manifest(
                stage_manifest,
                experiment="exp045-depth-objective",
                stage=f"cache-layer{layer}-{role}",
                status="completed",
                details={"manifest": str(manifest.resolve()), "reused": reused},
            )
            jax.clear_caches()
            gc.collect()

    layer_results = {}
    # Layer 18 runs first and leaves its downloaded later-layer shards available.
    # Layer 0 then consumes the full checkpoint and safely prunes it at the end.
    for layer in (18, 0):
        layer_json = output_dir / f"exp045-layer{layer}-result.json"
        completed = _valid_completed_result(layer_json, layer) if args.resume else None
        if completed is not None:
            print(f"depth_objective_layer={layer} RESUME-PASS")
            layer_results[str(layer)] = completed
            continue
        arguments = [
            "--activation-cache-manifest", str(manifests[(layer, "train")]),
            "--activation-cache-dir", str(artifact_dirs[(layer, "train")]),
            "--evaluation-cache-manifest", str(manifests[(layer, "validation")]),
            "--evaluation-cache-dir", str(artifact_dirs[(layer, "validation")]),
            "--qwen-cache-dir", qwen_cache_dir,
            "--target-layer", str(layer),
            "--protocol", "exp045-depth-objective",
            "--seeds", args.seeds,
            "--compute-dtype", args.compute_dtype,
            "--data-parallel",
            "--per-device-windows", str(args.per_device_windows),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--bootstrap-seed", str(args.bootstrap_seed + layer),
            "--endpoint-checkpoint-dir", str(
                output_dir / f"exp045-layer{layer}-endpoints"
            ),
            "--result-json", str(layer_json),
            "--output-dir", str(output_dir / f"exp045-layer{layer}"),
            "--stage-manifest", str(stage_manifest),
            "--prune-consumed-shards",
        ]
        if args.resume:
            arguments.append("--resume-endpoints")
        if args.skip_hash_verification:
            arguments.append("--skip-hash-verification")
        print(f"depth_objective_layer={layer} START")
        layer_results[str(layer)] = run_layer(arguments)
        update_stage_manifest(
            stage_manifest,
            experiment="exp045-depth-objective",
            stage=f"end-to-end-layer{layer}",
            status="completed",
            details={
                "result_json": str(layer_json.resolve()),
                "scientific_gate_passed": layer_results[str(layer)][
                    "scientific_gate_passed"
                ],
            },
        )
        jax.clear_caches()
        gc.collect()

    all_finite = all(result["passed"] for result in layer_results.values())
    scientific_gate_passed = bool(
        all_finite
        and all(
            result["scientific_gate_passed"]
            for result in layer_results.values()
        )
    )
    result = {
        "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
        "method": "cross_depth_counterfactual_contribution_objective",
        "protocol": "exp045-layer0-layer18",
        "target_layers": list(TARGET_LAYERS),
        "seeds": [int(value) for value in args.seeds.split(",")],
        "qwen_cache_storage": qwen_cache_storage,
        "qwen_cache_dir": qwen_cache_dir,
        "layer_results": layer_results,
        "scientific_gate_passed": scientific_gate_passed,
        "passed": all_finite,
        "notes": [
            "The counterfactual objective must beat mixer-only and decoder-aware controls at both frozen depths.",
            "Each depth requires a negative mean NLL delta, at least two of three seed wins, and a negative paired-bootstrap upper bound versus each control.",
            "A completed per-layer result is a durable restart boundary; no large checkpoint download to the user's computer is required.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, str(output_dir)
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    print(
        "DEPTH-OBJECTIVE-COMPARISON-PASS"
        if scientific_gate_passed
        else "DEPTH-OBJECTIVE-COMPARISON-GATE-FAIL"
    )
    return result


if __name__ == "__main__":
    main()
