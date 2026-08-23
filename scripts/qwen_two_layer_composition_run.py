from __future__ import annotations

import argparse
import gc
from pathlib import Path

import jax

from scripts.qwen_activation_cache import main as build_activation_cache
from scripts.qwen_two_layer_composition import main as run_composition
from extent.experiment_stage import update_stage_manifest
from extent.teacher_activation_cache import load_activation_cache


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
    artifact_dir = output_dir / f"exp044-layer{layer}-{role}-cache"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = artifact_dir / f"exp044-layer{layer}-{role}-manifest.json"
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


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Build all EXP-044 caches and run the experiment in one session."
    )
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp044-weights"
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache"
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--result-json",
        default="/kaggle/working/output/exp044-two-layer-composition.json",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float16"), default="float16"
    )
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse verified caches and endpoint checkpoints (default: enabled)",
    )
    args = parser.parse_args(argv)
    if args.per_device_windows < 1:
        raise ValueError("per-device-windows must be positive")
    if len(jax.devices()) != 8:
        raise ValueError("EXP-044 one-shot runner requires exactly eight TPU devices")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = output_dir / "exp044-stage-manifest.json"
    manifests: dict[tuple[int, str], Path] = {}
    artifact_dirs: dict[tuple[int, str], Path] = {}
    for evaluation_only in (False, True):
        for layer in (0, 18):
            cache_args, manifest, artifact_dir = _cache_arguments(
                layer=layer,
                evaluation_only=evaluation_only,
                qwen_cache_dir=args.qwen_cache_dir,
                dataset_cache_dir=args.dataset_cache_dir,
                output_dir=output_dir,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
                per_device_windows=args.per_device_windows,
            )
            role = "validation" if evaluation_only else "train"
            stage_name = f"cache-layer{layer}-{role}"
            cache_reused = False
            if args.resume and manifest.exists():
                try:
                    load_activation_cache(
                        manifest,
                        artifact_dir=artifact_dir,
                        verify_hashes=not args.skip_hash_verification,
                    )
                    cache_reused = True
                    print(f"building_activation_cache=layer{layer}-{role} RESUME-PASS")
                except (FileNotFoundError, ValueError) as exc:
                    print(
                        f"building_activation_cache=layer{layer}-{role} "
                        f"RESUME-REJECTED reason={exc}"
                    )
            if not cache_reused:
                update_stage_manifest(
                    stage_manifest,
                    experiment="exp044-layer0-layer18",
                    stage=stage_name,
                    status="running",
                )
                print(f"building_activation_cache=layer{layer}-{role} START")
                build_activation_cache(cache_args)
            manifests[(layer, role)] = manifest
            artifact_dirs[(layer, role)] = artifact_dir
            update_stage_manifest(
                stage_manifest,
                experiment="exp044-layer0-layer18",
                stage=stage_name,
                status="completed",
                details={
                    "manifest": str(manifest.resolve()),
                    "reused": cache_reused,
                },
            )
            print(f"building_activation_cache=layer{layer}-{role} DONE")
            jax.clear_caches()
            gc.collect()

    composition_args = [
        "--layer0-activation-cache-manifest", str(manifests[(0, "train")]),
        "--layer0-activation-cache-dir", str(artifact_dirs[(0, "train")]),
        "--layer18-activation-cache-manifest", str(manifests[(18, "train")]),
        "--layer18-activation-cache-dir", str(artifact_dirs[(18, "train")]),
        "--layer0-evaluation-cache-manifest", str(manifests[(0, "validation")]),
        "--layer0-evaluation-cache-dir", str(artifact_dirs[(0, "validation")]),
        "--layer18-evaluation-cache-manifest", str(manifests[(18, "validation")]),
        "--layer18-evaluation-cache-dir", str(artifact_dirs[(18, "validation")]),
        "--qwen-cache-dir", args.qwen_cache_dir,
        "--compute-dtype", args.compute_dtype,
        "--data-parallel",
        "--prune-consumed-shards",
        "--per-device-windows", str(args.per_device_windows),
        "--result-json", args.result_json,
        "--output-dir", str(output_dir),
        "--stage-manifest", str(stage_manifest),
    ]
    if args.resume:
        composition_args.append("--resume")
    if args.skip_hash_verification:
        composition_args.append("--skip-hash-verification")
    return run_composition(composition_args)


if __name__ == "__main__":
    main()
