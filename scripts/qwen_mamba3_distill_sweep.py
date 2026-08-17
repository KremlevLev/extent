from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_mamba3_distill_pilot import (
    _replace_output,
    _window_metrics,
    _write_json_with_output_mirror,
)
from singularity.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from singularity.config import Mamba3Config
from singularity.hardware import recommended_compute_dtype
from singularity.layerwise_distillation import (
    create_layerwise_train_step,
    create_teacher_mixer_runner,
)
from singularity.layers.common import RMSNorm
from singularity.layers.mamba3 import Mamba3MIMO
from singularity.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from singularity.optimizer import create_lion
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_mixer_arrays,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.readout_calibration import fit_dual_ridge_readout
from singularity.teacher_activation_cache import load_activation_cache
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_unique_ints(value: str, label: str, *, positive: bool) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError(f"{label} must not be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must contain unique values")
    if positive and any(item <= 0 for item in parsed):
        raise ValueError(f"{label} must contain positive values")
    if not positive and any(item < 0 for item in parsed):
        raise ValueError(f"{label} must contain non-negative values")
    return parsed


def aggregate_runs(
    runs: dict,
    variants: tuple[str, ...],
    checkpoints: tuple[int, ...],
) -> dict:
    """Aggregate paired seeds without hiding the individual measurements."""
    aggregate = {}
    for variant in variants:
        by_checkpoint = {}
        initial = np.asarray(
            [run[variant]["checkpoints"]["0"]["relative_l2"] for run in runs.values()],
            dtype=np.float64,
        )
        for checkpoint in checkpoints:
            values = [run[variant]["checkpoints"][str(checkpoint)] for run in runs.values()]
            l2 = np.asarray([value["relative_l2"] for value in values], dtype=np.float64)
            cosine = np.asarray(
                [value["cosine_similarity"] for value in values], dtype=np.float64
            )
            reduction = (initial - l2) / np.maximum(initial, np.finfo(np.float64).tiny)
            by_checkpoint[str(checkpoint)] = {
                "relative_l2_mean": float(l2.mean()),
                "relative_l2_std": float(l2.std(ddof=1)) if len(l2) > 1 else 0.0,
                "cosine_mean": float(cosine.mean()),
                "cosine_std": float(cosine.std(ddof=1)) if len(cosine) > 1 else 0.0,
                "relative_l2_reduction_from_step0_mean": float(reduction.mean()),
            }
        aggregate[variant] = by_checkpoint

    paired = {}
    if len(variants) == 2:
        first, second = variants
        for checkpoint in checkpoints:
            first_l2 = np.asarray(
                [run[first]["checkpoints"][str(checkpoint)]["relative_l2"] for run in runs.values()]
            )
            second_l2 = np.asarray(
                [run[second]["checkpoints"][str(checkpoint)]["relative_l2"] for run in runs.values()]
            )
            difference = first_l2 - second_l2
            paired[str(checkpoint)] = {
                "first_variant": first,
                "second_variant": second,
                "first_minus_second_relative_l2_mean": float(difference.mean()),
                "first_wins": int(np.sum(difference < 0)),
                "second_wins": int(np.sum(difference > 0)),
                "ties": int(np.sum(difference == 0)),
            }
    return {"by_variant": aggregate, "paired_comparison": paired}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired multi-seed Qwen3-to-Mamba3 distillation sweep."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/singularity-calibration-cache"
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=8)
    parser.add_argument("--checkpoint-steps", default="20,80")
    parser.add_argument("--evaluation-windows", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--variants", default="INIT-A-random,INIT-C-prior-qkvo-port"
    )
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--activation-cache-manifest")
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    args = parser.parse_args(argv)
    checkpoints = tuple(
        sorted(_parse_unique_ints(args.checkpoint_steps, "checkpoint-steps", positive=True))
    )
    seeds = _parse_unique_ints(args.seeds, "seeds", positive=False)
    variants_to_run = tuple(
        item.strip() for item in args.variants.split(",") if item.strip()
    )
    if not variants_to_run:
        raise ValueError("variants must not be empty")
    if min(args.sequence_length, args.calibration_windows, args.evaluation_windows) < 1:
        raise ValueError("sequence length and window counts must be positive")
    if args.readout_ridge <= 0:
        raise ValueError("readout ridge must be positive")
    max_steps = checkpoints[-1]
    evaluation_checkpoints = (0, *checkpoints)
    jax.config.update("jax_default_matmul_precision", "high")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    parameter_dtype = compute_dtype
    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba_config = Mamba3Config()
    window_count = args.calibration_windows + max_steps + args.evaluation_windows
    model_dir, shards = ensure_layer_checkpoint(
        args.cache_dir,
        index_payload["weight_map"],
        source,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    arrays = load_mixer_arrays(reader, source, args.layer_index)
    activation_cache = None
    activation_cache_paths = None
    if args.activation_cache_manifest:
        activation_cache, cached_arrays, activation_cache_paths = load_activation_cache(
            args.activation_cache_manifest,
            artifact_dir=args.activation_cache_dir,
            verify_hashes=True,
        )
        expected_layout = {
            "calibration": [0, args.calibration_windows],
            "training": [args.calibration_windows, args.calibration_windows + max_steps],
            "evaluation": [
                args.calibration_windows + max_steps,
                window_count,
            ],
            "total_windows": window_count,
        }
        checks = {
            "source": (activation_cache.get("source"), f"{spec.repo_id}@{spec.revision}"),
            "target_layer": (activation_cache.get("target_layer"), args.layer_index),
            "sequence_length": (
                activation_cache.get("sequence_length"),
                args.sequence_length,
            ),
            "window_layout": (activation_cache.get("window_layout"), expected_layout),
        }
        mismatches = {
            name: values for name, values in checks.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(f"activation cache does not match sweep: {mismatches}")
        normalized = cached_arrays["normalized_input"]
        teacher_targets = cached_arrays["attention_target"]
        expected_activation_shape = (
            window_count,
            args.sequence_length,
            source.hidden_size,
        )
        if tuple(normalized.shape) != expected_activation_shape:
            raise ValueError(
                f"cached normalized input shape {normalized.shape} != {expected_activation_shape}"
            )
        print(
            f"activation_cache=PASS manifest={Path(args.activation_cache_manifest).resolve()}"
        )
    else:
        tokens = load_wikitext2_tokens(
            window_count * args.sequence_length,
            args.dataset_cache_dir,
            tokenizer_repo=spec.repo_id,
            tokenizer_revision=spec.revision,
        )
        embedding_shard = index_payload["weight_map"]["model.embed_tokens.weight"]
        if embedding_shard not in shards:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                filename=embedding_shard,
                local_dir=model_dir,
            )
        attention_params = jax_attention_params(arrays, source, args.layer_index)
        hidden = jnp.asarray(
            reader.read_rows("model.embed_tokens.weight", tokens)
        ).reshape(window_count, args.sequence_length, source.hidden_size)
        norm_scale = jnp.asarray(
            arrays[f"model.layers.{args.layer_index}.input_layernorm.weight"]
        )
        normalized = RMSNorm(
            source.hidden_size, source.rms_norm_eps, jnp.float32
        ).apply({"params": {"scale": norm_scale}}, hidden).astype(compute_dtype)
        positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None]
        mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
        teacher_module = Qwen3GQAAttention(source)
        run_teacher = create_teacher_mixer_runner(teacher_module, positions, mask)
        print(f"precomputing_teacher_windows={window_count}")
        teacher_targets = np.stack(
            [
                np.asarray(
                    run_teacher(
                        attention_params, normalized[index : index + 1]
                    )[0],
                    dtype=np.float32,
                )
                for index in range(window_count)
            ]
        )
        del attention_params, run_teacher, teacher_module
        jax.clear_caches()

    calibration_slice = slice(0, args.calibration_windows)
    training_start = args.calibration_windows
    evaluation_slice = slice(training_start + max_steps, window_count)
    calibration_inputs = normalized[calibration_slice]
    calibration_targets = teacher_targets[calibration_slice]
    evaluation_inputs = normalized[evaluation_slice]
    evaluation_targets = teacher_targets[evaluation_slice]
    mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=compute_dtype,
        param_dtype=parameter_dtype,
    )
    apply_mamba = lambda params, inputs: mamba.apply({"params": params}, inputs)
    run_mamba = jax.jit(apply_mamba)
    run_features = jax.jit(
        lambda params, inputs: mamba.apply(
            {"params": params}, inputs, return_features=True
        )[1]
    )
    tx = create_lion(
        learning_rate=args.learning_rate,
        warmup_steps=min(2, max(0, max_steps - 1)),
        total_steps=max_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    train_step = create_layerwise_train_step(
        apply_mamba,
        tx,
        bf16_gradients=parameter_dtype == jnp.bfloat16,
    )
    runs = {}
    passed = True
    trainable_parameters = None
    partial_path = (
        Path(args.result_json).with_suffix(".partial.json")
        if args.result_json
        else None
    )

    for seed in seeds:
        base = mamba.init(jax.random.key(seed), normalized[:1])["params"]
        variants, reports = build_qwen3_to_mamba3_transplant_variants(
            base, arrays, source, mamba_config, args.layer_index
        )
        unknown = sorted(set(variants_to_run) - set(variants))
        if unknown:
            raise ValueError(f"unknown variants: {unknown}; available={sorted(variants)}")
        seed_results = {}
        for name in variants_to_run:
            params = variants[name]
            calibration_features = np.asarray(
                run_features(params, calibration_inputs), dtype=np.float32
            )
            readout, readout_report = fit_dual_ridge_readout(
                jnp.asarray(
                    calibration_features.reshape(-1, calibration_features.shape[-1])
                ),
                jnp.asarray(calibration_targets.reshape(-1, source.hidden_size)),
                relative_ridge=args.readout_ridge,
            )
            params = _replace_output(params, readout, parameter_dtype)
            trainable_parameters = int(
                sum(value.size for value in jax.tree.leaves(params))
            )
            opt_state = tx.init(params)
            metrics_by_checkpoint = {
                "0": _window_metrics(
                    evaluation_targets,
                    np.asarray(run_mamba(params, evaluation_inputs), dtype=np.float32),
                )
            }
            loss_curve = []
            finite = True
            for step in range(1, max_steps + 1):
                index = training_start + step - 1
                params, opt_state, metrics = train_step(
                    params,
                    opt_state,
                    normalized[index : index + 1],
                    jnp.asarray(
                        teacher_targets[index : index + 1], dtype=compute_dtype
                    ),
                )
                jax.block_until_ready(metrics)
                record = {
                    key: (
                        int(value)
                        if key in {"grads_finite", "nonfinite_grad_leaves"}
                        else float(value)
                    )
                    for key, value in metrics.items()
                }
                record["step"] = step
                loss_curve.append(record)
                finite = (
                    finite
                    and bool(metrics["grads_finite"])
                    and bool(np.isfinite(record["loss"]))
                )
                if step in checkpoints:
                    prediction = np.asarray(
                        run_mamba(params, evaluation_inputs), dtype=np.float32
                    )
                    finite = finite and bool(np.all(np.isfinite(prediction)))
                    metrics_by_checkpoint[str(step)] = _window_metrics(
                        evaluation_targets, prediction
                    )
                    print(
                        f"seed={seed} {name} checkpoint={step} "
                        f"heldout_l2={metrics_by_checkpoint[str(step)]['relative_l2']:.6g} "
                        f"cosine={metrics_by_checkpoint[str(step)]['cosine_similarity']:.6g}"
                    )
            passed = passed and finite
            seed_results[name] = {
                "mapping": asdict(reports[name]),
                "readout_calibration": asdict(readout_report),
                "checkpoints": metrics_by_checkpoint,
                "loss_curve": loss_curve,
                "finite": finite,
            }
            runs[str(seed)] = seed_results
            if partial_path:
                mirror = _write_json_with_output_mirror(
                    partial_path,
                    {
                        "status": "in_progress",
                        "layer_index": args.layer_index,
                        "checkpoint_steps": evaluation_checkpoints,
                        "completed_runs": runs,
                    },
                    args.output_dir,
                )
                print(f"partial_result_json={partial_path.resolve()}")
                if mirror:
                    print(f"partial_output_json={mirror.resolve()}")

    aggregate = aggregate_runs(runs, variants_to_run, evaluation_checkpoints)
    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "paired_multi_seed_Mamba3_layerwise_distillation_sweep",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "calibration_windows": args.calibration_windows,
        "checkpoint_steps": evaluation_checkpoints,
        "maximum_training_tokens_per_variant_seed": max_steps * args.sequence_length,
        "evaluation_windows": args.evaluation_windows,
        "learning_rate": args.learning_rate,
        "readout_relative_ridge": args.readout_ridge,
        "seeds": seeds,
        "compute_dtype": dtype_decision.dtype,
        "dtype_reason": dtype_decision.reason,
        "jax_backend": jax.default_backend(),
        "mamba_config": asdict(mamba_config),
        "trainable_parameters": trainable_parameters,
        "activation_source": (
            "portable_activation_cache"
            if args.activation_cache_manifest
            else "on_the_fly_layer0_teacher"
        ),
        "activation_cache": (
            {
                "manifest": str(Path(args.activation_cache_manifest).resolve()),
                "producer_compute_dtype": activation_cache["compute_dtype"],
                "producer_storage_dtype": activation_cache["storage_dtype"],
                "resolved_artifacts": {
                    name: str(path) for name, path in activation_cache_paths.items()
                },
            }
            if activation_cache is not None
            else None
        ),
        "variants": variants_to_run,
        "runs": runs,
        "aggregate": aggregate,
        "passed": passed,
        "notes": [
            "Seeds are paired: both variants use the same random base for each seed.",
            "All checkpoints use one fixed held-out set after the maximum-budget training region.",
            "Checkpoint 20 is an intermediate point on the 80-step learning-rate schedule, not a separately scheduled run.",
            "Calibration, training, and evaluation windows are disjoint.",
            "The pass flag certifies finite execution only; it is not a scientific success criterion.",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        mirror = _write_json_with_output_mirror(output, result, args.output_dir)
        print(f"result_json={output.resolve()}")
        if mirror:
            print(f"output_json={mirror.resolve()}")
    if not passed:
        raise SystemExit("MAMBA3-DISTILL-SWEEP-FAIL")
    print("MAMBA3-DISTILL-SWEEP-PASS")


if __name__ == "__main__":
    main()
