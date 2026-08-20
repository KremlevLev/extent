from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_mamba3_distill_pilot import (
    _replace_output,
    _window_metrics,
    _write_json_with_output_mirror,
)
from extent.config import Mamba3Config
from extent.hardware import recommended_compute_dtype
from extent.layerwise_distillation import create_layerwise_train_step
from extent.layers.mamba3 import Mamba3MIMO
from extent.offline_distillation import (
    deterministic_batch_indices,
    restore_offline_checkpoint,
    save_offline_checkpoint,
)
from extent.optimizer import create_lion
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B
from extent.readout_calibration import fit_dual_ridge_readout
from extent.teacher_activation_cache import load_activation_cache


def _slice(layout: dict, name: str) -> slice:
    bounds = layout[name]
    if len(bounds) != 2 or int(bounds[0]) < 0 or int(bounds[1]) <= int(bounds[0]):
        raise ValueError(f"invalid activation-cache {name} bounds: {bounds}")
    return slice(int(bounds[0]), int(bounds[1]))


def _evaluate(run_mamba, params, inputs, targets, batch_windows: int) -> dict:
    predictions = []
    for start in range(0, len(inputs), batch_windows):
        stop = min(start + batch_windows, len(inputs))
        predictions.append(
            np.asarray(run_mamba(params, inputs[start:stop]), dtype=np.float32)
        )
    return _window_metrics(targets, np.concatenate(predictions, axis=0))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resume mixer-only Mamba distillation from an offline Qwen activation cache."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument(
        "--max-run-steps",
        type=int,
        default=0,
        help="Maximum updates in this process; zero runs through total-steps.",
    )
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--evaluation-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    args = parser.parse_args(argv)
    positive = (
        args.total_steps,
        args.batch_windows,
        args.evaluation_batch_windows,
        args.checkpoint_every,
        args.evaluation_every,
    )
    if min(positive) < 1 or args.max_run_steps < 0:
        raise ValueError("step and batch arguments must be positive (max-run-steps may be zero)")
    if min(args.seed, args.data_seed) < 0:
        raise ValueError("seeds must be non-negative")
    if min(args.learning_rate, args.readout_ridge) <= 0:
        raise ValueError("learning rate and readout ridge must be positive")

    manifest, arrays, paths = load_activation_cache(
        args.activation_cache_manifest,
        artifact_dir=args.activation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if manifest.get("source") != source_name:
        raise ValueError("activation cache does not match the pinned Qwen3 source")
    layout = manifest["window_layout"]
    calibration_slice = _slice(layout, "calibration")
    training_slice = _slice(layout, "training")
    evaluation_slice = _slice(layout, "evaluation")
    normalized = arrays["normalized_input"]
    targets = arrays["attention_target"]
    training_count = training_slice.stop - training_slice.start

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    parameter_dtype = compute_dtype
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba_config = Mamba3Config()
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
    example_input = jnp.asarray(normalized[calibration_slice.start : calibration_slice.start + 1], dtype=compute_dtype)
    base_params = mamba.init(jax.random.key(args.seed), example_input)["params"]
    tx = create_lion(
        learning_rate=args.learning_rate,
        warmup_steps=min(2000, max(1, args.total_steps // 100)),
        total_steps=args.total_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    compatibility = {
        "source": source_name,
        "dataset": manifest.get("dataset"),
        "target_layer": int(manifest["target_layer"]),
        "sequence_length": int(manifest["sequence_length"]),
        "normalized_input_sha256": manifest["artifacts"]["normalized_input"]["sha256"],
        "attention_target_sha256": manifest["artifacts"]["attention_target"]["sha256"],
        "mamba_config": asdict(mamba_config),
        "compute_dtype": dtype_decision.dtype,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_windows": args.batch_windows,
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "readout_ridge": args.readout_ridge,
    }
    checkpoint_dir = Path(args.checkpoint_dir)
    history_path = checkpoint_dir / "metrics.json"

    if args.resume:
        params, opt_state, start_step, restored_metadata = restore_offline_checkpoint(
            checkpoint_dir,
            base_params,
            tx.init(base_params),
            expected_compatibility=compatibility,
        )
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else []
        )
        print(f"resume=PASS step={start_step}")
    else:
        calibration_inputs = jnp.asarray(
            normalized[calibration_slice], dtype=compute_dtype
        )
        calibration_targets = np.asarray(
            targets[calibration_slice], dtype=np.float32
        )
        features = np.asarray(
            run_features(base_params, calibration_inputs), dtype=np.float32
        )
        readout, readout_report = fit_dual_ridge_readout(
            jnp.asarray(features.reshape(-1, features.shape[-1])),
            jnp.asarray(calibration_targets.reshape(-1, source.hidden_size)),
            relative_ridge=args.readout_ridge,
        )
        params = _replace_output(base_params, readout, parameter_dtype)
        opt_state = tx.init(params)
        start_step = 0
        history = []
        restored_metadata = None
        print(
            f"readout_calibration=PASS relative_l2={readout_report.calibration_relative_l2:.6g}"
        )

    if start_step > args.total_steps:
        raise ValueError("checkpoint step exceeds configured total-steps")
    run_limit = (
        args.total_steps
        if args.max_run_steps == 0
        else min(args.total_steps, start_step + args.max_run_steps)
    )
    train_step = create_layerwise_train_step(
        apply_mamba,
        tx,
        bf16_gradients=parameter_dtype == jnp.bfloat16,
    )
    evaluation_inputs = jnp.asarray(
        normalized[evaluation_slice], dtype=compute_dtype
    )
    evaluation_targets = np.asarray(targets[evaluation_slice], dtype=np.float32)

    def record_evaluation(step: int) -> dict:
        metrics = _evaluate(
            run_mamba,
            params,
            evaluation_inputs,
            evaluation_targets,
            args.evaluation_batch_windows,
        )
        record = {"step": step, **metrics}
        history.append(record)
        _atomic_json(history_path, history)
        print(
            f"evaluation step={step} relative_l2={metrics['relative_l2']:.6g} "
            f"cosine={metrics['cosine_similarity']:.6g}"
        )
        return record

    if not history or int(history[-1]["step"]) != start_step:
        record_evaluation(start_step)

    finite = True
    last_train_metrics = None

    def write_result(completed_step: int, metadata: dict) -> None:
        result = {
            "status": "complete" if completed_step == args.total_steps else "in_progress",
            "method": "resumable_offline_Mamba3_activation_distillation",
            "source": source_name,
            "activation_cache_manifest": str(
                Path(args.activation_cache_manifest).resolve()
            ),
            "resolved_activation_artifacts": {
                name: str(path) for name, path in paths.items()
            },
            "target_layer": int(manifest["target_layer"]),
            "sequence_length": int(manifest["sequence_length"]),
            "training_windows": training_count,
            "unique_training_tokens": training_count
            * int(manifest["sequence_length"]),
            "processed_training_tokens": completed_step
            * args.batch_windows
            * int(manifest["sequence_length"]),
            "step": completed_step,
            "total_steps": args.total_steps,
            "batch_windows": args.batch_windows,
            "compute_dtype": dtype_decision.dtype,
            "jax_backend": jax.default_backend(),
            "mamba_config": asdict(mamba_config),
            "checkpoint": metadata,
            "history": history,
            "last_train_metrics": last_train_metrics,
            "finite": finite,
            "passed": bool(finite and completed_step == args.total_steps),
            "notes": [
                "No Qwen checkpoint or teacher module is loaded by this process.",
                "Training examples follow a deterministic per-epoch permutation recoverable from step and data seed.",
                "A completed checkpoint contains Mamba parameters and Lion optimizer state.",
            ],
        }
        mirror = _write_json_with_output_mirror(
            Path(args.result_json), result, args.output_dir
        )
        if mirror:
            print(f"output_json={mirror.resolve()}")

    if start_step == run_limit:
        if restored_metadata is None:
            raise RuntimeError("a zero-update run requires a restored checkpoint")
        write_result(start_step, restored_metadata)

    for zero_based_step in range(start_step, run_limit):
        indices = deterministic_batch_indices(
            zero_based_step,
            args.batch_windows,
            training_count,
            args.data_seed,
        )
        cache_indices = training_slice.start + indices
        batch_inputs = jnp.asarray(normalized[cache_indices], dtype=compute_dtype)
        batch_targets = jnp.asarray(targets[cache_indices], dtype=compute_dtype)
        params, opt_state, train_metrics = train_step(
            params, opt_state, batch_inputs, batch_targets
        )
        jax.block_until_ready(train_metrics)
        completed_step = zero_based_step + 1
        last_train_metrics = {
            key: int(value) if key in {"grads_finite", "nonfinite_grad_leaves"} else float(value)
            for key, value in train_metrics.items()
        }
        finite = finite and bool(train_metrics["grads_finite"]) and bool(
            np.isfinite(last_train_metrics["loss"])
        )
        should_evaluate = (
            completed_step % args.evaluation_every == 0
            or completed_step == run_limit
            or completed_step == args.total_steps
        )
        if should_evaluate:
            record_evaluation(completed_step)
        should_checkpoint = (
            completed_step % args.checkpoint_every == 0
            or completed_step == run_limit
            or completed_step == args.total_steps
        )
        if should_checkpoint:
            metadata = save_offline_checkpoint(
                checkpoint_dir,
                params,
                opt_state,
                step=completed_step,
                compatibility=compatibility,
            )
            write_result(completed_step, metadata)
            print(
                f"checkpoint step={completed_step} sha256={metadata['checkpoint_sha256']}"
            )
    if not finite:
        raise SystemExit("OFFLINE-MAMBA-DISTILL-NONFINITE")
    if run_limit < args.total_steps:
        print(f"OFFLINE-MAMBA-DISTILL-PAUSED step={run_limit}; rerun with --resume")
    else:
        print("OFFLINE-MAMBA-DISTILL-PASS")


if __name__ == "__main__":
    main()
