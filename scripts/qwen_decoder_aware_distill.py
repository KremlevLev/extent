from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.offline_mamba_distill import _slice
from scripts.qwen_mamba3_distill_pilot import (
    _replace_output,
    _window_metrics,
    _write_json_with_output_mirror,
)
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.hardware import recommended_compute_dtype
from extent.layerwise_distillation import create_decoder_aware_train_step
from extent.layers.mamba3 import Mamba3MIMO
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_parity import ensure_layer_checkpoint
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.readout_calibration import fit_dual_ridge_readout
from extent.teacher_activation_cache import load_activation_cache
from extent.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _checkpoint_steps(value: str, total_steps: int) -> tuple[int, ...]:
    steps = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not steps or steps[0] != 0 or steps[-1] != total_steps:
        raise ValueError("checkpoints must include zero and total-steps")
    if steps[0] < 0 or steps[-1] > total_steps:
        raise ValueError("checkpoint is outside the training schedule")
    return steps


def _teacher_decoder_outputs(
    runner,
    tail_params,
    residual_inputs,
    mixer_targets,
    batch_windows: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(residual_inputs), batch_windows):
        stop = min(start + batch_windows, len(residual_inputs))
        output = runner(
            tail_params,
            residual_inputs[start:stop],
            mixer_targets[start:stop],
        )
        outputs.append(np.asarray(output, dtype=np.float32))
    return np.concatenate(outputs)


def _evaluate(
    runner,
    params,
    tail_params,
    residual_inputs,
    normalized_inputs,
    mixer_targets: np.ndarray,
    decoder_targets: np.ndarray,
    batch_windows: int,
) -> dict:
    mixer_predictions = []
    decoder_predictions = []
    for start in range(0, len(residual_inputs), batch_windows):
        stop = min(start + batch_windows, len(residual_inputs))
        mixer, decoder = runner(
            params,
            tail_params,
            residual_inputs[start:stop],
            normalized_inputs[start:stop],
        )
        jax.block_until_ready((mixer, decoder))
        mixer_predictions.append(np.asarray(mixer, dtype=np.float32))
        decoder_predictions.append(np.asarray(decoder, dtype=np.float32))
    mixer_predictions = np.concatenate(mixer_predictions)
    decoder_predictions = np.concatenate(decoder_predictions)
    return {
        "mixer_output": _window_metrics(mixer_targets, mixer_predictions),
        "decoder_output": _window_metrics(decoder_targets, decoder_predictions),
        "finite": bool(
            np.all(np.isfinite(mixer_predictions))
            and np.all(np.isfinite(decoder_predictions))
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare mixer-only and decoder-aware offline Mamba distillation."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--qwen-cache-dir", default="/content/qwen3-layer0-weights")
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--checkpoints", default="0,128,256,512,1024")
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--evaluation-batch-windows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/content/output")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--compute-dtype", choices=("auto", "float32", "bfloat16"), default="auto"
    )
    args = parser.parse_args(argv)
    if min(
        args.total_steps,
        args.batch_windows,
        args.evaluation_batch_windows,
        args.learning_rate,
        args.readout_ridge,
    ) <= 0:
        raise ValueError("step, batch, learning-rate, and ridge values must be positive")
    if args.decoder_loss_weight <= 0:
        raise ValueError("decoder-loss-weight must be positive for the joint arm")
    if min(args.seed, args.data_seed) < 0:
        raise ValueError("seeds must be non-negative")
    checkpoint_steps = _checkpoint_steps(args.checkpoints, args.total_steps)

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
    training_count = training_slice.stop - training_slice.start
    if args.total_steps * args.batch_windows != training_count:
        raise ValueError(
            "EXP-038 requires exactly one pass: total-steps * batch-windows "
            "must equal the number of training windows"
        )

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
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
        param_dtype=compute_dtype,
    )
    apply_mixer = lambda params, inputs: mamba.apply({"params": params}, inputs)
    calibration_inputs = jnp.asarray(
        arrays["normalized_input"][calibration_slice], dtype=compute_dtype
    )
    calibration_targets = np.asarray(
        arrays["attention_target"][calibration_slice], dtype=np.float32
    )
    base_params = mamba.init(jax.random.key(args.seed), calibration_inputs[:1])["params"]
    run_features = jax.jit(
        lambda params, inputs: mamba.apply(
            {"params": params}, inputs, return_features=True
        )[1]
    )
    features = np.asarray(run_features(base_params, calibration_inputs), dtype=np.float32)
    readout, readout_report = fit_dual_ridge_readout(
        jnp.asarray(features.reshape(-1, features.shape[-1])),
        jnp.asarray(calibration_targets.reshape(-1, source.hidden_size)),
        relative_ridge=args.readout_ridge,
    )
    initial_params = _replace_output(base_params, readout, compute_dtype)

    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(QWEN3_14B.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir,
        index_payload["weight_map"],
        source,
        int(manifest["target_layer"]),
        repo_id=QWEN3_14B.repo_id,
        revision=QWEN3_14B.revision,
    )
    tail_params = qwen3_decoder_tail_params(
        QwenCheckpointReader(model_dir), int(manifest["target_layer"])
    )
    tail = Qwen3DecoderTail(
        source.hidden_size,
        source.intermediate_size,
        source.rms_norm_eps,
        compute_dtype,
        jnp.float32,
    )
    apply_tail = lambda params, residual, mixer_output: tail.apply(
        {"params": params}, residual, mixer_output
    )
    teacher_runner = create_batched_teacher_tail_runner(tail)
    replacement_runner = create_batched_replacement_runner(mamba, tail)

    residual_eval = jnp.asarray(
        arrays["residual_input"][evaluation_slice], dtype=compute_dtype
    )
    normalized_eval = jnp.asarray(
        arrays["normalized_input"][evaluation_slice], dtype=compute_dtype
    )
    mixer_eval = jnp.asarray(
        arrays["attention_target"][evaluation_slice], dtype=compute_dtype
    )
    mixer_eval_np = np.asarray(mixer_eval, dtype=np.float32)
    decoder_eval_np = _teacher_decoder_outputs(
        teacher_runner,
        tail_params,
        residual_eval,
        mixer_eval,
        args.evaluation_batch_windows,
    )

    arms = {
        "MIXER-ONLY": 0.0,
        "JOINT-MIXER-DECODER": args.decoder_loss_weight,
    }
    results = {}
    all_finite = True
    for name, decoder_weight in arms.items():
        params = initial_params
        tx = create_lion(
            learning_rate=args.learning_rate,
            warmup_steps=min(2000, max(1, args.total_steps // 100)),
            total_steps=args.total_steps,
            weight_decay=0.0,
            max_grad_norm=1.0,
        )
        opt_state = tx.init(params)
        train_step = create_decoder_aware_train_step(
            apply_mixer,
            apply_tail,
            tx,
            decoder_loss_weight=decoder_weight,
            bf16_gradients=compute_dtype == jnp.bfloat16,
        )
        evaluations = {
            "0": _evaluate(
                replacement_runner,
                params,
                tail_params,
                residual_eval,
                normalized_eval,
                mixer_eval_np,
                decoder_eval_np,
                args.evaluation_batch_windows,
            )
        }
        finite = bool(evaluations["0"]["finite"])
        max_grad_norm = 0.0
        last_train_metrics = None
        for zero_based_step in range(args.total_steps):
            indices = deterministic_batch_indices(
                zero_based_step,
                args.batch_windows,
                training_count,
                args.data_seed,
            )
            cache_indices = training_slice.start + indices
            batch_residual = jnp.asarray(
                arrays["residual_input"][cache_indices], dtype=compute_dtype
            )
            batch_normalized = jnp.asarray(
                arrays["normalized_input"][cache_indices], dtype=compute_dtype
            )
            batch_mixer_targets = jnp.asarray(
                arrays["attention_target"][cache_indices], dtype=compute_dtype
            )
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                tail_params,
                batch_residual,
                batch_normalized,
                batch_mixer_targets,
            )
            jax.block_until_ready(metrics)
            last_train_metrics = {
                key: int(value)
                if key in {"grads_finite", "nonfinite_grad_leaves"}
                else float(value)
                for key, value in metrics.items()
            }
            max_grad_norm = max(max_grad_norm, last_train_metrics["grad_norm"])
            finite = finite and bool(metrics["grads_finite"]) and bool(
                np.isfinite(last_train_metrics["loss"])
            )
            completed_step = zero_based_step + 1
            if completed_step in checkpoint_steps:
                evaluation = _evaluate(
                    replacement_runner,
                    params,
                    tail_params,
                    residual_eval,
                    normalized_eval,
                    mixer_eval_np,
                    decoder_eval_np,
                    args.evaluation_batch_windows,
                )
                evaluations[str(completed_step)] = evaluation
                finite = finite and bool(evaluation["finite"])
                print(
                    f"{name} step={completed_step} "
                    f"mixer_l2={evaluation['mixer_output']['relative_l2']:.6g} "
                    f"decoder_l2={evaluation['decoder_output']['relative_l2']:.6g}"
                )
        results[name] = {
            "decoder_loss_weight": decoder_weight,
            "evaluations": evaluations,
            "last_train_metrics": last_train_metrics,
            "max_grad_norm": max_grad_norm,
            "finite": finite,
        }
        all_finite = all_finite and finite

    mixer_final = results["MIXER-ONLY"]["evaluations"][str(args.total_steps)]
    joint_final = results["JOINT-MIXER-DECODER"]["evaluations"][str(args.total_steps)]
    mixer_decoder_l2 = mixer_final["decoder_output"]["relative_l2"]
    joint_decoder_l2 = joint_final["decoder_output"]["relative_l2"]
    mixer_mixer_l2 = mixer_final["mixer_output"]["relative_l2"]
    joint_mixer_l2 = joint_final["mixer_output"]["relative_l2"]
    decoder_improvement = (mixer_decoder_l2 - joint_decoder_l2) / mixer_decoder_l2
    mixer_degradation = (joint_mixer_l2 - mixer_mixer_l2) / mixer_mixer_l2
    scientific_gate_passed = bool(
        all_finite and decoder_improvement >= 0.10 and mixer_degradation <= 0.10
    )
    result = {
        "source": source_name,
        "dataset": manifest.get("dataset"),
        "method": "paired_mixer_only_vs_decoder_aware_Mamba3_distillation",
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {name: str(path) for name, path in paths.items()},
        "target_layer": int(manifest["target_layer"]),
        "sequence_length": int(manifest["sequence_length"]),
        "training_windows": training_count,
        "unique_training_tokens_per_arm": training_count
        * int(manifest["sequence_length"]),
        "total_steps": args.total_steps,
        "batch_windows": args.batch_windows,
        "checkpoints": list(checkpoint_steps),
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "learning_rate": args.learning_rate,
        "readout_ridge": args.readout_ridge,
        "readout_calibration_relative_l2": float(
            readout_report.calibration_relative_l2
        ),
        "mamba_config": asdict(mamba_config),
        "arms": results,
        "comparison": {
            "decoder_relative_l2_improvement_fraction": decoder_improvement,
            "mixer_relative_l2_degradation_fraction": mixer_degradation,
            "required_decoder_improvement_fraction": 0.10,
            "maximum_allowed_mixer_degradation_fraction": 0.10,
        },
        "scientific_gate_passed": scientific_gate_passed,
        "passed": all_finite,
        "notes": [
            "Both arms start from identical readout-calibrated Mamba parameters and consume identical batches.",
            "The joint objective is the normalized mean of mixer and decoder relative MSE when decoder-loss-weight is one.",
            "The frozen Qwen decoder tail is shared and is never optimized.",
            "passed reports numerical execution; scientific_gate_passed reports the pre-registered quality threshold.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, args.output_dir
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if not all_finite:
        raise SystemExit("DECODER-AWARE-DISTILL-NONFINITE")
    print(
        "DECODER-AWARE-DISTILL-PASS"
        if scientific_gate_passed
        else "DECODER-AWARE-DISTILL-GATE-FAIL"
    )


if __name__ == "__main__":
    main()
