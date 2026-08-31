from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from scripts.offline_mamba_distill import _slice
from scripts.qwen_bridge_ablation import (
    _calibrate_readout,
    _checkpoint_steps,
    _deadline_reached,
    _train_recovery,
)
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import Qwen3DecoderTail, qwen3_decoder_tail_params
from extent.hardware import recommended_compute_dtype
from extent.layers.mamba3 import Mamba3MIMO
from extent.mamba3_aware_bridge import (
    create_exact_dual_bridge_train_step,
    dual_recurrent_parity,
    zero_complex_projection,
)
from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_parity import ensure_layer_checkpoint, load_mixer_arrays
from extent.qwen_source import (
    QWEN_SOURCES,
    load_remote_source_metadata,
    teacher_config_from_spec,
)
from extent.teacher_activation_cache import load_activation_cache, validate_external_evaluation_cache
from extent.weight_mapping import QwenCheckpointReader


PROTOCOL = "exp056-operator-preserving-mimo-lift-screen"
ARM_ORDER = (
    "CONTROL-RANDOM",
    "CONTROL-FLAT-QKVO",
    "SINGLE-CHANNEL-LIFT",
    "BALANCED-RANK-LIFT",
)
M3Q_HOMOTOPY_ARMS = (
    "M3Q-EXACT-LINEAR",
    "M3Q-EXACT-COSINE",
    "M3Q-EXACT-DELAYED-COSINE",
)
M3Q_DUAL_BRIDGE_ARMS = (
    "M3Q-DUAL-RANDOM",
    "M3Q-DUAL-EXACT",
    "M3Q-DUAL-EXACT-NO-COMPLEX",
)
REGISTERED_ARMS = ARM_ORDER + M3Q_HOMOTOPY_ARMS + M3Q_DUAL_BRIDGE_ARMS
VARIANT_BY_ARM = {
    "CONTROL-RANDOM": "INIT-A-random",
    "CONTROL-FLAT-QKVO": "INIT-C-prior-qkvo-port",
    "SINGLE-CHANNEL-LIFT": "INIT-J-single-channel-qkvo-lift",
    "BALANCED-RANK-LIFT": "INIT-K-balanced-qkvo-lift",
    "M3Q-EXACT-LINEAR": "INIT-K-balanced-qkvo-lift",
    "M3Q-EXACT-COSINE": "INIT-K-balanced-qkvo-lift",
    "M3Q-EXACT-DELAYED-COSINE": "INIT-K-balanced-qkvo-lift",
    "M3Q-DUAL-RANDOM": "INIT-A-random",
    "M3Q-DUAL-EXACT": "INIT-K-balanced-qkvo-lift",
    "M3Q-DUAL-EXACT-NO-COMPLEX": "INIT-K-balanced-qkvo-lift",
}
HOMOTOPY_BY_ARM = {
    "M3Q-EXACT-LINEAR": "linear",
    "M3Q-EXACT-COSINE": "cosine",
    "M3Q-EXACT-DELAYED-COSINE": "delayed-cosine",
}


def _metric_record(metrics: dict) -> dict:
    return {
        key: bool(value)
        if key == "grads_finite"
        else int(value)
        if key == "nonfinite_grad_leaves"
        else float(value)
        for key, value in metrics.items()
    }


def _train_exact_dual_bridge(
    *,
    params,
    recurrent: Mamba3MIMO,
    dual: Mamba3MIMO,
    config: Mamba3Config,
    training_arrays,
    training_slice: slice,
    compute_dtype,
    steps: int,
    learning_rate: float,
    data_seed: int,
    freeze_complex: bool,
    parity_inputs,
    deadline_monotonic: float | None,
):
    if freeze_complex:
        params = zero_complex_projection(params, recurrent.hidden_size, config)
    tx = create_lion(
        learning_rate=learning_rate,
        warmup_steps=min(64, max(1, steps // 100)),
        total_steps=steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    apply_dual = lambda candidate, inputs: dual.apply({"params": candidate}, inputs)
    apply_recurrent = lambda candidate, inputs: recurrent.apply(
        {"params": candidate}, inputs
    )
    train_step = create_exact_dual_bridge_train_step(
        apply_dual,
        tx,
        hidden_size=recurrent.hidden_size,
        config=config,
        bf16_gradients=compute_dtype == jnp.bfloat16,
        freeze_complex=freeze_complex,
    )
    opt_state = tx.init(params)
    training_count = training_slice.stop - training_slice.start
    first = None
    last = None
    max_grad_norm = 0.0
    finite = True
    completed = 0
    for zero_step in range(steps):
        if zero_step % 8 == 0 and _deadline_reached(deadline_monotonic):
            break
        index = int(
            deterministic_batch_indices(
                zero_step, 1, training_count, data_seed
            )[0]
        )
        cache_index = training_slice.start + index
        inputs = jnp.asarray(
            training_arrays["normalized_input"][cache_index : cache_index + 1],
            dtype=compute_dtype,
        )
        targets = jnp.asarray(
            training_arrays["attention_target"][cache_index : cache_index + 1],
            dtype=compute_dtype,
        )
        params, opt_state, metrics = train_step(params, opt_state, inputs, targets)
        jax.block_until_ready(metrics)
        record = _metric_record(metrics)
        first = record if first is None else first
        last = record
        max_grad_norm = max(max_grad_norm, record["grad_norm"])
        finite = bool(
            finite and record["grads_finite"] and np.isfinite(record["loss"])
        )
        completed = zero_step + 1
    parity = dual_recurrent_parity(
        apply_dual,
        apply_recurrent,
        params,
        jnp.asarray(parity_inputs, dtype=compute_dtype),
    )
    parity_passed = bool(
        parity["finite"]
        and parity["relative_l2"] <= (2e-2 if compute_dtype == jnp.bfloat16 else 2e-5)
    )
    return params, {
        "kind": "canonical_mamba3_exact_ssd_dual_preconditioning",
        "requested_steps": steps,
        "completed_steps": completed,
        "learning_rate": learning_rate,
        "freeze_complex_projection": freeze_complex,
        "first_metrics": first,
        "last_metrics": last,
        "max_grad_norm": max_grad_norm,
        "finite": bool(finite and parity_passed),
        "complete": completed == steps,
        "bridge_to_recurrent_parity": parity,
        "parity_threshold": 2e-2 if compute_dtype == jnp.bfloat16 else 2e-5,
        "execution_switch_has_no_parameter_conversion": True,
    }, completed == steps and finite and parity_passed


def main(
    argv: list[str] | None = None,
    *,
    deadline_monotonic: float | None = None,
    return_endpoint_params: bool = False,
) -> dict | tuple[dict, dict[str, dict[str, dict]]]:
    parser = argparse.ArgumentParser(description="Screen exact SISO-to-MIMO QKVO lifts at one Qwen3 layer.")
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--evaluation-cache-manifest", required=True)
    parser.add_argument("--evaluation-cache-dir")
    parser.add_argument("--qwen-cache-dir", required=True)
    parser.add_argument("--source-model", choices=tuple(QWEN_SOURCES), default="14b")
    parser.add_argument("--total-steps", type=int, default=4096)
    parser.add_argument("--checkpoints", default="0,256,1024,2048,4096")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--dual-bridge-steps", type=int, default=1024)
    parser.add_argument("--dual-bridge-learning-rate", type=float, default=3e-5)
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--data-seed", type=int, default=20260827)
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-protocol", default=PROTOCOL)
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args(argv)
    checkpoints = _checkpoint_steps(args.checkpoints, args.total_steps)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("seeds must be unique non-negative integers")
    if not arms or len(arms) != len(set(arms)) or any(arm not in REGISTERED_ARMS for arm in arms):
        raise ValueError("arms must be unique registered MIMO-lift arms")
    if "CONTROL-RANDOM" not in arms:
        raise ValueError("arms must include CONTROL-RANDOM")
    if min(args.dual_bridge_steps, args.dual_bridge_learning_rate) <= 0:
        raise ValueError("dual bridge steps and learning rate must be positive")

    train_manifest, train_arrays, train_paths = load_activation_cache(
        args.activation_cache_manifest, artifact_dir=args.activation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    eval_manifest, eval_arrays, eval_paths = load_activation_cache(
        args.evaluation_cache_manifest, artifact_dir=args.evaluation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    evaluation_slice = validate_external_evaluation_cache(
        train_manifest, eval_manifest, allow_cross_split=True, required_dataset_split="validation"
    )
    calibration_slice = _slice(train_manifest["window_layout"], "calibration")
    training_slice = _slice(train_manifest["window_layout"], "training")
    if training_slice.stop - training_slice.start != args.total_steps:
        raise ValueError("MIMO lift screen requires one recovery window per requested step")
    layer = int(train_manifest["target_layer"])
    if int(eval_manifest["target_layer"]) != layer:
        raise ValueError("training and evaluation caches target different layers")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    spec = QWEN_SOURCES[args.source_model]
    expected_source = f"{spec.repo_id}@{spec.revision}"
    if train_manifest.get("source") != expected_source:
        raise ValueError(
            f"activation cache source is {train_manifest.get('source')!r}; "
            f"expected {expected_source!r}"
        )
    source = teacher_config_from_spec(
        spec,
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba_config = Mamba3Config()
    _, index_payload = load_remote_source_metadata(spec)
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir, index_payload["weight_map"], source, layer,
        repo_id=spec.repo_id, revision=spec.revision,
    )
    if spec.shard_count == 1:
        index_path = Path(model_dir) / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps(index_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    reader = QwenCheckpointReader(model_dir)
    mixer_arrays = load_mixer_arrays(reader, source, layer)
    tail_params = qwen3_decoder_tail_params(reader, layer)
    mamba = Mamba3MIMO(source.hidden_size, mamba_config, dtype=compute_dtype, param_dtype=compute_dtype)
    dual_mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=compute_dtype,
        param_dtype=compute_dtype,
        execution_mode="dual",
    )
    tail = Qwen3DecoderTail(
        source.hidden_size, source.intermediate_size, source.rms_norm_eps, compute_dtype, jnp.float32
    )
    calibration_inputs = np.asarray(train_arrays["normalized_input"][calibration_slice])
    calibration_targets = np.asarray(train_arrays["attention_target"][calibration_slice], np.float32)
    result_path = Path(args.result_json)
    partial_path = result_path.with_suffix(".partial.json")
    seed_results: dict[str, dict] = {}
    endpoint_params: dict[str, dict[str, dict]] = {}
    complete = True
    numerical_pass = True

    def persist(status: str) -> None:
        _write_json_with_output_mirror(partial_path, {
            "protocol": args.experiment_protocol, "status": status, "target_layer": layer,
            "arm_order": list(arms), "seeds": seed_results,
            "complete": complete, "passed": numerical_pass,
        }, args.output_dir)

    for seed in seeds:
        if _deadline_reached(deadline_monotonic):
            complete = False
            break
        base = mamba.init(jax.random.key(seed), jnp.asarray(calibration_inputs[:1], dtype=compute_dtype))["params"]
        variants, reports = build_qwen3_to_mamba3_transplant_variants(
            base, mixer_arrays, source, mamba_config, layer
        )
        seed_record = {"arms": {}}
        seed_results[str(seed)] = seed_record
        endpoint_params[str(seed)] = {}
        for arm in arms:
            if _deadline_reached(deadline_monotonic):
                complete = False
                break
            print(f"exp056_layer={layer} seed={seed} arm={arm} recovery=START")
            variant = VARIANT_BY_ARM[arm]
            initial = variants[variant]
            bridge_result = None
            if arm in M3Q_DUAL_BRIDGE_ARMS:
                print(
                    f"exp056_layer={layer} seed={seed} arm={arm} "
                    "exact_dual_bridge=START"
                )
                initial, bridge_result, bridge_complete = _train_exact_dual_bridge(
                    params=initial,
                    recurrent=mamba,
                    dual=dual_mamba,
                    config=mamba_config,
                    training_arrays=train_arrays,
                    training_slice=training_slice,
                    compute_dtype=compute_dtype,
                    steps=args.dual_bridge_steps,
                    learning_rate=args.dual_bridge_learning_rate,
                    data_seed=args.data_seed + seed + 30_000,
                    freeze_complex=arm == "M3Q-DUAL-EXACT-NO-COMPLEX",
                    parity_inputs=calibration_inputs[:1],
                    deadline_monotonic=deadline_monotonic,
                )
                numerical_pass = numerical_pass and bool(bridge_result["finite"])
                if not bridge_complete:
                    seed_record["arms"][arm] = {
                        "variant": variant,
                        "initializer": asdict(reports[variant]),
                        "dual_bridge": bridge_result,
                        "recovery": None,
                    }
                    persist("deadline_partial")
                    complete = False
                    break
            calibrated, readout_report = _calibrate_readout(
                mamba, initial, calibration_inputs, calibration_targets,
                compute_dtype, args.readout_ridge,
            )
            trained, recovery, arm_complete = _train_recovery(
                params=calibrated, mamba=mamba, tail=tail, tail_params=tail_params,
                training_arrays=train_arrays, training_slice=training_slice,
                evaluation_arrays=eval_arrays, evaluation_slice=evaluation_slice,
                compute_dtype=compute_dtype, total_steps=args.total_steps,
                checkpoints=checkpoints, learning_rate=args.learning_rate,
                decoder_loss_weight=args.decoder_loss_weight,
                data_seed=args.data_seed + seed,
                evaluation_batch_windows=args.evaluation_batch_windows,
                deadline_monotonic=deadline_monotonic,
                homotopy_schedule=HOMOTOPY_BY_ARM.get(arm, "none"),
            )
            seed_record["arms"][arm] = {
                "variant": variant, "initializer": asdict(reports[variant]),
                "dual_bridge": bridge_result,
                "readout_calibration": readout_report,
                "recovery_recipe": (
                    "mamba3_exact_dual_then_recurrent_decoder_aware"
                    if arm in M3Q_DUAL_BRIDGE_ARMS
                    else "standard_decoder_aware"
                    if arm not in HOMOTOPY_BY_ARM
                    else f"m3q_{HOMOTOPY_BY_ARM[arm]}_attention_to_mamba3_homotopy"
                ),
                "recovery": recovery,
            }
            numerical_pass = numerical_pass and bool(recovery["finite"])
            if arm_complete and return_endpoint_params:
                endpoint_params[str(seed)][arm] = trained
            persist("in_progress" if arm_complete else "deadline_partial")
            if not arm_complete:
                complete = False
                break
        if not complete:
            break
        jax.clear_caches()
        gc.collect()

    result = {
        "protocol": args.experiment_protocol,
        "status": "completed" if complete else "deadline_partial",
        "source": f"{spec.repo_id}@{spec.revision}",
        "method": (
            "m3q_exact_ssd_dual_complex_bridge"
            if any(arm in M3Q_DUAL_BRIDGE_ARMS for arm in arms)
            else "m3q_operator_lift_with_attention_to_mamba3_homotopy"
            if any(arm in HOMOTOPY_BY_ARM for arm in arms)
            else "operator_preserving_siso_to_mimo_qkvo_lift"
        ),
        "target_layer": layer, "sequence_length": int(train_manifest["sequence_length"]),
        "seeds_requested": list(seeds), "arm_order": list(arms),
        "training": {
            "recovery_steps_per_arm": args.total_steps,
            "checkpoints": list(checkpoints),
            "dual_bridge_steps_per_dual_arm": args.dual_bridge_steps,
            "dual_bridge_learning_rate": args.dual_bridge_learning_rate,
        },
        "compute_dtype": dtype_decision.dtype, "mamba_config": asdict(mamba_config),
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "evaluation_cache_manifest": str(Path(args.evaluation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {k: str(v) for k, v in train_paths.items()},
        "resolved_evaluation_artifacts": {k: str(v) for k, v in eval_paths.items()},
        "seeds": seed_results, "complete": complete, "passed": numerical_pass,
        "notes": [
            "All arms use paired data order, readout calibration, optimizer, and held-out deployable evaluation.",
            "M3Q arms alter only the training-time decoder bridge schedule; evaluation always runs standalone Mamba-3.",
            "M3Q-DUAL arms train the canonical Mamba-3 tree in exact quadratic SSD-dual mode and then switch the same parameters to recurrent execution.",
            "The NO-COMPLEX arm freezes only the Mamba-3 angle projection during bridge preconditioning; recovery remains canonical and trainable.",
            "Single-channel and balanced-rank lifts are algebraically equal before training; optimization geometry is the intervention.",
            "This is a layer-local decoder-aware screen, not full-model NLL evidence.",
        ],
    }
    mirror = _write_json_with_output_mirror(result_path, result, args.output_dir)
    print(f"result_json={result_path.resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    return (result, endpoint_params) if return_endpoint_params else result


if __name__ == "__main__":
    main()
