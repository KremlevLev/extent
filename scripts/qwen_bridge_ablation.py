from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.offline_mamba_distill import _slice
from scripts.qwen_decoder_aware_distill import (
    _evaluate,
    _teacher_decoder_outputs,
)
from scripts.qwen_mamba3_distill_pilot import (
    _replace_output,
    _window_metrics,
    _write_json_with_output_mirror,
)
from extent.attention_bridge import (
    APPLE_ATTENTION_TO_MAMBA_ARXIV,
    MOHAWK_ARXIV,
    AttentionBridgeConfig,
    apply_attention_bridge,
    build_bridge_mamba3_initialization,
    create_bridge_train_step,
    create_orientation_train_step,
    initialize_bridge_params,
    qwen3_attention_components,
    relative_mse,
)
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.hardware import recommended_compute_dtype
from extent.layerwise_distillation import (
    create_decoder_aware_train_step,
    create_homotopy_decoder_train_step,
)
from extent.layers.mamba3 import Mamba3MIMO
from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_mixer_arrays,
)
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.readout_calibration import fit_dual_ridge_readout
from extent.teacher_activation_cache import (
    load_activation_cache,
    validate_external_evaluation_cache,
)
from extent.weight_mapping import QwenCheckpointReader


PROTOCOL = "exp054-attention-bridge-initialization-screen"
ARM_ORDER = (
    "CONTROL-RANDOM",
    "APPLE-BRIDGE",
    "BRIDGE-PLUS-ORIENTATION",
    "MOHAWK-ORIENTATION",
    "CONTROL-QKVO",
)


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _checkpoint_steps(value: str, total_steps: int) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not parsed or parsed[0] != 0 or parsed[-1] != total_steps:
        raise ValueError("checkpoints must include zero and total-steps")
    if any(step < 0 or step > total_steps for step in parsed):
        raise ValueError("checkpoint is outside the recovery schedule")
    return parsed


def _deadline_reached(deadline_monotonic: float | None) -> bool:
    return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic


def homotopy_alpha(step: int, total_steps: int, schedule: str) -> float:
    """Return the registered deployable-Mamba fraction for one recovery step."""
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError("homotopy step is outside the recovery schedule")
    progress = (step + 1) / total_steps
    if schedule == "linear":
        return float(progress)
    if schedule == "cosine":
        return float(0.5 - 0.5 * np.cos(np.pi * progress))
    if schedule == "delayed-cosine":
        # 15% teacher-only decoder path, transition through 70%, then 30%
        # fully deployable training. Mixer matching remains active throughout.
        if progress <= 0.15:
            return 0.0
        if progress >= 0.70:
            return 1.0
        local = (progress - 0.15) / 0.55
        return float(0.5 - 0.5 * np.cos(np.pi * local))
    raise ValueError(f"unknown homotopy schedule: {schedule}")


def _metric_record(metrics: dict) -> dict:
    return {
        key: int(value)
        if key in {"grads_finite", "nonfinite_grad_leaves"}
        else float(value)
        for key, value in metrics.items()
    }


def _bridge_eval(
    bridge_params,
    probe,
    attention_params,
    normalized_inputs,
    positions,
    attention_mask,
    bridge_config,
    batch_windows: int,
) -> dict:
    output_targets = []
    output_predictions = []
    matrix_losses = []
    for start in range(0, len(normalized_inputs), batch_windows):
        stop = min(start + batch_windows, len(normalized_inputs))
        inputs = jnp.asarray(normalized_inputs[start:stop])
        teacher = probe(attention_params, inputs)
        batch_positions = jnp.broadcast_to(positions, (len(inputs), positions.shape[1]))
        batch_mask = jnp.broadcast_to(
            attention_mask, (len(inputs), attention_mask.shape[1])
        )
        bridge = apply_attention_bridge(
            bridge_params,
            teacher,
            batch_positions,
            batch_mask,
            attention_params["o_proj"]["kernel"],
            bridge_config,
        )
        jax.block_until_ready(bridge)
        output_targets.append(np.asarray(teacher.output, np.float32))
        output_predictions.append(np.asarray(bridge.output, np.float32))
        matrix_losses.append(
            float(relative_mse(bridge.matrix, teacher.matrix))
        )
    targets = np.concatenate(output_targets)
    predictions = np.concatenate(output_predictions)
    return {
        "output": _window_metrics(targets, predictions),
        "matrix_relative_mse": float(np.mean(matrix_losses)),
        "finite": bool(np.all(np.isfinite(predictions))),
    }


def _train_bridge(
    *,
    seed: int,
    probe,
    attention_params,
    training_inputs,
    evaluation_inputs,
    positions,
    attention_mask,
    bridge_config: AttentionBridgeConfig,
    steps: int,
    learning_rate: float,
    matrix_loss_weight: float,
    data_seed: int,
    evaluation_batch_windows: int,
    deadline_monotonic: float | None,
) -> tuple[dict, dict, bool]:
    params = initialize_bridge_params(
        jax.random.key(seed + 54_000),
        int(attention_params["q_norm"]["scale"].shape[0]),
        bridge_config,
    )
    tx = create_lion(
        learning_rate=learning_rate,
        warmup_steps=min(8, max(1, steps // 10)),
        total_steps=steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    train_step = create_bridge_train_step(
        tx, bridge_config, matrix_loss_weight=matrix_loss_weight
    )
    opt_state = tx.init(params)
    before = _bridge_eval(
        params,
        probe,
        attention_params,
        evaluation_inputs,
        positions,
        attention_mask,
        bridge_config,
        evaluation_batch_windows,
    )
    finite = bool(before["finite"])
    max_grad_norm = 0.0
    completed = 0
    for zero_based_step in range(steps):
        if zero_based_step % 8 == 0 and _deadline_reached(deadline_monotonic):
            break
        index = int(
            deterministic_batch_indices(
                zero_based_step, 1, len(training_inputs), data_seed
            )[0]
        )
        inputs = jnp.asarray(training_inputs[index : index + 1])
        teacher = probe(attention_params, inputs)
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            teacher,
            positions,
            attention_mask,
            attention_params["o_proj"]["kernel"],
        )
        jax.block_until_ready(metrics)
        record = _metric_record(metrics)
        max_grad_norm = max(max_grad_norm, record["grad_norm"])
        finite = finite and bool(record["grads_finite"]) and np.isfinite(
            record["loss"]
        )
        completed = zero_based_step + 1
    after = _bridge_eval(
        params,
        probe,
        attention_params,
        evaluation_inputs,
        positions,
        attention_mask,
        bridge_config,
        evaluation_batch_windows,
    )
    finite = finite and bool(after["finite"])
    result = {
        "requested_steps": steps,
        "completed_steps": completed,
        "learning_rate": learning_rate,
        "matrix_loss_weight": matrix_loss_weight,
        "before": before,
        "after": after,
        "max_grad_norm": max_grad_norm,
        "finite": finite,
        "complete": completed == steps,
    }
    return params, result, completed == steps


def _train_orientation(
    *,
    params,
    probe,
    attention_params,
    training_inputs,
    mamba_config,
    steps: int,
    learning_rate: float,
    data_seed: int,
    bf16_gradients: bool,
    deadline_monotonic: float | None,
) -> tuple[dict, dict, bool]:
    tx = create_lion(
        learning_rate=learning_rate,
        warmup_steps=min(8, max(1, steps // 10)),
        total_steps=steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    train_step = create_orientation_train_step(
        tx, mamba_config, bf16_gradients=bf16_gradients
    )
    opt_state = tx.init(params)
    finite = True
    max_grad_norm = 0.0
    first_loss = None
    last_loss = None
    completed = 0
    for zero_based_step in range(steps):
        if zero_based_step % 8 == 0 and _deadline_reached(deadline_monotonic):
            break
        index = int(
            deterministic_batch_indices(
                zero_based_step, 1, len(training_inputs), data_seed
            )[0]
        )
        inputs = jnp.asarray(training_inputs[index : index + 1])
        teacher = probe(attention_params, inputs)
        params, opt_state, metrics = train_step(
            params, opt_state, inputs, teacher.matrix
        )
        jax.block_until_ready(metrics)
        record = _metric_record(metrics)
        first_loss = record["loss"] if first_loss is None else first_loss
        last_loss = record["loss"]
        max_grad_norm = max(max_grad_norm, record["grad_norm"])
        finite = finite and bool(record["grads_finite"]) and np.isfinite(
            record["loss"]
        )
        completed = zero_based_step + 1
    return params, {
        "requested_steps": steps,
        "completed_steps": completed,
        "learning_rate": learning_rate,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "max_grad_norm": max_grad_norm,
        "finite": finite,
        "complete": completed == steps,
        "scope": "head-averaged Mamba-3 matrix; value gate, D skip and output projection excluded",
    }, completed == steps


def _calibrate_readout(
    mamba,
    params,
    calibration_inputs,
    calibration_targets,
    compute_dtype,
    relative_ridge: float,
):
    run_features = jax.jit(
        lambda candidate, inputs: mamba.apply(
            {"params": candidate}, inputs, return_features=True
        )[1]
    )
    features = np.asarray(
        run_features(params, jnp.asarray(calibration_inputs, dtype=compute_dtype)),
        np.float32,
    )
    readout, report = fit_dual_ridge_readout(
        jnp.asarray(features.reshape(-1, features.shape[-1])),
        jnp.asarray(calibration_targets.reshape(-1, calibration_targets.shape[-1])),
        relative_ridge=relative_ridge,
    )
    return _replace_output(params, readout, compute_dtype), asdict(report)


def _train_recovery(
    *,
    params,
    mamba,
    tail,
    tail_params,
    training_arrays,
    training_slice,
    evaluation_arrays,
    evaluation_slice,
    compute_dtype,
    total_steps: int,
    checkpoints: tuple[int, ...],
    learning_rate: float,
    decoder_loss_weight: float,
    data_seed: int,
    evaluation_batch_windows: int,
    deadline_monotonic: float | None,
    homotopy_schedule: str = "none",
) -> tuple[dict, dict, bool]:
    apply_mixer = lambda candidate, inputs: mamba.apply(
        {"params": candidate}, inputs
    )
    apply_tail = lambda candidate, residual, mixer_output: tail.apply(
        {"params": candidate}, residual, mixer_output
    )
    tx = create_lion(
        learning_rate=learning_rate,
        warmup_steps=min(64, max(1, total_steps // 100)),
        total_steps=total_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    if homotopy_schedule == "none":
        train_step = create_decoder_aware_train_step(
            apply_mixer,
            apply_tail,
            tx,
            decoder_loss_weight=decoder_loss_weight,
            bf16_gradients=compute_dtype == jnp.bfloat16,
        )
    else:
        # Validate before compilation so protocol mistakes fail cheaply.
        homotopy_alpha(0, total_steps, homotopy_schedule)
        train_step = create_homotopy_decoder_train_step(
            apply_mixer,
            apply_tail,
            tx,
            decoder_loss_weight=decoder_loss_weight,
            bf16_gradients=compute_dtype == jnp.bfloat16,
        )
    opt_state = tx.init(params)
    teacher_runner = create_batched_teacher_tail_runner(tail)
    replacement_runner = create_batched_replacement_runner(mamba, tail)
    residual_eval = jnp.asarray(
        evaluation_arrays["residual_input"][evaluation_slice], dtype=compute_dtype
    )
    normalized_eval = jnp.asarray(
        evaluation_arrays["normalized_input"][evaluation_slice], dtype=compute_dtype
    )
    mixer_eval = jnp.asarray(
        evaluation_arrays["attention_target"][evaluation_slice], dtype=compute_dtype
    )
    mixer_eval_np = np.asarray(mixer_eval, np.float32)
    decoder_eval_np = _teacher_decoder_outputs(
        teacher_runner,
        tail_params,
        residual_eval,
        mixer_eval,
        evaluation_batch_windows,
    )
    evaluations = {
        "0": _evaluate(
            replacement_runner,
            teacher_runner,
            params,
            tail_params,
            residual_eval,
            normalized_eval,
            mixer_eval_np,
            decoder_eval_np,
            evaluation_batch_windows,
        )
    }
    finite = bool(evaluations["0"]["finite"])
    max_grad_norm = 0.0
    last_train_metrics = None
    completed = 0
    training_count = training_slice.stop - training_slice.start
    for zero_based_step in range(total_steps):
        if zero_based_step % 8 == 0 and _deadline_reached(deadline_monotonic):
            break
        index = int(
            deterministic_batch_indices(
                zero_based_step, 1, training_count, data_seed
            )[0]
        )
        cache_index = training_slice.start + index
        step_arguments = (
            params,
            opt_state,
            tail_params,
            jnp.asarray(
                training_arrays["residual_input"][cache_index : cache_index + 1],
                dtype=compute_dtype,
            ),
            jnp.asarray(
                training_arrays["normalized_input"][cache_index : cache_index + 1],
                dtype=compute_dtype,
            ),
            jnp.asarray(
                training_arrays["attention_target"][cache_index : cache_index + 1],
                dtype=compute_dtype,
            ),
        )
        if homotopy_schedule == "none":
            params, opt_state, metrics = train_step(*step_arguments)
        else:
            alpha = homotopy_alpha(
                zero_based_step, total_steps, homotopy_schedule
            )
            params, opt_state, metrics = train_step(*step_arguments, alpha)
        jax.block_until_ready(metrics)
        last_train_metrics = _metric_record(metrics)
        max_grad_norm = max(max_grad_norm, last_train_metrics["grad_norm"])
        finite = finite and bool(last_train_metrics["grads_finite"]) and np.isfinite(
            last_train_metrics["loss"]
        )
        completed = zero_based_step + 1
        if completed in checkpoints:
            evaluations[str(completed)] = _evaluate(
                replacement_runner,
                teacher_runner,
                params,
                tail_params,
                residual_eval,
                normalized_eval,
                mixer_eval_np,
                decoder_eval_np,
                evaluation_batch_windows,
            )
            finite = finite and bool(evaluations[str(completed)]["finite"])
            print(
                f"recovery_step={completed} "
                f"decoder_l2={evaluations[str(completed)]['decoder_output']['relative_l2']:.6g}"
            )
    if str(completed) not in evaluations:
        evaluations[str(completed)] = _evaluate(
            replacement_runner,
            teacher_runner,
            params,
            tail_params,
            residual_eval,
            normalized_eval,
            mixer_eval_np,
            decoder_eval_np,
            evaluation_batch_windows,
        )
    return params, {
        "requested_steps": total_steps,
        "completed_steps": completed,
        "evaluations": evaluations,
        "last_train_metrics": last_train_metrics,
        "max_grad_norm": max_grad_norm,
        "homotopy_schedule": homotopy_schedule,
        "finite": finite,
        "complete": completed == total_steps,
    }, completed == total_steps


def aggregate_bridge_screen(
    layer_results: dict[str, dict],
    *,
    primary_arm: str = "BRIDGE-PLUS-ORIENTATION",
) -> dict:
    """Aggregate only complete paired cells; never reinterpret partial runs."""
    if primary_arm == "CONTROL-RANDOM" or primary_arm not in ARM_ORDER:
        raise ValueError("primary_arm must be a non-random registered arm")
    comparisons = {}
    screening_gate = True
    complete_layers = 0
    for layer, result in sorted(layer_results.items(), key=lambda item: int(item[0])):
        by_arm = {arm: [] for arm in ARM_ORDER}
        for seed_result in result.get("seeds", {}).values():
            for arm in ARM_ORDER:
                recovery = seed_result.get("arms", {}).get(arm, {}).get("recovery")
                if not recovery or not recovery.get("complete"):
                    continue
                final = recovery["evaluations"][str(recovery["completed_steps"])]
                by_arm[arm].append(
                    float(final["decoder_output"]["relative_l2"])
                )
        random_values = by_arm["CONTROL-RANDOM"]
        layer_comparisons = {}
        if len(random_values) == len(result.get("seeds", {})) and random_values:
            complete_layers += 1
            random_array = np.asarray(random_values)
            for arm in ARM_ORDER[1:]:
                values = by_arm[arm]
                if len(values) != len(random_values):
                    continue
                array = np.asarray(values)
                difference = array - random_array
                layer_comparisons[arm] = {
                    "decoder_relative_l2_mean": float(array.mean()),
                    "random_relative_l2_mean": float(random_array.mean()),
                    "arm_minus_random_mean": float(difference.mean()),
                    "wins_over_random": int(np.sum(difference < 0)),
                    "paired_seeds": len(values),
                }
            combined = layer_comparisons.get(primary_arm)
            screening_gate = screening_gate and bool(
                combined
                and combined["wins_over_random"] >= 2
                and combined["arm_minus_random_mean"] < 0
            )
        else:
            screening_gate = False
        comparisons[layer] = layer_comparisons
    screening_gate = bool(
        screening_gate and complete_layers == len(layer_results) and complete_layers > 0
    )
    return {
        "layers": comparisons,
        "complete_layers": complete_layers,
        "screening_gate_passed": screening_gate,
        "gate_definition": (
            f"{primary_arm} beats CONTROL-RANDOM in at least 2/3 paired seeds "
            "and in mean final decoder relative-L2 at every completed target layer"
        ),
        "primary_arm": primary_arm,
    }


def main(
    argv: list[str] | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> dict:
    parser = argparse.ArgumentParser(
        description="Run one Qwen3-to-Mamba3 bridge initialization screen."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--evaluation-cache-manifest", required=True)
    parser.add_argument("--evaluation-cache-dir")
    parser.add_argument("--qwen-cache-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--checkpoints", default="0,256,512,1024")
    parser.add_argument("--bridge-steps", type=int, default=128)
    parser.add_argument("--orientation-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--bridge-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bridge-rope-fraction", type=float, default=1.0)
    parser.add_argument("--orientation-learning-rate", type=float, default=3e-5)
    parser.add_argument("--bridge-matrix-loss-weight", type=float, default=0.1)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--data-seed", type=int, default=20260827)
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-protocol", default=PROTOCOL)
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args(argv)
    log_prefix = args.experiment_protocol.split("-", 1)[0]
    checkpoints = _checkpoint_steps(args.checkpoints, args.total_steps)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if len(seeds) != len(set(seeds)) or not seeds or min(seeds) < 0:
        raise ValueError("seeds must be unique non-negative integers")
    if min(
        args.total_steps,
        args.bridge_steps,
        args.orientation_steps,
        args.evaluation_batch_windows,
    ) < 1:
        raise ValueError("step and batch counts must be positive")

    train_manifest, train_arrays, train_paths = load_activation_cache(
        args.activation_cache_manifest,
        artifact_dir=args.activation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    eval_manifest, eval_arrays, eval_paths = load_activation_cache(
        args.evaluation_cache_manifest,
        artifact_dir=args.evaluation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    evaluation_slice = validate_external_evaluation_cache(
        train_manifest,
        eval_manifest,
        allow_cross_split=True,
        required_dataset_split="validation",
    )
    layout = train_manifest["window_layout"]
    calibration_slice = _slice(layout, "calibration")
    training_slice = _slice(layout, "training")
    training_count = training_slice.stop - training_slice.start
    if training_count != args.total_steps:
        raise ValueError("bridge screen requires one recovery window per requested step")
    layer = int(train_manifest["target_layer"])
    if int(eval_manifest["target_layer"]) != layer:
        raise ValueError("training and evaluation caches target different layers")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba_config = Mamba3Config()
    bridge_config = AttentionBridgeConfig(
        feature_dim=mamba_config.d_state,
        rope_fraction=args.bridge_rope_fraction,
    )
    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(QWEN3_14B.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir,
        index_payload["weight_map"],
        source,
        layer,
        repo_id=QWEN3_14B.repo_id,
        revision=QWEN3_14B.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    mixer_arrays = load_mixer_arrays(reader, source, layer)
    attention_params = jax_attention_params(mixer_arrays, source, layer)
    tail_params = qwen3_decoder_tail_params(reader, layer)
    sequence_length = int(train_manifest["sequence_length"])
    positions = jnp.arange(sequence_length, dtype=jnp.int32)[None]
    attention_mask = jnp.ones((1, sequence_length), dtype=jnp.bool_)

    @jax.jit
    def probe(params, inputs):
        batch_positions = jnp.broadcast_to(positions, (inputs.shape[0], sequence_length))
        batch_mask = jnp.broadcast_to(
            attention_mask, (inputs.shape[0], sequence_length)
        )
        return qwen3_attention_components(
            params, inputs, batch_positions, source, batch_mask
        )

    mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=compute_dtype,
        param_dtype=compute_dtype,
    )
    tail = Qwen3DecoderTail(
        source.hidden_size,
        source.intermediate_size,
        source.rms_norm_eps,
        compute_dtype,
        jnp.float32,
    )
    calibration_inputs = np.asarray(
        train_arrays["normalized_input"][calibration_slice]
    )
    calibration_targets = np.asarray(
        train_arrays["attention_target"][calibration_slice], np.float32
    )
    training_inputs = train_arrays["normalized_input"][training_slice]
    bridge_evaluation_inputs = eval_arrays["normalized_input"][evaluation_slice]
    result_path = Path(args.result_json)
    partial_path = result_path.with_suffix(".partial.json")
    seed_results = {}
    complete = True
    numerical_pass = True

    def persist(status: str) -> None:
        payload = {
            "protocol": args.experiment_protocol,
            "status": status,
            "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
            "target_layer": layer,
            "seeds": seed_results,
            "complete": complete,
            "passed": numerical_pass,
        }
        _write_json_with_output_mirror(partial_path, payload, args.output_dir)

    for seed in seeds:
        if _deadline_reached(deadline_monotonic):
            complete = False
            break
        print(f"{log_prefix}_layer={layer} seed={seed} bridge_stage=START")
        bridge_params, bridge_result, bridge_complete = _train_bridge(
            seed=seed,
            probe=probe,
            attention_params=attention_params,
            training_inputs=training_inputs,
            evaluation_inputs=bridge_evaluation_inputs,
            positions=positions,
            attention_mask=attention_mask,
            bridge_config=bridge_config,
            steps=args.bridge_steps,
            learning_rate=args.bridge_learning_rate,
            matrix_loss_weight=args.bridge_matrix_loss_weight,
            data_seed=args.data_seed + seed,
            evaluation_batch_windows=args.evaluation_batch_windows,
            deadline_monotonic=deadline_monotonic,
        )
        seed_record = {"bridge_stage": bridge_result, "arms": {}}
        seed_results[str(seed)] = seed_record
        numerical_pass = numerical_pass and bool(bridge_result["finite"])
        persist("in_progress" if bridge_complete else "deadline_partial")
        if not bridge_complete:
            complete = False
            break

        base = mamba.init(
            jax.random.key(seed),
            jnp.asarray(calibration_inputs[:1], dtype=compute_dtype),
        )["params"]
        mapped, _ = build_qwen3_to_mamba3_transplant_variants(
            base, mixer_arrays, source, mamba_config, layer
        )
        apple, apple_report = build_bridge_mamba3_initialization(
            base,
            mixer_arrays,
            bridge_params,
            source,
            mamba_config,
            layer,
        )
        mohawk, mohawk_result, mohawk_complete = _train_orientation(
            params=base,
            probe=probe,
            attention_params=attention_params,
            training_inputs=training_inputs,
            mamba_config=mamba_config,
            steps=args.orientation_steps,
            learning_rate=args.orientation_learning_rate,
            data_seed=args.data_seed + seed + 1_000,
            bf16_gradients=compute_dtype == jnp.bfloat16,
            deadline_monotonic=deadline_monotonic,
        )
        seed_record["orientation_stages"] = {
            "MOHAWK-ORIENTATION": mohawk_result,
        }
        numerical_pass = numerical_pass and bool(mohawk_result["finite"])
        persist("in_progress" if mohawk_complete else "deadline_partial")
        if not mohawk_complete:
            complete = False
            break
        combined, combined_result, combined_complete = _train_orientation(
            params=apple,
            probe=probe,
            attention_params=attention_params,
            training_inputs=training_inputs,
            mamba_config=mamba_config,
            steps=args.orientation_steps,
            learning_rate=args.orientation_learning_rate,
            data_seed=args.data_seed + seed + 2_000,
            bf16_gradients=compute_dtype == jnp.bfloat16,
            deadline_monotonic=deadline_monotonic,
        )
        seed_record["orientation_stages"][
            "BRIDGE-PLUS-ORIENTATION"
        ] = combined_result
        numerical_pass = numerical_pass and bool(combined_result["finite"])
        persist("in_progress" if combined_complete else "deadline_partial")
        if not combined_complete:
            complete = False
            break
        initializers = {
            "CONTROL-RANDOM": (base, {"kind": "canonical_random"}),
            "APPLE-BRIDGE": (apple, asdict(apple_report)),
            "BRIDGE-PLUS-ORIENTATION": (
                combined,
                {
                    "kind": "apple_bridge_then_mohawk_orientation",
                    "apple": asdict(apple_report),
                    "orientation": combined_result,
                },
            ),
            "MOHAWK-ORIENTATION": (
                mohawk,
                {
                    "kind": "mohawk_inspired_head_averaged_orientation",
                    "source_arxiv": MOHAWK_ARXIV,
                    "orientation": mohawk_result,
                },
            ),
            "CONTROL-QKVO": (
                mapped["INIT-C-prior-qkvo-port"],
                {"kind": "rejected_direct_qkvo_control"},
            ),
        }
        for arm in ARM_ORDER:
            if _deadline_reached(deadline_monotonic):
                complete = False
                break
            print(f"{log_prefix}_layer={layer} seed={seed} arm={arm} recovery=START")
            initial, initializer_report = initializers[arm]
            calibrated, readout_report = _calibrate_readout(
                mamba,
                initial,
                calibration_inputs,
                calibration_targets,
                compute_dtype,
                args.readout_ridge,
            )
            _, recovery, recovery_complete = _train_recovery(
                params=calibrated,
                mamba=mamba,
                tail=tail,
                tail_params=tail_params,
                training_arrays=train_arrays,
                training_slice=training_slice,
                evaluation_arrays=eval_arrays,
                evaluation_slice=evaluation_slice,
                compute_dtype=compute_dtype,
                total_steps=args.total_steps,
                checkpoints=checkpoints,
                learning_rate=args.learning_rate,
                decoder_loss_weight=args.decoder_loss_weight,
                data_seed=args.data_seed + seed,
                evaluation_batch_windows=args.evaluation_batch_windows,
                deadline_monotonic=deadline_monotonic,
            )
            seed_record["arms"][arm] = {
                "initializer": initializer_report,
                "readout_calibration": readout_report,
                "recovery": recovery,
            }
            numerical_pass = numerical_pass and bool(recovery["finite"])
            persist("in_progress" if recovery_complete else "deadline_partial")
            if not recovery_complete:
                complete = False
                break
        if not complete:
            break
        jax.clear_caches()

    result = {
        "protocol": args.experiment_protocol,
        "status": "completed" if complete else "deadline_partial",
        "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
        "method": "apple_bridge_and_mohawk_orientation_initialization_screen",
        "target_layer": layer,
        "sequence_length": sequence_length,
        "seeds_requested": list(seeds),
        "arm_order": list(ARM_ORDER),
        "training": {
            "bridge_steps_per_seed": args.bridge_steps,
            "orientation_steps_per_oriented_arm": args.orientation_steps,
            "recovery_steps_per_arm": args.total_steps,
            "recovery_checkpoints": list(checkpoints),
            "unique_recovery_tokens_per_arm": args.total_steps * sequence_length,
            "initializer_tokens_are_reported_separately": True,
        },
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "mamba_config": asdict(mamba_config),
        "bridge_config": asdict(bridge_config),
        "source_methods": {
            "apple_arxiv": APPLE_ATTENTION_TO_MAMBA_ARXIV,
            "mohawk_arxiv": MOHAWK_ARXIV,
        },
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "evaluation_cache_manifest": str(Path(args.evaluation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {
            name: str(path) for name, path in train_paths.items()
        },
        "resolved_evaluation_artifacts": {
            name: str(path) for name, path in eval_paths.items()
        },
        "seeds": seed_results,
        "complete": complete,
        "passed": numerical_pass,
        "notes": [
            f"All five arms receive the same {args.total_steps}-step decoder-aware recovery schedule and paired data order.",
            "Bridge and orientation construction cost is recorded separately instead of hidden inside recovery tokens.",
            "APPLE-BRIDGE folds learned pre-softmax Hedgehog features into canonical Mamba-3; it is not exact HedgeMamba equivalence.",
            "MOHAWK-ORIENTATION matches a head-averaged Mamba-3 matrix proxy because Qwen and Mamba use unequal head counts.",
            "This is an initialization screen, not end-to-end 14B recovery evidence.",
        ],
    }
    mirror = _write_json_with_output_mirror(result_path, result, args.output_dir)
    print(f"result_json={result_path.resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    label = log_prefix.upper()
    print(f"{label}-LAYER-COMPLETE" if complete else f"{label}-LAYER-DEADLINE-PARTIAL")
    return result


if __name__ == "__main__":
    main()
