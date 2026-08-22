from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.offline_mamba_distill import _slice
from scripts.qwen_activation_cache import (
    _create_data_parallel_decoder_runner,
    _create_decoder_runner,
    _safe_prune_shards,
)
from scripts.qwen_decoder_aware_distill import (
    _teacher_decoder_outputs,
    main as train_replacements,
)
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.hardware import recommended_compute_dtype
from extent.layers.common import RMSNorm
from extent.layers.mamba3 import Mamba3MIMO
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
)
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.streamed_lm_eval import (
    create_lm_metrics_runner,
    end_to_end_loss_comparison,
    full_model_shard_last_use,
    hidden_relative_l2,
)
from extent.teacher_activation_cache import (
    load_activation_cache,
    run_host_data_parallel,
    run_host_microbatches,
)
from extent.weight_mapping import QwenCheckpointReader


BRANCHES = (
    "ORIGINAL-CACHED-QWEN",
    "CALIBRATED-STEP0",
    "MIXER-ONLY-STEP1024",
    "JOINT-STEP1024",
)


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_replacement_windows(
    runner,
    params,
    tail_params,
    residual_inputs,
    normalized_inputs,
    batch_windows: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(residual_inputs), batch_windows):
        stop = min(start + batch_windows, len(residual_inputs))
        _, decoder = runner(
            params,
            tail_params,
            residual_inputs[start:stop],
            normalized_inputs[start:stop],
        )
        outputs.append(np.asarray(decoder, dtype=np.float32))
    return np.concatenate(outputs)


def _branch_divergence(hidden: np.ndarray, windows: int) -> dict:
    values = hidden.reshape(len(BRANCHES), windows, *hidden.shape[1:])
    reference = values[0]
    return {
        name: hidden_relative_l2(reference, values[index])
        for index, name in enumerate(BRANCHES)
    }


def _run_lm_metrics(
    runner,
    norm_params,
    lm_head_kernel,
    hidden: np.ndarray,
    tokens: np.ndarray,
    *,
    devices,
    per_device_windows: int,
    compute_dtype,
    return_window_nll: bool = False,
) -> dict:
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    window_nll = []
    if devices:
        device_count = len(devices)
        global_batch = device_count * per_device_windows
        for start in range(0, len(hidden), global_batch):
            stop = min(start + global_batch, len(hidden))
            valid = stop - start
            host_hidden = np.zeros(
                (global_batch, *hidden.shape[1:]), dtype=np.dtype(compute_dtype)
            )
            host_tokens = np.zeros(
                (global_batch, tokens.shape[1]), dtype=np.int32
            )
            host_valid = np.zeros((global_batch,), dtype=np.bool_)
            host_hidden[:valid] = np.asarray(
                hidden[start:stop], dtype=np.dtype(compute_dtype)
            )
            host_tokens[:valid] = tokens[start:stop]
            host_valid[:valid] = True
            shaped_hidden = jnp.asarray(host_hidden).reshape(
                device_count, per_device_windows, *hidden.shape[1:]
            )
            shaped_tokens = jnp.asarray(host_tokens).reshape(
                device_count, per_device_windows, tokens.shape[1]
            )
            shaped_valid = jnp.asarray(host_valid).reshape(
                device_count, per_device_windows
            )
            loss, count, correct = runner(
                norm_params,
                lm_head_kernel,
                shaped_hidden,
                shaped_tokens,
                shaped_valid,
            )
            jax.block_until_ready((loss, count, correct))
            loss_values = np.asarray(loss, dtype=np.float64)
            count_values = np.asarray(count, dtype=np.int64)
            correct_values = np.asarray(correct, dtype=np.int64)
            total_loss += float(np.sum(loss_values))
            total_tokens += int(np.sum(count_values))
            total_correct += int(np.sum(correct_values))
            if return_window_nll:
                valid_losses = loss_values.reshape(-1)[:valid]
                valid_counts = count_values.reshape(-1)[:valid]
                window_nll.extend((valid_losses / valid_counts).tolist())
    else:
        for start in range(0, len(hidden), per_device_windows):
            stop = min(start + per_device_windows, len(hidden))
            batch_hidden = jnp.asarray(hidden[start:stop], dtype=compute_dtype)
            batch_tokens = jnp.asarray(tokens[start:stop], dtype=jnp.int32)
            valid = jnp.ones((stop - start,), dtype=jnp.bool_)
            loss, count, correct = runner(
                norm_params, lm_head_kernel, batch_hidden, batch_tokens, valid
            )
            jax.block_until_ready((loss, count, correct))
            loss_values = np.asarray(loss, dtype=np.float64)
            count_values = np.asarray(count, dtype=np.int64)
            correct_values = np.asarray(correct, dtype=np.int64)
            total_loss += float(np.sum(loss_values))
            total_tokens += int(np.sum(count_values))
            total_correct += int(np.sum(correct_values))
            if return_window_nll:
                window_nll.extend((loss_values / count_values).tolist())
    mean_nll = total_loss / total_tokens
    result = {
        "nll_sum": total_loss,
        "token_count": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
        "top1_accuracy": total_correct / total_tokens,
        "top1_correct": total_correct,
        "finite": bool(np.isfinite(mean_nll)),
    }
    if return_window_nll:
        if len(window_nll) != len(hidden):
            raise ValueError("per-window NLL count does not match evaluated windows")
        result["window_mean_nll"] = window_nll
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Stream Qwen3 after a layer-0 Mamba replacement and measure end-to-end NLL shock."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--qwen-cache-dir", default="/kaggle/working/qwen3-e2e-weights")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--training-checkpoints", default="0,512,1024")
    parser.add_argument("--evaluation-batch-windows", type=int, default=1)
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--compute-dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--prune-consumed-shards", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    args = parser.parse_args(argv)
    if min(
        args.total_steps,
        args.batch_windows,
        args.evaluation_batch_windows,
        args.per_device_windows,
    ) < 1:
        raise ValueError("step and batch settings must be positive")
    if args.total_steps != 1024:
        raise ValueError("EXP-040 branch names and frozen protocol require 1024 steps")

    jax.config.update("jax_default_matmul_precision", "high")
    manifest, arrays, paths = load_activation_cache(
        args.activation_cache_manifest,
        artifact_dir=args.activation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if manifest.get("source") != source_name or int(manifest["target_layer"]) != 0:
        raise ValueError("EXP-040 requires the pinned Qwen3 source and target layer zero")
    layout = manifest["window_layout"]
    evaluation_slice = _slice(layout, "evaluation")
    evaluation_windows = evaluation_slice.stop - evaluation_slice.start
    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    devices = list(jax.devices())
    if args.data_parallel and len(devices) < 2:
        raise ValueError("--data-parallel requires at least two visible devices")
    parallel_devices = devices if args.data_parallel else None

    training_json = Path(args.output_dir) / "exp040-layer0-training.json"
    training_args = [
        "--activation-cache-manifest", args.activation_cache_manifest,
        "--qwen-cache-dir", args.qwen_cache_dir,
        "--total-steps", str(args.total_steps),
        "--checkpoints", args.training_checkpoints,
        "--batch-windows", str(args.batch_windows),
        "--evaluation-batch-windows", str(args.evaluation_batch_windows),
        "--learning-rate", str(args.learning_rate),
        "--readout-ridge", str(args.readout_ridge),
        "--decoder-loss-weight", str(args.decoder_loss_weight),
        "--seed", str(args.seed),
        "--data-seed", str(args.data_seed),
        "--compute-dtype", args.compute_dtype,
        "--result-json", str(training_json),
        "--output-dir", args.output_dir,
    ]
    if args.activation_cache_dir:
        training_args.extend(["--activation-cache-dir", args.activation_cache_dir])
    if args.skip_hash_verification:
        training_args.append("--skip-hash-verification")
    training_result, calibrated_params, endpoint_params = train_replacements(
        training_args, return_endpoint_params=True
    )

    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(QWEN3_14B.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba = Mamba3MIMO(
        source.hidden_size,
        Mamba3Config(),
        dtype=compute_dtype,
        param_dtype=compute_dtype,
    )
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir,
        index_payload["weight_map"],
        source,
        0,
        repo_id=QWEN3_14B.repo_id,
        revision=QWEN3_14B.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    tail_params = qwen3_decoder_tail_params(reader, 0)
    tail = Qwen3DecoderTail(
        source.hidden_size,
        source.intermediate_size,
        source.rms_norm_eps,
        compute_dtype,
        jnp.float32,
    )
    teacher_tail_runner = create_batched_teacher_tail_runner(tail)
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
    original_layer0 = _teacher_decoder_outputs(
        teacher_tail_runner,
        tail_params,
        residual_eval,
        mixer_eval,
        args.evaluation_batch_windows,
    )
    calibrated_layer0 = _run_replacement_windows(
        replacement_runner,
        calibrated_params,
        tail_params,
        residual_eval,
        normalized_eval,
        args.evaluation_batch_windows,
    )
    mixer_layer0 = _run_replacement_windows(
        replacement_runner,
        endpoint_params["MIXER-ONLY"],
        tail_params,
        residual_eval,
        normalized_eval,
        args.evaluation_batch_windows,
    )
    joint_layer0 = _run_replacement_windows(
        replacement_runner,
        endpoint_params["JOINT-MIXER-DECODER"],
        tail_params,
        residual_eval,
        normalized_eval,
        args.evaluation_batch_windows,
    )
    hidden = np.concatenate(
        (original_layer0, calibrated_layer0, mixer_layer0, joint_layer0), axis=0
    )
    divergence = {"0": _branch_divergence(hidden, evaluation_windows)}
    del (
        tail_params,
        calibrated_params,
        endpoint_params,
        residual_eval,
        normalized_eval,
        mixer_eval,
        original_layer0,
        calibrated_layer0,
        mixer_layer0,
        joint_layer0,
        mamba,
        tail,
        teacher_tail_runner,
        replacement_runner,
    )
    jax.clear_caches()
    gc.collect()

    last_use = full_model_shard_last_use(
        index_payload["weight_map"], source.num_layers
    )
    removed_shards = []
    if args.prune_consumed_shards:
        removed_shards.extend(_safe_prune_shards(model_dir, last_use, 0))
    positions = jnp.arange(int(manifest["sequence_length"]), dtype=jnp.int32)[None]
    attention_mask = jnp.ones(
        (1, int(manifest["sequence_length"])), dtype=jnp.bool_
    )
    decoder = Qwen3DecoderLayer(source)
    layer_runner = (
        _create_data_parallel_decoder_runner(
            decoder, positions, attention_mask, devices
        )
        if args.data_parallel
        else _create_decoder_runner(decoder, positions, attention_mask)
    )
    divergence_layers = {9, 19, 29, 39}
    for layer_index in range(1, source.num_layers):
        print(f"streaming_decoder_layer={layer_index}/{source.num_layers - 1}")
        ensure_layer_checkpoint(
            model_dir,
            index_payload["weight_map"],
            source,
            layer_index,
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
        )
        reader = QwenCheckpointReader(model_dir)
        layer_arrays = load_layer_arrays(reader, source, layer_index)
        layer_params = jax_layer_params(layer_arrays, source, layer_index)
        hidden = (
            run_host_data_parallel(
                layer_runner,
                layer_params,
                hidden,
                args.per_device_windows,
                len(devices),
                input_dtype=compute_dtype,
            )
            if args.data_parallel
            else run_host_microbatches(
                layer_runner,
                layer_params,
                hidden,
                args.evaluation_batch_windows,
                input_dtype=compute_dtype,
            )
        )
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError(
                f"non-finite streamed residual after layer {layer_index}"
            )
        if layer_index in divergence_layers:
            divergence[str(layer_index)] = _branch_divergence(
                hidden, evaluation_windows
            )
        del layer_params, layer_arrays, reader
        gc.collect()
        if args.prune_consumed_shards:
            removed_shards.extend(
                _safe_prune_shards(model_dir, last_use, layer_index)
            )

    from huggingface_hub import hf_hub_download

    for tensor_name in ("model.norm.weight", "lm_head.weight"):
        hf_hub_download(
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
            filename=index_payload["weight_map"][tensor_name],
            local_dir=model_dir,
        )
    reader = QwenCheckpointReader(model_dir)
    norm_params = {
        "scale": jnp.asarray(reader.read("model.norm.weight"), dtype=jnp.float32)
    }
    lm_head_kernel = jnp.asarray(
        reader.read("lm_head.weight").T, dtype=compute_dtype
    )
    norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    lm_runner = create_lm_metrics_runner(
        norm, data_parallel_devices=parallel_devices
    )
    token_ids = np.asarray(arrays["token_ids"][evaluation_slice], dtype=np.int32)
    branch_values = hidden.reshape(
        len(BRANCHES), evaluation_windows, *hidden.shape[1:]
    )
    lm_metrics = {}
    for branch_index, branch_name in enumerate(BRANCHES):
        print(f"evaluating_lm_head={branch_name}")
        lm_metrics[branch_name] = _run_lm_metrics(
            lm_runner,
            norm_params,
            lm_head_kernel,
            branch_values[branch_index],
            token_ids,
            devices=parallel_devices,
            per_device_windows=args.per_device_windows,
            compute_dtype=compute_dtype,
        )
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, source.num_layers)
        )

    original_nll = lm_metrics["ORIGINAL-CACHED-QWEN"]["mean_nll"]
    calibrated_nll = lm_metrics["CALIBRATED-STEP0"]["mean_nll"]
    mixer_nll = lm_metrics["MIXER-ONLY-STEP1024"]["mean_nll"]
    joint_nll = lm_metrics["JOINT-STEP1024"]["mean_nll"]
    all_finite = all(metrics["finite"] for metrics in lm_metrics.values())
    comparison = end_to_end_loss_comparison(
        original_nll=original_nll,
        calibrated_nll=calibrated_nll,
        mixer_only_nll=mixer_nll,
        joint_nll=joint_nll,
        all_finite=all_finite,
    )
    scientific_gate_passed = comparison["scientific_gate_passed"]
    result = {
        "source": source_name,
        "dataset": manifest.get("dataset"),
        "method": "streamed_end_to_end_Qwen3_layer0_Mamba_loss_shock",
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {name: str(path) for name, path in paths.items()},
        "target_layer": 0,
        "sequence_length": int(manifest["sequence_length"]),
        "evaluation_windows": evaluation_windows,
        "evaluation_tokens": evaluation_windows
        * (int(manifest["sequence_length"]) - 1),
        "branches": list(BRANCHES),
        "seed": args.seed,
        "training_result": training_result,
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "data_parallel": args.data_parallel,
        "per_device_windows": args.per_device_windows,
        "hidden_relative_l2_to_original": divergence,
        "lm_metrics": lm_metrics,
        "comparison": comparison,
        "prune_consumed_shards": args.prune_consumed_shards,
        "removed_checkpoint_shards": sorted(set(removed_shards)),
        "scientific_gate_passed": scientific_gate_passed,
        "passed": all_finite,
        "notes": [
            "The original branch reconstructs layer zero from the cached frozen-attention target, then all branches use identical frozen layers 1-39, final norm, and lm_head.",
            "Each evaluation window is an independent causal sequence and NLL covers only within-window next-token labels.",
            "Only one Qwen decoder layer is materialized on device at a time during streamed propagation.",
            "passed reports numerical execution; scientific_gate_passed reports the frozen end-to-end NLL threshold.",
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
        raise SystemExit("STREAMED-END-TO-END-SHOCK-NONFINITE")
    print(
        "STREAMED-END-TO-END-SHOCK-PASS"
        if scientific_gate_passed
        else "STREAMED-END-TO-END-SHOCK-GATE-FAIL"
    )
    return result


if __name__ == "__main__":
    main()
