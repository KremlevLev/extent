from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_activation_cache import (
    _create_data_parallel_decoder_runner,
    _create_decoder_runner,
    _create_norm_runner,
    _safe_prune_shards,
)
from scripts.qwen_decoder_aware_confirmation import (
    main as train_multiseed,
    parse_seeds,
)
from scripts.qwen_decoder_aware_distill import _teacher_decoder_outputs
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_streamed_end_to_end_shock import (
    _read_json,
    _run_lm_metrics,
    _run_replacement_windows,
)
from scripts.qwen_streamed_multiseed_end_to_end import branch_divergence
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.endpoint_checkpoint import (
    restore_endpoint_checkpoint,
    save_endpoint_checkpoint,
)
from extent.experiment_stage import update_stage_manifest
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
    aggregate_two_layer_composition,
    bootstrap_two_layer_composition_inflation,
    create_lm_metrics_runner,
    full_model_shard_last_use,
    hidden_relative_l2,
)
from extent.teacher_activation_cache import (
    file_sha256,
    load_activation_cache,
    run_host_data_parallel,
    run_host_microbatches,
    validate_external_evaluation_cache,
)
from extent.weight_mapping import QwenCheckpointReader


TARGET_LAYERS = (0, 18)


def composition_branch_names(seeds: tuple[int, ...]) -> tuple[str, ...]:
    names = ["ORIGINAL-CACHED-QWEN"]
    for seed in seeds:
        names.extend(
            (
                f"SEED-{seed}-LAYER0-ONLY",
                f"SEED-{seed}-LAYER18-ONLY",
                f"SEED-{seed}-LAYER0-LAYER18",
            )
        )
    return tuple(names)


def _validate_cache_pair(
    layer0_manifest: dict,
    layer18_manifest: dict,
    layer0_arrays: dict,
    layer18_arrays: dict,
    *,
    evaluation: bool,
) -> None:
    if int(layer0_manifest["target_layer"]) != 0:
        raise ValueError("the layer-zero cache must target layer 0")
    if int(layer18_manifest["target_layer"]) != 18:
        raise ValueError("the second cache must target layer 18")
    shared = (
        "source",
        "dataset",
        "sequence_length",
        "dataset_split",
        "token_offset",
        "token_range",
        "window_layout",
        "evaluation_only",
    )
    mismatches = {
        key: (layer0_manifest.get(key), layer18_manifest.get(key))
        for key in shared
        if layer0_manifest.get(key) != layer18_manifest.get(key)
    }
    if mismatches:
        raise ValueError(f"layer cache provenance mismatch: {mismatches}")
    if bool(layer0_manifest.get("evaluation_only")) != evaluation:
        raise ValueError("cache evaluation_only status differs from its role")
    if not np.array_equal(layer0_arrays["token_ids"], layer18_arrays["token_ids"]):
        raise ValueError("layer caches contain different token IDs")


def _train_layer(
    *,
    layer: int,
    activation_manifest: str,
    activation_dir: str | None,
    evaluation_manifest: str,
    evaluation_dir: str | None,
    qwen_cache_dir: str,
    seeds: str,
    data_seed: int,
    total_steps: int,
    checkpoints: str,
    batch_windows: int,
    evaluation_batch_windows: int,
    learning_rate: float,
    readout_ridge: float,
    decoder_loss_weight: float,
    compute_dtype: str,
    skip_hash_verification: bool,
    output_dir: Path,
    resume: bool,
) -> tuple[dict, dict[str, dict[str, dict]], dict]:
    layer_output = output_dir / f"layer{layer}-training"
    result_json = layer_output / f"exp044-layer{layer}-training.json"
    endpoint_dir = output_dir / "exp044-endpoints" / f"layer{layer}"
    compatibility = {
        "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
        "experiment": "exp044-layer0-layer18",
        "target_layer": layer,
        "activation_manifest_sha256": file_sha256(Path(activation_manifest)),
        "evaluation_manifest_sha256": file_sha256(Path(evaluation_manifest)),
        "seeds": [int(value) for value in seeds.split(",")],
        "data_seed": data_seed,
        "total_steps": total_steps,
        "checkpoints": checkpoints,
        "batch_windows": batch_windows,
        "evaluation_batch_windows": evaluation_batch_windows,
        "learning_rate": learning_rate,
        "readout_ridge": readout_ridge,
        "decoder_loss_weight": decoder_loss_weight,
        "compute_dtype": compute_dtype,
    }
    if resume and result_json.exists() and (endpoint_dir / "checkpoint.json").exists():
        training = json.loads(result_json.read_text(encoding="utf-8"))
        if not training.get("passed") or int(training.get("target_layer", -1)) != layer:
            raise ValueError(f"invalid completed training result for layer {layer}")
        endpoints, metadata = restore_endpoint_checkpoint(
            endpoint_dir, expected_compatibility=compatibility
        )
        print(
            f"training_replacement_layer={layer} RESUME-PASS "
            f"sha256={metadata['checkpoint_sha256']}"
        )
        return training, endpoints, metadata
    arguments = [
        "--activation-cache-manifest", activation_manifest,
        "--evaluation-cache-manifest", evaluation_manifest,
        "--allow-cross-split-evaluation",
        "--qwen-cache-dir", qwen_cache_dir,
        "--seeds", seeds,
        "--data-seed", str(data_seed),
        "--total-steps", str(total_steps),
        "--checkpoints", checkpoints,
        "--batch-windows", str(batch_windows),
        "--evaluation-batch-windows", str(evaluation_batch_windows),
        "--learning-rate", str(learning_rate),
        "--readout-ridge", str(readout_ridge),
        "--decoder-loss-weight", str(decoder_loss_weight),
        "--compute-dtype", compute_dtype,
        "--result-json", str(result_json),
        "--output-dir", str(layer_output),
    ]
    if activation_dir:
        arguments.extend(["--activation-cache-dir", activation_dir])
    if evaluation_dir:
        arguments.extend(["--evaluation-cache-dir", evaluation_dir])
    if skip_hash_verification:
        arguments.append("--skip-hash-verification")
    print(f"training_replacement_layer={layer} START")
    training, _, endpoints = train_multiseed(
        arguments, return_endpoint_params=True
    )
    metadata = save_endpoint_checkpoint(
        endpoint_dir,
        endpoints,
        compatibility=compatibility,
    )
    print(f"training_replacement_layer={layer} DONE")
    print(
        f"endpoint_checkpoint_layer={layer} "
        f"sha256={metadata['checkpoint_sha256']}"
    )
    jax.clear_caches()
    gc.collect()
    return training, endpoints, metadata


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Measure Qwen3 layer-0 plus layer-18 Mamba composition shock."
    )
    parser.add_argument("--layer0-activation-cache-manifest", required=True)
    parser.add_argument("--layer0-activation-cache-dir")
    parser.add_argument("--layer18-activation-cache-manifest", required=True)
    parser.add_argument("--layer18-activation-cache-dir")
    parser.add_argument("--layer0-evaluation-cache-manifest", required=True)
    parser.add_argument("--layer0-evaluation-cache-dir")
    parser.add_argument("--layer18-evaluation-cache-manifest", required=True)
    parser.add_argument("--layer18-evaluation-cache-dir")
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp044-weights"
    )
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--training-checkpoints", default="0,512,1024")
    parser.add_argument("--evaluation-batch-windows", type=int, default=1)
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--prune-consumed-shards", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stream-parity-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--maximum-mean-inflation", type=float, default=1.25)
    parser.add_argument("--maximum-seed-inflation", type=float, default=1.50)
    parser.add_argument("--maximum-bootstrap-upper", type=float, default=1.50)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--stage-manifest")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if args.total_steps != 1024:
        raise ValueError("EXP-044 fixes training at 1,024 steps per layer and arm")
    if min(
        args.batch_windows,
        args.evaluation_batch_windows,
        args.per_device_windows,
        args.bootstrap_samples,
    ) < 1:
        raise ValueError("batch sizes and bootstrap samples must be positive")
    if args.bootstrap_samples < 100:
        raise ValueError("EXP-044 requires at least 100 bootstrap samples")
    if min(
        args.stream_parity_tolerance,
        args.maximum_mean_inflation,
        args.maximum_seed_inflation,
        args.maximum_bootstrap_upper,
    ) <= 0:
        raise ValueError("gate thresholds must be positive")

    jax.config.update("jax_default_matmul_precision", "high")
    cache_specs = {
        0: (
            args.layer0_activation_cache_manifest,
            args.layer0_activation_cache_dir,
            args.layer0_evaluation_cache_manifest,
            args.layer0_evaluation_cache_dir,
        ),
        18: (
            args.layer18_activation_cache_manifest,
            args.layer18_activation_cache_dir,
            args.layer18_evaluation_cache_manifest,
            args.layer18_evaluation_cache_dir,
        ),
    }
    loaded = {}
    for layer, (train_manifest_path, train_dir, eval_manifest_path, eval_dir) in cache_specs.items():
        train_manifest, train_arrays, train_paths = load_activation_cache(
            train_manifest_path,
            artifact_dir=train_dir,
            verify_hashes=not args.skip_hash_verification,
        )
        eval_manifest, eval_arrays, eval_paths = load_activation_cache(
            eval_manifest_path,
            artifact_dir=eval_dir,
            verify_hashes=not args.skip_hash_verification,
        )
        evaluation_slice = validate_external_evaluation_cache(
            train_manifest,
            eval_manifest,
            required_token_offset=0,
            required_evaluation_windows=256,
            required_dataset_split="validation",
            allow_cross_split=True,
        )
        loaded[layer] = {
            "training_manifest": train_manifest,
            "training_arrays": train_arrays,
            "training_paths": train_paths,
            "evaluation_manifest": eval_manifest,
            "evaluation_arrays": eval_arrays,
            "evaluation_paths": eval_paths,
            "evaluation_slice": evaluation_slice,
        }
    _validate_cache_pair(
        loaded[0]["training_manifest"],
        loaded[18]["training_manifest"],
        loaded[0]["training_arrays"],
        loaded[18]["training_arrays"],
        evaluation=False,
    )
    _validate_cache_pair(
        loaded[0]["evaluation_manifest"],
        loaded[18]["evaluation_manifest"],
        loaded[0]["evaluation_arrays"],
        loaded[18]["evaluation_arrays"],
        evaluation=True,
    )
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if any(
        loaded[layer]["training_manifest"].get("source") != source_name
        for layer in TARGET_LAYERS
    ):
        raise ValueError("EXP-044 requires the pinned Qwen3-14B source")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    devices = list(jax.devices())
    if args.data_parallel and len(devices) < 2:
        raise ValueError("--data-parallel requires at least two visible devices")
    parallel_devices = devices if args.data_parallel else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_results = {}
    endpoint_params = {}
    endpoint_checkpoints = {}
    for layer in TARGET_LAYERS:
        spec = cache_specs[layer]
        (
            training_results[str(layer)],
            endpoint_params[str(layer)],
            endpoint_checkpoints[str(layer)],
        ) = _train_layer(
            layer=layer,
            activation_manifest=spec[0],
            activation_dir=spec[1],
            evaluation_manifest=spec[2],
            evaluation_dir=spec[3],
            qwen_cache_dir=args.qwen_cache_dir,
            seeds=args.seeds,
            data_seed=args.data_seed,
            total_steps=args.total_steps,
            checkpoints=args.training_checkpoints,
            batch_windows=args.batch_windows,
            evaluation_batch_windows=args.evaluation_batch_windows,
            learning_rate=args.learning_rate,
            readout_ridge=args.readout_ridge,
            decoder_loss_weight=args.decoder_loss_weight,
            compute_dtype=dtype_decision.dtype,
            skip_hash_verification=args.skip_hash_verification,
            output_dir=output_dir,
            resume=args.resume,
        )
        if args.stage_manifest:
            update_stage_manifest(
                args.stage_manifest,
                experiment="exp044-layer0-layer18",
                stage=f"training-layer{layer}",
                status="completed",
                details={
                    "checkpoint_sha256": endpoint_checkpoints[str(layer)][
                        "checkpoint_sha256"
                    ],
                    "checkpoint_bytes": endpoint_checkpoints[str(layer)][
                        "checkpoint_bytes"
                    ],
                },
            )

    if args.stage_manifest:
        update_stage_manifest(
            args.stage_manifest,
            experiment="exp044-layer0-layer18",
            stage="streamed-evaluation",
            status="running",
            details={"restart_boundary": "decoder-layer-0"},
        )

    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(
        QWEN3_14B.resolve_url("model.safetensors.index.json")
    )
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
    names = composition_branch_names(seeds)
    evaluation_slice = loaded[0]["evaluation_slice"]
    evaluation_windows = evaluation_slice.stop - evaluation_slice.start
    evaluation_arrays0 = loaded[0]["evaluation_arrays"]
    residual0 = jnp.asarray(
        evaluation_arrays0["residual_input"][evaluation_slice],
        dtype=compute_dtype,
    )
    normalized0 = jnp.asarray(
        evaluation_arrays0["normalized_input"][evaluation_slice],
        dtype=compute_dtype,
    )
    attention0 = jnp.asarray(
        evaluation_arrays0["attention_target"][evaluation_slice],
        dtype=compute_dtype,
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
    tail0_params = qwen3_decoder_tail_params(reader, 0)
    tail = Qwen3DecoderTail(
        source.hidden_size,
        source.intermediate_size,
        source.rms_norm_eps,
        compute_dtype,
        jnp.float32,
    )
    teacher_tail_runner = create_batched_teacher_tail_runner(tail)
    replacement_runner = create_batched_replacement_runner(mamba, tail)
    original_layer0 = _teacher_decoder_outputs(
        teacher_tail_runner,
        tail0_params,
        residual0,
        attention0,
        args.evaluation_batch_windows,
    )
    layer0_replacements = {}
    for seed in seeds:
        layer0_replacements[str(seed)] = _run_replacement_windows(
            replacement_runner,
            endpoint_params["0"][str(seed)]["JOINT-MIXER-DECODER"],
            tail0_params,
            residual0,
            normalized0,
            args.evaluation_batch_windows,
        )
    branch_outputs = [original_layer0]
    for seed in seeds:
        replacement = layer0_replacements[str(seed)]
        branch_outputs.extend((replacement, original_layer0, replacement))
    hidden = np.concatenate(branch_outputs, axis=0)
    divergence = {
        "0": branch_divergence(hidden, names, evaluation_windows)
    }
    del (
        branch_outputs,
        layer0_replacements,
        tail0_params,
        residual0,
        normalized0,
        attention0,
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
    sequence_length = int(loaded[0]["training_manifest"]["sequence_length"])
    positions = jnp.arange(sequence_length, dtype=jnp.int32)[None]
    attention_mask = jnp.ones((1, sequence_length), dtype=jnp.bool_)
    decoder = Qwen3DecoderLayer(source)
    layer_runner = (
        _create_data_parallel_decoder_runner(
            decoder, positions, attention_mask, devices
        )
        if args.data_parallel
        else _create_decoder_runner(decoder, positions, attention_mask)
    )

    def run_teacher_layer(params, values):
        return (
            run_host_data_parallel(
                layer_runner,
                params,
                values,
                args.per_device_windows,
                len(devices),
                input_dtype=compute_dtype,
            )
            if args.data_parallel
            else run_host_microbatches(
                layer_runner,
                params,
                values,
                args.evaluation_batch_windows,
                input_dtype=compute_dtype,
            )
        )

    for layer_index in range(1, 18):
        print(f"streaming_decoder_layer={layer_index}/39")
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
        hidden = run_teacher_layer(layer_params, hidden)
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError(
                f"non-finite streamed residual after layer {layer_index}"
            )
        del layer_params, layer_arrays, reader
        gc.collect()
        if args.prune_consumed_shards:
            removed_shards.extend(
                _safe_prune_shards(model_dir, last_use, layer_index)
            )
    divergence["17"] = branch_divergence(hidden, names, evaluation_windows)

    layer18_eval = loaded[18]["evaluation_arrays"]
    layer18_slice = loaded[18]["evaluation_slice"]
    cached_layer18_residual = np.asarray(
        layer18_eval["residual_input"][layer18_slice], dtype=np.float32
    )
    branch_values = hidden.reshape(
        len(names), evaluation_windows, *hidden.shape[1:]
    )
    stream_parity_l2 = hidden_relative_l2(
        cached_layer18_residual, branch_values[0]
    )
    stream_parity_passed = stream_parity_l2 <= args.stream_parity_tolerance
    if not stream_parity_passed:
        raise ValueError(
            "streamed original input at layer 18 does not reproduce its cache: "
            f"relative_l2={stream_parity_l2}"
        )

    print("applying_composed_replacement_layer=18")
    ensure_layer_checkpoint(
        model_dir,
        index_payload["weight_map"],
        source,
        18,
        repo_id=QWEN3_14B.repo_id,
        revision=QWEN3_14B.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    layer18_arrays = load_layer_arrays(reader, source, 18)
    layer18_teacher_params = jax_layer_params(layer18_arrays, source, 18)
    layer18_tail_params = qwen3_decoder_tail_params(reader, 18)
    norm18 = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    norm18_runner = _create_norm_runner(norm18)
    norm18_params = {
        "scale": jnp.asarray(
            layer18_arrays["model.layers.18.input_layernorm.weight"],
            dtype=jnp.float32,
        )
    }
    replacement18_runner = create_batched_replacement_runner(mamba, tail)
    next_values = []
    for branch_index, name in enumerate(names):
        values = branch_values[branch_index]
        if name == "ORIGINAL-CACHED-QWEN" or name.endswith("LAYER0-ONLY"):
            output = run_teacher_layer(layer18_teacher_params, values)
        else:
            seed = name.split("-")[1]
            normalized = run_host_microbatches(
                norm18_runner,
                norm18_params,
                values,
                args.evaluation_batch_windows,
                input_dtype=compute_dtype,
            )
            output = _run_replacement_windows(
                replacement18_runner,
                endpoint_params["18"][seed]["JOINT-MIXER-DECODER"],
                layer18_tail_params,
                jnp.asarray(values, dtype=compute_dtype),
                jnp.asarray(normalized, dtype=compute_dtype),
                args.evaluation_batch_windows,
            )
        next_values.append(output)
    hidden = np.concatenate(next_values, axis=0)
    divergence["18"] = branch_divergence(hidden, names, evaluation_windows)
    del (
        next_values,
        branch_values,
        cached_layer18_residual,
        layer18_arrays,
        layer18_teacher_params,
        layer18_tail_params,
        norm18,
        norm18_runner,
        norm18_params,
        replacement18_runner,
        endpoint_params,
        mamba,
        tail,
        reader,
    )
    jax.clear_caches()
    gc.collect()
    if args.prune_consumed_shards:
        removed_shards.extend(_safe_prune_shards(model_dir, last_use, 18))

    for layer_index in range(19, source.num_layers):
        print(f"streaming_decoder_layer={layer_index}/39")
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
        hidden = run_teacher_layer(layer_params, hidden)
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError(
                f"non-finite streamed residual after layer {layer_index}"
            )
        if layer_index in {29, 39}:
            divergence[str(layer_index)] = branch_divergence(
                hidden, names, evaluation_windows
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
    final_norm_params = {
        "scale": jnp.asarray(reader.read("model.norm.weight"), dtype=jnp.float32)
    }
    lm_head_kernel = jnp.asarray(
        reader.read("lm_head.weight").T, dtype=compute_dtype
    )
    final_norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    lm_runner = create_lm_metrics_runner(
        final_norm,
        data_parallel_devices=parallel_devices,
        per_window=True,
    )
    token_ids = np.asarray(
        evaluation_arrays0["token_ids"][evaluation_slice], dtype=np.int32
    )
    branch_values = hidden.reshape(
        len(names), evaluation_windows, *hidden.shape[1:]
    )
    lm_metrics = {}
    for branch_index, name in enumerate(names):
        print(f"evaluating_lm_head={name}")
        lm_metrics[name] = _run_lm_metrics(
            lm_runner,
            final_norm_params,
            lm_head_kernel,
            branch_values[branch_index],
            token_ids,
            devices=parallel_devices,
            per_device_windows=args.per_device_windows,
            compute_dtype=compute_dtype,
            return_window_nll=True,
        )
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, source.num_layers)
        )

    seed_metrics = {
        str(seed): {
            "layer0_only": lm_metrics[f"SEED-{seed}-LAYER0-ONLY"],
            "layer18_only": lm_metrics[f"SEED-{seed}-LAYER18-ONLY"],
            "layer0_layer18": lm_metrics[f"SEED-{seed}-LAYER0-LAYER18"],
        }
        for seed in seeds
    }
    aggregate = aggregate_two_layer_composition(
        original_nll=lm_metrics["ORIGINAL-CACHED-QWEN"]["mean_nll"],
        seed_metrics=seed_metrics,
        maximum_mean_inflation=args.maximum_mean_inflation,
        maximum_seed_inflation=args.maximum_seed_inflation,
    )
    bootstrap = bootstrap_two_layer_composition_inflation(
        original_window_nll=lm_metrics["ORIGINAL-CACHED-QWEN"][
            "window_mean_nll"
        ],
        seed_window_nll={
            str(seed): {
                key: metrics[key]["window_mean_nll"]
                for key in ("layer0_only", "layer18_only", "layer0_layer18")
            }
            for seed, metrics in seed_metrics.items()
        },
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    all_finite = bool(
        all(result["passed"] for result in training_results.values())
        and all(metrics["finite"] for metrics in lm_metrics.values())
        and aggregate["all_finite"]
        and stream_parity_passed
    )
    scientific_gate_passed = bool(
        all_finite
        and aggregate["scientific_gate_passed"]
        and bootstrap["mean_inflation_confidence_interval"][1]
        <= args.maximum_bootstrap_upper
    )
    result = {
        "source": source_name,
        "dataset": loaded[0]["training_manifest"].get("dataset"),
        "method": "two_layer_Mamba3_composition_interaction",
        "protocol": "exp044-layer0-layer18",
        "target_layers": list(TARGET_LAYERS),
        "training_cache_manifests": {
            str(layer): str(Path(cache_specs[layer][0]).resolve())
            for layer in TARGET_LAYERS
        },
        "evaluation_cache_manifests": {
            str(layer): str(Path(cache_specs[layer][2]).resolve())
            for layer in TARGET_LAYERS
        },
        "resolved_training_artifacts": {
            str(layer): {
                name: str(path)
                for name, path in loaded[layer]["training_paths"].items()
            }
            for layer in TARGET_LAYERS
        },
        "resolved_evaluation_artifacts": {
            str(layer): {
                name: str(path)
                for name, path in loaded[layer]["evaluation_paths"].items()
            }
            for layer in TARGET_LAYERS
        },
        "evaluation_dataset_split": "validation",
        "evaluation_token_offset": 0,
        "evaluation_token_range": loaded[0]["evaluation_manifest"].get(
            "token_range"
        ),
        "sequence_length": sequence_length,
        "evaluation_windows": evaluation_windows,
        "evaluation_tokens_per_branch": evaluation_windows
        * (sequence_length - 1),
        "seeds": list(seeds),
        "branches": list(names),
        "training_results": training_results,
        "endpoint_checkpoints": endpoint_checkpoints,
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "data_parallel": args.data_parallel,
        "per_device_windows": args.per_device_windows,
        "stream_to_layer18_cache_parity": {
            "relative_l2": stream_parity_l2,
            "maximum_relative_l2": args.stream_parity_tolerance,
            "passed": stream_parity_passed,
        },
        "hidden_relative_l2_to_original": divergence,
        "lm_metrics": lm_metrics,
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "maximum_bootstrap_upper": args.maximum_bootstrap_upper,
        "prune_consumed_shards": args.prune_consumed_shards,
        "removed_checkpoint_shards": sorted(set(removed_shards)),
        "scientific_gate_passed": scientific_gate_passed,
        "passed": all_finite,
        "notes": [
            "Each seed uses independently trained decoder-aware layer-0 and layer-18 Mamba endpoints.",
            "Layer-18 Mamba receives a freshly computed RMSNorm input from each branch, including the shifted layer-0 replacement stream.",
            "Composition inflation is composed excess NLL divided by the sum of the two standalone excess NLL values.",
            "The original, both single replacements, and the composed replacement share identical validation windows and frozen Qwen decoder weights.",
            "passed reports numerical/provenance validity; scientific_gate_passed applies the pre-registered interaction and bootstrap limits.",
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
        raise SystemExit("TWO-LAYER-COMPOSITION-NONFINITE")
    if args.stage_manifest:
        update_stage_manifest(
            args.stage_manifest,
            experiment="exp044-layer0-layer18",
            stage="final",
            status="completed",
            details={
                "result_json": str(Path(args.result_json).resolve()),
                "scientific_gate_passed": scientific_gate_passed,
            },
        )
    print(
        "TWO-LAYER-COMPOSITION-PASS"
        if scientific_gate_passed
        else "TWO-LAYER-COMPOSITION-GATE-FAIL"
    )
    return result


if __name__ == "__main__":
    main()
