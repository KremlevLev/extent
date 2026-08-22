from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from extent.hardware import recommended_compute_dtype
from extent.layers.common import RMSNorm
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    jax_layer_params,
    load_layer_arrays,
    load_mixer_arrays,
)
from extent.qwen3_teacher import (
    Qwen3DecoderLayer,
    Qwen3GQAAttention,
    Qwen3TeacherConfig,
)
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.teacher_activation_cache import (
    activation_window_layout,
    array_artifact,
    atomic_save_array,
    checkpoint_shard_last_use,
    file_sha256,
    run_host_data_parallel,
    run_host_microbatches,
)
from extent.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _create_decoder_runner(module, positions, attention_mask):
    @jax.jit
    def run(params, inputs):
        batch_positions = jnp.broadcast_to(
            positions, (inputs.shape[0], positions.shape[1])
        )
        batch_mask = jnp.broadcast_to(
            attention_mask, (inputs.shape[0], attention_mask.shape[1])
        )
        return module.apply(
            {"params": params}, inputs, batch_positions, batch_mask
        )

    return run


def _create_norm_runner(module):
    @jax.jit
    def run(params, inputs):
        return module.apply({"params": params}, inputs)

    return run


def _create_data_parallel_decoder_runner(module, positions, attention_mask, devices):
    def apply(params, inputs):
        batch_positions = jnp.broadcast_to(
            positions, (inputs.shape[0], positions.shape[1])
        )
        batch_mask = jnp.broadcast_to(
            attention_mask, (inputs.shape[0], attention_mask.shape[1])
        )
        return module.apply(
            {"params": params}, inputs, batch_positions, batch_mask
        )

    return jax.pmap(apply, in_axes=(None, 0), devices=devices)


def _create_data_parallel_norm_runner(module, devices):
    return jax.pmap(
        lambda params, inputs: module.apply({"params": params}, inputs),
        in_axes=(None, 0),
        devices=devices,
    )


def _safe_prune_shards(
    model_dir: Path,
    last_use: dict[str, int],
    completed_layer: int,
) -> list[str]:
    root = model_dir.resolve()
    removed = []
    for shard, layer in last_use.items():
        if layer != completed_layer:
            continue
        candidate = (root / shard).resolve()
        if root not in candidate.parents:
            raise ValueError(f"refusing to prune shard outside checkpoint cache: {candidate}")
        if candidate.exists():
            candidate.unlink()
            removed.append(shard)
    return removed


def _partial_payload(
    *,
    source_name: str,
    target_layer: int,
    completed_layer: int,
    shape: tuple[int, ...],
    compute_dtype: str,
    storage_dtype: str,
    tokens_sha256: str,
    token_offset: int,
    dataset_split: str,
    evaluation_only: bool,
    residual_path: Path,
    removed_shards: list[str],
) -> dict:
    return {
        "status": "propagating" if completed_layer < target_layer - 1 else "target_input_ready",
        "source": source_name,
        "target_layer": target_layer,
        "completed_decoder_layer": completed_layer,
        "activation_shape": list(shape),
        "compute_dtype": compute_dtype,
        "storage_dtype": storage_dtype,
        "tokens_sha256": tokens_sha256,
        "token_offset": token_offset,
        "dataset_split": dataset_split,
        "evaluation_only": evaluation_only,
        "residual_input_path": str(residual_path.resolve()),
        "removed_checkpoint_shards": removed_shards,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cache true Qwen3 residual inputs and attention targets for one layer."
    )
    parser.add_argument(
        "--cache-dir", default="/kaggle/working/qwen3-activation-checkpoint"
    )
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-calibration-cache"
    )
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=8)
    parser.add_argument("--training-windows", type=int, default=80)
    parser.add_argument("--evaluation-windows", type=int, default=4)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--microbatch-windows", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--per-device-windows", type=int, default=2)
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--result-json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune-consumed-shards", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.target_layer < 40:
        raise ValueError("target-layer must be in [0, 40)")
    if min(args.sequence_length, args.microbatch_windows, args.per_device_windows) < 1:
        raise ValueError("sequence-length and batch sizes must be positive")
    if args.token_offset < 0:
        raise ValueError("token-offset must be non-negative")
    layout = activation_window_layout(
        args.calibration_windows,
        args.training_windows,
        args.evaluation_windows,
        allow_evaluation_only=args.evaluation_only,
    )
    jax.config.update("jax_default_matmul_precision", "high")
    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    storage_dtype = np.dtype(args.storage_dtype)
    spec = QWEN3_14B
    source_name = f"{spec.repo_id}@{spec.revision}"
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    if args.target_layer >= source.num_layers:
        raise ValueError("target-layer exceeds the pinned teacher depth")
    devices = list(jax.devices())
    if args.data_parallel and len(devices) < 2:
        raise ValueError("--data-parallel requires at least two visible devices")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qwen3-layer{args.target_layer}-activation-cache"
    residual_path = output_dir / f"{stem}-residual-input.npy"
    normalized_path = output_dir / f"{stem}-normalized-input.npy"
    target_path = output_dir / f"{stem}-attention-target.npy"
    tokens_path = output_dir / f"{stem}-token-ids.npy"
    requested_manifest = Path(
        args.result_json or f"/kaggle/working/{stem}-manifest.json"
    )
    partial_manifest = requested_manifest.with_suffix(".partial.json")
    output_partial_manifest = output_dir / partial_manifest.name

    tokens = load_wikitext2_tokens(
        layout.total_windows * args.sequence_length,
        args.dataset_cache_dir,
        tokenizer_repo=spec.repo_id,
        tokenizer_revision=spec.revision,
        token_offset=args.token_offset,
        dataset_split=args.dataset_split,
    ).reshape(layout.total_windows, args.sequence_length)
    atomic_save_array(tokens_path, tokens, np.dtype(np.int32))
    tokens_sha256 = file_sha256(tokens_path)
    model_dir = Path(args.cache_dir)
    from huggingface_hub import hf_hub_download

    for filename in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=filename,
            local_dir=model_dir,
        )
    reader = QwenCheckpointReader(model_dir)
    last_use = checkpoint_shard_last_use(index_payload["weight_map"], args.target_layer)
    removed_shards: list[str] = []
    expected_shape = (layout.total_windows, args.sequence_length, source.hidden_size)
    completed_layer = -1

    if args.resume:
        resume_manifest_path = (
            output_partial_manifest
            if output_partial_manifest.exists()
            else partial_manifest
        )
        if not resume_manifest_path.exists() or not residual_path.exists():
            raise FileNotFoundError("resume requires partial manifest and residual input")
        resume = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        expected = {
            "source": source_name,
            "target_layer": args.target_layer,
            "activation_shape": list(expected_shape),
            "compute_dtype": dtype_decision.dtype,
            "storage_dtype": args.storage_dtype,
            "tokens_sha256": tokens_sha256,
            "token_offset": args.token_offset,
            "dataset_split": args.dataset_split,
            "evaluation_only": args.evaluation_only,
        }
        mismatches = {
            key: (resume.get(key), value)
            for key, value in expected.items()
            if resume.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume manifest does not match this run: {mismatches}")
        hidden = np.asarray(np.load(residual_path, mmap_mode="r"), dtype=np.float32)
        completed_layer = int(resume["completed_decoder_layer"])
        removed_shards = list(resume.get("removed_checkpoint_shards", []))
        print(f"resume=PASS completed_decoder_layer={completed_layer}")
    else:
        embedding_shard = index_payload["weight_map"]["model.embed_tokens.weight"]
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=embedding_shard,
            local_dir=model_dir,
        )
        hidden = reader.read_rows(
            "model.embed_tokens.weight", tokens.reshape(-1)
        ).reshape(expected_shape)
        atomic_save_array(residual_path, hidden, storage_dtype)
        if args.prune_consumed_shards:
            removed_shards.extend(_safe_prune_shards(model_dir, last_use, -1))

    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None]
    attention_mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    decoder = Qwen3DecoderLayer(source)
    if args.data_parallel:
        run_decoder = _create_data_parallel_decoder_runner(
            decoder, positions, attention_mask, devices
        )
        run_batches = lambda runner, params, values: run_host_data_parallel(
            runner,
            params,
            values,
            args.per_device_windows,
            len(devices),
            input_dtype=compute_dtype,
        )
    else:
        run_decoder = _create_decoder_runner(decoder, positions, attention_mask)
        run_batches = lambda runner, params, values: run_host_microbatches(
            runner,
            params,
            values,
            args.microbatch_windows,
            input_dtype=compute_dtype,
        )
    for layer in range(completed_layer + 1, args.target_layer):
        print(f"propagating_decoder_layer={layer}/{args.target_layer - 1}")
        ensure_layer_checkpoint(
            model_dir,
            index_payload["weight_map"],
            source,
            layer,
            repo_id=spec.repo_id,
            revision=spec.revision,
        )
        arrays = load_layer_arrays(reader, source, layer)
        params = jax_layer_params(arrays, source, layer)
        hidden = run_batches(run_decoder, params, hidden)
        finite = bool(np.all(np.isfinite(hidden)))
        if not finite:
            raise FloatingPointError(f"non-finite residual stream after layer {layer}")
        atomic_save_array(residual_path, hidden, storage_dtype)
        completed_layer = layer
        del params, arrays
        gc.collect()
        if args.prune_consumed_shards:
            removed_shards.extend(_safe_prune_shards(model_dir, last_use, layer))
        partial = _partial_payload(
            source_name=source_name,
            target_layer=args.target_layer,
            completed_layer=completed_layer,
            shape=expected_shape,
            compute_dtype=dtype_decision.dtype,
            storage_dtype=args.storage_dtype,
            tokens_sha256=tokens_sha256,
            token_offset=args.token_offset,
            dataset_split=args.dataset_split,
            evaluation_only=args.evaluation_only,
            residual_path=residual_path,
            removed_shards=removed_shards,
        )
        _write_json_with_output_mirror(
            partial_manifest, partial, str(output_dir)
        )
        print(f"layer={layer} residual_cache={residual_path.resolve()}")

    # Ensure a resumable manifest also exists for target layer zero.
    partial = _partial_payload(
        source_name=source_name,
        target_layer=args.target_layer,
        completed_layer=completed_layer,
        shape=expected_shape,
        compute_dtype=dtype_decision.dtype,
        storage_dtype=args.storage_dtype,
        tokens_sha256=tokens_sha256,
        token_offset=args.token_offset,
        dataset_split=args.dataset_split,
        evaluation_only=args.evaluation_only,
        residual_path=residual_path,
        removed_shards=removed_shards,
    )
    _write_json_with_output_mirror(partial_manifest, partial, str(output_dir))

    ensure_layer_checkpoint(
        model_dir,
        index_payload["weight_map"],
        source,
        args.target_layer,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    target_arrays = load_mixer_arrays(reader, source, args.target_layer)
    norm_name = f"model.layers.{args.target_layer}.input_layernorm.weight"
    norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    run_norm = (
        _create_data_parallel_norm_runner(norm, devices)
        if args.data_parallel
        else _create_norm_runner(norm)
    )
    norm_params = {"scale": jnp.asarray(target_arrays[norm_name])}
    normalized = run_batches(run_norm, norm_params, hidden)
    attention = Qwen3GQAAttention(source)
    run_attention = (
        _create_data_parallel_decoder_runner(
            attention, positions, attention_mask, devices
        )
        if args.data_parallel
        else _create_decoder_runner(attention, positions, attention_mask)
    )
    attention_params = jax_attention_params(
        target_arrays, source, args.target_layer
    )
    targets = run_batches(run_attention, attention_params, normalized)
    finite = bool(
        np.all(np.isfinite(hidden))
        and np.all(np.isfinite(normalized))
        and np.all(np.isfinite(targets))
    )
    atomic_save_array(residual_path, hidden, storage_dtype)
    atomic_save_array(normalized_path, normalized, storage_dtype)
    atomic_save_array(target_path, targets, storage_dtype)
    del attention_params, target_arrays
    gc.collect()
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, args.target_layer)
        )

    result = {
        "source": source_name,
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "streamed_Qwen3_residual_and_attention_activation_cache",
        "target_layer": args.target_layer,
        "sequence_length": args.sequence_length,
        "dataset_split": args.dataset_split,
        "token_offset": args.token_offset,
        "token_range": [
            args.token_offset,
            args.token_offset + layout.total_windows * args.sequence_length,
        ],
        "evaluation_only": args.evaluation_only,
        "window_layout": {
            "calibration": [layout.calibration.start, layout.calibration.stop],
            "training": [layout.training.start, layout.training.stop],
            "evaluation": [layout.evaluation.start, layout.evaluation.stop],
            "total_windows": layout.total_windows,
        },
        "compute_dtype": dtype_decision.dtype,
        "dtype_reason": dtype_decision.reason,
        "storage_dtype": args.storage_dtype,
        "microbatch_windows": args.microbatch_windows,
        "data_parallel": args.data_parallel,
        "per_device_windows": args.per_device_windows,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "execution": (
            f"data-parallel pmap across {len(devices)} devices"
            if args.data_parallel
            else "single-device jit; additional visible accelerator devices are not used"
        ),
        "prune_consumed_shards": args.prune_consumed_shards,
        "removed_checkpoint_shards": sorted(set(removed_shards)),
        "artifacts": {
            "token_ids": array_artifact(tokens_path),
            "residual_input": array_artifact(residual_path),
            "normalized_input": array_artifact(normalized_path),
            "attention_target": array_artifact(target_path),
        },
        "passed": finite,
        "notes": [
            "Each window is an independent causal sequence whose positions restart at zero.",
            "Residual input is the exact frozen-teacher input before target_layer.",
            "Normalized input and attention target are sufficient for mixer-only distillation.",
            "This script performs teacher inference and caching only; it does not train a model.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        requested_manifest, result, str(output_dir)
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={requested_manifest.resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if not finite:
        raise SystemExit("QWEN3-ACTIVATION-CACHE-FAIL")
    print("QWEN3-ACTIVATION-CACHE-PASS")


if __name__ == "__main__":
    main()
