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
from scripts.qwen_decoder_aware_distill import _teacher_decoder_outputs
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_streamed_end_to_end_shock import (
    _read_json,
    _run_lm_metrics,
    _run_replacement_windows,
)
from scripts.qwen_streamed_multiseed_end_to_end import branch_divergence
from extent.boundary_composition import analyze_boundary_scaling
from extent.composition_onset import analyze_composition_onset
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.endpoint_checkpoint import restore_endpoint_checkpoint
from extent.hardware import recommended_compute_dtype
from extent.layers.common import RMSNorm
from extent.layers.mamba3 import Mamba3MIMO
from extent.progressive_composition import (
    analyze_progressive_composition,
    progressive_branch_names,
    validate_progressive_layer_sets,
)
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
)
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.streamed_lm_eval import (
    create_lm_metrics_runner,
    full_model_shard_last_use,
)
from extent.teacher_activation_cache import (
    file_sha256,
    load_activation_cache,
    run_host_data_parallel,
    run_host_microbatches,
)
from extent.weight_mapping import QwenCheckpointReader


def _resolve_index_path(index_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else index_path.parent / path


def _load_endpoint_bundle(
    index_path: Path,
    layer_spec: dict,
    *,
    source_name: str,
    layer: int,
    seeds: tuple[int, ...],
) -> tuple[dict, dict]:
    directory = _resolve_index_path(index_path, layer_spec["endpoint_dir"])
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compatibility = metadata.get("compatibility", {})
    expected = {
        "source": source_name,
        "target_layer": layer,
        "seeds": list(seeds),
        "total_steps": int(layer_spec["steps"]),
    }
    mismatches = {
        key: (compatibility.get(key), value)
        for key, value in expected.items()
        if compatibility.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"endpoint compatibility mismatch for layer {layer}: {mismatches}"
        )
    endpoints, verified = restore_endpoint_checkpoint(
        directory, expected_compatibility=compatibility
    )
    for seed in seeds:
        try:
            endpoints[str(seed)]["JOINT-MIXER-DECODER"]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} endpoint is missing JOINT seed {seed}"
            ) from exc
    return endpoints, verified


def _branch_specs(
    seeds: tuple[int, ...],
    layer_sets: dict[int, tuple[int, ...]],
    branch_sets: dict[str, tuple[int, ...]] | None = None,
) -> list[dict]:
    specs = [{"name": "ORIGINAL-CACHED-QWEN", "seed": None, "layers": ()}]
    if branch_sets is None:
        branch_sets = {
            f"COMPOSED-{count}": layers
            for count, layers in sorted(layer_sets.items())
        }
    for seed in seeds:
        for label, layers in branch_sets.items():
            specs.append(
                {
                    "name": f"SEED-{seed}-{label}",
                    "seed": seed,
                    "layers": layers,
                }
            )
    return specs


def analyze_exact_lift_pilot(*, original_metric: dict, composed_metrics: dict, **_) -> dict:
    random_metric = composed_metrics["SEED-123-COMPOSED-2"]
    lift_metric = composed_metrics["SEED-456-COMPOSED-2"]
    random_nll = float(random_metric["mean_nll"])
    lift_nll = float(lift_metric["mean_nll"])
    original_nll = float(original_metric["mean_nll"])
    passed = bool(
        np.isfinite([original_nll, random_nll, lift_nll]).all()
        and lift_nll < random_nll
    )
    return {
        "original_mean_nll": original_nll,
        "random_composition_mean_nll": random_nll,
        "exact_lift_composition_mean_nll": lift_nll,
        "exact_minus_random_nll": lift_nll - random_nll,
        "exact_minus_original_nll": lift_nll - original_nll,
        "random_minus_original_nll": random_nll - original_nll,
        "pilot_gate_passed": passed,
        "scientific_gate_passed": passed,
        "all_finite": bool(np.isfinite([original_nll, random_nll, lift_nll]).all()),
        "gate_definition": "The paired two-layer exact-lift composition has lower frozen end-to-end mean NLL than the random-initialized composition.",
    }


def analyze_exact_lift_multiseed(
    *, original_metric: dict, composed_metrics: dict, bootstrap_samples: int, bootstrap_seed: int, **_
) -> dict:
    seeds = (123, 456, 789)
    pairs = []
    window_differences = []
    for seed in seeds:
        random_metric = composed_metrics[f"SEED-{seed}-COMPOSED-2"]
        exact_metric = composed_metrics[f"SEED-{seed + 1000}-COMPOSED-2"]
        delta = float(exact_metric["mean_nll"] - random_metric["mean_nll"])
        pairs.append({
            "seed": seed, "random_mean_nll": float(random_metric["mean_nll"]),
            "exact_mean_nll": float(exact_metric["mean_nll"]), "exact_minus_random_nll": delta,
        })
        window_differences.append(
            np.asarray(exact_metric["window_mean_nll"], np.float64)
            - np.asarray(random_metric["window_mean_nll"], np.float64)
        )
    differences = np.stack(window_differences)
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, np.float64)
    for index in range(bootstrap_samples):
        seed_indices = rng.integers(0, len(seeds), size=len(seeds))
        window_indices = rng.integers(0, differences.shape[1], size=differences.shape[1])
        draws[index] = differences[seed_indices][:, window_indices].mean()
    mean_delta = float(np.mean([pair["exact_minus_random_nll"] for pair in pairs]))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    finite = bool(np.isfinite(differences).all() and np.isfinite(draws).all())
    wins = sum(pair["exact_minus_random_nll"] < 0 for pair in pairs)
    gate = bool(finite and mean_delta < 0 and wins >= 2 and upper < 0)
    return {
        "original_mean_nll": float(original_metric["mean_nll"]),
        "paired_seeds": pairs, "mean_exact_minus_random_nll": mean_delta,
        "wins": wins, "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": bootstrap_samples, "pilot_gate_passed": gate,
        "scientific_gate_passed": gate, "all_finite": finite,
        "gate_definition": "Exact lift has lower mean NLL, wins >=2/3 seeds, and the paired hierarchical-bootstrap 95% upper bound is below zero.",
    }


def analyze_exact_lift_scaling(
    *, layer_sets: dict[int, tuple[int, ...]], original_metric: dict,
    composed_metrics: dict, bootstrap_samples: int, bootstrap_seed: int, **_
) -> dict:
    seeds = (123, 456, 789)
    stages = []
    all_finite = True
    for count in sorted(layer_sets):
        pairs, window_differences = [], []
        for seed in seeds:
            random_metric = composed_metrics[f"SEED-{seed}-COMPOSED-{count}"]
            exact_metric = composed_metrics[f"SEED-{seed + 1000}-COMPOSED-{count}"]
            random_nll = float(random_metric["mean_nll"])
            exact_nll = float(exact_metric["mean_nll"])
            pairs.append({"seed": seed, "random_mean_nll": random_nll, "exact_mean_nll": exact_nll, "exact_minus_random_nll": exact_nll - random_nll})
            window_differences.append(np.asarray(exact_metric["window_mean_nll"], np.float64) - np.asarray(random_metric["window_mean_nll"], np.float64))
        differences = np.stack(window_differences)
        finite = bool(np.isfinite(differences).all())
        all_finite = all_finite and finite
        rng = np.random.default_rng(bootstrap_seed + count)
        draws = np.empty(bootstrap_samples, np.float64)
        for index in range(bootstrap_samples):
            seed_indices = rng.integers(0, len(seeds), size=len(seeds))
            window_indices = rng.integers(0, differences.shape[1], size=differences.shape[1])
            draws[index] = differences[seed_indices][:, window_indices].mean()
        mean_delta = float(np.mean([pair["exact_minus_random_nll"] for pair in pairs]))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        wins = sum(pair["exact_minus_random_nll"] < 0 for pair in pairs)
        stage_gate = bool(finite and mean_delta < 0 and wins >= 2 and upper < 0)
        stages.append({
            "replacement_count": count, "layers": list(layer_sets[count]),
            "paired_seeds": pairs, "mean_exact_minus_random_nll": mean_delta,
            "wins": wins, "bootstrap_95_ci": [float(lower), float(upper)],
            "stage_gate_passed": stage_gate,
        })
    gate = bool(all_finite and len(stages) == 3 and all(stage["stage_gate_passed"] for stage in stages))
    return {
        "original_mean_nll": float(original_metric["mean_nll"]),
        "stages": stages, "bootstrap_samples": bootstrap_samples,
        "all_finite": all_finite, "scientific_gate_passed": gate,
        "gate_definition": "At every 2/4/8-layer stage, exact lift wins >=2/3 seeds and mean paired NLL plus its 95% hierarchical-bootstrap upper bound are below zero.",
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Stream frozen Qwen around registered Mamba compositions."
    )
    parser.add_argument("--endpoint-index", required=True)
    parser.add_argument(
        "--qwen-cache-dir", default="/dev/shm/extent-qwen3-exp051-weights"
    )
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--evaluation-batch-windows", type=int, default=4)
    parser.add_argument("--per-device-windows", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--maximum-original-nll-difference", type=float, default=0.01)
    parser.add_argument("--maximum-mean-inflation", type=float, default=1.25)
    parser.add_argument("--maximum-seed-inflation", type=float, default=1.50)
    parser.add_argument("--maximum-bootstrap-upper", type=float, default=1.50)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--prune-consumed-shards", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    args = parser.parse_args(argv)
    if min(
        args.evaluation_batch_windows,
        args.per_device_windows,
        args.bootstrap_samples,
    ) < 1:
        raise ValueError("batch and bootstrap settings must be positive")

    jax.config.update("jax_default_matmul_precision", "high")
    devices = list(jax.devices())
    if args.data_parallel and len(devices) < 2:
        raise ValueError("--data-parallel requires multiple visible devices")
    parallel_devices = devices if args.data_parallel else None
    index_path = Path(args.endpoint_index).resolve()
    endpoint_index = json.loads(index_path.read_text(encoding="utf-8"))
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if endpoint_index.get("source") != source_name:
        raise ValueError("endpoint index does not use pinned Qwen3-14B")
    seeds = tuple(int(seed) for seed in endpoint_index["seeds"])
    raw_layer_sets = {
        int(count): tuple(layers)
        for count, layers in endpoint_index["layer_sets"].items()
    }
    layer_sets = (
        raw_layer_sets
        if endpoint_index.get("analysis_mode") in {"exact_lift_pilot", "exact_lift_multiseed", "exact_lift_scaling"}
        else validate_progressive_layer_sets(raw_layer_sets)
    )
    target_layers = tuple(
        sorted({layer for values in layer_sets.values() for layer in values})
    )
    if tuple(endpoint_index["target_layers"]) != target_layers:
        raise ValueError("endpoint index target layers do not match nested sets")
    if set(endpoint_index["layers"]) != {str(layer) for layer in target_layers}:
        raise ValueError("endpoint index is incomplete")
    if target_layers[0] != 0:
        raise ValueError("progressive evaluation requires layer 0 as its first replacement")
    if any(0 not in layers for layers in layer_sets.values()):
        raise ValueError("every progressive branch must include replacement layer 0")
    analysis_mode = endpoint_index.get("analysis_mode", "progressive")
    if analysis_mode not in {
        "progressive",
        "boundary_scaling",
        "onset_localization",
        "exact_lift_pilot",
        "exact_lift_multiseed",
        "exact_lift_scaling",
    }:
        raise ValueError(f"unsupported composition analysis mode: {analysis_mode}")
    raw_branch_sets = endpoint_index.get("branch_sets")
    branch_sets = (
        {
            str(label): tuple(int(layer) for layer in layers)
            for label, layers in raw_branch_sets.items()
        }
        if raw_branch_sets is not None
        else None
    )
    if branch_sets is not None:
        if not branch_sets or any(not layers for layers in branch_sets.values()):
            raise ValueError("custom composition branches cannot be empty")
        for label, layers in branch_sets.items():
            if len(layers) != len(set(layers)) or tuple(sorted(layers)) != layers:
                raise ValueError(f"branch {label} must contain sorted unique layers")
            if not set(layers) <= set(target_layers):
                raise ValueError(f"branch {label} uses an unregistered endpoint")
        if {layer for layers in branch_sets.values() for layer in layers} != set(
            target_layers
        ):
            raise ValueError("custom branches must cover every registered endpoint")

    evaluation = endpoint_index["evaluation_cache"]
    evaluation_manifest_path = _resolve_index_path(
        index_path, evaluation["manifest"]
    )
    evaluation_dir = _resolve_index_path(index_path, evaluation["artifact_dir"])
    manifest, arrays, resolved_paths = load_activation_cache(
        evaluation_manifest_path,
        artifact_dir=evaluation_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    if (
        manifest.get("source") != source_name
        or int(manifest.get("target_layer", -1)) != 0
        or not manifest.get("evaluation_only")
        or manifest.get("dataset_split") != "validation"
    ):
        raise ValueError("layer-0 evaluation cache has incompatible provenance")
    layout = manifest["window_layout"]["evaluation"]
    evaluation_slice = slice(int(layout[0]), int(layout[1]))
    evaluation_windows = evaluation_slice.stop - evaluation_slice.start
    if evaluation_windows != 256:
        raise ValueError("composition evaluation requires 256 validation windows")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
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
    sequence_length = int(manifest["sequence_length"])
    specs = _branch_specs(seeds, layer_sets, branch_sets)
    names = tuple(spec["name"] for spec in specs)
    if branch_sets is None and names != progressive_branch_names(seeds, layer_sets):
        raise AssertionError("branch metadata and names diverged")

    mamba = Mamba3MIMO(
        source.hidden_size,
        Mamba3Config(),
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
    replacement_runner = create_batched_replacement_runner(mamba, tail)
    teacher_tail_runner = create_batched_teacher_tail_runner(tail)
    residual0 = jnp.asarray(
        arrays["residual_input"][evaluation_slice], dtype=compute_dtype
    )
    normalized0 = jnp.asarray(
        arrays["normalized_input"][evaluation_slice], dtype=compute_dtype
    )
    attention0 = jnp.asarray(
        arrays["attention_target"][evaluation_slice], dtype=compute_dtype
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
    original0 = _teacher_decoder_outputs(
        teacher_tail_runner,
        tail0_params,
        residual0,
        attention0,
        args.evaluation_batch_windows,
    )
    endpoints0, endpoint_metadata0 = _load_endpoint_bundle(
        index_path,
        endpoint_index["layers"]["0"],
        source_name=source_name,
        layer=0,
        seeds=seeds,
    )
    seed_layer0 = {}
    for seed in seeds:
        seed_layer0[seed] = _run_replacement_windows(
            replacement_runner,
            endpoints0[str(seed)]["JOINT-MIXER-DECODER"],
            tail0_params,
            residual0,
            normalized0,
            args.evaluation_batch_windows,
        )
    hidden = np.concatenate(
        [original0]
        + [
            (
                seed_layer0[int(spec["seed"])]
                if 0 in spec["layers"]
                else original0
            )
            for spec in specs[1:]
        ],
        axis=0,
    )
    divergence = {"0": branch_divergence(hidden, names, evaluation_windows)}
    endpoint_checkpoints = {"0": endpoint_metadata0}
    del (
        endpoints0,
        seed_layer0,
        tail0_params,
        residual0,
        normalized0,
        attention0,
        original0,
        teacher_tail_runner,
        reader,
    )
    jax.clear_caches()
    gc.collect()

    last_use = full_model_shard_last_use(
        index_payload["weight_map"], source.num_layers
    )
    removed_shards = []
    if args.prune_consumed_shards:
        removed_shards.extend(_safe_prune_shards(model_dir, last_use, 0))
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
        if args.data_parallel:
            return run_host_data_parallel(
                layer_runner,
                params,
                values,
                args.per_device_windows,
                len(devices),
                input_dtype=compute_dtype,
            )
        return run_host_microbatches(
            layer_runner,
            params,
            values,
            args.evaluation_batch_windows,
            input_dtype=compute_dtype,
        )

    diagnostic_layers = set(target_layers) | {9, 19, 29, 39}
    for layer in range(1, source.num_layers):
        print(f"streaming_progressive_decoder_layer={layer}/{source.num_layers - 1}")
        ensure_layer_checkpoint(
            model_dir,
            index_payload["weight_map"],
            source,
            layer,
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
        )
        reader = QwenCheckpointReader(model_dir)
        layer_arrays = load_layer_arrays(reader, source, layer)
        teacher_params = jax_layer_params(layer_arrays, source, layer)
        teacher_hidden = run_teacher_layer(teacher_params, hidden)
        if layer in target_layers:
            print(f"applying_progressive_replacement_layer={layer}")
            branch_inputs = hidden.reshape(
                len(names), evaluation_windows, *hidden.shape[1:]
            )
            teacher_branches = teacher_hidden.reshape(
                len(names), evaluation_windows, *teacher_hidden.shape[1:]
            )
            next_branches = [teacher_branches[index] for index in range(len(names))]
            norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
            norm_runner = _create_norm_runner(norm)
            norm_params = {
                "scale": jnp.asarray(
                    layer_arrays[
                        f"model.layers.{layer}.input_layernorm.weight"
                    ],
                    dtype=jnp.float32,
                )
            }
            tail_params = qwen3_decoder_tail_params(reader, layer)
            endpoints, endpoint_metadata = _load_endpoint_bundle(
                index_path,
                endpoint_index["layers"][str(layer)],
                source_name=source_name,
                layer=layer,
                seeds=seeds,
            )
            endpoint_checkpoints[str(layer)] = endpoint_metadata
            for model_seed in seeds:
                included = [
                    index
                    for index, spec in enumerate(specs)
                    if spec["seed"] == model_seed and layer in spec["layers"]
                ]
                if not included:
                    continue
                values = np.concatenate(
                    [branch_inputs[index] for index in included], axis=0
                )
                normalized = run_host_microbatches(
                    norm_runner,
                    norm_params,
                    values,
                    args.evaluation_batch_windows,
                    input_dtype=compute_dtype,
                )
                replaced = _run_replacement_windows(
                    replacement_runner,
                    endpoints[str(model_seed)]["JOINT-MIXER-DECODER"],
                    tail_params,
                    jnp.asarray(values, dtype=compute_dtype),
                    jnp.asarray(normalized, dtype=compute_dtype),
                    args.evaluation_batch_windows,
                )
                chunks = np.split(replaced, len(included), axis=0)
                for branch_index, output in zip(included, chunks):
                    next_branches[branch_index] = output
            hidden = np.concatenate(next_branches, axis=0)
            del (
                branch_inputs,
                teacher_branches,
                next_branches,
                norm,
                norm_runner,
                norm_params,
                tail_params,
                endpoints,
            )
            jax.clear_caches()
            gc.collect()
        else:
            hidden = teacher_hidden
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError(
                f"non-finite progressive residual after layer {layer}"
            )
        if layer in diagnostic_layers:
            divergence[str(layer)] = branch_divergence(
                hidden, names, evaluation_windows
            )
        del teacher_hidden, teacher_params, layer_arrays, reader
        gc.collect()
        if args.prune_consumed_shards:
            removed_shards.extend(
                _safe_prune_shards(model_dir, last_use, layer)
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
    token_ids = np.asarray(arrays["token_ids"][evaluation_slice], dtype=np.int32)
    branch_values = hidden.reshape(
        len(names), evaluation_windows, *hidden.shape[1:]
    )
    lm_metrics = {}
    for index, name in enumerate(names):
        print(f"evaluating_progressive_lm_head={name}")
        lm_metrics[name] = _run_lm_metrics(
            lm_runner,
            final_norm_params,
            lm_head_kernel,
            branch_values[index],
            token_ids,
            devices=parallel_devices,
            per_device_windows=args.per_device_windows,
            compute_dtype=compute_dtype,
            return_window_nll=True,
        )
    raw_metrics_path = Path(args.result_json).with_name(
        f"{Path(args.result_json).stem}-lm-metrics.json"
    )
    evaluation_protocol = endpoint_index.get(
        "evaluation_protocol", "exp051-progressive-composition"
    )
    raw_metrics = {
        "source": source_name,
        "protocol": evaluation_protocol,
        "analysis_mode": analysis_mode,
        "branches": list(names),
        "lm_metrics": lm_metrics,
        "complete": True,
    }
    _write_json_with_output_mirror(
        raw_metrics_path, raw_metrics, args.output_dir
    )
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, source.num_layers)
        )

    standalone_metrics = {
        str(layer): endpoint_index["layers"][str(layer)][
            "standalone_joint_metrics"
        ]
        for layer in target_layers
    }
    analysis_function = {
        "progressive": analyze_progressive_composition,
        "boundary_scaling": analyze_boundary_scaling,
        "onset_localization": analyze_composition_onset,
        "exact_lift_pilot": analyze_exact_lift_pilot,
        "exact_lift_multiseed": analyze_exact_lift_multiseed,
        "exact_lift_scaling": analyze_exact_lift_scaling,
    }[analysis_mode]
    aggregate = analysis_function(
        seeds=seeds,
        layer_sets=layer_sets,
        original_metric=lm_metrics["ORIGINAL-CACHED-QWEN"],
        standalone_metrics=standalone_metrics,
        composed_metrics=lm_metrics,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        maximum_original_nll_difference=args.maximum_original_nll_difference,
        maximum_mean_inflation=args.maximum_mean_inflation,
        maximum_seed_inflation=args.maximum_seed_inflation,
        maximum_bootstrap_upper=args.maximum_bootstrap_upper,
    )
    result = {
        "source": source_name,
        "method": endpoint_index.get(
            "method", "asymmetric_budget_progressive_Mamba3_composition"
        ),
        "protocol": endpoint_index.get(
            "protocol", "exp051-progressive-2-4-8-layer-composition"
        ),
        "analysis_mode": analysis_mode,
        "endpoint_index": str(index_path),
        "endpoint_index_sha256": file_sha256(index_path),
        "target_layers": list(target_layers),
        "layer_sets": {str(key): list(value) for key, value in layer_sets.items()},
        "branch_sets": (
            {key: list(value) for key, value in branch_sets.items()}
            if branch_sets is not None
            else None
        ),
        "seeds": list(seeds),
        "sequence_length": sequence_length,
        "evaluation_windows": evaluation_windows,
        "evaluation_tokens_per_branch": evaluation_windows * (sequence_length - 1),
        "evaluation_cache_manifest": str(evaluation_manifest_path),
        "resolved_evaluation_artifacts": {
            key: str(value) for key, value in resolved_paths.items()
        },
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "branches": list(names),
        "hidden_relative_l2_to_original": divergence,
        "lm_metrics": lm_metrics,
        "endpoint_checkpoints": endpoint_checkpoints,
        "aggregate": aggregate,
        "removed_checkpoint_shards": sorted(set(removed_shards)),
        "scientific_gate_passed": aggregate["scientific_gate_passed"],
        "passed": aggregate["all_finite"],
        "notes": endpoint_index.get(
            "notes",
            [
                "Replacement sets are nested and use independently trained JOINT endpoints for each seed.",
                "Additive expected excess NLL uses paired standalone evaluations on the same validation windows.",
            ],
        ),
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, args.output_dir
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    print(f"raw_lm_metrics_json={raw_metrics_path.resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if not result["passed"]:
        raise SystemExit("COMPOSITION-EVALUATION-NONFINITE")
    print(
        "COMPOSITION-SCIENTIFIC-GATE-PASS"
        if result["scientific_gate_passed"]
        else "COMPOSITION-SCIENTIFIC-GATE-FAIL"
    )
    return result


if __name__ == "__main__":
    main()
