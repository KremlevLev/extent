from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from singularity.calibration_data import WIKITEXT_REPO, WIKITEXT_REVISION, load_wikitext2_tokens
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
from singularity.qwen3_parity import ensure_layer_checkpoint, jax_layer_params, load_layer_arrays, parity_metrics
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.readout_calibration import fit_dual_ridge_readout
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _replace_output(params, kernel, dtype):
    mutable = unfreeze(params)
    mutable["out_proj"]["kernel"] = kernel.astype(dtype)
    return freeze(jax.tree.map(lambda value: value.astype(dtype), mutable))


def _window_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    return asdict(parity_metrics(targets.reshape(-1, targets.shape[-1]), predictions.reshape(-1, predictions.shape[-1])))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a controlled one-layer Qwen3-to-Mamba3 distillation pilot.")
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/singularity-calibration-cache")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--evaluation-windows", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--compute-dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--variants", default="INIT-A-random,INIT-C-prior-qkvo-port")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    if min(args.sequence_length, args.calibration_windows, args.steps, args.evaluation_windows) < 1:
        raise ValueError("sequence length, window counts, and steps must be positive")
    if args.readout_ridge <= 0:
        raise ValueError("readout ridge must be positive")
    jax.config.update("jax_default_matmul_precision", "high")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    parameter_dtype = compute_dtype
    selected_variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not selected_variants:
        raise ValueError("at least one distillation variant is required")
    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(param_dtype="float32", compute_dtype=dtype_decision.dtype, remat_policy="none")
    mamba_config = Mamba3Config()

    window_count = args.calibration_windows + args.steps + args.evaluation_windows
    token_count = window_count * args.sequence_length
    tokens = load_wikitext2_tokens(token_count, args.dataset_cache_dir, tokenizer_repo=spec.repo_id, tokenizer_revision=spec.revision)
    model_dir, shards = ensure_layer_checkpoint(args.cache_dir, index_payload["weight_map"], source, args.layer_index, repo_id=spec.repo_id, revision=spec.revision)
    embedding_shard = index_payload["weight_map"]["model.embed_tokens.weight"]
    if embedding_shard not in shards:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=spec.repo_id, revision=spec.revision, filename=embedding_shard, local_dir=model_dir)
    reader = QwenCheckpointReader(model_dir)
    arrays = load_layer_arrays(reader, source, args.layer_index)
    teacher_params = jax_layer_params(arrays, source, args.layer_index)
    hidden = jnp.asarray(reader.read_rows("model.embed_tokens.weight", tokens)).reshape(window_count, args.sequence_length, source.hidden_size)
    normalized = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": teacher_params["input_layernorm"]["scale"]}}, hidden
    ).astype(compute_dtype)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    teacher_module = Qwen3GQAAttention(source)
    run_teacher = create_teacher_mixer_runner(teacher_module, positions, mask)
    print(f"precomputing_teacher_windows={window_count}")
    teacher_targets = np.stack([
        np.asarray(
            run_teacher(teacher_params["self_attn"], normalized[index:index + 1])[0],
            dtype=np.float32,
        )
        for index in range(window_count)
    ])
    del teacher_params, run_teacher, teacher_module
    jax.clear_caches()

    mamba = Mamba3MIMO(source.hidden_size, mamba_config, dtype=compute_dtype, param_dtype=parameter_dtype)
    base = mamba.init(jax.random.key(args.seed), normalized[:1])["params"]
    variants, reports = build_qwen3_to_mamba3_transplant_variants(base, arrays, source, mamba_config, args.layer_index)
    unknown = sorted(set(selected_variants) - set(variants))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(variants)}")
    apply_mamba = lambda params, inputs: mamba.apply({"params": params}, inputs)
    run_features = jax.jit(lambda params, inputs: mamba.apply({"params": params}, inputs, return_features=True)[1])
    run_mamba = jax.jit(apply_mamba)
    calibration_slice = slice(0, args.calibration_windows)
    training_start = args.calibration_windows
    evaluation_slice = slice(training_start + args.steps, window_count)
    calibration_inputs = normalized[calibration_slice]
    evaluation_inputs = normalized[evaluation_slice]
    calibration_targets = teacher_targets[calibration_slice]
    evaluation_targets = teacher_targets[evaluation_slice]
    results = {}
    passed = True
    tx = create_lion(
        learning_rate=args.learning_rate,
        warmup_steps=min(2, max(0, args.steps - 1)),
        total_steps=args.steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    train_step = create_layerwise_train_step(
        apply_mamba,
        tx,
        bf16_gradients=parameter_dtype == jnp.bfloat16,
    )

    for name in selected_variants:
        params = variants[name]
        calibration_features = np.asarray(run_features(params, calibration_inputs), dtype=np.float32)
        readout, readout_report = fit_dual_ridge_readout(
            jnp.asarray(calibration_features.reshape(-1, calibration_features.shape[-1])),
            jnp.asarray(calibration_targets.reshape(-1, source.hidden_size)),
            relative_ridge=args.readout_ridge,
        )
        params = _replace_output(params, readout, parameter_dtype)
        pre = np.asarray(run_mamba(params, evaluation_inputs), dtype=np.float32)
        opt_state = tx.init(params)
        curve = []
        finite = True
        for step in range(args.steps):
            index = training_start + step
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                normalized[index:index + 1],
                jnp.asarray(teacher_targets[index:index + 1], dtype=compute_dtype),
            )
            jax.block_until_ready(metrics)
            record = {key: float(value) if key not in {"grads_finite", "nonfinite_grad_leaves"} else int(value) for key, value in metrics.items()}
            record["step"] = step
            curve.append(record)
            finite = (
                finite
                and bool(metrics["grads_finite"])
                and bool(np.isfinite(record["loss"]))
            )
            print(f"{name} step={step} loss={record['loss']:.6g} grad_norm={record['grad_norm']:.6g} finite={bool(metrics['grads_finite'])}")
        post = np.asarray(run_mamba(params, evaluation_inputs), dtype=np.float32)
        finite &= bool(np.all(np.isfinite(post)))
        passed = passed and finite
        results[name] = {
            "mapping": asdict(reports[name]),
            "readout_calibration": asdict(readout_report),
            "pre_distillation_heldout": _window_metrics(evaluation_targets, pre),
            "post_distillation_heldout": _window_metrics(evaluation_targets, post),
            "loss_curve": curve,
            "finite": finite,
        }
        print(f"{name}=DONE pre_l2={results[name]['pre_distillation_heldout']['relative_l2']:.6g} post_l2={results[name]['post_distillation_heldout']['relative_l2']:.6g}")
        if args.result_json:
            partial = Path(args.result_json).with_suffix(".partial.json")
            _write_json(
                partial,
                {
                    "status": "in_progress",
                    "source": f"{spec.repo_id}@{spec.revision}",
                    "method": "trainable_Mamba3_layerwise_distillation_pilot",
                    "completed_variants": results,
                },
            )
            print(f"partial_result_json={partial.resolve()}")

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "trainable_Mamba3_layerwise_distillation_pilot",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "calibration_windows": args.calibration_windows,
        "training_steps": args.steps,
        "training_tokens_per_variant": args.steps * args.sequence_length,
        "evaluation_windows": args.evaluation_windows,
        "learning_rate": args.learning_rate,
        "readout_relative_ridge": args.readout_ridge,
        "seed": args.seed,
        "compute_dtype": dtype_decision.dtype,
        "dtype_reason": dtype_decision.reason,
        "jax_backend": jax.default_backend(),
        "mamba_config": asdict(mamba_config),
        "trainable_parameters": int(sum(value.size for value in jax.tree.leaves(params))),
        "variants": results,
        "passed": passed,
        "notes": [
            "Calibration, training, and evaluation use disjoint contiguous WikiText windows.",
            "Every variant receives an independently ridge-calibrated output projection before training.",
            "All Mamba parameters are trainable with identical Lion settings and token budgets.",
            "T4 uses FP32 for numerical safety; TPU and BF16-capable accelerators use BF16 parameters, compute, gradients, and Lion momentum.",
        ],
    }
    serialized = json.dumps(result, indent=2, default=_json_default)
    print(serialized)
    if args.result_json:
        output = Path(args.result_json)
        _write_json(output, result)
        print(f"result_json={output.resolve()}")
    if not passed:
        raise SystemExit("MAMBA3-DISTILL-PILOT-FAIL")
    print("MAMBA3-DISTILL-PILOT-PASS")


if __name__ == "__main__":
    main()
