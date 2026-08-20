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
from extent.layers.mamba3 import Mamba3MIMO
from extent.offline_distillation import restore_offline_checkpoint
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


def _run_windows(runner, args: tuple, windows: int, batch_windows: int):
    mixer_outputs = []
    decoder_outputs = []
    for start in range(0, windows, batch_windows):
        stop = min(start + batch_windows, windows)
        batch_args = tuple(value[start:stop] if hasattr(value, "shape") and value.shape[0] == windows else value for value in args)
        mixer, decoder = runner(*batch_args)
        jax.block_until_ready((mixer, decoder))
        mixer_outputs.append(np.asarray(mixer, dtype=np.float32))
        decoder_outputs.append(np.asarray(decoder, dtype=np.float32))
    return np.concatenate(mixer_outputs), np.concatenate(decoder_outputs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure full decoder-layer shock from a recovered offline Mamba checkpoint."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--qwen-cache-dir", default="/content/qwen3-decoder-tail")
    parser.add_argument("--evaluation-batch-windows", type=int, default=2)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/content/output")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--compute-dtype", choices=("auto", "float32", "bfloat16"), default="auto"
    )
    args = parser.parse_args(argv)
    if args.evaluation_batch_windows < 1:
        raise ValueError("evaluation-batch-windows must be positive")

    manifest, arrays, paths = load_activation_cache(
        args.activation_cache_manifest,
        artifact_dir=args.activation_cache_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if manifest.get("source") != source_name:
        raise ValueError("activation cache does not match the pinned Qwen3 source")
    checkpoint_metadata = json.loads(
        (Path(args.checkpoint_dir) / "checkpoint.json").read_text(encoding="utf-8")
    )
    compatibility = checkpoint_metadata["compatibility"]
    required_contract = {
        "source": source_name,
        "dataset": manifest.get("dataset"),
        "target_layer": int(manifest["target_layer"]),
        "sequence_length": int(manifest["sequence_length"]),
        "normalized_input_sha256": manifest["artifacts"]["normalized_input"]["sha256"],
        "attention_target_sha256": manifest["artifacts"]["attention_target"]["sha256"],
    }
    mismatches = {
        key: (compatibility.get(key), expected)
        for key, expected in required_contract.items()
        if compatibility.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint and activation cache mismatch: {mismatches}")

    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    if compatibility["compute_dtype"] != dtype_decision.dtype:
        raise ValueError(
            "evaluation compute dtype must match the recovered checkpoint dtype"
        )
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba_config = Mamba3Config(**compatibility["mamba_config"])
    mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=compute_dtype,
        param_dtype=compute_dtype,
    )
    layout = manifest["window_layout"]
    calibration_slice = _slice(layout, "calibration")
    evaluation_slice = _slice(layout, "evaluation")
    normalized = arrays["normalized_input"]
    calibration_inputs = jnp.asarray(normalized[calibration_slice], dtype=compute_dtype)
    calibration_targets = np.asarray(
        arrays["attention_target"][calibration_slice], dtype=np.float32
    )
    base_params = mamba.init(
        jax.random.key(int(compatibility["seed"])), calibration_inputs[:1]
    )["params"]
    run_features = jax.jit(
        lambda params, inputs: mamba.apply(
            {"params": params}, inputs, return_features=True
        )[1]
    )
    features = np.asarray(run_features(base_params, calibration_inputs), dtype=np.float32)
    readout, readout_report = fit_dual_ridge_readout(
        jnp.asarray(features.reshape(-1, features.shape[-1])),
        jnp.asarray(calibration_targets.reshape(-1, source.hidden_size)),
        relative_ridge=float(compatibility["readout_ridge"]),
    )
    calibrated_params = _replace_output(base_params, readout, compute_dtype)
    tx = create_lion(
        learning_rate=float(compatibility["learning_rate"]),
        warmup_steps=min(2000, max(1, int(compatibility["total_steps"]) // 100)),
        total_steps=int(compatibility["total_steps"]),
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    recovered_params, _, recovered_step, restored_metadata = restore_offline_checkpoint(
        args.checkpoint_dir,
        base_params,
        tx.init(base_params),
        expected_compatibility=compatibility,
    )
    if recovered_step != int(compatibility["total_steps"]):
        raise ValueError("decoder evaluation requires a completed offline checkpoint")

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
    teacher_runner = create_batched_teacher_tail_runner(tail)
    replacement_runner = create_batched_replacement_runner(mamba, tail)

    residual_eval = jnp.asarray(arrays["residual_input"][evaluation_slice], dtype=compute_dtype)
    normalized_eval = jnp.asarray(normalized[evaluation_slice], dtype=compute_dtype)
    target_eval = jnp.asarray(arrays["attention_target"][evaluation_slice], dtype=compute_dtype)
    windows = int(residual_eval.shape[0])
    teacher_decoder = []
    for start in range(0, windows, args.evaluation_batch_windows):
        stop = min(start + args.evaluation_batch_windows, windows)
        output = teacher_runner(
            tail_params, residual_eval[start:stop], target_eval[start:stop]
        )
        teacher_decoder.append(np.asarray(output, dtype=np.float32))
    teacher_decoder = np.concatenate(teacher_decoder)
    target_eval_np = np.asarray(target_eval, dtype=np.float32)

    variants = {
        "INIT-A-random": base_params,
        "INIT-A-readout-calibrated-step0": calibrated_params,
        f"OFFLINE-recovered-step{recovered_step}": recovered_params,
    }
    results = {}
    passed = True
    for name, params in variants.items():
        mixer_output, decoder_output = _run_windows(
            replacement_runner,
            (params, tail_params, residual_eval, normalized_eval),
            windows,
            args.evaluation_batch_windows,
        )
        finite = bool(
            np.all(np.isfinite(mixer_output))
            and np.all(np.isfinite(decoder_output))
        )
        passed &= finite
        results[name] = {
            "mixer_output": _window_metrics(target_eval_np, mixer_output),
            "decoder_output": _window_metrics(teacher_decoder, decoder_output),
            "finite": finite,
        }
        print(
            f"{name}=DONE mixer_l2={results[name]['mixer_output']['relative_l2']:.6g} "
            f"decoder_l2={results[name]['decoder_output']['relative_l2']:.6g}"
        )

    result = {
        "source": source_name,
        "method": "offline_recovered_Mamba3_full_decoder_shock",
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {name: str(path) for name, path in paths.items()},
        "target_layer": int(manifest["target_layer"]),
        "sequence_length": int(manifest["sequence_length"]),
        "evaluation_windows": windows,
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "checkpoint_step": recovered_step,
        "checkpoint_sha256": restored_metadata["checkpoint_sha256"],
        "readout_calibration_relative_l2": float(
            readout_report.calibration_relative_l2
        ),
        "variants": results,
        "passed": passed,
        "notes": [
            "Every arm uses the identical frozen Qwen residual, post-attention RMSNorm, and MLP path.",
            "Teacher decoder targets are reconstructed from cached exact attention outputs plus frozen Qwen decoder-tail weights.",
            "The evaluation split is disjoint from readout calibration and offline training windows.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, args.output_dir
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if not passed:
        raise SystemExit("RECOVERED-MAMBA-DECODER-EVAL-FAIL")
    print("RECOVERED-MAMBA-DECODER-EVAL-PASS")


if __name__ == "__main__":
    main()
