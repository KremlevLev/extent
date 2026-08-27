from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
from urllib import request

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
from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from extent.qwen3_parity import ensure_layer_checkpoint, load_mixer_arrays
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.teacher_activation_cache import load_activation_cache, validate_external_evaluation_cache
from extent.weight_mapping import QwenCheckpointReader


PROTOCOL = "exp056-operator-preserving-mimo-lift-screen"
ARM_ORDER = (
    "CONTROL-RANDOM",
    "CONTROL-FLAT-QKVO",
    "SINGLE-CHANNEL-LIFT",
    "BALANCED-RANK-LIFT",
)
VARIANT_BY_ARM = {
    "CONTROL-RANDOM": "INIT-A-random",
    "CONTROL-FLAT-QKVO": "INIT-C-prior-qkvo-port",
    "SINGLE-CHANNEL-LIFT": "INIT-J-single-channel-qkvo-lift",
    "BALANCED-RANK-LIFT": "INIT-K-balanced-qkvo-lift",
}


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


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
    parser.add_argument("--total-steps", type=int, default=4096)
    parser.add_argument("--checkpoints", default="0,256,1024,2048,4096")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
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
    if not arms or len(arms) != len(set(arms)) or any(arm not in ARM_ORDER for arm in arms):
        raise ValueError("arms must be unique registered MIMO-lift arms")
    if "CONTROL-RANDOM" not in arms:
        raise ValueError("arms must include CONTROL-RANDOM")

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
    source = Qwen3TeacherConfig(param_dtype="float32", compute_dtype=dtype_decision.dtype, remat_policy="none")
    mamba_config = Mamba3Config()
    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(QWEN3_14B.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir, index_payload["weight_map"], source, layer,
        repo_id=QWEN3_14B.repo_id, revision=QWEN3_14B.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    mixer_arrays = load_mixer_arrays(reader, source, layer)
    tail_params = qwen3_decoder_tail_params(reader, layer)
    mamba = Mamba3MIMO(source.hidden_size, mamba_config, dtype=compute_dtype, param_dtype=compute_dtype)
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
            calibrated, readout_report = _calibrate_readout(
                mamba, variants[variant], calibration_inputs, calibration_targets,
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
            )
            seed_record["arms"][arm] = {
                "variant": variant, "initializer": asdict(reports[variant]),
                "readout_calibration": readout_report, "recovery": recovery,
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
        "source": f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}",
        "method": "operator_preserving_siso_to_mimo_qkvo_lift",
        "target_layer": layer, "sequence_length": int(train_manifest["sequence_length"]),
        "seeds_requested": list(seeds), "arm_order": list(arms),
        "training": {"recovery_steps_per_arm": args.total_steps, "checkpoints": list(checkpoints)},
        "compute_dtype": dtype_decision.dtype, "mamba_config": asdict(mamba_config),
        "activation_cache_manifest": str(Path(args.activation_cache_manifest).resolve()),
        "evaluation_cache_manifest": str(Path(args.evaluation_cache_manifest).resolve()),
        "resolved_activation_artifacts": {k: str(v) for k, v in train_paths.items()},
        "resolved_evaluation_artifacts": {k: str(v) for k, v in eval_paths.items()},
        "seeds": seed_results, "complete": complete, "passed": numerical_pass,
        "notes": [
            "All arms use paired data order, readout calibration, optimizer, and decoder-aware recovery.",
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
