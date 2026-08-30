from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import time
import traceback

import jax
import numpy as np

from scripts.qwen_activation_cache import main as build_cache
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from extent import HybridForCausalLM
from extent.config import load_config
from extent.full_hybrid_materialization import (
    materialize_exact_lift_layer,
    materialize_rorope_bkv_layer,
)
from extent.initialization import (
    initialize_sharded_optimizer_state,
    initialize_sharded_parameters,
)
from extent.model import causal_lm_loss
from extent.optimizer import create_lion
from extent.preflight import allocated_bytes_by_device
from extent.qwen3_parity import load_mixer_arrays
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata, write_source_marker
from extent.sharding import batch_sharding, create_v5e_mesh, replicated_sharding
from extent.teacher_activation_cache import load_activation_cache
from extent.weight_mapping import QwenCheckpointReader, stream_direct_qwen_weights


PROTOCOL = "exp062-full-hybrid-materialization-bringup"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gib(value: int) -> float:
    return value / 2**30


def _write_partial(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ensure_checkpoint(model_dir: Path) -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
            filename=filename,
            local_dir=model_dir,
        )
    config_payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index_payload = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    for index, filename in enumerate(sorted(set(index_payload["weight_map"].values())), start=1):
        hf_hub_download(
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
            filename=filename,
            local_dir=model_dir,
        )
        print(f"checkpoint_shard={index}/{QWEN3_14B.shard_count} READY")
    write_source_marker(model_dir, QWEN3_14B)
    return config_payload, index_payload


def _cache_manifest(output: Path, layer: int) -> Path:
    return output / f"exp062-mla-layer{layer}-cache" / "manifest.json"


def _ensure_mla_cache(
    output: Path,
    qwen_dir: Path,
    dataset_dir: Path,
    layer: int,
    calibration_windows: int,
    sequence_length: int,
) -> Path:
    directory = output / f"exp062-mla-layer{layer}-cache"
    manifest = _cache_manifest(output, layer)
    if manifest.exists():
        try:
            cached, _, _ = load_activation_cache(manifest, artifact_dir=directory)
            if (
                cached["target_layer"] == layer
                and cached["sequence_length"] == sequence_length
                and cached["window_layout"]["calibration"] == [0, calibration_windows]
            ):
                print(f"mla_calibration_layer={layer} RESUME-PASS")
                return manifest
        except Exception as exc:
            print(f"mla_calibration_layer={layer} RESUME-REJECT {type(exc).__name__}: {exc}")
    directory.mkdir(parents=True, exist_ok=True)
    build_cache([
        "--cache-dir", str(qwen_dir), "--dataset-cache-dir", str(dataset_dir),
        "--target-layer", str(layer), "--sequence-length", str(sequence_length),
        "--calibration-windows", str(calibration_windows), "--training-windows", "1",
        "--evaluation-windows", "1", "--token-offset", "393216",
        "--data-parallel", "--per-device-windows", "2", "--compute-dtype", "bfloat16",
        "--storage-dtype", "float16", "--output-dir", str(directory),
        "--result-json", str(manifest),
    ])
    return manifest


def _forward_probe(model, params, layout, mesh, length: int) -> dict:
    tokens = np.arange(length, dtype=np.int32)[None] % model.config.vocab_size
    host = {
        "input_ids": tokens,
        "labels": tokens.copy(),
        "attention_mask": np.ones_like(tokens, dtype=np.bool_),
        "loss_mask": np.ones_like(tokens, dtype=np.bool_),
    }
    batch_layout = batch_sharding(mesh)
    batch = {name: jax.device_put(value, batch_layout) for name, value in host.items()}

    def loss_fn(candidate, values):
        logits = model.apply(
            {"params": candidate},
            values["input_ids"],
            attention_mask=values["attention_mask"],
        )
        return causal_lm_loss(logits, values["labels"], values["loss_mask"])

    compiled = jax.jit(
        loss_fn,
        in_shardings=(layout, {name: batch_layout for name in batch}),
        out_shardings=replicated_sharding(mesh),
    )
    started = time.monotonic()
    loss = compiled(params, batch)
    jax.block_until_ready(loss)
    return {
        "sequence_length": length,
        "loss": float(loss),
        "finite": bool(np.isfinite(float(loss))),
        "compile_and_execute_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Materialize and probe the full Extent-14B hybrid.")
    parser.add_argument("--config", default="config/hybrid_14b_v5e8.yaml")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-exp062-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--calibration-windows", type=int, default=16)
    parser.add_argument("--calibration-sequence-length", type=int, default=32)
    parser.add_argument("--probe-contexts", default="8,32,128")
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-062 requires one TPU v5e-8")
    if args.calibration_windows * args.calibration_sequence_length < 448:
        raise ValueError("RoRoPE-BKV rank 448 requires at least 448 calibration tokens")
    contexts = tuple(sorted({int(value) for value in args.probe_contexts.split(",")}))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    qwen_dir, dataset_dir = Path(args.qwen_cache_dir), Path(args.dataset_cache_dir)
    result_path = output / "extent-full-hybrid-materialization.json"
    partial_path = output / "exp062-materialization-partial.json"
    started_clock, started, stage = time.monotonic(), _now(), "startup"
    _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=started\nexperiment=EXP-062 full hybrid materialization\nhost={socket.gethostname()}")
    result = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at_utc": started,
        "stage": stage,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "mixer_reports": [],
        "forward_probes": [],
    }
    try:
        config, extras = load_config(args.config)
        source = Qwen3TeacherConfig(param_dtype="float32", compute_dtype="bfloat16", remat_policy="none")
        stage = "mla-calibration-caches"
        manifests = {}
        for layer in config.attention_layer_indices:
            manifests[layer] = _ensure_mla_cache(output, qwen_dir, dataset_dir, layer, args.calibration_windows, args.calibration_sequence_length)
        stage = "checkpoint-download"
        _, index_payload = _ensure_checkpoint(qwen_dir)
        reader = QwenCheckpointReader(qwen_dir)

        stage = "sharded-random-initialization"
        mesh = create_v5e_mesh(devices)
        model = HybridForCausalLM(config)
        init_tokens = jax.device_put(np.zeros((mesh.shape["data"], 1), np.int32), batch_sharding(mesh))
        initialized = initialize_sharded_parameters(model, jax.random.key(620), init_tokens, mesh)
        params = initialized.params
        stage = "direct-qwen-import"
        params, direct_report = stream_direct_qwen_weights(params, reader, config)
        result["direct_mapping"] = {"tensor_count": direct_report.tensor_count, "parameter_count": direct_report.parameter_count}

        for layer in range(config.num_layers):
            stage = f"mixer-layer-{layer}"
            arrays = load_mixer_arrays(reader, source, layer)
            if layer in config.attention_layer_indices:
                manifest, cache, _ = load_activation_cache(manifests[layer], artifact_dir=manifests[layer].parent)
                start, stop = manifest["window_layout"]["calibration"]
                calibration = np.asarray(cache["normalized_input"][start:stop], dtype=np.float32)
                params, report = materialize_rorope_bkv_layer(params, arrays, calibration, source, config, layer)
                del cache, calibration
            else:
                params, report = materialize_exact_lift_layer(params, arrays, source, config, layer)
            result["mixer_reports"].append(report.to_dict())
            result["stage"] = stage
            _write_partial(partial_path, result)
            print(f"materialized_layer={layer}/39 method={report.method}")
            del arrays
            gc.collect()

        stage = "optimizer-allocation"
        training = extras["training"]
        tx = create_lion(learning_rate=float(training["learning_rate"]), warmup_steps=int(training["warmup_steps"]), total_steps=int(training["max_steps"]), weight_decay=float(training["weight_decay"]), max_grad_norm=float(training["max_grad_norm"]), accumulation_steps=1)
        optimizer = initialize_sharded_optimizer_state(tx, params, initialized.abstract_params, initialized.layout, mesh)
        jax.block_until_ready(optimizer.opt_state)
        result["allocated_parameter_gib_by_device"] = {device: _gib(size) for device, size in allocated_bytes_by_device(params).items()}
        result["allocated_optimizer_gib_by_device"] = {device: _gib(size) for device, size in allocated_bytes_by_device(optimizer.opt_state).items()}

        stage = "forward-probes"
        for length in contexts:
            probe = _forward_probe(model, params, initialized.layout, mesh, length)
            result["forward_probes"].append(probe)
            _write_partial(partial_path, result)
            print(f"forward_probe_context={length} loss={probe['loss']:.6f} finite={probe['finite']}")
            if not probe["finite"]:
                raise FloatingPointError(f"non-finite full-hybrid forward at context {length}")
        result.update({"status": "completed", "passed": True, "stage": "completed", "completed_at_utc": _now(), "duration_hours": (time.monotonic() - started_clock) / 3600, "source_tensor_count": len(index_payload["weight_map"]), "materialized_mamba_layers": len(config.mamba_layer_indices), "materialized_mla_layers": len(config.attention_layer_indices)})
        _write_json_with_output_mirror(result_path, result, str(output))
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=completed\nexperiment=EXP-062\nduration_hours={result['duration_hours']:.3f}\ncontexts={contexts}")
        print(f"EXP062-MATERIALIZATION-PASS\nresult_json={result_path.resolve()}")
        return result
    except BaseException as exc:
        result.update({"status": "failed", "passed": False, "stage": stage, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "completed_at_utc": _now(), "duration_hours": (time.monotonic() - started_clock) / 3600})
        _write_json_with_output_mirror(output / "exp062-materialization-failure.json", result, str(output))
        _safe_notify(args.telegram, f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-062\nstage={stage}\nerror={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
