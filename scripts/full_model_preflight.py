from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from extent import HybridForCausalLM
from extent.config import load_config
from extent.initialization import (
    ShardedParameters,
    abstract_parameter_tree,
    initialize_sharded_optimizer_state,
    initialize_sharded_parameters,
)
from extent.optimizer import create_lion
from extent.preflight import allocated_bytes_by_device, build_preflight_report
from extent.qwen_source import (
    QWEN3_14B,
    validate_source_marker,
    validate_source_metadata,
)
from extent.sharding import batch_sharding, create_v5e_mesh
from extent.weight_mapping import (
    QwenCheckpointReader,
    stream_direct_qwen_weights,
    validate_local_direct_shapes,
)


def _gib(value: int) -> float:
    return value / 2**30


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shape-only 14B v5e-8 audit and guarded parameter init.")
    parser.add_argument("--config", default="config/hybrid_14b_v5e8.yaml")
    parser.add_argument("--initialize-params", action="store_true")
    parser.add_argument(
        "--initialize-optimizer",
        action="store_true",
        help="Also allocate the Lion state; implies --initialize-params.",
    )
    parser.add_argument("--init-sequence-length", type=int, default=1)
    parser.add_argument("--hbm-per-device-gib", type=float, default=16.0)
    parser.add_argument(
        "--qwen-model-dir",
        help="Import pinned Qwen embeddings/MLPs/norms/head after sharded initialization.",
    )
    args = parser.parse_args(argv)

    config, extras = load_config(args.config)
    axes = tuple(extras["mesh"]["axes"])
    shape = tuple(int(size) for size in extras["mesh"]["shape"])
    axis_sizes = dict(zip(axes, shape))
    model = HybridForCausalLM(config)

    print("Tracing exact parameter shapes (no model arrays are allocated)...")
    abstract_params = abstract_parameter_tree(model, batch_size=shape[0], sequence_length=1)
    report = build_preflight_report(abstract_params, axis_sizes)
    headroom = args.hbm_per_device_gib - report.training_gib_per_device
    print(f"target_mesh={axis_sizes} mesh_size={report.mesh_size}")
    print(f"exact_parameters={report.parameter_count:,} tensors={report.tensor_count}")
    print(f"partitioned_tensors={report.partitioned_tensor_count}/{report.tensor_count}")
    print(f"global_bf16_weights={report.global_weight_gib:.3f} GiB")
    print(f"weights_per_device={report.weight_gib_per_device:.3f} GiB")
    print(f"weights_grad_lion_per_device={report.training_gib_per_device:.3f} GiB")
    print(f"persistent_headroom_at_{args.hbm_per_device_gib:g}GiB={headroom:.3f} GiB")
    print("largest local tensors:")
    for tensor in report.largest_tensors:
        print(
            f"  {tensor.path} shape={tensor.shape} shards={tensor.partitions} "
            f"local={_gib(tensor.bytes_per_device):.3f} GiB"
        )
    if headroom < 4.0:
        print("verdict=NO-GO: less than 4 GiB remains for activations/XLA temporaries")
    else:
        print("verdict=PREFLIGHT-PASS: persistent state fits; a real HBM check is still required")

    if not (args.initialize_params or args.initialize_optimizer or args.qwen_model_dir):
        print("mode=shape-only; no full model arrays were allocated")
        return

    devices = list(jax.devices())
    if len(devices) != report.mesh_size:
        raise RuntimeError(
            f"refusing full initialization: need {report.mesh_size} devices, found {len(devices)}"
        )
    mesh = create_v5e_mesh(devices)
    if dict(mesh.shape) != axis_sizes:
        raise RuntimeError(f"actual mesh {dict(mesh.shape)} does not match target {axis_sizes}")
    host_tokens = np.zeros((mesh.shape["data"], args.init_sequence_length), dtype=np.int32)
    tokens = jax.device_put(host_tokens, batch_sharding(mesh))
    print("initializing parameters directly into device shards...")
    initialized = initialize_sharded_parameters(model, jax.random.key(0), tokens, mesh)
    jax.block_until_ready(initialized.params)
    allocated = allocated_bytes_by_device(initialized.params)
    print("allocated_parameter_bytes_by_device:")
    for device, size in sorted(allocated.items()):
        print(f"  {device}: {_gib(size):.3f} GiB")
    print("initialization=PASS (parameters only; optimizer and train step were not created)")

    if args.qwen_model_dir:
        qwen_dir = Path(args.qwen_model_dir)
        validate_source_marker(qwen_dir, QWEN3_14B)
        source_config = json.loads((qwen_dir / "config.json").read_text(encoding="utf-8"))
        source_index = json.loads(
            (qwen_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        validate_source_metadata(source_config, source_index, QWEN3_14B)
        reader = QwenCheckpointReader(qwen_dir)
        validate_local_direct_shapes(reader, config)
        print("streaming direct Qwen tensors into their final device shards...")
        def report_import_progress(index, total, entry):
            if index == 1 or index % 16 == 0 or index == total:
                print(f"  imported={index}/{total} latest={entry.source}")

        loaded_params, mapping_report = stream_direct_qwen_weights(
            initialized.params,
            reader,
            config,
            progress=report_import_progress,
        )
        jax.block_until_ready(loaded_params)
        initialized = ShardedParameters(
            loaded_params, initialized.layout, initialized.abstract_params
        )
        print(
            f"qwen_direct_import=PASS tensors={mapping_report.tensor_count} "
            f"parameters={mapping_report.parameter_count:,}"
        )

    if not args.initialize_optimizer:
        return

    training = extras["training"]
    if int(training.get("gradient_accumulation_steps", 1)) != 1:
        raise RuntimeError(
            "optimizer HBM probe requires gradient_accumulation_steps=1; "
            "MultiSteps adds a full gradient buffer"
        )
    tx = create_lion(
        learning_rate=float(training["learning_rate"]),
        warmup_steps=int(training["warmup_steps"]),
        total_steps=int(training["max_steps"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        accumulation_steps=1,
    )
    print("initializing Lion state directly into device shards...")
    initialized_optimizer = initialize_sharded_optimizer_state(
        tx,
        initialized.params,
        initialized.abstract_params,
        initialized.layout,
        mesh,
    )
    jax.block_until_ready(initialized_optimizer.opt_state)
    optimizer_allocated = allocated_bytes_by_device(initialized_optimizer.opt_state)
    print("allocated_optimizer_bytes_by_device:")
    for device, size in sorted(optimizer_allocated.items()):
        print(f"  {device}: {_gib(size):.3f} GiB")
    print("allocated_persistent_bytes_by_device:")
    for device in sorted(set(allocated) | set(optimizer_allocated)):
        total = allocated.get(device, 0) + optimizer_allocated.get(device, 0)
        print(f"  {device}: {_gib(total):.3f} GiB")
    print(
        "optimizer_initialization=PASS "
        "(weights + Lion state only; no gradients, activations, or train step)"
    )


if __name__ == "__main__":
    main()
