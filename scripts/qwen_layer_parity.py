from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
    parity_metrics,
    required_layer_shards,
    torch_layer_state,
)
from singularity.qwen3_teacher import Qwen3DecoderLayer, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata, write_source_marker
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _torch_reference(
    arrays: dict[str, np.ndarray],
    hidden_states: np.ndarray,
    config_path: Path,
    layer_index: int,
) -> np.ndarray:
    import torch
    import transformers
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

    if transformers.__version__ != "4.51.0":
        raise RuntimeError(
            "transformers==4.51.0 is required for the pinned reference; "
            f"got {transformers.__version__}"
        )
    config = Qwen3Config.from_json_file(str(config_path))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config, layer_idx=layer_index)
    state = torch_layer_state(arrays, Qwen3TeacherConfig(), layer_index)
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval()

    hidden = torch.from_numpy(hidden_states)
    positions = torch.arange(hidden.shape[1], dtype=torch.long)[None, :]
    rotary = Qwen3RotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions)
    minimum = torch.finfo(torch.float32).min
    causal_mask = torch.full(
        (1, 1, hidden.shape[1], hidden.shape[1]), minimum, dtype=torch.float32
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    with torch.inference_mode():
        output = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )[0]
    return output.detach().cpu().numpy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare one pinned Qwen3 layer in PyTorch and JAX."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-3)
    parser.add_argument("--relative-l2-tolerance", type=float, default=5e-4)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)

    if args.sequence_length < 1:
        raise ValueError("sequence length must be positive")
    spec = QWEN3_14B
    source_config = _read_json(spec.resolve_url("config.json"))
    source_index = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(source_config, source_index, spec)
    config = Qwen3TeacherConfig()
    shards = required_layer_shards(source_index["weight_map"], config, args.layer_index)
    print(f"source={spec.repo_id}@{spec.revision}")
    print(
        f"layer={args.layer_index} required_tensors=11 required_shards={len(shards)} "
        f"files={list(shards)}"
    )
    model_dir, _ = ensure_layer_checkpoint(
        args.cache_dir,
        source_index["weight_map"],
        config,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    write_source_marker(model_dir, spec)
    print(f"checkpoint_subset=PASS directory={model_dir}")

    arrays = load_layer_arrays(QwenCheckpointReader(model_dir), config, args.layer_index)
    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal(
        (1, args.sequence_length, config.hidden_size), dtype=np.float32
    )
    torch_output = _torch_reference(
        arrays, hidden, model_dir / "config.json", args.layer_index
    )

    fp32_config = replace(
        config, param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    params = jax_layer_params(arrays, config, args.layer_index)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    jax_layer = Qwen3DecoderLayer(fp32_config)
    apply_layer = jax.jit(jax_layer.apply)
    jax_output = apply_layer(
        {"params": params}, jnp.asarray(hidden), positions, attention_mask
    )
    jax.block_until_ready(jax_output)
    metrics = parity_metrics(torch_output, np.asarray(jax_output))
    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "transformers_version": "4.51.0",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "compute_dtype": "float32",
        "metrics": asdict(metrics),
        "tolerances": {
            "max_abs": args.max_abs_tolerance,
            "relative_l2": args.relative_l2_tolerance,
        },
        "passed": metrics.passes(
            max_abs_tolerance=args.max_abs_tolerance,
            relative_l2_tolerance=args.relative_l2_tolerance,
        ),
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")
    if not result["passed"]:
        raise SystemExit("PARITY-FAIL: metrics exceed the frozen tolerances")
    print("PARITY-PASS: official PyTorch and JAX decoder-layer outputs agree")


if __name__ == "__main__":
    main()
