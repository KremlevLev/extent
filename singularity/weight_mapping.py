"""Qwen safetensors -> hybrid initialization scaffolding.

The functions operate on one array/layer at a time so a 14B checkpoint never has
to be materialized twice in host RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from singularity.config import HybridConfig


@dataclass(frozen=True)
class MappingEntry:
    source: str
    target: str
    transform: str = "transpose"


def direct_qwen_mappings(config: HybridConfig) -> list[MappingEntry]:
    entries = [
        MappingEntry("model.embed_tokens.weight", "embed_tokens/embedding", "identity"),
        MappingEntry("model.norm.weight", "norm/scale", "identity"),
        MappingEntry("lm_head.weight", "lm_head/kernel", "transpose"),
    ]
    for layer in range(config.num_layers):
        source = f"model.layers.{layer}"
        target = f"layers_{layer}"
        entries.extend(
            [
                MappingEntry(f"{source}.input_layernorm.weight", f"{target}/input_layernorm/scale", "identity"),
                MappingEntry(f"{source}.post_attention_layernorm.weight", f"{target}/post_attention_layernorm/scale", "identity"),
                MappingEntry(f"{source}.mlp.gate_proj.weight", f"{target}/mlp/gate_proj/kernel"),
                MappingEntry(f"{source}.mlp.up_proj.weight", f"{target}/mlp/up_proj/kernel"),
                MappingEntry(f"{source}.mlp.down_proj.weight", f"{target}/mlp/down_proj/kernel"),
            ]
        )
    return entries


def truncated_svd(
    weight: np.ndarray,
    rank: int,
    *,
    exact_threshold: int = 2_048,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor [input, output] into low-rank factors without a huge full SVD."""
    matrix = weight.astype(np.float32)
    if max(matrix.shape) <= exact_threshold:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    else:
        width = min(rank + oversample, min(matrix.shape))
        random = np.random.default_rng(seed).standard_normal((matrix.shape[1], width), dtype=np.float32)
        q, _ = np.linalg.qr(matrix @ random, mode="reduced")
        for _ in range(power_iterations):
            q_right, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
            q, _ = np.linalg.qr(matrix @ q_right, mode="reduced")
        small = q.T @ matrix
        u_small, singular, vh = np.linalg.svd(small, full_matrices=False)
        u = q @ u_small
    rank = min(rank, singular.size)
    root = np.sqrt(singular[:rank])
    return (u[:, :rank] * root).astype(weight.dtype), (root[:, None] * vh[:rank]).astype(weight.dtype)


class QwenCheckpointReader:
    """Lazy safetensors reader: only one requested tensor is resident at a time."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as stream:
            self.weight_map: dict[str, str] = json.load(stream)["weight_map"]

    def read(self, name: str) -> np.ndarray:
        from safetensors import safe_open

        if name not in self.weight_map:
            raise KeyError(f"tensor not present in checkpoint: {name}")
        with safe_open(str(self.model_dir / self.weight_map[name]), framework="np", device="cpu") as shard:
            return shard.get_tensor(name)

    def qkvo(self, layer: int) -> dict[str, np.ndarray]:
        prefix = f"model.layers.{layer}.self_attn"
        return {name: self.read(f"{prefix}.{name}_proj.weight").T for name in ("q", "k", "v", "o")}


def fit_matrix(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Deterministically tile/crop a projection for transplant ablations."""
    row_repeats = (target_shape[0] + source.shape[0] - 1) // source.shape[0]
    col_repeats = (target_shape[1] + source.shape[1] - 1) // source.shape[1]
    tiled = np.tile(source, (row_repeats, col_repeats))
    scale = np.sqrt(source.shape[1] / target_shape[1])
    return (tiled[: target_shape[0], : target_shape[1]] * scale).astype(source.dtype)


def mamba_transplant_in_projection(
    q_kernel: np.ndarray,
    k_kernel: np.ndarray,
    v_kernel: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Experimental Q/K/V seed for Mamba in_proj; suitable as an ablation baseline.

    This is intentionally labelled a heuristic, not a pretrained-equivalence map.
    It repeats V for x/z and interleaves K/Q over the remaining controller slots.
    """
    width = target_shape[1]
    first = width // 2
    xz = fit_matrix(np.concatenate((v_kernel, q_kernel), axis=1), (target_shape[0], first))
    controllers = fit_matrix(np.concatenate((k_kernel, q_kernel, v_kernel), axis=1), (target_shape[0], width - first))
    return np.concatenate((xz, controllers), axis=1)
