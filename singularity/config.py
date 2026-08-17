from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Mamba3Config:
    d_state: int = 64
    # 0.75 keeps the hybrid near Qwen-14B's parameter budget. The canonical
    # expand=2 variant is an intentionally larger (~18B) ablation here.
    expand: float = 0.75
    head_dim: int = 64
    groups: int = 1
    mimo_rank: int = 4
    rope_fraction: float = 0.5
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    a_floor: float = 1e-4
    conv_kernel: int = 4


@dataclass(frozen=True)
class MLAConfig:
    num_heads: int = 40
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 1_000_000.0
    qk_head_chunk_size: int = 4


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 152_064
    hidden_size: int = 5_120
    intermediate_size: int = 13_824
    num_layers: int = 48
    attention_layer_indices: tuple[int, ...] = field(
        default_factory=lambda: (5, 12, 19, 26, 33, 40, 47)
    )
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8_192
    tie_word_embeddings: bool = False
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    logits_dtype: str = "float32"
    remat_policy: str = "full"
    scan_layers: bool = False
    mamba: Mamba3Config = field(default_factory=Mamba3Config)
    mla: MLAConfig = field(default_factory=MLAConfig)

    def __post_init__(self) -> None:
        if not self.attention_layer_indices:
            raise ValueError("at least one MLA layer is required")
        if len(set(self.attention_layer_indices)) != len(self.attention_layer_indices):
            raise ValueError("attention_layer_indices must be unique")
        if min(self.attention_layer_indices) < 0 or max(self.attention_layer_indices) >= self.num_layers:
            raise ValueError("attention layer index is outside the decoder")
        if self.mamba.mimo_rank < 1:
            raise ValueError("mimo_rank must be positive")
        inner = int(self.hidden_size * self.mamba.expand)
        if inner % self.mamba.head_dim:
            raise ValueError("expanded hidden size must be divisible by Mamba head_dim")
        if self.mamba.d_state % 2:
            raise ValueError("Mamba-3 complex rotation requires an even d_state")
        if self.mla.qk_rope_head_dim % 2:
            raise ValueError("MLA RoPE head dimension must be even")

    @property
    def mamba_layer_indices(self) -> tuple[int, ...]:
        attention = set(self.attention_layer_indices)
        return tuple(i for i in range(self.num_layers) if i not in attention)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tiny_config() -> HybridConfig:
    """Small CPU-friendly configuration used by smoke tests."""
    return HybridConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=3,
        attention_layer_indices=(1,),
        max_position_embeddings=128,
        mamba=Mamba3Config(
            d_state=8,
            expand=1.0,
            head_dim=16,
            groups=1,
            mimo_rank=2,
            conv_kernel=3,
        ),
        mla=MLAConfig(
            num_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            qk_head_chunk_size=2,
        ),
    )


def load_config(path: str | Path) -> tuple[HybridConfig, dict[str, Any]]:
    """Load the project YAML while keeping training settings separate."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    model = dict(payload["model"])
    model["attention_layer_indices"] = tuple(model["attention_layer_indices"])
    model["mamba"] = Mamba3Config(**model["mamba"])
    model["mla"] = MLAConfig(**model["mla"])
    return HybridConfig(**model), {key: value for key, value in payload.items() if key != "model"}
