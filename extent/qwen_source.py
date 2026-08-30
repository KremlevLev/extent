from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from urllib import request


@dataclass(frozen=True)
class QwenSourceSpec:
    repo_id: str
    revision: str
    architecture: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    tensor_count: int
    shard_count: int
    total_size_bytes: int

    def resolve_url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"


QWEN3_14B = QwenSourceSpec(
    repo_id="Qwen/Qwen3-14B",
    revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    num_hidden_layers=40,
    hidden_size=5120,
    intermediate_size=17408,
    num_attention_heads=40,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    max_position_embeddings=40960,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    tensor_count=443,
    shard_count=8,
    total_size_bytes=29_536_614_400,
)

QWEN3_1_7B_BASE = QwenSourceSpec(
    repo_id="Qwen/Qwen3-1.7B-Base",
    revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    num_hidden_layers=28,
    hidden_size=2048,
    intermediate_size=6144,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    max_position_embeddings=32768,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    tensor_count=310,
    shard_count=1,
    total_size_bytes=3_441_149_952,
)

QWEN_SOURCES = {
    "14b": QWEN3_14B,
    "1.7b-base": QWEN3_1_7B_BASE,
}


def teacher_config_from_spec(
    spec: QwenSourceSpec,
    *,
    param_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    remat_policy: str = "none",
):
    """Construct the JAX teacher contract from an immutable source pin."""
    from extent.qwen3_teacher import Qwen3TeacherConfig

    return Qwen3TeacherConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        head_dim=spec.head_dim,
        max_position_embeddings=spec.max_position_embeddings,
        rope_theta=spec.rope_theta,
        rms_norm_eps=spec.rms_norm_eps,
        tie_word_embeddings=spec.tie_word_embeddings,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        remat_policy=remat_policy,
    )


def load_remote_source_metadata(spec: QwenSourceSpec) -> tuple[dict, dict]:
    """Read pinned config/index metadata without downloading model weights."""
    with request.urlopen(spec.resolve_url("config.json"), timeout=30) as response:
        config_payload = json.loads(response.read().decode("utf-8"))
    if spec.shard_count == 1:
        from extent.weight_mapping import teacher_qwen_mappings

        teacher = teacher_config_from_spec(spec)
        weight_map = {
            entry.source: "model.safetensors"
            for entry in teacher_qwen_mappings(teacher)
        }
        index_payload = {
            "metadata": {"total_size": spec.total_size_bytes},
            "weight_map": weight_map,
        }
    else:
        with request.urlopen(
            spec.resolve_url("model.safetensors.index.json"), timeout=30
        ) as response:
            index_payload = json.loads(response.read().decode("utf-8"))
    validate_source_metadata(config_payload, index_payload, spec)
    return config_payload, index_payload

SOURCE_MARKER = ".extent_source.json"


@dataclass(frozen=True)
class SourceValidationReport:
    tensor_count: int
    shard_count: int
    total_size_bytes: int


def validate_source_metadata(
    config_payload: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    spec: QwenSourceSpec = QWEN3_14B,
) -> SourceValidationReport:
    """Reject a checkpoint whose architecture or index differs from the pin."""
    expected_config = {
        "architectures": [spec.architecture],
        "model_type": spec.model_type,
        "num_hidden_layers": spec.num_hidden_layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "vocab_size": spec.vocab_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "rope_theta": spec.rope_theta,
        "rms_norm_eps": spec.rms_norm_eps,
        "tie_word_embeddings": spec.tie_word_embeddings,
    }
    mismatches = {
        key: (config_payload.get(key), expected)
        for key, expected in expected_config.items()
        if config_payload.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} expected {expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Qwen config does not match pinned source: {details}")

    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("checkpoint index has no weight_map")
    tensor_count = len(weight_map)
    shard_count = len(set(weight_map.values()))
    total_size = int(index_payload.get("metadata", {}).get("total_size", -1))
    actual = (tensor_count, shard_count, total_size)
    expected = (spec.tensor_count, spec.shard_count, spec.total_size_bytes)
    if actual != expected:
        raise ValueError(
            "checkpoint index does not match pinned source: "
            f"tensors/shards/bytes={actual}, expected={expected}"
        )
    return SourceValidationReport(tensor_count, shard_count, total_size)


def write_source_marker(
    model_dir: str | Path,
    spec: QwenSourceSpec = QWEN3_14B,
) -> Path:
    path = Path(model_dir) / SOURCE_MARKER
    path.write_text(
        json.dumps({"repo_id": spec.repo_id, "revision": spec.revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def validate_source_marker(
    model_dir: str | Path,
    spec: QwenSourceSpec = QWEN3_14B,
) -> None:
    path = Path(model_dir) / SOURCE_MARKER
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {SOURCE_MARKER}; use the pinned download launcher so revision provenance is recorded"
        )
    marker = json.loads(path.read_text(encoding="utf-8"))
    expected = {"repo_id": spec.repo_id, "revision": spec.revision}
    if marker != expected:
        raise ValueError(f"source marker mismatch: {marker!r}, expected {expected!r}")
