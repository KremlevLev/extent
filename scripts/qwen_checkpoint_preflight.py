from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

from singularity import HybridForCausalLM
from singularity.config import load_config
from singularity.initialization import abstract_parameter_tree
from singularity.qwen_source import (
    QWEN3_14B,
    validate_source_marker,
    validate_source_metadata,
)
from singularity.weight_mapping import (
    QwenCheckpointReader,
    validate_direct_mapping_plan,
    validate_local_direct_shapes,
)


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the immutable Qwen source and direct map.")
    parser.add_argument("--model-dir", help="Optional downloaded checkpoint directory.")
    args = parser.parse_args(argv)
    spec = QWEN3_14B
    print(f"source={spec.repo_id}@{spec.revision}")
    if args.model_dir:
        model_dir = Path(args.model_dir)
        print(f"reading downloaded checkpoint metadata from {model_dir.resolve()}...")
        validate_source_marker(model_dir, spec)
        source_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        source_index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        missing = sorted(
            shard for shard in set(source_index["weight_map"].values())
            if not (model_dir / shard).is_file()
        )
        if missing:
            raise FileNotFoundError(f"checkpoint is missing shards: {missing}")
        print("source_revision_marker=PASS")
    else:
        print("fetching pinned config and safetensors index only (model shards are not downloaded)...")
        source_config = _read_json(spec.resolve_url("config.json"))
        source_index = _read_json(spec.resolve_url("model.safetensors.index.json"))
    source_report = validate_source_metadata(source_config, source_index, spec)

    config, _ = load_config("config/hybrid_14b_v5e8.yaml")
    abstract_params = abstract_parameter_tree(HybridForCausalLM(config), sequence_length=1)
    mapping_report = validate_direct_mapping_plan(config, abstract_params, source_index["weight_map"])
    gib = source_report.total_size_bytes / 2**30
    print(
        f"checkpoint_index=PASS tensors={source_report.tensor_count} "
        f"shards={source_report.shard_count} size={gib:.3f} GiB"
    )
    print(
        f"direct_mapping=PASS tensors={mapping_report.tensor_count} "
        f"parameters={mapping_report.parameter_count:,}"
    )
    print("scope=embeddings + final norm + lm_head + all layer norms/MLPs")
    if args.model_dir:
        validate_local_direct_shapes(QwenCheckpointReader(args.model_dir), config)
        print("local_safetensors_headers=PASS")
    print("mixer_conversion=NOT_RUN (MLA and Mamba-3 transplant remain separate experiments)")


if __name__ == "__main__":
    main()
