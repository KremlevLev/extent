from __future__ import annotations

import argparse
from pathlib import Path

from singularity.qwen_source import QWEN3_14B, write_source_marker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the immutable Qwen3-14B source checkpoint.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    spec = QWEN3_14B
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"downloading {spec.repo_id}@{spec.revision} to {output}")
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=output,
        allow_patterns=["config.json", "model.safetensors.index.json", "model-*.safetensors"],
    )
    marker = write_source_marker(output, spec)
    print(f"source_marker={marker}")
    print("download=PASS; run checkpoint validation before loading weights")


if __name__ == "__main__":
    main()
