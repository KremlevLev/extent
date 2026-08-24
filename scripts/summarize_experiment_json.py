from __future__ import annotations

import argparse
import json
from pathlib import Path

from extent.artifact_summary import write_compact_summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Create compact JSON and Markdown summaries of an Extent artifact."
    )
    parser.add_argument("artifact")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    artifact = Path(args.artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    result = write_compact_summary(
        payload,
        artifact_path=artifact,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
