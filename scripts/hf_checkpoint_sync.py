from __future__ import annotations

import argparse
import os

from extent.hf_checkpoint_sync import (
    download_checkpoint_from_hub,
    upload_checkpoint_to_hub,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Push or pull a verified Extent checkpoint using Hugging Face Hub."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("push", "pull"):
        command = subparsers.add_parser(operation)
        command.add_argument("--checkpoint-dir", required=True)
        command.add_argument("--repo-id", required=True)
        command.add_argument("--path-in-repo", required=True)
        command.add_argument("--revision", default="main")
    subparsers.choices["push"].add_argument("--public", action="store_true")
    subparsers.choices["push"].add_argument("--commit-message")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or None

    if args.operation == "push":
        info = upload_checkpoint_to_hub(
            args.checkpoint_dir,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            private=not args.public,
            commit_message=args.commit_message,
            token=token,
        )
        print(f"HF-CHECKPOINT-PUSH-PASS commit={getattr(info, 'oid', 'unknown')}")
    else:
        metadata = download_checkpoint_from_hub(
            args.checkpoint_dir,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            token=token,
        )
        print(
            f"HF-CHECKPOINT-PULL-PASS step={metadata['step']} "
            f"sha256={metadata['checkpoint_sha256']}"
        )


if __name__ == "__main__":
    main()
