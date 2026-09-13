"""Read-only size audit of Extent experiment artifacts on Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import artifact_config_from_env
from extent.hf_retention import DEFAULT_PROTECTED_PREFIXES, classify_repo_files


def _gib(value: int) -> float:
    return value / 1024**3


def _render(audit: dict) -> str:
    lines = [
        "# Extent Hugging Face checkpoint-size audit",
        "",
        f"- Repository working-tree size: `{_gib(audit['repository_bytes']):.3f} GiB`",
        f"- Checkpoint payloads to review: `{audit['reviewable_payload_files']}` "
        f"(`{_gib(audit['reviewable_payload_bytes']):.3f} GiB`)",
        f"- Currently reusable source payloads: `{audit['protected_payload_files']}` "
        f"(`{_gib(audit['protected_payload_bytes']):.3f} GiB`)",
        "- Reusable sources: `EXP-069`, `EXP-072-v2`",
        "",
        "| Experiment | Payloads | Reviewable GiB |",
        "|---|---:|---:|",
    ]
    for experiment, row in audit["reviewable_by_experiment"].items():
        lines.append(
            f"| {experiment} | {row['files']} | {_gib(row['bytes']):.3f} |"
        )
    lines += [
        "",
        "This audit is read-only and never changes the repository.",
        "A listed payload is only a review candidate, not an automatic deletion decision.",
        "Scientific JSON/Markdown artifacts are not included in the table.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument(
        "--reusable-prefix", action="append", dest="protected_prefixes",
        help="replace defaults with one or more checkpoint prefixes still in use",
    )
    args = parser.parse_args(argv)

    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")

    from huggingface_hub import HfApi

    api = HfApi(token=hub.token)
    files = list(api.list_repo_tree(
        repo_id=hub.repo_id,
        repo_type=hub.repo_type,
        revision=hub.revision,
        path_in_repo="experiments",
        recursive=True,
        expand=True,
    ))
    prefixes = tuple(args.protected_prefixes or DEFAULT_PROTECTED_PREFIXES)
    audit = classify_repo_files(files, prefixes)
    audit.update(
        repo_id=hub.repo_id,
        repo_type=hub.repo_type,
        revision=hub.revision,
        mode="read_only",
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "extent-hf-retention-audit.json"
    summary_path = output / "extent-hf-retention-audit-summary.md"
    write_json_atomic(json_path, audit)
    summary_path.write_text(_render(audit), encoding="utf-8")
    print(_render(audit), flush=True)
    print("mode=READ-ONLY remote repository unchanged", flush=True)
    print(f"result_json={json_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return audit


if __name__ == "__main__":
    main()
