"""EXP-091: long paired recovery from on-policy endpoints with a Qwen anchor.

Run once per Kaggle TPU session. Each complete trajectory is checkpointed to
the configured HF dataset, so a deadline-partial run continues on rerun.
"""

from __future__ import annotations

from pathlib import Path

from extent.config import load_config
from extent.hf_artifact_sync import artifact_config_from_env
import scripts.m3q_long_recoverability_campaign as campaign
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072, configured_contract


def required_source_manifests(seeds, order):
    required = set()
    for seed in seeds:
        for layer in order:
            required.add(
                f"{EXP069_PREFIX}/checkpoints/prep/seed-{seed}/layer-{layer}/checkpoint.json"
            )
            required.add(
                f"{EXP072['HF_PREFIX']}/checkpoints/seed-{seed}/ONPOLICY/"
                f"layer-{layer}/checkpoint.json"
            )
    return required


def preflight_source_endpoints(*, api=None):
    """Fail quickly if the 96 prerequisite remote checkpoints are absent."""
    from huggingface_hub import HfApi

    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    config, _ = load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )
    order = configured_contract(config)["replacement_order"]
    required = required_source_manifests(campaign.SEEDS, order)
    client = HfApi(token=hub.token) if api is None else api
    available = set(client.list_repo_files(
        repo_id=hub.repo_id, repo_type=hub.repo_type,
        revision=hub.revision, token=hub.token,
    ))
    missing = sorted(required - available)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(required)} source manifests on HF; "
            f"first missing: {missing[0]}"
        )
    print(f"exp091_source_preflight=PASS manifests={len(required)}", flush=True)


def main(argv: list[str] | None = None) -> dict:
    campaign.PROTOCOL = "exp091-onpolicy-backbone-anchor-v1"
    campaign.HF_PREFIX = "experiments/exp091-onpolicy-backbone-anchor"
    campaign.ARMS = ("ONPOLICY-PLAIN", "ONPOLICY-ANCHOR")
    campaign.CONTROL_ARM = "ONPOLICY-PLAIN"
    campaign.PRIMARY_ARM = "ONPOLICY-ANCHOR"
    campaign.EXPERIMENT_ID = "EXP-091"
    campaign.ARTIFACT_STEM = "extent-m3q-onpolicy-anchor"
    campaign.EXPERIMENT_LABEL = "on-policy warm start with protected Qwen backbone"
    campaign.PRIMARY_QUESTION = (
        "From identical EXP-072 on-policy Mamba endpoints, does 0.03x Lion "
        "update scaling on copied Qwen weights improve whole-model recovery "
        "versus ordinary Lion updates at equal steps, text, and optimizer?"
    )
    campaign.FROZEN_DT_STEPS = 0
    campaign.WARM_START_ARMS = campaign.ARMS
    campaign.BACKBONE_UPDATE_SCALE = {"ONPOLICY-ANCHOR": 0.03}
    campaign.SEEDS = (123, 456)
    campaign.TOTAL_STEPS = 24_576
    campaign.TOKENS_PER_TRAJECTORY = campaign.TOTAL_STEPS * campaign.SEQUENCE_LENGTH
    campaign.CHECKPOINTS = (0, 3_072, 8_192, 16_384, 24_576)
    campaign.SAVE_EVERY = 8_192
    campaign.EXECUTION_ORDER = (
        (123, "ONPOLICY-PLAIN"),
        (123, "ONPOLICY-ANCHOR"),
        (456, "ONPOLICY-ANCHOR"),
        (456, "ONPOLICY-PLAIN"),
    )
    preflight_source_endpoints()
    return campaign.main(argv)


if __name__ == "__main__":
    main()
