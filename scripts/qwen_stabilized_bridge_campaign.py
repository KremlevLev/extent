"""Locked EXP-055 correction for the unstable full-RoPE bridge in EXP-054."""

from __future__ import annotations

from scripts.qwen_bridge_ablation_campaign import main as run_bridge_campaign


CAMPAIGN_PROTOCOL = "exp055-stabilized-bridge-campaign"
LAYER_PROTOCOL = "exp055-stabilized-bridge-recovery"

DEFAULT_ARGUMENTS = (
    "--artifact-prefix", "exp055",
    "--campaign-protocol", CAMPAIGN_PROTOCOL,
    "--layer-protocol", LAYER_PROTOCOL,
    "--experiment-name", "EXP-055 stabilized partial-RoPE bridge confirmation",
    "--summary-filename", "extent-stabilized-bridge-campaign-summary.md",
    "--result-json", "/kaggle/working/output/extent-stabilized-bridge-campaign.json",
    "--qwen-cache-dir", "/kaggle/working/qwen3-exp055-weights",
    "--recovery-steps", "4096",
    "--checkpoints", "0,256,1024,2048,4096",
    "--bridge-steps", "256",
    "--bridge-learning-rate", "3e-4",
    "--bridge-matrix-loss-weight", "0.0",
    "--bridge-rope-fraction", "0.5",
    "--orientation-steps", "256",
    "--primary-arm", "APPLE-BRIDGE",
    "--max-wall-hours", "3.4",
)


def main(argv: list[str] | None = None) -> dict:
    """Run the pre-registered correction, allowing explicit final overrides."""
    return run_bridge_campaign([*DEFAULT_ARGUMENTS, *(argv or [])])


if __name__ == "__main__":
    main()
