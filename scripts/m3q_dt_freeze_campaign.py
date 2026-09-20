"""EXP-090 entry point: long full-model random recovery with an early dt freeze."""

from __future__ import annotations

import scripts.m3q_long_recoverability_campaign as campaign


def main(argv: list[str] | None = None) -> dict:
    campaign.PROTOCOL = "exp090-random-dt-freeze-v1"
    campaign.HF_PREFIX = "experiments/exp090-random-dt-freeze"
    campaign.ARMS = ("RANDOM", "RANDOM-DT-FROZEN")
    campaign.CONTROL_ARM = "RANDOM"
    campaign.PRIMARY_ARM = "RANDOM-DT-FROZEN"
    campaign.EXPERIMENT_ID = "EXP-090"
    campaign.ARTIFACT_STEM = "extent-m3q-dt-freeze"
    campaign.EXPERIMENT_LABEL = "long-horizon dt-freeze recovery"
    campaign.FROZEN_DT_STEPS = 32_768
    campaign.PRIMARY_QUESTION = (
        "Does freezing Mamba-3 dt for the first 8.389M tokens improve "
        "25.166M-token whole-model recovery from canonical random initialization?"
    )
    return campaign.main(argv)


if __name__ == "__main__":
    main()
