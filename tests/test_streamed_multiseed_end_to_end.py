from __future__ import annotations

from scripts.qwen_streamed_multiseed_end_to_end import (
    branch_names,
    reference_reproduction,
)


def test_multiseed_branch_order_is_frozen():
    assert branch_names((123, 456, 789), 1024) == (
        "ORIGINAL-CACHED-QWEN",
        "SEED-123-CALIBRATED-STEP0",
        "SEED-123-MIXER-ONLY-STEP1024",
        "SEED-123-JOINT-STEP1024",
        "SEED-456-CALIBRATED-STEP0",
        "SEED-456-MIXER-ONLY-STEP1024",
        "SEED-456-JOINT-STEP1024",
        "SEED-789-CALIBRATED-STEP0",
        "SEED-789-MIXER-ONLY-STEP1024",
        "SEED-789-JOINT-STEP1024",
    )


def test_reference_reproduction_applies_absolute_nll_tolerance():
    reference = {
        "lm_metrics": {
            "ORIGINAL-CACHED-QWEN": {"mean_nll": 1.0},
            "CALIBRATED-STEP0": {"mean_nll": 2.0},
            "MIXER-ONLY-STEP1024": {"mean_nll": 1.8},
            "JOINT-STEP1024": {"mean_nll": 1.5},
        }
    }
    metrics = {
        "ORIGINAL-CACHED-QWEN": {"mean_nll": 1.001},
        "SEED-123-CALIBRATED-STEP0": {"mean_nll": 2.001},
        "SEED-123-MIXER-ONLY-STEP1024": {"mean_nll": 1.799},
        "SEED-123-JOINT-STEP1024": {"mean_nll": 1.501},
    }
    report = reference_reproduction(
        reference,
        metrics,
        seed=123,
        total_steps=1024,
        tolerance=0.01,
    )
    assert report["passed"] is True
    metrics["SEED-123-JOINT-STEP1024"]["mean_nll"] = 1.6
    report = reference_reproduction(
        reference,
        metrics,
        seed=123,
        total_steps=1024,
        tolerance=0.01,
    )
    assert report["passed"] is False
