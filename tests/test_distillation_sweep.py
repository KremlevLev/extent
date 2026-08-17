from __future__ import annotations

import pytest

from scripts.qwen_mamba3_distill_sweep import (
    _parse_unique_ints,
    aggregate_runs,
    sweep_notes,
)


def _metrics(l2: float, cosine: float) -> dict:
    return {"relative_l2": l2, "cosine_similarity": cosine}


def test_parse_unique_ints_rejects_invalid_sweep_axes():
    assert _parse_unique_ints("123, 456", "seeds", positive=False) == (123, 456)
    with pytest.raises(ValueError, match="unique"):
        _parse_unique_ints("20,20", "checkpoints", positive=True)
    with pytest.raises(ValueError, match="positive"):
        _parse_unique_ints("0,20", "checkpoints", positive=True)


def test_aggregate_runs_preserves_paired_seed_ranking():
    runs = {
        "1": {
            "random": {"checkpoints": {"0": _metrics(0.9, 0.5), "20": _metrics(0.7, 0.7)}},
            "port": {"checkpoints": {"0": _metrics(1.0, 0.4), "20": _metrics(0.8, 0.6)}},
        },
        "2": {
            "random": {"checkpoints": {"0": _metrics(0.8, 0.6), "20": _metrics(0.6, 0.8)}},
            "port": {"checkpoints": {"0": _metrics(0.9, 0.5), "20": _metrics(0.7, 0.7)}},
        },
    }
    result = aggregate_runs(runs, ("random", "port"), (0, 20))
    random_final = result["by_variant"]["random"]["20"]
    paired = result["paired_comparison"]["20"]
    assert random_final["relative_l2_mean"] == pytest.approx(0.65)
    assert random_final["relative_l2_reduction_from_step0_mean"] > 0.2
    assert paired["first_minus_second_relative_l2_mean"] == pytest.approx(-0.1)
    assert paired["first_wins"] == 2


def test_sweep_notes_describe_arbitrary_variant_and_budget_counts():
    notes = sweep_notes(("a", "b", "c", "d", "e"), 20)
    assert "all 5 variants" in notes[0]
    assert "20-step optimization schedule" in notes[2]
    assert "80-step" not in " ".join(notes)
