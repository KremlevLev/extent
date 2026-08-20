from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze

from scripts.qwen_mamba3_distill_sweep import (
    _parse_unique_ints,
    aggregate_runs,
    linear_prior_scale,
    sweep_notes,
    transient_prior_offset,
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


def test_linear_prior_scale_reaches_zero_at_frozen_decay_step():
    assert linear_prior_scale(0, 20) == 1.0
    assert linear_prior_scale(10, 20) == 0.5
    assert linear_prior_scale(20, 20) == 0.0
    assert linear_prior_scale(80, 20) == 0.0
    with pytest.raises(ValueError, match="positive"):
        linear_prior_scale(0, 0)


def test_transient_prior_offset_excludes_independent_output_readout():
    random = freeze(
        {
            "in_proj": {"kernel": jnp.ones((2, 2))},
            "b_norm": {"scale": jnp.ones((2,))},
            "c_norm": {"scale": jnp.ones((2,))},
            "out_proj": {"kernel": jnp.ones((2, 2))},
            "D": jnp.ones((2,)),
        }
    )
    blended = freeze(
        {
            "in_proj": {"kernel": 1.25 * jnp.ones((2, 2))},
            "b_norm": {"scale": 1.1 * jnp.ones((2,))},
            "c_norm": {"scale": 0.9 * jnp.ones((2,))},
            "out_proj": {"kernel": 3.0 * jnp.ones((2, 2))},
            "D": jnp.ones((2,)),
        }
    )
    offset = transient_prior_offset(random, blended)
    np.testing.assert_allclose(offset["in_proj"]["kernel"], 0.25)
    np.testing.assert_allclose(offset["out_proj"]["kernel"], 0.0)
    np.testing.assert_allclose(offset["D"], 0.0)
