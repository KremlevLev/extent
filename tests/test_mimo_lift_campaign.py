from scripts.qwen_mimo_lift import ARM_ORDER, VARIANT_BY_ARM
from scripts.qwen_mimo_lift_campaign import TARGET_LAYERS, aggregate_mimo_lift, render_summary


def _recovery(values):
    return {
        "complete": True,
        "evaluations": {
            str(step): {"decoder_output": {"relative_l2": value}}
            for step, value in zip((0, 256, 1024, 2048, 4096), values)
        },
    }


def _layer(layer):
    seeds = {}
    for offset, seed in enumerate((123, 456, 789)):
        random = [1.0, .8, .7, .6, .5 + offset * .001]
        seeds[str(seed)] = {"arms": {
            "CONTROL-RANDOM": {"recovery": _recovery(random)},
            "CONTROL-FLAT-QKVO": {"recovery": _recovery([1.1, .9, .72, .59, .48])},
            "SINGLE-CHANNEL-LIFT": {"recovery": _recovery([1.0, .79, .68, .56, .46])},
            "BALANCED-RANK-LIFT": {"recovery": _recovery([1.0, .77, .65, .53, .44])},
        }}
    return {
        "target_layer": layer,
        "training": {"checkpoints": [0, 256, 1024, 2048, 4096]},
        "seeds": seeds,
    }


def test_protocol_has_paired_controls_and_depth_order():
    assert TARGET_LAYERS == (18, 6, 29, 0)
    assert ARM_ORDER[0] == "CONTROL-RANDOM"
    assert set(ARM_ORDER) == set(VARIANT_BY_ARM)
    assert VARIANT_BY_ARM["SINGLE-CHANNEL-LIFT"].startswith("INIT-J")
    assert VARIANT_BY_ARM["BALANCED-RANK-LIFT"].startswith("INIT-K")


def test_aggregate_tracks_crossover_and_excludes_boundary_from_gate():
    results = {str(layer): _layer(layer) for layer in TARGET_LAYERS}
    aggregate = aggregate_mimo_lift(results)
    assert aggregate["screening_gate_passed"]
    assert aggregate["layers"]["18"]["BALANCED-RANK-LIFT"]["earliest_mean_crossover_step"] == 256

    # Layer zero is deliberately diagnostic and cannot veto the internal-layer gate.
    for seed in results["0"]["seeds"].values():
        seed["arms"]["BALANCED-RANK-LIFT"]["recovery"] = _recovery([2, 2, 2, 2, 2])
    aggregate = aggregate_mimo_lift(results)
    assert aggregate["screening_gate_passed"]

    campaign = {
        "status": "completed", "duration_hours": 7.0, "complete": True,
        "passed": True, "aggregate": aggregate,
    }
    summary = render_summary(campaign)
    assert "BALANCED-RANK-LIFT" in summary
    assert "boundary diagnostic" in summary
