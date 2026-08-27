from scripts.qwen_exact_lift_confirmation_campaign import (
    ARMS,
    LAYERS,
    SEQUENCE_LENGTH,
    TRAIN_OFFSET,
    VALIDATION_OFFSET,
    aggregate_confirmation,
    render_summary,
)


def _layer(lift_delta=-0.1):
    seeds = {}
    for seed in (123, 456, 789):
        seeds[str(seed)] = {"arms": {
            "CONTROL-RANDOM": {"recovery": {"complete": True, "evaluations": {"4096": {"decoder_output": {"relative_l2": 1.0}}}}},
            "BALANCED-RANK-LIFT": {"recovery": {"complete": True, "evaluations": {"4096": {"decoder_output": {"relative_l2": 1.0 + lift_delta}}}}},
        }}
    return {"seeds": seeds}


def test_confirmation_protocol_is_locked_and_minimal():
    assert LAYERS == (18, 6, 29, 0)
    assert ARMS == ("CONTROL-RANDOM", "BALANCED-RANK-LIFT")
    assert SEQUENCE_LENGTH == 64
    assert TRAIN_OFFSET == 131072
    assert VALIDATION_OFFSET == 8192


def test_confirmation_gate_requires_every_layer():
    results = {str(layer): _layer() for layer in LAYERS}
    aggregate = aggregate_confirmation(results)
    assert aggregate["confirmation_gate_passed"]
    del results["0"]
    assert not aggregate_confirmation(results)["confirmation_gate_passed"]
    summary = render_summary({"status": "completed", "duration_hours": 1.5, "complete": True, "aggregate": aggregate})
    assert "locked fresh-text" in summary
    assert "10.000%" in summary
