from __future__ import annotations

import json

from scripts.qwen_extended_horizon_campaign import (
    _combine_budget_layers,
    _prune_completed_layer,
    _read_completed_layer,
)


def test_exp049_combines_sequential_layer_results():
    per_layer = {
        "0": {
            "passed": True,
            "scientific_gate_passed": True,
            "layer_results": {"0": {"passed": True}},
        },
        "18": {
            "passed": True,
            "scientific_gate_passed": False,
            "layer_results": {"18": {"passed": True}},
        },
    }
    combined = _combine_budget_layers("short", per_layer)
    assert combined["passed"]
    assert not combined["scientific_gate_passed"]
    assert set(combined["layer_results"]) == {"0", "18"}


def test_exp049_resume_boundary_checks_layer_and_budget(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "exp049-long-layer18",
                "total_steps_per_arm": 8192,
                "target_layers": [18],
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    assert _read_completed_layer(
        path,
        experiment="exp049-long",
        layer=18,
        steps=8192,
    ) is not None
    assert _read_completed_layer(
        path,
        experiment="exp049-long",
        layer=0,
        steps=8192,
    ) is None


def test_exp049_prunes_only_completed_layer_payloads(tmp_path):
    target_cache = tmp_path / "exp049-long-layer18-train-cache"
    other_cache = tmp_path / "exp049-long-layer0-train-cache"
    endpoint_dir = tmp_path / "exp049-long-layer18-endpoints"
    target_cache.mkdir()
    other_cache.mkdir()
    endpoint_dir.mkdir()
    target_array = target_cache / "activation.npy"
    other_array = other_cache / "activation.npy"
    endpoint = endpoint_dir / "endpoint_params.msgpack"
    target_array.write_bytes(b"cache")
    other_array.write_bytes(b"keep")
    endpoint.write_bytes(b"params")

    removed = _prune_completed_layer(
        tmp_path,
        experiment="exp049-long",
        layer=18,
    )

    assert len(removed) == 2
    assert not target_array.exists()
    assert not endpoint.exists()
    assert other_array.exists()
