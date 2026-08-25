from __future__ import annotations

import json

from extent.artifact_summary import (
    build_compact_summary,
    render_summary_markdown,
    write_compact_summary,
)


def _payload() -> dict:
    return {
        "source": "teacher@revision",
        "method": "scale",
        "protocol": "exp-test",
        "started_at_utc": "2026-08-24T00:00:00+00:00",
        "completed_at_utc": "2026-08-24T06:00:00+00:00",
        "optimizer_visible_tokens_total": 1234,
        "passed": True,
        "scientific_gate_passed": True,
        "aggregate": {
            "primary_arm": "joint",
            "layers": [
                {
                    "target_layer": 0,
                    "primary_joint_gate_passed": True,
                    "arms": {
                        "joint": {
                            "mean_long_minus_short_nll": -0.2,
                            "long_minus_short_95ci": [-0.3, -0.1],
                            "long_wins": 3,
                            "per_seed": [
                                {
                                    "seed": 123,
                                    "short_nll": 5.0,
                                    "long_nll": 4.8,
                                    "long_minus_short_nll": -0.2,
                                }
                            ],
                        }
                    },
                }
            ],
        },
    }


def test_compact_summary_removes_full_training_payload():
    payload = _payload()
    payload["budget_results"] = {"huge": {"training_result": [1, 2, 3]}}
    summary = build_compact_summary(payload)
    assert summary["duration_hours"] == 6.0
    assert "budget_results" not in summary
    assert summary["scale_layers"][0]["arms"]["joint"]["long_wins"] == 3
    assert "Mean ΔNLL" in render_summary_markdown(summary)


def test_compact_summary_writes_json_and_markdown(tmp_path):
    artifact = tmp_path / "campaign.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    outputs = write_compact_summary(_payload(), artifact_path=artifact)
    summary_json = tmp_path / "campaign-summary.json"
    summary_md = tmp_path / "campaign-summary.md"
    assert outputs["summary_json"] == str(summary_json.resolve())
    assert summary_json.exists()
    assert summary_md.exists()


def test_compact_summary_includes_depth_trend():
    payload = _payload()
    payload["aggregate"]["depth_trend"] = {
        "arms": {
            "joint": {
                "nll_delta_per_layer_index_slope": 0.01,
                "slope_95ci": [0.005, 0.015],
            }
        },
        "scientific_gate_passed": True,
    }
    summary = build_compact_summary(payload)
    assert summary["depth_trend"]["scientific_gate_passed"]
    assert "Depth trend" in render_summary_markdown(summary)


def test_compact_summary_includes_progressive_composition_table():
    payload = _payload()
    payload["aggregate"] = {
        "primary_replacement_count": 8,
        "baseline_reproduction": {
            "passed": True,
            "maximum_absolute_mean_nll_difference": 0.001,
        },
        "stages": [
            {
                "replacement_count": 8,
                "layers": [0, 6, 12, 18, 23, 29, 34, 39],
                "additive_expected_excess_nll_mean": 0.2,
                "observed_composed_excess_nll_mean": 0.22,
                "interaction_nll_mean": 0.02,
                "composition_inflation_ratio_mean": 1.1,
                "seed_inflation_passes": 3,
                "bootstrap": {"mean_inflation_ratio_95ci": [1.0, 1.2]},
                "scientific_gate_passed": True,
            }
        ],
    }
    summary = build_compact_summary(payload)
    assert summary["progressive_composition"]["primary_replacement_count"] == 8
    rendered = render_summary_markdown(summary)
    assert "Progressive composition" in rendered
    assert "0,6,12,18,23,29,34,39" in rendered
