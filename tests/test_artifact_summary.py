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
