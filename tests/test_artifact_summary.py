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


def test_compact_summary_includes_boundary_and_incremental_tables():
    payload = _payload()
    payload["aggregate"] = {
        "primary_replacement_count": 16,
        "baseline_reproduction": {
            "passed": True,
            "maximum_absolute_mean_nll_difference": 0.001,
        },
        "stages": [],
        "composition_scaling_gate_passed": True,
        "boundary_mechanism_gate_passed": True,
        "boundary_analysis": {
            "stages": [
                {
                    "replacement_count": 16,
                    "internal_replacement_count": 15,
                    "layer0_only_excess_nll_mean": 0.80,
                    "internal_only_excess_nll_mean": 0.15,
                    "full_composition_excess_nll_mean": 0.72,
                    "boundary_interaction_nll_mean": -0.23,
                    "bootstrap": {
                        "internal_minus_layer0_95ci": [-0.70, -0.60],
                        "boundary_interaction_nll_95ci": [-0.30, -0.15],
                    },
                    "mechanism_gate_passed": True,
                }
            ]
        },
        "incremental_scaling": {
            "lower_replacement_count": 8,
            "upper_replacement_count": 16,
            "added_layers": [1, 3, 9, 15, 21, 26, 32, 36],
            "expected_added_excess_nll_mean": 0.08,
            "observed_added_excess_nll_mean": 0.02,
            "incremental_interaction_nll_mean": -0.06,
            "incremental_inflation_ratio_mean": 0.25,
            "seed_inflation_passes": 3,
            "bootstrap": {
                "incremental_inflation_ratio_95ci": [0.1, 0.4]
            },
            "scientific_gate_passed": True,
        },
    }
    summary = build_compact_summary(payload)
    assert summary["boundary_scaling"]["boundary_mechanism_gate_passed"]
    rendered = render_summary_markdown(summary)
    assert "Layer-0 boundary decomposition" in rendered
    assert "Incremental 8-to-16 scaling" in rendered


def test_compact_summary_includes_onset_single_and_layout_tables():
    payload = _payload()
    payload["aggregate"] = {
        "stages": [],
        "onset_localization_gate_passed": True,
        "first_order_attribution_gate_passed": True,
        "incremental_onset": {
            "earliest_detected_upper_count": 10,
            "stages": [
                {
                    "lower_replacement_count": 8,
                    "upper_replacement_count": 10,
                    "added_layers": [1, 36],
                    "expected_added_excess_nll_mean": 0.02,
                    "observed_added_excess_nll_mean": 0.08,
                    "incremental_interaction_nll_mean": 0.06,
                    "incremental_inflation_ratio_mean": 4.0,
                    "positive_interaction_wins": 3,
                    "bootstrap": {
                        "adjusted_confidence_level": 0.9875,
                        "incremental_interaction_nll_adjusted_ci": [0.04, 0.08],
                        "incremental_inflation_ratio_adjusted_ci": [2.0, 6.0],
                    },
                    "superadditive_onset_detected": True,
                }
            ],
        },
        "single_addition_attribution": {
            "context_sensitive_layers": [1],
            "layers": [
                {
                    "layer": 1,
                    "standalone_expected_excess_nll_mean": 0.01,
                    "conditional_added_excess_nll_mean": 0.05,
                    "conditional_interaction_nll_mean": 0.04,
                    "conditional_amplification_ratio_mean": 5.0,
                    "positive_interaction_wins": 3,
                    "bootstrap": {
                        "adjusted_confidence_level": 0.99375,
                        "conditional_interaction_nll_adjusted_ci": [0.02, 0.06],
                        "conditional_amplification_ratio_adjusted_ci": [2.0, 8.0],
                    },
                    "context_sensitive_layer_detected": True,
                }
            ],
        },
        "matched_layouts": {
            "layouts": [
                {"layout": "EARLY", "excess_nll_mean": 0.9},
                {"layout": "BALANCED", "excess_nll_mean": 0.8},
                {"layout": "LATE", "excess_nll_mean": 0.7},
            ],
            "pairwise_comparisons": [
                {
                    "left": "EARLY",
                    "right": "LATE",
                    "mean_nll_difference": 0.2,
                    "adjusted_ci": [0.1, 0.3],
                    "resolved": True,
                }
            ],
            "best_layout": "LATE",
            "worst_layout": "EARLY",
            "layout_spread_nll": 0.2,
            "bootstrap": {},
        },
    }
    summary = build_compact_summary(payload)
    assert summary["composition_onset"]["earliest_detected_upper_count"] == 10
    rendered = render_summary_markdown(summary)
    assert "Incremental onset localization" in rendered
    assert "Conditional single additions" in rendered
    assert "Matched 12-layer layouts" in rendered
    assert "EARLY−LATE" in rendered
