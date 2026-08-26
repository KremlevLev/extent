from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def _duration_hours(payload: dict) -> float | None:
    try:
        start = datetime.fromisoformat(payload["started_at_utc"])
        end = datetime.fromisoformat(payload["completed_at_utc"])
    except (KeyError, TypeError, ValueError):
        return None
    return (end - start).total_seconds() / 3600


def _compact_scale_layers(payload: dict) -> list[dict]:
    compact = []
    for layer in payload.get("aggregate", {}).get("layers", []):
        arms = {}
        for name, metrics in layer.get("arms", {}).items():
            arms[name] = {
                "mean_long_minus_short_nll": metrics.get(
                    "mean_long_minus_short_nll"
                ),
                "long_minus_short_95ci": metrics.get(
                    "long_minus_short_95ci"
                ),
                "long_wins": metrics.get("long_wins"),
                "per_seed": [
                    {
                        "seed": record.get("seed"),
                        "short_nll": record.get("short_nll"),
                        "long_nll": record.get("long_nll"),
                        "long_minus_short_nll": record.get(
                            "long_minus_short_nll"
                        ),
                    }
                    for record in metrics.get("per_seed", [])
                ],
            }
        compact.append(
            {
                "target_layer": layer.get("target_layer"),
                "primary_gate_passed": layer.get(
                    "primary_joint_gate_passed"
                ),
                "arms": arms,
            }
        )
    return compact


def _compact_depth_results(payload: dict) -> list[dict]:
    compact = []
    for experiment, result in payload.get("depth_results", {}).items():
        for layer, layer_result in result.get("layer_results", {}).items():
            aggregate = layer_result.get("aggregate", {})
            compact.append(
                {
                    "experiment": experiment,
                    "target_layer": int(layer),
                    "scientific_gate_passed": layer_result.get(
                        "scientific_gate_passed"
                    ),
                    "mean_contribution_minus_mixer_nll": aggregate.get(
                        "mean_contribution_minus_mixer_nll"
                    ),
                    "mean_contribution_minus_joint_nll": aggregate.get(
                        "mean_contribution_minus_joint_nll"
                    ),
                    "contribution_wins_vs_mixer": aggregate.get(
                        "contribution_wins_vs_mixer"
                    ),
                    "contribution_wins_vs_joint": aggregate.get(
                        "contribution_wins_vs_joint"
                    ),
                    "bootstrap": layer_result.get("bootstrap"),
                }
            )
    return compact


def _compact_progressive_stages(payload: dict) -> list[dict]:
    compact = []
    for stage in payload.get("aggregate", {}).get("stages", []):
        bootstrap = stage.get("bootstrap", {})
        compact.append(
            {
                "replacement_count": stage.get("replacement_count"),
                "layers": stage.get("layers"),
                "additive_expected_excess_nll_mean": stage.get(
                    "additive_expected_excess_nll_mean"
                ),
                "observed_composed_excess_nll_mean": stage.get(
                    "observed_composed_excess_nll_mean"
                ),
                "interaction_nll_mean": stage.get("interaction_nll_mean"),
                "composition_inflation_ratio_mean": stage.get(
                    "composition_inflation_ratio_mean"
                ),
                "seed_inflation_passes": stage.get(
                    "seed_inflation_passes"
                ),
                "mean_inflation_ratio_95ci": bootstrap.get(
                    "mean_inflation_ratio_95ci"
                ),
                "scientific_gate_passed": stage.get(
                    "scientific_gate_passed"
                ),
            }
        )
    return compact


def _compact_boundary_stages(payload: dict) -> list[dict]:
    compact = []
    boundary = payload.get("aggregate", {}).get("boundary_analysis", {})
    for stage in boundary.get("stages", []):
        bootstrap = stage.get("bootstrap", {})
        compact.append(
            {
                "replacement_count": stage.get("replacement_count"),
                "internal_replacement_count": stage.get(
                    "internal_replacement_count"
                ),
                "layer0_only_excess_nll_mean": stage.get(
                    "layer0_only_excess_nll_mean"
                ),
                "internal_only_excess_nll_mean": stage.get(
                    "internal_only_excess_nll_mean"
                ),
                "full_composition_excess_nll_mean": stage.get(
                    "full_composition_excess_nll_mean"
                ),
                "boundary_interaction_nll_mean": stage.get(
                    "boundary_interaction_nll_mean"
                ),
                "internal_minus_layer0_95ci": bootstrap.get(
                    "internal_minus_layer0_95ci"
                ),
                "boundary_interaction_nll_95ci": bootstrap.get(
                    "boundary_interaction_nll_95ci"
                ),
                "mechanism_gate_passed": stage.get(
                    "mechanism_gate_passed"
                ),
            }
        )
    return compact


def build_compact_summary(payload: dict) -> dict:
    """Reduce a full campaign artifact to its scientific decision surface."""
    summary: dict[str, Any] = {
        "source": payload.get("source"),
        "method": payload.get("method"),
        "protocol": payload.get("protocol"),
        "started_at_utc": payload.get("started_at_utc"),
        "completed_at_utc": payload.get("completed_at_utc"),
        "duration_hours": _duration_hours(payload),
        "optimizer_visible_tokens_total": payload.get(
            "optimizer_visible_tokens_total"
        ),
        "passed": payload.get("passed"),
        "scientific_gate_passed": payload.get("scientific_gate_passed"),
    }
    scale_layers = _compact_scale_layers(payload)
    if scale_layers:
        summary["primary_arm"] = payload.get("aggregate", {}).get(
            "primary_arm"
        )
        summary["scale_layers"] = scale_layers
        if payload.get("aggregate", {}).get("depth_trend"):
            summary["depth_trend"] = payload["aggregate"]["depth_trend"]
    depth_results = _compact_depth_results(payload)
    if depth_results:
        summary["depth_results"] = depth_results
    progressive = _compact_progressive_stages(payload)
    if progressive:
        summary["progressive_composition"] = {
            "primary_replacement_count": payload.get("aggregate", {}).get(
                "primary_replacement_count"
            ),
            "baseline_reproduction": payload.get("aggregate", {}).get(
                "baseline_reproduction"
            ),
            "stages": progressive,
        }
    boundary = _compact_boundary_stages(payload)
    if boundary:
        aggregate = payload.get("aggregate", {})
        incremental = aggregate.get("incremental_scaling", {})
        incremental_bootstrap = incremental.get("bootstrap", {})
        summary["boundary_scaling"] = {
            "composition_scaling_gate_passed": aggregate.get(
                "composition_scaling_gate_passed"
            ),
            "boundary_mechanism_gate_passed": aggregate.get(
                "boundary_mechanism_gate_passed"
            ),
            "stages": boundary,
            "incremental": {
                "lower_replacement_count": incremental.get(
                    "lower_replacement_count"
                ),
                "upper_replacement_count": incremental.get(
                    "upper_replacement_count"
                ),
                "added_layers": incremental.get("added_layers"),
                "expected_added_excess_nll_mean": incremental.get(
                    "expected_added_excess_nll_mean"
                ),
                "observed_added_excess_nll_mean": incremental.get(
                    "observed_added_excess_nll_mean"
                ),
                "incremental_interaction_nll_mean": incremental.get(
                    "incremental_interaction_nll_mean"
                ),
                "incremental_inflation_ratio_mean": incremental.get(
                    "incremental_inflation_ratio_mean"
                ),
                "incremental_inflation_ratio_95ci": incremental_bootstrap.get(
                    "incremental_inflation_ratio_95ci"
                ),
                "seed_inflation_passes": incremental.get(
                    "seed_inflation_passes"
                ),
                "scientific_gate_passed": incremental.get(
                    "scientific_gate_passed"
                ),
            },
        }
    context = payload.get("context_aggregate")
    if context:
        summary["context_transfer"] = {
            "contribution_best_cells": context.get("contribution_best_cells"),
            "required_contribution_best_cells": context.get(
                "required_contribution_best_cells"
            ),
            "maximum_contribution_regret_nll": context.get(
                "maximum_contribution_regret_nll"
            ),
            "maximum_allowed_contribution_regret_nll": context.get(
                "maximum_allowed_contribution_regret_nll"
            ),
            "scientific_gate_passed": context.get("scientific_gate_passed"),
            "cells": context.get("cells"),
        }
    return summary


def render_summary_markdown(summary: dict) -> str:
    lines = [
        f"# {summary.get('protocol', 'Experiment')} compact summary",
        "",
        f"- Numerical pass: `{summary.get('passed')}`",
        f"- Scientific gate: `{summary.get('scientific_gate_passed')}`",
    ]
    if summary.get("duration_hours") is not None:
        lines.append(f"- Duration: `{summary['duration_hours']:.3f}` hours")
    if summary.get("optimizer_visible_tokens_total") is not None:
        lines.append(
            "- Optimizer-visible tokens: "
            f"`{summary['optimizer_visible_tokens_total']:,}`"
        )
    if summary.get("scale_layers"):
        lines.extend(["", "## Long-minus-short NLL", ""])
        for layer in summary["scale_layers"]:
            lines.append(
                f"### Layer {layer['target_layer']} "
                f"(primary gate: `{layer['primary_gate_passed']}`)"
            )
            lines.append("")
            lines.append("| Arm | Mean ΔNLL | 95% CI | Wins |")
            lines.append("|---|---:|---:|---:|")
            for arm, metrics in layer["arms"].items():
                interval = metrics.get("long_minus_short_95ci") or [None, None]
                lines.append(
                    f"| {arm} | {metrics.get('mean_long_minus_short_nll'):.8f} "
                    f"| [{interval[0]:.8f}, {interval[1]:.8f}] "
                    f"| {metrics.get('long_wins')}/3 |"
                )
            lines.append("")
    if summary.get("depth_trend"):
        trend = summary["depth_trend"]
        lines.extend(["", "## Depth trend", ""])
        for arm, metrics in trend.get("arms", {}).items():
            interval = metrics["slope_95ci"]
            lines.append(
                f"- {arm}: slope `{metrics['nll_delta_per_layer_index_slope']:.8f}` "
                f"with 95% CI `[{interval[0]:.8f}, {interval[1]:.8f}]`."
            )
        lines.append(
            "- Depth-trend scientific gate: "
            f"`{trend.get('scientific_gate_passed')}`"
        )
    if summary.get("depth_results"):
        lines.extend(["", "## Depth objective results", ""])
        lines.append("| Experiment | Layer | Gate | Contribution−Mixer | Contribution−Joint |")
        lines.append("|---|---:|---:|---:|---:|")
        for result in summary["depth_results"]:
            lines.append(
                f"| {result['experiment']} | {result['target_layer']} "
                f"| {result['scientific_gate_passed']} "
                f"| {result['mean_contribution_minus_mixer_nll']:.8f} "
                f"| {result['mean_contribution_minus_joint_nll']:.8f} |"
            )
    progressive = summary.get("progressive_composition")
    if progressive:
        lines.extend(["", "## Progressive composition", ""])
        lines.append(
            "| Replacements | Layers | Additive excess | Observed excess "
            "| Interaction | Inflation | 95% CI | Seed passes | Gate |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for stage in progressive["stages"]:
            interval = stage.get("mean_inflation_ratio_95ci") or [None, None]
            interval_text = (
                f"[{interval[0]:.4f}, {interval[1]:.4f}]"
                if interval[0] is not None and interval[1] is not None
                else "n/a"
            )
            inflation = stage.get("composition_inflation_ratio_mean")
            inflation_text = f"{inflation:.4f}" if inflation is not None else "n/a"
            lines.append(
                f"| {stage['replacement_count']} "
                f"| {','.join(str(layer) for layer in stage['layers'])} "
                f"| {stage['additive_expected_excess_nll_mean']:.8f} "
                f"| {stage['observed_composed_excess_nll_mean']:.8f} "
                f"| {stage['interaction_nll_mean']:.8f} "
                f"| {inflation_text} | {interval_text} "
                f"| {stage['seed_inflation_passes']}/3 "
                f"| {stage['scientific_gate_passed']} |"
            )
        baseline = progressive.get("baseline_reproduction") or {}
        lines.append("")
        lines.append(
            "- Original-Qwen reproduction: "
            f"`{baseline.get('passed')}`; maximum |ΔNLL| "
            f"`{baseline.get('maximum_absolute_mean_nll_difference')}`."
        )
        lines.append(
            "- Primary replacement count: "
            f"`{progressive.get('primary_replacement_count')}`."
        )
    boundary = summary.get("boundary_scaling")
    if boundary:
        lines.extend(["", "## Layer-0 boundary decomposition", ""])
        lines.append(
            "| Full replacements | Internal replacements | Layer 0 excess "
            "| Internal excess | Full excess | Boundary interaction "
            "| Internal−layer 0 CI | Interaction CI | Gate |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for stage in boundary["stages"]:
            dominance_ci = stage.get("internal_minus_layer0_95ci") or [None, None]
            interaction_ci = stage.get("boundary_interaction_nll_95ci") or [
                None,
                None,
            ]
            lines.append(
                f"| {stage['replacement_count']} "
                f"| {stage['internal_replacement_count']} "
                f"| {stage['layer0_only_excess_nll_mean']:.8f} "
                f"| {stage['internal_only_excess_nll_mean']:.8f} "
                f"| {stage['full_composition_excess_nll_mean']:.8f} "
                f"| {stage['boundary_interaction_nll_mean']:.8f} "
                f"| [{dominance_ci[0]:.4f}, {dominance_ci[1]:.4f}] "
                f"| [{interaction_ci[0]:.4f}, {interaction_ci[1]:.4f}] "
                f"| {stage['mechanism_gate_passed']} |"
            )
        incremental = boundary["incremental"]
        interval = incremental.get("incremental_inflation_ratio_95ci") or [
            None,
            None,
        ]
        ratio = incremental.get("incremental_inflation_ratio_mean")
        ratio_text = f"{ratio:.4f}" if ratio is not None else "n/a"
        interval_text = (
            f"[{interval[0]:.4f}, {interval[1]:.4f}]"
            if interval[0] is not None and interval[1] is not None
            else "n/a"
        )
        lines.extend(["", "## Incremental 8-to-16 scaling", ""])
        lines.append(
            "- Added layers: `"
            + ",".join(str(layer) for layer in incremental["added_layers"])
            + "`."
        )
        lines.append(
            "- Expected / observed added excess NLL: "
            f"`{incremental['expected_added_excess_nll_mean']:.8f}` / "
            f"`{incremental['observed_added_excess_nll_mean']:.8f}`."
        )
        lines.append(
            "- Incremental interaction / inflation: "
            f"`{incremental['incremental_interaction_nll_mean']:.8f}` / "
            f"`{ratio_text}`; 95% CI `{interval_text}`."
        )
        lines.append(
            "- Composition-scaling gate: "
            f"`{boundary['composition_scaling_gate_passed']}`; "
            "boundary-mechanism gate: "
            f"`{boundary['boundary_mechanism_gate_passed']}`."
        )
    context = summary.get("context_transfer")
    if context:
        lines.extend(
            [
                "",
                "## Context transfer",
                "",
                f"- Best cells: `{context['contribution_best_cells']}` / "
                f"`{context['required_contribution_best_cells']}`",
                "- Maximum regret: "
                f"`{context['maximum_contribution_regret_nll']:.8f}` "
                "(allowed "
                f"`{context['maximum_allowed_contribution_regret_nll']:.8f}`)",
            ]
        )
    if summary.get("scale_layers"):
        lines.extend(
            [
                "",
                "Negative long-minus-short NLL means the larger recovery budget is better.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_compact_summary(
    payload: dict,
    *,
    artifact_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    artifact = Path(artifact_path)
    base = artifact.with_suffix("")
    summary_json = base.with_name(f"{base.name}-summary.json")
    summary_md = base.with_name(f"{base.name}-summary.md")
    summary = build_compact_summary(payload)
    _atomic_write_text(summary_json, json.dumps(summary, indent=2) + "\n")
    _atomic_write_text(summary_md, render_summary_markdown(summary))
    mirrors = []
    if output_dir:
        mirror_root = Path(output_dir)
        for path in (summary_json, summary_md):
            mirror = mirror_root / path.name
            if mirror.resolve() != path.resolve():
                _atomic_write_text(mirror, path.read_text(encoding="utf-8"))
                mirrors.append(str(mirror.resolve()))
    return {
        "summary_json": str(summary_json.resolve()),
        "summary_markdown": str(summary_md.resolve()),
        "mirrors": mirrors,
    }
