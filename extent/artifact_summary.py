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
    depth_results = _compact_depth_results(payload)
    if depth_results:
        summary["depth_results"] = depth_results
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
    lines.extend(
        [
            "",
            "Negative long-minus-short NLL means the larger recovery budget is better.",
            "",
        ]
    )
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
