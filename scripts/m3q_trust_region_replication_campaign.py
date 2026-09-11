"""EXP-082: eight fresh-data replications of the frozen EXP-081 method."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import traceback

import jax
import numpy as np

from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from scripts import m3q_trust_region_sweep_campaign as sweep
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp082-multisplit-trust-region-replication-v1"
HF_PREFIX = "experiments/exp082-trust-region-replication"
RESULT_STEM = "extent-m3q-trust-region-replication"
REPLICATIONS = tuple(range(8))
BASE_TRAIN_OFFSET = 1_642_496
TRAIN_STRIDE = 69_632
BASE_EVAL_OFFSET = 0
EVAL_STRIDE = 8_192
SOURCE_SEEDS = (123, 456)
MILESTONES = (0, 6, 12, 18, 24)


def experiment_contract():
    return {
        "protocol": PROTOCOL,
        "confirmation_of": sweep.PROTOCOL,
        "replications": list(REPLICATIONS),
        "source_seeds": list(SOURCE_SEEDS),
        "frozen_method": {
            "arms": list(sweep.ARMS),
            "primary_arm": sweep.PRIMARY_ARM,
            "alphas": {key: list(value) for key, value in sweep.ALPHAS.items()},
            "minimum_relative_gain": sweep.MIN_RELATIVE_GAIN,
            "steps_per_layer": sweep.STEPS_PER_LAYER,
            "optimizer": "BF16 Lion lr=3e-5 warmup=64 cosine no-decay clip=1.0",
        },
        "proposal_offsets": [
            BASE_TRAIN_OFFSET + replication * TRAIN_STRIDE
            for replication in REPLICATIONS
        ],
        "calibration_offsets": [
            BASE_TRAIN_OFFSET + replication * TRAIN_STRIDE
            + sweep.TRAIN_WINDOWS * sweep.TRAIN_LENGTH
            for replication in REPLICATIONS
        ],
        "locked_validation_offsets": [
            BASE_EVAL_OFFSET + replication * EVAL_STRIDE
            for replication in REPLICATIONS
        ],
        "locked_evaluation_dataset": (
            "deepmind/pg19@4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b/"
            "validation manifest; storage.googleapis.com/deepmind-gutenberg assets"
        ),
        "uncertainty_unit": "data replication; source model seeds are reported separately",
        "gate": (
            "For each source seed: all 8 replications; negative mean TRUST-LINE "
            "final-minus-start NLL; upper normal 95% bound below zero; at least "
            "6/8 negative replications; negative mean TRUST-LINE-minus-HARD final "
            "NLL; <=1.25x excursion; and a nonzero accepted coordinate in every run."
        ),
    }


def _branch_record(inner, seed, arm):
    return inner["branches"][str(seed)][arm]


def aggregate(result):
    per_seed = {}
    all_complete = len(result.get("replications", {})) == len(REPLICATIONS)
    for seed in SOURCE_SEEDS:
        deltas, control_deltas, max_ratios, accepted_counts = [], [], [], []
        window_deltas = []
        accepted_layers = {}
        completed = 0
        for replication in REPLICATIONS:
            inner = result.get("replications", {}).get(str(replication))
            if not inner or inner.get("status") != "completed":
                continue
            trust = _branch_record(inner, seed, "TRUST-LINE")
            hard = _branch_record(inner, seed, "HARD-ACCEPT")
            if not trust.get("complete") or not hard.get("complete"):
                continue
            completed += 1
            start = float(trust["evaluations"]["0"]["student_nll"])
            final = float(trust["evaluations"]["24"]["student_nll"])
            hard_final = float(hard["evaluations"]["24"]["student_nll"])
            deltas.append(final - start)
            control_deltas.append(final - hard_final)
            max_ratios.append(max(
                float(trust["evaluations"][str(step)]["student_nll"]) / start
                for step in MILESTONES
            ))
            accepted = [
                value for value in trust.get("decisions", {}).values()
                if float(value["selected_alpha"]) > 0
            ]
            accepted_counts.append(len(accepted))
            for decision in accepted:
                layer = str(decision["layer"])
                accepted_layers[layer] = accepted_layers.get(layer, 0) + 1
            before = np.asarray(trust["evaluations"]["0"]["window_nll"], np.float64)
            after = np.asarray(trust["evaluations"]["24"]["window_nll"], np.float64)
            window_deltas.extend((after - before).tolist())

        values = np.asarray(deltas, np.float64)
        mean = float(np.mean(values)) if len(values) else None
        se = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else None
        )
        upper = mean + 1.96 * se if se is not None else None
        seed_passed = bool(
            completed == len(REPLICATIONS)
            and mean < 0 and upper < 0
            and sum(value < 0 for value in deltas) >= 6
            and float(np.mean(control_deltas)) < 0
            and max(max_ratios) <= 1.25
            and min(accepted_counts) > 0
        )
        per_seed[str(seed)] = {
            "completed_replications": completed,
            "trust_final_minus_start_nll": deltas,
            "mean_delta": mean,
            "replication_standard_error": se,
            "normal_95_upper": upper,
            "negative_replications": int(sum(value < 0 for value in deltas)),
            "trust_minus_hard_final_nll": control_deltas,
            "mean_trust_minus_hard": (
                float(np.mean(control_deltas)) if control_deltas else None
            ),
            "maximum_nll_ratio": max(max_ratios) if max_ratios else None,
            "accepted_coordinates": accepted_counts,
            "accepted_layer_frequency": accepted_layers,
            "descriptive_window_delta_mean": (
                float(np.mean(window_deltas)) if window_deltas else None
            ),
            "descriptive_window_delta_standard_error": (
                float(np.std(window_deltas, ddof=1) / np.sqrt(len(window_deltas)))
                if len(window_deltas) > 1 else None
            ),
            "gate_passed": seed_passed,
        }
    return {
        "completed_replications": sum(
            result.get("replications", {}).get(str(replication), {}).get("status")
            == "completed" for replication in REPLICATIONS
        ),
        "per_seed": per_seed,
        "scientific_gate_passed": bool(
            all_complete
            and all(per_seed[str(seed)]["gate_passed"] for seed in SOURCE_SEEDS)
        ),
        "gate_definition": experiment_contract()["gate"],
    }


def render_summary(result):
    aggregate_result = result["aggregate"]
    lines = [
        "# EXP-082 multi-split trust-region replication", "",
        f"- Status: `{result['status']}`",
        f"- Invocation duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Completed replications: `{aggregate_result['completed_replications']}/8`",
        f"- Scientific gate: `{aggregate_result['scientific_gate_passed']}`",
        "", "| Seed | Repeats | Mean ΔNLL | Replicate SE | 95% upper | "
        "Negative | Mean vs hard | Max ratio | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in SOURCE_SEEDS:
        row = aggregate_result["per_seed"][str(seed)]
        number = lambda value: "pending" if value is None else f"{value:.6f}"
        lines.append(
            f"| {seed} | {row['completed_replications']}/8 | {number(row['mean_delta'])} "
            f"| {number(row['replication_standard_error'])} "
            f"| {number(row['normal_95_upper'])} "
            f"| {row['negative_replications']}/8 "
            f"| {number(row['mean_trust_minus_hard'])} "
            f"| {number(row['maximum_nll_ratio'])} "
            f"| {row['gate_passed']} |"
        )
    lines += [
        "", "Primary uncertainty is computed across fresh data replications.",
        "Per-window statistics are retained as descriptive evidence only.",
    ]
    return "\n".join(lines) + "\n"


def replication_overrides(replication):
    train_offset = BASE_TRAIN_OFFSET + replication * TRAIN_STRIDE
    return {
        "PROTOCOL": f"exp082-trust-region-replication-{replication}-v1",
        "HF_PREFIX": f"{HF_PREFIX}/replication-{replication}",
        "RESULT_STEM": f"exp082-trust-region-replication-{replication}",
        "OUTPUT_SUBDIR": f"exp082/replication-{replication}",
        "TRAIN_OFFSET": train_offset,
        "CALIBRATION_OFFSET": train_offset + sweep.TRAIN_WINDOWS * sweep.TRAIN_LENGTH,
        "EVAL_OFFSET": BASE_EVAL_OFFSET + replication * EVAL_STRIDE,
        "EVAL_SPLIT": "validation",
        "EVAL_DATASET_CONFIG": "pg19-pinned-manifest",
        "DATA_SEED": 82_000 + replication * 1_000,
        "CHECKPOINT_RETRY_DELAYS": (2, 5, 15, 30),
    }


def run_replication(replication, args):
    overrides = replication_overrides(replication)
    original = {name: getattr(sweep, name) for name in overrides}
    arguments = [
        "--output-dir", args.output_dir,
        "--state-dir", f"/dev/shm/extent-exp082-rep-{replication}-state",
        "--exp069-state-dir", args.exp069_state_dir,
        "--exp072-state-dir", args.exp072_state_dir,
        "--qwen-cache-dir", args.qwen_cache_dir,
        "--dataset-cache-dir", args.dataset_cache_dir,
        "--max-wall-hours", "1.5",
        "--no-telegram",
    ]
    try:
        for name, value in overrides.items():
            setattr(sweep, name, value)
        return sweep.main(arguments)
    finally:
        for name, value in original.items():
            setattr(sweep, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--exp069-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--exp072-state-dir", default="/dev/shm/extent-exp072-v2-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp082-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.75)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.5 <= args.max_wall_hours <= 8.25:
        raise ValueError("EXP-082 wall budget must be 1.5-8.25 hours")

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / "exp082"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{RESULT_STEM}.json"
    summary_path = output / f"{RESULT_STEM}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    contract = experiment_contract()
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {"contract": contract, "status": "running", "replications": {},
              "durability_warnings": []}
    )
    if result["contract"] != contract:
        raise ValueError("EXP-082 resume contract mismatch")
    # An early development snapshot used a list; reject it before expensive work.
    if not isinstance(result.get("replications"), dict):
        result["replications"] = {}
    if result.get("status") == "completed":
        result["aggregate"] = aggregate(result)
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        print("EXP-082 already completed; restored result is unchanged", flush=True)
        return result
    for key in ("error", "error_type", "traceback"):
        result.pop(key, None)
    result.update(
        status="running",
        git_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    )
    stage = "startup"

    def persist(upload=False):
        result.update(
            duration_hours=(time.monotonic() - started) / 3600,
            aggregate=aggregate(result),
        )
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload:
            for local, remote in ((result_path, "latest.json"),
                                  (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(
                        local, f"{HF_PREFIX}/{remote}", hub,
                        commit_message=f"{PROTOCOL} progress",
                    )
                except Exception as exc:
                    warning = {"artifact": remote, "error_type": type(exc).__name__}
                    result.setdefault("durability_warnings", []).append(warning)
                    print(f"EXP-082 hf_summary=FAILED {warning}; local retained", flush=True)

    _safe_notify(args.telegram, f"Extent {PROTOCOL} started")
    try:
        for replication in REPLICATIONS:
            existing = result["replications"].get(str(replication))
            if existing and existing.get("status") == "completed":
                print(f"EXP-082 replication={replication} SKIP completed", flush=True)
                continue
            if time.monotonic() + 75 * 60 >= deadline:
                result["status"] = "deadline_partial"
                return result
            stage = f"replication-{replication}"
            print(f"EXP-082 replication={replication}/7 START", flush=True)
            inner = run_replication(replication, args)
            result["replications"][str(replication)] = inner
            persist(upload=True)
            if inner.get("status") != "completed":
                result["status"] = "deadline_partial"
                return result
            jax.clear_caches()
            gc.collect()
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result.update(
            status="failed", stage=stage, error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(),
        )
        raise
    finally:
        persist(upload=True)
        _safe_notify(
            args.telegram,
            f"Extent {PROTOCOL} {result['status']} "
            f"replications={result['aggregate']['completed_replications']}/8 "
            f"gate={result['aggregate']['scientific_gate_passed']}",
        )


if __name__ == "__main__":
    main()
