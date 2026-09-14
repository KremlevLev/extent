"""EXP-083: eight replications of dual-domain assembled trust recovery."""
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


PROTOCOL = "exp083-dual-domain-assembled-trust-v1"
HF_PREFIX = "experiments/exp083-dual-domain-trust"
RESULT_STEM = "extent-m3q-dual-domain-trust"
SUMMARY_TITLE = "EXP-083 dual-domain assembled trust recovery"
OUTPUT_SUBDIR = "exp083"
STATE_PREFIX = "extent-exp083"
QWEN_CACHE_DIR = "/dev/shm/qwen3-1.7b-exp083-weights"
DERIVED_FROM = "EXP-082 single-domain cross-corpus failure"
REPLICATIONS = tuple(range(8))
SOURCE_SEEDS = (123, 456)
PRIMARY_ARM = "DUAL-CONSENSUS"
CONTROL_ARM = "WIKI-ONLY"
MILESTONES = (0, 6, 12, 18, 24)
TRAIN_STRIDE = 69_632
SECONDARY_STRIDE = 4_096
EVAL_STRIDE = 8_192
BASE_TRAIN_OFFSET = 0
BASE_SECONDARY_OFFSET = 0
BASE_EVAL_OFFSET = 0
SECONDARY_WINDOWS = 16
CONTROL_SELECTION_MODE = "primary"
PRIMARY_SELECTION_MODE = "dual_consensus"
DECISION_GROUP_SIZE = 1
CONTROL_GROUP_SELECTION_MODE = "coordinate"
PRIMARY_GROUP_SELECTION_MODE = "coordinate"
REQUIRED_NEGATIVE_REPLICATIONS = 6
REQUIRED_NONZERO_REPLICATIONS = 6
REQUIRED_JOINT_RESCUE_REPLICATIONS = 0
INNER_MAX_WALL_HOURS = 1.75
MINIMUM_START_HEADROOM_MINUTES = 90
INNER_PROTOCOL_PREFIX = "exp083-dual-domain-replication"
INNER_RESULT_PREFIX = "exp083-dual-domain-replication"
INNER_OUTPUT_PREFIX = "exp083/replication"
BASE_DATA_SEED = 83_000


def experiment_contract():
    contract = {
        "protocol": PROTOCOL,
        "derived_from": DERIVED_FROM,
        "replications": list(REPLICATIONS),
        "source_seeds": list(SOURCE_SEEDS),
        "arms": [CONTROL_ARM, PRIMARY_ARM],
        "primary_arm": PRIMARY_ARM,
        "alpha_grid": list(sweep.ALPHAS["TRUST-LINE"]),
        "minimum_relative_gain_per_domain": sweep.MIN_RELATIVE_GAIN,
        "steps_per_layer": sweep.STEPS_PER_LAYER,
        "proposal_and_primary_calibration": {
            "dataset": (
                "Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3/"
                "wikitext-103-raw-v1/train"
            ),
            "proposal_offsets": [
                BASE_TRAIN_OFFSET + rep * TRAIN_STRIDE for rep in REPLICATIONS
            ],
            "calibration_offsets": [
                BASE_TRAIN_OFFSET + rep * TRAIN_STRIDE
                + sweep.TRAIN_WINDOWS * sweep.TRAIN_LENGTH
                for rep in REPLICATIONS
            ],
        },
        "secondary_calibration": {
            "dataset": (
                "deepmind/pg19@4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b/"
                "validation manifest and Google-hosted assets"
            ),
            "offsets": [
                BASE_SECONDARY_OFFSET + rep * SECONDARY_STRIDE
                for rep in REPLICATIONS
            ],
            "windows": SECONDARY_WINDOWS,
            "length": 256,
        },
        "locked_evaluation": {
            "dataset": (
                "deepmind/pg19@4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b/"
                "test manifest and Google-hosted assets"
            ),
            "offsets": [
                BASE_EVAL_OFFSET + rep * EVAL_STRIDE for rep in REPLICATIONS
            ],
            "windows": 32,
            "length": 256,
        },
        "uncertainty_unit": "data replication, reported separately by source seed",
        "gate": (
            f"For each seed: all {len(REPLICATIONS)} runs; negative {PRIMARY_ARM} "
            "mean final-minus-start and upper normal 95% bound; "
            f">={REQUIRED_NEGATIVE_REPLICATIONS}/{len(REPLICATIONS)} negative; "
            f"negative mean final versus {CONTROL_ARM}; <=1.25x excursion; "
            f"nonzero acceptance in >={REQUIRED_NONZERO_REPLICATIONS}/"
            f"{len(REPLICATIONS)}."
        ),
    }
    if PRIMARY_SELECTION_MODE == "robust_consensus":
        contract["robust_selection_extension"] = {
            "primary_selection_mode": PRIMARY_SELECTION_MODE,
            "control_selection_mode": CONTROL_SELECTION_MODE,
            "minimum_window_improvement_fraction_per_domain": (
                sweep.MIN_WINDOW_IMPROVEMENT_FRACTION
            ),
        }
    if DECISION_GROUP_SIZE != 1:
        contract["interaction_aware_extension"] = {
            "decision_group_size": DECISION_GROUP_SIZE,
            "control_group_selection_mode": CONTROL_GROUP_SELECTION_MODE,
            "primary_group_selection_mode": PRIMARY_GROUP_SELECTION_MODE,
            "required_joint_only_rescue_replications_per_seed": (
                REQUIRED_JOINT_RESCUE_REPLICATIONS
            ),
            "proposal_compute": (
                "matched: two independent local proposals from each group-start state"
            ),
            "selection_compute": (
                "matched: both arms evaluate the complete two-alpha Cartesian grid "
                "on both calibration domains"
            ),
        }
        if REQUIRED_JOINT_RESCUE_REPLICATIONS:
            contract["gate"] += (
                f" Joint-only rescue in >={REQUIRED_JOINT_RESCUE_REPLICATIONS}/"
                f"{len(REPLICATIONS)} replications per seed."
            )
    return contract


def aggregate(result):
    per_seed = {}
    for seed in SOURCE_SEEDS:
        deltas, controls, ratios, accepted_counts, window_deltas = [], [], [], [], []
        joint_rescue_counts = []
        accepted_layers = {}
        completed = 0
        for replication in REPLICATIONS:
            inner = result.get("replications", {}).get(str(replication))
            if not inner or inner.get("status") != "completed":
                continue
            branches = inner["branches"][str(seed)]
            primary, control = branches[PRIMARY_ARM], branches[CONTROL_ARM]
            if not primary.get("complete") or not control.get("complete"):
                continue
            completed += 1
            start = float(primary["evaluations"]["0"]["student_nll"])
            final = float(primary["evaluations"]["24"]["student_nll"])
            control_final = float(control["evaluations"]["24"]["student_nll"])
            deltas.append(final - start)
            controls.append(final - control_final)
            ratios.append(max(
                float(primary["evaluations"][str(step)]["student_nll"]) / start
                for step in MILESTONES
            ))
            accepted = []
            for decision in primary.get("decisions", {}).values():
                accepted.extend(sweep.selected_layers(decision))
            accepted_counts.append(len(accepted))
            joint_rescue_counts.append(sum(
                bool(decision.get("joint_only_rescue"))
                for decision in primary.get("decisions", {}).values()
            ))
            for layer in accepted:
                layer = str(layer)
                accepted_layers[layer] = accepted_layers.get(layer, 0) + 1
            before = np.asarray(primary["evaluations"]["0"]["window_nll"], np.float64)
            after = np.asarray(primary["evaluations"]["24"]["window_nll"], np.float64)
            window_deltas.extend((after - before).tolist())

        values = np.asarray(deltas, np.float64)
        mean = float(np.mean(values)) if len(values) else None
        se = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else None
        )
        upper = mean + 1.96 * se if se is not None else None
        passed = bool(
            completed == len(REPLICATIONS)
            and mean < 0 and upper < 0
            and sum(value < 0 for value in deltas) >= REQUIRED_NEGATIVE_REPLICATIONS
            and float(np.mean(controls)) < 0
            and max(ratios) <= 1.25
            and sum(count > 0 for count in accepted_counts)
            >= REQUIRED_NONZERO_REPLICATIONS
            and sum(count > 0 for count in joint_rescue_counts)
            >= REQUIRED_JOINT_RESCUE_REPLICATIONS
        )
        per_seed[str(seed)] = {
            "completed_replications": completed,
            "dual_final_minus_start_nll": deltas,
            "mean_delta": mean,
            "replication_standard_error": se,
            "normal_95_upper": upper,
            "negative_replications": int(sum(value < 0 for value in deltas)),
            "dual_minus_wiki_final_nll": controls,
            "mean_dual_minus_wiki": float(np.mean(controls)) if controls else None,
            "mean_primary_minus_control": (
                float(np.mean(controls)) if controls else None
            ),
            "maximum_nll_ratio": max(ratios) if ratios else None,
            "accepted_coordinates": accepted_counts,
            "nonzero_acceptance_replications": int(sum(x > 0 for x in accepted_counts)),
            "joint_only_rescue_pairs": joint_rescue_counts,
            "joint_only_rescue_replications": int(sum(
                count > 0 for count in joint_rescue_counts
            )),
            "accepted_layer_frequency": accepted_layers,
            "descriptive_window_delta_mean": (
                float(np.mean(window_deltas)) if window_deltas else None
            ),
            "descriptive_window_delta_standard_error": (
                float(np.std(window_deltas, ddof=1) / np.sqrt(len(window_deltas)))
                if len(window_deltas) > 1 else None
            ),
            "gate_passed": passed,
        }
    return {
        "completed_replications": sum(
            result.get("replications", {}).get(str(rep), {}).get("status") == "completed"
            for rep in REPLICATIONS
        ),
        "per_seed": per_seed,
        "scientific_gate_passed": bool(
            len(result.get("replications", {})) == len(REPLICATIONS)
            and all(per_seed[str(seed)]["gate_passed"] for seed in SOURCE_SEEDS)
        ),
        "gate_definition": experiment_contract()["gate"],
    }


def render_summary(result):
    agg = result["aggregate"]
    lines = [
        f"# {SUMMARY_TITLE}", "",
        f"- Status: `{result['status']}`",
        f"- Invocation duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Completed replications: `{agg['completed_replications']}/{len(REPLICATIONS)}`",
        f"- Scientific gate: `{agg['scientific_gate_passed']}`", "",
        "| Seed | Repeats | Mean ΔNLL | Replicate SE | 95% upper | Negative | "
        f"Mean vs {CONTROL_ARM} | Nonzero | Max ratio | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    number = lambda value: "pending" if value is None else f"{value:.6f}"
    for seed in SOURCE_SEEDS:
        row = agg["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {row['completed_replications']}/{len(REPLICATIONS)} "
            f"| {number(row['mean_delta'])} "
            f"| {number(row['replication_standard_error'])} "
            f"| {number(row['normal_95_upper'])} | {row['negative_replications']}/"
            f"{len(REPLICATIONS)} "
            f"| {number(row['mean_primary_minus_control'])} "
            f"| {row['nonzero_acceptance_replications']}/{len(REPLICATIONS)} "
            f"| {number(row['maximum_nll_ratio'])} | {row['gate_passed']} |"
        )
    lines += [
        "", "Both arms compute both calibration domains; only selection differs.",
        "Primary uncertainty uses data-replication means, not correlated windows.",
    ]
    if DECISION_GROUP_SIZE != 1:
        lines += [
            "", "Joint-only rescue replications: " + ", ".join(
                f"seed {seed}="
                f"{agg['per_seed'][str(seed)]['joint_only_rescue_replications']}/"
                f"{len(REPLICATIONS)}"
                for seed in SOURCE_SEEDS
            ),
        ]
    return "\n".join(lines) + "\n"


def replication_overrides(replication):
    alpha_grid = tuple(sweep.ALPHAS["TRUST-LINE"])
    train_offset = BASE_TRAIN_OFFSET + replication * TRAIN_STRIDE
    return {
        "PROTOCOL": f"{INNER_PROTOCOL_PREFIX}-{replication}-v1",
        "HF_PREFIX": f"{HF_PREFIX}/replication-{replication}",
        "RESULT_STEM": f"{INNER_RESULT_PREFIX}-{replication}",
        "OUTPUT_SUBDIR": f"{INNER_OUTPUT_PREFIX}-{replication}",
        "SUMMARY_TITLE": f"{SUMMARY_TITLE} replication {replication}",
        "ARMS": (CONTROL_ARM, PRIMARY_ARM),
        "PRIMARY_ARM": PRIMARY_ARM,
        "CONTROL_ARM": CONTROL_ARM,
        "ALPHAS": {CONTROL_ARM: alpha_grid, PRIMARY_ARM: alpha_grid},
        "ARM_SELECTION_MODE": {
            CONTROL_ARM: CONTROL_SELECTION_MODE,
            PRIMARY_ARM: PRIMARY_SELECTION_MODE,
        },
        "DECISION_GROUP_SIZE": DECISION_GROUP_SIZE,
        "GROUP_SELECTION_MODE": {
            CONTROL_ARM: CONTROL_GROUP_SELECTION_MODE,
            PRIMARY_ARM: PRIMARY_GROUP_SELECTION_MODE,
        },
        "TRAIN_OFFSET": train_offset,
        "CALIBRATION_OFFSET": train_offset + sweep.TRAIN_WINDOWS * sweep.TRAIN_LENGTH,
        "TRAIN_DATASET_CONFIG": "wikitext-103-raw-v1",
        "CALIBRATION_DATASET_CONFIG": "wikitext-103-raw-v1",
        "SECONDARY_CALIBRATION": {
            "dataset_config": "pg19-pinned-manifest",
            "split": "validation",
            "offset": BASE_SECONDARY_OFFSET + replication * SECONDARY_STRIDE,
            "windows": SECONDARY_WINDOWS,
            "length": 256,
        },
        "EVAL_OFFSET": BASE_EVAL_OFFSET + replication * EVAL_STRIDE,
        "EVAL_SPLIT": "test",
        "EVAL_DATASET_CONFIG": "pg19-pinned-manifest",
        "DATA_SEED": BASE_DATA_SEED + replication * 1_000,
        "CHECKPOINT_RETRY_DELAYS": (2, 5, 15, 30),
    }


def run_replication(replication, args):
    overrides = replication_overrides(replication)
    original = {name: getattr(sweep, name) for name in overrides}
    arguments = [
        "--output-dir", args.output_dir,
        "--state-dir", f"/dev/shm/{STATE_PREFIX}-rep-{replication}-state",
        "--exp069-state-dir", args.exp069_state_dir,
        "--exp072-state-dir", args.exp072_state_dir,
        "--qwen-cache-dir", args.qwen_cache_dir,
        "--dataset-cache-dir", args.dataset_cache_dir,
        "--max-wall-hours", str(INNER_MAX_WALL_HOURS), "--no-telegram",
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
    parser.add_argument("--qwen-cache-dir", default=QWEN_CACHE_DIR)
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.1)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 2.0 <= args.max_wall_hours <= 8.25:
        raise ValueError(f"{PROTOCOL} wall budget must be 2-8.25 hours")

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / OUTPUT_SUBDIR
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
        raise ValueError("EXP-083 resume contract mismatch")
    if result.get("status") == "completed":
        result["aggregate"] = aggregate(result)
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        print(f"{PROTOCOL} already completed; restored result is unchanged", flush=True)
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
                    print(f"{PROTOCOL} hf_summary=FAILED {warning}; local retained", flush=True)

    _safe_notify(args.telegram, f"Extent {PROTOCOL} started")
    try:
        for replication in REPLICATIONS:
            existing = result["replications"].get(str(replication))
            if existing and existing.get("status") == "completed":
                print(f"{PROTOCOL} replication={replication} SKIP completed", flush=True)
                continue
            if time.monotonic() + MINIMUM_START_HEADROOM_MINUTES * 60 >= deadline:
                result["status"] = "deadline_partial"
                return result
            stage = f"replication-{replication}"
            print(
                f"{PROTOCOL} replication={replication + 1}/{len(REPLICATIONS)} START",
                flush=True,
            )
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
            f"replications={result['aggregate']['completed_replications']}/"
            f"{len(REPLICATIONS)} "
            f"gate={result['aggregate']['scientific_gate_passed']}",
        )


if __name__ == "__main__":
    main()
