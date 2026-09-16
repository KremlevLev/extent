"""EXP-081: assembled-model trust-region coordinate recovery."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_pg19_tokens, load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import full_model_eval_metrics
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import abstract_parameter_tree, initialize_sharded_parameters
from extent.model import HybridDecoderLayer
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sequential_recovery import make_conditional_recovery_step
from extent.sharding import batch_sharding, named_sharding_tree, validate_partition_specs
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_allocation_campaign import contract_for
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072
from scripts.m3q_full_depth_sequential_confirmation import configured_contract
from scripts.m3q_sequential_recovery_campaign import endpoint_contract as sequential_endpoint_contract
from scripts.m3q_sequential_recovery_campaign import local_layer_params, require_v5e8
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp081-assembled-trust-region-sweep-v1"
HF_PREFIX = "experiments/exp081-trust-region-sweep"
RESULT_STEM = "extent-m3q-trust-region-sweep"
OUTPUT_SUBDIR = "exp081"
SUMMARY_TITLE = "EXP-081 assembled-model trust-region sweep"
ARMS = ("HARD-ACCEPT", "TRUST-LINE")
PRIMARY_ARM = "TRUST-LINE"
CONTROL_ARM = "HARD-ACCEPT"
SEEDS = (123, 456)
ALPHAS = {
    "HARD-ACCEPT": (0.0, 1.0),
    "TRUST-LINE": (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
}
MIN_RELATIVE_GAIN = 0.001
MIN_WINDOW_IMPROVEMENT_FRACTION = 0.60
STEPS_PER_LAYER = 2048
MILESTONES = (6, 12, 18, 24)
TRAIN_WINDOWS, TRAIN_LENGTH, TRAIN_OFFSET = 512, 128, 1_572_864
CALIBRATION_WINDOWS, CALIBRATION_LENGTH, CALIBRATION_OFFSET = 32, 128, 1_638_400
TRAIN_DATASET_CONFIG = "wikitext-2-raw-v1"
CALIBRATION_DATASET_CONFIG = "wikitext-2-raw-v1"
SECONDARY_CALIBRATION = None
ARM_SELECTION_MODE = {arm: "primary" for arm in ARMS}
ARM_MIN_RELATIVE_GAIN = {arm: MIN_RELATIVE_GAIN for arm in ARMS}
EVAL_WINDOWS, EVAL_LENGTH, EVAL_OFFSET = 32, 256, 196_608
EVAL_SPLIT = "test"
EVAL_DATASET_CONFIG = "wikitext-2-raw-v1"
DATA_SEED = 81_000
CHECKPOINT_RETRY_DELAYS = None
DECISION_GROUP_SIZE = 1
GROUP_SELECTION_MODE = {arm: "coordinate" for arm in ARMS}


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def pair_key(first: float, second: float) -> str:
    return f"{first:g},{second:g}"


def restored_sequence(values):
    """Read sequence fields from memory or numeric-key checkpoint JSON."""
    if isinstance(values, dict):
        return [values[key] for key in sorted(values, key=lambda key: int(key))]
    return values


def selected_layers(decision: dict) -> list[int]:
    if "selected_alphas" in decision:
        return [
            int(layer) for layer, alpha in zip(
                restored_sequence(decision["layers"]),
                restored_sequence(decision["selected_alphas"]), strict=True
            ) if float(alpha) > 0
        ]
    return [int(decision["layer"])] if float(decision["selected_alpha"]) > 0 else []


def selected_coordinate_count(decisions: dict) -> int:
    total = 0
    for decision in decisions.values():
        if "selected_alphas" in decision:
            total += sum(
                float(alpha) > 0
                for alpha in restored_sequence(decision["selected_alphas"])
            )
        else:
            total += int(float(decision["selected_alpha"]) > 0)
    return int(total)


def choose_trust_alpha(scores: dict[str, float], minimum_relative_gain=MIN_RELATIVE_GAIN):
    """Choose the lowest-KL alpha, retaining alpha zero without enough gain."""
    baseline = float(scores[alpha_key(0.0)])
    best_key = min(scores, key=lambda key: (float(scores[key]), float(key)))
    best = float(scores[best_key])
    gain = (baseline - best) / max(abs(baseline), 1e-12)
    if float(best_key) == 0.0 or gain < minimum_relative_gain:
        return 0.0, float(max(gain, 0.0))
    return float(best_key), float(gain)


def choose_consensus_alpha(
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
    minimum_relative_gain=MIN_RELATIVE_GAIN,
):
    """Minimize worst-domain KL ratio among candidates improving both domains."""
    primary_base = float(primary_scores[alpha_key(0.0)])
    secondary_base = float(secondary_scores[alpha_key(0.0)])
    eligible = []
    for key in primary_scores:
        alpha = float(key)
        if alpha == 0.0:
            continue
        primary_gain = (primary_base - float(primary_scores[key])) / max(
            abs(primary_base), 1e-12
        )
        secondary_gain = (secondary_base - float(secondary_scores[key])) / max(
            abs(secondary_base), 1e-12
        )
        if primary_gain >= minimum_relative_gain and secondary_gain >= minimum_relative_gain:
            worst_ratio = max(
                float(primary_scores[key]) / max(abs(primary_base), 1e-12),
                float(secondary_scores[key]) / max(abs(secondary_base), 1e-12),
            )
            eligible.append((worst_ratio, alpha, primary_gain, secondary_gain))
    if not eligible:
        return 0.0, 0.0, 0.0
    _, alpha, primary_gain, secondary_gain = min(eligible)
    return float(alpha), float(primary_gain), float(secondary_gain)


def choose_robust_consensus_alpha(
    primary_metrics: dict[str, dict],
    secondary_metrics: dict[str, dict],
    minimum_relative_gain=MIN_RELATIVE_GAIN,
    minimum_window_fraction=MIN_WINDOW_IMPROVEMENT_FRACTION,
):
    """Require mean and paired-window improvement in both calibration domains."""
    primary_scores = {
        key: float(value["prediction_kl"]) for key, value in primary_metrics.items()
    }
    secondary_scores = {
        key: float(value["prediction_kl"]) for key, value in secondary_metrics.items()
    }
    primary_base = primary_scores[alpha_key(0.0)]
    secondary_base = secondary_scores[alpha_key(0.0)]
    primary_windows = np.asarray(
        primary_metrics[alpha_key(0.0)]["window_prediction_kl"], np.float64
    )
    secondary_windows = np.asarray(
        secondary_metrics[alpha_key(0.0)]["window_prediction_kl"], np.float64
    )
    eligible = []
    for key in primary_scores:
        alpha = float(key)
        if alpha == 0.0:
            continue
        primary_gain = (primary_base - primary_scores[key]) / max(abs(primary_base), 1e-12)
        secondary_gain = (secondary_base - secondary_scores[key]) / max(abs(secondary_base), 1e-12)
        primary_fraction = float(np.mean(
            np.asarray(primary_metrics[key]["window_prediction_kl"], np.float64)
            < primary_windows
        ))
        secondary_fraction = float(np.mean(
            np.asarray(secondary_metrics[key]["window_prediction_kl"], np.float64)
            < secondary_windows
        ))
        if (
            primary_gain >= minimum_relative_gain
            and secondary_gain >= minimum_relative_gain
            and primary_fraction >= minimum_window_fraction
            and secondary_fraction >= minimum_window_fraction
        ):
            worst_ratio = max(
                primary_scores[key] / max(abs(primary_base), 1e-12),
                secondary_scores[key] / max(abs(secondary_base), 1e-12),
            )
            eligible.append((
                worst_ratio, -min(primary_fraction, secondary_fraction), alpha,
                primary_gain, secondary_gain, primary_fraction, secondary_fraction,
            ))
    if not eligible:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    (_, _, alpha, primary_gain, secondary_gain,
     primary_fraction, secondary_fraction) = min(eligible)
    return (
        float(alpha), float(primary_gain), float(secondary_gain),
        float(primary_fraction), float(secondary_fraction),
    )


def choose_pair_consensus(
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
    minimum_relative_gain=MIN_RELATIVE_GAIN,
):
    """Choose a two-coordinate point that improves both domain means."""
    baseline_key = pair_key(0.0, 0.0)
    primary_base = float(primary_scores[baseline_key])
    secondary_base = float(secondary_scores[baseline_key])
    eligible = []
    for key in primary_scores:
        first, second = (float(value) for value in key.split(","))
        if first == 0.0 and second == 0.0:
            continue
        primary_gain = (primary_base - float(primary_scores[key])) / max(
            abs(primary_base), 1e-12
        )
        secondary_gain = (secondary_base - float(secondary_scores[key])) / max(
            abs(secondary_base), 1e-12
        )
        if primary_gain >= minimum_relative_gain and secondary_gain >= minimum_relative_gain:
            worst_ratio = max(
                float(primary_scores[key]) / max(abs(primary_base), 1e-12),
                float(secondary_scores[key]) / max(abs(secondary_base), 1e-12),
            )
            eligible.append((
                worst_ratio, first + second, first, second,
                primary_gain, secondary_gain,
            ))
    if not eligible:
        return (0.0, 0.0), 0.0, 0.0
    _, _, first, second, primary_gain, secondary_gain = min(eligible)
    return (float(first), float(second)), float(primary_gain), float(secondary_gain)


def choose_greedy_from_pair_grid(
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
    alphas,
    minimum_relative_gain=MIN_RELATIVE_GAIN,
):
    """Make two sequential decisions while paying for the complete pair grid."""
    first_primary = {
        alpha_key(alpha): primary_scores[pair_key(alpha, 0.0)] for alpha in alphas
    }
    first_secondary = {
        alpha_key(alpha): secondary_scores[pair_key(alpha, 0.0)] for alpha in alphas
    }
    first, _, _ = choose_consensus_alpha(
        first_primary, first_secondary, minimum_relative_gain
    )
    second_primary = {
        alpha_key(alpha): primary_scores[pair_key(first, alpha)] for alpha in alphas
    }
    second_secondary = {
        alpha_key(alpha): secondary_scores[pair_key(first, alpha)] for alpha in alphas
    }
    second, _, _ = choose_consensus_alpha(
        second_primary, second_secondary, minimum_relative_gain
    )
    baseline_key = pair_key(0.0, 0.0)
    selected_key = pair_key(first, second)
    primary_base = float(primary_scores[baseline_key])
    secondary_base = float(secondary_scores[baseline_key])
    primary_gain = (primary_base - float(primary_scores[selected_key])) / max(
        abs(primary_base), 1e-12
    )
    secondary_gain = (secondary_base - float(secondary_scores[selected_key])) / max(
        abs(secondary_base), 1e-12
    )
    return (float(first), float(second)), float(primary_gain), float(secondary_gain)


def is_joint_only_rescue(
    selected: tuple[float, float],
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
    minimum_relative_gain=MIN_RELATIVE_GAIN,
) -> bool:
    """True when a selected pair works although neither component does alone."""
    first, second = selected
    if first <= 0 or second <= 0:
        return False
    base = pair_key(0.0, 0.0)

    def eligible(key):
        return all(
            (float(scores[base]) - float(scores[key]))
            / max(abs(float(scores[base])), 1e-12) >= minimum_relative_gain
            for scores in (primary_scores, secondary_scores)
        )

    return (
        eligible(pair_key(first, second))
        and not eligible(pair_key(first, 0.0))
        and not eligible(pair_key(0.0, second))
    )


def blend_parameters(current, proposal, alpha: float):
    def blend(old, new):
        value = old.astype(jnp.float32) + alpha * (
            new.astype(jnp.float32) - old.astype(jnp.float32)
        )
        return value.astype(old.dtype)
    return jax.tree.map(blend, current, proposal)


def experiment_contract(config):
    contract = {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_sequential_contract": configured_contract(config),
        "arms": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "seeds": list(SEEDS),
        "forward_order": list(config.mamba_layer_indices),
        "alphas": {key: list(value) for key, value in ALPHAS.items()},
        "minimum_relative_calibration_kl_gain": MIN_RELATIVE_GAIN,
        "steps_per_layer": STEPS_PER_LAYER,
        "milestones": list(MILESTONES),
        "train": ["train", TRAIN_OFFSET, TRAIN_WINDOWS, TRAIN_LENGTH],
        "calibration": [
            "train", CALIBRATION_OFFSET, CALIBRATION_WINDOWS, CALIBRATION_LENGTH
        ],
        "locked_evaluation": (
            [EVAL_SPLIT, EVAL_OFFSET, EVAL_WINDOWS, EVAL_LENGTH]
            if EVAL_DATASET_CONFIG == "wikitext-2-raw-v1"
            else [EVAL_DATASET_CONFIG, EVAL_SPLIT, EVAL_OFFSET, EVAL_WINDOWS, EVAL_LENGTH]
        ),
        "data_seed": DATA_SEED,
        "proposal_objective": "conditional decoder-contribution relative MSE",
        "acceptance_objective": "assembled-model prediction KL against frozen Qwen",
        "optimizer": "BF16 Lion lr=3e-5 warmup=64 cosine no-decay clip=1.0",
        "trainable": "one proposed Mamba subtree; accepted interpolation only",
    }
    if (
        TRAIN_DATASET_CONFIG != "wikitext-2-raw-v1"
        or CALIBRATION_DATASET_CONFIG != "wikitext-2-raw-v1"
        or SECONDARY_CALIBRATION is not None
        or CONTROL_ARM != "HARD-ACCEPT"
        or any(ARM_SELECTION_MODE.get(arm) != "primary" for arm in ARMS)
        or DECISION_GROUP_SIZE != 1
    ):
        extension = {
            "train_dataset_config": TRAIN_DATASET_CONFIG,
            "calibration_dataset_config": CALIBRATION_DATASET_CONFIG,
            "secondary_calibration": SECONDARY_CALIBRATION,
            "arm_selection_mode": dict(ARM_SELECTION_MODE),
            "control_arm": CONTROL_ARM,
        }
        if DECISION_GROUP_SIZE != 1:
            extension.update(
                decision_group_size=DECISION_GROUP_SIZE,
                group_selection_mode=dict(GROUP_SELECTION_MODE),
                pair_proposal_contract=(
                    "Both proposals are trained from the unchanged group-start "
                    "assembled state; both arms evaluate the complete Cartesian "
                    "alpha grid on both calibration domains."
                ),
            )
            if any(
                ARM_MIN_RELATIVE_GAIN.get(arm, MIN_RELATIVE_GAIN)
                != MIN_RELATIVE_GAIN
                for arm in ARMS
            ):
                extension["arm_minimum_relative_calibration_kl_gain"] = dict(
                    ARM_MIN_RELATIVE_GAIN
                )
        if any(ARM_SELECTION_MODE.get(arm) == "robust_consensus" for arm in ARMS):
            extension["minimum_window_improvement_fraction"] = (
                MIN_WINDOW_IMPROVEMENT_FRACTION
            )
        contract["dataset_and_selection_extension"] = extension
    return contract


def checkpoint_contract(contract, seed, arm, position, source_hashes):
    return dict(
        contract,
        kind="assembled_trust_region_milestone",
        seed=int(seed), arm=arm, position=int(position),
        source_exp072_checkpoint_sha256={str(k): v for k, v in source_hashes.items()},
    )


def aggregate(result):
    rows, pairs = [], []
    for seed in SEEDS:
        branches = result.get("branches", {}).get(str(seed), {})
        for arm in ARMS:
            row = branches.get(arm, {})
            if not row.get("complete"):
                continue
            curve = np.asarray([
                row["evaluations"][str(step)]["student_nll"]
                for step in (0, *MILESTONES)
            ], np.float64)
            rows.append({
                "seed": seed, "arm": arm,
                "initial_nll": float(curve[0]), "final_nll": float(curve[-1]),
                "final_delta": float(curve[-1] - curve[0]),
                "maximum_nll_ratio": float(np.max(curve) / curve[0]),
                "accepted_coordinates": selected_coordinate_count(
                    row.get("decisions", {})
                ),
            })
        if all(branches.get(arm, {}).get("complete") for arm in ARMS):
            pairs.append({
                "seed": seed,
                "trust_minus_hard_final_nll": float(
                    branches[PRIMARY_ARM]["evaluations"]["24"]["student_nll"]
                    - branches[CONTROL_ARM]["evaluations"]["24"]["student_nll"]
                ),
            })
    primary = [row for row in rows if row["arm"] == PRIMARY_ARM]
    passed = bool(
        len(primary) == len(SEEDS) and len(pairs) == len(SEEDS)
        and all(row["final_delta"] < 0 for row in primary)
        and all(row["maximum_nll_ratio"] <= 1.25 for row in primary)
        and all(row["accepted_coordinates"] > 0 for row in primary)
        and all(row["trust_minus_hard_final_nll"] < 0 for row in pairs)
    )
    return {
        "completed_primary_seeds": len(primary), "completed_pairs": len(pairs),
        "results": rows, "paired_results": pairs,
        "scientific_gate_passed": passed,
        "gate_definition": (
            f"At both seeds {PRIMARY_ARM} improves locked final NLL, stays within "
            f"1.25x start, accepts a nonzero coordinate, and beats {CONTROL_ARM}."
        ),
    }


def render_summary(result):
    lines = [
        f"# {SUMMARY_TITLE}", "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
        f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`",
        "", "| Seed | Arm | Position | Test NLL | KL | Accepted |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, branches in result.get("branches", {}).items():
        for arm, row in branches.items():
            accepted = selected_coordinate_count(row.get("decisions", {}))
            for position, metrics in row.get("evaluations", {}).items():
                lines.append(
                    f"| {seed} | {arm} | {position} | {metrics['student_nll']:.6f} "
                    f"| {metrics['prediction_kl']:.6f} | {accepted} |"
                )
    lines += [
        "", "Alpha selection uses only the disjoint calibration split.",
        "Locked test metrics are observed only at registered milestones.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp081-state")
    parser.add_argument("--exp069-state-dir", default="/dev/shm/extent-exp069-state")
    parser.add_argument("--exp072-state-dir", default="/dev/shm/extent-exp072-v2-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp081-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=7.5)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.0 <= args.max_wall_hours <= 8.25:
        raise ValueError("EXP-081 wall budget must be 1-8.25 hours")

    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir) / OUTPUT_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{RESULT_STEM}.json"
    summary_path = output / f"{RESULT_STEM}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    config, _ = load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )
    contract = experiment_contract(config)
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {"contract": contract, "status": "running", "branches": {},
              "checkpoint_events": [], "durability_warnings": []}
    )
    if result["contract"] != contract:
        raise ValueError("EXP-081 resume contract mismatch")
    if result.get("status") == "completed":
        print("EXP-081 already completed; restored result is unchanged", flush=True)
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
    store = CampaignCheckpointStore(
        Path(args.state_dir), f"{HF_PREFIX}/checkpoints", hub,
        retry_delays=CHECKPOINT_RETRY_DELAYS,
    )
    seq_store = CampaignCheckpointStore(
        Path(args.exp072_state_dir), f"{EXP072['HF_PREFIX']}/checkpoints", hub
    )
    base_store = CampaignCheckpointStore(
        Path(args.exp069_state_dir), f"{EXP069_PREFIX}/checkpoints", hub
    )
    stage = "startup"

    def persist(upload=False):
        result.update(
            duration_hours=(time.monotonic() - started) / 3600,
            checkpoint_events=store.events + seq_store.events + base_store.events,
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
                    print(f"hf_summary=FAILED {warning}; local retained", flush=True)

    _safe_notify(args.telegram, f"Extent {PROTOCOL} started")
    try:
        mesh, devices = require_v5e8()
        result["devices"] = [str(device) for device in devices]
        batch_layout = batch_sharding(mesh)
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16",
            remat_policy="full",
        )
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        initialized = initialize_sharded_parameters(
            teacher, jax.random.key(810),
            jax.device_put(np.zeros((1, 1), np.int32), batch_layout), mesh,
        )
        teacher_params, _ = stream_teacher_qwen_weights(initialized.params, reader, source)
        del initialized

        def tokens(count, length, offset, split, dataset_config="wikitext-2-raw-v1"):
            if dataset_config == "pg19-pinned-manifest":
                return load_pg19_tokens(
                    count * length, args.dataset_cache_dir,
                    tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                    tokenizer_revision=QWEN3_1_7B_BASE.revision,
                    token_offset=offset, dataset_split=split,
                ).reshape(count, length)
            return load_wikitext2_tokens(
                count * length, args.dataset_cache_dir,
                tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision,
                token_offset=offset, dataset_split=split,
                dataset_config=dataset_config,
            ).reshape(count, length)

        train_tokens = tokens(
            TRAIN_WINDOWS, TRAIN_LENGTH, TRAIN_OFFSET, "train",
            TRAIN_DATASET_CONFIG,
        )
        calibration_tokens = tokens(
            CALIBRATION_WINDOWS, CALIBRATION_LENGTH, CALIBRATION_OFFSET, "train",
            CALIBRATION_DATASET_CONFIG,
        )
        secondary_calibration_tokens = None
        if SECONDARY_CALIBRATION is not None:
            secondary_calibration_tokens = tokens(
                SECONDARY_CALIBRATION["windows"], SECONDARY_CALIBRATION["length"],
                SECONDARY_CALIBRATION["offset"], SECONDARY_CALIBRATION["split"],
                SECONDARY_CALIBRATION["dataset_config"],
            )
        eval_tokens = tokens(
            EVAL_WINDOWS, EVAL_LENGTH, EVAL_OFFSET, EVAL_SPLIT,
            EVAL_DATASET_CONFIG,
        )
        hashes = {
            "train": hashlib.sha256(train_tokens.tobytes()).hexdigest(),
            "calibration": hashlib.sha256(calibration_tokens.tobytes()).hexdigest(),
            "eval": hashlib.sha256(eval_tokens.tobytes()).hexdigest(),
        }
        if secondary_calibration_tokens is not None:
            hashes["secondary_calibration"] = hashlib.sha256(
                secondary_calibration_tokens.tobytes()
            ).hexdigest()
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-081 token content changed")
        result["data_sha256"] = hashes

        order = tuple(config.mamba_layer_indices)
        source_contract = contract_for(config)
        seq_contract = contract["source_sequential_contract"]
        base, base_hashes = {}, {}
        for seed in SEEDS:
            base[seed], base_hashes[seed] = {}, {}
            for depth, layer in enumerate(order, 1):
                base_slot = f"prep/seed-{seed}/layer-{layer}"
                base_meta = base_store.metadata(
                    base_slot,
                    dict(source_contract, seed=seed, layer=layer, kind="prepared_mamba"),
                )
                if base_meta is None:
                    raise FileNotFoundError(f"missing EXP-069 endpoint: {base_slot}")
                source_slot = f"seed-{seed}/ONPOLICY/layer-{layer}"
                source_c = sequential_endpoint_contract(
                    seq_contract, seed, "ONPOLICY", layer, depth,
                    base_meta["checkpoint_sha256"],
                )
                restored = seq_store.restore(source_slot, source_c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP-072 endpoint: {source_slot}")
                payload, meta = restored
                base[seed][layer] = payload["params"]
                base_hashes[seed][layer] = meta["checkpoint_sha256"]

        teacher_forward = jax.jit(lambda p, value: teacher.apply({"params": p}, value))

        def teacher_logits(windows):
            values = []
            for window in windows:
                value = teacher_forward(
                    teacher_params, jax.device_put(window[None], batch_layout)
                )
                values.append(np.asarray(jax.device_get(value), np.float32))
            return values

        calibration_teacher_logits = teacher_logits(calibration_tokens)
        secondary_calibration_teacher_logits = (
            teacher_logits(secondary_calibration_tokens)
            if secondary_calibration_tokens is not None else None
        )
        eval_teacher_logits = teacher_logits(eval_tokens)

        generic_cfg = replace(
            config,
            attention_layer_indices=tuple(
                index for index in range(config.num_layers) if index != 0
            ),
        )
        student_layer = HybridDecoderLayer(generic_cfg, 0)
        teacher_layer = Qwen3DecoderLayer(source)
        tx = create_lion(
            learning_rate=3e-5, warmup_steps=64, total_steps=STEPS_PER_LAYER,
            weight_decay=0.0, max_grad_norm=1.0,
        )
        train_step = jax.jit(
            make_conditional_recovery_step(student_layer, tx, bf16_gradients=True)
        )
        conditional_teacher = jax.jit(
            lambda p, x: teacher_layer.apply(
                {"params": p}, x,
                jnp.arange(x.shape[1], dtype=jnp.int32)[None], None,
            )
        )
        embedding_table = teacher_params["embed_tokens"]["embedding"]
        model_cache, prefix_forward_cache = {}, {}

        def model_parts(replaced):
            replaced = tuple(replaced)
            if replaced not in model_cache:
                cfg = replace(
                    config,
                    attention_layer_indices=tuple(
                        index for index in range(config.num_layers)
                        if index not in replaced
                    ),
                )
                model = HybridForCausalLM(cfg)
                abstract = abstract_parameter_tree(model)
                validate_partition_specs(abstract, mesh)
                model_cache[replaced] = (
                    model, abstract, named_sharding_tree(abstract, mesh)
                )
            return model_cache[replaced]

        def assemble(replaced, endpoints):
            model, abstract, layout = model_parts(replaced)
            return model, compose_parameters(
                teacher_params, endpoints, replaced, abstract, layout
            )

        full_model, _, _ = model_parts(order)
        eval_fn = jax.jit(
            lambda p, value, target: full_model_eval_metrics(
                full_model.apply({"params": p}, value), target, value,
                temperature=2.0,
            )
        )

        def evaluate(endpoints, windows, targets):
            _, params = assemble(order, endpoints)
            records = []
            for index, window in enumerate(windows):
                metrics = eval_fn(
                    params, jax.device_put(window[None], batch_layout),
                    jnp.asarray(targets[index]),
                )
                jax.block_until_ready(metrics)
                records.append(_metric_record(metrics))
            del params
            summary = {
                name: bool(all(row[name] for row in records))
                if name == "finite"
                else float(np.mean([row[name] for row in records]))
                for name in records[0]
            }
            summary["window_nll"] = [row["student_nll"] for row in records]
            summary["window_prediction_kl"] = [
                row["prediction_kl"] for row in records
            ]
            return summary

        def branch_inputs(endpoints, layer):
            if layer == 0:
                return np.asarray(jax.device_get(
                    jnp.take(embedding_table, jnp.asarray(train_tokens), axis=0)
                ), np.float32)
            prefix = tuple(index for index in order if index < layer)
            model, params = assemble(prefix, endpoints)
            if layer not in prefix_forward_cache:
                prefix_forward_cache[layer] = jax.jit(
                    lambda p, value, model=model, layer=layer:
                    model.apply(
                        {"params": p}, value, return_hidden_states=True
                    )[1][layer - 1]
                )
            pieces = []
            for start in range(0, TRAIN_WINDOWS, 4):
                value = prefix_forward_cache[layer](
                    params,
                    jax.device_put(train_tokens[start:start + 4], batch_layout),
                )
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            del params
            return np.concatenate(pieces)

        def condition_targets(layer, inputs):
            pieces = []
            for start in range(0, len(inputs), 4):
                value = conditional_teacher(
                    teacher_params[f"layers_{layer}"],
                    jax.device_put(inputs[start:start + 4], batch_layout),
                )
                pieces.append(np.asarray(jax.device_get(value), np.float32))
            return np.concatenate(pieces)

        for seed in SEEDS:
            arm_order = ARMS if seed == SEEDS[0] else tuple(reversed(ARMS))
            for arm in arm_order:
                endpoints = dict(base[seed])
                row = {"complete": False, "evaluations": {}, "decisions": {}}
                start_position = 0
                for milestone in reversed(MILESTONES):
                    slot = f"seed-{seed}/{arm}/position-{milestone}"
                    restored = store.restore(
                        slot, checkpoint_contract(
                            contract, seed, arm, milestone, base_hashes[seed]
                        )
                    )
                    if restored is not None:
                        payload, _ = restored
                        endpoints = {int(key): value for key, value in payload["endpoints"].items()}
                        row = payload["row"]
                        start_position = milestone
                        print(f"EXP-081 restore=PASS slot={slot}", flush=True)
                        break
                result["branches"].setdefault(str(seed), {})[arm] = row
                if "0" not in row["evaluations"]:
                    row["evaluations"]["0"] = evaluate(
                        endpoints, eval_tokens, eval_teacher_logits
                    )
                    persist(upload=True)
                if DECISION_GROUP_SIZE == 2:
                    if secondary_calibration_tokens is None:
                        raise ValueError("paired selection requires secondary calibration")
                    if len(order) % 2 or any(step % 2 for step in MILESTONES):
                        raise ValueError("paired selection requires even order and milestones")
                    for group_start in range(0, len(order), 2):
                        position = group_start + 2
                        if position <= start_position:
                            continue
                        if time.monotonic() + 20 * 60 >= deadline:
                            result["status"] = "deadline_partial"
                            return result
                        layers = (order[group_start], order[group_start + 1])
                        stage = (
                            f"seed-{seed}-{arm}-positions-{group_start + 1}-{position}"
                            f"-layers-{layers[0]}-{layers[1]}"
                        )
                        print(f"{stage} START", flush=True)
                        group_endpoints = dict(endpoints)
                        currents, proposals, proposal_records = {}, {}, {}
                        for local_index, layer in enumerate(layers):
                            coordinate_position = group_start + local_index + 1
                            inputs = branch_inputs(group_endpoints, layer)
                            targets = condition_targets(layer, inputs)
                            current = jax.tree.map(jnp.asarray, group_endpoints[layer])
                            proposal = current
                            frozen = local_layer_params(
                                teacher_params[f"layers_{layer}"], proposal
                            )
                            opt_state = tx.init(proposal)
                            first = last = None
                            for zero_step in range(STEPS_PER_LAYER):
                                index = deterministic_batch_indices(
                                    zero_step, 1, TRAIN_WINDOWS,
                                    DATA_SEED + seed + coordinate_position,
                                )[0]
                                proposal, opt_state, metrics = train_step(
                                    proposal, opt_state, frozen,
                                    jnp.asarray(inputs[index:index + 1], jnp.bfloat16),
                                    jnp.asarray(targets[index:index + 1], jnp.bfloat16),
                                )
                                if zero_step in (0, STEPS_PER_LAYER - 1):
                                    jax.block_until_ready(metrics)
                                    record = _metric_record(metrics)
                                    first = record if first is None else first
                                    last = record
                                    if not record["grads_finite"]:
                                        raise FloatingPointError(
                                            f"non-finite pair proposal: {stage} layer={layer}"
                                        )
                            currents[layer] = current
                            proposals[layer] = proposal
                            proposal_records[str(layer)] = {
                                "first_loss": first["loss"],
                                "final_loss": last["loss"],
                                "final_grad_norm": last["grad_norm"],
                            }
                            del inputs, targets, opt_state, frozen

                        primary_scores, secondary_scores = {}, {}
                        primary_metrics, secondary_metrics = {}, {}
                        alpha_grid = tuple(ALPHAS[arm])
                        for first_alpha in alpha_grid:
                            for second_alpha in alpha_grid:
                                trial_endpoints = dict(group_endpoints)
                                for layer, alpha in zip(
                                    layers, (first_alpha, second_alpha), strict=True
                                ):
                                    trial_endpoints[layer] = blend_parameters(
                                        currents[layer], proposals[layer], alpha
                                    )
                                key = pair_key(first_alpha, second_alpha)
                                metrics = evaluate(
                                    trial_endpoints, calibration_tokens,
                                    calibration_teacher_logits,
                                )
                                secondary = evaluate(
                                    trial_endpoints, secondary_calibration_tokens,
                                    secondary_calibration_teacher_logits,
                                )
                                primary_scores[key] = metrics["prediction_kl"]
                                secondary_scores[key] = secondary["prediction_kl"]
                                primary_metrics[key] = metrics
                                secondary_metrics[key] = secondary

                        mode = GROUP_SELECTION_MODE[arm]
                        minimum_relative_gain = ARM_MIN_RELATIVE_GAIN[arm]
                        if mode == "pair_consensus":
                            selected, primary_gain, secondary_gain = (
                                choose_pair_consensus(
                                    primary_scores,
                                    secondary_scores,
                                    minimum_relative_gain,
                                )
                            )
                        elif mode == "greedy_pair_grid":
                            selected, primary_gain, secondary_gain = (
                                choose_greedy_from_pair_grid(
                                    primary_scores,
                                    secondary_scores,
                                    alpha_grid,
                                    minimum_relative_gain,
                                )
                            )
                        else:
                            raise ValueError(f"unsupported group selection mode: {mode}")

                        for layer, alpha in zip(layers, selected, strict=True):
                            if alpha > 0:
                                endpoints[layer] = jax.device_get(
                                    blend_parameters(
                                        currents[layer], proposals[layer], alpha
                                    )
                                )
                        rescue = is_joint_only_rescue(
                            selected,
                            primary_scores,
                            secondary_scores,
                            minimum_relative_gain,
                        )
                        row["decisions"][str(position)] = {
                            "layers": [int(layer) for layer in layers],
                            "selected_alphas": [float(alpha) for alpha in selected],
                            "relative_calibration_kl_gain": primary_gain,
                            "secondary_relative_calibration_kl_gain": secondary_gain,
                            "selection_mode": mode,
                            "joint_only_rescue": rescue,
                            "calibration": primary_metrics,
                            "secondary_calibration": secondary_metrics,
                            "proposal_metrics": proposal_records,
                        }
                        print(
                            f"{stage} selected_alphas={selected} "
                            f"primary_gain={primary_gain:.6g} "
                            f"secondary_gain={secondary_gain:.6g} "
                            f"joint_only_rescue={rescue}",
                            flush=True,
                        )
                        del (
                            group_endpoints, currents, proposals, proposal_records,
                            primary_scores, secondary_scores, primary_metrics,
                            secondary_metrics, trial_endpoints,
                        )
                        if position in MILESTONES:
                            row["evaluations"][str(position)] = evaluate(
                                endpoints, eval_tokens, eval_teacher_logits
                            )
                            slot = f"seed-{seed}/{arm}/position-{position}"
                            meta = store.save(
                                slot,
                                {
                                    "endpoints": {
                                        str(key): value
                                        for key, value in endpoints.items()
                                    },
                                    "row": row,
                                },
                                contract=checkpoint_contract(
                                    contract, seed, arm, position,
                                    base_hashes[seed],
                                ),
                                step=position,
                                metrics={
                                    "student_nll": row["evaluations"][str(position)][
                                        "student_nll"
                                    ],
                                    "accepted_coordinates": selected_coordinate_count(
                                        row["decisions"]
                                    ),
                                },
                            )
                            synced = any(
                                event.get("operation") == "upload"
                                and event.get("slot") == slot
                                and event.get("passed")
                                for event in reversed(store.events)
                            )
                            if not synced:
                                result.setdefault("durability_warnings", []).append({
                                    "slot": slot,
                                    "checkpoint_sha256": meta["checkpoint_sha256"],
                                    "reason": "hf_checkpoint_upload_failed_after_retries",
                                })
                                print(
                                    f"EXP-081 durability warning: {slot}; continuing",
                                    flush=True,
                                )
                            persist(upload=True)
                        else:
                            persist()
                        gc.collect()
                    row["complete"] = True
                    persist(upload=True)
                    continue
                if DECISION_GROUP_SIZE != 1:
                    raise ValueError(
                        f"unsupported decision group size: {DECISION_GROUP_SIZE}"
                    )
                for position, layer in enumerate(order, 1):
                    if position <= start_position:
                        continue
                    if time.monotonic() + 20 * 60 >= deadline:
                        result["status"] = "deadline_partial"
                        return result
                    stage = f"seed-{seed}-{arm}-position-{position}-layer-{layer}"
                    print(f"{stage} START", flush=True)
                    inputs = branch_inputs(endpoints, layer)
                    targets = condition_targets(layer, inputs)
                    current = jax.tree.map(jnp.asarray, endpoints[layer])
                    proposal = current
                    frozen = local_layer_params(
                        teacher_params[f"layers_{layer}"], proposal
                    )
                    opt_state = tx.init(proposal)
                    first = last = None
                    for zero_step in range(STEPS_PER_LAYER):
                        index = deterministic_batch_indices(
                            zero_step, 1, TRAIN_WINDOWS,
                            DATA_SEED + seed + position,
                        )[0]
                        proposal, opt_state, metrics = train_step(
                            proposal, opt_state, frozen,
                            jnp.asarray(inputs[index:index + 1], jnp.bfloat16),
                            jnp.asarray(targets[index:index + 1], jnp.bfloat16),
                        )
                        if zero_step in (0, STEPS_PER_LAYER - 1):
                            jax.block_until_ready(metrics)
                            record = _metric_record(metrics)
                            first = record if first is None else first
                            last = record
                            if not record["grads_finite"]:
                                raise FloatingPointError(f"non-finite proposal: {stage}")

                    scores, trial_metrics = {}, {}
                    secondary_scores, secondary_trial_metrics = {}, {}
                    trial_endpoints = None
                    for alpha in ALPHAS[arm]:
                        trial_endpoints = dict(endpoints)
                        trial_endpoints[layer] = blend_parameters(current, proposal, alpha)
                        metrics = evaluate(
                            trial_endpoints, calibration_tokens,
                            calibration_teacher_logits,
                        )
                        key = alpha_key(alpha)
                        scores[key] = metrics["prediction_kl"]
                        trial_metrics[key] = metrics
                        if secondary_calibration_tokens is not None:
                            secondary_metrics = evaluate(
                                trial_endpoints, secondary_calibration_tokens,
                                secondary_calibration_teacher_logits,
                            )
                            secondary_scores[key] = secondary_metrics["prediction_kl"]
                            secondary_trial_metrics[key] = secondary_metrics
                    if ARM_SELECTION_MODE[arm] == "dual_consensus":
                        selected_alpha, relative_gain, secondary_relative_gain = (
                            choose_consensus_alpha(scores, secondary_scores)
                        )
                        primary_window_fraction = secondary_window_fraction = None
                    elif ARM_SELECTION_MODE[arm] == "robust_consensus":
                        (
                            selected_alpha, relative_gain, secondary_relative_gain,
                            primary_window_fraction, secondary_window_fraction,
                        ) = choose_robust_consensus_alpha(
                            trial_metrics, secondary_trial_metrics
                        )
                    elif ARM_SELECTION_MODE[arm] == "primary":
                        selected_alpha, relative_gain = choose_trust_alpha(scores)
                        secondary_relative_gain = None
                        primary_window_fraction = secondary_window_fraction = None
                    else:
                        raise ValueError(
                            f"unsupported alpha selection mode: {ARM_SELECTION_MODE[arm]}"
                        )
                    if selected_alpha > 0:
                        endpoints[layer] = jax.device_get(
                            blend_parameters(current, proposal, selected_alpha)
                        )
                    row["decisions"][str(position)] = {
                        "layer": int(layer), "selected_alpha": selected_alpha,
                        "relative_calibration_kl_gain": relative_gain,
                        "calibration": trial_metrics,
                        "proposal_first_loss": first["loss"],
                        "proposal_final_loss": last["loss"],
                        "proposal_final_grad_norm": last["grad_norm"],
                    }
                    if secondary_calibration_tokens is not None:
                        row["decisions"][str(position)].update(
                            secondary_calibration=secondary_trial_metrics,
                            secondary_relative_calibration_kl_gain=secondary_relative_gain,
                            selection_mode=ARM_SELECTION_MODE[arm],
                        )
                        if primary_window_fraction is not None:
                            row["decisions"][str(position)].update(
                                primary_window_improvement_fraction=(
                                    primary_window_fraction
                                ),
                                secondary_window_improvement_fraction=(
                                    secondary_window_fraction
                                ),
                            )
                    print(
                        f"{stage} selected_alpha={selected_alpha:g} "
                        f"relative_kl_gain={relative_gain:.6g}", flush=True,
                    )
                    del inputs, targets, current, proposal, opt_state, frozen, trial_endpoints
                    if position in MILESTONES:
                        row["evaluations"][str(position)] = evaluate(
                            endpoints, eval_tokens, eval_teacher_logits
                        )
                        slot = f"seed-{seed}/{arm}/position-{position}"
                        meta = store.save(
                            slot,
                            {"endpoints": {str(k): v for k, v in endpoints.items()},
                             "row": row},
                            contract=checkpoint_contract(
                                contract, seed, arm, position, base_hashes[seed]
                            ),
                            step=position,
                            metrics={
                                "student_nll": row["evaluations"][str(position)]["student_nll"],
                                "accepted_coordinates": selected_coordinate_count(
                                    row["decisions"]
                                ),
                            },
                        )
                        synced = any(
                            event.get("operation") == "upload"
                            and event.get("slot") == slot and event.get("passed")
                            for event in reversed(store.events)
                        )
                        if not synced:
                            result.setdefault("durability_warnings", []).append({
                                "slot": slot,
                                "checkpoint_sha256": meta["checkpoint_sha256"],
                                "reason": "hf_checkpoint_upload_failed_after_retries",
                            })
                            print(
                                f"EXP-081 durability warning: {slot}; continuing",
                                flush=True,
                            )
                        persist(upload=True)
                    else:
                        persist()
                    gc.collect()
                row["complete"] = True
                persist(upload=True)
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
            f"gate={result['aggregate']['scientific_gate_passed']}",
        )


if __name__ == "__main__":
    main()
