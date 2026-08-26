from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.offline_mamba_distill import _slice
from scripts.qwen_activation_cache import (
    _create_data_parallel_decoder_runner,
    _create_decoder_runner,
    _safe_prune_shards,
)
from scripts.qwen_decoder_aware_confirmation import (
    main as train_multiseed,
    parse_seeds,
)
from scripts.qwen_decoder_aware_distill import _teacher_decoder_outputs
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror
from scripts.qwen_streamed_end_to_end_shock import (
    _read_json,
    _run_lm_metrics,
    _run_replacement_windows,
)
from extent.config import Mamba3Config
from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_replacement_runner,
    create_batched_teacher_tail_runner,
    qwen3_decoder_tail_params,
)
from extent.hardware import recommended_compute_dtype
from extent.endpoint_checkpoint import (
    endpoint_checkpoint_supports_arms,
    restore_endpoint_checkpoint,
    save_endpoint_checkpoint,
)
from extent.experiment_stage import update_stage_manifest
from extent.layers.common import RMSNorm
from extent.layers.mamba3 import Mamba3MIMO
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
)
from extent.qwen3_teacher import Qwen3DecoderLayer, Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.streamed_lm_eval import (
    aggregate_multiseed_end_to_end,
    bootstrap_excess_nll_recovery,
    create_lm_metrics_runner,
    full_model_shard_last_use,
    hidden_relative_l2,
)
from extent.teacher_activation_cache import (
    file_sha256,
    load_activation_cache,
    run_host_data_parallel,
    run_host_microbatches,
    validate_external_evaluation_cache,
)
from extent.weight_mapping import QwenCheckpointReader


OBJECTIVE_COMPARISON_PROTOCOLS = frozenset(
    {
        "exp045-depth-objective",
        "exp046-depth-objective",
        "exp047-context-transfer",
        "exp048-long-horizon",
        "exp049-extended-horizon",
        "exp050-depth-scaling-atlas",
        "exp051-progressive-composition",
        "exp052-boundary-scaling",
        "exp053-onset-localization",
    }
)
EXTERNAL_EVALUATION_PROTOCOLS = OBJECTIVE_COMPARISON_PROTOCOLS | {
    "exp042-fresh",
    "exp043-validation",
}
PROTOCOL_METHODS = {
    "exp041": "three_seed_streamed_end_to_end_Qwen3_layer0_Mamba_confirmation",
    "exp042-fresh": "locked_fresh_text_three_seed_streamed_end_to_end_confirmation",
    "exp043-validation": "cross_split_end_to_end_NLL_adjudication",
    "exp045-depth-objective": "depth_aware_counterfactual_objective_comparison",
    "exp046-depth-objective": "depth_generalization_counterfactual_objective_comparison",
    "exp047-context-transfer": "zero_shot_context_transfer_of_recovered_Mamba_objectives",
    "exp048-long-horizon": "long_horizon_transplant_scaling_comparison",
    "exp049-extended-horizon": "extended_horizon_transplant_scaling_comparison",
    "exp050-depth-scaling-atlas": "depth_scaling_atlas_objective_comparison",
    "exp051-progressive-composition": "progressive_composition_standalone_objective_comparison",
    "exp052-boundary-scaling": "boundary_scaling_standalone_objective_comparison",
    "exp053-onset-localization": "composition_onset_standalone_objective_comparison",
}
PROTOCOL_NOTES = {
    "exp041": "The first configured seed must reproduce the archived EXP-040 NLL values within the frozen tolerance.",
    "exp042-fresh": "The external evaluation cache is disjoint from the training cache and locked to tokens [65536, 69632).",
    "exp043-validation": "The primary endpoint uses the pinned WikiText-2 validation split; local decoder L2 is diagnostic only.",
    "exp045-depth-objective": "The primary endpoint compares counterfactual contribution matching against both controls on pinned validation windows.",
    "exp046-depth-objective": "The primary endpoint extends the frozen objective comparison to additional decoder depths.",
    "exp047-context-transfer": "Endpoints trained only at sequence length 32 are evaluated without updates on the frozen 8,192-token validation prefix.",
    "exp048-long-horizon": "The short and long arms use the same frozen protocol except for one-pass unique-token budget and its pre-registered Lion schedule.",
    "exp049-extended-horizon": "The 2,048-step and 8,192-step arms test whether the depth-dependent scaling sign persists at a larger one-pass recovery budget.",
    "exp050-depth-scaling-atlas": "Three new decoder depths test whether long-budget benefit decays systematically with layer index.",
    "exp051-progressive-composition": "Standalone JOINT endpoints provide paired additive controls for the frozen 2/4/8-layer composition test.",
    "exp052-boundary-scaling": "Standalone JOINT endpoints provide paired controls for 8-to-16-layer scaling and layer-0 boundary ablations.",
    "exp053-onset-localization": "Standalone JOINT endpoints support multiplicity-corrected onset, conditional-addition, and matched-layout tests.",
}
PROTOCOL_LABELS = {
    "exp041": "MULTISEED-STREAMED-END-TO-END",
    "exp042-fresh": "LOCKED-FRESH-TEXT",
    "exp043-validation": "CROSS-SPLIT-NLL-ADJUDICATION",
    "exp045-depth-objective": "DEPTH-OBJECTIVE-COMPARISON",
    "exp046-depth-objective": "DEPTH-GENERALIZATION-COMPARISON",
    "exp047-context-transfer": "CONTEXT-TRANSFER-COMPARISON",
    "exp048-long-horizon": "LONG-HORIZON-OBJECTIVE-COMPARISON",
    "exp049-extended-horizon": "EXTENDED-HORIZON-OBJECTIVE-COMPARISON",
    "exp050-depth-scaling-atlas": "DEPTH-SCALING-ATLAS-COMPARISON",
    "exp051-progressive-composition": "PROGRESSIVE-COMPOSITION-STANDALONE",
    "exp052-boundary-scaling": "BOUNDARY-SCALING-STANDALONE",
    "exp053-onset-localization": "COMPOSITION-ONSET-STANDALONE",
}


def branch_names(
    seeds: tuple[int, ...],
    total_steps: int,
    *,
    include_contribution: bool = False,
) -> tuple[str, ...]:
    names = ["ORIGINAL-CACHED-QWEN"]
    for seed in seeds:
        if include_contribution:
            names.extend(
                (
                    f"SEED-{seed}-MIXER-ONLY-STEP{total_steps}",
                    f"SEED-{seed}-JOINT-STEP{total_steps}",
                    f"SEED-{seed}-CONTRIBUTION-STEP{total_steps}",
                )
            )
        else:
            names.extend(
                (
                    f"SEED-{seed}-CALIBRATED-STEP0",
                    f"SEED-{seed}-MIXER-ONLY-STEP{total_steps}",
                    f"SEED-{seed}-JOINT-STEP{total_steps}",
                )
            )
    return tuple(names)


def collect_per_seed_lm_metrics(
    lm_metrics: dict[str, dict],
    seeds: tuple[int, ...],
    total_steps: int,
    *,
    objective_comparison: bool,
) -> dict[str, dict[str, dict]]:
    """Collect only the branches that the selected protocol evaluated."""
    per_seed_metrics: dict[str, dict[str, dict]] = {}
    for seed in seeds:
        metrics = {
            "mixer_only": lm_metrics[
                f"SEED-{seed}-MIXER-ONLY-STEP{total_steps}"
            ],
            "joint": lm_metrics[f"SEED-{seed}-JOINT-STEP{total_steps}"],
        }
        if objective_comparison:
            metrics["contribution"] = lm_metrics[
                f"SEED-{seed}-CONTRIBUTION-STEP{total_steps}"
            ]
        else:
            metrics["calibrated"] = lm_metrics[
                f"SEED-{seed}-CALIBRATED-STEP0"
            ]
        per_seed_metrics[str(seed)] = metrics
    return per_seed_metrics


def protocol_collects_window_nll(protocol: str) -> bool:
    """Return whether paired bootstrap inputs must be retained per window."""
    return protocol == "exp043-validation" or protocol in OBJECTIVE_COMPARISON_PROTOCOLS


def aggregate_depth_objectives(
    original_nll: float,
    seed_metrics: dict[str, dict[str, dict]],
) -> dict:
    """Compare objective endpoints without unstable ratios around zero shock."""
    records = []
    for seed, metrics in seed_metrics.items():
        mixer = float(metrics["mixer_only"]["mean_nll"])
        joint = float(metrics["joint"]["mean_nll"])
        contribution = float(metrics["contribution"]["mean_nll"])
        records.append(
            {
                "seed": int(seed),
                "mixer_only_excess_nll": mixer - original_nll,
                "joint_excess_nll": joint - original_nll,
                "contribution_excess_nll": contribution - original_nll,
                "contribution_minus_mixer_nll": contribution - mixer,
                "contribution_minus_joint_nll": contribution - joint,
                "contribution_beats_mixer": contribution < mixer,
                "contribution_beats_joint": contribution < joint,
            }
        )
    delta_mixer = np.asarray(
        [record["contribution_minus_mixer_nll"] for record in records]
    )
    delta_joint = np.asarray(
        [record["contribution_minus_joint_nll"] for record in records]
    )
    return {
        "per_seed": records,
        "mean_contribution_minus_mixer_nll": float(np.mean(delta_mixer)),
        "mean_contribution_minus_joint_nll": float(np.mean(delta_joint)),
        "contribution_wins_vs_mixer": int(np.sum(delta_mixer < 0)),
        "contribution_wins_vs_joint": int(np.sum(delta_joint < 0)),
        "all_finite": bool(
            np.all(np.isfinite(delta_mixer)) and np.all(np.isfinite(delta_joint))
        ),
    }


def bootstrap_depth_objective_deltas(
    seed_window_nll: dict[str, dict[str, list[float]]],
    *,
    samples: int,
    seed: int,
) -> dict:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    arrays = {
        key: {name: np.asarray(values, np.float64) for name, values in arms.items()}
        for key, arms in seed_window_nll.items()
    }
    required_arms = {"mixer_only", "joint", "contribution"}
    if not arrays or any(set(values) != required_arms for values in arrays.values()):
        raise ValueError("each seed must provide all three objective arms")
    all_lengths = {
        len(array)
        for values in arrays.values()
        for array in values.values()
    }
    window_counts = {len(values["contribution"]) for values in arrays.values()}
    if (
        len(window_counts) != 1
        or len(all_lengths) != 1
        or not window_counts
        or min(window_counts) < 2
    ):
        raise ValueError("all seeds must share at least two paired windows")
    windows = window_counts.pop()
    rng = np.random.default_rng(seed)
    mixer_deltas = np.empty(samples, np.float64)
    joint_deltas = np.empty(samples, np.float64)
    for index in range(samples):
        selected = rng.integers(0, windows, size=windows)
        mixer_deltas[index] = np.mean(
            [
                np.mean(values["contribution"][selected] - values["mixer_only"][selected])
                for values in arrays.values()
            ]
        )
        joint_deltas[index] = np.mean(
            [
                np.mean(values["contribution"][selected] - values["joint"][selected])
                for values in arrays.values()
            ]
        )
    return {
        "samples": samples,
        "seed": seed,
        "contribution_minus_mixer_95ci": [
            float(value) for value in np.percentile(mixer_deltas, [2.5, 97.5])
        ],
        "contribution_minus_joint_95ci": [
            float(value) for value in np.percentile(joint_deltas, [2.5, 97.5])
        ],
    }


def branch_divergence(
    hidden: np.ndarray,
    names: tuple[str, ...],
    windows: int,
) -> dict[str, float]:
    values = hidden.reshape(len(names), windows, *hidden.shape[1:])
    reference = values[0]
    return {
        name: hidden_relative_l2(reference, values[index])
        for index, name in enumerate(names)
    }


def reference_reproduction(
    reference: dict,
    lm_metrics: dict[str, dict],
    *,
    seed: int,
    total_steps: int,
    tolerance: float,
) -> dict:
    mappings = {
        "ORIGINAL-CACHED-QWEN": "ORIGINAL-CACHED-QWEN",
        "CALIBRATED-STEP0": f"SEED-{seed}-CALIBRATED-STEP0",
        "MIXER-ONLY-STEP1024": f"SEED-{seed}-MIXER-ONLY-STEP{total_steps}",
        "JOINT-STEP1024": f"SEED-{seed}-JOINT-STEP{total_steps}",
    }
    differences = {}
    for reference_name, current_name in mappings.items():
        expected = float(reference["lm_metrics"][reference_name]["mean_nll"])
        actual = float(lm_metrics[current_name]["mean_nll"])
        differences[reference_name] = {
            "reference_mean_nll": expected,
            "current_mean_nll": actual,
            "absolute_difference": abs(actual - expected),
        }
    passed = all(
        record["absolute_difference"] <= tolerance
        for record in differences.values()
    )
    return {
        "seed": seed,
        "absolute_nll_tolerance": tolerance,
        "branches": differences,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Confirm the streamed Qwen3 layer-0 Mamba NLL advantage across three seeds."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--evaluation-cache-manifest")
    parser.add_argument("--evaluation-cache-dir")
    parser.add_argument(
        "--qwen-cache-dir", default="/kaggle/working/qwen3-exp041-weights"
    )
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--training-checkpoints", default="0,512,1024")
    parser.add_argument("--evaluation-batch-windows", type=int, default=1)
    parser.add_argument("--per-device-windows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--contribution-mixer-weight", type=float, default=1.0)
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--prune-consumed-shards", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--reference-exp040-json",
        default="results/EXP-040-qwen3-streamed-end-to-end-layer0.json",
    )
    parser.add_argument("--reference-nll-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOL_METHODS),
        default="exp041",
    )
    parser.add_argument("--required-recovery-fraction", type=float)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--endpoint-checkpoint-dir")
    parser.add_argument("--resume-endpoints", action="store_true")
    parser.add_argument("--stage-manifest")
    parser.add_argument("--training-result-json")
    parser.add_argument("--endpoint-source-evaluation-manifest")
    parser.add_argument("--endpoint-source-experiment")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if args.protocol == "exp048-long-horizon":
        if args.total_steps not in {1024, 4096}:
            raise ValueError("EXP-048 requires 1,024 or 4,096 training steps")
    elif args.protocol in {
        "exp049-extended-horizon",
        "exp050-depth-scaling-atlas",
        "exp051-progressive-composition",
        "exp052-boundary-scaling",
        "exp053-onset-localization",
    }:
        if args.total_steps not in {2048, 8192}:
            raise ValueError(
                "EXP-049/050/051/052/053 requires 2,048 or 8,192 training steps"
            )
    elif args.total_steps != 1024:
        raise ValueError("the selected protocol requires exactly 1,024 steps")
    if min(
        args.batch_windows,
        args.evaluation_batch_windows,
        args.per_device_windows,
    ) < 1:
        raise ValueError("batch settings must be positive")
    if args.reference_nll_tolerance < 0:
        raise ValueError("reference NLL tolerance must be non-negative")
    if not 0 <= args.target_layer < 40:
        raise ValueError("target-layer must be in [0, 40)")
    if (
        args.protocol not in OBJECTIVE_COMPARISON_PROTOCOLS
        and args.target_layer != 0
    ):
        raise ValueError("legacy streamed protocols are frozen to layer zero")

    jax.config.update("jax_default_matmul_precision", "high")
    context_transfer = args.protocol == "exp047-context-transfer"
    if context_transfer:
        manifest = json.loads(
            Path(args.activation_cache_manifest).read_text(encoding="utf-8")
        )
        arrays, paths = {}, {}
    else:
        manifest, arrays, paths = load_activation_cache(
            args.activation_cache_manifest,
            artifact_dir=args.activation_cache_dir,
            verify_hashes=not args.skip_hash_verification,
        )
    source_name = f"{QWEN3_14B.repo_id}@{QWEN3_14B.revision}"
    if (
        manifest.get("source") != source_name
        or int(manifest["target_layer"]) != args.target_layer
    ):
        raise ValueError("activation cache source or target layer does not match")
    evaluation_manifest = manifest
    evaluation_arrays = arrays
    evaluation_paths = paths
    evaluation_slice = _slice(manifest["window_layout"], "evaluation")
    if args.protocol in EXTERNAL_EVALUATION_PROTOCOLS:
        if not args.evaluation_cache_manifest:
            raise ValueError("the selected protocol requires an external evaluation cache")
        evaluation_manifest, evaluation_arrays, evaluation_paths = (
            load_activation_cache(
                args.evaluation_cache_manifest,
                artifact_dir=args.evaluation_cache_dir,
                verify_hashes=not args.skip_hash_verification,
            )
        )
        if args.protocol == "exp042-fresh":
            evaluation_slice = validate_external_evaluation_cache(
                manifest,
                evaluation_manifest,
                required_token_offset=65_536,
                required_evaluation_windows=128,
                required_dataset_split="train",
            )
        elif args.protocol == "exp047-context-transfer":
            evaluation_slice = validate_external_evaluation_cache(
                manifest,
                evaluation_manifest,
                required_token_offset=0,
                required_dataset_split="validation",
                allow_cross_split=True,
                allow_sequence_length_mismatch=True,
            )
            if evaluation_manifest.get("token_range") != [0, 8192]:
                raise ValueError("EXP-047 requires the frozen 8,192-token range")
        else:
            evaluation_slice = validate_external_evaluation_cache(
                manifest,
                evaluation_manifest,
                required_token_offset=0,
                required_evaluation_windows=256,
                required_dataset_split="validation",
                allow_cross_split=True,
            )
    evaluation_windows = evaluation_slice.stop - evaluation_slice.start
    evaluation_sequence_length = int(evaluation_manifest["sequence_length"])
    required_recovery = args.required_recovery_fraction
    if required_recovery is None:
        required_recovery = {
            "exp041": 0.10,
            "exp042-fresh": 0.25,
            "exp043-validation": 0.20,
            "exp045-depth-objective": 0.0,
            "exp046-depth-objective": 0.0,
            "exp047-context-transfer": 0.0,
            "exp048-long-horizon": 0.0,
            "exp049-extended-horizon": 0.0,
            "exp050-depth-scaling-atlas": 0.0,
            "exp051-progressive-composition": 0.0,
            "exp052-boundary-scaling": 0.0,
            "exp053-onset-localization": 0.0,
        }[args.protocol]
    if not 0.0 <= required_recovery <= 1.0:
        raise ValueError("required recovery fraction must be in [0, 1]")
    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    compute_dtype = jnp.dtype(dtype_decision.dtype)
    devices = list(jax.devices())
    if args.data_parallel and len(devices) < 2:
        raise ValueError("--data-parallel requires at least two visible devices")
    parallel_devices = devices if args.data_parallel else None
    objective_comparison = args.protocol in OBJECTIVE_COMPARISON_PROTOCOLS
    stage_experiment = args.protocol if objective_comparison else "legacy"
    names = branch_names(
        seeds,
        args.total_steps,
        include_contribution=objective_comparison,
    )

    experiment_id = {
        "exp041": "exp041",
        "exp042-fresh": "exp042",
        "exp043-validation": "exp043",
        "exp045-depth-objective": f"exp045-layer{args.target_layer}",
        "exp046-depth-objective": f"exp046-layer{args.target_layer}",
        "exp047-context-transfer": (
            args.endpoint_source_experiment
            or f"exp045-layer{args.target_layer}"
        ),
        "exp048-long-horizon": (
            f"exp048-step{args.total_steps}-layer{args.target_layer}"
        ),
        "exp049-extended-horizon": (
            f"exp049-step{args.total_steps}-layer{args.target_layer}"
        ),
        "exp050-depth-scaling-atlas": (
            f"exp050-step{args.total_steps}-layer{args.target_layer}"
        ),
        "exp051-progressive-composition": (
            f"exp051-step{args.total_steps}-layer{args.target_layer}"
        ),
        "exp052-boundary-scaling": (
            f"exp052-step{args.total_steps}-layer{args.target_layer}"
        ),
        "exp053-onset-localization": (
            f"exp053-step{args.total_steps}-layer{args.target_layer}"
        ),
    }[args.protocol]
    if context_transfer:
        if not all(
            (
                args.training_result_json,
                args.endpoint_checkpoint_dir,
                args.endpoint_source_evaluation_manifest,
                args.endpoint_source_experiment,
            )
        ):
            raise ValueError(
                "EXP-047 requires training result, endpoint checkpoint, "
                "source evaluation manifest, and source experiment"
            )
        training_json = Path(args.training_result_json)
    else:
        training_json = (
            Path(args.output_dir) / f"{experiment_id}-training-confirmation.json"
        )
    training_args = [
        "--activation-cache-manifest", args.activation_cache_manifest,
        "--qwen-cache-dir", args.qwen_cache_dir,
        "--seeds", args.seeds,
        "--data-seed", str(args.data_seed),
        "--total-steps", str(args.total_steps),
        "--checkpoints", args.training_checkpoints,
        "--batch-windows", str(args.batch_windows),
        "--evaluation-batch-windows", str(args.evaluation_batch_windows),
        "--learning-rate", str(args.learning_rate),
        "--readout-ridge", str(args.readout_ridge),
        "--decoder-loss-weight", str(args.decoder_loss_weight),
        "--contribution-mixer-weight", str(args.contribution_mixer_weight),
        "--compute-dtype", args.compute_dtype,
        "--result-json", str(training_json),
        "--output-dir", args.output_dir,
    ]
    if args.activation_cache_dir:
        training_args.extend(
            ["--activation-cache-dir", args.activation_cache_dir]
        )
    if args.evaluation_cache_manifest:
        training_args.extend(
            ["--evaluation-cache-manifest", args.evaluation_cache_manifest]
        )
    if args.evaluation_cache_dir:
        training_args.extend(
            ["--evaluation-cache-dir", args.evaluation_cache_dir]
        )
    if args.protocol == "exp043-validation":
        training_args.append("--allow-cross-split-evaluation")
    if objective_comparison:
        training_args.extend(
            ["--allow-cross-split-evaluation", "--include-contribution-arm"]
        )
    if args.skip_hash_verification:
        training_args.append("--skip-hash-verification")
    checkpoint_dir = (
        Path(args.endpoint_checkpoint_dir)
        if args.endpoint_checkpoint_dir
        else None
    )
    compatibility = {
        "source": source_name,
        "experiment": experiment_id,
        "target_layer": args.target_layer,
        "activation_manifest_sha256": file_sha256(
            Path(args.activation_cache_manifest)
        ),
        "evaluation_manifest_sha256": file_sha256(
            Path(
                args.endpoint_source_evaluation_manifest
                if context_transfer
                else args.evaluation_cache_manifest
            )
        ) if args.evaluation_cache_manifest else None,
        "seeds": list(seeds),
        "data_seed": args.data_seed,
        "total_steps": args.total_steps,
        "training_checkpoints": args.training_checkpoints,
        "learning_rate": args.learning_rate,
        "readout_ridge": args.readout_ridge,
        "decoder_loss_weight": args.decoder_loss_weight,
        "contribution_mixer_weight": args.contribution_mixer_weight,
        "compute_dtype": dtype_decision.dtype,
    }
    restored = False
    if (
        objective_comparison
        and (args.resume_endpoints or context_transfer)
        and checkpoint_dir is not None
        and training_json.exists()
        and (checkpoint_dir / "checkpoint.json").exists()
        and endpoint_checkpoint_supports_arms(
            checkpoint_dir,
            {
                "MIXER-ONLY",
                "JOINT-MIXER-DECODER",
                "COUNTERFACTUAL-CONTRIBUTION",
            },
        )
    ):
        training_result = json.loads(training_json.read_text(encoding="utf-8"))
        if not training_result.get("passed"):
            raise ValueError("saved objective training result is not numerically valid")
        endpoint_by_seed, checkpoint_metadata = restore_endpoint_checkpoint(
            checkpoint_dir,
            expected_compatibility=compatibility,
        )
        initial_by_seed = {}
        restored = True
        print(
            "training_endpoints=RESUME-PASS "
            f"sha256={checkpoint_metadata['checkpoint_sha256']}"
        )
    else:
        if context_transfer:
            raise ValueError("EXP-047 cannot retrain missing source endpoints")
        training_result, initial_by_seed, endpoint_by_seed = train_multiseed(
            training_args, return_endpoint_params=True
        )
        checkpoint_metadata = None
        if objective_comparison and checkpoint_dir is not None:
            checkpoint_metadata = save_endpoint_checkpoint(
                checkpoint_dir,
                endpoint_by_seed,
                compatibility=compatibility,
            )
            print(
                "training_endpoints=CHECKPOINT-PASS "
                f"sha256={checkpoint_metadata['checkpoint_sha256']}"
            )
    if objective_comparison and args.stage_manifest:
        stage_suffix = (
            f"layer{args.target_layer}-seq{evaluation_sequence_length}"
            if context_transfer
            else f"layer{args.target_layer}"
        )
        update_stage_manifest(
            args.stage_manifest,
            experiment=stage_experiment,
            stage=f"endpoints-{stage_suffix}",
            status="completed",
            details={
                "restored": restored,
                "checkpoint_sha256": checkpoint_metadata[
                    "checkpoint_sha256"
                ] if checkpoint_metadata else None,
                "checkpoint_bytes": checkpoint_metadata[
                    "checkpoint_bytes"
                ] if checkpoint_metadata else None,
            },
        )
        update_stage_manifest(
            args.stage_manifest,
            experiment=stage_experiment,
            stage=f"streamed-evaluation-{stage_suffix}",
            status="running",
            details={"restart_boundary": f"decoder-layer-{args.target_layer}"},
        )

    config_payload = _read_json(QWEN3_14B.resolve_url("config.json"))
    index_payload = _read_json(
        QWEN3_14B.resolve_url("model.safetensors.index.json")
    )
    validate_source_metadata(config_payload, index_payload, QWEN3_14B)
    source = Qwen3TeacherConfig(
        param_dtype="float32",
        compute_dtype=dtype_decision.dtype,
        remat_policy="none",
    )
    mamba = Mamba3MIMO(
        source.hidden_size,
        Mamba3Config(),
        dtype=compute_dtype,
        param_dtype=compute_dtype,
    )
    model_dir, _ = ensure_layer_checkpoint(
        args.qwen_cache_dir,
        index_payload["weight_map"],
        source,
        args.target_layer,
        repo_id=QWEN3_14B.repo_id,
        revision=QWEN3_14B.revision,
    )
    reader = QwenCheckpointReader(model_dir)
    tail_params = qwen3_decoder_tail_params(reader, args.target_layer)
    tail = Qwen3DecoderTail(
        source.hidden_size,
        source.intermediate_size,
        source.rms_norm_eps,
        compute_dtype,
        jnp.float32,
    )
    teacher_tail_runner = create_batched_teacher_tail_runner(tail)
    replacement_runner = create_batched_replacement_runner(mamba, tail)
    residual_eval = jnp.asarray(
        evaluation_arrays["residual_input"][evaluation_slice], dtype=compute_dtype
    )
    normalized_eval = jnp.asarray(
        evaluation_arrays["normalized_input"][evaluation_slice], dtype=compute_dtype
    )
    mixer_eval = jnp.asarray(
        evaluation_arrays["attention_target"][evaluation_slice], dtype=compute_dtype
    )
    layer0_outputs = [
        _teacher_decoder_outputs(
            teacher_tail_runner,
            tail_params,
            residual_eval,
            mixer_eval,
            args.evaluation_batch_windows,
        )
    ]
    for seed in seeds:
        seed_key = str(seed)
        if not objective_comparison:
            layer0_outputs.append(
                _run_replacement_windows(
                    replacement_runner,
                    initial_by_seed[seed_key],
                    tail_params,
                    residual_eval,
                    normalized_eval,
                    args.evaluation_batch_windows,
                )
            )
        layer0_outputs.extend(
            (
                _run_replacement_windows(
                    replacement_runner,
                    endpoint_by_seed[seed_key]["MIXER-ONLY"],
                    tail_params,
                    residual_eval,
                    normalized_eval,
                    args.evaluation_batch_windows,
                ),
                _run_replacement_windows(
                    replacement_runner,
                    endpoint_by_seed[seed_key]["JOINT-MIXER-DECODER"],
                    tail_params,
                    residual_eval,
                    normalized_eval,
                    args.evaluation_batch_windows,
                ),
            )
        )
        if objective_comparison:
            layer0_outputs.append(
                _run_replacement_windows(
                    replacement_runner,
                    endpoint_by_seed[seed_key]["COUNTERFACTUAL-CONTRIBUTION"],
                    tail_params,
                    residual_eval,
                    normalized_eval,
                    args.evaluation_batch_windows,
                )
            )
    hidden = np.concatenate(layer0_outputs, axis=0)
    divergence = {
        str(args.target_layer): branch_divergence(
            hidden, names, evaluation_windows
        )
    }
    del (
        layer0_outputs,
        initial_by_seed,
        endpoint_by_seed,
        tail_params,
        residual_eval,
        normalized_eval,
        mixer_eval,
        mamba,
        tail,
        teacher_tail_runner,
        replacement_runner,
    )
    jax.clear_caches()
    gc.collect()

    last_use = full_model_shard_last_use(
        index_payload["weight_map"], source.num_layers
    )
    removed_shards = []
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, args.target_layer)
        )
    positions = jnp.arange(
        evaluation_sequence_length, dtype=jnp.int32
    )[None]
    attention_mask = jnp.ones(
        (1, evaluation_sequence_length), dtype=jnp.bool_
    )
    decoder = Qwen3DecoderLayer(source)
    layer_runner = (
        _create_data_parallel_decoder_runner(
            decoder, positions, attention_mask, devices
        )
        if args.data_parallel
        else _create_decoder_runner(decoder, positions, attention_mask)
    )
    divergence_layers = {9, 19, 29, 39}
    for layer_index in range(args.target_layer + 1, source.num_layers):
        print(f"streaming_decoder_layer={layer_index}/{source.num_layers - 1}")
        ensure_layer_checkpoint(
            model_dir,
            index_payload["weight_map"],
            source,
            layer_index,
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
        )
        reader = QwenCheckpointReader(model_dir)
        layer_arrays = load_layer_arrays(reader, source, layer_index)
        layer_params = jax_layer_params(layer_arrays, source, layer_index)
        hidden = (
            run_host_data_parallel(
                layer_runner,
                layer_params,
                hidden,
                args.per_device_windows,
                len(devices),
                input_dtype=compute_dtype,
            )
            if args.data_parallel
            else run_host_microbatches(
                layer_runner,
                layer_params,
                hidden,
                args.evaluation_batch_windows,
                input_dtype=compute_dtype,
            )
        )
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError(
                f"non-finite streamed residual after layer {layer_index}"
            )
        if layer_index in divergence_layers:
            divergence[str(layer_index)] = branch_divergence(
                hidden, names, evaluation_windows
            )
        del layer_params, layer_arrays, reader
        gc.collect()
        if args.prune_consumed_shards:
            removed_shards.extend(
                _safe_prune_shards(model_dir, last_use, layer_index)
            )

    from huggingface_hub import hf_hub_download

    for tensor_name in ("model.norm.weight", "lm_head.weight"):
        hf_hub_download(
            repo_id=QWEN3_14B.repo_id,
            revision=QWEN3_14B.revision,
            filename=index_payload["weight_map"][tensor_name],
            local_dir=model_dir,
        )
    reader = QwenCheckpointReader(model_dir)
    norm_params = {
        "scale": jnp.asarray(
            reader.read("model.norm.weight"), dtype=jnp.float32
        )
    }
    lm_head_kernel = jnp.asarray(
        reader.read("lm_head.weight").T, dtype=compute_dtype
    )
    norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    collect_window_nll = protocol_collects_window_nll(args.protocol)
    lm_runner = create_lm_metrics_runner(
        norm,
        data_parallel_devices=parallel_devices,
        per_window=collect_window_nll,
    )
    token_ids = np.asarray(
        evaluation_arrays["token_ids"][evaluation_slice], dtype=np.int32
    )
    branch_values = hidden.reshape(
        len(names), evaluation_windows, *hidden.shape[1:]
    )
    lm_metrics = {}
    for branch_index, name in enumerate(names):
        print(f"evaluating_lm_head={name}")
        lm_metrics[name] = _run_lm_metrics(
            lm_runner,
            norm_params,
            lm_head_kernel,
            branch_values[branch_index],
            token_ids,
            devices=parallel_devices,
            per_device_windows=args.per_device_windows,
            compute_dtype=compute_dtype,
            return_window_nll=collect_window_nll,
        )
    raw_metrics_path = Path(args.result_json).with_name(
        f"{Path(args.result_json).stem}-lm-metrics.json"
    )
    raw_metrics_payload = {
        "source": source_name,
        "protocol": args.protocol,
        "target_layer": args.target_layer,
        "sequence_length": evaluation_sequence_length,
        "evaluation_windows": evaluation_windows,
        "branches": list(names),
        "lm_metrics": lm_metrics,
        "complete": True,
    }
    raw_metrics_mirror = _write_json_with_output_mirror(
        raw_metrics_path,
        raw_metrics_payload,
        args.output_dir,
    )
    print(f"raw_lm_metrics_json={raw_metrics_path.resolve()}")
    if raw_metrics_mirror:
        print(f"raw_lm_metrics_output_json={raw_metrics_mirror.resolve()}")
    if args.prune_consumed_shards:
        removed_shards.extend(
            _safe_prune_shards(model_dir, last_use, source.num_layers)
        )

    reference_path = Path(args.reference_exp040_json)
    if args.protocol == "exp041":
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("source") != source_name or not reference.get("passed"):
            raise ValueError("reference EXP-040 artifact is incompatible or invalid")
        reproduction = reference_reproduction(
            reference,
            lm_metrics,
            seed=seeds[0],
            total_steps=args.total_steps,
            tolerance=args.reference_nll_tolerance,
        )
    else:
        reproduction = {
            "required": False,
            "reason": (
                "EXP-042 evaluates a disjoint locked token range"
                if args.protocol == "exp042-fresh"
                else "This protocol evaluates the pinned validation split"
            ),
            "passed": True,
        }
    per_seed_metrics = collect_per_seed_lm_metrics(
        lm_metrics,
        seeds,
        args.total_steps,
        objective_comparison=objective_comparison,
    )
    if objective_comparison:
        aggregate = aggregate_depth_objectives(
            lm_metrics["ORIGINAL-CACHED-QWEN"]["mean_nll"],
            per_seed_metrics,
        )
    else:
        aggregate = aggregate_multiseed_end_to_end(
            original_nll=lm_metrics["ORIGINAL-CACHED-QWEN"]["mean_nll"],
            seed_metrics=per_seed_metrics,
            reference_reproduced=reproduction["passed"],
            reference_required=args.protocol == "exp041",
            required_recovery_fraction=required_recovery,
        )
    bootstrap = None
    if args.protocol == "exp043-validation":
        bootstrap = bootstrap_excess_nll_recovery(
            original_window_nll=lm_metrics["ORIGINAL-CACHED-QWEN"][
                "window_mean_nll"
            ],
            seed_window_nll={
                str(seed): {
                    "mixer_only": lm_metrics[
                        f"SEED-{seed}-MIXER-ONLY-STEP{args.total_steps}"
                    ]["window_mean_nll"],
                    "joint": lm_metrics[
                        f"SEED-{seed}-JOINT-STEP{args.total_steps}"
                    ]["window_mean_nll"],
                }
                for seed in seeds
            },
        )
    elif objective_comparison:
        bootstrap = bootstrap_depth_objective_deltas(
            {
                str(seed): {
                    arm: per_seed_metrics[str(seed)][arm]["window_mean_nll"]
                    for arm in ("mixer_only", "joint", "contribution")
                }
                for seed in seeds
            },
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    all_finite = bool(
        training_result["passed"]
        and lm_metrics["ORIGINAL-CACHED-QWEN"]["finite"]
        and aggregate["all_finite"]
    )
    if objective_comparison:
        scientific_gate_passed = bool(
            all_finite
            and aggregate["mean_contribution_minus_mixer_nll"] < 0
            and aggregate["mean_contribution_minus_joint_nll"] < 0
            and aggregate["contribution_wins_vs_mixer"] >= 2
            and aggregate["contribution_wins_vs_joint"] >= 2
            and bootstrap is not None
            and bootstrap["contribution_minus_mixer_95ci"][1] < 0
            and bootstrap["contribution_minus_joint_95ci"][1] < 0
        )
    elif args.protocol == "exp043-validation":
        scientific_gate_passed = bool(
            all_finite
            and aggregate["scientific_gate_passed"]
            and bootstrap is not None
            and bootstrap["mean_recovery_confidence_interval"][0] >= 0.10
        )
    else:
        scientific_gate_passed = bool(
            training_result["scientific_gate_passed"]
            and aggregate["scientific_gate_passed"]
        )
    result = {
        "source": source_name,
        "dataset": manifest.get("dataset"),
        "method": PROTOCOL_METHODS[args.protocol],
        "protocol": args.protocol,
        "activation_cache_manifest": str(
            Path(args.activation_cache_manifest).resolve()
        ),
        "resolved_activation_artifacts": {
            name: str(path) for name, path in paths.items()
        },
        "evaluation_cache_manifest": (
            str(Path(args.evaluation_cache_manifest).resolve())
            if args.evaluation_cache_manifest
            else None
        ),
        "resolved_evaluation_artifacts": {
            name: str(path) for name, path in evaluation_paths.items()
        },
        "evaluation_token_offset": evaluation_manifest.get("token_offset", 0),
        "evaluation_token_range": evaluation_manifest.get("token_range"),
        "evaluation_dataset_split": evaluation_manifest.get(
            "dataset_split", "train"
        ),
        "reference_exp040_json": (
            str(reference_path.resolve()) if args.protocol == "exp041" else None
        ),
        "target_layer": args.target_layer,
        "training_sequence_length": int(manifest["sequence_length"]),
        "sequence_length": evaluation_sequence_length,
        "evaluation_windows": evaluation_windows,
        "evaluation_tokens_per_branch": evaluation_windows
        * (evaluation_sequence_length - 1),
        "seeds": list(seeds),
        "branches": list(names),
        "training_result": training_result,
        "endpoint_checkpoint": checkpoint_metadata,
        "training_endpoints_restored": restored,
        "compute_dtype": dtype_decision.dtype,
        "jax_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "data_parallel": args.data_parallel,
        "per_device_windows": args.per_device_windows,
        "hidden_relative_l2_to_original": divergence,
        "lm_metrics": lm_metrics,
        "reference_reproduction": reproduction,
        "required_recovery_fraction": required_recovery,
        "required_bootstrap_lower_bound": (
            0.10 if args.protocol == "exp043-validation" else None
        ),
        "local_training_gate_required": (
            args.protocol != "exp043-validation"
            and args.protocol not in OBJECTIVE_COMPARISON_PROTOCOLS
        ),
        "objective_gate": (
            {
                "required_wins_per_control": 2,
                "required_mean_delta_sign": "negative",
                "required_bootstrap_upper_sign": "negative",
            }
            if objective_comparison
            else None
        ),
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "prune_consumed_shards": args.prune_consumed_shards,
        "removed_checkpoint_shards": sorted(set(removed_shards)),
        "scientific_gate_passed": scientific_gate_passed,
        "passed": all_finite,
        "notes": [
            "The original branch is shared; every objective arm uses an identical calibrated start and paired batches within each seed.",
            "All branches stream together from the target replacement through identical frozen later Qwen layers, final norm, and lm_head.",
            PROTOCOL_NOTES[args.protocol],
            "passed reports numerical execution; scientific_gate_passed also requires the local training and full-depth aggregate gates.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, args.output_dir
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if objective_comparison and args.stage_manifest:
        update_stage_manifest(
            args.stage_manifest,
            experiment=stage_experiment,
            stage=f"streamed-evaluation-{stage_suffix}",
            status="completed",
            details={
                "result_json": str(Path(args.result_json).resolve()),
                "scientific_gate_passed": scientific_gate_passed,
            },
        )
    if not all_finite:
        raise SystemExit("MULTISEED-STREAMED-END-TO-END-NONFINITE")
    label = PROTOCOL_LABELS[args.protocol]
    print(f"{label}-{'PASS' if scientific_gate_passed else 'GATE-FAIL'}")
    return result


if __name__ == "__main__":
    main()
