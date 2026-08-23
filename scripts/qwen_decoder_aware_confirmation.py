from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import jax
import numpy as np

from scripts.qwen_decoder_aware_distill import main as run_single_seed
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(","))
    if len(seeds) != 3 or len(set(seeds)) != 3 or min(seeds) < 0:
        raise ValueError("confirmation requires exactly three distinct non-negative seeds")
    return seeds


def aggregate_confirmation(seed_results: dict[str, dict], total_steps: int) -> dict:
    endpoint = str(total_steps)
    paired = []
    step0_identical = True
    all_finite = True
    for seed, result in seed_results.items():
        mixer_arm = result["arms"]["MIXER-ONLY"]
        joint_arm = result["arms"]["JOINT-MIXER-DECODER"]
        mixer_final = mixer_arm["evaluations"][endpoint]
        joint_final = joint_arm["evaluations"][endpoint]
        mixer_decoder_l2 = mixer_final["decoder_output"]["relative_l2"]
        joint_decoder_l2 = joint_final["decoder_output"]["relative_l2"]
        mixer_mixer_l2 = mixer_final["mixer_output"]["relative_l2"]
        joint_mixer_l2 = joint_final["mixer_output"]["relative_l2"]
        decoder_improvement = (
            mixer_decoder_l2 - joint_decoder_l2
        ) / mixer_decoder_l2
        mixer_degradation = (joint_mixer_l2 - mixer_mixer_l2) / mixer_mixer_l2
        identical = (
            mixer_arm["evaluations"]["0"] == joint_arm["evaluations"]["0"]
        )
        finite = bool(result["passed"] and mixer_arm["finite"] and joint_arm["finite"])
        paired.append(
            {
                "seed": int(seed),
                "mixer_only_decoder_relative_l2": mixer_decoder_l2,
                "joint_decoder_relative_l2": joint_decoder_l2,
                "decoder_improvement_fraction": decoder_improvement,
                "mixer_only_mixer_relative_l2": mixer_mixer_l2,
                "joint_mixer_relative_l2": joint_mixer_l2,
                "mixer_degradation_fraction": mixer_degradation,
                "joint_wins_decoder": joint_decoder_l2 < mixer_decoder_l2,
                "step0_identical": identical,
                "finite": finite,
            }
        )
        step0_identical = step0_identical and identical
        all_finite = all_finite and finite

    improvements = np.asarray(
        [record["decoder_improvement_fraction"] for record in paired],
        dtype=np.float64,
    )
    degradations = np.asarray(
        [record["mixer_degradation_fraction"] for record in paired],
        dtype=np.float64,
    )
    mixer_decoder = np.asarray(
        [record["mixer_only_decoder_relative_l2"] for record in paired],
        dtype=np.float64,
    )
    joint_decoder = np.asarray(
        [record["joint_decoder_relative_l2"] for record in paired],
        dtype=np.float64,
    )
    wins = sum(record["joint_wins_decoder"] for record in paired)
    mean_improvement = float(np.mean(improvements))
    maximum_degradation = float(np.max(degradations))
    scientific_gate_passed = bool(
        all_finite
        and step0_identical
        and mean_improvement >= 0.10
        and wins >= 2
        and maximum_degradation <= 0.10
    )
    return {
        "paired_endpoints": paired,
        "mixer_only_decoder_relative_l2_mean": float(np.mean(mixer_decoder)),
        "mixer_only_decoder_relative_l2_std": float(np.std(mixer_decoder)),
        "joint_decoder_relative_l2_mean": float(np.mean(joint_decoder)),
        "joint_decoder_relative_l2_std": float(np.std(joint_decoder)),
        "decoder_improvement_fraction_mean": mean_improvement,
        "decoder_improvement_fraction_std": float(np.std(improvements)),
        "mixer_degradation_fraction_mean": float(np.mean(degradations)),
        "mixer_degradation_fraction_max": maximum_degradation,
        "joint_decoder_wins": wins,
        "required_joint_decoder_wins": 2,
        "required_decoder_improvement_fraction_mean": 0.10,
        "maximum_allowed_mixer_degradation_per_seed": 0.10,
        "step0_identical_all_seeds": step0_identical,
        "all_finite": all_finite,
        "scientific_gate_passed": scientific_gate_passed,
    }


def main(
    argv: list[str] | None = None,
    *,
    return_endpoint_params: bool = False,
) -> dict | tuple[dict, dict[str, dict], dict[str, dict[str, dict]]]:
    parser = argparse.ArgumentParser(
        description="Confirm decoder-aware Mamba distillation across three paired seeds."
    )
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--activation-cache-dir")
    parser.add_argument("--evaluation-cache-manifest")
    parser.add_argument("--evaluation-cache-dir")
    parser.add_argument("--allow-cross-split-evaluation", action="store_true")
    parser.add_argument("--qwen-cache-dir", default="/content/qwen3-layer0-weights")
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--data-seed", type=int, default=20260820)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--checkpoints", default="0,128,256,512,1024")
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--evaluation-batch-windows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--readout-ridge", type=float, default=1e-2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--contribution-mixer-weight", type=float, default=1.0)
    parser.add_argument("--include-contribution-arm", action="store_true")
    parser.add_argument("--compute-dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", default="/content/output")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = {}
    initial_params_by_seed = {}
    endpoint_params_by_seed = {}
    for seed in seeds:
        seed_json = output_dir / f"exp039-decoder-aware-seed{seed}.json"
        single_args = [
            "--activation-cache-manifest", args.activation_cache_manifest,
            "--qwen-cache-dir", args.qwen_cache_dir,
            "--total-steps", str(args.total_steps),
            "--checkpoints", args.checkpoints,
            "--batch-windows", str(args.batch_windows),
            "--evaluation-batch-windows", str(args.evaluation_batch_windows),
            "--learning-rate", str(args.learning_rate),
            "--readout-ridge", str(args.readout_ridge),
            "--decoder-loss-weight", str(args.decoder_loss_weight),
            "--contribution-mixer-weight", str(args.contribution_mixer_weight),
            "--seed", str(seed),
            "--data-seed", str(args.data_seed),
            "--compute-dtype", args.compute_dtype,
            "--result-json", str(seed_json),
            "--output-dir", str(output_dir),
        ]
        if args.activation_cache_dir:
            single_args.extend(
                ["--activation-cache-dir", args.activation_cache_dir]
            )
        if args.evaluation_cache_manifest:
            single_args.extend(
                ["--evaluation-cache-manifest", args.evaluation_cache_manifest]
            )
        if args.evaluation_cache_dir:
            single_args.extend(
                ["--evaluation-cache-dir", args.evaluation_cache_dir]
            )
        if args.allow_cross_split_evaluation:
            single_args.append("--allow-cross-split-evaluation")
        if args.skip_hash_verification:
            single_args.append("--skip-hash-verification")
        if args.include_contribution_arm:
            single_args.append("--include-contribution-arm")
        print(f"EXP-039 seed={seed} START")
        if return_endpoint_params:
            seed_result, initial_params, endpoint_params = run_single_seed(
                single_args, return_endpoint_params=True
            )
            seed_results[str(seed)] = seed_result
            initial_params_by_seed[str(seed)] = initial_params
            endpoint_params_by_seed[str(seed)] = endpoint_params
        else:
            seed_results[str(seed)] = run_single_seed(single_args)
        print(f"EXP-039 seed={seed} DONE")
        jax.clear_caches()
        gc.collect()

    aggregate = aggregate_confirmation(seed_results, args.total_steps)
    first = seed_results[str(seeds[0])]
    result = {
        "source": first["source"],
        "dataset": first["dataset"],
        "method": "three_seed_decoder_aware_Mamba3_confirmation",
        "activation_cache_manifest": first["activation_cache_manifest"],
        "evaluation_cache_manifest": first["evaluation_cache_manifest"],
        "evaluation_windows": first["evaluation_windows"],
        "evaluation_token_offset": first["evaluation_token_offset"],
        "target_layer": first["target_layer"],
        "sequence_length": first["sequence_length"],
        "seeds": list(seeds),
        "data_seed": args.data_seed,
        "total_steps_per_arm": args.total_steps,
        "unique_training_tokens_per_arm": first["unique_training_tokens_per_arm"],
        "optimizer_visible_tokens_total": (
            first["unique_training_tokens_per_arm"]
            * (3 if args.include_contribution_arm else 2)
            * len(seeds)
        ),
        "checkpoints": first["checkpoints"],
        "compute_dtype": first["compute_dtype"],
        "jax_backend": first["jax_backend"],
        "learning_rate": args.learning_rate,
        "readout_ridge": args.readout_ridge,
        "decoder_loss_weight": args.decoder_loss_weight,
        "contribution_mixer_weight": args.contribution_mixer_weight,
        "include_contribution_arm": args.include_contribution_arm,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "scientific_gate_passed": aggregate["scientific_gate_passed"],
        "passed": aggregate["all_finite"],
        "notes": [
            "Every seed uses identical calibrated starts and batches for every enabled objective arm.",
            "All seeds share the frozen cache, data order, optimizer schedule, and decoder tail.",
            "passed reports numerical execution; scientific_gate_passed reports the frozen multi-seed threshold.",
        ],
    }
    mirror = _write_json_with_output_mirror(
        Path(args.result_json), result, args.output_dir
    )
    print(json.dumps(result, indent=2))
    print(f"result_json={Path(args.result_json).resolve()}")
    if mirror:
        print(f"output_json={mirror.resolve()}")
    if not aggregate["all_finite"]:
        raise SystemExit("DECODER-AWARE-CONFIRMATION-NONFINITE")
    print(
        "DECODER-AWARE-CONFIRMATION-PASS"
        if aggregate["scientific_gate_passed"]
        else "DECODER-AWARE-CONFIRMATION-GATE-FAIL"
    )
    if return_endpoint_params:
        return result, initial_params_by_seed, endpoint_params_by_seed
    return result


if __name__ == "__main__":
    main()
