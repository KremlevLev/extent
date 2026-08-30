from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.config import load_config
from extent.full_hybrid_materialization import (
    materialize_exact_lift_layer,
    materialize_qwen_gqa_layer,
)
from extent.full_model_distillation import (
    full_model_eval_metrics,
    make_hidden_bridge_distill_step,
    make_prediction_distill_step,
)
from extent.initialization import (
    initialize_sharded_optimizer_state,
    initialize_sharded_parameters,
)
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen3_parity import load_mixer_arrays
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import (
    QWEN3_1_7B_BASE,
    load_remote_source_metadata,
    teacher_config_from_spec,
)
from extent.sharding import batch_sharding, create_v5e_mesh, replicated_sharding
from extent.weight_mapping import (
    QwenCheckpointReader,
    stream_direct_qwen_weights,
    stream_teacher_qwen_weights,
)
from scripts.qwen_extended_horizon_campaign import _safe_notify
from scripts.qwen_mamba3_distill_pilot import _write_json_with_output_mirror


PROTOCOL = "exp064-qwen17-full-model-m3q-distillation"
ARMS = ("EXACT-KL", "M3Q-HIDDEN-BRIDGE-KL", "RANDOM-KL")
CHECKPOINTS = (0, 128, 512, 1024, 2048, 3072)
TRAIN_OFFSET = 393_216
VALIDATION_OFFSET = 0
TEMPERATURE = 2.0
CROSS_ENTROPY_WEIGHT = 0.1
HIDDEN_STAGE_FRACTION = 0.25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_checkpoint(directory: Path) -> QwenCheckpointReader:
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    _, index = load_remote_source_metadata(QWEN3_1_7B_BASE)
    filenames = sorted(set(index["weight_map"].values()))
    for filename in ("config.json", *filenames):
        hf_hub_download(
            repo_id=QWEN3_1_7B_BASE.repo_id,
            revision=QWEN3_1_7B_BASE.revision,
            filename=filename,
            local_dir=directory,
        )
    if not (directory / "model.safetensors.index.json").exists():
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    return QwenCheckpointReader(directory)


def _materialize_student(
    model,
    config,
    source,
    reader,
    mesh,
    init_tokens,
    *,
    exact_mamba: bool,
):
    initialized = initialize_sharded_parameters(
        model, jax.random.key(640), init_tokens, mesh
    )
    params, direct_report = stream_direct_qwen_weights(
        initialized.params, reader, config
    )
    mixer_reports = []
    for layer in range(config.num_layers):
        arrays = load_mixer_arrays(reader, source, layer)
        if layer in config.attention_layer_indices:
            params, report = materialize_qwen_gqa_layer(
                params, arrays, source, layer
            )
            mixer_reports.append(report.to_dict())
        elif exact_mamba:
            params, report = materialize_exact_lift_layer(
                params, arrays, source, config, layer
            )
            mixer_reports.append(report.to_dict())
        del arrays
    return initialized, params, direct_report, mixer_reports


def _metric_record(metrics: dict) -> dict:
    return {
        key: bool(value)
        if key in {"grads_finite", "finite"}
        else int(value)
        if key == "nonfinite_grad_leaves"
        else float(value)
        for key, value in metrics.items()
    }


def aggregate_results(arms: dict[str, dict], checkpoints: tuple[int, ...]) -> dict:
    complete = {
        arm: row for arm, row in arms.items() if row.get("complete")
    }
    curves = {
        arm: np.asarray([
            row["evaluations"][str(step)]["excess_nll"]
            for step in checkpoints
        ], dtype=np.float64)
        for arm, row in complete.items()
    }
    x = np.asarray(checkpoints, np.float64) / checkpoints[-1]
    auc = {
        arm: float(np.sum(0.5 * (curve[:-1] + curve[1:]) * np.diff(x)))
        for arm, curve in curves.items()
    }
    exact = complete.get("EXACT-KL")
    staged = complete.get("M3Q-HIDDEN-BRIDGE-KL")
    if exact is None or staged is None:
        return {
            "complete_primary_pair": False,
            "excess_nll_auc": auc,
            "scientific_gate_passed": False,
        }
    exact_final = exact["evaluations"][str(checkpoints[-1])]
    staged_final = staged["evaluations"][str(checkpoints[-1])]
    final_gain = 1.0 - staged_final["excess_nll"] / exact_final["excess_nll"]
    auc_gain = 1.0 - auc["M3Q-HIDDEN-BRIDGE-KL"] / auc["EXACT-KL"]
    return {
        "complete_primary_pair": True,
        "excess_nll_auc": auc,
        "final_exact_excess_nll": exact_final["excess_nll"],
        "final_staged_excess_nll": staged_final["excess_nll"],
        "final_staged_relative_gain": float(final_gain),
        "staged_auc_relative_gain": float(auc_gain),
        "final_exact_prediction_kl": exact_final["prediction_kl"],
        "final_staged_prediction_kl": staged_final["prediction_kl"],
        "scientific_gate_passed": bool(
            exact_final["excess_nll"] > 0
            and staged_final["excess_nll"] < exact_final["excess_nll"]
            and staged_final["prediction_kl"] < exact_final["prediction_kl"]
            and auc["M3Q-HIDDEN-BRIDGE-KL"] < auc["EXACT-KL"]
        ),
        "gate_definition": (
            "The staged exact-init arm must beat ordinary exact-init KL in "
            "held-out final excess NLL, final prediction KL, and normalized "
            "excess-NLL AUC. Random-KL is contextual and not required if the "
            "wall deadline is reached after the primary pair."
        ),
    }


def render_summary(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# EXP-064 Qwen3-1.7B full-model M3Q distillation",
        "",
        f"- Status: `{result['status']}`",
        f"- Duration: `{result['duration_hours']:.3f}` hours",
        f"- Completed arms: `{len([x for x in result['arms'].values() if x.get('complete')])}/3`",
        f"- Scientific gate: `{aggregate['scientific_gate_passed']}`",
        "",
        "| Arm | Steps | Final excess NLL | Final KL | Final agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in result["arms"].items():
        if not row.get("evaluations"):
            continue
        step = str(row["completed_steps"])
        metrics = row["evaluations"][step]
        lines.append(
            f"| {arm} | {step} | {metrics['excess_nll']:.7f} | "
            f"{metrics['prediction_kl']:.7f} | {metrics['top1_agreement']:.5f} |"
        )
    if aggregate.get("complete_primary_pair"):
        lines += [
            "",
            f"- Staged final excess-NLL gain: `{100 * aggregate['final_staged_relative_gain']:+.3f}%`",
            f"- Staged excess-NLL AUC gain: `{100 * aggregate['staged_auc_relative_gain']:+.3f}%`",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run full-model M3Q distillation on Qwen3-1.7B."
    )
    parser.add_argument("--config", default="config/hybrid_1_7b_gqa_v5e8.yaml")
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp064-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--total-steps", type=int, default=3072)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--max-wall-hours", type=float, default=7.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.total_steps != CHECKPOINTS[-1]:
        raise ValueError(f"registered EXP-064 total steps are {CHECKPOINTS[-1]}")
    devices = list(jax.devices())
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise ValueError("EXP-064 requires one TPU v5e-8")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "exp064-partial.json"
    result_path = output / "extent-qwen17-full-model-m3q.json"
    summary_path = output / "extent-qwen17-full-model-m3q-summary.md"
    started_clock = time.monotonic()
    deadline = started_clock + args.max_wall_hours * 3600
    stage = "startup"
    result = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at_utc": _now(),
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "arms": {},
        "checkpoints": list(CHECKPOINTS),
        "sequence_length": args.sequence_length,
        "training_token_offset": TRAIN_OFFSET,
        "validation_token_offset": VALIDATION_OFFSET,
        "temperature": TEMPERATURE,
        "cross_entropy_weight": CROSS_ENTROPY_WEIGHT,
        "hidden_stage_fraction": HIDDEN_STAGE_FRACTION,
    }

    def persist() -> None:
        _write_json_with_output_mirror(partial_path, result, str(output))

    _safe_notify(
        args.telegram,
        "Extent TPU campaign\nstatus=started\nexperiment=EXP-064 full-model M3Q"
        f"\nhost={socket.gethostname()}\nhard_budget_hours={args.max_wall_hours}",
    )
    try:
        config, extras = load_config(args.config)
        if config.mla.implementation != "qwen3_gqa":
            raise ValueError("EXP-064 isolates Mamba recovery and requires retained GQA")
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE,
            param_dtype="bfloat16",
            compute_dtype="bfloat16",
            remat_policy="full",
        )
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        replicated = replicated_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)

        stage = "checkpoint"
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        stage = "teacher"
        teacher_model = Qwen3ForCausalLM(source)
        teacher_initialized = initialize_sharded_parameters(
            teacher_model, jax.random.key(641), init_tokens, mesh
        )
        teacher_params, teacher_report = stream_teacher_qwen_weights(
            teacher_initialized.params, reader, source
        )
        result["teacher_mapping"] = {
            "tensor_count": teacher_report.tensor_count,
            "parameter_count": teacher_report.parameter_count,
        }

        stage = "data"
        train_tokens = load_wikitext2_tokens(
            args.total_steps * args.sequence_length,
            args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=TRAIN_OFFSET,
            dataset_split="train",
        ).reshape(args.total_steps, args.sequence_length)
        validation_tokens = load_wikitext2_tokens(
            args.validation_windows * args.sequence_length,
            args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=VALIDATION_OFFSET,
            dataset_split="validation",
        ).reshape(args.validation_windows, args.sequence_length)
        student_model = HybridForCausalLM(config)

        def teacher_apply(params, tokens, return_hidden):
            if return_hidden:
                return teacher_model.apply(
                    {"params": params}, tokens, return_hidden_states=True
                )
            return teacher_model.apply({"params": params}, tokens), ()

        def student_apply(params, tokens, return_hidden):
            if return_hidden:
                return student_model.apply(
                    {"params": params}, tokens, return_hidden_states=True
                )
            return student_model.apply({"params": params}, tokens), ()

        metric_layout = {
            name: replicated for name in (
                "loss", "prediction_kl", "cross_entropy", "hidden_loss",
                "grad_norm", "grads_finite", "nonfinite_grad_leaves", "max_abs_grad",
            )
        }
        eval_layout = {
            name: replicated for name in (
                "student_nll", "teacher_nll", "excess_nll", "prediction_kl",
                "top1_agreement", "finite",
            )
        }

        for arm in ARMS:
            if time.monotonic() >= deadline:
                break
            stage = f"arm-{arm}"
            exact = arm != "RANDOM-KL"
            initialized, params, direct_report, mixer_reports = _materialize_student(
                student_model, config, source, reader, mesh, init_tokens,
                exact_mamba=exact,
            )
            training = extras["training"]
            tx = create_lion(
                learning_rate=float(training["learning_rate"]),
                warmup_steps=int(training["warmup_steps"]),
                total_steps=args.total_steps,
                weight_decay=float(training["weight_decay"]),
                max_grad_norm=float(training["max_grad_norm"]),
            )
            optimizer = initialize_sharded_optimizer_state(
                tx, params, initialized.abstract_params, initialized.layout, mesh
            )
            opt_state = optimizer.opt_state
            prediction_step = jax.jit(
                make_prediction_distill_step(
                    student_apply, teacher_apply, tx,
                    temperature=TEMPERATURE,
                    cross_entropy_weight=CROSS_ENTROPY_WEIGHT,
                    bf16_gradients=True,
                ),
                in_shardings=(initialized.layout, optimizer.layout,
                              teacher_initialized.layout, batch_layout),
                out_shardings=(initialized.layout, optimizer.layout, metric_layout),
                donate_argnums=(0, 1),
            )
            hidden_step = None
            if arm == "M3Q-HIDDEN-BRIDGE-KL":
                hidden_step = jax.jit(
                    make_hidden_bridge_distill_step(
                        student_apply, teacher_apply, tx,
                        layer_indices=config.mamba_layer_indices,
                        temperature=TEMPERATURE,
                        cross_entropy_weight=CROSS_ENTROPY_WEIGHT,
                        prediction_weight=1.0,
                        hidden_weight=1.0,
                        bf16_gradients=True,
                    ),
                    in_shardings=(initialized.layout, optimizer.layout,
                                  teacher_initialized.layout, batch_layout),
                    out_shardings=(initialized.layout, optimizer.layout, metric_layout),
                    donate_argnums=(0, 1),
                )

            def evaluate(student_parameters, teacher_parameters, tokens):
                student_logits = student_model.apply(
                    {"params": student_parameters}, tokens
                )
                teacher_logits = teacher_model.apply(
                    {"params": teacher_parameters}, tokens
                )
                return full_model_eval_metrics(
                    student_logits, teacher_logits, tokens,
                    temperature=TEMPERATURE,
                )

            evaluate = jax.jit(
                evaluate,
                in_shardings=(initialized.layout, teacher_initialized.layout,
                              batch_layout),
                out_shardings=eval_layout,
            )

            def evaluate_validation(student_parameters) -> dict:
                records = []
                for window in validation_tokens:
                    batch = jax.device_put(window[None, :], batch_layout)
                    metrics = evaluate(student_parameters, teacher_params, batch)
                    jax.block_until_ready(metrics)
                    records.append(_metric_record(metrics))
                return {
                    name: bool(all(record[name] for record in records))
                    if name == "finite"
                    else float(np.mean([record[name] for record in records]))
                    for name in records[0]
                }
            arm_result = {
                "initializer": "exact_mimo_lift" if exact else "random_mamba3",
                "recipe": (
                    "hidden_bridge_then_prediction_kl"
                    if hidden_step is not None else "prediction_kl"
                ),
                "direct_tensor_count": direct_report.tensor_count,
                "mixer_reports": mixer_reports,
                "evaluations": {},
                "training_metrics": {},
                "completed_steps": 0,
                "complete": False,
                "finite": True,
            }
            result["arms"][arm] = arm_result
            initial_metrics = evaluate_validation(params)
            arm_result["evaluations"]["0"] = initial_metrics
            persist()
            hidden_steps = int(args.total_steps * HIDDEN_STAGE_FRACTION)
            for zero_step in range(args.total_steps):
                if zero_step % 8 == 0 and time.monotonic() >= deadline:
                    break
                index = int(deterministic_batch_indices(
                    zero_step, 1, len(train_tokens), 20260830
                )[0])
                batch = jax.device_put(train_tokens[index:index + 1], batch_layout)
                if hidden_step is not None and zero_step < hidden_steps:
                    params, opt_state, metrics = hidden_step(
                        params, opt_state, teacher_params, batch
                    )
                else:
                    params, opt_state, metrics = prediction_step(
                        params, opt_state, teacher_params, batch
                    )
                jax.block_until_ready(metrics)
                step = zero_step + 1
                arm_result["completed_steps"] = step
                arm_result["finite"] = bool(
                    arm_result["finite"] and bool(metrics["grads_finite"])
                    and np.isfinite(float(metrics["loss"]))
                )
                if step in CHECKPOINTS:
                    record = _metric_record(metrics)
                    arm_result["training_metrics"][str(step)] = record
                    evaluated = evaluate_validation(params)
                    arm_result["evaluations"][str(step)] = evaluated
                    arm_result["finite"] = bool(
                        arm_result["finite"] and evaluated["finite"]
                    )
                    persist()
                    print(
                        f"exp064_arm={arm} step={step} "
                        f"excess_nll={evaluated['excess_nll']:.6g} "
                        f"kl={evaluated['prediction_kl']:.6g}"
                    )
            final_step = str(arm_result["completed_steps"])
            if final_step not in arm_result["evaluations"]:
                # A hard wall deadline can fall between registered checkpoints.
                # Preserve one compact deployable endpoint before returning.
                evaluated = evaluate_validation(params)
                arm_result["evaluations"][final_step] = evaluated
                arm_result["finite"] = bool(
                    arm_result["finite"] and evaluated["finite"]
                )
            arm_result["complete"] = arm_result["completed_steps"] == args.total_steps
            persist()
            del params, opt_state, optimizer, initialized, prediction_step, hidden_step
            jax.clear_caches()
            gc.collect()
            if not arm_result["complete"]:
                break

        aggregate = aggregate_results(result["arms"], CHECKPOINTS)
        primary_complete = aggregate["complete_primary_pair"]
        all_complete = all(
            result["arms"].get(arm, {}).get("complete") for arm in ARMS
        )
        result.update({
            "status": "completed" if all_complete else "deadline_partial",
            "complete": all_complete,
            "passed": bool(primary_complete and all(
                row.get("finite") for row in result["arms"].values()
            )),
            "aggregate": aggregate,
            "completed_at_utc": _now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
        })
        _write_json_with_output_mirror(result_path, result, str(output))
        summary_path.write_text(render_summary(result), encoding="utf-8")
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment=EXP-064"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\ngate={aggregate['scientific_gate_passed']}",
        )
        print(f"EXP064-{result['status'].upper()}\nsummary={summary_path.resolve()}")
        return result
    except BaseException as exc:
        result.update({
            "status": "failed", "stage": stage,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_at_utc": _now(),
            "duration_hours": (time.monotonic() - started_clock) / 3600,
        })
        _write_json_with_output_mirror(
            output / "exp064-failure.json", result, str(output)
        )
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-064"
            f"\nstage={stage}\nerror={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
