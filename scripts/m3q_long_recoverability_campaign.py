"""EXP-089: long-horizon full-model exact-lift versus random recovery.

One invocation intentionally consumes almost one Kaggle TPU session.  Complete
model and Lion states are uploaded to the configured HF dataset and the next
invocation resumes the same trajectory instead of restarting it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
import traceback

import jax
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.config import load_config
from extent.full_model_distillation import (
    full_model_eval_metrics,
    make_prediction_distill_step,
)
from extent.hf_artifact_sync import artifact_config_from_env, upload_artifact
from extent.initialization import (
    initialize_sharded_optimizer_state,
    initialize_sharded_parameters,
)
from extent.offline_distillation import deterministic_batch_indices
from extent.optimizer import create_lion
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.sharding import batch_sharding, create_v5e_mesh, replicated_sharding
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.qwen17_full_model_distill_campaign import (
    _ensure_checkpoint,
    _materialize_student,
    _metric_record,
)
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp089-long-full-model-recoverability-v1"
HF_PREFIX = "experiments/exp089-long-recoverability"
SEEDS = (123, 456)
ARMS = ("RANDOM", "EXACT-LIFT")
TOTAL_STEPS = 98_304
SEQUENCE_LENGTH = 256
TOKENS_PER_TRAJECTORY = TOTAL_STEPS * SEQUENCE_LENGTH
CHECKPOINTS = (0, 3_072, 8_192, 16_384, 32_768, 65_536, 98_304)
SAVE_EVERY = 8_192
TRAIN_OFFSET = 5_242_880
EVAL_OFFSET = 0
EVAL_WINDOWS = 32
DATA_SEED = 89000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_contract(config) -> dict:
    return {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "model": asdict(config),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "total_steps": TOTAL_STEPS,
        "sequence_length": SEQUENCE_LENGTH,
        "tokens_per_trajectory": TOKENS_PER_TRAJECTORY,
        "checkpoints": list(CHECKPOINTS),
        "save_every": SAVE_EVERY,
        "train": {
            "dataset": "Salesforce/wikitext",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "offset": TRAIN_OFFSET,
        },
        "evaluation": {
            "config": "wikitext-103-raw-v1",
            "split": "validation",
            "offset": EVAL_OFFSET,
            "windows": EVAL_WINDOWS,
        },
        "optimizer": {
            "name": "lion",
            "learning_rate": 3e-5,
            "warmup_steps": 512,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "objective": {
            "teacher_to_student_kl_temperature": 2.0,
            "true_token_cross_entropy_weight": 0.1,
        },
        "primary_question": (
            "Does exact MIMO QKVO lift cross below canonical random Mamba-3 "
            "initialization during 25.166M-token whole-model recovery?"
        ),
    }


def aggregate(result: dict) -> dict:
    paired = {}
    for seed in SEEDS:
        rows = result.get("trajectories", {}).get(str(seed), {})
        random = rows.get("RANDOM", {})
        exact = rows.get("EXACT-LIFT", {})
        shared = sorted(
            set(map(int, random.get("evaluations", {})))
            & set(map(int, exact.get("evaluations", {})))
        )
        comparisons = {}
        for step in shared:
            r = random["evaluations"][str(step)]
            e = exact["evaluations"][str(step)]
            comparisons[str(step)] = {
                "exact_minus_random_excess_nll": float(
                    e["excess_nll"] - r["excess_nll"]
                ),
                "exact_minus_random_prediction_kl": float(
                    e["prediction_kl"] - r["prediction_kl"]
                ),
                "exact_wins_nll": bool(e["excess_nll"] < r["excess_nll"]),
                "exact_wins_kl": bool(e["prediction_kl"] < r["prediction_kl"]),
            }
        paired[str(seed)] = {
            "shared_checkpoints": shared,
            "comparisons": comparisons,
            "complete_pair": bool(random.get("complete") and exact.get("complete")),
        }
    complete = all(row["complete_pair"] for row in paired.values())
    final_wins = 0
    if complete:
        for row in paired.values():
            endpoint = row["comparisons"][str(TOTAL_STEPS)]
            final_wins += int(endpoint["exact_wins_nll"] and endpoint["exact_wins_kl"])
    return {
        "paired": paired,
        "completed_trajectories": sum(
            int(row.get("complete", False))
            for rows in result.get("trajectories", {}).values()
            for row in rows.values()
        ),
        "complete": complete,
        "exact_endpoint_wins": final_wins,
        "scientific_gate_passed": bool(complete and final_wins == len(SEEDS)),
        "gate_definition": (
            "At step 98,304 exact lift must beat random in both held-out "
            "excess NLL and prediction KL for both paired seeds. Curves and "
            "first crossover checkpoints remain primary diagnostic outputs."
        ),
    }


def render_summary(result: dict) -> str:
    agg = result["aggregate"]
    lines = [
        "# EXP-089 long-horizon full-model recoverability",
        "",
        f"- Status: `{result['status']}`",
        f"- This invocation: `{result['duration_hours']:.3f}` hours",
        f"- Completed trajectories: `{agg['completed_trajectories']}/4`",
        f"- Tokens per complete trajectory: `{TOKENS_PER_TRAJECTORY:,}`",
        f"- Scientific gate: `{agg['scientific_gate_passed']}`",
        "",
        "| Seed | Arm | Step | Excess NLL | Prediction KL | Complete |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, rows in result.get("trajectories", {}).items():
        for arm, row in rows.items():
            step = int(row.get("completed_steps", 0))
            metrics = row.get("evaluations", {}).get(str(step))
            if metrics is None:
                prior = [int(x) for x in row.get("evaluations", {})]
                if not prior:
                    continue
                step = max(prior)
                metrics = row["evaluations"][str(step)]
            lines.append(
                f"| {seed} | {arm} | {step:,} | {metrics['excess_nll']:.7f} | "
                f"{metrics['prediction_kl']:.7f} | {row.get('complete', False)} |"
            )
    lines += [
        "",
        "The campaign is intentionally multi-session. Rerunning the same entry "
        "point restores the complete model, Lion state, step cursor, and metrics "
        "from Hugging Face and continues without repeating durable work.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp089-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp089-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.35)
    parser.add_argument("--hf-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1.0 <= args.max_wall_hours <= 8.5:
        raise ValueError("max-wall-hours must be between 1.0 and 8.5")

    started = time.monotonic()
    # Reserve time for the final multi-GB checkpoint upload and compact report.
    deadline = started + args.max_wall_hours * 3600 - 35 * 60
    output = Path(args.output_dir)
    state_root = Path(args.state_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    existing = sum(p.stat().st_size for p in state_root.rglob("*") if p.is_file())
    if shutil.disk_usage(state_root).free + existing < 24 * 1024**3:
        raise ValueError("state-dir requires at least 24 GiB free including existing checkpoints")
    hub = artifact_config_from_env() if args.hf_sync else None
    if args.hf_sync and hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    store = CampaignCheckpointStore(state_root, f"{HF_PREFIX}/checkpoints", hub)
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = experiment_contract(config)
    result_path = output / "extent-m3q-long-recoverability.json"
    summary_path = output / "extent-m3q-long-recoverability-summary.md"
    result = {
        "protocol": PROTOCOL,
        "contract": contract,
        "status": "running",
        "started_at_utc": _now(),
        "trajectories": {},
        "checkpoint_events": [],
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=False,
    )
    result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    result["software"] = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "flax", "optax", "huggingface_hub")
    }
    result["software"]["python"] = platform.python_version()
    stage = "startup"

    def persist(upload: bool = True) -> None:
        result["duration_hours"] = (time.monotonic() - started) / 3600
        result["checkpoint_events"] = store.events
        result["aggregate"] = aggregate(result)
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload and hub is not None:
            for local, remote in (
                (result_path, "latest.json"),
                (summary_path, "latest-summary.md"),
            ):
                try:
                    upload_artifact(local, f"{HF_PREFIX}/{remote}", hub,
                                    commit_message="EXP-089 progress")
                except Exception as exc:
                    print(f"hf_summary=FAILED type={type(exc).__name__}; local retained", flush=True)

    _safe_notify(
        args.telegram,
        "Extent TPU campaign\nstatus=started\nexperiment=EXP-089 long recoverability"
        f"\nhost={socket.gethostname()}\nsoft_budget_hours={args.max_wall_hours}",
    )
    try:
        devices = list(jax.devices())
        result["devices"] = [str(device) for device in devices]
        if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
            raise ValueError("EXP-089 requires one TPU v5e-8")
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        replicated = replicated_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16",
            remat_policy="full",
        )
        stage = "checkpoint-and-data"
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        train = load_wikitext2_tokens(
            TOKENS_PER_TRAJECTORY, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=TRAIN_OFFSET, dataset_split="train",
            dataset_config="wikitext-103-raw-v1",
        ).reshape(TOTAL_STEPS, SEQUENCE_LENGTH)
        validation = load_wikitext2_tokens(
            EVAL_WINDOWS * SEQUENCE_LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
            tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=EVAL_OFFSET, dataset_split="validation",
            dataset_config="wikitext-103-raw-v1",
        ).reshape(EVAL_WINDOWS, SEQUENCE_LENGTH)
        result["data_sha256"] = {
            "train": hashlib.sha256(train.tobytes()).hexdigest(),
            "validation": hashlib.sha256(validation.tobytes()).hexdigest(),
        }

        stage = "teacher"
        teacher_model = Qwen3ForCausalLM(source)
        teacher_init = initialize_sharded_parameters(
            teacher_model, jax.random.key(891), init_tokens, mesh
        )
        teacher_params, teacher_report = stream_teacher_qwen_weights(
            teacher_init.params, reader, source
        )
        teacher_layout = teacher_init.layout
        result["teacher_tensor_count"] = teacher_report.tensor_count
        del teacher_init

        def teacher_apply(params, tokens, return_hidden):
            return teacher_model.apply({"params": params}, tokens), ()

        execution_order = (
            (123, "RANDOM"), (123, "EXACT-LIFT"),
            (456, "EXACT-LIFT"), (456, "RANDOM"),
        )
        for seed, arm in execution_order:
            if time.monotonic() + 15 * 60 >= deadline:
                break
            stage = f"seed-{seed}-{arm}"
            slot = f"seed-{seed}/{arm.lower()}"
            checkpoint_contract = dict(
                contract, kind="full_model_trajectory", seed=seed, arm=arm
            )
            meta = store.metadata(slot, checkpoint_contract)
            if meta is not None and meta["metrics"].get("complete"):
                result["trajectories"].setdefault(str(seed), {})[arm] = meta["metrics"]
                continue

            model = HybridForCausalLM(config)
            initialized, params, direct_report, mixer_reports = _materialize_student(
                model, config, source, reader, mesh, init_tokens,
                exact_mamba=arm == "EXACT-LIFT",
            )
            layout, abstract = initialized.layout, initialized.abstract_params
            tx = create_lion(
                learning_rate=3e-5, warmup_steps=512, total_steps=TOTAL_STEPS,
                weight_decay=0.0, max_grad_norm=1.0,
            )
            optimizer = initialize_sharded_optimizer_state(
                tx, params, abstract, layout, mesh
            )
            opt_state, opt_layout = optimizer.opt_state, optimizer.layout
            row = {
                "initializer": "random_mamba3" if arm == "RANDOM" else "exact_mimo_lift",
                "complete": False, "finite": True, "completed_steps": 0,
                "tokens_seen": 0, "evaluations": {}, "training_metrics": {},
                "direct_tensor_count": direct_report.tensor_count,
                "mixer_reports": mixer_reports,
            }
            if meta is not None:
                restored, meta = store.restore(
                    slot, checkpoint_contract,
                    {"params": params, "opt_state": opt_state},
                )
                params = jax.tree.map(jax.device_put, restored["params"], layout)
                opt_state = jax.tree.map(jax.device_put, restored["opt_state"], opt_layout)
                row = meta["metrics"]
                del restored
            result["trajectories"].setdefault(str(seed), {})[arm] = row

            def student_apply(p, tokens, return_hidden):
                return model.apply({"params": p}, tokens), ()

            metric_layout = {
                name: replicated for name in (
                    "loss", "prediction_kl", "cross_entropy", "hidden_loss",
                    "grad_norm", "grads_finite", "nonfinite_grad_leaves",
                    "max_abs_grad",
                )
            }
            train_step = jax.jit(
                make_prediction_distill_step(
                    student_apply, teacher_apply, tx, temperature=2.0,
                    cross_entropy_weight=0.1, bf16_gradients=True,
                ),
                in_shardings=(layout, opt_layout, teacher_layout, batch_layout),
                out_shardings=(layout, opt_layout, metric_layout),
                donate_argnums=(0, 1),
            )

            @jax.jit
            def eval_batch(p, teacher, tokens):
                return full_model_eval_metrics(
                    model.apply({"params": p}, tokens),
                    teacher_model.apply({"params": teacher}, tokens), tokens,
                    temperature=2.0,
                )

            def evaluate(p):
                records = []
                for window in validation:
                    metrics = eval_batch(
                        p, teacher_params,
                        jax.device_put(window[None, :], batch_layout),
                    )
                    jax.block_until_ready(metrics)
                    records.append(_metric_record(metrics))
                return {
                    key: bool(all(record[key] for record in records))
                    if key == "finite"
                    else float(np.mean([record[key] for record in records]))
                    for key in records[0]
                }

            def save() -> None:
                store.save(
                    slot, {"params": params, "opt_state": opt_state},
                    contract=checkpoint_contract, step=row["completed_steps"],
                    metrics=row,
                )
                persist(upload=True)

            if not row["evaluations"]:
                row["evaluations"]["0"] = evaluate(params)
                save()
            print(
                f"exp089 seed={seed} arm={arm} resume_step={row['completed_steps']}",
                flush=True,
            )
            last_saved = row["completed_steps"]
            for zero_step in range(row["completed_steps"], TOTAL_STEPS):
                if zero_step % 32 == 0 and time.monotonic() >= deadline:
                    break
                index = int(deterministic_batch_indices(
                    zero_step, 1, len(train), DATA_SEED + seed
                )[0])
                batch = jax.device_put(train[index:index + 1], batch_layout)
                params, opt_state, metrics = train_step(
                    params, opt_state, teacher_params, batch
                )
                jax.block_until_ready(metrics)
                if not bool(metrics["grads_finite"]) or not np.isfinite(float(metrics["loss"])):
                    raise FloatingPointError(
                        f"non-finite seed={seed} arm={arm} step={zero_step + 1}; "
                        f"last durable step={last_saved}"
                    )
                step = zero_step + 1
                row["completed_steps"] = step
                row["tokens_seen"] = step * SEQUENCE_LENGTH
                if step in CHECKPOINTS:
                    row["evaluations"][str(step)] = evaluate(params)
                    row["training_metrics"][str(step)] = _metric_record(metrics)
                    print(
                        f"exp089 seed={seed} arm={arm} step={step} "
                        f"tokens={row['tokens_seen']} "
                        f"excess_nll={row['evaluations'][str(step)]['excess_nll']:.6f}",
                        flush=True,
                    )
                row["complete"] = step == TOTAL_STEPS
                if step % SAVE_EVERY == 0 or row["complete"]:
                    save()
                    last_saved = step
            if str(row["completed_steps"]) not in row["evaluations"]:
                row["evaluations"][str(row["completed_steps"])] = evaluate(params)
            row["complete"] = row["completed_steps"] == TOTAL_STEPS
            if row["completed_steps"] != last_saved:
                save()
            del params, opt_state, optimizer, initialized, train_step, eval_batch
            jax.clear_caches()
            gc.collect()
            if not row["complete"]:
                break

        result["aggregate"] = aggregate(result)
        result["status"] = "completed" if result["aggregate"]["complete"] else "deadline_partial"
        result["completed_at_utc"] = _now()
        persist(upload=True)
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus={result['status']}\nexperiment=EXP-089"
            f"\nduration_hours={result['duration_hours']:.3f}"
            f"\ntrajectories={result['aggregate']['completed_trajectories']}/4",
        )
        print(f"EXP089-{result['status'].upper()}\nsummary={summary_path.resolve()}")
        return result
    except BaseException as exc:
        result.update(
            status="failed", stage=stage, error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(), completed_at_utc=_now(),
        )
        persist(upload=True)
        _safe_notify(
            args.telegram,
            f"Extent TPU campaign\nstatus=failed\nexperiment=EXP-089"
            f"\nstage={stage}\nerror={type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
