"""EXP-093: foldable whole-model Mamba-gain calibration from ONPOLICY starts.

One Kaggle TPU invocation resumes small optimizer states from HF when needed.
No full-model weights are updated or uploaded by this experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import causal_cross_entropy
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import initialize_sharded_parameters
from extent.mixer_gain import effective_gains, scaled_mamba_parameters
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sharding import batch_sharding, create_v5e_mesh
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072, configured_contract
from scripts import m3q_sequential_recovery_campaign as sequential
from scripts import m3q_onpolicy_anchor_campaign as exp091
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp093-foldable-mixer-gains-v1"
HF_PREFIX = "experiments/exp093-foldable-mixer-gains"
SEEDS = (123, 456)
ARMS = ("GLOBAL", "PER-LAYER")
STEPS = 2048
SAVE_EVERY = 128
LENGTH = 256
TRAIN_WINDOWS = 2048
TRAIN_OFFSET = 7_340_032
VALIDATION_WINDOWS = 16
VALIDATION_OFFSET = 16_384
TEST_WINDOWS = 32
TEST_OFFSET = 0
LEARNING_RATE = 0.01


def contract_for(config):
    order = configured_contract(config)["replacement_order"]
    return {
        "protocol": PROTOCOL,
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_protocol": EXP072["PROTOCOL"],
        "model": json.loads(json.dumps(asdict(config))),
        "seeds": list(SEEDS), "arms": list(ARMS),
        "replacement_order": list(order),
        "steps_per_arm": STEPS, "sequence_length": LENGTH,
        "student_train_tokens": len(SEEDS) * len(ARMS) * STEPS * LENGTH,
        "train": {"split": "train", "offset": TRAIN_OFFSET, "windows": TRAIN_WINDOWS},
        "validation": {"split": "validation", "offset": VALIDATION_OFFSET,
                       "windows": VALIDATION_WINDOWS},
        "locked_test": {"split": "test", "offset": TEST_OFFSET, "windows": TEST_WINDOWS},
        "optimizer": {"name": "adam", "lr": LEARNING_RATE,
                      "coordinates": "FP32, zero-init"},
        "gain": "1 + 0.5*tanh(raw); scales Mamba out_proj kernel; copied Qwen and Mamba weights frozen",
        "objective": "true-token causal cross-entropy, same text/order for paired arms",
        "selection": "registered final step only; validation is diagnostic; test is final evaluation only",
        "gate": "PER-LAYER test NLL <= GLOBAL minus 0.05 and below unchanged start at both seeds",
    }


def aggregate(result):
    branches = result.get("branches", {})
    complete = all(
        branches.get(str(seed), {}).get(arm, {}).get("complete", False)
        for seed in SEEDS for arm in ARMS
    )
    wins = {}
    for seed in SEEDS:
        pair = branches.get(str(seed), {})
        if all(pair.get(arm, {}).get("complete", False) for arm in ARMS):
            global_nll = pair["GLOBAL"]["locked_test_nll"]
            local_nll = pair["PER-LAYER"]["locked_test_nll"]
            start_nll = pair["GLOBAL"]["start_test_nll"]
            wins[str(seed)] = bool(local_nll <= global_nll - 0.05 and local_nll < start_nll)
    return {"complete": complete, "scientific_gate_passed": bool(complete and all(wins.values())),
            "seed_wins": wins}


def render_summary(result):
    lines = ["# EXP-093 foldable Mamba-gain calibration", "",
             f"- Status: `{result['status']}`",
             f"- Duration: `{result.get('duration_hours', 0):.3f}` hours",
             f"- Scientific gate: `{result['aggregate']['scientific_gate_passed']}`", "",
             "| Seed | Arm | Step | Start test NLL | Final test NLL |",
             "|---:|---|---:|---:|---:|"]
    for seed, pair in result.get("branches", {}).items():
        for arm, row in pair.items():
            lines.append(f"| {seed} | {arm} | {row['step']} | "
                         f"{row.get('start_test_nll', float('nan')):.6f} | "
                         f"{row.get('locked_test_nll', float('nan')):.6f} |")
    lines += ["", "The locked test is evaluated only at the registered final endpoint. "
              "This is a gain-calibration test, not full-model recovery or an initializer comparison."]
    return "\n".join(lines) + "\n"


def _restore_optimizer(tx, raw, saved):
    state = tx.init(raw)
    if saved is None:
        return state
    adam = state[0]
    return (type(adam)(jnp.asarray(saved["count"], adam.count.dtype),
                       jnp.asarray(saved["mu"], adam.mu.dtype),
                       jnp.asarray(saved["nu"], adam.nu.dtype)), *state[1:])


def _optimizer_record(state):
    adam = state[0]
    return {"count": int(adam.count), "mu": np.asarray(adam.mu).tolist(),
            "nu": np.asarray(adam.nu).tolist()}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp093-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp093-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.25)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall_hours <= 8.5:
        raise ValueError("max-wall-hours must be between 1 and 8.5")
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 20 * 60
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "extent-m3q-mixer-gains.json"
    summary_path = output / "extent-m3q-mixer-gains-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")
    stage = "preflight"
    _safe_notify(args.telegram, f"Extent EXP-093 started host={socket.gethostname()}")
    try:
        exp091.preflight_source_endpoints()
        config, _ = load_config(
            Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
        )
        contract = contract_for(config)
        if not result_path.exists():
            restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "contract": contract, "status": "running", "branches": {},
        }
        if result.get("status") == "failed" and "contract" not in result:
            result = {"contract": contract, "status": "running", "branches": {}}
        if result["contract"] != contract:
            raise ValueError("EXP-093 resume contract mismatch")
        result.pop("traceback", None)
        result["status"] = "running"
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if result.get("git_revision", revision) != revision:
            raise ValueError("EXP-093 code revision changed during resume")
        result["git_revision"] = revision

        def persist():
            result["duration_hours"] = (time.monotonic() - started) / 3600
            result["aggregate"] = aggregate(result)
            write_json_atomic(result_path, result)
            summary_path.write_text(render_summary(result), encoding="utf-8")
            for path, name in ((result_path, "latest.json"),
                               (summary_path, "latest-summary.md")):
                for attempt in range(3):
                    try:
                        upload_artifact(path, f"{HF_PREFIX}/{name}", hub,
                                        commit_message="EXP-093 resumable progress")
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep((2, 5)[attempt])

        stage = "tpu-and-data"
        devices = list(jax.devices())
        if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
            raise ValueError("EXP-093 requires one TPU v5e-8")
        result["devices"] = [str(d) for d in devices]
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        token_sets = {}
        for name, split, offset, windows in (
            ("train", "train", TRAIN_OFFSET, TRAIN_WINDOWS),
            ("validation", "validation", VALIDATION_OFFSET, VALIDATION_WINDOWS),
            ("locked_test", "test", TEST_OFFSET, TEST_WINDOWS),
        ):
            token_sets[name] = load_wikitext2_tokens(
                windows * LENGTH, args.dataset_cache_dir,
                tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision,
                token_offset=offset, dataset_split=split,
                dataset_config="wikitext-103-raw-v1",
            ).reshape(windows, LENGTH)
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest()
                  for key, value in token_sets.items()}
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-093 data changed during resume")
        result["data_sha256"] = hashes

        stage = "load-model"
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16",
            remat_policy="full",
        )
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        teacher_init = initialize_sharded_parameters(
            teacher, jax.random.key(930), init_tokens, mesh
        )
        teacher_params, report = stream_teacher_qwen_weights(
            teacher_init.params, reader, source
        )
        result["teacher_tensor_count"] = report.tensor_count
        del teacher_init
        model = HybridForCausalLM(config)
        model_init = initialize_sharded_parameters(
            model, jax.random.key(931), init_tokens, mesh
        )
        abstract, layout = model_init.abstract_params, model_init.layout
        del model_init
        seq_contract = configured_contract(config)
        order = tuple(seq_contract["replacement_order"])
        if len(order) != 24 or set(order) != set(range(config.num_layers)) - set(config.attention_layer_indices):
            raise ValueError("EXP-093 replacement contract differs from production 1.7B layout")
        base_store = CampaignCheckpointStore(
            Path(args.state_dir) / "exp069", f"{EXP069_PREFIX}/checkpoints", hub
        )
        seq_store = CampaignCheckpointStore(
            Path(args.state_dir) / "exp072", f"{EXP072['HF_PREFIX']}/checkpoints", hub
        )
        tx = optax.adam(LEARNING_RATE)

        # Keep the arm static for JAX; each arm has a different raw-coordinate shape.
        def make_step(arm):
            @jax.jit
            def step(base, raw, state, tokens):
                def objective(coordinates):
                    gains = effective_gains(coordinates, arm, len(order))
                    candidate = scaled_mamba_parameters(base, order, gains)
                    logits = model.apply({"params": candidate}, tokens)
                    return causal_cross_entropy(logits, tokens)
                loss, grad = jax.value_and_grad(objective)(raw)
                updates, new_state = tx.update(grad, state, raw)
                return optax.apply_updates(raw, updates), new_state, loss
            return step

        # Each arm has a different raw-coordinate shape, so compile once per arm.
        eval_runners = {arm: jax.jit(lambda b, r, t, arm=arm: causal_cross_entropy(
                model.apply({"params": scaled_mamba_parameters(
                    b, order, effective_gains(r, arm, len(order)))}, t), t
            )) for arm in ARMS}

        def evaluate(base, raw, arm, name):
            values = []
            run = eval_runners[arm]
            for window in token_sets[name]:
                loss = run(base, raw, jax.device_put(window[None, :], batch_layout))
                values.append(float(jax.block_until_ready(loss)))
            return float(np.mean(values))

        for seed in SEEDS:
            if all(result["branches"].get(str(seed), {}).get(arm, {}).get("complete") for arm in ARMS):
                continue
            stage = f"compose-seed-{seed}"
            endpoints, endpoint_hashes = {}, {}
            for depth, layer in enumerate(order, 1):
                base_slot = f"prep/seed-{seed}/layer-{layer}"
                base_contract = dict(seq_contract["source_contract"], seed=seed,
                                     layer=layer, kind="prepared_mamba")
                base_meta = base_store.metadata(base_slot, base_contract)
                if base_meta is None:
                    raise FileNotFoundError(f"missing EXP-069 source: {base_slot}")
                seq_slot = f"seed-{seed}/ONPOLICY/layer-{layer}"
                seq_c = sequential.endpoint_contract(
                    seq_contract, seed, "ONPOLICY", layer, depth,
                    base_meta["checkpoint_sha256"],
                )
                restored = seq_store.restore(seq_slot, seq_c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP-072 source: {seq_slot}")
                payload, seq_meta = restored
                endpoints[layer] = payload["params"]
                endpoint_hashes[str(layer)] = seq_meta["checkpoint_sha256"]
            base = compose_parameters(teacher_params, endpoints, order, abstract, layout)
            del endpoints
            gc.collect()
            for arm in ARMS:
                stage = f"seed-{seed}-{arm}"
                row = result["branches"].setdefault(str(seed), {}).setdefault(arm, {})
                if row.get("complete"):
                    continue
                if row.get("endpoint_hashes", endpoint_hashes) != endpoint_hashes:
                    raise ValueError("EXP-072 source hashes changed on resume")
                row["endpoint_hashes"] = endpoint_hashes
                raw = jnp.asarray(row.get("raw", [0.0] * (1 if arm == "GLOBAL" else len(order))),
                                  dtype=jnp.float32)
                state = _restore_optimizer(tx, raw, row.get("optimizer"))
                start_step = int(row.get("step", 0))
                if start_step == 0 and "start_validation_nll" not in row:
                    row["start_validation_nll"] = evaluate(base, raw, arm, "validation")
                    row.update(step=0, raw=np.asarray(raw).tolist(),
                               optimizer=_optimizer_record(state))
                    persist()
                train_step = make_step(arm)
                for step_number in range(start_step + 1, STEPS + 1):
                    if time.monotonic() >= deadline:
                        row.update(step=step_number - 1, raw=np.asarray(raw).tolist(),
                                   optimizer=_optimizer_record(state))
                        result["status"] = "deadline_partial"
                        persist()
                        _safe_notify(args.telegram, f"Extent EXP-093 deadline_partial "
                                     f"seed={seed} arm={arm} step={step_number - 1}")
                        return result
                    window = token_sets["train"][(step_number - 1) % TRAIN_WINDOWS]
                    raw, state, loss = train_step(
                        base, raw, state, jax.device_put(window[None, :], batch_layout)
                    )
                    jax.block_until_ready(loss)
                    if not np.isfinite(float(loss)) or not bool(jnp.all(jnp.isfinite(raw))):
                        raise FloatingPointError(f"nonfinite EXP-093 step {step_number}")
                    if step_number % SAVE_EVERY == 0 or step_number == STEPS:
                        row.update(step=step_number, raw=np.asarray(raw).tolist(),
                                   optimizer=_optimizer_record(state),
                                   latest_train_nll=float(loss))
                        row.setdefault("validation", {})[str(step_number)] = evaluate(
                            base, raw, arm, "validation"
                        )
                        persist()
                        print(f"exp093 seed={seed} arm={arm} step={step_number} "
                              f"train_nll={float(loss):.6f} "
                              f"val_nll={row['validation'][str(step_number)]:.6f}", flush=True)
                row["locked_test_nll"] = evaluate(base, raw, arm, "locked_test")
                row["start_test_nll"] = evaluate(
                    base, jnp.zeros_like(raw), arm, "locked_test"
                )
                row["effective_gains"] = np.asarray(
                    effective_gains(raw, arm, len(order))
                ).tolist()
                row["complete"] = True
                persist()
                del raw, state, train_step
                gc.collect()
            del base
            gc.collect()
        result["status"] = "completed"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["checkpoint_events"] = base_store.events + seq_store.events
        persist()
        _safe_notify(args.telegram, "Extent EXP-093 completed "
                     f"gate={result['aggregate']['scientific_gate_passed']}")
        print(f"EXP093-COMPLETED summary={summary_path.resolve()}", flush=True)
        return result
    except BaseException as exc:
        failure = locals().get("result")
        if failure is not None:
            failure.update(status="failed", stage=stage, error_type=type(exc).__name__,
                           error=str(exc), traceback=traceback.format_exc(),
                           completed_at_utc=datetime.now(timezone.utc).isoformat())
            try:
                persist()
            except Exception:
                write_json_atomic(result_path, failure)
        else:
            write_json_atomic(result_path, {
                "status": "failed", "stage": stage,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        _safe_notify(args.telegram, f"Extent EXP-093 failed stage={stage} "
                     f"error={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
