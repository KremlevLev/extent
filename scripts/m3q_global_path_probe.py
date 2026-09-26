"""EXP-092: calibrate a full-model path from ONPOLICY start to EXP-091 endpoints.

This is a go/no-go diagnostic before another long TPU training campaign. It
does not optimize on the locked test text or create new trained checkpoints.
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
import shutil

import jax
import numpy as np

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import full_model_eval_metrics
from extent.global_interpolation import interpolate_parameters, select_alpha
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifact
from extent.initialization import initialize_sharded_parameters
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.sharding import batch_sharding, create_v5e_mesh
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072
from scripts import m3q_sequential_recovery_campaign as sequential
from scripts import m3q_onpolicy_anchor_campaign as exp091
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint, _metric_record
from scripts.qwen_extended_horizon_campaign import _safe_notify


PROTOCOL = "exp092-global-onpolicy-path-v1"
HF_PREFIX = "experiments/exp092-global-onpolicy-path"
ARMS_BY_SEED = {123: ("ONPOLICY-PLAIN", "ONPOLICY-ANCHOR"),
                456: ("ONPOLICY-ANCHOR",)}
ALPHAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
MIN_CALIBRATION_GAIN = 0.01
CALIBRATION_SPLIT = "validation"
CALIBRATION_OFFSET = 8192
CALIBRATION_WINDOWS = 16
LOCKED_SPLIT = "test"
LOCKED_OFFSET = 0
LOCKED_WINDOWS = 32
SEQUENCE_LENGTH = 256


def contract_for(config, exp091_result):
    return {
        "protocol": PROTOCOL,
        "source": exp091_result["contract"]["source"],
        "source_protocol": exp091_result["contract"]["protocol"],
        "source_data_sha256": exp091_result["data_sha256"],
        "model": exp091_result["contract"]["model"],
        "arms_by_seed": {str(k): list(v) for k, v in ARMS_BY_SEED.items()},
        "alphas": list(ALPHAS),
        "min_calibration_gain": MIN_CALIBRATION_GAIN,
        "calibration": {"split": CALIBRATION_SPLIT, "offset": CALIBRATION_OFFSET,
                        "windows": CALIBRATION_WINDOWS, "length": SEQUENCE_LENGTH},
        "locked_test": {"split": LOCKED_SPLIT, "offset": LOCKED_OFFSET,
                         "windows": LOCKED_WINDOWS, "length": SEQUENCE_LENGTH},
        "selection": "minimum calibration mean NLL; ties prefer smaller alpha; "
                     "accept only if gain >= 0.01; locked test never selects alpha",
        "scope": "whole-model parameter path, not a training result or initializer win",
    }


def aggregate(result):
    rows = result.get("branches", {})
    complete = all(
        arm in rows.get(str(seed), {}) for seed, arms in ARMS_BY_SEED.items()
        for arm in arms
    )
    anchor_rows = [rows.get(str(seed), {}).get("ONPOLICY-ANCHOR")
                   for seed in ARMS_BY_SEED]
    passed = bool(complete and all(
        row is not None and row["selected_alpha"] > 0
        and row["selected_locked"]["student_nll"]
        < row["start_locked"]["student_nll"] - MIN_CALIBRATION_GAIN
        for row in anchor_rows
    ))
    return {
        "complete": complete,
        "completed_branches": sum(len(v) for v in rows.values()),
        "scientific_gate_passed": passed,
        "gate_definition": (
            "Calibrated alpha must be nonzero and reduce locked-test NLL by "
            ">0.01 versus alpha zero for ONPOLICY-ANCHOR at both seeds. "
            "The seed-123 PLAIN path is contextual, not part of the gate."
        ),
    }


def render_summary(result):
    agg = result["aggregate"]
    lines = ["# EXP-092 whole-model ONPOLICY parameter-path probe", "",
             f"- Status: `{result['status']}`",
             f"- Duration: `{result['duration_hours']:.3f}` hours",
             f"- Completed branches: `{agg['completed_branches']}/3`",
             f"- Scientific gate: `{agg['scientific_gate_passed']}`", "",
             "| Seed | Trained endpoint | Selected α | Start locked NLL | "
             "Selected locked NLL | Raw endpoint locked NLL |",
             "|---:|---|---:|---:|---:|---:|"]
    for seed, branches in result.get("branches", {}).items():
        for arm, row in branches.items():
            lines.append(
                f"| {seed} | {arm} | {row['selected_alpha']:.5f} | "
                f"{row['start_locked']['student_nll']:.6f} | "
                f"{row['selected_locked']['student_nll']:.6f} | "
                f"{row['raw_locked']['student_nll']:.6f} |"
            )
    lines += ["", "Alpha is selected only on fresh WikiText-103 validation "
              "windows; the locked WikiText-103 test split is evaluation-only. "
              "A positive result would motivate a separate staged training "
              "campaign, not establish a recovered 1.7B/14B model."]
    return "\n".join(lines) + "\n"


def required_source_manifests():
    required = set()
    for seed, arms in ARMS_BY_SEED.items():
        for arm in arms:
            required.add(
                f"{exp091.campaign.HF_PREFIX}/checkpoints/seed-{seed}/"
                f"{arm.lower()}/checkpoint.json"
            )
    return required


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default="/dev/shm/extent-exp092-state")
    parser.add_argument("--qwen-cache-dir", default="/dev/shm/qwen3-1.7b-exp092-weights")
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
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.state_dir).free < 24 * 1024**3:
        raise ValueError("EXP-092 state-dir requires at least 24 GiB free")
    result_path = output / "extent-m3q-global-path.json"
    summary_path = output / "extent-m3q-global-path-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO are required")

    stage = "source-preflight"
    _safe_notify(args.telegram, f"Extent EXP-092 started host={socket.gethostname()}")
    try:
        exp091.configure()
        exp091.preflight_source_endpoints()
        from huggingface_hub import HfApi

        available = set(HfApi(token=hub.token).list_repo_files(
            repo_id=hub.repo_id, repo_type=hub.repo_type,
            revision=hub.revision, token=hub.token,
        ))
        missing = sorted(required_source_manifests() - available)
        if missing:
            raise FileNotFoundError(f"missing EXP-091 endpoint: {missing[0]}")
        exp091_result_path = output / "exp092-source-exp091.json"
        if not restore_artifact(
            exp091_result_path, f"{exp091.campaign.HF_PREFIX}/latest.json", hub
        ):
            raise FileNotFoundError("missing EXP-091 latest.json")
    except BaseException as exc:
        _safe_notify(args.telegram, f"Extent EXP-092 failed stage={stage} "
                     f"error={type(exc).__name__}: {exc}")
        raise
    source_result = json.loads(exp091_result_path.read_text(encoding="utf-8"))
    if source_result["contract"]["protocol"] != exp091.campaign.PROTOCOL:
        raise ValueError("EXP-091 source protocol mismatch")
    config, _ = load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )
    if json.loads(json.dumps(asdict(config))) != source_result["contract"]["model"]:
        raise ValueError("EXP-091 source model configuration mismatch")
    contract = contract_for(config, source_result)
    if not result_path.exists():
        restore_artifact(result_path, f"{HF_PREFIX}/latest.json", hub)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "contract": contract, "status": "running", "branches": {},
    }
    if result["contract"] != contract:
        raise ValueError("EXP-092 resume contract mismatch")
    result.pop("traceback", None)
    result.update(
        status="running",
        source_artifact_sha256=hashlib.sha256(exp091_result_path.read_bytes()).hexdigest(),
        git_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    )
    store = CampaignCheckpointStore(
        Path(args.state_dir) / "exp091", f"{exp091.campaign.HF_PREFIX}/checkpoints", hub
    )
    base_store = CampaignCheckpointStore(
        Path(args.state_dir) / "exp069", f"{EXP069_PREFIX}/checkpoints", hub
    )
    seq_store = CampaignCheckpointStore(
        Path(args.state_dir) / "exp072", f"{EXP072['HF_PREFIX']}/checkpoints", hub
    )

    def persist(upload=True):
        result["duration_hours"] = (time.monotonic() - started) / 3600
        result["aggregate"] = aggregate(result)
        result["checkpoint_events"] = store.events + base_store.events + seq_store.events
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(result), encoding="utf-8")
        if upload:
            for local, remote in ((result_path, "latest.json"),
                                  (summary_path, "latest-summary.md")):
                try:
                    upload_artifact(local, f"{HF_PREFIX}/{remote}", hub,
                                    commit_message="EXP-092 progress")
                except Exception as exc:
                    print(f"exp092_upload=FAILED file={remote} type={type(exc).__name__}", flush=True)

    try:
        stage = "tpu-and-data"
        devices = list(jax.devices())
        if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
            raise ValueError("EXP-092 requires one TPU v5e-8")
        result["devices"] = [str(device) for device in devices]
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        token_sets = {}
        for name, split, offset, windows in (
            ("calibration", CALIBRATION_SPLIT, CALIBRATION_OFFSET, CALIBRATION_WINDOWS),
            ("locked_test", LOCKED_SPLIT, LOCKED_OFFSET, LOCKED_WINDOWS),
        ):
            token_sets[name] = load_wikitext2_tokens(
                windows * SEQUENCE_LENGTH, args.dataset_cache_dir,
                tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision,
                token_offset=offset, dataset_split=split,
                dataset_config="wikitext-103-raw-v1",
            ).reshape(windows, SEQUENCE_LENGTH)
        hashes = {name: hashlib.sha256(value.tobytes()).hexdigest()
                  for name, value in token_sets.items()}
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("EXP-092 dataset content changed during resume")
        result["data_sha256"] = hashes

        stage = "teacher-and-model"
        source = teacher_config_from_spec(
            QWEN3_1_7B_BASE, param_dtype="bfloat16", compute_dtype="bfloat16",
            remat_policy="full",
        )
        reader = _ensure_checkpoint(Path(args.qwen_cache_dir))
        teacher = Qwen3ForCausalLM(source)
        teacher_init = initialize_sharded_parameters(
            teacher, jax.random.key(920), init_tokens, mesh
        )
        teacher_params, teacher_report = stream_teacher_qwen_weights(
            teacher_init.params, reader, source
        )
        result["teacher_tensor_count"] = teacher_report.tensor_count
        del teacher_init
        model = HybridForCausalLM(config)
        model_init = initialize_sharded_parameters(
            model, jax.random.key(921), init_tokens, mesh
        )
        abstract, layout = model_init.abstract_params, model_init.layout
        del model_init

        @jax.jit
        def eval_batch(params, teacher_weights, tokens):
            student_logits = model.apply({"params": params}, tokens)
            teacher_logits = teacher.apply({"params": teacher_weights}, tokens)
            return full_model_eval_metrics(
                student_logits, teacher_logits, tokens, temperature=2.0
            )

        def evaluate(params, name):
            records = []
            for window in token_sets[name]:
                metrics = eval_batch(
                    params, teacher_params,
                    jax.device_put(window[None, :], batch_layout),
                )
                jax.block_until_ready(metrics)
                records.append(_metric_record(metrics))
            result_metrics = {
                key: bool(all(record[key] for record in records)) if key == "finite"
                else float(np.mean([record[key] for record in records]))
                for key in records[0]
            }
            result_metrics["window_nll"] = [record["student_nll"] for record in records]
            return result_metrics

        seq_contract = source_result["contract"]["warm_start"]["source_contract"]
        order = tuple(seq_contract["replacement_order"])
        if set(order) != set(range(config.num_layers)) - set(config.attention_layer_indices):
            raise ValueError("EXP-091 and production replacement layouts differ")
        for seed, arms in ARMS_BY_SEED.items():
            for arm in arms:
                if arm in result["branches"].get(str(seed), {}):
                    continue
                if time.monotonic() + 15 * 60 >= deadline:
                    result["status"] = "deadline_partial"
                    persist()
                    _safe_notify(args.telegram, "Extent EXP-092 deadline_partial "
                                 f"branches={result['aggregate']['completed_branches']}/3")
                    return result
                stage = f"seed-{seed}-{arm}"
                print(f"exp092 seed={seed} arm={arm} START", flush=True)
                endpoints, endpoint_hashes = {}, {}
                for depth, layer in enumerate(order, 1):
                    base_slot = f"prep/seed-{seed}/layer-{layer}"
                    base_contract = dict(
                        seq_contract["source_contract"], seed=seed, layer=layer,
                        kind="prepared_mamba",
                    )
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
                start_params = compose_parameters(
                    teacher_params, endpoints, order, abstract, layout
                )
                del endpoints
                slot = f"seed-{seed}/{arm.lower()}"
                candidate_contract = dict(
                    source_result["contract"], kind="full_model_trajectory",
                    seed=seed, arm=arm,
                    data_sha256=source_result["data_sha256"],
                )
                restored = store.restore(slot, candidate_contract)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP-091 complete endpoint: {slot}")
                payload, candidate_meta = restored
                if not candidate_meta["metrics"].get("complete"):
                    raise ValueError(f"EXP-091 endpoint is not complete: {slot}")
                if candidate_meta["metrics"].get("warm_start_hashes") != endpoint_hashes:
                    raise ValueError(f"EXP-091 endpoint source hashes changed: {slot}")
                trained_params = jax.tree.map(jax.device_put, payload["params"], layout)
                del payload, restored
                calibration = {}
                for alpha in ALPHAS:
                    params = interpolate_parameters(start_params, trained_params, alpha)
                    calibration[str(alpha)] = evaluate(params, "calibration")
                    print(
                        f"exp092 seed={seed} arm={arm} alpha={alpha:g} "
                        f"cal_nll={calibration[str(alpha)]['student_nll']:.6f}",
                        flush=True,
                    )
                    del params
                selected = select_alpha(
                    {alpha: calibration[str(alpha)]["student_nll"] for alpha in ALPHAS},
                    min_gain=MIN_CALIBRATION_GAIN,
                )
                locked = {}
                for alpha in sorted({0.0, selected, 1.0}):
                    params = interpolate_parameters(start_params, trained_params, alpha)
                    locked[str(alpha)] = evaluate(params, "locked_test")
                    del params
                row = {
                    "source_endpoint_sha256": candidate_meta["checkpoint_sha256"],
                    "warm_start_hashes": endpoint_hashes,
                    "source_steps": candidate_meta["step"],
                    "selected_alpha": selected,
                    "calibration": calibration,
                    "locked_test": locked,
                    "start_locked": locked["0.0"],
                    "selected_locked": locked[str(selected)],
                    "raw_locked": locked["1.0"],
                }
                result["branches"].setdefault(str(seed), {})[arm] = row
                persist()
                del start_params, trained_params, row
                gc.collect()
        result["status"] = "completed"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        _safe_notify(args.telegram, "Extent EXP-092 completed "
                     f"gate={result['aggregate']['scientific_gate_passed']}")
        print(f"EXP092-COMPLETED summary={summary_path.resolve()}", flush=True)
        return result
    except BaseException as exc:
        result.update(
            status="failed", stage=stage, error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        persist()
        _safe_notify(args.telegram, f"Extent EXP-092 failed stage={stage} "
                     f"error={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
