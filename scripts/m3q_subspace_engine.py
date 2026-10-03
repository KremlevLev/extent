"""Resumable, paired full-depth low-rank recovery campaigns (094--096).

Base model weights are frozen. Only FP32 correction coordinates and Adam states
are checkpointed. Shared controls are deterministic consistency checks, not
additional independent seeds. These are recovery tests, not MLA experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
import optax

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens, load_pg19_tokens, PG19_REVISION, WIKITEXT_REVISION
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.composition_diagnostics import compose_parameters
from extent.config import load_config
from extent.full_model_distillation import causal_cross_entropy, forward_kl, hidden_delta_relative_mse
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifacts_together
from extent.initialization import initialize_sharded_parameters
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.recovery_subspace import initialize_corrections, apply_corrections, coordinate_count, coordinates_finite
from extent.sharding import batch_sharding, create_v5e_mesh
from extent.weight_mapping import stream_teacher_qwen_weights
from scripts.m3q_allocation_campaign import HF_PREFIX as EXP069_PREFIX
from scripts.m3q_full_depth_sequential_confirmation import OVERRIDES as EXP072, configured_contract
from scripts import m3q_sequential_recovery_campaign as sequential
from scripts import m3q_onpolicy_anchor_campaign as exp091
from scripts.qwen17_full_model_distill_campaign import _ensure_checkpoint
from scripts.qwen_extended_horizon_campaign import _safe_notify

SEEDS = (123, 456)
HORIZONS = (2048, 4096, 8192)
LENGTH = 256
TRAIN_WINDOWS = 8192
DATA = (("train", "train", 8_388_608, TRAIN_WINDOWS),
        ("validation", "validation", 49_152, 32),
        ("locked_test", "test", 32_768, 64))
LR = 3e-4
# Frozen scientific implementation digest for the completed legacy protocols.
# New hooks below do not change their objective, optimizer or step function.
LEGACY_ENGINE_SHA256 = "c3e17745b12e2d1bbab225ead5db3db705343327c1caa5241e10a8512dfb36e6"
# New schedule/evaluation extensions are opt-in. EXP-097 keeps its unchanged
# diagnostic factory and its completed/resumable scientific source contract.
EXP097_ENGINE_SHA256 = "dbd4803090b19c5d35634e332ddeceb2619dd904b9849d49326592dfe3213e9c"

@dataclass(frozen=True)
class Arm:
    name: str
    subspace: str
    rank: int
    objective: str = "ce"
    protected: bool = False

@dataclass(frozen=True)
class Campaign:
    number: int
    name: str
    arms: tuple[Arm, ...]
    primary: str
    control: str

    @property
    def prefix(self):
        return f"experiments/exp{self.number:03d}-{self.name}"

    @property
    def stem(self):
        return f"extent-m3q-{self.name}"


@dataclass(frozen=True)
class Schedule:
    """Per-invocation immutable settings; never mutate the legacy module globals."""
    horizons: tuple[int, ...]
    train_windows: int
    data: tuple[tuple[str, str, int, int], ...]
    pg19_test_offset: int
    pg19_test_windows: int = 64
    max_wall_hours: float = 8.0
    reserve_minutes: int = 45

    def __post_init__(self):
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons or self.horizons[0] < 1:
            raise ValueError("horizons must be strictly increasing positive steps")
        rows = {name: (split, offset, windows) for name, split, offset, windows in self.data}
        if len(rows) != len(self.data) or set(rows) != {"train", "validation", "locked_test"}:
            raise ValueError("schedule must contain unique train/validation/locked_test rows")
        if rows["train"][2] != self.train_windows or self.train_windows < self.horizons[-1]:
            raise ValueError("new campaigns require enough unique train windows for the final horizon")
        if any(offset < 0 or windows < 1 for _, offset, windows in rows.values()):
            raise ValueError("invalid data range")
        if self.pg19_test_offset < 0 or self.pg19_test_windows < 1 or not 0 < self.reserve_minutes < self.max_wall_hours * 60:
            raise ValueError("invalid evaluation range or saving reserve")


def terminal_result(spec, result):
    return all(result.get("branches", {}).get(str(s), {}).get(a.name, {}).get("complete")
               or result.get("branches", {}).get(str(s), {}).get(a.name, {}).get("failed")
               for s in SEEDS for a in spec.arms)

def contract_for(spec, config):
    root = Path(__file__).resolve().parents[1]
    return {
        "protocol": f"exp{spec.number:03d}-{spec.name}-v1",
        "campaign": asdict(spec), "model": json.loads(json.dumps(asdict(config))),
        "source": f"{QWEN3_1_7B_BASE.repo_id}@{QWEN3_1_7B_BASE.revision}",
        "source_protocol": EXP072["PROTOCOL"],
        "replacement_order": list(configured_contract(config)["replacement_order"]),
        "seeds": list(SEEDS), "horizons": list(HORIZONS), "sequence_length": LENGTH,
        "data": [list(row) for row in DATA],
        "student_training_tokens": len(SEEDS) * len(spec.arms) * HORIZONS[-1] * LENGTH,
        "optimizer": {"name": "adam", "learning_rate": LR, "clip_global_norm": 1.0,
                      "coordinates_and_moments": "float32", "base_weights": "frozen BF16"},
        "objectives": {"ce": "true-token CE", "teacher": "KL(T=2)+0.1CE",
                       "self": "CE+0.1KL(T=2) to unchanged assembled hybrid",
                       "delta_then_teacher": "steps 1..2048: decoder-delta relative MSE +0.1CE; later KL(T=2)+0.1CE"},
        "delta_layers": "replaced layers >0 only; teacher/student see their own upstream streams",
        "protected": "in_proj dt/raw_a/trap/angle projection columns unchanged; B/C/x/z may adapt",
        "selection": "no early stopping/model selection; validation diagnostic; locked test only at8192",
        "gate": "registered primary beats control by >=0.1 test NLL and unchanged start at BOTH seeds",
        "comparison": "equal student tokens, NOT equal parameters/FLOPs; repeated rank8 CE controls are not independent evidence",
        "implementation_sha256": {p: ((LEGACY_ENGINE_SHA256 if spec.number in (94, 95, 96) else EXP097_ENGINE_SHA256)
            if spec.number in (94, 95, 96, 97) and p == "scripts/m3q_subspace_engine.py"
            else hashlib.sha256((root / p).read_bytes()).hexdigest())
            for p in ("scripts/m3q_subspace_engine.py", "extent/recovery_subspace.py",
                      "extent/full_model_distillation.py")},
    }

def coordinate_plan(config, arm, layer_count):
    inner = int(config.hidden_size * config.mamba.expand)
    heads = inner // config.mamba.head_dim
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    packed = active + 3 * heads + int(config.mamba.d_state * config.mamba.rope_fraction) // 2
    if arm.subspace == "HEAD-GAIN":
        count = layer_count * heads
    elif arm.subspace == "MLP-READOUT-LORA":
        count = layer_count * arm.rank * (config.intermediate_size + config.hidden_size)
    else:
        count = layer_count * arm.rank * (inner + config.hidden_size)
        if arm.subspace == "INOUT-LORA":
            count += layer_count * arm.rank * (config.hidden_size + packed)
    effective = count - (layer_count * arm.rank * (packed - active) if arm.protected else 0)
    return {"allocated_coordinates": count, "effective_coordinates": effective,
            "replicated_adapter_adam_bytes_per_device": count * 4 * 3,
            "adapter_gradient_bytes_per_device": count * 4,
            "excludes": "frozen model weights, activations and compiler temporaries"}

def needs_work(row, target, final_target=None):
    final_target = HORIZONS[-1] if final_target is None else final_target
    return not row.get("failed") and (row.get("step", 0) < target
            or target == final_target and not row.get("complete"))

def aggregate(spec, result):
    branches = result.get("branches", {})
    complete = all(branches.get(str(seed), {}).get(arm.name, {}).get("complete", False)
                   for seed in SEEDS for arm in spec.arms)
    wins = {}
    for seed in SEEDS:
        rows = branches.get(str(seed), {})
        a, b = rows.get(spec.primary, {}), rows.get(spec.control, {})
        if a.get("complete") and b.get("complete"):
            wins[str(seed)] = bool(a["locked_test_nll"] <= b["locked_test_nll"] - 0.1
                                   and a["locked_test_nll"] < a["start_test_nll"])
    return {"complete": complete, "seed_wins": wins,
            "scientific_gate_passed": bool(complete and len(wins) == len(SEEDS) and all(wins.values()))}

def render_summary(spec, result):
    lines = [f"# EXP-{spec.number:03d}: {spec.name}", "",
             f"Status: {result['status']}; gate: {result['aggregate']['scientific_gate_passed']}",
             f"Cumulative invocation time: {result.get('cumulative_hours', 0):.3f} h; pending HF: {result.get('remote_sync_pending', False)}", "",
             "| Seed | Method | Step | Coordinates | Start val NLL | Latest val NLL | Start test NLL | Final test NLL | State |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.6f}"
    for seed, rows in result.get("branches", {}).items():
        for name, row in rows.items():
            validations = row.get("validation", {})
            last_val = validations[max(validations, key=int)] if validations else None
            state = "failed" if row.get("failed") else "complete" if row.get("complete") else "partial"
            lines.append(f"| {seed} | {name} | {row.get('step', 0)} | {row.get('coordinates', 0):,} | "
                         f"{fmt(row.get('start_validation_nll'))} | {fmt(last_val)} | "
                         f"{fmt(row.get('start_test_nll'))} | {fmt(row.get('locked_test_nll'))} | {state} |")
    lines += ["", f"Primary: {spec.primary} vs {spec.control}. Final endpoint only; two paired source seeds.",
              "Frozen Qwen3-1.7B hybrid, 24 Mamba + 4 GQA. No claim of full recovery or MLA quality."]
    if "original_teacher_test_nll" in result:
        lines += [f"Original Qwen locked-test NLL: {fmt(result['original_teacher_test_nll'])}."]
    if "original_teacher_pg19_nll" in result:
        lines += [f"Original Qwen PG-19 test NLL: {fmt(result['original_teacher_pg19_nll'])}.", "",
            "| Seed | Arm | Start PG-19 NLL | Final PG-19 NLL |", "|---|---|---:|---:|"]
        for seed, rows in result.get("branches", {}).items():
            for name, row in rows.items():
                lines.append(f"| {seed} | {name} | {fmt(row.get('start_pg19_nll'))} | {fmt(row.get('pg19_test_nll'))} |")
        lines += ["", f"Cross-domain gate: {result['aggregate'].get('cross_domain_gate_passed', False)}."]
    if any("latest_health" in row for rows in result.get("branches", {}).values() for row in rows.values()):
        lines += ["", "## Numerical health (last attempted training step)", "",
            "| Seed | Arm | Forward finite | Gradients finite | Naive norm finite | Safe proposal finite | log10 norm | Norm-only events | Failure step |",
            "|---|---|---|---|---|---|---:|---:|---:|"]
        for seed, rows in result.get("branches", {}).items():
            for name, row in rows.items():
                h = row.get("latest_health", {})
                lines.append(f"| {seed} | {name} | {h.get('forward_finite')} | {h.get('gradients_finite')} | "
                    f"{h.get('naive_norm_finite')} | {h.get('safe_proposal_finite')} | {fmt(h.get('log10_grad_norm'))} | "
                    f"{row.get('norm_only_overflow_steps', 0)} | {row.get('failure_diagnostics', {}).get('attempted_step', '—')} |")
        lines += ["", "The displayed scalar norm may be saturated; log10 norm is authoritative. "
                  "Matched shadow proposals diagnose clipping, not separate-trajectory causality."]
    return "\n".join(lines) + "\n"

def make_train_step(model, teacher, tx, arm, order, protected_columns, head_dim):
    def corrected(base, coordinates):
        return apply_corrections(base, coordinates, head_dim=head_dim,
                                 protected_input_columns=protected_columns if arm.protected else None)

    @jax.jit
    def step(base, coordinates, state, teacher_params, tokens, step_number):
        # Teacher/anchor forward passes are outside the differentiated closure.
        teacher_output = None
        if arm.objective in ("teacher", "delta_then_teacher"):
            teacher_output = jax.tree.map(jax.lax.stop_gradient, teacher.apply(
                {"params": teacher_params}, tokens,
                return_hidden_states=arm.objective == "delta_then_teacher"))
        elif arm.objective == "self":
            teacher_output = jax.lax.stop_gradient(model.apply({"params": base}, tokens))

        def objective(c):
            candidate = corrected(base, c)
            if arm.objective == "delta_then_teacher":
                logits, states = model.apply({"params": candidate}, tokens, return_hidden_states=True)
                teacher_logits, teacher_states = teacher_output
                alignment = jax.lax.cond(step_number <= HORIZONS[0],
                    lambda: hidden_delta_relative_mse(states, teacher_states, tuple(i for i in order if i > 0)),
                    lambda: forward_kl(logits, teacher_logits, temperature=2.0))
                return alignment + 0.1 * causal_cross_entropy(logits, tokens)
            logits = model.apply({"params": candidate}, tokens)
            ce = causal_cross_entropy(logits, tokens)
            if arm.objective == "teacher":
                return forward_kl(logits, teacher_output, temperature=2.0) + 0.1 * ce
            if arm.objective == "self":
                return ce + 0.1 * forward_kl(logits, teacher_output, temperature=2.0)
            return ce
        loss, grad = jax.value_and_grad(objective)(coordinates)
        updates, new_state = tx.update(grad, state, coordinates)
        new = optax.apply_updates(coordinates, updates)
        norm = optax.global_norm(grad)
        finite = (jnp.isfinite(loss) & jnp.isfinite(norm) & coordinates_finite(grad)
                  & coordinates_finite(new) & coordinates_finite(new_state))
        return new, new_state, loss, norm, finite
    return step

def run_campaign(spec, argv=None, *, step_factory=None, contract_extra=None, aggregate_factory=None, schedule=None):
    horizons = HORIZONS if schedule is None else schedule.horizons
    train_windows = TRAIN_WINDOWS if schedule is None else schedule.train_windows
    data = DATA if schedule is None else schedule.data
    reserve_minutes = 30 if schedule is None else schedule.reserve_minutes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default=f"/dev/shm/extent-exp{spec.number}-state")
    parser.add_argument("--qwen-cache-dir", default=f"/dev/shm/qwen3-exp{spec.number}-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8.25 if schedule is None else schedule.max_wall_hours)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall_hours <= 8.5:
        raise ValueError("max-wall-hours must be in [1, 8.5]")
    if schedule is not None and args.max_wall_hours > schedule.max_wall_hours:
        raise ValueError("new campaigns cannot exceed the registered eight-hour session cap")
    if args.max_wall_hours * 60 <= reserve_minutes:
        raise ValueError("wall budget must leave time after the saving reserve")
    config, _ = load_config(Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml")
    contract = json.loads(json.dumps(contract_for(spec, config)))
    if contract_extra:
        contract.update(contract_extra)
    if schedule is not None:
        contract.update(horizons=list(horizons), data=[list(row) for row in data],
            student_training_tokens=len(SEEDS) * len(spec.arms) * horizons[-1] * LENGTH,
            schedule=asdict(schedule),
            execution_order={str(s): [a.name for a in (spec.arms if s == SEEDS[0] else tuple(reversed(spec.arms)))]
                             for s in SEEDS},
            datasets={"wiki": f"Salesforce/wikitext@{WIKITEXT_REVISION}/wikitext-103-raw-v1",
                      "pg19": f"deepmind/pg19@{PG19_REVISION}/test"},
            selection=f"no selection; validation diagnostic; both locked tests only at start/final {horizons[-1]}")
    # Schedule tuples must survive JSON/Hub roundtrips identically on resume.
    contract = json.loads(json.dumps(contract))
    summarize = (lambda result: aggregate(spec, result)) if aggregate_factory is None else aggregate_factory
    if args.plan_only:
        plan = {"contract": contract, "memory": {a.name: coordinate_plan(config, a,
                    len(contract["replacement_order"])) for a in spec.arms}}
        print(json.dumps(plan, indent=2))
        return plan
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - reserve_minutes * 60
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path, summary_path = output / f"{spec.stem}.json", output / f"{spec.stem}-summary.md"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO required before TPU initialization")
    if not result_path.exists():
        restore_artifact(result_path, f"{spec.prefix}/latest.json", hub)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "contract": contract, "status": "running", "branches": {}, "sessions": [], "pending_slots": []}
    if result["contract"] != contract:
        raise ValueError("resume protocol mismatch; refusing to mix experiments")
    if result.get("status") == "failed":
        result.setdefault("previous_failures", []).append({key: result[key] for key in
            ("stage", "error_type", "error") if key in result})
        for key in ("stage", "error_type", "error", "traceback"):
            result.pop(key, None)
    store = CampaignCheckpointStore(Path(args.state_dir) / "adapters", f"{spec.prefix}/checkpoints")
    remote_store = CampaignCheckpointStore(store.root, store.prefix, hub, retry_delays=())
    next_sync = 0.0
    result["sessions"].append({"started_at_utc": datetime.now(timezone.utc).isoformat(), "hours": 0.0})
    session = result["sessions"][-1]

    def persist(sync=False):
        nonlocal next_sync
        session["hours"] = (time.monotonic() - started) / 3600
        result["cumulative_hours"] = sum(s["hours"] for s in result["sessions"])
        result["aggregate"] = summarize(result)
        result["remote_sync_pending"] = bool(result["pending_slots"])
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(spec, result), encoding="utf-8")
        if not sync or time.monotonic() < next_sync:
            return
        files = [(result_path, f"{spec.prefix}/latest.json"),
                 (summary_path, f"{spec.prefix}/latest-summary.md")]
        pending = list(result["pending_slots"])
        for slot in pending:
            directory = store.root / slot
            meta = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
            remote_meta = directory / "remote-checkpoint.json"
            write_json_atomic(remote_meta, dict(meta, checkpoint_file="state.msgpack"))
            files += [(directory / meta["checkpoint_file"], f"{store.prefix}/{slot}/state.msgpack"),
                      (remote_meta, f"{store.prefix}/{slot}/checkpoint.json")]
        # Clear pending fields in the same atomic Hub commit as their payloads.
        result["pending_slots"], result["remote_sync_pending"] = [], False
        write_json_atomic(result_path, result)
        summary_path.write_text(render_summary(spec, result), encoding="utf-8")
        try:
            upload_artifacts_together(files, hub, commit_message=f"EXP-{spec.number} bundled recovery progress")
        except Exception as exc:
            result["pending_slots"], result["remote_sync_pending"] = pending, True
            result["remote_sync_error_type"] = type(exc).__name__
            write_json_atomic(result_path, result)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise
            next_sync = time.monotonic() + (3600 if status == 429 else 120)
            print(f"HF-DEFERRED type={type(exc).__name__}; local atomic states retained", flush=True)
        else:
            next_sync = time.monotonic() + 10 * 60
            result.pop("remote_sync_error_type", None)
            write_json_atomic(result_path, result)

    if args.sync_only:
        persist(sync=True)
        return result
    if summarize(result)["complete"]:
        persist(sync=True)
        print("Already completed; no TPU model allocation.", flush=True)
        return result
    if schedule is not None and terminal_result(spec, result):
        result["status"] = "branch_failure"
        persist(sync=True)
        print("All branches terminal; no TPU model allocation.", flush=True)
        return result
    stage = "preflight"
    _safe_notify(args.telegram, f"Extent EXP-{spec.number} started; budget={args.max_wall_hours}h")
    try:
        result["git_revision"] = subprocess.run(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout.strip()
        if not result["branches"]:
            exp091.preflight_source_endpoints()
        result["status"] = "running"
        persist(sync=True)  # Verify write authorization before expensive model allocation.
        token_sets = {name: load_wikitext2_tokens(windows * LENGTH, args.dataset_cache_dir,
            tokenizer_repo=QWEN3_1_7B_BASE.repo_id, tokenizer_revision=QWEN3_1_7B_BASE.revision,
            token_offset=offset, dataset_split=split, dataset_config="wikitext-103-raw-v1"
        ).reshape(windows, LENGTH) for name, split, offset, windows in data}
        if schedule is not None:
            token_sets["pg19_test"] = load_pg19_tokens(schedule.pg19_test_windows * LENGTH,
                args.dataset_cache_dir, tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
                tokenizer_revision=QWEN3_1_7B_BASE.revision, dataset_split="test",
                token_offset=schedule.pg19_test_offset).reshape(schedule.pg19_test_windows, LENGTH)
        hashes = {name: hashlib.sha256(x.tobytes()).hexdigest() for name, x in token_sets.items()}
        if contract.get("expected_data_sha256") and hashes != contract["expected_data_sha256"]:
            raise ValueError("token SHA differs from registered local data preflight")
        if result.get("data_sha256", hashes) != hashes:
            raise ValueError("resume token SHA mismatch")
        result["data_sha256"] = hashes
        devices = list(jax.devices())
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise ValueError("requires a single-host TPU v5e-8")
        mesh = create_v5e_mesh(devices)
        batch_layout = batch_sharding(mesh)
        init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_layout)
        if schedule is not None and time.monotonic() >= deadline:
            result["status"] = "deadline_partial"
            next_sync = 0.0
            persist(sync=True)
            _safe_notify(args.telegram, f"Extent EXP-{spec.number} deadline_partial during data preflight")
            return result
        stage = "load-model"
        source = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype="bfloat16",
                                          compute_dtype="bfloat16", remat_policy="full")
        teacher = Qwen3ForCausalLM(source)
        initial = initialize_sharded_parameters(teacher, jax.random.key(940), init_tokens, mesh)
        teacher_params, _ = stream_teacher_qwen_weights(initial.params, _ensure_checkpoint(Path(args.qwen_cache_dir)), source)
        del initial
        model = HybridForCausalLM(config)
        initial = initialize_sharded_parameters(model, jax.random.key(941), init_tokens, mesh)
        abstract, layout = initial.abstract_params, initial.layout
        del initial
        seq_contract = configured_contract(config)
        order = tuple(seq_contract["replacement_order"])
        inner = int(config.hidden_size * config.mamba.expand)
        protected_columns = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
        source_stores = (
            CampaignCheckpointStore(Path(args.state_dir) / "exp069", f"{EXP069_PREFIX}/checkpoints", hub),
            CampaignCheckpointStore(Path(args.state_dir) / "exp072", f"{EXP072['HF_PREFIX']}/checkpoints", hub))
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
        factory = make_train_step if step_factory is None else step_factory
        steps = {a.name: factory(model, teacher, tx, a, order, protected_columns, config.mamba.head_dim)
                 for a in spec.arms}
        evals = {a.name: jax.jit(lambda base, c, t, a=a: causal_cross_entropy(model.apply({"params":
            apply_corrections(base, c, head_dim=config.mamba.head_dim,
                protected_input_columns=protected_columns if a.protected else None)}, t), t)) for a in spec.arms}

        evaluation_windows = {}

        def evaluate(base, c, arm, name):
            values = [float(jax.block_until_ready(evals[arm.name](base, c,
                jax.device_put(w[None, :], batch_layout)))) for w in token_sets[name]]
            evaluation_windows[name] = values
            return float(np.mean(values))

        def checked_evaluate(row, base, c, arm, name):
            value = evaluate(base, c, arm, name)
            if step_factory is not None and not np.isfinite(value):
                row.update(failed=True, error=f"nonfinite {name} evaluation at step {row['step']}",
                           evaluation_failure={"split": name, "step": row["step"]})
                return None
            return value

        if contract.get("evaluate_teacher_baseline") and ("original_teacher_test_nll" not in result
                or schedule is not None and "original_teacher_pg19_nll" not in result):
            stage = "original-qwen-locked-test-baseline"
            teacher_eval = jax.jit(lambda weights, t: causal_cross_entropy(teacher.apply({"params": weights}, t), t))
            values = [float(jax.block_until_ready(teacher_eval(teacher_params,
                jax.device_put(w[None, :], batch_layout)))) for w in token_sets["locked_test"]]
            baseline = float(np.mean(values))
            if not np.isfinite(baseline):
                raise FloatingPointError("original Qwen baseline is nonfinite")
            result["original_teacher_test_nll"] = baseline
            if schedule is not None:
                result["original_teacher_test_windows"] = values
                values = [float(jax.block_until_ready(teacher_eval(teacher_params,
                    jax.device_put(w[None, :], batch_layout)))) for w in token_sets["pg19_test"]]
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError("original Qwen PG19 baseline is nonfinite")
                result["original_teacher_pg19_nll"] = float(np.mean(values))
                result["original_teacher_pg19_windows"] = values
            persist(sync=True)

        def compose(seed):
            endpoints, source_hashes = {}, {}
            for depth, layer in enumerate(order, 1):
                slot = f"prep/seed-{seed}/layer-{layer}"
                meta = source_stores[0].metadata(slot, dict(seq_contract["source_contract"],
                    seed=seed, layer=layer, kind="prepared_mamba"))
                if meta is None:
                    raise FileNotFoundError(f"missing EXP069 {slot}")
                c = sequential.endpoint_contract(seq_contract, seed, "ONPOLICY", layer, depth, meta["checkpoint_sha256"])
                restored = source_stores[1].restore(f"seed-{seed}/ONPOLICY/layer-{layer}", c)
                if restored is None:
                    raise FileNotFoundError(f"missing EXP072 seed={seed} layer={layer}")
                payload, metadata = restored
                endpoints[layer], source_hashes[str(layer)] = payload["params"], metadata["checkpoint_sha256"]
            return compose_parameters(teacher_params, endpoints, order, abstract, layout), source_hashes

        result["status"] = "running"
        # Rotate every method/seed through each horizon before extending it.
        for target in horizons:
            for seed in SEEDS:
                rows = result["branches"].setdefault(str(seed), {})
                if not any(needs_work(rows.get(a.name, {}), target, horizons[-1]) for a in spec.arms):
                    continue
                if time.monotonic() >= deadline:
                    break
                stage = f"compose-seed-{seed}"
                base, source_hashes = compose(seed)
                arm_order = spec.arms if schedule is None or seed == SEEDS[0] else tuple(reversed(spec.arms))
                for arm in arm_order:
                    row = rows.setdefault(arm.name, {"step": 0})
                    if not needs_work(row, target, horizons[-1]):
                        continue
                    stage = f"seed-{seed}-{arm.name}"
                    slot = f"seed-{seed}/{arm.name.lower()}"
                    checkpoint_contract = dict(contract, seed=seed, arm=asdict(arm),
                                               source_sha256=source_hashes, data_sha256=hashes)
                    coords = initialize_corrections(base, order, arm.subspace, seed=seed,
                                                   rank=arm.rank, head_dim=config.mamba.head_dim)
                    state = tx.init(coords)
                    template = {"coordinates": coords, "optimizer": state}
                    restored = store.restore(slot, checkpoint_contract, template)
                    if restored is None:
                        restored = remote_store.restore(slot, checkpoint_contract, template)
                    if restored:
                        payload, meta = restored
                        coords, state, row["step"] = payload["coordinates"], payload["optimizer"], meta["step"]
                    elif row["step"]:
                        raise FileNotFoundError(f"missing recovery state for {slot}; refusing restart")
                    row["coordinates"] = coordinate_count(coords)
                    row["effective_coordinates"] = row["coordinates"] - (len(order) * arm.rank *
                        (base[f"layers_{order[0]}"]["mamba"]["in_proj"]["kernel"].shape[1] - protected_columns)
                        if arm.protected else 0)
                    if row["coordinates"] != coordinate_plan(config, arm, len(order))["allocated_coordinates"]:
                        raise ValueError("runtime adapter shapes differ from registered memory plan")
                    if "start_validation_nll" not in row:
                        row["start_validation_nll"] = checked_evaluate(row, base, coords, arm, "validation")
                    if schedule is not None and not row.get("failed") and any(
                            key not in row for key in ("start_test_nll", "start_test_windows",
                                                       "start_pg19_nll", "start_pg19_windows")):
                        zero = initialize_corrections(base, order, arm.subspace, seed=seed, rank=arm.rank, head_dim=config.mamba.head_dim)
                        row["start_test_nll"] = checked_evaluate(row, base, zero, arm, "locked_test")
                        if not row.get("failed"):
                            row["start_test_windows"] = evaluation_windows["locked_test"]
                            row["start_pg19_nll"] = checked_evaluate(row, base, zero, arm, "pg19_test")
                        if not row.get("failed"):
                            row["start_pg19_windows"] = evaluation_windows["pg19_test"]
                        del zero

                    def save():
                        meta = store.save(slot, {"coordinates": coords, "optimizer": state},
                            contract=checkpoint_contract, step=row["step"], metrics={"finite": not row.get("failed", False)})
                        row["checkpoint_sha256"], row["checkpoint_bytes"] = meta["checkpoint_sha256"], meta["checkpoint_bytes"]
                        if slot not in result["pending_slots"]:
                            result["pending_slots"].append(slot)
                        persist(sync=True)

                    if restored is None:
                        save()  # Durable step zero also permits recovery from an early compile failure.
                    for number in range(row["step"] + 1, target + 1):
                        if row.get("failed"):
                            break
                        if time.monotonic() >= deadline:
                            save()
                            break
                        tokens = jax.device_put(token_sets["train"][(number - 1) % train_windows][None, :], batch_layout)
                        step_started = time.monotonic()
                        proposal = steps[arm.name](base, coords, state, teacher_params, tokens, jnp.asarray(number))
                        new, new_state, loss, norm, finite = proposal[:5]
                        jax.block_until_ready(loss)
                        if schedule is not None:
                            row["training_step_seconds"] = row.get("training_step_seconds", 0.0) + time.monotonic() - step_started
                            row["attempted_training_steps"] = row.get("attempted_training_steps", 0) + 1
                        if len(proposal) == 6:
                            health = jax.device_get(proposal[5])
                            def encode(value):
                                if isinstance(value, dict):
                                    return {k: encode(v) for k, v in value.items()}
                                if isinstance(value, (tuple, list)):
                                    return [encode(v) for v in value]
                                value = np.asarray(value).item()
                                return value if not isinstance(value, float) or np.isfinite(value) else None
                            row["latest_health"] = encode(health)
                            if schedule is not None:
                                logarithm = row["latest_health"].get("log10_grad_norm")
                                if logarithm is not None:
                                    row["max_observed_log10_grad_norm"] = max(row.get("max_observed_log10_grad_norm", -30.0), logarithm)
                                row["forward_gradient_nonfinite_steps"] = row.get("forward_gradient_nonfinite_steps", 0) + int(
                                    not bool(health.get("forward_finite", False)) or not bool(health.get("gradients_finite", False)))
                            if bool(health.get("norm_only_overflow", False)):
                                row["norm_only_overflow_steps"] = row.get("norm_only_overflow_steps", 0) + 1
                                if "first_norm_only_overflow" not in row:
                                    event_slot = f"{slot}/norm-only-event"
                                    event = {"step": number, "train_window": (number - 1) % train_windows,
                                             "health": encode(health)}
                                    metadata = store.save(event_slot, {"coordinates": coords, "optimizer": state},
                                        contract=dict(checkpoint_contract, kind="norm_only_event"),
                                        step=row["step"], metrics=event)
                                    event["checkpoint_sha256"] = metadata["checkpoint_sha256"]
                                    event["checkpoint_slot"] = event_slot
                                    row["first_norm_only_overflow"] = event
                                    result["pending_slots"].append(event_slot)
                        if not bool(finite):
                            row.update(failed=True, error=f"nonfinite step {number}; last finite state retained")
                            if len(proposal) == 6:
                                row["failure_diagnostics"] = {"attempted_step": number,
                                    "train_window": (number - 1) % train_windows, "health": row["latest_health"]}
                            save()
                            break
                        coords, state = new, new_state
                        row.update(step=number, latest_loss=float(loss), latest_grad_norm=float(norm))
                        if number % 256 == 0 or number == target:
                            if number % 1024 == 0:
                                row.setdefault("validation", {})[str(number)] = checked_evaluate(row, base, coords, arm, "validation")
                            save()
                            print(f"EXP{spec.number} seed={seed} arm={arm.name} step={number} loss={float(loss):.6f}", flush=True)
                    if row["step"] == horizons[-1] and not row.get("failed") and not row.get("complete"):
                        row["locked_test_nll"] = checked_evaluate(row, base, coords, arm, "locked_test")
                        if schedule is None:
                            zero = initialize_corrections(base, order, arm.subspace, seed=seed, rank=arm.rank, head_dim=config.mamba.head_dim)
                            row["start_test_nll"] = checked_evaluate(row, base, zero, arm, "locked_test")
                        elif not row.get("failed"):
                            row["locked_test_windows"] = evaluation_windows["locked_test"]
                            row["pg19_test_nll"] = checked_evaluate(row, base, coords, arm, "pg19_test")
                            if not row.get("failed"):
                                row["pg19_test_windows"] = evaluation_windows["pg19_test"]
                        row["complete"] = not row.get("failed", False)
                        save()
                    del coords, state, template, restored
                    gc.collect()
                    if time.monotonic() >= deadline:
                        break
                del base
                gc.collect()
            if time.monotonic() >= deadline:
                break
        done = summarize(result)["complete"]
        any_failed = any(r.get("failed") for rows in result["branches"].values() for r in rows.values())
        result["status"] = "completed" if done else "branch_failure" if any_failed else "deadline_partial"
        next_sync = 0.0
        persist(sync=True)
        _safe_notify(args.telegram, f"Extent EXP-{spec.number} {result['status']} gate={result['aggregate']['scientific_gate_passed']} remote_synced={not result['remote_sync_pending']}")
        print(f"result_json={result_path}\nsummary={summary_path}", flush=True)
        return result
    except BaseException as exc:
        result.update(status="failed", stage=stage, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        try:
            next_sync = 0.0  # A caught failure must attempt to publish recovery states now.
            persist(sync=True)
        except Exception:
            write_json_atomic(result_path, result)
        _safe_notify(args.telegram, f"Extent EXP-{spec.number} failed stage={stage} type={type(exc).__name__}")
        raise
