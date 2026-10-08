"""EXP106/107: matched long full-decoder recovery with global/layer clipping."""
import argparse
import ast
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from jax.sharding import NamedSharding, PartitionSpec as P

from extent import HybridForCausalLM
from extent.calibration_data import load_wikitext2_tokens, load_pg19_tokens, WIKITEXT_REVISION, PG19_REVISION
from extent.campaign_checkpoint import CampaignCheckpointStore, write_json_atomic
from extent.chunked_checkpoint import digest
from extent.config import load_config
from extent.decoder_recovery import split_decoder, forward_parameters
from extent.layer_clip_recovery import optimizer, make_step, schedule, TOTAL_STEPS, AGC_RATIO, MIN_WEIGHT_NORM
from extent.recovery_durability import RecoveryCheckpointStore as ChunkedCheckpointStore, cloud_roundtrip
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout, memory_estimate, evaluate, load_sources
from extent.full_model_distillation import causal_cross_entropy
from extent.initialization import abstract_parameter_tree, initialize_sharded_optimizer_state
from extent.recovery_subspace import initialize_corrections, apply_corrections, coordinates_finite
from extent.sharding import create_v5e_mesh, batch_sharding, replicated_sharding, named_sharding_tree
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifacts_together
from scripts import m3q_subspace_engine as engine
from scripts.m3q_plateau_campaign import source_initializer
from scripts.qwen_extended_horizon_campaign import _safe_notify

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (123, 456)
ARMS = ("DECODER-CE",)
HORIZONS = tuple(range(4096, TOTAL_STEPS + 1, 4096))
DATA = (("train", "train", 40_370_176, TOTAL_STEPS),
        ("validation", "validation", 114_688, 32),
        ("locked_test", "test", 294_912, 24))


def load_tokens(cache):
    source = engine.QWEN3_1_7B_BASE
    kwargs = dict(tokenizer_repo=source.repo_id, tokenizer_revision=source.revision)
    tokens = {name: load_wikitext2_tokens(n * 256, cache, token_offset=offset,
              dataset_split=split, dataset_config="wikitext-103-raw-v1", **kwargs).reshape(n,256)
              for name,split,offset,n in DATA}
    tokens["pg19_test"] = load_pg19_tokens(128*256, cache, dataset_split="test",
        token_offset=327_680, **kwargs).reshape(128,256)
    return tokens


def token_manifest(tokens):
    return {k:dict(sha256=hashlib.sha256(x.tobytes()).hexdigest(),shape=list(x.shape),dtype=str(x.dtype))
            for k,x in tokens.items()}


def contract_for(config, experiment=106):
    from dataclasses import asdict
    if experiment not in (106,107):
        raise ValueError("unregistered layer clipping experiment")
    files = ("scripts/m3q_layer_clip_campaign.py", "extent/layer_clip_recovery.py",
             "extent/recovery_durability.py", "scripts/m3q_paper_decoder_recovery_campaign.py",
             "extent/decoder_recovery.py", "extent/chunked_checkpoint.py", "extent/model.py",
             "extent/initialization.py", "extent/stable_gradient_clip.py", "extent/calibration_data.py",
             "extent/recovery_subspace.py")
    contract = dict(experiment=experiment,model=asdict(config),
        source=json.loads((ROOT/"results/EXP-101-103-source-manifest.json").read_text()),
        data=[list(row) for row in DATA],pg19=[327680,128],
        data_manifest=json.loads((ROOT/"results/EXP-106-107-data-preflight.json").read_text()),
        datasets=dict(wiki=WIKITEXT_REVISION,pg19=PG19_REVISION),
        tokenizer=dict(repo=engine.QWEN3_1_7B_BASE.repo_id,revision=engine.QWEN3_1_7B_BASE.revision),
        seeds=list(SEEDS),arms=list(ARMS),horizons=list(HORIZONS),
        optimizer=dict(name="AdamW",lr=1e-5,final_lr=1e-6,warmup=512,betas=[.9,.95],
                       eps=1e-8,matrix_decay=.1,fresh_at_transition=True,
                       clipping="layer-group-relative" if experiment==107 else "global-1",
                       agc_ratio=AGC_RATIO if experiment==107 else None,
                       min_weight_norm=MIN_WEIGHT_NORM if experiment==107 else None),
        evaluation_reuse="Wiki validation/test reused: exploratory; new PG19 test is primary confirmation",
        selection="fixed EXP098 final16384; no best endpoint; registered final32768",
        implementation={name:hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for name in files})
    return json.loads(json.dumps(contract))


def aggregate(result):
    rows=[result.get("branches",{}).get(str(s),{}).get("DECODER-CE",{}) for s in SEEDS]
    complete=all(r.get("step")==TOTAL_STEPS and "final" in r and not r.get("failed") for r in rows)
    gains={str(s):{n:r["start"][n]["nll"]-r["final"][n]["nll"] for n in ("locked_test","pg19_test")}
           for s,r in zip(SEEDS,rows) if "final" in r}
    return dict(complete=complete,primary_gate=None,transfer_gate=None,own_start_gains=gains,
                interpretation="comparison gate requires BOTH account results; use compare_campaigns")


def compare_campaigns(control, adaptive, target=TOTAL_STEPS):
    # Reject mismatched settings; hashes naturally differ only by experiment/clip.
    left,right=json.loads(json.dumps(control["contract"])),json.loads(json.dumps(adaptive["contract"]))
    if left.pop("experiment")!=106 or right.pop("experiment")!=107:
        raise ValueError("expected EXP106 global and EXP107 adaptive")
    for c in (left,right):
        for key in ("clipping","agc_ratio","min_weight_norm"):
            c["optimizer"].pop(key)
    if left!=right:
        raise ValueError("unmatched cross-account scientific contracts")
    rows={}
    for seed in SEEDS:
        c=control.get("branches",{}).get(str(seed),{}).get("DECODER-CE",{})
        a=adaptive.get("branches",{}).get(str(seed),{}).get("DECODER-CE",{})
        if c.get("failed") or a.get("failed"):
            return dict(complete=False,primary_gate=None,reason="numerical/resource failure")
        if target==TOTAL_STEPS:
            cm,am=c.get("final"),a.get("final")
            if c.get("step")!=target or a.get("step")!=target:
                cm,am=None,None
        else:
            cm=c.get("milestones",{}).get(str(target))
            am=a.get("milestones",{}).get(str(target))
        if cm is None or am is None:
            return dict(complete=False,primary_gate=None,reason="missing matched endpoint")
        for name in ("locked_test","pg19_test"):
            windows=[metric[name]["windows"] for metric in (c["start"],a["start"],cm,am)]
            if not windows[0] or any(len(w)!=len(windows[0]) for w in windows):
                raise ValueError("unmatched evaluation window counts")
            if any(not np.all(np.isfinite(w)) for w in windows) or any(
                not np.isfinite(metric[name]["nll"]) for metric in (c["start"],a["start"],cm,am)):
                raise ValueError("nonfinite comparison metrics")
            if max(abs(x-y) for x,y in zip(c["start"][name]["windows"],a["start"][name]["windows"]))>1e-4:
                raise ValueError("cross-account initial NLL parity failed")
        rows[str(seed)]=dict(pg19_control_gain=cm["pg19_test"]["nll"]-am["pg19_test"]["nll"],
            pg19_start_gain=a["start"]["pg19_test"]["nll"]-am["pg19_test"]["nll"],
            wiki_start_regression=am["locked_test"]["nll"]-a["start"]["locked_test"]["nll"],
            control_pg19_start_gain=c["start"]["pg19_test"]["nll"]-cm["pg19_test"]["nll"],
            control_wiki_start_gain=c["start"]["locked_test"]["nll"]-cm["locked_test"]["nll"],
            adaptive_wiki_start_gain=a["start"]["locked_test"]["nll"]-am["locked_test"]["nll"],
            paired_pg19=[x-y for x,y in zip(cm["pg19_test"]["windows"],am["pg19_test"]["windows"])])
        teacher=adaptive.get("original_qwen")
        if teacher:
            rows[str(seed)]["qwen_gaps"]={n:am[n]["nll"]-teacher[n]["nll"] for n in ("locked_test","pg19_test")}
    passed=all(r["pg19_control_gain"]>=.1 and r["pg19_start_gain"]>=.1 and r["wiki_start_regression"]<=.1 for r in rows.values())
    return dict(complete=True,primary_gate=bool(passed) if target==TOTAL_STEPS else None,
                target=target,exploratory=target!=TOTAL_STEPS,comparisons=rows)


def main(argv=None, *, experiment=106):
    if experiment not in (106,107):
        raise ValueError("unregistered experiment")
    adaptive = experiment == 107
    arms = ARMS
    prefix = f"experiments/exp{experiment}-layer-clip-recovery"
    stem = f"extent-m3q-layer-clip-{experiment}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--state-dir", default=f"/kaggle/working/extent-exp{experiment}-state")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--qwen-cache-dir", default="/kaggle/working/qwen3-exp104-weights")
    parser.add_argument("--dataset-cache-dir", default="/kaggle/working/extent-dataset-cache")
    parser.add_argument("--max-wall-hours", type=float, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--data-preflight-only", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if not 1.5 < args.max_wall_hours <= 8:
        raise ValueError("wall budget must be >1.5h and <=8h")
    config, _ = load_config(ROOT / "config/hybrid_1_7b_gqa_v5e8.yaml")
    if args.data_preflight_only:
        data = token_manifest(load_tokens(args.dataset_cache_dir))
        write_json_atomic(ROOT / "results/EXP-106-107-data-preflight.json", data)
        return data
    contract = contract_for(config, experiment)
    if args.plan_only:
        plan = dict(contract=contract, memory=json.loads((ROOT / "results/EXP-104-memory-preflight.json").read_text()))
        print(json.dumps(plan, indent=2))
        return plan
    started = time.monotonic()
    deadline = started + args.max_wall_hours * 3600 - 5400
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.json"
    summary_path = output / f"{stem}-summary.txt"
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO required")
    if not path.exists():
        restore_artifact(path, f"{prefix}/latest.json", hub)
    result = json.loads(path.read_text()) if path.exists() else dict(contract=contract, branches={}, pending_slots=[], sessions=[])
    if result["contract"] != contract:
        raise ValueError("EXP106/107 resume contract mismatch")
    if result.get("error"):
        result.setdefault("previous_failures", []).append({k:result[k] for k in
            ("status", "stage", "error_type", "error", "traceback") if k in result})
        for key in ("stage", "error_type", "error", "traceback"):
            result.pop(key, None)
    # Kaggle supplies much more host RAM than writable disk. Explicit state-dir
    # overrides retain the legacy/test location unless checkpoint-dir is given.
    explicit_state = any(x == "--state-dir" or x.startswith("--state-dir=") for x in arguments)
    checkpoint_root = args.checkpoint_dir or (args.state_dir if explicit_state or not Path("/dev/shm").is_dir()
                       else f"/dev/shm/extent-exp{experiment}-checkpoints")
    store = ChunkedCheckpointStore(checkpoint_root, f"{prefix}/checkpoints", hub)
    result["runtime_revision"] = dict(evaluation="canonical-BF16-materialization-and-layout-v2",
        implementation={p:digest(ROOT/p) for p in contract.get("implementation", {})}, checkpoint_root=str(checkpoint_root))
    session = dict(started_at_utc=datetime.now(timezone.utc).isoformat())
    result["sessions"].append(session)
    next_summary_sync = 0.0

    # Result files may be newer than local state after a disk/upload interruption.
    # Check the authoritative manifest before trusting completed branch cursors.
    for seed, rows in result["branches"].items():
        for arm, row in rows.items():
            if not row.get("step"):
                continue
            slot = f"seed-{seed}/{arm.lower()}"
            meta = store.metadata(slot, dict(contract, seed=int(seed), arm=arm))
            if meta is None:
                raise FileNotFoundError("reported progress has no durable optimizer manifest")
            if meta["step"] != row["step"]:
                row["step"] = meta["step"]
                row.pop("final", None)
                row.pop("layer_statistics", None)
                row["diagnostics"] = [d for d in row["diagnostics"] if d["step"] <= row["step"]]
                row["milestones"] = {k:v for k,v in row.get("milestones", {}).items() if int(k)<=row["step"]}

    def persist(sync=False, force=False):
        nonlocal next_summary_sync
        session["hours"] = (time.monotonic() - started) / 3600
        result["aggregate"] = aggregate(result)
        write_json_atomic(path, result)
        summary_path.write_text(json.dumps(dict(status=result.get("status"), aggregate=result["aggregate"],
            branches={s: {a: {k:v for k,v in r.items() if k in ("step", "failed", "start", "final", "error")}
                           for a,r in rows.items()} for s,rows in result["branches"].items()}), indent=2), encoding="utf-8")
        if not sync or not force and time.monotonic() < next_summary_sync:
            return
        for slot in list(result["pending_slots"]):
            if store.sync(slot, force=force, deadline=started + args.max_wall_hours * 3600):
                result["pending_slots"].remove(slot)
                result.setdefault("upload_measurements", {})[slot] = json.loads((store.directory(slot) / "synced.json").read_text())
                seed_name, arm_name = slot.split("/")
                result["branches"][seed_name.removeprefix("seed-")][arm_name.upper()]["durable_step"] = result["upload_measurements"][slot]["step"]
        result["checkpoint_sync_errors"] = dict(store.sync_errors)
        result["checkpoint_upload_status"] = "pending" if result["pending_slots"] else "durable"
        write_json_atomic(path, result)
        for seed, rows in result["branches"].items():
            for arm, row in rows.items():
                if not row.get("step"):
                    continue
                meta = store.metadata(f"seed-{seed}/{arm.lower()}", dict(contract, seed=int(seed), arm=arm))
                if meta is None or meta["step"] != row["step"]:
                    result["durability_error"] = "result ahead of safe local checkpoint; previous remote cursor retained"
                    write_json_atomic(path, result)
                    break
        try:
            upload_artifacts_together([(path, f"{prefix}/latest.json"), (summary_path, f"{prefix}/latest-summary.txt")],
                                      hub, commit_message="EXP106/107 progress")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise
            next_summary_sync = time.monotonic() + (3600 if status == 429 else 120)
            result["summary_sync_error"] = type(exc).__name__
            write_json_atomic(path, result)
        else:
            next_summary_sync = time.monotonic() + 600
            result.pop("summary_sync_error", None)

    def notify(event):
        counts = ", ".join(f"{s}/{a}={r.get('step',0)}" for s,rows in result["branches"].items() for a,r in rows.items())
        message = (f"Extent EXP-{experiment}: {event}; status={result.get('status')}; "
                   f"gate={aggregate(result)['primary_gate']}; HF pending={len(result['pending_slots'])}; {counts}")
        result.setdefault("notifications", []).append(dict(event=event, **_safe_notify(args.telegram, message)))
        write_json_atomic(path, result)

    if args.sync_only or aggregate(result)["complete"]:
        persist(sync=True, force=True)
        notify("sync finished" if not result["pending_slots"] else "sync incomplete: local states not durable")
        return result
    notify("started")
    try:
        result["status"] = "preflight"
        persist(sync=True, force=True)
        if result.get("summary_sync_error"):
            raise RuntimeError("HF write preflight not durable; models were not allocated")
        tokens = load_tokens(args.dataset_cache_dir)
        if token_manifest(tokens) != contract["data_manifest"]:
            raise ValueError("EXP106/107 token SHA/capacity differs from registered preflight")
        result["cloud_roundtrip"] = cloud_roundtrip(Path(checkpoint_root), prefix, hub,
            started + args.max_wall_hours*3600)
        persist()
        if time.monotonic() >= deadline:
            raise TimeoutError("training deadline reached during data preflight")
        devices = list(jax.devices())
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise ValueError("EXP106/107 requires single-host TPU v5e8")
        mesh, batch = create_v5e_mesh(devices), batch_sharding(create_v5e_mesh(devices))
        result["resources"] = dict(disk_free_bytes=shutil.disk_usage(output).free,
                                    devices=[str(d) for d in devices])
        try:
            import psutil
            result["resources"]["host_available_ram_bytes"] = psutil.virtual_memory().available
            shapes = abstract_parameter_tree(HybridForCausalLM(config))
            required_ram = 2 * sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(shapes)) + (2 << 30)
            if str(checkpoint_root).startswith("/dev/shm/"):
                shape_trainable = ({k:v for k,v in shapes.items() if k not in ("embed_tokens", "lm_head")}
                    )
                required_ram += 3 * sum(x.size*12 for x in jax.tree.leaves(shape_trainable))
            result["resources"]["source_host_cache_required_with_staging_bytes"] = required_ram
            if result["resources"]["host_available_ram_bytes"] < required_ram:
                raise MemoryError("host RAM cannot hold both fixed BF16 sources and bounded staging")
        except ImportError:
            result["resources"]["host_available_ram_bytes"] = None
        model, order, sources = load_sources(config, mesh, args, hub, tokens, result, persist, deadline)
        compiled_steps = {}
        columns = 2 * int(config.hidden_size * config.mamba.expand) + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
        materializer = jax.jit(lambda b,c: apply_corrections(b,c, protected_input_columns=columns,
                                   head_dim=config.mamba.head_dim),
                                   out_shardings=named_sharding_tree(sources[123][0], mesh))
        # Fix historical starts from the ORIGINAL warm source, before any
        # resumed state is restored or any new trajectory is trained.
        for seed in SEEDS:
            rows = result["branches"].setdefault(str(seed), {})
            if all(rows.get(a, {}).get("canonical_start_checked_v2") for a in arms):
                continue
            base, c = sources[seed]
            base = jax.device_put(base, named_sharding_tree(base, mesh))
            c = jax.device_put(c, trainable_layout(c, mesh))
            recovered = materializer(base, c)
            masters, frozen = split_decoder(recovered)
            reconstructed = forward_parameters(arms[0], masters, frozen, columns, config.mamba.head_dim)
            if not all(bool(jnp.all(x == y)) for x,y in zip(jax.tree.leaves(recovered),jax.tree.leaves(reconstructed))):
                raise ValueError("canonical seed preflight changed initial weights")
            reference = {name:evaluate(model,recovered,tokens[name],batch) for name in ("validation", "locked_test", "pg19_test")}
            check = {name:evaluate(model,reconstructed,tokens[name],batch) for name in reference}
            if any(max(abs(x-y) for x,y in zip(reference[n]["windows"],check[n]["windows"]))>1e-4 for n in reference):
                raise ValueError("canonical source start NLL differs >1e-4")
            for arm in arms:
                row = rows.setdefault(arm, dict(step=0, diagnostics=[]))
                if "start" in row and not row.get("canonical_start_checked_v2"):
                    row["start_uncanonical_v1"] = row["start"]
                    for diagnostic in row["diagnostics"]:
                        diagnostic.setdefault("evaluation_layout", "v1-uncanonical")
                row["start"] = reference
                row["canonical_start_checked_v2"] = True
            del base,c,recovered,masters,frozen,reconstructed
            gc.collect()
            persist()
        for target in HORIZONS:
            for seed in (SEEDS if (target//4096)%2 else SEEDS[::-1]):
                rows = result["branches"].setdefault(str(seed), {})
                for arm in (arms if seed == 123 else arms[::-1]):
                    row = rows.setdefault(arm, dict(step=0, diagnostics=[]))
                    if row.get("failed") or row["step"] >= target and (target != TOTAL_STEPS or "final" in row):
                        continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError("registered training deadline")
                    result["stage"] = f"{seed}/{arm}/{target}"
                    host_base, host_c = sources[seed]
                    base = jax.device_put(host_base, named_sharding_tree(host_base, mesh))
                    c = jax.device_put(host_c, trainable_layout(host_c, mesh))
                    recovered = materializer(base,c)
                    dense_arm = arms[0]
                    masters, frozen = split_decoder(recovered)
                    restored_start = forward_parameters(dense_arm, masters, frozen, columns, config.mamba.head_dim)
                    if not all(bool(jnp.all(x == y)) for x,y in zip(jax.tree.leaves(recovered), jax.tree.leaves(restored_start))):
                        raise ValueError("FP32 master/BF16 forward differs from adapter materialization")
                    del restored_start
                    initial_parameters = (masters, frozen) if arm == dense_arm else (c, base)
                    parameters, fixed = initial_parameters
                    layout, fixed_layout = trainable_layout(parameters, mesh), named_sharding_tree(fixed, mesh)
                    parameters, fixed = jax.device_put(parameters, layout), jax.device_put(fixed, fixed_layout)
                    # Always evaluate the INITIAL registered source before a
                    # checkpoint restore; never relabel step512 as a new start.
                    initial = materializer(fixed, parameters) if arm == "ADAPTER-CE" else forward_parameters(arm, parameters, fixed, columns, config.mamba.head_dim)
                    if not all(bool(jnp.all(x == y)) for x,y in zip(jax.tree.leaves(recovered),jax.tree.leaves(initial))):
                        raise ValueError("initial materialized kernels differ after sharding")
                    if not row.get("canonical_start_checked_v2"):
                        if "start" in row:
                            row["start_uncanonical_v1"] = row.pop("start")
                        row["start"] = {name:evaluate(model, initial, tokens[name], batch)
                                        for name in ("validation", "locked_test", "pg19_test")}
                        row["canonical_start_checked_v2"] = True
                    del initial
                    del base, c, recovered, masters, frozen, initial_parameters
                    plan = memory_estimate(parameters, fixed, layout, fixed_layout)
                    row["memory_estimate"] = plan
                    stats = devices[0].memory_stats() or {}
                    row["allocator_before_optimizer"] = stats
                    limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
                    if limit is None:
                        raise RuntimeError("TPU allocator did not report HBM capacity; refusing unsupported memory budget")
                    if plan["conservative_bytes_per_device"] > limit:
                        raise MemoryError("full registered strategy exceeds conservative per-device HBM budget")
                    tx = optimizer(parameters, adaptive=adaptive)
                    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters)
                    sharded = initialize_sharded_optimizer_state(tx, parameters, abstract, layout, mesh)
                    state, state_layout = sharded.opt_state, sharded.layout
                    del sharded
                    slot = f"seed-{seed}/{arm.lower()}"
                    ccontract = dict(contract, seed=seed, arm=arm)
                    template = {"parameters": parameters, "optimizer": state}
                    payload_layout = {"parameters": layout, "optimizer": state_layout}
                    saved = store.restore(slot, ccontract, template, payload_layout)
                    resumed = saved is not None
                    if saved is not None:
                        payload, meta = saved
                        parameters, state, row["step"] = payload["parameters"], payload["optimizer"], meta["step"]
                        row["diagnostics"] = [d for d in row["diagnostics"] if d["step"] <= row["step"]]
                        if row["step"] < TOTAL_STEPS:
                            row.pop("final", None)
                        row["milestones"] = {k:v for k,v in row.get("milestones",{}).items() if int(k)<=row["step"]}
                        del payload
                    elif row["step"]:
                        raise FileNotFoundError("result has progress but compatible optimizer checkpoint is missing")
                    del template
                    del saved
                    if not bool(coordinates_finite(parameters)) or not bool(coordinates_finite(state)):
                        raise FloatingPointError("nonfinite restored state")
                    if int(state[1].count) != row["step"] or int(state[3].count) != row["step"]:
                        raise ValueError("restored Adam/schedule counters differ from checkpoint cursor")
                    def current():
                        return materializer(fixed, parameters) if arm == "ADAPTER-CE" else forward_parameters(arm, parameters, fixed, columns, config.mamba.head_dim)
                    if "start" not in row:
                        row["start"] = {name: evaluate(model, current(), tokens[name], batch)
                                        for name in ("validation", "locked_test", "pg19_test")}
                    other = rows.get(arms[0] if arm == arms[0] else arms[0], {})
                    if other.get("canonical_start_checked_v2"):
                        for name in row["start"]:
                            if max(abs(x-y) for x,y in zip(row["start"][name]["windows"], other["start"][name]["windows"])) > 1e-4:
                                raise ValueError("initial adapter/decoder NLL differs >1e-4")
                    if arm not in compiled_steps:
                        step = make_step(model, tx, arm, columns, config.mamba.head_dim,
                                         (layout, state_layout, fixed_layout, batch, replicated_sharding(mesh)), adaptive=adaptive)
                        compiled_steps[arm] = step.lower(parameters, state, fixed,
                            jax.device_put(tokens["train"][0][None], batch)).compile()
                    compiled = compiled_steps[arm]
                    analysis = compiled.memory_analysis()
                    row["compiler_memory"] = None if analysis is None else {
                        k: int(getattr(analysis, k)) for k in ("argument_size_in_bytes", "output_size_in_bytes",
                        "temp_size_in_bytes", "alias_size_in_bytes")}
                    if analysis is not None:
                        measured = analysis.argument_size_in_bytes + analysis.output_size_in_bytes + analysis.temp_size_in_bytes - analysis.alias_size_in_bytes
                        if measured > limit:
                            raise MemoryError("compiled full step exceeds reported HBM capacity")
                    step = compiled
                    last_saved = row["step"] if resumed else -1
                    def save():
                        nonlocal last_saved
                        meta = store.save(slot, ccontract, {"parameters": parameters, "optimizer": state}, row["step"])
                        row["checkpoint_bytes"] = meta["payload_bytes"]
                        row["checkpoint_generation"] = meta["generation"]
                        row["remaining_save_reserve_seconds"] = started + args.max_wall_hours*3600 - time.monotonic()
                        if slot not in result["pending_slots"]:
                            result["pending_slots"].append(slot)
                        last_saved = row["step"]
                        persist()
                    if not resumed:
                        save()
                        persist(sync=True, force=True)
                        if slot in result["pending_slots"]:
                            if not store.sync_until(slot,min(deadline,time.monotonic()+1200)):
                                raise RuntimeError("initial full checkpoint not remotely verified; training not started")
                            persist(sync=True,force=True)
                    try:
                        proposed, proposed_state = parameters, state
                        while row["step"] < target and time.monotonic() < deadline:
                            update_started = time.monotonic()
                            proposed, proposed_state, health = step(parameters, state, fixed,
                                jax.device_put(tokens["train"][row["step"]][None], batch))
                            health = jax.tree.map(lambda v:np.asarray(v).item(), jax.device_get(health))
                            row["measured_update_seconds"] = row.get("measured_update_seconds", 0.) + time.monotonic()-update_started
                            row["measured_updates"] = row.get("measured_updates", 0) + 1
                            if not health["finite"]:
                                row.update(failed=True, error="nonfinite proposal rejected", failure_step=row["step"]+1)
                                proposed, proposed_state = parameters, state
                                break
                            parameters, state = proposed, proposed_state
                            row["step"] += 1
                            row["last_health"] = health
                            stats = row.setdefault("layer_statistics", {})
                            for name, values in health["layers"].items():
                                summary = stats.setdefault(name, dict(steps=0, clipped_steps=0,
                                    min_gradient_retained_fraction=1., max_log10_grad_norm=-30.,
                                    max_log10_relative_update=-30.))
                                summary["steps"] += 1
                                summary["clipped_steps"] += int(values["gradient_retained_fraction"] < .999999)
                                for metric in ("log10_grad_norm","log10_relative_update"):
                                    summary["max_"+metric] = max(summary["max_"+metric],values[metric])
                                summary["min_gradient_retained_fraction"] = min(
                                    summary["min_gradient_retained_fraction"], values["gradient_retained_fraction"])
                            row["norm_only_overflow_count"] = row.get("norm_only_overflow_count", 0) + int(not health.get("naive_norm_finite", True))
                            row["last_learning_rate"] = float(schedule(row["step"] - 1))
                            if row["step"] % 1024 == 0:
                                row["diagnostics"].append(dict(step=row["step"], health=health, evaluation_layout="canonical-v2",
                                    validation=evaluate(model, current(), tokens["validation"], batch)))
                                print(f"EXP{experiment} seed={seed} arm={arm} step={row['step']} CE={health['loss']:.5f}", flush=True)
                            if row["step"] % 4096 == 0:
                                save()
                                persist(sync=True)
                        if row["step"] == target and not row.get("failed"):
                            row.setdefault("milestones", {})[str(target)] = {name:evaluate(model,current(),tokens[name],batch)
                                for name in ("locked_test","pg19_test")}
                        if row["step"] == TOTAL_STEPS and not row.get("failed"):
                            row["final"] = {name: evaluate(model, current(), tokens[name], batch)
                                             for name in ("locked_test", "pg19_test")}
                            row["allocator_after_training"] = devices[0].memory_stats()
                            for key in ("embed_tokens", "lm_head"):
                                if key in host_base:
                                    expected = jax.device_put(host_base[key], named_sharding_tree({key: host_base[key]}, mesh)[key])
                                    actual = current()[key]
                                    if not all(bool(jnp.all(x == y)) for x,y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual))):
                                        raise ValueError("frozen vocabulary weights changed")
                                    del expected, actual
                            row["frozen_vocabulary_bitwise_verified"] = True
                    finally:
                        if last_saved != row["step"]:
                            save()
                    if row.get("failed"):
                        raise FloatingPointError("nonfinite proposal rejected; campaign stopped with last safe state")
                    del parameters, state, fixed, step, proposed, proposed_state, current
                    gc.collect()
                    persist(sync=True)
        result["status"] = "completed" if aggregate(result)["complete"] else "branch_failure"
    except TimeoutError as exc:
        result.update(status="deadline_partial", error=str(exc))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        notify("failed")
        persist(sync=True, force=True)
        raise
    notify("training finished" if aggregate(result)["complete"] else "training stopped")
    persist(sync=True, force=True)
    while result["pending_slots"] and time.monotonic() < started + args.max_wall_hours*3600 - 60:
        wait = max(1., store.next_sync-time.monotonic())
        if wait > started + args.max_wall_hours*3600-time.monotonic()-60:
            break
        time.sleep(min(wait,60))
        persist(sync=True, force=True)
    notify("finished and saved" if not result["pending_slots"] else "finished; HF states still pending")
    return result


if __name__ == "__main__":
    main()
