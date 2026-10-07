"""EXP-104: paper stage-2 CE strategy on the fixed EXP098 recovered hybrid."""
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
from extent.chunked_checkpoint import ChunkedCheckpointStore, digest
from extent.config import load_config
from extent.decoder_recovery import split_decoder, split_mamba, forward_parameters, optimizer, make_step, schedule
from extent.full_model_distillation import causal_cross_entropy
from extent.initialization import abstract_parameter_tree, initialize_sharded_optimizer_state
from extent.recovery_subspace import initialize_corrections, apply_corrections, coordinates_finite
from extent.sharding import create_v5e_mesh, batch_sharding, replicated_sharding, named_sharding_tree
from extent.hf_artifact_sync import artifact_config_from_env, restore_artifact, upload_artifacts_together
from scripts import m3q_subspace_engine as engine
from scripts.m3q_plateau_campaign import source_initializer

ROOT = Path(__file__).resolve().parents[1]
STEM = "extent-m3q-paper-decoder-recovery"
PREFIX = "experiments/exp104-paper-decoder-recovery"
SEEDS = (123, 456)
ARMS = ("ADAPTER-CE", "DECODER-CE")
HORIZONS = (512, 1024, 2048)
DATA = (("train", "train", 39_845_888, 2048),
        ("validation", "validation", 114_688, 32),
        ("locked_test", "test", 294_912, 24))
_EVALUATORS = {}


def load_tokens(cache):
    source = engine.QWEN3_1_7B_BASE
    kwargs = dict(tokenizer_repo=source.repo_id, tokenizer_revision=source.revision)
    tokens = {name: load_wikitext2_tokens(n * 256, cache, token_offset=offset,
              dataset_split=split, dataset_config="wikitext-103-raw-v1", **kwargs).reshape(n, 256)
              for name, split, offset, n in DATA}
    tokens["pg19_test"] = load_pg19_tokens(64 * 256, cache, dataset_split="test",
                                          token_offset=294_912, **kwargs).reshape(64, 256)
    return tokens


def token_manifest(tokens):
    return {k: dict(sha256=hashlib.sha256(x.tobytes()).hexdigest(), shape=list(x.shape),
                   dtype=str(x.dtype)) for k, x in tokens.items()}


def contract_for(config, experiment=104):
    from dataclasses import asdict
    source = json.loads((ROOT / "results/EXP-101-103-source-manifest.json").read_text())
    data = json.loads((ROOT / "results/EXP-104-data-preflight.json").read_text())
    files = ("scripts/m3q_paper_decoder_recovery_campaign.py", "extent/decoder_recovery.py",
             "extent/chunked_checkpoint.py", "extent/initialization.py", "extent/model.py",
             "extent/recovery_subspace.py", "extent/stable_gradient_clip.py", "extent/calibration_data.py")
    contract = dict(experiment=experiment, model=asdict(config), source=source,
        data=[list(row) for row in DATA], pg19=[294_912, 64], data_manifest=data,
        datasets=dict(wiki=WIKITEXT_REVISION, pg19=PG19_REVISION),
        tokenizer=dict(repo=engine.QWEN3_1_7B_BASE.repo_id, revision=engine.QWEN3_1_7B_BASE.revision),
        horizons=list(HORIZONS), seeds=list(SEEDS), arms=list(ARMS),
        optimizer=dict(name="AdamW", lr=1e-5, final_lr=1e-6, warmup=64, betas=[.9, .95],
                       eps=1e-8, matrix_decay=.1, stable_clip=1, fresh_at_transition=True),
        precision="FP32 masters and moments; BF16 forward; frozen input/output vocabulary",
        selection="fixed EXP098 final16384; no validation selection; locked start/final2048",
        implementation={name: digest(ROOT / name) for name in files})
    contract = json.loads(json.dumps(contract))
    if experiment == 104:
        # Explicit, narrowly scoped continuation of the measured v1 run:
        # objective/optimizer/source/data contract remains exactly unchanged.
        legacy = json.loads((ROOT / "results/EXP-104-v1-contract.json").read_text(encoding="utf-8-sig"))
        contract["implementation"] = legacy["implementation"]
        if contract != legacy:
            raise ValueError("EXP104 scientific settings differ from approved v1 continuation")
        for path, expected in legacy["implementation"].items():
            if path not in ("scripts/m3q_paper_decoder_recovery_campaign.py", "extent/decoder_recovery.py"):
                actual = hashlib.sha256((ROOT/path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if actual != expected:
                    raise ValueError(f"EXP104 unchanged scientific dependency differs: {path}")
        functions = {n.name:n for n in ast.parse((ROOT/"extent/decoder_recovery.py").read_text()).body if isinstance(n,ast.FunctionDef)}
        fingerprints = json.loads((ROOT/"results/EXP-104-v1-training-fingerprints.json").read_text())
        nodes = {name:functions[name] for name in ("optimizer", "schedule", "make_step", "split_decoder", "decoder_parameters")}
        nodes.update(decoder_branch=functions["forward_parameters"].body[0], adapter_branch=functions["forward_parameters"].body[2])
        if any(hashlib.sha256(ast.dump(node).encode()).hexdigest() != fingerprints[name] for name,node in nodes.items()):
            raise ValueError("EXP104 v1 training math changed; explicit new contract required")
    elif experiment == 105:
        contract["arms"] = ["ADAPTER-CE", "MAMBA-CE"]
        contract["precision"] = "FP32 Mamba-only masters/moments; BF16 forward; all non-Mamba source leaves frozen"
        contract["evaluation_reuse"] = "pre-registered shared EXP104 windows; prior starts visible, no final outcome used for selection; not independent replication"
    else:
        raise ValueError("unregistered experiment")
    return contract


def trainable_layout(tree, mesh):
    flat = traverse_util.flatten_dict(tree)
    ordinary = traverse_util.flatten_dict(named_sharding_tree(tree, mesh))
    result = {}
    parts = mesh.shape["fsdp"] * mesh.shape["tensor"]
    for path, x in flat.items():
        layout = ordinary[path]
        if x.ndim >= 2 and layout.is_fully_replicated and parts > 1:
            dimension = next((i for i, n in enumerate(x.shape) if n % parts == 0), None)
            if dimension is None:
                raise ValueError(f"unshardable trainable matrix {path}: {x.shape}")
            axes = [None] * x.ndim
            axes[dimension] = ("fsdp", "tensor")
            layout = NamedSharding(mesh, P(*axes))
        result[path] = layout
    return traverse_util.unflatten_dict(result)


def memory_estimate(parameters, fixed, layout, fixed_layout):
    def per_device(tree, shards):
        return sum(int(x.size * x.dtype.itemsize / s.num_devices) if not s.is_fully_replicated
                   else int(x.size * x.dtype.itemsize) for x, s in zip(jax.tree.leaves(tree), jax.tree.leaves(shards)))
    # Conservative coexistence: current/proposed masters and moments (6x),
    # gradients (1x), optimizer updates (1x), BF16 forward copy (0.5x).
    train = per_device(parameters, layout)
    frozen = per_device(fixed, fixed_layout)
    reserve = 6 << 30  # policy allowance, not measured activation/compile memory
    return dict(trainable_count=sum(int(x.size) for x in jax.tree.leaves(parameters)),
        trainable_master_bytes_per_device=train, fixed_bytes_per_device=frozen,
        current_and_proposed_weights_moments_gradients_updates_forward_bytes=int(8.5 * train + frozen),
        activation_compile_workspace_allowance_bytes=reserve,
        conservative_bytes_per_device=int(8.5 * train + frozen + reserve),
        teacher_device_resident=False, source_other_seed_device_resident=False,
        limitation="policy estimate; actual compiler/allocator measurements required on TPU")


def evaluate(model, parameters, tokens, batch):
    # Reuse one executable for all equal-length windows and branch evaluations.
    layout = named_sharding_tree(parameters, batch.mesh)
    parameters = jax.device_put(parameters, layout)
    call = _EVALUATORS.setdefault(id(model), jax.jit(
        lambda p, t: causal_cross_entropy(model.apply({"params": p}, t), t),
        in_shardings=(layout, batch), out_shardings=replicated_sharding(batch.mesh)))
    values = [float(call(parameters, jax.device_put(w[None], batch))) for w in tokens]
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("nonfinite evaluation")
    return dict(nll=float(np.mean(values)), windows=values)


def load_sources(config, mesh, args, hub, tokens, result, persist, deadline):
    """Teacher exists only during source assembly/evaluation, never full training."""
    seq = engine.configured_contract(config)
    order = tuple(seq["replacement_order"])
    class ReadOnlySourceStore(CampaignCheckpointStore):
        def sync(self, slot):
            return False
    stores = (ReadOnlySourceStore(Path(args.state_dir) / "exp069", f"{engine.EXP069_PREFIX}/checkpoints", hub),
              ReadOnlySourceStore(Path(args.state_dir) / "exp072", f"{engine.EXP072['HF_PREFIX']}/checkpoints", hub))
    manifest = json.loads((ROOT / "results/EXP-101-103-source-manifest.json").read_text())
    pinned_hub = replace(hub, repo_id=manifest["repo_id"], repo_type="dataset", revision=manifest["revision"])
    pinned_store = ReadOnlySourceStore(Path(args.state_dir) / "exp098", manifest["prefix"], pinned_hub)
    # Full binary/source SHA checks precede production weight allocation.
    for seed in SEEDS:
        entry = manifest["branches"][str(seed)]
        expected = entry["metadata"]
        source_state = pinned_store.restore(entry["slot"], expected["contract"])
        if source_state is None:
            raise FileNotFoundError("missing pinned EXP098 recovery source")
        source_payload, source_meta = source_state
        if source_meta["checkpoint_sha256"] != expected["checkpoint_sha256"] or source_meta["step"] != 16384:
            raise ValueError("pinned EXP098 binary SHA/step mismatch")
        if not all(np.all(np.isfinite(x)) for x in jax.tree.leaves(source_payload["coordinates"])):
            raise FloatingPointError("nonfinite EXP098 source coordinates")
        del source_payload, source_state
        for depth, layer in enumerate(order, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("training deadline reached during binary source preflight")
            prep = stores[0].metadata(f"prep/seed-{seed}/layer-{layer}", dict(seq["source_contract"],
                                      seed=seed, layer=layer, kind="prepared_mamba"))
            if prep is None:
                raise FileNotFoundError("missing EXP069 source")
            c = engine.sequential.endpoint_contract(seq, seed, "ONPOLICY", layer, depth, prep["checkpoint_sha256"])
            meta = stores[1].metadata(f"seed-{seed}/ONPOLICY/layer-{layer}", c)
            if meta is None or meta["checkpoint_sha256"] != expected["contract"]["source_sha256"][str(layer)]:
                raise ValueError("EXP072 binary SHA differs from fixed recovered source")
        print(f"EXP104 source binary preflight PASS seed={seed}", flush=True)
    source = engine.teacher_config_from_spec(engine.QWEN3_1_7B_BASE,
        param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="full")
    teacher = engine.Qwen3ForCausalLM(source)
    init_tokens = jax.device_put(np.zeros((1, 1), np.int32), batch_sharding(mesh))
    initial = engine.initialize_sharded_parameters(teacher, jax.random.key(940), init_tokens, mesh)
    teacher_params, _ = engine.stream_teacher_qwen_weights(initial.params,
                         engine._ensure_checkpoint(Path(args.qwen_cache_dir)), source)
    del initial
    model = HybridForCausalLM(config)
    abstract = abstract_parameter_tree(model)
    layout = named_sharding_tree(abstract, mesh)
    initialize = source_initializer(manifest, pinned_store)
    if not result.get("original_qwen_canonical_v2"):
        if "original_qwen" in result:
            result["original_qwen_uncanonical_v1"] = result["original_qwen"]
        result["original_qwen"] = {name: evaluate(teacher, teacher_params, tokens[name], batch_sharding(mesh))
                                   for name in ("locked_test", "pg19_test")}
        result["original_qwen_canonical_v2"] = True
        persist()
    sources = {}
    for seed in SEEDS:
        if time.monotonic() >= deadline:
            raise TimeoutError("training deadline reached during source loading")
        endpoints, hashes = {}, {}
        for depth, layer in enumerate(order, 1):
            prep = stores[0].metadata(f"prep/seed-{seed}/layer-{layer}", dict(seq["source_contract"],
                                      seed=seed, layer=layer, kind="prepared_mamba"))
            if prep is None:
                raise FileNotFoundError("missing prepared EXP069 source")
            c = engine.sequential.endpoint_contract(seq, seed, "ONPOLICY", layer, depth, prep["checkpoint_sha256"])
            restored = stores[1].restore(f"seed-{seed}/ONPOLICY/layer-{layer}", c)
            if restored is None:
                raise FileNotFoundError("missing EXP072 endpoint")
            payload, meta = restored
            endpoints[layer], hashes[str(layer)] = payload["params"], meta["checkpoint_sha256"]
        base = engine.compose_parameters(teacher_params, endpoints, order, abstract, layout)
        coordinates = initialize_corrections(base, order, "INOUT-LORA", seed=seed, rank=32,
                                             head_dim=config.mamba.head_dim)
        old_tx = engine.optax.chain(engine.optax.clip_by_global_norm(1), engine.optax.adam(3e-4))
        coordinates = initialize(seed, None, {"coordinates": coordinates, "optimizer": old_tx.init(coordinates)}, hashes)
        sources[seed] = (jax.device_get(base), jax.device_get(coordinates))
        del base, coordinates, endpoints, payload
        gc.collect()
    del teacher_params, teacher, stores, pinned_store
    gc.collect()
    jax.clear_caches()
    return model, order, sources


def aggregate(result):
    arms = result.get("contract", {}).get("arms", ARMS)
    rows = [result.get("branches", {}).get(str(seed), {}).get(arm, {}) for seed in SEEDS for arm in arms]
    complete = all(r.get("step") == 2048 and "final" in r and not r.get("failed") for r in rows)
    if not complete:
        return dict(complete=False, primary_gate=None, transfer_gate=None)
    comparisons = {}
    gate, transfer = True, True
    for seed in SEEDS:
        c, p = (result["branches"][str(seed)][a] for a in arms)
        wiki_gain = c["final"]["locked_test"]["nll"] - p["final"]["locked_test"]["nll"]
        start_gain = p["start"]["locked_test"]["nll"] - p["final"]["locked_test"]["nll"]
        pg_gain = c["final"]["pg19_test"]["nll"] - p["final"]["pg19_test"]["nll"]
        pg_regression = p["final"]["pg19_test"]["nll"] - p["start"]["pg19_test"]["nll"]
        comparisons[str(seed)] = dict(wiki_control_gain=wiki_gain, wiki_start_gain=start_gain,
                                     pg19_control_gain=pg_gain, pg19_start_regression=pg_regression,
                                     qwen_wiki_gap=p["final"]["locked_test"]["nll"] - result["original_qwen"]["locked_test"]["nll"],
                                     paired_wiki=[x-y for x,y in zip(c["final"]["locked_test"]["windows"], p["final"]["locked_test"]["windows"])])
        gate &= wiki_gain >= .1 and start_gain >= .1 and pg_regression <= .1
        transfer &= pg_gain >= .1
    return dict(complete=True, primary_gate=bool(gate), transfer_gate=bool(gate and transfer), comparisons=comparisons)


def main(argv=None, *, experiment=104):
    arms = ARMS if experiment == 104 else ("ADAPTER-CE", "MAMBA-CE")
    prefix = PREFIX if experiment == 104 else "experiments/exp105-mamba-capacity-recovery"
    stem = STEM if experiment == 104 else "extent-m3q-mamba-capacity-recovery"
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
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if not 1.5 < args.max_wall_hours <= 8:
        raise ValueError("wall budget must be >1.5h and <=8h")
    config, _ = load_config(ROOT / "config/hybrid_1_7b_gqa_v5e8.yaml")
    if args.data_preflight_only:
        data = token_manifest(load_tokens(args.dataset_cache_dir))
        write_json_atomic(ROOT / "results/EXP-104-data-preflight.json", data)
        return data
    contract = contract_for(config) if experiment == 104 else contract_for(config, experiment)
    if args.plan_only:
        plan = dict(contract=contract, memory=json.loads((ROOT / f"results/EXP-{experiment}-memory-preflight.json").read_text()))
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
        raise ValueError("EXP104 resume contract mismatch")
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
                row["diagnostics"] = [d for d in row["diagnostics"] if d["step"] <= row["step"]]

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
        write_json_atomic(path, result)
        if result["pending_slots"]:
            # Never publish progress ahead of its remotely durable optimizer.
            return
        for seed, rows in result["branches"].items():
            for arm, row in rows.items():
                if not row.get("step"):
                    continue
                meta = store.metadata(f"seed-{seed}/{arm.lower()}", dict(contract, seed=int(seed), arm=arm))
                if meta is None or meta["step"] != row["step"]:
                    result["durability_error"] = "result ahead of safe local checkpoint; previous remote cursor retained"
                    write_json_atomic(path, result)
                    return
        try:
            upload_artifacts_together([(path, f"{prefix}/latest.json"), (summary_path, f"{prefix}/latest-summary.txt")],
                                      hub, commit_message="EXP104 progress")
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

    if args.sync_only or aggregate(result)["complete"]:
        persist(sync=True, force=True)
        return result
    try:
        result["status"] = "preflight"
        persist(sync=True, force=True)
        if result.get("summary_sync_error"):
            raise RuntimeError("HF write preflight not durable; models were not allocated")
        tokens = load_tokens(args.dataset_cache_dir)
        if token_manifest(tokens) != contract["data_manifest"]:
            raise ValueError("EXP104 token SHA/capacity differs from registered preflight")
        if time.monotonic() >= deadline:
            raise TimeoutError("training deadline reached during data preflight")
        devices = list(jax.devices())
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise ValueError("EXP104 requires single-host TPU v5e8")
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
                    if experiment == 104 else {k:{"mamba":v["mamba"]} for k,v in shapes.items() if "mamba" in v})
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
            masters, frozen = split_decoder(recovered) if experiment == 104 else split_mamba(recovered)
            reconstructed = forward_parameters(arms[1], masters, frozen, columns, config.mamba.head_dim)
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
            for seed in SEEDS:
                rows = result["branches"].setdefault(str(seed), {})
                for arm in (arms if seed == 123 else arms[::-1]):
                    row = rows.setdefault(arm, dict(step=0, diagnostics=[]))
                    if row.get("failed") or row["step"] >= target and (target != 2048 or "final" in row):
                        continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError("registered training deadline")
                    result["stage"] = f"{seed}/{arm}/{target}"
                    host_base, host_c = sources[seed]
                    base = jax.device_put(host_base, named_sharding_tree(host_base, mesh))
                    c = jax.device_put(host_c, trainable_layout(host_c, mesh))
                    recovered = materializer(base,c)
                    dense_arm = arms[1]
                    masters, frozen = split_decoder(recovered) if experiment == 104 else split_mamba(recovered)
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
                    tx = optimizer(parameters)
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
                        if row["step"] < 2048:
                            row.pop("final", None)
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
                    other = rows.get(arms[1] if arm == arms[0] else arms[0], {})
                    if other.get("canonical_start_checked_v2"):
                        for name in row["start"]:
                            if max(abs(x-y) for x,y in zip(row["start"][name]["windows"], other["start"][name]["windows"])) > 1e-4:
                                raise ValueError("initial adapter/decoder NLL differs >1e-4")
                    if arm not in compiled_steps:
                        step = make_step(model, tx, arm, columns, config.mamba.head_dim,
                                         (layout, state_layout, fixed_layout, batch, replicated_sharding(mesh)))
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
                    try:
                        proposed, proposed_state = parameters, state
                        while row["step"] < target and time.monotonic() < deadline:
                            proposed, proposed_state, health = step(parameters, state, fixed,
                                jax.device_put(tokens["train"][row["step"]][None], batch))
                            health = {k: np.asarray(v).item() for k,v in jax.device_get(health).items()}
                            if not health["finite"]:
                                row.update(failed=True, error="nonfinite proposal rejected", failure_step=row["step"]+1)
                                proposed, proposed_state = parameters, state
                                break
                            parameters, state = proposed, proposed_state
                            row["step"] += 1
                            row["last_health"] = health
                            row["norm_only_overflow_count"] = row.get("norm_only_overflow_count", 0) + int(not health.get("naive_norm_finite", True))
                            row["last_learning_rate"] = float(schedule(row["step"] - 1))
                            if row["step"] % 256 == 0:
                                row["diagnostics"].append(dict(step=row["step"], health=health, evaluation_layout="canonical-v2",
                                    validation=evaluate(model, current(), tokens["validation"], batch)))
                                print(f"EXP{experiment} seed={seed} arm={arm} step={row['step']} CE={health['loss']:.5f}", flush=True)
                            if row["step"] % 512 == 0:
                                save()
                                persist(sync=True)
                        if row["step"] == 2048 and not row.get("failed"):
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
                    del parameters, state, fixed, step, proposed, proposed_state, current
                    gc.collect()
                    persist(sync=True)
        result["status"] = "completed" if aggregate(result)["complete"] else "branch_failure"
    except TimeoutError as exc:
        result.update(status="deadline_partial", error=str(exc))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        persist(sync=True, force=True)
        raise
    persist(sync=True, force=True)
    return result


if __name__ == "__main__":
    main()
