"""EXP176: bounded-RAM PyTorch CE continuation from received EXP175 CURRENT511."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from extent.broad_corpus import digest_file, write_json


PREFIX = "experiments/exp176-torch-full-ce-v1"
CONTEXT, EFFECTIVE, SOURCE_STEP = 1024, 16, 511
INPUTS = CONTEXT * EFFECTIVE
REFERENCE = Path(__file__).parent / "results/EXP-176-source-reference.json"

def history_at_checkpoint(previous, step):
    return dict(
        history=[row for row in previous.get("history", []) if row["step"] <= step],
        evaluations=[row for row in previous.get("evaluations", []) if row["step"] <= step],
        excluded_updates=max(0, previous.get("step", step) - step),
    )

def compatible_contract(saved, expected):
    if saved == expected:
        return True
    # 60f93a3 differs only in diagnostic sample-index arithmetic. Model,
    # loss, optimizer and recurrence operations are unchanged.
    previous = dict(expected, code=dict(expected["code"]))
    previous["code"]["extent/torch_continue_176.py"] = "624a9a35a5b4cac433d37b7ebf1e4178671f031683cfb770b81363b4619074da"
    return saved == previous


def transport_probe(root, policy):
    import io
    from huggingface_hub import HfApi, hf_hub_download
    data = (b"Extent176 durable transport\n" * 4096)
    repo = os.environ.get("EXTENT_HF_CHECKPOINT_REPO", "levkremlev888/extent")
    api = HfApi(token=os.environ["HF_TOKEN"])
    limit = min(policy.deadline, time.time() + 900)
    uploaded = policy.call(
        api.upload_file, repo_id=repo, repo_type="dataset", path_in_repo=PREFIX + "/preflight.bin",
        path_or_fileobj=io.BytesIO(data), commit_message="EXP176 bounded transport check", _retry_deadline=limit,
    )
    file = policy.call(hf_hub_download, repo_id=repo, repo_type="dataset",
                       filename=PREFIX + "/preflight.bin", revision=uploaded.oid,
                       local_dir=str(root / "transport-probe"), token=os.environ["HF_TOKEN"],
                       _retry_deadline=limit)
    if Path(file).read_bytes() != data:
        raise ValueError("HF transport round-trip differs")
    return dict(verified=True, bytes=len(data), revision=uploaded.oid)


def contract(reference):
    from extent.quality_172 import SOURCE, SOURCE_REVISION as REVISION
    base = Path(__file__).parent
    return dict(
        experiment=176, format="torch-full-ce-v1", source=reference,
        original_model=SOURCE, original_revision=REVISION,
        context=CONTEXT, effective_batch=EFFECTIVE, loss="next-token CE",
        scope="all active gqa7 student weights", dtype="FP32 master/Adam, BF16 projection GEMMs, FP32 state",
        lr=1e-6, optimizer="AdamW(.9,.95), eps1e-8, wd0, clip1",
        code={name: digest_file(base / name) for name in
              ("extent/torch_continue_176.py", "extent/torch_clean_170.py")},
    )


def prepare(root, work, policy):
    from huggingface_hub import hf_hub_download
    from extent.quality_172 import SOURCE, SOURCE_REVISION as REVISION
    from extent.broad_corpus import PackedCorpus

    reference = json.loads(REFERENCE.read_text())
    source = root / "source175"
    source.mkdir(exist_ok=True)

    def get(repo, name, revision, destination, repo_type="dataset"):
        downloaded = Path(policy.call(
            hf_hub_download, repo_id=repo, filename=name, revision=revision, repo_type=repo_type,
            local_dir=str(destination), token=os.environ.get("HF_TOKEN"),
        ))
        return downloaded

    get(SOURCE, "config.json", REVISION, root / "config", repo_type="model")
    if not (root / "CURRENT.pt").exists():
        from engine_torch170 import restore_remote
        restore_remote(root, policy, prefix=PREFIX)
        if not (root / "CURRENT.pt").exists() and (work / "remote-current176.json").exists():
            raise ValueError("Previously verified176 CURRENT unavailable; inspect remote descriptor")
        if (root / "CURRENT.pt").exists():
            subprocess.run([sys.executable, "-m", "engine_torch176", "--stage", "describe",
                            "--work-dir", str(work), "--ram-dir", str(root)], check=True)

    metadata_path = get(reference["repo"], reference["prefix"] + "/checkpoint.json",
                        reference["revision"], source)
    metadata = json.loads(metadata_path.read_text())
    actual = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (actual != reference["metadata_sha256"] or metadata["step"] != SOURCE_STEP
            or metadata["generation"] != reference["generation"]):
        raise ValueError("Source175 metadata differs from received checkpoint")
    if not (root / "CURRENT.pt").exists():
        print("EXP176 restoring pinned full175 parameters and Adam", flush=True)
        for row in metadata["leaves"]:
            for chunk in row["chunks"]:
                name = Path(chunk["file"])
                if name.is_absolute() or ".." in name.parts or name.parts[0] != metadata["generation"]:
                    raise ValueError("Unsafe pinned source chunk")
                file = get(reference["repo"], reference["prefix"] + "/" + chunk["file"],
                           reference["revision"], source)
                if file.stat().st_size != chunk["bytes"] or digest_file(file) != chunk["sha256"]:
                    raise ValueError("Pinned source checksum mismatch")
        config_file = get(SOURCE, "config.json", REVISION, root / "config", repo_type="model")
        if config_file.name != "config.json":
            raise ValueError("Unexpected config layout")
        command = [sys.executable, "-u", "-m", "engine_torch176", "--stage", "convert",
                   "--work-dir", str(work), "--ram-dir", str(root)]
        subprocess.run(command, check=True)
        # Exact pinned source is independently available remotely. The converted
        # complete CURRENT and descriptor have already been written by the child.
        current = json.loads((root / "CURRENT.json").read_text())
        if current["step"] != SOURCE_STEP or digest_file(root / "CURRENT.pt") != current["sha256"]:
            raise ValueError("Converted CURRENT verification failed")
        if source.is_symlink() or source.resolve().parent != root:
            raise ValueError("Unsafe import cache")
        shutil.rmtree(source)

    pack_prefix = "experiments/exp169-data-c4-edu600m-v1"
    bundle = root / "bundle"
    inventory_file = get(reference["repo"], pack_prefix + "/inventory.json", reference["revision"], bundle)
    inventory = json.loads(inventory_file.read_text())
    corpus = root / "corpus"
    corpus.mkdir(exist_ok=True)
    for name in ("manifest.json", "train.bin", "validation.bin"):
        relative = "data-pack-edu-600m/" + name
        expected = inventory["files"][relative]
        target = corpus / name
        if target.exists() and target.stat().st_size == expected["bytes"] and digest_file(target) == expected["sha256"]:
            continue
        file = get(reference["repo"], pack_prefix + "/" + relative, reference["revision"], bundle)
        if file.stat().st_size != expected["bytes"] or digest_file(file) != expected["sha256"]:
            raise ValueError("Corpus checksum mismatch")
        file.replace(target)
    if PackedCorpus(corpus).identity != reference["corpus_identity"]:
        raise ValueError("Corpus identity differs from EXP175")
    write_json(work / "prepared176.json", dict(source_verified=True, source_step=SOURCE_STEP,
                                             source_revision=reference["revision"]))


def convert(root):
    from extent.torch_continue_176 import FullStudent, convert_source, load_config, save_current
    reference = json.loads(REFERENCE.read_text())
    directory = root / "source175" / reference["prefix"]
    metadata = json.loads((directory / "checkpoint.json").read_text())
    model = FullStudent(load_config(root / "config"), backend="chunked")
    optimizer, cursor = convert_source(model, metadata, directory)
    if cursor != reference["cursor"]:
        raise ValueError("Converted cursor mismatch")
    save_current(root, dict(version=1, contract=contract(reference), step=SOURCE_STEP,
                            cursor=cursor, model=model.state_dict(), optimizer=optimizer.state_dict()))
    print("EXP176 source conversion completed; active parameters=", sum(p.numel() for p in model.parameters()), flush=True)

def describe_current(root):
    import torch
    reference = json.loads(REFERENCE.read_text())
    saved = torch.load(root / "CURRENT.pt", map_location="cpu", mmap=True, weights_only=True)
    if saved["contract"] != contract(reference):
        raise ValueError("Restored176 contract differs")
    if saved["cursor"] != reference["cursor"] + (saved["step"] - SOURCE_STEP) * INPUTS:
        raise ValueError("Restored176 cursor differs")
    write_json(root / "CURRENT.json", dict(
        step=saved["step"], cursor=saved["cursor"], contract=saved["contract"],
        sha256=digest_file(root / "CURRENT.pt"), bytes=(root / "CURRENT.pt").stat().st_size,
    ))

def upload_worker(args, root):
    from engine_torch170 import sync_remote
    from extent.hf_download_retry import DownloadPolicy
    snapshot = root / "upload"
    metadata = json.loads((snapshot / "CURRENT.json").read_text())
    policy = DownloadPolicy(root / "transport", deadline=min(args.deadline - 60, time.time() + 1800))
    meta = sync_remote(snapshot, snapshot / "runtime-snapshot.json", policy, prefix=PREFIX,
                       remove_local=False, checkpoint_step=metadata["step"])
    write_json(Path(args.result), dict(status="verified", remote=meta))


def worker(args, root, work):
    import numpy as np
    import torch
    import transformers
    from extent.broad_corpus import PackedCorpus
    from extent.torch_clean_170 import microbatches
    from extent.torch_continue_176 import (
        FullStudent, host_precision_changes, load_config, sample_master_weights, save_current,
    )
    from extent.weekend_storage_176 import disk_guard

    if torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "4.57.1":
        raise ValueError("Use torch2.11.0 and transformers4.57.1")
    result_path = Path(args.result)
    report = dict(status="preflight", contract={"experiment": 176, "format": "torch-full-ce-v1"}, branches={})
    write_json(result_path, report)
    free, total = torch.cuda.mem_get_info()
    budget = min(args.memory_gib * 2**30, free - 5 * 2**30)
    if budget < 40 * 2**30:
        raise MemoryError("Less than40GiB available for full student, leaving5GiB GPU headroom")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.set_float32_matmul_precision("highest")
    reference = json.loads(REFERENCE.read_text())
    expected = contract(reference)
    saved = torch.load(root / "CURRENT.pt", map_location="cpu", mmap=True, weights_only=True)
    if not compatible_contract(saved["contract"], expected):
        raise ValueError("CURRENT contract differs")
    model = FullStudent(load_config(root / "config"), backend=args.backend)
    model.load_state_dict(saved["model"], strict=True, assign=True)
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9, .95), eps=1e-8,
                                 weight_decay=0., fused=True)
    for group in saved["optimizer"]["param_groups"]:
        group.update(fused=True, foreach=False)
    optimizer.load_state_dict(saved["optimizer"])
    step, cursor = int(saved["step"]), int(saved["cursor"])
    if cursor != reference["cursor"] + (step - SOURCE_STEP) * INPUTS:
        raise ValueError("CURRENT cursor is inconsistent")
    del saved
    corpus = PackedCorpus(root / "corpus")
    if corpus.identity != reference["corpus_identity"]:
        raise ValueError("Wrong training corpus")
    report.update(contract=expected, physical_batch=args.batch, backend=args.backend,
                  gpu=torch.cuda.get_device_name(), allocated_budget_bytes=int(budget), pending_slots=["CURRENT"],
                  trainable_parameters=sum(p.numel() for p in model.parameters()))
    branch = dict(step=step, source_step=SOURCE_STEP, accepted_input_tokens=(step - SOURCE_STEP) * INPUTS,
                  selected_batch=args.batch, history=[], evaluations=[])
    previous_path = work / "previous176.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text()).get("branches", {}).get("FULL-CE", {})
        restored_history = history_at_checkpoint(previous, step)
        branch["history"] = restored_history["history"]
        branch["evaluations"] = restored_history["evaluations"]
        report["uncheckpointed_previous_updates_excluded"] = restored_history["excluded_updates"]
    report["branches"]["FULL-CE"] = branch
    write_json(result_path, report)
    started = last_save = time.monotonic()
    origin = step
    checkpoint_safe = True

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def batch(at, count, split="train"):
        values = corpus.arrays[split]
        end = at + count * CONTEXT
        if end > len(values):
            raise EOFError("Fresh corpus exhausted")
        return torch.from_numpy(np.array(values[at:end], dtype=np.int64).reshape(count, CONTEXT)).to("cuda")

    def evaluate(count, label):
        values = []
        model.eval()
        with torch.no_grad(), autocast():
            for index in range(count):
                if (work / "FAST_STOP").exists():
                    model.train()
                    return None
                values.append(float(model(batch(index * CONTEXT, 1, "validation"))))
        model.train()
        row = dict(step=step, label=label, windows=count, nll=float(np.mean(values)), per_window_nll=values)
        if not math.isfinite(row["nll"]):
            raise FloatingPointError("Nonfinite validation NLL")
        branch["evaluations"].append(row)
        branch.update(last_validation_nll=row["nll"], validation_windows=count, validation_step=step)
        write_json(result_path, report)
        print("EXP176 validation", label, "step", step, "NLL", row["nll"], flush=True)
        return row

    def update(at, capture_precision=False):
        nonlocal checkpoint_safe
        samples_before = sample_master_weights(model) if capture_precision else None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        events = []
        def stamp():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        for offset, count in microbatches(EFFECTIVE, args.batch):
            forward_start = stamp()
            with autocast():
                loss = model(batch(at + offset * CONTEXT, count))
            forward_end = stamp()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite CE before optimizer")
            backward_start = stamp()
            (loss * (count / EFFECTIVE)).backward()
            backward_end = stamp()
            events.extend((("student_forward", forward_start, forward_end),
                           ("student_backward", backward_start, backward_end)))
            loss_sum += float(loss.detach()) * (count / EFFECTIVE)
        clip_start = stamp()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
        clip_end = stamp()
        optimizer_start = stamp()
        checkpoint_safe = False
        optimizer.step()
        check = list(model.parameters())
        for state in optimizer.state.values():
            check.extend((state["exp_avg"], state["exp_avg_sq"]))
        if not bool(torch.stack([torch.isfinite(value).all() for value in check]).all()):
            raise FloatingPointError("Nonfinite parameters/Adam; previous complete CURRENT retained")
        checkpoint_safe = True
        if samples_before is not None:
            report["first_update_precision_samples"] = host_precision_changes(samples_before, sample_master_weights(model))
        optimizer_end = stamp()
        events.extend((("clip", clip_start, clip_end), ("optimizer", optimizer_start, optimizer_end)))
        torch.cuda.synchronize()
        phases = {}
        for name, start, end in events:
            phases[name] = phases.get(name, 0.) + start.elapsed_time(end) / 1000
        return loss_sum, float(norm), phases

    def snapshot():
        file = save_current(root, dict(version=1, contract=expected, step=step, cursor=cursor,
                                       model=model.state_dict(), optimizer=optimizer.state_dict()))
        branch["local_checkpoint_step"] = step
        report["local_checkpoint_step"] = step
        write_json(result_path, report)
        return file

    if args.stage == "probe":
        # Real learned recurrence, same projections/precision and all parameters.
        mixer = model.mixers["0"]
        generator = torch.Generator(device="cuda").manual_seed(176)
        inputs = torch.randn(1, 128, model.qwen.config.hidden_size, device="cuda",
                             dtype=torch.bfloat16, generator=generator) * .1
        params = tuple(mixer.parameters())
        with autocast():
            reference_output = mixer(inputs, reference=True)
            actual_output = mixer(inputs)
            reference_grad = torch.autograd.grad(reference_output.float().square().mean(), params)
            actual_grad = torch.autograd.grad(actual_output.float().square().mean(), params)
        rel = float((actual_output.float() - reference_output.float()).norm() / reference_output.float().norm().clamp_min(1e-7))
        grad_rel = max(float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-5))
                       for a, b in zip(actual_grad, reference_grad))
        if rel > .01 or grad_rel > .05 or not math.isfinite(rel + grad_rel):
            raise ValueError("CUDA recurrence output/gradient gate failed")
        del reference_grad, actual_grad, reference_output, actual_output
        replay = evaluate(4, "source_replay")
        if replay is None:
            report.update(status="requested_stop", accepted_updates=0)
            write_json(result_path, report)
            return
        if step == SOURCE_STEP:
            delta = np.array(replay["per_window_nll"]) - np.array(reference["validation_window_ce"][:4])
            if abs(delta.mean()) > .02 or np.max(np.abs(delta)) > .08:
                raise ValueError("Ported EXP175 source NLL differs")
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            before = time.monotonic()
            loss, _, phases = update(cursor, capture_precision=not times)
            times.append(time.monotonic() - before)
        report.update(status="measured", finite=math.isfinite(loss), accepted_updates=0,
                      engineering_input_presentations=3 * INPUTS,
                      input_tokens_per_second=INPUTS / float(np.median(times[1:])),
                      seconds_per_update=times, last_cuda_phases_seconds=phases,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      kernel_gate=dict(seed=176, output_relative_l2=rel, gradient_max_relative_l2=grad_rel))
        write_json(result_path, report)
        return

    initial = evaluate(128, "initial")
    if initial is None:
        report["status"] = "fast_paused"
        write_json(result_path, report)
        return
    if step == SOURCE_STEP and abs(initial["nll"] - reference["expected_validation_nll"]) > .02:
        raise ValueError("Expanded source replay differs; training not started")
    report["status"] = "running"
    write_json(result_path, report)
    try:
        while time.time() < args.deadline - args.save_reserve_minutes * 60:
            if (work / "STOP").exists() or (work / "FAST_STOP").exists():
                break
            if cursor + INPUTS > len(corpus.arrays["train"]):
                report["status"] = "corpus_exhausted"
                break
            if step % 32 == 0:
                try:
                    disk_guard(work, args.min_free_gib)
                except OSError:
                    report["status"] = "storage_limit"
                    break
            begin = time.monotonic()
            loss, norm, phases = update(cursor, capture_precision=step == origin)
            seconds = time.monotonic() - begin
            step += 1
            cursor += INPUTS
            branch.update(step=step, accepted_input_tokens=(step - SOURCE_STEP) * INPUTS)
            branch["invocation_input_tokens"] = (step - origin) * INPUTS
            branch["history"].append(dict(step=step, ce=loss, loss=loss, lr=1e-6,
                                          seconds=seconds, grad_norm=norm, input_tokens_per_second=INPUTS / seconds,
                                          cuda_phases_seconds=phases))
            if step % 32 == 0:
                report["launcher_elapsed_hours"] = (time.monotonic() - started) / 3600
                branch["loop_tokens_per_second"] = (step - origin) * INPUTS / (time.monotonic() - started)
                ack = root / "remote-ack.json"
                if ack.exists():
                    branch["durable_step"] = json.loads(ack.read_text())["step"]
                write_json(result_path, report)
                print("EXP176 step", step, "new_inputs", branch["accepted_input_tokens"],
                      "CE", loss, "loop_tok/s", round(branch["loop_tokens_per_second"]), flush=True)
            if step % 512 == 0:
                evaluate(128, "periodic")
            if step == origin + 1 or time.monotonic() - last_save > 1800:
                snapshot()
                last_save = time.monotonic()
        if report["status"] not in ("corpus_exhausted", "storage_limit"):
            report["status"] = "fast_paused" if (work / "FAST_STOP").exists() else "deadline_partial"
            if (work / "STOP").exists():
                report["status"] = "stopped"
        if report["status"] != "fast_paused":
            evaluate(128, "final")
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:600])
        raise
    finally:
        if checkpoint_safe:
            snapshot()
        else:
            report["previous_current_retained"] = True
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(result_path, report)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work-dir", required=True)
    end = cli.add_mutually_exclusive_group()
    end.add_argument("--until")
    end.add_argument("--max-wall-hours", type=float)
    cli.add_argument("--memory-gib", type=float, default=90)
    cli.add_argument("--min-free-gib", type=float, default=30)
    cli.add_argument("--max-ram-files-gib", type=float, default=120)
    cli.add_argument("--max-new-hf-gib", type=float, default=64)
    cli.add_argument("--save-reserve-minutes", type=float, default=18)
    cli.add_argument("--planner-minutes", type=float, default=35)
    cli.add_argument("--probe-timeout-minutes", type=float, default=8)
    cli.add_argument("--preflight-only", action="store_true")
    cli.add_argument("--clear-stop", action="store_true")
    cli.add_argument("--stage", choices=("parent", "convert", "describe", "upload", "probe", "train"), default="parent")
    cli.add_argument("--ram-dir")
    cli.add_argument("--result")
    cli.add_argument("--batch", type=int, default=1)
    cli.add_argument("--backend", choices=("chunked", "chunked-compiled", "tilelang"), default="chunked-compiled")
    cli.add_argument("--deadline", type=float)
    cli.add_argument("--gpu", default="0")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.stage != "parent":
        try:
            if args.stage == "convert":
                convert(Path(args.ram_dir))
            elif args.stage == "describe":
                describe_current(Path(args.ram_dir))
            elif args.stage == "upload":
                upload_worker(args, Path(args.ram_dir))
            else:
                worker(args, Path(args.ram_dir), work)
        except BaseException as exc:
            if args.result:
                path = Path(args.result)
                report = json.loads(path.read_text()) if path.exists() else {}
                report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:600],
                              traceback=traceback.format_exc())
                write_json(path, report)
            raise
        return

    from filelock import FileLock
    from extent.hf_download_retry import DownloadPolicy, install_download_retries
    from extent.isolated_tpu_planner import grow, run_probe, terminate_group
    from extent.weekend_storage_176 import (
        check_ram_growth, cleanup_after_verified_sync, create_ram_workspace, disk_guard, upload_snapshot,
    )
    from engine_torch170 import sync_remote

    if args.until:
        target = datetime.fromisoformat(args.until)
        if target.tzinfo is None:
            raise ValueError("--until must include timezone")
        deadline = target.timestamp()
    elif args.max_wall_hours:
        deadline = time.time() + args.max_wall_hours * 3600
    else:
        raise ValueError("Specify --until or --max-wall-hours")
    if not 3600 < deadline - time.time() < 96 * 3600:
        raise ValueError("Need between one and96hours remaining")
    for name in ("memory_gib", "min_free_gib", "max_ram_files_gib", "max_new_hf_gib",
                 "save_reserve_minutes", "planner_minutes", "probe_timeout_minutes"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError("Positive finite " + name + " required")
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN is required")
    if not os.environ.get("TELEGRAM_CHAT_ID") and os.environ.get("TELEGRAM_USER_ID"):
        os.environ["TELEGRAM_CHAT_ID"] = os.environ["TELEGRAM_USER_ID"]
    if args.save_reserve_minutes < 18:
        raise ValueError("Reserve at least18minutes for verified final upload")
    (work / "results").mkdir(exist_ok=True)
    report_path = work / "results/runtime-torch176.json"
    launch_path = work / "results/runtime-torch176-launch.json"
    launch = dict(status="preparing", contract={"experiment": 176}, deadline_unix=deadline,
                  started_unix=time.time(), probes=[])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
    os.environ.setdefault("EXTENT_HF_CHECKPOINT_REPO", "levkremlev888/extent")

    def notify(message):
        from extent.notifications import send_telegram_message
        try:
            send_telegram_message("Extent EXP176: " + message)
        except Exception as exc:
            print("telegram_error", type(exc).__name__, flush=True)

    with FileLock(str(work / "server.lock"), timeout=0):
        for name in ("STOP", "FAST_STOP"):
            if args.clear_stop:
                (work / name).unlink(missing_ok=True)
            elif (work / name).exists():
                raise ValueError(name + " exists; use --clear-stop")
        disk_guard(work, args.min_free_gib)
        root = create_ram_workspace(work)
        for name in ("tmp", "cache", "probes", "cache/xet", "cache/inductor", "cache/triton",
                     "cache/extensions", "cache/cuda", "cache/xdg"):
            (root / name).mkdir(parents=True, exist_ok=True)
        os.environ.update(TMPDIR=str(root / "tmp"), HF_HOME=str(root / "cache"),
                          HF_XET_CACHE=str(root / "cache/xet"),
                          TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
                          TRITON_CACHE_DIR=str(root / "cache/triton"),
                          TORCH_EXTENSIONS_DIR=str(root / "cache/extensions"),
                          CUDA_CACHE_PATH=str(root / "cache/cuda"),
                          XDG_CACHE_HOME=str(root / "cache/xdg"),
                          EXTENT_HF_RETRY_STATE_DIR=str(root / "transport"),
                          EXTENT_HF_DOWNLOAD_DEADLINE_UNIX=str(deadline - args.save_reserve_minutes * 60),
                          EXTENT_HF_SAVE_DEADLINE_UNIX=str(deadline - 60))
        policy = DownloadPolicy(root / "transport", deadline=deadline - 60)
        install_download_retries()
        if report_path.exists():
            previous_report = json.loads(report_path.read_text())
            if previous_report.get("branches", {}).get("FULL-CE"):
                archives = work / "results/archive"
                archives.mkdir(exist_ok=True)
                shutil.copy2(report_path, archives / f"runtime176-{time.time_ns()}.json")
                shutil.copy2(report_path, work / "previous176.json")
        write_json(launch_path, launch)
        write_json(report_path, dict(status="preparing", contract={"experiment": 176}, branches={}))
        previous = {sig: signal.signal(sig, lambda _s, _f: (work / "STOP").touch())
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        child = None
        budget_path = work / "remote-budget176.json"
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(
            attempted_bytes=0, last_upload_unix=0, last_attempt_unix=time.time(),
        )
        write_json(budget_path, budget)
        verified = None
        uploader = None
        upload_metadata = None
        upload_started = 0.

        def stage_upload(final=False):
            snapshot, metadata = upload_snapshot(root)
            size = metadata["bytes"]
            remaining = args.max_new_hf_gib * 2**30 - budget["attempted_bytes"]
            if remaining < size * (1 if final else 2):
                print("HF budget reserved for final CURRENT", flush=True)
                return None
            budget["attempted_bytes"] += size
            write_json(budget_path, budget)
            snapshot_report = snapshot / "runtime-snapshot.json"
            snapshot_value = json.loads(report_path.read_text())
            snapshot_value["remote_snapshot_step"] = metadata["step"]
            write_json(snapshot_report, snapshot_value)
            return snapshot, metadata

        def accept_upload(meta, metadata):
            nonlocal verified
            if meta["sha256"] != metadata["sha256"]:
                raise ValueError("Uploaded snapshot SHA differs")
            verified = meta
            budget["last_upload_unix"] = time.time()
            write_json(budget_path, budget)
            write_json(work / "remote-current176.json", meta)
            write_json(root / "remote-ack.json", dict(step=metadata["step"], sha256=meta["sha256"]))
            root.joinpath("upload/CURRENT.pt").unlink(missing_ok=True)

        def synchronize(final=False):
            staged = stage_upload(final)
            if staged is None:
                return False
            snapshot, metadata = staged
            meta = sync_remote(snapshot, snapshot / "runtime-snapshot.json", policy, prefix=PREFIX, remove_local=False,
                               checkpoint_step=metadata["step"])
            accept_upload(meta, metadata)
            return True

        def command(stage, result, batch, backend):
            return [sys.executable, "-u", "-m", "engine_torch176", "--stage", stage,
                    "--work-dir", str(work), "--ram-dir", str(root), "--result", str(result),
                    "--batch", str(batch), "--backend", backend, "--deadline", str(deadline),
                    "--memory-gib", str(args.memory_gib), "--min-free-gib", str(args.min_free_gib),
                    "--save-reserve-minutes", str(args.save_reserve_minutes)]

        try:
            if os.environ.get("TELEGRAM_BOT_TOKEN") and (os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_USER_ID")):
                from extent.status_autostart import ensure_listener
                try:
                    ensure_listener(work, "results/runtime-torch176.json", force_restart=True)
                except Exception as exc:
                    launch["status_bot_error"] = type(exc).__name__
                    print("status_bot_error", type(exc).__name__, flush=True)
            notify("started; PyTorch full CE from175; RAM staging; disk reserve30GiB")
            launch["transport_probe"] = transport_probe(root, policy)
            write_json(launch_path, launch)
            prepare(root, work, policy)
            descriptor = json.loads((root / "CURRENT.json").read_text())
            if digest_file(root / "CURRENT.pt") != descriptor["sha256"]:
                raise ValueError("CURRENT bytes differ from its atomic descriptor")
            launch["starting_checkpoint_step"] = descriptor["step"]
            check_ram_growth(root, args.max_ram_files_gib)
            stop = lambda: (work / "STOP").exists() or (work / "FAST_STOP").exists()
            fits = []
            cache_path = work / "planner-selected176.json"
            current_code = contract(json.loads(REFERENCE.read_text()))["code"]
            if cache_path.exists() and not stop():
                cached = json.loads(cache_path.read_text())
                if cached.get("code") == current_code:
                    candidate = cached["selected"]
                    result = root / "probes/cached-plan.json"
                    row = run_probe(
                        command("probe", result, candidate["batch"], candidate["backend"]),
                        result, result.with_suffix(".log"), args.probe_timeout_minutes * 60, stop=stop,
                    )
                    row.update(batch=candidate["batch"], backend=candidate["backend"])
                    launch["probes"].append(row)
                    if row["status"] == "measured":
                        fits.append(row)
            planner_end = min(time.monotonic() + args.planner_minutes * 60,
                              time.monotonic() + deadline - time.time() - args.save_reserve_minutes * 60)
            variants = ["chunked-compiled", "chunked"]
            if importlib.util.find_spec("mamba_ssm") is not None:
                variants.append("tilelang")
            for number, backend in enumerate([] if fits else variants):
                limit = time.monotonic() + max(0., planner_end - time.monotonic()) / (len(variants) - number)
                def probe(batch):
                    result = root / "probes" / f"{backend}-b{batch}.json"
                    row = run_probe(command("probe", result, batch, backend), result, result.with_suffix(".log"),
                                    min(args.probe_timeout_minutes * 60, max(.1, limit - time.monotonic())), stop=stop)
                    if row["status"] != "measured" and result.exists():
                        row["report"] = json.loads(result.read_text())
                    return dict(row, backend=backend)
                outcome = grow(probe, first=1, max_batch=8, deadline=limit)
                launch["probes"].extend(outcome["trials"])
                fits.extend(row for row in outcome["trials"] if row["status"] == "measured")
                # Keep probe failures and timings on SSD for audit after RAM cleanup.
                for file in (root / "probes").iterdir():
                    shutil.copy2(file, work / "results" / file.name)
                write_json(launch_path, launch)
            if stop():
                launch["status"] = "fast_paused" if (work / "FAST_STOP").exists() else "stopped"
                write_json(report_path, dict(status=launch["status"], contract={"experiment": 176}, branches={}))
                notify("stopped during preflight; GPU workers exited; RAM CURRENT retained")
                return
            if not fits:
                raise RuntimeError("No verified full-CE CUDA fit; inspect results/*b*.log")
            best = max(fits, key=lambda row: row["report"]["input_tokens_per_second"])
            for file in (root / "probes").iterdir():
                shutil.copy2(file, work / "results" / file.name)
            write_json(cache_path, dict(code=current_code, selected=best))
            launch.update(status="preflight_completed", selected=best)
            write_json(launch_path, launch)
            notify(f"preflight passed; {best['backend']} batch={best['batch']}; measured={best['report']['input_tokens_per_second']:.0f}tok/s")
            if args.preflight_only or stop():
                write_json(report_path, dict(status="preflight_completed", contract={"experiment": 176}, planner=best))
                return
            with (work / "runtime-torch176.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command("train", report_path, best["batch"], best["backend"]),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                last_check = 0.
                while child.poll() is None:
                    if time.time() >= deadline:
                        terminate_group(child)
                        raise TimeoutError("Hard deadline reached; local CURRENT retained")
                    if time.monotonic() - last_check > 60:
                        try:
                            disk_guard(work, args.min_free_gib)
                            check_ram_growth(root, args.max_ram_files_gib)
                        except (OSError, MemoryError):
                            (work / "STOP").touch()
                        if report_path.exists():
                            progress = json.loads(report_path.read_text()).get("branches", {}).get("FULL-CE", {})
                            print("EXP176 heartbeat step=", progress.get("step"),
                                  "new_inputs=", progress.get("accepted_input_tokens"),
                                  "NLL=", progress.get("last_validation_nll"), flush=True)
                        last_check = time.monotonic()
                    if (time.time() < deadline - args.save_reserve_minutes * 60
                            and not stop() and uploader is None
                            and time.time() - budget.get("last_attempt_unix", budget["last_upload_unix"]) > 24 * 3600
                            and (root / "CURRENT.json").exists()):
                        budget["last_attempt_unix"] = time.time()
                        write_json(budget_path, budget)
                        try:
                            staged = stage_upload()
                            if staged is not None:
                                _, upload_metadata = staged
                                output = root / "upload-result.json"
                                output.unlink(missing_ok=True)
                                with (work / "upload176.log").open("a") as upload_log:
                                    uploader = subprocess.Popen(command("upload", output, 1, "chunked"),
                                        stdout=upload_log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                                upload_started = time.monotonic()
                        except Exception as exc:
                            launch["periodic_sync_error"] = type(exc).__name__
                            write_json(launch_path, launch)
                    if uploader is not None:
                        if uploader.poll() is None and time.monotonic() - upload_started > 1800:
                            terminate_group(uploader)
                        if uploader.poll() is not None:
                            try:
                                if uploader.returncode:
                                    raise RuntimeError("Periodic upload worker failed")
                                upload_result = json.loads((root / "upload-result.json").read_text())
                                accept_upload(upload_result["remote"], upload_metadata)
                            except Exception as exc:
                                launch["periodic_sync_error"] = type(exc).__name__
                                write_json(launch_path, launch)
                            uploader = None
                    time.sleep(1)
            value = json.loads(report_path.read_text())
            if child.returncode != 0:
                raise RuntimeError("CUDA worker failed; latest complete local CURRENT retained")
            if value["status"] == "fast_paused":
                if uploader is not None:
                    terminate_group(uploader)
                notify("fast pause ready; GPU worker exited; current state retained in RAM")
                launch["status"] = "fast_paused"
                return
            if uploader is not None:
                # The final snapshot supersedes an unfinished periodic transfer.
                terminate_group(uploader)
            if not synchronize(final=True):
                raise RuntimeError("Final HF budget insufficient; latest local CURRENT retained")
            value.update(hf_verified=True, remote=verified)
            value["pending_slots"] = []
            value["branches"]["FULL-CE"]["durable_step"] = value["local_checkpoint_step"]
            cleanup_after_verified_sync(root, work, verified["sha256"])
            value["ram_cleanup_verified"] = True
            write_json(report_path, value)
            launch["status"] = "completed"
            notify("finished, HF verified, RAM caches/states removed; report=" + str(report_path))
        except BaseException as exc:
            launch.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:600],
                          traceback=traceback.format_exc())
            notify("failed; owned RAM recovery files retained; inspect runtime-torch176-launch.json")
            raise
        finally:
            if child is not None and child.poll() is None:
                terminate_group(child)
            if uploader is not None and uploader.poll() is None:
                terminate_group(uploader)
            launch["finished_unix"] = time.time()
            launch["elapsed_seconds"] = launch["finished_unix"] - launch["started_unix"]
            if report_path.exists() and "starting_checkpoint_step" in launch:
                current = json.loads(report_path.read_text()).get("branches", {}).get("FULL-CE", {})
                if current.get("step", 0) >= launch["starting_checkpoint_step"]:
                    inputs = (current["step"] - launch["starting_checkpoint_step"]) * INPUTS
                    launch["invocation_accepted_input_tokens"] = inputs
                    launch["overall_input_tokens_per_second"] = inputs / max(launch["elapsed_seconds"], 1e-6)
            write_json(launch_path, launch)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
