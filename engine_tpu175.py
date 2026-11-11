"""EXP175: two-hour full-student objective diagnosis from clean EXP171."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

from engine_tpu173 import environment, prepare, wait_child
from extent.broad_corpus import digest_file, write_json


PREFIX = "experiments/exp175-objective-diagnostic-v1"
CONTEXT = 1024
EFFECTIVE = 16
INPUTS = CONTEXT * EFFECTIVE
SOURCE_CURSOR = 83_886_080
ARMS = ("ce", "kl", "mixed")


def phase(root, name, **details):
    write_json(root / "phase.json", dict(phase=name, unix=time.time(), pid=os.getpid(), **details))
    print("EXP175", name, details, flush=True)


def worker(args, root):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from extent.broad_corpus import PackedCorpus
    from extent.clean_tpu_171 import initialize_host, layout, optimizer
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.hf_download_retry import install_download_retries
    from extent.quality_172 import row_array, validate_manifest
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.training_diagnostic_175 import (
        WEIGHTS, gradient_diagnostics, make_objective, update_diagnostics,
    )

    report_path = Path(args.report)
    report = dict(status="preflight", arm=args.arm, step=0, history=[], evaluations=[],
                  accepted_input_tokens=0, physical_batch=args.batch, effective_batch=EFFECTIVE)
    write_json(report_path, report)
    install_download_retries()
    phase(root, "load_pinned_source", arm=args.arm)
    config, teacher_config, bank, layers, last, embedding, norm = initialize_host(
        root / "qwen", lambda name: phase(root, name),
    )
    manifest = root / "bank/checkpoint.json"
    for row in validate_manifest(json.loads(manifest.read_text())):
        node = bank
        for key in row["path"][1:-1]:
            node = node[key]
        value = row_array(row, root / "bank")
        key = row["path"][-1]
        if value.shape != node[key].shape:
            raise ValueError("Source bank shape mismatch")
        node[key] = value

    devices = jax.devices("tpu")
    if len(devices) != 8 or jax.process_count() != 1:
        raise ValueError("Eight local TPU chips required")
    mesh = Mesh(np.array(devices), ("model",))
    rep = NamedSharding(mesh, PartitionSpec())
    jax.config.update("jax_compilation_cache_dir", str(root / "compile-cache"))
    donor_layout = (layout(layers, mesh), *(layout(x, mesh, stacked=False) for x in (last, embedding, norm)))
    donor = jax.device_put((layers, last, embedding, norm), donor_layout)
    parameter_layout = dict(mamba=layout(bank, mesh), qwen=donor_layout)
    parameters = jax.device_put(dict(
        mamba=bank, qwen=jax.tree.map(lambda x: x.astype(np.float32), (layers, last, embedding, norm)),
    ), parameter_layout)
    tx = optimizer()
    state = tx.init(parameters)
    state_layout = layout(state, mesh, stacked=False)
    state = jax.device_put(state, state_layout)
    corpus = PackedCorpus(root / "edu")
    code_dir = Path(__file__).resolve().parent
    weights = jax.device_put(np.asarray(WEIGHTS[args.arm], np.float32), rep)
    contract = dict(
        experiment=175, arm=args.arm, source_step=5120, source_cursor=SOURCE_CURSOR,
        source_manifest_sha256=digest_file(manifest), corpus=corpus.identity,
        scope="all active student weights", teacher="immutable original Qwen",
        loss_weights=WEIGHTS[args.arm], peak_lr=1e-6, warmup_updates=32,
        context=CONTEXT, effective_batch=EFFECTIVE, optimizer="fresh Adam(.9,.95), clip1",
        code={name: digest_file(code_dir / name) for name in (
            "engine_tpu175.py", "extent/training_diagnostic_175.py",
            "extent/end_to_end_174.py", "extent/end_to_end_173.py", "extent/clean_tpu_171.py",
        )},
    )
    # JSON round-trip keeps contract tuples identical to their checkpoint representation.
    contract = json.loads(json.dumps(contract))
    report["contract"] = contract
    hub = artifact_config_from_env(include_kaggle_secrets=False) if args.stage == "continue" else None
    store = RecoveryCheckpointStore(root / "states", PREFIX, hub)
    cursor, step = SOURCE_CURSOR, 0

    def payload():
        return dict(parameters=parameters, optimizer=state, cursor=np.asarray(cursor, np.int64))

    if args.stage == "continue":
        phase(root, "restore_selected_pilot", arm=args.arm)
        restored = store.restore(args.arm, contract, payload())
        if restored is None:
            raise ValueError("Selected pilot checkpoint missing")
        saved, metadata = restored
        step, cursor = metadata["step"], int(saved["cursor"])
        if step < args.steps or cursor != SOURCE_CURSOR + step * INPUTS:
            raise ValueError("Selected checkpoint cursor mismatch")
        parameters = jax.device_put(saved["parameters"], parameter_layout)
        state = jax.device_put(saved["optimizer"], state_layout)
        del saved, restored
        report["local_checkpoint_step"] = step
    report.update(step=step, accepted_input_tokens=step * INPUTS)

    def device(value):
        return jax.device_put(np.asarray(value, np.float32), rep)

    def tokens(at, count, split="train"):
        end = at + count * CONTEXT
        if end > len(corpus.arrays[split]):
            raise EOFError("Corpus exhausted")
        return jax.device_put(np.array(corpus.arrays[split][at:end]).reshape(count, CONTEXT), rep)

    objective = make_objective(config, teacher_config)
    grad_fn = jax.jit(jax.value_and_grad(objective, has_aux=True), out_shardings=((rep, rep), parameter_layout))
    metrics_fn = jax.jit(lambda p, t, x: objective(p, t, x, weights)[1], out_shardings=rep)
    add = jax.jit(lambda a, b: jax.tree.map(lambda x, y: x + y, a, b), out_shardings=parameter_layout)
    average = jax.jit(lambda tree: jax.tree.map(lambda x: x * args.batch / EFFECTIVE, tree),
                      out_shardings=parameter_layout)
    finite = jax.jit(lambda tree: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)
    ])), out_shardings=rep)

    def apply(p, state, grad, rate):
        updates, state = tx.update(grad, state, p)
        return optax.apply_updates(p, jax.tree.map(lambda x: -rate * x, updates)), state

    update = jax.jit(apply, out_shardings=(parameter_layout, state_layout))

    def batch_gradient(p, at, objective_weights):
        grad, values = None, device([0., 0.])
        for offset in range(0, EFFECTIVE, args.batch):
            (_, metrics), part = grad_fn(p, donor, tokens(at + offset * CONTEXT, args.batch), objective_weights)
            grad = part if grad is None else add(grad, part)
            values += metrics * (args.batch / EFFECTIVE)
        grad = average(grad)
        if not bool(finite((grad, values))):
            raise FloatingPointError("Nonfinite gradient or loss")
        return grad, values

    def batch_metrics(p, at):
        values = device([0., 0.])
        for offset in range(0, EFFECTIVE, args.batch):
            values += metrics_fn(p, donor, tokens(at + offset * CONTEXT, args.batch)) * (args.batch / EFFECTIVE)
        return np.asarray(values)

    def complete(p, state, at, rate):
        grad, values = batch_gradient(p, at, weights)
        new, new_state = update(p, state, grad, device(rate))
        if not bool(finite((new, new_state))):
            raise FloatingPointError("Nonfinite proposed optimizer update")
        return new, new_state, np.asarray(values), grad

    def components(values):
        ce, kl = map(float, values)
        return dict(ce=ce, kl=kl, objective=ce * WEIGHTS[args.arm][0] + kl * WEIGHTS[args.arm][1])

    def validate(label, count=128):
        rows = []
        for index in range(count):
            if time.time() > args.deadline - 120:
                break
            values = np.asarray(metrics_fn(parameters, donor, tokens(index * CONTEXT, 1, "validation")))
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("Nonfinite validation metrics")
            rows.append(values)
        entry = dict(step=step, panel=label, windows=len(rows),
                     per_window=[components(row) for row in rows])
        if rows:
            entry.update(components(np.mean(rows, axis=0)))
        report["evaluations"].append(entry)
        write_json(report_path, report)
        phase(root, "validation", step=step, nll=entry.get("ce"), windows=len(rows))
        return entry

    def serial(tree):
        return jax.tree.map(lambda x: np.asarray(x).item(), tree)

    phase(root, "source_replay", stage=args.stage, arm=args.arm)
    initial = validate("start")
    if initial["windows"] != 128:
        raise TimeoutError("Incomplete start validation")
    if step == 0:
        monitor = sum(x["ce"] for x in initial["per_window"][:4]) / 4
        if abs(monitor - 2.58102) > .08:
            raise ValueError("Frozen EXP171 replay mismatch")
    origin = step
    started = time.monotonic()
    last_save = started
    report["status"] = "running"
    write_json(report_path, report)
    try:
        if args.stage == "diagnose":
            phase(root, "CE_KL_gradient_decomposition")
            ce_grad, _ = batch_gradient(parameters, SOURCE_CURSOR, device(WEIGHTS["ce"]))
            kl_grad, _ = batch_gradient(parameters, SOURCE_CURSOR, device(WEIGHTS["kl"]))
            summary_fn = jax.jit(gradient_diagnostics, out_shardings=rep)
            report["gradient_components"] = serial(summary_fn(ce_grad, kl_grad))
            del ce_grad, kl_grad
            report["same_batch_start"] = components(batch_metrics(parameters, SOURCE_CURSOR))
            timings = []
            for repeat in range(32):
                if time.time() >= args.work_deadline or (root / "STOP").exists():
                    break
                began = time.monotonic()
                new, new_state, _, grad = complete(parameters, state, SOURCE_CURSOR, 1e-6)
                timings.append(time.monotonic() - began)
                if repeat == 0:
                    changes_fn = jax.jit(update_diagnostics, out_shardings=rep)
                    report["first_actual_Adam_update"] = serial(changes_fn(parameters, new, grad))
                    report["same_batch_after_first_update"] = components(batch_metrics(new, SOURCE_CURSOR))
                parameters, state = new, new_state
                del grad
            report["diagnostic_training_presentations"] = len(timings) * INPUTS
            report["diagnostic_updates"] = len(timings)
            report["component_gradient_presentations"] = 2 * INPUTS
            report["same_batch_end"] = components(batch_metrics(parameters, SOURCE_CURSOR))
            if len(timings) > 1:
                report["seconds_per_update"] = float(np.median(timings[1:]))
            validate("after_repeated_batch", count=4)
            report["status"] = "diagnostic_completed" if len(timings) == 32 else "diagnostic_partial"
            return

        while (time.time() < args.work_deadline and not (root / "STOP").exists()
               and cursor + INPUTS <= len(corpus.arrays["train"])):
            if args.stage == "pilot" and step >= args.steps:
                break
            before = batch_metrics(parameters, cursor) if step in (0, args.steps - 1) else None
            began = time.monotonic()
            rate = 1e-6 * min(1., (step + 1) / 32)
            new, new_state, metrics, grad = complete(parameters, state, cursor, rate)
            seconds = time.monotonic() - began
            del grad
            if before is not None:
                after = batch_metrics(new, cursor)
                report.setdefault("same_batch_updates", []).append(dict(
                    step=step + 1, cursor=cursor, before=components(before), after=components(after),
                ))
            parameters, state = new, new_state
            step += 1
            cursor += INPUTS
            report["history"].append(dict(step=step, lr=rate, seconds=seconds,
                                          tokens_per_second=INPUTS / seconds, **components(metrics)))
            report.update(step=step, accepted_input_tokens=step * INPUTS,
                          invocation_input_tokens=(step - origin) * INPUTS,
                          loop_tokens_per_second=(step - origin) * INPUTS / (time.monotonic() - started))
            if step % 32 == 0:
                phase(root, "train", arm=args.arm, step=step,
                      tokens_per_second=round(report["loop_tokens_per_second"], 1))
                write_json(report_path, report)
            if step % 64 == 0:
                validate("monitor", count=4)
            if args.stage == "continue" and time.monotonic() - last_save > 900:
                store.save(args.arm, contract, payload(), step)
                report["local_checkpoint_step"] = step
                last_save = time.monotonic()
        final = validate("final")
        if final["windows"] == 128:
            report["final_validation_nll"] = final["ce"]
            report["final_validation_kl"] = final["kl"]
        if args.stage == "pilot":
            report["status"] = "pilot_completed" if step == args.steps and final["windows"] == 128 else "pilot_partial"
        else:
            report["status"] = "stopped" if (root / "STOP").exists() else "deadline_partial"
            if cursor + INPUTS > len(corpus.arrays["train"]):
                report["status"] = "corpus_exhausted"
    except Exception as exc:
        report.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        write_json(report_path, report)
        if args.stage in ("pilot", "continue"):
            phase(root, "save", arm=args.arm, step=step)
            if report.get("local_checkpoint_step") != step:
                store.save(args.arm, contract, payload(), step)
                report["local_checkpoint_step"] = step
                write_json(report_path, report)
            if args.stage == "continue":
                report["hf_verified"] = hub is not None and store.sync_until(
                    args.arm, time.monotonic() + max(0., args.deadline - time.time() - 60),
                )
                write_json(report_path, report)


def discard(root, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    states = root / "states"
    directory = states / arm
    if states.is_symlink() or directory.is_symlink() or directory.resolve().parent != states.resolve():
        raise ValueError("Unsafe disposable state directory")
    if directory.exists():
        shutil.rmtree(directory)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work-dir", default="/dev/shm/extent175")
    cli.add_argument("--output-dir", default="/kaggle/working/extent-results")
    cli.add_argument("--max-wall-hours", type=float, default=1.9)
    cli.add_argument("--save-reserve-minutes", type=float, default=18)
    cli.add_argument("--stage", choices=("parent", "prepare", "diagnose", "pilot", "continue"), default="parent")
    cli.add_argument("--arm", choices=ARMS, default="mixed")
    cli.add_argument("--batch", type=int, choices=(4, 8), default=8)
    cli.add_argument("--steps", type=int, default=256)
    cli.add_argument("--deadline", type=float)
    cli.add_argument("--work-deadline", type=float)
    cli.add_argument("--report")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage != "parent":
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        faulthandler.dump_traceback_later(180, repeat=True)
        try:
            prepare(root) if args.stage == "prepare" else worker(args, root)
        except Exception as exc:
            if args.report:
                path = Path(args.report)
                report = json.loads(path.read_text()) if path.exists() else {}
                report.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
                write_json(path, report)
            raise
        return

    from engine_tpu169 import ram_root
    from extent.notifications import send_telegram_message
    from extent.training_diagnostic_175 import plan_matched_steps, select_winner
    from filelock import FileLock

    if not 1 <= args.max_wall_hours <= 2 or not 10 <= args.save_reserve_minutes <= 30:
        raise ValueError("Use one to two hours and a ten to thirty minute save reserve")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "extent-m3q-tpu175.json"
    deadline = time.time() + args.max_wall_hours * 3600
    work_deadline = deadline - args.save_reserve_minutes * 60
    report = dict(experiment=175, status="preparing", pilots=[], resources=ram_root(root),
                  requested_hours=args.max_wall_hours, save_reserve_minutes=args.save_reserve_minutes)
    base = [sys.executable, "-u", "-m", "engine_tpu175", "--work-dir", str(root),
            "--deadline", str(deadline), "--work-deadline", str(work_deadline)]

    def notify(message):
        try:
            send_telegram_message("Extent EXP175: " + message)
        except Exception:
            pass

    def stop(_signum, _frame):
        (root / "STOP").touch()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(stage, arm, batch, steps, timeout):
        name = f"{stage}175-{arm}-b{batch}"
        report_path = output / (name + ".json")
        report_path.unlink(missing_ok=True)
        phase(root, name)
        result = wait_child(base + ["--stage", stage, "--arm", arm, "--batch", str(batch),
                                    "--steps", str(steps), "--report", str(report_path)],
                            root, output / (name + ".log"), max(1., timeout), label="EXP175")
        result.update(arm=arm, batch=batch, report=json.loads(report_path.read_text()) if report_path.exists() else {})
        return result

    with FileLock(str(root / "server.lock"), timeout=0):
        try:
            if (root / "STOP").exists() or (root / "states").exists():
                raise ValueError("Use a fresh work directory for this matched comparison")
            write_json(path, report)
            os.environ.setdefault("EXTENT_HF_CHECKPOINT_REPO", "levkremlev888/extent")
            os.environ.update(environment(root, "cpu"))
            os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(work_deadline)
            os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(deadline - 60)
            notify("started; CE/KL diagnosis, three matched full-student pilots")
            prep = wait_child(base + ["--stage", "prepare"], root, output / "prepare175.log",
                              min(1800., work_deadline - time.time()), label="EXP175")
            report["preparation"] = prep
            if prep["returncode"] != 0:
                raise RuntimeError("Source preparation failed; inspect prepare175.log")
            os.environ.update(environment(root, "tpu,cpu"))
            batch = 8
            diag = launch("diagnose", "mixed", batch, args.steps, min(900., work_deadline - time.time()))
            report["diagnostic_attempts"] = [diag]
            log = output / "diagnose175-mixed-b8.log"
            if diag["returncode"] != 0 and any(
                marker in log.read_text(errors="replace").lower()
                for marker in ("resource_exhausted", "out of memory", "resource exhausted")
            ):
                batch = 4
                diag = launch("diagnose", "mixed", batch, args.steps, min(900., work_deadline - time.time()))
                report["diagnostic_attempts"].append(diag)
            report["diagnostic"] = diag
            seconds = diag["report"].get("seconds_per_update", 5.2 if batch == 8 else 8.)
            steps = min(args.steps, plan_matched_steps(deadline - time.time(), seconds, args.save_reserve_minutes))
            if steps < 64:
                raise TimeoutError("Insufficient time for three matched pilots; diagnostics retained")
            report.update(planned_steps=steps, selected_batch=batch, status="pilots")
            write_json(path, report)
            winner = None
            for arm in ARMS:
                if (root / "STOP").exists() or time.time() >= work_deadline:
                    break
                trial = launch("pilot", arm, batch, steps,
                               min(steps * seconds * 1.4 + 180, deadline - time.time() - 300))
                report["pilots"].append(trial)
                previous = winner
                winner = select_winner(report["pilots"], steps)
                report["winner"] = winner["arm"] if winner else None
                write_json(path, report)
                if previous and winner and previous["arm"] != winner["arm"]:
                    discard(root, previous["arm"])
                if winner is None or winner["arm"] != arm:
                    discard(root, arm)
            if winner is None:
                raise RuntimeError("No complete pilot; diagnostic and worker reports retained")
            notify("matched pilots ended; selected " + winner["arm"])
            report["status"] = "continuing"
            write_json(path, report)
            result = launch("continue", winner["arm"], batch, steps, deadline - time.time())
            report["continuation"] = result
            if result["returncode"] != 0:
                raise RuntimeError("Selected worker failed; saved report and logs retained")
            complete = all(select_winner([row], steps) is not None for row in report["pilots"]) and len(report["pilots"]) == 3
            report["matched_comparison_complete"] = complete
            saved = result["report"].get("hf_verified", False)
            diagnostic_complete = diag["report"].get("status") == "diagnostic_completed"
            report["status"] = "completed" if saved and complete and diagnostic_complete else "partial"
        except Exception as exc:
            report.update(status="partial_or_failed", error=dict(type=type(exc).__name__, message=str(exc)),
                          supervisor_traceback=traceback.format_exc())
        finally:
            if (root / "phase.json").exists():
                report["last_phase"] = json.loads((root / "phase.json").read_text())
            write_json(path, report)
            notify(report["status"] + "; download extent-m3q-tpu175.json")
    return report


if __name__ == "__main__":
    main()
