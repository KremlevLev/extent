"""EXP174: matched CE+KL learning-rate pilots, then continue the winner."""

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from engine_tpu173 import environment, prepare, wait_child
from extent.broad_corpus import digest_file, write_json


PREFIX = "experiments/exp174-gqa7-lr-v1"
SCOPES = ("mamba", "full")
RATES = (1e-6, 3e-6, 1e-5)
BATCHES = (8, 16)
PILOT_STEPS = 128
CONTEXT = 1024
EFFECTIVE = 16
SOURCE_CURSOR = 83_886_080
INPUTS_PER_STEP = CONTEXT * EFFECTIVE
SELECTION_WINDOWS = tuple(range(32, 48))
MONITOR_WINDOWS = tuple(range(4))


def candidate_id(scope, rate):
    return f"{scope}-{rate:.0e}"


CANDIDATES = {candidate_id(scope, rate): (scope, rate) for scope in SCOPES for rate in RATES}


def select_winner(rows):
    complete = [
        row for row in rows
        if row["returncode"] == 0 and row["report"].get("status") == "pilot_completed"
        and row["report"].get("step") == PILOT_STEPS
        and math.isfinite(row["report"].get("selection_nll", math.inf))
    ]
    return min(complete, key=lambda row: (row["report"]["selection_nll"], row["candidate"]), default=None)


def phase(root, name, **details):
    write_json(root / "phase.json", dict(phase=name, unix=time.time(), pid=os.getpid(), **details))
    print("EXP174", name, details, flush=True)


def worker(args, root):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from extent.broad_corpus import PackedCorpus
    from extent.clean_tpu_171 import initialize_host, layout, optimizer
    from extent.end_to_end_174 import make_gradient, student_hidden
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.hf_download_retry import install_download_retries
    from extent.quality_172 import row_array, validate_manifest
    from extent.recovery_durability import RecoveryCheckpointStore

    install_download_retries()
    scope, peak = CANDIDATES[args.candidate]
    phase(root, "load_pinned_teacher_and_171_bank", scope=scope)
    config, teacher_config, bank, layers, last, embedding, norm = initialize_host(
        root / "qwen", lambda name: phase(root, name),
    )
    manifest_path = root / "bank/checkpoint.json"
    for row in validate_manifest(json.loads(manifest_path.read_text())):
        node = bank
        for name in row["path"][1:-1]:
            node = node[name]
        name = row["path"][-1]
        value = row_array(row, root / "bank")
        if value.shape != node[name].shape:
            raise ValueError("EXP171 parameter shape mismatch")
        node[name] = value

    phase(root, "TPU_discovery_and_sharding")
    devices = jax.devices("tpu")
    if len(devices) != 8 or jax.process_count() != 1:
        raise ValueError("EXP174 requires eight local TPU chips")
    mesh = Mesh(np.array(devices), ("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    jax.config.update("jax_compilation_cache_dir", str(root / "compile-cache"))
    donor_layout = (layout(layers, mesh), layout(last, mesh, stacked=False),
                    layout(embedding, mesh, stacked=False), layout(norm, mesh, stacked=False))
    donor = jax.device_put((layers, last, embedding, norm), donor_layout)
    parameters = {"mamba": bank}
    parameter_layout = {"mamba": layout(bank, mesh)}
    if scope == "full":
        parameters["qwen"] = jax.tree.map(lambda value: value.astype(jnp.float32), (layers, last, embedding, norm))
        parameter_layout["qwen"] = donor_layout
    parameters = jax.device_put(parameters, parameter_layout)
    tx = optimizer()
    state = tx.init(parameters)
    state_layout = layout(state, mesh, stacked=False)
    state = jax.device_put(state, state_layout)
    corpus = PackedCorpus(root / "edu")
    code_dir = Path(__file__).resolve().parent
    contract = dict(
        experiment=174, scope=scope, peak_lr=peak, warmup_updates=32, source_step=5120,
        source_manifest_sha256=digest_file(manifest_path), source_cursor=SOURCE_CURSOR,
        context=CONTEXT, effective_batch=EFFECTIVE, corpus=corpus.identity,
        loss=".5 CE + .5 exact forward KL; T=1", optimizer="fresh Adam(.9,.95), clip1",
        code={name: digest_file(code_dir / name) for name in
              ("engine_tpu174.py", "extent/end_to_end_174.py", "extent/end_to_end_173.py", "extent/clean_tpu_171.py")},
    )
    hub = artifact_config_from_env(include_kaggle_secrets=False) if args.stage == "continue" else None
    store = RecoveryCheckpointStore(root / "states", PREFIX, hub)
    step = 0
    cursor = SOURCE_CURSOR
    report_path = Path(args.report)
    report = dict(status="preflight", scope=scope, peak_lr=peak, physical_batch=args.batch,
                  contract=contract, step=step, history=[], evaluations=[])
    write_json(report_path, report)

    def payload():
        return dict(parameters=parameters, optimizer=state, cursor=np.asarray(cursor, np.int64))

    if args.stage == "continue":
        phase(root, "restore_winner", candidate=args.candidate)
        restored = store.restore(args.candidate, contract, payload())
        if restored is None:
            raise ValueError("Winner checkpoint unavailable")
        saved, metadata = restored
        step = metadata["step"]
        cursor = int(saved["cursor"])
        if step < PILOT_STEPS or cursor != SOURCE_CURSOR + step * INPUTS_PER_STEP:
            raise ValueError("Winner checkpoint cursor mismatch")
        parameters = jax.device_put(saved["parameters"], parameter_layout)
        state = jax.device_put(saved["optimizer"], state_layout)
        del saved, restored
        report.update(step=step, accepted_input_tokens=step * INPUTS_PER_STEP,
                      invocation_input_tokens=0, local_checkpoint_step=step)
        write_json(report_path, report)

    def tokens(at, count, split):
        end = at + count * CONTEXT
        values = corpus.arrays[split]
        if end > len(values):
            raise EOFError("Corpus exhausted")
        return jax.device_put(np.array(values[at:end]).reshape(count, CONTEXT), replicated)

    gradient = jax.jit(make_gradient(config, teacher_config, scope == "full"),
                       out_shardings=(replicated, parameter_layout, replicated))

    def finite(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))

    finite = jax.jit(finite, out_shardings=replicated)

    def accumulate(previous, gradient_part):
        return jax.tree.map(lambda a, b: a + b, previous, gradient_part)

    accumulate = jax.jit(accumulate, out_shardings=parameter_layout)

    def apply(parameters, state, gradient, rate):
        updates, new_state = tx.update(gradient, state, parameters)
        updates = jax.tree.map(lambda x: -rate * x, updates)
        return optax.apply_updates(parameters, updates), new_state

    update = jax.jit(apply, out_shardings=(parameter_layout, state_layout))
    normalize = jax.jit(
        lambda tree: jax.tree.map(lambda value: value * (args.batch / EFFECTIVE), tree),
        out_shardings=parameter_layout,
    )

    def complete(parameters, state, cursor, rate):
        grad = None
        metrics = jax.device_put(np.zeros(2, np.float32), replicated)
        for offset in range(0, EFFECTIVE, args.batch):
            _, part, values = gradient(parameters, donor, tokens(cursor + offset * CONTEXT, args.batch, "train"))
            grad = part if grad is None else accumulate(grad, part)
            metrics += values
        factor = args.batch / EFFECTIVE
        grad = normalize(grad)
        metrics *= factor
        if not bool(finite((grad, metrics))):
            raise FloatingPointError("Nonfinite gradient or loss")
        new_parameters, new_state = update(
            parameters, state, grad, jax.device_put(np.asarray(rate, np.float32), replicated),
        )
        if not bool(finite((new_parameters, new_state))):
            raise FloatingPointError("Nonfinite proposed optimizer update")
        return new_parameters, new_state, np.asarray(metrics)

    @jax.jit
    def evaluate_window(parameters, donor, token):
        h = student_hidden(parameters, donor, token, config, teacher_config)
        head = parameters.get("qwen", donor)[2].astype(jnp.bfloat16)
        labels = token.reshape(-1)
        features = h.reshape(CONTEXT // 128, 128, h.shape[-1])
        targets = jnp.roll(labels, -1).reshape(-1, 128)

        def score(total, inputs):
            features, target, tile = inputs
            logits = (features @ head.T).astype(jnp.float32)
            losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)
            keep = (tile * 128 + jnp.arange(128)) < CONTEXT - 1
            return total + jnp.sum(jnp.where(keep, losses, 0.)), None

        total, _ = jax.lax.scan(score, jnp.zeros((), jnp.float32), (features, targets, jnp.arange(CONTEXT // 128)))
        return total

    def evaluate(windows, label):
        nll = sum(float(evaluate_window(parameters, donor, tokens(i * CONTEXT, 1, "validation")))
                  for i in windows) / (len(windows) * (CONTEXT - 1))
        if not math.isfinite(nll):
            raise FloatingPointError("Nonfinite evaluation NLL")
        report["evaluations"].append(dict(step=step, panel=label, nll=nll))
        print(f"EXP174 {args.candidate} step={step} {label}_nll={nll:.6f}", flush=True)
        return nll

    phase(root, "replay_and_compile", stage=args.stage, batch=args.batch)
    baseline = evaluate(MONITOR_WINDOWS, "monitor")
    if step == 0 and abs(baseline - 2.5812020288) > .08:
        raise ValueError("Frozen EXP171 source replay mismatch")
    if args.stage == "probe":
        timings = []
        for _ in range(3):
            began = time.monotonic()
            parameters, state, _ = complete(parameters, state, cursor, peak / 32)
            timings.append(time.monotonic() - began)
        report.update(status="measured", tokens_per_second=INPUTS_PER_STEP / (sum(timings[1:]) / 2),
                      accepted_input_tokens=0,
                      probe_seconds=timings, engineering_inputs=3 * INPUTS_PER_STEP)
        write_json(report_path, report)
        return

    if args.stage == "pilot":
        report["initial_selection_nll"] = evaluate(SELECTION_WINDOWS, "selection_start")

    deadline = args.deadline - (30 if scope == "full" else 18) * 60
    began = last_save = time.monotonic()
    origin_step = step
    report["status"] = "running"
    write_json(report_path, report)
    phase(root, "train", stage=args.stage, candidate=args.candidate, start_step=step)
    try:
        while (time.time() < deadline and not (root / "STOP").exists()
               and cursor + INPUTS_PER_STEP <= len(corpus.arrays["train"])):
            if args.stage == "pilot" and step == PILOT_STEPS:
                break
            rate = peak * min(1., (step + 1) / 32)
            started = time.monotonic()
            parameters, state, metrics = complete(parameters, state, cursor, rate)
            step += 1
            cursor += INPUTS_PER_STEP
            elapsed = time.monotonic() - started
            report["history"].append(dict(step=step, lr=rate, ce=float(metrics[0]), kl=float(metrics[1]),
                                          seconds=elapsed, tokens_per_second=INPUTS_PER_STEP / elapsed))
            report.update(step=step, accepted_input_tokens=step * INPUTS_PER_STEP,
                          invocation_input_tokens=(step - origin_step) * INPUTS_PER_STEP,
                          loop_tokens_per_second=(step - origin_step) * INPUTS_PER_STEP / (time.monotonic() - began))
            interval = 16 if args.stage == "pilot" else 128
            if step % interval == 0:
                nll = evaluate(MONITOR_WINDOWS, "monitor")
                phase(root, "train_progress", candidate=args.candidate, step=step, nll=nll,
                      tokens_per_second=round(report["loop_tokens_per_second"], 1))
                write_json(report_path, report)
            if args.stage == "continue" and time.monotonic() - last_save > 900:
                store.save(args.candidate, contract, payload(), step)
                last_save = time.monotonic()
                report["local_checkpoint_step"] = step
                write_json(report_path, report)
        if args.stage == "pilot":
            if step == PILOT_STEPS:
                report["selection_nll"] = evaluate(SELECTION_WINDOWS, "selection")
                report["selection_nll_delta"] = report["selection_nll"] - report["initial_selection_nll"]
                store.save(args.candidate, contract, payload(), step)
                report["status"] = "pilot_completed"
            else:
                report["status"] = "pilot_partial"
        else:
            report["status"] = "corpus_exhausted" if cursor + INPUTS_PER_STEP > len(corpus.arrays["train"]) else "deadline_partial"
            if (root / "STOP").exists():
                report["status"] = "stopped"
            evaluate(MONITOR_WINDOWS, "monitor_final")
            report["final_selection_nll"] = evaluate(SELECTION_WINDOWS, "selection_final")
    except Exception as exc:
        report.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        write_json(report_path, report)
        if args.stage == "continue":
            phase(root, "save_winner", step=step)
            if report.get("local_checkpoint_step") != step:
                store.save(args.candidate, contract, payload(), step)
            report["local_checkpoint_step"] = step
            report["hf_verified"] = False
            write_json(report_path, report)
            if hub is not None:
                report["hf_verified"] = store.sync_until(
                    args.candidate, time.monotonic() + max(0., args.deadline - time.time() - 60),
                )
            write_json(report_path, report)


def discard_pilot(root, candidate):
    """Remove only a known disposable pilot directory in this run's tmpfs."""
    if candidate not in CANDIDATES:
        raise ValueError("Unknown pilot")
    directory = root / "states" / candidate
    if directory.exists():
        if directory.parent.is_symlink() or directory.is_symlink() or directory.resolve().parent != directory.parent:
            raise ValueError("Unsafe pilot directory")
        shutil.rmtree(directory)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work-dir", default="/dev/shm/extent174")
    cli.add_argument("--output-dir", default="/kaggle/working/extent-results")
    cli.add_argument("--max-wall-hours", type=float, default=9.)
    cli.add_argument("--stage", choices=("parent", "prepare", "probe", "pilot", "continue"), default="parent")
    cli.add_argument("--candidate", choices=CANDIDATES)
    cli.add_argument("--batch", type=int, choices=BATCHES, default=8)
    cli.add_argument("--deadline", type=float)
    cli.add_argument("--report")
    cli.add_argument("--resume-candidate", choices=CANDIDATES)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage != "parent":
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        faulthandler.dump_traceback_later(180, repeat=True)
        if args.stage == "prepare":
            prepare(root)
        else:
            worker(args, root)
        return

    from engine_tpu169 import ram_root
    from extent.notifications import send_telegram_message
    from filelock import FileLock

    if not 2 <= args.max_wall_hours <= 9:
        raise ValueError("EXP174 requires a budget from two to nine hours")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "extent-m3q-tpu174.json"
    deadline = time.time() + args.max_wall_hours * 3600
    report = dict(experiment=174, status="preparing", resources=ram_root(root), probes=[], pilots=[])
    base = [sys.executable, "-u", "-m", "engine_tpu174", "--work-dir", str(root), "--deadline", str(deadline)]

    def notify(message):
        try:
            send_telegram_message("Extent EXP174: " + message)
        except Exception:
            pass

    def request_stop(_signum, _frame):
        (root / "STOP").touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def launch(stage, candidate, batch, timeout):
        name = f"{stage}174-{candidate}-b{batch}"
        path = output / f"{name}.json"
        path.unlink(missing_ok=True)
        phase(root, name)
        trial = wait_child(base + ["--stage", stage, "--candidate", candidate, "--batch", str(batch),
                                   "--report", str(path)], root, output / f"{name}.log", timeout, label="EXP174")
        trial.update(candidate=candidate, batch=batch, report=json.loads(path.read_text()) if path.exists() else {})
        return trial

    with FileLock(str(root / "server.lock"), timeout=0):
        try:
            if (root / "STOP").exists():
                raise ValueError("Remove STOP explicitly before starting a new invocation")
            write_json(report_path, report)
            os.environ.setdefault("EXTENT_HF_CHECKPOINT_REPO", "levkremlev888/extent")
            os.environ.update(environment(root, "cpu"))
            os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(deadline - 30 * 60)
            os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(deadline - 60)
            notify("started; matched LR pilots and winner continuation")
            prep = wait_child(base + ["--stage", "prepare"], root, output / "prepare174.log", 2700, label="EXP174")
            if prep["returncode"]:
                raise RuntimeError("Source preparation failed; see prepare174.log")
            os.environ.update(environment(root, "tpu,cpu"))
            winner = None
            if args.resume_candidate:
                winner = dict(candidate=args.resume_candidate, batch=args.batch)
            else:
                if (root / "states").exists() and any((root / "states" / key).exists() for key in CANDIDATES):
                    raise ValueError("Existing states found; use --resume-candidate or a fresh work directory")
                for scope in SCOPES:
                    if (root / "STOP").exists() or time.time() > deadline - 40 * 60:
                        break
                    probes = []
                    for batch in BATCHES:
                        if (root / "STOP").exists():
                            break
                        trial = launch("probe", candidate_id(scope, RATES[0]), batch, 720)
                        probes.append(trial)
                        report["probes"].append(trial)
                        write_json(report_path, report)
                    fits = [row for row in probes if row["returncode"] == 0 and row["report"].get("status") == "measured"]
                    if not fits:
                        continue
                    batch = max(fits, key=lambda row: row["report"]["tokens_per_second"])["batch"]
                    for rate in RATES:
                        if (root / "STOP").exists() or time.time() > deadline - 40 * 60:
                            break
                        candidate = candidate_id(scope, rate)
                        trial = launch("pilot", candidate, batch, min(2400, deadline - time.time() - 35 * 60))
                        report["pilots"].append(trial)
                        previous = winner
                        winner = select_winner(report["pilots"])
                        report["winner"] = winner["candidate"] if winner else None
                        write_json(report_path, report)
                        if previous and winner and previous["candidate"] != winner["candidate"]:
                            discard_pilot(root, previous["candidate"])
                        if winner is None or winner["candidate"] != candidate:
                            discard_pilot(root, candidate)
            if winner is None:
                raise RuntimeError("No complete comparable LR pilot")
            report.update(winner=winner["candidate"], selected_batch=winner["batch"], status="continuing",
                          complete_pilots=sum(select_winner([row]) is not None for row in report["pilots"]))
            write_json(report_path, report)
            trial = launch("continue", winner["candidate"], winner["batch"], max(1., deadline - time.time()))
            report["continuation"] = trial
            result = trial["report"]
            if trial["returncode"] != 0:
                raise RuntimeError("Winner worker failed; inspect its saved report and log")
            report["status"] = result["status"] if result.get("hf_verified") else "storage_partial"
        except Exception as exc:
            report.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        finally:
            if (root / "phase.json").exists():
                report["last_phase"] = json.loads((root / "phase.json").read_text())
            write_json(report_path, report)
            notify(report["status"] + "; download extent-m3q-tpu174.json")
    return report


if __name__ == "__main__":
    main()
