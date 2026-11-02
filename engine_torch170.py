"""Clean QKVO alignment launcher; isolated CUDA planner, bounded CURRENT storage."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from extent.broad_corpus import write_json, digest_file
from extent.hf_download_retry import DownloadPolicy
from extent.isolated_tpu_planner import run_probe, grow, terminate_group

PREFIX = 'experiments/exp170-torch-clean-align-v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--model-cache', required=True)
    p.add_argument('--max-wall-hours', type=float, required=True)
    p.add_argument('--memory-gib', type=float, default=85.)
    p.add_argument('--save-reserve-minutes', type=float, default=18.)
    p.add_argument('--planner-minutes', type=float, default=25.)
    p.add_argument('--probe-timeout-minutes', type=float, default=8.)
    p.add_argument('--telegram', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--hf', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--gpu', default='0')
    p.add_argument('--sequence-length', type=int, default=4096, choices=[256, 1024, 2048, 4096])
    p.add_argument('--effective-batch', type=int, default=16)
    p.add_argument('--total-updates', type=int, default=1526)
    p.add_argument('--eval-every', type=int, default=128)
    p.add_argument('--eval-windows', type=int, default=8)
    return p


def notify(enabled, message):
    if not enabled:
        return
    from extent.notifications import send_telegram_message
    try:
        send_telegram_message('Extent EXP170: '+message)
    except Exception as exc:
        print('notification_error='+type(exc).__name__, flush=True)


def remote_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get('HF_TOKEN'))


def restore_remote(root, policy):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    repo = os.environ.get('EXTENT_HF_CHECKPOINT_REPO', 'levkremlev888/extent')
    try:
        metadata = policy.call(hf_hub_download, repo_id=repo, repo_type='dataset',
            filename=PREFIX+'/latest.json', token=os.environ.get('HF_TOKEN'),
            local_dir=str(root/'remote-metadata'))
    except EntryNotFoundError:
        return
    meta = json.loads(Path(metadata).read_text())
    if meta.get('file') != PREFIX+'/CURRENT.pt':
        raise ValueError('remote CURRENT path differs')
    # Download directly to the CURRENT namespace; no second HF cache copy.
    downloaded = policy.call(hf_hub_download, repo_id=repo, repo_type='dataset',
        filename=PREFIX+'/CURRENT.pt', revision=meta['commit'],
        token=os.environ.get('HF_TOKEN'), local_dir=str(root/'remote-current'))
    if digest_file(downloaded) != meta['sha256']:
        raise ValueError('remote CURRENT SHA mismatch')
    Path(downloaded).replace(root/'CURRENT.pt')


def sync_remote(root, report_path, policy):
    """Commit CURRENT then a manifest; delete local ONLY after remote LFS SHA check."""
    api = remote_api()
    repo = os.environ.get('EXTENT_HF_CHECKPOINT_REPO', 'levkremlev888/extent')
    current = root/'CURRENT.pt'
    if current.is_symlink() or not current.is_file():
        raise ValueError('CURRENT must be a regular local file')
    sha = digest_file(current)
    commit = policy.call(api.upload_file, path_or_fileobj=str(current),
        path_in_repo=PREFIX+'/CURRENT.pt', repo_id=repo, repo_type='dataset',
        commit_message='EXP170 own mixer bank and optimizer CURRENT')
    commit_id = commit.oid
    info = policy.call(api.get_paths_info, repo_id=repo, repo_type='dataset',
        paths=[PREFIX+'/CURRENT.pt'], revision=commit_id)
    if len(info) != 1 or getattr(getattr(info[0], 'lfs', None), 'sha256', None) != sha:
        raise ValueError('remote LFS SHA not verified; retaining local CURRENT')
    meta = dict(file=PREFIX+'/CURRENT.pt', sha256=sha, commit=commit_id,
        step=json.loads(report_path.read_text()).get('local_checkpoint_step'))
    write_json(root/'latest.json', meta)
    from huggingface_hub import CommitOperationAdd
    policy.call(api.create_commit, repo_id=repo, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=PREFIX+'/latest.json', path_or_fileobj=str(root/'latest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'/runtime.json', path_or_fileobj=str(report_path))],
        commit_message='EXP170 verified CURRENT descriptor and report')
    # Re-read exact manifest commit, rather than interpreting an upload as proof.
    from huggingface_hub import hf_hub_download
    readback = policy.call(hf_hub_download, repo_id=repo, repo_type='dataset',
        filename=PREFIX+'/latest.json', token=os.environ.get('HF_TOKEN'),
        local_dir=str(root/'remote-verification'), force_download=True)
    if json.loads(Path(readback).read_text()) != meta:
        raise ValueError('remote manifest mismatch; retaining CURRENT')
    current.unlink()
    return meta


def worker_command(args, root, result, deadline, *, batch, chunk, backend='chunked', probe=False):
    cmd = [sys.executable, '-u', '-m', 'extent.torch_train_170',
        '--work-dir', str(root), '--corpus', str(Path(args.corpus).resolve()),
        '--model-cache', str(Path(args.model_cache).resolve()),
        '--memory-gib', str(args.memory_gib), '--result', str(result),
        '--deadline', str(deadline), '--save-reserve-minutes', str(args.save_reserve_minutes),
        '--sequence-length', str(args.sequence_length), '--effective-batch', str(args.effective_batch),
        '--total-updates', str(args.total_updates), '--physical-batch', str(batch), '--chunk', str(chunk), '--backend', backend,
        '--eval-every', str(args.eval_every), '--eval-windows', str(args.eval_windows)]
    if probe:
        cmd.append('--probe')
    elif args.profile:
        cmd.append('--profile')
    return cmd


def sample_gpu(root, stopped, gpu):
    """Small append-only utilization/VRAM/power trace, independent of CUDA worker."""
    with (root/'gpu-telemetry.csv').open('a', encoding='utf-8') as log:
        while not stopped.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '-i', gpu,
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)
                if result.returncode == 0:
                    log.write(str(time.time())+','+result.stdout.strip()+'\n'); log.flush()
            except (OSError, subprocess.TimeoutExpired):
                pass
            stopped.wait(5)


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('max_wall_hours', 'memory_gib', 'planner_minutes', 'probe_timeout_minutes', 'save_reserve_minutes'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite '+key+' required')
    if args.save_reserve_minutes < 18 or args.max_wall_hours*60 <= args.save_reserve_minutes+args.planner_minutes+5:
        raise ValueError('need18min reserve plus planner and training time')
    if args.effective_batch < 1 or args.total_updates < 1 or args.eval_every < 1 or args.eval_windows < 1:
        raise ValueError('positive counts required')
    if args.hf and (args.sequence_length, args.effective_batch, args.total_updates) != (4096,16,1526):
        raise ValueError('engineering shape/target overrides require --no-hf and a separate work directory')
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.hf and not os.environ.get('HF_TOKEN'):
        raise ValueError('HF_TOKEN required unless --no-hf')
    report = root/'results/runtime-torch170.json'
    from filelock import FileLock
    with FileLock(str(root/'server.lock'), timeout=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(root/'kernel-cache/inductor')
        os.environ['TRITON_CACHE_DIR'] = str(root/'kernel-cache/triton')
        for leaf in ('STOP', 'FAST_STOP'):
            if args.clear_stop:
                (root/leaf).unlink(missing_ok=True)
            elif (root/leaf).exists():
                raise ValueError(leaf+' exists; use --clear-stop')
        def request_stop(signum, frame):
            (root/'STOP').touch()
        old_signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        deadline = time.time()+args.max_wall_hours*3600
        policy = DownloadPolicy(root/'transport170', deadline=deadline-60)
        os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX'] = str(deadline-args.save_reserve_minutes*60)
        os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX'] = str(deadline-60)
        os.environ['EXTENT_HF_RETRY_STATE_DIR'] = str(root/'transport170')
        from extent.hf_download_retry import install_download_retries
        install_download_retries()
        write_json(report, dict(status='planning', contract={'experiment':170}, branches={}))
        if args.telegram:
            from extent.status_autostart import ensure_listener
            ensure_listener(root, str(report.relative_to(root)))
        notify(args.telegram, 'clean QKVO; original Qwen; PyTorch; planner starting')
        telemetry_stop = threading.Event()
        telemetry = threading.Thread(target=sample_gpu, args=(root, telemetry_stop, args.gpu), daemon=True)
        telemetry.start()
        try:
            stop = lambda: (root/'STOP').exists() or (root/'FAST_STOP').exists()
            fits = []
            planner_end = min(time.monotonic()+args.planner_minutes*60,
                time.monotonic()+deadline-time.time()-args.save_reserve_minutes*60)
            # Two legal kernel tilings. Each CUDA crash is confined to its child.
            variants = [('chunked-compiled', 16), ('chunked', 16)]
            if importlib.util.find_spec('mamba_ssm') is not None:
                variants += [('tilelang', 8), ('tilelang', 16)]
            for variant_index, (backend, chunk) in enumerate(variants):
                if stop():
                    break
                variant_end = time.monotonic()+max(0, planner_end-time.monotonic())/(len(variants)-variant_index)
                def probe(batch):
                    result = root/f'probes/{backend}-chunk{chunk}-batch{batch}.json'
                    log = result.with_suffix('.log')
                    return run_probe(worker_command(args, root, result, deadline,
                        batch=batch, chunk=chunk, backend=backend, probe=True), result, log,
                        min(args.probe_timeout_minutes*60, max(.1, variant_end-time.monotonic())), stop=stop)
                outcome = grow(probe, first=1, max_batch=min(8, args.effective_batch),
                    deadline=variant_end, record=lambda value: write_json(root/f'planner-{backend}-chunk{chunk}.json', value))
                fits.extend(dict(row, chunk=chunk, backend=backend) for row in outcome['trials'] if row['status']=='measured')
            if stop():
                notify(args.telegram, 'stopped during planner; no training updates accepted'); return
            if not fits:
                raise RuntimeError('no valid fused CUDA fit; inspect probes/*.log; no slow fallback')
            best = max(fits, key=lambda row: row['report']['input_tokens_per_second'])
            write_json(root/'planner-selected.json', best)
            print('selected_batch='+str(best['batch'])+' chunk='+str(best['chunk'])+
                ' measured_full_update_tokens_per_second='+str(best['report']['input_tokens_per_second']), flush=True)
            if args.preflight_only:
                write_json(report, dict(status='preflight_completed', contract={'experiment':170}, planner=best, accepted_updates=0))
                notify(args.telegram, 'preflight completed; no training updates accepted; planner-selected.json'); return
            if args.hf and not (root/'CURRENT.pt').exists():
                restore_remote(root, policy)
            if stop():
                notify(args.telegram, 'stopped before training worker; CURRENT retained'); return
            command = worker_command(args, root, report, deadline, batch=best['batch'], chunk=best['chunk'], backend=best['backend'])
            child = subprocess.Popen(command, start_new_session=(os.name=='posix'))
            try:
                while child.poll() is None:
                    if time.time() >= deadline:
                        terminate_group(child)
                        raise TimeoutError('hard wall deadline; preserve existing CURRENT')
                    time.sleep(1)
            finally:
                if child.poll() is None:
                    terminate_group(child)
            value = json.loads(report.read_text())
            if child.returncode or value['status'] in ('failed', 'memory_limit'):
                raise RuntimeError('training worker failed; CURRENT retained, see report/log')
            if args.hf and value['status'] != 'fast_paused' and (root/'CURRENT.pt').exists():
                value['remote'] = sync_remote(root, report, policy)
                value['pending_slots'] = []
                write_json(report, value)
            if value['status'] == 'fast_paused':
                write_json(root/'fast-pause-ready.json', dict(step=value.get('local_checkpoint_step'), gpu_worker_exited=True, local_current_retained=True))
            notify(args.telegram, value['status']+'; report='+str(report))
        except BaseException as exc:
            value = json.loads(report.read_text()) if report.exists() else {'contract':{'experiment':170}}
            value.update(status='failed', launcher_error_type=type(exc).__name__)
            write_json(report, value)
            notify(args.telegram, 'failed: '+type(exc).__name__+'; local CURRENT retained')
            raise
        finally:
            telemetry_stop.set()
            telemetry.join(timeout=6)
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    main()
