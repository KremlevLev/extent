"""Optional process-title wrapper; EXP162 scientific sources stay unchanged."""
import argparse
import ctypes
import os
import sys
from pathlib import Path


def nvml_process_name():
    """Read the driver's actual view, without allocating a CUDA context."""
    if not sys.platform.startswith('linux'):return None
    try:
        library=ctypes.CDLL('libnvidia-ml.so.1')
        library.nvmlSystemGetProcessName.argtypes=[ctypes.c_uint,ctypes.c_char_p,ctypes.c_uint]
        if library.nvmlInit_v2()!=0:return None
        try:
            buffer=ctypes.create_string_buffer(4096)
            if library.nvmlSystemGetProcessName(os.getpid(),buffer,len(buffer))==0:
                return buffer.value.decode('utf-8',errors='replace')
        finally:library.nvmlShutdown()
    except (OSError,AttributeError):pass
    return None


def main(argv=None):
    # Only consume this wrapper's option; preserve every training argument.
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--process-title',default='VLLM:EngineCore')
    parser.add_argument('--runner',choices=('compare','methods','full','full164'),default='compare')
    args,remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not args.process_title or len(args.process_title)>64 or any(ord(c)<32 for c in args.process_title):
        raise ValueError('process title must contain 1..64 printable characters')
    # Keep /proc/environ intact. Set title before code that changes environment.
    os.environ['SPT_NOENV']='1'
    try:
        import setproctitle
    except ImportError:
        raise SystemExit('Install optional requirements-label.txt using this Python first') from None
    setproctitle.setproctitle(args.process_title)
    print('process_title_requested='+args.process_title,
          'nvml_process_name='+str(nvml_process_name() or 'unavailable'),flush=True)
    # New full launcher registers after its training lock. Legacy runners keep
    # pinned scientific files; attach their observer here without changing math.
    if args.runner not in ('full','full164') and '--no-telegram' not in remaining and os.environ.get('TELEGRAM_BOT_TOKEN') and (os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('TELEGRAM_USER_ID')):
        observer=argparse.ArgumentParser(add_help=False)
        observer.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
        observed,_=observer.parse_known_args(remaining);root=Path(observed.work_dir).expanduser().resolve()
        from engine_dashboard import trainer_active
        if trainer_active(root) is not True:
            from extent.status_autostart import ensure_listener
            ensure_listener(root,'results/runtime-methods.json' if args.runner=='methods' else 'results/runtime-compare.json')
    # Same training process/PID/signals; observer is an independent read-only process.
    if args.runner=='full':
        from engine_full_next import main as run
    elif args.runner=='full164':
        from engine_full import main as run
    elif args.runner=='methods':
        from engine_methods import main as run
    else:
        from engine_compare import main as run
    return run(remaining)


if __name__=='__main__':main()
