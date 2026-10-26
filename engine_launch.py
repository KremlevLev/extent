"""Optional process-title wrapper; EXP162 scientific sources stay unchanged."""
import argparse
import ctypes
import os
import sys


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
    # Same process/PID/signals and stop logic; no subprocess, fork, or watchdog.
    from engine_compare import main as run
    return run(remaining)


if __name__=='__main__':main()
