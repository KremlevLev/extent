# This legacy file downloads and executes Qwen3-14B at import time.  It is a
# manual integration script, not an isolated pytest test, so keep it out of the
# fast/offline suite. Run it explicitly only in a prepared TPU environment.
collect_ignore = ["test_qwen_base.py"]
