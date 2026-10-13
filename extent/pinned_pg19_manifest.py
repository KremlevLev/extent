"""Seed the exact registered PG19 test manifest without an HF metadata request.

Only transport changes: the legacy loaders and token SHA guards remain intact.
The cache snapshot layout is verified through the public Hub cache API.
"""
from pathlib import Path
import hashlib
import os
import uuid

REPO_ID = 'deepmind/pg19'
REVISION = '4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b'
FILENAME = 'data/test_files.txt'
SHA256 = 'c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c'
BUNDLE = Path(__file__).resolve().parents[1] / 'results/pinned-data/pg19-test-files.txt'

def prepare_pg19_test_manifest(cache_dir):
    from huggingface_hub import try_to_load_from_cache
    raw = BUNDLE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('bundled pinned PG19 test manifest SHA mismatch')
    cache_dir = Path(cache_dir)
    destination = cache_dir / 'datasets--deepmind--pg19' / 'snapshots' / REVISION / FILENAME
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError('cached pinned PG19 test manifest SHA mismatch')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            staging.write_bytes(raw)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
    found = try_to_load_from_cache(REPO_ID, FILENAME, repo_type='dataset',
                                  revision=REVISION, cache_dir=cache_dir)
    if not isinstance(found, str) or Path(found).resolve() != destination.resolve():
        raise RuntimeError('installed HF client cannot resolve seeded pinned PG19 manifest')
    print('pg19_test_manifest=PINNED_LOCAL sha256=' + SHA256, flush=True)
    return destination
