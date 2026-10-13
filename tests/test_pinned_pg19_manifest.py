"""Transport-only regression: the original loader must resolve without HF HTTP."""
import hashlib
from pathlib import Path
import pytest
from huggingface_hub import hf_hub_download
import huggingface_hub.file_download as download
from extent import pinned_pg19_manifest as pinned

def test_manifest_download_uses_fixed_snapshot_without_http(tmp_path, monkeypatch):
    def no_http(*args, **kwargs):
        raise AssertionError('unexpected HTTP request for pinned manifest')
    monkeypatch.setattr(download, '_request_wrapper', no_http)
    expected = pinned.prepare_pg19_test_manifest(tmp_path)
    found = hf_hub_download(pinned.REPO_ID, pinned.FILENAME, repo_type='dataset',
                           revision=pinned.REVISION, cache_dir=tmp_path)
    assert Path(found) == expected
    assert hashlib.sha256(Path(found).read_bytes()).hexdigest() == pinned.SHA256
    assert len(Path(found).read_text().splitlines()) == 100
    assert pinned.prepare_pg19_test_manifest(tmp_path) == expected

def test_corruption_is_rejected_without_silent_replacement(tmp_path, monkeypatch):
    destination = pinned.prepare_pg19_test_manifest(tmp_path)
    destination.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='cached.*SHA mismatch'):
        pinned.prepare_pg19_test_manifest(tmp_path)
    assert destination.read_bytes() == b'wrong'
    monkeypatch.setattr(pinned, 'BUNDLE', destination)
    with pytest.raises(ValueError, match='bundled.*SHA mismatch'):
        pinned.prepare_pg19_test_manifest(tmp_path / 'other')

def test_entry_preserves_args_and_skips_cache_for_nontraining(monkeypatch, tmp_path):
    from scripts import m3q_efficiency_112_campaign as entry
    calls = []
    monkeypatch.setattr(pinned, 'prepare_pg19_test_manifest', lambda cache: calls.append(cache))
    monkeypatch.setattr(entry, 'run', lambda args, experiment: (args, experiment))
    args = ['--dataset-cache-dir', str(tmp_path), '--max-wall-hours', '1.8']
    assert entry.main(args) == (args, 112)
    assert calls == [str(tmp_path)]
    for flag in ('--plan-only', '--sync-only'):
        assert entry.main(args + [flag]) == (args + [flag], 112)
    assert calls == [str(tmp_path)]

def test_registered_scientific_contract_is_unchanged():
    import json
    from extent.config import load_config
    from scripts.m3q_efficiency_campaign import contract_for
    root = Path(__file__).resolve().parents[1]
    config, _ = load_config(root / 'config/hybrid_1_7b_gqa_v5e8.yaml')
    registered = json.loads((root / 'results/EXP-112-v1-contract.json').read_text())
    assert contract_for(config, 112) == registered
