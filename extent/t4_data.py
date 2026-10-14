"""Small explicit GPU panels; validation reuse is labelled, not a new test."""
from pathlib import Path
import json
import numpy as np
from extent.plateau_data import pack_panels, hash_array
from extent.calibration_data import WIKITEXT_CONFIG_FILES, WIKITEXT_REPO, WIKITEXT_REVISION
from extent.qwen_source import QWEN3_1_7B_BASE

PANELS = {'A': ('validation', 122880, 64), 'B': ('validation', 180224, 32),
          'C': ('validation', 196608, 64), 'FIT': ('train', 74973184, 128)}


def load_data(directory, check=lambda: None):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    file = directory/'gpu-panels.npz'; manifest = directory/'gpu-panels.json'
    if file.exists() and manifest.exists():
        meta = json.loads(manifest.read_text())
        arrays = dict(np.load(file, allow_pickle=False))
        if meta['spec'] != json.loads(json.dumps(PANELS)):
            raise ValueError('GPU panel spec changed')
        if any(hash_array(a) != meta['sha256'][k] for k, a in arrays.items()):
            raise ValueError('GPU data cache SHA mismatch')
        return arrays, meta
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,
        revision=QWEN3_1_7B_BASE.revision, cache_dir=directory, trust_remote_code=False)
    arrays = {}; provenance = {}
    for split in ('validation', 'train'):
        specs = {k: (o, n) for k, (s, o, n) in PANELS.items() if s == split}
        def rows():
            for name in WIKITEXT_CONFIG_FILES['wikitext-103-raw-v1'][split]:
                check()
                p = hf_hub_download(WIKITEXT_REPO, name, repo_type='dataset',
                    revision=WIKITEXT_REVISION, cache_dir=directory)
                for text in pq.read_table(p, columns=['text'])['text'].to_pylist():
                    check()
                    yield text
        found, prov = pack_panels(tokenizer, rows(), specs)
        arrays.update(found); provenance[split] = prov
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root/'results/EXP-114-data-preflight.json').read_text())['panels']
    for key, legacy in (('A','A'), ('B','B'), ('C','C_wiki')):
        if hash_array(arrays[key]) != expected[legacy]['sha256']:
            raise ValueError('GPU validation token hash differs from TPU panel')
    meta = dict(spec=PANELS, sha256={k: hash_array(a) for k, a in arrays.items()},
        wiki_revision=WIKITEXT_REVISION, tokenizer_revision=QWEN3_1_7B_BASE.revision,
        provenance=provenance, interpretation='A/B/C reused EXP114 validation; FIT overlaps EXP115 train; no independent locked test')
    tmp = directory/'gpu-panels.tmp.npz'; np.savez(tmp, **arrays); tmp.replace(file)
    manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return arrays, meta
