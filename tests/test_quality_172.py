import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from extent.quality_172 import (MAPPING, validate_manifest, row_array, sha,
    install_bank, Engine, summarize, EXPECTED_STEP, SOURCE, SOURCE_REVISION, fetch_bank, PREFIX)
from extent.torch_clean_170 import CleanMamba, chunked_scan
from engine_quality172 import load_journal


def tiny():
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, vocab_size=37,
        rms_norm_eps=1e-6)
    cfg._attn_implementation = 'sdpa'
    model = Qwen3Model(cfg).eval().requires_grad_(False)
    bank = torch.nn.ModuleDict({'0': CleanMamba(model.layers[0].self_attn, cfg)})
    return model, bank


def manifest():
    return dict(step=EXPECTED_STEP, generation='step-5120-test',
        contract=dict(experiment=171, protocol='clean-qkvo-tpu-v1', source=SOURCE,
            revision=SOURCE_REVISION, geometry='H16/G8/P128/N128/R4', context=1024,
            effective_batch=16, code='fef7b022fa077843b57e8943e8abe4c12c4a886aad945f2ab00b14cb71c81466'),
        leaves=[dict(path=['parameters', *path], shape=[27, 1], dtype='float32', chunks=[])
                for path in MAPPING])


def test_frozen_manifest_rejects_old_lineage_and_unsafe_chunks():
    m = manifest(); assert len(validate_manifest(m)) == 12
    for key, value in [('step', 260386), ('contract', {})]:
        broken = copy.deepcopy(m); broken[key] = value
        with pytest.raises(ValueError): validate_manifest(broken)
    for path in ('../secret', '/tmp/secret', 'step-5120-test\\x.npy'):
        broken = copy.deepcopy(m); broken['leaves'][0]['chunks'] = [dict(file=path)]
        with pytest.raises(ValueError): validate_manifest(broken)
    m['leaves'].append(m['leaves'][0])
    with pytest.raises(ValueError): validate_manifest(m)


def test_chunk_integrity_extent_and_dtype(tmp_path):
    path = tmp_path/'part.npy'; np.save(path, np.arange(8, dtype=np.float32))
    row = dict(shape=[2, 4], chunks=[dict(file='part.npy', sha256=sha(path))])
    np.testing.assert_array_equal(row_array(row, tmp_path), np.arange(8).reshape(2, 4))
    row['shape'] = [2, 5]
    with pytest.raises(ValueError, match='truncated'): row_array(row, tmp_path)
    row['chunks'][0]['sha256'] = 'bad'
    with pytest.raises(ValueError, match='changed'): row_array(row, tmp_path)


def test_hf_nested_download_layout_is_readable_and_reused(tmp_path, monkeypatch):
    import huggingface_hub
    meta = manifest()
    payloads = {}
    for i, row in enumerate(meta['leaves']):
        name = meta['generation']+f'/{i}.npy'
        file = tmp_path/f'source-{i}.npy'
        np.save(file, np.arange(27, dtype=np.float32)+i)
        payloads[PREFIX+'/'+name] = file.read_bytes()
        row['chunks'] = [dict(file=name, bytes=file.stat().st_size, sha256=sha(file))]
    payloads[PREFIX+'/checkpoint.json'] = json.dumps(meta).encode()
    calls = []
    def download(repo, filename, **kwargs):
        calls.append(filename)
        file = Path(kwargs['local_dir'])/filename  # actual SDK local_dir semantics
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(payloads[filename]); return str(file)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    root = tmp_path/'bank'
    assert fetch_bank(root) == meta
    assert json.loads((root/'checkpoint.json').read_text()) == meta
    for i,row in enumerate(meta['leaves']):
        np.testing.assert_array_equal(row_array(row,root), (np.arange(27,dtype=np.float32)+i).reshape(27,1))
    count = len(calls)
    assert fetch_bank(root) == meta
    assert len(calls) == count  # no second copy/redownload of valid flat files
    # Detect a corrupt cached leaf instead of quietly reporting benchmark scores.
    (root/meta['leaves'][0]['chunks'][0]['file']).write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'): fetch_bank(root)


def test_exact_jax_checkpoint_to_torch_mapping_all_leaves(tmp_path):
    model, _ = tiny()
    bank = torch.nn.ModuleDict({str(i): CleanMamba(model.layers[0].self_attn, model.config) for i in range(27)})
    meta = manifest()
    for row in meta['leaves']:
        name, transpose = MAPPING[tuple(row['path'][1:])]
        shape = tuple(getattr(bank['0'], name).shape)
        if transpose: shape = shape[::-1]
        values = np.arange(27*np.prod(shape), dtype=np.float32).reshape(27, *shape)/10000
        path = tmp_path/('part'+str(len(list(tmp_path.iterdir())))+'.npy')
        np.save(path, values.flatten()); row.update(shape=list(values.shape), chunks=[dict(file=path.name, sha256=sha(path))])
    # Safe-generation paths in the real checkpoint.
    generation = tmp_path/meta['generation']; generation.mkdir()
    for row in meta['leaves']:
        ch = row['chunks'][0]; (tmp_path/ch['file']).replace(generation/ch['file']); ch['file'] = meta['generation']+'/'+ch['file']
    install_bank(bank, meta, tmp_path)
    for row in meta['leaves']:
        name, transpose = MAPPING[tuple(row['path'][1:])]; values = row_array(row, tmp_path)
        for i in (0, 13, 26):
            expected = values[i].T if transpose else values[i]
            np.testing.assert_array_equal(getattr(bank[str(i)], name).detach().numpy(), expected)


def test_torch_evaluator_matches_tpu171_math_nonzero_learned_dynamics():
    import jax.numpy as jnp
    from extent.clean_tpu_171 import mamba_forward
    from extent.config import Mamba3Config
    model, bank = tiny(); m = bank['0']
    with torch.no_grad():
        m.in_weight[-13:].normal_(0, .02); m.mimo_o.normal_(0, .2)
    p = dict(in_proj={'kernel':m.in_weight.detach().numpy().T}, out_proj={'kernel':m.out_weight.detach().numpy().T},
        b_norm={'scale':m.b_norm.detach().numpy()}, c_norm={'scale':m.c_norm.detach().numpy()},
        readout_norm={'scale':m.readout_norm.detach().numpy()})
    for name in ('D','b_bias','c_bias','dt_bias','mimo_x','mimo_z','mimo_o'): p[name] = getattr(m,name).detach().numpy()
    cfg = SimpleNamespace(hidden_size=16, rms_norm_eps=1e-6,
        mamba=Mamba3Config(head_dim=4,d_state=4,groups=2,expand=1.,mimo_rank=4))
    u = torch.randn(1,128,16)*.2
    with torch.no_grad():
        y = chunked_scan(*m.components(u),m,checkpoint=False)
        actual = torch.nn.functional.linear(y.flatten(-2),m.out_weight).numpy()
    expected = np.asarray(mamba_forward(jnp.asarray(u.numpy()),p,cfg))
    np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=3e-5)


def test_full_teacher_matches_hf_and_hybrid_is_causal(monkeypatch):
    import extent.torch_clean_170 as core
    monkeypatch.setitem(core.MASKS,'gqa4',(1,))
    model, bank = tiny(); engine = Engine('.',device='cpu',model=model,bank=bank)
    tokens = torch.tensor([[1,5,3,7,2,4]])
    with torch.inference_mode():
        torch.testing.assert_close(engine.hidden(tokens,'teacher'),model(tokens,use_cache=False).last_hidden_state)
        before = engine.hidden(tokens,'gqa4'); changed = tokens.clone(); changed[:,4:] = 9
        torch.testing.assert_close(before[:,:4],engine.hidden(changed,'gqa4')[:,:4],rtol=1e-4,atol=1e-5)
        values = engine.nll(tokens[0].tolist(),'teacher')
        h = model(tokens,use_cache=False).last_hidden_state
        expected = torch.nn.functional.cross_entropy((h[:,:-1]@model.embed_tokens.weight.T).flatten(0,1),tokens[:,1:].flatten(),reduction='none')
        np.testing.assert_allclose(values,expected.numpy(),rtol=1e-5)


def test_interrupted_journal_and_matched_summary(tmp_path):
    row = dict(id='n1',kind='nll',panel='edu_1024',targets=10,
        teacher=dict(sum_nll=20),gqa4=dict(sum_nll=30),gqa7=dict(sum_nll=25))
    path = tmp_path/'progress.jsonl'; path.write_text(json.dumps(row)+'\n{"partial":')
    items = load_journal(path); assert list(items) == ['n1']
    result = summarize(items)
    assert result['nll']['gqa4']['edu_1024']['nll'] == 3
    assert result['nll']['teacher']['edu_1024']['targets'] == 10


def test_continuation_scoring_and_truncation(monkeypatch):
    model, bank = tiny(); engine = Engine('.',device='cpu',model=model,bank=bank)
    class Tokenizer:
        eos_token_id=0
        def encode(self,text,add_special_tokens=False): return [ord(c)%30+1 for c in text]
    observed=[]
    def nll(ids,role): observed.append(ids); return list(range(1,len(ids)))
    monkeypatch.setattr(engine,'nll',nll)
    score,removed=engine.pair(Tokenizer(),('abcde',' xy'),'teacher',max_context=6)
    assert removed==2 and len(observed[0])==6
    assert score==-(3+4+5)  # only continuation; context loss excluded
    with pytest.raises(ValueError): engine.pair(Tokenizer(),('a','long answer'),'teacher',max_context=6)
