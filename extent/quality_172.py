"""Frozen EXP171 -> PyTorch inference; no optimizer, training, or legacy bank.

Choice templates/metrics reuse EXP156, not an official lm-eval CLI result.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = 'levkremlev888/extent'
PREFIX = 'experiments/exp171-clean-qkvo-tpu-v1/ALIGN'
REVISION = 'd341d2b8f87592f617c44c92c5d7532669c75755'
EXPECTED_STEP = 5120
SOURCE = 'Qwen/Qwen3-1.7B-Base'
SOURCE_REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
MAPPING = {
    ('in_proj', 'kernel'): ('in_weight', True),
    ('out_proj', 'kernel'): ('out_weight', True),
    ('b_norm', 'scale'): ('b_norm', False),
    ('c_norm', 'scale'): ('c_norm', False),
    ('readout_norm', 'scale'): ('readout_norm', False),
    **{(name,): (name, False) for name in
       ('D', 'b_bias', 'c_bias', 'dt_bias', 'mimo_o', 'mimo_x', 'mimo_z')},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(8 << 20), b''): h.update(piece)
    return h.hexdigest()


def validate_manifest(meta):
    c = meta['contract']
    expected = dict(experiment=171, protocol='clean-qkvo-tpu-v1', source=SOURCE,
                    revision=SOURCE_REVISION, geometry='H16/G8/P128/N128/R4',
                    context=1024, effective_batch=16,
                    code='fef7b022fa077843b57e8943e8abe4c12c4a886aad945f2ab00b14cb71c81466')
    if meta['step'] != EXPECTED_STEP or any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('not the frozen received EXP171 CURRENT5120')
    rows = [r for r in meta['leaves'] if r['path'][0] == 'parameters']
    keys = [tuple(r['path'][1:]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(MAPPING):
        raise ValueError('incomplete/duplicate mixer parameter tree')
    for r in rows:
        if r['shape'][0] != 27 or r['dtype'] != 'float32':
            raise ValueError('wrong bank shape/dtype')
        for ch in r['chunks']:
            p = Path(ch['file'])
            if p.is_absolute() or '..' in p.parts or '\\' in ch['file'] or p.parts[0] != meta['generation']:
                raise ValueError('unsafe chunk path')
    return rows


def fetch_bank(root):
    """Only ~3.2GB mixer parameters; do NOT download Adam's two extra copies."""
    from huggingface_hub import hf_hub_download
    root = Path(root)
    def get(name):
        return Path(hf_hub_download(REPO, PREFIX+'/'+name, revision=REVISION,
                                   repo_type='dataset', local_dir=root))
    meta = json.loads(get('checkpoint.json').read_text())
    for row in validate_manifest(meta):
        for ch in row['chunks']:
            file = get(ch['file'])
            if file.stat().st_size != ch['bytes'] or sha(file) != ch['sha256']:
                raise ValueError('checkpoint chunk checksum mismatch')
    return meta


def row_array(row, root):
    # Chunk hashes were checked by prepare; workers check again before use.
    result = np.empty(int(np.prod(row['shape'])), dtype=np.float32)
    cursor = 0
    for ch in row['chunks']:
        path = Path(root)/ch['file']
        if sha(path) != ch['sha256']: raise ValueError('checkpoint changed after prepare')
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.float32 or x.ndim != 1 or cursor+x.size > result.size:
            raise ValueError('invalid chunk extent/dtype')
        result[cursor:cursor+x.size] = x
        cursor += x.size
    if cursor != result.size: raise ValueError('truncated parameter row')
    return result.reshape(row['shape'])


def install_bank(bank, meta, root):
    import torch
    for row in validate_manifest(meta):
        name, transpose = MAPPING[tuple(row['path'][1:])]
        array = row_array(row, root)
        for index in range(27):
            value = array[index].T if transpose else array[index]
            target = getattr(bank[str(index)], name)
            if tuple(value.shape) != tuple(target.shape): raise ValueError('mapped shape differs: '+name)
            with torch.no_grad(): target.copy_(torch.from_numpy(value.copy()))
            if not torch.equal(target.detach().cpu(), torch.from_numpy(value.copy())):
                raise ValueError('parameter conversion was not exact: '+name)


class Engine:
    def __init__(self, root, *, device='cuda', model=None, bank=None):
        import torch
        from extent.torch_clean_170 import CleanMamba
        self.device = device
        if model is None:
            import transformers
            from transformers import AutoModel
            if transformers.__version__ != '4.57.1': raise ValueError('requires transformers4.57.1')
            model = AutoModel.from_pretrained(SOURCE, revision=SOURCE_REVISION,
                torch_dtype=torch.float32, attn_implementation='sdpa',
                cache_dir=str(Path(root)/'hf'), trust_remote_code=False)
            if (model.config.num_hidden_layers, model.config.hidden_size,
                model.config.num_attention_heads, model.config.num_key_value_heads,
                model.config.head_dim, model.config.tie_word_embeddings) != (28, 2048, 16, 8, 128, True):
                raise ValueError('original Qwen geometry/tied head differs')
            bank = torch.nn.ModuleDict({str(i): CleanMamba(layer.self_attn, model.config)
                for i, layer in enumerate(model.layers[:-1])})
            meta = json.loads((Path(root)/'bank/checkpoint.json').read_text())
            install_bank(bank, meta, Path(root)/'bank')
        self.model = model.eval().requires_grad_(False).to(device)
        self.bank = bank.eval().requires_grad_(False).to(device)
        self.precision = 'float32'
        self.set_precision('float16' if device.startswith('cuda') else 'float32')

    def set_precision(self, precision):
        import torch
        self.precision = precision
        self.model.to(dtype=getattr(torch, precision))
        # Preserve learned dynamics/norms in FP32; projection GEMMs use autocast.
        self.bank.float()

    def autocast(self):
        import torch
        return torch.autocast('cuda', dtype=torch.float16) if self.precision == 'float16' else contextlib.nullcontext()

    def hidden(self, tokens, role):
        import torch
        from extent.torch_clean_170 import MASKS, chunked_scan
        from torch.nn import functional as F
        retained = set(range(len(self.model.layers))) if role == 'teacher' else set(MASKS[role])
        h = self.model.embed_tokens(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        rope = self.model.rotary_emb(h, positions)
        with self.autocast():
            for i, layer in enumerate(self.model.layers):
                if i in retained:
                    h = layer(h, position_ids=positions, position_embeddings=rope, use_cache=False)
                else:
                    m = self.bank[str(i)]
                    u = layer.input_layernorm(h)
                    parts = m.components(u)
                    y = chunked_scan(*parts, m, checkpoint=False)
                    middle = h + F.linear(y.flatten(-2).to(u.dtype), m.out_weight)
                    h = middle + layer.mlp(layer.post_attention_layernorm(middle))
            h = self.model.norm(h)
        if not torch.isfinite(h).all(): raise FloatingPointError('nonfinite composed hidden states')
        return h

    def nll(self, ids, role):
        import torch
        from torch.nn import functional as F
        tokens = torch.as_tensor(ids, dtype=torch.long, device=self.device)[None]
        h = self.hidden(tokens, role)
        output = []
        with self.autocast():
            for start in range(0, tokens.shape[1]-1, 16):
                end = min(tokens.shape[1]-1, start+16)
                logits = F.linear(h[:, start:end], self.model.embed_tokens.weight).float()
                values = F.cross_entropy(logits.flatten(0, 1), tokens[:, start+1:end+1].flatten(), reduction='none')
                if not torch.isfinite(values).all(): raise FloatingPointError('nonfinite token NLL')
                output.extend(values.tolist())
        return output

    def pair(self, tokenizer, pair, role, max_context=1024):
        from extent.quality_benchmark_156 import encode_pair
        ids, prefix = encode_pair(tokenizer, *pair)
        # Never truncate answer tokens; left-truncate context identically in all roles.
        if len(ids)-prefix >= max_context: raise ValueError('answer exceeds context limit')
        removed = max(0, len(ids)-max_context)
        ids, prefix = ids[removed:], prefix-removed
        return -sum(self.nll(ids, role)[prefix-1:]), removed

    def generate(self, tokenizer, text, role, count=48, check=lambda: None):
        import torch
        from torch.nn import functional as F
        ids = tokenizer.encode(text, add_special_tokens=False)
        generated = []
        # Correct full-prefix replay. No claim of recurrent-cache speed.
        for _ in range(count):
            check()
            tokens = torch.tensor([ids+generated], device=self.device)
            h = self.hidden(tokens, role)
            with self.autocast(): logits = F.linear(h[:, -1], self.model.embed_tokens.weight).float()
            if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite generation logits')
            token = int(logits.argmax(-1)); generated.append(token)
            if token == tokenizer.eos_token_id: break
        return dict(text=tokenizer.decode(generated), tokens=generated, greedy=True, max_new_tokens=count)


def summarize(items):
    from extent.quality_benchmark_156 import summarize_items, paired_results
    choices = {role: {k: v[role] for k, v in items.items() if v.get('kind') == 'choice'}
               for role in ('teacher', 'gqa4', 'gqa7')}
    return dict(tasks={r: summarize_items(x) for r, x in choices.items()},
        paired={r: paired_results(choices['teacher'], choices[r]) for r in ('gqa4', 'gqa7')},
        nll={r: {panel: dict(n_windows=len(rows), targets=sum(x['targets'] for x in rows),
                    nll=sum(x[r]['sum_nll'] for x in rows)/sum(x['targets'] for x in rows))
                 for panel in sorted({v['panel'] for v in items.values() if v.get('kind') == 'nll'})
                 if (rows := [v for v in items.values() if v.get('kind') == 'nll' and v['panel'] == panel])}
             for r in choices})
