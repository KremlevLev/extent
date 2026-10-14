"""Layer-streamed inference on one GPU. No full model/Adam on the device.

The immutable EXP111 patient is reconstructed on HOST; a single layer is sent
to the selected GPU. Stored BF16 weights and recurrence are preserved.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import ml_dtypes
import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import unfreeze
from extent.config import HybridConfig, Mamba3Config, MLAConfig
from extent.qwen_source import QWEN3_1_7B_BASE, teacher_config_from_spec
from extent.weight_mapping import teacher_qwen_mappings, QwenCheckpointReader, expected_qwen_shape
from extent.hf_artifact_sync import restore_artifact
from extent.chunked_checkpoint import digest
from extent.efficiency_sources import ReadOnlySourceStore, tree_digest
from extent.model import HybridDecoderLayer
from extent.plateau_forensics import parts, norm, mlp, mamba_trace
from extent.recovery_subspace import apply_corrections

ROOT = Path(__file__).resolve().parents[1]
BF16 = np.dtype(ml_dtypes.bfloat16)


def patient_config(pin):
    spec = dict(pin['source_contract']['model'])
    spec['mamba'] = Mamba3Config(**spec['mamba'])
    spec['mla'] = MLAConfig(**spec['mla'])
    spec['attention_layer_indices'] = tuple(spec['attention_layer_indices'])
    return HybridConfig(**spec)


def check_path(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('unsafe immutable source path')
    return p


def read_parameter_chunks(meta, directory, prefix, hub, check=lambda: None):
    """Only correction parameters; do not load Adam moments or initialize a model."""
    flat = {}
    for row in meta['leaves']:
        if row['path'][0] != 'parameters':
            continue
        check()
        array = np.empty(int(np.prod(row['shape'])), dtype=row['dtype'])
        cursor = 0
        for chunk in row['chunks']:
            relative = check_path(chunk['file'])
            if relative.parts[0] != meta['generation']:
                raise ValueError('chunk generation differs')
            file = directory / relative
            if not file.exists():
                restore_artifact(file, f'{prefix}/{relative.as_posix()}', hub)
            if not file.exists() or digest(file) != chunk['sha256']:
                raise ValueError('patient correction SHA mismatch')
            value = np.load(file, allow_pickle=False)
            if value.dtype != array.dtype or cursor + value.size > array.size:
                raise ValueError('invalid patient correction chunk')
            array[cursor:cursor + value.size] = value.reshape(-1)
            cursor += value.size
        if cursor != array.size:
            raise ValueError('truncated patient correction')
        flat[tuple(row['path'][1:])] = array.reshape(row['shape'])
    if not flat:
        raise ValueError('empty patient correction')
    return traverse_util.unflatten_dict(flat)


def ensure_qwen(cache, check):
    from huggingface_hub import hf_hub_download
    cache.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'model.safetensors'):
        check()
        hf_hub_download(QWEN3_1_7B_BASE.repo_id, filename,
                       revision=QWEN3_1_7B_BASE.revision, local_dir=cache)
    return QwenCheckpointReader(cache)


class Patient:
    def __init__(self, cache, hub, device, check=lambda: None):
        self.check, self.device = check, device
        self.pin = json.loads((ROOT / 'results/EXP-114-source-pin.json').read_text())
        self.cfg = patient_config(self.pin)
        self.tc = teacher_config_from_spec(QWEN3_1_7B_BASE, param_dtype='bfloat16',
                                          compute_dtype='bfloat16', remat_policy='none')
        self.reader = ensure_qwen(Path(cache) / 'qwen', check)
        # All Qwen arrays are host BF16; teacher layers and preserved tensors
        # share their host arrays with the composed hybrid.
        flat = {}
        for entry in teacher_qwen_mappings(self.tc):
            check()
            value = self.reader.read(entry.source)
            if value.shape != expected_qwen_shape(entry, self.tc):
                raise ValueError('Qwen source tensor shape differs')
            if entry.transform == 'transpose':
                value = value.T
            flat[tuple(entry.target.split('/'))] = np.asarray(value, dtype=BF16)
        self.teacher = traverse_util.unflatten_dict(flat)
        self.base = unfreeze(self.teacher)
        # Import the legacy contract builder only during reconstruction.
        from scripts import m3q_subspace_engine as engine
        source_pin = json.loads((ROOT / 'results/EXP-111-113-source-pin.json').read_text())
        oldhub = replace(hub, repo_id=source_pin['repo_id'], repo_type='dataset', revision=source_pin['revision'])
        seq = engine.configured_contract(self.cfg)
        stores = (ReadOnlySourceStore(Path(cache)/'069', f'{engine.EXP069_PREFIX}/checkpoints', oldhub),
                  ReadOnlySourceStore(Path(cache)/'072', f'{engine.EXP072["HF_PREFIX"]}/checkpoints', oldhub))
        for depth, layer in enumerate(seq['replacement_order'], 1):
            check()
            prep = stores[0].metadata(f'prep/seed-123/layer-{layer}',
                dict(seq['source_contract'], seed=123, layer=layer, kind='prepared_mamba'))
            if prep is None:
                raise FileNotFoundError('pinned EXP069 preparation missing')
            contract = engine.sequential.endpoint_contract(seq, 123, 'ONPOLICY', layer, depth, prep['checkpoint_sha256'])
            saved = stores[1].restore(f'seed-123/ONPOLICY/layer-{layer}', contract)
            if saved is None or saved[1]['checkpoint_sha256'] != source_pin['seeds']['123'][str(layer)]:
                raise ValueError('EXP072 endpoint pin mismatch')
            self.base[f'layers_{layer}'].pop('self_attn')
            self.base[f'layers_{layer}']['mamba'] = saved[0]['params']
        if tree_digest(self.base) != self.pin['source_models']['123']['sha256']:
            raise ValueError('reconstructed ONPOLICY host tree differs from pin')
        meta = self.pin['manifests']['123']['metadata']
        prefix = 'experiments/exp111-recovery-efficiency/checkpoints/123/MAIN'
        pinned = replace(hub, repo_id=self.pin['repo_id'], repo_type='dataset', revision=self.pin['revision'])
        pointer = Path(cache)/'111/checkpoint.json'
        if not pointer.exists():
            restore_artifact(pointer, f'{prefix}/checkpoint.json', pinned)
        if json.loads(pointer.read_text()) != meta or meta['step'] != 32768:
            raise ValueError('EXP111 manifest pin differs')
        self.coordinates = read_parameter_chunks(meta, pointer.parent, prefix, pinned, check)
        self.columns = (2*int(self.cfg.hidden_size*self.cfg.mamba.expand)
                        + 2*self.cfg.mamba.mimo_rank*self.cfg.mamba.groups*self.cfg.mamba.d_state)
        # CPU folding uses the same JAX FP32 formula and BF16 cast as EXP111;
        # the original tree digest has already been checked independently.
        cpu = jax.devices('cpu')[0]
        self.current = unfreeze(self.base)
        with jax.default_device(cpu):
            for layer in self.cfg.mamba_layer_indices:
                root = f'layers_{layer}'
                projections = {name: self.base[root]['mamba'][name] for name in ('in_proj','out_proj')}
                subset = {name: value for name,value in self.coordinates.items() if name.startswith(root+'/')}
                # Explicit JAX arrays avoid accidentally taking NumPy/BLAS @
                # for output projections. Stage only two kernels, not the MLP.
                cpu_base = jax.device_put({root:{'mamba':projections}},cpu)
                cpu_q = jax.device_put(subset,cpu)
                folded = jax.jit(lambda b,q:apply_corrections(b,q,
                    head_dim=self.cfg.mamba.head_dim,protected_input_columns=self.columns))(cpu_base,cpu_q)
                values = jax.device_get(folded[root]['mamba'])
                for name in projections:self.current[root]['mamba'][name]=values[name]
        self.folded_sha = tree_digest(self.current)
        self.functions = {}

    def placed(self, x):
        return jax.device_put(x, self.device)

    def layer(self, layer, x, *, teacher=False, mode='normal', period=0,
              compute='bfloat16', override=None):
        self.check()
        cfg = replace(self.cfg, compute_dtype=compute, remat_policy='none')
        # Two shared executables (GQA and Mamba) for normal forward. Layer
        # index is only used for selecting the mixer type, not its numerics.
        attention = teacher or layer in cfg.attention_layer_indices
        key = (attention, teacher, mode, period, compute)
        if key not in self.functions:
            if teacher:
                tc = replace(self.tc, compute_dtype=compute)
                fn = lambda p, h: parts(h, p, cfg, layer, tc)['output']
            elif mode == 'reset':
                def fn(p, h):
                    n = norm(h, p['input_layernorm'], cfg)
                    mixed = mamba_trace(n, p['mamba'], cfg, period)[0]
                    mid = h + mixed
                    return mid + mlp(norm(mid, p['post_attention_layernorm'], cfg), p, cfg)
            else:
                index = cfg.attention_layer_indices[0] if attention else cfg.mamba_layer_indices[0]
                fn = lambda p, h: HybridDecoderLayer(cfg, index).apply({'params': p}, h,
                    jnp.arange(h.shape[1])[None], None)
            self.functions[key] = jax.jit(fn)
        p = override if override is not None else (self.teacher if teacher else self.current)[f'layers_{layer}']
        # FP32 experiment changes projections/MLP dtype; stored BF16 weights,
        # norms and readout casting are unchanged. It is not a retrained model.
        h = x.astype(getattr(jnp, compute))
        return jax.block_until_ready(self.functions[key](self.placed(p), self.placed(h)))

    def embed(self, tokens, compute='bfloat16'):
        return self.placed(self.current['embed_tokens']['embedding'][np.asarray(tokens)].astype(getattr(np, 'float32') if compute=='float32' else BF16))

    def forward_hidden(self, tokens, *, teacher=False, mode='normal', period=0,
                       affected=(), compute='bfloat16', overrides=None, stop=None, precision_map=None):
        x = self.embed(tokens, compute)
        for layer in range(self.cfg.num_layers if stop is None else stop):
            x = self.layer(layer, x, teacher=teacher,
                mode=mode if layer in affected else 'normal', period=period,
                compute=(precision_map or {}).get(layer, compute), override=(overrides or {}).get(layer))
        return x

    def token_losses(self, hidden, tokens, compute='bfloat16'):
        """Tied vocabulary head in 32-position tiles, bounding logit memory."""
        key = ('head', compute)
        if key not in self.functions:
            cfg = replace(self.cfg, compute_dtype=compute)
            def fn(h, scale, embedding, labels):
                h = norm(h, {'scale': scale}, cfg).astype(jnp.bfloat16)
                # Same embedding.attend BF16 output as the canonical tied head.
                logits = jnp.matmul(h, embedding.T).astype(jnp.bfloat16).astype(jnp.float32)
                return -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), labels[..., None], -1)[..., 0]
            self.functions[key] = jax.jit(fn)
        scale = self.placed(self.current['norm']['scale'])
        embedding = self.placed(self.current['embed_tokens']['embedding'])
        values = []
        for offset in range(0, tokens.shape[1]-1, 32):
            self.check()
            end = min(offset+32, tokens.shape[1]-1)
            losses = self.functions[key](hidden[:, offset:end], scale, embedding,
                                        self.placed(tokens[:, offset+1:end+1]))
            values.append(np.asarray(jax.device_get(losses)))
        return np.concatenate(values, axis=1)

    def evaluate(self, windows, **kwargs):
        rows = []
        for t in windows:
            t = np.asarray(t, np.int32)[None]
            h = self.forward_hidden(t, **kwargs)
            rows.append(self.token_losses(h, t, kwargs.get('compute', 'bfloat16'))[0])
        a = np.asarray(rows)
        if not np.isfinite(a).all():
            raise FloatingPointError('nonfinite full-model NLL')
        return dict(nll=float(a.mean()), windows=a.mean(1).tolist(), token_nll=a.tolist())
