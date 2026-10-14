"""Real GPU setup + tiny production-vs-streamed parity; no patient download."""
from dataclasses import replace
import json
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from extent import HybridForCausalLM, tiny_config
from extent.model import _attention_source_config
from extent.t4_patient import Patient


def tiny_patient(device):
    cfg=replace(tiny_config(), mla=replace(tiny_config().mla,implementation='qwen3_gqa'),
        param_dtype='bfloat16',compute_dtype='bfloat16',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    model=HybridForCausalLM(cfg);t=jnp.arange(40,dtype=jnp.int32)[None]%cfg.vocab_size
    params=model.init(jax.random.key(116),t)['params']
    patient=Patient.__new__(Patient)
    patient.cfg=cfg;patient.tc=_attention_source_config(cfg);patient.device=device
    patient.check=lambda:None;patient.current=jax.device_get(params);patient.teacher=patient.current
    patient.functions={}
    return patient,model,params,np.asarray(t)


def check_streaming(device):
    p,model,params,t=tiny_patient(device)
    with jax.default_device(device):
        reference=jax.jit(lambda weights,t: model.apply({'params':weights},t))(p.placed(params),p.placed(t))
        expected=-jnp.take_along_axis(jax.nn.log_softmax(reference[:,:-1].astype(jnp.float32),-1),
                                     p.placed(t[:,1:,None]),-1)[...,0]
        hidden=p.forward_hidden(t)
        actual=p.token_losses(hidden,t)
    error=float(np.max(np.abs(actual-np.asarray(expected))))
    # Separate layer executables change BF16 fusion/rounding at GQA residual
    # boundaries. This new GPU protocol registers a bounded token-NLL check,
    # not bitwise parity. The actual patient also needs its separate .03 gate.
    if not np.isfinite(error) or error>.01:
        raise ValueError(f'tiled streamed NLL differs from production model: {error}')
    return dict(max_abs_token_nll=error,tolerance=.01,passed=True,
        scope='bounded BF16 differences from executable boundaries; not bitwise identity')


def main():
    from importlib.metadata import version
    required={'jax':'0.11.0','jaxlib':'0.11.0','flax':'0.12.8','optax':'0.2.8'}
    actual={name:version(name) for name in required}
    if actual!=required:raise ValueError(f'GPU dependency versions differ: {actual}')
    devices=jax.devices('gpu')
    report=dict(versions=actual,devices=[d.device_kind for d in devices],
        parity=[check_streaming(d) for d in devices])
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':main()
