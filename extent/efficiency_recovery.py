"""EXP111-113 optimizer, data and deterministic validation-only LR selection."""
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import optax
from extent.calibration_data import load_wikitext2_tokens,load_pg19_tokens
from extent.qwen_source import QWEN3_1_7B_BASE
from extent.stable_gradient_clip import stable_clip_by_global_norm
from extent.batch_recovery import make_step

TOTAL_STEPS=32768
PILOT_STEPS=1024
SEEDS=(123,456)
PEAKS={'LR-1e-5':1e-5,'LR-1e-4':1e-4,'LR-3e-4':3e-4}
PROBE_STEPS=(0,1024,4096,8192,16384,24576,32768)
DATA=(("train","train",65536000,TOTAL_STEPS),("validation","validation",122880,64),
      ("locked_test","validation",147456,64),("reproduction","test",32768,64))

def load_tokens(cache):
    kw=dict(tokenizer_repo=QWEN3_1_7B_BASE.repo_id,tokenizer_revision=QWEN3_1_7B_BASE.revision)
    tokens={name:load_wikitext2_tokens(n*256,cache,token_offset=offset,dataset_split=split,
       dataset_config="wikitext-103-raw-v1",**kw).reshape(n,256) for name,split,offset,n in DATA}
    tokens['pg19_test']=load_pg19_tokens(128*256,cache,token_offset=393216,dataset_split='test',**kw).reshape(128,256)
    return tokens

def token_manifest(tokens):
    return {k:dict(sha256=hashlib.sha256(v.tobytes()).hexdigest(),shape=list(v.shape),dtype=str(v.dtype)) for k,v in tokens.items()}

def schedule(step,peak,adapter=False):
    return jnp.asarray(peak,jnp.float32) if adapter else peak*jnp.minimum(jnp.asarray(step,jnp.float32)/128,1.)

def optimizer(parameters,adaptive=False,accumulation=1,*,peak=3e-4,adapter=False):
    return optax.chain(stable_clip_by_global_norm(1),optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
        optax.identity(),optax.scale_by_schedule(lambda n:-schedule(n,peak,adapter)))

def choose_lr(branches):
    scores={}
    for arm in PEAKS:
        rows=[branches.get(str(s),{}).get(arm,{}) for s in SEEDS]
        if all(r.get('step')==PILOT_STEPS and not r.get('failed') and 'pilot_validation' in r for r in rows):
            value=float(np.mean([r['pilot_validation']['nll'] for r in rows]))
            if np.isfinite(value):scores[arm]=value
    if not scores:raise ValueError('no LR pilot finite at BOTH seeds; no speed superiority conclusion')
    return min(scores,key=lambda a:(scores[a],PEAKS[a])),scores

def crossing_intervals(diagnostics,threshold):
    rows=[d for d in diagnostics if 'reproduction' in d]
    for i in range(1,len(rows)):
        if rows[i-1]['reproduction']['nll']<=threshold and rows[i]['reproduction']['nll']<=threshold:
            before=next((r for r in reversed(rows[:i-1]) if r['reproduction']['nll']>threshold),None)
            return dict(reached=True,last_above=None if before is None else before['step'],
                first_below=rows[i-1]['step'],confirmed_at=rows[i]['step'])
    return dict(reached=False,right_censored_at=rows[-1]['step'] if rows else None)

def cost_totals(branches):
    rows=[r for arms in branches.values() for r in arms.values()]
    attempts=sum(r.get('attempted_updates',0) for r in rows)
    return dict(attempted_updates=attempts,input_tokens=256*attempts,ce_targets=255*attempts,
        training_seconds=sum(r.get('training_seconds',0.) for r in rows),
        diagnostic_seconds=sum(r.get('diagnostic_seconds',0.) for r in rows),
        preparation_cost='UNKNOWN',replays_and_failed_attempts_included=True)

def pilot_parity(row,pilot,tolerance=1e-4):
    error=abs(row['validation']['nll']-pilot['pilot_validation']['nll'])
    if not np.isfinite(error) or error>tolerance:raise ValueError('selected pilot replay differs; stop comparison')
    return dict(validation_absolute_error=error,tolerance=tolerance)
