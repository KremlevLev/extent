"""Frozen EXP156 evaluation. Parent/data helpers do not import JAX.

Choice formats follow EleutherAI lm-evaluation-harness task definitions;
this is a separately versioned zero-shot JAX evaluator, not its CLI output.
"""
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import replace
import hashlib
import json
import math
import re
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASKS=('arc_easy','arc_challenge','piqa','winogrande','hellaswag','mmlu')
SEED=156123


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def pin():
    return json.loads((ROOT/'results/EXP-156-source-pin.json').read_text())


def contract():
    code={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in
          ('extent/quality_benchmark_156.py','scripts/m3q_quality_benchmark_156.py')}
    return dict(protocol='exp156-frozen-quality-v1',source_step=80260,source_subset='CURRENT parameters only',
                source_pin_sha=digest_json(pin()),data_pin_sha=digest_json(json.loads((ROOT/'results/EXP-156-data-pin.json').read_text())),
                tasks=list(TASKS),fewshot=0,seed=SEED,choice_batch=2,max_context=2048,
                primary_metric={k:('acc' if k in ('winogrande','mmlu') else 'acc_norm') for k in TASKS},
                precision='source BF16 weights/activations, FP32 correction arithmetic and log-softmax; no FP16 conversion',
                selection='snapshot fixed before benchmarks; no task result selects weights',
                normalization='acc and acc_norm (character count of unprefixed answer text); Wino/MMLU acc primary',
                code=code,limitations='single checkpoint; zero-shot diagnostic, not official 5-shot MMLU or generation pass@1; fresh means unused in our adaptation/evaluation, not absent from Qwen pretraining')


def stable_order(rows,key='id'):
    return sorted(rows,key=lambda r:hashlib.sha256(f'{SEED}:{r[key]}'.encode()).digest())


def format_doc(task,doc,index):
    """Return explicit (context, continuation) pairs, including Wino suffix scoring."""
    subject=doc.get('subject','')
    if task.startswith('arc_'):
        texts=doc['choices']['text'];labels=doc['choices']['label']
        gold=list(labels).index(str(doc['answerKey']))
        ctx='Question: '+doc['question']+'\nAnswer:'
        pairs=[(ctx,' '+x) for x in texts];norms=[max(1,len(x)) for x in texts]
    elif task=='piqa':
        texts=[doc['sol1'],doc['sol2']];gold=int(doc['label'])
        ctx='Question: '+doc['goal']+'\nAnswer:'
        pairs=[(ctx,' '+x) for x in texts];norms=[max(1,len(x)) for x in texts]
    elif task=='winogrande':
        pre,post=doc['sentence'].split('_',1);gold=int(doc['answer'])-1
        suffix=' '+post.strip();pairs=[(pre+doc[k],suffix) for k in ('option1','option2')]
        if not post.strip():raise ValueError('empty Winogrande suffix')
        norms=[len(suffix)]*2
    elif task=='hellaswag':
        def clean(x):return re.sub(r'\[.*?\]','',x.strip().replace(' [title]','. ')).replace('  ',' ')
        ctx=clean(doc['activity_label']+': '+doc['ctx_a']+' '+doc['ctx_b'].capitalize())
        texts=[clean(x) for x in doc['endings']];gold=int(doc['label'])
        pairs=[(ctx,' '+x) for x in texts];norms=[max(1,len(x)) for x in texts]
    elif task=='mmlu':
        ctx='The following are multiple choice questions (with answers) about '+subject.replace('_',' ')+'.\n\n'
        ctx+=doc['question'].strip()+'\n'+'\n'.join(f'{letter}. {text}' for letter,text in zip('ABCD',doc['choices']))+'\nAnswer:'
        pairs=[(ctx,' '+x) for x in 'ABCD'];norms=[1]*4;gold=int(doc['answer'])
    else:raise ValueError(task)
    if not 0<=gold<len(pairs):raise ValueError('gold label outside options')
    if any(not y.strip() for x,y in pairs):raise ValueError('empty answer')
    return dict(id=f'{task}:{index}',task=task,index=index,subject=subject,gold=gold,pairs=pairs,norms=norms)


def balanced_mmlu(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['subject']].append(r)
    queues=[deque(stable_order(groups[k])) for k in sorted(groups)]
    out=[]
    while any(queues):
        for q in queues:
            if q:out.append(q.popleft())
    return out


def interleave(tasks):
    """All tasks advance together, so a deadline doesn't leave only ARC results."""
    queues={k:deque(v) for k,v in tasks.items()};out=[]
    while any(queues.values()):
        for k in TASKS:
            if queues[k]:out.append(queues[k].popleft())
    return out


def encode_pair(tokenizer,context,continuation):
    # Canonical trailing-whitespace transfer keeps token boundaries unambiguous.
    n=len(context)-len(context.rstrip())
    if n:continuation=context[-n:]+continuation;context=context[:-n]
    prefix=tokenizer.encode(context,add_special_tokens=False)
    whole=tokenizer.encode(context+continuation,add_special_tokens=False)
    if whole[:len(prefix)]!=prefix:raise ValueError('non-prefix BPE boundary; not scored silently')
    if len(whole)<=len(prefix):raise ValueError('empty tokenized continuation')
    if not prefix:
        prefix=[tokenizer.eos_token_id];whole=prefix+whole
    return whole,len(prefix)


def wilson(correct,n):
    if not n:return None
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [middle-half,middle+half]


def summarize_items(items):
    out={}
    for task in TASKS:
        rows=[x for x in items.values() if x.get('task')==task and x.get('status')=='scored']
        if not rows:continue
        r={}
        for metric in ('acc','acc_norm'):
            correct=sum(x[metric] for x in rows);r[metric]=correct/len(rows);r[metric+'_ci95']=wilson(correct,len(rows))
        r.update(n=len(rows),chance=float(np.mean([1/len(x['scores']) for x in rows])))
        if task=='mmlu':
            r['subjects']={s:dict(n=sum(x['subject']==s for x in rows),acc=float(np.mean([x['acc'] for x in rows if x['subject']==s]))) for s in sorted({x['subject'] for x in rows})}
        out[task]=r
    return out


def paired_results(a,b):
    out={};rng=np.random.default_rng(SEED)
    for task in TASKS:
        ids=sorted(k for k in a.keys()&b.keys() if a[k].get('task')==task and a[k].get('status')==b[k].get('status')=='scored')
        if not ids:continue
        assert all(a[k]['prompt_sha256']==b[k]['prompt_sha256'] and a[k]['gold']==b[k]['gold'] for k in ids)
        result={'n_common':len(ids)}
        for metric in ('acc','acc_norm'):
            diffs=np.array([int(b[k][metric])-int(a[k][metric]) for k in ids],np.float64)
            boot=[float(rng.choice(diffs,size=len(diffs),replace=True).mean()) for _ in range(2000)]
            result[metric]=dict(teacher=float(np.mean([a[k][metric] for k in ids])),hybrid=float(np.mean([b[k][metric] for k in ids])),
                                hybrid_minus_teacher=float(diffs.mean()),paired_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
        out[task]=result
    return out


def prepare_data(cache,tokenizer,check=lambda:None):
    """Pinned Parquet reads, no executable HF dataset scripts or mutable branches."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from extent.campaign_checkpoint import write_json_atomic
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    specs=json.loads((ROOT/'results/EXP-156-data-pin.json').read_text());tasks={};provenance={};panels={}
    for task,spec in specs.items():
        check();file=Path(hf_hub_download(spec['repo'],spec['file'],revision=spec['revision'],repo_type='dataset',local_dir=cache/task))
        actual=file_sha(file)
        if spec.get('sha256',actual)!=actual:raise ValueError('pinned dataset SHA differs: '+task)
        provenance[task]=dict(spec,sha256=actual)
        if task in TASKS:
            docs=pq.read_table(file).to_pylist();rows=[format_doc(task,doc,i) for i,doc in enumerate(docs)]
            tasks[task]=balanced_mmlu(rows) if task=='mmlu' else stable_order(rows)
        else:
            # Bound host memory for the much larger Russian Wikipedia shard.
            docs=[]
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128,columns=['text']):
                docs.extend(batch.to_pylist())
                if len(docs)>=4096:break
            order=np.random.default_rng(SEED).permutation(len(docs));tokens=[];used=[]
            for idx in order:
                check();text=docs[int(idx)]['text'];ids=tokenizer.encode(text,add_special_tokens=False)
                if len(ids)<64:continue
                tokens.extend(ids+[tokenizer.eos_token_id]);used.append(int(idx))
                if len(tokens)>=40960:break
            if len(tokens)<40960:raise ValueError('insufficient fresh token panel: '+task)
            arr=np.asarray(tokens[:40960],np.int32);panels[task]=arr
            provenance[task].update(document_indices=used,token_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                                    windows256=128,context_windows1024=8,eos_between_documents=True)
    ordered=interleave(tasks)
    manifest=dict(protocol='zero-shot-fixed-v1',seed=SEED,counts={k:len(v) for k,v in tasks.items()},
                  questions_sha256=digest_json(ordered),provenance=provenance)
    write_json_atomic(cache/'questions.json',ordered);write_json_atomic(cache/'data-manifest.json',manifest)
    np.savez(cache/'fresh-panels.npz',**panels)
    return ordered,panels,manifest


GENERATION_PROMPTS=(
    ('en','The old lighthouse keeper opened the door and saw'),
    ('en','Water freezes when'),('en','A computer program is'),
    ('en','Question: Why do leaves usually look green?\nAnswer:'),
    ('en','Question: What is the capital of France?\nAnswer:'),
    ('en','Question: Alice has 3 apples and buys 2 more. How many apples does she have?\nAnswer:'),
    ('en','The difference between a list and a dictionary in Python is'),
    ('en','A short explanation of how rain forms:\n'),
    ('ru','Однажды утром молодой инженер открыл ноутбук и обнаружил'),
    ('ru','Вода замерзает при'),('ru','Компьютерная программа — это'),
    ('ru','Вопрос: Почему листья обычно зелёные?\nОтвет:'),
    ('ru','Вопрос: Какая столица Франции?\nОтвет:'),
    ('ru','Вопрос: У Маши было 3 яблока, она купила ещё 2. Сколько стало?\nОтвет:'),
    ('ru','Разница между списком и словарём в Python заключается в том, что'),
    ('ru','Краткое объяснение того, как образуется дождь:\n'),
    ('code','def add(a, b):\n    """Return the sum of a and b."""\n'),
    ('code','def is_even(n):\n    """Return True if n is even."""\n'),
    ('code','def reverse_string(s):\n    """Return the reversed string."""\n'),
    ('code','def factorial(n):\n    """Return n factorial for nonnegative integer n."""\n'),
    ('code','def maximum(values):\n    """Return the largest element of a nonempty list."""\n'),
    ('code','def count_vowels(text):\n    """Count English vowels in text."""\n'),
    ('code','def unique_in_order(values):\n    """Remove duplicates while preserving the original order."""\n'),
    ('code','def binary_search(values, target):\n    """Return index of target in sorted values, or -1."""\n'),
)


class InferenceEngine:
    """Inference-only, right padding, bounded head tiles, no optimizer allocation."""
    def __init__(self,cfg,tc,base,q,columns,spec,role,check=lambda:None):
        import jax
        import jax.numpy as jnp
        from extent.reset_streamed_gpu import MemoryCheckedJit
        from extent.rank_growth_153 import block_forward
        from extent.plateau_forensics import norm,mixer,mlp
        self.cfg,self.tc,self.base,self.q,self.columns,self.spec,self.role=cfg,tc,base,q,columns,spec,role
        if not cfg.tie_word_embeddings or (role=='teacher' and not tc.tie_word_embeddings):
            raise ValueError('EXP156 requires the tied-head Qwen3-1.7B-Base contract')
        self.check=check;self.functions={};self.fallbacks=[]
        for attention in (False,True) if role=='hybrid' else (True,):
            index=cfg.attention_layer_indices[0] if attention else cfg.mamba_layer_indices[0]
            if role=='hybrid':
                fn=lambda p,r,h,index=index:block_forward(cfg,index,p,r,h,columns,spec,jnp.asarray(1.,jnp.float32))[0]
            else:
                def fn(p,r,h):
                    h=h+mixer(norm(h,p['input_layernorm'],cfg),p,cfg,0,tc)
                    return h+mlp(norm(h,p['post_attention_layernorm'],cfg),p,cfg)
            self.functions[attention]=MemoryCheckedJit(fn)
        def head(features,labels,embedding,norm_params):
            shape=labels.shape;count=labels.size;tile=16;pad=(-count)%tile
            features=jnp.pad(features.reshape(-1,cfg.hidden_size),((0,pad),(0,0))).reshape(-1,tile,cfg.hidden_size)
            labels=jnp.pad(labels.reshape(-1),(0,pad)).reshape(-1,tile)
            def body(_,xy):
                x,y=xy;x=norm(x,norm_params,cfg).astype(getattr(jnp,cfg.param_dtype))
                logits=(x@embedding.T).astype(getattr(jnp,cfg.compute_dtype)).astype(jnp.float32)
                return None,-jnp.take_along_axis(jax.nn.log_softmax(logits,-1),y[:,None],axis=-1)[:,0]
            _,v=jax.lax.scan(body,None,(features,labels));return v.reshape(-1)[:count].reshape(shape)
        self._head=MemoryCheckedJit(head)
        def logits(x,embedding,norm_params):
            x=norm(x,norm_params,cfg).astype(getattr(jnp,cfg.param_dtype))
            return (x@embedding.T).astype(getattr(jnp,cfg.compute_dtype)).astype(jnp.float32)
        self._logits=MemoryCheckedJit(logits)

    def head(self,features,labels):
        # Keep the 622MB real embedding as a dynamic argument, not duplicated
        # executable constants for every answer-length bucket.
        return self._head(features,labels,self.base['embed_tokens']['embedding'],self.base['norm'])

    def logits(self,features):
        return self._logits(features,self.base['embed_tokens']['embedding'],self.base['norm'])

    def hidden(self,tokens):
        import jax.numpy as jnp
        self.check();h=self.base['embed_tokens']['embedding'][jnp.asarray(tokens)].astype(getattr(jnp,self.cfg.compute_dtype))
        for i in range(self.cfg.num_layers):
            self.check();kind=self.role=='teacher' or i in self.cfg.attention_layer_indices
            h=self.functions[kind](self.base[f'layers_{i}'],self.q[str(i)] if self.role=='hybrid' else {},h)
        return h

    def token_nll(self,tokens):
        import jax.numpy as jnp
        h=self.hidden(tokens);return np.asarray(self.head(h[:,:-1],jnp.asarray(tokens[:,1:])))

    def score_pairs(self,pairs,tokenizer,max_context=2048,batch_size=2):
        import jax.numpy as jnp
        requests=[encode_pair(tokenizer,*pair) for pair in pairs]
        if max(len(x) for x,n in requests)>max_context:raise OverflowError('context exceeds fixed evaluation cap; no truncation')
        out=[]
        for offset in range(0,len(requests),batch_size):
            group=requests[offset:offset+batch_size];length=max(len(x) for x,n in group)
            bucket=next(n for n in (128,256,512,1024,2048) if n>=length)
            width=max(len(x)-n for x,n in group);width=16*math.ceil(width/16)
            t=np.full((len(group),bucket),tokenizer.eos_token_id,np.int32)
            indices=np.zeros((len(group),width),np.int32);labels=np.zeros_like(indices);mask=np.zeros_like(indices,dtype=bool)
            for j,(ids,start) in enumerate(group):
                t[j,:len(ids)]=ids;count=len(ids)-start
                indices[j,:count]=np.arange(start-1,len(ids)-1);labels[j,:count]=ids[start:];mask[j,:count]=True
            try:h=self.hidden(t)
            except (MemoryError,RuntimeError) as exc:
                if len(group)==1 or not (isinstance(exc,MemoryError) or 'RESOURCE_EXHAUSTED' in str(exc) or 'out of memory' in str(exc).lower()):raise
                self.fallbacks.append(dict(batch=len(group),bucket=bucket,reason='memory; retry same choices individually'))
                out.extend(self.score_pairs(pairs[offset:offset+batch_size],tokenizer,max_context,batch_size=1));continue
            selected=h[jnp.arange(len(group))[:,None],jnp.asarray(indices)]
            losses=np.asarray(self.head(selected,jnp.asarray(labels)))
            if not np.all(np.isfinite(losses)):raise FloatingPointError('nonfinite choice losses')
            for j,(ids,start) in enumerate(group):out.append(dict(loglikelihood=-float(losses[j][mask[j]].sum(dtype=np.float64)),tokens=len(ids)-start,context_tokens=start,total_tokens=len(ids)))
        return out

    def generate(self,tokenizer,prompt,mode,index,max_new=64):
        import jax.numpy as jnp
        ids=tokenizer.encode(prompt,add_special_tokens=False);initial=len(ids)
        if not ids or initial+max_new>512:raise ValueError('generation prompt does not fit fixed512 budget')
        rng=np.random.default_rng(SEED+index);generated=[];reason='max_new_tokens'
        for _ in range(max_new):
            self.check();bucket=256 if len(ids)<=256 else 512
            arr=np.full((1,bucket),tokenizer.eos_token_id,np.int32);arr[0,:len(ids)]=ids
            h=self.hidden(arr);z=np.asarray(self.logits(h[:,len(ids)-1]))[0].astype(np.float64)
            if not np.all(np.isfinite(z)):raise FloatingPointError('nonfinite generation logits')
            if mode=='greedy':token=int(np.argmax(z))
            else:
                top=np.argpartition(z,-50)[-50:];v=z[top]/.7;prob=np.exp(v-v.max());prob/=prob.sum();token=int(rng.choice(top,p=prob))
            generated.append(token);ids.append(token)
            if token==tokenizer.eos_token_id:reason='eos';break
        grams=[tuple(generated[i:i+4]) for i in range(max(0,len(generated)-3))]
        return dict(prompt=prompt,mode=mode,text=tokenizer.decode(generated,skip_special_tokens=True),
                    token_ids=generated,prompt_tokens=initial,stop_reason=reason,
                    repeated_4gram_fraction=(1-len(set(grams))/len(grams)) if grams else 0.,
                    caveat='raw base-model continuation; generated code not executed; no pass@1 or inference speed claim')


def load_engine(role,cache,hub,check=lambda:None,device=None):
    import jax
    import ml_dtypes
    from flax import traverse_util
    from extent.qwen_source import QWEN3_1_7B_BASE,teacher_config_from_spec
    from extent.weight_mapping import teacher_qwen_mappings
    from extent.config import load_config
    from extent.t4_patient import ensure_qwen
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml');cfg=replace(cfg,remat_policy='none')
    tc=teacher_config_from_spec(QWEN3_1_7B_BASE,param_dtype='bfloat16',compute_dtype='bfloat16',remat_policy='none')
    cache=Path(cache);q={};columns=0;spec={}
    if role=='teacher':
        reader=ensure_qwen(cache/'patient/qwen',check);flat={}
        for entry in teacher_qwen_mappings(tc):
            check();value=reader.read(entry.source)
            if entry.transform=='transpose':value=value.T
            flat[tuple(entry.target.split('/'))]=np.asarray(value,dtype=ml_dtypes.bfloat16)
        base=traverse_util.unflatten_dict(flat)
    else:
        from extent.reset_long_patient import Patient
        from extent.continuation_matrix import base_coordinates,prepare
        from extent.reset_long_fixed_sources import complete_coordinates
        from extent.t4_patient import read_parameter_chunks
        from extent.hf_artifact_sync import restore_artifact
        p=pin();sourcehub=replace(hub,repo_id=p['repo_id'],repo_type='dataset',revision=p['revision'])
        directory=cache/'source156';pointer=directory/'checkpoint.json';directory.mkdir(parents=True,exist_ok=True)
        if not pointer.exists():restore_artifact(pointer,p['path'],sourcehub)
        if hashlib.sha256(pointer.read_bytes()).hexdigest()!=p['manifest_sha256']:raise ValueError('source156 manifest SHA mismatch')
        meta=json.loads(pointer.read_text())
        if meta!=p['metadata']:raise ValueError('source156 metadata differs')
        for name,expected in meta['contract']['code'].items():
            if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('source numerical code differs: '+name)
        q=complete_coordinates(read_parameter_chunks(meta,directory,p['path'].rsplit('/',1)[0],sourcehub,check),meta['contract']['model'])
        patient=Patient(cache/'patient',hub,jax.devices('cpu')[0],check,seed=123)
        columns=patient.columns;spec=dict(meta['contract']['recipes'][0],mode='exact')
        ancestor=base_coordinates(hub,cache/'immutable',check)
        with jax.default_device(jax.devices('cpu')[0]):base,q,_=prepare(patient,spec,dict(parameters=q),p,ancestor)
        del patient,ancestor
    device=device or jax.devices('gpu')[0]
    base=jax.device_put(base,device);q=jax.device_put(q,device)
    return InferenceEngine(cfg,tc,base,q,columns,spec,role,check)
