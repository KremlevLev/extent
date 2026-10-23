"""Actual tiny BF16 inference, answer boundaries, masks, statistics and source pins."""
from dataclasses import replace
from pathlib import Path
import json
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_rank_growth_153 import mature_source
from extent import quality_benchmark_156 as bench
from extent.rank_growth_153 import grow_coordinates,block_forward
from extent.competitor_tpu import token_nll
from extent.qwen3_teacher import Qwen3ForCausalLM


class Tokenizer:
    eos_token_id=0
    def encode(self,text,**kw):return [ord(c)%120+1 for c in text]
    def decode(self,ids,**kw):return ' '.join(map(str,ids))


def test_task_formats_boundaries_and_balanced_mmlu():
    arc=bench.format_doc('arc_easy',dict(question='q',choices={'text':['a','b'],'label':['A','B']},answerKey='B'),7)
    assert arc['gold']==1 and arc['pairs']==[('Question: q\nAnswer:',' a'),('Question: q\nAnswer:',' b')]
    w=bench.format_doc('winogrande',dict(sentence='The _ was tired.',option1='boy',option2='girl',answer='2'),0)
    assert w['pairs']==[('The boy',' was tired.'),('The girl',' was tired.')]
    t=Tokenizer();ids,start=bench.encode_pair(t,'hello  ',' world')
    assert ids==t.encode('hello   world') and start==5
    class Broken(Tokenizer):
        def encode(self,text,**kw):return [len(text)]
    with pytest.raises(ValueError,match='BPE boundary'):bench.encode_pair(Broken(),'a','b')
    rows=[dict(id=str(i),subject='a' if i<8 else 'b') for i in range(10)]
    order=bench.balanced_mmlu(rows)
    assert {r['subject'] for r in order[:2]}=={'a','b'} and len(order)==10
    queues={k:[dict(id=k+str(i)) for i in range(2)] for k in bench.TASKS}
    assert [r['id'] for r in bench.interleave(queues)[:6]]==[k+'0' for k in bench.TASKS]


def test_real_trained_growth_scores_only_continuation_and_is_causal():
    p,t,q=mature_source();spec=bench.pin()['metadata']['contract']['recipes'][0]
    q=grow_coordinates(p.base,q,p.cfg,256)
    for i in p.cfg.mamba_layer_indices:
        for f in q[str(i)]['extra_lora'].values():f['b']=jnp.full_like(f['b'],.001)
    e=bench.InferenceEngine(p.cfg,p.tc,p.base,q,p.columns,spec,'hybrid')
    tokenizer=Tokenizer();pairs=[('Q:',' aa'),('Q:',' b')]
    scores=e.score_pairs(pairs,tokenizer)
    for pair,score in zip(pairs,scores):
        ids,start=bench.encode_pair(tokenizer,*pair)
        ref=np.asarray(token_nll(p.cfg,p.base,q,jnp.asarray([ids]),p.columns,spec,jnp.asarray(80260.),block_fn=block_forward))
        expected=-float(ref[0,start-1:].sum(dtype=np.float64))
        assert abs(score['loglikelihood']-expected)/score['tokens']<.03
    # Extra right-hand tokens/padding cannot change likelihood of existing targets.
    first=np.array([[1,2,3,4,5,6]],np.int32)
    second=np.array([[1,2,3,4,99,100]],np.int32)
    np.testing.assert_allclose(e.token_nll(first)[:,:3],e.token_nll(second)[:,:3],atol=.003,rtol=0)
    separate=e.score_pairs(pairs,tokenizer,batch_size=1)
    assert all(abs(a['loglikelihood']-b['loglikelihood'])/a['tokens']<.03 for a,b in zip(scores,separate))
    with pytest.raises(OverflowError):e.score_pairs([('q'*2100,' a')],tokenizer)
    for mode in ('greedy','sample'):
        a=e.generate(tokenizer,'Q:',mode,0,max_new=3);b=e.generate(tokenizer,'Q:',mode,0,max_new=3)
        assert a['token_ids']==b['token_ids'] and len(a['token_ids'])<=3


def test_original_teacher_matches_independent_full_flax_forward():
    p,t,_=mature_source();tc=replace(p.tc,tie_word_embeddings=True);model=Qwen3ForCausalLM(tc)
    base=model.init(jax.random.key(156),t)['params']
    e=bench.InferenceEngine(p.cfg,tc,base,{},0,{},'teacher')
    logits=model.apply({'params':base},t)
    ref=-jnp.take_along_axis(jax.nn.log_softmax(logits[:,:-1],-1),t[:,1:,None],-1)[...,0]
    np.testing.assert_allclose(e.token_nll(np.asarray(t)),ref,atol=.03,rtol=0)
    selected=np.asarray(e.logits(e.hidden(np.asarray(t))[:,-1]))
    np.testing.assert_allclose(selected,np.asarray(logits[:,-1]),atol=.03,rtol=0)
    traced=jax.make_jaxpr(e._head.function)(e.hidden(np.asarray(t))[:,:2],t[:,1:3],base['embed_tokens']['embedding'],base['norm'])
    assert not traced.consts  # real622MB embedding must not be an executable constant
    assert any(v.shape==base['embed_tokens']['embedding'].shape for v in traced.in_avals)


def test_actual_transformers_fp32_bridge_and_bf16_diagnostic(tmp_path,monkeypatch):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM as TorchQwen
    from flax import traverse_util
    from extent.weight_mapping import teacher_qwen_mappings
    from scripts import m3q_quality_benchmark_156 as runner
    p,t,_=mature_source();tc=replace(p.tc,tie_word_embeddings=True)
    base=Qwen3ForCausalLM(tc).init(jax.random.key(156),t)['params']
    cfg=Qwen3Config(vocab_size=tc.vocab_size,hidden_size=tc.hidden_size,
        intermediate_size=tc.intermediate_size,num_hidden_layers=tc.num_layers,
        num_attention_heads=tc.num_attention_heads,num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,rms_norm_eps=tc.rms_norm_eps,rope_theta=tc.rope_theta,
        max_position_embeddings=tc.max_position_embeddings,tie_word_embeddings=True)
    cfg._attn_implementation='eager';model=TorchQwen(cfg).eval()
    flat=traverse_util.flatten_dict(base);state=model.state_dict()
    for entry in teacher_qwen_mappings(tc):
        value=np.asarray(flat[tuple(entry.target.split('/'))].astype(jnp.float32))
        if entry.transform=='transpose':value=value.T
        state[entry.source]=torch.from_numpy(value.copy())
    state['lm_head.weight']=state['model.embed_tokens.weight']
    model.load_state_dict(state,strict=True);folder=tmp_path/'patient/qwen';folder.mkdir(parents=True)
    model.save_pretrained(folder)
    engine=bench.InferenceEngine(p.cfg,tc,base,{},0,{},'teacher')
    result=runner.teacher_reference_check(engine,Tokenizer(),tmp_path)
    assert result['passed'] and result['tolerance']==.003
    assert len(result['rows'])==5 and all(r['passed'] for r in result['rows'])
    assert result['benchmark_dtype']=='bfloat16'
    # Reproduce the class of observed startup failure: a BF16 backend delta
    # must be visible, while the independent FP32 formula gate remains strict.
    original=engine.token_nll
    monkeypatch.setattr(engine,'token_nll',lambda tokens:original(tokens)+.2)
    shifted=runner.teacher_reference_check(engine,Tokenizer(),tmp_path)
    assert shifted['passed'] and not shifted['bf16_backend_diagnostic']['used_as_equivalence_gate']
    assert max(r['max_abs_token_nll_error'] for r in shifted['bf16_backend_diagnostic']['rows'])>.1
    # A real mapping/formula error in the original checkpoint is still rejected.
    with torch.no_grad():model.model.norm.weight.mul_(2)
    model.save_pretrained(folder)
    invalid=runner.teacher_reference_check(engine,Tokenizer(),tmp_path)
    assert not invalid['passed']
    (tmp_path/'teacher-reference-test.json').write_text(json.dumps(dict(
        reference=result,bf16_drift_injected=shifted,wrong_hf_norm=invalid)),encoding='utf-8')


def test_only_empty_failed_preflight_can_cross_code_repair():
    from scripts import m3q_quality_benchmark_156 as runner
    c=bench.contract();old=dict(c,code={'old':'hash'})
    prior=dict(status='failed',contract=old,workers={'teacher':dict(status='failed',items={},nll={},generations={})})
    assert runner.empty_failed_preflight(prior,c)
    prior['workers']['teacher']['items']={'piqa:0':{}}
    assert not runner.empty_failed_preflight(prior,c)
    prior['workers']['teacher']['items']={};prior['contract']=dict(old,source_step=0)
    assert not runner.empty_failed_preflight(prior,c)


def test_paired_statistics_use_common_items_not_different_endpoints():
    a={};b={}
    for i in range(5):
        row=dict(task='piqa',status='scored',gold=0,prompt_sha256=str(i),scores=[{},{}],acc=i%2==0,acc_norm=i%2==0)
        a[str(i)]=row;b[str(i)]=dict(row,acc=True,acc_norm=True)
    b['unmatched']=dict(b['0'])
    result=bench.paired_results(a,b)['piqa']
    assert result['n_common']==5 and result['acc']['hybrid_minus_teacher']==pytest.approx(.4)
    assert bench.summarize_items(a)['piqa']['acc']==.6
    assert 0<bench.wilson(0,10)[1]<.5
    b['0']['prompt_sha256']='changed'
    with pytest.raises(AssertionError):bench.paired_results(a,b)


def test_pin_no_training_code_changes_and_plan():
    from scripts.m3q_quality_benchmark_156 import main
    p=bench.pin();assert p['source_step']==p['metadata']['step']==80260
    for file,expected in p['metadata']['contract']['code'].items():
        assert hashlib.sha256((bench.ROOT/file).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected
    c=main(['--plan-only']);assert c['fewshot']==0 and c['source_step']==80260
    assert len(c['tasks'])==6 and c['max_context']==2048


def test_actual_worker_deadline_journal_resume_and_source_gate(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from importlib.metadata import version
    from scripts import m3q_quality_benchmark_156 as runner
    import extent.hf_artifact_sync,extent.t4_scale_data
    from extent.hf_artifact_sync import HubArtifactConfig
    import transformers
    c=bench.contract();monkeypatch.setattr(runner,'contract',lambda:c)
    output=tmp_path/'out';output.mkdir();cache=tmp_path/'cache';data=cache/'benchmark-data';data.mkdir(parents=True)
    rows=[]
    for task in bench.TASKS:
        rows.append(dict(id=task+':0',task=task,gold=0,subject='',pairs=[('Q:',' a'),('Q:',' b')],norms=[1,1]))
    arrays={key:np.ones(40960,np.int32) for key in ('english_stories','russian_wikipedia')}
    manifest=dict(questions_sha256=bench.digest_json(rows),provenance={k:dict(token_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()})
    (data/'questions.json').write_text(json.dumps(rows));(data/'data-manifest.json').write_text(json.dumps(manifest));np.savez(data/'fresh-panels.npz',**arrays)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    monkeypatch.setattr(runner,'pin',lambda:dict(runtime=runtime,data={},reference={p:dict(windows=[1.]) for p in 'ABC'}))
    real_devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:real_devices('cpu')[:1] if platform=='gpu' else real_devices(platform))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:HubArtifactConfig('offline','dataset',''))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({p:np.ones((1,256),np.int32) for p in 'ABC'},{}))
    class Engine:
        fallbacks=[]
        def __init__(self):self.count=0;self.stop=True
        def token_nll(self,t):return np.ones((1,t.shape[1]-1),np.float32)
        def score_pairs(self,pairs,*a,**kw):
            self.count+=1
            if self.stop and self.count>1:raise TimeoutError('test interruption')
            return [dict(loglikelihood=-float(i+1),tokens=1) for i in range(len(pairs))]
        def generate(self,tokenizer,prompt,mode,index):return dict(prompt=prompt,mode=mode,text='test',token_ids=[1])
    engine=Engine();monkeypatch.setattr(bench,'load_engine',lambda *a:engine)
    args=SimpleNamespace(worker='hybrid',output_dir=str(output),cache_dir=str(cache),deadline_unix=time.time()+3600)
    r=runner.worker(args);assert r['status']=='deadline_partial' and len(r['items'])==1
    path=output/'extent-m3q-quality-156-hybrid.json';saved=json.loads(path.read_text());saved['items']={};path.write_text(json.dumps(saved))
    engine.stop=False;engine.count=0
    r=runner.worker(args)
    assert r['status']=='completed' and len(r['items'])==6 and len(r['generations'])==36
    assert engine.count==5  # first completed item restored from its journal
    assert r['source_replay']['passed'] and all(r['summary'][k]['n']==1 for k in bench.TASKS)


def test_parent_dispatch_two_gpus_one_report_no_remote_training(tmp_path,monkeypatch):
    import transformers,huggingface_hub
    from scripts import m3q_quality_benchmark_156 as runner
    import extent.hf_artifact_sync
    from extent.hf_artifact_sync import HubArtifactConfig
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:HubArtifactConfig('offline','dataset',''))
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    uploads=[];monkeypatch.setattr(extent.hf_artifact_sync,'upload_artifacts_together',lambda files,*a,**kw:uploads.append(files))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda *a,**kw:None)
    manifest=dict(counts={k:1 for k in bench.TASKS})
    monkeypatch.setattr(runner,'prepare_data',lambda *a:([],{},manifest))
    commands=[];c=bench.contract()
    old=dict(c,code={'legacy':'preflight-v1'})
    failed_workers={role:dict(contract=old,role=role,status=status,items={},nll={},generations={})
                    for role,status in (('teacher','failed'),('hybrid','loading'))}
    failed=dict(contract=old,status='failed',workers=failed_workers)
    (tmp_path/'extent-m3q-quality-156.json').write_text(json.dumps(failed))
    for role,row in failed_workers.items():
        (tmp_path/f'extent-m3q-quality-156-{role}.json').write_text(json.dumps(row))
    class Child:
        returncode=0
        def __init__(self,cmd,**kw):
            role=cmd[cmd.index('--worker')+1];commands.append((role,kw['env']['CUDA_VISIBLE_DEVICES'],cmd))
            assert kw['env']['JAX_PLATFORMS']=='cuda,cpu' and '--deadline-unix' in cmd
            item=dict(task='piqa',status='scored',gold=0,prompt_sha256='same',scores=[{},{}],acc=True,acc_norm=True)
            r=dict(contract=c,role=role,status='completed',items={'piqa:0':item},generations={},nll={})
            (tmp_path/f'extent-m3q-quality-156-{role}.json').write_text(json.dumps(r))
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    r=runner.main(['--max-wall-hours','11.5','--output-dir',str(tmp_path),'--cache-dir',str(tmp_path/'cache'),'--no-telegram'])
    assert r['status']=='completed' and r['paired']['piqa']['n_common']==1
    archive=Path(r['retired_failed_preflight']['path'])
    assert json.loads(archive.read_text())==failed
    assert [(role,gpu) for role,gpu,cmd in commands]==[('teacher','0'),('hybrid','1')]
    assert uploads and (tmp_path/'extent-m3q-quality-156.json').exists()
