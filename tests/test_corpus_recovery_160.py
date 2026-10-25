import numpy as np
import pytest
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from tests.test_hidden_distillation_159 import fixture
from extent import corpus_recovery_160 as distill


def test_streamed_teacher_reference_matches_full_teacher_nll():
    from extent.hidden_distillation_159 import DistillationStep,optimizer
    from extent.competitor_tpu import teacher_hidden
    from extent.context_recovery import tiled_nll
    p,t,q,spec,tp=fixture(0)
    engine=DistillationStep(p.cfg,p.tc,p.columns,spec,optimizer(q,spec))
    result=distill.teacher_validation(p.cfg,engine,p.base,tp,np.asarray(t),lambda:None)
    expected=tiled_nll(p.cfg,p.base,teacher_hidden(p.cfg,p.tc,tp,t),t)
    assert abs(result['nll']-float(expected.mean()))<.01
    assert result['input_tokens']==t.size


def test_real_contract_pins_corpus_and_identical_kd(tmp_path):
    from extent.broad_corpus import build_corpus,C4_REVISION
    from extent.qwen_source import QWEN3_1_7B_BASE as donor
    from scripts.m3q_corpus_recovery_160 import contract_for,parser
    contract=dict(dataset='allenai/c4',revision=C4_REVISION,config='en',tokenizer=donor.repo_id,tokenizer_revision=donor.revision,
                  vocab_size=donor.vocab_size,eos_token_id=1,sequence_length=256,
                  train_tokens=1024,validation_tokens=128*256)
    stream=[dict(split='validation',index=i,text=f'validation {i} '+('a '*110)) for i in range(128)]
    stream += [dict(split='train',index=i,text=f'train {i} '+('a '*110)) for i in range(4)]
    build_corpus(tmp_path,stream,lambda t:[2]*255,contract,min_free_bytes=0)
    a,b=[contract_for(160,i,corpus_dir=tmp_path) for i in (0,1)]
    assert a['corpus_identity']==b['corpus_identity']
    left,right=a['recipes'][0],b['recipes'][0]
    assert {k:v for k,v in left.items() if k not in ('name','data_stream')}=={k:v for k,v in right.items() if k not in ('name','data_stream')}
    assert left['hidden_beta']==right['hidden_beta']==0
    assert 'extent/broad_corpus.py' in a['code'] and 'extent/hidden_distillation_159.py' in a['code']
    assert parser().parse_args(['--corpus-dir',str(tmp_path)]).max_steps==262144
    server=contract_for(160,None,1000000000,tmp_path,'server-balanced')
    assert len(server['recipes'])==2 and server['execution']=='server-balanced'
    assert server['execution_namespace']=='experiments/exp160-corpus-server-v1'

@pytest.mark.parametrize('bad_replay',[False,True])
@pytest.mark.parametrize('execution',['t4-pair','server-balanced'])
def test_actual_worker_c4_cursor_resume_and_source_gate(tmp_path,monkeypatch,bad_replay,execution):
    import sys,json,time
    from importlib.metadata import version
    from scripts import m3q_corpus_recovery_160 as runner
    from extent.competitor_tpu import token_nll
    from extent.recovery_durability import RecoveryCheckpointStore
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.competitor_tpu
    p,t,q,spec,tp=fixture(1);spec.update(hidden_beta=0.,data_stream='c4');p.teacher=tp;cursor=118304
    import extent.broad_corpus
    seen=[]
    class Corpus:
        identity='fixture'
        def __init__(self,*a):pass
        def validation(self):return np.asarray(t)
        def batch(self,index):
            seen.append(index);return np.asarray(t)
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',Corpus)
    from scripts.m3q_hidden_distillation_159 import contract_for
    c=contract_for(159,1);c['recipes']=[spec];c['horizon']=2;c['eval_steps']=[0]
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    ref=float(token_nll(p.cfg,p.base,q,t,p.columns,spec,jnp.asarray(cursor),block_fn=distill.block_forward).mean())
    pin=dict(source_step=cursor,adam_step=0,best_step=cursor,runtime=runtime,data={},reference={k:{'windows':[ref+int(bad_replay)]} for k in 'ABC'})
    payload=dict(parameters=q,best=q,best_step=jnp.asarray(cursor,jnp.int32))
    monkeypatch.setattr(distill,'source_pin',lambda:pin)
    monkeypatch.setattr(distill,'restore_source',lambda *a:(payload,pin));monkeypatch.setattr(distill,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(distill,'prepare',lambda *a:(p.base,q,cursor))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:extent.hf_artifact_sync.HubArtifactConfig('offline','dataset',''))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*args):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    class Session:
        def __init__(self):
            self.result=dict(branches={},runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),pending_slots=[])
            self.store=RecoveryCheckpointStore(tmp_path/'states','159');self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=1:raise TimeoutError('test work deadline')
        def emit(self,*a):pass
        def persist(self,**kw):return True
        def checkpoint(self,n,v,step):self.store.save(n,dict(c,arm=n),v,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *args:session)
    args=SimpleNamespace(experiment=160,worker=1,execution=execution,corpus_dir='fixture',max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert session.result['cost'].get('accepted_updates',0)==0
    else:
        result=runner.run(args);row=next(iter(result['branches'].values()))
        assert result['cost']['accepted_updates']==1 and result['cost']['teacher_input_tokens']==t.size
        assert result['teacher_shared_head_verified'] and row['step']==118305
        assert row['first_update_health']['health']['hidden_mse']==0
        assert row['adam_count_at_restore']==[0]*6
        assert session.store.metadata(spec['name'],dict(c,arm=spec['name']))['step']==118305
        if execution=='server-balanced':assert result['teacher_resident_bytes']>0 and 'resident GPU' in result['teacher_execution']


        assert seen==[0]
        session.result['cost'].update(accepted_updates=0)
        runner.run(args)
        assert seen==[0,1]

@pytest.mark.parametrize('fail_worker',[False,True])
def test_parent_spawns_two_isolated_gpus_and_preserves_failure(tmp_path,monkeypatch,fail_worker):
    import json
    from scripts import m3q_corpus_recovery_160 as runner
    from extent.campaign_checkpoint import write_json_atomic
    import extent.hf_artifact_sync,extent.hf_download_retry,extent.reset_quiet
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    uploaded=[];envs=[]
    def upload(path,*args,**kw):uploaded.append(json.loads(path.read_text()))
    monkeypatch.setattr(extent.hf_artifact_sync,'upload_artifact',upload)
    monkeypatch.setattr(extent.hf_download_retry,'install_download_retries',lambda:None)
    class Notice:
        def __init__(self,*a):pass
        def emit(self,*a):pass
    monkeypatch.setattr(extent.reset_quiet,'ParentNotifier',Notice)
    class Process:
        def __init__(self,cmd,env,**kw):
            envs.append(env['CUDA_VISIBLE_DEVICES']);i=int(cmd[cmd.index('--worker')+1]);self.failed=fail_worker and i==1
            out=Path(cmd[cmd.index('--output-dir')+1]);c=runner.contract_for(160,i,corpus_dir='fixture')
            write_json_atomic(out/f'extent-m3q-corpus-recovery-160-worker-{i}.json',dict(contract=c,status='failed' if self.failed else 'completed',branches={}))
        def poll(self):return 1 if self.failed else 0
    from pathlib import Path
    from scripts.m3q_hidden_distillation_159 import contract_for
    monkeypatch.setattr(runner,'contract_for',lambda exp,i,*a,**kw:contract_for(159,i))
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    args=['--corpus-dir','fixture','--output-dir',str(tmp_path/'out'),'--cache-dir',str(tmp_path/'cache'),'--checkpoint-dir',str(tmp_path/'states'),'--no-telegram']
    if fail_worker:
        with pytest.raises(RuntimeError):runner.main(args)
        assert uploaded[-1]['status']=='failed'
    else:
        r=runner.main(args);assert r['status']=='completed' and len(r['workers'])==2
    assert envs==['0','1']
