import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tests.test_batch_recovery_161 import initialized_parent
from extent.corpus_switch_162 import ForkStore,training_windows,windows_seen


def test_both_children_restore_same_current_and_adam_reset_best_only(tmp_path):
    from extent.batch_recovery_161 import sha
    from extent.corpus_switch_162 import adam_counts
    p,t,q,spec,tp,tx,payload,parent,source=initialized_parent(tmp_path)
    source['source_arm']=spec['name']
    names=['C4-CONTINUE','EDU-FRESH']
    plan={'branches':{n:dict(source,parent_windows=2 if n.startswith('C4') else 0) for n in names}}
    store=ForkStore(tmp_path/'child','child',parent_root=parent.root,source_plan=plan)
    result=[]
    for name in names:
        saved,meta=store.restore(name,{'arm':name},payload)
        result.append(saved)
        assert meta['step']==118306 and adam_counts(saved['optimizer'])==[2]*6
        for a,b in zip(jax.tree.leaves(saved['parameters']),jax.tree.leaves(payload['parameters'])):
            np.testing.assert_array_equal(a,b)
        for a,b in zip(jax.tree.leaves(saved['best']),jax.tree.leaves(saved['parameters'])):
            np.testing.assert_array_equal(a,b)
        assert int(saved['best_step'])==118306
    for a,b in zip(jax.tree.leaves(result[0]),jax.tree.leaves(result[1])):np.testing.assert_array_equal(a,b)
    assert sha(parent.directory(spec['name'])/'checkpoint.json')==source['manifest_sha']
    c=dict(batch=6,parent=plan,recipes=[dict(name=names[0],data_stream='c4'),dict(name=names[1],data_stream='edu')])
    class Corpus:
        def batch(self,i):return np.full((1,256),i,np.int32)
    assert training_windows(None,dict(c4=Corpus(),edu=Corpus()),c,names[0],118306)[:,0].tolist()==list(range(2,8))
    assert training_windows(None,dict(c4=Corpus(),edu=Corpus()),c,names[1],118306)[:,0].tolist()==list(range(6))


def test_edu_reserved_docs_exclude_all_c4_and_train_excludes_c4_validation(tmp_path,monkeypatch):
    import sqlite3
    from extent.broad_corpus import document_digest
    from scripts import prepare_edu_corpus as edu
    db=sqlite3.connect(tmp_path/'documents.sqlite')
    db.execute('CREATE TABLE documents (digest BLOB PRIMARY KEY, split TEXT)')
    db.executemany('INSERT INTO documents VALUES (?,?)',[(document_digest('c4 train'),'train'),(document_digest('c4 held'),'validation')])
    db.commit();db.close()
    monkeypatch.setattr(edu,'VALIDATION_DOCUMENTS',2)
    docs=[dict(text=t) for t in ('c4 train','edu held','c4 held','edu train')]
    rows=list(edu.rows_from_documents(docs,tmp_path/'documents.sqlite',lambda:None,lambda split:False))
    assert [(r['split'],r['index'],r['text']) for r in rows]==[
        ('validation',0,''),('validation',1,'edu held'),('train',0,''),('train',1,'edu train')]


def test_contract_limits_both_streams_to_unused_tokens_and_pins_revision(tmp_path,monkeypatch):
    import extent.broad_corpus
    import scripts.m3q_corpus_recovery_160
    from extent.corpus_switch_162 import contract
    from scripts.prepare_edu_corpus import REVISION
    parent=dict(format='corpus-switch-parent-v1',corpus_identity='c4',branches={
        'C4-CONTINUE':dict(step=154852,parent_windows=10),'EDU-FRESH':dict(step=154852,parent_windows=0)})
    plan=tmp_path/'source.json';plan.write_text(json.dumps(parent))
    c4=SimpleNamespace(identity='c4',manifest=dict(contract=dict(tokenizer_revision='tok')),arrays={'train':np.zeros(100*256)})
    edu=SimpleNamespace(identity='edu',manifest=dict(contract=dict(dataset='HuggingFaceFW/fineweb-edu',revision=REVISION,
        excluded_c4_identity='c4',sequence_length=256,tokenizer_revision='tok')),arrays={'train':np.zeros(80*256)})
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda path:c4 if path=='c4' else edu)
    monkeypatch.setattr(scripts.m3q_corpus_recovery_160,'contract_for',lambda *a:dict(corpus_identity='c4',code={}))
    c=contract(162,None,1000,'c4','server-balanced',plan,'edu')
    assert c['horizon']==13 and c['batch']==6 and c['selection'].startswith('C4 D')
    assert c['eval_steps'][-1]==13
    a,b=c['recipes']
    assert {k:v for k,v in a.items() if k not in ('name','data_stream')}=={k:v for k,v in b.items() if k not in ('name','data_stream')}
    edu.manifest['contract']['revision']='mutable/wrong'
    with pytest.raises(ValueError):contract(162,None,1000,'c4','server-balanced',plan,'edu')


def test_freeze_rejects_wrong_current_and_pins_same_manifest_for_both(tmp_path):
    from extent.corpus_switch_162 import freeze_parent
    root=tmp_path
    c=dict(protocol='exp161-batch6-v1',code={},corpus_identity='fixed')
    pointer=root/'states/runtime-b6/KD-C4-FRESH/checkpoint.json';pointer.parent.mkdir(parents=True)
    meta=dict(step=154852,contract=dict(c,arm='KD-C4-FRESH'))
    pointer.write_text(json.dumps(meta))
    path=root/'results/runtime-b6.json';path.parent.mkdir()
    report=dict(contract=c,status='deadline_partial',pending_slots=[],runtime={},branches={'KD-C4-FRESH':dict(
        step=154851,durable_step=154852,training_windows_seen=208713,
        final=dict(last=dict(step=154852,panels={'D':dict(windows=[5.231])})))})
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='CURRENT154852'):freeze_parent(root)
    report['branches']['KD-C4-FRESH']['step']=154852;path.write_text(json.dumps(report))
    plan,p=freeze_parent(root)
    a,b=p['branches'].values()
    assert a['metadata']==b['metadata']==meta and a['adam_count']==b['adam_count']==36548
    assert a['parent_windows']==208713 and b['parent_windows']==0
    assert freeze_parent(root)[1]==p


def test_launcher_routes_new_report_and_shared_caches_without_training_reset(tmp_path,monkeypatch):
    import sys,engine_compare
    from scripts import m3q_corpus_switch_162 as runner
    import extent.broad_corpus,extent.corpus_switch_162,extent.hf_artifact_sync,extent.hf_download_retry
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    for name in ('data-pack','data-pack-edu'):
        path=tmp_path/name;path.mkdir();(path/'manifest.json').write_text('{}')
    monkeypatch.setenv('HF_TOKEN','offline-test-not-a-token')
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:SimpleNamespace(manifest=expected))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda **kw:object())
    plan=tmp_path/'corpus-switch/source.json';plan.parent.mkdir();plan.write_text('{}')
    monkeypatch.setattr(extent.corpus_switch_162,'freeze_parent',lambda *a,**kw:(plan,{}))
    monkeypatch.setattr(extent.hf_download_retry,'install_download_retries',lambda:None)
    device=SimpleNamespace(device_kind='CPU mocked launcher',memory_stats=lambda:{})
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(devices=lambda p:[device],config=SimpleNamespace(update=lambda *a:None)))
    called=[]
    def run(args):called.append(args);return dict(status='deadline_partial',pending_slots=[])
    monkeypatch.setattr(runner,'run',run)
    result=engine_compare.main(['--work-dir',str(tmp_path),'--max-wall-hours','9','--no-telegram'])
    assert result['status']=='deadline_partial' and called[0].experiment==162
    assert called[0].edu_dir==str(tmp_path/'data-pack-edu')
    assert called[0].checkpoint_dir==str(tmp_path/'states')
    assert (tmp_path/'runtime-compare-status.json').exists()
    assert not (tmp_path/'runtime-b6-status.json').exists()
@pytest.mark.parametrize('bad_replay',[False,True])
def test_actual_runner_batch6_resume_tokens_and_source_gate(tmp_path,monkeypatch,bad_replay):
    import sys
    from importlib.metadata import version
    from scripts import m3q_corpus_switch_162 as runner
    from scripts.m3q_hidden_distillation_159 import contract_for
    from extent import corpus_switch_162 as model
    from extent.competitor_tpu import token_nll
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.broad_corpus
    p,t,q,spec,tp,tx,parent_payload,parent,source=initialized_parent(tmp_path)
    p.teacher=tp
    source['source_arm']=spec['name']
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    ref=float(token_nll(p.cfg,p.base,q,t,p.columns,spec,jnp.asarray(118306),block_fn=model.block_forward).mean())
    source['current_reference']={'panels':{k:{'windows':[ref+int(bad_replay)]} for k in ('A','B','C','D')}}
    c=contract_for(159,0);c.update(recipes=[spec],horizon=2,eval_steps=[0],batch=6,parent=dict(branches={spec['name']:source}))
    pin=dict(source_step=118304,adam_step=0,best_step=118304,runtime=runtime,data={})
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    monkeypatch.setattr(model,'source_pin',lambda:pin)
    monkeypatch.setattr(model,'restore_source',lambda *a:({'parameters':q},pin))
    monkeypatch.setattr(model,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(model,'prepare',lambda *a:(p.base,q,118304))
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    requested=[]
    class Corpus:
        def __init__(self,*a):pass
        arrays={'train':np.zeros(1000,dtype=np.int32)}
        identity='tiny'
        def validation(self):return np.asarray(t)
        def batch(self,i):
            requested.append(i);return (np.asarray(t)+i)%p.cfg.vocab_size
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',Corpus)
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*a):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    class Session:
        def __init__(self):
            self.result=dict(runtime=runtime,pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),
                branches={spec['name']:dict(step=118306,source_step=118306,history=[],evaluations=[])})
            self.store=ForkStore(tmp_path/'child-run','child',parent_root=parent.root,source_plan={'branches':{spec['name']:source}})
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=1:raise TimeoutError('unit work deadline')
        def persist(self,**kw):return True
        def emit(self,*a):pass
        def checkpoint(self,n,v,step):self.store.save(n,dict(c,arm=n),v,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=162,worker=None,execution='server-balanced',source_plan='unit',corpus_dir='unit',edu_dir='edu',max_steps=1000000000,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert not requested and session.result['cost'].get('accepted_updates',0)==0
    else:
        result=runner.run(args);row=result['branches'][spec['name']]
        assert requested==list(range(2,8))
        assert result['cost']['accepted_input_tokens']==6*t.size
        assert result['cost']['ce_targets']==6*(t.shape[1]-1)
        assert row['step']==118307 and row['training_windows_seen']==8
        assert row['adam_count_at_restore']==[2]*6
        result['cost']['accepted_updates']=0
        runner.run(args)
        assert requested==list(range(2,14)) and row['step']==118308 and row['training_windows_seen']==14
        assert row['adam_count_at_restore']==[3]*6
