import json,sys,subprocess,os
from pathlib import Path
import numpy as np
import pytest


def test_cpu_supervisor_import_does_not_import_jax():
    result=subprocess.run([sys.executable,'-c',"import engine_tpu169,sys;assert 'jax' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_large_work_directory_rejects_ssd(tmp_path,monkeypatch):
    import engine_tpu169
    original=Path.read_text
    def read(path,*a,**kw):
        if str(path).replace('\\','/')=='/proc/self/mountinfo':return '1 0 0:1 / / rw - ext4 /dev/x rw'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='tmpfs'):engine_tpu169.ram_root(tmp_path)


def test_supervisor_keeps_known_fit_and_runs_same_batch_pilots_then_resume(tmp_path,monkeypatch):
    import engine_tpu169 as launch
    import extent.isolated_tpu_planner as planner
    import extent.notifications
    commands=[];notices=[]
    monkeypatch.setattr(launch,'ram_root',lambda p:dict(verified_tmpfs=True))
    monkeypatch.setattr(extent.notifications,'send_telegram_message',lambda text:notices.append(text))
    class Child:
        def wait(self,timeout=None):return 0
    def spawn(command,**kw):commands.append(command);return Child()
    monkeypatch.setattr(launch.subprocess,'Popen',spawn)
    def probe(command,*a,**kw):
        batch=int(command[command.index('--batch')+1])
        return dict(status='worker_failed',returncode=-9) if batch==4 else dict(status='measured',report=dict(input_tokens_per_second={1:1000,2:3000,3:2000}[batch],accepted_updates=0,finite=True))
    monkeypatch.setattr(planner,'run_probe',probe)
    # environment mutates only this test's copy; restore it afterwards.
    for key in ('JAX_PLATFORMS','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TMPDIR','XLA_PYTHON_CLIENT_PREALLOCATE','EXTENT_HF_CHECKPOINT_REPO','EXTENT_HF_REPO_TYPE','EXTENT_HF_DOWNLOAD_DEADLINE_UNIX','EXTENT_HF_SAVE_DEADLINE_UNIX'):
        monkeypatch.setenv(key,os.environ.get(key,''))
    result=launch.main(['--work-dir',str(tmp_path/'ram'),'--output-dir',str(tmp_path/'out')])
    assert result['selected_batch']==2 and result['status']=='finished'
    assert result['trials']['compiled']['last_successful_batch']==3
    workers=[c for c in commands if 'worker' in c]
    assert [c[c.index('--mode')+1] for c in workers]==['control','ce-only','control']
    assert all(c[c.index('--batch')+1]=='2' for c in workers)
    assert len(notices)==2


def test_supervisor_preflight_failure_writes_report_without_starting_worker(tmp_path,monkeypatch):
    import engine_tpu169 as launch
    import extent.notifications
    monkeypatch.setattr(launch,'ram_root',lambda p:(_ for _ in ()).throw(MemoryError('cgroup RAM limit')))
    monkeypatch.setattr(extent.notifications,'send_telegram_message',lambda text:None)
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not start worker'))
    for key in ('EXTENT_HF_CHECKPOINT_REPO','EXTENT_HF_REPO_TYPE','EXTENT_HF_DOWNLOAD_DEADLINE_UNIX','EXTENT_HF_SAVE_DEADLINE_UNIX'):
        monkeypatch.setenv(key,os.environ.get(key,''))
    report=launch.main(['--work-dir',str(tmp_path/'ram'),'--output-dir',str(tmp_path/'out')])
    assert report['status']=='failed' and report['error']['type']=='MemoryError'
    assert json.loads((tmp_path/'out/extent-m3q-tpu169.json').read_text())['status']=='failed'


def test_pair_comparison_refuses_unequal_tails_and_missing_panels():
    from engine_tpu169 import compare_pilots
    row=dict(status='pilot_completed',accepted_input_tokens=1024,data_windows=4,
        final=dict(last=dict(panels={p:dict(nll=5.) for p in 'ABCDE'})))
    assert compare_pilots({'control':row,'ce-only':row},{'input_tokens':1024})['comparable']
    assert not compare_pilots({'control':row,'ce-only':dict(row,accepted_input_tokens=768)},{'input_tokens':1024})['comparable']
    assert not compare_pilots({'control':row,'ce-only':dict(row,final={})},{'input_tokens':1024})['comparable']


@pytest.mark.parametrize('corrupt,wrong_source',[(False,False),(True,False),(False,True)])
def test_reuse_data_bundle_checks_source_and_every_file(tmp_path,monkeypatch,corrupt,wrong_source):
    import hashlib
    from extent import tpu_data_169 as bundle
    import extent.hf_artifact_sync
    content=b'fixed token bytes'
    files={n+'/'+f:dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest()) for n in bundle.NAMES for f in ('manifest.json','train.bin','validation.bin')}
    value=dict(format='exp169-data-v1',identities={'data-pack':'c4','data-pack-edu-600m':'edu'},files=files)
    (tmp_path/'tpu169-data.json').write_text(json.dumps(value))
    def download(path,remote,hub):path.write_bytes(b'corrupt' if corrupt else content);return True
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',download)
    monkeypatch.setattr(bundle,'inventory',lambda root:value)
    report=dict(corpus_identity='wrong' if wrong_source else 'c4',edu_identity='edu')
    if corrupt or wrong_source:
        with pytest.raises(ValueError):bundle.restore(tmp_path,None,report)
    else:assert bundle.restore(tmp_path,None,report)


def test_contract_source_clock_mode_and_unbounded_integer_batch(tmp_path,monkeypatch):
    from tests.test_full_stream_166 import small_contract
    from extent import tpu_diagnostic_169 as model
    import scripts.prepare_extended_edu
    plan,_=small_contract(tmp_path,monkeypatch);source=json.loads(plan.read_text());source['lr_multiplier']=1
    source['full_source'].update(step=260386,windows=838601,adam_count=105534);plan.write_text(json.dumps(source))
    monkeypatch.setattr(scripts.prepare_extended_edu,'verify_prefix',lambda *a:dict(verified=True))
    monkeypatch.setenv('EXTENT169_BATCH','13');monkeypatch.setenv('EXTENT169_MODE','control')
    c=model.contract(169,None,1000000000,'c4','server-balanced',plan,'edu')
    assert c['batch']==13 and c['full_source']['adam_count']==105534 and not c['eval_steps']
    assert c['recipes'][0]['alpha']==.5
    monkeypatch.setenv('EXTENT169_MODE','ce-only')
    other=model.contract(169,None,1000000000,'c4','server-balanced',plan,'edu')
    assert other['recipes'][0]['alpha']==0 and other['full_source']==c['full_source']
    assert len(json.dumps(c))<1000000


def test_source_fork_preserves_adam_without_copy_and_own_state_wins(tmp_path,monkeypatch):
    import jax.numpy as jnp
    from extent import tpu_diagnostic_169 as model
    from extent.recovery_durability import RecoveryCheckpointStore
    q={'w':jnp.arange(4,dtype=jnp.float32)}
    payload=dict(parameters=q,optimizer=tuple(jnp.asarray(105534) for _ in range(6)),best=q,
        best_step=jnp.asarray(260386),data_windows=jnp.asarray(838601))
    donor=RecoveryCheckpointStore(tmp_path/'source','old');donor.save('FULL-KD',{'parent':168},payload,260386)
    pointer=donor.directory('FULL-KD')/'checkpoint.json';manifest=tmp_path/'manifest.json';manifest.write_bytes(pointer.read_bytes())
    child=model.ForkStore(tmp_path/'own','new',source=dict(manifest_sha=model.sha(manifest),source_prefix='old'),
        parent_root=donor.root,manifest_file=manifest)
    monkeypatch.setattr(model,'adam_counts',lambda state:[int(v) for v in state])
    restored,meta=child.restore('FULL-KD',{'child':169},payload)
    assert meta['step']==260386 and int(restored['data_windows'])==838601
    assert not child.directory('FULL-KD').exists()
    child.save('FULL-KD',{'child':169},restored,260387)
    pointer.write_text('parent missing')
    assert child.restore('FULL-KD',{'child':169},payload)[1]['step']==260387


def test_real_compiled_probe_audit_two_updates_diagnostics_do_not_mutate_source(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jnp
    from tests.test_batch_recovery_161 import initialized_parent
    from extent.method_sweep_163 import coordinates,optimizer,adam_counts
    from extent import tpu_diagnostic_169 as model
    from extent.tpu_step_169 import plan_kernel,diagnostics
    p,t,current,_,teacher,_,_,_,_=initialized_parent(tmp_path)
    spec=dict(model.recipes(169)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=optimizer(q,spec);state=tx.init(q)
    before=jax.tree.map(np.array,(q,state))
    tokens=jnp.concatenate([t,(t+1)%p.cfg.vocab_size,(t+2)%p.cfg.vocab_size])
    monkeypatch.setenv('EXTENT169_ENGINE','compiled');monkeypatch.setenv('EXTENT169_PROBE','0')
    engine,plan=plan_kernel(p.cfg,p.tc,p.columns,spec,tx,q,state,p.base,teacher,tokens,jnp.asarray(260386.,jnp.float32),lambda:None,lambda v:None)
    assert plan['selected_batch']==3 and plan['gradient_audit']['passed'] and plan['accepted_updates']==0
    first=engine(q,state,p.base,teacher,tokens,jnp.asarray(260386.,jnp.float32))
    second=engine(first[0],first[1],p.base,teacher,tokens,jnp.asarray(260387.,jnp.float32))
    assert adam_counts(second[1])==[2]*6 and bool(second[2]['finite'])
    result=diagnostics(p.cfg,p.tc,p.columns,spec,tx,dict(parameters=q,optimizer=state),p.base,teacher,tokens,260386,lambda:None)
    assert result['accepted_updates']==0 and len(result['layer_profile']['layers'])==p.cfg.num_layers
    assert len(result['objective_pairs'])==2
    assert all(np.isfinite(v['cosine']) for v in result['ce_kl_by_layer'].values())
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves((q,state))):np.testing.assert_array_equal(a,b)
    json.dumps(result,allow_nan=False)


def test_eight_virtual_devices_odd_batch_reference_and_teacher_head(tmp_path):
    code='''
import tempfile
from pathlib import Path
import jax,jax.numpy as jnp
from tests.test_batch_recovery_161 import initialized_parent
from extent.method_sweep_163 import coordinates,optimizer,adam_counts
from extent.tpu_diagnostic_169 import recipes,teacher_validation
from extent.tpu_step_169 import plan_kernel,shard_bytes
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
p,t,current,_,teacher,_,_,_,_=initialized_parent(Path(tempfile.mkdtemp()))
spec=dict(recipes(169)[0],align_layers=list(range(p.cfg.num_layers)))
mesh=create_v5e_mesh(jax.devices('cpu'));assert len(jax.devices())==8
q=coordinates(p.base,current,p.cfg,p.columns,spec);layout=coordinate_layout(q,mesh)
q=jax.device_put(q,layout);tx=optimizer(q,spec);state,_=sharded_optimizer(tx,q,layout,mesh)
base=jax.device_put(p.base,named_sharding_tree(p.base,mesh));teacher=jax.device_put(teacher,named_sharding_tree(teacher,mesh))
tokens=jax.device_put(jnp.concatenate([t,t,t]),batch_sharding(mesh))
engine,report=plan_kernel(p.cfg,p.tc,p.columns,spec,tx,q,state,base,teacher,tokens,jnp.asarray(260386.,jnp.float32),lambda:None,lambda v:None)
assert report['gradient_audit']['passed'] and len(shard_bytes(q))==8
out=engine(q,state,base,teacher,tokens,jnp.asarray(260386.,jnp.float32))
assert bool(out[2]['finite']) and adam_counts(out[1])==[1]*6
from extent.tpu_step_169 import TPUReference
reference=TPUReference(p.cfg,p.tc,p.columns,spec,tx)
value=teacher_validation(p.cfg,reference,base,teacher,tokens[:1],lambda:None)
assert value['nll']>0
print('eight_device_odd_batch_teacher_and_gradient=PASS')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=8',EXTENT169_ENGINE='compiled',EXTENT169_PROBE='0')
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
