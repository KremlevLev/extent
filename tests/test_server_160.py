import time
from types import SimpleNamespace
import pytest
from extent.server_control import install_session_control,storage_status
from scripts.m3q_server_160 import parser,remaining_hours


def test_stop_file_shortens_save_deadline_and_allows_finalization(tmp_path):
    stop=tmp_path/'STOP';stop.touch()
    calls=[]
    s=SimpleNamespace(check=lambda:calls.append(1),wall=time.monotonic()+21*3600,result={})
    args=SimpleNamespace(stop_file=str(stop),work_root=None,save_reserve_minutes=18)
    install_session_control(s,args)
    with pytest.raises(TimeoutError,match='soft stop'):s.check()
    assert s.wall-time.monotonic()<=18*60 and not calls
    s.finalizing=True;s.check();assert calls==[1]


def test_storage_guard_preserves_files_and_enters_save(tmp_path):
    p=tmp_path/'keep.bin';p.write_bytes(b'x'*100)
    with pytest.raises(OSError,match='budget'):storage_status(tmp_path,1e-9,0)
    assert p.read_bytes()==b'x'*100
    s=SimpleNamespace(check=lambda:None,wall=time.monotonic()+3600,result={})
    args=SimpleNamespace(stop_file=None,work_root=str(tmp_path),save_reserve_minutes=18,max_local_gb=1e-9,min_free_gb=0)
    install_session_control(s,args)
    with pytest.raises(TimeoutError,match='storage budget'):s.check()
    assert p.exists()


def test_hours_include_preparation_and_can_be_changed():
    assert parser().parse_args(['--max-wall-hours','21']).max_wall_hours==21
    assert remaining_hours(100,21,clock=lambda:3700)==20
    from scripts.m3q_corpus_recovery_160 import session_budget
    assert session_budget(160,21,18,'server-balanced')[0]==21*3600-18*60
    with pytest.raises(ValueError):session_budget(160,21,18)


def test_fast_steps_do_not_produce_hundreds_of_local_snapshots(tmp_path):
    saved=[]
    s=SimpleNamespace(check=lambda:None,checkpoint=lambda n,p,k:saved.append((n,k)),
                      wall=time.monotonic()+3600,result={})
    args=SimpleNamespace(stop_file=None,work_root=None,save_reserve_minutes=18)
    install_session_control(s,args)
    for step in range(100):s.checkpoint('a',{},step)
    assert saved==[('a',0)]
    s.finalizing=True;s.checkpoint('a',{},100)
    assert saved==[('a',0),('a',100)]


def test_launcher_passes_single_gpu_and_remaining_time_to_real_runner_entry(tmp_path,monkeypatch):
    import json,sys
    from pathlib import Path
    from scripts import m3q_server_160 as server
    from scripts import m3q_corpus_recovery_160 as runner
    import extent.broad_corpus
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    corpus=tmp_path/'data-pack';corpus.mkdir();(corpus/'manifest.json').write_text('{}')
    monkeypatch.setenv('HF_TOKEN','offline-test-not-a-token')
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:SimpleNamespace(manifest=expected))
    import extent.hf_download_retry
    monkeypatch.setattr(extent.hf_download_retry,'install_download_retries',lambda:None)
    monkeypatch.setattr(server,'remaining_hours',lambda *a:20.5)
    device=SimpleNamespace(device_kind='CPU fake only',memory_stats=lambda:{})
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(devices=lambda p:[device]))
    calls=[]
    def run(args):
        calls.append(args)
        return dict(status='deadline_partial',pending_slots=[])
    monkeypatch.setattr(runner,'run',run)
    result=server.main(['--work-dir',str(tmp_path),'--max-wall-hours','21','--no-telegram'])
    assert result['status']=='deadline_partial' and len(calls)==1
    assert calls[0].execution=='server-balanced' and calls[0].worker is None
    assert calls[0].max_wall_hours==20.5 and calls[0].max_new_hf_gb==32
    assert (tmp_path/'runtime-status.json').exists()


def test_server_session_uses_neutral_local_names_and_separate_hf_namespace(tmp_path,monkeypatch):
    import extent.hf_artifact_sync
    from scripts.m3q_corpus_recovery_160 import parser,session
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    args=parser().parse_args(['--execution','server-balanced','--corpus-dir','unused',
        '--max-wall-hours','21','--output-dir',str(tmp_path/'results'),
        '--checkpoint-dir',str(tmp_path/'states')])
    s=session(args,dict(protocol='unit'),None)
    assert s.path.name=='runtime-report.json'
    assert s.prefix=='experiments/exp160-corpus-server-v1'
