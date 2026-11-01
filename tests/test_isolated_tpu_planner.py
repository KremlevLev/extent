import json,sys,time
from extent.isolated_tpu_planner import run_probe,grow


def test_signal_failure_retains_prior_successes():
    def probe(batch):
        if batch==5:return dict(status='worker_failed',returncode=-9)
        return dict(status='measured',report=dict(input_tokens_per_second={1:100,2:300,3:270,4:250}[batch]))
    result=grow(probe)
    assert result['selected_batch']==2 and result['last_successful_batch']==4
    assert result['trials'][-1]['batch']==5 and result['reason']=='failed_probe'


def test_real_worker_exception_missing_result_and_timeout(tmp_path):
    report=tmp_path/'result.json';log=tmp_path/'worker.log'
    row=run_probe([sys.executable,'-c','raise MemoryError("simulated OOM")'],report,log,10)
    assert row['status']=='worker_failed' and 'MemoryError' in log.read_text()
    row=run_probe([sys.executable,'-c','pass'],report,log,10)
    assert row['status']=='invalid_result'
    row=run_probe([sys.executable,'-c','import time;time.sleep(30)'],report,log,.1)
    assert row['status']=='probe_timeout'


def test_worker_must_report_finite_unaccepted_measurement(tmp_path):
    report=tmp_path/'result.json';log=tmp_path/'worker.log'
    code='from pathlib import Path;Path('+repr(str(report))+').write_text('+repr(json.dumps(dict(finite=True,input_tokens_per_second=123.,accepted_updates=0)))+')'
    result=run_probe([sys.executable,'-c',code],report,log,10)
    assert result['status']=='measured'


def test_no_fit_does_not_select_or_loop_forever():
    result=grow(lambda b:dict(status='worker_failed'))
    assert result['selected_batch'] is None and len(result['trials'])==1
    result=grow(lambda b:dict(status='measured',report=dict(input_tokens_per_second=b)),max_batch=8)
    assert result['selected_batch']==8 and len(result['trials'])==8


def test_every_integer_including_7_and_13_is_measured():
    seen=[]
    def probe(batch):
        seen.append(batch)
        return dict(status='worker_failed') if batch==14 else dict(status='measured',report=dict(input_tokens_per_second=batch))
    result=grow(probe)
    assert seen==list(range(1,15))
    assert result['selected_batch']==13 and result['last_successful_batch']==13
