import sys
from types import SimpleNamespace

import pytest
import engine_launch


def test_wrapper_sets_title_before_runner_and_preserves_arguments(monkeypatch,capsys):
    calls=[]
    monkeypatch.setitem(sys.modules,'setproctitle',SimpleNamespace(setproctitle=lambda title:calls.append(('title',title))))
    def run(args):
        calls.append(('run',args));return {'status':'deadline_partial'}
    monkeypatch.setitem(sys.modules,'engine_compare',SimpleNamespace(main=run))
    monkeypatch.setattr(engine_launch,'nvml_process_name',lambda:'driver-visible-title')
    result=engine_launch.main(['--work-dir','/work','--max-wall-hours','4','--clear-stop'])
    assert calls==[('title','VLLM:EngineCore'),('run',['--work-dir','/work','--max-wall-hours','4','--clear-stop'])]
    assert result['status']=='deadline_partial'
    assert 'nvml_process_name=driver-visible-title' in capsys.readouterr().out


def test_custom_title_does_not_become_a_training_argument(monkeypatch):
    titles=[];arguments=[]
    monkeypatch.setitem(sys.modules,'setproctitle',SimpleNamespace(setproctitle=titles.append))
    monkeypatch.setitem(sys.modules,'engine_compare',SimpleNamespace(main=arguments.append))
    monkeypatch.setattr(engine_launch,'nvml_process_name',lambda:None)
    engine_launch.main(['--process-title','EngineCore','--no-telegram'])
    assert titles==['EngineCore'] and arguments==[['--no-telegram']]
    with pytest.raises(ValueError):engine_launch.main(['--process-title','bad\nname'])


def test_methods_route_does_not_change_default_compare_or_training_args(monkeypatch):
    arguments=[]
    monkeypatch.setitem(sys.modules,'setproctitle',SimpleNamespace(setproctitle=lambda title:None))
    monkeypatch.setitem(sys.modules,'engine_methods',SimpleNamespace(main=arguments.append))
    monkeypatch.setattr(engine_launch,'nvml_process_name',lambda:None)
    engine_launch.main(['--runner','methods','--max-wall-hours','14','--clear-stop'])
    assert arguments==[['--max-wall-hours','14','--clear-stop']]
