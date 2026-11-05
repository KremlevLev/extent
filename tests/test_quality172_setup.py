"""Exercise the Kaggle pip-free environment route without CUDA downloads."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def test_worker_diagnostic_exposes_early_crash_without_report(tmp_path):
    from engine_quality172 import worker_diagnostic
    (tmp_path/'worker-0.log').write_text('Traceback: missing CUDA runtime\n')
    row = worker_diagnostic(tmp_path, 0, 1)
    assert row['returncode'] == 1 and row['status'] == 'missing_report'
    assert 'missing CUDA runtime' in row['log_tail']
    (tmp_path/'worker-1.json').write_text(json.dumps(dict(status='failed', error='actual failure')))
    row = worker_diagnostic(tmp_path, 1, 2)
    assert row['error'] == 'actual failure' and row['returncode'] == 2


def test_notebook_avoids_ensurepip_and_targets_isolated_python():
    notebook = json.loads((Path(__file__).resolve().parents[1]/'notebooks/extent-t4-quality-172.ipynb').read_text())
    source = ''.join(notebook['cells'][2]['source'])
    ast.parse(source)
    assert "'venv', '--without-pip', ENV" in source
    assert "PIP = [sys.executable, '-m', 'pip', '--python', PYTHON]" in source
    assert source.count("subprocess.run(PIP + ['install'") == 2
    assert "[PYTHON, '-m', 'pip'" not in source


def test_real_install_into_pip_free_venv_with_ensurepip_forbidden(tmp_path):
    env = tmp_path/'env'
    # Fail loudly if the environment creation ever invokes ensurepip.
    code = '''import runpy,sys,venv
def forbidden(*args,**kwargs):
    raise AssertionError("ensurepip must never run")
venv.EnvBuilder._setup_pip = forbidden
sys.argv = ['venv','--without-pip',sys.argv[1]]
runpy.run_module('venv',run_name='__main__')
'''
    subprocess.run([sys.executable, '-c', code, str(env)], check=True, capture_output=True, text=True)
    target = env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    probe = 'import importlib.util; assert importlib.util.find_spec("pip") is None'
    subprocess.run([str(target), '-c', probe], check=True, capture_output=True)
    wheel = tmp_path/'extent_setup_probe-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('extent_setup_probe.py', 'VALUE = 172\n')
        z.writestr('extent_setup_probe-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: extent-setup-probe\nVersion: 1.0\n')
        z.writestr('extent_setup_probe-1.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        z.writestr('extent_setup_probe-1.0.dist-info/RECORD', '')
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(target), 'install',
                    '--no-index', '--no-deps', str(wheel)], check=True, capture_output=True, text=True)
    subprocess.run([str(target), '-c', 'import extent_setup_probe; assert extent_setup_probe.VALUE == 172'],
                   check=True, capture_output=True)
