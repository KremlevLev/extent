"""Build a clean Kaggle T4 notebook from the user's existing deploy-key cell."""
import argparse
import json
from pathlib import Path


def build(source, target):
    original=json.loads(Path(source).read_text(encoding='utf-8'))
    clone=original['cells'][0]['source']
    if isinstance(clone,list):clone=''.join(clone)
    old='''subprocess.run(
    ["git", "clone", "git@github.com:KremlevLev/extent.git"],
    cwd="/kaggle/working",
    env=git_env,
    check=True,
)'''
    new='''repo_dir = Path("/kaggle/working/extent")
if repo_dir.exists():
    subprocess.run(["git", "pull", "--ff-only"], cwd=repo_dir, env=git_env, check=True)
else:
    subprocess.run(["git", "clone", "git@github.com:KremlevLev/extent.git"],
                   cwd="/kaggle/working", env=git_env, check=True)'''
    if old not in clone:raise ValueError('User clone cell differs; inspect before adapting')
    clone=clone.replace(old,new)
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text))
    def code(text):cells.append(dict(cell_type='code',metadata={},source=text,outputs=[],execution_count=None))
    md('''# Extent: Kaggle GPU T4 x2

В Settings выбери **GPU T4 x2**, включи **Internet** и доступ к Secrets:
`EXTENT_DEPLOY_KEY_B64`, `HF_TOKEN`, `EXTENT_HF_CHECKPOINT_REPO`.
Для Telegram нужны те же secrets, что в TPU-ноутбуке; отключение: `--no-telegram`.

Каждый notebook запускает **один** EXP116/117/118/119, обе карты работают
на независимых задачах. Для четырёх экспериментов создай четыре копии этого
ноутбука и замени последнюю ячейку по `results/EXP-116-119-t4-launch.md`.
TPU-сессии и их checkpoints не меняются.

Сначала инженерная проверка на обеих GPU; затем source-NLL replay на реальной
1.7B перед экспериментом. GPU-результатов на момент подготовки ещё нет.
''')
    code(clone)
    md('Установка выполняется до импортов JAX. CUDA12 выбрана для совместимости с драйвером T4; версия среды закреплена отдельно от TPU.')
    code('''import os
import sys
import subprocess
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["EXTENT_HF_REPO_TYPE"] = "dataset"
os.environ.pop("LD_LIBRARY_PATH", None)

gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    text=True,
)
print(gpu_info)
gpu_rows = [r.strip() for r in gpu_info.splitlines() if r.strip()]
if len(gpu_rows) != 2 or any("T4" not in r for r in gpu_rows):
    raise RuntimeError("В Kaggle Settings выбери GPU T4 x2")
if any(int(r.split(",")[1].strip().split(".")[0]) < 525 for r in gpu_rows):
    raise RuntimeError("Драйвер слишком старый для CUDA12 JAX: нужен >=525")

repo_dir = Path("/kaggle/working/extent")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-t4.txt"],
               cwd=repo_dir, check=True)
print("t4_dependencies=PASS")''')
    code('''from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
# Не печатать значения secrets.
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN").strip()
os.environ["EXTENT_HF_CHECKPOINT_REPO"] = secrets.get_secret("EXTENT_HF_CHECKPOINT_REPO").strip()
os.chdir("/kaggle/working/extent")
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

subprocess.run([sys.executable, "-u", "-m", "scripts.m3q_t4_preflight"], check=True)
print("t4_engineering_preflight=PASS")''')
    md('''## Последняя ячейка: один эксперимент

Ниже EXP116 по умолчанию. Меняй только эту ячейку на EXP117/118/119 из
launch-документа. `11.5` — максимум часов **с момента вызова runner**, включая
загрузку и 45 минут резерва. Если настройка ноутбука уже заняла больше
30 минут, уменьшай `--max-wall-hours` по оставшемуся времени сессии.
Работа закончится раньше, если выполнены все зарегистрированные задачи:
runner не расходует квоту пустым ожиданием. `deadline_partial` продолжай той же
ячейкой в новой сессии; данные/source скачиваются заново, completed tasks
читаются с HF. У EXP119 отдельно смотри `durable_step` и pending saves.

JSON, summary и GPU-логи — в `/kaggle/working/extent-results`; небольшие отчёты
и fit-states — в HF `experiments/expXXX-t4/worker-N`. Checkpoint моделей не
дублируется в HF. На Save Version выбирай запуск всех ячеек и сохранение Output.
''')
    code('''from scripts.m3q_t4_campaign import main as run_t4

result = run_t4([
    "--experiment", "116",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python',version='3.12'),
        kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True,
                    language='python',sourceType='notebook',dataSources=[])),cells=cells)
    # All imported cells are sanitized; execution timestamps and outputs removed.
    for index,cell in enumerate(cells):cell['id']=f'extent-t4-{index:02d}'
    Path(target).parent.mkdir(parents=True,exist_ok=True)
    Path(target).write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return notebook


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target')
    a=p.parse_args();build(a.source,a.target)
