import os,sys,time,json
from pathlib import Path
sys.path.insert(0,os.getcwd())
work=Path('../extent176/reference-diagnostic').resolve();work.mkdir(exist_ok=True)
from extent.weekend_storage_176 import create_ram_workspace
from extent.hf_download_retry import DownloadPolicy
from engine_torch170 import restore_remote
from huggingface_hub import hf_hub_download
root=create_ram_workspace(work)
os.environ['HF_HOME']=str(root/'hf-cache');os.environ['HF_XET_CACHE']=str(root/'xet');os.environ['TMPDIR']=str(root/'tmp');(root/'tmp').mkdir(exist_ok=True)
policy=DownloadPolicy(root/'transport',deadline=time.time()+2400)
print('restoring verified CURRENT9227 to owned tmpfs',flush=True)
restore_remote(root,policy,prefix='experiments/exp176-torch-full-ce-v1')
policy.call(hf_hub_download,repo_id='Qwen/Qwen3-1.7B-Base',filename='config.json',revision='ea980cb0a6c2ae4b936e82123acc929f1cec04c1',local_dir=str(root/'config'),token=os.environ.get('HF_TOKEN'))
print('RESTORE COMPLETE',root,flush=True)
