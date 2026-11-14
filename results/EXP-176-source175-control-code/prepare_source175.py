import os,sys,time,json,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,os.getcwd())
from extent.weekend_storage_176 import create_ram_workspace,verify_owned
from extent.hf_download_retry import DownloadPolicy
from extent.broad_corpus import digest_file
from huggingface_hub import hf_hub_download
from engine_torch176 import REFERENCE,convert
work=Path('../extent176/source175-diagnostic').resolve();work.mkdir(exist_ok=True)
root=create_ram_workspace(work)
for k,v in {'HF_HOME':'hf-cache','HF_XET_CACHE':'xet','TMPDIR':'tmp'}.items():os.environ[k]=str(root/v);(root/v).mkdir(exist_ok=True)
os.environ.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0')
policy=DownloadPolicy(root/'transport',deadline=time.time()+2400,interval=.25)
r=json.loads(REFERENCE.read_text());directory=root/'source175'
def get(name):return Path(policy.call(hf_hub_download,repo_id=r['repo'],repo_type='dataset',revision=r['revision'],filename=r['prefix']+'/'+name,local_dir=str(directory),token=os.environ['HF_TOKEN']))
meta=json.loads(get('checkpoint.json').read_text());assert hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()==r['metadata_sha256']
assert meta['generation']==r['generation'] and meta['step']==511
chunks={c['file']:c for leaf in meta['leaves'] for c in leaf['chunks']}
print('PINNED SOURCE175',len(chunks),'chunks',sum(c['bytes'] for c in chunks.values()),'bytes',flush=True)
def fetch(item):
 name,c=item;f=get(name)
 assert f.stat().st_size==c['bytes'] and digest_file(f)==c['sha256']
 return c['bytes']
finished=0
with ThreadPoolExecutor(max_workers=4) as pool:
 for index,future in enumerate(as_completed([pool.submit(fetch,c) for c in chunks.items()]),1):
  finished+=future.result()
  if index%25==0 or index==len(chunks):print('downloaded',index,'/',len(chunks),'bytes',finished,flush=True)
policy.call(hf_hub_download,repo_id='Qwen/Qwen3-1.7B-Base',revision=r.get('original_revision','ea980cb0a6c2ae4b936e82123acc929f1cec04c1'),filename='config.json',local_dir=str(root/'config'),token=os.environ['HF_TOKEN'])
convert(root)
d=json.loads((root/'CURRENT.json').read_text());assert d['step']==511 and d['sha256']==digest_file(root/'CURRENT.pt')
verify_owned(root);assert directory.resolve().parent==root.resolve() and not directory.is_symlink();shutil.rmtree(directory)
print('SOURCE175 READY',root,flush=True)
