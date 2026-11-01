"""Publish/reuse existing verified packs, avoiding600M retokenization on queued TPU."""
import argparse,json,time,os
from pathlib import Path
from extent.broad_corpus import PackedCorpus,digest_file,write_json

PREFIX='experiments/exp169-data-c4-edu600m-v1'
NAMES=('data-pack','data-pack-edu','data-pack-edu-600m')


def inventory(root):
    from scripts.prepare_extended_edu import verify_prefix
    root=Path(root);packs={n:PackedCorpus(root/n) for n in NAMES}
    proof=verify_prefix(root/NAMES[1],root/NAMES[2])
    if [packs[n].manifest['contract']['train_tokens'] for n in NAMES]!=[100000000,100000000,600000000]:
        raise ValueError('require original100M C4,100M Edu and verified600M Edu')
    files={}
    for name in NAMES:
        for leaf in ('manifest.json','train.bin','validation.bin'):
            key=name+'/'+leaf;path=root/key;files[key]=dict(bytes=path.stat().st_size,sha256=digest_file(path))
    return dict(format='exp169-data-v1',identities={n:p.identity for n,p in packs.items()},proof=proof,files=files)


def restore(root,hub,source_report):
    from extent.hf_artifact_sync import restore_artifact
    root=Path(root);descriptor=root/'tpu169-data.json'
    if not descriptor.exists() and not restore_artifact(descriptor,PREFIX+'/inventory.json',hub):return False
    value=json.loads(descriptor.read_text())
    expected={n+'/'+f for n in NAMES for f in ('manifest.json','train.bin','validation.bin')}
    if value.get('format')!='exp169-data-v1' or set(value['files'])!=expected:raise ValueError('invalid pack inventory paths')
    if value['identities']['data-pack']!=source_report['corpus_identity'] or value['identities']['data-pack-edu-600m']!=source_report['edu_identity']:
        raise ValueError('published packs differ from immutable EXP168 source')
    for name,meta in value['files'].items():
        path=root/name
        if path.exists() and path.stat().st_size==meta['bytes'] and digest_file(path)==meta['sha256']:continue
        path.parent.mkdir(parents=True,exist_ok=True)
        if not restore_artifact(path,PREFIX+'/'+name,hub):raise FileNotFoundError('published pack file missing: '+name)
        if path.stat().st_size!=meta['bytes'] or digest_file(path)!=meta['sha256']:raise ValueError('published pack SHA mismatch: '+name)
    if inventory(root)!=value:raise ValueError('restored corpus inventory/proof differs')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',required=True)
    a=p.parse_args(argv);root=Path(a.work_dir).resolve();value=inventory(root)
    from extent.hf_artifact_sync import artifact_config_from_env,upload_artifacts_together,restore_artifact
    from extent.hf_download_retry import DownloadPolicy
    os.environ.setdefault('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent')
    os.environ.setdefault('EXTENT_HF_REPO_TYPE','dataset')
    hub=artifact_config_from_env(include_kaggle_secrets=False)
    if hub is None:raise ValueError('HF_TOKEN required')
    existing=root/'tpu169-remote-inventory.json'
    if restore_artifact(existing,PREFIX+'/inventory.json',hub):
        if json.loads(existing.read_text())!=value:raise ValueError('published immutable corpus pack differs; refusing overwrite')
        print('EXP169 matching corpus inventory already published');return
    deadline=time.time()+3600;policy=DownloadPolicy(root/'tpu169-transport',deadline=deadline)
    for name in value['files']:
        policy.call(upload_artifacts_together,[(root/name,PREFIX+'/'+name)],hub,commit_message='EXP169 verified corpus pack')
    descriptor=root/'tpu169-data.json';write_json(descriptor,value)
    policy.call(upload_artifacts_together,[(descriptor,PREFIX+'/inventory.json')],hub,commit_message='EXP169 corpus inventory published last')
    print('EXP169 existing corpus packs published; no weights or extra corpus copies created')


if __name__=='__main__':main()
