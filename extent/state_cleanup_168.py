"""Remove only known local state namespaces whose remote commit manifests match."""
import json,shutil,tempfile,time
from pathlib import Path

PREFIXES={
    'runtime-long':'experiments/exp168-full-long-v1/checkpoints',
    **{f'runtime-lr-x{x}':f'experiments/exp167-lr-sweep-v1/x{x}/checkpoints' for x in (1,3,10)},
    'runtime-full-stream':'experiments/exp166-full-stream-v1/checkpoints',
    'runtime-full-next':'experiments/exp165-full-steady-v1/checkpoints',
    'runtime-full':'experiments/exp164-full-adaptive-v1/checkpoints',
    'runtime-methods':'experiments/exp163-method-sweep-v1/checkpoints',
    'runtime-b6':'experiments/exp161-batch6-v1/checkpoints',
}


def cleanup(root,hub,deadline=None,restore=None):
    if restore is None:
        from extent.hf_artifact_sync import restore_artifact as restore
    states=(Path(root)/'states').resolve();rows=[]
    for namespace,prefix in PREFIXES.items():
        directory=states/namespace
        if not directory.exists():continue
        if directory.is_symlink() or directory.resolve().parent!=states:raise ValueError('unsafe state namespace')
        for slot in directory.iterdir():
            if deadline and time.monotonic()>=deadline:return rows
            pointer=slot/'checkpoint.json'
            if not slot.is_dir() or not pointer.exists():continue
            row=dict(namespace=namespace,slot=slot.name,deleted=False);rows.append(row)
            if slot.is_symlink() or any(p.is_symlink() for p in slot.rglob('*')):
                row['reason']='symlink refused';continue
            try:
                local=json.loads(pointer.read_text())
                # Publication of this manifest is the commit marker: checkpoint
                # upload protocol publishes it only after all immutable chunks.
                with tempfile.TemporaryDirectory(prefix='state-verify-') as tmp:
                    remote=Path(tmp)/'checkpoint.json'
                    if not restore(remote,prefix+'/'+slot.name+'/checkpoint.json',hub):
                        row['reason']='remote manifest unavailable';continue
                    if json.loads(remote.read_text())!=local:
                        row['reason']='remote CURRENT differs; retain local';continue
                target=slot.resolve()
                if target.parent!=directory.resolve():raise ValueError('state slot escaped namespace')
                row['bytes']=sum(p.stat().st_size for p in slot.rglob('*') if p.is_file())
                shutil.rmtree(target);row.update(deleted=True,step=local['step'])
            except Exception as exc:row.update(reason=str(exc)[:500],error_type=type(exc).__name__)
    return rows
