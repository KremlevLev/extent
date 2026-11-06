"""EXP172: matched frozen teacher/gqa4/gqa7 quality on two independent T4s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from extent.broad_corpus import write_json
from extent.quality_172 import REPO, REVISION, SOURCE, SOURCE_REVISION, fetch_bank, sha

ROLES = ('teacher', 'gqa4', 'gqa7')


def notify(text):
    # Same Bot API credentials as existing runs; never poll getUpdates.
    if not all(os.environ.get(k) for k in ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID')): return
    import urllib.request, urllib.parse
    try:
        data = urllib.parse.urlencode(dict(chat_id=os.environ['TELEGRAM_CHAT_ID'], text=text)).encode()
        request = urllib.request.Request('https://api.telegram.org/bot'+os.environ['TELEGRAM_BOT_TOKEN']+'/sendMessage', data=data)
        with urllib.request.urlopen(request, timeout=15) as response: response.read()
    except Exception as exc: print('notification_failed='+type(exc).__name__, flush=True)


def prepare(root, cap):
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import numpy as np
    import pyarrow.parquet as pq
    from extent.quality_benchmark_156 import ROOT, TASKS, format_doc, stable_order, balanced_mmlu, interleave, GENERATION_PROMPTS
    meta = fetch_bank(root/'bank')
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, revision=SOURCE_REVISION, cache_dir=root/'hf', trust_remote_code=False)
    # Download once in parent, then workers open the local cache.
    for name in ('config.json', 'model.safetensors'):
        hf_hub_download(SOURCE, name, revision=SOURCE_REVISION, cache_dir=root/'hf')
    pins = json.loads((ROOT/'results/EXP-156-data-pin.json').read_text())
    tasks, provenance, panels = {}, {}, {}
    for task, pin in pins.items():
        print('EXP172 prepare dataset='+task, flush=True)
        file = Path(hf_hub_download(pin['repo'], pin['file'], revision=pin['revision'], repo_type='dataset', cache_dir=root/'hf'))
        if sha(file) != pin['sha256']: raise ValueError('dataset SHA mismatch: '+task)
        provenance[task] = pin
        if task in TASKS:
            rows = [format_doc(task, doc, i) for i, doc in enumerate(pq.read_table(file).to_pylist())]
            tasks[task] = (balanced_mmlu(rows) if task == 'mmlu' else stable_order(rows))[:cap]
        else:
            tokens, used = [], []
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128, columns=['text']):
                for doc in batch.to_pylist():
                    ids = tokenizer.encode(doc['text'], add_special_tokens=False)
                    if len(ids) < 64: continue
                    tokens.extend(ids+[tokenizer.eos_token_id]); used.append(hashlib.sha256(doc['text'].encode()).hexdigest())
                    if len(tokens) >= 131072: break
                if len(tokens) >= 131072: break
            if len(tokens) < 131072: raise ValueError('short independent NLL panel')
            panels[task] = np.asarray(tokens[:131072], np.int32)
            provenance[task] = dict(pin, token_sha256=hashlib.sha256(panels[task].tobytes()).hexdigest(), document_hashes=used)
    # Full Edu validation only (~0.5MB), no 2.4GB training corpus or old Adam.
    prefix = 'experiments/exp169-data-c4-edu600m-v1'
    inv = json.loads(Path(hf_hub_download(REPO, prefix+'/inventory.json', revision=REVISION, repo_type='dataset', cache_dir=root/'hf')).read_text())
    key = 'data-pack-edu-600m/validation.bin'
    f = Path(hf_hub_download(REPO, prefix+'/'+key, revision=REVISION, repo_type='dataset', cache_dir=root/'hf'))
    if sha(f) != inv['files'][key]['sha256']: raise ValueError('Edu validation SHA mismatch')
    panels['edu'] = np.fromfile(f, np.int32)
    np.savez(root/'panels.npz', **panels)
    jobs = []
    # Early independent text results and generation, followed by round-robin tasks.
    for panel in panels:
        for length in (256, 1024):
            for i in range(32):
                jobs.append(dict(id=f'nll:{panel}:{length}:{i}', kind='nll', panel=panel, length=length, window=i))
    generations = [dict(id=f'generate:{i}', kind='generate', language=language, prompt=text)
                   for i, (language, text) in enumerate(GENERATION_PROMPTS)]
    spaced = []
    while jobs or generations:
        spaced.extend(jobs[:6]); del jobs[:6]
        if generations: spaced.append(generations.pop(0))
    jobs = spaced
    choices = [dict(row, kind='choice') for row in interleave(tasks)]
    # Spread diagnostics through the beginning instead of leaving them behind MMLU.
    ordered = []
    while jobs or choices:
        ordered.extend(choices[:6]); del choices[:6]
        if jobs: ordered.append(jobs.pop(0))
    frozen = dict(experiment=172, source_step=meta['step'], hf_revision=REVISION,
        checkpoint_manifest_sha256=sha(root/'bank/checkpoint.json'), max_per_task=cap,
        questions_sha256=hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest(),
        data=provenance, precision='FP16 projection/MLP; FP32 mixer coordinates/state/log-softmax',
        context=1024, fewshot=0, templates='EXP156 separate evaluator; not official lm-eval CLI',
        selection='CURRENT5120 fixed before benchmarks; no tuning on task outcomes',
        limitations='not decontaminated from original Qwen pretraining; NLL panels not benchmark accuracy; partial matched subsets explicitly counted')
    write_json(root/'contract.json', frozen); write_json(root/'jobs.json', ordered)
    return frozen


def load_journal(path):
    items = {}
    if path.exists():
        # JSONL records use LF only. str.splitlines also splits U+0085/U+2028/
        # U+2029 INSIDE valid JSON strings (benchmark text), corrupting records.
        lines = path.read_text(encoding='utf-8').split('\n')
        for i, line in enumerate(lines):
            try: row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1: raise
                break  # interrupted final write; redo that one item
            items[row['id']] = row
    return items


def aggregate(root, workers=None, error=None):
    """CPU-only report rebuild from existing journals; no weights or GPU work."""
    from extent.quality_172 import summarize
    root = Path(root); items = {}
    if workers is None:
        workers = [json.loads(p.read_text(encoding='utf-8'))
                   for i in range(2) if (p := root/f'worker-{i}.json').exists()]
    for i in range(2):
        shard = load_journal(root/f'worker-{i}.jsonl')
        if items.keys() & shard.keys(): raise ValueError('duplicate item IDs across GPU shards')
        items.update(shard)
    complete = (error is None and len(workers)==2 and
                all(r.get('status')=='completed' and r.get('returncode',0)==0 for r in workers) and
                sum(r.get('completed_items',0) for r in workers)==len(items))
    result = dict(experiment=172,status='completed' if complete else 'partial_or_failed',
                  error=error,workers=workers,items=items,summary={})
    try: result['summary'] = summarize(items)
    except Exception as exc:
        result.update(status='aggregation_failed',error=dict(type=type(exc).__name__,message=str(exc)[:250]))
        raise
    finally: write_json(root/'extent-m3q-quality172.json',result)
    return result


def worker_diagnostic(root, shard, returncode):
    root = Path(root)
    path = root/f'worker-{shard}.json'
    state = json.loads(path.read_text()) if path.exists() else dict(status='missing_report')
    state.update(shard=shard, returncode=returncode)
    if returncode != 0 or state.get('status') in ('failed', 'missing_report'):
        log = root/f'worker-{shard}.log'
        if log.exists():
            # Bounded tail also covers failures before Python can write a report.
            with log.open('rb') as f:
                f.seek(0, 2); size = f.tell(); f.seek(max(0, size-12000))
                state['log_tail'] = f.read().decode('utf-8', errors='replace')
    return state


def gate(engine, panels):
    import torch
    import numpy as np
    from extent.torch_clean_170 import chunked_scan
    from torch.nn import functional as F
    rows = []
    # Actual learned weights, independent token recurrence, including chunk boundary.
    for index in (0, 13, 26):
        m = engine.bank[str(index)]
        generator = torch.Generator(device=engine.device).manual_seed(172+index)
        u = torch.randn(1, 65, m.h*m.n, device=engine.device, generator=generator)*.2
        expected = m(u, reference=True)
        parts = m.components(u)
        actual = F.linear(chunked_scan(*parts, m, checkpoint=False).flatten(-2), m.out_weight)
        error = float((actual-expected).norm()/expected.norm().clamp_min(1e-8))
        if not np.isfinite(error) or error > .003: raise ValueError('learned recurrent gate failed')
        rows.append(dict(layer=index, recurrent_relative_l2=error))
    # Full composed FP16 versus FP32 on identical real tokens, all masks.
    ids = panels['edu'][:128].tolist()
    engine.set_precision('float32')
    token_tensor = torch.tensor([ids], device=engine.device)
    expected_hidden = engine.model(token_tensor, use_cache=False).last_hidden_state
    actual_hidden = engine.hidden(token_tensor, 'teacher')
    teacher_error = float((actual_hidden-expected_hidden).norm()/expected_hidden.norm().clamp_min(1e-8))
    if not np.isfinite(teacher_error) or teacher_error > .001:
        raise ValueError('manual teacher path differs from original HF model')
    full = {r: np.asarray(engine.nll(ids, r)) for r in ROLES}
    engine.set_precision('float16')
    differences = {}
    for r in ROLES:
        half = np.asarray(engine.nll(ids, r)); delta = np.abs(full[r]-half)
        differences[r] = dict(mean_abs_token_nll=float(delta.mean()), max_abs_token_nll=float(delta.max()))
        if not np.isfinite(delta).all() or delta.mean() > .03 or delta.max() > .25:
            raise ValueError('FP16 gate failed for '+r+'; refusing silent precision change')
    # Reproduce the same four TPU validation windows, bounded BF16/FP16 difference.
    reproduced = {}
    for r, expected in zip(ROLES, (2.4763966744, 2.5981243316, 2.5832235722)):
        values = [v for i in range(4) for v in engine.nll(panels['edu'][i*1024:(i+1)*1024].tolist(), r)]
        nll = float(np.mean(values)); reproduced[r] = dict(nll=nll, reference=expected)
        if abs(nll-expected) > .08: raise ValueError('TPU checkpoint replay differs for '+r)
    return dict(recurrence=rows, teacher_hf_relative_l2=teacher_error,
                fp16_vs_fp32=differences, tpu_replay=reproduced)


def worker(a, root):
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from extent.quality_172 import Engine
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.9)
    engine = Engine(root)
    panels = np.load(root/'panels.npz')
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, revision=SOURCE_REVISION, cache_dir=root/'hf', local_files_only=True)
    report = root/f'worker-{a.shard}.json'; journal = root/f'worker-{a.shard}.jsonl'
    contract = json.loads((root/'contract.json').read_text())
    prior = json.loads(report.read_text()) if report.exists() else {}
    if prior and prior.get('contract') != contract: raise ValueError('resume contract changed')
    state = dict(status='preflight', contract=contract, device=torch.cuda.get_device_name(), shard=a.shard)
    write_json(report, state)
    with torch.inference_mode():
        state['gates'] = gate(engine, panels); state['status'] = 'running'; write_json(report, state)
        jobs = json.loads((root/'jobs.json').read_text()); items = load_journal(journal)
        # Remove interrupted tail before appending, preserving all complete rows.
        journal.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in items.values()))
        def check():
            if time.time() >= a.deadline-90 or (root/'STOP').exists(): raise InterruptedError('deadline/STOP')
        try:
            for index, job in enumerate(jobs):
                if index%2 != a.shard or job['id'] in items: continue
                check(); begin = time.monotonic(); row = dict(job)
                if job['kind'] == 'choice':
                    for role in ROLES:
                        check(); scores, truncations = zip(*(engine.pair(tokenizer, pair, role) for pair in job['pairs']))
                        prediction = int(np.argmax(scores)); normalized = int(np.argmax(np.asarray(scores)/job['norms']))
                        row[role] = dict(status='scored', task=job['task'], subject=job['subject'], gold=job['gold'],
                            scores=list(scores), acc=prediction==job['gold'], acc_norm=normalized==job['gold'],
                            prompt_sha256=hashlib.sha256(json.dumps(job['pairs'], ensure_ascii=False).encode()).hexdigest(),
                            truncated_context_tokens=list(truncations))
                elif job['kind'] == 'nll':
                    start = job['window']*job['length']; ids = panels[job['panel']][start:start+job['length']].tolist()
                    row['panel'] = job['panel']+'_'+str(job['length']); row['targets'] = len(ids)-1
                    for role in ROLES:
                        check(); row[role] = dict(sum_nll=sum(engine.nll(ids, role)))
                else:
                    for role in ROLES: row[role] = engine.generate(tokenizer, job['prompt'], role, check=check)
                row['seconds'] = time.monotonic()-begin
                with journal.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
                items[row['id']] = row
                state.update(completed_items=len(items), last_id=row['id'], last_item_seconds=row['seconds'], updated_unix=time.time())
                write_json(report, state)
                if len(items)%10 == 0: print('EXP172 shard', a.shard, 'matched_items', len(items), 'last', row['id'], flush=True)
            state['status'] = 'completed'
        except InterruptedError: state['status'] = 'deadline_partial'
        finally:
            state['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(); write_json(report, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', default='/kaggle/working/extent-quality172')
    p.add_argument('--max-wall-hours', type=float, default=11.5)
    p.add_argument('--max-per-task', type=int, default=1000)
    p.add_argument('--shard', type=int, choices=[0, 1])
    p.add_argument('--deadline', type=float)
    p.add_argument('--aggregate-only', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv); root = Path(a.work_dir); root.mkdir(parents=True, exist_ok=True)
    if a.aggregate_only:
        result=aggregate(root)
        print(json.dumps(dict(status=result['status'],summary=result['summary']),ensure_ascii=False),flush=True)
        return result
    if a.shard is not None:
        try: worker(a, root)
        except BaseException as exc:
            path = root/f'worker-{a.shard}.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:250]); write_json(path, state)
            raise
        return
    if not 0 < a.max_wall_hours <= 12 or a.max_per_task < 1: raise ValueError('invalid budget/sample cap')
    deadline = time.time()+a.max_wall_hours*3600
    os.environ.update(USE_TORCH='1', USE_TF='0', USE_FLAX='0', HF_HUB_DISABLE_PROGRESS_BARS='1',
        EXTENT_HF_DOWNLOAD_DEADLINE_UNIX=str(deadline-180), EXTENT_HF_RETRY_STATE_DIR=str(root/'transport'))
    # Parent never initializes CUDA; children each see one GPU.
    from filelock import FileLock
    with FileLock(str(root/'run.lock'), timeout=0):
        notify('Extent EXP172: preparing frozen171 benchmark; 2 GPU workers; matched teacher/gqa4/gqa7')
        children = []; handles = []; parent_error = None
        try:
            previous = json.loads((root/'contract.json').read_text()) if (root/'contract.json').exists() else None
            if previous and previous['max_per_task'] != a.max_per_task: raise ValueError('use new work-dir for changed sample cap')
            prepare(root, a.max_per_task)
            for i in range(2):
                log = (root/f'worker-{i}.log').open('w'); handles.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
                command = [sys.executable, '-u', '-m', 'engine_quality172', '--work-dir', str(root),
                    '--shard', str(i), '--deadline', str(deadline)]
                children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            last = 0
            while any(p.poll() is None for p in children):
                if time.time() > deadline:
                    for p in children:
                        if p.poll() is None: p.terminate()
                    break
                if time.time()-last > 60:
                    last = time.time()
                    for i in range(2):
                        f = root/f'worker-{i}.json'
                        if f.exists():
                            r = json.loads(f.read_text()); print('EXP172 GPU', i, r.get('status'), 'matched', r.get('completed_items', 0), flush=True)
                time.sleep(2)
            for p in children:
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
        except BaseException as exc:
            parent_error = dict(type=type(exc).__name__, message=str(exc)[:250])
            raise
        finally:
            for p in children:
                if p.poll() is None: p.kill(); p.wait()
            for log in handles: log.close()
            workers = []
            for i in range(2):
                if i < len(children):
                    diagnostic = worker_diagnostic(root, i, children[i].returncode)
                    workers.append(diagnostic)
                    if diagnostic.get('status') in ('failed', 'missing_report') or diagnostic['returncode'] != 0:
                        print('EXP172 WORKER FAILURE '+json.dumps(diagnostic, ensure_ascii=False), flush=True)
            result=aggregate(root,workers,parent_error)
            notify('Extent EXP172: '+result['status']+'; matched items='+str(len(result['items']))+'; download extent-m3q-quality172.json')
            print(json.dumps(dict(status=result['status'], summary=result['summary']), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__': main()
