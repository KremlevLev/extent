"""Pinned, bounded, restartable document packing; no implicit training epochs.

Exact normalized-document deduplication is not semantic/benchmark decontamination.
The SQLite journal and two token files are one recovery unit. Only manifest.json
marks a completed corpus. No model weights or teacher targets are cached here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import unicodedata
import numpy as np

C4_REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def document_digest(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).split())
    return hashlib.sha256(normalized.encode('utf-8')).digest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


class CorpusExhausted(Exception):
    """The registered stream is finished; do not cycle or silently fetch new data."""


class PackedCorpus:
    def __init__(self, directory, *, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['format'] != 'extent-c4-packed-v1':
            raise ValueError('unsupported corpus format')
        self.identity = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()
        self.arrays = {}
        for split, item in self.manifest['files'].items():
            path = self.directory / (split + '.bin')
            if path.stat().st_size != item['tokens'] * 4:
                raise ValueError('corpus token file length differs')
            if verify and digest_file(path) != item['sha256']:
                raise ValueError('corpus token file SHA differs')
            self.arrays[split] = np.memmap(path, dtype='<i4', mode='r')
        self.length = self.manifest['contract']['sequence_length']

    def batch(self, index):
        if index < 0 or index >= len(self.arrays['train']) // self.length:
            raise CorpusExhausted('C4 stream exhausted; explicit new corpus required')
        start = int(index) * self.length
        return np.array(self.arrays['train'][start:start+self.length])[None]

    def validation(self):
        return self.arrays['validation'].reshape(-1, self.length)


def build_corpus(directory, rows, tokenize, contract, *, check=lambda: None,
                 progress=lambda state: None, commit_every=256,
                 min_free_bytes=2 * 1024**3, max_local_bytes=8 * 1024**3):
    """rows yield split/index/text/url; stable indices belong to pinned sources.

    On retry rows may restart at zero. A committed per-split cursor skips them.
    Append files are fsynced BEFORE the SQLite commit; uncommitted tails are
    truncated on restart. Caller must hold an exclusive builder lock.
    """
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        corpus = PackedCorpus(root)
        if corpus.manifest['contract'] != contract:
            raise ValueError('completed corpus contract differs; use a new directory')
        return corpus.manifest
    if contract['sequence_length'] < 2 or commit_every < 1:
        raise ValueError('invalid sequence/commit length')
    goals = {k: int(contract[k+'_tokens']) for k in ('train', 'validation')}
    if any(n <= 0 or n % contract['sequence_length'] for n in goals.values()):
        raise ValueError('positive whole-window token budgets required')
    if 4 * sum(goals.values()) > max_local_bytes:
        raise ValueError('packed tokens exceed local storage budget')
    db = sqlite3.connect(root / 'documents.sqlite')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS documents (digest BLOB PRIMARY KEY, split TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT)')
    record = db.execute('SELECT value FROM state WHERE id=1').fetchone()
    state = json.loads(record[0]) if record else dict(contract=contract, counts={k:0 for k in goals},
        cursor={k:-1 for k in goals}, documents={k:0 for k in goals}, duplicates=0, rejected=0)
    if state['contract'] != contract:
        db.close(); raise ValueError('unfinished corpus contract differs')
    files = {}
    try:
        for split in goals:
            path = root / (split+'.bin')
            if path.exists() and path.stat().st_size < state['counts'][split]*4:
                raise ValueError('committed corpus bytes missing')
            f = open(path, 'r+b' if path.exists() else 'w+b')
            f.truncate(state['counts'][split]*4); f.seek(0, 2); files[split] = f

        def commit():
            for f in files.values():
                f.flush(); os.fsync(f.fileno())
            db.execute('INSERT OR REPLACE INTO state VALUES (1,?)', (json.dumps(state),))
            db.commit()

        commit(); progress(state); processed = 0
        for row in rows:
            check()
            if all(state['counts'][k] == goals[k] for k in goals): break
            split = row['split']; index = int(row['index'])
            if split not in goals: raise ValueError('unknown source split')
            if index <= state['cursor'][split]: continue
            if index != state['cursor'][split]+1: raise ValueError('source cursor gap')
            state['cursor'][split] = index; processed += 1
            if state['counts'][split] < goals[split]:
                text = row['text']
                # Avoid importing old Wiki diagnostic articles through C4 URLs.
                if len(text.strip()) < 200 or len(text) > 1_000_000 or 'wikipedia.org' in row.get('url','').lower():
                    state['rejected'] += 1
                else:
                    key = document_digest(text)
                    added = db.execute('INSERT OR IGNORE INTO documents VALUES (?,?)', (key,split)).rowcount
                    if not added: state['duplicates'] += 1
                    else:
                        ids = np.asarray(tokenize(text), dtype='<i4')
                        if ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0) or np.any(ids >= contract['vocab_size']):
                            raise ValueError('invalid tokenizer output')
                        ids = np.concatenate((ids, np.asarray([contract['eos_token_id']], dtype='<i4')))
                        remaining = goals[split]-state['counts'][split]
                        ids = ids[:remaining]; files[split].write(ids.tobytes())
                        state['counts'][split] += len(ids); state['documents'][split] += 1
            progress(state)
            if processed % commit_every == 0:
                used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if used > max_local_bytes or shutil.disk_usage(root).free < min_free_bytes:
                    raise OSError('corpus storage budget/reserve exhausted')
                commit()
        commit()
        if state['counts'] != goals:
            raise CorpusExhausted('source stopped before corpus budget; partial build is resumable')
        for f in files.values(): f.close()
        manifest = dict(format='extent-c4-packed-v1', contract=contract, statistics=state,
            limitations='normalized exact-document dedup only; no semantic or benchmark decontamination; sequential packed windows, EOS between documents',
            files={k:dict(tokens=goals[k],sha256=digest_file(root/(k+'.bin'))) for k in goals})
        write_json(root/'manifest.json', manifest)
        return manifest
    finally:
        for f in files.values():
            if not f.closed: f.close()
        db.close()
