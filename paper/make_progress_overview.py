"""One Russian overview, with source-backed points and separate token accounting.

Offline only. Historical controls and changed evaluation panels are NOT stitched
into an uninterrupted learning curve. Token axis starts at EXP111, not transfer.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'paper/figures'
SOURCES = []
POINTS = []
OFFSET = 32768 * 256
COLORS = {'A': '#c2762b', 'B': '#8172ac', 'C': '#087f78'}


def read(relative):
    path = ROOT / relative
    SOURCES.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return json.loads(path.read_text(encoding='utf-8'))


def csv_read(relative):
    path = ROOT / relative
    SOURCES.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def branch(report):
    assert len(report['branches']) == 1
    return next(iter(report['branches'].values()))


def worker(relative, index):
    return next(w['report'] for w in read(relative)['workers'] if w['worker'] == index)


def point(exp, panel, nll, step, source, kind='registered'):
    value = dict(experiment=exp, panel=panel, nll=float(nll), global_step=step,
                 full_model_input_positions=OFFSET+step*256 if step is not None else None,
                 kind=kind, source=source)
    POINTS.append(value)
    return value


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.hashsalt': 'extent-progress-20261024',
                         'axes.spines.top': False, 'axes.spines.right': False})
    historical = csv_read('paper/figures/points.csv')
    early = {}
    for arm in ('TEACHER', 'ONPOLICY'):
        row = next(r for r in historical if r['experiment'] == 'EXP-072-v2'
                   and r['seed'] == '123' and r['arm'] == arm and r['x'] == '24')
        early[arm] = point('072', 'historical072', row['nll'], None,
                           'paper/figures/points.csv: seed123, depth24, '+arm, 'historical_comparison')
    quality = csv_read('paper/figures/quality_points.csv')
    qprov = read('paper/figures/quality_provenance.json')
    diagnostic = read('results/EXP-097-numerical-stability.json')['branches']['123']['SAFE-PROTECTED-R32']
    assert diagnostic['step']==8192 and diagnostic['complete']
    assert abs(diagnostic['locked_test_nll']-7.266784)<1e-6
    for relative in ('paper/draft.ru.md','paper/make_progress_overview.py','paper/progress_layout.py','results/EXP-111-113-recovery-efficiency-protocol.md',
                     'results/EXP-072-completed-summary.md'):
        path=ROOT/relative
        SOURCES.append(dict(path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    q111 = sorted([r for r in quality if r['experiment']=='111' and r['seed']=='123'
                   and r['panel']=='reproduction'], key=lambda r: int(r['step']))
    assert int(q111[-1]['step']) == 32768
    old = [dict(point('111', 'historical111', r['nll'], None,
                      'quality_points.csv: reproduction seed123 step'+r['step']),
                full_model_input_positions=int(r['input_tokens'])) for r in q111]
    # Replace the stored records with the same explicit EXP111 token coordinates.
    POINTS[-len(old):] = old
    r136 = worker('results/EXP-136-streamed-v2-session-1.json', 0)
    r145 = worker('results/EXP-145-long-v1-session-1.json', 1)
    r147 = worker('results/EXP-147-fold-continue-session-1.json', 0)
    r153 = read('results/kaggle-recovery-2026-10-20/exp153-rank-growth-v2-worker-1.json')
    r155 = read('results/EXP-155-session-2.json')
    reports = [('136', r136), ('145', r145), ('147', r147), ('153 v2', r153), ('155', r155)]
    assert all(r['data']['panels'] == r136['data']['panels'] for _, r in reports)
    assert [branch(r)['source_step'] for _,r in reports] == [0,12586,24965,31126,37910]
    assert branch(r155)['final']['last']['step'] == 118304
    same = {p: [] for p in 'ABC'}
    for p in 'ABC':
        panel = branch(r136)['source_controls'][p]
        assert len(panel['windows']) == (32 if p=='B' else 64)
        same[p].append(point('136', p, panel['nll'], 0, '136/source_controls/'+p, 'source_no_intervention'))
    for exp,r in reports:
        b = branch(r)
        es = list(b['evaluations'])
        if 'final' in b:
            last = b['final']['last']
            if not any(e['step']==last['step'] and e.get('panels')==last['panels'] for e in es):
                es.append(dict(last, endpoint=True))
        for e in es:
            for p in 'ABC':
                panel = e.get('panels',{}).get(p)
                if panel is None or len(panel['windows']) != (32 if p=='B' else 64):
                    continue  # Never mix A16 ramp monitors with A64.
                same[p].append(point(exp,p,panel['nll'],e['step'],
                                      f'{exp}/evaluations-or-final/step{e["step"]}/{p}',
                                      'deadline' if e.get('endpoint') else 'registered'))
    token_end = OFFSET + 118304*256
    assert token_end == 38674432
    prep = {'069_exact_dual_local':24*1024*64,
            '069_recurrent_local':24*2048*64,
            '072_onpolicy_local':24*4096*128}
    assert sum(prep.values()) == 17301504

    late=read('paper/latest_recovery_data.json')
    for row in late['points']:
        for panel,v in row['panels'].items():
            value=point(str(row['experiment']),panel,v['nll'],None,
                'paper/latest_recovery_data.json:'+row['arm']+'/step'+str(row['step']),row['kind'])
            value.update(global_step=row['step'],full_model_input_positions=row['full_model_input_positions'],arm=row['arm'])
    from paper.progress_layout import draw
    fig=draw(early,old,same,late)
    token_end=late['token_ledger']['full_model_selected_path']
    for ext in ('png','pdf','svg'):
        path=OUT/f'progress_overview.ru.{ext}'
        fig.savefig(path,dpi=180,facecolor=fig.get_facecolor())
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    with (OUT/'progress_overview_points.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(POINTS[0])+['arm']);writer.writeheader();writer.writerows(POINTS)
    audit=dict(date='2026-10-29',sources=SOURCES,selected_seed=123,points=len(POINTS),
               full_model_training_positions=token_end,local_preparation_positions=prep,
               local_preparation_total=sum(prep.values()),ridge_calibration_positions_separate=12288,
               stage_updates=dict(EXP111=32768,EXP136=12586,EXP145=12379,EXP147=6161,EXP153v2=6784,EXP155=80394),
               scope='Selected lineage only. Not total research compute. No smooth curve across changed evaluation panels. Not unique tokens or FLOP equivalent.',
               preparation_basis='paper/draft.ru.md section3.4 and EXP111 protocol: reuse EXP069 local checkpoints -> EXP072 ONPOLICY -> EXP111; no EXP097 state reuse.',
               historical_raw_sha256=qprov['sources']['111']['sha256'],late_ledger=late['token_ledger'],
               late_offsets=late['offsets'],late_extracted_sources=late['sources'])
    (OUT/'progress_overview_provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(POINTS)} source-backed points; {token_end:,} full-model positions; {sum(prep.values()):,} local positions separately.')


if __name__=='__main__':
    main()
