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
    for relative in ('paper/draft.ru.md','results/EXP-111-113-recovery-efficiency-protocol.md',
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
    r155 = read('results/EXP-155-session-1.json')
    reports = [('136', r136), ('145', r145), ('147', r147), ('153 v2', r153), ('155', r155)]
    assert all(r['data']['panels'] == r136['data']['panels'] for _, r in reports)
    assert [branch(r)['source_step'] for _,r in reports] == [0,12586,24965,31126,37910]
    assert branch(r155)['final']['last']['step'] == 80260
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
    token_end = OFFSET + 80260*256
    assert token_end == 28935168
    prep = {'069_exact_dual_local':24*1024*64,
            '069_recurrent_local':24*2048*64,
            '072_onpolicy_local':24*4096*128}
    assert sum(prep.values()) == 17301504

    fig = plt.figure(figsize=(18,10), facecolor='#fafbf9')
    ax = fig.add_axes([.065,.285,.865,.53], facecolor='#fafbf9')
    ax.set_yscale('log')
    ax.set_xlim(-7.7,32)
    ax.set_ylim(4.25,27)
    ax.yaxis.set_major_locator(FixedLocator([4.5,5,6,7,10,15,20,25]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28])
    ax.tick_params(axis='both',labelsize=10)
    ax.grid(axis='y',alpha=.16)
    ax.set_ylabel('NLL ↓  ·  логарифмическая шкала', labelpad=12)
    ax.set_xlabel('Млн входных позиций при сквозном обучении с начала EXP-111', labelpad=13)
    ax.axvspan(-7.7,-1.05,color='#e8ecea',alpha=.8)
    ax.axvline(-1.05,color='#9ba7a4',ls=(0,(3,4)),lw=1)
    ax.axvline(OFFSET/1e6,color='#9ba7a4',ls=(0,(3,4)),lw=1)

    # A comparison of two preparations, not a timed descent of one checkpoint.
    xs=[-6.7,-3.1];ys=[early['TEACHER']['nll'],early['ONPOLICY']['nll']]
    ax.scatter(xs,ys,s=85,color=['#9a5e56','#557768'],zorder=6)
    ax.plot(xs,ys,color='#8b9993',lw=1.5,ls=(0,(3,4)))
    ax.annotate('22,35', (xs[0],ys[0]),xytext=(0,14),textcoords='offset points',ha='center',fontsize=15,fontweight='bold',color='#7b4a44')
    ax.annotate('10,55', (xs[1],ys[1]),xytext=(0,-25),textcoords='offset points',ha='center',fontsize=15,fontweight='bold',color='#43685b')
    ax.text(-5.9,17.5,'Чистые входы\nQwen при подготовке',ha='center',fontsize=9.5,color='#6d514c')
    ax.text(-3.7,7.8,'ONPOLICY:\nвходы гибридного\nпрефикса',ha='center',fontsize=10,color='#43685b')
    ax.text(-4.4,5.0,'EXP-072 · 24 замены\nДругая оценка;\nздесь X не считает токены',ha='center',fontsize=10,color='#51615b',linespacing=1.4)

    ax.plot([v['full_model_input_positions']/1e6 for v in old],[v['nll'] for v in old],
            color='#337caa',lw=2.5,marker='o',ms=3,zorder=4)
    ax.annotate('12,97',(0,old[0]['nll']),xytext=(10,10),textcoords='offset points',fontsize=14,fontweight='bold',color='#337caa')
    ax.annotate('7,29',(OFFSET/1e6,old[-1]['nll']),xytext=(-10,23),textcoords='offset points',ha='right',fontsize=13,fontweight='bold',color='#337caa')
    ax.text(4.15,16.8,'Защищённый rank 32\n+ устойчивый клиппинг\nEXP-111',ha='center',fontsize=12,color='#255e82',linespacing=1.5)
    ax.text(4.1,5.1,'Быстрый спад, затем плато\n8,39 млн позиций',ha='center',fontsize=11,color='#255e82')
    ax.text(4.35,9.1,'EXP-097: 12,97 → 7,267 за 2,10 млн\nТот же рецепт; отдельный прогон, не предок EXP-111',
            ha='center',fontsize=9,color='#337caa',linespacing=1.5)
    ax.text(OFFSET/1e6+.22,22,'Смена оценочного набора → A / B / C',fontsize=10,color='#67736d',rotation=0)

    stage_bounds=[OFFSET/1e6]+[(OFFSET+s*256)/1e6 for s in (12586,24965,31126,37910,80260)]
    for i,(lo,hi) in enumerate(zip(stage_bounds,stage_bounds[1:])):
        ax.axvspan(lo,hi,color=['#d1e9df','#e8dfcf','#e7e2ef','#cee7e3','#c8e4df'][i],alpha=.24,zorder=0)
        if i:ax.axvline(lo,color='#a2b4ad',lw=.8,ls=(0,(2,5)),zorder=0)
        ax.add_patch(Rectangle((lo,1.018),hi-lo,.035,transform=ax.get_xaxis_transform(),
                               color=['#789e8b','#b39665','#9383aa','#4e9b90','#197e74'][i],clip_on=False,lw=0))
        ax.text((lo+hi)/2,1.067,['136','145','147','153 v2','155'][i],
                transform=ax.get_xaxis_transform(),ha='center',fontsize=10,fontweight='bold',color='#39564d')
    for p in 'ABC':
        rows=same[p]
        ax.plot([v['full_model_input_positions']/1e6 for v in rows],[v['nll'] for v in rows],
                lw=2.7 if p=='C' else 1.4,color=COLORS[p],alpha=1 if p=='C' else .75,
                marker='o',ms=3.1 if p=='C' else 2,zorder=5 if p=='C' else 3)
        end=branch(r155)['final']['last']['panels'][p]['nll']
        assert abs(rows[-1]['nll']-end)<1e-9
        ax.scatter(token_end/1e6,end,s=48 if p=='C' else 24,color=COLORS[p],zorder=8)
        label_y={'A':5.51,'B':5.15,'C':4.54}[p]
        ax.annotate(f'{p}: {end:.3f}'.replace('.',','),(token_end/1e6,end),xytext=(30.0,label_y),
                    textcoords='data',ha='left',va='center',fontsize=12,fontweight='bold',
                    color=COLORS[p],arrowprops=dict(arrowstyle='-',lw=.9,color=COLORS[p]))
    ax.text(19,13.1,'Нормализация + tanh + сброс динамики\n→ сворачивание поправки и новая R32\n→ продолжение → рост до R256',
            ha='center',fontsize=13,color='#245e53',linespacing=1.55)
    ax.annotate('4,646 — checkpoint 78 870, выбранный по A\n4,666 — последний checkpoint 80 260',
                (token_end/1e6,4.6655588076),xytext=(20.8,7.6),fontsize=10.5,color='#087f78',
                arrowprops=dict(arrowstyle='->',color='#087f78',connectionstyle='arc3,rad=-.1'))
    fig.legend(handles=[Line2D([0],[0],color='#337caa',lw=2,label='Историческая оценка EXP-111')]+
              [Line2D([0],[0],color=COLORS[p],lw=2,label='Панель '+p) for p in 'ABC'],
              loc='upper right',bbox_to_anchor=(.932,.887),ncol=4,frameon=False,fontsize=10)

    fig.text(.065,.952,'Extent: от неудачной композиции к NLL 4,67',fontsize=24,fontweight='bold',color='#173f34')
    fig.text(.065,.912,'Одна история исследования · реальные измерения без сглаживания · исходный Qwen3-1.7B → 24 Mamba-3 MIMO + 4 GQA',
             fontsize=12,color='#5e7069')
    stages=[('111','Защищённая R32','8,389 млн'),('136','Норма + tanh\n+ совместный сброс','3,222 млн'),
            ('145','Сворачивание\n+ новая R32 / Adam','3,169 млн'),('147','Сохранение R32\nи Adam','1,577 млн'),
            ('153 v2','R32 → R256\n+ свежий Adam','1,737 млн'),('155','Та же R256 / Adam\nпродолжение на TPU','10,842 млн')]
    for x,(exp,title,tokens) in zip([.07,.218,.366,.514,.662,.81],stages):
        fig.text(x,.181,'EXP-'+exp,fontsize=11,fontweight='bold',color='#355e50')
        fig.text(x,.139,title,fontsize=10,color='#50665c',linespacing=1.35)
        fig.text(x,.111,'+ '+tokens,fontsize=12,fontweight='bold',color='#173f34')
    fig.text(.065,.070,'Всего 46,237 млн предъявлений: 28,935 млн при сквозном обучении + 17,302 млн при послойной подготовке EXP-069/072.',
             fontsize=11,fontweight='bold',color='#254b3d')
    fig.text(.065,.044,'Блок и вся сеть требуют разных вычислений. Это не уникальные токены; поиск гипотез, кэши, оценки и ridge-калибровка (12 288 позиций) не включены.',
             fontsize=9.5,color='#617269')
    fig.text(.065,.021,'EXP-130 предложил рецепт сброса; EXP-136 повторил его из EXP-111 (затраты EXP-130 не прибавлены). Один сид; повторно используемая валидация. Разрыв оценки отмечен.',
             fontsize=9.5,color='#617269')
    for ext in ('png','pdf','svg'):
        path=OUT/f'progress_overview.ru.{ext}'
        fig.savefig(path,dpi=180,facecolor=fig.get_facecolor())
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    with (OUT/'progress_overview_points.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(POINTS[0]));writer.writeheader();writer.writerows(POINTS)
    audit=dict(date='2026-10-24',sources=SOURCES,selected_seed=123,points=len(POINTS),
               full_model_training_positions=token_end,local_preparation_positions=prep,
               local_preparation_total=sum(prep.values()),ridge_calibration_positions_separate=12288,
               stage_updates=dict(EXP111=32768,EXP136=12586,EXP145=12379,EXP147=6161,EXP153v2=6784,EXP155=42350),
               scope='Selected lineage only. Not total research compute. No smooth curve across changed evaluation panels. Not unique tokens or FLOP equivalent.',
               preparation_basis='paper/draft.ru.md section3.4 and EXP111 protocol: reuse EXP069 local checkpoints -> EXP072 ONPOLICY -> EXP111; no EXP097 state reuse.',
               historical_raw_sha256=qprov['sources']['111']['sha256'])
    (OUT/'progress_overview_provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(POINTS)} source-backed points; {token_end:,} full-model positions; {sum(prep.values()):,} local positions separately.')


if __name__=='__main__':
    main()
