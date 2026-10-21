"""Offline, auditable figures from archived continuation reports; no model execution."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
SOURCES = []
POINTS = []
COLORS = ['#247b9c', '#b76d29', '#7654a3', '#16856c']

def read(path):
    p = ROOT / path
    SOURCES.append({'path': path, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return json.loads(p.read_text(encoding='utf-8'))

def worker(path, index):
    d = read(path)
    return next(w['report'] for w in d['workers'] if w['worker'] == index)

def branch(r):
    assert len(r['branches']) == 1
    return next(iter(r['branches'].values()))

def nll(e, p):
    return e['panels'][p]['nll']

def series(r, p, final=False):
    b = branch(r)
    # Early reset monitors use A16; never splice them onto the full A64 curve.
    count = 32 if p == 'B' else 64
    es = [e for e in b['evaluations'] if p in e.get('panels', {})
          and len(e['panels'][p]['windows']) == count]
    values = [(e['step'], nll(e, p), 'registered') for e in es]
    if final and 'final' in b:
        e = b['final']['last']
        values.append((e['step'], nll(e, p), 'deadline'))
    return values

def save(fig, name, footer):
    fig.text(.5, .018, footer, ha='center', va='bottom', fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(0, .105, 1, .93))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=190, bbox_inches='tight')
        if ext == 'svg':
            path = OUT / f'{name}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'extent-continuation-20261020'})
    r136 = worker('results/EXP-136-streamed-v2-session-1.json', 0)
    r145 = worker('results/EXP-145-long-v1-session-1.json', 1)
    r147 = worker('results/EXP-147-fold-continue-session-1.json', 0)
    r154 = worker('results/EXP-154-continuation-matrix-session-1.json', 0)
    growth = [read(f'results/kaggle-recovery-2026-10-20/exp153-rank-growth-v2-worker-{w}.json') for w in (0, 1)]
    reports = [r136, r145, r147, r154, *growth]
    assert all(r['data']['panels'] == r136['data']['panels'] for r in reports)
    for r in [r147, r154, *growth]:
        assert branch(r)['source_replay']['passed']
    for r, expected in zip(growth, [38294, 37910]):
        b = branch(r)
        assert r['status'] == 'running' and b['step'] == b['durable_step'] == expected
        assert b['source_step'] == 31126
        assert b['best_step'] == 37270
        assert b['source_adam_step'] == 0
    # Same panel/data throughout the selected lineage; do not splice historical test panels.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    stages = [(r136, '136: combined reset', COLORS[0]),
              (r145, '145: fold + fresh R32/Adam', COLORS[1]),
              (r147, '147: keep R32/Adam', COLORS[2]),
              (growth[1], '153 v2: grow to R256 + fresh Adam', COLORS[3])]
    for ax, p in zip(axes, 'ABC'):
        for r, label, color in stages:
            values = series(r, p, final=True)
            ax.plot([x*256/1e6 for x,y,k in values], [y for x,y,k in values],
                    color=color, lw=2, marker='o', ms=3, label=label)
            x,y,k = values[-1]
            ax.annotate(f'{y:.3f}', (x*256/1e6,y), xytext=(4,6), textcoords='offset points', color=color, fontsize=9)
            for x,y,k in values:
                POINTS.append(dict(figure='continuation_lineage', arm=label, panel=p,
                                   step=x, input_tokens=x*256, nll=y, kind=k))
        for boundary in (12586,24965,31126):
            ax.axvline(boundary*256/1e6, color='#9ca3af', lw=.8, ls='--', alpha=.6)
        ax.set_title(f'Validation {p} ({64 if p != "B" else 32} windows)', fontweight='bold')
        ax.set_xlabel('Input-token presentations since EXP-136 (millions)')
        ax.set_ylabel('NLL (lower is better)'); ax.grid(alpha=.18); ax.margins(x=.12, y=.12)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.935), ncol=2, fontsize=9)
    fig.suptitle('The selected continuation path moves below the earlier plateau', fontsize=15, y=1.01)
    save(fig, 'continuation_lineage',
         'One preparation seed; exposed validation panels. Dashed lines mark recipe changes, not independent runs.\n'
         'Tokens exclude EXP-111 and earlier preparation; repeated FIT data. EXP-153 ends at its last saved evaluation, not a final test.')
    # Equal additional tokens from the exact same trained source.
    fig,axes = plt.subplots(1,3,figsize=(13.2,4.7))
    comparisons = [(r154,'R32, preserved Adam',COLORS[0]),
                   (growth[0],'R64, fresh Adam',COLORS[1]),
                   (growth[1],'R256, fresh Adam',COLORS[3])]
    matched = []
    maps = [{e['new_steps']:e for e in branch(r)['evaluations']} for r,_,_ in comparisons]
    for s in sorted(set.intersection(*(set(m) for m in maps))):
        matched.append({'new_updates':s, 'input_tokens':s*256,
                        'arms':{label:{p:nll(m[s],p) for p in 'ABC'} for m,(_,label,_) in zip(maps,comparisons)}})
    for ax,p in zip(axes,'ABC'):
        for r,label,color in comparisons:
            values = series(r,p)
            xs=[(x-31126)*256/1e6 for x,y,k in values];ys=[y for x,y,k in values]
            ax.plot(xs,ys,marker='o',ms=4,lw=2,color=color,label=label)
            ax.annotate(f'{ys[-1]:.3f}',(xs[-1],ys[-1]),xytext=(5,5),textcoords='offset points',color=color)
            if r is r154:
                last=branch(r)['final']['last'];lx=(last['step']-31126)*256/1e6;ly=nll(last,p)
                ax.plot([xs[-1],lx],[ys[-1],ly],':',color=color)
                ax.scatter(lx,ly,marker='X',s=48,color=color)
            for x,y,k in values:
                POINTS.append(dict(figure='rank_growth_matched',arm=label,panel=p,step=x-31126,
                                   input_tokens=(x-31126)*256,nll=y,kind=k))
        ax.set_title(f'Validation {p}',fontweight='bold');ax.set_xlabel('Additional input tokens (millions)')
        ax.set_ylabel('NLL (lower is better)');ax.grid(alpha=.18);ax.margins(x=.15,y=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=9)
    fig.suptitle('Source-preserving growth: R256 improves later in the observed window',fontsize=15,y=1.01)
    save(fig,'rank_growth_matched',
         'EXP-153 v2 versus EXP-154 ordinary continuation; same source at global step 31,126.\n'
         'Fresh Adam is shared by R64/R256, but differs from R32. Circles: registered evaluations; X: deadline endpoint. No extrapolation.')
    # Two control screens with dense numerical annotations; no causal cross-row contrast.
    control = read('results/EXP-148-fold-continue-analysis.json')['matched145']
    lr = read('results/EXP-147-fold-continue-analysis.json')['matched']
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.5))
    for j,p in enumerate('ABC'):
        ax=axes[0,j]
        for key,label,color in [('control','Retain factors + fresh Adam',COLORS[0]),('fold','Fold + fresh R32/Adam',COLORS[1])]:
            ax.plot([e['new_steps']*256/1e6 for e in control],[e[key][p] for e in control],marker='o',ms=3,lw=2,color=color,label=label)
            for e in control:
                POINTS.append(dict(figure='continuation_controls_fold',arm=label,panel=p,step=e['new_steps'],
                                   input_tokens=e['new_steps']*256,nll=e[key][p],kind='matched'))
        e=control[-1];adv=e['control'][p]-e['fold'][p]
        ax.text(.04,.06,f'Fold advantage at {e["new_steps"]:,}: {adv:.3f} NLL',transform=ax.transAxes,fontsize=9)
        ax.set_title(f'Fold control | panel {p}',fontweight='bold');ax.legend(fontsize=8,loc='upper right')
        ax=axes[1,j]
        for key,label,color in [('original','Original LR',COLORS[0]),('div3','LR / 3',COLORS[1]),('div10','LR / 10',COLORS[2])]:
            ax.plot([e['new_steps']*256/1e6 for e in lr],[e[key][p] for e in lr],marker='o',ms=3,lw=2,color=color,label=label)
            for e in lr:
                POINTS.append(dict(figure='continuation_controls_lr',arm=label,panel=p,step=e['new_steps'],
                                   input_tokens=e['new_steps']*256,nll=e[key][p],kind='matched'))
        ax.set_title(f'Continuation LR screen | panel {p}',fontweight='bold');ax.legend(fontsize=8,loc='upper right')
    for ax in axes.flat:
        ax.set_xlabel('Additional input tokens (millions)');ax.set_ylabel('NLL');ax.grid(alpha=.18);ax.margins(y=.18)
    fig.suptitle('Controls constrain the explanation: extra training and optimization both matter',fontsize=14)
    save(fig,'continuation_controls',
         'Top: EXP-145 versus repaired EXP-148 control, same source and fresh Adam. Bottom: EXP-147/148, another shared source.\n'
         'Only shared update counts are plotted in each row. One seed, reused validation; folding does not isolate capacity.')
    audit={'date':'2026-10-20','sources':SOURCES,'scope':'One seed; same exposed A/B/C panels; partial horizons; frozen report snapshots.',
           'matched_growth':matched,
           'growth_snapshot_status':[{'arm':r['contract']['recipes'][0]['name'],'status':r['status'],
                'durable_step':branch(r)['durable_step'],'latest_evaluation_step':branch(r)['evaluations'][-1]['step'],
                'best_step':branch(r)['best_step'],'cost':r['cost']} for r in growth]}
    (OUT/'continuation_provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'continuation_points.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(POINTS[0]));writer.writeheader();writer.writerows(POINTS)
    print(f'Rebuilt three figures; {len(POINTS)} traceable points; {len(matched)} common growth checkpoints.')

if __name__ == '__main__':
    main()
