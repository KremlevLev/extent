"""Offline EXP155 report audit and unsmoothed full-panel learning curves."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=ROOT/'results/EXP-155-session-1.json'
    d=json.loads(raw.read_text(encoding='utf-8'));b=d['branches']['CONTINUE-R256-ADAM']
    plan=json.loads((ROOT/'results/EXP-155-engineering-plan.json').read_text())
    assert d['contract']==plan['contract']
    assert d['status']=='deadline_partial' and d['stop_reason']=='registered work deadline'
    assert b['step']==b['durable_step']==80260 and b['source_step']==37910
    assert b['adam_count_at_restore']==[6784]*6 and not b.get('failed')
    assert b['finalization_status']=='completed' and b['source_replay']['passed']
    assert not d['pending_slots'] and not d.get('error') and not d.get('report_sync_error')
    assert d['cost']['accepted_updates']==d['cost']['attempted_updates']==42350
    assert d['cost']['accepted_input_tokens']==42350*256
    assert b['execution_checks'][0]['passed'] and b['execution_checks'][0]['selected']=='compiled'
    es=b['evaluations'];start=es[0];last=b['final']['last'];best=b['final']['best']
    assert best['step']==b['best_step']==78870
    assert best['panels']==next(e['panels'] for e in es if e['step']==78870)
    for label,e in b['final'].items():
        assert all(x['accepted'] for x in e['profile']['layers']),label
    points=[]
    for e in es+[dict(last,new_steps=last['step']-37910,endpoint='deadline')]:
        for p in 'ABC':
            v=e['panels'][p];assert len(v['windows'])==(32 if p=='B' else 64)
            points.append(dict(step=e['step'],new_steps=e['new_steps'],input_tokens=e['new_steps']*256,
                               panel=p,nll=v['nll'],kind=e.get('endpoint','registered')))
    analysis={'date':'2026-10-23','raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
              'status':d['status'],'source_step':37910,'last_step':80260,'registered_best_step':78870,
              'accepted_updates':42350,'accepted_input_tokens':42350*256,
              'audit_extra_input_tokens':d['cost']['engine_audit_extra_input_tokens'],
              'wall_hours':d['sessions'][-1]['hours'],'training_hours':d['cost']['training_seconds']/3600,
              'notification_count':len(d['notifications']),
              'source':{p:start['panels'][p]['nll'] for p in 'ABC'},
              'last':{p:last['panels'][p]['nll'] for p in 'ABC'},
              'registered_best':{p:best['panels'][p]['nll'] for p in 'ABC'},
              'source_minus_last':{p:start['panels'][p]['nll']-last['panels'][p]['nll'] for p in 'ABC'},
              'late_32768_to40960_gain':{p:es[-2]['panels'][p]['nll']-es[-1]['panels'][p]['nll'] for p in 'ABC'},
              'interpretation':'Successful work-deadline continuation, not completed262144 horizon. No hard plateau in observed late window; no causal rank claim. Single seed, reused validation, cyclic FIT, prep excluded. Last is distinct from A-selected registered best.'}
    (ROOT/'results/EXP-155-analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'results/EXP-155-points.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.4))
    for ax,p,color in zip(axes,'ABC',['#247b9c','#7654a3','#16856c']):
        xs=[e['new_steps']*256/1e6 for e in es];ys=[e['panels'][p]['nll'] for e in es]
        ax.plot(xs,ys,color=color,marker='o',ms=4,lw=2)
        x=(last['step']-37910)*256/1e6;y=last['panels'][p]['nll']
        ax.plot([xs[-1],x],[ys[-1],y],':',color=color);ax.scatter(x,y,marker='X',s=60,color=color)
        ax.annotate(f'{y:.3f}',(x,y),xytext=(5,5),textcoords='offset points',color=color,fontweight='bold')
        ax.annotate(f'{ys[0]:.3f}',(xs[0],ys[0]),xytext=(5,7),textcoords='offset points',color=color)
        ax.set_title(f'Validation {p} ({32 if p=="B" else 64} windows)',fontweight='bold')
        ax.set_xlabel('Additional input tokens (millions)');ax.set_ylabel('NLL (lower is better)')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.margins(x=.12,y=.12)
    fig.suptitle('EXP-155: R256 continues improving on TPU without another reset',fontsize=14)
    fig.text(.5,.02,'Same prepared model and Adam; 42,350 accepted updates / 10.842M input positions.\n'
             'Circles: registered measurements; X: deadline endpoint. Separate panels; no smoothing. One seed, reused validation; preparation excluded.',
             ha='center',fontsize=9,color='#4b5563')
    fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ('png','pdf','svg'):
        path=ROOT/f'results/EXP-155-trajectory.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    print(f'Audited {len(points)} panel points; all clocks, gates and endpoint profiles pass.')


if __name__=='__main__':main()
