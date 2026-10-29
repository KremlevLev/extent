"""Offline matched-scope and late full-decoder recovery figures; no model run."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from paper.progress_layout import COLORS,rows_for


def main():
    data=json.loads((ROOT/'paper/latest_recovery_data.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'extent-latest-20261029'})
    fig,(ax,curve)=plt.subplots(1,2,figsize=(17,6.5),gridspec_kw={'width_ratios':[1.1,1]},facecolor='#fafbf9')
    fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.23,wspace=.30)
    arms=['R256-CONTROL','UPPER8-MLP','FULL-KD']
    matrix=np.array([[next(r for r in data['points'] if r['experiment']==163 and r['arm']==arm and r['step']==156900)['panels'][p]['nll'] for p in 'ABCDE'] for arm in arms])
    delta=matrix-matrix[0]
    image=ax.imshow(delta,cmap='BrBG_r',vmin=-.08,vmax=.08,aspect='auto')
    for i in range(3):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}\n{delta[i,j]:+.3f}',ha='center',va='center',fontsize=11,color='#233b31')
    ax.set_xticks(range(5),['Wiki A','Wiki B','Wiki C','C4 D','Edu E'])
    ax.set_yticks(range(3),['R256\n57,2 млн','+ верхние MLP\n359,2 млн','Весь декодер\n1 364,2 млн'])
    ax.set_title('EXP-163 · одинаковые 3,146 млн токенов',loc='left',fontweight='bold',pad=14)
    bar=fig.colorbar(image,ax=ax,fraction=.04,pad=.04);bar.set_label('Δ NLL относительно R256; меньше лучше')
    for p in 'ABCDE':
        rows=rows_for(data,166,'FULL-KD',p);source=rows[0]['panels'][p]['nll']
        x=[(r['full_model_input_positions']-data['offsets']['exp166_start'])/1e6 for r in rows]
        y=[r['panels'][p]['nll']-source for r in rows]
        curve.plot(x,y,color=COLORS[p],lw=2,marker='o',ms=3,label=p)
    curve.axhline(0,color='#555',lw=.8);curve.grid(alpha=.15)
    curve.set_xlabel('Дополнительные входные токены EXP-166, млн')
    curve.set_ylabel('Δ NLL относительно CURRENT157685')
    curve.set_title('EXP-166 · +48,05 млн токенов',loc='left',fontweight='bold',pad=14)
    curve.legend(ncol=5,frameon=False,loc='lower left')
    fig.text(.10,.94,'Полное обучение: небольшой выигрыш и продолжающееся медленное улучшение',fontsize=19,fontweight='bold',color='#1d4939')
    fig.text(.10,.87,'Слева — сопоставимый бюджет разных рецептов; справа — одна траектория полного декодера.',fontsize=11,color='#5f7067')
    fig.text(.10,.12,'EXP-163: в ячейке NLL и разница с R256. LR различается между рецептами; это не чистая проверка числа параметров.',fontsize=10,color='#53695d')
    fig.text(.10,.075,'EXP-166: одинаковые фиксированные панели; отображены реальные точки без сглаживания. Один сид; панель E — FineWeb-Edu.',fontsize=10,color='#53695d')
    out=ROOT/'paper/figures'
    for ext in ('png','svg','pdf'):fig.savefig(out/f'latest_full_decoder.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    (out/'latest_full_decoder_provenance.json').write_text(json.dumps(dict(date='2026-10-29',
        source='paper/latest_recovery_data.json',sha256=hashlib.sha256((ROOT/'paper/latest_recovery_data.json').read_bytes()).hexdigest(),
        matched_updates=2048,matched_inputs=3145728,full_continuation_inputs=48046080,
        scope='diagnostic validation, one seed; LR and capacity differ in EXP163'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
