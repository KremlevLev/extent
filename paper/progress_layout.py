"""Offline Russian historical overview; model stages and domains stay separate."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter

COLORS={'A':'#a95f1c','B':'#9176b6','C':'#128378','D':'#326ab2','E':'#ba4561'}


def rows_for(data,experiment,arm,panel):
    return sorted([r for r in data['points'] if r['experiment']==experiment and r['arm']==arm and panel in r['panels']],
                  key=lambda r:r['full_model_input_positions'])


def line(ax,rows,panel,**kwargs):
    return ax.plot([r['full_model_input_positions']/1e6 for r in rows],
        [r['panels'][panel]['nll'] for r in rows],color=COLORS[panel],**kwargs)


def draw(early,old,same,data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'extent-progress-20261029',
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(20,12),facecolor='#fafbf9')
    grid=fig.add_gridspec(2,3,left=.065,right=.95,bottom=.20,top=.86,hspace=.40,wspace=.30,height_ratios=[1.5,1])
    ax=fig.add_subplot(grid[0,:]);zoom=fig.add_subplot(grid[1,0]);domain=fig.add_subplot(grid[1,1]);table=fig.add_subplot(grid[1,2])
    for axis in (ax,zoom,domain):
        axis.set_facecolor('#fafbf9');axis.grid(alpha=.14);axis.set_ylabel('NLL ↓')
    ax.set_yscale('log');ax.set_ylim(3.7,27);ax.set_xlim(-22,153)
    ax.yaxis.set_major_locator(FixedLocator([4,5,6,7,10,15,20,25]));ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
    ax.set_xticks([0,8.39,20,38.67,60,80,92.10,120,144.50]);ax.set_xticklabels(['0','8,39','20','38,67','60','80','92,10','120','144,50'])
    ax.set_xlabel('Млн позиций сквозного обучения вдоль выбранного пути; альтернативная Wiki-ветка показана отдельно',labelpad=10)
    ax.axvspan(-22,-2,color='#e7ece9');ax.axvline(-2,color='#9aa9a1',ls=':',lw=1)
    for x,key in [(-17,'TEACHER'),(-8,'ONPOLICY')]:
        nll=early[key]['nll'];ax.scatter(x,nll,color='#627e70',s=50,zorder=8)
        ax.text(x,nll*1.12,f'{nll:.2f}'.replace('.',','),ha='center',fontweight='bold',fontsize=12)
    ax.text(-17,13.0,'Чистые\nвходы',ha='center',fontsize=9)
    ax.text(-8,7.5,'Входы\nгибрида',ha='center',fontsize=9)
    ax.text(-12,4.6,'EXP-072\nДва разных старта;\nX здесь не считает токены',ha='center',fontsize=8.5,color='#53675e')
    ax.plot([r['full_model_input_positions']/1e6 for r in old],[r['nll'] for r in old],color='#596d9d',lw=2.5)
    ax.annotate('12,97 → 7,29\nзащищённая R32',xy=(2.1,7.3),xytext=(5,15.3),fontsize=10,
        arrowprops=dict(arrowstyle='->',color='#596d9d'),color='#3a5183')
    ax.axvline(8.388608,color='#8e9f94',ls=':',lw=1)
    ax.text(9.5,23,'Новая оценка: A / B / C',fontsize=9,color='#67766d')
    for panel in 'ABC':
        x=[r['full_model_input_positions']/1e6 for r in same[panel]];y=[r['nll'] for r in same[panel]]
        for axis in (ax,zoom):axis.plot(x,y,color=COLORS[panel],lw=1.7,alpha=.85)
    fork=data['offsets']['exp155_end']/1e6;c4end=data['offsets']['exp161_c4_end']/1e6
    fullstart=data['offsets']['exp166_start']/1e6;end=data['offsets']['exp166_end']/1e6
    for lo,hi,color in [(8.388608,fork,'#d3e8dd'),(fork,c4end,'#dbe5f2'),(c4end,end,'#eadfe8')]:ax.axvspan(lo,hi,color=color,alpha=.25)
    for x in (fork,c4end,fullstart):ax.axvline(x,color='#a8b6af',ls=':',lw=.8)
    ax.text((8.39+fork)/2,10,'Норма + tanh + сброс\n→ сворачивание → R256',ha='center',fontsize=10,color='#2e6658')
    ax.text((fork+c4end)/2,10,'CE + дистилляция\nWiki ↔ свежий C4',ha='center',fontsize=10,color='#325e85')
    ax.text((fullstart+end)/2,10,'Свежий FineWeb-Edu\nвесь декодер + Adam',ha='center',fontsize=10,color='#784d6d')
    for panel in 'ABC':
        line(ax,rows_for(data,161,'KD-C4-FRESH',panel),panel,lw=1.7)
        line(ax,rows_for(data,163,'FULL-KD',panel),panel,lw=1.7)
        line(ax,rows_for(data,166,'FULL-KD',panel),panel,lw=1.7)
    wiki=rows_for(data,161,'KD-WIKI-CYCLIC','C');line(ax,wiki,'C',ls='--',lw=1.8,alpha=.8)
    ax.annotate('Wiki-ветка: C = 4,094\nДругая ветка, не предок EXP-166',
        xy=(wiki[-1]['full_model_input_positions']/1e6,wiki[-1]['panels']['C']['nll']),xytext=(57,4.67),
        fontsize=9,color=COLORS['C'],arrowprops=dict(arrowstyle='-',color=COLORS['C']))
    for panel in 'DE':
        for exp,arm in [(161,'KD-C4-FRESH'),(163,'FULL-KD'),(166,'FULL-KD')]:
            rs=rows_for(data,exp,arm,panel)
            if rs:
                line(ax,rs,panel,lw=2.3)
                line(domain,rs,panel,lw=2.2,marker='o',ms=2.5)
    final=next(r for r in reversed(data['points']) if r['experiment']==166 and r['kind']=='final')
    for panel,label_y in [('A',6.35),('B',5.99),('C',5.58),('D',5.18),('E',4.78)]:
        value=final['panels'][panel]['nll']
        ax.annotate(f'{panel} {value:.3f}'.replace('.',','),xy=(end,value),xytext=(147,label_y),
            fontsize=10,fontweight='bold',color=COLORS[panel],arrowprops=dict(arrowstyle='-',color=COLORS[panel],lw=.7))
    ax.legend(handles=[Line2D([0],[0],color='#596d9d',lw=2,label='Исторический EXP-111')]+
        [Line2D([0],[0],color=COLORS[p],lw=2,label='Wiki '+p if p in 'ABC' else 'C4 D' if p=='D' else 'Edu E') for p in 'ABCDE'],
        ncol=6,frameon=False,loc='upper center',bbox_to_anchor=(.53,1.17),fontsize=10)
    zoom.set_title('Раннее восстановление: увеличение',loc='left',fontweight='bold',fontsize=12)
    zoom.set_xlim(8.3,39.5);zoom.set_ylim(4.1,7.2);zoom.set_xlabel('Млн позиций с начала EXP-111')
    for step,label in [(0,'136'),(12586,'145'),(24965,'147'),(31126,'153'),(37910,'155')]:
        x=(8388608+step*256)/1e6;zoom.axvline(x,color='#acb7b0',ls=':',lw=.6)
        zoom.text(x+.15,7.0,label,fontsize=8,color='#566a60')
    zoom.text(24.5,6.75,'Сброс → fold → рост → продолжение',fontsize=9,ha='center',color='#51675b')
    domain.set_title('Свежие домены: C4 и FineWeb-Edu',loc='left',fontweight='bold',fontsize=12)
    domain.set_xlabel('Млн позиций на выбранном пути');domain.set_xlim(38,148);domain.set_ylim(4.95,7.2)
    domain.axvline(c4end,color='#aaa',ls=':',lw=1);domain.axvline(fullstart,color='#aaa',ls=':',lw=1)
    domain.text(98,6.35,'EXP-166: +48,05 млн\nD: 5,176 → 5,133\nE: 5,110 → 5,018',fontsize=10,linespacing=1.6,color='#53605a')
    domain.legend(handles=[Line2D([0],[0],color=COLORS[p],lw=2,label='C4 D' if p=='D' else 'FineWeb-Edu E') for p in 'DE'],frameon=False,fontsize=9,loc='upper left')
    table.axis('off');table.set_title('Последний CURRENT: EXP-166',loc='left',fontweight='bold',fontsize=12)
    table.text(0,.89,'Панель                         NLL       Δ за этап',fontweight='bold',fontsize=11)
    source=next(r for r in data['points'] if r['experiment']==166)
    for i,p in enumerate('ABCDE'):
        nll=final['panels'][p]['nll'];delta=nll-source['panels'][p]['nll']
        table.text(0,.76-i*.12,f'{"Wiki "+p if p in "ABC" else "C4 D" if p=="D" else "FineWeb-Edu E":<22} {nll:6.3f}    {delta:+.3f}',fontfamily='DejaVu Sans Mono',fontsize=10,color=COLORS[p])
    table.text(0,.09,'Учитель: D = 2,884; E = 2,616\nРазрыв сохраняется; NLL ≠ восстановленные способности.',fontsize=9,color='#5c6a61',linespacing=1.5)
    fig.text(.065,.958,'Extent: от ошибки композиции к восстановлению на новых данных',fontsize=23,fontweight='bold',color='#193f33')
    fig.text(.065,.923,'Qwen3-1.7B → 24 Mamba-3 MIMO + 4 GQA  ·  результаты по EXP-166 включительно  ·  один выбранный путь и явно отмеченная альтернатива',fontsize=11,color='#5b7065')
    fig.text(.065,.126,'Выбранный путь: 144,501 млн позиций сквозного обучения + 17,302 млн позиций послойной подготовки.',fontsize=13,fontweight='bold',color='#244c3b')
    fig.text(.065,.095,'Арифметически 161,803 млн предъявлений. Блок и вся сеть требуют разных вычислений; это не уникальные токены и не полная стоимость исследования.',fontsize=10,color='#53695b')
    fig.text(.065,.064,'Учтены EXP-160 C4 (0,541 млн), EXP-161 C4 (52,889 млн), EXP-163 FULL (4,350 млн), EXP-164 (256 позиций) и EXP-166 (48,046 млн).',fontsize=10,color='#53695b')
    fig.text(.065,.035,'Другие ветки, поисковые пилоты, оценки и проходы учителя считаются отдельно. Панели не смешаны; пропуски не интерполированы; EXP-167 пока без результатов.',fontsize=9.5,color='#65746b')
    return fig
