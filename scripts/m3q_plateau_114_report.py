"""CPU-only CSV export and static figures from EXP114 JSON; no model loading."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def export_tables(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);files=[]
    def write(name,rows):
        if not rows:return
        path=output/name;fields=sorted(set(k for row in rows for k in row))
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        files.append(path)
    rows=[]
    for row in result.get('tables',{}).get('interventions',{}).values():
        rows.append({k:v for k,v in row.items() if not isinstance(v,(list,dict))}|{'indices':','.join(map(str,row['indices']))})
    write('extent-m3q-plateau-114-interventions.csv',rows)
    rows=[]
    for seed,data in result.get('readout',{}).items():
        rows += [dict(seed=seed,step=step,A_nll=row['nll']) for step,row in data.get('curve',{}).items()]
    write('extent-m3q-plateau-114-readout.csv',rows)
    rows=[]
    for key,task in result.get('tasks',{}).items():
        if key.startswith('profile/') and '/layer' in key:
            _,seed,layer,window=key.split('/');data=task['value']
            rows.append(dict(seed=seed,layer=int(layer.removeprefix('layer')),window=int(window),
                hidden_rms=data['hidden']['rms'],mixed_rms=data['mixed']['rms'],mlp_rms=data['mlp']['rms'],
                mixer_teacher_error=data['mixer_teacher_input_error']['relative_l2'],
                mixer_student_error=data['mixer_student_input_error']['relative_l2'],
                input_drift=data['input_drift']['relative_l2'],amplification=data['amplification']))
    write('extent-m3q-plateau-114-layer-profile.csv',rows)
    return files

def figures(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True);saved=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf','svg'):
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=160);saved.append(path)
        plt.close(fig)
    tasks=result.get('tasks',{});choices=result.get('choices',{})
    rows=list(result.get('tables',{}).get('interventions',{}).values())
    seeds=[s for s in ('123','456') if any(k.startswith(f'profile/{s}/layer') for k in tasks)]
    if seeds:
        fig,axes=plt.subplots(1,len(seeds),figsize=(6*len(seeds),7),squeeze=False)
        fields=('hidden_rms','mixed_rms','input_drift','teacher_input_error','student_input_error')
        for seed,ax in zip(seeds,axes[0]):
            data={}
            for k,r in tasks.items():
                if k.startswith(f'profile/{seed}/layer'):
                    j=int(k.split('/')[2].removeprefix('layer'));row=r['value']
                    data.setdefault(j,[]).append([row['hidden']['rms'],row['mixed']['rms'],row['input_drift']['relative_l2'],
                        row['mixer_teacher_input_error']['relative_l2'],row['mixer_student_input_error']['relative_l2']])
            layers=sorted(data);matrix=np.asarray([np.median(np.asarray(data[j],float),axis=0) for j in layers])
            image=ax.imshow(np.log10(np.maximum(matrix,1e-12)),aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(fields)),fields,rotation=45,ha='right');ax.set_yticks(range(len(layers)),layers)
            ax.set_ylabel('Decoder layer');ax.set_title(f'Seed {seed}: log10 discovery medians')
            fig.colorbar(image,ax=ax)
        save(fig,'extent-m3q-plateau-114-layer-profile')
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(12,4),squeeze=False)
        for seed,ax in zip(('123','456'),axes[0]):
            selected=choices.get(seed,{}).get('single',{})
            for family,j in selected.items():
                found=[next((r for r in rows if str(r['seed'])==seed and r['family']==family and
                    r['indices']==[j] and not r['neighbor'] and r['panel']==panel and r['tail'] is None),None)
                    for panel in ('A','B','C_wiki','C_pg19')]
                xx=[i for i,r in enumerate(found) if r is not None];yy=[r['delta_nll'] for r in found if r is not None]
                ax.plot(xx,yy,'o--',label=f'{family}, layer {j}')
            ax.axhline(0,color='black',linewidth=.7);ax.axhline(-.1,color='gray',linestyle=':')
            ax.set_xticks(range(4),('A reused','B','C Wiki','C PG19'));ax.set_ylabel('Paired NLL delta (negative improves)')
            ax.set_title(f'Seed {seed}; A-selected layers');ax.legend(fontsize=8)
        save(fig,'extent-m3q-plateau-114-rescue-verification')
    if any(data.get('curve') for data in result.get('readout',{}).values()):
        fig,ax=plt.subplots(figsize=(6,4))
        for seed,data in result['readout'].items():
            points=sorted((int(step),row['nll']) for step,row in data.get('curve',{}).items())
            if points:ax.plot([s*256 for s,n in points],[n for s,n in points],'o--',label=f'Seed {seed}')
        ax.set_xlabel('Readout FIT input tokens (frozen decoder forwards extra)');ax.set_ylabel('A NLL (selection)')
        ax.set_title('Observed readout endpoints; no interpolated measurements');ax.legend()
        save(fig,'extent-m3q-plateau-114-readout-fit')
    return saved

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('json');p.add_argument('--output-dir')
    args=p.parse_args(argv);source=Path(args.json);result=json.loads(source.read_text())
    output=Path(args.output_dir) if args.output_dir else source.parent/'exp114-report'
    files=export_tables(result,output)+figures(result,output)
    print(f'Exported {len(files)} files into {output}');return files

if __name__=='__main__':main()
