"""Reuse deploy-key/setup cells; default to long EXP120, not old EXP116."""
import argparse
import json
from pathlib import Path
from scripts.build_t4_notebook import build


def build_long(source,target):
    notebook=build(source,target)
    for cell in notebook['cells']:
        cell['source']=cell['source'].replace('EXP116/117/118/119','EXP120/121/122/123')
        cell['source']=cell['source'].replace('EXP-116-119-t4-launch.md','EXP-120-123-t4-launch.md')
        cell['source']=cell['source'].replace('EXP116 по умолчанию','EXP120 по умолчанию')
        cell['source']=cell['source'].replace('EXP117/118/119','EXP121/122/123')
        cell['source']=cell['source'].replace('scripts.m3q_t4_campaign','scripts.m3q_t4_scale_campaign')
        cell['source']=cell['source'].replace('"--experiment", "116"','"--experiment", "120"')
        cell['source']=cell['source'].replace('У EXP119 отдельно смотри','У EXP121–123 отдельно смотри')
        cell['source']=cell['source'].replace('GPU-результатов на момент подготовки ещё нет.','EXP116–119 завершены; новые EXP120–123 ещё не запускались на GPU.')
        if cell['cell_type']=='markdown' and 'EXP120 по умолчанию' in cell['source']:
            cell['source']+='\n\nEXP121–123: 65 536 полных CE/KL обновлений на GPU, с продолжением по HF checkpoint. EXP120: 461 условие масштаба, A64 каждое; объём делится между картами. Это плановый объём, не обещание закончить за одну сессию.\n'
    Path(target).write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return notebook


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');a=p.parse_args()
    build_long(a.source,a.target)
