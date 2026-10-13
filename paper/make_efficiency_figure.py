"""Rebuild Figure 3 from audited extracted points; no model or remote access."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'figures'

def main():
    with (ROOT / 'quality_points.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    styles = {111: ('Protected rank32', '#167d9a'),
              112: ('Same-start decoder CE (partial)', '#e59f3a'),
              113: ('Random Mamba, decoder CE', '#bb4b47')}
    for j, seed in enumerate(('123', '456')):
        for i, panel in enumerate(('reproduction', 'locked_test')):
            ax = axes[i, j]
            for exp, (label, color) in styles.items():
                points = sorted((int(r['input_tokens']) / 1e6, float(r['nll']))
                                for r in rows if int(r['experiment']) == exp
                                and r['seed'] == seed and r['panel'] == panel)
                ax.plot([x for x, y in points], [y for x, y in points],
                        marker='o', markersize=2.5, color=color, label=label)
            ax.set_title(f'Seed {seed}: {"historical reused" if i == 0 else "fresh locked Wiki"}')
            ax.set_ylabel('NLL (lower is better)')
            ax.grid(alpha=.15)
            if i == 1:
                ax.set_xlabel('MAIN input tokens (millions)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('Recovery controls: recorded MAIN evaluations; pilots and warm-start cost excluded')
    fig.tight_layout(rect=(0, .045, 1, .96))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(ROOT / f'quality_curves.{ext}', dpi=160)
    svg = ROOT / 'quality_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)
    print(f'Figure 3 rebuilt from {len(rows)} recorded points.')

if __name__ == '__main__':
    main()
