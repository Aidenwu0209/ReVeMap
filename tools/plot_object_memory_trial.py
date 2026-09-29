"""Plot measured second-pass timings and call budgets, not projected speedups."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'evaluation/DETAILS.json').read_text())
scenes=json.loads((ROOT/'docs/object-memory-plan.json').read_text())['scenes']
fig,axs=plt.subplots(1,2,figsize=(13,5.7))
fig.subplots_adjust(left=.065,right=.98,top=.79,bottom=.25,wspace=.18)
colors={'A':'#8896a7','B':'#167d9a'}
labels={'A':'A: original reference','B':'B: object memory'}
x=np.arange(len(scenes));w=.35
for i,arm in enumerate('AB'):
    values=[next(r for r in rows if r['repeat']==2 and r['scene']==s and r['arm']==arm) for s in scenes]
    for ax,key in zip(axs,['tail_wall_seconds','vlm_calls']):
        bars=ax.bar(x+(i-.5)*w,[r[key] for r in values],width=w,color=colors[arm],label=labels[arm])
        ax.bar_label(bars,labels=[f'{r[key]:.1f}' if key=='tail_wall_seconds' else str(r[key]) for r in values],fontsize=8,padding=3)
for ax,title in zip(axs,['Measured post-SAM3 tail (seconds)','Fresh VLM calls']):
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=12);ax.set_xticks(x,[s.replace('scene','') for s in scenes])
    ax.set_xlabel('ScanNet scene | 25 fixed views each');ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.16);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
fig.suptitle('ssh33 / RTX 4060 Laptop: 28.3% shorter tail with unchanged outputs',fontsize=15,fontweight='bold',y=.96)
handles,names=axs[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,frameon=False)
fig.text(.065,.035,'Second pass, warm filesystem; fresh model processes. Frozen geometry + cached SAM3; not full RGB-D pipeline timing.',fontsize=9,color='#555555')
out=ROOT/'evaluation/object-memory-comparison.png';fig.savefig(out,dpi=170,facecolor='white');print(out)
