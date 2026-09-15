#!/usr/bin/env python3
"""Render Fig. 3 from coverage_audit.py output; contains no audit counts."""
from pathlib import Path
from collections import Counter
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COL_W=3.45
BLUE_LIGHT,PEACH,BLUE,PALE='#A8B6C8','#F1B98B','#5C6F94','#F2F2F2'
INK,FRAME='#35435A','#758197'
ORDER=['PeerExchange','DispatchExecute','ParallelAggregate','EvaluateRefine']
NAME={'PeerExchange':'Peer exchange','DispatchExecute':'Dispatch','ParallelAggregate':'Parallel agg.','EvaluateRefine':'Eval.-refine'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'axes.labelsize':6.5,
        'xtick.labelsize':6,'ytick.labelsize':6,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':FRAME,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})

def boxed(ax,grid=None):
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color(FRAME);spine.set_linewidth(.6)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color=PALE,linewidth=.6)

def save(fig,path):fig.savefig(path,bbox_inches=None,metadata={'Creator':'MASBench evaluation/plot_coverage.py'});plt.close(fig);print(path)

def upset(metrics,out):
    raw=metrics['plot_data']['family_intersections']
    intersections=sorted([(tuple(r['families']),r['papers']) for r in raw],key=lambda p:(-p[1],len(p[0]),p[0]))
    marginal=Counter()
    for combo,count in intersections:
        for family in combo:marginal[family]+=count
    fig=plt.figure(figsize=(COL_W,1.78));left=fig.add_axes([.76/COL_W,.34/1.78,.43/COL_W,.67/1.78])
    matrix=fig.add_axes([1.38/COL_W,.34/1.78,1.99/COL_W,.67/1.78]);top=fig.add_axes([1.38/COL_W,1.08/1.78,1.99/COL_W,.62/1.78])
    x=np.arange(len(intersections));counts=[c for _,c in intersections];ceiling=max(counts+[1])
    top.bar(x,counts,color=BLUE,width=.72,zorder=3)
    for xi,count in zip(x,counts):top.text(xi,count+ceiling*.025,str(count),ha='center',va='bottom',fontsize=5.7)
    top.set(xlim=(-.65,len(x)-.35),ylim=(0,ceiling*1.2),ylabel='Papers');top.set_xticks([]);boxed(top,'y')
    y=np.arange(4);values=[marginal[m] for m in ORDER];left.barh(y,values,height=.58,color=PEACH,zorder=3)
    left.set_yticks(y,labels=[NAME[m] for m in ORDER]);left.set(xlim=(0,max(values+[1])*1.18),ylim=(3.55,-.55),xlabel='Papers');left.tick_params(axis='y',length=0);boxed(left)
    for yi in range(4):
        if yi%2==0:matrix.axhspan(yi-.5,yi+.5,color=PALE,zorder=0)
    for xi,(combo,_) in enumerate(intersections):
        matrix.scatter([xi]*4,y,s=6,color=BLUE_LIGHT,alpha=.55,zorder=1);active=[i for i,m in enumerate(ORDER) if m in combo]
        if len(active)>1:matrix.plot([xi,xi],[min(active),max(active)],color=BLUE,lw=.9,zorder=2)
        matrix.scatter([xi]*len(active),active,s=13,color=BLUE,zorder=3)
    matrix.set(xlim=(-.65,len(x)-.35),ylim=(3.55,-.55),xlabel='Family combination');matrix.set_xticks([]);matrix.set_yticks([]);boxed(matrix)
    save(fig,out/'fig3a_workflow_composition_upset.pdf')

def bars(metrics,out):
    rows=metrics['plot_data']['metrics'];labels=[r['label'] for r in rows]
    values=[r.get('value',r.get('numerator',0)/r.get('denominator',1))*100 for r in rows]
    fig=plt.figure(figsize=(COL_W,1.65));ax=fig.add_axes([1.03/COL_W,.34/1.65,2.34/COL_W,1.23/1.65]);y=np.arange(len(rows))*1.1
    colors=[BLUE_LIGHT,PEACH]+[BLUE]*(len(rows)-2);ax.barh(y,values,height=.67,color=colors,zorder=3)
    ax.set_yticks(y,labels=labels);ax.tick_params(axis='y',length=0);ax.set(xlim=(0,105),ylim=(y[-1]+.6,-.6),xlabel='Coverage / F1 (%)');ax.set_xticks([0,25,50,75,100])
    for yy,pct,row,color in zip(y,values,rows,colors):
        suffix=f" ({row['numerator']}/{row['denominator']})" if 'numerator' in row else ''
        ax.text(pct/2,yy,f'{pct:.1f}%{suffix}',color='white' if color==BLUE else INK,ha='center',va='center',fontsize=6,zorder=4)
    boxed(ax,'x');save(fig,out/'fig3b_specification_coverage.pdf')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    metrics=json.loads(a.metrics.read_text());a.output_dir.mkdir(parents=True,exist_ok=True);style();upset(metrics,a.output_dir);bars(metrics,a.output_dir)

if __name__=='__main__':main()
