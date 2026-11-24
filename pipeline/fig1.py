"""Figure 1: derived criteria against official values, from v3."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import core, water, ladder, tab1

plt.rcParams.update({"font.size":9,"axes.linewidth":.8,
                     "axes.spines.top":False,"axes.spines.right":False})
INK,ACC,WARM,GREY="#1a1a1a","#1f5fa8","#c0392b","#8a8a8a"

# official freshwater aquatic-life values, ng/L
OFFICIAL={"PFOS":[("US EPA 2024",250.0),("EU direct toxicity",230.0),
                  ("ANZG 95 %",130.0),("ANZG 99 %",9.1),
                  ("ECCC Canada",6800.0)],
          "PFOA":[("US EPA 2024",1e5)]}
NAGAI={"PFOS":(330.0,2500.0),"PFOA":(16000.0,720000.0)}
ORDER=[("C1","C1"),("C3","C3"),("C5","C5"),("C6","C6"),
       ("C6p","C6$'$"),("C7a","C7a"),("C7b","C7b")]
KEY={"PFOS":"Perfluorooctane sulfonic acid","PFOA":"PFOA ion"}


def panel(ax,sub,rule="r10"):
    C,out=tab1.crit_table(rule)
    a=water.load(verbose=False)
    nm=a[["parent","name"]].drop_duplicates("parent").set_index("parent").name
    par=[p for p in C.index if str(nm.get(p,""))==KEY[sub]]
    if not par: ax.text(.5,.5,"not found",transform=ax.transAxes); return
    p=par[0]
    vals=[(lab,C.loc[p,col]) for col,lab in ORDER]
    ys=np.arange(len(vals))[::-1]
    for y,(lab,v) in zip(ys,vals):
        if pd.isna(v): continue
        screened = lab in ("C6$'$","C7a","C7b")
        ax.plot(v,y,"o",ms=7,color=ACC if screened else GREY,
                mec="white",mew=.9,zorder=4)
    # official band
    offs=OFFICIAL[sub]
    lo=min(v for _,v in offs); hi=max(v for _,v in offs)
    ax.axvspan(lo,hi,color=WARM,alpha=.10,zorder=0)
    offs_sorted=sorted(offs,key=lambda z:z[1])
    for nmo,v in offs_sorted:
        ax.axvline(v,color=WARM,lw=.9,ls=(0,(4,3)),alpha=.75,zorder=1)
    # label only the extremes: on a log axis spanning eight decades the middle
    # values (230 and 250 ng/L) are indistinguishable and their labels collide
    if len(offs_sorted)==1:
        nmo,v=offs_sorted[0]
        ax.text(v,len(vals)-0.30,"  "+nmo+"  ",rotation=90,fontsize=6.4,
                color=WARM,ha="left",va="top")
    else:
        for nmo,v,ha in ((offs_sorted[0][0],offs_sorted[0][1],"right"),
                         (offs_sorted[-1][0],offs_sorted[-1][1],"left")):
            ax.text(v,len(vals)-0.30,"  "+nmo+"  ",rotation=90,fontsize=6.4,
                    color=WARM,ha=ha,va="top")
    if len(offs_sorted)>2:
        ax.text(np.sqrt(offs_sorted[0][1]*offs_sorted[-1][1]),len(vals)-0.30,
                f"{len(offs_sorted)} authorities",fontsize=6.4,color=WARM,
                ha="center",va="top")
    # Nagai consistent recalculation
    nlo,nhi=NAGAI[sub]
    ax.annotate("",xy=(nlo,-0.62),xytext=(nhi,-0.62),
                arrowprops=dict(arrowstyle="|-|,widthA=.3,widthB=.3",
                                color=INK,lw=1.1))
    ax.text(np.sqrt(nlo*nhi),-0.95,"consistent recalculation",
            fontsize=6.4,color=INK,ha="center",va="top")
    ax.set_xscale("log"); ax.set_yticks(ys)
    ax.set_yticklabels([l for l,_ in vals],fontsize=8)
    ax.tick_params(axis="y",length=0)
    ax.set_ylim(-1.5,len(vals)+0.15)
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1,numticks=99))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("derived criterion (ng/L)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",which="major",color="#e8e8e8",lw=.6,zorder=0)
    return vals


if __name__=="__main__":
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
    for ax,sub,t in zip(axes,("PFOS","PFOA"),("A   PFOS","B   PFOA")):
        panel(ax,sub)
        ax.set_title(t,loc="left",fontsize=9.5,color=INK,pad=6)
    axes[0].text(.02,.965,"blue: screened routes",transform=axes[0].transAxes,
                 fontsize=7,color=ACC,va="top")
    fig.tight_layout()
    fig.savefig(OUT + "/fig_03_official.png",dpi=300,
                bbox_inches="tight",facecolor="white")
    print("wrote fig_03_official.png")
