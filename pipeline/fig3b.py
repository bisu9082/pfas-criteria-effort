"""Figure 3: the broad set -- depth equalisation and the mediator control."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import mech

RNG=np.random.default_rng(mech.SEED)
plt.rcParams.update({"font.size":9,"axes.linewidth":.8,
                     "axes.spines.top":False,"axes.spines.right":False})
INK,ACC,WARM,GREY="#1a1a1a","#1f5fa8","#c0392b","#8a8a8a"

B=mech._read(mech.SRC)
byc={c:v.logTox.values for c,v in B.groupby("test_cas")}
n_all=len(byc)
stat={c:(len(v),v.min(),np.median(v)) for c,v in byc.items()}
D=pd.DataFrame([(c,n,mn,md) for c,(n,mn,md) in stat.items()],
               columns=["cas","n","mn","md"])
D["ln"]=np.log10(D.n)

# qcut on a heavily tied log-count axis drops bins: 14 requested leaves 9.
# The caption quotes the realised number, so it is computed once here.
NBIN_REQ=14
NBIN=pd.qcut(D.ln,NBIN_REQ,duplicates="drop").cat.categories.size


def panelA(ax):
    for col,c,lab,mk in (("mn",WARM,"minimum","o"),("md",GREY,"median","s")):
        q=pd.qcut(D.ln,NBIN_REQ,duplicates="drop"); g=D.groupby(q,observed=True)
        x=g.ln.mean().values; y=g[col].mean().values
        se=(g[col].std()/np.sqrt(g[col].size())).values
        ax.errorbar(x,y,yerr=1.96*se,fmt=mk+"-",color=c,ms=3.6,lw=1.3,
                    capsize=1.8,elinewidth=.8,label=lab,alpha=.95)
        r,_=stats.spearmanr(D.ln,D[col])
        ax.text(x[-1]+0.07,y[-1],f"$\\rho={r:+.3f}$",fontsize=7.4,color=c,va="center")
    ax.set_xlabel(r"$\log_{10}$ acute records per chemical")
    ax.set_ylabel("$\\log_{10}$ acute value (mg L$^{-1}$)")
    ax.legend(frameon=False,fontsize=7.6,loc="lower left",handlelength=1.6)
    q=pd.qcut(D.ln,NBIN_REQ,duplicates="drop")
    xmax=D.groupby(q,observed=True).ln.mean().max()
    ax.set_xlim(D.ln.min()-0.05,xmax+0.33)
    ax.set_title(f"A   the minimum falls with effort; the median barely moves "
                 f"($n$ = {n_all:,})",loc="left",fontsize=9,color=INK,pad=6)

def panelB(ax,ndraw=200):
    ks=[3,5,10,20]; base={}; eq={}
    for k in ks:
        cas=[c for c,v in byc.items() if len(v)>=k]
        ln=np.array([np.log10(len(byc[c])) for c in cas])
        bmn=np.array([byc[c].min() for c in cas])
        bmd=np.array([np.median(byc[c]) for c in cas])
        base[k]=(stats.spearmanr(ln,bmn)[0],stats.spearmanr(ln,bmd)[0],len(cas))
        # F-MEDX: the equalised minimum is the exact expectation of the
        # minimum of k drawn without replacement, not a Monte Carlo average.
        # The median has no comparable closed form and stays averaged.
        emn=np.array([mech.exp_min_wor(byc[c],k) for c in cas])
        emd=np.zeros(len(cas))
        for _ in range(ndraw):
            for i,c in enumerate(cas):
                emd[i]+=np.median(RNG.choice(byc[c],k,replace=False))
        emd/=ndraw
        eq[k]=(stats.spearmanr(ln,emn)[0],stats.spearmanr(ln,emd)[0])
    x=np.arange(len(ks)); w=0.34
    for j,(lab,c,idx) in enumerate((("minimum",WARM,0),("median",GREY,1))):
        bb=[base[k][idx] for k in ks]; ee=[eq[k][idx] for k in ks]
        off=(j-0.5)*w
        ax.bar(x+off,bb,w*0.46,color=c,alpha=.30,
               label=f"{lab}, full depth",zorder=2)
        ax.bar(x+off+w*0.46,ee,w*0.46,color=c,alpha=.95,
               label=f"{lab}, depth equalised",zorder=2)
        for xi,b_,e_ in zip(x,bb,ee):
            ax.annotate(f"{e_-b_:+.2f}",xy=(xi+off+w*0.23,min(b_,e_)-0.012),
                        fontsize=6.4,color=c,ha="center",va="top")
    ax.axhline(0,color=INK,lw=.7)
    ax.set_xticks(x); ax.set_xticklabels([f"$k$ = {k}\n$n$ = {base[k][2]:,}" for k in ks],
                                         fontsize=7.6)
    ax.set_ylabel(r"Spearman $\rho$ with record count")
    ax.legend(frameon=False,fontsize=6.8,loc="upper center",ncol=2,
              bbox_to_anchor=(0.5,-0.22),
              handlelength=1.1,columnspacing=1.1,labelspacing=.3)
    ax.set_title("B   equalising sampling depth removes most of the minimum's "
                 "dependence,\n      and nothing of the median's",
                 loc="left",fontsize=9,color=INK,pad=6)
    ax.set_ylim(min(min(base[k][0] for k in ks),
                    min(eq[k][0] for k in ks))*1.30,0.05)

if __name__=="__main__":
    fig,axes=plt.subplots(2,1,figsize=(7.1,6.6))
    panelA(axes[0]); panelB(axes[1])
    fig.subplots_adjust(hspace=.62)
    print(f"panel A quantile bins realised: {NBIN} (of {NBIN_REQ} requested)")
    fig.savefig(OUT + "/fig_05_broad.png",dpi=300,
                bbox_inches="tight",facecolor="white")
    print("wrote fig_05_broad.png")
