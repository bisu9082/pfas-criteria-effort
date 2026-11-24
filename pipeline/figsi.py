"""Supplementary figures: composition of the extract, and SSD normality."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import core

plt.rcParams.update({"font.size":9,"axes.linewidth":.8,
                     "axes.spines.top":False,"axes.spines.right":False})
INK,ACC,WARM,GREY="#1a1a1a","#1f5fa8","#c0392b","#8a8a8a"
EFFECT={"GEN":"gene expression","BCM":"biochemistry","MOR":"mortality",
        "GRO":"growth","ENZ":"enzyme activity","POP":"population",
        "REP":"reproduction","HRM":"hormone","PHY":"physiology",
        "CEL":"cellular","HIS":"histology","ACC":"accumulation",
        "BEH":"behaviour","DVP":"development","MPH":"morphology",
        "IMM":"immune","GRO/":"growth"}
APICAL={"MOR","GRO","REP","POP","DVP"}


def data_fig(d):
    fig=plt.figure(figsize=(7.1,7.0))
    gs=fig.add_gridspec(3,1,height_ratios=[1,1.25,1.05],hspace=.62)

    # A endpoint types
    ax=fig.add_subplot(gs[0])
    ep=d.ep.astype(str).value_counts().head(9)[::-1]
    cols=[WARM if k in ("LOEC","LOEL") else (ACC if k in ("NOEC","NOEL") else GREY)
          for k in ep.index]
    ax.barh(range(len(ep)),ep.values,color=cols,height=.68)
    ax.set_yticks(range(len(ep))); ax.set_yticklabels(ep.index,fontsize=8)
    for i,v in enumerate(ep.values):
        ax.text(v*1.02,i,f"{v:,}",va="center",fontsize=7,color=INK)
    ax.set_xlim(0,ep.max()*1.16); ax.set_xlabel("result rows")
    ax.tick_params(axis="y",length=0)
    ax.set_title(f"A   endpoint types  (red: {int((d.ep=='LOEC').sum()):,} "
                 f"\\textsf{{LOEC}} rows report a concentration at which an "
                 f"effect was seen)".replace("\\textsf{LOEC}","LOEC"),
                 loc="left",fontsize=9,color=INK,pad=6)
    ax.spines["left"].set_visible(False)

    # B effect types
    ax=fig.add_subplot(gs[1])
    ef=d.effect.astype(str).str.strip().value_counts().head(12)[::-1]
    lab=[EFFECT.get(k,k) for k in ef.index]
    cols=[ACC if k.rstrip("/") in APICAL else GREY for k in ef.index]
    ax.barh(range(len(ef)),ef.values,color=cols,height=.68)
    ax.set_yticks(range(len(ef))); ax.set_yticklabels(lab,fontsize=8)
    for i,v in enumerate(ef.values):
        ax.text(v*1.02,i,f"{v:,}",va="center",fontsize=7,color=INK)
    ax.set_xlim(0,ef.max()*1.16); ax.set_xlabel("result rows")
    ax.tick_params(axis="y",length=0)
    ax.set_title(f"B   reported effects  (blue: apical, "
                 f"{100*d.apical.mean():.1f}\\% of all rows)".replace("\\%","%"),
                 loc="left",fontsize=9,color=INK,pad=6)
    ax.spines["left"].set_visible(False)

    # C records per substance
    ax=fig.add_subplot(gs[2])
    n=d.groupby("parent").size().sort_values(ascending=False)
    ax.plot(np.arange(1,len(n)+1),n.values,"o-",color=INK,ms=3,lw=1.1)
    ax.set_yscale("log")
    med=n.median(); top2=100*n.head(2).sum()/n.sum()
    ax.axhline(med,color=WARM,lw=.9,ls=(0,(4,3)))
    ax.text(len(n)*0.52,med*1.18,f"median {med:.0f} rows",
            fontsize=7.4,color=WARM)
    ax.annotate(f"the two most-studied PFAS\nhold {top2:.0f}\\% of all rows".replace("\\%","%"),
                xy=(1.6,n.iloc[1]),xytext=(7,n.iloc[1]*0.30),fontsize=7.4,color=INK,
                arrowprops=dict(arrowstyle="->",color=INK,lw=.8))
    ax.set_xlabel("substance, rank-ordered"); ax.set_ylabel("result rows")
    ax.set_title(f"C   {len(n)} substances, {len(d):,} result rows",
                 loc="left",fontsize=9,color=INK,pad=6)
    fig.savefig(OUT + "/fig_si_data.png",dpi=300,
                bbox_inches="tight",facecolor="white")
    print(f"wrote fig_si_data.png  ({len(d):,} rows, {len(n)} substances, "
          f"median {med:.0f}, top2 {top2:.1f}%)")


def _fmtcas(v):
    """ECOTOX stores CAS without punctuation; print the hyphenated form."""
    t="".join(ch for ch in str(v) if ch.isdigit())
    return f"{t[:-3]}-{t[-3:-1]}-{t[-1]}" if len(t) >= 5 else str(v)


def qq_fig(rule="operational"):
    out,d,_=core.build(rule,verbose=False)
    base=d[d.conc1_type=="A"]
    sel=base[base.ep.isin(core.CHRONIC_EP) & base.apical & base.chronic_ok]
    sm=sel.groupby(["parent","latin_name"]).logTox.mean().reset_index()
    cnt=sm.groupby("parent").size()
    keep=cnt[cnt>=5].sort_values(ascending=False).index.tolist()
    _adrej=core.ssd_diagnostics(rule).set_index("parent").AD_reject5.to_dict()
    import water
    a=water.load(verbose=False)
    nm=a[["parent","name"]].drop_duplicates("parent").set_index("parent").name
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    # Panel titles are referred to by name in the SI prose, so a formula
    # fallback (C17H8Cl2F8N2O3 for the substance the text calls lufenuron)
    # left the reader with no bridge between text and figure.  Parents with no
    # name in the extract are named from their CAS number instead, and long
    # names are abbreviated only at a word boundary.
    _cas=(d.dropna(subset=["parent"]).groupby("parent").test_cas
            .agg(lambda v: v.dropna().iloc[0] if v.notna().any() else None))
    _BY_CAS={"103055-07-8":"Lufenuron","76-05-1":"Trifluoroacetic acid"}

    def lbl(p):
        s=nm.get(p)
        if not isinstance(s,str):
            c=_cas.get(p)
            c=str(int(c)) if isinstance(c,float) and c==c else (
                str(c) if c is not None else None)
            s=_BY_CAS.get(_fmtcas(c)) if c else None
        if isinstance(s,str):
            if len(s)<=26:
                return s
            return s[:26].rsplit(" ",1)[0]+"\u2026"
        m=Chem.MolFromSmiles(str(p))
        return CalcMolFormula(m) if m else str(p)[:22]
    ncol=4; nrow=int(np.ceil(len(keep)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(7.1,1.75*nrow))
    axes=np.atleast_1d(axes).ravel()
    for ax,p in zip(axes,keep):
        v=np.sort(sm[sm.parent==p].logTox.values); n=len(v)
        q=stats.norm.ppf((np.arange(1,n+1)-.5)/n)
        ax.plot(q,v,"o",ms=3.2,color=ACC,mec="white",mew=.5)
        b=np.polyfit(q,v,1); xs=np.array([q.min(),q.max()])
        ax.plot(xs,np.polyval(b,xs),"-",color=GREY,lw=.9)
        # Marked cases must be the Anderson-Darling rejections, which is what
        # the SI prose and si_tab_fit's own columns argue from; keying the red
        # titles to a Shapiro-Wilk p marked a different set of four.
        a2=float(stats.anderson(v,dist="norm").statistic) if n>=3 else np.nan
        rj=bool(_adrej.get(p,False))
        ax.set_title(f"{lbl(p)}\n$n$ = {n},  $A^2$ = {a2:.2f}",fontsize=6.6,
                     color=WARM if rj else INK,pad=3)
        ax.tick_params(labelsize=6)
    for ax in axes[len(keep):]: ax.axis("off")
    fig.supxlabel("normal quantile",fontsize=8)
    fig.supylabel(r"species mean chronic value, $\log_{10}$",fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + "/fig_si_qq.png",dpi=300,
                bbox_inches="tight",facecolor="white")
    rej=[lbl(p) for p in keep if bool(_adrej.get(p,False))]
    print(f"wrote fig_si_qq.png  ({len(keep)} substances under '{rule}', "
          f"normality rejected for {len(rej)}: {rej})")


if __name__=="__main__":
    d=core.load("r10")
    data_fig(d)
    qq_fig("operational")
