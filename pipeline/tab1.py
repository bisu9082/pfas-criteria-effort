"""Table 1: derived criteria and exceedance under the conventions, from v3."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import core, water, ladder

# F-REGEN: Table 1 of the main text.  The output path was hardcoded, so a
# verifier could not regenerate it into a scratch directory and the paper's
# central table sat outside byte verification entirely.
OUT = os.environ.get("SITABS_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables"))

AF_REF=1000.0
ROWS=[("C1","lowest value, any endpoint","C1"),
      ("C3","lowest acute value","C3"),
      ("C5","median acute, by species","C5"),
      ("C6","lowest chronic NOEC / AF","C6"),
      ("C6$'$","as C6, apical, chronic","C6p"),
      ("C7a","lowest apical SMCV / 10","C7a"),
      ("C7b","SSD fifth percentile","C7b")]

def crit_table(rule):
    out,d,_=core.build(rule,verbose=False)
    C=ladder.criteria_ngL(out,d).set_index("parent")
    o=out.set_index("parent")
    C["C7a"]=[(10**o.minSMCV_d[p])*1e6/10.0 if pd.notna(o.minSMCV_d.get(p)) else np.nan
              for p in C.index]
    return C,out

def fmt(x):
    if pd.isna(x): return "---"
    if x>=1e4 or x<1e-2: return rf"\num{{{x:.2e}}}"
    if x>=1000: return f"{x:,.0f}".replace(",",r"\,")
    return f"{x:.3g}"

def build():
    a=water.load(verbose=False); det=a[a.detected]
    nst=det.MonitoringLocationIdentifier.nunique()   # stations with ANY detection
    # LEFT: own substance set, under R.10 (the primary rule)
    C10,out10=crit_table("r10")
    meas=[p for p in det.parent.dropna().unique() if p in C10.index]
    left={}
    for lab,desc,col in ROWS:
        sub=[p for p in meas if pd.notna(C10.loc[p,col])]
        if not sub: left[col]=(0,np.nan,np.nan,0); continue
        dd=det[det.parent.isin(sub)]
        thr=dd.parent.map(C10[col])
        left[col]=(len(sub), C10.loc[sub,col].median(), 100*(dd.ngL>thr).mean(),
                   len(dd))
    # RIGHT: common subset, operational rule (R.10 leaves only two substances)
    Cop,outop=crit_table("operational")
    cols=[c for _,_,c in ROWS]
    measop=[p for p in det.parent.dropna().unique() if p in Cop.index]
    common=[p for p in measop if Cop.loc[p,cols].notna().all()]
    sub=det[det.parent.isin(common)]
    right={}
    for lab,desc,col in ROWS:
        thr=sub.parent.map(Cop[col]); ex=sub.ngL>thr
        right[col]=(Cop.loc[common,col].median(), 100*ex.mean(),
                    100*sub[ex].MonitoringLocationIdentifier.nunique()
                    /sub.MonitoringLocationIdentifier.nunique())
    cap=(r"\caption{Derived criteria and exceedance. The left block applies each "
         r"convention, under ECHA R.10 durations, to every monitored substance it can "
         r"assess. \emph{Each row's percentage is over the detections of the substances "
         r"that convention can assess}, given as $n_{\mathrm{det}}$; the record holds "
         r"\num{%d} detections in all, at %s stations with any detection. The right "
         r"block restricts all conventions to the substances every one can assess: under "
         r"R.10 only two qualify, so it is given under the operational rule, where %d "
         r"substances and \num{%d} detections remain. Assessment factors come from the "
         r"same screened pool as the endpoint they divide; PNEC is the median across "
         r"substances. Column \emph{det.} is the percentage of detections above "
         r"the criterion and \emph{stat.} the percentage of stations with at "
         r"least one such detection.}"
         % (len(det), f"{nst:,}".replace(",",r"\,"), len(common), len(sub)))
    L=[r"\begin{table}[t]",r"\centering",r"\scriptsize",
       r"\setlength{\tabcolsep}{2.0pt}",cap,r"\label{tab:ladder}",
       r"\begin{tabular}{@{}ll rrrr rrr@{}}",r"\toprule",
       r"& & \multicolumn{4}{c}{own set, R.10} & "
       r"\multicolumn{3}{c}{common subset ($n=%d$), oper.}\\" % len(common),
       r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
       r"& convention & subs. & $n_{\mathrm{det}}$ & PNEC & det. & PNEC & det. & stat.\\",
       r"& & & & (ng/L) & (\%) & (ng/L) & (\%) & (\%)\\",r"\midrule"]
    for lab,desc,col in ROWS:
        n,p,s,nd=left[col]; pr,sr,st=right[col]
        L.append(f"{lab} & {desc} & {n} & {nd:,} & {fmt(p)} & {s:.2f} & "
                 f"{fmt(pr)} & {sr:.2f} & {st:.2f}\\\\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(os.path.join(OUT, "tab_ladder_v2.tex"), "w").write("\n".join(L)+"\n")
    print(f"left block: {len(meas)} monitored substances, {len(det):,} detections, {nst:,} stations")
    print(f"right block: {len(common)} common substances, {len(sub):,} detections")
    for lab,desc,col in ROWS:
        print(f"  {lab:6s} own n={left[col][0]:2d} ndet={left[col][3]:6,d} PNEC={left[col][1]:>11,.4g} det={left[col][2]:6.2f}%"
              f" | common PNEC={right[col][0]:>11,.4g} det={right[col][1]:6.2f}% stat={right[col][2]:6.2f}%")
    return common,sub

if __name__=="__main__":
    build()
