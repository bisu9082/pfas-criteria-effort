"""
v3 robustness: the checks bearing on alternative explanations.

Replaces v2/robust.py, which (a) omitted the AF = 1000 divisor on C1, so every
exceedance figure it produced was wrong, (b) read the v2 ladder with the
assessment factor taken from the unscreened pool, and (c) worked on the
detections-only water frame with the QC records still in it.
"""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import core, water, ladder, mech

AF_REF=1000.0
CS=[("C6p","C6$'$ EU-TGD apical + duration"),
    ("C7a","C7a lowest apical SMCV"),
    ("C7b","C7b SSD fifth percentile")]


def crit_cols(rule):
    out,d,_=core.build(rule,verbose=False)
    C=ladder.criteria_ngL(out,d).set_index("parent")
    o=out.set_index("parent")
    C["C7a"]=[(10**o.minSMCV_d[p])*1e6/10.0 if pd.notna(o.minSMCV_d.get(p)) else np.nan
              for p in C.index]
    C["n_records"]=[o.n_records.get(p,np.nan) for p in C.index]
    C["n_tests"]=[o.n_tests.get(p,np.nan) for p in C.index]
    return C,out,d


def boot_rho(x,y,B=5000,seed=1):
    r=np.random.default_rng(seed); n=len(x)
    x=np.asarray(x); y=np.asarray(y); acc=[]
    for _ in range(B):
        j=r.integers(0,n,n)
        if len(set(x[j]))<3: continue
        acc.append(stats.spearmanr(x[j],y[j])[0])
    return np.nanpercentile(acc,[2.5,97.5])


def run(rule="operational"):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    C,out,d=crit_cols(rule)
    def nC(s):
        m=Chem.MolFromSmiles(str(s))
        return sum(1 for a in m.GetAtoms() if a.GetSymbol()=="C") if m else np.nan
    C["nC"]=[nC(p) for p in C.index]
    lg=lambda s: np.log10(s[core.EFFORT])

    w=94
    print("="*w); print(f"ROBUSTNESS  |  duration rule = {rule}"); print("="*w)
    rows={}
    print("\n[1] as reported, and the chain-length confound")
    for c,nm in CS:
        s=C.dropna(subset=[c,"nC",core.EFFORT])
        r=stats.spearmanr(lg(s),s[c])[0]
        p=mech.rank_partial(lg(s),s[c],s.nC)[0]
        rnc=stats.spearmanr(lg(s),s.nC)[0]
        rows[c]={"raw":r,"nC":p,"rho_nC":rnc,"n":len(s)}
        print(f"  {nm:34s} rho={r:+.3f}  partial|nC={p:+.3f}  "
              f"(rho(log n, nC)={rnc:+.2f}, n={len(s)})")

    print("\n[2] alternative effort variables")
    KEY=["test_id","ep","conc1_mean","conc1_unit"]
    ndedup=d.drop_duplicates(subset=KEY).groupby("parent").size()
    C["n_dedup"]=[ndedup.get(p,np.nan) for p in C.index]
    for c,nm in CS:
        s=C.dropna(subset=[c,core.EFFORT_ALT,"n_dedup"])
        rr=stats.spearmanr(np.log10(s[core.EFFORT_ALT]),s[c])[0]
        rd=stats.spearmanr(np.log10(s.n_dedup),s[c])[0]
        # 'rows' is now the declared sensitivity; 'raw' is the primary (tests)
        rows[c]["rows"]=rr; rows[c]["tests"]=rows[c]["raw"]; rows[c]["dedup"]=rd
        print(f"  {nm:34s} tests={rows[c]['raw']:+.3f}  rows={rr:+.3f}  "
              f"collapsed={rd:+.3f}")

    print("\n[3] dropping the three non-OECD-PFAS parents")
    bad=["O=Cc1c(F)c(F)c(F)c(F)c1F","Nc1c(F)c(F)c(F)c(F)c1F","O=C(O)C(F)F"]
    for c,nm in CS:
        s=C.dropna(subset=[c]); s2=s[~s.index.isin(bad)]
        r2=stats.spearmanr(lg(s2),s2[c])[0]
        rows[c]["pfas_only"]=r2
        # The three non-OECD parents are already removed by the apical and
        # duration screens, so this row drops nothing on the screened pools.
        # Record that, so the table can say so instead of printing a row
        # identical to the baseline under a caption calling it a check.
        rows[c]["pfas_drop"]=len(s)-len(s2)
        print(f"  {nm:34s} all={rows[c]['raw']:+.3f}  PFAS-only={r2:+.3f}  "
              f"(n {len(s)} -> {len(s2)})")

    print("\n[4] bootstrap 95% CI, fixed seed, 5000 draws")
    for i,(c,nm) in enumerate(CS):
        s=C.dropna(subset=[c])
        lo,hi=boot_rho(lg(s),s[c],seed=100+i)
        rows[c]["ci"]=(lo,hi)
        print(f"  {nm:34s} rho={rows[c]['raw']:+.3f}  CI=[{lo:+.2f}, {hi:+.2f}]  n={len(s)}")

    print("\n[5] permutation null for the SSD set")
    s=C.dropna(subset=["C7b"]); n=len(s)
    if n>=5:
        r0=np.random.default_rng(9)
        nl=np.array([stats.spearmanr(lg(s),r0.permutation(s.C7b.values))[0]
                     for _ in range(20000)])
        obs=stats.spearmanr(lg(s),s.C7b)[0]
        print(f"  n={n}  observed {obs:+.3f};  null mean {nl.mean():+.3f} sd {nl.std():.3f};"
              f"  95% band [{np.percentile(nl,2.5):+.2f}, {np.percentile(nl,97.5):+.2f}];"
              f"  P={np.mean(nl<=obs):.4f}")
        rows["perm"]=(n,obs,nl.mean(),nl.std(),np.percentile(nl,2.5),
                      np.percentile(nl,97.5),float(np.mean(nl<=obs)))

    # ---- monitoring side, with the AF = 1000 divisor restored (the v2 bug)
    print("\n[6] C1 exceedance: leave one reporting organisation out")
    a=water.load(verbose=False); det=a[a.detected]
    lo_=d.groupby("parent").logTox.min()
    pm=(10**lo_)*1e6/AF_REF                      # <-- v2 omitted / AF_REF
    pm_bug=(10**lo_)*1e6                         # what v2 actually computed
    x=det[det.parent.isin(pm.index)].copy()
    x["RQ"]=x.ngL/x.parent.map(pm)
    x["RQ_bug"]=x.ngL/x.parent.map(pm_bug)
    full=100*(x.RQ>1).mean(); full_bug=100*(x.RQ_bug>1).mean()
    o="OrganizationFormalName"
    top=x[o].value_counts().head(8)
    vals=[100*(x[x[o]!=k].RQ>1).mean() for k in top.index]
    stn=x.groupby(["MonitoringLocationIdentifier","parent"]).RQ.median()
    allres=det  # detections; the all-results denominator:
    xa=a[a.parent.isin(pm.index)].copy()
    xa["RQ"]=np.where(xa.detected, xa.ngL/xa.parent.map(pm), np.nan)
    allres_frac=100*(xa.RQ>1).sum()/len(xa)
    print(f"  full set, detections basis            {full:.2f}%")
    print(f"     (v2, with the AF divisor omitted:  {full_bug:.2f}%)")
    print(f"  leave-one-organisation-out, 8 largest [{min(vals):.2f}, {max(vals):.2f}]")
    print(f"  largest contributor supplies          {100*top.iloc[0]/len(x):.1f}% of detections")
    print(f"  one median per station x substance    {100*(stn>1).mean():.2f}%")
    print(f"  over ALL retrieved results            {allres_frac:.2f}%  "
          f"(non-detections counted as not exceeding)")
    # F-NOISE: measurement-grid sensitivity.  The SI quoted this on the v2
    # base of 76.50%; it had no generator.  200 replicates of independent
    # multiplicative noise on the measured value.
    _rng=np.random.default_rng(20261122)
    _thr=x.parent.map(pm).values; _v=x.ngL.values
    _f=np.array([100*np.mean(_v*(1+_rng.uniform(-0.10,0.10,_v.size))>_thr)
                 for _ in range(200)])
    _lo,_hi=np.percentile(_f,[2.5,97.5])
    print(f"  +/-10% multiplicative noise, 200 reps  mean {_f.mean():.2f}%"
          f"  95% range [{_lo:.2f}, {_hi:.2f}]")
    rows["water"]=dict(full=full,full_bug=full_bug,lo=min(vals),hi=max(vals),
                       top=100*top.iloc[0]/len(x),station=100*(stn>1).mean(),
                       allres=allres_frac,
                       noise=_f.mean(),noise_lo=_lo,noise_hi=_hi)
    return rows


if __name__=="__main__":
    for r in ("operational","r10"):
        run(r); print()
