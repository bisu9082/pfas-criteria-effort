"""v3 exceedance ladder, on the common subset, with detection status explicit."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import core, water

AF_REF = 1000.0   # acute base-set factor, used for C1-C5


def criteria_ngL(out, d):
    """ng/L criterion per parent for each convention."""
    lo = d.groupby("parent").logTox.min().rename("C1")
    noLOEC = d[~d.ep.isin(["LOEC","LOEL"])].groupby("parent").logTox.min().rename("C2")
    ac = d[d.ep.isin(core.ACUTE)]
    C3 = ac.groupby("parent").logTox.min().rename("C3")
    C4 = ac.groupby("parent").logTox.median().rename("C4")
    C5 = (ac.groupby(["parent","latin_name"]).logTox.median()
            .groupby("parent").median().rename("C5"))
    C = pd.concat([lo,noLOEC,C3,C4,C5],axis=1).reset_index()
    for c in ("C1","C2","C3","C4","C5"):
        C[c] = (10**C[c])*1e6/AF_REF
    o = out.set_index("parent")
    C["C6"]  = [(10**o.C6_ap[p])*1e6/o.AF_C6_ap[p]
                if p in o.index and pd.notna(o.C6_ap.get(p)) and pd.notna(o.AF_C6_ap.get(p))
                else np.nan for p in C.parent]
    # F-AF1: the factor must come from the SAME screened pool as the endpoint
    # it divides.  AF_tgd is assessed on the wider chronic pool (it admits MATC
    # and EC10 for the derivability question) and differs for one or two
    # substances; using it here would make Table 1 disagree with the text.
    C["C6p"] = [(10**o.C6_prime[p])*1e6/o.AF_C6_prime[p]
                if p in o.index and o.route.get(p)=="chronic"
                   and pd.notna(o.C6_prime.get(p)) and pd.notna(o.AF_C6_prime.get(p))
                else np.nan for p in C.parent]
    C["C7b"] = [(10**o.HC5_d[p])*1e6 if p in o.index and pd.notna(o.HC5_d.get(p))
                else np.nan for p in C.parent]
    return C


def run(rule="r10"):
    out, d, _ = core.build(rule, verbose=False)
    a = water.load(verbose=False); det = a[a.detected]
    nm = a[["parent","name"]].drop_duplicates("parent").set_index("parent").name
    C = criteria_ngL(out, d)
    cols = ["C1","C2","C3","C4","C5","C6","C6p","C7b"]

    ci0 = C.set_index("parent")
    meas = {p for p in det.parent.dropna().unique() if p in ci0.index}
    common = [p for p in meas if ci0.loc[p, cols].notna().all()]
    sub = det[det.parent.isin(common)]
    print("="*92); print(f"EXCEEDANCE LADDER  |  rule = {rule}"); print("="*92)
    print(f"  substances every convention can assess : {len(common)}")
    print(f"  detections in that common subset       : {len(sub):,}")
    print(f"  stations                               : "
          f"{sub.MonitoringLocationIdentifier.nunique():,}")
    print(f"  [{', '.join(sorted(str(nm.get(p,p))[:26] for p in common))}]")
    ci = ci0
    print(f"\n  {'conv':>5} {'median crit ng/L':>18} {'% detections above':>20} "
          f"{'% stations w/ any':>18}")
    res={}
    for c in cols:
        thr = sub.parent.map(ci[c])
        ex = sub.ngL > thr
        st = sub[ex].MonitoringLocationIdentifier.nunique()
        res[c]=(ci.loc[common,c].median(), 100*ex.mean(),
                100*st/sub.MonitoringLocationIdentifier.nunique())
        print(f"  {c:>5} {res[c][0]:>18,.3g} {res[c][1]:>20.1f} {res[c][2]:>18.1f}")
    lo,hi = min(r[1] for r in res.values()), max(r[1] for r in res.values())
    mlo,mhi = min(r[0] for r in res.values()), max(r[0] for r in res.values())
    print(f"\n  exceedance spans {hi:.1f}% -> {lo:.1f}% on derivation choice alone")
    print(f"  median criterion spans {np.log10(mhi/mlo):.1f} orders of magnitude")
    print(f"  the two routes agencies use: C6' {res['C6p'][0]:,.3g} ng/L "
          f"({res['C6p'][1]:.1f}% of detections, {res['C6p'][2]:.1f}% of stations) "
          f"vs C7b {res['C7b'][0]:,.3g} ng/L ({res['C7b'][1]:.1f}%, {res['C7b'][2]:.1f}%)")
    print(f"     unscreened C6 for comparison: {res['C6'][1]:.1f}% of detections")
    print(f"     C1 -> C2 (dropping LOEC): {res['C1'][1]:.1f}% -> {res['C2'][1]:.1f}%")
    return res, common, sub


if __name__ == "__main__":
    for r in ("r10","operational"):
        run(r); print()
