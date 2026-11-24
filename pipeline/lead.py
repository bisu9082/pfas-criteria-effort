"""
v3 lead: the paper's lead result, on the cleaned full-pull water frame.

Co-occurring PFAS -- the same substances, measured in the same samples at
the same stations by the same methods, at indistinguishable concentrations
-- receive criteria orders of magnitude apart, and for several of them no
criterion is derivable at all.  The exceedance verdict follows the
evidence base behind the criterion, not the water.

Note on thresholds: v2's water frame (m_ladder.pkl) retains QC records and
unmappable units, which inflates station counts (PFOA 1,013 stations there
against 872 here).  A ">=800 stations" cut tuned on that frame is an
artefact of it, so the block below is reported with a threshold sweep.
"""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import core, water


def criterion_ngL(row):
    """C6' criterion in ng/L, or NaN when no PNEC is derivable (F-DRV).

    The factor is AF_C6_prime, assessed on the same screened NOEC/NOEL pool as
    the endpoint (F-AF1), not AF_tgd, which is assessed on the wider pool used
    for the derivability question."""
    if row.route == "none" or pd.isna(row.AF_C6_prime) or pd.isna(row.C6_prime):
        return np.nan
    return (10 ** row.C6_prime) * 1e6 / row.AF_C6_prime


def block(det, out, nm, min_st, verbose=True):
    st = det.groupby("parent").MonitoringLocationIdentifier.nunique()
    subs = st[st >= min_st].index.tolist()
    if len(subs) < 2:
        return None
    sets = [set(det.loc[det.parent == p, "MonitoringLocationIdentifier"])
            for p in subs]
    shared = set.intersection(*sets)
    sub = det[det.parent.isin(subs) &
              det.MonitoringLocationIdentifier.isin(shared)]
    rows = []
    for p in subs:
        s = sub[sub.parent == p]
        r = out[out.parent == p].iloc[0]
        c = criterion_ngL(r)
        rows.append(dict(name=str(nm.get(p, p))[:30], n_det=len(s),
                         med=s.ngL.median(), n_rec=int(r.n_records),
                         chron_sp=int(r.chronic_sp or 0), route=r.route,
                         crit_ngL=c,
                         exceed=100 * (s.ngL > c).mean() if pd.notna(c) else np.nan))
    return pd.DataFrame(rows).sort_values("med", ascending=False), shared, sub


def run(a, out, rule, min_st=700):
    det = a[a.detected]
    nm = a[["parent", "name"]].drop_duplicates("parent").set_index("parent").name
    R, shared, sub = block(det, out, nm, min_st)

    print("=" * 98)
    print(f"LEAD RESULT  |  duration rule = {rule}  |  threshold >= {min_st} stations")
    print("=" * 98)
    print(f"  co-occurring substances                : {len(R)}")
    print(f"  stations measuring all of them         : {len(shared):,}")
    print(f"  detections in that block               : {len(sub):,} "
          f"({100*len(sub)/len(det):.1f}% of all {len(det):,} detections)")
    print()
    print(R.to_string(index=False, float_format=lambda x: f"{x:,.3g}"))

    fin = R.dropna(subset=["crit_ngL"])
    none = R[R.route == "none"]
    print(f"\n  measured median concentration : {R.med.min():.2f} - {R.med.max():.2f} ng/L"
          f"   ({R.med.max()/R.med.min():.1f}-fold)")
    print(f"  apical chronic species        : {R.chron_sp.min()} - {R.chron_sp.max()}")
    print(f"  ECOTOX records                : {R.n_rec.min():,} - {R.n_rec.max():,}"
          f"   ({R.n_rec.max()/R.n_rec.min():.0f}-fold)")
    if len(fin) > 1:
        print(f"  C6' criterion                 : {fin.crit_ngL.min():,.3g} - "
              f"{fin.crit_ngL.max():,.3g} ng/L   "
              f"({fin.crit_ngL.max()/fin.crit_ngL.min():,.0f}-fold)")
        print(f"  detections in exceedance      : {fin.exceed.max():.1f}% -> "
              f"{fin.exceed.min():.1f}%")
    if len(none):
        print(f"  NO PNEC derivable             : {len(none)} of {len(R)} "
              f"({', '.join(none.name)}), {none.n_det.sum():,} detections "
              f"= {100*none.n_det.sum()/len(sub):.1f}% of the block")

    # threshold sweep
    print("\n  -- threshold sweep --")
    print(f"     {'min_st':>7} {'subs':>5} {'stations':>9} {'detections':>11} "
          f"{'conc fold':>10} {'crit fold':>12} {'exceed span':>14}")
    for m in (400, 500, 600, 700, 750, 800):
        b = block(det, out, nm, m, verbose=False)
        if b is None:
            continue
        Rb, sh, sb = b
        f = Rb.dropna(subset=["crit_ngL"])
        cf = f"{f.crit_ngL.max()/f.crit_ngL.min():,.0f}x" if len(f) > 1 else "-"
        ex = f"{f.exceed.max():.1f}->{f.exceed.min():.1f}%" if len(f) > 1 else "-"
        print(f"     {m:>7} {len(Rb):>5} {len(sh):>9,} {len(sb):>11,} "
              f"{Rb.med.max()/Rb.med.min():>9.1f}x {cf:>12} {ex:>14}")
    return R, shared, sub


if __name__ == "__main__":
    a = water.load(verbose=False)
    for rule in ("r10", "operational"):
        out, _, _ = core.build(rule, verbose=False)
        run(a, out, rule)
        print()
