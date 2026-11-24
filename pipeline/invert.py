"""
v3 invert: does complying with the guidance make criteria MORE or LESS
protective?

A stricter chronic-duration screen discards studies.  Discarding studies
shortens the species list.  A shorter species list truncates the estimated
spread of the sensitivity distribution and raises the fitted lower
percentile, and it also removes trophic levels, which lowers the
assessment factor.  Both effects push the criterion UP, i.e. make it less
protective.  This module measures that on the substances themselves.
"""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import core, water


def compare(verbose=True):
    o10, _, _ = core.build("r10", verbose=False)
    oop, _, _ = core.build("operational", verbose=False)
    k = ["parent", "n_records", "C6_prime", "AF_C6_prime", "AF_tgd", "route",
         "chronic_tl", "chronic_sp", "n_sp_d", "sd_d", "mu_d", "HC5_d"]
    m = o10[k].merge(oop[k], on="parent", suffixes=("_r10", "_op"))

    def crit(c6, af):
        return np.where(pd.notna(c6) & pd.notna(af), (10 ** c6) * 1e6 / af, np.nan)

    m["crit_r10"] = crit(m.C6_prime_r10, m.AF_tgd_r10)
    m["crit_op"] = crit(m.C6_prime_op, m.AF_tgd_op)
    both = m.dropna(subset=["crit_r10", "crit_op"]).copy()
    both["lr"] = np.log10(both.crit_r10 / both.crit_op)

    a = water.load(verbose=False)
    det = a[a.detected]
    nm = a[["parent", "name"]].drop_duplicates("parent").set_index("parent").name
    ndet = det.groupby("parent").size()

    if verbose:
        print("=" * 96)
        print("Does complying with ECHA R.10 make the criterion more protective?")
        print("=" * 96)
        print(f"  substances with a derivable C6' criterion under BOTH rules: {len(both)}")
        up = (both.lr > 0.01).sum(); dn = (both.lr < -0.01).sum()
        print(f"    criterion RAISED by compliance (less protective): {up}")
        print(f"    criterion LOWERED by compliance (more protective): {dn}")
        print(f"    unchanged: {len(both)-up-dn}")
        print(f"  median log10 change: {both.lr.median():+.3f} "
              f"({10**both.lr.median():.2f}x)")
        w = stats.wilcoxon(both.lr) if len(both) > 5 else None
        if w is not None:
            print(f"  Wilcoxon signed-rank on log10 ratio: "
                  f"W={w.statistic:.0f}, p={w.pvalue:.4g}")

        print("\n  -- substances that are monitored, ranked by detections --")
        mm = both.copy()
        mm["name"] = [str(nm.get(p, p))[:30] for p in mm.parent]
        mm["n_det"] = [int(ndet.get(p, 0)) for p in mm.parent]
        mm = mm[mm.n_det > 0].sort_values("n_det", ascending=False)
        show = mm[["name", "n_det", "n_sp_d_op", "n_sp_d_r10",
                   "AF_tgd_op", "AF_tgd_r10", "crit_op", "crit_r10", "lr"]]
        print(show.rename(columns={
            "n_sp_d_op": "sp_oper", "n_sp_d_r10": "sp_R10",
            "AF_tgd_op": "AF_oper", "AF_tgd_r10": "AF_R10",
            "crit_op": "crit_oper", "crit_r10": "crit_R10",
            "lr": "log10_ratio"}).to_string(
            index=False, float_format=lambda x: f"{x:,.3g}"))

        print("\n  -- substances that LOSE a derivable criterion under R.10 --")
        lost = m[(m.route_op != "none") & (m.route_r10 == "none")].copy()
        lost["name"] = [str(nm.get(p, p))[:30] for p in lost.parent]
        lost["n_det"] = [int(ndet.get(p, 0)) for p in lost.parent]
        lost = lost.sort_values("n_det", ascending=False)
        print(f"     {len(lost)} substances; "
              f"{int(lost.n_det.sum()):,} detections "
              f"({100*lost.n_det.sum()/len(det):.1f}% of all detections)")
        print(lost[lost.n_det > 0][["name", "n_det", "n_records_op",
                                    "chronic_sp_op", "chronic_sp_r10"]]
              .rename(columns={"n_records_op": "n_rec",
                               "chronic_sp_op": "sp_oper",
                               "chronic_sp_r10": "sp_R10"})
              .to_string(index=False))

        print("\n  -- the species-list truncation, on substances with an SSD "
              "under both rules --")
        s = m.dropna(subset=["HC5_d_r10", "HC5_d_op"]).copy()
        if len(s):
            s["name"] = [str(nm.get(p, p))[:30] for p in s.parent]
            s["dHC5"] = s.HC5_d_r10 - s.HC5_d_op
            s["dsd"] = s.sd_d_r10 - s.sd_d_op
            print(s[["name", "n_sp_d_op", "n_sp_d_r10", "sd_d_op", "sd_d_r10",
                     "HC5_d_op", "HC5_d_r10", "dsd", "dHC5"]]
                  .rename(columns={"n_sp_d_op": "sp_oper", "n_sp_d_r10": "sp_R10"})
                  .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
            print(f"\n     species lost: median "
                  f"{(s.n_sp_d_op - s.n_sp_d_r10).median():.0f}")
            print(f"     change in fitted sd : median {s.dsd.median():+.3f} log10")
            print(f"     change in HC5       : median {s.dHC5.median():+.3f} log10"
                  f"  ({10**s.dHC5.median():.2f}x)")
    return m, both


if __name__ == "__main__":
    compare()
