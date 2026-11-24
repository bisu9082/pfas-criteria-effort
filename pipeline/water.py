"""
v3 water: the monitoring side, rebuilt on the FULL WQP pull.

The v2 water frame (wqp/m_ladder.pkl, 20,682 rows) contains detections only:
every row is "detected" or "Present Above Quantification Limit", and there
are no non-detects in it at all.  The draft nevertheless calls those rows
"measurements" and computes exceedance fractions over them, so every such
fraction is in fact conditional on detection.  This module starts from the
full pull instead (wqp/all_raw.pkl, 117,034 results) and keeps detection
status explicit, so the denominator is whatever the sentence says it is.

Fixes:
  F-DEN  detection frequency and exceedance are computed against the full
         retrieved record, not against the detections-only subset.
  F-QC   % recovery results and identifiable laboratory / field duplicates
         and control spikes are removed.  v2 said they "cannot be
         excluded"; they can.
  F-GEO  the geographic concentration of the record is reported, so the
         dataset is not described as "United States surface water".
"""
import re, sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import core


# ---------------------------------------------------------------- data paths
# Resolved from $PFAS_DATA, defaulting to a "data" directory beside the code,
# so the deposited repository runs with no edit.  The working tree sets
# PFAS_DATA to its own location.  Both the pickle and the csv.gz form of each
# frame are accepted; the repository ships the csv.gz, which is the portable
# one.
import os as _os


def _data(*names):
    root = _os.environ.get(
        "PFAS_DATA",
        _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), "data"))
    for n in names:
        p = _os.path.join(root, n)
        if _os.path.exists(p):
            return p
    return _os.path.join(root, names[0])


def _read(path, **kw):
    """Read a deposited frame with the dtypes the pipeline was built on.

    A csv round-trip silently retypes identifier columns (test_cas,
    species_number) from str to int64, which reorders every groupby and
    therefore reseeds every Monte Carlo draw downstream: the rarefaction
    slope moved 0.163 -> 0.156 and the hazard share 63 -> 61 % on that
    alone.  Each deposited frame carries a <name>.dtypes.json sidecar and
    it is applied on read, so the csv and the original pickle give byte
    -identical results.
    """
    import json as _json
    import pandas as _pd
    if path.endswith(".pkl"):
        return _pd.read_pickle(path)
    if path.endswith(".parquet"):
        import numpy as _np
        f = _pd.read_parquet(path)
        for c in f.columns:
            if str(f[c].dtype) == "string":
                # pandas writes NA-semantics StringDtype; the frames the
                # analysis was built on carry the NaN-semantics "str" dtype,
                # and the two group and drop differently.
                f[c] = f[c].astype("str")
        return f
    return _pd.read_csv(path, **kw)


RAW = _data("wqp_results.parquet", "all_raw.pkl")
CHARMAP = _data("char_map.csv")

CONC_UNITS = {"ng/l": 1.0, "ug/l": 1000.0}          # -> ng/L
NONDETECT = {
    "Not Detected", "Below Method Detection Limit",
    "Not Detected at Detection Limit", "Not Detected at Reporting Limit",
    "Below Reporting Limit", "Present Below Quantification Limit",
    "Detected Not Quantified",
}
DETECTED = {"Present Above Quantification Limit"}
DROP_COND = {"Not Reported", "Systematic Contamination"}

QC_PAT = re.compile(r"(LAB|LABORATORY)[-_ ]?(CONTROL|BLANK|SPIKE|DUP)"
                    r"|FIELD[-_ ]?(DUP|BLANK|REP)"
                    r"|\bLCS\b|\bMSD?\b|\bDUP\b|\bFD\b(?![A-Z])", re.I)


def load(verbose=True):
    a = _read(RAW, low_memory=False)
    n0 = len(a)

    # --- canonical name -> parent structure (collapses salts) -------------
    cm = pd.read_csv(CHARMAP)
    # one display name per parent: the shortest free-acid name available
    cm["_score"] = cm.char.str.contains("acid", case=False).astype(int) * -10 \
                   + cm.char.str.len()
    disp = (cm.sort_values("_score").groupby("parent")
              .agg(name=("char", "first"), cas=("cas", "first")).reset_index())
    a = a.merge(cm[["char", "parent"]].rename(columns={"char": "CharacteristicName"}),
                on="CharacteristicName", how="inner")
    n_mapped = len(a)

    # --- units -----------------------------------------------------------
    u = a["ResultMeasure/MeasureUnitCode"].astype(str).str.lower().str.strip()
    a["_mult"] = u.map(CONC_UNITS)
    cond = a.ResultDetectionConditionText
    # a non-detect legitimately carries no unit
    is_nd_txt = cond.isin(NONDETECT)
    bad_unit = a._mult.isna() & ~is_nd_txt
    a = a[~bad_unit].copy()

    # --- QC ---------------------------------------------------------------
    aid = a.ActivityIdentifier.astype(str)
    is_qc = aid.str.contains(QC_PAT, na=False, regex=True)
    conflag = a.MeasureQualifierCode.astype(str).str.upper().str.contains(
        r"\bCON\b", na=False, regex=True)
    drop_cond = a.ResultDetectionConditionText.isin(DROP_COND)
    a = a[~(is_qc | conflag | drop_cond)].copy()

    # --- detection status -------------------------------------------------
    cond = a.ResultDetectionConditionText
    val = pd.to_numeric(a.ResultMeasureValue, errors="coerce")
    a["detected"] = cond.isna() | cond.isin(DETECTED)
    a["ngL"] = np.where(a.detected, val * a._mult, np.nan)
    a = a[~(a.detected & a.ngL.isna())].copy()      # detected but unparseable
    # F-ZERO: 26 rows are flagged as detections but carry a value of 0 ng/L,
    # which is not a concentration.  They do not move the medians but they
    # destroy the per-substance minimum, which the analytical-floor argument
    # depends on.  Dropped.
    n_zero = int((a.detected & (a.ngL <= 0)).sum())
    a = a[~(a.detected & (a.ngL <= 0))].copy()
    a.attrs["n_zero_dropped"] = n_zero

    a = a.merge(disp, on="parent", how="left")

    if verbose:
        print("=" * 92)
        print("v3 water: full WQP pull, detection status explicit")
        print("=" * 92)
        print(f"  retrieved results                 {n0:>9,}")
        print(f"  matched a PFAS name list          {n_mapped:>9,}")
        print(f"  after unit and QC screens         {len(a):>9,}")
        print(f"    detections                      {a.detected.sum():>9,}"
              f"   ({100*a.detected.mean():.1f}%)")
        print(f"    non-detections                  {(~a.detected).sum():>9,}")
        _np = a.parent.notna()
        print(f"    resolve to a parent structure    {int(_np.sum()):>9,}"
              f"  ({a.parent.nunique()} substances,"
              f" {int(a[_np].detected.sum()):,} detections)")
        print(f"    resolve to no parent structure   {int((~_np).sum()):>9,}"
              f"  ({a[~_np].CharacteristicName.nunique()} names,"
              f" {int(a[~_np].detected.sum()):,} detections)")
        print(f"  distinct stations                 "
              f"{a.MonitoringLocationIdentifier.nunique():>9,}")
        print(f"  F-ZERO: dropped {n_zero} rows flagged as detections "
              f"but reporting 0 ng/L")
        print(f"  QC removed: {is_qc.sum():,} QC activity ids, "
              f"{conflag.sum():,} CON-flagged, {drop_cond.sum():,} unusable condition")
        print(f"  F-QC note: v2's water frame (m_ladder.pkl, 20,682 rows) is "
              f"detections only\n             and contains no non-detects at all.")
    return a


def geography(a, verbose=True):
    det = a[a.detected]
    org = det.OrganizationFormalName.value_counts()
    k80 = int((org.cumsum() / org.sum() < 0.80).sum() + 1)
    if verbose:
        print(f"\n-- F-GEO --")
        print(f"   {len(org)} reporting organisations; "
              f"{k80} carry 80% of detections")
        print(org.head(7).to_string())
    return org


def substance_table(a, duration="r10", verbose=True):
    out, d, pools = core.build(duration, verbose=False)
    g = a.groupby("parent")
    T = pd.DataFrame({
        "n_result": g.size(),
        "n_det": g.detected.sum(),
        "n_station": g.MonitoringLocationIdentifier.nunique(),
    }).reset_index()
    det = a[a.detected]
    T = T.merge(det.groupby("parent").ngL.median().rename("det_median_ngL"),
                on="parent", how="left")
    T["det_freq"] = 100 * T.n_det / T.n_result
    T = T.merge(a[["parent", "name", "cas"]].drop_duplicates("parent"),
                on="parent", how="left")
    T = T.merge(out[["parent", "n_records", "route", "AF_tgd", "C_acute", "acute_tl",
                     "chronic_tl", "chronic_sp", "C6_prime", "AF_C6_prime",
                     "n_sp_d", "HC5_d"]], on="parent", how="left")
    T["ssd_eligible"] = T.n_sp_d.fillna(0) >= 5
    T = T.sort_values("n_det", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*92}")
        print(f"The PFAS measured in surface water: can they be assessed at all?"
              f"   (duration = {duration})")
        print(f"{'='*92}")
        s = T[["name", "n_result", "n_det", "det_freq", "det_median_ngL",
               "n_records", "chronic_tl", "chronic_sp", "route", "n_sp_d"]].copy()
        s["name"] = s.name.str.slice(0, 32)
        print(s.to_string(index=False, float_format=lambda x: f"{x:,.3g}"))

        tot = T.n_det.sum(); tst = T.n_station.sum()
        print(f"\n-- occurrence-weighted, {len(T)} substances, "
              f"{tot:,} detections --")
        for lab, mask in (("SSD-eligible (>=5 species, screened)", T.ssd_eligible),
                          ("a PNEC is derivable", T.route.isin(["chronic", "acute"])),
                          ("NO PNEC derivable", T.route == "none")):
            x = T[mask]
            print(f"   {lab:38s} {len(x):2d}/{len(T)}  "
                  f"{100*x.n_det.sum()/tot:5.1f}% of detections")

        # eligibility vs support over ALL 87 substances, not just the measured
        # F-AUCVAR: this was computed on n_records while the paper's declared
        # exposure variable is core.EFFORT (distinct tests).  The two agree to
        # two decimals here (0.922 against 0.924), which is exactly how such a
        # mismatch survives unnoticed.  Read the single definition instead.
        elig = out.n_sp_d.fillna(0) >= 5
        _ev = out[core.EFFORT]
        rk = _ev.rank(ascending=False, method="min")
        mw = stats.mannwhitneyu(_ev[elig], _ev[~elig], alternative="greater")
        auc = mw.statistic / (elig.sum() * (~elig).sum())
        print(f"\n-- eligibility vs testing effort, all {len(out)} substances --")
        print(f"   SSD-eligible: {elig.sum()}; exposure-variable ranks "
              f"{sorted(rk[elig].astype(int))}")
        print(f"   AUC = {auc:.3f}   p = {mw.pvalue:.3g}")
    return T, out


if __name__ == "__main__":
    a = load()
    geography(a)
    for rule in ("r10", "operational"):
        substance_table(a, rule)
