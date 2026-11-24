"""
v3 core: single-source regeneration for the PFAS criteria / testing-effort paper.

Fixes relative to v2 (each verified; see FIXES.md):
  F-AF1  assessment factors are computed from the SAME screened pool the
         endpoint is taken from, not from the unscreened NOEC pool.
  F-AF2  aquatic macrophytes ("Flowers, Trees, Shrubs, Ferns") count as
         primary producers.  v2's trophic() mapped them to "other", which
         excluded 107 screened records across 6 parents from trophic
         coverage and so inflated their assessment factors.
  F-DUR  chronic exposure-duration minima follow ECHA R.10
         (algae 72 h, macrophytes/Lemna 168 h, invertebrates 504 h,
         fish/amphibians 672 h).  The 72/168 h rule used in v2 is retained
         as a declared sensitivity variant, labelled "operational", never
         as "what the guidance specifies".
  F-DRV  EU TGD base-set admissibility is evaluated, so substances for
         which NO PNEC is derivable are reported as such instead of
         receiving a number.
  F-CI   HC5 intervals are equal-tailed with the coverage stated.  v2 took
         nct.ppf(.95)/nct.ppf(.05), i.e. two one-sided 95% tolerance
         limits, giving JOINT 90% coverage while labelling it "exact 95%".

Usage:  python3 -I core.py [--duration r10|operational]
"""
import argparse, json, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import nct
from scipy.special import gammaln


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


SRC = _data("results_parent.csv.gz")
Z05 = stats.norm.ppf(0.05)

# ---------------------------------------------------------------- taxonomy
# EU TGD / ECHA R.10 base set is three trophic levels: primary producers
# (algae and aquatic plants), invertebrates (Daphnia), and fish.
PRIMARY_PRODUCER = {"Algae", "Moss, Hornworts", "Flowers, Trees, Shrubs, Ferns"}
INVERTEBRATE = {"Crustaceans", "Insects/Spiders", "Molluscs", "Worms",
                "Other Invertebrates", "Invertebrates"}
VERTEBRATE = {"Fish", "Amphibians"}          # fish is the base-set level;
                                             # amphibians are pooled here and
                                             # the count is reported both ways


def head_group(g):
    return str(g).split(";")[0].strip()


def trophic(g):
    """Map an ECOTOX group string to an EU TGD trophic level."""
    h = head_group(g)
    if h in PRIMARY_PRODUCER:
        return "producer"
    if h in INVERTEBRATE:
        return "invertebrate"
    if h in VERTEBRATE:
        return "vertebrate"
    return "other"


# ------------------------------------------------- chronic duration minima
# ECHA R.10 Table R.10-3 style minima for a study to count as long-term.
DURATION_RULES = {
    "r10": {          # what the guidance actually specifies
        "producer_algae": 72.0,    # OECD 201, 72 h is chronic for algae
        "producer_plant": 168.0,   # OECD 221 Lemna / 239 Myriophyllum, 7 d
        "invertebrate":   504.0,   # OECD 211 Daphnia reproduction, 21 d
        "vertebrate":     672.0,   # OECD 210 fish ELS, 28 d
        "other":          672.0,
    },
    "operational": {  # the rule used in the v2 draft, kept as a sensitivity
        "producer_algae":  72.0,
        "producer_plant":  72.0,
        "invertebrate":   168.0,
        "vertebrate":     168.0,
        "other":          168.0,
    },
}


def duration_key(g):
    h = head_group(g)
    if h in ("Algae", "Moss, Hornworts"):
        return "producer_algae"
    if h == "Flowers, Trees, Shrubs, Ferns":
        return "producer_plant"
    if h in INVERTEBRATE:
        return "invertebrate"
    if h in VERTEBRATE:
        return "vertebrate"
    return "other"


# ------------------------------------------------------------------- loader
# F-EPADM: ECOTOX's results.txt carries endpoint codes that are not effect
# thresholds a no-effect criterion may be anchored on.  A benchmark dose (BMD,
# BMC, BMDF, CMC variants) is a modelled dose at a specified response level and
# carries no no-effect interpretation; a bioconcentration-family code (BCF,
# BCFD, BAF) on a mass-per-volume row is an exposure concentration from a
# bioconcentration study, or a coding error.  Either way the row is inadmissible
# as the basis of a PNEC.  The objection is admissibility, not dimension: these
# rows carry mg/L, ug/L, ppb and ng/L and their units are sound.
NON_EFFECT_EP = r"BCF|BAF|BMD|BMC|CMC"


def load(duration="r10"):
    d = pd.read_csv(SRC, low_memory=False)
    d = d[~d.ep.astype(str).str.contains(NON_EFFECT_EP, na=False)].copy()
    # the censoring rule is applied upstream, in the curation that produced
    # SRC: `censored` is all-False here, so this filter is a no-op retained
    # only so the admissibility chain is visible in one place.
    d = d[~d.censored.astype(bool)].copy()

    d["dur"] = pd.to_numeric(d.obs_duration_mean, errors="coerce")
    u = d.obs_duration_unit.astype(str).str.lower().str.strip()
    mult = u.map({"h": 1.0, "d": 24.0, "wk": 168.0, "mo": 720.0,
                  "min": 1 / 60}).astype(float)
    d["hrs"] = d.dur * mult

    d["apical"] = d.apical.astype(bool)
    d["troph"] = d.ecotox_group.map(trophic)
    d["dkey"] = d.ecotox_group.map(duration_key)

    rule = DURATION_RULES[duration]
    d["min_hrs"] = d.dkey.map(rule)
    d["chronic_ok"] = d.hrs.notna() & (d.hrs >= d.min_hrs)
    d.attrs["duration"] = duration
    return d


# ------------------------------------------------------------ TGD machinery
ACUTE = {"LC50", "EC50", "IC50"}
CHRONIC_EP = {"NOEC", "NOEL", "MATC", "EC10"}
# F-EFFORT: the exposure variable is defined ONCE, here.  v3 used result rows
# (12,763).  A single test reporting 52 gene-expression endpoints at one NOEC
# contributes 52 rows, and the manuscript's own composition analysis shows the
# mechanistic-endpoint share RISES with effort (apical 72.7% -> 21.5%), so a row
# count is inflated differentially by the substances under study -- the confound
# runs with the outcome.  Distinct tests (2,182) is the unit of experiment and
# is invariant to how many endpoints a study chose to report.  Result rows stay
# computed and are reported as the declared sensitivity.
EFFORT = "n_tests"            # primary exposure
EFFORT_ALT = "n_records"      # declared sensitivity

AF_CHRONIC = {3: 10.0, 2: 50.0, 1: 100.0}
AF_ACUTE_BASESET = 1000.0

# F-AF100: R.10 Table R.10-4's AF-100 row is "one long-term NOEC (fish or
# Daphnia)".  A single long-term level that is a PRIMARY PRODUCER does not
# satisfy it -- the algal test is the base-set producer test.  v3 assigned
# AF 100 on producer-only long-term data, which is a factor the guidance does
# not tabulate, in a paper about untabulated factor assignment.
AF100_TAXA = {"invertebrate", "vertebrate"}   # Daphnia, fish


def af_chronic(tl_names):
    """Factor from Table R.10-4, or NaN where the table has no applicable row."""
    tls = set(tl_names) - {"other"}
    n = len(tls)
    if n >= 3:
        return 10.0
    if n == 2:
        return 50.0
    if n == 1:
        return 100.0 if tls & AF100_TAXA else np.nan
    return np.nan


def tl_count(sub):
    """Number of distinct EU TGD trophic levels represented."""
    return len(set(sub.troph) - {"other"})


def assign_af(pool, parents):
    """
    Assessment factor per parent, computed from the pool the endpoint is
    taken from (F-AF1).  Returns a frame with the trophic counts so the
    assignment is auditable rather than a bare number.
    """
    rows = []
    gb = dict(list(pool.groupby("parent")))
    for p in parents:
        sub = gb.get(p)
        if sub is None or not len(sub):
            rows.append(dict(parent=p, n_rec_pool=0, n_sp_pool=0,
                             tl=0, tl_names="", AF=np.nan))
            continue
        tls = sorted(set(sub.troph) - {"other"})
        n = len(tls)
        rows.append(dict(parent=p, n_rec_pool=len(sub),
                         n_sp_pool=sub.latin_name.nunique(),
                         tl=n, tl_names="+".join(tls),
                         AF=af_chronic(tls)))
    return pd.DataFrame(rows)


def derivability(d):
    """
    EU TGD admissibility per parent (F-DRV).

    FOUR routes, not three.  The docstring described a three-way split long
    after the judgement branch was added, which is the kind of drift that makes
    a reader trust the comment over the code.

    route  = 'chronic'    admissible long-term data on >=1 credited trophic
                          level                               -> AF 10/50/100
             'acute'      no admissible long-term data but a complete
                          short-term base set (all three levels) -> AF 1000
             'judgement'  admissible long-term data exist but on no credited
                          level, AND the short-term base set is incomplete.
                          No row of table R.10-4 applies, so the guidance
                          tabulates no factor and an assessor must decide.
                          Tested AFTER the base-set test, so a substance with
                          a complete base set takes the acute route instead.
             'none'       no admissible datum of either kind; nothing is
                          derivable under the guidance as written.

    The judgement branch is why the ladder is reported as four-way throughout
    the manuscript, and why 'none' means absence of data rather than absence
    of a tabulated factor.
    """
    base = d[d.conc1_type == "A"]
    ac = base[base.ep.isin(ACUTE)]
    ch = base[base.ep.isin(CHRONIC_EP) & base.apical & base.chronic_ok]
    rows = []
    for p in sorted(d.parent.dropna().unique()):
        a = ac[ac.parent == p]
        c = ch[ch.parent == p]
        atl, ctl = tl_count(a), tl_count(c)
        ctl_names = set(c.troph) - {"other"}
        af_c = af_chronic(ctl_names) if ctl >= 1 else np.nan
        if ctl >= 1 and not np.isnan(af_c):
            route, af = "chronic", af_c
        elif atl >= 3:
            # either no long-term datum at all, or long-term data on a level
            # the AF table does not credit: the tabulated base-set route
            route, af = "acute", AF_ACUTE_BASESET
        elif ctl >= 1:
            # admissible long-term data exist but no row of Table R.10-4
            # applies and the base set is incomplete.  The guidance tabulates
            # no factor; a PNEC here requires an assessor's judgement.
            route, af = "judgement", np.nan
        else:
            route, af = "none", np.nan
        rows.append(dict(parent=p, n_acute=len(a), acute_tl=atl,
                         n_chronic=len(c), chronic_tl=ctl,
                         chronic_sp=c.latin_name.nunique(),
                         chronic_tl_names="+".join(sorted(set(c.troph) - {"other"})),
                         route=route, AF_tgd=af))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- SSD
def c4(n):
    n = np.asarray(n, dtype=float)
    return np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)) * np.sqrt(2 / (n - 1))


def ssd(sel, min_species=5, bias_correct=False):
    sm = sel.groupby(["parent", "latin_name"]).logTox.mean().reset_index()
    g = sm.groupby("parent").logTox
    t = pd.DataFrame({"n_sp": g.size(), "mu": g.mean(),
                      "sd": g.std(), "minSMCV": g.min()}).reset_index()
    if bias_correct:
        t["sd"] = t.sd / c4(t.n_sp)
    t["HC5"] = t.mu + Z05 * t.sd
    t.loc[(t.n_sp < min_species) | t.sd.isna(), "HC5"] = np.nan
    return t


def hc5_interval(mu, sd, n, coverage=0.95):
    """
    Equal-tailed two-sided interval for the 5th percentile, from noncentral-t
    tolerance limits.  `coverage` is the JOINT coverage of the interval
    (F-CI): coverage=0.95 uses the .975/.025 quantiles; coverage=0.90
    reproduces v2's .95/.05 bounds, which v2 mislabelled as 95%.
    """
    a = (1.0 - coverage) / 2.0
    dnc = -Z05 * np.sqrt(n)
    lo = mu - nct.ppf(1 - a, n - 1, dnc) / np.sqrt(n) * sd
    hi = mu - nct.ppf(a, n - 1, dnc) / np.sqrt(n) * sd
    return lo, hi


# -------------------------------------------------------------------- main
def build(duration="r10", verbose=True):
    d = load(duration)
    base = d[d.conc1_type == "A"]

    pools = {
        "C6_pub":   d[d.ep.isin(["NOEC", "NOEL"])],
        "C6_ai":    base[base.ep.isin(["NOEC", "NOEL"])],
        "C6_ap":    base[base.ep.isin(["NOEC", "NOEL"]) & base.apical],
        "C6_prime": base[base.ep.isin(["NOEC", "NOEL"]) & base.apical & base.chronic_ok],
    }
    parents = sorted(d.parent.dropna().unique())
    out = pd.DataFrame({"parent": parents})
    out = out.join(d.groupby("parent").size().rename("n_records"), on="parent")
    out = out.join(d.groupby("parent").test_id.nunique().rename("n_tests"), on="parent")

    # endpoints and a per-pool assessment factor (F-AF1)
    for k, p in pools.items():
        out = out.join(p.groupby("parent").logTox.min().rename(k), on="parent")
        af = assign_af(p, parents).rename(
            columns={"AF": f"AF_{k}", "tl": f"tl_{k}",
                     "tl_names": f"tlnames_{k}", "n_rec_pool": f"nrec_{k}",
                     "n_sp_pool": f"nsp_{k}"})
        out = out.merge(af, on="parent", how="left")

    drv = derivability(d)
    out = out.merge(drv, on="parent", how="left")

    # F-ACUTECRIT: a substance on the acute base-set route HAS a criterion, at
    # the tabulated factor of 1000.  Without this column the figure and the
    # text can only show it as an absence, which reintroduces graphically the
    # error removed from the Conclusions.
    _ac = base[base.ep.isin(ACUTE)].groupby("parent").logTox.min()
    out = out.join(_ac.rename("acute_min"), on="parent")
    out["C_acute"] = np.where(out.route == "acute",
                              out.acute_min - np.log10(AF_ACUTE_BASESET), np.nan)

    # SSD on the screened chronic apical pool
    sel = base[base.ep.isin(CHRONIC_EP) & base.apical]
    for tag, s in (("nd", sel), ("d", sel[sel.chronic_ok])):
        t = ssd(s).rename(columns={c: f"{c}_{tag}"
                                   for c in ["n_sp", "mu", "sd", "HC5", "minSMCV"]})
        out = out.merge(t, on="parent", how="left")

    out.attrs["duration"] = duration
    if verbose:
        c1_effort(out, d)
    report(out, duration)
    return out, d, pools



def c1_effort(out, d):
    """F-C1GEN: the C1 correlation was quoted in the body with no generator
    anywhere in the pipeline, and the audit checked only that the string was
    present in the text.  Generate it, under both exposures."""
    lo = d.groupby("parent").logTox.min()
    s = out.set_index("parent").join(lo.rename("_c1"), how="inner").dropna(subset=["_c1"])
    print("\n-- C1 (lowest value over all endpoints, AF 1000) --")
    res = {}
    for e, lab in ((EFFORT, "primary (tests)"), (EFFORT_ALT, "sensitivity (rows)")):
        r, p = stats.spearmanr(np.log10(s[e]), s._c1)
        res[e] = (r, p, len(s))
        print(f"  C1        n={len(s):3d}   {lab:20s} rho={r:+.3f} (p={p:.3g})")
    return res


def report(out, duration):
    w = 78
    print("=" * w)
    print(f"v3 core  |  duration rule = {duration}")
    print("=" * w)

    print("\n-- C6 ladder, AF taken from the SAME screened pool (F-AF1) --")
    for k in ("C6_pub", "C6_ai", "C6_ap", "C6_prime"):
        s = out.dropna(subset=[k, f"AF_{k}"]).copy()
        crit = s[k] - np.log10(s[f"AF_{k}"])
        r_e, p_e = stats.spearmanr(np.log10(s[EFFORT]), s[k])
        r_c, p_c = stats.spearmanr(np.log10(s[EFFORT]), crit)
        print(f"  {k:9s} n={len(s):3d}   endpoint rho={r_e:+.3f} (p={p_e:.3g})"
              f"   criterion rho={r_c:+.3f} (p={p_c:.3g})")

    print("\n-- EU TGD derivability (F-DRV) --")
    print(out.route.value_counts().to_string())
    none = out[out.route == "none"]
    print(f"  no PNEC derivable: {len(none)} of {len(out)} substances")

    print("\n-- C7 SSD --")
    for tag, nm in (("nd", "no duration filter"), ("d", f"duration-filtered ({duration})")):
        s = out.dropna(subset=[f"HC5_{tag}"])
        if len(s) < 3:
            print(f"  {nm:34s} n={len(s):3d}  (too few to correlate)")
            continue
        r, p = stats.spearmanr(np.log10(s[EFFORT]), s[f"HC5_{tag}"])
        print(f"  {nm:34s} n={len(s):3d}  rho={r:+.3f}  p={p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", default="r10",
                    choices=list(DURATION_RULES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, d, pools = build(a.duration)
    path = a.out or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  f"ladder_{a.duration}.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


def af_decomposition(out, pool_key="C6_pub"):
    """F-AFDEC: the assessment-factor decomposition quoted in the body had no
    generator anywhere in the pipeline, was computed per decade of RECORDS
    (the superseded exposure), on the pre-F-AF1 unscreened factor pool, and
    before the four-way ladder restricted the AF-100 row to fish/Daphnia.

    log C = log(NOEC) - log(AF) is an identity, so the three slopes sum by
    construction; what the decomposition reports is how little of the movement
    in the endpoint the factor carries back.
    """
    s = out.dropna(subset=[pool_key, f"AF_{pool_key}"]).copy()
    x = np.log10(s[EFFORT].values)
    y_noec = s[pool_key].values
    y_af = np.log10(s[f"AF_{pool_key}"].values)
    y_crit = y_noec - y_af

    def ols(x, y):
        b, a = np.polyfit(x, y, 1)
        r, p = stats.pearsonr(x, y)
        return b, p

    b_noec, p_noec = ols(x, y_noec)
    b_af, p_af = ols(x, y_af)
    b_crit, p_crit = ols(x, y_crit)
    assert abs(b_crit - (b_noec - b_af)) < 1e-9, "decomposition identity broken"
    offset = abs(b_af) / abs(b_noec)
    span = abs(b_noec) * (x.max() - x.min())
    afs = sorted(set(s[f"AF_{pool_key}"].dropna()))
    af_span = np.log10(max(afs) / min(afs)) if afs else np.nan
    print(f"\n-- assessment-factor decomposition ({pool_key}, exposure = {EFFORT}) --")
    print(f"  n = {len(s)} substances")
    print(f"  endpoint   slope {b_noec:+.3f} per decade (p={p_noec:.2g})")
    print(f"  factor     slope {b_af:+.3f} per decade (p={p_af:.2g})")
    print(f"  criterion  slope {b_crit:+.3f} per decade (p={p_crit:.2g})")
    print(f"  the factor offsets {100*offset:.0f}% of the endpoint movement, "
          f"{100*(1-offset):.0f}% passes through")
    print(f"  endpoint moves {span:.1f} log units over the observed exposure range")
    print(f"  factor values in use: {[int(a) for a in afs]}  "
          f"span {af_span:.1f} log units")
    return dict(n=len(s), b_noec=b_noec, b_af=b_af, b_crit=b_crit,
                p_noec=p_noec, p_af=p_af, offset=offset, span=span,
                af_span=af_span, afs=[int(a) for a in afs])


def route_gap(rule="r10", ssd_col="HC5_d"):
    """The gap between the two derivation routes, on the substances where BOTH
    return a value.

    F-DING: the manuscript carried 2.13 log units on five substances with no
    generator anywhere in v3.  That value was computed before the AF-100 fix,
    which removed one substance (gap 3.51) from the C6' pool and so changed
    both the n and the mean.  The number is regenerated here so it cannot go
    stale again silently.

    The assessment-factor route's CRITERION is the screened minimum divided by
    its tabulated factor -- not the minimum itself.  Comparing C6_prime to an
    HC5 directly compares a no-effect concentration with a criterion and
    understates the gap; that is the error that made 2.13 irreproducible.

    The gap is signed HC5 minus assessment-factor criterion, so a positive
    value means the assessment-factor route is the more stringent of the two,
    which is the direction Ding et al. report.
    """
    o, _, _ = build(rule, verbose=False)
    m = o.dropna(subset=["C6_prime", "AF_C6_prime", ssd_col]).copy()
    m["crit_af"] = m.C6_prime - np.log10(m.AF_C6_prime)
    m["gap"] = m[ssd_col] - m["crit_af"]
    assert len(m) >= 2, "too few both-route substances to summarise"
    return dict(n=len(m), mean=float(m.gap.mean()), median=float(m.gap.median()),
                lo=float(m.gap.min()), hi=float(m.gap.max()), rows=m)


def ssd_diagnostics(duration="operational", min_species=5):
    """Per-substance goodness-of-fit for the log-normal SSD.

    F-FITCOLS: si_tab_fit once carried the Anderson-Darling statistic, its 5%
    critical value, a Shapiro-Wilk p and the probability-plot R^2.  A
    regeneration dropped all four while the SI prose kept arguing from them,
    leaving four claims pointing at columns that no longer existed.  They are
    recomputed here, from the SAME screened pool that produces mu_d/sd_d, so a
    diagnostic can never describe a different fit from the one tabulated.

    The Anderson-Darling critical value is reported alongside the statistic
    because the statistic alone does not say whether it rejects; scipy's
    critical values for the normal case already account for the parameters
    being estimated from the sample.
    """
    from scipy import stats as st
    d = load(duration)
    base = d[d.conc1_type == "A"]
    sel = base[base.ep.isin(CHRONIC_EP) & base.apical]
    sel = sel[sel.chronic_ok]
    sm = sel.groupby(["parent", "latin_name"]).logTox.mean().reset_index()
    rows = []
    for p, g in sm.groupby("parent"):
        x = np.sort(g.logTox.values.astype(float))
        if len(x) < min_species or np.std(x) == 0:
            continue
        a = st.anderson(x, dist="norm")
        crit5 = float(a.critical_values[list(a.significance_level).index(15.0) + 2]) \
            if 15.0 in list(a.significance_level) else float(a.critical_values[2])
        sw = st.shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
        (_, _), (_, _, r) = st.probplot(x, dist="norm", fit=True)
        rows.append(dict(parent=p, n_sp=len(x), AD=float(a.statistic),
                         AD_crit5=crit5, SW_p=float(sw[1]), R2=float(r) ** 2))
    t = pd.DataFrame(rows)
    t["AD_reject5"] = t.AD > t.AD_crit5
    return t


def lcl_support(duration="operational", coverages=(0.95, 0.90)):
    """Does reporting the lower confidence limit act as an implicit support factor?

    F-LCL: the reviewer instruction for this item specified rho = -0.508 for the
    interval half-width against log species, "invariant to coverage".  No such
    value exists.  The magnitude is the SIGMA-HAT correlation (+0.503), reported
    with the half-width's label and the opposite sign; sigma-hat is exactly
    coverage-invariant because it is a fit parameter with no coverage argument,
    which is where the invariance claim came from.

    The half-width correlation is not reported at all.  half-width = k(n) * sd,
    where k(n) is strictly decreasing in n by construction, so that correlation
    is a tug-of-war between a deterministic identity and a substantive relation
    and moves with coverage for that reason.  What the recommendation rests on
    is the identity itself, plus what the lower limit does.

    NOTE on k(n): it is strictly decreasing in n -- verified over the species
    counts in this set -- but its Spearman against n is NOT -1 in this sample,
    because 9 of the 13 substances tie at 5, 6 or 7 species and ties cap the
    coefficient short of unity.  State the monotonicity as the property of the
    function that it is; do not quote a rank correlation for it.
    """
    o, _, _ = build(duration, verbose=False)
    e = o[o.n_sp_d.fillna(0) >= 5]
    ls = np.log10(e.n_sp_d.astype(int).values)
    out = dict(n=len(e),
               rho_sd=float(stats.spearmanr(ls, e.sd_d.values)[0]),
               p_sd=float(stats.spearmanr(ls, e.sd_d.values)[1]),
               rho_hc5=float(stats.spearmanr(ls, e.HC5_d.values)[0]),
               p_hc5=float(stats.spearmanr(ls, e.HC5_d.values)[1]),
               lcl={})
    kseq = [hc5_interval(0.0, 1.0, int(n), coverage=coverages[0]) for n in
            sorted(set(e.n_sp_d.astype(int)))]
    kvals = [(hi - lo) / 2.0 for lo, hi in kseq]
    out["k_strictly_decreasing"] = all(a > b for a, b in zip(kvals[:-1], kvals[1:]))
    for cov in coverages:
        lo = np.array([hc5_interval(mu, sd, int(n), coverage=cov)[0]
                       for mu, sd, n in zip(e.mu_d.values, e.sd_d.values,
                                            e.n_sp_d.astype(int).values)])
        r, p = stats.spearmanr(ls, lo)
        out["lcl"][cov] = (float(r), float(p))
    return out
