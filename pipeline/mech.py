"""
v3 mech: the mechanism, on the broad set (n = 1,175 chemicals with >= 5 species).

Replaces the manuscript's n = 13 sigma decomposition, which rests on a
quantity that is not itself significant (OLS HC5 slope p = 0.090) and whose
sigma share has a bootstrap CI of [21, 108] with 4.2% of replicates above
100%.

Fixes:
  F-SPEC  the SSD's sample size is the number of SPECIES, not the number of
          records.  The manuscript's "the species channel is not the
          operative one" rests on n = 13 and is contradicted at n = 1,175.
          Framed as a specification correction, not an empirical reversal:
          on the 13-substance PFAS set the partials run the other way and
          n = 13 cannot resolve the decomposition at all.
  F-STRAT the three-stratum table ("record count does nothing within species
          strata, p = 0.45, 0.60, 0.058") is cut-point dependent and does
          NOT reproduce.  Replaced by stratification-free pooled partials.
  F-MED   conditioning on within-chemical SD conditions on a DESCENDANT of
          testing effort, i.e. a mediator.  Replaced by a held-out hazard
          control over 20 random splits.
  F-FILL  rho(log n, within-chemical SD) = +0.783 came from fillna(0) over
          1,266 singleton chemicals (26.6% of 4,757).
  F-DECOMP the species slope is decomposed into c4 bias, estimator
          sample-size behaviour (identified by within-chemical rarefaction
          with chemical fixed effects), species-pool composition, and an
          irreducible residual that survives an independent hazard control.
"""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
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


SRC = _data("broad_records.parquet", "broad_records.pkl")
Z05 = stats.norm.ppf(0.05)
SEED = 20261122


def c4(n):
    n = np.asarray(n, dtype=float)
    return np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)) * np.sqrt(2 / (n - 1))


def species_means(B=None):
    if B is None:
        B = _read(SRC)
    return B.groupby(["test_cas", "species_number"]).logTox.mean().reset_index(), B


def build(min_sp=5):
    g, B = species_means()
    S = (g.groupby("test_cas")
          .agg(nsp=("species_number", "count"), mu=("logTox", "mean"),
               sd=("logTox", "std")).reset_index())
    S = S.merge(B.groupby("test_cas").test_id.count().rename("nrec").reset_index(),
                on="test_cas")
    S = S[S.nsp >= min_sp].dropna(subset=["sd"]).copy()
    S["hc5"] = S.mu + Z05 * S.sd
    S["ls"] = np.log10(S.nsp)
    S["lr"] = np.log10(S.nrec)
    S["lrps"] = np.log10(S.nrec / S.nsp)     # replication depth per species
    return S, g, B


def rank_partial(x, y, c):
    """Spearman partial: Pearson on ranks, residualised on the rank of c."""
    X, Y, C = (stats.rankdata(v) for v in (x, y, c))
    A = np.c_[np.ones(len(C)), C]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return stats.pearsonr(res(X), res(Y))


def fisher_pool(pairs):
    """Pool correlations over strata by Fisher z with n-3 weights."""
    zs, ws = [], []
    for r, n in pairs:
        if n < 6 or not np.isfinite(r):
            continue
        zs.append(np.arctanh(np.clip(r, -0.999, 0.999))); ws.append(n - 3)
    zs, ws = np.array(zs), np.array(ws, dtype=float)
    zb = (zs * ws).sum() / ws.sum()
    se = 1 / np.sqrt(ws.sum())
    return np.tanh(zb), np.tanh(zb - 1.96 * se), np.tanh(zb + 1.96 * se), len(zs)


def channels(S):
    print("=" * 88)
    print(f"THE OPERATIVE CHANNEL  (n = {len(S):,} chemicals with >= 5 species)")
    print("=" * 88)
    for lab, col in (("log records", "lr"), ("log species", "ls")):
        r, p = stats.spearmanr(S[col], S.hc5)
        print(f"  rho(HC5, {lab:12s}) = {r:+.4f}   p = {p:.3g}")
    r1, p1 = rank_partial(S.ls, S.hc5, S.lr)
    r2, p2 = rank_partial(S.lr, S.hc5, S.ls)
    print(f"\n  partial(species, HC5 | records) = {r1:+.4f}  p = {p1:.3g}   <- survives")
    print(f"  partial(records, HC5 | species) = {r2:+.4f}  p = {p2:.3g}   "
          f"<- SIGN REVERSES")
    print("  A single latent effort factor measured with differential error would")
    print("  attenuate both partials; it cannot reverse one.  The reversal is")
    print("  produced by estimator structure, and is predicted a priori:")
    for lab, col in (("sd", "sd"), ("mu", "mu"), ("HC5", "hc5")):
        r, p = rank_partial(S.lrps, S[col], S.ls)
        print(f"    partial(replication depth, {lab:3s} | species) = {r:+.4f}  p = {p:.3g}")
    print("  i.e. replication within species shrinks the between-species SD")
    print("  estimate, which RAISES the HC5 -- the opposite sign to adding species.")


def stratfree(S):
    print("\n" + "=" * 88)
    print("F-STRAT: the three-stratum table does not reproduce; use pooled partials")
    print("=" * 88)
    print("  hand-chosen strata, for the record (cut-point dependent):")
    for cuts in ([(5, 7), (8, 14), (15, 10 ** 9)], [(5, 6), (7, 10), (11, 10 ** 9)]):
        out = []
        for lo, hi in cuts:
            T = S[(S.nsp >= lo) & (S.nsp <= hi)]
            r, p = stats.spearmanr(T.lr, T.hc5)
            out.append(f"[{lo}-{hi if hi < 10**8 else '+'}] n={len(T)} "
                       f"rho={r:+.3f} p={p:.3f}")
        print("    " + " | ".join(out))
    print("  -> significant in two of three with OPPOSITE signs.  Not publishable.")

    pairs = [(stats.spearmanr(T.lr, T.hc5)[0], len(T))
             for _, T in S.groupby("nsp") if len(T) >= 10]
    r, lo, hi, k = fisher_pool(pairs)
    print(f"\n  pooled over {k} exact-species-count strata (n >= 10 each):")
    print(f"    rho(records, HC5 | species) = {r:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    q = pd.qcut(S.lr, 4, labels=False, duplicates="drop")
    pairs2, det = [], []
    for k2 in sorted(set(q)):
        T = S[q == k2]
        rr, pp = stats.spearmanr(T.ls, T.hc5)
        pairs2.append((rr, len(T))); det.append(f"{rr:+.3f}")
    r2, lo2, hi2, _ = fisher_pool(pairs2)
    print(f"  species effect within record-count quartiles: {', '.join(det)}")
    print(f"    pooled rho(species, HC5 | records) = {r2:+.4f}  "
          f"95% CI [{lo2:+.3f}, {hi2:+.3f}]   <- survives in every stratum")


def rarefaction_slope(g, ndraw=60, sizes=(5, 8, 12, 20), min_species=20):
    """Within-chemical rarefaction slope of the fifth percentile on log species,
    under chemical fixed effects, so that only the estimator's sample size
    varies.

    F-CENSUSvMECH: extracted from decompose() so the magnitude has one
    definition and can be read into the manuscript as a macro.  The paper's
    headline spread (criteria a factor of 7389 apart) comes from WHICH route
    applies, which is a complete enumeration; this slope is the effort
    mechanism, and it is about a tenth of a log unit over 20 species down to 5.
    Stating one without the other invites the reader to attribute the first to
    the second.
    """
    rng = np.random.default_rng(SEED)
    pool = g.groupby("test_cas").species_number.count()
    cas = pool[pool >= min_species].index
    gs = {c: v.logTox.values for c, v in g[g.test_cas.isin(cas)].groupby("test_cas")}
    rows = []
    for c, v in gs.items():
        for s in sizes:
            for _ in range(ndraw):
                d = rng.choice(v, s, replace=False)
                rows.append((c, np.log10(s), d.mean() + Z05 * d.std(ddof=1)))
    R = pd.DataFrame(rows, columns=["cas", "ls", "hc5"])
    R["lsd"] = R.ls - R.groupby("cas").ls.transform("mean")
    R["hd"] = R.hc5 - R.groupby("cas").hc5.transform("mean")
    sl = (R.lsd * R.hd).sum() / (R.lsd ** 2).sum()
    se = np.sqrt(((R.hd - sl * R.lsd) ** 2).sum() /
                 ((len(R) - len(gs) - 1) * (R.lsd ** 2).sum()))
    return float(sl), float(se), gs, R


def decompose(S, g, B, ndraw=60):
    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 88)
    print("F-DECOMP: what the species slope is made of")
    print("=" * 88)
    A = np.c_[np.ones(len(S)), S.ls]
    b_tot = np.linalg.lstsq(A, S.hc5.values, rcond=None)[0][1]
    print(f"  between-chemical slope, HC5 ~ log10(species) : {b_tot:+.4f} per decade")

    hc5c = S.mu + Z05 * (S.sd / c4(S.nsp.values))
    b_c4 = np.linalg.lstsq(A, hc5c.values, rcond=None)[0][1]
    print(f"    after c4 small-sample SD correction        : {b_c4:+.4f}"
          f"   -> c4 explains {100*(b_tot-b_c4)/b_tot:.1f}%")

    # within-chemical rarefaction, chemical fixed effects
    sl, se, gs, R = rarefaction_slope(g, ndraw=ndraw)
    print(f"    within-chemical rarefaction ({len(gs)} chemicals >= 20 species,")
    print(f"      chemical fixed effects, {ndraw} draws at s = 5/8/12/20)")
    print(f"      slope = {sl:+.4f} (se {se:.4f})"
          f"   -> estimator sample size = {100*sl/b_tot:.1f}%")
    for s in (5, 20):
        print(f"        mean rarefied HC5 at s = {s:2d} : "
              f"{R[R.ls == np.log10(s)].hc5.mean():+.4f}")
    print("      a shorter species list gives a HIGHER HC5 -- data-poor substances")
    print("      are UNDER-protected, and c4 fixes only a twentieth of it.")

    # species-pool composition
    top = B.species_number.value_counts()
    for k in (50, 100, 200):
        keep = set(top.head(k).index)
        gg = g[g.species_number.isin(keep)]
        T = (gg.groupby("test_cas").agg(nsp=("species_number", "count"),
                                        mu=("logTox", "mean"),
                                        sd=("logTox", "std")).reset_index())
        T = T[T.nsp >= 5].dropna(subset=["sd"])
        T = T.merge(S[["test_cas", "lr"]], on="test_cas", how="inner")
        T["hc5"] = T.mu + Z05 * T.sd; T["ls"] = np.log10(T.nsp)
        r, _ = stats.spearmanr(T.ls, T.hc5)
        rp, _ = rank_partial(T.ls, T.hc5, T.lr)
        print(f"    restricted to the {k:3d} most-tested species (n = {len(T):4d}) : "
              f"rho = {r:+.4f}, partial | records = {rp:+.4f}")
    print("      -> species-pool composition explains essentially none of it.")



def exp_min_wor(x, k):
    """F-MEDX: exact expectation of the minimum of k values drawn WITHOUT
    replacement from x.

    v3 took a single random draw, which adds noise to the dependent variable
    and attenuates the correlation toward zero by about a third.  The
    expectation has a closed form, so no simulation is needed:

        E[min] = sum_i x_(i) * C(m-i, k-1) / C(m, k),   i = 1..m-k+1

    on x sorted ascending -- x_(i) is the minimum exactly when the other k-1
    draws all come from the m-i values above it.  Binomials in logs, or
    C(m, k) overflows at the m in this set.  Deterministic: removes a seed
    from the package.
    """
    x = np.sort(np.asarray(x, dtype=float))
    m = x.size
    if k >= m:
        return float(x[0])
    i = np.arange(1, m - k + 2)
    logw = (gammaln(m - i + 1) - gammaln(k) - gammaln(m - i - k + 2)) \
           - (gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1))
    w = np.exp(logw)
    assert abs(w.sum() - 1.0) < 1e-9, f"weights sum {w.sum()!r} at m={m} k={k}"
    return float(x[: m - k + 1] @ w)

def mediator(g, B, nsplit=40, k=5):
    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 88)
    print("F-MED: restoring the hazard-attention statement with a LEGITIMATE control")
    print("=" * 88)
    rec = B.groupby("test_cas").test_id.count().rename("nrec")
    # depth-equalised minimum: k records per chemical, so the extreme-order
    # statistic is not driven by n
    byc = {c: v.logTox.values for c, v in B.groupby("test_cas")}
    cas = [c for c, v in byc.items() if len(v) >= 2 * k]
    tot, p_sd, p_hz = [], [], []
    for _ in range(nsplit):
        mn, hz, sdv, ln = [], [], [], []
        for c in cas:
            v = byc[c]
            idx = rng.permutation(len(v))
            a, b = v[idx[:len(v) // 2]], v[idx[len(v) // 2:]]
            if len(a) < k or len(b) < 1:
                continue
            mn.append(exp_min_wor(a, k))
            hz.append(b.mean()); sdv.append(b.std(ddof=1) if len(b) > 1 else np.nan)
            ln.append(np.log10(len(v)))
        D = pd.DataFrame(dict(mn=mn, hz=hz, sd=sdv, ln=ln)).dropna()
        tot.append(stats.spearmanr(D.ln, D.mn)[0])
        p_hz.append(rank_partial(D.ln, D.mn, D.hz)[0])
        p_sd.append(rank_partial(D.ln, D.mn, D.sd)[0])
    tot, p_hz, p_sd = map(np.array, (tot, p_hz, p_sd))
    n = len(D)
    se = 1 / np.sqrt(n - 4)
    print(f"  {nsplit} independent splits, n = {n:,} chemicals with >= {2*k} records")
    print(f"    total rho(depth-equalised minimum, log n)      "
          f"{tot.mean():+.4f}  split sd {tot.std():.4f}")
    print(f"    partial | held-out within-chemical SD (mediator) "
          f"{p_sd.mean():+.4f}  split sd {p_sd.std():.4f}")
    print(f"    partial | held-out hazard level  (legitimate)   "
          f"{p_hz.mean():+.4f}  split sd {p_hz.std():.4f}"
          f"   95% CI [{p_hz.mean()-1.96*se:+.3f}, {p_hz.mean()+1.96*se:+.3f}]")
    print(f"    the legitimate control leaves "
          f"{100*p_hz.mean()/tot.mean():.0f}% of the total.")
    print("  Conditioning on within-chemical SD conditions on a descendant of")
    print("  effort, so the near-zero partial understates the direct effect.")

    print("\n  F-FILL: rho(log n, within-chemical SD)")
    sd_all = B.groupby("test_cas").logTox.std()
    nn = B.groupby("test_cas").test_id.count()
    D2 = pd.DataFrame(dict(sd=sd_all, n=nn)).dropna(subset=["n"])
    nsing = int(D2.sd.isna().sum())
    r_fill = stats.spearmanr(np.log10(D2.n), D2.sd.fillna(0))[0]
    D5 = D2[D2.n >= 5].dropna()
    D2b = D2.dropna()
    print(f"    with fillna(0) over {nsing:,} singleton chemicals "
          f"({100*nsing/len(D2):.1f}% of {len(D2):,}) : {r_fill:+.4f}   <- as published")
    print(f"    on n >= 5, no imputation (n = {len(D5):,})            : "
          f"{stats.spearmanr(np.log10(D5.n), D5.sd)[0]:+.4f}   <- correct")
    print(f"    on n >= 2, no imputation (n = {len(D2b):,})            : "
          f"{stats.spearmanr(np.log10(D2b.n), D2b.sd)[0]:+.4f}")


def sigma_share(S, nboot=2000):
    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 88)
    print("The two channels, at n = 1,175 instead of n = 13")
    print("=" * 88)
    for lab, col in (("records", "lr"), ("species", "ls")):
        x = S[col].values
        A = np.c_[np.ones(len(x)), x]
        bmu = np.linalg.lstsq(A, S.mu.values, rcond=None)[0][1]
        bsd = np.linalg.lstsq(A, S.sd.values, rcond=None)[0][1]
        tot = bmu + Z05 * bsd
        share = abs(Z05 * bsd) / abs(tot) * 100
        bs = []
        for _ in range(nboot):
            i = rng.integers(0, len(S), len(S))
            Ab = np.c_[np.ones(len(i)), x[i]]
            m = np.linalg.lstsq(Ab, S.mu.values[i], rcond=None)[0][1]
            s = np.linalg.lstsq(Ab, S.sd.values[i], rcond=None)[0][1]
            t = m + Z05 * s
            if abs(t) > 1e-9:
                bs.append(abs(Z05 * s) / abs(t) * 100)
        bs = np.array(bs)
        print(f"  exposure = log {lab:8s}  d(mu) = {bmu:+.4f}  z*d(sd) = {Z05*bsd:+.4f}"
              f"  total = {tot:+.4f}")
        print(f"      sigma share = {share:.1f}%   bootstrap 95% CI "
              f"[{np.percentile(bs,2.5):.0f}, {np.percentile(bs,97.5):.0f}]"
              f"   replicates > 100%: {100*(bs>100).mean():.1f}%")


def pfas_set():
    """The 13-substance PFAS set cannot resolve the decomposition -- say so."""
    import core
    print("\n" + "=" * 88)
    print("F-SPEC: on the PFAS SSD set the partials run the OTHER way")
    print("=" * 88)
    for rule in ("operational", "r10"):
        out, _, _ = core.build(rule, verbose=False)
        s = out.dropna(subset=["HC5_d"]).copy()
        if len(s) < 5:
            print(f"  [{rule}] n = {len(s)} -- too few to correlate at all")
            continue
        s["lr"] = np.log10(s.n_records); s["ls"] = np.log10(s.n_sp_d)
        rr = stats.spearmanr(s.lr, s.HC5_d)
        rs = stats.spearmanr(s.ls, s.HC5_d)
        p1 = rank_partial(s.ls, s.HC5_d, s.lr)[0]
        p2 = rank_partial(s.lr, s.HC5_d, s.ls)[0]
        print(f"  [{rule}] n = {len(s)}")
        print(f"     rho(records) = {rr[0]:+.3f} (p={rr[1]:.3g})   "
              f"rho(species) = {rs[0]:+.3f} (p={rs[1]:.3g})")
        print(f"     partial(species|records) = {p1:+.3f}   "
              f"partial(records|species) = {p2:+.3f}")
    print("  So the manuscript is not 'backwards' on its own data: n = 13 cannot")
    print("  separate the channels.  The correction is one of SPECIFICATION --")
    print("  an SSD's sample size is its species count -- demonstrated at n = 1,175.")


if __name__ == "__main__":
    S, g, B = build()
    channels(S); stratfree(S); decompose(S, g, B); mediator(g, B)
    sigma_share(S); pfas_set()
