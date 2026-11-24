"""Figure 3 rebuilt: the convention ladder, the operative channel, and the
decomposition of the species slope."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.special import gammaln
import core, mech

RNG = np.random.default_rng(20261122)
Z05 = stats.norm.ppf(0.05)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, WARM, GREY = "#1a1a1a", "#1f5fa8", "#c0392b", "#8a8a8a"

# ---------------------------------------------------------------- panel A
# F-FIGLAB: these rows were labelled C1 / C5 / C6 while plotting C6_pub /
# C6_ai / C6_ap.  C1 is a different convention (lowest value over ANY endpoint,
# n = 87) and was not plotted at all, although the body quotes it as the
# headline.  Labels now name the column each row actually plots, in the
# vocabulary of Supplementary Table S1, and C1 is added.
# C6ai and C6ap are intermediate steps of the C6 screen and exist nowhere
# else in the submission, so the row text names the step rather than leaving
# two undefined codes on the paper's central result.
LADDER = [("C6",   "C6_pub",   "lowest chronic NOEC, as published"),
          ("C6ai", "C6_ai",    "C6 + active-ingredient basis (step)"),
          ("C6ap", "C6_ap",    "C6 + apical effects only (step)"),
          ("C6'",  "C6_prime", "+ adequate duration (fully screened)")]


def panelA(ax, rule="r10", nboot=2000):
    out, d, pools = core.build(rule, verbose=False)
    rows = []
    # C1: lowest value over any endpoint, AF 1000 -- a constant divisor, so the
    # rank correlation is that of the endpoint itself
    _c1 = d.groupby("parent").logTox.min()
    _s1 = out.set_index("parent").join(_c1.rename("_c1"), how="inner").dropna(subset=["_c1"])
    _x = np.log10(_s1[core.EFFORT].values); _y = _s1._c1.values
    _r = stats.spearmanr(_x, _y)[0]
    _bs = [stats.spearmanr(_x[i], _y[i])[0]
           for i in (RNG.integers(0, len(_x), len(_x)) for _ in range(nboot))]
    _lo, _hi = np.nanpercentile(_bs, [2.5, 97.5])
    rows.append(("C1", "lowest value, any endpoint", _r, _lo, _hi, len(_s1)))
    for lab, col, desc in LADDER:
        s = out.dropna(subset=[col, f"AF_{col}"]).copy()
        crit = s[col] - np.log10(s[f"AF_{col}"])
        x = np.log10(s[core.EFFORT].values); y = crit.values
        r = stats.spearmanr(x, y)[0]
        bs = [stats.spearmanr(x[i], y[i])[0]
              for i in (RNG.integers(0, len(x), len(x)) for _ in range(nboot))]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append((lab, desc, r, lo, hi, len(s)))
    # C7a: lowest apical species mean chronic value over a constant divisor.
    # It was absent from this panel although the body calls the screened
    # estimates the central result and reports C7a as the one that reaches
    # conventional significance.
    _sa = out.dropna(subset=["minSMCV_d"])
    if len(_sa) >= 5:
        xa = np.log10(_sa[core.EFFORT].values); ya = _sa.minSMCV_d.values
        ra = stats.spearmanr(xa, ya)[0]
        bsa = [stats.spearmanr(xa[i], ya[i])[0]
               for i in (RNG.integers(0, len(xa), len(xa)) for _ in range(nboot))]
        rows.append(("C7a", "lowest apical species mean chronic value", ra,
                     *np.nanpercentile(bsa, [2.5, 97.5]), len(_sa)))
    # the SSD route
    s = out.dropna(subset=["HC5_d"])
    if len(s) >= 5:
        x = np.log10(s[core.EFFORT].values); y = s.HC5_d.values
        r = stats.spearmanr(x, y)[0]
        bs = [stats.spearmanr(x[i], y[i])[0]
              for i in (RNG.integers(0, len(x), len(x)) for _ in range(nboot))]
        rows.append(("C7b", "SSD fifth percentile", r,
                     *np.nanpercentile(bs, [2.5, 97.5]), len(s)))
    ypos = np.arange(len(rows))[::-1]
    XL, XR = -1.02, 0.42          # fixed, readable range
    for yp, (lab, desc, r, lo, hi, n) in zip(ypos, rows):
        excl = hi < 0 or lo > 0
        c = WARM if excl else GREY
        lc, hc = max(lo, XL), min(hi, XR)
        ax.plot([lc, hc], [yp, yp], color=c, lw=2.2, solid_capstyle="round",
                alpha=.85, zorder=2)
        # mark an interval that runs off the panel rather than rescaling for it
        if hi > XR:
            ax.annotate("", xy=(XR + 0.035, yp), xytext=(XR - 0.005, yp),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4,
                                        shrinkA=0, shrinkB=0), zorder=2)
        if lo < XL:
            ax.annotate("", xy=(XL - 0.035, yp), xytext=(XL + 0.005, yp),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4,
                                        shrinkA=0, shrinkB=0), zorder=2)
        ax.plot(r, yp, "o", ms=5.5, color=c, zorder=3, mec="white", mew=.8)
        note = f"$n$ = {n}"
        if hi > XR or lo < XL:
            note += "  (interval uninformative)"
        ax.text(XR + 0.065, yp, note, fontsize=6.8, color=GREY,
                va="center", ha="left")
    ax.axvline(0, color=INK, lw=.7, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lab}   {desc}" for lab, desc, *_ in rows], fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(XL - 0.05, XR + 0.40)
    ax.set_xticks(np.arange(-1.0, 0.45, 0.25))
    ax.set_xlabel(r"Spearman $\rho$ (testing effort vs derived criterion)")
    ax.set_title("A   each screening step narrows the dependence; it survives "
                 "the screen but the screened intervals are wide",
                 loc="left", fontsize=9, color=INK, pad=6)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------- panel B
def panelB(axes, S):
    """Two panels sharing y: each exposure on its OWN axis.  Plotting both
    against one shared 'count' axis would invite a comparison at equal x,
    which is meaningless -- species and record counts are different units."""
    axL, axR = axes
    specs = [(axL, "ls", ACC, "o", "species count", "number of species"),
             (axR, "lr", GREY, "s", "record count", "number of records")]
    for ax, col, c, mk, lab, xlab in specs:
        q = pd.qcut(S[col], 12, duplicates="drop")
        g = S.groupby(q, observed=True)
        x = g[col].mean().values; y = g.hc5.mean().values
        se = (g.hc5.std() / np.sqrt(g.hc5.size())).values
        ax.errorbar(x, y, yerr=1.96 * se, fmt=mk + "-", color=c, ms=3.8, lw=1.4,
                    capsize=2.0, elinewidth=.8, alpha=.95)
        b = np.polyfit(S[col], S.hc5, 1)
        xs = np.linspace(S[col].min(), S[col].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=c, lw=.9, ls=(0, (4, 3)), alpha=.65)
        ax.set_xlabel(f"$\\log_{{10}}$ {xlab}")
        r = stats.spearmanr(S[col], S.hc5)[0]
        # labelled "raw" so it cannot be confused with the boxed partial
        ax.text(.04, .06, f"raw $\\rho = {r:+.3f}$", transform=ax.transAxes,
                fontsize=8, color=c, va="bottom")
    r1 = mech.rank_partial(S.ls, S.hc5, S.lr)[0]
    r2 = mech.rank_partial(S.lr, S.hc5, S.ls)[0]
    axL.set_ylabel("fitted fifth percentile\n($\\log_{10}$ mg L$^{-1}$)")
    axR.tick_params(labelleft=False)
    axL.text(.04, .97, f"holding records fixed\n$\\rho = \\mathbf{{{r1:+.3f}}}$  survives",
             transform=axL.transAxes, fontsize=7.2, color=ACC, va="top",
             bbox=dict(fc="white", ec=ACC, lw=.5, alpha=.9, pad=3))
    axR.text(.04, .97, f"holding species fixed\n$\\rho = \\mathbf{{{r2:+.3f}}}$  reverses",
             transform=axR.transAxes, fontsize=7.2, color=WARM, va="top",
             bbox=dict(fc="white", ec=WARM, lw=.5, alpha=.9, pad=3))
    axL.set_title(f"B   species coverage is the operative channel "
                  f"($n$ = {len(S):,} chemicals)",
                  loc="left", fontsize=9, color=INK, pad=6)


# ---------------------------------------------------------------- panel C
def c4(n):
    n = np.asarray(n, float)
    return np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)) * np.sqrt(2 / (n - 1))


def panelC(ax, S, g, ndraw=60):
    # F-RAREONE: this panel drew its own rarefaction with a different seed and
    # annotated the endpoint difference (0.11), while the abstract, body and
    # \rarEffect carried the fitted slope over the same range (0.10).  It now
    # reads the one definition in mech.rarefaction_slope, so the figure and the
    # text cannot disagree.
    levels = (5, 8, 12, 20)
    sl, _se, gs, R = mech.rarefaction_slope(g, ndraw=ndraw, sizes=levels,
                                            min_species=20)
    R = R.assign(s=(10 ** R.ls).round().astype(int))
    m = R.groupby("s").hd.agg(["mean", "std", "size"])
    se = m["std"] / np.sqrt(m["size"])
    ax.errorbar(levels, m["mean"], yerr=1.96 * se, fmt="o-", color=WARM,
                ms=5, lw=1.6, capsize=2.5, elinewidth=.9, zorder=3)
    ax.axhline(0, color=INK, lw=.7, ls=(0, (3, 3)), zorder=1)
    span = abs(sl) * np.log10(levels[-1] / levels[0])
    ax.set_xlim(4.5, 22.5)
    ax.text(.985, .94,
            "fewer species  $\\Rightarrow$  higher percentile  $\\Rightarrow$  "
            "less protective\n"
            f"{span:.2f} $\\log_{{10}}$ units from 20 species down to 5",
            transform=ax.transAxes, fontsize=7.2, color=WARM, ha="right",
            va="top", linespacing=1.5,
            bbox=dict(fc="white", ec=WARM, lw=.5, alpha=.9, pad=3.4))
    ax.set_xscale("log"); ax.set_xticks(levels)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("species retained (same chemical, rarefied)")
    ax.set_ylabel("deviation in fifth percentile\n($\\log_{10}$, within chemical)")
    ax.set_title(f"C   the identified part: {len(gs)} chemicals, chemical fixed effects",
                 loc="left", fontsize=9, color=INK, pad=6)
    return R


if __name__ == "__main__":
    S, g, B = mech.build()
    fig = plt.figure(figsize=(7.1, 7.6))
    gsx = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.05],
                           hspace=.58, wspace=.08)
    panelA(fig.add_subplot(gsx[0, :]))
    axB = [fig.add_subplot(gsx[1, 0])]
    axB.append(fig.add_subplot(gsx[1, 1], sharey=axB[0]))
    panelB(axB, S)
    panelC(fig.add_subplot(gsx[2, :]), S, g)
    fig.savefig(OUT + "/fig_04_effort.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    print("wrote " + OUT + "/fig_04_effort.png")
