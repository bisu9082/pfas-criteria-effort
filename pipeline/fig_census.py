"""Figure 2: the derivability census.  Every PFAS in the monitoring record and
what the guidance returns for it, plus one homologous series.

Reads v3 outputs only (substances_r10.csv, ladder_r10.csv) -- no new analysis.
Design constraints that must not be relaxed:
  * non-derivable substances sit in a gutter OFF the value axis.  Placed at the
    axis edge they read as "criterion ~ 1e6", which is the worst error this
    figure can make.
  * three states, not two: the acute base-set route HAS a criterion, at a
    factor of 1000.  Collapsing it into "none" reintroduces graphically the
    error removed from the Conclusions.
  * no error bar, no band, no fitted line, no p-value.  This is a complete
    enumeration; nothing here has sampling uncertainty.
  * one glyph, one meaning.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os as _os
V = _os.path.dirname(_os.path.abspath(__file__))
OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))

S = list(csv.DictReader(open(f"{V}/substances_r10.csv")))
L = {x["parent"]: x for x in csv.DictReader(open(f"{V}/ladder_r10.csv"))}

INK = "#1b1b1b"
COL = {"chronic": "#2b6cb0", "acute": "#b7791f", "judgement": "#718096", "none": "#c53030"}
LAB = {"chronic": "long-term route, AF 10-100", "acute": "acute base set, AF 1000",
       "judgement": "no tabulated factor", "none": "no admissible datum"}

SHORT = {
 "PFOA ion": "PFOA", "Perfluorooctane sulfonic acid": "PFOS",
 "Perfluorobutanesulfonic acid": "PFBS", "Perfluorohexanoic acid": "PFHxA",
 "Perfluoroheptanoic acid": "PFHpA", "Perfluorobutanoic acid": "PFBA",
 "Perfluorovaleric acid": "PFPeA", "Perfluorohexanesulfonic acid": "PFHxS",
 "Perfluorononanoic acid": "PFNA", "Perfluorodecanoic acid": "PFDA",
 "Perfluorooctanesulfonamide": "FOSA", "Perfluoroundecanoic acid": "PFUnDA",
 "Perfluorododecanoic acid": "PFDoDA", "6:2 Fluorotelomer sulfonate acid": "6:2 FTS",
 "Perfluorotridecanoic acid": "PFTrDA", "3-perfluoropentyl propanoate": "5:3 FTCA ester",
 "Perfluorobutylsulfonamide": "FBSA", "Perfluorohexanesulfonamide": "FHxSA",
 "3-Perfluoroheptylpropanoic acid (7:3 FTCA)": "7:3 FTCA",
 "2-(N-methylperfluoro-1-octanesulfonamido)acetic acid": "N-MeFOSAA",
 "Perfluoro(2-propoxypropanoate)": "HFPO-DA",
 "N-ethyl perfluoro-1-octanesulfonamide": "N-EtFOSA",
}
def short(n):
    for k, v in SHORT.items():
        if n.startswith(k[:22]):
            return v
    return n[:12]

def crit(x):
    """The criterion the guidance returns, by whichever route applies.

    F-ACUTECRIT: a substance on the acute base-set route HAS a criterion, at
    the tabulated factor of 1000.  Showing it as an absence would reintroduce
    graphically the error that was removed from the Conclusions.  Only the
    'none' and 'judgement' routes return nothing.
    """
    c = x["C6_prime"]
    if c:
        return 10 ** float(c) * 1e6 / float(x["AF_C6_prime"])
    ca = x.get("C_acute")
    if ca:
        return 10 ** float(ca) * 1e6
    return None

rows = sorted(S, key=lambda r: -int(r["n_det"]))
y = np.arange(len(rows))[::-1]

fig = plt.figure(figsize=(7.4, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[2.75, 0.42, 0.95], height_ratios=[1.52, 1.0],
                      wspace=0.10, hspace=0.52, left=0.135, right=0.975,
                      top=0.925, bottom=0.145)
axA = fig.add_subplot(gs[0, 0]); axG = fig.add_subplot(gs[0, 1], sharey=axA)
axD = fig.add_subplot(gs[0, 2], sharey=axA)
axB = fig.add_subplot(gs[1, :])

# ---------------- A: the census
for yy, x in zip(y, rows):
    r = x["route"]; c = crit(x)
    axA.axhline(yy, color="#e8e8e8", lw=.6, zorder=0)
    if c is not None:
        axA.plot([c], [yy], "o", ms=6.2, color=COL[r], zorder=3)
        # the species count is a property of the LONG-TERM pool; annotating it
        # on an acute-route substance reads as "this criterion rests on 0
        # species", which is false -- it rests on the acute base set.
        nsp = L[x["parent"]]["nsp_C6_prime"] if r == "chronic" else ""
        if nsp:
            axA.annotate(f"{nsp} sp", (c, yy), textcoords="offset points",
                         xytext=(9, -2.6), fontsize=6.2, color=INK)
    else:
        axG.plot([0], [yy], "X", ms=7.0, color=COL[r], zorder=3)
axA.set_yticks(y); axA.set_yticklabels([short(x["name"]) for x in rows], fontsize=7)
_vals = [c for c in (crit(x) for x in rows) if c is not None]
assert _vals, "no criteria to plot"
axA.set_xscale("log")
axA.set_xlim(10 ** np.floor(np.log10(min(_vals)) - 0.35),
             10 ** np.ceil(np.log10(max(_vals)) + 0.35))
# every substance must land somewhere: on the axis or in the gutter.
_on = sum(1 for x in rows if crit(x) is not None)
_gut = sum(1 for x in rows if crit(x) is None)
assert _on + _gut == len(rows), f"{_on}+{_gut} != {len(rows)}"
assert min(_vals) >= axA.get_xlim()[0] and max(_vals) <= axA.get_xlim()[1], \
    "a criterion falls outside the drawn axis and would be clipped"
axA.set_xlabel("derived criterion, by whichever route applies  (ng L$^{-1}$)", fontsize=8)
axA.tick_params(labelsize=7)
axA.set_title("A   every PFAS detected in the monitoring record, and what the guidance returns for it",
              fontsize=8.2, loc="left", pad=6)
axG.set_xlim(-1, 1); axG.set_xticks([])
axG.set_xlabel("no criterion\nof any kind", fontsize=7.4, color=COL["none"])
axG.set_facecolor("#fbf1f1")
plt.setp(axG.get_yticklabels(), visible=False)
for s_ in ("top", "right", "left", "bottom"):
    axG.spines[s_].set_visible(False)
for yy, x in zip(y, rows):
    axG.axhline(yy, color="#eee5e5", lw=.6, zorder=0)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[k], label=LAB[k], ms=5.6)
                    for k in ("chronic", "acute")] +
                   [Line2D([], [], marker="X", ls="", color=COL[k], label=LAB[k], ms=5.8)
                    for k in ("none",)],
           fontsize=6.6, loc="upper left", bbox_to_anchor=(0.0, -0.21),
           ncol=3, frameon=False, handletextpad=.3, columnspacing=1.4)

# detections side panel
axD.barh(y, [int(x["n_det"]) for x in rows], color="#cbd5e0", height=.62)
axD.set_xscale("log"); axD.set_xlim(5, 6000)
axD.set_xlabel("detections", fontsize=8); axD.tick_params(labelsize=7)
plt.setp(axD.get_yticklabels(), visible=False)
for s in ("top", "right", "left"):
    axD.spines[s].set_visible(False)

# ---------------- B: the PFCA homologous series
series = [("C4", "Perfluorobutanoic acid"), ("C5", "Perfluorovaleric acid"),
          ("C6", "Perfluorohexanoic acid"), ("C7", "Perfluoroheptanoic acid"),
          ("C8", "PFOA ion"), ("C9", "Perfluorononanoic acid"),
          ("C10", "Perfluorodecanoic acid"), ("C11", "Perfluoroundecanoic acid"),
          ("C12", "Perfluorododecanoic acid")]
byname = {x["name"]: x for x in S}
xs, ys, cs, tags = [], [], [], []
for i, (lab, nm) in enumerate(series):
    x = byname[nm]; c = crit(x); r = x["route"]
    xs.append(i); cs.append(COL[r])
    if c is None:
        ys.append(None); tags.append("")
    else:
        ys.append(c)
        tags.append((L[x["parent"]]["nsp_C6_prime"] + " sp") if r == "chronic"
                    else "base set")
# F-FIGSEG: a dashed/solid "adjacent vs skipped" convention is unreadable when
# most segments span a gap.  Join only adjacent derivable members; the dots and
# the status strip carry the rest.
dd = [(i, v) for i, v in zip(xs, ys) if v is not None]
for a, b in zip(dd[:-1], dd[1:]):
    if b[0] - a[0] == 1:
        axB.plot([a[0], b[0]], [a[1], b[1]], "-", color="#a0aec0", lw=1.0, zorder=1)
for i, v, c, t in zip(xs, ys, cs, tags):
    if v is None:
        pass
    else:
        axB.plot([i], [v], "o", ms=7.0, color=c, zorder=3)
        axB.annotate(t, (i, v), textcoords="offset points", xytext=(0, 9),
                     fontsize=6.4, ha="center", color=INK)
axB.set_yscale("log"); axB.set_ylim(40, 2.0e6)
axB.set_xlim(-.5, len(series) - .5)
axB.set_xticks(xs); axB.set_xticklabels([s[0] for s in series], fontsize=7.4)
axB.set_xlabel("perfluoroalkyl carboxylate chain length  (toxicity increases to the right)", fontsize=8)
axB.set_ylabel("criterion  (ng L$^{-1}$)", fontsize=8)
axB.tick_params(labelsize=7)
axB.set_title("B   one homologous series: the criterion is non-monotone in chain length",
              fontsize=8.2, loc="left", pad=10)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.text(2.28, 4.2e5, "225$\\times$ more lenient than C4,\non long-term data from one\ntrophic level",
         fontsize=6.6, color=INK, va="top", ha="left")
# categorical status strip, outside the value scale
div = axB.inset_axes([0, -0.30, 1, 0.13], transform=axB.transAxes)
div.set_xlim(-.5, len(series) - .5); div.set_ylim(0, 1)
div.set_xticks([]); div.set_yticks([])
for s_ in ("top", "right", "left", "bottom"):
    div.spines[s_].set_visible(False)
ROUTE_TXT = {"chronic": "long-term", "acute": "acute\nbase set", "none": "no\ndatum",
             "judgement": "no factor"}
for i, (lab, nm) in enumerate(series):
    r = byname[nm]["route"]
    div.add_patch(plt.Rectangle((i - .42, .08), .84, .84,
                                color=COL[r], alpha=.16, lw=0))
    div.text(i, .5, ROUTE_TXT[r], ha="center", va="center",
             fontsize=5.9, color=COL[r])
div.text(-.48, -0.55, "route the guidance returns", fontsize=6.6,
         color=INK, ha="left", va="top")

fig.savefig(f"{OUT}/fig_02_census.png", dpi=250, bbox_inches="tight")
print("wrote fig_02_census.png")
