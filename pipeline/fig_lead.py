"""Figure 1: the co-occurrence block.  Seven PFAS at the shared stations --
the water they are measured in, against the criteria they receive.

One shared logarithmic concentration axis, because both quantities are
concentrations in ng/L and the whole claim is a comparison of their ranges:
the measured distributions sit inside about one order of magnitude
while the criteria span a factor of 7389.  Two panels with separate axes would destroy
exactly that.

Design constraints:
  * all seven rows carry a measured distribution, because all seven are
    measured.  A row with no criterion is a GAP in a row that otherwise has
    data -- absence shown, not a row omitted.
  * the two absences are different states and are drawn differently: one
    falls back to the acute base set and HAS a criterion at AF 1000; the
    other has nothing of either kind.
  * route colours are shared with Figure 2, which carries the full
    taxonomy; this figure shows only the three states the seven occupy.
  * no route taxonomy is repeated here; Figure 2 carries it.
"""
import sys, csv
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import core, water, lead

OUT = _os.environ.get("PAPER_OUT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
INK = "#1b1b1b"
COL = {"chronic": "#2b6cb0", "acute": "#b7791f", "none": "#c53030"}
WATER = "#4a7fb5"

NAME = {"PFOA ion": "Perfluorooctanoic acid",
        "Perfluorovaleric acid": "Perfluoropentanoic acid"}

a = water.load(verbose=False)
det = a[a.detected]
o, d, _ = core.build("r10", verbose=False)
nm = a[["parent", "name"]].drop_duplicates("parent").set_index("parent").name
R, shared, sub = lead.block(det, o, nm, 700)
S = o.set_index("parent")

name2parent = {v: k for k, v in nm.items()}
rows = []
for _, r in R.iterrows():
    p = name2parent[r["name"]]
    g = sub[sub.parent == p]
    v = np.log10(g.ngL[g.ngL > 0].values)
    s = S.loc[p]
    route = s.route
    if route == "chronic":
        crit = float(s.C6_prime) - np.log10(float(s.AF_C6_prime))
    elif route == "acute":
        crit = float(s.C_acute)
    else:
        crit = None
    rows.append(dict(name=NAME.get(str(r["name"]), str(r["name"])),
                     tests=int(S.loc[p].n_tests), v=v,
                     crit=crit + 6 if crit is not None else None,
                     exceed=(float((g.ngL[g.ngL > 0] > 10 ** (crit + 6)).mean()) * 100
                             if crit is not None else None),
                     route=route))
rows.sort(key=lambda z: -z["tests"])
y = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(figsize=(7.4, 3.5))
for yy, r in zip(y, rows):
    ax.axhline(yy, color="#eeeeee", lw=.6, zorder=0)
    q = np.percentile(r["v"], [5, 25, 50, 75, 95])
    ax.plot([r["v"].min(), r["v"].max()], [yy, yy], color="#c8cfd6", lw=.7, zorder=1)
    ax.plot([r["v"].max()], [yy], "|", ms=5.0, color="#9aa5b1", mew=.9, zorder=2)
    ax.plot([q[0], q[4]], [yy, yy], color="#9aa5b1", lw=1.0, zorder=2)
    ax.plot([q[1], q[3]], [yy, yy], color=WATER, lw=5.2, solid_capstyle="butt",
            alpha=.85, zorder=3)
    ax.plot([q[2]], [yy], "o", ms=3.4, color="white", zorder=4)
    xr = r["v"].max() if r["crit"] is None else max(r["crit"], r["v"].max())
    if r["crit"] is not None:
        ax.plot([r["crit"]], [yy], "D", ms=7.0, color=COL[r["route"]], zorder=5)
        # F-ROUND: "%.1f" turns PFBA's 0.069% into "0.1%", rounding a
        # sub-tenth-of-a-percent exceedance UP past the threshold a reader
        # would take it for.  Below 0.1 state the bound, not a rounded value.
        e = r["exceed"]
        if e == 0:
            frac = "no detection above"
        elif e < 0.1:
            frac = "<0.1% of detections above"
        else:
            frac = f"{e:.1f}% of detections above"
        if r["route"] == "acute":
            txt, col = f"acute base set, AF 1000 — {frac}", COL["acute"]
        else:
            txt, col = frac, (COL["chronic"] if e else "#8a949e")
        ax.annotate(txt, (xr, yy), textcoords="offset points",
                    xytext=(12, -2.4), fontsize=6.0, color=col, va="center")
    else:
        ax.annotate("no criterion of any kind", (xr, yy),
                    textcoords="offset points", xytext=(12, -2.4),
                    fontsize=6.0, color=COL["none"], style="italic",
                    va="center", ha="left")
ax.set_yticks(y)
ax.set_yticklabels([f"{r['name']}\n{r['tests']} ecotoxicity tests" for r in rows], fontsize=6.8)
ax.set_xlim(-1.3, 8.4)
ax.set_xticks([0, 2, 4, 6])
ax.set_xlabel("$\\log_{10}$ concentration or criterion  (ng L$^{-1}$)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("the same water, criteria a factor of 7389 apart",
             fontsize=8.6, loc="left", pad=6)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(handles=[
    Line2D([], [], color=WATER, lw=5.2, label="measured, 25–75%"),
    Line2D([], [], color="#9aa5b1", lw=1.0, label="5–95%"),
    Line2D([], [], color="#c8cfd6", lw=.7, label="full range"),
    Line2D([], [], marker="D", ls="", color=COL["chronic"], ms=5.6,
           label="criterion, long-term route"),
    Line2D([], [], marker="D", ls="", color=COL["acute"], ms=5.6,
           label="criterion, acute base set")],
    fontsize=6.4, loc="upper left", bbox_to_anchor=(0.0, -0.24), ncol=5,
    frameon=False, handletextpad=.4, columnspacing=1.3)
fig.savefig(f"{OUT}/fig_01_lead.png", dpi=250, bbox_inches="tight")
print("wrote fig_01_lead.png;", len(rows), "substances,",
      sum(1 for r in rows if r["crit"] is None), "with no criterion")
