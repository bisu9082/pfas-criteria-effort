"""Single-command regeneration.  Everything the manuscript quotes comes from here."""
import sys, io, contextlib
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import core, water, lead, invert, ladder, mech, robust

HERE = _os.path.dirname(_os.path.abspath(__file__))
DERIVED = _os.environ.get("DERIVED_OUT",
    _os.path.join(_os.path.dirname(HERE), "derived"))
_os.makedirs(DERIVED, exist_ok=True)
LOG = _os.path.join(HERE, "run_all.log")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a = water.load()
    water.geography(a)
    for rule in ("r10", "operational"):
        out, d, pools = core.build(rule)
        out.to_csv(f"{DERIVED}/ladder_{rule}.csv", index=False)
        T, _ = water.substance_table(a, rule)
        T.to_csv(f"{DERIVED}/substances_{rule}.csv", index=False)
        core.af_decomposition(out)
        R, shared, sub = lead.run(a, out, rule)
        R.to_csv(f"{DERIVED}/lead_{rule}.csv", index=False)
        print()
    invert.compare()
    for r in ("r10","operational"):
        ladder.run(r)
    robust.run("operational")
    robust.run("r10")
    # F-MECHLOG: mech was imported but never called, so every mechanism
    # number in the body was outside the regeneration log.
    _S, _g, _B = mech.build()
    mech.channels(_S); mech.stratfree(_S)
    mech.decompose(_S, _g, _B); mech.mediator(_g, _B)
    mech.sigma_share(_S); mech.pfas_set()
open(LOG, "w").write(buf.getvalue())

# figures and tables
import subprocess
# F-RUNALL: fig_lead and fig_census were absent, so the single command that
# the cover letter will claim regenerates everything did not regenerate
# Figures 1 and 2.  tab1 is dropped from this list because sitabs now calls
# tab1.build() itself; running it here as well rebuilt the same file twice.
for mod in ("sitabs","fig_lead","fig_census","fig3","fig1","fig3b","figsi"):
    subprocess.run([sys.executable,"-I",f"{HERE}/{mod}.py"],
                   check=True, capture_output=True)
print("[run_all] regenerated Table 1, supplementary tables S1-S6, "
      "main-text Figures 1-5 and the two supplementary figures")
print(buf.getvalue())
print(f"\n[run_all] wrote {LOG} and ladder_*/substances_*/lead_*.csv")
