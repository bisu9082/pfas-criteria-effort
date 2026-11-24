"""Generate the supplementary tables the manuscript references, from v3."""
import os as _os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from scipy import stats
import core, water, mech, lead, ladder

OUT = os.environ.get("SITABS_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables"))

# Macros produced while building S1--S5 and emitted with the rest by s6().
# s6() asserts this is non-empty rather than silently writing a macro file
# missing half its definitions, which is how a \\newcommand can go absent
# and leave an undefined control sequence in a document that still compiles
# in a later run from stale .aux files.
_MAC = []
def esc(s): return str(s).replace("&",r"\&").replace("_",r"\_").replace("%",r"\%")

def _lcl_macros():
    """F-LCL: five coefficients, all macro-fed.  The half-width correlation is
    deliberately NOT among them -- see core.lcl_support for why it is not a
    reportable quantity."""
    r = core.lcl_support()
    assert r["k_strictly_decreasing"], "k(n) is not monotone; the body claim fails"
    lo95, p95 = r["lcl"][0.95]
    lo90, p90 = r["lcl"][0.90]
    return "".join([
        "%% LCL as an implicit support factor -- see core.lcl_support()\n",
        "\\newcommand{\\lclN}{%d}\n" % r["n"],
        "\\newcommand{\\rhoSigmaSp}{%+.3f}\n" % r["rho_sd"],
        "\\newcommand{\\pSigmaSp}{%.2f}\n" % r["p_sd"],
        "\\newcommand{\\rhoHcSp}{%+.3f}\n" % r["rho_hc5"],
        "\\newcommand{\\pHcSp}{%.2f}\n" % r["p_hc5"],
        "\\newcommand{\\rhoLclA}{%+.3f}\n" % lo95,
        "\\newcommand{\\pLclA}{%.2f}\n" % p95,
        "\\newcommand{\\rhoLclB}{%+.3f}\n" % lo90,
        "\\newcommand{\\pLclB}{%.2f}\n" % p90,
    ])


def _rarefy_macros():
    """F-CENSUSvMECH: the manuscript's magnitude claim (criteria 7389-fold
    apart) comes from WHICH ROUTE applies -- a complete enumeration -- while
    the effort mechanism is a within-chemical effect of about a tenth of a log
    unit.  A reader who meets both in one abstract attributes the first to the
    second.  Emit the mechanism's own magnitude so the abstract can state it."""
    import numpy as _np
    _S, _g, _B = mech.build()
    b, se, _gs, _R = mech.rarefaction_slope(_g)
    dec = _np.log10(20.0 / 5.0)
    return "".join([
        "%% within-chemical rarefaction effect -- see mech.py\n",
        "\\newcommand{\\rarSlope}{%.3f}\n" % abs(b),
        "\\newcommand{\\rarEffect}{%.2f}\n" % (abs(b) * dec),
    ])


def _station_macros():
    """F-WRSCOPE: Water Research rejects up front for supporting-discipline
    depth 'without making a good link with water research in general'.  The
    paper already computes what the derivation choice does to a monitoring
    record, but reports it only as percentages of detections and stations.  A
    water scientist reads counts.  These are the SAME figures on the SAME
    common subset, converted -- not a new analysis and not a new denominator.
    """
    import ladder as _ladder
    res, common, sub = _ladder.run("operational")
    n = sub.MonitoringLocationIdentifier.nunique()
    af = round(res["C6p"][2] / 100.0 * n)
    ssd = round(res["C7b"][2] / 100.0 * n)
    ratio = res["C7b"][0] / res["C6p"][0]
    return "".join([
        "%% derivation choice expressed in stations -- see ladder.run()\n",
        "\\newcommand{\\stTotal}{%d}\n" % n,
        "\\newcommand{\\stAF}{%d}\n" % af,
        "\\newcommand{\\stSSD}{%d}\n" % ssd,
        "\\newcommand{\\stDiff}{%d}\n" % (af - ssd),
        "\\newcommand{\\stSubs}{%d}\n" % len(common),
        # the DERIVATION-CHOICE movement of one threshold, which is NOT the
        # 7389-fold span across seven different substances
        "\\newcommand{\\routeRatio}{%.0f}\n" % ratio,
        "\\newcommand{\\monOrgs}{%d}\n" % _orgs(),
    ])


def _orgs():
    import water as _w
    return int(_w.load(verbose=False).OrganizationFormalName.nunique())


def _infeasible_macros():
    """F-CHRONPOOL: the sharpest referee objection is that the effort mechanism
    is demonstrated on an acute-only set after the chronic target population
    was declared unable to resolve it.  The answer is not an apology: the
    chronic pool cannot support a within-chemical rarefaction at all.  Emit the
    two counts so the manuscript states a reason rather than a concession."""
    import mech as _m, core as _c
    _S, _g, _B = _m.build()
    _sl, _se, _gs, _R = _m.rarefaction_slope(_g)
    d = _c.load("operational")
    b = d[d.conc1_type == "A"]
    sel = b[b.ep.isin(_c.CHRONIC_EP) & b.apical & b.chronic_ok]
    sp = (sel.groupby(["test_cas", "latin_name"]).size()
             .groupby("test_cas").size())
    return "".join([
        "%% why the mechanism is resolved on acute data -- see core/mech\n",
        "\\newcommand{\\acutePanel}{%d}\n" % len(_gs),
        "\\newcommand{\\chronPanel}{%d}\n" % int((sp >= 20).sum()),
        "\\newcommand{\\chronMaxSp}{%d}\n" % int(sp.max()),
    ])


def _lead_macros():
    """F-LEADPOTENCY: the seven-substance paragraph quantified support by
    record count and species count, but the paper's declared exposure variable
    is the number of DISTINCT TESTS, and that is what Figure 1 annotates on
    every row.  The body lagged its own figure.  Emit the range so the two
    cannot disagree.  Reported as tests, not species, which also steers round
    the chronic_sp / nsp_C6_prime collision -- those two differ for PFHxA (2
    against 1) and PFOS (34 against 33)."""
    import water as _w, core as _c, lead as _l
    a = _w.load(verbose=False)
    det = a[a.detected]
    o, _, _ = _c.build("r10", verbose=False)
    nm = a[["parent", "name"]].drop_duplicates("parent").set_index("parent").name
    R, shared, sub = _l.block(det, o, nm, 700)
    n2p = {v: k for k, v in nm.items()}
    S = o.set_index("parent")
    t = [int(S.loc[n2p[r["name"]]].n_tests) for _, r in R.iterrows()]
    return "".join([
        "%% seven-substance block, support measured as distinct tests\n",
        "\\newcommand{\\leadTestLo}{%d}\n" % min(t),
        "\\newcommand{\\leadTestHi}{%d}\n" % max(t),
        "\\newcommand{\\leadTestFac}{%d}\n" % round(max(t) / min(t)),
    ])


def _baseset_macros():
    """F-BASESET: 17 of the 23 chronic-route substances have an INCOMPLETE
    short-term base set, and none of those 17 has a complete one.  So the
    chronic route is not the stricter of two available routes for them -- it is
    the only route they have.  Withdraw the long-term datum and they fall to
    'no tabulated factor', not to the acute base set at 1000.  This is a
    statement about derivability, not about stringency."""
    import core as _c
    o, _, _ = _c.build("r10", verbose=False)
    ch = o[o.route == "chronic"]
    inc = ch[ch.acute_tl < 3]
    vc = inc.acute_tl.value_counts().to_dict()
    return "".join([
        "%% base-set backstop under the four-way ladder -- see core.derivability()\n",
        "\\newcommand{\\chronN}{%d}\n" % len(ch),
        "\\newcommand{\\noBackstop}{%d}\n" % len(inc),
        "\\newcommand{\\withBackstop}{%d}\n" % int((ch.acute_tl >= 3).sum()),
        "\\newcommand{\\noAcuteAtAll}{%d}\n" % int(vc.get(0, 0)),
    ])


def _auc_macros():
    """F-AUCCI: the abstract and body state AUC = 0.92 with no interval.  Its
    upper bound touches 1.000 -- the statistic is boundary-censored at five
    eligible substances -- and that has to be declared rather than left for a
    referee to discover.  Computed on core.EFFORT, the declared exposure
    variable, which is also what water.py now reads."""
    import core as _c, numpy as _np
    from scipy import stats as _st
    o, _, _ = _c.build("r10", verbose=False)
    elig = (o.n_sp_d.fillna(0) >= 5).values
    x = o[_c.EFFORT].values
    mw = _st.mannwhitneyu(x[elig], x[~elig], alternative="greater")
    auc = mw.statistic / (elig.sum() * (~elig).sum())
    rng = _np.random.default_rng(20260101)
    ip, iq = _np.where(elig)[0], _np.where(~elig)[0]
    bs = []
    for _ in range(4000):
        a = rng.choice(ip, len(ip), replace=True)
        b = rng.choice(iq, len(iq), replace=True)
        bs.append(_st.mannwhitneyu(x[a], x[b], alternative="greater").statistic
                  / (len(a) * len(b)))
    lo, hi = _np.percentile(bs, [2.5, 97.5])
    return "".join([
        "%% AUC with a bootstrap interval, boundary-censored above\n",
        "\\newcommand{\\aucVal}{%.3f}\n" % auc,
        "\\newcommand{\\aucLo}{%.2f}\n" % lo,
        "\\newcommand{\\aucHi}{%.2f}\n" % hi,
        "\\newcommand{\\aucNelig}{%d}\n" % int(elig.sum()),
        "\\newcommand{\\aucP}{%.4f}\n" % mw.pvalue,
    ])


def _routegap_macros():
    """F-DING: the two-route gap is quoted in the abstract, the introduction
    and the results.  Emit it as macros so the three cannot drift apart, which
    is exactly how 2.13 survived in all three after the AF-100 fix changed it."""
    r = core.route_gap()
    return "".join([
        "%% two-route gap, regenerated -- see core.route_gap()\n",
        "\\newcommand{\\gapN}{%d}\n" % r["n"],
        "\\newcommand{\\gapNword}{%s}\n" % ("four" if r["n"] == 4 else str(r["n"])),
        "\\newcommand{\\gapMean}{%.2f}\n" % r["mean"],
        "\\newcommand{\\gapMedian}{%.2f}\n" % r["median"],
        # F-GAPRANGE: the reviewer's wording paired the mean with the median
        # and set them in an en-dash range, which a reader takes for the spread
        # across substances.  The real spread is wider on both sides.  Emit the
        # actual extremes so the text cannot pass a central pair off as a range.
        "\\newcommand{\\gapLo}{%.2f}\n" % r["lo"],
        "\\newcommand{\\gapHi}{%.2f}\n" % r["hi"],
    ])
def _hc5_width_macros():
    """F-CI2: the HC5 interval widths are quoted in BOTH main.tex and si.tex.
    Emit them as macros so the two documents cannot state different widths for
    the same quantity, which is how the joint-90 numbers survived in the SI
    after the body was corrected.  Coverage is an explicit argument, and the
    nominal coverage is written into the macro name, not into a caption."""
    COV = 0.95
    out = [f"%% HC5 two-sided {COV:.0%} interval widths, coverage explicit\n"]
    assert abs((1 - COV) / 2 - 0.025) < 1e-12, "coverage/tail mismatch"
    for rule, tag in (("r10", "RTen"), ("operational", "Op")):
        o, _, _ = core.build(rule, verbose=False)
        e = o[o.n_sp_d.fillna(0) >= 5]
        W, NS = [], []
        for _, r in e.iterrows():
            lo, hi = core.hc5_interval(r.mu_d, r.sd_d, int(r.n_sp_d), coverage=COV)
            W.append(hi - lo); NS.append(int(r.n_sp_d))
        W = np.asarray(W)
        i_lo, i_hi = int(W.argmin()), int(W.argmax())
        out += [f"\\newcommand{{\\hcWidthLo{tag}}}{{{W.min():.2f}}}\n",
                f"\\newcommand{{\\hcWidthHi{tag}}}{{{W.max():.2f}}}\n",
                f"\\newcommand{{\\hcWidthLoSp{tag}}}{{{NS[i_lo]}}}\n",
                f"\\newcommand{{\\hcWidthHiSp{tag}}}{{{NS[i_hi]}}}\n",
                f"\\newcommand{{\\hcN{tag}}}{{{len(W)}}}\n"]
    return "".join(out)

def s1to5():
    """F-IMPORTSIDE: S1--S5 were written at module IMPORT time, so merely
    importing sitabs overwrote five checked-in tables in paperA.  That made
    them impossible to verify by regeneration -- importing the generator
    destroyed the thing being compared -- and made any import a silent write.
    They are generated on call now, like every other table here."""
    # ---------------- S1: the conventions in full
    rows=[("C1","lowest value, any endpoint or duration","1000","reference case; target of published screening work"),
          ("C2","lowest value, \\textsf{LOEC} removed","1000","isolates an inadmissible endpoint type"),
          ("C3","lowest acute \\textsf{LC50}/\\textsf{EC50}/\\textsf{IC50}","1000","single endpoint type"),
          ("C4","median acute value over records","1000","central, record-weighted"),
          ("C5","median over species of species median acute","1000","central, species-weighted"),
          ("C6","lowest chronic \\textsf{NOEC}","10/50/100","EU TGD, factor by chronic trophic coverage"),
          ("C6$'$","lowest \\emph{apical} chronic \\textsf{NOEC} of adequate duration","10/50/100","factor from the \\emph{screened} pool"),
          ("C7a","lowest apical chronic species mean","---","species mean chronic value"),
          ("C7b","fifth percentile of a log-normal SSD","---","requires $\\geq 5$ species")]
    s1="\n".join([r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
     r"\caption{The derivation conventions in full. The assessment factor for C6 and C6$'$ follows the EU Technical Guidance ladder set by the number of trophic levels with an admissible long-term datum: 10 for three or more, 50 for two, 100 for one. For C6$'$ that count is taken from the screened pool, not from the unscreened compilation.}",
     r"\label{tab:s1}",
     r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.34\textwidth} l >{\raggedright\arraybackslash}p{0.31\textwidth}@{}}\toprule",
     r"Code & Statistic & AF & Note\\\midrule"]+
     [f"{a} & {b} & {c} & {d}\\\\" for a,b,c,d in rows]+
     [r"\bottomrule\end{tabular}\end{table}"])
    open(f"{OUT}/si_tab_s1.tex","w").write(s1+"\n")

    # ---------------- S2: HC5 confidence intervals, both coverages
    blocks=[]
    for rule in ("r10","operational"):
        out,_,_=core.build(rule,verbose=False)
        a=water.load(verbose=False)
        nm=a[["parent","name"]].drop_duplicates("parent").set_index("parent").name
        s=out.dropna(subset=["HC5_d"]).sort_values("n_sp_d")
        for _,r in s.iterrows():
            n=int(r.n_sp_d)
            lo90,hi90=core.hc5_interval(r.mu_d,r.sd_d,n,0.90)
            lo95,hi95=core.hc5_interval(r.mu_d,r.sd_d,n,0.95)
            who=nm.get(r.parent)
            if not isinstance(who,str):
                who=r"---"
            elif len(who)>30:
                who=who[:30].rsplit(" ",1)[0]+"."
            blocks.append((rule,who,n,
                           (10**r.HC5_d)*1e6,hi90-lo90,hi95-lo95))
    B=pd.DataFrame(blocks,columns=["rule","name","n_sp","hc5","w90","w95"])
    lines=[r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
     r"\caption{Confidence intervals for the fifth percentile, from noncentral-$t$ tolerance limits, widths in $\log_{10}$ units. Stacking two one-sided 95\,\% limits, as is easily done, gives an interval of \emph{joint} 90\,\% coverage; the equal-tailed two-sided 95\,\% interval is 1.20 to 1.31 times wider. Both are given. A dash in the substance column marks a substance that supports a distribution but is not in the monitoring set.}",
     r"\label{tab:s2}",
     r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.26\textwidth} rrrr@{}}\toprule",
     r"Rule & Substance & $n_\mathrm{sp}$ & HC5 (ng/L) & width 90\,\% & width 95\,\%\\\midrule"]
    for rule,grp in B.groupby("rule",sort=False):
        for _,r in grp.iterrows():
            lines.append(f"{'R.10' if r.rule=='r10' else 'operational'} & {esc(r['name'])} & {r.n_sp} & "
                         f"{r.hc5:,.3g} & {r.w90:.2f} & {r.w95:.2f}\\\\")
        lines.append(r"\addlinespace")
    lines+=[r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_s2.tex","w").write("\n".join(lines)+"\n")

    # ---------------- S3: exceedance ladder, both rules
    cols=["C1","C2","C3","C4","C5","C6","C6p","C7b"]
    lab={"C6p":"C6$'$"}
    secs=[]
    for rule in ("r10","operational"):
        res,common,sub=ladder.run(rule)
        secs.append((rule,res,len(common),len(sub)))
    lines=[r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
     r"\caption{Exceedance on the common subset: every convention applied to the substances all of them can assess. Under R.10 only two substances qualify, which is the gatekeeping result of the main text arriving a second way; the common-subset figures quoted in the main text are therefore from the operational rule.}",
     r"\label{tab:s3}",
     r"\begin{tabular}{l"+"r"*len(cols)+r"}\toprule",
     r"& "+" & ".join(lab.get(c,c) for c in cols)+r"\\\midrule"]
    for rule,res,ns,nd in secs:
        tag="R.10" if rule=="r10" else "operational"
        lines.append(r"\multicolumn{%d}{l}{\emph{%s} --- %d substances, %s detections}\\"%(len(cols)+1,tag,ns,f"{nd:,}"))
        lines.append("median criterion (ng/L) & "+" & ".join(f"{res[c][0]:,.3g}" for c in cols)+r"\\")
        lines.append(r"\% of detections above & "+" & ".join(f"{res[c][1]:.1f}" for c in cols)+r"\\")
        lines.append(r"\% of stations & "+" & ".join(f"{res[c][2]:.1f}" for c in cols)+r"\\\addlinespace")
    lines+=[r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_s3.tex","w").write("\n".join(lines)+"\n")

    # ---------------- S4: co-occurrence threshold sweep
    a=water.load(verbose=False); det=a[a.detected]
    nm=a[["parent","name"]].drop_duplicates("parent").set_index("parent").name
    o10,_,_=core.build("r10",verbose=False)
    lines=[r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
     r"\caption{The co-occurrence block is not a tuned selection. Station threshold swept from 400 to 750; for each, the substances detected at that many stations or more, the stations measuring all of them, and the resulting spans.}",
     r"\label{tab:s4}",
     r"\begin{tabular}{rrrrrrr}\toprule",
     r"Min.\ stations & Substances & Shared stations & Detections & Conc.\ span & Criterion span\\\midrule"]
    for m in (400,500,600,700,750):
        b=lead.block(det,o10,nm,m)
        if b is None: continue
        R,sh,sb=b; f=R.dropna(subset=["crit_ngL"])
        cs=f"{f.crit_ngL.max()/f.crit_ngL.min():,.0f}$\\times$" if len(f)>1 else "---"
        lines.append(f"{m} & {len(R)} & {len(sh):,} & {len(sb):,} & "
                     f"{R.med.max()/R.med.min():.1f}$\\times$ & {cs}\\\\")
    lines+=[r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_s4.tex","w").write("\n".join(lines)+"\n")

    # ---------------- S5: mediator depth sensitivity
    B2=mech._read(mech.SRC)
    byc={c:v.logTox.values for c,v in B2.groupby("test_cas")}
    rows=[]
    NSPLIT=40
    for k in (3,4,5,6,8,10):
        rng=np.random.default_rng(mech.SEED)
        cas=[c for c,v in byc.items() if len(v)>=2*k]
        tot=[];phz=[];psd=[];N=0
        for _ in range(NSPLIT):
            mn=[];hz=[];sd=[];ln=[]
            for c in cas:
                v=byc[c]; idx=rng.permutation(len(v))
                A,Bh=v[idx[:len(v)//2]],v[idx[len(v)//2:]]
                if len(A)<k or len(Bh)<2: continue
                mn.append(mech.exp_min_wor(A,k)); hz.append(Bh.mean())
                sd.append(Bh.std(ddof=1)); ln.append(np.log10(len(v)))
            D=pd.DataFrame(dict(mn=mn,hz=hz,sd=sd,ln=ln)).dropna(); N=len(D)
            tot.append(stats.spearmanr(D.ln,D.mn)[0])
            phz.append(mech.rank_partial(D.ln,D.mn,D.hz)[0])
            psd.append(mech.rank_partial(D.ln,D.mn,D.sd)[0])
        tot,phz,psd=map(np.array,(tot,phz,psd)); se=1/np.sqrt(N-4)
        rows.append((k,N,tot.mean(),psd.mean(),phz.mean(),phz.mean()-1.96*se,
                     phz.mean()+1.96*se,100*phz.mean()/tot.mean(),phz.std()))
    # F-HAZSHARE: the share the legitimate control leaves was stated three
    # times in the manuscript with two different referents -- "of the total"
    # in the Results and "of the remainder" in the Conclusions -- and with no
    # referent at all in the Abstract.  Only "of the total" is what the
    # computation produces.  Emit it, and the range over depth choices, so the
    # three sites read one number with one meaning.
    _sh = [100*r[4]/r[2] for r in rows]
    _k5 = next(100*r[4]/r[2] for r in rows if r[0] == 5)
    _MAC.extend([
        "%% hazard-control share of the TOTAL association (not of a remainder)\n",
        "\\newcommand{\\hazShare}{%.0f}\n" % _k5,
        "\\newcommand{\\hazShareLo}{%.0f}\n" % min(_sh),
        "\\newcommand{\\hazShareHi}{%.0f}\n" % max(_sh),
        "\\newcommand{\\hazPartial}{%.2f}\n" % abs(next(r[4] for r in rows if r[0] == 5)),
        "\\newcommand{\\hazN}{%s}\n" % f"{next(r[1] for r in rows if r[0]==5):,}",
    ])
    lines=[r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
     r"\caption{Controlling a mediator against controlling a confounder. The depth-equalised minimum is taken from a random half of each chemical's records and the control estimated from the held-out half, over 40 independent splits. It is the exact expectation of the minimum of $k$ records drawn without replacement, not a single realisation. Because each chemical is split in two, a chemical enters at depth $k$ only if it carries at least $2k$ records, so $n$ here is smaller at a given $k$ than in Fig.~5B, where no split is taken and the requirement is $k$ records. Conditioning on the within-chemical spread, a descendant of testing effort, removes the association; conditioning on an independently estimated hazard level leaves \hazShareLo{} to \hazShareHi{}\,\% of the total, depending on the depth $k$, with every interval excluding zero.}",
     r"\label{tab:s5}",
     r"\begin{tabular}{rrrrrrr}\toprule",
     r"$k$ & $n$ & total $\rho$ & $\mid$ spread (mediator) & $\mid$ hazard & 95\,\% CI & split s.d.\\\midrule"]
    for k,N,t_,ps,ph,lo,hi,frac,sdsp in rows:
        lines.append(f"{k} & {N:,} & ${t_:+.3f}$ & ${ps:+.3f}$ & ${ph:+.3f}$ & "
                     f"$[{lo:+.3f},\\,{hi:+.3f}]$ & {sdsp:.3f}\\\\")
    lines+=[r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_s5.tex","w").write("\n".join(lines)+"\n")
    print("wrote si_tab_s1..s5.tex")





# ---------------- S6: robustness (appended; regenerated by run_all)
def s6():
    assert _MAC, "s1to5() must run before s6(): the S5 macros are built there"
    import robust
    rows=robust.run("operational")
    CS=[("C6p","C6$'$"),("C7a","C7a"),("C7b","C7b")]
    _nodrop=all(rows[c].get("pfas_drop",0)==0 for c,_ in CS)
    L=[r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
       r"\caption{Robustness of the correlation between testing effort and the "
       r"derived criterion, under the operational duration rule where enough "
       r"substances survive to estimate it. Testing effort is distinct tests throughout the manuscript; the row and collapsed frames are the "
       r"declared sensitivities. Partial correlations control the "
       r"stated variable. None of the checks removes the association, and no "
       r"sign changes between frames.}",
       r"\label{tab:rob}",
       r"\begin{tabular}{@{}p{62mm} rrr@{}}\toprule",
       r"check & " + " & ".join(n for _,n in CS) + r"\\\midrule",
       "as reported (effort as distinct tests) & " + " & ".join(f"${rows[c]['raw']:+.3f}$" for c,_ in CS) + r"\\",
       "partial, controlling carbon number & " +
         " & ".join(f"${rows[c]['nC']:+.3f}$" for c,_ in CS) + r"\\",
       "effort as result rows & " +
         " & ".join(f"${rows[c]['rows']:+.3f}$" for c,_ in CS) + r"\\",
       "effort as rows collapsed on test, endpoint and concentration & " +
         " & ".join(f"${rows[c]['dedup']:+.3f}$" for c,_ in CS) + r"\\",
       ("excluding three non-OECD-PFAS parents"
        + ("$^{\\dagger}$" if _nodrop else "") + " & ") +
         " & ".join(f"${rows[c]['pfas_only']:+.3f}$" for c,_ in CS) + r"\\",
       r"bootstrap 95\,\% CI (5000 draws) & " +
         " & ".join(f"$[{rows[c]['ci'][0]:+.2f},{rows[c]['ci'][1]:+.2f}]$" for c,_ in CS)
         + r"\\",
       "substances & " + " & ".join(str(rows[c]['n']) for c,_ in CS) + r"\\",
       r"\bottomrule\end{tabular}", ""]
    if _nodrop:
        L.append(r"\footnotesize $^{\dagger}$ None of the three survives the "
                 r"apical and duration screens, so this row drops no substance "
                 r"and reproduces the baseline by construction.")
    L.append(r"\end{table}")
    open(f"{OUT}/si_tab_s6.tex","w").write("\n".join(L)+"\n")
    # the prose numbers, for the SI text
    w=rows["water"]; p=rows["perm"]
    open(f"{OUT}/si_rob_numbers.tex","w").write(
      "%% generated -- do not edit\n"
      f"\\newcommand{{\\robFull}}{{{w['full']:.2f}}}\n"
      f"\\newcommand{{\\robLo}}{{{w['lo']:.2f}}}\n"
      f"\\newcommand{{\\robHi}}{{{w['hi']:.2f}}}\n"
      f"\\newcommand{{\\robTop}}{{{w['top']:.1f}}}\n"
      f"\\newcommand{{\\robStation}}{{{w['station']:.2f}}}\n"
      f"\\newcommand{{\\robAllres}}{{{w['allres']:.2f}}}\n"
      f"\\newcommand{{\\robNoise}}{{{w['noise']:.2f}}}\n"
      f"\\newcommand{{\\robNoiseLo}}{{{w['noise_lo']:.2f}}}\n"
      f"\\newcommand{{\\robNoiseHi}}{{{w['noise_hi']:.2f}}}\n"
      f"\\newcommand{{\\permN}}{{{p[0]}}}\n"
      f"\\newcommand{{\\permObs}}{{{p[1]:+.3f}}}\n"
      f"\\newcommand{{\\permSd}}{{{p[3]:.2f}}}\n"
      f"\\newcommand{{\\permLo}}{{{p[4]:+.2f}}}\n"
      f"\\newcommand{{\\permHi}}{{{p[5]:+.2f}}}\n"
      f"\\newcommand{{\\permP}}{{{p[6]:.4f}}}\n"
      + _hc5_width_macros() + _routegap_macros() + _lcl_macros() + _rarefy_macros() + _station_macros() + _infeasible_macros() + _lead_macros() + _baseset_macros() + _auc_macros()
      + "".join(_MAC))
    print("wrote si_tab_s6.tex and si_rob_numbers.tex")



# ---------------- S-INV: WQP characteristic-name inventory (was si_tab1,
# a hand-maintained v2 artefact outside the pipeline: 20 of its 66 rows
# carried pre-QC detection counts and it summed to 22,255 against 20,998,
# and it omitted two retired names present in the frame).
def inventory():
    import water
    a = water.load(verbose=False)
    # water.load() inner-joins the character map on name->parent, so the CAS
    # column survives only for names that resolve.  The table then printed
    # "nan" for the other 25 and the 18 CAS numbers that carry no ECOTOX
    # record -- the ones the surrounding text counts -- were invisible in the
    # table meant to document them.  CAS is read from the map itself.
    cm = pd.read_csv(water.CHARMAP)
    g = (a.groupby("CharacteristicName")
           .agg(n=("detected", "sum"), parent=("parent", "first"))
           .reset_index())
    g = g.merge(cm[["char", "cas"]].rename(columns={"char": "CharacteristicName"}),
                on="CharacteristicName", how="left")
    g["n"] = g.n.astype(int)
    g["mapped"] = g.parent.notna()
    g = g.sort_values(["mapped", "n"], ascending=[False, False])
    tot, nmap, nun = int(g.n.sum()), int(g.mapped.sum()), int((~g.mapped).sum())
    ncas = int(cm.cas.dropna().nunique())
    ncas_norec = int(cm[cm.parent.isna()].cas.dropna().nunique())
    npar = int(cm.parent.nunique())
    assert tot == int(a.detected.sum()), "inventory does not sum to the frame"
    assert g.cas.notna().all(), "a characteristic name lost its CAS number"

    def esc(t):
        return str(t).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")

    L = [r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{72mm} l r c@{}}",
         r"\caption{Water Quality Portal characteristic names with quantified "
         r"data after the unit and quality-control screens, the CAS registry "
         r"number assigned to each, the number of detections, and whether the "
         rf"name resolves to a parent structure with an aquatic toxicity "
         rf"record. The {len(g)} names carry {tot:,} detections in total, "
         rf"{nmap} of them resolving to a parent structure and {nun} not. "
         rf"They carry {ncas} distinct CAS numbers between them, which "
         rf"collapse to {npar} parent structures; the remaining "
         rf"{ncas_norec} CAS numbers appear in no ECOTOX aquatic toxicity "
         rf"record, so no criterion can be derived for them. "
         r"The detection column sums "
         r"to the frame by construction.}",
         r"\label{tab:inv}\\", r"\toprule",
         r"Characteristic name & CAS & $n$ & resolves \\", r"\midrule",
         r"\endfirsthead", r"\toprule",
         r"Characteristic name & CAS & $n$ & resolves \\", r"\midrule",
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in g.iterrows():
        _c = esc(r.cas) if pd.notna(r.cas) else "---"
        L.append(f"\\sloppy {esc(r.CharacteristicName)} & {_c} "
                 f"& {r.n:,} & {'yes' if r.mapped else 'no'} \\\\")
    L.append(r"\end{longtable}")
    open(f"{OUT}/si_tab_inv.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote si_tab_inv.tex: {len(g)} names, {tot:,} detections, "
          f"{nmap} resolved / {nun} unresolved")


# ---------------- S-FIT: SSD fit diagnostics (was si_tab2, a hand-maintained
# v2 artefact outside the pipeline whose species counts were the UNFILTERED
# ones -- 57 and 39 for PFOS and PFOA against the screened 47 and 34 -- so the
# table, the surrounding prose and fig_si_qq.png described three different sets.
# The screen and the substance count are now written from the data, not the
# caption, and the species counts are asserted against core.build.
def fitdiag(rule="operational"):
    """F-FITCOLS: a regeneration of this table dropped the Anderson-Darling
    statistic, its critical value, the Shapiro-Wilk p and the probability-plot
    R^2, while four claims in si.tex went on arguing from them.  All four are
    restored, from core.ssd_diagnostics(), which refits on the same screened
    pool that produces mu_d/sd_d.  CAS is carried because the parent SMILES was
    being silently truncated to 34 characters, which made rows ambiguous."""
    import core
    o, d, _ = core.build(rule, verbose=False)
    diag = core.ssd_diagnostics(rule).set_index("parent")
    def _cas(v):
        t = str(v).strip()
        return t if not t.isdigit() or len(t) < 5 else f"{t[:-3]}-{t[-3:-1]}-{t[-1]}"
    cas = (d.dropna(subset=["parent"]).groupby("parent").test_cas
             .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "")
             .map(_cas))
    e = o[o.n_sp_d.fillna(0) >= 5].copy().sort_values("n_sp_d", ascending=False)
    rulename = "strict ECHA R.10" if rule == "r10" else "operational"
    cap = (
        r"\caption{Fitted log-normal species sensitivity distributions under "
        "the " + rulename + " duration rule, for every substance with at least "
        r"five species after the apical and chronic-duration screen. $n$ is the "
        r"screened species count, which is what the fifth percentile's sample "
        "size is. " + str(len(e)) + r" substances qualify. Values are "
        r"$\log_{10}$ \si{\milli\gram\per\litre}; the fifth percentile is "
        r"$\hat\mu + z_{0.05}\hat\sigma$. $A^2$ is the Anderson--Darling "
        r"statistic and $A^2_{0.05}$ its 5\,\% critical value for the normal "
        r"case, so $A^2 > A^2_{0.05}$ is a rejection; $R^2$ is the squared "
        r"correlation of the normal quantile plot.}")
    L = [r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
         cap,
         r"\label{tab:fit}",
         r"\begin{tabular}{@{}l r r r r r r r@{}}\toprule",
         r"CAS & $n$ species & $\hat\mu$ & $\hat\sigma$ & HC5 & $A^2$ & "
         r"$A^2_{0.05}$ & $R^2$ \\\midrule"]
    for _, r in e.iterrows():
        has = r.parent in diag.index
        g = diag.loc[r.parent] if has else None
        a2 = ("$%.3f$" % g.AD) if has else "---"
        a2c = ("$%.3f$" % g.AD_crit5) if has else "---"
        r2 = ("$%.2f$" % g.R2) if has else "---"
        # The daggers must agree with the A^2 columns printed beside them and
        # with the SI prose, both of which argue from Anderson-Darling.  They
        # were keyed to the Shapiro-Wilk p, which marks a DIFFERENT set of
        # four substances (it drops PFOS and adds a six-species parent).
        mark = r"\,$^{\dagger}$" if (has and bool(g.AD_reject5)) else ""
        L.append("%s%s & %d & $%+.2f$ & $%.2f$ & $%+.2f$ & %s & %s & %s \\\\"
                 % (cas.get(r.parent, ""), mark, int(r.n_sp_d), r.mu_d,
                    r.sd_d, r.HC5_d, a2, a2c, r2))
    nsw = int(diag.AD_reject5.sum())
    L += [r"\bottomrule\end{tabular}", "",
          r"\footnotesize $^{\dagger}$ Anderson--Darling rejects normality at the "
          r"5\,\% level (" + str(nsw) + " substances).", r"\end{table}"]
    open(OUT + "/si_tab_fit.tex", "w").write("\n".join(L) + "\n")
    ns = sorted(e.n_sp_d.astype(int).tolist(), reverse=True)
    nrej = int(diag.AD_reject5.sum())
    print("wrote si_tab_fit.tex: rule=%s, %d substances, AD rejects %d, "
          "SW rejects %d, min R2 %.3f"
          % (rule, len(e), nrej, nsw, diag.R2.min()))
    return ns


# ---------------- S-SENS: min-species / distributional-form sensitivity.
# Was hard-coded in si.tex with no generator and computed on the v2 4,744-row
# frame and the unfiltered species set (n = 24/16/16/8/5, rho = -0.666/-0.691/
# -0.700/-0.286/-0.100).  Recomputed on the v3 screened pool.
def sens(rule="operational", nboot=4000):
    import core
    from scipy import stats as st
    d = core.load(rule)
    sel = d[d.apical & d.chronic_ok & (d.conc1_type == "A")
            & d.ep.isin(["NOEC", "NOEL", "MATC", "EC10"])]
    smcv = sel.groupby(["parent", "latin_name"]).logTox.mean().reset_index()
    # F-SENSEFF: this used the result-row count while the rest of the
    # manuscript defines testing effort as distinct tests, so Table S8 printed
    # -0.703 where Table S10's baseline is -0.710 and the main text quoted
    # both for the same quantity.  One definition, from core.
    eff = {k: float(v) for k, v in
           d.groupby("parent").test_id.nunique().items()}
    rng = np.random.default_rng(20261122)
    out = []
    for minsp in (3, 5, 8, 10):
        for form in ("log-normal", "log-logistic"):
            if minsp != 5 and form == "log-logistic":
                continue
            rows = []
            for p, v in smcv.groupby("parent"):
                x = v.logTox.values
                if len(x) < minsp:
                    continue
                mu, sd = x.mean(), x.std(ddof=1)
                if form == "log-normal":
                    hc5 = mu + core.Z05 * sd if hasattr(core, "Z05") else mu - 1.6449 * sd
                else:                     # logistic with matched variance
                    hc5 = mu - (np.pi / np.sqrt(3)) * sd * 0.7368825
                e = eff.get(p)
                if e is None or e <= 0:
                    continue
                rows.append((float(np.log10(e)), float(hc5)))
            D = pd.DataFrame(rows, columns=["leff", "hc5"]).dropna()
            if len(D) < 3:
                out.append((minsp, form, len(D), np.nan, np.nan, np.nan)); continue
            r = st.spearmanr(D.leff, D.hc5)[0]
            bs = []
            for _ in range(nboot):
                Z = D.values[rng.integers(0, len(D), len(D))]
                bs.append(st.spearmanr(Z[:, 0], Z[:, 1])[0])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            out.append((minsp, form, len(D), r, lo, hi))
    L = [r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
         r"\caption{Spearman correlation between testing effort and the SSD "
         r"fifth percentile, under varying minimum species requirements and "
         rf"distributional form, on the screened pool under the "
         rf"{'strict ECHA R.10' if rule=='r10' else 'operational'} duration "
         r"rule. Testing effort is distinct tests, as throughout. "
         r"Bootstrap 95\,\% confidence intervals from "
         rf"{nboot} resamples.}}",
         r"\label{tab:sens}",
         r"\begin{tabular}{@{}l l r r l@{}}\toprule",
         r"Minimum species & Distribution & substances & $\rho$ & 95\,\% CI \\\midrule"]
    for minsp, form, n, r, lo, hi in out:
        ci = "---" if np.isnan(r) else f"$[{lo:+.3f}, {hi:+.3f}]$"
        rv = "---" if np.isnan(r) else f"${r:+.3f}$"
        L.append(f"{minsp} & {form} & {n} & {rv} & {ci} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_sens.tex", "w").write("\n".join(L) + "\n")
    for o in out:
        print(f"  minsp={o[0]:2d} {o[1]:13s} n={o[2]:3d} rho={o[3]:+.3f}")
    return out


# ---------------- S-PERSUB: per-substance measured values, criteria and
# exceedance.  Was the last \input with no generator in v3 (tab_persub.tex),
# and its caption cross-referenced a Table 1 count that no longer existed.
_PERSUB_RENAME = {
    "Perfluorovaleric acid": "Perfluoropentanoic acid",
}


def _canon(nm):
    """Canonical substance name for printed tables.

    Two defects lived here: the ECOTOX synonym "Perfluorovaleric acid" was
    printed where the body and Figure 1 say "Perfluoropentanoic acid", and a
    hard [:30] slice truncated names mid-word ("6:2 Fluorotelomer sulfonate
    ac").  Names are now mapped, then abbreviated only at a word boundary
    with an explicit ellipsis so a truncation is visible as one.
    """
    nm = _PERSUB_RENAME.get(nm, nm)
    if len(nm) <= 33:
        return nm
    cut = nm[:32].rsplit(" ", 1)[0]
    return cut + "\\,\\ldots"


def persub(rule="operational"):
    import core, water
    a = water.load(verbose=False)
    det = a[a.detected]
    o, d, _ = core.build(rule, verbose=False)
    T, _ = water.substance_table(a, rule, verbose=False)
    name = (a[["parent", "name"]].dropna().drop_duplicates("parent")
            .set_index("parent").name)
    C = o.set_index("parent")
    rows = []
    for p, g in det[det.parent.notna()].groupby("parent"):
        if p not in C.index:
            continue
        r = C.loc[p]
        c1 = (10 ** d[d.parent == p].logTox.min()) * 1e6 / 1000.0 \
             if (d.parent == p).any() else np.nan
        # C6 is a CRITERION, not the pooled minimum: the tabulated factor has
        # to be divided out before the unit conversion.  Printing C6_pub raw
        # put log10 mg/L values (negative numbers) under a ng/L heading and
        # made every exceedance 100 %.
        a6 = r.get("AF_C6_pub")
        c6 = r.get("C6_pub")
        c6 = (10 ** (c6 - np.log10(a6))) * 1e6 \
            if (pd.notna(c6) and pd.notna(a6)) else np.nan
        c7 = r.get("HC5_d")
        c7 = (10 ** c7) * 1e6 if pd.notna(c7) else np.nan
        ex = lambda t: 100 * (g.ngL > t).mean() if pd.notna(t) else np.nan
        rows.append(dict(nm=_canon(str(name.get(p, p))), n=len(g),
                         sites=g.MonitoringLocationIdentifier.nunique(),
                         med=g.ngL.median(), recs=int(r.n_records),
                         c1=c1, c6=c6, c7=c7,
                         e1=ex(c1), e6=ex(c6), e7=ex(c7)))
    D = pd.DataFrame(rows).sort_values("n", ascending=False)
    n7 = int(D.c7.notna().sum())
    nscr = int(o.n_sp_d.fillna(0).ge(5).sum())
    f = lambda v: "---" if pd.isna(v) else (f"{v:,.3g}")
    e = lambda v: "---" if pd.isna(v) else f"{v:.1f}"
    L = [r"\begin{table}[p]\centering\tiny\setlength{\tabcolsep}{2.2pt}",
         r"\caption{Measured concentrations, derived criteria and exceedance by "
         r"substance. Measurements are Water Quality Portal results for United "
         r"States stream and lake stations queried from 1 January 2015. "
         r"The C1 and C6 columns are computed on the unscreened pool; the "
         r"C7b column is the fifth percentile of the distribution fitted to "
         r"the pool that survives the apical and chronic-duration screen, so "
         rf"it covers {n7} of these substances against the {nscr} that "
         rf"survive the screen across the whole set under the "
         rf"{'strict ECHA R.10' if rule=='r10' else 'operational'} rule. "
         r"Criteria in \si{\nano\gram\per\litre}; exceedance is the percentage "
         r"of that substance's own detections above its own criterion. "
         r"}",
         r"\label{tab:persub}",
         r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{37mm} rrr r rrr rrr@{}}\toprule",
         r"& \multicolumn{3}{c}{measured} & tox & "
         r"\multicolumn{3}{c}{criterion} & \multicolumn{3}{c}{exceedance (\%)}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"substance & $n$ & sites & median & recs & C1 & C6 & C7b & C1 & C6 & C7b\\\midrule"]
    for _, r in D.iterrows():
        L.append(f"\\textsf{{{r.nm}}} & {r.n:,} & {r.sites:,} & {r.med:.2f} & "
                 f"{r.recs:,} & {f(r.c1)} & {f(r.c6)} & {f(r.c7)} & "
                 f"{e(r.e1)} & {e(r.e6)} & {e(r.e7)}\\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/tab_persub.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote tab_persub.tex: {len(D)} substances, C7b on {n7}, screened {nscr}")




# ---------------- S-AFDEC: assessment-factor decomposition.  Was hard-coded
# in si.tex against the superseded exposure (records), the pre-F-AF1 factor
# pool and n = 62.  The identity is asserted in core.af_decomposition.
def afdec(rule="r10"):
    import core
    o, _, _ = core.build(rule, verbose=False)
    r = core.af_decomposition(o)
    ex = "distinct tests" if core.EFFORT == "n_tests" else "records"
    fmt = lambda p: f"${p:.1g}$".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + ("}$" if "e" in f"{p:.1g}" else "$")
    def pf(p):
        m, e = f"{p:.1e}".split("e")
        return f"${float(m):.1f}\\times10^{{{int(e)}}}$"
    L = [r"\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
         r"\caption{Decomposition of the effort dependence of the EU-TGD "
         r"criterion, over the "
         rf"{r['n']} substances that have both a chronic \textsf{{NOEC}} and a "
         rf"factor under the "
         rf"{'strict ECHA R.10' if rule=='r10' else 'operational'} duration "
         rf"rule. Slopes are ordinary least squares against $\log_{{10}}$ "
         rf"{ex}. Because the criterion is the numerator divided by the "
         rf"factor, the three slopes satisfy "
         rf"${r['b_crit']:.3f} = {r['b_noec']:.3f} - ({r['b_af']:.3f})$ "
         r"exactly; the decomposition reports not that they sum but how little "
         r"of the sum the factor carries.}",
         r"\label{tab:afdec}",
         r"\begin{tabular}{@{}l r r@{}}\toprule",
         rf"Quantity & slope per decade of {ex} & $p$ \\\midrule",
         rf"$\log_{{10}}$ lowest chronic \textsf{{NOEC}} & ${r['b_noec']:.3f}$ & {pf(r['p_noec'])} \\",
         rf"$\log_{{10}}$ assessment factor            & ${r['b_af']:.3f}$ & {pf(r['p_af'])} \\",
         rf"$\log_{{10}}$ derived criterion            & ${r['b_crit']:.3f}$ & --- \\",
         r"\bottomrule\end{tabular}\end{table}"]
    open(f"{OUT}/si_tab_afdec.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote si_tab_afdec.tex: n={r['n']}, offset={100*r['offset']:.0f}%")
    return r


if __name__=="__main__":
    import tab1          # F-REGEN: Table 1 regenerates with the rest
    tab1.build()
    s1to5()
    s6()
    inventory()
    fitdiag("operational")
    sens("operational")
    persub("operational")
    afdec("r10")
