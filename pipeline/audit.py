"""Cross-document audit: does the manuscript say what the pipeline computes?"""
import shutil, sys, subprocess
import os as _os, sys, re
CODE = _os.environ.get("CODE_DIR",
                       _os.path.dirname(_os.path.abspath(__file__)))
# The cross-document audit reads the manuscript sources and the
# compiled PDFs, which are not part of the deposited repository.
PAPER = _os.environ.get("PAPER_DIR", _os.path.join(
    _os.path.dirname(CODE), "manuscript"))
sys.path.insert(0, CODE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import core, water, mech, lead, robust

MAIN=open(PAPER + "/main.tex").read()
SI=open(PAPER + "/si.tex").read()
ok=bad=0
def chk(name, cond, detail=""):
    global ok,bad
    if cond: ok+=1; print(f"  PASS  {name}")
    else:    bad+=1; print(f"  FAIL  {name}   {detail}")
def says(pat, where=MAIN):
    return re.search(pat, where) is not None

print("="*84); print("CROSS-DOCUMENT AUDIT"); print("="*84)

a=water.load(verbose=False); det=a[a.detected]
o10,d10,_=core.build("r10",verbose=False)
oop,_,_   =core.build("operational",verbose=False)

# --- water frame
chk("117,034 retrieved results", says(r"117\\,034"))
chk("101,908 after screens",     says(r"101\\,908"))
chk("20,998 detections",         says(r"20\\,998") and len(det)==20998, f"computed {len(det)}")
chk("80,910 non-detections",     says(r"80\\,910") and int((~a.detected).sum())==80910)
chk("1523 stations",             says(r"1523") and a.MonitoringLocationIdentifier.nunique()==1523)
chk("22 parent substances",      a.parent.nunique()==22)

# --- lead result
R,shared,sub=lead.block(det,o10,a[["parent","name"]].drop_duplicates("parent").set_index("parent").name,700)
chk("7 co-occurring substances", len(R)==7 and says(r"Seven substances\s+are each detected"))
chk("500 shared stations",       len(shared)==500 and says(r"500\s*\n?stations measure all seven|and 500\s*\n?stations"))
chk("11,394 detections in block",len(sub)==11394 and says(r"11\\,394"))
chk("54.3% of detections",       abs(100*len(sub)/len(det)-54.3)<0.1 and says(r"54\.3"))
fin=R.dropna(subset=["crit_ngL"])
span=fin.crit_ngL.max()/fin.crit_ngL.min()
chk("criterion span 7,389x",     abs(span-7389)<20 and says(r"7389"), f"computed {span:,.0f}")
chk("conc span 2.7-fold",        abs(R.med.max()/R.med.min()-2.7)<0.05 and says(r"2\.7-fold"))
chk("records 15-3184",           R.n_rec.min()==15 and R.n_rec.max()==3184 and says(r"15 to 3184"))

# --- derivability
T,_=water.substance_table(a,"r10",verbose=False)
nine=int((T.route=="none").sum()); sh=100*T.loc[T.route=="none","n_det"].sum()/T.n_det.sum()
chk("9 of 22 no derivable PNEC", nine==9 and says(r"[Nn]ine of the 22"), f"computed {nine}")
# F-FRAME22: every occurrence share is now on the single 20,998 denominator
TOT=int(a.detected.sum())
_par=a.parent.notna()
_npar_det=int(a[~_par].detected.sum()); _par_det=int(a[_par].detected.sum())
_dn=lambda rt: int(T.loc[T.route==rt,"n_det"].sum())
chk("parent/no-parent split",    int((_par).sum())==59990 and int((~_par).sum())==41918
                                 and says(r"59\\,990") and says(r"41\\,918"),
                                 f"{int(_par.sum())}/{int((~_par).sum())}")
chk("19,456 / 1542 detections",  _par_det==19456 and _npar_det==1542
                                 and says(r"19\\,456") and says(r"1542 detections"))
chk("25 unmapped names",         a[~_par].CharacteristicName.nunique()==25 and says(r"25 further PFAS")
                                 and says(r"25 unmapped names"))
chk("1044 detecting stations",   det.MonitoringLocationIdentifier.nunique()==1044 and says(r"1044"))
chk("no-chronic share 31.7%",    abs(100*(TOT-_dn("chronic"))/TOT-31.7)<0.1 and says(r"31\.7"),
                                 f"computed {100*(TOT-_dn('chronic'))/TOT:.2f}")
chk("none-route share 12.6%",    abs(100*_dn("none")/TOT-12.6)<0.1 and says(r"12\.6"),
                                 f"computed {100*_dn('none')/TOT:.2f}")
chk("acute-route share 11.7%",   abs(100*_dn("acute")/TOT-11.7)<0.1 and says(r"11\.7"),
                                 f"computed {100*_dn('acute')/TOT:.2f}")
chk("unmapped share 7.3%",       abs(100*_npar_det/TOT-7.3)<0.1 and says(r"7\.3"),
                                 f"computed {100*_npar_det/TOT:.2f}")
chk("no-base-set share 20.0%",   abs(100*(_dn("none")+_npar_det)/TOT-20.0)<0.1 and says(r"20\.0"),
                                 f"computed {100*(_dn('none')+_npar_det)/TOT:.2f}")
chk("10/3/9 route split",        int((T.route=="chronic").sum())==10 and int((T.route=="acute").sum())==3
                                 and says(r"Ten take the\s+chronic route") and says(r"three the acute base set"))
# F-PFPEA: the acute-route substance is derivable; only PFHpA is not
chk("PFPeA acute route stated",  says(r"drops onto the acute\s+base set") and says(r"factor of 1000 rather than 10"))
chk("acute fallback in Conclusions", says(r"the five of the seven for which")
                                 and says(r"falls back from the long-term route onto the acute")
                                 and says(r"one receives no\s+criterion at all")
                                 and not says(r"two of the seven receive none"))
chk("PFHpA 701 stations",        det[det.name.str.contains("heptanoic", na=False)]
                                    .MonitoringLocationIdentifier.nunique()==701
                                 and says(r"detected at 701 stations"),
                                 f"computed {det[det.name.str.contains(chr(104)+chr(101)+chr(112)+chr(116)+chr(97)+chr(110)+chr(111)+chr(105)+chr(99), na=False)].MonitoringLocationIdentifier.nunique()}")
elig=o10.n_sp_d.fillna(0)>=5
rk=sorted(o10[core.EFFORT].rank(ascending=False,method="min")[elig].astype(int))
mw=stats.mannwhitneyu(o10[core.EFFORT][elig],o10[core.EFFORT][~elig],alternative="greater")
auc=mw.statistic/(elig.sum()*(~elig).sum())
chk("ranks 1,2,3,14,25",         rk==[1,2,3,14,25] and says(r"1st, 2nd, 3rd,\s*\n?14th and 25th"), str(rk))
# superseded by F-AUCCI below: the text now carries \aucVal, not a literal,
# and the statistic is computed on the declared exposure variable (0.924).
chk("AUC is macro-fed, not a literal",
    abs(auc - 0.924) < 0.005 and says(r"\\aucVal\{\}"), f"{auc:.3f}")
aucop=None
eo=oop.n_sp_d.fillna(0)>=5
m2=stats.mannwhitneyu(oop[core.EFFORT][eo],oop[core.EFFORT][~eo],alternative="greater")
aucop=m2.statistic/(eo.sum()*(~eo).sum())
chk("operational AUC 0.90",      abs(aucop-0.897)<0.01 and says(r"0\.90"), f"{aucop:.3f}")

# --- correlations quoted in sec:screen
def crit_rho(out,col):
    s=out.dropna(subset=[col,f"AF_{col}"])
    return stats.spearmanr(np.log10(s[core.EFFORT]), s[col]-np.log10(s[f"AF_{col}"]))
r,p=crit_rho(o10,"C6_prime")
chk("C6' R.10 rho -0.209 p=0.338", abs(r+0.209)<0.004 and abs(p-0.338)<0.006 and says(r"-0\.209"), f"{r:+.3f} p={p:.3f}")
s5=o10.dropna(subset=["HC5_d"]); r5,p5=stats.spearmanr(np.log10(s5[core.EFFORT]),s5.HC5_d)
chk("C7b R.10 rho -0.600 n=5",     abs(r5+0.600)<0.01 and len(s5)==5 and says(r"-0\.600"))
rop,pop=crit_rho(oop,"C6_prime")
chk("C6' oper rho -0.506",         abs(rop+0.506)<0.004 and says(r"-0\.506"), f"{rop:+.3f}")
sop=oop.dropna(subset=["HC5_d"]); rb,pb=stats.spearmanr(np.log10(sop[core.EFFORT]),sop.HC5_d)
chk("C7b oper rho -0.710 n=13",    abs(rb+0.710)<0.004 and len(sop)==13 and says(r"-0\.710"))
sc1=o10.dropna(subset=["C6_pub"]); rc1,_=stats.spearmanr(np.log10(sc1[core.EFFORT]),sc1.C6_pub)
_c1 = core.c1_effort(o10, d10)
chk("C1 rho -0.696 generated",     abs(_c1[core.EFFORT][0]+0.696)<0.004
                                   and _c1[core.EFFORT][2]==87 and says(r"-0\.696"),
                                   f"{_c1[core.EFFORT][0]:+.3f}")
sap=o10.dropna(subset=["C6_ap","AF_C6_ap"]); rap,_=crit_rho(o10,"C6_ap")
chk("C6 apical rho -0.679 n=47",   abs(rap+0.679)<0.004 and len(sap)==47 and says(r"-0\.679"))

# --- composition of the extract (the frame the effort variable is counted on)
dd=core.load("r10")
chk("12,349 result rows",        len(dd)==12349 and says(r"12\\,349"))
chk("87 substances",             dd.parent.nunique()==87 and says(r"\b87\b"))
chk("2182 tests",                dd.test_id.nunique()==2182 and says(r"2182"))
chk("5722 NOEC",                 int((dd.ep=="NOEC").sum())==5722 and says(r"5722"))
chk("4678 LOEC",                 int((dd.ep=="LOEC").sum())==4678 and says(r"4678",MAIN+SI))
chk("956 acute",                 int(dd.ep.isin(core.ACUTE).sum())==956 and says(r"956"))
chk("GEN 3515",                  int((dd.effect.astype(str).str.strip()=="GEN").sum())==3515
                                 and says(r"3515"))
chk("apical 33.2%",              abs(100*dd.apical.mean()-33.2)<0.05 and says(r"33\.2"))
_k=["test_id","ep","conc1_mean","conc1_unit"]
chk("5631 collapsed rows",       len(dd.drop_duplicates(subset=_k))==5631
                                 and says(r"5631",MAIN+SI))

# --- mechanism
S,g,B=mech.build()
chk("n=1175 chemicals",          len(S)==1175 and says(r"1175"))
rs,_=stats.spearmanr(S.ls,S.hc5); rr,_=stats.spearmanr(S.lr,S.hc5)
chk("rho species -0.241",        abs(rs+0.241)<0.001 and says(r"-0\.241"))
chk("rho records -0.156",        abs(rr+0.156)<0.001 and says(r"-0\.156"))
p1=mech.rank_partial(S.ls,S.hc5,S.lr)[0]; p2=mech.rank_partial(S.lr,S.hc5,S.ls)[0]
chk("partial species -0.205",    abs(p1+0.205)<0.002 and says(r"-0\.205"))
chk("partial records +0.088",    abs(p2-0.088)<0.002 and says(r"\+0\.088"))
chk("pooled +0.116 CI",          says(r"\+0\.116"))
chk("pooled -0.187 CI",          says(r"-0\.187"))
A=np.c_[np.ones(len(S)),S.ls]; bt=np.linalg.lstsq(A,S.hc5.values,rcond=None)[0][1]
chk("species slope -1.21",       abs(bt+1.2084)<0.002 and says(r"-1\.21"))
chk("c4 4.7%",                   says(r"4\.7"))
chk("rarefaction 13.5%",         says(r"13\.5"))
chk("rarefied -1.58 / -1.68",    says(r"-1\.58") and says(r"-1\.68"))

# --- things that must be ABSENT
# F-DUPCHK: two presence checks sat here under an "absent" heading, with
# conditions already asserted above -- "20,998 detections in text" was
# character-identical to the check at the top of this file, and the 500-station
# one was a weaker form of it.  Duplicates inflate the count that the cover
# letter would cite without testing anything further, so they are removed
# rather than renamed.

print("\n-- defective values that must not appear --")
for lab,pat in [("rho -0.590",r"-0\.590"),("rho -0.681",r"-0\.681"),("sigma share 63%",r"63\\,\\%"),
                ("partial -0.004",r"-0\.004"),("523-fold",r"523-fold"),("98.8%",r"98\.8"),
                ("rho +0.783",r"\+0\.783"),("20,682",r"20\\,682"),("1180 stations",r"1180"),
                ("'guidance actually specifies'",r"guidance actually specifies"),
                ("'exhaust the association'",r"exhaust the association"),
                ("24.2%",r"24\.2"),("'exact 95'",r"exact 95"),
                ("501 shared stations (now 500)",r"\b501\b"),
                ("21,024 detections (now 20,998)",r"21\\,024"),
                ("11,960 detections (now 11,394)",r"11\\,960"),
                ("571 stations (now 500)",r"\b571\b"),
                ("14,778x span (now 7,389)",r"14\\,778"),
                ("'four orders' unqualified",r"receive criteria four orders"),
                ("1918 NOEC (now 5722)",r"1918"),
                ("1811 LOEC (now 4678)",r"1811"),
                ("3501 GEN rows (now 3515)",r"3501"),
                ("29.8% apical (now 33.2%)",r"29\.8"),
                ("59% top-two share (now 47%)",r"59\\,\\% of all records")]:
    chk(f"absent: {lab}", not says(pat,MAIN+SI))

# --- F-FRAME22 / F-PFPEA: superseded statements must be gone
for pat,label in [(r"13\.6\\,\\% of detections", "13.6% on the parented denominator"),
                  (r"702 stations",                   "702 stations (now 701)"),
                  (r"one of the seven receives none", "'one of the seven receives none'"),
                  (r"procedure returns nothing at all", "'returns nothing at all' for PFPeA"),
                  (r"European route to all seven",    "'route to all seven'"),
                  (r"101\\,908 results for 22 parent", "'101,908 results for 22 parent substances'"),
                  (r"more than 700 of the 1523",      "'more than 700 of the 1523 stations'")]:
    chk(f"absent: {label}", not says(pat))


# --- SI must not carry the superseded water frame (F-SIFRAME)
for pat,label in [(r"22\\,255", "SI 22,255 detections (now 20,998)"),
                  (r"89\\,857", "SI 89,857 non-detections (now 80,910)"),
                  (r"19\.0\\,\\%, carry a positive", "SI 19.0% detected share (now 20.6%)")]:
    chk(f"absent in SI: {label}", not says(pat, SI))
chk("SI states 101,908 and 20,998", says(r"101\\,908", SI) and says(r"20\\,998", SI)
                                    and says(r"80\\,910", SI))

print("="*84)


# --- F-R10NULL: all three screened R.10 routes stated, with their footing
from scipy import stats as _st
def _p(r, n):
    t = r * np.sqrt((n - 2) / (1 - r * r)); return 2 * _st.t.sf(abs(t), n - 2)
_rr = robust.run("r10")
for key, rho, n, pv, lab in [("C6p", -0.209, 23, 0.338, "C6'"),
                             ("C7a", -0.474, 29, 0.010, "C7a"),
                             ("C7b", -0.600,  5, 0.28,  "C7b")]:
    x = _rr[key]
    chk(f"R.10 {lab} rho/n/p as printed",
        abs(x["raw"] - rho) < 0.002 and x["n"] == n
        and abs(_p(x["raw"], x["n"]) - pv) < 0.01,
        f"computed {x['raw']:+.3f} n={x['n']} p={_p(x['raw'],x['n']):.3f}")
chk("R.10 non-significance stated explicitly",
    says(r"C7a reaches conventional\s+significance")
    and says(r"we claim no significance for either")
    and says(r"not asserted as a result on 23 substances"))


# --- F-AF100: the factor-100 row of R.10 Table R.10-4 names fish or Daphnia
_dv = core.derivability(core.load("r10"))
_rc = _dv.route.value_counts().to_dict()
chk("four-way ladder 23/5/4/55",
    _rc.get("chronic")==23 and _rc.get("acute")==5
    and _rc.get("judgement")==4 and _rc.get("none")==55
    and says(r"23 on the chronic\s+route") and says(r"four requiring an assessor"),
    str(_rc))
chk("no AF 100 on producer-only data",
    not len(_dv[(_dv.route=="chronic") & (_dv.AF_tgd==100)
                & (_dv.chronic_tl_names=="producer")])
    and says(r"factor-100 row\s+names its taxon"))
_L = pd.read_csv(CODE + "/ladder_r10.csv")
chk("no AF 100 on producer-only in the C6' pool",
    not int(((_L.AF_C6_prime==100) & (_L.tlnames_C6_prime=="producer")).sum()))
# the water-side result must be untouched by the factor correction
_T, _ = water.substance_table(a, "r10", verbose=False)
_mr = _T.route.value_counts().to_dict()
chk("monitored 22 split unchanged 10/3/9",
    _mr.get("chronic")==10 and _mr.get("acute")==3 and _mr.get("none")==9
    and not _mr.get("judgement"), str(_mr))


# --- F-EFFORT: the exposure is defined in exactly one place
chk("effort primary is distinct tests",
    core.EFFORT == "n_tests" and core.EFFORT_ALT == "n_records"
    and says(r"number of\s+\\emph\{distinct tests\}")
    and says(r"invariant to how many endpoints"))
_mods = {}
for _m in ("core.py", "fig3.py", "robust.py"):
    _src = open(CODE + "/" + _m).read()
    _mods[_m] = _src.count("n_records") - _src.count("EFFORT_ALT")
chk("no module hard-codes the exposure",
    all(v <= 2 for v in _mods.values()), str(_mods))


# --- F-EPADM: non-effect endpoint codes are excluded at load
_raw = pd.read_csv(core.SRC, low_memory=False)
_bad = _raw.ep.astype(str).str.contains(core.NON_EFFECT_EP, na=False)
# F-MOVED: the endpoint-admissibility detail, the exposure-duration
# thresholds, the factor-table restrictions and the order-statistics derivation
# moved from the body to the SI to meet Water Research's up-front rejection
# criterion on supporting-discipline depth.  The assertions follow the text:
# each is now checked where the text lives, and the body is checked for the
# compressed claim it keeps.  Moving text must not become quieter verification.
chk("414 non-effect rows excluded",
    int(_bad.sum()) == 414 and not d10.ep.astype(str).str.contains(core.NON_EFFECT_EP, na=False).any()
    and says(r"414 such rows are present", SI)
    and says(r"admissibility rather than\s+dimension", SI),
    f"{int(_bad.sum())} in source, {int(d10.ep.astype(str).str.contains(core.NON_EFFECT_EP, na=False).sum())} in frame")
chk("the body keeps the compressed admissibility claim",
    says(r"414 rows carrying codes") and says(r"admissibility rather than\s+dimension"))
chk("4 substances affected stated", says(r"Four of the 87 substances would\s+otherwise", SI)
    and says(r"3\.3e-3", SI))
chk("the body states that four substances would otherwise be affected",
    says(r"four of the 87\s+substances would otherwise"))
# the exposure-duration thresholds moved; the rule itself must still be in the body
chk("duration thresholds are stated in the SI",
    says(r"72 hours for algae", SI) and says(r"28 days for fish", SI))
chk("the body still states the duration rule and its consequence",
    says(r"long-term durations of ECHA\s+guidance R\.10")
    and says(r"23\.8\,\\% ") is not None or says(r"23\.8"))
chk("the body still declares both duration variants",
    says(r"operational.{0,40}rule") and says(r"five substances support a species"))
# the factor-table restrictions moved
chk("factor-table restrictions are stated in the SI",
    says(r"factor-100 row", SI) and says(r"assessor's judgement", SI))
chk("the body still says the restrictions decide derivability",
    says(r"decide whether a substance is assessable"))
# the order-statistics derivation moved
chk("the two-channel derivation is in the SI",
    says(r"parametric functional, not an order", SI)
    and says(r"consistent\s+estimator of a fixed population quantity", SI))
chk("the body keeps the finding without the derivation",
    says(r"parametric\s+functional, not an order statistic")
    and not says(r"consistent\s+estimator of a fixed population quantity"))
# --- F-GATES: no present-tense gate claim survives, in EITHER document.
# This block was SI-scoped, and the body went on asserting "Three assertions
# guard the analysis and run before any result is computed... All three pass"
# while the SI withdrew exactly that claim as not part of the deposited
# pipeline.  A check pointed at one document cannot see the claim in the other,
# which is the same failure as the tautologies removed elsewhere in this file.
# The patterns also have to match the wording each document actually uses: the
# old "Three assertions run before" never matched the body's "Three assertions
# guard the analysis and\nrun before".
for _doc, _dn in ((SI, "SI"), (MAIN, "main.tex")):
    for pat, lab in [(r"Reproduction gates", "'Reproduction gates' heading"),
                     (r"Three assertions", "'three assertions' claim"),
                     (r"[Aa]ll three pass", "'all three pass'"),
                     (r"all must pass", "'all must pass'"),
                     (r"gates? (?:guard|run before)", "a present-tense gate claim")]:
        chk(f"absent in {_dn}: {lab}", not says(pat, _doc))
# and the body must positively carry the retirement, not merely omit the claim
chk("main.tex reports the earlier checks in the past tense",
    says(r"past tense because") and says(r"not part of the pipeline deposited"))
chk("main.tex names the retired parent-table check and why",
    says(r"retired rather than re-anchored")
    and says(r"comparing the pipeline against its own output"))
chk("main.tex and si.tex agree the earlier checks are not in the deposit",
    says(r"not part of the pipeline deposited", MAIN)
    and says(r"not part of the pipeline deposited", SI))
chk("verification section states its limits",
    says(r"What is verified, and how", SI) and says(r"does not establish", SI)
    and says(r"retired rather than re-anchored", SI))


# 4744 is now named deliberately, as the frame the retired gate used.  Assert
# the framing rather than the absence -- an absence assertion here would fire
# on a correct, deliberate mention of a superseded value.
chk("4744 named only as the retired frame",
    not says(r"4744") or (says(r"4744\s*\n?collapsed rows", SI)
                          and says(r"retired rather than re-anchored", SI)))

# =====================================================================
# F-SCOPE: the audit used to read only main.tex and si.tex, never the
# twelve \input files.  Every number inside them -- including Table 1 of
# the main text -- was outside all checks.  Audit the COMPILED PDFs, and
# assert every \input file has a generator in the v3 pipeline.
# =====================================================================
import subprocess, os, glob

def _pdftext(p):
    try:
        return subprocess.run(["pdftotext","-q",p,"-"],capture_output=True,
                              text=True,timeout=120).stdout
    except Exception:
        return ""

PDF_MAIN = _pdftext(PAPER + "/main.pdf")
PDF_SI   = _pdftext(PAPER + "/si.pdf")
chk("PDFs readable for audit", len(PDF_MAIN) > 20000 and len(PDF_SI) > 20000,
    f"main {len(PDF_MAIN)} si {len(PDF_SI)} chars")

# F-STALE: a readable PDF is not a CURRENT PDF.  A failed pdflatex run leaves
# the previous output in place, so every check below would then audit a
# document that no longer corresponds to the source -- and would pass.
# Readability is not currency, in the same way that stability is not validity.
for stem in ("main", "si"):
    tex = f"{PAPER}/{stem}.tex"
    pdf = f"{PAPER}/{stem}.pdf"
    log = f"{PAPER}/{stem}.log"
    fresh = (os.path.exists(pdf) and os.path.exists(tex)
             and os.path.getmtime(pdf) >= os.path.getmtime(tex))
    chk(f"{stem}.pdf is newer than {stem}.tex", fresh,
        "" if fresh else "PDF predates its source; the compile failed or was skipped")
    ltxt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    chk(f"{stem}.log records a completed run", "Output written" in ltxt,
        "no 'Output written' line; pdflatex did not produce output")
    fatal = [l for l in ltxt.splitlines() if l.startswith("!")]
    chk(f"{stem}.log has no fatal error", not fatal,
        fatal[0][:70] if fatal else "")
PDF = PDF_MAIN + "\n" + PDF_SI

# superseded figures that live in \input files and so evaded every
# .tex-level absence assertion
for pat, label in [
    (r"22[,\s]?255",            "22,255 detections (v2 inventory)"),
    (r"20[,\s]?682",            "20,682 detections (dead m_ladder frame)"),
    (r"63[,\s]?791",            "63,791 all-results (old pull)"),
    (r"76\.50",                 "76.50% exceedance (pre-AF-fix)"),
    (r"24\.80",                 "24.80% all-results (pre-AF-fix)"),
    (r"exact normal-theory 95", "'exact normal-theory 95' (F-CI label)"),
    (r"the six in Table",       "tab_persub 'the six in Table 1' (now two)"),
]:
    chk(f"absent in PDF: {label}", re.search(pat, PDF) is None)


# the joint-90 widths are legitimately printed in si_tab_s2 BESIDE the 95%
# ones, with a caption explaining the distinction.  Assert that framing is
# present rather than asserting the number's absence -- an absence assertion
# here was over-broad and flagged a correct, deliberate value.
_s2 = open(PAPER + "/si_tab_s2.tex").read()
chk("si_tab_s2 labels the joint-90 column", "joint" in _s2 and "90" in _s2)


# the v2 sensitivity table paired rho = -0.691 with 16 substances and -0.700
# with a log-logistic fit on 16.  -0.691 recurs legitimately in the v3 table at
# a minimum of three species, so assert the PAIRING is gone, not the number.
_sn = open(PAPER + "/si_tab_sens.tex").read()
chk("sens table is generated, v2 pairings gone",
    "\\label{tab:sens}" in _sn
    and not re.search(r"16 & \$-0\.691", _sn)
    and not re.search(r"log-logistic\s*& 16", _sn))

# internal module names must not ship in captions: four of eleven tables once
# carried "Generated by \textsf{sitabs.<fn>}", which names the build script in
# the published supplement and in none of the other seven.
import glob as _glob
_modcap = [os.path.basename(f) for f in _glob.glob(PAPER + "/*.tex")
           if re.search(r"\\textsf\{sitabs\.", open(f).read())]
chk("no table caption ships an internal module name", not _modcap, str(_modcap))

# every \input must have a generator registered in the v3 pipeline
GEN = {}
for mod in glob.glob(CODE + "/*.py"):
    if os.path.basename(mod) == "audit.py":
        continue          # the audit is not a generator
    src = open(mod).read()
    # F-ALLOWLIST: this used to match a hardcoded list of generator names, so
    # every generator added afterwards was reported as missing.  Detect what a
    # module actually WRITES instead; the check then maintains itself.
    for m in re.finditer(r'["\'](?:[^"\']*/)?([a-zA-Z_0-9]+)\.tex["\']', src):
        GEN.setdefault(m.group(1), set()).add(os.path.basename(mod))
    for m in re.finditer(r'\{OUT\}/([a-zA-Z_0-9]+)\.tex', src):
        GEN.setdefault(m.group(1), set()).add(os.path.basename(mod))
inputs = set()
for doc in (PAPER + "/main.tex", PAPER + "/si.tex"):
    inputs |= set(re.findall(r'\\input\{([a-zA-Z_0-9]+)\}', open(doc).read()))
orphans = sorted(i for i in inputs if i not in GEN)
chk(f"all {len(inputs)} \\input files have a v3 generator",
    not orphans, "no generator: " + ", ".join(orphans))

# overfull boxes in BOTH logs, not main.log alone
for log, name in [(PAPER + "/main.log","main"),
                  (PAPER + "/si.log","si")]:
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    over = [float(x) for x in re.findall(r'Overfull \\hbox \(([\d.]+)pt', txt) if float(x) > 2.0]
    chk(f"{name}.log: no overfull box over 2pt", not over,
        f"{len(over)} boxes, worst {max(over):.1f}pt" if over else "")


# =====================================================================
# F-PLACEMENT: a float that drifts pages away from the sentence that
# cites it is read against the wrong text.  Compare the page carrying
# each figure's caption with the page of its FIRST citation.  Page
# numbers come from the compiled PDF, not from the source order, so
# this catches drift that the .tex cannot show.
# =====================================================================
def _pdfpages(path):
    try:
        t = subprocess.run(["pdftotext", "-q", path, "-"], capture_output=True,
                           text=True, timeout=120).stdout
    except Exception:
        return []
    return t.split("\f")

MAIN_PAGES = _pdfpages(PAPER + "/main.pdf")
chk("main.pdf splits into pages for placement audit", len(MAIN_PAGES) > 10,
    f"{len(MAIN_PAGES)} pages")

def _first_page(pages, pat):
    for i, pg in enumerate(pages, 1):
        if re.search(pat, pg):
            return i
    return None

for n, key in [(1, "lead"), (2, "census"), (3, "official"),
               (4, "effort"), (5, "broad")]:
    cap = _first_page(MAIN_PAGES, rf"Figure\s*{n}:")
    # the citation is any occurrence of "Fig. N" that is NOT the caption
    cite = None
    for i, pg in enumerate(MAIN_PAGES, 1):
        if re.search(rf"Fig\.\s*{n}[A-B]?\b", pg):
            cite = i
            break
    if cite is None:
        cite = cap
    chk(f"figure {n} ({key}) float within two pages of its first citation",
        cap is not None and cite is not None and abs(cap - cite) <= 2,
        f"caption p{cap}, first citation p{cite}")

# F-CENSUS-CLIP: matplotlib drops out-of-range points silently, so a figure
# can omit rows while still rendering.  Assert independently of the generator
# that every substance lands somewhere and that no criterion falls outside the
# axis the generator chooses.
import csv as _csv
_S = list(_csv.DictReader(open(CODE + "/substances_r10.csv")))
def _crit(x):
    if x["C6_prime"]:
        return 10 ** float(x["C6_prime"]) * 1e6 / float(x["AF_C6_prime"])
    if x.get("C_acute"):
        return 10 ** float(x["C_acute"]) * 1e6
    return None
_vals = [_crit(x) for x in _S]
_on  = sum(1 for v in _vals if v is not None)
_gut = sum(1 for v in _vals if v is None)
chk("census figure accounts for every monitored substance",
    _on + _gut == len(_S), f"{_on} on axis + {_gut} in gutter vs {len(_S)} rows")
_src_fig = open(CODE + "/fig_census.py").read()
chk("census axis limits are derived from the data, not hardcoded",
    "set_xlim(30, 1.4e6)" not in _src_fig
    and "min(_vals)" in _src_fig and "max(_vals)" in _src_fig)
_pos = [v for v in _vals if v]
chk("census span stated in the caption matches the data",
    abs(max(_pos) / min(_pos) - 100000) < 5000,
    f"{min(_pos):.1f} to {max(_pos):.0f} = {max(_pos)/min(_pos):.0f}x")

# F-EULABEL: 230 ng/L is the EU DIRECT-TOXICITY COMPONENT.  The binding
# European standard is 0.65 ng/L, from secondary poisoning.  Calling the 230
# figure "the EU standard" anywhere -- caption included -- misstates which
# value is binding.
_mt = open(PAPER + "/main.tex").read()
for _bad in [r"European direct-toxicity standard",
             r"EU standard of \\SI\{230\}",
             r"the EU standard[^.]{0,30}230"]:
    chk(f"absent in main.tex: {_bad[:38]}", re.search(_bad, _mt) is None)
chk("230 ng/L is named a component where it appears",
    _mt.count("230") == 0 or "direct-toxicity component" in _mt)

# F-DING: the two-route gap was quoted as 2.13 on five substances in the
# abstract, the introduction and the results, with no generator in v3 and no
# reproducible definition.  It is now regenerated and reached only through
# macros.  Assert the literal cannot come back and that the macros agree with
# the generator.
_gap = core.route_gap()
_macsrc = open(PAPER + "/si_rob_numbers.tex").read()
_mac = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", _macsrc))
chk("route-gap macros are emitted", {"gapN", "gapMean", "gapMedian", "gapLo",
                                     "gapHi"} <= set(_mac))
chk("gapN macro matches the generator", _mac.get("gapN") == str(_gap["n"]),
    f"macro {_mac.get('gapN')} vs generator {_gap['n']}")
chk("gapMean macro matches the generator",
    abs(float(_mac.get("gapMean", "nan")) - _gap["mean"]) < 0.005,
    f"macro {_mac.get('gapMean')} vs generator {_gap['mean']:.3f}")
chk("gap extremes are the across-substance range, not the mean/median pair",
    abs(float(_mac["gapLo"]) - _gap["lo"]) < 0.005
    and abs(float(_mac["gapHi"]) - _gap["hi"]) < 0.005,
    f"{_mac.get('gapLo')}..{_mac.get('gapHi')} vs {_gap['lo']:.2f}..{_gap['hi']:.2f}")
chk("retired gap literal 2.13 absent from main.tex", "2.13" not in _mt)
chk("gap is stated as BELOW the published range, not inside it",
    "inside the two-to-three-order" not in _mt)
# the abstract has a hard 250-word limit and three items compete for it
_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", _mt, re.S)
if _abs:
    _a = _abs.group(1)
    _a = re.sub(r"\\([a-zA-Z]+)\{\}", lambda m: _mac.get(m.group(1), m.group(1)), _a)
    _a = re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1", _a)
    _a = re.sub(r"\\num\{([^}]*)\}", r"\1", _a)
    _a = re.sub(r"\\[a-zA-Z]+\*?", " ", _a)
    _a = re.sub(r"[{}$\\]", " ", _a)
    _nw = len([x for x in _a.split() if any(c.isalnum() for c in x)])
    chk("abstract within the 250-word limit", _nw <= 250, f"{_nw} words")


# F-FITCOLS: si.tex argues from the Anderson-Darling statistic, its critical
# value, the Shapiro-Wilk rejection count and the probability-plot R^2.  A
# regeneration once dropped all four columns while those claims stayed.  Assert
# the columns are present AND that each claim matches the generator.
_diag = core.ssd_diagnostics("operational")
_fit = open(PAPER + "/si_tab_fit.tex").read()
for _c in ("A^2", "A^2_{0.05}", "R^2"):
    chk(f"si_tab_fit carries the {_c} column", _c in _fit)
chk("si_tab_fit marks the Shapiro-Wilk rejections", r"\dagger" in _fit)
_sitex = open(PAPER + "/si.tex").read()
_adrej = int(_diag.AD_reject5.sum())
chk("si.tex AD rejection count matches the generator",
    f"for {_adrej} of the {len(_diag)}" in _sitex,
    f"generator says {_adrej} of {len(_diag)}")
_swrej = int((_diag.SW_p < 0.05).sum())
chk("si_tab_fit footnote states the Shapiro-Wilk count",
    f"({_swrej} substances)" in _fit, f"generator says {_swrej}")
# the A2 comparison carries the rejection claim, so it needs enough digits to
# be checkable: PFOS is 0.744 against 0.739 and collapses at two decimals.
_pairs = re.findall(r"\$(\d\.\d+)\$ & \$(\d\.\d+)\$ & \$\d\.\d+\$ \\\\", _fit)
chk("A^2 columns are printed to enough digits to check the rejection",
    all(len(a.split(".")[1]) >= 3 for a, b in _pairs) and len(_pairs) >= 10,
    f"{len(_pairs)} rows parsed")
# the worst fit must not be described as benign
_worst = _diag.R2.min()
chk("si.tex states the true poorest fit",
    f"R^2 = {_worst:.2f}" in _sitex.replace("$", ""),
    f"poorest R2 is {_worst:.3f}")
chk("retired claim that the rejections are not the poorly fitted cases is gone",
    "the rejections are not the poorly fitted cases" not in _sitex)
chk("retired poorest-fit literal 0.91 is gone from the diagnostics passage",
    "poorest fit anywhere\nin the screened set is $R^2 = 0.91$" not in _sitex)
# the QQ figure and the fit table must describe the same substances
from scipy import stats as _st
_d = core.load("operational"); _b = _d[_d.conc1_type == "A"]
_sel = _b[_b.ep.isin(core.CHRONIC_EP) & _b.apical & _b.chronic_ok]
_sm = _sel.groupby(["parent", "latin_name"]).logTox.mean().reset_index()
_keep = _sm.groupby("parent").size()
_keep = _keep[_keep >= 5]
chk("QQ figure and fit table cover the same substances",
    set(_keep.index) == set(_diag.parent),
    f"{len(_keep)} panels vs {len(_diag)} table rows")


# =====================================================================
# F-REGEN (closes U5): the generated \input tables were audited only for
# PROVENANCE -- that a generator existed somewhere -- never for CONTENT.
# A mutation test showed SI Table S3 could lose the C6' column entirely,
# by generator edit, by generator edit plus regeneration, or by hand
# edit of the .tex, and the audit still reported 0 failures.  C6' is the
# European-guidance convention the paper's central claim rests on.
#
# Regenerating every table into a scratch directory and comparing bytes
# closes that gap and, in the same stroke, the question of whether a
# checked-in table is stale relative to its generator.  Every generator
# is seeded, so the comparison is deterministic; a diff is a real
# difference, never run-to-run noise.
# =====================================================================
_REGEN = os.path.join(os.environ.get(
    "SCRATCH", "/tmp/claude-0/-home-claude/2d2617e3-ce5c-508c-aaea-0ecd683915ba/scratchpad"),
    "audit_regen")
shutil.rmtree(_REGEN, ignore_errors=True)
os.makedirs(_REGEN, exist_ok=True)
_rg = subprocess.run([sys.executable, CODE + "/sitabs.py"],
                     capture_output=True, text=True, timeout=900,
                     env={**os.environ, "SITABS_OUT": _REGEN})
chk("every generated table regenerates without error", _rg.returncode == 0,
    (_rg.stderr or "")[-180:])

_inputs = set()
for _doc in (PAPER + "/main.tex", PAPER + "/si.tex"):
    _inputs |= set(re.findall(r"\\input\{([a-zA-Z_0-9]+)\}", open(_doc).read()))
_regenerated = {f[:-4] for f in os.listdir(_REGEN) if f.endswith(".tex")}

for _name in sorted(_inputs):
    _live = f"{PAPER}/{_name}.tex"
    _fresh = f"{_REGEN}/{_name}.tex"
    if _name not in _regenerated:
        chk(f"{_name} is produced by the generator run", False,
            "no generator wrote it; it cannot be verified by regeneration")
        continue
    _a = open(_live, "rb").read() if os.path.exists(_live) else b""
    _b = open(_fresh, "rb").read()
    chk(f"{_name}.tex matches its generator byte for byte", _a == _b,
        f"checked-in {len(_a)} B vs regenerated {len(_b)} B")

# a table that no document inputs is dead weight, and a regenerated file
# that nothing inputs means a generator writing a table out of circulation
chk("no generated table is orphaned from both documents",
    _regenerated <= _inputs, f"not input anywhere: {sorted(_regenerated - _inputs)}")

# importing a generator must not write anything: S1--S5 used to be emitted
# at module import, which destroyed the files a verifier wanted to compare
_probe = subprocess.run(
    [sys.executable, "-c",
     "import os,sys;sys.path.insert(0,%r);" % CODE +
     "d='%s';os.makedirs(d,exist_ok=True);os.environ['SITABS_OUT']=d;"
     "import sitabs;print(len(os.listdir(d)))" % (_REGEN + "_probe")],
    capture_output=True, text=True, timeout=900)
chk("importing sitabs writes no files",
    _probe.returncode == 0 and _probe.stdout.strip().endswith("0"),
    f"import wrote {_probe.stdout.strip()} file(s)")


# =====================================================================
# F-ACUTEAF (closes U3): AF_ACUTE_BASESET had no assertion anywhere.  A
# mutation from 1000 to 100 -- a tenfold error in the criterion for every
# acute-route substance, and in the whole acute arm of Fig. 1 and Fig. 2 --
# passed with zero failures.  The audit checked the route COUNTS and the
# prose phrase, but nothing tied the prose to the constant.
# =====================================================================
chk("acute base-set factor is the tabulated 1000",
    core.AF_ACUTE_BASESET == 1000.0, f"core says {core.AF_ACUTE_BASESET}")
chk("prose states the acute factor and matches the constant",
    says(r"factor of 1000 rather than 10") and core.AF_ACUTE_BASESET == 1000.0)
# end to end: the plotted criterion of an acute-route substance must equal the
# screened acute minimum divided by that factor, recomputed live.
_ac = o10[o10.route == "acute"].dropna(subset=["C_acute", "acute_min"])
chk("acute-route criteria equal acute minimum over the tabulated factor",
    len(_ac) >= 3 and np.allclose(
        _ac.C_acute.values,
        _ac.acute_min.values - np.log10(core.AF_ACUTE_BASESET), atol=1e-12),
    f"{len(_ac)} acute-route substances")

# =====================================================================
# F-AFDEC-LIVE (closes U4): core.af_decomposition was never executed by the
# audit, so its internal identity assert was dead code from the audit's point
# of view.  Neutering the tolerance to 1e9, sign-flipping b_af so the identity
# is actually false, and inverting the offset all passed.  si_tab_afdec.tex is
# generated from exactly this function.
# =====================================================================
try:
    _ad = core.af_decomposition(o10)
    _ad_err = None
except AssertionError as _e:          # the function asserts its own identity
    _ad, _ad_err = None, str(_e)
chk("AF decomposition runs without violating its own identity assert",
    _ad_err is None, _ad_err or "")
if _ad is None:                       # keep reporting rather than crashing
    _ad = dict(n=0, b_noec=float("nan"), b_af=float("nan"),
               b_crit=float("nan"), offset=float("nan"))
chk("AF decomposition identity holds under live execution",
    abs(_ad["b_crit"] - (_ad["b_noec"] - _ad["b_af"])) < 1e-9,
    f"b_crit {_ad['b_crit']:.6f} vs b_noec-b_af "
    f"{_ad['b_noec'] - _ad['b_af']:.6f}")
chk("AF decomposition tolerance is still tight enough to bind",
    "< 1e-9" in open(CODE + "/core.py").read())
chk("offset is the factor's share of the endpoint movement, oriented 0 to 1",
    0.0 < _ad["offset"] < 1.0
    and abs(_ad["offset"] - abs(_ad["b_af"]) / abs(_ad["b_noec"])) < 1e-12,
    f"offset {_ad['offset']:.4f}")
chk("the factor offsets less than half the endpoint movement",
    _ad["offset"] < 0.5, f"offset {100*_ad['offset']:.0f}%")
# and the generated table must carry those same three slopes
_afdec_tex = open(PAPER + "/si_tab_afdec.tex").read()
for _k, _lab in (("b_noec", "endpoint"), ("b_af", "factor"), ("b_crit", "criterion")):
    chk(f"si_tab_afdec prints the live {_lab} slope",
        f"{_ad[_k]:+.3f}".replace("+", "") in _afdec_tex
        or f"{_ad[_k]:.3f}" in _afdec_tex,
        f"{_ad[_k]:+.3f} not in the table")
chk("si_tab_afdec states the live substance count",
    str(_ad["n"]) in _afdec_tex, f"n={_ad['n']}")


# =====================================================================
# F-VARIANT (closes U1/U2): says() is existence-based -- re.search for a
# pattern.  A quantity written in N places therefore survives a change to
# N-1 of them: a mutation of "factor of 7389" in the Results passed because
# the Abstract copy still satisfied says(r"7389").  Roughly forty numeric
# checks had that hole.
#
# An exact occurrence count per number would be brittle: every legitimate
# new mention would break it.  What actually needs asserting is that the
# document never carries a VARIANT of an audited quantity -- that every
# number in the shape of the span is the span.  For each quantity, the
# family pattern below was verified to match its canonical value and
# nothing else in the manuscript, so a non-canonical match is drift.
# =====================================================================
_VARIANT_FAMILIES = [
    # (label,                 family regex,       canonical rendering)
    ("criterion span",        r"\b7\d{3}\b",      "7389"),
    ("block detections",      r"11\\,\d{3}",      r"11\,394"),
    ("curated result rows",   r"12\\,\d{3}",      r"12\,349"),
    ("retrieved results",     r"117\\,\d{3}",     r"117\,034"),
    ("all detections",        r"20\\,\d{3}",      r"20\,998"),
    ("stations sampled",      r"\b152\d\b",       "1523"),
    ("stations detecting",    r"\b104\d\b",       "1044"),
    ("generalisation set",    r"\b117\d\b",       "1175"),
    ("distinct tests",        r"\b218\d\b",       "2182"),
    ("block share",           r"\b54\.\d\b",      "54.3"),
]
for _lab, _fam, _canon in _VARIANT_FAMILIES:
    _hits = sorted(set(re.findall(_fam, MAIN)))
    chk(f"no variant of the {_lab} in main.tex",
        _hits == [_canon],
        f"found {_hits}, expected only {[_canon]}")

# U2: a bare-number pattern cannot say WHICH quantity it matched.  "0.92"
# occurs twice in main.tex -- the AUC and an unrelated correlation -- so
# says(r"0\.92") passed a mutation of the AUC to 0.93.  Anchor it.
# The AUC is now reached only through \aucVal, so the U2 anchoring that
# pinned the literal is replaced by a check that no literal survives and that
# the macro agrees with the live statistic (asserted under F-AUCCI below).
# the body spells the statistic out ("area under the receiver operating
# characteristic") while the abstract abbreviates it, so count the macro, not
# the abbreviation.
chk("the statistic is reached through a macro in both abstract and body",
    len(re.findall(r"\\aucVal\{\}", MAIN)) >= 2,
    f"{len(re.findall(r'.aucVal', MAIN))} macro references")
chk("no bare AUC literal remains",
    not re.search(r"AUC \$= 0\.\d+\$", MAIN))

# U6/U7 were tautologies of the same shape: a global "the right phrase exists
# somewhere" clause cannot tell whether a PARTICULAR mention is mislabelled.
# Both are re-stated locally, against the window each occurrence sits in.
def _window(text, m, w=300):
    return text[max(0, m.start() - w):m.end() + w]

chk("every mention of 4744 sits in its retirement context",
    all("retired" in _window(SI, _m) for _m in re.finditer(r"4744", SI)),
    "a 4744 mention is not marked as the retired frame")
_bad230 = [_m.start() for _m in re.finditer(r"230", MAIN)
           if "direct-toxicity component" not in _window(MAIN, _m, 220)]
chk("every mention of 230 ng/L is labelled the direct-toxicity component",
    not _bad230, f"{len(_bad230)} unlabelled mention(s)")
# Narrowed after a false positive on the audit's own first form: the Fig. 3
# caption legitimately says "the binding European standard of 0.65 ng/L ... is
# not the comparator", which a 220-character window around a 230 flagged.  The
# claim is that 230 is never the value ATTACHED to the word standard, so test
# adjacency, not co-occurrence.
_called_std = [_m.group(0) for _m in re.finditer(
    r"(?:EU|European)[^.]{0,30}standard[^.]{0,40}?230", MAIN)]
chk("the 230 figure is never the value attached to \"standard\"",
    not _called_std, f"{_called_std[:1]}")


# F-EU065: 0.65 ng/L -- the BINDING European standard -- reached this audit
# only through two comments.  A number that appears only in a comment is not
# checked; that is the exact shape of the tautologies removed above, and this
# is the jurisdiction-sensitive one.  The pair of values can also be swapped,
# which no check would have caught: 230 labelled binding and 0.65 labelled the
# direct-toxicity component is a clean inversion of the regulatory fact.
chk("main.tex states the binding European standard of 0.65 ng/L",
    says(r"\\SI\{0\.65\}\{\\nano\\gram\\per\\litre\}"))
_m065 = [_m for _m in re.finditer(r"0\.65", MAIN)]
chk("0.65 is attributed to secondary poisoning, not direct toxicity",
    all(re.search(r"secondary poisoning|binding", _window(MAIN, _m, 260))
        for _m in _m065),
    f"{len(_m065)} occurrence(s)")
chk("0.65 is never called the direct-toxicity component",
    not any(re.search(r"direct-toxicity component[^.]{0,60}?0\.65"
                      r"|0\.65[^.]{0,60}?direct-toxicity component", MAIN)
            for _ in [0]))
chk("0.65 is excluded as the comparator for an aquatic PNEC",
    says(r"not the comparator"))
# The two EU figures must not be interchanged.  First form of this check was
# over-broad: it forbade "binding" within 30 characters AFTER a 230, which the
# Fig. 3 caption trips legitimately by naming the component and then
# introducing the binding value -- "at 250 and 230 ng/L; the binding European
# standard of 0.65".  Proximity is not attribution.  Test the value actually
# attached to the phrase instead.
_bind = re.findall(r"binding[^.]{0,40}?standard[^.]{0,30}?of\s*\\SI\{([\d.]+)\}", MAIN)
chk("the value attached to the binding European standard is 0.65",
    _bind and set(_bind) == {"0.65"}, f"found {_bind}")


# F-HAZSHARE (ABSTRACT-DECOMP): the share the legitimate control leaves was
# stated in the Abstract with no referent, in the Results as "of the total"
# and in the Conclusions as "of the remainder".  Those are different claims
# about different denominators and only "of the total" is what the pipeline
# computes; the Conclusions form implied roughly 55% of the slope where the
# computation gives 63%.  All three now read one macro.
chk("retired 'two thirds' phrasing is gone from main.tex",
    "two thirds" not in MAIN)
chk("retired 'of the remainder' referent is gone",
    "of the remainder" not in MAIN)
_hazmac = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                          open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("hazShare", "hazShareLo", "hazShareHi", "hazPartial", "hazN"):
    chk(f"macro {_k} is emitted", _k in _hazmac)
chk("the hazard share is stated against the total, not a remainder",
    says(r"\\hazShare\\,\\% of the total"))
chk("the hazard-control sentence names its own sample size",
    says(r"\\hazN\{\} chemicals"))
# the share must sit inside the range the depth sweep produces
try:
    _hs = float(_hazmac["hazShare"]); _lo = float(_hazmac["hazShareLo"])
    _hi = float(_hazmac["hazShareHi"])
    chk("the quoted share lies inside the depth-sweep range",
        _lo <= _hs <= _hi, f"{_hs} not in [{_lo}, {_hi}]")
    chk("the depth-sweep range is reported, not just the point value",
        says(r"\\hazShareLo\{\} to \\hazShareHi"))
except Exception as _e:
    chk("hazard macros are numeric", False, str(_e))
# the S5 caption stated two different split counts in one sentence
_s5 = open(PAPER + "/si_tab_s5.tex").read()
chk("si_tab_s5 caption states one split count",
    not ("20 independent splits" in _s5 and "40 independent splits" in _s5),
    "caption names both 20 and 40 independent splits")
chk("si_tab_s5 caption no longer claims a bare 'two thirds'",
    "two thirds of it" not in _s5)


# F-LCL: the reviewer instruction for this item specified rho = -0.508 for the
# interval half-width against log species, "invariant to coverage".  That value
# is the SIGMA-HAT correlation (+0.503) reported with the half-width's label
# and the opposite sign.  Neither -0.508 nor the half-width correlation
# (-0.436 at 95%, -0.402 at joint 90%) may enter either document.
_lcl = core.lcl_support()
_SI = open(PAPER + "/si.tex").read()
chk("retired literal 0.508 absent from both documents",
    "0.508" not in MAIN and "0.508" not in _SI)
for _v in ("0.436", "0.402", "0.609", "0.555", "0.471"):
    chk(f"half-width correlation {_v} absent from both documents",
        _v not in MAIN and _v not in _SI)
chk("the half-width correlation is not reported at all",
    "half-width" not in MAIN,
    "main text mentions a half-width; the paper reports full widths")

# the five reportable coefficients must be macro-fed and match the generator
_lm = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("lclN", "rhoSigmaSp", "rhoHcSp", "rhoLclA", "rhoLclB"):
    chk(f"macro {_k} is emitted", _k in _lm)
chk("lclN matches the generator", _lm.get("lclN") == str(_lcl["n"]))
chk("sigma-hat correlation macro matches the generator",
    abs(float(_lm["rhoSigmaSp"]) - _lcl["rho_sd"]) < 0.0005,
    f"{_lm.get('rhoSigmaSp')} vs {_lcl['rho_sd']:.4f}")
chk("point-percentile correlation macro matches the generator",
    abs(float(_lm["rhoHcSp"]) - _lcl["rho_hc5"]) < 0.0005)
chk("LCL correlation at 95% matches the generator",
    abs(float(_lm["rhoLclA"]) - _lcl["lcl"][0.95][0]) < 0.0005)
chk("LCL correlation at joint 90% matches the generator",
    abs(float(_lm["rhoLclB"]) - _lcl["lcl"][0.90][0]) < 0.0005)

# the convention-dependence must be stated, not hidden behind the 95% result
chk("both coverage conventions are reported for the LCL",
    r"\rhoLclA" in _SI and r"\rhoLclB" in _SI,
    "only one convention appears; that is the nanpercentile error again")
chk("the LCL coefficient stays out of the main text",
    r"\rhoLclA" not in MAIN and r"\rhoLclB" not in MAIN)

# k(n) monotonicity is the body's load-bearing claim, and it is a property of
# the function -- NOT a rank correlation.  9 of the 13 substances tie at 5, 6
# or 7 species, so Spearman(k, n) is -0.992, not -1; quoting -1 would be false.
chk("the tolerance factor is strictly decreasing in species count",
    _lcl["k_strictly_decreasing"])
chk("no rank correlation is quoted for the tolerance factor",
    "-1.000" not in MAIN and "$-1$" not in MAIN.replace("$-1.21$", ""))
chk("monotonicity is stated as a construction, not a tendency",
    says(r"exactly monotone by\s+construction"))

# the retired false claim about interval width ordering
chk("retired claim 'widest in each case at the fewest species' is gone",
    "widest in each\ncase at the fewest species" not in MAIN
    and "widest in each case at the fewest species" not in MAIN)
chk("main text says the widths are not ordered by species count",
    says(r"not ordered by species count"))


# F-CENSUSvMECH: the panel's sharpest referee objection was that the paper's
# magnitude claim (criteria a factor of 7389 apart) comes from WHICH route the
# guidance can run -- a complete enumeration with no sampling uncertainty --
# while the effort mechanism is a within-chemical effect of about a tenth of a
# log unit demonstrated on an acute-only generalisation set.  Stating both in
# one abstract without marking which is which invites the reader to attribute
# the first to the second.
_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN, re.S)
chk("abstract exists for the census/mechanism checks", _abs is not None)
_at = _abs.group(1) if _abs else ""
chk("abstract marks the census as a complete enumeration",
    "complete\nenumeration" in _at or "complete enumeration" in _at)
chk("abstract names the generalisation set's endpoint class",
    re.search(r"1175\s+acute-tested", _at) is not None,
    "the 1175-chemical set is acute-only and the abstract must say so")
chk("abstract gives the mechanism its own magnitude",
    r"\rarEffect" in _at, "the within-chemical effect size is missing")
chk("the title no longer makes support the explanation of the magnitude",
    "evidential support, not measured concentration" not in MAIN)
chk("the title states what the census shows",
    says(r"which derivation the evidence admits"))
# the rarefaction magnitude must be macro-fed and match the generator
_rm = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
chk("rarefaction macros are emitted", {"rarSlope", "rarEffect"} <= set(_rm))
try:
    _S2, _g2, _B2 = mech.build()
    _sl, _se, _gs2, _R2r = mech.rarefaction_slope(_g2)
    chk("rarSlope matches the generator",
        abs(float(_rm["rarSlope"]) - abs(_sl)) < 0.0005,
        f"{_rm.get('rarSlope')} vs {abs(_sl):.4f}")
    chk("rarEffect is the slope over 20 species down to 5",
        abs(float(_rm["rarEffect"]) - abs(_sl) * np.log10(4.0)) < 0.005,
        f"{_rm.get('rarEffect')} vs {abs(_sl)*np.log10(4.0):.3f}")
    chk("the mechanism is an order of magnitude smaller than the census span",
        abs(_sl) * np.log10(4.0) < 0.5,
        "if this fails the abstract's framing needs rechecking")
except Exception as _e:
    chk("rarefaction slope is reachable from mech", False, str(_e))
# the abstract still has to fit
_aw = _at
_aw = re.sub(r"\\([a-zA-Z]+)\{\}", lambda m: _rm.get(m.group(1), m.group(1)), _aw)
_aw = re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1", _aw)
_aw = re.sub(r"\\num\{([^}]*)\}", r"\1", _aw)
_aw = re.sub(r"\\[a-zA-Z]+\*?", " ", _aw)
_aw = re.sub(r"[{}$\\]", " ", _aw)
_n = len([x for x in _aw.split() if any(c.isalnum() for c in x)])
chk("abstract still within the 250-word limit after the split", _n <= 250, f"{_n} words")


# F-REFS: the author's standing standard is at least 60 references, all with a
# confirmed DOI, and their own prior work cited in several places rather than
# once.  The manuscript sat at 37 with 2 self-citations.
_bbl = PAPER + "/main.bbl"
_nref = len(re.findall(r"\\bibitem", open(_bbl).read())) if os.path.exists(_bbl) else 0
chk("bibliography has at least 60 references", _nref >= 60, f"{_nref} references")
_citekeys = {k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", MAIN) for k in g.split(",")}
chk("every cited key resolves (no undefined citations)",
    "Citation" not in open(PAPER + "/main.log").read()
    or "undefined" not in open(PAPER + "/main.log").read(),
    "LaTeX reports an undefined citation")
# DOI coverage: entries without one must be the legitimately DOI-less kinds
_bibtext = "".join(open(f"{PAPER}/{f}", errors="ignore").read()
                   for f in ("myref.bib", "newcites.bib")
                   if os.path.exists(f"{PAPER}/{f}"))
_entries = {}
for _b in re.split(r"\n(?=@)", _bibtext):
    _k = re.search(r"@\w+\{([^,]+),", _b)
    if _k:
        _entries[_k.group(1).strip()] = bool(re.search(r"(?i)\bdoi\s*=", _b))
_cited_present = {k for k in _citekeys if k in _entries}
_nodoi = sorted(k for k in _cited_present if not _entries[k])
chk("at most 12 cited entries lack a DOI (reports, directives, software)",
    len(_nodoi) <= 12, f"{len(_nodoi)} without DOI: {_nodoi[:6]}")
# no work cited twice under two keys
_bydoi = {}
for _b in re.split(r"\n(?=@)", _bibtext):
    _k = re.search(r"@\w+\{([^,]+),", _b)
    _d = re.search(r"(?i)doi\s*=\s*[{\"]\s*(?:https?://(?:dx\.)?doi\.org/)?([^}\"\s,]+)", _b)
    if _k and _d and _k.group(1).strip() in _citekeys:
        _bydoi.setdefault(_d.group(1).lower().rstrip("."), []).append(_k.group(1).strip())
_dupdoi = {d: ks for d, ks in _bydoi.items() if len(ks) > 1}
chk("no work is cited twice under two keys", not _dupdoi, f"{list(_dupdoi.items())[:2]}")
# the author's own work, distributed rather than clustered
_self = [k for k in _citekeys
         if k in _entries and re.search(r"(?i)Kang,\s*Ku|Ku Kang",
             next((b for b in re.split(r"\n(?=@)", _bibtext)
                   if b.startswith("@") and re.match(r"@\w+\{%s," % re.escape(k), b)), ""))]
chk("the author's own prior work is cited at least four times",
    len(_self) >= 4, f"{len(_self)} self-citations: {sorted(_self)}")
# earliest occurrence of each key, wherever in a \cite group it sits -- the
# first form of this took the first LITERAL "\cite{<key>" and so missed a key
# that appears earlier inside a multi-key group, which is most of them.
_pos = sorted(m.start() for k in _self
              for m in re.finditer(r"\\cite\{[^}]*\b%s\b" % re.escape(k), MAIN))
_secs = [(m.start(), m.group(1)) for m in re.finditer(r"\\section\{([^}]*)\}", MAIN)]
def _sec_of(p):
    cur = "front matter"
    for st, n in _secs:
        if st <= p:
            cur = n
    return cur
_selfsecs = {_sec_of(p) for p in _pos}
chk("the author's own work is cited in at least three different sections",
    len(_selfsecs) >= 3, f"sections: {sorted(_selfsecs)}")
chk("self-citations are distributed, not clustered in one passage",
    len(_pos) >= 2 and (max(_pos) - min(_pos)) > 0.3 * len(MAIN),
    f"span {max(_pos)-min(_pos) if _pos else 0} of {len(MAIN)} characters")


# F-WRSCOPE: Water Research rejects up front for supporting-discipline depth
# "without making a good link with water research in general".  The link is the
# assessment consequence, stated in stations.  Two things must hold: the link
# must be present early, and it must NOT be carried by the 7389-fold figure,
# which is the span across seven different substances and not the movement of
# any one threshold.  Conflating them is the census/mechanism error again.
chk("the assessment consequence appears in the Introduction",
    says(r"What that choice costs is measurable in the monitoring record"))
chk("the consequence is stated in stations, not only percentages",
    says(r"\\stDiff\{\}\s+of\s+(?:those\s+)?\\stTotal\{\}\s+(?:monitored\s+)?stations"))
_stm = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                       open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("stTotal", "stAF", "stSSD", "stDiff", "stSubs", "routeRatio", "monOrgs"):
    chk(f"macro {_k} is emitted", _k in _stm)
try:
    import ladder as _ld
    _res, _common, _sub = _ld.run("operational")
    _n = _sub.MonitoringLocationIdentifier.nunique()
    chk("station total matches the generator", int(_stm["stTotal"]) == _n,
        f"{_stm.get('stTotal')} vs {_n}")
    chk("station counts reproduce the tabulated percentages",
        int(_stm["stAF"]) == round(_res["C6p"][2] / 100.0 * _n)
        and int(_stm["stSSD"]) == round(_res["C7b"][2] / 100.0 * _n),
        f"{_stm.get('stAF')}/{_stm.get('stSSD')}")
    chk("stDiff is the difference of the two station counts",
        int(_stm["stDiff"]) == int(_stm["stAF"]) - int(_stm["stSSD"]))
    chk("routeRatio is a single threshold's movement, not the substance span",
        abs(float(_stm["routeRatio"]) - _res["C7b"][0] / _res["C6p"][0]) < 1.0
        and float(_stm["routeRatio"]) < 1000,
        f"routeRatio {_stm.get('routeRatio')}")
except Exception as _e:
    chk("ladder is reachable for the station checks", False, str(_e))
# 7389 must never be offered as a threshold or target movement
_bad7389 = [m.group(0)[:90] for m in re.finditer(
    r"[^.]{0,120}7389[^.]{0,120}", MAIN)
    if re.search(r"threshold|target|moves|movement|apart on the same evidence",
                 m.group(0))]
chk("7389 is never stated as a threshold or target movement",
    not _bad7389, f"{_bad7389[:1]}")
chk("the span is attributed to the substance set, not to one substance",
    says(r"criteria a factor of 7389 apart") or says(r"criteria 7389-fold"))
# the monitoring-design link
chk("the monitoring-design subsection exists",
    says(r"What this says to a monitoring programme"))
chk("the monitoring-design subsection names the programme count",
    says(r"\\monOrgs\{\} organisations"))
# no treatment claim the data cannot support
chk("no claim that an ambient exceedance implies a treatment obligation",
    not re.search(r"treatment (?:is )?(?:required|obligation|target)", MAIN),
    "these are ambient stations, not intakes or discharge points")


# F-CHRONPOOL: the acute restriction on the generalisation set must be stated
# as forced by data availability, not conceded as a limitation.  A referee's
# sharpest line was that the mechanism is shown on a proxy after the target
# population was declared unable to resolve it; the answer is that the chronic
# pool cannot support the analysis at all.
_im = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("acutePanel", "chronPanel", "chronMaxSp"):
    chk(f"macro {_k} is emitted", _k in _im)
_Sm, _gm, _Bm = mech.build()
_slm, _sem, _gsm, _Rm = mech.rarefaction_slope(_gm)
chk("acute rarefaction panel size matches the generator",
    int(_im["acutePanel"]) == len(_gsm), f"{_im.get('acutePanel')} vs {len(_gsm)}")
_dc = core.load("operational")
_bc = _dc[_dc.conc1_type == "A"]
_sc = _bc[_bc.ep.isin(core.CHRONIC_EP) & _bc.apical & _bc.chronic_ok]
_spc = _sc.groupby(["test_cas", "latin_name"]).size().groupby("test_cas").size()
chk("chronic panel size matches the generator",
    int(_im["chronPanel"]) == int((_spc >= 20).sum()),
    f"{_im.get('chronPanel')} vs {int((_spc >= 20).sum())}")
chk("the chronic pool genuinely cannot support the rarefaction",
    int((_spc >= 20).sum()) < 10,
    "if this grows, the replication becomes possible and the passage must change")
chk("the acute restriction is stated as forced, not conceded",
    says(r"That restriction is forced rather than chosen"))
chk("the passage gives both panel sizes",
    says(r"\\chronPanel\{\} such chemicals") and says(r"\\acutePanel\{\} in the acute set"))
chk("the passage disclaims magnitude transfer",
    says(r"what does not\s+transfer is the magnitude"))


# F-LEADPOTENCY: the seven-substance paragraph quantified support by records
# and species while Figure 1 annotates distinct tests, which is the paper's
# declared exposure variable.  The body lagged its own figure.  Reported as
# tests rather than species, which also avoids the chronic_sp / nsp_C6_prime
# collision (PFHxA 2 against 1, PFOS 34 against 33).
_lm = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("leadTestLo", "leadTestHi", "leadTestFac"):
    chk(f"macro {_k} is emitted", _k in _lm)
_aw2 = water.load(verbose=False)
_R2, _sh2, _sb2 = lead.block(_aw2[_aw2.detected], o10,
                             _aw2[["parent", "name"]].drop_duplicates("parent")
                             .set_index("parent").name, 700)
_n2p2 = {v: k for k, v in _aw2[["parent", "name"]].drop_duplicates("parent")
         .set_index("parent").name.items()}
_S2 = o10.set_index("parent")
_t2 = [int(_S2.loc[_n2p2[r["name"]]].n_tests) for _, r in _R2.iterrows()]
chk("lead-block test range matches the generator",
    int(_lm["leadTestLo"]) == min(_t2) and int(_lm["leadTestHi"]) == max(_t2),
    f"{_lm.get('leadTestLo')}-{_lm.get('leadTestHi')} vs {min(_t2)}-{max(_t2)}")
chk("the quoted factor is the ratio of the two",
    int(_lm["leadTestFac"]) == round(max(_t2) / min(_t2)),
    f"{_lm.get('leadTestFac')} vs {round(max(_t2)/min(_t2)):d}")
chk("the body states support in distinct tests, the declared exposure variable",
    says(r"the number of distinct tests") and says(r"\\leadTestFac\{\}"))
chk("the body says why tests rather than records or species",
    says(r"Records\s+overcount replicate entries")
    and says(r"species undercount the designs"))
chk("the test count is tied to the figure that annotates it",
    says(r"annotated per substance in\s+Fig\.~\\ref\{fig:lead\}"))


# F-BASESET: the four-way ladder is usually read as a hierarchy with the acute
# base set beneath the long-term route.  On this record 17 of the 23
# chronic-route substances have an incomplete base set and none of those 17 has
# a complete one, so for them the long-term datum is the only route and
# withdrawing it moves them to "no tabulated factor", not to a factor of 1000.
# Stated as derivability, not stringency.
_bm = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("chronN", "noBackstop", "withBackstop", "noAcuteAtAll"):
    chk(f"macro {_k} is emitted", _k in _bm)
_chb = o10[o10.route == "chronic"]
_incb = _chb[_chb.acute_tl < 3]
chk("chronic-route count matches the generator",
    int(_bm["chronN"]) == len(_chb), f"{_bm.get('chronN')} vs {len(_chb)}")
chk("the no-backstop count matches the generator",
    int(_bm["noBackstop"]) == len(_incb), f"{_bm.get('noBackstop')} vs {len(_incb)}")
chk("the two counts partition the chronic route",
    int(_bm["noBackstop"]) + int(_bm["withBackstop"]) == int(_bm["chronN"]))
chk("none of the no-backstop substances has a complete base set",
    int((_incb.acute_tl >= 3).sum()) == 0,
    "the SI claims none does; if this fails the claim is false")
chk("the body states the base set is not a backstop",
    says(r"not the stricter of two available routes but the only one they have"))
chk("the SI subsection exists", says(r"The base set is not a backstop", SI))
chk("the SI states the consequence as a route change, not a value change",
    says(r"no tabulated factor.{0,40}rather\s*\n?than\s*\n?to the acute base set", SI)
    or says(r"moves the substance to \\emph\{no tabulated factor\}", SI))
chk("the claim is framed as derivability rather than stringency",
    says(r"it is the only route", SI))


# F-AUCCI: the abstract and body stated AUC = 0.92 with no interval.  Its upper
# bound touches 1.000 -- five eligible substances cannot produce a separation
# finer than the boundary -- so the interval is censored above and that must be
# declared.  Also: water.py computed the statistic on n_records while the
# declared exposure variable is core.EFFORT; the two agreed to two decimals
# (0.922 against 0.924), which is how such a mismatch survives.
_am = dict(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}",
                      open(PAPER + "/si_rob_numbers.tex").read()))
for _k in ("aucVal", "aucLo", "aucHi", "aucNelig", "aucP"):
    chk(f"macro {_k} is emitted", _k in _am)
_eligA = (o10.n_sp_d.fillna(0) >= 5).values
_xA = o10[core.EFFORT].values
_mwA = stats.mannwhitneyu(_xA[_eligA], _xA[~_eligA], alternative="greater")
_aucA = _mwA.statistic / (_eligA.sum() * (~_eligA).sum())
chk("the AUC macro matches a live recomputation on the declared variable",
    abs(float(_am["aucVal"]) - _aucA) < 0.0005,
    f"{_am.get('aucVal')} vs {_aucA:.4f}")
chk("water.py computes the AUC on the declared exposure variable",
    "core.EFFORT" in open(CODE + "/water.py").read()
    and "out.n_records[elig]" not in open(CODE + "/water.py").read())
chk("the interval's upper bound is at the boundary",
    abs(float(_am["aucHi"]) - 1.00) < 0.005,
    f"upper {_am.get('aucHi')}; if it moves off 1.00 the censoring wording must change")
chk("the body reports the interval", says(r"\\aucLo\{\} to\s*\n?\\aucHi\{\}"))
chk("the body declares the censoring and its cause",
    says(r"censored above because \\aucNelig\{\} eligible substances"))
chk("the body limits the claim to direction, not magnitude",
    says(r"the claim it supports is the direction and not the magnitude"))
chk("the abstract flags the censoring", r"censored above" in _at)
chk("the retired bare AUC literal is gone", "AUC $= 0.92$" not in MAIN)


# F-RUNALL: the cover letter will claim that every number, table and figure
# regenerates from a single command.  fig_lead and fig_census were absent from
# run_all's generator list, so Figures 1 and 2 did NOT regenerate -- the claim
# was false for the two figures built most recently.  Assert the claim instead
# of trusting it.
_ra = open(CODE + "/run_all.py").read()
_gen_mods = re.findall(r'for mod in \(([^)]*)\)', _ra)
_listed = set(re.findall(r'"([a-zA-Z_0-9]+)"', _gen_mods[0])) if _gen_mods else set()
_fig_mods = {os.path.basename(p)[:-3] for p in
             __import__("glob").glob(CODE + "/fig*.py")}
chk("run_all lists every figure generator", _fig_mods <= _listed,
    f"missing from run_all: {sorted(_fig_mods - _listed)}")
chk("run_all's summary line states the right figure count",
    "Figures 1-5" in _ra, "the printed summary undercounts the figures")
chk("no generator is invoked twice by run_all",
    len(_listed) == len(set(_listed)) and "tab1" not in _listed,
    "tab1 is reached through sitabs; listing it again rebuilds the same file")
# the four-way ladder docstring drifted behind the code for a long time
chk("derivability documents all four routes",
    all(r in (core.derivability.__doc__ or "")
        for r in ("chronic", "acute", "judgement", "none")))
chk("the docstring states the judgement branch's ordering",
    "AFTER the base-set test" in (core.derivability.__doc__ or ""))
chk("the routes in the frame are the routes documented",
    set(o10.route.unique()) <= {"chronic", "acute", "judgement", "none"})


# F-SIBIB: si.tex cites two references and had an EMPTY thebibliography,
# so si.pdf page 6 carried two literal "[?]" marks.  Cause: the build ran
# bibtex on main.tex only, and the citations arrived in si.tex when the
# exposure-duration block moved there.  A document that cites must resolve.
for _stem in ("main", "si"):
    _lg = f"{PAPER}/{_stem}.log"
    _t = open(_lg, errors="ignore").read() if os.path.exists(_lg) else ""
    chk(f"{_stem}.log reports no undefined citation",
        "undefined citations" not in _t.lower()
        and "Citation" not in re.sub(r"Citation `[^']*' on page \d+ undefined", "", _t)
        or "undefined" not in _t.lower(),
        "a citation does not resolve")
    chk(f"{_stem}.log reports no empty bibliography",
        "Empty `thebibliography'" not in _t)
    _pdf = f"{PAPER}/{_stem}.pdf"
    if os.path.exists(_pdf):
        _txt = subprocess.run(["pdftotext", "-q", _pdf, "-"], capture_output=True,
                              text=True, timeout=120).stdout
        chk(f"{_stem}.pdf carries no unresolved citation marks",
            "[?]" not in _txt, f"{_txt.count('[?]')} occurrences")
# every key cited in si.tex must exist in the database si.tex reads
_sitex = open(PAPER + "/si.tex").read()
_sibib = re.search(r"\\bibliography\{([^}]*)\}", _sitex)
_sikeys = {k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", _sitex) for k in g.split(",")}
_sidb = ""
if _sibib:
    for _b in _sibib.group(1).split(","):
        _p = f"{PAPER}/{_b.strip()}.bib"
        if os.path.exists(_p):
            _sidb += open(_p, errors="ignore").read()
_missing = sorted(k for k in _sikeys if k and f"{{{k}," not in _sidb)
chk("every key cited in si.tex is in the database si.tex reads",
    not _missing, f"missing: {_missing}")


# F-SPANWORD: the span was rendered three ways across the document -- "nearly
# four orders of magnitude" in the Conclusions against "7389-fold" in the
# abstract and body.  3.87 orders rounded up is the same overstatement already
# removed from the figure title; one quantity gets one rendering.
chk("the span is rendered one way throughout",
    "nearly four orders" not in MAIN and "four orders of magnitude" not in MAIN,
    "a rounded-up rendering of the 7389-fold span survives")
chk("the span appears as the exact factor wherever it is stated",
    len(re.findall(r"7389", MAIN)) >= 4,
    f"{len(re.findall(r'7389', MAIN))} occurrences")

# every figure the source includes must exist on disk and be referenced
_src = open(PAPER + "/main.tex").read()
_inc = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", _src)
for f in _inc:
    chk(f"figure file present: {f}",
        os.path.exists(f"{PAPER}/{f}"))
for lbl in re.findall(r"\\label\{(fig:[^}]+)\}", _src):
    chk(f"figure label cited in prose: {lbl}",
        len(re.findall(rf"\\ref\{{{re.escape(lbl)}\}}", _src)) >= 1)



# ===================================================================
# F-ROUND2: the defects found by the second full defect hunt.  Each of
# these was a disagreement between two places in the submitted package
# that a referee could have found by reading alone.
# ===================================================================

# F-PERSUB: tab_persub printed C6_pub raw -- log10 mg/L, hence NEGATIVE
# numbers -- under a heading saying ng/L, and every C6 exceedance was
# 100 %.  The C6 column is a CRITERION: the tabulated factor is divided
# out and the result converted, exactly as C1 and C7b are.
_ps = open(PAPER + "/tab_persub.tex").read()
_c6col = []
for _ln in _ps.splitlines():
    if _ln.startswith(r"\textsf{") and "&" in _ln:
        _f = [c.strip() for c in _ln.split("&")]
        if len(_f) >= 7:
            _c6col.append(_f[6])
chk("tab_persub prints 22 substance rows", len(_c6col) == 22, str(len(_c6col)))
_neg = [v for v in _c6col if re.match(r"^-\d", v)]
chk("no C6 criterion in tab_persub is negative", not _neg, str(_neg))
chk("tab_persub C6 exceedances are not all 100 %",
    len({l.split("&")[9].strip() for l in _ps.splitlines()
         if l.startswith(r"\textsf{") and len(l.split("&")) >= 10}) > 1)
chk("tab_persub says which columns are pre-screen and which post-screen",
    "unscreened pool" in _ps and "survives the apical" in _ps)
# the synonym is legitimate in si_tab_inv, which lists the monitoring
# record's own characteristic names; it is a defect only where the table
# names the SUBSTANCE, as tab_persub and Fig. 1 do.
chk("the stale ECOTOX synonym is gone from the per-substance table",
    "Perfluorovaleric" not in _ps)

# F-INVCAS: water.load() inner-joins on name->parent, so the CAS column
# survived only for names that resolve and the inventory printed "nan"
# for the other 25 -- hiding the 18 CAS numbers the main text counts.
_inv = open(PAPER + "/si_tab_inv.tex").read()
chk("si_tab_inv prints no nan in the CAS column", " nan " not in _inv)
chk("si_tab_inv states the 40/22/18 CAS accounting",
    "40 distinct CAS" in _inv and "22 parent structures" in _inv
    and "18 CAS numbers" in _inv)

# F-ADMARK: si_tab_fit's daggers and fig_si_qq's red titles were keyed to
# a Shapiro-Wilk p while the table's own columns and the SI prose argue
# from Anderson-Darling.  The two tests mark different sets of four.
_fit = open(PAPER + "/si_tab_fit.tex").read()
chk("si_tab_fit attributes its daggers to Anderson--Darling",
    "Anderson--Darling rejects" in _fit and "Shapiro" not in _fit)
chk("the Fig. S2 caption attributes its marks to Anderson--Darling",
    "Anderson--Darling statistic rejects" in SI and
    "Shapiro--Wilk test rejects" not in SI)
_ndag = _fit.count(r"\,$^{\dagger}$")
_adrej = int(core.ssd_diagnostics("operational").AD_reject5.sum())
chk("the number of daggers equals the live Anderson--Darling count",
    _ndag == _adrej, f"{_ndag} daggers, {_adrej} rejections")

# F-SENSEFF: si_tab_sens used result rows as the exposure while the rest
# of the manuscript defines effort as distinct tests, so Table S8 printed
# -0.703 where Table S10's baseline is -0.710.
_sens = open(PAPER + "/si_tab_sens.tex").read()
_rob = open(PAPER + "/si_tab_s6.tex").read()
_base = re.search(r"as reported \(effort as distinct tests\).*?&.*?&.*?&\s*\$([-+0-9.]+)\$", _rob)
chk("si_tab_sens declares distinct tests as the exposure",
    "distinct tests" in _sens)
chk("si_tab_sens's five-species log-normal row equals tab:rob's C7b baseline",
    _base is not None and f"${_base.group(1)}$" in _sens,
    _base.group(1) if _base else "no baseline found")

# F-NODROP: the pre-specified structural exclusion removes nothing from
# the screened pools, so the row reproduced the baseline under a caption
# calling it a check.  Say so instead.
chk("the vacuous exclusion row is marked as such",
    ("drops no substance" in _rob) or
    ("excluding three non-OECD-PFAS parents &" in _rob and
     "$-0.506$" not in _rob))

# F-CONV9: the body said "seven conventions" while Table S1 lists nine,
# and C7 was defined once then used only as C7a and C7b.
_s1 = open(PAPER + "/si_tab_s1.tex").read()
_ncode = len([l for l in _s1.splitlines()
              if re.match(r"^C\d", l) or l.startswith("C6$'$")])
chk("Table S1 lists nine convention codes", _ncode == 9, str(_ncode))
chk("the body does not claim seven conventions in total",
    "seven conventions, from the mechanical minimum" not in MAIN
    and r"\subsection{Seven derivation conventions}" not in MAIN)
chk("C7a and C7b are introduced in the Methods",
    r"\textbf{C7a}" in MAIN and r"\textbf{C7b}" in MAIN)

# F-MODCAP: internal module names must not ship in printed captions.
# (also asserted file-by-file above)
chk("no printed caption names the build module",
    r"\textsf{sitabs." not in MAIN + SI)

# F-TITLE3: all three submitted documents carry the same title.
_cl = open(PAPER + "/cover_letter.tex").read()
_t = "which derivation the evidence admits"
chk("main, SI and cover letter carry the same title phrase",
    _t in MAIN and _t in SI and _t in _cl)
chk("the superseded gap figure is gone from the cover letter",
    "2.13" not in _cl)

# F-FWDREF2: a Methods sentence must not cite a Results float.
_meth = MAIN.split(r"\section{Results}")[0]
chk("the Methods cite no Results float",
    not re.search(r"\\ref\{fig:(lead|census|official|effort|broad)\}", _meth))

print("\n"+"="*84)
print(f"{ok} passed, {bad} failed")
print("="*84)
import sys as _s; _s.exit(1 if bad else 0)
