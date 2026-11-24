# What the audit checks, and what it cannot

`pipeline/audit.py` is the cross-document audit. It runs against the
manuscript sources and the compiled PDFs, which are **not** part of this
repository; point it at them with

```bash
PAPER_DIR=/path/to/manuscript python pipeline/audit.py
```

It currently runs 386 checks. What they do:

**Manuscript against pipeline.** Every quantity stated in the text is
recomputed and compared with the value as it is *printed in the PDF*, not as
it is written in the LaTeX source, so a macro that expands to the wrong thing
is caught. Superseded values are asserted absent from both documents, with one
exception: the supplement's robustness section deliberately prints three
superseded figures, and the audit knows about that passage.

**Generated tables.** All thirteen `\input` tables are regenerated into a
scratch directory and compared byte for byte with the committed ones. A
separate probe imports each generator in a subprocess and asserts that the
import writes nothing — the S1–S5 generators once emitted at module import,
which destroyed the very files a verifier would have compared.

**Internal consistency.** Figure floats within two pages of their first
citation; no Methods sentence citing a Results float; no overfull box wider
than two points; no undefined reference or citation; the abstract within its
250-word limit; every figure label cited in prose; the thousands separator and
the headline span rendered one way throughout.

**The specific defects found during this work.** Each one that was fixed has a
check that would catch its return. `docs/FIXES.md` is the list. Examples: no
criterion column may print a negative concentration (a factor that was never
divided out); the daggers in the SSD fit table must equal a live
Anderson–Darling recomputation (they were keyed to a different test, marking a
different four substances); the sensitivity table's baseline must equal the
robustness table's baseline (the two had different definitions of testing
effort); the assessment-factor count must match the factor tally three
sections away.

## What it cannot establish

These checks guard the link between the manuscript and the pipeline. **They do
not test the specification.** If a duration rule, a trophic assignment or an
endpoint-admissibility rule is wrong, every number derived from it is
internally consistent and the audit passes.

Several such defects were found during this work, and none of them was found
by the audit. They were found by reading the rendered output — which is why
two independent defect hunts were run over the compiled PDFs and the LaTeX
sources after the audit was already passing at 365 checks, and why those hunts
found a criterion column printing negative concentrations under a `ng/L`
heading with every exceedance at 100 %.

The lesson is in `FIXES.md` rather than in the check count: an audit tells you
that the paper says what the code computes. It does not tell you that the code
computes the right thing, and it does not tell you that the table makes sense.
