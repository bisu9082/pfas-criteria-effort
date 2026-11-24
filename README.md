# Support dependence of PFAS aquatic water-quality criteria

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24487897.svg)](https://doi.org/10.5281/zenodo.24487897)

Code and derived data for:

> Kim, J.\*, Yoo, J.\*, Kang, K. **Co-occurring PFAS receive aquatic criteria
> orders of magnitude apart, or none at all: which derivation the evidence
> admits, not measured concentration, decides the risk verdict.**
> (\*equal contribution)

Aquatic water-quality criteria are anchored on an extreme of the available
toxicity data — the lowest reported no-effect concentration, or the fifth
percentile of a species sensitivity distribution. Both move as the evidence
behind them grows. This repository derives PFAS criteria under nine
conventions from one curated ECOTOX extract, joins them to the United States
Water Quality Portal monitoring record, and reports what the derivation
choice costs in the water: on the six substances every convention can assess,
286 of 969 monitoring stations change compliance status on the derivation
choice alone, with no change in a single measured concentration.

## What is here

```
pipeline/     the analysis, one module per concern
data/         the curated inputs the pipeline reads
derived/      criterion and station tables the pipeline writes
tables/       the LaTeX tables as they appear in the paper
figures/      the figures as they appear in the paper
docs/         what the audit does and does not check
```

## Reproducing

```bash
pip install -r pipeline/requirements.txt
python pipeline/run_all.py
```

`run_all.py` regenerates every figure, every `\input` table and the macro file
that feeds the manuscript's repeated numbers, from the deposited data, in one
command. It takes a few minutes; the Monte Carlo steps are seeded.

To check the result against the paper, `python pipeline/audit.py` runs the
cross-document audit. It reads the compiled PDFs rather than the LaTeX
sources, so every value is checked as it is printed, and it regenerates each
generated table into a scratch directory and asserts byte equality with the
committed one. The audit needs `main.pdf` and `si.pdf`, which are not
deposited here; without them it reports the checks it cannot run.

## Data

| file | what it is | rows |
|---|---|---|
| `data/results_parent.csv.gz` | the curated ECOTOX extract, collapsed to parent structures | 12,349 |
| `data/broad_records.parquet` | the acute generalisation set, all structural classes | 106,095 |
| `data/wqp_results.parquet` | the Water Quality Portal pull, the columns the pipeline uses | 117,034 |
| `data/char_map.csv` | characteristic name to parent structure, with CAS and verification formula | 66 |

These are derived extracts, deposited so the analysis runs without a download.
They are not the source releases: the toxicity data are the ECOTOX ASCII
release of 15 September 2026 (US EPA) and the measured concentrations are
public records of the Water Quality Portal, both of which should be obtained
from their publishers for any use beyond reproducing this analysis.

`data/*.parquet` preserve the exact dtypes the analysis was built on. That
matters more than it looks: a CSV round-trip retypes the identifier columns
from string to integer, which reorders every `groupby` and therefore reseeds
every Monte Carlo draw downstream — on that alone the rarefaction slope moved
0.163 to 0.156 and the hazard-control share 63 % to 61 %.

## A note on what the pipeline does not verify

`docs/AUDIT.md` sets out what the audit checks and, more usefully, what it
cannot. The audit guards the link between the manuscript and the pipeline; it
cannot test the specification. If a duration rule, a trophic assignment or an
endpoint-admissibility rule is wrong, every number derived from it is
internally consistent and the audit passes. Several such defects were found
during this work, and none of them was found by the audit.

## Citing this repository

Archived on Zenodo. Cite the concept DOI
[10.5281/zenodo.24487897](https://doi.org/10.5281/zenodo.24487897), which
always resolves to the newest archived version, rather than a version DOI.
`CITATION.cff` carries the same metadata in machine-readable form.

## Licence

Code and documentation: MIT (`LICENSE`).
Derived data tables: CC BY 4.0 (`LICENSE-DATA`).
