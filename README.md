# Pre-Injury Hepatic Transcriptome Predicts Later-Onset Drug-Induced Liver Injury in Rat

Analysis code and derived data for a study asking whether rat liver gene expression,
measured while histopathology is still normal, predicts injury that appears only later
in a repeat-dose study — and whether it does so beyond what administered dose alone
explains.

**Headline result.** Under leave-one-compound-out validation across 139 compounds, day-4
expression predicted day-15/29 histopathological injury with an arm-level AUC of
**0.832** (compound-clustered 95% CI 0.754–0.894), against **0.690** for dose level alone
(difference +0.142, clustered 95% CI +0.020 to +0.248, *p* = 0.027).
A conventional random split on the same data returns 0.933, so compound leakage inflates
apparent performance by **0.101 AUC**.

All intervals resample **whole compounds**, not arms: arms from one compound share
chemistry, dose series and batch, so the effective sample size is the 139 compounds.
Arm-level resampling and DeLong's test assume independence and are reported only as
comparators. The model's inputs are **expression values alone** — administered dose is
never a feature.

---

## Why the task is defined this way

Transcriptomic hepatotoxicity classifiers are almost always trained against histopathology
scored in the *same* sample from which expression was measured. Such a model detects injury
that is already present; it does not predict injury yet to occur, and its accuracy says
nothing about early-warning value.

This analysis separates predictor from outcome in time:

| | |
|---|---|
| **Predictor** | day-4 expression, from arms where **no** day-4 animal shows a hepatic finding |
| **Outcome** | confirmed injury at day 15 or 29 in the same compound–dose arm |
| **Validation** | leave-one-compound-out — every prediction concerns unseen chemistry |
| **Baseline** | ordinal dose level, specified in advance |
| **Scoring unit** | the arm (n = 396), not the rat — animals within an arm are replicates sharing one outcome |

Restricting to pre-injury arms also weakens the dose confound structurally: arms already
injured at day 4 are predominantly high-dose, so excluding them drops the dose-only AUC
from 0.893 on the conventional detection task to 0.690 here.

---

## Relation to our other work on this resource

We have used Open TG-GATEs before, in a study that classified hepatotoxicity from
expression and histopathology scored in the **same** sample
([dili-transcriptomic-biomarkers-xai](https://github.com/Thata-ux/dili-transcriptomic-biomarkers-xai)).
That is a concurrent-detection task and a separate piece of work. The present study asks a
different question — whether expression measured while tissue is still normal anticipates
injury that appears later — and shares no model, no label definition and no evaluation
design with it. Section "Why the task is defined this way" explains why the distinction
matters.

---

## Data sources

Nothing in this repository requires special access.

### Open TG-GATEs annotation (~288 KB)

From [dbarchive.biosciencedbc.jp/en/open-tggates](https://dbarchive.biosciencedbc.jp/en/open-tggates/),
LATEST release:

| File | Size | Contents |
|---|---|---|
| `open_tggates_pathology.zip` | 89 KB | histopathology findings: organ, finding type, topography, grade, spontaneous flag |
| `open_tggates_cel_file_attribute.zip` | 200 KB | array → compound, dose level, sacrifice period, animal |

Both are Shift-JIS (`cp932`) encoded, not UTF-8.

**The pathology table records findings only.** An animal with normal histopathology has no
row at all. This was verified against control animals, where 323 of 327 liver findings carry
the spontaneous-lesion flag — consistent with all animals having been examined and only
findings recorded. Absence of a record is therefore treated as absence of a finding.

### Expression matrix (666 MB)

`affy_tggates_repeat.tsv` from [doi.org/10.5281/zenodo.20437047](https://doi.org/10.5281/zenodo.20437047)
(CC BY 4.0) — RMA with BrainArray ENTREZG v22, 6,765 arrays, 143 compounds, 14,132 loci.
Columns are TG-GATEs barcodes zero-padded to 12 digits, so they join one-to-one to the
annotation tables.

```python
bc = metadata.BARCODE.astype(str).str.zfill(12)   # matches the matrix column names
```

This is a third-party deposit, so its array labelling was verified against an independent
RMA normalisation of 189 of the same arrays (see `verify_provenance.py`): the between-sample
similarity structures agree at **r = 0.999**, against r = −0.002 ± 0.133 under permuted
labels.

---

## Data provenance, in detail

Enough to let a reader judge whether the deposit was used correctly.

**Version and identity.** The expression matrix is `affy_tggates_repeat.tsv` from Zenodo
record [20437047](https://doi.org/10.5281/zenodo.20437047), the only version of that record;
6,765 arrays × 14,132 loci, RMA with BrainArray ENTREZG v22 CDF for Rat230_2 (GPL1355).
Annotation is the LATEST release of `open_tggates_pathology` and
`open_tggates_cel_file_attribute`, both Shift-JIS (`cp932`).

**Barcode mapping.** Matrix column names are TG-GATEs barcodes zero-padded to 12 digits.
The join is `metadata.BARCODE.astype(str).str.zfill(12)` and it is one-to-one: all 6,765
barcodes in `array_labels_union.csv` are unique, with no duplicate, no many-to-one collapse
and no array matched to more than one annotation row. Barcodes that fail to join are dropped
rather than imputed, and the counts in `arms_{rule}.csv` are computed after the join so that
every reported arm is backed by arrays actually present in the matrix.

**Replication structure.** Each compound–dose arm at day 4 carries exactly 3 arrays
(min = median = max = 3 across all arms; 1,704 day-4 arrays). Animals within an arm are
biological replicates sharing one arm-level outcome, so sample-level predictions are averaged
within the arm before scoring. No array is reused across arms.

**What the verification covers, and what it does not.** `verify_provenance.py` compares the
deposit against an independent RMA normalisation of raw CEL files. The overlap used was
**189 of the 6,765 arrays (2.8%)**, and within it 30 samples were compared pairwise; the
between-sample similarity structures agree at r = 0.999, against r = −0.002 ± 0.133 under
permuted labels (`provenance_check.json`). This establishes that array **labelling** in the
deposit is correct on the subset tested — that column *n* really is the barcode it claims to
be. It does **not** verify normalisation accuracy across the whole matrix, and it is not a
re-derivation of the deposit. A reader wanting full independence should renormalise the CEL
files for the compounds of interest and rerun Step 2; the script accepts any normalised
matrix with barcode column names.

---

## Execution order

Each step consumes the previous step's output. Runtimes are for Google Colab.

| Step | File | Needs | Runtime |
|---|---|---|---|
| 1 | `Step1b_Feasibility_Corrected.ipynb` | annotation files only | minutes |
| 2 | `Step2_EarlyWarning_Model.ipynb` | step 1 output + expression matrix | 1–2 h |
| 3 | `Step2b_DeLong_Permutation.ipynb` | as step 2 | 3–5 h |
| 4 | `make_figures.py` | step 2/3 output (no matrix) | minutes |
| 3b | `Step2c_Sensitivity_Rerun.ipynb` | step 1 output + expression matrix | ~1.5 h |
| 3c | `Step2d_Unit_And_Null_Audit.ipynb` | step 1 output + expression matrix | ~1 h |
| 5 | `cluster_inference.py` | step 3 output (no matrix) | ~2 min |
| — | `verify_provenance.py` | expression matrix + your own normalisation | minutes |

### Step 1 — feasibility gate

Derives histopathology labels under four rules and counts how many compound–dose arms have
a usable pre-injury window. Downloads only the 288 KB of annotation, so the design can be
judged before committing to the 666 MB matrix.

```
rule               admitted   pos   neg  cmpd  cmpd+  day4 n   pos%
any_finding             345    81   264   136     63    1035  23.5
moderate_plus           409    41   368   140     34    1227  10.0
union (primary)         396    48   348   139     36    1188  12.1
injury (strict)         400    18   382   139     12    1200   4.5
```

Each of these cohorts is evaluated in full by `Step2c_Sensitivity_Rerun.ipynb`; no admitted
arm lacks day-4 expression under any rule.

| rule | arms | pos | AUC (clustered CI) | AP | dose AUC | margin |
|---|---|---|---|---|---|---|
| any_finding | 345 | 81 | 0.840 (0.786-0.888) | 0.629 | 0.708 | +0.132 |
| moderate_plus | 409 | 41 | 0.871 (0.801-0.928) | 0.593 | 0.761 | +0.110 |
| **union** (primary) | 396 | 48 | **0.832** (0.754-0.894) | 0.497 | 0.690 | +0.142 |
| injury | 400 | 18 | 0.782 (0.665-0.881) | 0.120 | 0.570 | +0.212 |

The margin over dose is **widest under the strictest rule**, but `injury` has no usable
operating point: at its 4.5% prevalence threshold the 18 top-scoring arms contain one true
positive (balanced accuracy 0.506, MCC 0.011). The ranking still carries signal — 6 of 18
positives in the top decile — but ranking ability and threshold usability are different
claims and are reported separately.

### Step 2 — model and validation audit

Leave-one-compound-out model, the random-split comparator, the dose baseline, SHAP
attribution on held-out compounds, and sensitivity across all four label rules.

### Step 3 — per-arm scores, DeLong, permutation, fibrate sensitivity

Repeats the LOCO pass while retaining per-arm scores, then runs DeLong's test, a
100-permutation null, and the PPARα-agonist exclusion analysis.

### Step 3b — label-rule sensitivity on each rule's own cohort

The sensitivity cell in Step 2 filtered arms against the barcodes loaded for the primary
`union` rule, so `moderate_plus` (409 admitted arms) and `injury` (400) were evaluated on the
396-arm union cohort rather than their own. This notebook loads the day-4 arrays needed by
all four rules first — 1,239 barcodes in total — then runs leave-one-compound-out separately
per rule. It also computes balanced accuracy and MCC at an a-priori prevalence threshold
rather than the median of held-out predictions, adds compound-clustered intervals, and
checkpoints after each rule. Re-running `union` is deliberate: it must reproduce AUC 0.8324,
and the notebook checks that it does.

### Step 3c — unit-of-analysis and null-structure audit

Two checks a reviewer can reasonably demand. **A:** the model is fitted on arrays (three per
arm) but scored on arms, so the differential-expression test sees ~3x more observations than
there are independent outcomes; this repeats the analysis with arrays averaged to arms before
feature selection. **B:** the published permutation shuffles arm labels freely across
compounds, so the null lacks the within-compound label coherence the real data have; this
builds a compound-respecting null by swapping whole label vectors between compounds with the
same arm count. Neither affects leakage — held-out compounds are never seen in training — but
both affect how the result should be described.

### Step 5 — compound-clustered inference and fold composition

Recomputes every interval by resampling compounds instead of arms, and reports what each
leave-one-compound-out fold actually contained. Reads the deposited score files only.

```
python3 cluster_inference.py --scores derived/per_arm_scores.csv \
                             --scores-nf derived/per_arm_scores_no_fibrates.csv \
                             --out derived/
```

### Step 4 — figures

Reads the deposited score files; the expression matrix is **not** needed.

---

## Label definition

The primary rule, `union`, counts an animal as injured when it has either an injurious
lesion at any grade (necrosis, degeneration, fibrosis, hemorrhage, cholestasis) **or** any
finding graded moderate or severe.

The second clause governs hypertrophy, the most frequent finding in the resource (1,146
non-spontaneous liver records) and generally an adaptive response. Retaining it at moderate
or severe grade while excluding it at minimal or slight grade is **an operational decision,
not a statement of biological fact** — no severity threshold separates adaptation from
injury in any absolute sense. The strict comparator `injury` admits no hypertrophy at any
grade and is reported throughout beside the primary rule.

Spontaneous lesions are excluded everywhere: they are 39% of all findings and almost all
control-animal findings.

**Terminology audit.** Of 54 non-spontaneous liver finding types, 12 match the injurious
pattern. Two categories — mortality (`DEAD`, 148 records) and hepatocellular adenoma (4) —
carry grade `P` rather than a severity term and so escape the grade clause by construction;
counting them changes **no** arm's status or outcome. A deliberately over-inclusive rule
adding mortality, neoplasia, inflammation, infiltration, ductular and oval-cell reaction,
pyknosis, granuloma, edema and atrophy alters **4 of 425 arms** (admitted 396 → 393,
positive 48 → 47). The result is insensitive to the terminology mapping.

---

## Results summary

| Evaluation | Arm-level AUC | Compound-clustered 95% CI |
|---|---|---|
| Random split (conventional design) | 0.933 | — |
| **Leave-one-compound-out** | **0.832** | **0.754–0.894** |
| Transcriptome + dose (two-feature combiner) | 0.802 | — |
| Dose level only, LOCO | 0.580 | — |
| Dose level only, raw ordinal | 0.690 | — |

Adding dose on top of the transcriptomic score does not help (−0.030, clustered CI
−0.087 to +0.034, *p* = 0.34): early expression already carries the predictive content
of exposure level.

**Significance.** Compound-clustered paired bootstrap vs raw dose: difference +0.142,
95% CI +0.020 to +0.248, *p* = 0.027. Arm-level resampling gives +0.050 to +0.238
(*p* = 0.0040) and DeLong *z* = 2.89 (*p* = 0.0039); both assume independence and so
overstate the evidence.

**Fold composition.** Of 139 folds, 29 held both outcome classes, 7 positives only and
103 negatives only, with a median of 3 arms per fold. Per-fold AUC takes only the values
0, 0.5 and 1 and must not be read as between-fold variability; the pooled estimate with
its clustered interval is the meaningful figure. See `derived/fold_composition.csv`.

**Threshold.** Balanced accuracy 0.727 and Matthews correlation 0.455 are computed at an
a-priori threshold flagging the top 12.1% of arms (the design prevalence). An earlier
version used the median of held-out predictions, which leaks the test distribution and
*understates* both figures (0.702 and 0.263) by flagging half of all arms. Permutation over 100 arm-label shuffles: null 0.493 ± 0.066, zero permutations
reached the observed value, so *p* is bounded at 1/101 = 0.0099, with the observed statistic
5.3 SD above the null mean.

**PPARα agonists.** Peroxisome proliferation is a rat-specific mechanism. Removing
clofibrate, fenofibrate and gemfibrozil from training *and* evaluation raised AUC to
**0.890** over the remaining 136 compounds (dose 0.711, difference +0.179,
*p* = 1.5 × 10⁻⁴). WY-14643, the prototypical peroxisome proliferator, never enters the
admitted set: it injures the liver by day 4.

**Genes.** 49 genes were selected at least once across the 139 folds, 41 in at least half,
and **25 in every fold**. The stable set is dominated by the Nrf2 antioxidant battery
(Gstm3, Gsta3, Gstt3, Nqo1, Gpx2, Akr7a3, Ephx1, Aldh1a1) with phase I–III xenobiotic
metabolism (Cyp1a1, Ugt2b1, Ugt2b7, Abcc3) and lipid handling (Acot1, Acaa1a, Acsm5, Me1,
Cidea, Apoa4). The signature is therefore **adaptive rather than injurious**: the magnitude
of the early stress response, not any early damage marker, is what anticipates whether
injury follows.

One stable feature is not protein-coding. `LOC108349824`
([RGD:11493970](https://rgd.mcw.edu/rgdweb/report/gene/main.html?id=11493970)) is an
uncharacterized lncRNA on rat chromosome 1q51, selected in all 139 folds, expressed at the
85th percentile across loci, and 1.39 log₂ higher at day 4 in arms that later develop injury
(*p* = 5.1 × 10⁻²³). Note that BrainArray reassigns probes designed in 2003 to loci annotated
since, so this locus needs targeted confirmation before being treated as a biomarker.

---

## Repository contents

```
Step1b_Feasibility_Corrected.ipynb    arm inventory and label rules
Step2_EarlyWarning_Model.ipynb        LOCO model, validation audit, SHAP
Step2b_DeLong_Permutation.ipynb       per-arm scores, DeLong, permutation, fibrates
make_figures.py                       Figures 1-4 (300 dpi PNG + vector PDF)
Step2c_Sensitivity_Rerun.ipynb        label-rule sensitivity, each rule on its own cohort
cluster_inference.py                  compound-clustered intervals, fold composition
verify_provenance.py                  deposit verification against your own normalisation

derived/
  arms_{rule}.csv                     arm classification under each label rule
  array_labels_{rule}.csv             per-array labels joined to barcodes
  per_arm_scores.csv                  transcriptomic, dose-model and ordinal-dose scores
  per_arm_scores_no_fibrates.csv      same, PPARalpha agonists excluded
  permutation_null.csv                100 null AUCs
  feature_stability.csv               cross-fold selection frequency
  shap_stable_genes.csv               SHAP attributions
  provenance_check.json               deposit verification result
  step2_results.json                  evaluation designs and label sensitivity
  step2b_results.json                 DeLong, permutation, fibrate sensitivity
  cluster_inference.json              compound-clustered intervals for every statistic
  fold_composition.csv                arms, classes and AUC per leave-one-compound-out fold

figures/                              Figures 1-4 as published
```

---

## Reproducing a specific number

Most reported values need no refitting, because per-arm scores are deposited:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sc = pd.read_csv("derived/per_arm_scores.csv", index_col=0)
roc_auc_score(sc.y, sc.score_transcriptomic)   # 0.8324
roc_auc_score(sc.y, sc.dose_level_ordinal)     # 0.6901
average_precision_score(sc.y, sc.score_transcriptomic)   # 0.4968
```

The DeLong implementation is embedded in `Step2b_DeLong_Permutation.ipynb` and self-tests
on load: a score compared with itself must return *p* = 1, and its AUC must match
`sklearn.metrics.roc_auc_score` exactly.

---

## Known limitations

- Outcomes are linked at **compound–dose arm** level, not within animal: TG-GATEs uses
  terminal sampling, so no animal is followed over time. An arm label derived from three
  animals is correspondingly coarse.
- The positive class is 48 arms (12.1%), so intervals are wide and per-compound AUC is
  uninformative — of the 29 folds containing both classes, most hold two or three arms,
  forcing AUC to 0 or 1. Pooled performance is the meaningful figure.
- Compounds that injure the liver **within** the predictor window are excluded by
  construction, so the approach says nothing about rapidly acting hepatotoxicants.
- Microarray, not RNA-seq. Rat, not human. Both require separate validation.
- **No external validation.** Nothing here is external validation, and it should not be
  described as such. That would need an independent repeat-dose study with early
  transcriptomics, early histopathology to establish the pre-injury state, and late
  histopathology for the outcome; we know of no public dataset meeting all three.
  Compound withholding and clustered intervals are a stricter internal test than the
  random splits usual in this literature, but internal resampling cannot expose platform,
  laboratory or scoring-convention effects.

---

## Citation

Andika T.H., Abdul Karim D.D., Gameiro J.F., Ardhy F.
*Pre-Injury Hepatic Transcriptome Predicts Later-Onset Drug-Induced Liver Injury in Rat:
Compound-Held-Out Evidence Beyond Dose.* Manuscript in preparation.

Corresponding author: Dewi Damayanti Abdul Karim, Institut Teknologi Sumatera —
dewi.abdul@fa.itera.ac.id

Please also cite the underlying data:

- Igarashi Y. *et al.* (2015) Open TG-GATEs: a large-scale toxicogenomics database.
  *Nucleic Acids Research* 43(D1):D921–D927. [doi:10.1093/nar/gku955](https://doi.org/10.1093/nar/gku955)
- Kumar A. (2026) Unified rat-liver microarray expression matrix for 446 DILI-relevant
  drugs. Zenodo. [doi:10.5281/zenodo.20437047](https://doi.org/10.5281/zenodo.20437047)

## Licence

Code released under the MIT Licence. Derived data files are CC BY 4.0, consistent with the
upstream Zenodo deposit.
