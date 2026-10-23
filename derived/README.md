# Derived data

Every file here is produced by the notebooks in the repository root. Together they let a
reader recompute all reported figures and statistics **without downloading the 666 MB
expression matrix and without refitting any model**.

## Arm-level labels and inventory

Produced by `Step1b_Feasibility_Corrected.ipynb`, one set per label rule
(`union`, `injury`, `any_finding`, `moderate_plus`, `injury_moderate`).

| File | Contents |
|---|---|
| `arms_{rule}.csv` | one row per compound–dose arm: `early_rate`, `late_rate_max`, `status`, `outcome` |
| `inventory_{rule}.csv` | finding rate per compound × dose × day, before arm classification |
| `array_labels_{rule}.csv` | per-array labels joined to TG-GATEs barcodes |

`status` takes one of `admitted`, `already_injured_at_early`, `no_early_timepoint`,
`no_late_timepoint`. Only `admitted` arms enter the model; `outcome` is defined for those
rows alone.

The primary rule is `union`: an animal counts as injured when it has an injurious lesion at
any grade (necrosis, degeneration, fibrosis, hemorrhage, cholestasis) **or** any finding
graded moderate or severe. The strict comparator is `injury`, which admits only the
injurious lesion types and no hypertrophy at any grade.

## Model predictions

Produced by `Step2b_DeLong_Permutation.ipynb`.

| File | Contents |
|---|---|
| `per_arm_scores.csv` | 396 arms: `y`, `compound`, `score_transcriptomic`, `score_dose_model`, `dose_level_ordinal` |
| `per_arm_scores_no_fibrates.csv` | 389 arms, PPARα agonists removed from training and evaluation |
| `permutation_null.csv` | 100 null AUCs from arm-label permutation |

Scores are out-of-fold: each arm was predicted by a model that never saw its compound.
Sample-level predictions were averaged within an arm, because animals in an arm are
replicates sharing one outcome.

## Feature attribution

| File | Contents |
|---|---|
| `feature_stability.csv` | every gene selected at least once: `gene` (Entrez), `symbol`, `folds`, `pct` |
| `shap_stable_genes.csv` | mean absolute SHAP on held-out compounds for genes passing 50% fold selection |

49 genes were selected at least once across the 139 folds, 41 in at least half, 25 in every
fold. Interpret only genes recurring across folds; one selected in a minority of partitions
may reflect that partition rather than a reproducible response.

`symbol` falls back to the Entrez identifier when MyGene has no mapping. One such entry,
`108349824`, is `LOC108349824` — an uncharacterized lncRNA on rat chromosome 1q51
(RGD:11493970), selected in all 139 folds.

## Results

| File | Contents |
|---|---|
| `step2_results.json` | evaluation designs (random split, LOCO, dose), per-compound breakdown, label-rule sensitivity |
| `step2b_results.json` | per-arm AUC and AP, both DeLong tests, permutation summary, fibrate sensitivity |
| `provenance_check.json` | agreement between the Zenodo deposit and an independent normalisation |
| `cluster_inference.json` | every interval recomputed by resampling compounds rather than arms |
| `fold_composition.csv` | per-fold arm counts, class balance and AUC for all 139 folds |

## Recomputing the headline numbers

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sc = pd.read_csv("per_arm_scores.csv", index_col=0)

roc_auc_score(sc.y, sc.score_transcriptomic)           # 0.8324
roc_auc_score(sc.y, sc.dose_level_ordinal)             # 0.6901
roc_auc_score(sc.y, sc.score_dose_model)               # 0.5799
average_precision_score(sc.y, sc.score_transcriptomic) # 0.4968

nf = pd.read_csv("per_arm_scores_no_fibrates.csv", index_col=0)
roc_auc_score(nf.y, nf.score_transcriptomic)           # 0.8899
roc_auc_score(nf.y, nf.dose_level_ordinal)             # 0.7113
```

The permutation p-value follows from `permutation_null.csv`: no null value reaches the
observed statistic of 0.8461, so the empirical p is bounded at 1/101 = 0.0099 rather than
estimated.

```python
import numpy as np
null = pd.read_csv("permutation_null.csv").null_auc
(null >= 0.8461446360153256).sum()        # 0
(null.mean(), null.std())                 # 0.4928, 0.0666
```

## Compound-clustered intervals

`cluster_inference.json` is the authoritative source for every confidence interval in the
manuscript. Produced by `cluster_inference.py` from the score files above — no refitting,
no expression matrix. Each entry carries both `arm_resampled` and `compound_clustered`
variants so the effect of the dependence structure is visible:

```python
import json
R = json.load(open("cluster_inference.json"))
R["auc"]["transcriptomic"]["compound_clustered"]     # 0.8324, [0.754, 0.894]
R["difference"]["vs raw dose"]["compound_clustered"] # +0.142, [+0.020, +0.248], p=0.027
R["folds"]["with_both_classes"]                      # 29 of 139
```

`fold_composition.csv` records what each fold held. The `auc` column is null for the 110
folds containing a single outcome class, and among the other 29 it takes only 0, 0.5 or 1,
because those folds hold two or three arms. It is deposited for transparency about the
design, not as a performance measure.
