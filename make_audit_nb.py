"""Generate Step2d_Unit_And_Null_Audit.ipynb.

Two checks a reviewer can reasonably demand, both answerable with the data already
in the repository plus the expression matrix:

  A. Unit of analysis. The model is fitted on arrays (3 per arm) while scoring is on
     arms. Does aggregating to arms BEFORE feature selection change the result?
  B. Null structure. The permutation shuffles arm labels across compounds, destroying
     within-compound label coherence. Does a compound-respecting null sit higher?
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "repo_out" / "Step2d_Unit_And_Null_Audit.ipynb"


def _lines(t):
    parts = t.rstrip("\n").split("\n")
    return [x + "\n" for x in parts[:-1]] + [parts[-1]]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(t)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(t)}


cells = [
md("""# Step 2d — Unit-of-analysis and null-structure audit

Two properties of the primary pipeline deserve an empirical answer rather than an
argument. Neither affects whether held-out compounds leak — they do not — but both
affect how the result should be described.

**A. Unit of analysis.** Scoring is on arms, but fitting is on arrays: each arm
contributes three day-4 arrays carrying that arm's label. The differential-expression
*t*-test therefore sees roughly three times as many observations as there are
independent outcomes, so its FDR threshold is anticonservative. Every arm contributes
exactly three arrays, so no arm is overweighted relative to another, and selection runs
inside the training fold only — but the feature list may favour genes with low
within-arm variance. This notebook repeats the whole analysis with arrays **averaged to
arms before** differential expression, and compares.

**B. Null structure.** The permutation shuffles labels across arms freely, so an arm can
receive a label from a different chemical. The real data have within-compound label
coherence through dose-response; the null does not. This notebook builds a
**compound-respecting** null, swapping whole label vectors between compounds that have
the same number of arms, and compares the two nulls.

Runtime: roughly 10 minutes for A, and 30-60 minutes for B."""),

md("## 1. Setup — same inputs as Step 2c"),
code("""!pip install xgboost -q

import json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE, EARLY_DAY, N_FEATURES, N_PERM = 42, 4, 100, 100
RULE = "union"

WORK = Path("/content/step2d"); WORK.mkdir(parents=True, exist_ok=True)
STEP1 = Path("/content/derived")
MATRIX = Path("/content/affy_tggates_repeat.tsv")
np.random.seed(RANDOM_STATE)"""),

code("""REPO = "https://github.com/Thata-ux/dili-preinjury-early-warning.git"
if not STEP1.exists():
    !git clone --depth 1 $REPO /content/repo
    STEP1 = Path("/content/repo/derived")

ZEN = "https://zenodo.org/records/20437047/files/affy_tggates_repeat.tsv"
if not MATRIX.exists():
    !wget -q --show-progress -O $MATRIX "$ZEN?download=1"
print("inputs ready")"""),

md("## 2. Pipeline"),
code("""def differential_genes(Xtr, ytr, alpha=0.05, min_lfc=0.5):
    pos, neg = Xtr.values[ytr == 1], Xtr.values[ytr == 0]
    if len(pos) < 2 or len(neg) < 2:
        return Xtr.columns[:500]
    _, p = ttest_ind(pos, neg, axis=0, equal_var=False)
    lfc = pos.mean(0) - neg.mean(0)
    ok = ~np.isnan(p); padj = np.ones_like(p)
    padj[ok] = multipletests(p[ok], method="fdr_bh")[1]
    keep = (padj < alpha) & (np.abs(lfc) >= min_lfc)
    return Xtr.columns[keep] if keep.sum() >= 20 else Xtr.columns[np.argsort(p)[:500]]


def make_model(ytr):
    pos = max(int(ytr.sum()), 1)
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=(len(ytr) - pos) / pos,
                         random_state=RANDOM_STATE, eval_metric="logloss")


def fit_predict(Xtr, ytr, Xte):
    deg = differential_genes(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr[deg], ytr)
    feats = Xtr[deg].columns[np.argsort(rf.feature_importances_)[::-1][:N_FEATURES]]
    return make_model(ytr).fit(Xtr[feats], ytr).predict_proba(Xte[feats])[:, 1]


def fit_predict_fast(Xtr, ytr, Xte, n=40):
    _, p = ttest_ind(Xtr.values[ytr == 1], Xtr.values[ytr == 0], axis=0, equal_var=False)
    feats = Xtr.columns[np.argsort(np.nan_to_num(p, nan=1.0))[:n]]
    pos = max(int(ytr.sum()), 1)
    m = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1,
                      scale_pos_weight=(len(ytr) - pos) / pos,
                      random_state=RANDOM_STATE, eval_metric="logloss")
    return m.fit(Xtr[feats], ytr).predict_proba(Xte[feats])[:, 1]"""),

md("## 3. Load the primary cohort"),
code("""ar = pd.read_csv(STEP1 / f"arms_{RULE}.csv")
ad = ar[ar.status == "admitted"]
om = {(c, d): int(o) for c, d, o in zip(ad.compound, ad.dose_level, ad.outcome)}

lb = pd.read_csv(STEP1 / f"array_labels_{RULE}.csv")
lb["bc"] = lb.BARCODE.astype(str).str.zfill(12)
s = lb[lb.day == EARLY_DAY].copy()
s = s[[(c, d) in om for c, d in zip(s.COMPOUND_NAME, s.DOSE_LEVEL)]]

head = pd.read_csv(MATRIX, sep="\\t", nrows=0); idx_col = head.columns[0]
keep = sorted(set(s.bc) & set(head.columns))
X_all = (pd.read_csv(MATRIX, sep="\\t", usecols=[idx_col] + keep).set_index(idx_col).T)
X_all.index = X_all.index.astype(str).str.zfill(12)
X_all.columns = X_all.columns.astype(str)      # mixed int/str gene ids break sklearn

s = s[s.bc.isin(X_all.index)]
y_s = np.array([om[(c, d)] for c, d in zip(s.COMPOUND_NAME, s.DOSE_LEVEL)])
g_s = s.COMPOUND_NAME.values
a_s = (s.COMPOUND_NAME + " | " + s.DOSE_LEVEL).values
X_s = X_all.loc[s.bc.values]
print(f"arrays {X_s.shape[0]} | arms {len(set(a_s))} | compounds {len(set(g_s))}")
print(f"arrays per arm: {pd.Series(a_s).value_counts().describe()[['min','50%','max']].to_dict()}")"""),

md("""## 4. Check A — aggregate to arms before feature selection

Averaging replicates first makes the *t*-test's sample size equal to the number of
independent outcomes. If the arm-level AUC barely moves, array-level fitting was not
buying anything the arm-level design cannot."""),
code("""Xa = X_s.copy(); Xa["__arm"] = a_s
X_arm = Xa.groupby("__arm").mean()
meta = (pd.DataFrame({"arm": a_s, "y": y_s, "compound": g_s})
          .groupby("arm").agg(y=("y", "max"), compound=("compound", "first")))
meta = meta.loc[X_arm.index]
y_a, g_a = meta.y.to_numpy(), meta.compound.to_numpy()
print(f"arm-level matrix {X_arm.shape} | positives {int(y_a.sum())}")

t0 = time.time()
o = np.full(len(y_a), np.nan)
for tr, te in LeaveOneGroupOut().split(X_arm, y_a, g_a):
    if len(np.unique(y_a[tr])) < 2:
        continue
    o[te] = fit_predict(X_arm.iloc[tr], y_a[tr], X_arm.iloc[te])
k = ~np.isnan(o)
auc_arm = roc_auc_score(y_a[k], o[k])

codes, uniq = pd.factorize(meta.compound[k])
groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
rng = np.random.default_rng(RANDOM_STATE); boot = []
yk, ok_ = y_a[k], o[k]
for _ in range(2000):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    if len(np.unique(yk[idx])) > 1:
        boot.append(roc_auc_score(yk[idx], ok_[idx]))
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

print(f"\\narm-aggregated AUC {auc_arm:.4f}  clustered CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
      f"({time.time()-t0:.0f}s)")
print(f"array-level AUC    0.8324  clustered CI [0.754, 0.894]  (published)")
print(f"difference         {auc_arm-0.8324:+.4f}")
pd.DataFrame({"arm": X_arm.index[k], "y": yk, "p": ok_}).to_csv(
    WORK / "scores_arm_aggregated.csv", index=False)
A = {"auc_arm_aggregated": float(auc_arm), "ci": ci,
     "auc_array_level_published": 0.8324, "difference": float(auc_arm - 0.8324)}
json.dump(A, open(WORK / "check_A_unit.json", "w"), indent=2)"""),

md("""## 5. Check B — a null that respects compound structure

The published null shuffles arm labels freely. Here label vectors are swapped **between
whole compounds that have the same number of arms**, so each compound keeps a label
pattern that some real compound actually had. Both nulls use the light pipeline, and the
observed statistic is recomputed with it, so all three are comparable."""),
code("""arm_truth = meta.loc[sorted(set(a_s))] if False else meta
arm_index = list(X_arm.index)

def loco_light(lbl_by_arm):
    yy = np.array([lbl_by_arm[a] for a in a_s])
    o = np.full(len(yy), np.nan)
    for tr, te in LeaveOneGroupOut().split(X_s, yy, g_s):
        if len(np.unique(yy[tr])) < 2:
            continue
        o[te] = fit_predict_fast(X_s.iloc[tr], yy[tr], X_s.iloc[te])
    k = ~np.isnan(o)
    sc = pd.DataFrame({"arm": a_s[k], "p": o[k]}).groupby("arm").p.mean()
    tr_ = pd.DataFrame({"arm": a_s, "y": yy}).groupby("arm").y.max()
    j = pd.concat([tr_.rename("y"), sc.rename("p")], axis=1).dropna()
    return roc_auc_score(j.y, j.p) if j.y.nunique() > 1 else np.nan

truth_map = dict(zip(meta.index, meta.y))
obs = loco_light(truth_map)
print(f"observed (light pipeline): {obs:.4f}\\n")

# compounds grouped by how many arms they have; vectors are swapped within a size class
by_cmpd = {}
for arm in arm_index:
    by_cmpd.setdefault(meta.loc[arm, "compound"], []).append(arm)
size_classes = {}
for c, arms in by_cmpd.items():
    size_classes.setdefault(len(arms), []).append(c)
print("compounds per arm-count class:",
      {k: len(v) for k, v in sorted(size_classes.items())})

rng = np.random.default_rng(RANDOM_STATE)
null_c = []
t0 = time.time()
for i in range(N_PERM):
    m = {}
    for size, cmpds in size_classes.items():
        donors = list(rng.permutation(cmpds))
        for recv, give in zip(cmpds, donors):
            for arm_r, arm_g in zip(by_cmpd[recv], by_cmpd[give]):
                m[arm_r] = truth_map[arm_g]
    a = loco_light(m)
    if not np.isnan(a):
        null_c.append(a)
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{N_PERM}  mean {np.mean(null_c):.3f}  ({time.time()-t0:.0f}s)")

null_c = np.array(null_c)
null_free = pd.read_csv(STEP1 / "permutation_null.csv").null_auc.values
B = {"observed_light": float(obs),
     "compound_respecting": {"mean": float(null_c.mean()), "sd": float(null_c.std()),
         "n": int(len(null_c)), "n_ge_observed": int((null_c >= obs).sum()),
         "p_value": float(((null_c >= obs).sum() + 1) / (len(null_c) + 1)),
         "z": float((obs - null_c.mean()) / null_c.std())},
     "unrestricted_published": {"mean": float(null_free.mean()),
         "sd": float(null_free.std()), "n": int(len(null_free))}}
pd.DataFrame({"null_auc": null_c}).to_csv(WORK / "permutation_null_compound.csv", index=False)
json.dump(B, open(WORK / "check_B_null.json", "w"), indent=2)
print("\\n", json.dumps(B, indent=2))"""),

md("## 6. Verdict"),
code("""print("CHECK A - unit of analysis")
print(f"  array-level (published) 0.8324 | arm-aggregated {A['auc_arm_aggregated']:.4f} "
      f"| difference {A['difference']:+.4f}")
print("  -> " + ("array-level fitting is not inflating the estimate; the two agree"
                 if abs(A["difference"]) < 0.03 else
                 "the two designs disagree by more than 0.03 AUC - report both"))

print("\\nCHECK B - null structure")
c, u = B["compound_respecting"], B["unrestricted_published"]
print(f"  unrestricted null        {u['mean']:.3f} +/- {u['sd']:.3f}")
print(f"  compound-respecting null {c['mean']:.3f} +/- {c['sd']:.3f}")
print(f"  observed {B['observed_light']:.4f} | p = {c['p_value']:.4f} | z = {c['z']:.2f}")
print("  -> " + ("the stricter null is also near chance; the published p stands"
                 if c["mean"] < 0.58 else
                 "the stricter null is raised - quote the compound-respecting p instead"))

import shutil
shutil.make_archive("/content/step2d_outputs", "zip", WORK)
from google.colab import files
files.download("/content/step2d_outputs.zip")"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
