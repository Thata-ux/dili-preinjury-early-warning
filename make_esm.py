"""Build the Electronic Supplementary Material files for submission.

Archives of Toxicology asks that each ESM file carry the article title, journal,
authors and the corresponding author's affiliation and email. A bare CSV has
nowhere to put that, so each Online Resource is an .xlsx with an `About` sheet
holding the metadata and a `Data` sheet holding the table itself.

    python3 make_esm.py --derived repo_out/derived --out repo_out/esm
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

TITLE = ("Pre-Injury Hepatic Transcriptome Predicts Later-Onset Drug-Induced Liver "
         "Injury in Rat: Compound-Held-Out Evidence Beyond Dose")
JOURNAL = "Archives of Toxicology"
AUTHORS = ("Tahta Herdian Andika; Dewi Damayanti Abdul Karim; "
           "Joana Freire Gameiro; Ferly Ardhy")
CORRESPONDING = ("Dewi Damayanti Abdul Karim, Institut Teknologi Sumatera, "
                 "Indonesia — dewi.abdul@fa.itera.ac.id")
REPO = "https://github.com/Thata-ux/dili-preinjury-early-warning"
RULES = ["union", "any_finding", "moderate_plus", "injury"]


def about(caption: str, source: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Field": ["Article title", "Journal", "Authors", "Corresponding author",
                  "Caption", "Source file in repository", "Repository"],
        "Value": [TITLE, JOURNAL, AUTHORS, CORRESPONDING, caption, source, REPO]})


def write(out: Path, n: int, caption: str, source: str, data: pd.DataFrame):
    path = out / f"ESM_{n}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as x:
        about(caption, source).to_excel(x, sheet_name="About", index=False)
        data.to_excel(x, sheet_name="Data", index=False)
        ws = x.sheets["About"]
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 110
    print(f"  ESM_{n}.xlsx  {len(data):>6,} rows x {data.shape[1]:>2} cols   {caption[:58]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, out = Path(a.derived), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1 — arm inventory under every label rule, stacked with a rule column
    arms = pd.concat([pd.read_csv(d / f"arms_{r}.csv").assign(label_rule=r)
                      for r in RULES], ignore_index=True)
    write(out, 1, "Compound-dose arm inventory with histopathological status by timepoint "
                  "and resulting task labels, under all four label rules.",
          "derived/arms_{rule}.csv", arms)

    # 2 — per-arm scores, primary analysis
    write(out, 2, "Per-arm prediction scores for the transcriptomic model, the fitted dose "
                  "model and ordinal dose level (primary cohort, 396 arms).",
          "derived/per_arm_scores.csv",
          pd.read_csv(d / "per_arm_scores.csv").rename(columns={"Unnamed: 0": "arm"}))

    # 3 — cross-fold selection frequency
    write(out, 3, "Cross-fold feature selection frequency for all genes selected at least "
                  "once, with Entrez identifiers mapped to gene symbols where available.",
          "derived/feature_stability.csv", pd.read_csv(d / "feature_stability.csv"))

    # 4 — SHAP
    write(out, 4, "SHAP attributions on held-out compounds for genes selected in a majority "
                  "of folds.", "derived/shap_stable_genes.csv",
          pd.read_csv(d / "shap_stable_genes.csv"))

    # 5 — permutation null
    write(out, 5, "Permutation null distribution of arm-level AUC over 100 arm-label "
                  "permutations.", "derived/permutation_null.csv",
          pd.read_csv(d / "permutation_null.csv"))

    # 6 — no-fibrate cohort
    write(out, 6, "Per-arm prediction scores with PPAR-alpha agonists excluded from training "
                  "and evaluation (389 arms).", "derived/per_arm_scores_no_fibrates.csv",
          pd.read_csv(d / "per_arm_scores_no_fibrates.csv")
            .rename(columns={"Unnamed: 0": "arm"}))

    # 7 — label-rule re-run, each rule on its own admitted cohort
    rerun = pd.concat([pd.read_csv(d / f"scores_{r}.csv").assign(label_rule=r)
                       for r in RULES], ignore_index=True)
    write(out, 7, "Per-arm prediction scores from the label-rule sensitivity re-run, each "
                  "rule evaluated on its own admitted cohort.",
          "derived/scores_{rule}.csv", rerun)

    # 8 — operating characteristics across thresholds (Table 3 before relocation)
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    sc = pd.read_csv(d / "per_arm_scores.csv", index_col=0)
    y, pr = sc.y.to_numpy(), sc.score_transcriptomic.to_numpy()
    codes, uniq = pd.factorize(sc.compound)
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(42)
    rows = []
    for pct in [0.05, 0.075, 0.10, 0.121, 0.15, 0.20, 0.25, 0.30]:
        k = max(1, int(round(pct * len(y))))
        thr = np.sort(pr)[::-1][k - 1]
        flag = (pr >= thr).astype(int)
        tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
        boot = []
        for _ in range(2000):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if len(np.unique(y[idx])) > 1:
                boot.append(balanced_accuracy_score(y[idx], flag[idx]))
        rows.append({"arms_flagged_pct": round(pct * 100, 1), "arms_flagged": int(flag.sum()),
                     "threshold": round(float(thr), 4),
                     "true_positives": tp, "false_positives": fp,
                     "sensitivity": round(tp / y.sum(), 4),
                     "specificity": round(1 - fp / (len(y) - y.sum()), 4),
                     "precision": round(tp / flag.sum(), 4),
                     "balanced_accuracy": round(balanced_accuracy_score(y, flag), 4),
                     "bacc_ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                     "bacc_ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                     "mcc": round(matthews_corrcoef(y, flag), 4)})
    write(out, 8, "Operating characteristics of the primary model across decision thresholds, "
                  "with compound-clustered confidence intervals on balanced accuracy.",
          "computed from derived/per_arm_scores.csv", pd.DataFrame(rows))

    # 9 — genes selected in every fold, ranked by attribution
    stab = pd.read_csv(d / "feature_stability.csv")
    shap = pd.read_csv(d / "shap_stable_genes.csv")
    key = "gene" if "gene" in shap.columns else shap.columns[0]
    every = stab[stab.folds == stab.folds.max()]
    g9 = (every.merge(shap, on=key, how="left", suffixes=("", "_shap"))
               .sort_values([c for c in shap.columns if "shap" in c.lower()][0],
                            ascending=False))
    write(out, 9, "Mean absolute SHAP attribution on held-out compounds for the genes selected "
                  "in every compound-held-out fold.",
          "derived/feature_stability.csv + shap_stable_genes.csv", g9)

    print(f"\n{len(list(out.glob('ESM_*.xlsx')))} Online Resources in {out}")


if __name__ == "__main__":
    main()
