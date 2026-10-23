"""
Compound-clustered inference for the DILI early-warning analysis.

Arms from one compound share chemistry, dose series and experimental batch, so they
are not independent. Resampling arms treats 396 observations as exchangeable when the
effective sample size is closer to the 139 compounds. This script recomputes every
reported interval by resampling COMPOUNDS, and reports the per-fold composition that
the arm-level summary hides.

    python3 cluster_inference.py --scores derived/per_arm_scores.csv \
                                 --scores-nf derived/per_arm_scores_no_fibrates.csv \
                                 --out derived/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = np.random.default_rng(42)
N_BOOT = 2000


# --------------------------------------------------------------------------- #
def cluster_index_map(df: pd.DataFrame, cluster_col: str):
    """Precompute cluster -> row indices once; resampling then just concatenates."""
    codes, uniques = pd.factorize(df[cluster_col])
    return [np.flatnonzero(codes == k) for k in range(len(uniques))]


def cluster_boot_indices(groups, rng):
    """Resample whole compounds with replacement, then take all their arms."""
    drawn = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in drawn])


def boot_ci(df, score_col, cluster_col=None, n=N_BOOT, stat=roc_auc_score, seed=42):
    """Bootstrap CI. cluster_col=None reproduces the naive arm-level resampling."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    s = df[score_col].to_numpy()
    groups = cluster_index_map(df, cluster_col) if cluster_col else None
    out = []
    for _ in range(n):
        idx = (cluster_boot_indices(groups, rng) if cluster_col
               else rng.integers(0, len(df), len(df)))
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(stat(y[idx], s[idx]))
    out = np.asarray(out)
    return {"point": float(stat(y, s)),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "width": float(np.percentile(out, 97.5) - np.percentile(out, 2.5)),
            "n_resamples": int(len(out))}


def boot_diff(df, col_a, col_b, cluster_col=None, n=N_BOOT, seed=42):
    """Paired bootstrap of the AUC difference, resampled on the same units."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    a, b = df[col_a].to_numpy(), df[col_b].to_numpy()
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    groups = cluster_index_map(df, cluster_col) if cluster_col else None
    out = []
    for _ in range(n):
        idx = (cluster_boot_indices(groups, rng) if cluster_col
               else rng.integers(0, len(df), len(df)))
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    out = np.asarray(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    p = max(p, 1.0 / (len(out) + 1))
    return {"difference": float(obs), "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "p_bootstrap": float(p),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_resamples": int(len(out))}


def fold_composition(df, cluster_col="compound"):
    """What each leave-one-compound-out fold actually contained."""
    rows = []
    for c, g in df.groupby(cluster_col):
        n_pos = int(g.y.sum())
        rows.append({"held_out_compound": c, "n_arms": len(g), "n_positive": n_pos,
                     "n_negative": len(g) - n_pos,
                     "both_classes": n_pos > 0 and n_pos < len(g),
                     "auc": float(roc_auc_score(g.y, g.score_transcriptomic))
                            if 0 < n_pos < len(g) else None,
                     # what the model saw when this compound was held out
                     "train_compounds": int(df[cluster_col].nunique() - 1),
                     "train_positive_arms": int(df.y.sum() - n_pos)})
    return pd.DataFrame(rows).sort_values(["both_classes", "n_positive"], ascending=False)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scores-nf", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sc = pd.read_csv(a.scores, index_col=0)
    print(f"arms {len(sc)} | positive {int(sc.y.sum())} | compounds {sc.compound.nunique()}")

    R = {"n_arms": int(len(sc)), "n_positive": int(sc.y.sum()),
         "n_compounds": int(sc.compound.nunique())}

    # ---- 1. AUC intervals, naive vs compound-clustered ---------------------
    print("\n=== AUC, arm-resampled vs compound-clustered ===")
    R["auc"] = {}
    for col, label in [("score_transcriptomic", "transcriptomic"),
                       ("dose_level_ordinal", "dose (raw ordinal)"),
                       ("score_dose_model", "dose (fitted, LOCO)")]:
        naive = boot_ci(sc, col, None)
        clust = boot_ci(sc, col, "compound")
        R["auc"][label] = {"arm_resampled": naive, "compound_clustered": clust}
        print(f"  {label:22s} AUC {naive['point']:.4f}")
        print(f"      arm-resampled      [{naive['lo']:.3f}, {naive['hi']:.3f}]"
              f"  width {naive['width']:.3f}")
        print(f"      compound-clustered [{clust['lo']:.3f}, {clust['hi']:.3f}]"
              f"  width {clust['width']:.3f}"
              f"   ({clust['width']/naive['width']:.2f}x wider)")

    # ---- 2. AUC difference ------------------------------------------------
    print("\n=== AUC difference, transcriptomic minus dose ===")
    R["difference"] = {}
    for comp, label in [("dose_level_ordinal", "vs raw dose"),
                        ("score_dose_model", "vs fitted dose model")]:
        naive = boot_diff(sc, "score_transcriptomic", comp, None)
        clust = boot_diff(sc, "score_transcriptomic", comp, "compound")
        R["difference"][label] = {"arm_resampled": naive, "compound_clustered": clust}
        print(f"  {label}: difference {clust['difference']:+.4f}")
        print(f"      arm-resampled      [{naive['lo']:+.3f}, {naive['hi']:+.3f}]"
              f"  p={naive['p_bootstrap']:.4f}  excludes 0: {naive['excludes_zero']}")
        print(f"      compound-clustered [{clust['lo']:+.3f}, {clust['hi']:+.3f}]"
              f"  p={clust['p_bootstrap']:.4f}  excludes 0: {clust['excludes_zero']}")

    # ---- 3. average precision ---------------------------------------------
    print("\n=== average precision, compound-clustered ===")
    R["average_precision"] = {}
    for col, label in [("score_transcriptomic", "transcriptomic"),
                       ("dose_level_ordinal", "dose (raw ordinal)")]:
        c = boot_ci(sc, col, "compound", stat=average_precision_score)
        R["average_precision"][label] = c
        print(f"  {label:22s} AP {c['point']:.4f}  [{c['lo']:.3f}, {c['hi']:.3f}]")

    # ---- 4. fold composition ----------------------------------------------
    print("\n=== leave-one-compound-out fold composition ===")
    fc = fold_composition(sc)
    fc.to_csv(out / "fold_composition.csv", index=False)
    both = fc[fc.both_classes]
    pos_only = fc[(fc.n_positive > 0) & (~fc.both_classes)]
    neg_only = fc[fc.n_positive == 0]
    R["folds"] = {
        "total": int(len(fc)),
        "with_both_classes": int(len(both)),
        "positive_only": int(len(pos_only)),
        "negative_only": int(len(neg_only)),
        "arms_per_fold_median": float(fc.n_arms.median()),
        "arms_per_fold_range": [int(fc.n_arms.min()), int(fc.n_arms.max())],
        "auc_median_evaluable_folds": float(both.auc.median()),
        "auc_iqr_evaluable_folds": [float(both.auc.quantile(.25)),
                                    float(both.auc.quantile(.75))],
        "auc_distribution_evaluable": both.auc.value_counts().to_dict()}
    print(f"  folds total                 : {len(fc)}")
    print(f"    both classes present      : {len(both)}")
    print(f"    positive arms only        : {len(pos_only)}")
    print(f"    negative arms only        : {len(neg_only)}")
    print(f"  arms per fold               : median {fc.n_arms.median():.0f}, "
          f"range {fc.n_arms.min()}-{fc.n_arms.max()}")
    print(f"  per-fold AUC (evaluable)    : median {both.auc.median():.3f}, "
          f"values {sorted(both.auc.unique())}")
    print("  -> per-fold AUC is degenerate: most evaluable folds hold 2-3 arms,")
    print("     so the statistic can only take a few values. Pooled is the")
    print("     meaningful estimate; per-fold AUC should not be interpreted.")

    # ---- 5. no-fibrate cohort ---------------------------------------------
    if a.scores_nf and Path(a.scores_nf).exists():
        nf = pd.read_csv(a.scores_nf, index_col=0)
        # the no-fibrate file has no compound column; recover it from the arm name
        if "compound" not in nf.columns:
            nf["compound"] = [str(i).split(" | ")[0] for i in nf.index]
        print(f"\n=== PPARa-excluded cohort ({len(nf)} arms, "
              f"{nf.compound.nunique()} compounds) ===")
        d = boot_diff(nf, "score_transcriptomic", "dose_level_ordinal", "compound")
        c = boot_ci(nf, "score_transcriptomic", "compound")
        R["no_fibrates"] = {"auc_compound_clustered": c, "difference_compound_clustered": d}
        print(f"  AUC {c['point']:.4f}  [{c['lo']:.3f}, {c['hi']:.3f}]")
        print(f"  difference {d['difference']:+.4f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]"
              f"  p={d['p_bootstrap']:.4f}  excludes 0: {d['excludes_zero']}")

    (out / "cluster_inference.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {out/'cluster_inference.json'} and {out/'fold_composition.csv'}")


if __name__ == "__main__":
    main()
