"""
Publication figures for the DILI early-warning manuscript.

Reads the outputs of Step 1b / Step 2 / Step 2b and writes four figures at
300 dpi (PNG) plus vector PDF.

    python3 make_figures.py --step1 /content/tggates_annot \
                            --step2 /content/step2 \
                            --out   /content/figures

Figure 1  Study design and arm selection
Figure 2  ROC and precision-recall, transcriptomic vs dose
Figure 3  Performance across evaluation designs and sensitivity analyses
Figure 4  Cross-fold gene stability and SHAP attribution

Palette is the validated CVD-safe categorical set (deutan dE 9.2, normal dE 27.6).
Every series also carries a direct label or legend, so identity never rests on
colour alone.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# ---- design tokens ---------------------------------------------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.7,
    "xtick.color": INK3, "ytick.color": INK3,
    "text.color": INK, "axes.labelcolor": INK2,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

GENE_GROUPS = {
    "Nrf2 / antioxidant": {"Gstm3", "Gsta3", "Gstt3", "Nqo1", "Gpx2", "Akr7a3",
                           "Ephx1", "Aldh1a1", "Srxn1", "Gclc", "Gclm", "Txnrd1"},
    "Xenobiotic metabolism": {"Cyp1a1", "Ugt2b1", "Ugt2b7", "Abcc3", "Cyp2c11",
                              "Aadac", "Nat1", "Cyp4b1", "Abcb1b"},
    "Lipid handling": {"Acot1", "Acot2", "Acaa1a", "Acsm5", "Me1", "Cidea",
                       "Apoa4", "Lipc", "Angptl3", "Scd"},
}
GROUP_COLOR = {"Nrf2 / antioxidant": BLUE, "Xenobiotic metabolism": ORANGE,
               "Lipid handling": AQUA, "Other": VIOLET}

def gene_group(sym: str) -> str:
    for g, members in GENE_GROUPS.items():
        if str(sym) in members:
            return g
    return "Other"


def boot_ci(y, s, clusters=None, n=2000):
    """Bootstrap AUC interval.

    `clusters` gives a compound label per arm. Arms from one compound share chemistry,
    dose series and batch, so they are resampled together: the effective sample size is
    the number of compounds, not the number of arms. Passing clusters=None reproduces
    the naive arm-level interval and is kept only for comparison.
    """
    y, s = np.asarray(y), np.asarray(s)
    groups = None
    if clusters is not None:
        codes, uniques = pd.factorize(np.asarray(clusters))
        groups = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    out = []
    for _ in range(n):
        if groups is None:
            i = RNG.integers(0, len(y), len(y))
        else:
            i = np.concatenate([groups[k] for k in
                                RNG.integers(0, len(groups), len(groups))])
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def compounds_of(df):
    """Compound label per arm; the no-fibrate file carries it only in the index."""
    if "compound" in df.columns:
        return df["compound"].to_numpy()
    return np.array([str(i).split(" | ")[0] for i in df.index])


#: Springer submission names, keyed by this script's descriptive name.
SPRINGER_NAME = {"Figure1_study_design": "Fig1", "Figure2_roc_pr": "Fig2",
                 "Figure3_forest": "Fig3", "Figure4_gene_stability": "Fig4"}

#: Set by main(). When true, each figure is also written as the submission
#: wants it: EPS vector plus 600 dpi TIFF (combination art), named FigN.
SPRINGER = False


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    written = f"{name}.png / .pdf"

    if SPRINGER and name in SPRINGER_NAME:
        sub = out / "springer"; sub.mkdir(exist_ok=True)
        stem = SPRINGER_NAME[name]
        # EPS has no alpha channel; a transparent patch would rasterise oddly.
        fig.patch.set_alpha(1.0)
        fig.savefig(sub / f"{stem}.eps", format="eps")
        fig.savefig(sub / f"{stem}.tif", format="tiff", dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"})
        written += f"  ->  springer/{stem}.eps / .tif"

    plt.close(fig)
    print(f"  wrote {written}")


# ---------------------------------------------------------------- figure 1 --
def figure1(arms: pd.DataFrame, out: Path):
    counts = arms.status.value_counts()
    n_total = len(arms)
    n_adm = int(counts.get("admitted", 0))
    n_inj = int(counts.get("already_injured_at_early", 0))
    n_nolate = int(counts.get("no_late_timepoint", 0))
    adm = arms[arms.status == "admitted"]
    n_pos, n_neg = int(adm.outcome.sum()), int((adm.outcome == 0).sum())
    n_cmp = int(adm.compound.nunique())

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f6f6f4", ec=INK3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.10",
                                    facecolor=fc, edgecolor=ec, linewidth=0.9))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                fontsize=7.1, color=INK2, linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=INK3, linewidth=0.8))

    box(2.9, 8.05, 4.4, 1.75, "Treated compound-dose arms",
        f"{n_total} arms with a day-4\nand a later timepoint")

    # exclusion branches off the trunk, not off the admitted box
    arrow(5.1, 8.05, 5.1, 6.55)
    arrow(5.1, 7.62, 2.52, 7.62)
    box(0.05, 6.95, 2.35, 1.42, "Excluded",
        f"injured at day 4: {n_inj}\nno later timepoint: {n_nolate}",
        fc="#fbf1ec", ec=ORANGE)

    box(2.5, 4.95, 5.2, 1.6, "Admitted: pre-injury arms",
        f"{n_adm} arms | {n_cmp} compounds\nno day-4 animal shows a finding",
        fc="#eef4fc", ec=BLUE)

    arrow(4.2, 4.95, 3.3, 3.95)
    arrow(6.0, 4.95, 6.9, 3.95)
    box(1.45, 2.35, 2.7, 1.6, "Progress to injury",
        f"{n_pos} arms\n({100*n_pos/max(n_adm,1):.1f}%)", fc="#fbf1ec", ec=ORANGE)
    box(6.05, 2.35, 2.7, 1.6, "No later injury",
        f"{n_neg} arms\n({100*n_neg/max(n_adm,1):.1f}%)", fc="#eef9f4", ec=AQUA)

    ax.text(5.0, 1.55, "Predictor: day-4 expression     \u2192     "
                       "Outcome: histopathology at day 15 / 29",
            ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.text(5.0, 0.62, "Whole compounds are held out; arms are the scoring unit, because\n"
                       "animals within an arm are replicates sharing a single outcome.",
            ha="center", va="center", fontsize=7.2, color=INK2, linespacing=1.6)
    save(fig, out, "Figure1_study_design")


# ---------------------------------------------------------------- figure 2 --
def figure2(sc: pd.DataFrame, out: Path):
    y = sc.y.values
    tx, dose = sc.score_transcriptomic.values, sc.dose_level_ordinal.values
    auc_t, auc_d = roc_auc_score(y, tx), roc_auc_score(y, dose)
    ap_t, ap_d = average_precision_score(y, tx), average_precision_score(y, dose)
    lo, hi = boot_ci(y, tx, compounds_of(sc))
    prev = y.mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for s, c, lab in [(tx, BLUE, "Day-4 transcriptome"),
                      (dose, ORANGE, "Dose level alone")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, lw=2.0, label=lab, zorder=3,
                solid_capstyle="round")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("A   Discrimination of later injury", loc="left")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.97, 0.16, f"AUC {auc_t:.3f}\n95% CI {lo:.3f}–{hi:.3f}", ha="right",
            va="bottom", fontsize=7.6, color=BLUE, fontweight="bold", linespacing=1.5)
    ax.text(0.97, 0.05, f"AUC {auc_d:.3f}", ha="right", va="bottom",
            fontsize=7.6, color=ORANGE, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)

    ax = axes[1]
    ax.axhline(prev, color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for s, c, lab in [(tx, BLUE, "Day-4 transcriptome"),
                      (dose, ORANGE, "Dose level alone")]:
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, color=c, lw=2.0, label=lab, zorder=3, solid_capstyle="round")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("B   Precision–recall (12% positive)", loc="left")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.97, 0.95, f"Transcriptome  AP {ap_t:.3f}", ha="right", va="top",
            fontsize=7.4, color=BLUE, fontweight="bold")
    ax.text(0.97, 0.86, f"Dose alone  AP {ap_d:.3f}", ha="right", va="top",
            fontsize=7.4, color=ORANGE, fontweight="bold")
    ax.text(0.03, prev + 0.03, "prevalence", fontsize=7, color=INK3)

    fig.tight_layout()
    save(fig, out, "Figure2_roc_pr")


# ---------------------------------------------------------------- figure 3 --
def figure3(sc, sc_nf, res2, res2b, out: Path, sens_dir: Path = None):
    """Forest plot. CI is drawn where per-arm scores exist; otherwise a diamond
    marks the point estimate and the caption says so."""
    SEC_FULL = "Full cohort (396 arms)"
    SEC_NOFIB = "PPAR\u03b1 agonists excluded (389 arms)"
    SEC_LABEL = "Label-rule sensitivity"
    entries = []          # (section, label, auc, lo, hi, colour, has_ci)

    def with_ci(section, label, y, score, colour, clusters):
        a = roc_auc_score(y, score)
        lo, hi = boot_ci(y, score, clusters)
        entries.append((section, label, a, lo, hi, colour, True))

    def point(section, label, a, colour):
        entries.append((section, label, a, None, None, colour, False))

    cmp_full = compounds_of(sc)
    with_ci(SEC_FULL, "Day-4 transcriptome", sc.y.values,
            sc.score_transcriptomic.values, BLUE, cmp_full)
    with_ci(SEC_FULL, "Dose level alone", sc.y.values,
            sc.dose_level_ordinal.values, ORANGE, cmp_full)
    rs = ((res2 or {}).get("A_random_split") or {}).get("auc")
    if rs is not None:
        point(SEC_FULL, "Random split (conventional)", rs, VIOLET)

    if sc_nf is not None:
        cmp_nf = compounds_of(sc_nf)
        with_ci(SEC_NOFIB, "Day-4 transcriptome", sc_nf.y.values,
                sc_nf.score_transcriptomic.values, BLUE, cmp_nf)
        with_ci(SEC_NOFIB, "Dose level alone", sc_nf.y.values,
                sc_nf.dose_level_ordinal.values, ORANGE, cmp_nf)

    # Label rules: prefer the Step 2c re-run, where each rule is evaluated on its own
    # admitted cohort and per-arm scores exist, so these carry clustered intervals too.
    labels = [("union", "union (primary)"), ("any_finding", "any finding"),
              ("moderate_plus", "moderate or worse"), ("injury", "injurious lesions")]
    sens = (res2 or {}).get("sensitivity", {})
    for key, lbl in labels:
        f = sens_dir / f"scores_{key}.csv" if sens_dir else None
        if f is not None and f.exists():
            g = pd.read_csv(f)
            with_ci(SEC_LABEL, lbl, g.y.values, g.p.values, VIOLET, compounds_of(g))
        elif key in sens and sens[key].get("auc") is not None:
            point(SEC_LABEL, lbl, sens[key]["auc"], VIOLET)

    sections = []
    for sec, *_ in entries:
        if sec not in sections:
            sections.append(sec)

    n_rows = len(entries)
    fig, ax = plt.subplots(figsize=(6.5, 0.33 * n_rows + 0.62 * len(sections) + 0.85))

    ypos, ylab = [], []
    hdr = []
    y = 0.0
    for sec in sections:
        y -= 0.95
        hdr.append((y, sec))
        for e in [e for e in entries if e[0] == sec]:
            _, label, a, lo, hi, c, has_ci = e
            y -= 0.78
            if has_ci:
                ax.plot([lo, hi], [y, y], color=c, lw=1.7,
                        solid_capstyle="round", zorder=2)
                ax.plot([a], [y], "o", color=c, ms=6, mec="white", mew=1.2, zorder=3)
                tx_x = hi
            else:
                ax.plot([a], [y], "D", color=c, ms=5.2, mec="white", mew=1.0, zorder=3)
                tx_x = a
            ax.text(tx_x + 0.008, y, f"{a:.3f}", va="center", fontsize=7.3,
                    color=c, fontweight="bold")
            ypos.append(y); ylab.append(label)

    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(y - 0.7, 0.55)
    ax.axvline(0.5, color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(0.5, 0.18, "chance", ha="center", va="center", fontsize=7, color=INK3)

    for yy, sec in hdr:
        ax.text(0.447, yy, sec, fontsize=7.7, fontweight="bold",
                color=INK, va="center", ha="left")

    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=7.7, color=INK2)
    ax.set_xlabel("Arm-level AUC")
    ax.set_title("Performance across evaluation designs and sensitivity analyses",
                 loc="left", pad=14)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.text(0.015, 0.005,
             "Circles carry a 95% compound-clustered bootstrap CI (whole compounds "
             "resampled, not arms). Diamonds are point estimates: per-arm scores were "
             "not retained for those variants.",
             fontsize=6.7, color=INK3)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, out, "Figure3_forest")


# ---------------------------------------------------------------- figure 4 --
def figure4(stab: pd.DataFrame, shap_df: pd.DataFrame, out: Path, top=20):
    d = shap_df.merge(stab[["gene", "folds", "pct"]], on="gene", how="left")
    d["symbol"] = d.symbol.fillna(d.gene.astype(str))
    d = d.sort_values("mean_abs_shap", ascending=False).head(top).iloc[::-1]
    d["group"] = d.symbol.map(gene_group)

    fig, ax = plt.subplots(figsize=(6.3, 0.30 * len(d) + 1.6))
    ypos = np.arange(len(d))
    colors = [GROUP_COLOR[g] for g in d.group]
    ax.barh(ypos, d.mean_abs_shap, height=0.62, color=colors,
            edgecolor="white", linewidth=0.8, zorder=3)

    xmax = d.mean_abs_shap.max()
    for i, (v, pct) in enumerate(zip(d.mean_abs_shap, d.pct)):
        lbl = f"{v:.2f}" + (f"   {pct:.0f}% of folds" if pd.notna(pct) else "")
        ax.text(v + xmax * 0.015, i, lbl, va="center", fontsize=6.9, color=INK2)

    ax.set_yticks(ypos); ax.set_yticklabels(d.symbol, fontsize=7.6,
                                            fontstyle="italic", color=INK)
    ax.set_xlabel("Mean |SHAP| on held-out compounds")
    ax.set_title("Genes carrying the anticipatory signal", loc="left")
    ax.set_xlim(0, xmax * 1.34)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    present = [g for g in GROUP_COLOR if g in set(d.group)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in present]
    ax.legend(handles, present, frameon=False, fontsize=7.2, ncol=2,
              loc="lower right", handlelength=1.1, borderpad=0.2)
    fig.tight_layout()
    save(fig, out, "Figure4_gene_stability")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", default="/content/tggates_annot")
    ap.add_argument("--step2", default="/content/step2")
    ap.add_argument("--out", default="/content/figures")
    ap.add_argument("--rule", default="union")
    ap.add_argument("--springer", action="store_true",
                    help="also write FigN.eps and 600 dpi FigN.tif for submission")
    a = ap.parse_args()
    global SPRINGER
    SPRINGER = a.springer
    s1, s2, out = Path(a.step1), Path(a.step2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def maybe(path, loader, what):
        if path.exists():
            return loader(path)
        print(f"  ! {what} not found at {path} - skipping what needs it")
        return None

    arms = maybe(s1 / f"arms_{a.rule}.csv", pd.read_csv, "arm classification")
    sc = maybe(s2 / "per_arm_scores.csv", lambda p: pd.read_csv(p, index_col=0),
               "per-arm scores")
    sc_nf = maybe(s2 / "per_arm_scores_no_fibrates.csv",
                  lambda p: pd.read_csv(p, index_col=0), "no-fibrate scores")
    stab = maybe(s2 / "feature_stability.csv", pd.read_csv, "feature stability")
    shap_df = maybe(s2 / "shap_stable_genes.csv", pd.read_csv, "SHAP table")
    res2 = maybe(s2 / "step2_results.json", lambda p: json.loads(p.read_text()),
                 "step2 results")
    res2b = maybe(s2 / "step2b_results.json", lambda p: json.loads(p.read_text()),
                  "step2b results")

    print("building figures ...")
    if arms is not None:
        figure1(arms, out)
    if sc is not None:
        figure2(sc, out)
        figure3(sc, sc_nf, res2, res2b, out, sens_dir=s2)
    if stab is not None and shap_df is not None:
        figure4(stab, shap_df, out)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
