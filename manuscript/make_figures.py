"""Rebuilds every figure of the manuscript from the original experiment outputs.

Run:   python make_figures.py        (needs numpy, pandas, matplotlib; no GPU, no internet)
Input: ../results/  = the output of the merged Kaggle session publict/eksperiment-v2 (version 6),
       i.e. what train.py / xai_utils.py / analysis.py wrote for the 45 trained models.
Output: figures/figN_*.png|pdf  and  figures/data/figN_*.csv (the exact numbers each figure draws).

Figure -> source files (all under ../results/)
  fig1_design_drawio  none: study-design diagram drawn by the authors in draw.io (no measured data)
  fig2_accuracy_cost  summary/runs.csv (test_macro_f1 of each run) + summary/efficiency_benchmark.csv (gmacs)
  fig3_forest         summary/xai_paired_tests.csv + summary/reliance_paired_tests.csv (paired bootstrap per image)
  fig4_sanity         summary/sanity_cascade.csv (Spearman correlation after cascading randomisation)
  fig5_erf            results_v3/main/<dataset>/<variant>/<run>/erf.npy (gradient maps saved by each trained model)
Nothing is typed in by hand: the script only reads these files, and check() fails if they disagree.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results")
SUM = os.path.join(RES, "summary")
OUT = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 300})
COL = {"s": "#2a78d6", "m": "#eb6834", "l": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS = [("busi", "BUSI"), ("ham10000", "HAM10000"), ("siim", "SIIM-ACR")]
V = [("s", "S"), ("m", "M"), ("l", "L")]


def dump(name, df):
    """Write the numbers behind a figure next to it, so each plotted value can be traced and re-plotted."""
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "data", f"{name}.csv"), index=False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: accuracy vs cost
def fig_acc_cost():
    runs = pd.read_csv(os.path.join(SUM, "runs.csv"))
    bench = pd.read_csv(os.path.join(SUM, "efficiency_benchmark.csv"))
    gm = bench[bench.img_size == 384].set_index("variant").gmacs
    dump("fig2_accuracy_cost", runs[["dataset", "variant", "run", "seed", "test_macro_f1"]]
         .assign(gmacs=runs.variant.map(gm)).sort_values(["dataset", "variant", "run", "seed"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True)
    for ax, (d, name) in zip(axes, DS):
        r = runs[runs.dataset == d]
        for v, lab in V:
            y = r[r.variant == v].test_macro_f1
            ax.scatter(np.full(len(y), gm[v]), y, s=10, color=COL[v], alpha=0.35, lw=0, zorder=2)
            ax.errorbar(gm[v], y.mean(), yerr=y.std(ddof=1), fmt="o", ms=5, color=COL[v],
                        mec="white", mew=1, elinewidth=1.2, capsize=0, zorder=3)
            ax.annotate(lab, (gm[v], y.mean()), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.set_xscale("log"); ax.set_xticks([8.36, 15.74, 35.9]); ax.set_xticklabels(["8.4", "15.7", "35.9"])
        ax.minorticks_off(); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlim(6.5, 52)
        lo, hi = r.test_macro_f1.min(), r.test_macro_f1.max()
        ax.set_ylim(lo - 0.02, hi + 0.02)
    axes[0].set_ylabel("Test macro-F1")
    fig.supxlabel("GMAC per image at 384 px (log scale)", fontsize=8, color=INK, y=-0.04)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig2_accuracy_cost")


# ---------------------------------------------------------------- Fig 3: forest plot (RQ3, RQ5, RQ6)
def fig_forest():
    x = pd.read_csv(os.path.join(SUM, "xai_paired_tests.csv"))
    r = pd.read_csv(os.path.join(SUM, "reliance_paired_tests.csv"))
    x = x[(x.subset == "common_correct") & (x.metric == "energy_norm")]
    r = r[(r.subset == "common_correct") & (r.metric == "lri")]
    panels = [("Grad-CAM localisation\n(energy_norm, primary)", x[x.method == "gradcam"]),
              ("IG localisation\n(energy_norm)", x[x.method == "ig"]),
              ("Lesion reliance\n(occlusion LRI, no explainer)", r)]
    contrasts = [("m", "s", "M - S"), ("l", "s", "L - S"), ("l", "m", "L - M")]
    rows = []
    for d, name in DS:
        for a, b, lab in contrasts:
            rows.append((d, a, b, f"{name}  {lab}"))
    cols = ["dataset", "A", "B", "n_images", "mean_A", "mean_B", "delta", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_holm"]
    dump("fig3_forest", pd.concat([df[cols].assign(panel=t.replace("\n", " ")) for t, df in panels]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    ypos = []
    y = 0
    for i, row in enumerate(rows):
        if i and i % 3 == 0:
            y += 0.8
        ypos.append(y); y += 1
    ypos = -np.array(ypos)
    for ax, (title, df) in zip(axes, panels):
        ax.axvspan(-0.05, 0.05, color="#eeede9", zorder=0, lw=0)
        ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
        for yy, (d, a, b, lab) in zip(ypos, rows):
            q = df[(df.dataset == d) & (df.A == a) & (df.B == b)].iloc[0]
            outside = (q.ci90_lo < -0.05) or (q.ci90_hi > 0.05)
            c = INK if outside else MUTED
            ax.plot([q.ci95_lo, q.ci95_hi], [yy, yy], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(q.delta, yy, "o", ms=4.5, color=c, mfc=("white" if not outside else c), mew=1.1, zorder=3)
        ax.set_title(title, fontsize=7.8, color=INK)
        ax.set_xlabel("Paired difference (95% CI)")
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels([lab for *_, lab in rows], fontsize=7.2)
    for ax in axes:
        lo, hi = ax.get_xlim(); m = max(abs(lo), abs(hi), 0.07); ax.set_xlim(-m, m)
    fig.text(0.99, -0.07, "Shaded band: equivalence margin ±0.05. Filled marker: 90% CI extends beyond the margin.",
             ha="right", fontsize=6.8, color=MUTED)
    save(fig, "fig3_forest")


# ---------------------------------------------------------------- Fig 4: sanity cascade
def fig_sanity():
    s = pd.read_csv(os.path.join(SUM, "sanity_cascade.csv"), header=None)
    levels = list(s.iloc[1, 4:10])
    body = s.iloc[3:].copy()
    body.columns = ["tag", "dataset", "variant", "method"] + levels
    body[levels] = body[levels].astype(float)
    dump("fig4_sanity", body[body.method.isin(["gradcam", "ig"])][["dataset", "variant", "method"] + levels])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    mc = {"gradcam": ("Grad-CAM", "#2a78d6", "-"), "ig": ("Integrated Gradients", "#eb6834", "-")}
    xs = np.arange(len(levels))
    for ax, (d, name) in zip(axes, DS):
        for m, (lab, c, ls) in mc.items():
            g = body[(body.dataset == d) & (body.method == m)][levels].values
            ax.fill_between(xs, g.min(0), g.max(0), color=c, alpha=0.15, lw=0)
            ax.plot(xs, g.mean(0), ls, color=c, lw=1.6, marker="o", ms=3.5, mec="white", mew=0.8, label=lab)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(levels, rotation=35, ha="right", fontsize=6.8)
        ax.set_title(name, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Spearman correlation with\ntrained-model map")
    axes[0].set_ylim(-0.4, 1.05)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.16, "Parameters randomised cumulatively from the top (left) to all layers (right); "
             "line = mean over S/M/L, band = range.", ha="center", fontsize=6.8, color=MUTED)
    save(fig, "fig4_sanity")


# ---------------------------------------------------------------- Fig 5: ERF grid
def fig_erf():
    runs = pd.read_csv(os.path.join(SUM, "runs.csv"))
    dump("fig5_erf", runs[["dataset", "variant", "run", "seed", "erf_area90"]].sort_values(["dataset", "variant", "run", "seed"]))
    fig, axes = plt.subplots(3, 3, figsize=(5.6, 5.9))
    for i, (d, name) in enumerate(DS):
        for j, (v, lab) in enumerate(V):
            fs = glob.glob(os.path.join(RES, "results_v3", "main", d, v, "*", "erf.npy"))
            e = np.mean([np.load(f) / np.load(f).sum() for f in fs], axis=0)
            ax = axes[i, j]
            ax.imshow(np.log10(e + 1e-12), cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            a90 = runs[(runs.dataset == d) & (runs.variant == v)].erf_area90.mean()
            ax.set_title(f"{lab}   area90 = {a90:.3f}", fontsize=7.8, color=INK)
            if j == 0:
                ax.set_ylabel(name, fontsize=8.5, color=INK)
    fig.text(0.5, 0.01, "Mean normalised ERF over runs, log scale. area90: fraction of the image holding 90% of the gradient mass.",
             ha="center", fontsize=6.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig5_erf")


# ---------------------------------------------------------------- raw panels for checking (no styling)
RAW = os.path.join(ROOT, "..", "bukti_figure", "figure_mentah")


def raw_panels():
    """Every panel of figures 2-5 as its own file, in plain matplotlib defaults, stamped with its source.
    Lets a reader check each panel against the CSV/npy it was drawn from before any layout or styling."""
    os.makedirs(RAW, exist_ok=True)
    index = []

    def out(fig, fname, figure, panel, source):
        fig.text(0.01, -0.04, f"{figure}, {panel} | source: results/{source}", fontsize=7)
        fig.savefig(os.path.join(RAW, fname + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        index.append(dict(file=fname + ".png", figure=figure, panel=panel, source="results/" + source))

    runs = pd.read_csv(os.path.join(SUM, "runs.csv"))
    gm = pd.read_csv(os.path.join(SUM, "efficiency_benchmark.csv")).query("img_size == 384").set_index("variant").gmacs
    x = pd.read_csv(os.path.join(SUM, "xai_paired_tests.csv"))
    r = pd.read_csv(os.path.join(SUM, "reliance_paired_tests.csv"))
    sc = pd.read_csv(os.path.join(SUM, "sanity_cascade.csv"), header=None)
    levels = list(sc.iloc[1, 4:10]); sc = sc.iloc[3:].copy()
    sc.columns = ["tag", "dataset", "variant", "method"] + levels; sc[levels] = sc[levels].astype(float)
    with plt.style.context("default"):
        for d, name in DS:
            g = runs[runs.dataset == d]
            fig, ax = plt.subplots()
            for v, lab in V:
                y = g[g.variant == v].test_macro_f1
                ax.plot([gm[v]] * len(y), y, "o", label=f"{lab} runs (n={len(y)})")
                ax.plot(gm[v], y.mean(), "k_", ms=20)
            ax.set_xlabel("GMAC (efficiency_benchmark.csv, img_size=384)"); ax.set_ylabel("test_macro_f1 (runs.csv)")
            ax.set_title(f"{name}: each run; black bar = mean"); ax.legend()
            out(fig, f"fig2_{d}_macroF1_vs_GMAC", "Figure 2", name, "summary/runs.csv + summary/efficiency_benchmark.csv")
        panels = [("a_gradcam_energy_norm", x[(x.method == "gradcam") & (x.metric == "energy_norm")], "summary/xai_paired_tests.csv (method=gradcam, metric=energy_norm)"),
                  ("b_ig_energy_norm", x[(x.method == "ig") & (x.metric == "energy_norm")], "summary/xai_paired_tests.csv (method=ig, metric=energy_norm)"),
                  ("c_lesion_reliance_lri", r[r.metric == "lri"], "summary/reliance_paired_tests.csv (metric=lri)")]
        for pid, df, src in panels:
            df = df[df.subset == "common_correct"].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = [f"{a.dataset} {a.A}-{a.B}" for a in df.itertuples()]
            ax.errorbar(df.delta, range(len(df)), xerr=[df.delta - df.ci95_lo, df.ci95_hi - df.delta], fmt="o")
            ax.axvline(0, color="k"); ax.axvline(-0.05, ls="--"); ax.axvline(0.05, ls="--")
            ax.set_yticks(range(len(df))); ax.set_yticklabels(labels); ax.invert_yaxis()
            ax.set_xlabel("delta with ci95_lo..ci95_hi; dashed = margin 0.05"); ax.set_title(pid)
            out(fig, f"fig3_panel_{pid}", "Figure 3", pid, src + ", subset=common_correct")
        for d, name in DS:
            fig, ax = plt.subplots()
            g = sc[sc.dataset == d]
            for _, row in g[g.method.isin(["gradcam", "ig"])].iterrows():
                ax.plot(levels, row[levels].values, "o-", label=f"{row.method} {row.variant.upper()}")
            ax.axhline(0, color="k"); ax.set_ylabel("Spearman correlation"); ax.set_title(f"{name}: every variant, no averaging"); ax.legend(fontsize=7)
            out(fig, f"fig4_{d}_sanity_cascade", "Figure 4", name, "summary/sanity_cascade.csv")
        for d, name in DS:
            for v, lab in V:
                fs = sorted(glob.glob(os.path.join(RES, "results_v3", "main", d, v, "*", "erf.npy")))
                e = np.mean([np.load(f) / np.load(f).sum() for f in fs], axis=0)
                fig, ax = plt.subplots()
                im = ax.imshow(np.log10(e + 1e-12)); fig.colorbar(im, label="log10 normalised gradient")
                ax.set_title(f"{name} {lab}: mean of {len(fs)} erf.npy")
                out(fig, f"fig5_{d}_{v}_erf", "Figure 5", f"{name} {lab}", f"results_v3/main/{d}/{v}/*/erf.npy ({len(fs)} files)")
    pd.DataFrame(index).to_csv(os.path.join(RAW, "INDEX.csv"), index=False)
    print("raw panels:", len(index), "files in", os.path.normpath(RAW))


def check():
    """Fails if the inputs are not the ones the manuscript reports."""
    runs = pd.read_csv(os.path.join(SUM, "runs.csv"))
    assert len(runs) == 45 and runs.ok.all(), "expected the 45 completed runs of the merged session"
    # ERF: recompute area90 from every saved gradient map (same rule as xai_utils.effective_receptive_field)
    for _, r in runs.iterrows():
        f = glob.glob(os.path.join(RES, "results_v3", "main", r.dataset, r.variant, f"run-{r.run}_seed-{r.seed}_*", "erf.npy"))
        assert len(f) == 1, (r.dataset, r.variant, r.run, r.seed, f)
        v = np.sort(np.load(f[0]).ravel().astype(np.float64))[::-1]
        area = (np.searchsorted(np.cumsum(v) / v.sum(), 0.9) + 1) / v.size
        assert abs(area - r.erf_area90) < 1e-3, (f[0], area, r.erf_area90)
    # forest plot: the plotted difference is the difference of the two reported means
    x = pd.read_csv(os.path.join(SUM, "xai_paired_tests.csv"))
    x = x[(x.subset == "common_correct") & (x.metric == "energy_norm") & x.method.isin(["gradcam", "ig"])]
    assert len(x) == 18 and np.allclose(x.delta, x.mean_A - x.mean_B, atol=1e-6)
    print("check passed: 45 runs, 45 ERF maps match runs.csv, paired differences consistent")


if __name__ == "__main__":
    check()
    fig_acc_cost(); fig_forest(); fig_sanity(); fig_erf()
    raw_panels()
    print("figures:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
    print("data:   ", sorted(os.listdir(os.path.join(OUT, "data"))))
