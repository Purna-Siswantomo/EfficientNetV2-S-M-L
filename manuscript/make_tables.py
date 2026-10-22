"""Writes the LaTeX tables of the manuscript directly from results/summary, so every number traces to a CSV."""
import ast, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SUM = os.path.join(ROOT, "..", "results", "summary")
OUT = os.path.join(ROOT, "tables")
DS = [("busi", "BUSI"), ("ham10000", "HAM10000"), ("siim", "SIIM-ACR")]
V = [("s", "S"), ("m", "M"), ("l", "L")]
CON = [("m", "s", "M$-$S"), ("l", "s", "L$-$S"), ("l", "m", "L$-$M")]
rd = lambda f: pd.read_csv(os.path.join(SUM, f))


def n(x, d=3, sign=False):
    if pd.isna(x):
        return "--"
    if abs(x) < 0.5 * 10 ** (-d):
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci(q, d=3, eq=True):
    s = f"{n(q.delta, d, True)} [{n(q.ci95_lo, d)}, {n(q.ci95_hi, d)}]"
    if eq and bool(q.equivalent):
        s += "$^{\\mathrm{e}}$"
    return s


def write(name, body):
    if name != "tab_cost":  # wide tables are scaled to the text width (\fitwidth is defined in main.tex)
        body = body.replace("\\begin{tabular}", "\\fitwidth{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    open(os.path.join(OUT, name + ".tex"), "w").write(body)


# ---------------------------------------------------------------- performance
def t_perf():
    r = rd("runs.csv"); p = rd("perf_comparisons.csv")
    rows = []
    for d, dn in DS:
        for i, (v, vn) in enumerate(V):
            g = r[(r.dataset == d) & (r.variant == v)]
            cells = [f"{g[c].mean():.3f} ({g[c].std(ddof=1):.3f})" for c in ("test_macro_f1", "test_bal_acc", "test_auroc")]
            if v == "s":
                diff = "--"
            else:
                q = p[(p.dataset == d) & (p.metric == "macro_f1") & (p.A == v) & (p.B == "s")].iloc[0]
                diff = ci(q) + f" & {q.p_holm:.2f}"
            if v == "s":
                diff += " & --"
            rows.append(f"{dn if i == 0 else ''} & {vn} & " + " & ".join(cells) + f" & {diff} \\\\")
        rows.append("\\midrule" if d != "siim" else "")
    body = r"""\begin{table}[!htbp]
\centering
\caption{Classification performance on the test data (mean and standard deviation over five folds for BUSI and five seeds for HAM10000 and SIIM-ACR) and the paired-bootstrap difference in macro-F1 against S. $^{\mathrm{e}}$: equivalent within $\pm0.03$ (90\% CI inside the margin). $p_\mathrm{Holm}$: Holm-adjusted $p$ within the primary performance family.}
\label{tab:perf}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccll}
\toprule
Dataset & Model & Macro-F1 & Balanced acc. & AUROC & $\Delta$ macro-F1 vs S [95\% CI] & $p_\mathrm{Holm}$ \\
\midrule
""" + "\n".join(x for x in rows if x) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_perf", body)


# ---------------------------------------------------------------- cost
def t_cost():
    b = rd("efficiency_benchmark.csv"); b = b[b.img_size == 384].set_index("variant")
    r = rd("runs.csv")
    e = r.groupby(["dataset", "variant"]).energy_kwh.mean()
    lines = []
    spec = [("Parameters (M)", lambda v: f"{b.loc[v, 'params_m']:.1f}"),
            ("GMAC per image", lambda v: f"{b.loc[v, 'gmacs']:.2f}"),
            ("GPU latency, batch 1 (ms)", lambda v: f"{b.loc[v, 'gpu_latency_ms']:.1f}"),
            ("GPU throughput, batch 32 (img/s)", lambda v: f"{b.loc[v, 'gpu_throughput_img_s']:.0f}"),
            ("CPU latency, batch 1 (ms)", lambda v: f"{b.loc[v, 'cpu_latency_ms']:.0f}"),
            ("Training speed, batch 8 (img/s)", lambda v: f"{b.loc[v, 'train_img_s_b8']:.1f}"),
            ("Training memory, batch 8 (GB)", lambda v: f"{b.loc[v, 'train_mem_gb_b8']:.2f}")]
    cols = ["params_m", "gmacs", "gpu_latency_ms", "gpu_throughput_img_s", "cpu_latency_ms", "train_img_s_b8", "train_mem_gb_b8"]
    raw = {lab: (lambda c: lambda v: float(b.loc[v, c]))(c) for (lab, _), c in zip(spec, cols)}
    for lab, f in spec:
        vals = [f(v) for v, _ in V]
        ratio = raw[lab]("l") / raw[lab]("s")  # from unrounded values
        rs = f"{ratio:.2f}" if ratio < 1 else f"{ratio:.1f}"
        lines.append(f"{lab} & " + " & ".join(vals) + f" & {rs}$\\times$ \\\\")
    lines.append("\\midrule")
    for d, dn in DS:
        vals = [e[(d, v)] for v, _ in V]
        lines.append(f"Training energy per run, {dn} (kWh) & " + " & ".join(f"{x:.3f}" for x in vals) + f" & {vals[2] / vals[0]:.1f}$\\times$ \\\\")
    body = r"""\begin{table}[!htbp]
\centering
\caption{Computational cost at 384\,px. The upper block is a controlled benchmark run alone on one NVIDIA T4 GPU (GPU inference in FP16 and CPU inference in FP32; one training step in FP16 with the same batch of 8 for every variant). Training energy is the CodeCarbon estimate averaged over the five runs of each dataset; these runs shared the machine with a second run, so energy is reported as a secondary, approximate measure.}
\label{tab:cost}
\small
\begin{tabular}{lcccc}
\toprule
Measure & S & M & L & L / S \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_cost", body)


# ---------------------------------------------------------------- baselines (all evaluated lesion images)
def t_baseline():
    x = rd("xai_vs_baselines.csv")
    rows = []
    for d, dn in DS:
        g = x[x.dataset == d]
        val = lambda m, v="s", c="energy_norm": g[(g.method == m) & (g.variant == v)][c].iloc[0]
        cam = " & ".join(f"{val('gradcam', v):.3f}" for v, _ in V)
        ig = " & ".join(f"{val('ig', v):.3f}" for v, _ in V)
        frac = " / ".join(f"{val('gradcam', v, 'frac_beats_center'):.2f}" for v, _ in V)
        rows.append(f"{dn} & {n(val('base_random'))} & {val('base_center'):.3f} & {val('oracle_grid'):.3f} & {cam} & {ig} & {frac} \\\\")
    body = r"""\begin{table}[!htbp]
\centering
\caption{Chance-normalised lesion energy (\texttt{energy\_norm}) of the explanation maps and of the reference maps, averaged over all evaluated lesion images and runs. The last column gives the fraction of images on which Grad-CAM exceeds the centre-bias map (S / M / L). \texttt{oracle\_grid} is the mask itself rendered at the $12\times12$ CAM resolution; it is a resolution reference, not an upper bound for energy.}
\label{tab:baselines}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lccccccccccc}
\toprule
 & \multicolumn{3}{c}{Reference maps} & \multicolumn{3}{c}{Grad-CAM} & \multicolumn{3}{c}{IG} & Grad-CAM $>$ centre \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Dataset & Random & Centre & Oracle & S & M & L & S & M & L & S / M / L \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_baselines", body)


# ---------------------------------------------------------------- primary + RQ5 + RQ6
def t_primary():
    x = rd("xai_paired_tests.csv"); x = x[(x.subset == "common_correct") & (x.metric == "energy_norm")]
    r = rd("reliance_paired_tests.csv"); r = r[(r.subset == "common_correct") & (r.metric == "lri")]
    dd = rd("rq6_method_dependence.csv")
    dd = dd[(dd.subset == "common_correct") & (dd.metric == "energy_norm") & (dd.m1 == "gradcam") & (dd.m2 == "ig")]
    rows = []
    for d, dn in DS:
        for i, (a, b, lab) in enumerate(CON):
            f = lambda df: df[(df.dataset == d) & (df.A == a) & (df.B == b)].iloc[0]
            g, ig, lri, dod = f(x[x.method == "gradcam"]), f(x[x.method == "ig"]), f(r), f(dd)
            nn = f"{int(g.n_images)}" if i == 0 else ""
            rows.append(f"{dn if i == 0 else ''} & {nn} & {lab} & {ci(g)} & {ci(ig)} & {ci(dod)} & {ci(lri)} \\\\")
        if d != "siim":
            rows.append("\\midrule")
    body = r"""\begin{table*}[!htbp]
\centering
\caption{Paired differences on lesion images classified correctly by all three variants ($n$ images). Grad-CAM \texttt{energy\_norm} is the primary endpoint (RQ3); the difference-in-differences (DoD) is the Grad-CAM difference minus the IG difference for the same image (RQ6); LRI is the explainer-free lesion reliance index (RQ5). Values are the mean paired difference with its bootstrap 95\% CI. $^{\mathrm{e}}$: equivalent within $\pm0.05$. All Grad-CAM differences except BUSI M$-$S have Holm-adjusted $p<0.001$.}
\label{tab:primary}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrlllll}
\toprule
Dataset & $n$ & Contrast & Grad-CAM (primary) & IG & DoD (Grad-CAM $-$ IG) & LRI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_primary", body)


# ---------------------------------------------------------------- H4
def t_h4():
    h = rd("h4_lesion_size_lmm.csv")
    rows = []
    for d, dn in DS:
        q = h[h.dataset == d].iloc[0]
        c = {k: v["coef"] for k, v in ast.literal_eval(q.coefs).items()}
        main = lambda v: c.get(f"C(variant, Treatment('s'))[T.{v}]", 0.0)
        inter = lambda v, t: c.get(f"C(variant, Treatment('s'))[T.{v}]:C(size_t, Treatment('small'))[T.{t}]", 0.0)
        eff = lambda v, t: main(v) + (0 if t == "small" else inter(v, t))
        cells = [n(eff(v, t), 3, True) for v in ("l", "m") for t in ("small", "medium", "large")]
        p = f"{q.p_interaction:.1e}".replace("e-0", "e-").replace("e-", "\\times10^{-") + "}"
        rows.append(f"{dn} & {int(q.n_images)} & " + " & ".join(cells) + f" & {q.lrt_interaction:.1f} & ${p}$ \\\\")
    body = r"""\begin{table}[!htbp]
\centering
\caption{Lesion-size analysis (RQ4). Linear mixed model \texttt{energy\_norm} $\sim$ variant $\times$ lesion-size tertile $+\,(1\,|\,\text{image})$ for Grad-CAM, fitted by maximum likelihood. Cells give the model-implied difference to S within each tertile (small / medium / large lesions). The interaction is tested with a likelihood-ratio test (4 df).}
\label{tab:h4}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrccccccrl}
\toprule
 & & \multicolumn{3}{c}{L $-$ S} & \multicolumn{3}{c}{M $-$ S} & & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & Images & Small & Medium & Large & Small & Medium & Large & LR $\chi^2$ & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_h4", body)


# ---------------------------------------------------------------- secondary metrics, L - S
def t_secondary():
    x = rd("xai_paired_tests.csv"); x = x[(x.subset == "common_correct") & (x.A == "l") & (x.B == "s")]
    mets = [("rra", "RRA"), ("hit", "Pointing hit"), ("del_gain", "Deletion gain"), ("ins_gain", "Insertion gain")]
    nn = lambda k: ", ".join(str(int(x[(x.dataset == d) & (x.method == "gradcam") & (x.metric == k)].n_images.iloc[0])) for d, _ in DS)
    nid, nloc = nn("del_gain"), nn("rra")
    rows = []
    for d, dn in DS:
        for j, (m, mn) in enumerate([("gradcam", "Grad-CAM"), ("ig", "IG")]):
            cells = [ci(x[(x.dataset == d) & (x.method == m) & (x.metric == k)].iloc[0], 3) for k, _ in mets]
            rows.append(f"{dn if j == 0 else ''} & {mn} & " + " & ".join(cells) + " \\\\")
        if d != "siim":
            rows.append("\\midrule")
    body = r"""\begin{table*}[!htbp]
\centering
\caption{Secondary localisation and faithfulness metrics, L $-$ S on the common-correct lesion images (mean paired difference [95\% CI]). RRA: relevance rank accuracy \citep{arras2022clevr}. Deletion and insertion gains are measured against a random pixel order on the same image and model, so positive values mean the map is more faithful than chance. Deletion and insertion were computed on the first 100 explained images of each run, so these columns use """ + nid + r""" images and the other columns use """ + nloc + r""" (BUSI, HAM10000, SIIM-ACR). $^{\mathrm{e}}$: equivalent within $\pm0.05$.}
\label{tab:secondary}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllll l}
\toprule
Dataset & Method & """ + " & ".join(mn for _, mn in mets) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_secondary", body)


# ---------------------------------------------------------------- pre-specified reading
def t_reading():
    q = rd("rq5_rq6_reading.csv")
    short = {
        "Larger model is not more explainable (2.2, row 3)": "Equivalent: not more explainable, higher cost (R3)",
        "Not conclusive under the pre-specified rules": "Not conclusive",
        "Scaling improves localisation under a locked recipe (2.2, row 4)": "Scaling improves Grad-CAM localisation (R4)",
        "Method-dependent: Grad-CAM penalises the larger model, the model's lesion reliance does not change (2.2, RQ6 row)": "Method-dependent (R6)",
    }
    lab = {"lower": "lower", "higher": "higher", "equivalent": "equiv.", "inconclusive": "inconcl."}
    rows = []
    for d, dn in DS:
        g = q[q.dataset == d]
        for i, c in enumerate(["M-S", "L-S", "L-M"]):
            r = g[g.contrast == c].iloc[0]
            cells = [lab.get(r[k], r[k]) for k in ("gradcam_localisation", "lesion_reliance", "gradcam_deletion", "ig_localisation", "gradcam_vs_ig_dod")]
            rows.append(f"{dn if i == 0 else ''} & {c.replace('-', '$-$')} & " + " & ".join(cells) + f" & {short.get(r.reading, r.reading)} \\\\")
        if d != "siim":
            rows.append("\\midrule")
    body = r"""\begin{table*}[!htbp]
\centering
\caption{Automatic application of the pre-specified interpretation rules (Table~\ref{tab:rules}). ``Lower''/``higher'': 95\% CI excludes zero and Holm $p<0.05$; ``equiv.'': TOST within $\pm0.05$; otherwise ``inconcl.''. When a difference is both significant and inside the margin, the rule reports ``lower''/``higher''; Table~\ref{tab:primary} gives both properties.}
\label{tab:reading}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllllll}
\toprule
Dataset & Contrast & Grad-CAM loc. & LRI & Grad-CAM deletion & IG loc. & DoD & Reading \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_reading", body)


if __name__ == "__main__":
    for f in (t_perf, t_cost, t_baseline, t_primary, t_h4, t_secondary, t_reading):
        f()
    print(sorted(os.listdir(OUT)))
