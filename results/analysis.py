"""Pre-specified statistical analysis (methodology v2, section 6).

Reads every finished run under --results and writes CSV tables + analysis_report.txt to --out.

Estimation units
  * performance: test images (paired across variants), metric averaged over the repetitions of a variant
    (5 seeds on the fixed split for HAM10000/SIIM; pooled out-of-fold predictions for BUSI).
  * explanations: one score per image and variant (mean over repetitions), paired across variants.
Inference
  * paired bootstrap over images (B resamples) -> 95% CI, two-sided bootstrap p; 90% CI for TOST
    equivalence against the pre-registered margins; Wilcoxon signed-rank on per-image differences;
    paired dominance d = P(A > B) - P(A < B) as effect size; Holm correction within each family.
  * H4 (lesion size): linear mixed model  score ~ variant * size_tertile + (1 | image),
    likelihood-ratio test of the interaction (ML fit).
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PAIRS = [("m", "s"), ("l", "s"), ("l", "m")]
# pre-registered equivalence margins (change ONLY before the full experiment, and report them)
MARGINS = {"macro_f1": 0.03, "bal_acc": 0.03, "auroc": 0.02,
           "energy_norm": 0.05, "hit": 0.05, "iou_topk": 0.05, "del_gain": 0.05, "ins_gain": 0.05,
           "lri": 0.05, "drop_lesion": 0.05, "rra": 0.05, "dod": 0.05}
PRIMARY_XAI = ("gradcam", "energy_norm")
# RQ6 (method dependence): pre-specified contrast = Grad-CAM vs IG on energy_norm; the other pairs are sensitivity
RQ6_PRIMARY = (("gradcam", "ig"), "energy_norm")
RQ6_PAIRS = [("gradcam", "ig"), ("gradcam", "ig_raw"), ("gradcampp", "ig")]


# ---------------------------------------------------------------- helpers
def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, run = len(idx), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def boot_p(bs):
    bs = bs[np.isfinite(bs)]
    if not len(bs):
        return np.nan
    return float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))


def summarize_boot(delta, bs, margin):
    lo95, hi95 = np.nanpercentile(bs, [2.5, 97.5])
    lo90, hi90 = np.nanpercentile(bs, [5, 95])
    return dict(delta=delta, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90, p_boot=boot_p(bs),
                margin=margin, equivalent=bool(margin is not None and lo90 > -margin and hi90 < margin))


def macro_f1(y, pred, C):
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def bal_acc(y, pred, C):
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return float(rec[cm.sum(1) > 0].mean())


def auroc(y, P, C):
    try:
        return float(roc_auc_score(y, P[:, 1]) if C == 2 else
                     roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(C))))
    except ValueError:
        return np.nan


METRIC_FN = {"macro_f1": lambda y, P, C: macro_f1(y, P.argmax(1), C),
             "bal_acc": lambda y, P, C: bal_acc(y, P.argmax(1), C),
             "auroc": auroc}


# ---------------------------------------------------------------- loading
def load_runs(results):
    rows = []
    for p in glob.glob(f"{results}/**/summary.json", recursive=True):
        d = json.load(open(p))
        d["dir"] = os.path.dirname(p)
        d["archived"] = "__failed-attempt" in d["dir"]   # earlier failed attempt kept by the rescue protocol
        d.setdefault("tag", "main")
        d.setdefault("attempt", 0)
        rows.append(d)
    S = pd.DataFrame(rows)
    if len(S):
        S["rep"] = np.where(S.dataset == "busi", "fold" + S.run.astype(str), "seed" + S.seed.astype(str))
    return S


def load_per_image(S, fname):
    parts = []
    for r in S.itertuples():
        f = os.path.join(r.dir, fname)
        if os.path.exists(f) and os.path.getsize(f) > 1:
            try:
                d = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            for k in ("dataset", "variant", "rep", "tag"):
                d[k] = getattr(r, k)
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------- performance
def perf_tables(S, n_boot, rng):
    desc = S.groupby(["tag", "dataset", "variant"])[
        [c for c in ["test_macro_f1", "test_bal_acc", "test_auroc", "test_acc", "epochs_run"] if c in S]] \
        .agg(["mean", "std", "count"]).round(4)
    comp = []
    for (tag, ds), g in S.groupby(["tag", "dataset"]):
        preds = {}
        for v, gv in g.groupby("variant"):
            reps = []
            for r in gv.itertuples():
                f = os.path.join(r.dir, "test_predictions.csv")
                if os.path.exists(f):
                    reps.append(pd.read_csv(f))
            if not reps:
                continue
            if ds == "busi":   # folds have different test images: pool out-of-fold predictions
                reps = [pd.concat(reps, ignore_index=True)]
            preds[v] = [r.sort_values("uid").reset_index(drop=True) for r in reps]
        for A, B in PAIRS:
            if A not in preds or B not in preds:
                continue
            uids = sorted(set.intersection(*[set(r.uid) for r in preds[A] + preds[B]]))
            if not uids:
                continue
            pa = [r.set_index("uid").loc[uids] for r in preds[A]]
            pb = [r.set_index("uid").loc[uids] for r in preds[B]]
            pcols = [c for c in pa[0].columns if c.startswith("p") and c[1:].isdigit()]
            C = len(pcols)
            y = pa[0].label.values.astype(int)
            PA = [r[pcols].values for r in pa]
            PB = [r[pcols].values for r in pb]
            for met, fn in METRIC_FN.items():
                val = lambda Ps, idx: np.nanmean([fn(y[idx], P[idx], C) for P in Ps])
                full = np.arange(len(y))
                delta = val(PA, full) - val(PB, full)
                bs = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, len(y), len(y))
                    bs[b] = val(PA, idx) - val(PB, idx)
                comp.append(dict(tag=tag, dataset=ds, metric=met, A=A, B=B, n_images=len(y),
                                 reps_A=len(PA), reps_B=len(PB), **summarize_boot(delta, bs, MARGINS.get(met))))
    comp = pd.DataFrame(comp)
    if len(comp):
        comp["family"] = np.where(comp.metric == "macro_f1", "primary_perf", "secondary_perf")
        comp["p_holm"] = np.nan
        for (t, fam), idx in comp.groupby(["tag", "family"]).groups.items():
            comp.loc[idx, "p_holm"] = holm(comp.loc[idx, "p_boot"])
    return desc, comp


# ---------------------------------------------------------------- explanations
def add_derived(X):
    X = X.copy()
    if "deletion_auc" in X:
        X["del_gain"] = X["deletion_auc_random"] - X["deletion_auc"]
        X["ins_gain"] = X["insertion_auc"] - X["insertion_auc_random"]
    # RRA for runs made before it was computed directly (pilot v3.1): with |top-k| = |mask| = k,
    # IoU = I / (2k - I)  =>  RRA = I / k = 2 IoU / (1 + IoU)   (exact unless ties sit at the threshold)
    if "iou_topk" in X:
        derived = 2 * X["iou_topk"] / (1 + X["iou_topk"])
        if "rra" not in X:
            X["rra"] = np.nan
        X["rra_derived"] = X["rra"].isna() & derived.notna()
        X["rra"] = X["rra"].fillna(derived)
    return X


def common_correct(X):
    """Keep (dataset, rep, uid) only if every variant present in that repetition classified it correctly."""
    c = X.drop_duplicates(["tag", "dataset", "rep", "uid", "variant"])[["tag", "dataset", "rep", "uid", "variant", "correct"]]
    nvar = c.groupby(["tag", "dataset"]).variant.nunique().rename("nvar")
    agg = c.groupby(["tag", "dataset", "rep", "uid"]).agg(ok=("correct", "min"), n=("variant", "nunique")).reset_index()
    agg = agg.merge(nvar, on=["tag", "dataset"])
    keep = agg[(agg.ok == 1) & (agg.n == agg.nvar)][["tag", "dataset", "rep", "uid"]]
    return X.merge(keep, on=["tag", "dataset", "rep", "uid"])


def paired_tests(X, metrics, methods, n_boot, rng, subset):
    out = []
    for (tag, ds, mth), g in X[X.method.isin(methods)].groupby(["tag", "dataset", "method"]):
        per = g.groupby(["uid", "variant"])[metrics].mean()
        for met in metrics:
            if met not in per or per[met].isna().all():
                continue
            w = per[met].unstack("variant")
            for A, B in PAIRS:
                if A not in w or B not in w:
                    continue
                d = (w[A] - w[B]).dropna().values
                if len(d) < 5:
                    continue
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
                try:
                    pw = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    pw = np.nan
                out.append(dict(tag=tag, dataset=ds, method=mth, metric=met, subset=subset, A=A, B=B, n_images=len(d),
                                mean_A=float(w[A].mean()), mean_B=float(w[B].mean()),
                                dominance=float(np.mean(np.sign(d))), p_wilcoxon=pw,
                                **summarize_boot(float(d.mean()), bs, MARGINS.get(met))))
    return pd.DataFrame(out)


def _tertile(s):
    if s.nunique() < 3:
        return pd.Series(["medium"] * len(s), index=s.index)
    return pd.qcut(s.rank(method="first"), 3, labels=["small", "medium", "large"]).astype(str)


def add_size_tertiles(X):
    """Lesion-size tertiles per dataset, computed on ALL lesion images (thresholds do not depend on
    which images a variant classifies correctly)."""
    u = X.drop_duplicates(["tag", "dataset", "uid"])[["tag", "dataset", "uid", "mask_area"]].copy()
    u["size_t"] = u.groupby(["tag", "dataset"]).mask_area.transform(_tertile)
    return X.merge(u[["tag", "dataset", "uid", "size_t"]], on=["tag", "dataset", "uid"])


def lesion_size_lmm(X, method="gradcam", metric="energy_norm", min_images=15):
    import statsmodels.formula.api as smf
    from scipy.stats import chi2
    rows = []
    d0 = X[X.method == method]
    for (tag, ds), g in d0.groupby(["tag", "dataset"]):
        g = g.dropna(subset=[metric])
        if g.variant.nunique() < 2 or g.uid.nunique() < min_images:
            continue
        g = g.assign(size_t=g.size_t.astype(str))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = smf.mixedlm(f"{metric} ~ C(variant, Treatment('s')) * C(size_t, Treatment('small'))",
                                   g, groups=g["uid"]).fit(reml=False)
                red = smf.mixedlm(f"{metric} ~ C(variant, Treatment('s')) + C(size_t, Treatment('small'))",
                                  g, groups=g["uid"]).fit(reml=False)
            lr = 2 * (full.llf - red.llf)
            dfd = len(full.fe_params) - len(red.fe_params)
            coefs = pd.DataFrame({"coef": full.fe_params, "se": full.bse_fe,
                                  "p": full.pvalues[full.fe_params.index]})
            rows.append(dict(tag=tag, dataset=ds, method=method, metric=metric, n_obs=len(g),
                             n_images=g.uid.nunique(), converged=bool(full.converged and red.converged),
                             lrt_interaction=lr, df=dfd, p_interaction=float(chi2.sf(lr, dfd)),
                             coefs=coefs.round(4).to_dict("index")))
        except Exception as e:  # singular fits on tiny pilots
            rows.append(dict(tag=tag, dataset=ds, method=method, metric=metric, error=str(e)[:200]))
    return pd.DataFrame(rows)


def baseline_table(X):
    cols = [c for c in ["energy_norm", "hit", "iou_topk", "rra", "del_gain", "ins_gain"] if c in X]
    t = X.groupby(["tag", "dataset", "variant", "method"])[cols].mean().round(4)
    # share of images where the method beats the centre-bias baseline (energy_norm)
    piv = X.pivot_table(index=["tag", "dataset", "variant", "rep", "uid"], columns="method", values="energy_norm")
    if "base_center" in piv:
        beat = {m: (piv[m] > piv["base_center"]).groupby(level=[0, 1, 2]).mean()
                for m in piv.columns if not m.startswith(("base_", "oracle"))}
        beat = pd.DataFrame(beat).stack().rename("frac_beats_center").reset_index() \
            .rename(columns={"level_3": "method"})
        t = t.reset_index().merge(beat, on=["tag", "dataset", "variant", "method"], how="left") \
            .set_index(["tag", "dataset", "variant", "method"])
    return t


# ---------------------------------------------------------------- main
def method_dependence(X, n_boot, rng, subset, metrics=("energy_norm", "rra")):
    """RQ6: does the size effect depend on the explainer?  Per image:
    dod = (A - B) under method m1  -  (A - B) under method m2,  e.g. (L - S)_Grad-CAM - (L - S)_IG.
    dod < 0: the larger model looks worse with m1 than with m2 (explainer-specific penalty)."""
    out = []
    for (tag, ds), g in X.groupby(["tag", "dataset"]):
        for met in metrics:
            if met not in g:
                continue
            w = g.groupby(["uid", "method", "variant"])[met].mean().unstack(["method", "variant"])
            for m1, m2 in RQ6_PAIRS:
                for A, B in PAIRS:
                    cols = [(m1, A), (m1, B), (m2, A), (m2, B)]
                    if not all(c in w for c in cols):
                        continue
                    d1 = w[(m1, A)] - w[(m1, B)]
                    d2 = w[(m2, A)] - w[(m2, B)]
                    d = (d1 - d2).dropna().values
                    if len(d) < 5:
                        continue
                    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
                    try:
                        pw = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    except ValueError:
                        pw = np.nan
                    prim = (m1, m2) == RQ6_PRIMARY[0] and met == RQ6_PRIMARY[1] and subset == "common_correct"
                    out.append(dict(tag=tag, dataset=ds, metric=met, m1=m1, m2=m2, A=A, B=B, subset=subset,
                                    n_images=len(d), diff_m1=float(np.nanmean(d1)), diff_m2=float(np.nanmean(d2)),
                                    dominance=float(np.mean(np.sign(d))), p_wilcoxon=pw,
                                    family="rq6" if prim else "rq6_sensitivity",
                                    **summarize_boot(float(d.mean()), bs, MARGINS["dod"])))
    return pd.DataFrame(out)


def _direction(row):
    """'higher' / 'lower' (95% CI excludes 0 and Holm p < 0.05), 'equivalent' (TOST), else 'inconclusive'."""
    if row is None:
        return "missing"
    p = row.get("p_holm", np.nan)
    sig = np.isfinite(p) and p < 0.05
    if sig and row["ci95_lo"] > 0:
        return "higher"
    if sig and row["ci95_hi"] < 0:
        return "lower"
    if row.get("equivalent", False):
        return "equivalent"
    return "inconclusive"


def rq5_reading(T, RT, D, bt):
    """Applies the pre-specified interpretation table (design doc 2.2) per dataset and variant pair."""
    def pick(df, **kw):
        if df is None or not len(df):
            return None
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).values
        return df[m].iloc[0].to_dict() if m.any() else None

    rows = []
    for (tag, ds), _ in T.groupby(["tag", "dataset"]):
        for A, B in PAIRS:
            base = dict(tag=tag, dataset=ds, subset="common_correct", A=A, B=B)
            loc = _direction(pick(T, method="gradcam", metric="energy_norm", **base))
            dele = _direction(pick(T, method="gradcam", metric="del_gain", **base))
            ig = _direction(pick(T, method="ig", metric="energy_norm", **base))
            lri = _direction(pick(RT, method="occlusion", metric="lri", **base))
            dod = _direction(pick(D, metric="energy_norm", m1="gradcam", m2="ig", **base))
            if loc == "missing":
                continue
            beats = []
            if bt is not None and len(bt):
                for v in (A, B):
                    try:
                        beats.append(bt.loc[(tag, ds, v, "gradcam"), "energy_norm"] >
                                     bt.loc[(tag, ds, v, "base_center"), "energy_norm"])
                    except KeyError:
                        pass
            if beats and not all(beats):
                reading = "Grad-CAM does not beat base_center for at least one variant: localisation not interpretable here (2.2, row 5)"
            elif loc == "lower" and lri == "equivalent" and dod == "lower" and ig in ("higher", "equivalent"):
                reading = "Method-dependent: Grad-CAM penalises the larger model, the model's lesion reliance does not change (2.2, RQ6 row)"
            elif loc == "lower" and lri == "equivalent" and dele == "lower":
                reading = "Explanation degrades, model does not (2.2, row 1)"
            elif loc == "lower" and lri == "lower":
                reading = "Larger model relies less on the lesion; heatmap reflects that (2.2, row 2)"
            elif loc == "equivalent":
                reading = "Larger model is not more explainable (2.2, row 3)"
            elif loc == "higher":
                reading = "Scaling improves localisation under a locked recipe (2.2, row 4)"
            else:
                reading = "Not conclusive under the pre-specified rules"
            rows.append(dict(tag=tag, dataset=ds, contrast=f"{A.upper()}-{B.upper()}", gradcam_localisation=loc,
                             lesion_reliance=lri, gradcam_deletion=dele, ig_localisation=ig,
                             gradcam_vs_ig_dod=dod, reading=reading))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bench", default="")
    ap.add_argument("--exclude_rescued", action="store_true",
                    help="sensitivity analysis: drop runs that only succeeded after a rescue attempt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(2026)
    rep = []
    S = load_runs(a.results)
    if S.empty:
        print("no runs found")
        return
    say = lambda title, df: rep.append(f"\n=== {title} ===\n" + (df.to_string() if len(df) else "(empty)"))
    ok = S.status.astype(str).str.startswith(("completed", "early_stop"))
    S["ok"] = ok
    excluded = S.loc[~ok, [c for c in ["tag", "dataset", "variant", "run", "seed", "lr", "attempt", "amp", "status",
                                       "archived"] if c in S]]
    # training stability per variant is itself a result: report it, do not hide it
    fin = S[~S.archived]
    stab = fin.groupby(["tag", "dataset", "variant"]).agg(
        runs=("ok", "size"), ok=("ok", "sum"),
        ok_first_try=("attempt", lambda x: int(((x == 0) & fin.loc[x.index, "ok"]).sum())),
        rescued=("attempt", lambda x: int(((x > 0) & fin.loc[x.index, "ok"]).sum())),
        failed_final=("ok", lambda x: int((~x).sum())))
    arch = S[S.archived].groupby(["tag", "dataset", "variant"]).size().rename("failed_attempts")
    stab = stab.join(arch).fillna({"failed_attempts": 0}).astype({"failed_attempts": int})
    if "bad_loss_frac_total" in S:
        stab = stab.join(fin[fin.ok].groupby(["tag", "dataset", "variant"]).bad_loss_frac_total.max()
                         .rename("max_skipped_batch_frac"))
    stab.to_csv(f"{a.out}/training_stability.csv")
    say("Training stability per variant (rescue protocol: next-lower LR, then FP32)", stab)
    say("Excluded runs (diverged, collapsed, unfinished, or archived failed attempts)", excluded)
    S = S[ok & ~S.archived]
    if a.exclude_rescued:
        S = S[S.attempt == 0]
        say("SENSITIVITY ANALYSIS", pd.DataFrame({"note": ["runs that needed a rescue attempt are excluded"]}))
    S.to_csv(f"{a.out}/runs.csv", index=False)

    desc, comp = perf_tables(S, a.n_boot, rng)
    desc.to_csv(f"{a.out}/perf_descriptive.csv")
    comp.to_csv(f"{a.out}/perf_comparisons.csv", index=False)
    say("Performance (mean, sd, n over repetitions)", desc)
    say("Performance: paired bootstrap differences A - B (Holm within family; equivalent = 90% CI inside margin)",
        comp.round(4) if len(comp) else comp)

    eff_cols = [c for c in ["params_m", "train_sec", "epochs_run", "train_img_per_sec", "energy_kwh", "erf_area90",
                            "xai_sec", "sanity_sec"] if c in S]
    eff = S.groupby(["tag", "dataset", "variant"])[eff_cols].agg(["mean", "std"]).round(3)
    eff.to_csv(f"{a.out}/efficiency_and_erf_from_runs.csv")
    say("Run-level efficiency (confounded by parallel runs; see bench) and ERF area (descriptive)", eff)
    if a.bench and os.path.exists(a.bench):
        b = pd.DataFrame(json.load(open(a.bench)))
        b.to_csv(f"{a.out}/efficiency_benchmark.csv", index=False)
        say("Controlled efficiency benchmark", b.round(3))

    T = RT = D = bt = None
    X = load_per_image(S, "xai_per_image.csv")
    if len(X):
        X = add_size_tertiles(add_derived(X))
        X.to_csv(f"{a.out}/xai_per_image_all.csv", index=False)
        bt = baseline_table(X)
        bt.to_csv(f"{a.out}/xai_vs_baselines.csv")
        say("Explanation scores vs reference maps (base_center, base_random, oracle_grid)", bt)
        XC = common_correct(X)
        metrics = [m for m in ["energy_norm", "hit", "iou_topk", "rra", "del_gain", "ins_gain"] if m in X]
        methods = [m for m in X.method.unique() if not m.startswith(("base_", "oracle"))]
        T = pd.concat([paired_tests(XC, metrics, methods, a.n_boot, rng, "common_correct"),
                       paired_tests(X, metrics, methods, a.n_boot, rng, "all_lesion_images")], ignore_index=True)
        if len(T):
            prim = (T.method == PRIMARY_XAI[0]) & (T.metric == PRIMARY_XAI[1]) & (T.subset == "common_correct")
            T["family"] = np.where(prim, "primary_xai", np.where(T.subset == "common_correct", "secondary_xai", "sensitivity"))
            T["p_holm"] = np.nan
            for (t, fam), idx in T.groupby(["tag", "family"]).groups.items():
                T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_wilcoxon"])
            T.to_csv(f"{a.out}/xai_paired_tests.csv", index=False)
            say("PRIMARY endpoint: Grad-CAM energy_norm, images correct for all variants",
                T[T.family == "primary_xai"].round(4))
            say("Secondary explanation endpoints", T[T.family == "secondary_xai"].round(4))
        D = pd.concat([method_dependence(XC, a.n_boot, rng, "common_correct"),
                       method_dependence(X, a.n_boot, rng, "all_lesion_images")], ignore_index=True)
        if len(D):
            D["p_holm"] = np.nan
            for (t, fam), idx in D.groupby(["tag", "family"]).groups.items():
                D.loc[idx, "p_holm"] = holm(D.loc[idx, "p_wilcoxon"])
            D.to_csv(f"{a.out}/rq6_method_dependence.csv", index=False)
            say("RQ6: method dependence, dod = (A-B)_m1 - (A-B)_m2 (family rq6 = Grad-CAM vs IG, energy_norm, common-correct)",
                D[D.family == "rq6"].round(4))
        if "rra_derived" in X and X.rra_derived.any():
            say("Note", pd.DataFrame({"note": [f"RRA derived from iou_topk for {int(X.rra_derived.sum())} rows (runs made before RRA was computed directly)"]}))
        lmm = lesion_size_lmm(XC)
        lmm.to_csv(f"{a.out}/h4_lesion_size_lmm.csv", index=False)
        say("H4: variant x lesion-size interaction (LMM, LRT)", lmm.drop(columns=["coefs"], errors="ignore"))

    R = load_per_image(S, "reliance_per_image.csv")
    if len(R):
        R["method"] = "occlusion"
        R.to_csv(f"{a.out}/reliance_per_image_all.csv", index=False)
        rt = R.groupby(["tag", "dataset", "variant"])[["drop_lesion", "drop_ctrl", "lri", "p_lesion_only", "ctrl_overlap"]] \
            .mean().round(4)
        rt.to_csv(f"{a.out}/reliance_summary.csv")
        say("Lesion reliance (model property, no XAI)", rt)
        RT = pd.concat([paired_tests(common_correct(R), ["lri", "drop_lesion"], ["occlusion"], a.n_boot, rng, "common_correct"),
                        paired_tests(R, ["lri", "drop_lesion"], ["occlusion"], a.n_boot, rng, "all_lesion_images")])
        if len(RT):
            RT["p_holm"] = holm(RT.p_wilcoxon)
            RT.to_csv(f"{a.out}/reliance_paired_tests.csv", index=False)
            say("Lesion reliance: paired differences", RT.round(4))

    if T is not None and len(T):
        RD = rq5_reading(T, RT, D, bt)
        RD.to_csv(f"{a.out}/rq5_rq6_reading.csv", index=False)
        say("RQ5/RQ6 reading by the pre-specified interpretation table (design doc 2.2)", RD)

    SN = load_per_image(S, "sanity_per_image.csv")
    if len(SN):
        st = SN.groupby(["tag", "dataset", "variant", "method", "level", "level_name"]).rank_corr.mean() \
            .unstack(["level", "level_name"]).round(3)
        st.to_csv(f"{a.out}/sanity_cascade.csv")
        say("Sanity check: mean Spearman(trained map, randomized map) by cascade level (lower = passes)", st)

    open(f"{a.out}/analysis_report.txt", "w").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
