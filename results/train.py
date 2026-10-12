"""One training run: dataset x variant x split x seed x image size -> metrics, XAI scores, ERF, efficiency.

Usage (one process per GPU; the notebook launches these):
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset ham10000 --variant s --run fixed --seed 1 ...

Controls that keep S, M and L comparable (see methodology v2, section 4):
  * weights: timm tf_efficientnetv2_{s,m,l}.in1k, all ported from the same official Google checkpoints
    (same ImageNet-1k recipe) and normalized with each model's own pretrained_cfg mean/std;
  * BatchNorm running statistics frozen (bn_mode=frozen), so the per-step micro-batch (32/16/8) does not
    change the maths: with gradient accumulation to an effective batch of 32, every variant gets the
    same update rule;
  * same optimizer, schedule, augmentation, loss, early stopping and learning-rate grid.

Stability (v3, after the pilot where L diverged at lr 3e-4: NaN loss on HAM10000, one-class collapse on BUSI):
  * classifier initialised to zero, so the first steps train the head before large gradients reach the backbone;
  * warm-up of max(1 epoch, --warmup_steps) optimizer steps (capped at 1/4 of training) instead of 1 epoch;
  * non-finite losses are skipped; a run with too many is stopped as "diverged@<epoch>";
  * a run whose best checkpoint predicts a single class on validation is marked "collapsed";
    diverged/collapsed runs skip XAI and are excluded by analysis.py (status filter).
  The same rules apply to every variant.

v3.1 (review of v3):
  * --max_bad_loss lowered from 0.2 to 0.02: skipping many non-finite batches would silently train one variant
    on less data than the others; the run-level totals (bad_loss_frac_total / _max_epoch) go to summary.json and
    are reported per variant by analysis.py;
  * --attempt / amp recorded, so runs re-done under the pre-specified rescue protocol (notebook section 5) can be
    identified and reported (and excluded in a sensitivity analysis).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

import xai_utils as X

WEIGHTS = {"s": "tf_efficientnetv2_s.in1k", "m": "tf_efficientnetv2_m.in1k", "l": "tf_efficientnetv2_l.in1k"}
# (batch per step, accumulation steps) -> effective batch 32 on a 16 GB T4 (FP16)
BATCH = {384: {"s": (32, 1), "m": (16, 2), "l": (8, 4)},
         480: {"s": (16, 2), "m": (8, 4), "l": (4, 8)}}
NO_FLIP = {"siim"}  # chest X-ray: horizontal flip moves the heart to the wrong side


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", required=True)
    a.add_argument("--variant", required=True, choices=["s", "m", "l"])
    a.add_argument("--run", required=True, help="split id: fold number (busi) or 'fixed'")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--cache", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--img_size", type=int, default=384)
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--patience", type=int, default=7)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--weight_decay", type=float, default=0.05)
    a.add_argument("--drop_rate", type=float, default=0.2)
    a.add_argument("--warmup_steps", type=int, default=100, help="minimum warm-up optimizer steps")
    a.add_argument("--max_bad_loss", type=float, default=0.02,
                   help="stop as diverged if this fraction of an epoch's batches has a non-finite loss")
    a.add_argument("--no_amp", action="store_true", help="train in FP32 (slower; fallback if FP16 overflows)")
    a.add_argument("--attempt", type=int, default=0, help="0 = planned run; 1, 2 = rescue attempts (see notebook)")
    a.add_argument("--bn_mode", default="frozen", choices=["frozen", "train"])
    a.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (if L runs out of memory)")
    a.add_argument("--train_frac", type=float, default=1.0)
    a.add_argument("--tag", default="main", help="analysis group: main | native | datafrac")
    a.add_argument("--shard", default="", help="which collaborator account ran this job (traceability only)")
    a.add_argument("--xai_n", type=int, default=300, help="lesion images scored with CAM/IG")
    a.add_argument("--scorecam_n", type=int, default=0)
    a.add_argument("--insdel_n", type=int, default=100)
    a.add_argument("--reliance_n", type=int, default=300)
    a.add_argument("--sanity_n", type=int, default=30)
    a.add_argument("--sanity_methods", default="gradcam,ig")
    a.add_argument("--erf_n", type=int, default=32)
    a.add_argument("--methods", default="gradcam,gradcampp,ig")
    a.add_argument("--no_xai", action="store_true", help="used by the learning-rate search")
    a.add_argument("--deadline", type=float, default=0, help="unix time; stop training if exceeded")
    a.add_argument("--keep_ckpt", action="store_true")
    a.add_argument("--no_pretrained", action="store_true", help="offline smoke test only")
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--max_steps", type=int, default=0, help="smoke test: cap optimizer steps per epoch")
    a.add_argument("--ig_steps", type=int, default=32)
    a.add_argument("--insdel_steps", type=int, default=25)
    return a.parse_args()


class ImgDS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(T.functional.to_image(Image.open(r.image).convert("RGB")))
        return x, int(r.label), i


def transforms(train, size, mean, std, flip=True):
    base = [T.Resize((size, size), antialias=True)]
    aug = ([T.RandomHorizontalFlip()] if flip else []) + \
          [T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), T.ColorJitter(0.1, 0.1)]
    return T.Compose(base + (aug if train else []) + [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])


def build_model(variant, n_classes, pretrained, drop_rate=0.2):
    m = timm.create_model(WEIGHTS[variant], pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
    head = m.get_classifier()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    cfg = m.pretrained_cfg
    return m, list(cfg["mean"]), list(cfg["std"])


def freeze_bn_stats(model):
    """Keep ImageNet running mean/var (affine gamma/beta stay trainable)."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    P, Y = [], []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            P.append(model(x.to(device, non_blocking=True).to(memory_format=torch.channels_last))
                     .float().softmax(1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def cls_metrics(p, y, n_classes):
    pred = p.argmax(1)
    out = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               macro_f1=f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0))
    try:
        out["auroc"] = roc_auc_score(y, p[:, 1]) if n_classes == 2 else \
            roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    except ValueError:
        out["auroc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def main():
    a = parse()
    t_start = time.time()
    seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = True

    ds_dir = os.path.join(a.cache, a.dataset)
    size_dir = os.path.join(ds_dir, str(a.img_size))
    df = pd.read_csv(os.path.join(ds_dir, "index.csv"), keep_default_na=False)
    ms = pd.read_csv(os.path.join(size_dir, "masks.csv"))
    df = df.merge(ms, on="uid", how="left")
    df["image"] = [os.path.join(size_dir, "img", f"{u}.jpg") for u in df.uid]
    df["mask"] = [os.path.join(size_dir, "mask", f"{u}.png") if (ar == ar and ar > 0) else ""
                  for u, ar in zip(df.uid, df.mask_area)]
    split = json.load(open(os.path.join(ds_dir, "splits.json")))[a.run]
    n_classes = int(df.label.max()) + 1
    by = df.set_index("uid")
    tr, va, te = (by.loc[split[k]].reset_index() for k in ("train", "val", "test"))
    if a.train_frac < 1.0:
        tr = tr.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=a.train_frac, random_state=a.seed))

    model, mean, std = build_model(a.variant, n_classes, not a.no_pretrained, a.drop_rate)
    if a.grad_ckpt:
        model.set_grad_checkpointing(True)
    model = model.to(device).to(memory_format=torch.channels_last)
    flip = a.dataset not in NO_FLIP

    bs, accum = BATCH.get(a.img_size, BATCH[384])[a.variant]
    gen = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ImgDS(tr, transforms(True, a.img_size, mean, std, flip)), bs, shuffle=True,
                       drop_last=len(tr) > bs * accum, num_workers=a.workers, pin_memory=True, generator=gen,
                       persistent_workers=a.workers > 0)
    mk = lambda d, b=32: DataLoader(ImgDS(d, transforms(False, a.img_size, mean, std)), b,
                                    num_workers=a.workers, pin_memory=True)
    dl_va, dl_te = mk(va), mk(te)

    counts = np.bincount(tr.label, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(dl_tr) / accum) if not a.max_steps else a.max_steps
    total = a.epochs * steps_per_epoch
    warm = max(1, min(max(steps_per_epoch, a.warmup_steps), total // 4))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    tracker = None
    if device == "cuda":
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(output_dir=a.out, log_level="error", save_to_file=True,
                                       tracking_mode="process",
                                       gpu_ids=[int(os.environ.get("PHYSICAL_GPU", 0))])
            tracker.start()
        except Exception as e:  # energy logging is optional
            print("codecarbon disabled:", e)

    ckpt = os.path.join(a.out, "best.pt")
    best, bad, hist, n_seen, t_train = -1.0, 0, [], 0, 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stopped, best_n_pred = "completed", 0
    tot_bad, tot_batches, max_bad_ep = 0, 0, 0.0
    for ep in range(a.epochs):
        model.train()
        if a.bn_mode == "frozen":
            freeze_bn_stats(model)
        t0, loss_sum, nb, n_bad, has_grad = time.time(), 0.0, 0, 0, False
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(dl_tr):
            if a.max_steps and i >= a.max_steps * accum:
                break
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x).float(), y) / accum
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                has_grad = True
                loss_sum += loss.item() * accum
                nb += 1
            else:
                n_bad += 1
            if (i + 1) % accum == 0 or i + 1 == len(dl_tr):
                if has_grad:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                has_grad = False
            n_seen += x.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        p, y = predict(model, dl_va, device, amp)
        m = cls_metrics(p, y, n_classes)
        n_pred = len(np.unique(p.argmax(1))) if np.isfinite(p).all() else 0
        bad_frac = n_bad / max(n_bad + nb, 1)
        tot_bad, tot_batches, max_bad_ep = tot_bad + n_bad, tot_batches + n_bad + nb, max(max_bad_ep, bad_frac)
        hist.append(dict(epoch=ep, loss=loss_sum / max(nb, 1), epoch_sec=dt, lr=opt.param_groups[0]["lr"],
                         bad_loss_frac=bad_frac, val_n_pred_classes=n_pred,
                         **{f"val_{k}": v for k, v in m.items()}))
        pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
        print(f"[{a.dataset}/{a.variant}/{a.run}/s{a.seed}/{a.img_size}] ep {ep} loss {hist[-1]['loss']:.4f} "
              f"val_f1 {m['macro_f1']:.4f} pred_classes {n_pred}/{n_classes} bad_loss {bad_frac:.2f} ({dt:.0f}s)",
              flush=True)
        weights_ok = all(torch.isfinite(q).all() for q in model.parameters())
        if bad_frac > a.max_bad_loss or n_pred == 0 or not weights_ok:
            stopped = f"diverged@{ep}"
            print(f"DIVERGED at epoch {ep} (non-finite loss fraction {bad_frac:.2f}, finite weights {weights_ok}); "
                  "try a lower --lr or --no_amp", flush=True)
            if best < 0:   # no usable checkpoint yet: keep the current weights so the test/XAI code still runs
                torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, ckpt)
            break
        if m["macro_f1"] > best:
            best, bad, best_n_pred = m["macro_f1"], 0, n_pred
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                stopped = f"early_stop@{ep}"
                break
        if a.deadline and time.time() > a.deadline:
            stopped = f"deadline@{ep}"
            break

    energy = None
    if tracker is not None:
        try:
            tracker.stop()
            energy = float(tracker.final_emissions_data.energy_consumed)  # kWh
        except Exception:
            pass
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else float("nan")

    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    model.eval()
    p, y = predict(model, dl_te, device, amp)
    test = cls_metrics(p, y, n_classes)
    pd.DataFrame({"uid": te.uid, "label": y, "pred": p.argmax(1), **{f"p{c}": p[:, c] for c in range(n_classes)}}) \
        .to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)

    summary = dict(tag=a.tag, shard=a.shard, dataset=a.dataset, variant=a.variant, weights=WEIGHTS[a.variant], run=a.run, seed=a.seed,
                   lr=a.lr, img_size=a.img_size, bn_mode=a.bn_mode, train_frac=a.train_frac, n_train=len(tr),
                   n_val=len(va), n_test=len(te), status=stopped, epochs_run=len(hist), best_val_macro_f1=best,
                   train_sec=t_train, train_img_per_sec=n_seen / max(t_train, 1e-6), peak_train_mem_gb=peak_mem,
                   energy_kwh=energy, params_m=sum(q.numel() for q in model.parameters()) / 1e6,
                   batch=bs, accum=accum, attempt=a.attempt, amp=bool(amp),
                   bad_loss_frac_total=tot_bad / max(tot_batches, 1), bad_loss_frac_max_epoch=max_bad_ep,
                   **{f"test_{k}": v for k, v in test.items()})

    if not stopped.startswith(("diverged", "deadline")) and best_n_pred <= 1:
        stopped = f"collapsed({stopped})"
        print("COLLAPSED: best checkpoint predicts a single class on validation; excluded from analysis", flush=True)
    summary["status"] = stopped
    failed = stopped.startswith(("diverged", "collapsed"))
    if not a.no_xai and not failed:
        summary.update(run_xai(a, model, te, device, amp, mean, std))
    summary["total_sec"] = time.time() - t_start
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    if not a.keep_ckpt:
        os.remove(ckpt)
    if stopped.startswith("deadline"):
        print("Stopped by deadline; run will be repeated next session.")
    else:
        open(os.path.join(a.out, "DONE"), "w").write(stopped)
    print(json.dumps({k: summary[k] for k in ("test_macro_f1", "test_auroc", "train_img_per_sec")}), flush=True)


def run_xai(a, model, te, device, amp, mean, std):
    """Explanation quality on test images that contain a lesion, target = true class.
    The same images (fixed shuffle) are used for S, M and L, so every score is paired by image.
    Correctness is stored so the analysis can restrict to images all variants classify correctly."""
    methods = [m for m in a.methods.split(",") if m]
    if a.scorecam_n > 0 and "scorecam" not in methods:
        methods.append("scorecam")
    les = te[te["mask"] != ""].copy()
    les = les.sample(frac=1.0, random_state=0).head(a.xai_n).reset_index(drop=True)
    tf = transforms(False, a.img_size, mean, std)
    for q in model.parameters():
        q.requires_grad_(True)
    ex = X.make_explainers(model, methods)
    grid = a.img_size // 32
    rows, rel_rows, san_rows = [], [], []
    t0 = time.time()
    xs, masks = [], []
    for i, r in les.iterrows():
        x = tf(T.functional.to_image(Image.open(r.image).convert("RGB")))[None].to(device)
        mask = np.asarray(Image.open(r["mask"]).convert("L").resize((a.img_size, a.img_size), Image.NEAREST)) > 127
        with torch.no_grad():
            prob = model(x).float().softmax(1)[0]
        base = dict(uid=r.uid, label=int(r.label), pred=int(prob.argmax()), p_true=float(prob[int(r.label)]))
        base["correct"] = int(base["pred"] == base["label"])
        maps = {}
        for name, e in ex.items():
            if name == "scorecam" and i >= a.scorecam_n:
                continue
            maps.update(X.saliency(name, e, x, r.label, a.ig_steps))
        refs = X.reference_maps(mask, grid, seed=i)
        rnd_order = None
        for name, s in list(maps.items()) + list(refs.items()):
            row = dict(base, method=name, **X.localization_metrics(s, mask))
            if i < a.insdel_n and name in maps:
                row.update(X.insertion_deletion(model, x, r.label, X.saliency_order(s, seed=i), a.insdel_steps, amp))
                if rnd_order is None:  # random pixel order: per-image reference for faithfulness
                    rnd_order = np.random.default_rng(10_000 + i).permutation(s.size)
                    rnd = X.insertion_deletion(model, x, r.label, rnd_order, a.insdel_steps, amp)
                row.update(deletion_auc_random=rnd["deletion_auc"], insertion_auc_random=rnd["insertion_auc"])
            rows.append(row)
        if i < a.reliance_n:
            rel_rows.append(dict(base, mask_area=float(mask.mean()),
                                 **X.lesion_reliance(model, x, r.label, mask, amp=amp)))
        if i < a.sanity_n:
            xs.append((x, r, {k: maps[k] for k in maps}))
    X.release(ex)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "xai_per_image.csv"), index=False)
    pd.DataFrame(rel_rows).to_csv(os.path.join(a.out, "reliance_per_image.csv"), index=False)
    t_xai = time.time() - t0

    # cascading model-parameter randomization (Adebayo et al., 2018)
    san_methods = [m for m in a.sanity_methods.split(",") if m in methods]
    if xs and san_methods:
        for lvl, lname, rm in X.cascade_models(model, seed=a.seed):
            exr = X.make_explainers(rm, san_methods)
            for x, r, maps in xs:
                for name, e in exr.items():
                    for k, s_r in X.saliency(name, e, x, r.label, a.ig_steps).items():
                        san_rows.append(dict(uid=r.uid, method=k, level=lvl, level_name=lname,
                                             rank_corr=X.map_rank_corr(maps[k], s_r)))
            X.release(exr)
            del exr
    pd.DataFrame(san_rows).to_csv(os.path.join(a.out, "sanity_per_image.csv"), index=False)

    erf_loader = DataLoader(ImgDS(les.head(a.erf_n) if len(les) else te.head(a.erf_n), tf), 8)
    erf, erf_area = X.effective_receptive_field(model, erf_loader, device, a.erf_n)
    np.save(os.path.join(a.out, "erf.npy"), erf)

    d = pd.DataFrame(rows)
    out = dict(xai_n_images=int(les.shape[0]), xai_sec=t_xai, sanity_sec=time.time() - t0 - t_xai,
               erf_area90=erf_area)
    if len(d):
        for mth, g in d.groupby("method"):
            for c in ("hit", "energy_norm", "iou_topk", "rra", "insertion_auc", "deletion_auc"):
                if c in g:
                    out[f"xai_{mth}_{c}"] = float(g[c].mean())
    if rel_rows:
        out["reliance_lri"] = float(pd.DataFrame(rel_rows).lri.mean())
    return out


if __name__ == "__main__":
    main()
