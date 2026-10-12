"""Dataset indexing, caching (pre-resize), and splitting for the EfficientNetV2 S/M/L x XAI study.

Layout under <cache_root>/<dataset>/:
    index.csv            uid, image_src, mask_src, label, label_name, group   (size independent)
    splits.json          run_id -> {train, val, test} lists of uid        (size independent)
    <size>/img/<uid>.jpg
    <size>/mask/<uid>.png
    <size>/masks.csv     uid, mask_area, mask_lost  (mask_lost = lesion vanished after resizing)

Splits do not depend on the image size, so the 384 px and native-resolution runs use the same test images.
"""
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find(root, pattern):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def _fail(name, root, hint):
    tree = []
    for dirpath, dirnames, filenames in os.walk(root or "."):
        depth = dirpath[len(root or "."):].count(os.sep)
        if depth > 3:
            continue
        tree.append(f"{dirpath}  ({len(filenames)} files)")
        if len(tree) > 40:
            break
    raise RuntimeError(
        f"[{name}] could not index dataset under {root}.\n{hint}\n"
        "Folder tree (first levels):\n  " + "\n  ".join(tree)
        + "\nSet DATASET_PATHS in the CONFIG cell to the correct folder."
    )


# ---------------------------------------------------------------- BUSI (curated)
BUSI_CLASSES = ["normal", "benign", "malignant"]


def index_busi(root):
    """Works for the original BUSI layout (<class>/<name>.png + <name>_mask.png, extra masks <name>_mask_1.png)
    and for re-packaged versions that keep class folders or class prefixes in file names.
    Multiple masks for one image are merged (union). BUSI has no patient ID, so group = image
    (limitation reported in the paper)."""
    if not root:
        _fail("busi", root, "Path not set.")
    files = [f for f in _find(root, "*") if f.lower().endswith(IMG_EXT)]
    if not files:
        _fail("busi", root, "No image files found.")
    masks, masks_by_stem, images = {}, {}, []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"^(.*?)[ _-]?(mask|Mask|MASK)(?:[ _-]?\d+)?$", base)
        if m:
            stem = m.group(1).rstrip(" _-")
            masks.setdefault((os.path.dirname(f), stem), []).append(f)
            masks_by_stem.setdefault(stem, []).append(f)
        else:
            images.append(f)
    rows = []
    for f in images:
        base = os.path.splitext(os.path.basename(f))[0]
        low = f.lower()
        cls = None
        for c in BUSI_CLASSES:
            if f"{os.sep}{c}{os.sep}" in low or os.path.basename(low).startswith(c):
                cls = c
                break
        if cls is None:
            continue
        mk = masks.get((os.path.dirname(f), base), []) or masks_by_stem.get(base, [])
        uid = f"busi_{len(rows):05d}"
        rows.append(dict(uid=uid, image_src=f, mask_src="|".join(sorted(mk)),
                         label=BUSI_CLASSES.index(cls), label_name=cls, group=uid))
    df = pd.DataFrame(rows)
    if df.empty or df.label.nunique() < 3:
        _fail("busi", root, "Expected class folders or file prefixes normal/benign/malignant.")
    return df


# ---------------------------------------------------------------- HAM10000
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def index_ham(root_images, root_masks):
    metas = [p for p in _find(root_images, "*.csv") if "metadata" in os.path.basename(p).lower()]
    if not metas:
        _fail("ham10000", root_images, "HAM10000_metadata.csv (with lesion_id, image_id, dx) not found.")
    meta = pd.read_csv(metas[0])
    imgs = {}
    for p in _find(root_images, "ISIC_*.jpg") + _find(root_images, "ISIC_*.png"):
        if "segmentation" in os.path.basename(p).lower():
            continue
        imgs.setdefault(re.search(r"ISIC_\d+", p).group(0), p)
    msk = {}
    for p in _find(root_masks, "ISIC_*"):
        if p.lower().endswith(IMG_EXT):
            msk.setdefault(re.search(r"ISIC_\d+", p).group(0), p)
    rows = []
    for r in meta.itertuples():
        if r.image_id not in imgs:
            continue
        rows.append(dict(uid=r.image_id, image_src=imgs[r.image_id], mask_src=msk.get(r.image_id, ""),
                         label=HAM_CLASSES.index(r.dx), label_name=r.dx, group=r.lesion_id))
    df = pd.DataFrame(rows)
    if df.empty:
        _fail("ham10000", root_images, "No ISIC_*.jpg images matched the metadata.")
    print(f"[ham10000] {len(df)} images, {int((df['mask_src'] != '').sum())} with masks")
    return df


# ---------------------------------------------------------------- SIIM-ACR (PNG version)
def index_siim(root, n_neg_per_pos=1.0, seed=42):
    """Pairs every image with a mask of the same file name in a folder whose path contains 'mask'.
    Label = mask has any positive pixel. Negatives are sub-sampled with a fixed seed (class balancing)."""
    if not root:
        _fail("siim", root, "Path not set.")
    files = _find(root, "*.png")
    is_mask = lambda f: "mask" in os.path.relpath(f, root).lower()  # the Kaggle slug itself contains "masks"
    mask_files = {os.path.basename(f): f for f in files if is_mask(f)}
    img_files = [f for f in files if not is_mask(f)]
    if not img_files or not mask_files:
        _fail("siim", root, "Expected separate image and mask folders with matching PNG file names.")
    rows = []
    for f in img_files:
        m = mask_files.get(os.path.basename(f))
        if m is not None:
            rows.append(dict(uid=os.path.splitext(os.path.basename(f))[0], image_src=f, mask_src=m))
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor(4) as ex:
        pos = list(ex.map(_mask_positive, df["mask_src"], chunksize=64))
    df["label"] = np.array(pos, dtype=int)
    df["label_name"] = np.where(df.label == 1, "pneumothorax", "normal")
    df.loc[df.label == 0, "mask_src"] = ""
    n_pos = int(df.label.sum())
    neg = df[df.label == 0].sample(n=min(int(n_pos * n_neg_per_pos), int((df.label == 0).sum())),
                                   random_state=seed)
    df = pd.concat([df[df.label == 1], neg]).sort_values("uid").reset_index(drop=True)
    df["group"] = df["uid"]
    print(f"[siim] {n_pos} positive + {len(neg)} sampled negative")
    return df


def _mask_positive(p):
    return int(np.asarray(Image.open(p).convert("L")).max() > 0)


# ---------------------------------------------------------------- caching
def _cache_one(args):
    src, msrc, dst, mdst, size = args
    if not os.path.exists(dst):
        Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC).save(dst, quality=95)
    area, lost = np.nan, False
    if msrc:
        orig_pos = False
        m = None
        for p in msrc.split("|"):
            im = Image.open(p).convert("L")
            orig_pos = orig_pos or (np.asarray(im).max() > 127)
            a = np.asarray(im.resize((size, size), Image.NEAREST)) > 127
            m = a if m is None else (m | a)
        area = float(m.mean())
        lost = bool(orig_pos and area == 0)
        if not os.path.exists(mdst):
            Image.fromarray((m * 255).astype(np.uint8)).save(mdst)
    return area, lost


def build_cache(df, ds_dir, size, workers=4):
    d = os.path.join(ds_dir, str(size))
    os.makedirs(os.path.join(d, "img"), exist_ok=True)
    os.makedirs(os.path.join(d, "mask"), exist_ok=True)
    jobs = [(r.image_src, r.mask_src, os.path.join(d, "img", f"{r.uid}.jpg"),
             os.path.join(d, "mask", f"{r.uid}.png"), size) for r in df.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_cache_one, jobs, chunksize=32))
    ms = pd.DataFrame({"uid": df.uid.values, "mask_area": [r[0] for r in res], "mask_lost": [r[1] for r in res]})
    ms.to_csv(os.path.join(d, "masks.csv"), index=False)
    n_lost = int(ms.mask_lost.sum())
    if n_lost:
        print(f"  WARNING: {n_lost} lesion masks became empty after resizing to {size}px (excluded from XAI; report this)")
    return ms


# ---------------------------------------------------------------- splits
def make_splits(df, dataset, n_folds=5, test_frac=0.15, val_frac=0.15, seed=42):
    """Returns dict run_id -> dict(train=[uid], val=[uid], test=[uid]).
    busi: 5-fold CV (run_id = fold; val is carved out of that fold's train part, so learning-rate
    selection on it never touches the fold's test images).
    ham10000/siim: one fixed split grouped by `group` (runs differ only in seed), run_id 'fixed'."""
    y, g, uid = df.label.values, df.group.values, df.uid.values
    splits = {}
    if dataset == "busi":
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        for k, (trv, te) in enumerate(skf.split(uid, y)):
            tr, va = train_test_split(trv, test_size=val_frac / (1 - 1 / n_folds), stratify=y[trv],
                                      random_state=seed)
            splits[str(k)] = dict(train=uid[tr].tolist(), val=uid[va].tolist(), test=uid[te].tolist())
    else:
        sgkf = StratifiedGroupKFold(n_splits=int(round(1 / test_frac)), shuffle=True, random_state=seed)
        trv, te = next(sgkf.split(uid, y, g))
        sgkf2 = StratifiedGroupKFold(n_splits=int(round((1 - test_frac) / val_frac)), shuffle=True,
                                     random_state=seed)
        tr_i, va_i = next(sgkf2.split(uid[trv], y[trv], g[trv]))
        splits["fixed"] = dict(train=uid[trv][tr_i].tolist(), val=uid[trv][va_i].tolist(),
                               test=uid[te].tolist())
    # leakage guard: no group may appear in two partitions
    grp = dict(zip(uid, g))
    for rid, s in splits.items():
        gs = {k: {grp[u] for u in s[k]} for k in ("train", "val", "test")}
        assert not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"]) and not (gs["train"] & gs["val"]), \
            f"group leakage in split {rid}"
    return splits


def prepare(dataset, paths, cache_root, sizes, workers=4, siim_neg_ratio=1.0):
    """Index + split (once) and cache every requested size. Returns (index_df, splits)."""
    out_dir = os.path.join(cache_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    idx_csv, split_json = os.path.join(out_dir, "index.csv"), os.path.join(out_dir, "splits.json")
    if os.path.exists(idx_csv) and os.path.exists(split_json):
        df, splits = pd.read_csv(idx_csv, keep_default_na=False), json.load(open(split_json))
    else:
        if dataset == "busi":
            df = index_busi(paths["busi"])
        elif dataset == "ham10000":
            df = index_ham(paths["ham10000_images"], paths["ham10000_masks"])
        elif dataset == "siim":
            df = index_siim(paths["siim"], n_neg_per_pos=siim_neg_ratio)
        else:
            raise ValueError(dataset)
        splits = make_splits(df, dataset)
        df.to_csv(idx_csv, index=False)
        json.dump(splits, open(split_json, "w"))
        print(f"[{dataset}] classes: {df.label_name.value_counts().to_dict()}")
    for size in sizes:
        if not os.path.exists(os.path.join(out_dir, str(size), "masks.csv")):
            print(f"[{dataset}] caching {len(df)} images at {size}px ...")
            build_cache(df, out_dir, size, workers)
    return df, splits
