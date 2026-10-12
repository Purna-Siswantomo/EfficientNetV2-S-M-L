"""Saliency methods, explanation-quality metrics, reference maps, lesion-reliance test,
cascading sanity check and effective receptive field. Model = timm EfficientNetV2
(conv_stem, bn1, blocks[stages], conv_head, bn2, global_pool, classifier)."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- saliency
def target_layer(model):
    """Last conv block output (after BN + SiLU): same 1280 x (H/32) x (W/32) map for S, M and L."""
    return model.bn2


def make_explainers(model, methods):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    tl = [target_layer(model)]
    ex = {}
    if "gradcam" in methods:
        ex["gradcam"] = GradCAM(model=model, target_layers=tl)
    if "gradcampp" in methods:
        ex["gradcampp"] = GradCAMPlusPlus(model=model, target_layers=tl)
    if "scorecam" in methods:
        sc = ScoreCAM(model=model, target_layers=tl)
        sc.batch_size = 32
        ex["scorecam"] = sc
    if "ig" in methods:
        from captum.attr import IntegratedGradients
        ex["ig"] = IntegratedGradients(model)
    return ex


def release(explainers):
    for e in explainers.values():
        a = getattr(e, "activations_and_grads", None)
        if a is not None:
            a.release()


def saliency(name, explainer, x, target, ig_steps=32):
    """x: [1,3,H,W] normalized tensor. Returns dict of HxW non-negative float maps.
    IG returns two maps: 'ig_raw' (|attribution| summed over channels) and 'ig' (9x9 box-smoothed,
    declared modification so pixel-level IG is comparable to CAM maps for localization)."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    if name == "ig":
        attr = explainer.attribute(x, baselines=torch.zeros_like(x), target=int(target),
                                   n_steps=ig_steps, internal_batch_size=4)  # 4 keeps L (FP32, 384 px) well inside 16 GB
        s = attr.detach().abs().sum(1)[0].float()
        sm = F.avg_pool2d(s[None, None], 9, stride=1, padding=4)[0, 0]
        return {"ig_raw": s.cpu().numpy(), "ig": sm.cpu().numpy()}
    cam = explainer(input_tensor=x, targets=[ClassifierOutputTarget(int(target))])
    return {name: cam[0].astype(np.float32)}


# ---------------------------------------------------------------- reference maps (no model)
def reference_maps(mask, grid, seed=0):
    """Maps that tell how good a score is:
    base_center : isotropic Gaussian at the image centre (sigma = H/4) -> centre-bias baseline
    base_random : random grid x grid map, bilinear-upsampled (same resolution as a CAM)
    oracle_grid : lesion fraction per CAM cell, bilinear-upsampled -> resolution-matched reference
                  (what a CAM of this resolution scores if it follows the mask exactly). It is close to the
                  ceiling for `hit` and `iou_topk`, but NOT for `energy`: a map concentrated on the cells fully
                  inside the lesion can exceed it (seen in the pilot: Grad-CAM++ 0.83 vs oracle 0.78 on HAM10000)."""
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sig = H / 4
    center = np.exp(-(((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * sig ** 2))).astype(np.float32)
    rng = np.random.default_rng(seed)
    rnd = torch.from_numpy(rng.random((grid, grid)).astype(np.float32))[None, None]
    rnd = F.interpolate(rnd, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    orc = F.adaptive_avg_pool2d(m, grid)
    orc = F.interpolate(orc, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    return {"base_center": center, "base_random": rnd, "oracle_grid": orc}


# ---------------------------------------------------------------- metrics vs mask
def _norm(s):
    s = s - s.min()
    m = s.max()
    return s / m if m > 0 else None


def localization_metrics(sal, mask):
    """sal: HxW float, mask: HxW bool. Returns raw scores and the chance-normalized energy
    energy_norm = (energy - a) / (1 - a), a = lesion area: 0 = uniform map, 1 = all energy in lesion.
    rra = Relevance Rank Accuracy (Arras et al., 2022; used by Cedro & Chlebus, 2026): share of the
    top-k pixels (k = lesion size) that fall inside the lesion."""
    s = _norm(sal)
    area = float(mask.mean())
    if s is None:
        return dict(hit=np.nan, energy=np.nan, energy_norm=np.nan, iou50=np.nan, iou_topk=np.nan, rra=np.nan,
                    mask_area=area)
    flat = s.ravel()
    mx = flat.max()
    hit = float(mask.ravel()[flat >= mx - 1e-6].mean())       # ties at the max split the credit
    energy = float((s * mask).sum() / (s.sum() + 1e-8))
    b = s >= 0.5
    iou50 = float((b & mask).sum() / ((b | mask).sum() + 1e-8))
    k = max(1, int(mask.sum()))
    thr = np.partition(flat, -k)[-k]
    bk = s >= thr
    iou_topk = float((bk & mask).sum() / ((bk | mask).sum() + 1e-8))
    rra = float((bk & mask).sum() / max(int(bk.sum()), 1))   # ties at the threshold enlarge bk; count them all
    return dict(hit=hit, energy=energy, energy_norm=(energy - area) / max(1 - area, 1e-6),
                iou50=iou50, iou_topk=iou_topk, rra=rra, mask_area=area)


# ---------------------------------------------------------------- faithfulness
def blur(x, k=31, sigma=10.0):
    ax = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-(ax.float() ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k).repeat(3, 1, 1, 1).to(x.dtype)
    y = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g, groups=3)
    return F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3), groups=3)


@torch.no_grad()
def _probs(model, batch, target, amp, chunk=16):
    out = []
    for i in range(0, batch.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and batch.is_cuda):
            out.append(model(batch[i:i + chunk]).float().softmax(1)[:, int(target)])
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def insertion_deletion(model, x, target, order, steps=25, amp=True):
    """RISE-style insertion/deletion AUC (Petsiuk et al., 2018) for a given pixel order (most salient first).
    Deletion replaces pixels by 0 (= dataset mean after normalization); insertion starts from a blurred image.
    Scores must be compared with the same image's random-order AUC (see analysis), because models of
    different size react differently to these out-of-distribution inputs."""
    _, _, H, W = x.shape
    order = torch.as_tensor(order, device=x.device)
    bl = blur(x)
    counts = torch.linspace(0, H * W, steps + 1).long()
    dele, ins = [], []
    for c in counts:
        m = torch.zeros(H * W, device=x.device)
        m[order[:c]] = 1
        m = m.view(1, 1, H, W)
        dele.append(x * (1 - m))
        ins.append(bl * (1 - m) + x * m)
    p = _probs(model, torch.cat(dele + ins), target, amp)
    d, n = p[:steps + 1], p[steps + 1:]
    return dict(deletion_auc=float(_trapz(d, dx=1 / steps)), insertion_auc=float(_trapz(n, dx=1 / steps)))


def saliency_order(sal, seed=0):
    """Most-salient-first pixel order; exact ties (blocky upsampled CAMs) broken at random, reproducibly."""
    rng = np.random.default_rng(seed)
    flat = sal.ravel().astype(np.float64)
    return np.lexsort((rng.random(flat.size), -flat))


# ---------------------------------------------------------------- lesion reliance (model property, no XAI)
def _dilate(mask_t, px):
    return F.max_pool2d(mask_t, 2 * px + 1, stride=1, padding=px) if px > 0 else mask_t


def _control_region(m, shifts):
    """Same-shape region shifted to overlap the lesion as little as possible (torch.roll wrap-around)."""
    best, best_ov = None, 2.0
    tot = float(m.sum()) + 1e-8
    for dy, dx in shifts:
        c = torch.roll(m, shifts=(dy, dx), dims=(-2, -1))
        ov = float((c * m).sum()) / tot
        if ov < best_ov:
            best, best_ov = c, ov
    return best, best_ov


@torch.no_grad()
def lesion_reliance(model, x, target, mask, dilate_px=8, amp=True):
    """How much the model's confidence depends on the lesion region itself (occlusion test, no XAI).
    drop_lesion : p(orig) - p(lesion replaced by its blurred version)
    drop_ctrl   : same for an equally large region elsewhere (controls for the size of the perturbation)
    lri         : drop_lesion - drop_ctrl  (> 0: the model relies on the lesion more than on other tissue)
    p_lesion_only : p when everything outside the (dilated) lesion is blurred."""
    _, _, H, W = x.shape
    m = torch.from_numpy(mask.astype(np.float32)).to(x.device)[None, None]
    m = _dilate(m, dilate_px)
    shifts = [(int(H * a), int(W * b)) for a in (0, .25, .5, .75) for b in (0, .25, .5, .75) if (a, b) != (0, 0)]
    c, ov = _control_region(m, shifts)
    bl = blur(x)
    batch = torch.cat([x, bl * m + x * (1 - m), bl * c + x * (1 - c), x * m + bl * (1 - m)])
    p = _probs(model, batch, target, amp)
    return dict(p_orig=float(p[0]), drop_lesion=float(p[0] - p[1]), drop_ctrl=float(p[0] - p[2]),
                lri=float(p[2] - p[1]), p_lesion_only=float(p[3]), ctrl_overlap=float(ov),
                dilated_area=float(m.mean()))


# ---------------------------------------------------------------- cascading sanity check
def randomization_levels(model):
    """Top-down cascade (Adebayo et al., 2018), identical level names for S, M and L."""
    b = model.blocks
    return [("classifier", [model.classifier]),
            ("head", [model.conv_head, model.bn2]),
            ("stage-1", [b[-1]]),
            ("stage-2", [b[-2]]),
            ("stage-3", [b[-3]]),
            ("all", list(b[:-3]) + [model.conv_stem, model.bn1])]


@torch.no_grad()
def _randomize(mods, g):
    for mod in mods:
        for p in mod.parameters():
            scale = float(p.float().std()) if p.numel() > 1 else 1.0
            p.copy_(torch.randn(p.shape, generator=g).to(p) * (scale + 1e-3))


def cascade_models(model, seed=0):
    """Yields (level_index, level_name, randomized deep copy) with cumulative randomization."""
    m = copy.deepcopy(model).eval()
    g = torch.Generator().manual_seed(seed)
    for i, (name, _) in enumerate(randomization_levels(model)):
        mods = dict(randomization_levels(m))[name]
        _randomize(mods, g)
        yield i, name, m


def map_rank_corr(a, b, size=48):
    """Spearman correlation between the trained-model map and the randomized-model map.
    A constant randomized map means the explanation was destroyed: scored as 0."""
    ta = F.adaptive_avg_pool2d(torch.from_numpy(np.ascontiguousarray(a))[None, None].float(), size).ravel().numpy()
    tb = F.adaptive_avg_pool2d(torch.from_numpy(np.ascontiguousarray(b))[None, None].float(), size).ravel().numpy()
    if np.ptp(ta) == 0:
        return np.nan
    if np.ptp(tb) == 0 or not np.isfinite(tb).all():
        return 0.0
    r = spearmanr(ta, tb).correlation
    return float(r) if np.isfinite(r) else np.nan


# ---------------------------------------------------------------- effective receptive field
def effective_receptive_field(model, loader, device, n_images=32, mass=0.9):
    """Luo et al. (2016): gradient of the centre unit of the last feature map w.r.t. the input.
    Model-level descriptive quantity (one value per trained model), NOT an image-level predictor.
    Returns (erf_map HxW, fraction of image area holding `mass` of the gradient magnitude)."""
    model.eval()
    acc, seen = None, 0
    for x, *_ in loader:
        for i in range(x.shape[0]):
            xi = x[i:i + 1].to(device).requires_grad_(True)
            f = model.forward_features(xi)
            h, w = f.shape[-2:]
            f[0, :, h // 2, w // 2].sum().backward()
            gmap = xi.grad.detach().abs().sum(1)[0].float().cpu().numpy()
            acc = gmap if acc is None else acc + gmap
            seen += 1
            model.zero_grad(set_to_none=True)
            if seen >= n_images:
                break
        if seen >= n_images:
            break
    erf = acc / seen
    v = np.sort(erf.ravel())[::-1]
    cum = np.cumsum(v) / (v.sum() + 1e-12)
    area = float((np.searchsorted(cum, mass) + 1) / v.size)
    return erf.astype(np.float32), area
