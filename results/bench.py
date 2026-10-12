"""Efficiency benchmark per variant and image size, run alone on one GPU (no other job running).

Reported per variant:
  params_m, gmacs (multiply-accumulates; FlopCounterMode counts 1 MAC = 2 FLOPs, so gmacs = flops / 2,
  the unit used in the EfficientNetV2 paper), GPU latency (batch 1, FP16), GPU throughput (batch 32, FP16),
  CPU latency (batch 1, FP32), and a *controlled* training step: fixed batch 8, FP16, frozen BN, so peak
  training memory and training throughput are comparable across variants (the in-run numbers are not:
  they use different micro-batches and share 4 vCPUs with a parallel run)."""
import argparse
import json
import time

import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

WEIGHTS = {"s": "tf_efficientnetv2_s.in1k", "m": "tf_efficientnetv2_m.in1k", "l": "tf_efficientnetv2_l.in1k"}


def timed(fn, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--variants", default="s,m,l")
    a.add_argument("--img_sizes", default="384")
    a.add_argument("--n_classes", type=int, default=2)
    a.add_argument("--out", required=True)
    a.add_argument("--cpu_iters", type=int, default=20)
    a.add_argument("--gpu_iters", type=int, default=200)
    a.add_argument("--train_batch", type=int, default=8)
    a = a.parse_args()
    cuda = torch.cuda.is_available()
    torch.set_num_threads(4)
    res = []
    for size in [int(s) for s in a.img_sizes.split(",")]:
        for v in a.variants.split(","):
            m = timm.create_model(WEIGHTS[v], pretrained=False, num_classes=a.n_classes).eval()
            x1 = torch.randn(1, 3, size, size)
            with FlopCounterMode(display=False) as fc, torch.no_grad():
                m(x1)
            r = dict(variant=v, img_size=size, params_m=sum(p.numel() for p in m.parameters()) / 1e6,
                     gmacs=fc.get_total_flops() / 2e9)
            with torch.no_grad():
                r["cpu_latency_ms"] = 1000 * timed(lambda: m(x1), 3, a.cpu_iters, False)
            if cuda:
                g = m.cuda().to(memory_format=torch.channels_last)
                xg1 = x1.cuda().to(memory_format=torch.channels_last)
                xg32 = torch.randn(32, 3, size, size, device="cuda").to(memory_format=torch.channels_last)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    r["gpu_latency_ms"] = 1000 * timed(lambda: g(xg1), 20, a.gpu_iters, True)
                    torch.cuda.reset_peak_memory_stats()
                    r["gpu_throughput_img_s"] = 32 / timed(lambda: g(xg32), 5, 30, True)
                    r["gpu_infer_mem_gb_b32"] = torch.cuda.max_memory_allocated() / 2 ** 30
                # controlled training step
                g.train()
                for mod in g.modules():
                    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                        mod.eval()
                opt = torch.optim.AdamW(g.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler("cuda")
                xb = torch.randn(a.train_batch, 3, size, size, device="cuda").to(memory_format=torch.channels_last)
                yb = torch.randint(0, a.n_classes, (a.train_batch,), device="cuda")

                def step():
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = nn.functional.cross_entropy(g(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                try:
                    torch.cuda.reset_peak_memory_stats()
                    dt = timed(step, 3, 20, True)
                    r[f"train_img_s_b{a.train_batch}"] = a.train_batch / dt
                    r[f"train_mem_gb_b{a.train_batch}"] = torch.cuda.max_memory_allocated() / 2 ** 30
                except torch.cuda.OutOfMemoryError:
                    r[f"train_img_s_b{a.train_batch}"] = float("nan")
                    r[f"train_mem_gb_b{a.train_batch}"] = float("nan")
                r["gpu_name"] = torch.cuda.get_device_name(0)
                del g, opt
                torch.cuda.empty_cache()
            print(r, flush=True)
            res.append(r)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
