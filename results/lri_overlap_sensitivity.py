"""Post-hoc sensitivity of the HAM10000 lesion reliance (LRI) contrasts to the control-region overlap.

Not part of the pre-specified analysis. Re-runs the main paired test (analysis.paired_tests on the
common-correct images) after dropping images whose control region overlaps the dilated lesion by more
than a threshold. Each threshold uses a fresh generator with the same seed.

    python lri_overlap_sensitivity.py            # reads summary/reliance_per_image_all.csv
"""
import os
import numpy as np
import pandas as pd
import analysis as an

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_BOOT, THRESHOLDS = 0, 2000, (1.0, 0.5, 0.1)  # 1.0 = no filter (primary analysis subset)

R = pd.read_csv(os.path.join(HERE, "summary", "reliance_per_image_all.csv"))
R = R[R.dataset == "ham10000"]
rows = []
for thr in THRESHOLDS:
    T = an.paired_tests(an.common_correct(R[R.ctrl_overlap <= thr]), ["lri"], ["occlusion"], N_BOOT,
                        np.random.default_rng(SEED), f"ctrl_overlap<={thr}")
    T.insert(0, "max_ctrl_overlap", thr)
    T["seed"], T["n_boot"] = SEED, N_BOOT
    rows.append(T)
out = pd.concat(rows)[["max_ctrl_overlap", "dataset", "metric", "subset", "A", "B", "n_images", "delta",
                       "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "margin", "equivalent", "seed", "n_boot"]]
out.to_csv(os.path.join(HERE, "summary", "lri_overlap_sensitivity.csv"), index=False)
print(out.round(4).to_string(index=False))
