# Data splits

Image-level split assignments used in all 45 main runs (no images or masks are redistributed).

- `<dataset>_index.csv`: `uid`, source image and mask path inside the Kaggle/Zenodo dataset (`owner/dataset/...`), `label`, `label_name`, and `group` (HAM10000 `lesion_id`; BUSI and SIIM-ACR have no patient identifier, so each image is its own group).
- `<dataset>_splits.json`: run id -> `{train, val, test}` lists of `uid`. BUSI uses folds `0`-`4`; HAM10000 and SIIM-ACR use one split, `fixed`.

The files were regenerated with `data_utils.prepare` (seed 42) and checked against the run outputs: the test set of every one of the 45 main runs in `results/results_v3/main/*/*/run-*/test_predictions.csv` matches `splits.json` exactly, labels agree with the index, and no `group` spans two partitions.
