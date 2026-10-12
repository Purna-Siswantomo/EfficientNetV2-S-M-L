# Scaling EfficientNetV2 and explanation quality in medical imaging

Code and results for the manuscript *Scaling EfficientNetV2 Does Not Yield Better Explanations in Medical Imaging: A Controlled Study Separating Model and Explainer Effects* by Purna Siswantomo, Ahmad Ilham and Wendy Sarasjati (Department of Informatics, Universitas Muhammadiyah Semarang), submitted to *AI Open*.

EfficientNetV2-S, -M and -L were trained with one recipe on three medical imaging datasets with lesion masks (BUSI, HAM10000, SIIM-ACR). Their Grad-CAM, Grad-CAM++ and Integrated Gradients maps were scored against the masks, and an explainer-free lesion reliance index (LRI) was computed for every image.

## Repository layout

| Path | Contents |
|---|---|
| `effnetv2_sml_xai_kaggle_v3.ipynb` | Kaggle notebook that runs every experiment (comments and instructions are in Indonesian) |
| `results/train.py` | training and per-run evaluation |
| `results/data_utils.py` | dataset indexing, masks and splits |
| `results/xai_utils.py` | Grad-CAM, Grad-CAM++, IG, localisation metrics, LRI, faithfulness, sanity checks, ERF |
| `results/analysis.py` | merged analysis: bootstrap, Wilcoxon, Holm, TOST, mixed model, interpretation rules |
| `results/bench.py` | efficiency benchmark (parameters, GMAC, latency, throughput, memory) |
| `results/best_lr.json` | learning rate selected per dataset, variant (and BUSI fold) |
| `results/efficiency_benchmark.json`, `results/environment.txt` | raw benchmark output and the full `pip freeze` of the Kaggle environment |
| `results/results_lr_v3/` | 74 learning-rate search runs (history, test predictions, summary, log, CodeCarbon emissions) |
| `results/results_v3/main/` | 45 main runs, one folder per `<dataset>/<variant>/run-<run>_seed-<seed>_*` |
| `results/summary/` | merged tables used in the paper (`runs.csv`, `xai_per_image_all.csv`, `reliance_per_image_all.csv`, paired tests, `rq5_rq6_reading.csv`, `analysis_report.txt`, ...) |
| `manuscript/make_tables.py`, `manuscript/make_figures.py` | rebuild the LaTeX tables and Figures 2-5 from `results/` |
| `manuscript/tables/`, `manuscript/figures/` | the tables and figures as they appear in the manuscript; `figures/data/` holds the numbers behind each plot |

Each main run folder contains `history.csv`, `test_predictions.csv`, `summary.json`, `xai_per_image.csv`, `reliance_per_image.csv`, `sanity_per_image.csv`, `erf.npy`, `emissions.csv` and `log.txt`.

## Data

Images and masks are not redistributed here. The notebook reads them from these sources:

| Dataset | Source |
|---|---|
| Curated BUSI | Zenodo, https://doi.org/10.5281/zenodo.19047974 (upload the zip as a private Kaggle dataset) |
| HAM10000 images and metadata | Kaggle `kmader/skin-cancer-mnist-ham10000` |
| HAM10000 lesion masks | Kaggle `tschandl/ham10000-lesion-segmentations` |
| SIIM-ACR (PNG images and masks) | Kaggle `vbookshelf/pneumothorax-chest-xray-images-and-masks` |

Please cite the original dataset papers listed in the manuscript when you use them.

## Reproducing the experiments

1. Import the notebook into Kaggle and set **Accelerator: GPU T4 x2** and **Internet: On**.
2. Add the four datasets above as notebook inputs. Paths are detected automatically under `/kaggle/input`.
3. Set `MODE` in the configuration cell and run, in this order: `lr_search`, `lr_extend` (writes `best_lr.json`), then `full`. The `full` plan resumes unfinished runs, so it can span several sessions. It can also be split over three Kaggle accounts with `SHARD = 1/2/3`, each run by its own owner, followed by one merge session with `SHARD = None`.
4. The merge session runs the analysis and the efficiency benchmark and writes `results/summary/`.

The 45 main runs took about 47 GPU hours on T4 GPUs in total (sum of `total_sec` in `runs.csv`), one run per GPU. The notebook also defines `pilot`, `native_res` and `datafrac` plans; they are not part of the reported results.

## Rebuilding the tables and figures

```bash
cd manuscript
python make_tables.py    # writes tables/*.tex
python make_figures.py   # writes figures/*.pdf|png, figures/data/*.csv and runs consistency checks
```

`make_figures.py` checks that all 45 runs are present, that the ERF area recomputed from every `erf.npy` matches `runs.csv`, and that the forest plot values equal the paired differences in `results/summary/xai_paired_tests.csv`. It also writes unstyled single panels to `../bukti_figure/figure_mentah/` for checking.

## Software

PyTorch 2.10.0, torchvision 0.25.0, timm 1.0.26, pytorch-grad-cam 1.5.7, Captum 0.9.0, statsmodels 0.14.6 and CodeCarbon 3.3.1. The complete environment is in `results/environment.txt`.

## Citation

If you use this code or these results, please cite the manuscript (reference to be added after publication).
