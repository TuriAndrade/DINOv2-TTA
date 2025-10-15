# TTA — Test-Time Adaptation for DINOv2 on ImageNet-C/R

This repository contains code to **evaluate and adapt DINOv2 classifiers under distribution shift** using **Test-Time Adaptation (TTA)**. It targets robustness benchmarks on **ImageNet-C** and **ImageNet-R**, plus the standard **ImageNet val** split.

Currently implemented TTA methods:

- **[Tent](https://arxiv.org/abs/2006.10726)** 
- **[EATA](https://arxiv.org/abs/2204.02610)**
- **[TEA](https://arxiv.org/abs/2311.14402)**

> The code is designed so you can easily plug in **new TTA methods**: add a module under `tta/` respecting the structure of the provided methods and add the corresponding config to `config/tta.py`.

---

## What’s here

- **DINOv2 classifier eval** (no adaptation): `dinov2_clf_eval.py`
- **DINOv2 + TTA eval** (with adaptation): `dinov2_tta_clf_eval.py`
- **TTA method implementations**: `tta/` (e.g., Tent, EATA, TEA)
- **ImageNet data prep**: `data/imagenet/` (scripts/notebooks to fetch & preprocess)
- **HDF5 dataloader**: `dataloaders/` (efficient `.h5` loading for val/C/R)
- **Configs**: `config/` (model/data/eval knobs)
- **Experiment helpers**: `scripts/` (orchestrate full sweeps/baselines)
- **Env & deps**: `example.env`, `requirements.txt`

---

## Datasets & preprocessing

The ImageNet data used here was **downloaded via Kaggle** and then **converted to a single HDF5 file** for fast, sequential access during evaluation. The `data/imagenet/` folder contains small utilities to convert the images + labels into an **`.h5`** (HDF5) format consumed by `dataloaders/`. 

## Reproducing the HDF5 dataset

This repo expects a single HDF5 file (e.g., `imagenet.h5`) that packs **ImageNet (val)**, **ImageNet-C**, and **ImageNet-R** with consistent preprocessing and optional subset indices. Below is a clean, reproducible pipeline.

### 1) Download datasets
From Kaggle (via UI or `kaggle` CLI):
- [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)
- [ImageNet-C](https://www.kaggle.com/datasets/husnifdu/imagenet-c)
- [ImageNet-R](https://www.kaggle.com/datasets/my1nonly/imagenet-r)

Unzip everything under a common `root` directory.

### 2) Arrange the directory layout
Organize the extracted folders so these paths exist:

- `train_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "train"`
- `val_dir   = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"`
- `test_dir  = root / "ILSVRC" / "Data" / "CLS-LOC" / "test"`  (optional)
- `c_dir     = root / "imagenet-c"`
- `r_dir     = root / "imagenet-r"`

A concrete tree looks like:

    root/
    ├─ ILSVRC/
    │  └─ Data/
    │     └─ CLS-LOC/
    │        ├─ train/          # optional
    │        ├─ val/
    │        └─ test/           # optional
    ├─ imagenet-c/
    └─ imagenet-r/

### 3) Build the base HDF5 (ImageNet val)
Create the initial HDF5 with the **clean validation** split (and, optionally, train/test if your script supports it):

    python data/imagenet/build_imagenet_h5.py \
      --root /abs/path/to/root \
      --out  /abs/path/to/imagenet.h5

This step:
- Resizes and center-crops to **224×224**,
- Converts to **NCHW** tensors scaled to **[0,1]**,
- Applies **ImageNet mean/std** normalization,
- Stores global preprocessing attributes in the file.

### 4) Append ImageNet-C
Add the corruption families (grouped by category/corruption/severity) to the same HDF5:

    python data/imagenet/append_imagenet_c_h5.py \
      --root /abs/path/to/root \
      --h5   /abs/path/to/imagenet.h5

This will create groups like `c/blur/defocus_blur/5/{images,labels}`.

### 5) Append ImageNet-R
Add ImageNet-R into `r/{images,labels}`:

    python data/imagenet/append_imagenet_r_h5.py \
      --root /abs/path/to/root \
      --h5   /abs/path/to/imagenet.h5

### 6) Point the repo to your HDF5
Export the path (or edit `.env`):

    export IMAGENET_HDF5=/abs/path/to/imagenet.h5
    # or:
    cp example.env .env
    # then set IMAGENET_HDF5=/abs/path/to/imagenet.h5 in .env

Now you can pick any dataset config key (e.g., `imagenet/val`, `imagenet/c/contrast`) and run the evaluators:

    python dinov2_clf_eval.py --dataset-config imagenet/val
    python dinov2_tta_clf_eval.py --dataset-config imagenet/c/defocus_blur@10k --tta-config eata
    
---

## Results

**Top-1 (%)**. ImageNet-C numbers are for **severity = 5**. “val” = ImageNet validation; “r” = ImageNet-R.

| Corruption / Split | No-Adapt | EATA |
|---|---:|---:|
| brightness         | 81.6 | 81.8 |
| contrast           | 67.9 | 72.2 |
| defocus_blur       | 54.8 | 66.5 |
| elastic_transform  | 47.9 | 69.6 |
| fog                | 70.0 | 74.3 |
| frost              | 62.2 | 69.2 |
| gaussian_noise     | 56.5 | 65.3 |
| glass_blur         | 35.6 | 61.2 |
| impulse_noise      | 60.4 | 67.0 |
| jpeg_compression   | 75.2 | 76.0 |
| motion_blur        | 63.8 | 69.3 |
| pixelate           | 78.1 | 78.8 |
| r                  | 57.1 | 60.4 |
| shot_noise         | 57.8 | 66.7 |
| snow               | 72.9 | 75.2 |
| val                | 86.3 | 85.9 |
| zoom_blur          | 61.3 | 66.6 |

---

## Installation

```bash
# 1) Create env (example with conda; use your preferred tool)
conda create -n tta python=3.10 -y
conda activate tta

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set up environment variables (paths, etc.)
cp example.env .env
# edit .env to point to your ImageNet HDF5 and cache directories
```

## Running

### 1) Plain DINOv2 classifier (no TTA)

`dinov2_clf_eval.py` supports “zero-flag” execution. Key args:

- `--dataset-config` *(str, default: `imagenet/val`)* — template for the dataset (e.g., `imagenet/train`, `imagenet/val`).
- `--out-dir` *(Path, default: `./results`)* — parent dir to save configs/results.
- `--out-name` *(Path, default: `None`)* — optional subdir name under `--out-dir`.
- `--arch` *(choices: `vits14|vitb14|vitl14|vitg14`, default: `vitl14`)* — DINOv2 backbone.
- `--layers` *(choices: `1|4`, default: `4`)* — how many intermediate layers feed the linear head.
- `--batch-size` *(int, default: `64`)*
- `--num-workers` *(int, default: `4`)*
- `--topk` *(ints, default: `1 5`)* — top-k accuracies to compute.
- `--device` *(default: `cuda`)* — `cuda` or `cpu`.
- `--reg` *(flag)* — use the register-token variant (`_reg`).

### 2) DINOv2 + TTA

`dinov2_tta_clf_eval.py` supports “zero-flag” execution. Key args:

- `--dataset-config` *(str, default: `imagenet/val`)* — template for the dataset (e.g., `imagenet/val`, `imagenet/train`).
- `--tta-config` *(str, default: `eata`)* — TTA method config template (e.g., `eata`, `tent`, `tea`).
- `--optim-config` *(str, default: `norm`)* — optimizer config template.
- `--out-dir` *(Path, default: `./results`)* — parent dir to save configs/results.
- `--out-name` *(Path, default: `None`)* — optional subdir name under `--out-dir`.
- `--load-model-path` *(Path, default: `None`)* — optional checkpoint to load.
- `--arch` *(choices: `vits14|vitb14|vitl14|vitg14`, default: `vitl14`)* — DINOv2 backbone.
- `--layers` *(choices: `1|4`, default: `4`)* — how many intermediate layers feed the linear head.
- `--batch-size` *(int, default: `32`)*
- `--num-workers` *(int, default: `4`)*
- `--topk` *(ints, default: `1 5`)* — top-k accuracies to compute.
- `--device` *(default: `cuda`)* — `cuda` or `cpu`.
- `--reg` *(flag)* — use the register-token variant (`_reg`).

### Running the runlists in `scripts/`

The `scripts/` folder contains a simple batch executor and ready-made runlists to evaluate **DINOv2 Vit-L/14** across all baselines.

#### What’s included
- `scripts/script_executor.sh` — reads a text file and executes each line as a full shell command.
- Runlists (no-register variant across baselines):
  - `scripts/dinov2_vitl14_lc_no_adapt.txt`
  - `scripts/dinov2_vitl14_lc_eata.txt`
  - `scripts/dinov2_vitl14_lc_tent.txt`
  - `scripts/dinov2_vitl14_lc_energy_tta.txt`

Each `*.txt` file contains one command per line (e.g., `python dinov2_clf_eval.py ...` or `python dinov2_tta_clf_eval.py ...`). Empty lines or lines starting with `#` are ignored.

#### 1) Make the executor executable
```bash
chmod +x scripts/script_executor.sh
```

#### 2) Run any provided runlist
```bash
# No adaptation baselines (Vit-L/14, no registers)
./scripts/script_executor.sh scripts/dinov2_vitl14_lc_no_adapt.txt

# EATA baselines
./scripts/script_executor.sh scripts/dinov2_vitl14_lc_eata.txt

# TENT baselines
./scripts/script_executor.sh scripts/dinov2_vitl14_lc_tent.txt

# Energy-based TTA baselines
./scripts/script_executor.sh scripts/dinov2_vitl14_lc_energy_tta.txt
```

#### 3) Create your own runlist (optional)
```bash
# my_runs.txt
python dinov2_clf_eval.py --dataset-config imagenet/val --arch vitg14 --layers 4 --out-dir ./results --out-name vitg14_val
python dinov2_tta_clf_eval.py --dataset-config imagenet/val --tta-config eata --optim-config norm --arch vitg14 --layers 4 --out-dir ./results --out-name vitg14_val_eata
```

Then run:
```bash
./scripts/script_executor.sh my_runs.txt
```

## Extending the repo with a new TTA method

This repo expects **the model’s `forward()` to perform the full adaptation loop** and return logits. The evaluator calls:

```python
logits = self.model(imgs)  # forward must encapsulate adaptation
```

Your TTA module should mirror the lifecyle of the methods available in `tta/`.
