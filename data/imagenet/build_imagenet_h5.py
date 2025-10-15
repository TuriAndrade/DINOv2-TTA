import os, json, csv, sys, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm

# ---- Defaults (torchvision ImageNet) ----
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224
RESIZE_MIN = 256
IMG_EXTS = {".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".bmp"}

# ---------- Helpers ----------
def has_subdirs(path: Path) -> bool:
    try:
        for d in path.iterdir():
            if d.is_dir():
                return True
    except FileNotFoundError:
        return False
    return False

def center_crop_224(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    scale = RESIZE_MIN / min(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - IMG_SIZE) // 2
    top  = (new_h - IMG_SIZE) // 2
    return img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

def load_and_preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = center_crop_224(im)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))  # C,H,W float32

def read_synset_mapping(root: Path) -> Dict[str, str]:
    for cand in [root / "LOC_synset_mapping.txt", *root.rglob("LOC_synset_mapping.txt")]:
        if cand.exists():
            out = {}
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    wnid, *rest = line.split(" ", 1)
                    out[wnid] = rest[0] if rest else wnid
            return out
    return {}

def build_train_lists(train_dir: Path) -> Tuple[List[Path], List[str]]:
    images, labels = [], []
    if not train_dir.exists(): return images, labels
    if not has_subdirs(train_dir): return images, labels
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    for cdir in tqdm(class_dirs, desc="Indexing train classes", unit="cls", dynamic_ncols=True):
        wnid = cdir.name
        for p in cdir.iterdir():
            if p.is_file() and p.suffix in IMG_EXTS:
                images.append(p); labels.append(wnid)
    return images, labels

def read_val_labels_from_csv(root: Path) -> Dict[str, str]:
    # Map filename -> wnid using LOC_val_solution.csv (first wnid token per row)
    csv_path = None
    for cand in [root / "LOC_val_solution.csv", *root.rglob("LOC_val_solution.csv")]:
        if cand.exists():
            csv_path = cand; break
    if not csv_path: return {}
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            image_id = row[0].strip()
            rest = " ".join(row[1:]).strip()
            if not rest: continue
            wnid = rest.split()[0]
            candidates = [
                f"{image_id}",
                f"{image_id}.JPEG",
                f"{image_id}.jpeg",
                f"{image_id}.jpg",
            ]
            for c in candidates:
                mapping[c] = wnid
    return mapping

def build_val_lists(root: Path, val_dir: Path) -> Tuple[List[Path], List[str]]:
    images, labels = [], []
    if not val_dir.exists(): return images, labels
    if has_subdirs(val_dir):
        class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
        for cdir in tqdm(class_dirs, desc="Indexing val classes", unit="cls", dynamic_ncols=True):
            wnid = cdir.name
            for p in cdir.iterdir():
                if p.is_file() and p.suffix in IMG_EXTS:
                    images.append(p); labels.append(wnid)
        return images, labels
    # flat val/: rely on CSV
    val_map = read_val_labels_from_csv(root)
    files = sorted([x for x in val_dir.iterdir() if x.is_file() and x.suffix in IMG_EXTS])
    for p in tqdm(files, desc="Indexing val files", unit="img", dynamic_ncols=True):
        wnid = val_map.get(p.name)
        if wnid is not None:
            images.append(p); labels.append(wnid)
    return images, labels

def build_test_list(test_dir: Path) -> List[Path]:
    if not test_dir.exists(): return []
    return list(tqdm(sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix in IMG_EXTS]),
                     desc="Indexing test files", unit="img", dynamic_ncols=True))

def make_label_indices(train_wnids: List[str], val_wnids: List[str]) -> Tuple[Dict[str,int], List[str]]:
    all_wnids = sorted(set(train_wnids) | set(val_wnids))
    return {w: i for i, w in enumerate(all_wnids)}, all_wnids

def write_split(h5f, group_name: str, img_paths: List[Path], labels_idx: Optional[List[int]]):
    # create group & datasets
    if group_name in h5f:
        raise RuntimeError(f"Group '{group_name}' already exists. Use --overwrite to replace it.")
    grp = h5f.create_group(group_name)
    n = len(img_paths)
    if n == 0:
        return
    imgs_ds = grp.create_dataset(
        "images", shape=(n, 3, IMG_SIZE, IMG_SIZE), dtype="float32",
        chunks=(1, 3, IMG_SIZE, IMG_SIZE), compression="gzip", compression_opts=4, shuffle=True
    )
    labs_ds = None
    if labels_idx is not None:
        labs_ds = grp.create_dataset("labels", shape=(n,), dtype="int64",
                                     chunks=(1024,), compression="gzip", compression_opts=4, shuffle=True)

    num_errors = 0
    with tqdm(total=n, desc=f"Writing {group_name}", unit="img", dynamic_ncols=True) as bar:
        for i, p in enumerate(img_paths):
            try:
                arr = load_and_preprocess(p)
                imgs_ds[i] = arr
                if labs_ds is not None:
                    labs_ds[i] = labels_idx[i]
            except Exception as e:
                num_errors += 1
                imgs_ds[i] = 0.0
                if labs_ds is not None: labs_ds[i] = -1
                tqdm.write(f"[WARN] Failed to process {p}: {e}")
            if i % 50 == 0: bar.set_postfix(errors=num_errors)
            bar.update(1)
    if num_errors:
        grp.attrs["num_decode_errors"] = num_errors

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Convert Kaggle ImageNet to HDF5 (224x224 normalized).")
    ap.add_argument("--root", type=Path, required=True,
                    help="Root folder with train/, val/, (optional) test/ and possibly LOC_* files.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HDF5 path. If exists, will append specified splits; if missing, will be created.")
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"],
                    help="Which splits to process.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing split groups in the HDF5 if present.")
    return ap.parse_args()

def main():
    args = parse_args()
    root: Path = args.root
    out_path: Path = args.out if args.out is not None else (root / "imagenet224.h5")
    splits = set(args.splits)

    train_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "train"
    val_dir   = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    test_dir  = root / "ILSVRC" / "Data" / "CLS-LOC" / "test"

    # Build file lists (only for requested splits)
    train_imgs = train_wnids = val_imgs = val_wnids = test_imgs = None

    if "train" in splits:
        train_imgs, train_wnids = build_train_lists(train_dir)
    else:
        train_imgs, train_wnids = [], []

    if "val" in splits:
        val_imgs, val_wnids = build_val_lists(root, val_dir)
    else:
        val_imgs, val_wnids = [], []

    if "test" in splits:
        test_imgs = build_test_list(test_dir)
    else:
        test_imgs = []

    # If neither train nor val requested, label map may be empty, but that's fine for test-only.
    wnid_to_idx, ordered_wnids = make_label_indices(train_wnids, val_wnids)
    train_label_idx = [wnid_to_idx[w] for w in train_wnids] if train_wnids else None
    val_label_idx   = [wnid_to_idx[w] for w in val_wnids]   if val_wnids   else None

    wnid_to_name = read_synset_mapping(root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    with h5py.File(out_path, mode) as h5f:
        # Write / update global metadata only on create
        if "preprocess" not in h5f.attrs:
            h5f.attrs["preprocess"] = json.dumps({
                "resize_shorter_side": RESIZE_MIN,
                "center_crop": IMG_SIZE,
                "channel_order": "NCHW",
                "scale": "[0,1] via /255.0",
                "normalize_mean": IMAGENET_MEAN.tolist(),
                "normalize_std": IMAGENET_STD.tolist(),
            })
        if "class_wnids" not in h5f and ordered_wnids:
            h5f.create_dataset("class_wnids", data=np.array(ordered_wnids, dtype="S16"))
        if wnid_to_name and "class_names" not in h5f and ordered_wnids:
            names = [wnid_to_name.get(w, w) for w in ordered_wnids]
            maxlen = max(len(n) for n in names) if names else 1
            h5f.create_dataset("class_names",
                               data=np.array([n.encode("utf-8") for n in names]),
                               dtype=f"S{maxlen}")

        # Overwrite handling
        for split in ["train", "val", "test"]:
            if split in splits and split in h5f:
                if args.overwrite:
                    del h5f[split]
                else:
                    tqdm.write(f"[INFO] Split '{split}' already exists in HDF5. Skipping (use --overwrite to replace).")

        # Write requested splits
        if "train" in splits and train_imgs:
            write_split(h5f, "train", train_imgs, train_label_idx)
        if "val" in splits and val_imgs:
            write_split(h5f, "val",   val_imgs,   val_label_idx)
        if "test" in splits and test_imgs:
            write_split(h5f, "test",  test_imgs,  labels_idx=None)

    tqdm.write(f"Done. HDF5 at: {out_path}")
    tqdm.write(f"Counts — train: {len(train_imgs)}  val: {len(val_imgs)}  test: {len(test_imgs)}")

if __name__ == "__main__":
    main()
