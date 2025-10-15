#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm

# ---- Same defaults as your builder ----
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224
RESIZE_MIN = 256
IMG_EXTS = {".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".bmp"}


def center_crop_224(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    scale = RESIZE_MIN / min(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    return img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))


def load_and_preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = center_crop_224(im)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))  # C,H,W float32


def list_imagenet_r(r_dir: Path) -> Tuple[List[Path], List[str]]:
    """Return (image_paths, wnids) for imagenet-r directory structure."""
    if not r_dir.exists():
        raise FileNotFoundError(f"imagenet-r directory not found at: {r_dir}")
    img_paths: List[Path] = []
    wnids: List[str] = []
    class_dirs = sorted([d for d in r_dir.iterdir() if d.is_dir()])
    for cdir in tqdm(
        class_dirs, desc="Indexing ImageNet-R classes", unit="cls", dynamic_ncols=True
    ):
        wnid = cdir.name
        for p in cdir.iterdir():
            if p.is_file() and p.suffix in IMG_EXTS:
                img_paths.append(p)
                wnids.append(wnid)
    return img_paths, wnids


def write_split(h5f, group_name: str, img_paths: List[Path], labels_idx: List[int]):
    if group_name in h5f:
        raise RuntimeError(
            f"Group '{group_name}' already exists. Use --overwrite to replace it."
        )
    n = len(img_paths)
    grp = h5f.create_group(group_name)
    if n == 0:
        return
    imgs_ds = grp.create_dataset(
        "images",
        shape=(n, 3, IMG_SIZE, IMG_SIZE),
        dtype="float32",
        chunks=(1, 3, IMG_SIZE, IMG_SIZE),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    labs_ds = grp.create_dataset(
        "labels",
        shape=(n,),
        dtype="int64",
        chunks=(1024,),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

    num_errors = 0
    with tqdm(
        total=n, desc=f"Writing {group_name}", unit="img", dynamic_ncols=True
    ) as bar:
        for i, p in enumerate(img_paths):
            try:
                arr = load_and_preprocess(p)
                imgs_ds[i] = arr
                labs_ds[i] = labels_idx[i]
            except Exception as e:
                num_errors += 1
                imgs_ds[i] = 0.0
                labs_ds[i] = -1
                tqdm.write(f"[WARN] Failed to process {p}: {e}")
            if i % 50 == 0:
                bar.set_postfix(errors=num_errors)
            bar.update(1)
    if num_errors:
        grp.attrs["num_decode_errors"] = num_errors


def main():
    ap = argparse.ArgumentParser(
        description="Append ImageNet-R to existing ImageNet HDF5 as group 'r/'."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder that contains ILSVRC/ and imagenet-r/ as siblings (e.g., /data/imagenet).",
    )
    ap.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to existing ImageNet HDF5 (with /class_wnids).",
    )
    ap.add_argument(
        "--rdir",
        type=Path,
        default=None,
        help="Path to the imagenet-r directory (defaults to <root>/imagenet-r).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, deletes existing 'r' group before writing.",
    )
    args = ap.parse_args()

    r_dir = args.rdir if args.rdir is not None else (args.root / "imagenet-r")
    img_paths, wnids = list_imagenet_r(r_dir)

    # Open H5 and read the canonical mapping from /class_wnids
    mode = "a" if args.h5.exists() else "w"
    with h5py.File(args.h5, mode) as h5f:
        if "class_wnids" not in h5f:
            raise RuntimeError(
                "HDF5 is missing '/class_wnids'. Build train/val first so labels align."
            )
        # Decode bytes to str (dtype is usually 'S16')
        class_wnids = [
            w.decode("utf-8") if hasattr(w, "decode") else str(w)
            for w in h5f["class_wnids"][()]
        ]
        wnid_to_idx: Dict[str, int] = {w: i for i, w in enumerate(class_wnids)}

        # Map ImageNet-R wnids to the same indices; warn & drop unknown wnids
        mapped_img_paths: List[Path] = []
        mapped_labels: List[int] = []
        unknown_count = 0
        for p, w in zip(img_paths, wnids):
            idx = wnid_to_idx.get(w)
            if idx is None:
                unknown_count += 1
                continue
            mapped_img_paths.append(p)
            mapped_labels.append(idx)

        if unknown_count:
            tqdm.write(
                f"[INFO] Skipped {unknown_count} images whose WNID is not in /class_wnids."
            )

        if "r" in h5f:
            if args.overwrite:
                del h5f["r"]
            else:
                print("[INFO] Group 'r' already exists. Use --overwrite to replace it.")
                return

        write_split(h5f, "r", mapped_img_paths, mapped_labels)

        print(f"Done. Appended ImageNet-R as 'r/' → {len(mapped_img_paths)} images.")


if __name__ == "__main__":
    main()
