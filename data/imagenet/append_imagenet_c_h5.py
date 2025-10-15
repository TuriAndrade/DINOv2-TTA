#!/usr/bin/env python3
"""
Append ImageNet-C to an existing ImageNet HDF5 as:

c/<group>/<name>/<severity>/images
c/<group>/<name>/<severity>/labels

- Preserves preprocessing used in your HDF5.
- Maps labels via /class_wnids (must already exist).
- Skips missing corruptions gracefully.
- Overwrites individual corruption paths only if --overwrite is set.
"""

import os, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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

# ---- Corruption taxonomy you specified ----
C_GROUPS: Dict[str, List[str]] = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["frost", "snow", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra": ["speckle_noise", "spatter", "gaussian_blur", "saturate"],
}


# --------------------------------------------------------------------------- #
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


def find_severity_dir(corruption_dir: Path, severity: str) -> Optional[Path]:
    """
    Some trees might use '5', others '-5' (as the user example shows).
    Try both.
    """
    cand1 = corruption_dir / severity
    cand2 = corruption_dir / f"-{severity.lstrip('-')}"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def list_imagenet_c(
    c_root: Path, corruption_name: str, severity_str: str
) -> Tuple[List[Path], List[str]]:
    """
    Expected on disk:
       <c_root>/<corruption_name>/<severity>/nXXXXXXXX/<image files>
    Returns (image_paths, wnids).
    """
    corruption_dir = c_root / corruption_name
    sev_dir = find_severity_dir(corruption_dir, severity_str)
    if sev_dir is None:
        raise FileNotFoundError(
            f"Severity folder not found for {corruption_name}: tried '{corruption_dir / severity_str}' and '{corruption_dir / ('-'+severity_str)}'"
        )

    img_paths: List[Path] = []
    wnids: List[str] = []
    # each wnid subdir
    class_dirs = sorted([d for d in sev_dir.iterdir() if d.is_dir()])
    for cdir in tqdm(
        class_dirs,
        desc=f"Indexing {corruption_name}/{severity_str}",
        unit="cls",
        leave=False,
        dynamic_ncols=True,
    ):
        wnid = cdir.name
        for p in cdir.iterdir():
            if p.is_file() and p.suffix in IMG_EXTS:
                img_paths.append(p)
                wnids.append(wnid)
    return img_paths, wnids


def ensure_group(h5f: h5py.File, path: str) -> h5py.Group:
    """
    Create nested groups if needed. Returns the final group.
    """
    parts = [p for p in path.strip("/").split("/") if p]
    grp = h5f
    for p in parts:
        grp = grp.require_group(p)
    return grp


def write_images_and_labels(
    h5f: h5py.File,
    group_path: str,
    img_paths: List[Path],
    labels_idx: List[int],
    overwrite: bool = False,
):
    """
    Writes datasets 'images' and 'labels' under group_path.
    Skips if already present and overwrite=False.
    """
    grp = ensure_group(h5f, group_path)

    # If datasets exist
    if "images" in grp or "labels" in grp:
        if overwrite:
            if "images" in grp:
                del grp["images"]
            if "labels" in grp:
                del grp["labels"]
        else:
            tqdm.write(
                f"[SKIP] {group_path} already has datasets. Use --overwrite to replace."
            )
            return

    n = len(img_paths)
    if n == 0:
        tqdm.write(f"[WARN] No images for {group_path}. Skipping.")
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
        total=n, desc=f"Writing {group_path}", unit="img", dynamic_ncols=True
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
        grp.attrs["num_decode_errors"] = int(num_errors)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Append ImageNet-C to existing ImageNet HDF5 under group 'c/'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder that contains ILSVRC/, imagenet-r/, imagenet-c/ as siblings (e.g., /data/imagenet).",
    )
    ap.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to existing ImageNet HDF5 (must contain /class_wnids).",
    )
    ap.add_argument(
        "--cdir",
        type=Path,
        default=None,
        help="Path to the imagenet-c directory (defaults to <root>/imagenet-c).",
    )
    ap.add_argument(
        "--severity",
        type=str,
        default="5",
        help="Severity level present on disk; script will also try '-<severity>'.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing datasets for each corruption if they already exist.",
    )
    ap.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of corruption names to process (e.g., gaussian_noise fog pixelate).",
    )
    args = ap.parse_args()

    c_dir = args.cdir if args.cdir is not None else (args.root / "imagenet-c")
    if not c_dir.exists():
        raise FileNotFoundError(f"imagenet-c directory not found at: {c_dir}")

    # Open H5 & build WNID->index mapping
    mode = "a" if args.h5.exists() else "w"
    with h5py.File(args.h5, mode) as h5f:
        if "class_wnids" not in h5f:
            raise RuntimeError(
                "HDF5 is missing '/class_wnids'. Build train/val first so labels align."
            )

        class_wnids = [
            w.decode("utf-8") if hasattr(w, "decode") else str(w)
            for w in h5f["class_wnids"][()]
        ]
        wnid_to_idx: Dict[str, int] = {w: i for i, w in enumerate(class_wnids)}

        processed = 0
        missing = 0
        total_written = 0

        # Iterate groups/names in your requested taxonomy
        for group, names in C_GROUPS.items():
            for name in names:
                if args.only is not None and name not in set(args.only):
                    continue

                corruption_root = c_dir / name
                if not corruption_root.exists():
                    tqdm.write(f"[MISS] {name} not found under {c_dir}. Skipping.")
                    missing += 1
                    continue

                # Collect file paths and WNIDs
                try:
                    img_paths, wnids = list_imagenet_c(c_dir, name, args.severity)
                except FileNotFoundError as e:
                    tqdm.write(f"[MISS] {e}")
                    missing += 1
                    continue

                # Map to indices; drop unknown wnids
                mapped_paths: List[Path] = []
                mapped_labels: List[int] = []
                unknown_count = 0
                for p, w in zip(img_paths, wnids):
                    idx = wnid_to_idx.get(w)
                    if idx is None:
                        unknown_count += 1
                        continue
                    mapped_paths.append(p)
                    mapped_labels.append(idx)

                if unknown_count:
                    tqdm.write(
                        f"[INFO] {name}/{args.severity}: skipped {unknown_count} images with unknown WNIDs."
                    )

                # HDF5 path: c/<group>/<name>/<severity>
                group_path = f"c/{group}/{name}/{args.severity.lstrip('+').lstrip('-')}"
                before = len(mapped_paths)
                write_images_and_labels(
                    h5f,
                    group_path,
                    mapped_paths,
                    mapped_labels,
                    overwrite=args.overwrite,
                )
                processed += 1
                total_written += before

        print(
            f"Done. Processed corruptions: {processed}. Missing: {missing}. Images considered: {total_written}."
        )


if __name__ == "__main__":
    main()
