#!/usr/bin/env python3
"""
Create class-balanced 10k subsets (10 per class for 1000 classes)
for every ImageNet-C corruption at severity=5 in an HDF5 laid out as:

c/<family>/<corruption>/5/{images,labels}

The script:
  - finds all c/*/*/5 groups automatically
  - samples per-class indices with a fixed seed
  - saves indices as an HDF5 dataset "subset10k_indices" under each .../5 group
  - also writes a JSON mapping {group_path: [indices]} for reproducibility

No images are copied; only index lists are stored.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime


def find_c_sev5_groups(h5):
    """Return list of group paths like 'c/blur/defocus_blur/5' that contain images+labels."""
    groups = []
    if "c" not in h5:
        return groups
    for fam in h5["c"].keys():  # e.g., blur, digital, noise, weather, extra
        fam_grp = h5["c"][fam]
        for corr in fam_grp.keys():  # e.g., defocus_blur, fog, etc.
            corr_grp = fam_grp[corr]
            if "5" in corr_grp:
                sev_grp = corr_grp["5"]
                if "images" in sev_grp and "labels" in sev_grp:
                    groups.append(f"c/{fam}/{corr}/5")
    return sorted(groups)


def create_or_replace_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    return group.create_dataset(name, data=data, **kwargs)


def sample_balanced_indices(labels, n_classes, per_class, seed):
    """
    labels: 1D numpy array of int labels
    n_classes: number of classes (e.g., 1000)
    per_class: how many items per class to sample (e.g., 10)
    seed: int
    """
    rng = np.random.default_rng(seed)
    picks = []
    for cls in range(n_classes):
        cls_idx = np.flatnonzero(labels == cls)
        if cls_idx.size < per_class:
            raise ValueError(
                f"Class {cls} has only {cls_idx.size} samples, cannot pick {per_class}."
            )
        sel = rng.choice(cls_idx, size=per_class, replace=False)
        picks.append(sel)
    idx = np.sort(np.concatenate(picks))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path", type=Path, help="Path to imagenet.h5")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    ap.add_argument(
        "--subset_size",
        type=int,
        default=10_000,
        help="Total subset size per corruption (default 10k)",
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help='Name of dataset to create under each ".../5" group (default: subset_indices_<size>)',
    )
    ap.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to write a JSON with the selected indices",
    )
    args = ap.parse_args()

    with h5py.File(args.h5_path, "r+") as h5:
        # Infer number of classes from class_names or labels dtype range
        if "class_names" in h5:
            n_classes = int(h5["class_names"].shape[0])
        elif "val" in h5 and "labels" in h5["val"]:
            n_classes = int(np.max(h5["val/labels"][:]) + 1)
        else:
            raise RuntimeError(
                "Could not infer number of classes; please ensure 'class_names' or 'val/labels' exist."
            )

        if args.subset_size % n_classes != 0:
            raise ValueError(
                f"subset_size ({args.subset_size}) must be divisible by n_classes ({n_classes})."
            )
        per_class = args.subset_size // n_classes

        sev5_groups = find_c_sev5_groups(h5)
        if not sev5_groups:
            raise RuntimeError(
                "No ImageNet-C severity=5 groups found under 'c/'. Check your HDF5 layout."
            )

        ds_name = args.dataset_name or f"subset_indices_{args.subset_size}"

        summary = {}
        for gpath in sev5_groups:
            labels = h5[f"{gpath}/labels"][:]
            if labels.ndim != 1:
                raise ValueError(f"{gpath}/labels must be 1D; got shape {labels.shape}")

            indices = sample_balanced_indices(labels, n_classes, per_class, args.seed)

            # Optional verification: check per-class counts
            chosen_labels = labels[indices]
            counts = np.bincount(chosen_labels, minlength=n_classes)
            minc, maxc = counts.min(), counts.max()
            if not (minc == maxc == per_class):
                raise AssertionError(
                    f"{gpath}: per-class counts not uniform. min={minc}, max={maxc}, expected={per_class}"
                )

            # Write indices into the H5 under the same group
            g = h5[gpath]
            dset = create_or_replace_dataset(
                g,
                ds_name,
                indices.astype(np.int32),
                compression="gzip",
                shuffle=True,
                fletcher32=True,
            )
            # Annotate for reproducibility
            dset.attrs["seed"] = args.seed
            dset.attrs["per_class"] = per_class
            dset.attrs["subset_size"] = args.subset_size
            dset.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"

            summary[gpath] = indices.tolist()
            print(
                f"[OK] {gpath}: wrote {ds_name} with {len(indices)} indices "
                f"({per_class}/class over {n_classes} classes)."
            )

        # Optional JSON dump with all indices
        if args.json_out is not None:
            payload = {
                "h5_path": str(args.h5_path),
                "seed": args.seed,
                "subset_size": args.subset_size,
                "dataset_name": ds_name,
                "n_classes": n_classes,
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "groups": summary,
            }
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"[OK] Wrote JSON index file to: {args.json_out}")


if __name__ == "__main__":
    main()
