import h5py
import numpy as np
import random
from pathlib import Path
from PIL import Image

# ==== CONFIG ====
H5_PATH = Path("./imagenet.h5")   # path to your HDF5 file
SPLIT   = "val"                 # 'train', 'val', or 'test'
N_SAMPLES = 8                    # number of random images to save
SAVE_DIR = Path("./output_images") # folder to save the images

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==== Load file ====
with h5py.File(H5_PATH, "r") as h5f:
    # --- Load preprocessing info ---
    if "preprocess" in h5f.attrs:
        import json
        preprocess_info = json.loads(h5f.attrs["preprocess"])
        mean = np.array(preprocess_info.get("normalize_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(preprocess_info.get("normalize_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # --- Detect dataset paths ---
    # Works for both group and flat naming
    img_path = f"{SPLIT}/images"
    label_path = f"{SPLIT}/labels"
    has_labels = label_path in h5f

    if img_path not in h5f:
        raise KeyError(f"Could not find dataset '{img_path}' in {H5_PATH}")

    imgs = h5f[img_path]
    labels = h5f[label_path][:] if has_labels else None

    wnids = [w.decode("utf-8") for w in h5f["class_wnids"][:]] if "class_wnids" in h5f else None
    class_names = [n.decode("utf-8") for n in h5f["class_names"][:]] if "class_names" in h5f else None

    # --- Pick random indices ---
    idxs = random.sample(range(len(imgs)), min(N_SAMPLES, len(imgs)))

    for idx in idxs:
        img = imgs[idx]  # C,H,W normalized float32

        # unnormalize to [0,1]
        img = (img.transpose(1, 2, 0) * std) + mean
        img = np.clip(img, 0, 1)

        # convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        if has_labels:
            lbl_idx = int(labels[idx])
            wnid = wnids[lbl_idx] if wnids else str(lbl_idx)
            label_str = wnid
            if class_names:
                label_str += f"_{class_names[lbl_idx].replace(' ', '_')}"
        else:
            label_str = "no_label"

        filename = SAVE_DIR / f"{SPLIT}_{idx}_{label_str}.png"
        pil_img.save(filename)
        print(f"Saved {filename}")
