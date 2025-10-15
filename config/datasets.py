import torch
import os

base_config = {
    "hdf5_file": os.getenv("IMAGENET_HDF5"),
    "dtypes": [torch.float32, torch.long],
}

dataset_configs = {
    # Clean datasets
    "imagenet/val": {**base_config, "datasets": ["val/images", "val/labels"]},
    "imagenet/r": {**base_config, "datasets": ["r/images", "r/labels"]},
    # ---------------------- ImageNet-C (severity 5) FULL ---------------------- #
    # ---- Blur ----
    "imagenet/c/defocus_blur": {
        **base_config,
        "datasets": ["c/blur/defocus_blur/5/images", "c/blur/defocus_blur/5/labels"],
    },
    "imagenet/c/glass_blur": {
        **base_config,
        "datasets": ["c/blur/glass_blur/5/images", "c/blur/glass_blur/5/labels"],
    },
    "imagenet/c/motion_blur": {
        **base_config,
        "datasets": ["c/blur/motion_blur/5/images", "c/blur/motion_blur/5/labels"],
    },
    "imagenet/c/zoom_blur": {
        **base_config,
        "datasets": ["c/blur/zoom_blur/5/images", "c/blur/zoom_blur/5/labels"],
    },
    # ---- Digital ----
    "imagenet/c/contrast": {
        **base_config,
        "datasets": ["c/digital/contrast/5/images", "c/digital/contrast/5/labels"],
    },
    "imagenet/c/elastic_transform": {
        **base_config,
        "datasets": [
            "c/digital/elastic_transform/5/images",
            "c/digital/elastic_transform/5/labels",
        ],
    },
    "imagenet/c/jpeg_compression": {
        **base_config,
        "datasets": [
            "c/digital/jpeg_compression/5/images",
            "c/digital/jpeg_compression/5/labels",
        ],
    },
    "imagenet/c/pixelate": {
        **base_config,
        "datasets": ["c/digital/pixelate/5/images", "c/digital/pixelate/5/labels"],
    },
    # ---- Extra ----
    "imagenet/c/gaussian_blur": {
        **base_config,
        "datasets": [
            "c/extra/gaussian_blur/5/images",
            "c/extra/gaussian_blur/5/labels",
        ],
    },
    "imagenet/c/saturate": {
        **base_config,
        "datasets": ["c/extra/saturate/5/images", "c/extra/saturate/5/labels"],
    },
    "imagenet/c/spatter": {
        **base_config,
        "datasets": ["c/extra/spatter/5/images", "c/extra/spatter/5/labels"],
    },
    "imagenet/c/speckle_noise": {
        **base_config,
        "datasets": [
            "c/extra/speckle_noise/5/images",
            "c/extra/speckle_noise/5/labels",
        ],
    },
    # ---- Noise ----
    "imagenet/c/gaussian_noise": {
        **base_config,
        "datasets": [
            "c/noise/gaussian_noise/5/images",
            "c/noise/gaussian_noise/5/labels",
        ],
    },
    "imagenet/c/impulse_noise": {
        **base_config,
        "datasets": [
            "c/noise/impulse_noise/5/images",
            "c/noise/impulse_noise/5/labels",
        ],
    },
    "imagenet/c/shot_noise": {
        **base_config,
        "datasets": ["c/noise/shot_noise/5/images", "c/noise/shot_noise/5/labels"],
    },
    # ---- Weather ----
    "imagenet/c/brightness": {
        **base_config,
        "datasets": ["c/weather/brightness/5/images", "c/weather/brightness/5/labels"],
    },
    "imagenet/c/fog": {
        **base_config,
        "datasets": ["c/weather/fog/5/images", "c/weather/fog/5/labels"],
    },
    "imagenet/c/frost": {
        **base_config,
        "datasets": ["c/weather/frost/5/images", "c/weather/frost/5/labels"],
    },
    "imagenet/c/snow": {
        **base_config,
        "datasets": ["c/weather/snow/5/images", "c/weather/snow/5/labels"],
    },
    # ---------------------- ImageNet-C SUBSETS ---------------------- #
    # ---- Blur ----
    "imagenet/c/defocus_blur@10k": {
        **base_config,
        "datasets": ["c/blur/defocus_blur/5/images", "c/blur/defocus_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/defocus_blur/5/subset_indices_10000",
    },
    "imagenet/c/defocus_blur@5k": {
        **base_config,
        "datasets": ["c/blur/defocus_blur/5/images", "c/blur/defocus_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/defocus_blur/5/subset_indices_5000",
    },
    "imagenet/c/defocus_blur@1k": {
        **base_config,
        "datasets": ["c/blur/defocus_blur/5/images", "c/blur/defocus_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/defocus_blur/5/subset_indices_1000",
    },
    "imagenet/c/glass_blur@10k": {
        **base_config,
        "datasets": ["c/blur/glass_blur/5/images", "c/blur/glass_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/glass_blur/5/subset_indices_10000",
    },
    "imagenet/c/glass_blur@5k": {
        **base_config,
        "datasets": ["c/blur/glass_blur/5/images", "c/blur/glass_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/glass_blur/5/subset_indices_5000",
    },
    "imagenet/c/glass_blur@1k": {
        **base_config,
        "datasets": ["c/blur/glass_blur/5/images", "c/blur/glass_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/glass_blur/5/subset_indices_1000",
    },
    "imagenet/c/motion_blur@10k": {
        **base_config,
        "datasets": ["c/blur/motion_blur/5/images", "c/blur/motion_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/motion_blur/5/subset_indices_10000",
    },
    "imagenet/c/motion_blur@5k": {
        **base_config,
        "datasets": ["c/blur/motion_blur/5/images", "c/blur/motion_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/motion_blur/5/subset_indices_5000",
    },
    "imagenet/c/motion_blur@1k": {
        **base_config,
        "datasets": ["c/blur/motion_blur/5/images", "c/blur/motion_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/motion_blur/5/subset_indices_1000",
    },
    "imagenet/c/zoom_blur@10k": {
        **base_config,
        "datasets": ["c/blur/zoom_blur/5/images", "c/blur/zoom_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/zoom_blur/5/subset_indices_10000",
    },
    "imagenet/c/zoom_blur@5k": {
        **base_config,
        "datasets": ["c/blur/zoom_blur/5/images", "c/blur/zoom_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/zoom_blur/5/subset_indices_5000",
    },
    "imagenet/c/zoom_blur@1k": {
        **base_config,
        "datasets": ["c/blur/zoom_blur/5/images", "c/blur/zoom_blur/5/labels"],
        "subset_indices_h5_path": "c/blur/zoom_blur/5/subset_indices_1000",
    },
    # ---- Digital ----
    "imagenet/c/contrast@10k": {
        **base_config,
        "datasets": ["c/digital/contrast/5/images", "c/digital/contrast/5/labels"],
        "subset_indices_h5_path": "c/digital/contrast/5/subset_indices_10000",
    },
    "imagenet/c/contrast@5k": {
        **base_config,
        "datasets": ["c/digital/contrast/5/images", "c/digital/contrast/5/labels"],
        "subset_indices_h5_path": "c/digital/contrast/5/subset_indices_5000",
    },
    "imagenet/c/contrast@1k": {
        **base_config,
        "datasets": ["c/digital/contrast/5/images", "c/digital/contrast/5/labels"],
        "subset_indices_h5_path": "c/digital/contrast/5/subset_indices_1000",
    },
    "imagenet/c/elastic_transform@10k": {
        **base_config,
        "datasets": [
            "c/digital/elastic_transform/5/images",
            "c/digital/elastic_transform/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/elastic_transform/5/subset_indices_10000",
    },
    "imagenet/c/elastic_transform@5k": {
        **base_config,
        "datasets": [
            "c/digital/elastic_transform/5/images",
            "c/digital/elastic_transform/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/elastic_transform/5/subset_indices_5000",
    },
    "imagenet/c/elastic_transform@1k": {
        **base_config,
        "datasets": [
            "c/digital/elastic_transform/5/images",
            "c/digital/elastic_transform/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/elastic_transform/5/subset_indices_1000",
    },
    "imagenet/c/jpeg_compression@10k": {
        **base_config,
        "datasets": [
            "c/digital/jpeg_compression/5/images",
            "c/digital/jpeg_compression/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/jpeg_compression/5/subset_indices_10000",
    },
    "imagenet/c/jpeg_compression@5k": {
        **base_config,
        "datasets": [
            "c/digital/jpeg_compression/5/images",
            "c/digital/jpeg_compression/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/jpeg_compression/5/subset_indices_5000",
    },
    "imagenet/c/jpeg_compression@1k": {
        **base_config,
        "datasets": [
            "c/digital/jpeg_compression/5/images",
            "c/digital/jpeg_compression/5/labels",
        ],
        "subset_indices_h5_path": "c/digital/jpeg_compression/5/subset_indices_1000",
    },
    "imagenet/c/pixelate@10k": {
        **base_config,
        "datasets": ["c/digital/pixelate/5/images", "c/digital/pixelate/5/labels"],
        "subset_indices_h5_path": "c/digital/pixelate/5/subset_indices_10000",
    },
    "imagenet/c/pixelate@5k": {
        **base_config,
        "datasets": ["c/digital/pixelate/5/images", "c/digital/pixelate/5/labels"],
        "subset_indices_h5_path": "c/digital/pixelate/5/subset_indices_5000",
    },
    "imagenet/c/pixelate@1k": {
        **base_config,
        "datasets": ["c/digital/pixelate/5/images", "c/digital/pixelate/5/labels"],
        "subset_indices_h5_path": "c/digital/pixelate/5/subset_indices_1000",
    },
    # ---- Extra ----
    "imagenet/c/gaussian_blur@10k": {
        **base_config,
        "datasets": [
            "c/extra/gaussian_blur/5/images",
            "c/extra/gaussian_blur/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/gaussian_blur/5/subset_indices_10000",
    },
    "imagenet/c/gaussian_blur@5k": {
        **base_config,
        "datasets": [
            "c/extra/gaussian_blur/5/images",
            "c/extra/gaussian_blur/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/gaussian_blur/5/subset_indices_5000",
    },
    "imagenet/c/gaussian_blur@1k": {
        **base_config,
        "datasets": [
            "c/extra/gaussian_blur/5/images",
            "c/extra/gaussian_blur/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/gaussian_blur/5/subset_indices_1000",
    },
    "imagenet/c/saturate@10k": {
        **base_config,
        "datasets": ["c/extra/saturate/5/images", "c/extra/saturate/5/labels"],
        "subset_indices_h5_path": "c/extra/saturate/5/subset_indices_10000",
    },
    "imagenet/c/saturate@5k": {
        **base_config,
        "datasets": ["c/extra/saturate/5/images", "c/extra/saturate/5/labels"],
        "subset_indices_h5_path": "c/extra/saturate/5/subset_indices_5000",
    },
    "imagenet/c/saturate@1k": {
        **base_config,
        "datasets": ["c/extra/saturate/5/images", "c/extra/saturate/5/labels"],
        "subset_indices_h5_path": "c/extra/saturate/5/subset_indices_1000",
    },
    "imagenet/c/spatter@10k": {
        **base_config,
        "datasets": ["c/extra/spatter/5/images", "c/extra/spatter/5/labels"],
        "subset_indices_h5_path": "c/extra/spatter/5/subset_indices_10000",
    },
    "imagenet/c/spatter@5k": {
        **base_config,
        "datasets": ["c/extra/spatter/5/images", "c/extra/spatter/5/labels"],
        "subset_indices_h5_path": "c/extra/spatter/5/subset_indices_5000",
    },
    "imagenet/c/spatter@1k": {
        **base_config,
        "datasets": ["c/extra/spatter/5/images", "c/extra/spatter/5/labels"],
        "subset_indices_h5_path": "c/extra/spatter/5/subset_indices_1000",
    },
    "imagenet/c/speckle_noise@10k": {
        **base_config,
        "datasets": [
            "c/extra/speckle_noise/5/images",
            "c/extra/speckle_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/speckle_noise/5/subset_indices_10000",
    },
    "imagenet/c/speckle_noise@5k": {
        **base_config,
        "datasets": [
            "c/extra/speckle_noise/5/images",
            "c/extra/speckle_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/speckle_noise/5/subset_indices_5000",
    },
    "imagenet/c/speckle_noise@1k": {
        **base_config,
        "datasets": [
            "c/extra/speckle_noise/5/images",
            "c/extra/speckle_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/extra/speckle_noise/5/subset_indices_1000",
    },
    # ---- Noise ----
    "imagenet/c/gaussian_noise@10k": {
        **base_config,
        "datasets": [
            "c/noise/gaussian_noise/5/images",
            "c/noise/gaussian_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/gaussian_noise/5/subset_indices_10000",
    },
    "imagenet/c/gaussian_noise@5k": {
        **base_config,
        "datasets": [
            "c/noise/gaussian_noise/5/images",
            "c/noise/gaussian_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/gaussian_noise/5/subset_indices_5000",
    },
    "imagenet/c/gaussian_noise@1k": {
        **base_config,
        "datasets": [
            "c/noise/gaussian_noise/5/images",
            "c/noise/gaussian_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/gaussian_noise/5/subset_indices_1000",
    },
    "imagenet/c/impulse_noise@10k": {
        **base_config,
        "datasets": [
            "c/noise/impulse_noise/5/images",
            "c/noise/impulse_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/impulse_noise/5/subset_indices_10000",
    },
    "imagenet/c/impulse_noise@5k": {
        **base_config,
        "datasets": [
            "c/noise/impulse_noise/5/images",
            "c/noise/impulse_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/impulse_noise/5/subset_indices_5000",
    },
    "imagenet/c/impulse_noise@1k": {
        **base_config,
        "datasets": [
            "c/noise/impulse_noise/5/images",
            "c/noise/impulse_noise/5/labels",
        ],
        "subset_indices_h5_path": "c/noise/impulse_noise/5/subset_indices_1000",
    },
    "imagenet/c/shot_noise@10k": {
        **base_config,
        "datasets": ["c/noise/shot_noise/5/images", "c/noise/shot_noise/5/labels"],
        "subset_indices_h5_path": "c/noise/shot_noise/5/subset_indices_10000",
    },
    "imagenet/c/shot_noise@5k": {
        **base_config,
        "datasets": ["c/noise/shot_noise/5/images", "c/noise/shot_noise/5/labels"],
        "subset_indices_h5_path": "c/noise/shot_noise/5/subset_indices_5000",
    },
    "imagenet/c/shot_noise@1k": {
        **base_config,
        "datasets": ["c/noise/shot_noise/5/images", "c/noise/shot_noise/5/labels"],
        "subset_indices_h5_path": "c/noise/shot_noise/5/subset_indices_1000",
    },
    # ---- Weather ----
    "imagenet/c/brightness@10k": {
        **base_config,
        "datasets": ["c/weather/brightness/5/images", "c/weather/brightness/5/labels"],
        "subset_indices_h5_path": "c/weather/brightness/5/subset_indices_10000",
    },
    "imagenet/c/brightness@5k": {
        **base_config,
        "datasets": ["c/weather/brightness/5/images", "c/weather/brightness/5/labels"],
        "subset_indices_h5_path": "c/weather/brightness/5/subset_indices_5000",
    },
    "imagenet/c/brightness@1k": {
        **base_config,
        "datasets": ["c/weather/brightness/5/images", "c/weather/brightness/5/labels"],
        "subset_indices_h5_path": "c/weather/brightness/5/subset_indices_1000",
    },
    "imagenet/c/fog@10k": {
        **base_config,
        "datasets": ["c/weather/fog/5/images", "c/weather/fog/5/labels"],
        "subset_indices_h5_path": "c/weather/fog/5/subset_indices_10000",
    },
    "imagenet/c/fog@5k": {
        **base_config,
        "datasets": ["c/weather/fog/5/images", "c/weather/fog/5/labels"],
        "subset_indices_h5_path": "c/weather/fog/5/subset_indices_5000",
    },
    "imagenet/c/fog@1k": {
        **base_config,
        "datasets": ["c/weather/fog/5/images", "c/weather/fog/5/labels"],
        "subset_indices_h5_path": "c/weather/fog/5/subset_indices_1000",
    },
    "imagenet/c/frost@10k": {
        **base_config,
        "datasets": ["c/weather/frost/5/images", "c/weather/frost/5/labels"],
        "subset_indices_h5_path": "c/weather/frost/5/subset_indices_10000",
    },
    "imagenet/c/frost@5k": {
        **base_config,
        "datasets": ["c/weather/frost/5/images", "c/weather/frost/5/labels"],
        "subset_indices_h5_path": "c/weather/frost/5/subset_indices_5000",
    },
    "imagenet/c/frost@1k": {
        **base_config,
        "datasets": ["c/weather/frost/5/images", "c/weather/frost/5/labels"],
        "subset_indices_h5_path": "c/weather/frost/5/subset_indices_1000",
    },
    "imagenet/c/snow@10k": {
        **base_config,
        "datasets": ["c/weather/snow/5/images", "c/weather/snow/5/labels"],
        "subset_indices_h5_path": "c/weather/snow/5/subset_indices_10000",
    },
    "imagenet/c/snow@5k": {
        **base_config,
        "datasets": ["c/weather/snow/5/images", "c/weather/snow/5/labels"],
        "subset_indices_h5_path": "c/weather/snow/5/subset_indices_5000",
    },
    "imagenet/c/snow@1k": {
        **base_config,
        "datasets": ["c/weather/snow/5/images", "c/weather/snow/5/labels"],
        "subset_indices_h5_path": "c/weather/snow/5/subset_indices_1000",
    },
}
