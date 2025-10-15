import h5py
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        datasets=["train/data", "train/labels"],
        dtypes=[torch.float32, torch.long],
    ):
        assert len(datasets) > 0, "At least one dataset must be passed."
        assert len(datasets) == len(
            dtypes
        ), "Datasets and dtypes must have the same length."

        self.hdf5_file = hdf5_file
        self.datasets = datasets
        self.dtypes = dtypes

        self.hdf = None  # Will be initialized in __getitem__
        self.data = {dataset: None for dataset in self.datasets}

        # Store the length without keeping the file open
        with h5py.File(self.hdf5_file, "r") as f:
            self.len = len(f[self.datasets[0]])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.hdf is None:
            # Open the HDF5 file when needed (in the worker process, so each process has it's own file handler)
            self.hdf = h5py.File(self.hdf5_file, "r", swmr=True)

            for dataset in self.datasets:
                self.data[dataset] = self.hdf[dataset]

        data = tuple(
            torch.tensor(self.data[dataset][idx], dtype=dtype)
            for dataset, dtype in zip(self.datasets, self.dtypes)
        )

        return data

    @staticmethod
    def get_dataloader(
        hdf5_file,
        datasets,
        dtypes,
        data_frac=1.0,
        batch_size=128,
        num_workers=4,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        seed=0,
        collate_fn=None,
        subset_indices=None,  # e.g., a list/np.array/torch tensor of indices
        subset_indices_h5_path=None,  # e.g., "c/blur/defocus_blur/5/subset_indices_10000"
        **kwargs,
    ):
        # Create dataset
        dataset = HDF5Dataset(
            hdf5_file=hdf5_file,
            datasets=datasets,
            dtypes=dtypes,
        )

        if subset_indices is None and subset_indices_h5_path is not None:
            with h5py.File(hdf5_file, "r") as f:
                if subset_indices_h5_path not in f:
                    raise KeyError(
                        f"Indices dataset not found: {subset_indices_h5_path}"
                    )
                subset_indices = f[subset_indices_h5_path][:]

        if subset_indices is not None:
            subset_indices = np.asarray(subset_indices, dtype=np.int64).tolist()
            dataset = Subset(dataset, subset_indices)

        elif data_frac < 1:
            dataset_len = len(dataset)
            selected_len = int(data_frac * dataset_len)
            indices = list(range(dataset_len))
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(indices)
            dataset = Subset(dataset, indices[:selected_len])

        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            )
        else:
            if shuffle:
                generator = torch.Generator().manual_seed(seed)
                sampler = RandomSampler(dataset, replacement=False, generator=generator)
            else:
                sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # sampler controls shuffling
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )
        return loader


if __name__ == "__main__":

    # Create a DataLoader
    dataloader = HDF5Dataset.get_dataloader(
        hdf5_file="/scratch/turirezende/tta_registers/data/imagenet/imagenet.h5",
        datasets=["c/blur/defocus_blur/5/images", "c/blur/defocus_blur/5/labels"],
        dtypes=[torch.float32, torch.long],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        seed=42,
        rank=0,
        world_size=1,
        subset_indices_h5_path="c/blur/defocus_blur/5/subset_indices_10000",
    )

    # Iterate through the DataLoader
    for batch_idx, batch in tqdm(
        enumerate(dataloader), desc="BATCH COUNT", total=len(dataloader)
    ):

        data, labels = batch
        print(
            f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {labels.shape}"
        )

        if batch_idx == 5:
            break
