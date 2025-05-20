from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import torch
import numpy as np
from tqdm.auto import tqdm
import os
import pickle
import json
import hashlib
from collections import OrderedDict
import multiprocessing as mp
import atexit
from utils.dataloader_utils import passes_edge_checks, passes_bw_ratio, upscale_voxels, compute_sdf


H5_DATASET = None
FILTER_PARAMS = {}


def init_worker(h5_file_path, check_for_edges, edge_thickness, bw_ratio):
    """
    Initializer for each worker process.
    Opens the HDF5 file and sets global variables for filtering parameters.
    """
    global H5_DATASET
    global FILTER_PARAMS

    h5f = h5py.File(h5_file_path, "r")
    H5_DATASET = h5f["voxels"]

    FILTER_PARAMS = {
        "check_for_edges": check_for_edges,
        "edge_thickness": edge_thickness,
        "bw_ratio": bw_ratio,
    }

    atexit.register(lambda: h5f.close())


def worker(idx):
    """
    Worker function to process a single voxel index.
    Returns the index if it passes the filtering criteria; otherwise, returns None.
    """
    binary_voxel = H5_DATASET[idx]

    if FILTER_PARAMS["check_for_edges"]:
        if not passes_edge_checks(binary_voxel, FILTER_PARAMS["edge_thickness"]):
            return None

    if FILTER_PARAMS["bw_ratio"] > 0.0:
        if not passes_bw_ratio(binary_voxel, FILTER_PARAMS["bw_ratio"]):
            return None

    return idx


class VoxelDataset(Dataset):
    """
    Dataset class for loading voxel data from an HDF5 file.
    """

    def __init__(
        self,
        h5_file_path,
        voxel_size=(32, 64, 64),
        transform=None,
        small=False,
        patched=True,
        bw_ratio=0.0,
        check_for_edges=False,
        edge_thickness=2,
        cache_size=1000,
        mapping_cache_dir="dataset/.cache/",
        percentage=0.2,
        sdf=True,
        interpol=False,
        sdf_scale=5.0,
    ):
        """
        Initialize the VoxelDataset.
        """
        self.h5_file_path = h5_file_path
        self.voxel_size = voxel_size
        self.transform = transform
        self.patched = patched
        self.bw_ratio = bw_ratio
        self.check_for_edges = check_for_edges
        self.edge_thickness = edge_thickness
        self.cache_size = cache_size
        self.small = small
        self.percentage = percentage
        self.interpol = interpol
        self.target_size = (64, 64, 64) if interpol else voxel_size

        # Initialize variables
        self.axis_stats = None
        self.file_cache = OrderedDict()
        self.h5f = None
        self.voxels_h5 = None
        self.total_voxels_in_h5 = None

        # Generate cache key
        self.cache_key = self._generate_cache_key()
        self.mapping_cache_dir = mapping_cache_dir
        os.makedirs(self.mapping_cache_dir, exist_ok=True)
        self.cache_file_path = os.path.join(self.mapping_cache_dir, f"index_mapping_{self.cache_key}.pkl")

        self.index_mapping = None
        self.__len__()

        # Open file and load into memory
        self._open_file()
        self._load_data_into_memory()

        # Initialize SDF parameters
        self.sdf = sdf
        self.sdf_scale = sdf_scale
        self.sdf_cache = {}

    def _load_data_into_memory(self):
        """
        Load the data into memory from a cache file if it exists else generate the data in memory.
        Currently, the 3D dataset required around 200 Gb of memory to load with 4 workers for training.
        An LRU caching mechanism using HDF5 file is in development.
        """
        print("Loading data into memory...")

        # Determine the in-memory cache file based on cache_key
        data_in_memory_filename = f"data_in_memory_{self.cache_key}.npy"
        data_in_memory_path = os.path.join(self.mapping_cache_dir, data_in_memory_filename)

        # Check if in-memory cache exists
        if os.path.exists(data_in_memory_path):
            print(f"Found in-memory cache '{data_in_memory_filename}'. Verifying size...")
            cached_data = np.load(data_in_memory_path)
            if cached_data.shape[0] == len(self.index_mapping):
                print(f"Loading {cached_data.shape[0]} voxels from cache '{data_in_memory_filename}'.")
                self.data_in_memory = cached_data.astype(np.float32)
                return
            else:
                print(
                    f"Wrong cache detected (size {cached_data.shape[0]} vs expected {len(self.index_mapping)}). Removing '{data_in_memory_filename}'..."
                )
                try:
                    os.remove(data_in_memory_path)
                except Exception as e:
                    print(f"Error removing wrong cache file '{data_in_memory_filename}': {e}")

        # Cache miss or mismatch: generate in-memory data
        batch_size = 1000
        self.data_in_memory = []
        total = len(self.index_mapping)
        for i in tqdm(range(0, total, batch_size), desc="Loading Voxels into Memory"):
            batch_indices = self.index_mapping[i : i + batch_size]
            batch_data = self.voxels_h5[batch_indices]
            if self.interpol:
                # Upscale each voxel in the batch to the target size
                processed_batch = np.array([upscale_voxels(voxel, self.target_size) for voxel in batch_data])
            else:
                processed_batch = batch_data
            self.data_in_memory.append(processed_batch)

        self.data_in_memory = np.concatenate(self.data_in_memory, axis=0)
        self.voxels_h5 = None
        self.h5f.close()
        self.h5f = None
        print("Data loading complete.")

        # Save for future runs. This is a hacky method to avoid loading the data into memory again and again voxel grid by voxel grid
        try:
            np.save(data_in_memory_path, self.data_in_memory)
            print(f"Saved {len(self.data_in_memory)} voxels to in-memory cache '{data_in_memory_filename}'.")
        except Exception as e:
            print(f"Error saving in-memory cache '{data_in_memory_filename}': {e}")

    def _generate_cache_key(self):
        """
        Generate a cache key based on the dataset parameters, this is a unique identifier for the dataset and the
        parameters used to generate the dataset are exposed directly to the command line.
        """
        params = {
            "bw_ratio": self.bw_ratio,
            "check_for_edges": self.check_for_edges,
            "edge_thickness": self.edge_thickness,
            "small": self.small,
            "percentage": self.percentage,
        }
        # only if interpolate is True, add it to the cache key
        if self.interpol:
            params["interpol"] = self.interpol

        params_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(params_str.encode("utf-8")).hexdigest()
        return cache_key

    def _open_file(self):
        """
        Open the HDF5 file and load the data into memory.
        """
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file_path, "r")
            self.voxels_h5 = self.h5f["voxels"]
            self.total_voxels_in_h5 = self.voxels_h5.shape[0]
            print(f"[Worker {os.getpid()}] HDF5 file opened. Total voxels: {self.total_voxels_in_h5}")

            if self.small:
                subset_percentage = self.percentage
                self.total_voxels = int(self.total_voxels_in_h5 * subset_percentage)
                print(
                    f"[Worker {os.getpid()}] Using small dataset: {self.total_voxels} voxels ({subset_percentage * 100}%)"
                )
            else:
                self.total_voxels = self.total_voxels_in_h5
                print(f"[Worker {os.getpid()}] Using full dataset: {self.total_voxels} voxels")

    def _create_index_mapping(self):
        print(f"[Worker {os.getpid()}] Creating index mapping with filtering...")

        num_cores = int(
            mp.cpu_count() // 1.25
        )  # use only 80% of the cores for now...found out that perf is slightly better with this
        print(f"[Worker {os.getpid()}] Using {num_cores} cores for filtering")

        with mp.Pool(
            processes=num_cores,
            initializer=init_worker,
            initargs=(
                self.h5_file_path,
                self.check_for_edges,
                self.edge_thickness,
                self.bw_ratio,
            ),
        ) as pool:
            # Use imap with the calculated chunksize
            results = list(
                tqdm(
                    pool.imap(worker, range(self.total_voxels)),
                    total=self.total_voxels,
                    desc="Filtering Voxels",
                )
            )

        valid_indices = [idx for idx in results if idx is not None]

        print(f"[Worker {os.getpid()}] Total voxels after filtering: {len(valid_indices)}")
        return valid_indices

    def __len__(self):
        if self.index_mapping is None:
            self._open_file()
            print(f"[VoxelDataset] Using cache file path: {self.cache_file_path}")
            if os.path.exists(self.cache_file_path):
                print(f"[Worker {os.getpid()}] Loading index mapping from cache...")
                with open(self.cache_file_path, "rb") as f:
                    self.index_mapping = pickle.load(f)
                print(f"[Worker {os.getpid()}] Total voxels after filtering (cached): {len(self.index_mapping)}")
            else:
                self.index_mapping = self._create_index_mapping()
                print(f"[Worker {os.getpid()}] Saving index mapping to cache...")
                with open(self.cache_file_path, "wb") as f:
                    pickle.dump(self.index_mapping, f)
        return len(self.index_mapping)

    def __getitem__(self, idx):
        binary_voxel = self.data_in_memory[idx]
        if self.sdf:
            cache_key = (idx, self.sdf_scale)

            if cache_key not in self.sdf_cache:
                sdf = compute_sdf(binary_voxel, scale=self.sdf_scale)
                self.sdf_cache[cache_key] = sdf
            else:
                sdf = self.sdf_cache[cache_key]

            voxel_tensor = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)  # add channel dimension

            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)
            return voxel_tensor
        else:
            # remap between -1 and 1 since binary data might {0, 1} was not be good for training
            voxel_minus_one_one = (binary_voxel * 2.0) - 1.0

            voxel_tensor = torch.tensor(voxel_minus_one_one, dtype=torch.float32).unsqueeze(0)

            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)
            return voxel_tensor

    def __del__(self):
        if hasattr(self, "h5f") and self.h5f:
            self.h5f.close()


def get_voxel_dataloaders(
    h5_file_path,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    transform=None,
    small=False,
    val_split=0.2,
    patched=True,
    bw_ratio=0.0,
    check_for_edges=False,
    edge_thickness=2,
    pin_memory=False,
    drop_last=True,
    cache_size=1000,
    mapping_cache_dir=None,
    percentage=0.04,
    sdf=True,
    interpol=False,
    sdf_scale=5.0,
):
    dataset = VoxelDataset(
        h5_file_path=h5_file_path,
        voxel_size=(32, 64, 64),
        transform=transform,
        small=small,
        patched=patched,
        bw_ratio=bw_ratio,
        check_for_edges=check_for_edges,
        edge_thickness=edge_thickness,
        cache_size=cache_size,
        mapping_cache_dir=mapping_cache_dir,
        percentage=percentage,
        sdf=sdf,
        interpol=interpol,
        sdf_scale=sdf_scale,
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print(f"DataLoader using {num_workers} workers.")

    return train_loader, val_loader


if __name__ == "__main__":
    # test the dataloader with a small dataset. cache files are usually saved in the same directory as the base dataset.

    h5_file_path = "dataset/voxel_data.h5"
    batch_size = 16
    num_workers = 2
    pin_memory = False
    train_loader, val_loader = get_voxel_dataloaders(
        h5_file_path=h5_file_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=None,
        small=True,
        val_split=0.2,
        patched=True,
        bw_ratio=0.5,
        check_for_edges=True,
        edge_thickness=2,
        pin_memory=False,
        drop_last=True,
        cache_size=1000,
        mapping_cache_dir=None,
        percentage=0.3,
    )
