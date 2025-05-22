import os
import gzip
import h5py
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm


def process_file(file_path):
    """Process a single file and return tensors."""
    try:
        with gzip.open(file_path, "rb") as gz_f:
            data_list = torch.load(gz_f, map_location="cpu")
            tensors = []
            for tensor in data_list:
                # Each file has a list of 128 voxels, each of shape (32, 64, 64)
                if isinstance(tensor, torch.Tensor):
                    tensors.append(tensor.cpu().numpy().astype(np.uint8))
            return (file_path, tensors)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, None)


def create_hdf5_file(voxel_dir, hdf5_path, total_voxels=56704, num_workers=8):
    """
    Create HDF5 file from .pt.gz files using parallel processing.

    Args:
        voxel_dir (str): Directory containing .pt.gz voxel files.
        hdf5_path (str): Output HDF5 file path.
    """
    if os.path.exists(hdf5_path):
        print(f"HDF5 file already exists at {hdf5_path}. Remove it or specify a new path.")
        return

    print(f"Total voxels to process: {total_voxels}")

    files = [os.path.join(voxel_dir, f) for f in os.listdir(voxel_dir) if f.endswith(".pt.gz")]
    files.sort()

    with h5py.File(hdf5_path, "w") as f:
        dset = f.create_dataset(
            "voxels",
            shape=(total_voxels, 32, 64, 64),
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
        )

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(process_file, files)

            current_idx = 0
            progress_bar = tqdm(total=len(files), desc="Processing files")

            for file_path, tensors in results:
                if tensors is not None and len(tensors) > 0:
                    tensors_np = np.stack(tensors, axis=0)  # shape: (N, 32, 64, 64)

                    n_voxels = len(tensors_np)
                    if current_idx + n_voxels > total_voxels:
                        print(f"Warning: Exceeding total_voxels at file {file_path}")
                        n_voxels = total_voxels - current_idx
                        tensors_np = tensors_np[:n_voxels]

                    dset[current_idx : current_idx + n_voxels] = tensors_np
                    current_idx += n_voxels

                progress_bar.update(1)

            progress_bar.close()

        print(f"Created HDF5 file at {hdf5_path}")
        print(f"Total voxels processed: {current_idx}")


if __name__ == "__main__":
    # usage with the dataset provided in the repo

    voxel_dir = "dataset/voxels_shrink"
    hdf5_file_path = "dataset/voxels_shrink/voxels_int.h5"

    # adjust total_voxels based on how many files and how many voxels per file

    # currently the dataset contains 443 samples, with 128 voxels each of size (32, 64, 64), hence
    # total_voxels = 443 * 128 = 56704

    total_voxels = 56704

    # Create HDF5 file in batches
    if not os.path.exists(hdf5_file_path):
        create_hdf5_file(
            voxel_dir,
            hdf5_file_path,
            total_voxels=total_voxels,
            num_workers=40,  # change it according to the need
        )
    else:
        print("HDF5 file already exists.")
