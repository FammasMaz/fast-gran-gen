#!/usr/bin/env python3
import argparse
import os
import gzip
import torch
import sys
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing

# Add project root to path so we can import dataset_gen
sys.path.append(os.getcwd())


def viz_stack(stack, inp_shape):
    """Reconstruct dense 3D voxel grid from sparse slices."""
    max_rows, max_cols = inp_shape
    num_slices = len(stack)
    dense_stack = torch.zeros((num_slices, max_rows, max_cols), dtype=torch.uint8)

    for i, indices in enumerate(stack):
        if indices.shape[1] > 0:
            dense_stack[i, indices[0], indices[1]] = 1

    return dense_stack


def extract_blocks(voxel_grid, block_shape=(32, 64, 64)):
    """Extract non-overlapping blocks from the dense grid."""
    grid_shape = voxel_grid.shape  # (D, H, W)

    # Calculate number of blocks in each dimension
    num_blocks_d = grid_shape[0] // block_shape[0]
    num_blocks_h = grid_shape[1] // block_shape[1]
    num_blocks_w = grid_shape[2] // block_shape[2]

    blocks = []
    for i in range(num_blocks_d):
        for j in range(num_blocks_h):
            for k in range(num_blocks_w):
                start_d = i * block_shape[0]
                end_d = start_d + block_shape[0]
                start_h = j * block_shape[1]
                end_h = start_h + block_shape[1]
                start_w = k * block_shape[2]
                end_w = start_w + block_shape[2]

                block = voxel_grid[start_d:end_d, start_h:end_h, start_w:end_w]
                blocks.append(block.numpy().astype(np.uint8))

    return blocks


def process_file(args):
    fpath, img_size, block_shape = args
    try:
        with gzip.open(fpath, "rb") as f:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except TypeError:
                # Fallback for older torch versions that don't support weights_only
                data = torch.load(f, map_location="cpu", weights_only=False)

        if isinstance(data, list):
            # Filter and convert
            valid_blocks = []
            for b in data:
                if isinstance(b, torch.Tensor):
                    valid_blocks.append(b.numpy().astype(np.uint8))
                elif isinstance(b, np.ndarray):
                    valid_blocks.append(b.astype(np.uint8))
            return valid_blocks

        elif isinstance(data, dict) and "slices" in data:
            slices = data["slices"]
            # Reconstruct dense grid
            dense_grid = viz_stack(slices, img_size)
            # Extract blocks
            blocks = extract_blocks(dense_grid, block_shape)
            return blocks

        else:
            if isinstance(data, torch.Tensor):
                if data.ndim == 3:
                    blocks = extract_blocks(data, block_shape)
                    return blocks

            print(f"Warning: Unknown or empty data format in {fpath}")
            return []

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Convert folder of .pt.gz files (slices or blocks) to HDF5.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .pt.gz files")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output .h5 file")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size (H, W) for reconstruction",
    )
    parser.add_argument(
        "--block-shape",
        type=int,
        nargs=3,
        default=[32, 64, 64],
        help="Block shape (D, H, W)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file
    img_size = tuple(args.img_size)
    block_shape = tuple(args.block_shape)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pt.gz")]
    files.sort()

    if not files:
        print("No .pt.gz files found.")
        sys.exit(1)

    print(f"Found {len(files)} files. Analyzing first file to estimate total size...")

    # Analyze first file to determine block count per file
    sample_blocks = process_file((files[0], img_size, block_shape))
    if not sample_blocks:
        print("Error: Could not extract blocks from the first file. Check format or try another file.")
        # Try one more just in case first is bad
        if len(files) > 1:
            sample_blocks = process_file((files[1], img_size, block_shape))

        if not sample_blocks:
            sys.exit(1)

    blocks_per_file = len(sample_blocks)
    total_voxels = blocks_per_file * len(files)
    sample_shape = sample_blocks[0].shape

    print(f"Detected {blocks_per_file} blocks per file.")
    print(f"Block shape: {sample_shape}")
    print(f"Estimated total samples: {total_voxels}")

    if total_voxels == 0:
        print("Total voxels is 0. Check if block shape fits into the grid dimensions.")
        sys.exit(1)

    # Prepare HDF5
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Warning: Output file {output_file} already exists. Overwriting.")

    print(f"Creating HDF5 file at {output_file}...")
    with h5py.File(output_file, "w") as f:
        dset = f.create_dataset(
            "voxels",
            shape=(total_voxels, *sample_shape),
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
        )

        # Process files in parallel
        process_args = [(f, img_size, block_shape) for f in files]

        current_idx = 0

        # We use chunksize=1 to get updates frequently, or higher for speed
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for blocks in tqdm(
                pool.imap(process_file, process_args),
                total=len(files),
                desc="Converting",
            ):
                if blocks:
                    n = len(blocks)
                    # Safety check for overflow
                    if current_idx + n > total_voxels:
                        # This happens if some files produce more blocks than the first one
                        # HDF5 dataset resizing is possible but slow/complex here.
                        # We trunc or print warning.
                        if current_idx < total_voxels:
                            n = total_voxels - current_idx
                        else:
                            continue

                    if n > 0:
                        block_arr = np.stack(blocks[:n])
                        dset[current_idx : current_idx + n] = block_arr
                        current_idx += n

    print(f"Done! Saved {current_idx} samples to {output_file}")


if __name__ == "__main__":
    main()
