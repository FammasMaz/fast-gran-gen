"""One-stop script to generate lunar RVEs and pack them into an HDF5 dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dataset_gen.hdf5_maker import create_hdf5_file
from dataset_gen.lunar_rve_generator import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lunar RVEs and convert them into an HDF5 dataset.")
    parser.add_argument("--sysconfig", type=Path, required=True, help="Path to the simulant *-sysconfig.dat file.")
    parser.add_argument("--stack-dir", type=Path, required=True, help="Directory containing the *.mic.gz stacks referenced by sysconfig.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated .pt.gz shards.")
    parser.add_argument("--hdf5-path", type=Path, required=True, help="Destination HDF5 file (will be created).")
    parser.add_argument("--profile", default="lunar_oprl2n_75_300", help="Lunar RVE profile to use.")
    parser.add_argument("--num-volumes", type=int, default=256, help="Number of RVE volumes to generate.")
    parser.add_argument("--volumes-per-file", type=int, default=32, help="Number of volumes per .pt.gz shard.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation.")
    parser.add_argument("--margin-voxels", type=int, default=4, help="Boundary margin zeroed during segmentation.")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel workers for generation (0 = auto).")
    parser.add_argument(
        "--grain-cache",
        type=Path,
        help="Optional pickle file where segmented grains are cached for faster subsequent runs.",
    )
    parser.add_argument("--hdf5-workers", type=int, default=8, help="Workers used while packing the HDF5 file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    worker_count = args.num_workers if args.num_workers > 0 else max(1, os.cpu_count() or 1)

    summary = generate_dataset(
        profile_name=args.profile,
        output_dir=args.output_dir,
        sysconfig_path=args.sysconfig,
        stack_dir=args.stack_dir,
        num_volumes=args.num_volumes,
        volumes_per_file=args.volumes_per_file,
        seed=args.seed,
        margin_voxels=args.margin_voxels,
        num_workers=worker_count,
        grain_cache=args.grain_cache,
    )

    voxel_shape = tuple(int(x) for x in summary["profile_config"]["grid_shape"])
    total_samples = summary["num_volumes"]

    create_hdf5_file(
        voxel_dir=str(args.output_dir),
        hdf5_path=str(args.hdf5_path),
        total_voxels=total_samples,
        num_workers=args.hdf5_workers,
        voxel_shape=voxel_shape,
    )

    print("Lunar training dataset ready:")
    print(f"  Shards: {len(summary['saved_files'])} -> {args.output_dir}")
    print(f"  HDF5:   {args.hdf5_path} (samples={total_samples}, shape={voxel_shape})")


if __name__ == "__main__":
    main()
