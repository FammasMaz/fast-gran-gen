#!/usr/bin/env python3
"""Crop and mask a generated voxel volume into a realistic ballast profile."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_crop(volume, target_shape):
    slices = []
    for current, target in zip(volume.shape, target_shape):
        if target > current:
            raise ValueError(f"target axis {target} exceeds generated axis {current}")
        start = (current - target) // 2
        slices.append(slice(start, start + target))
    return volume[tuple(slices)]


def build_profile_mask(shape, spacing, crest_width, base_width, length):
    nz, ny, nx = shape
    dz, dy, dx = spacing
    depth = nz * dz
    z = (np.arange(nz) + 0.5) * dz
    y = (np.arange(ny) + 0.5 - 0.5 * ny) * dy
    x = (np.arange(nx) + 0.5 - 0.5 * nx) * dx
    vertical_fraction = np.clip(z / depth, 0.0, 1.0)
    half_width = 0.5 * (
        base_width + (crest_width - base_width) * vertical_fraction
    )
    transverse = np.abs(y)[None, :, None] <= half_width[:, None, None]
    longitudinal = np.abs(x)[None, None, :] <= 0.5 * length
    return transverse & longitudinal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument(
        "--isotropic-voxel-size",
        type=float,
        default=None,
        help="Use one physical voxel size on all axes (old LMGC90 grain-scale contract).",
    )
    parser.add_argument("--depth", type=float, default=0.30)
    parser.add_argument("--base-width", type=float, default=4.50)
    parser.add_argument("--crest-width", type=float, default=3.60)
    parser.add_argument("--length", type=float, default=4.20)
    parser.add_argument("--base-depth", type=float, default=0.10)
    parser.add_argument("--base-width-unit", type=float, default=0.30)
    parser.add_argument("--base-length-unit", type=float, default=0.30)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    source = np.load(args.input)
    if source.ndim != 3:
        raise ValueError(f"expected 3-D volume, got {source.shape}")
    if args.isotropic_voxel_size is not None:
        if args.isotropic_voxel_size <= 0.0:
            raise ValueError("isotropic voxel size must be positive")
        final_spacing = np.full(3, args.isotropic_voxel_size)
    else:
        base_spacing = np.array(
            [
                args.base_depth / 32.0,
                args.base_width_unit / 64.0,
                args.base_length_unit / 64.0,
            ]
        )
        final_spacing = base_spacing * args.scale_factor
    final_dimensions = np.array([args.depth, args.base_width, args.length])
    target_shape = np.rint(final_dimensions / final_spacing).astype(int)
    cropped = centered_crop(source, target_shape)
    mask = build_profile_mask(
        cropped.shape,
        final_spacing,
        args.crest_width,
        args.base_width,
        args.length,
    )
    profiled = ((cropped > args.threshold) & mask).astype(np.uint8)
    if not profiled.any():
        raise RuntimeError("profile mask removed all occupied voxels")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, profiled)
    vti_path = args.output.with_suffix(".vti")
    image = pv.ImageData(dimensions=profiled.shape)
    image.spacing = tuple(final_spacing)
    image.origin = (0.0, -0.5 * args.base_width, -0.5 * args.length)
    image["voxel_data"] = profiled.flatten(order="F")
    image.save(vti_path)

    occupied = np.argwhere(profiled > 0)
    record = {
        "schema": "diffusion_ballast_profile_v1",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_shape_depth_width_length": list(source.shape),
        "profile_shape_depth_width_length": list(profiled.shape),
        "axis_order": ["depth_z", "cross_track_y", "along_track_x"],
        "voxel_spacing_depth_width_length_m": final_spacing.tolist(),
        "uniform_scale_factor": args.scale_factor,
        "isotropic_voxel_size_m": args.isotropic_voxel_size,
        "dimensions_m": {
            "depth": args.depth,
            "base_width": args.base_width,
            "crest_width": args.crest_width,
            "length": args.length,
        },
        "shoulder_run_m": 0.5 * (args.base_width - args.crest_width),
        "shoulder_run_to_rise": 0.5
        * (args.base_width - args.crest_width)
        / args.depth,
        "occupied_voxels": int(profiled.sum()),
        "occupancy_fraction_in_profile_box": float(profiled.mean()),
        "occupied_index_bounds": {
            "minimum": occupied.min(axis=0).tolist(),
            "maximum": occupied.max(axis=0).tolist(),
        },
    }
    args.metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
