#!/usr/bin/env python3
"""Validate and volume-correct a completed physical track segmentation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--polyhedra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.003125)
    parser.add_argument("--minimum-particles", type=int, default=50_000)
    return parser.parse_args()


def main():
    args = parse_args()
    profile_path = args.source / "profiled_track.npy"
    profile = np.load(profile_path, mmap_mode="r")
    payload = json.loads(args.polyhedra.read_text())
    grains = payload.get("polyhedrons", {})
    if len(grains) < args.minimum_particles:
        raise RuntimeError(f"particle count too small: {len(grains)}")

    occupied_volume = float(np.count_nonzero(profile) * args.voxel_size**3)
    mesh_volume = float(sum(abs(float(g.get("volume", 0.0))) for g in grains.values()))
    retained_ratio = mesh_volume / occupied_volume
    if not 0.82 <= retained_ratio <= 1.10:
        raise RuntimeError(f"mesh/voxel volume ratio out of bounds: {retained_ratio}")
    volume_correction = retained_ratio ** (-1.0 / 3.0)

    diameters = []
    vertices_per_grain = []
    all_min = []
    all_max = []
    for key, grain in grains.items():
        vertices = np.asarray(grain["vertices"], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise RuntimeError(f"invalid grain {key}")
        center = vertices.mean(axis=0)
        diameters.append(2.0 * np.linalg.norm(vertices - center, axis=1).max())
        vertices_per_grain.append(len(vertices))
        all_min.append(vertices.min(axis=0))
        all_max.append(vertices.max(axis=0))
    diameters = np.asarray(diameters)
    corrected_diameters = diameters * volume_correction
    if not 0.040 <= corrected_diameters.mean() <= 0.060:
        raise RuntimeError(f"physical mean diameter invalid: {corrected_diameters.mean()}")

    corrected = json.loads(args.polyhedra.read_text())
    for grain in corrected["polyhedrons"].values():
        vertices = np.asarray(grain["vertices"], dtype=float)
        center = np.asarray(grain.get("centroid") or vertices.mean(axis=0), dtype=float)
        scaled = center + volume_correction * (vertices - center)
        grain["vertices"] = scaled.tolist()
        grain["centroid"] = center.tolist()
        if "volume" in grain:
            grain["volume"] = float(grain["volume"]) * volume_correction**3
    args.output.write_text(json.dumps(corrected, separators=(",", ":")))

    marker = {
        "status": "complete",
        "scope": "native_physical_five_sleeper_ballast",
        "source_generation": str(args.source),
        "particle_count": len(grains),
        "occupied_voxel_volume_m3": occupied_volume,
        "mesh_volume_before_correction_m3": mesh_volume,
        "mesh_to_voxel_volume_ratio": retained_ratio,
        "lmgc90_volume_correction_scale": volume_correction,
        "corrected_mean_diameter_m": float(corrected_diameters.mean()),
        "diameter_m_quantiles_after_volume_correction": np.quantile(
            corrected_diameters, [0.0, 0.05, 0.5, 0.95, 1.0]
        ).tolist(),
        "median_vertices": float(np.median(vertices_per_grain)),
        "source_axis_bounds_m": {
            "minimum": np.min(np.asarray(all_min), axis=0).tolist(),
            "maximum": np.max(np.asarray(all_max), axis=0).tolist(),
        },
        "polyhedra_sha256": sha256(args.output),
        "profile_sha256": sha256(profile_path),
    }
    args.marker.write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
