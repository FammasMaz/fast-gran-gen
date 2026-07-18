#!/usr/bin/env python3
"""Replicate a physical diffusion segmentation over a larger track profile.

The diffusion generator is convolutional and the generated prism currently
covers 3.35 m on each horizontal axis. This utility tiles its segmented grain
state into the missing margins, clips centers against the requested trapezoidal
profile, and assigns fresh IDs. Grain shapes and physical sizes are unchanged.
"""

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--length", type=float, default=4.20)
    parser.add_argument("--base-width", type=float, default=4.50)
    parser.add_argument("--crest-width", type=float, default=3.60)
    parser.add_argument("--depth", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=25100)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text())
    source = payload.get("polyhedrons", {})
    if not source:
        raise RuntimeError("input contains no polyhedra")

    records = []
    centers = []
    for key in sorted(source, key=lambda value: int(value)):
        grain = source[key]
        vertices = np.asarray(grain["vertices"], dtype=float)
        center = np.asarray(grain.get("centroid") or vertices.mean(axis=0), dtype=float)
        records.append((grain, vertices, center))
        centers.append(center)
    centers = np.asarray(centers)

    # Segmentation coordinates are z, y, x.
    spans = np.ptp(centers, axis=0)
    tile_y = spans[1]
    tile_x = spans[2]
    if tile_y <= 0.0 or tile_x <= 0.0:
        raise RuntimeError(f"invalid source spans: {spans}")
    ny = max(1, int(math.ceil(args.base_width / tile_y)))
    nx = max(1, int(math.ceil(args.length / tile_x)))
    offsets_y = (np.arange(ny) - 0.5 * (ny - 1)) * tile_y
    offsets_x = (np.arange(nx) - 0.5 * (nx - 1)) * tile_x

    rng = np.random.default_rng(args.seed)
    output = {}
    source_ids = np.array([int(key) for key in sorted(source, key=lambda value: int(value))])
    output_id = 1
    diameter_values = []
    bounds_min = []
    bounds_max = []
    for iy, offset_y in enumerate(offsets_y):
        for ix, offset_x in enumerate(offsets_x):
            order = rng.permutation(len(records)) if (iy or ix) else np.arange(len(records))
            for record_index in order:
                grain, vertices, center = records[int(record_index)]
                translated_center = center + np.array([0.0, offset_y, offset_x])
                z = translated_center[0]
                if not 0.0 <= z <= args.depth:
                    continue
                vertical = np.clip(z / args.depth, 0.0, 1.0)
                half_width = 0.5 * (
                    args.base_width
                    + (args.crest_width - args.base_width) * vertical
                )
                if abs(translated_center[1]) > half_width:
                    continue
                if abs(translated_center[2]) > 0.5 * args.length:
                    continue
                translated = vertices + np.array([0.0, offset_y, offset_x])
                entry = dict(grain)
                entry["id"] = output_id
                entry["vertices"] = translated.tolist()
                entry["centroid"] = translated_center.tolist()
                entry["bounding_box"] = {
                    "min": translated.min(axis=0).tolist(),
                    "max": translated.max(axis=0).tolist(),
                }
                entry["ranges"] = np.ptp(translated, axis=0).tolist()
                output[str(output_id)] = entry
                diameter_values.append(
                    2.0 * np.linalg.norm(translated - translated_center, axis=1).max()
                )
                bounds_min.append(translated.min(axis=0))
                bounds_max.append(translated.max(axis=0))
                output_id += 1

    diameters = np.asarray(diameter_values)
    minimum = np.min(np.asarray(bounds_min), axis=0)
    maximum = np.max(np.asarray(bounds_max), axis=0)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "total_count": len(output),
            "replication_source_sha256": sha256(args.input),
            "replication_seed": args.seed,
            "replication_tiles_y_x": [ny, nx],
            "target_profile_m": {
                "length": args.length,
                "base_width": args.base_width,
                "crest_width": args.crest_width,
                "depth": args.depth,
            },
        }
    )
    args.output.write_text(json.dumps({"polyhedrons": output, "metadata": metadata}, separators=(",", ":")))

    marker = {
        "status": "complete",
        "scope": "physical_five_sleeper_diffusion_replication",
        "particle_count": len(output),
        "source_particle_count": len(source),
        "source_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "mean_diameter_m": float(diameters.mean()),
        "diameter_quantiles_m": np.quantile(diameters, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
        "axis_order": ["depth_z", "cross_track_y", "along_track_x"],
        "bounds_minimum_m": minimum.tolist(),
        "bounds_maximum_m": maximum.tolist(),
        "bounds_size_m": (maximum - minimum).tolist(),
        "sleeper_spacing_m": 0.60,
        "sleeper_centers_along_track_m": [-1.2, -0.6, 0.0, 0.6, 1.2],
    }
    args.marker.write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
