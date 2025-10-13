"""
Heuristic RVE generator for civil geotechnical sand and recycled aggregate mixes.

This script produces binary voxel volumes that mimic dense granular packs using a
gravity-based sequential placement algorithm with superquadric grains. The output
volumes are compatible with the existing diffusion pipeline: volumes are saved as
lists of tensors inside gzip-compressed .pt files, matching the expected dataset
layout (each tensor has shape (depth, height, width) = (32, 64, 64)).

Example usage:
    python dataset_gen/heuristic_rve_generator.py \\
        --output-dir dataset/sand_mixed \\
        --num-volumes 256 \\
        --volumes-per-file 64 \\
        --profile sand_recycled_dense \\
        --seed 13
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Profiles
# -----------------------------------------------------------------------------

DEFAULT_PROFILES: Dict[str, Dict] = {
    "sand_recycled_dense": {
        "grid_shape": (32, 64, 64),  # (z, y, x)
        "packing_fraction": 0.58,
        "min_equiv_radius": 1.1,
        "placement": {
            "max_attempts": 24000,
            "per_grain_attempts": 80,
            "tolerance": 0.08,
            "wall_padding": 0.8,
            "floor_padding": 0.2,
            "ceiling_padding": 0.2,
        },
        "families": [
            {
                "name": "dense_sand",
                "weight": 0.40,
                "diameter_lognormal": {
                    "median": 5.2,
                    "sigma": 0.16,
                    "min": 3.5,
                    "max": 6.8,
                },
                "aspect_ratio_range": (0.75, 1.25),
                "flatness_range": (0.78, 1.35),
                "shape_exponent_range": (0.80, 0.98),
            },
            {
                "name": "fine_sand",
                "weight": 0.25,
                "diameter_lognormal": {
                    "median": 3.6,
                    "sigma": 0.14,
                    "min": 2.4,
                    "max": 5.0,
                },
                "aspect_ratio_range": (0.82, 1.20),
                "flatness_range": (0.88, 1.18),
                "shape_exponent_range": (0.90, 1.05),
            },
            {
                "name": "recycled_chunks",
                "weight": 0.35,
                "diameter_lognormal": {
                    "median": 7.5,
                    "sigma": 0.17,
                    "min": 6.0,
                    "max": 9.5,
                },
                "aspect_ratio_range": (0.55, 1.35),
                "flatness_range": (0.55, 1.28),
                "shape_exponent_range": (0.65, 0.92),
            },
        ],
        "filling": {
            "diameter_range": (1.2, 2.8),
            "max_attempts": 20000,
            "per_grain_attempts": 120,
            "aspect_ratio_range": (0.88, 1.20),
            "flatness_range": (0.88, 1.28),
            "shape_exponent_range": (0.94, 1.08),
            "min_equiv_radius": 0.8,
            "tolerance": -0.05,
        },
        "postprocess": {
            "closing_iterations": 1,
            "dilation_iterations": 1,
            "erosion_iterations": 0,
        },
    }
}


def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random rotation matrix using uniformly sampled quaternions.
    """
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)

    q = np.array([q1, q2, q3, q4])

    # Convert quaternion to rotation matrix
    q_norm = q / np.linalg.norm(q)
    w, x, y, z = q_norm

    rot = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float32,
    )
    return rot


def choose_family(families: List[Dict], rng: np.random.Generator) -> Dict:
    weights = np.array([f["weight"] for f in families], dtype=np.float32)
    probs = weights / weights.sum()
    idx = rng.choice(len(families), p=probs)
    return families[idx]


def sample_diameter(params: Dict, rng: np.random.Generator) -> float:
    """
    Sample an equivalent spherical diameter (voxel units).
    """
    median = params["median"]
    sigma = params["sigma"]
    d = rng.lognormal(mean=math.log(median), sigma=sigma)
    return float(np.clip(d, params["min"], params["max"]))


def sample_axes(equiv_radius: float, family: Dict, rng: np.random.Generator) -> np.ndarray:
    """
    Sample semi-axis lengths for a superquadric grain. The returned axes preserve
    the volume associated with the equivalent spherical radius.
    """
    aspect = rng.uniform(*family["aspect_ratio_range"])
    flatness = rng.uniform(*family["flatness_range"])

    # Start with ratios and enforce unit product to keep total volume consistent.
    ratios = np.array([1.0, aspect, flatness], dtype=np.float32)
    ratios *= (1.0 / (np.prod(ratios) ** (1.0 / 3.0)))
    rng.shuffle(ratios)

    axes = equiv_radius * ratios
    target_volume = equiv_radius**3
    current_volume = np.prod(axes)
    if current_volume <= 0.0:
        return np.array([equiv_radius, equiv_radius, equiv_radius], dtype=np.float32)

    scale = (target_volume / current_volume) ** (1.0 / 3.0)
    return axes * scale


def sample_shape_exponent(family: Dict, rng: np.random.Generator) -> float:
    lo, hi = family["shape_exponent_range"]
    return float(rng.uniform(lo, hi))


# -----------------------------------------------------------------------------
# Grain dataclass and placement
# -----------------------------------------------------------------------------


@dataclass
class Grain:
    center: np.ndarray  # (x, y, z)
    axes: np.ndarray  # semi-axis lengths (a, b, c)
    rotation: np.ndarray  # 3x3 rotation matrix
    shape_exponent: float
    bounding_radius: float
    family: str


def place_grain(
    bounding_radius: float,
    centers: np.ndarray,
    radii: np.ndarray,
    grid_shape: Tuple[int, int, int],
    rng: np.random.Generator,
    placement_cfg: Dict,
    override_tolerance: Optional[float] = None,
    override_attempts: Optional[int] = None,
) -> Tuple[bool, np.ndarray]:
    """
    Attempt to place a grain using a gravity-inspired sequential drop. Returns
    (success, center).
    """
    z_size, y_size, x_size = grid_shape

    tol = placement_cfg["tolerance"] if override_tolerance is None else override_tolerance
    wall_padding = placement_cfg["wall_padding"]
    floor_padding = placement_cfg["floor_padding"]
    ceiling_padding = placement_cfg["ceiling_padding"]

    x_min = bounding_radius + wall_padding
    x_max = x_size - bounding_radius - wall_padding
    y_min = bounding_radius + wall_padding
    y_max = y_size - bounding_radius - wall_padding
    z_min = bounding_radius + floor_padding
    z_max = z_size - bounding_radius - ceiling_padding

    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return False, np.zeros(3, dtype=np.float32)

    max_tries = override_attempts or placement_cfg["per_grain_attempts"]
    for _ in range(max_tries):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)

        required_z = z_min
        if centers.size:
            dx = x - centers[:, 0]
            dy = y - centers[:, 1]
            dist_xy_sq = dx * dx + dy * dy
            total_radius = bounding_radius + radii
            threshold = np.maximum(total_radius - tol, 0.0)
            mask = dist_xy_sq < threshold * threshold
            if np.any(mask):
                valid_total = total_radius[mask]
                valid_dxy = dist_xy_sq[mask]
                contact_heights = centers[mask, 2] + np.sqrt(np.clip(valid_total**2 - valid_dxy, 0.0, None))
                if contact_heights.size:
                    required_z = max(required_z, float(contact_heights.max()))

        if required_z > z_max:
            continue

        center = np.array([x, y, required_z], dtype=np.float32)

        if centers.size:
            diff = center - centers
            dist_sq = np.sum(diff * diff, axis=1)
            limit = np.maximum(bounding_radius + radii - tol, 0.0)
            if np.any(dist_sq < limit * limit):
                continue

            return True, center
        else:
            return True, center

    return False, np.zeros(3, dtype=np.float32)


# -----------------------------------------------------------------------------
# Rasterisation
# -----------------------------------------------------------------------------


def rasterize_grains(grains: Iterable[Grain], grid_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert placed grains into a binary occupancy grid (uint8).
    """
    z_size, y_size, x_size = grid_shape
    grid = np.zeros(grid_shape, dtype=np.uint8)

    for grain in grains:
        radius = grain.bounding_radius
        x_min = max(int(math.floor(grain.center[0] - radius - 1)), 0)
        x_max = min(int(math.ceil(grain.center[0] + radius + 1)), x_size - 1)
        y_min = max(int(math.floor(grain.center[1] - radius - 1)), 0)
        y_max = min(int(math.ceil(grain.center[1] + radius + 1)), y_size - 1)
        z_min = max(int(math.floor(grain.center[2] - radius - 1)), 0)
        z_max = min(int(math.ceil(grain.center[2] + radius + 1)), z_size - 1)

        inv_rot = grain.rotation.T
        exponent = grain.shape_exponent
        a, b, c = grain.axes

        pow_factor = 2.0 / max(exponent, 1e-3)

        for z_idx in range(z_min, z_max + 1):
            z_coord = z_idx + 0.5
            for y_idx in range(y_min, y_max + 1):
                y_coord = y_idx + 0.5
                for x_idx in range(x_min, x_max + 1):
                    if grid[z_idx, y_idx, x_idx]:
                        continue
                    x_coord = x_idx + 0.5
                    diff = np.array(
                        [x_coord - grain.center[0], y_coord - grain.center[1], z_coord - grain.center[2]],
                        dtype=np.float32,
                    )
                    local = inv_rot @ diff

                    value = (
                        (abs(local[0]) / max(a, 1e-3)) ** pow_factor
                        + (abs(local[1]) / max(b, 1e-3)) ** pow_factor
                        + (abs(local[2]) / max(c, 1e-3)) ** pow_factor
                    )
                    if value <= 1.0:
                        grid[z_idx, y_idx, x_idx] = 1

    return grid


# -----------------------------------------------------------------------------
# Morphological helpers
# -----------------------------------------------------------------------------


def binary_dilation(volume: np.ndarray, iterations: int = 1) -> np.ndarray:
    vol = volume.astype(np.uint8)
    if iterations <= 0:
        return vol

    for _ in range(iterations):
        padded = np.pad(vol, 1, mode="edge")
        neighbours = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbours.append(
                        padded[1 + dz : 1 + dz + vol.shape[0], 1 + dy : 1 + dy + vol.shape[1], 1 + dx : 1 + dx + vol.shape[2]]
                    )
        vol = np.maximum.reduce(neighbours)
    return vol


def binary_erosion(volume: np.ndarray, iterations: int = 1) -> np.ndarray:
    vol = volume.astype(np.uint8)
    if iterations <= 0:
        return vol

    for _ in range(iterations):
        padded = np.pad(vol, 1, mode="edge")
        neighbours = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbours.append(
                        padded[1 + dz : 1 + dz + vol.shape[0], 1 + dy : 1 + dy + vol.shape[1], 1 + dx : 1 + dx + vol.shape[2]]
                    )
        vol = np.minimum.reduce(neighbours)
    return vol


def morphological_postprocess(volume: np.ndarray, config: Dict) -> np.ndarray:
    closing_iterations = config.get("closing_iterations", 0)
    dilation_iterations = config.get("dilation_iterations", 0)
    erosion_iterations = config.get("erosion_iterations", 0)

    vol = volume
    for _ in range(max(closing_iterations, 0)):
        vol = binary_dilation(vol, 1)
        vol = binary_erosion(vol, 1)

    if dilation_iterations > 0:
        vol = binary_dilation(vol, dilation_iterations)

    if erosion_iterations > 0:
        vol = binary_erosion(vol, erosion_iterations)

    return vol


# -----------------------------------------------------------------------------
# Volume generation
# -----------------------------------------------------------------------------


def generate_volume(profile: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """
    Generate a single RVE volume and return it along with metadata.
    """
    grid_shape = tuple(profile["grid_shape"])
    packing_fraction = profile["packing_fraction"]
    families = profile["families"]
    placement_cfg = profile["placement"]

    z_size, y_size, x_size = grid_shape
    domain_volume = x_size * y_size * z_size
    target_volume = domain_volume * packing_fraction

    grains: List[Grain] = []
    centers = np.empty((0, 3), dtype=np.float32)
    radii = np.empty((0,), dtype=np.float32)
    accumulated_volume = 0.0

    attempt_budget = profile["placement"]["max_attempts"]
    made_attempts = 0
    min_equiv_radius = profile.get("min_equiv_radius", 0.5)

    while accumulated_volume < target_volume and made_attempts < attempt_budget:
        family = choose_family(families, rng)
        diameter = sample_diameter(family["diameter_lognormal"], rng)
        progress = accumulated_volume / max(target_volume, 1e-8)
        size_scale = np.interp(
            progress,
            [0.0, 0.55, 0.8, 1.0],
            [1.0, 0.92, 0.82, 0.7],
        )
        min_d = family["diameter_lognormal"]["min"] * 0.9
        max_d = family["diameter_lognormal"]["max"]
        diameter = float(np.clip(diameter * size_scale, min_d, max_d))
        equiv_radius = max(diameter / 2.0, min_equiv_radius)
        axes = sample_axes(equiv_radius, family, rng)
        shape_exponent = sample_shape_exponent(family, rng)
        rotation = random_rotation_matrix(rng)
        bounding_radius = float(np.max(axes))

        success, center = place_grain(bounding_radius, centers, radii, grid_shape, rng, placement_cfg)
        made_attempts += 1
        if not success:
            continue

        grain = Grain(center=center, axes=axes, rotation=rotation, shape_exponent=shape_exponent, bounding_radius=bounding_radius, family=family["name"])
        grains.append(grain)
        accumulated_volume += (4.0 / 3.0) * math.pi * np.prod(axes)
        centers = np.vstack([centers, center[None, :]])
        radii = np.append(radii, bounding_radius)

    filling_info = profile.get("filling")
    filling_attempts = 0
    filling_successes = 0
    if accumulated_volume < target_volume and filling_info:
        fill_budget = filling_info.get("max_attempts", 0)
        fill_family = {
            "aspect_ratio_range": filling_info["aspect_ratio_range"],
            "flatness_range": filling_info["flatness_range"],
            "shape_exponent_range": filling_info["shape_exponent_range"],
        }
        fill_min_radius = filling_info.get("min_equiv_radius", min_equiv_radius * 0.8)
        while accumulated_volume < target_volume and filling_attempts < fill_budget:
            diameter = rng.uniform(*filling_info["diameter_range"])
            equiv_radius = max(diameter / 2.0, fill_min_radius)
            axes = sample_axes(equiv_radius, fill_family, rng)
            shape_exponent = sample_shape_exponent(fill_family, rng)
            rotation = random_rotation_matrix(rng)
            bounding_radius = float(np.max(axes))

            success, center = place_grain(
                bounding_radius,
                centers,
                radii,
                grid_shape,
                rng,
                placement_cfg,
                override_tolerance=filling_info.get("tolerance"),
                override_attempts=filling_info.get("per_grain_attempts"),
            )
            filling_attempts += 1
            if not success:
                continue

            grain = Grain(center=center, axes=axes, rotation=rotation, shape_exponent=shape_exponent, bounding_radius=bounding_radius, family="fines_fill")
            grains.append(grain)
            accumulated_volume += (4.0 / 3.0) * math.pi * np.prod(axes)
            centers = np.vstack([centers, center[None, :]])
            radii = np.append(radii, bounding_radius)
            filling_successes += 1

    volume = rasterize_grains(grains, grid_shape)
    if "postprocess" in profile:
        volume = morphological_postprocess(volume, profile["postprocess"])
    actual_solid_fraction = float(volume.sum() / domain_volume)
    metadata = {
        "num_grains": len(grains),
        "target_packing_fraction": packing_fraction,
        "continuous_solid_fraction": float(accumulated_volume / domain_volume),
        "realized_solid_fraction": actual_solid_fraction,
        "attempts": made_attempts,
        "filling_attempts_used": filling_attempts,
        "filling_successes": filling_successes,
        "converged": accumulated_volume >= target_volume,
    }
    return volume, metadata


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------


def save_volume_chunk(
    tensors: List[torch.Tensor],
    output_dir: Path,
    chunk_index: int,
) -> str:
    """
    Save a list of tensors to a gzip-compressed .pt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"rve_{chunk_index:04d}.pt.gz"
    with gzip.open(file_path, "wb") as fp:
        torch.save(tensors, fp)
    return str(file_path)


def generate_dataset(
    profile_name: str,
    output_dir: Path,
    num_volumes: int,
    volumes_per_file: int,
    seed: int,
) -> Dict:
    """
    Generate a dataset of volumes and save them in pipeline-compatible chunks.
    """
    if profile_name not in DEFAULT_PROFILES:
        raise KeyError(f"Unknown profile '{profile_name}'. Available: {list(DEFAULT_PROFILES)}")

    profile = DEFAULT_PROFILES[profile_name]

    rng = np.random.default_rng(seed)
    saved_files: List[str] = []
    chunk: List[torch.Tensor] = []

    volume_stats: List[Dict] = []
    solid_fractions: List[float] = []

    for idx in tqdm(range(num_volumes), desc="Generating volumes"):
        volume, meta = generate_volume(profile, rng)
        tensor = torch.from_numpy(volume.astype(np.uint8))
        chunk.append(tensor)
        volume_stats.append(meta)
        solid_fractions.append(meta["realized_solid_fraction"])

        if len(chunk) == volumes_per_file or idx == num_volumes - 1:
            chunk_idx = len(saved_files)
            path = save_volume_chunk(chunk, output_dir, chunk_idx)
            saved_files.append(path)
            chunk = []

    return {
        "profile": profile_name,
        "output_dir": str(output_dir),
        "num_volumes": num_volumes,
        "volumes_per_file": volumes_per_file,
        "seed": seed,
        "saved_files": saved_files,
        "solid_fraction_mean": float(np.mean(solid_fractions)),
        "solid_fraction_std": float(np.std(solid_fractions)),
        "volume_stats": volume_stats,
        "profile_config": profile,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic RVE generator for sand/recycled aggregate media.")
    parser.add_argument("--profile", default="sand_recycled_dense", help=f"Profile to use. Choices: {list(DEFAULT_PROFILES)}")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for generated .pt.gz files.")
    parser.add_argument("--num-volumes", type=int, default=128, help="Total number of RVE volumes to generate.")
    parser.add_argument("--volumes-per-file", type=int, default=128, help="Number of volumes stored per .pt.gz chunk.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--metadata-path", type=Path, help="Optional path for JSON metadata summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_dataset(
        profile_name=args.profile,
        output_dir=args.output_dir,
        num_volumes=args.num_volumes,
        volumes_per_file=args.volumes_per_file,
        seed=args.seed,
    )

    metadata_path = args.metadata_path or (args.output_dir / "generation_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, default=to_serializable)

    print(f"Generated {summary['num_volumes']} volumes in {len(summary['saved_files'])} files.")
    print(f"Mean solid fraction: {summary['solid_fraction_mean']:.3f} ± {summary['solid_fraction_std']:.3f}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
