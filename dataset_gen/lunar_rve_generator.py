"""RVE generator for lunar simulant microstructures."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from dataset_gen.lunar_dataset import ImageStackInfo, load_raw_stack, parse_sysconfig, segment_stack, trim_border
from dataset_gen.lunar_profiles import DEFAULT_LUNAR_PROFILES, RVEProfile


@dataclass
class SegmentedGrain:
    voxels: np.ndarray
    volume_px: int
    bbox: Tuple[int, int, int]
    voxel_size_um: float

    @property
    def equiv_diameter_um(self) -> float:
        radius_px = (3.0 * self.volume_px / (4.0 * math.pi)) ** (1.0 / 3.0)
        return float(2.0 * radius_px * self.voxel_size_um)


def load_grain_cache(cache_path: Optional[Path]) -> Optional[List[SegmentedGrain]]:
    if cache_path is None:
        return None
    if cache_path.exists():
        tqdm.write(f"[lunar_rve] Loading cached grain library from {cache_path}")
        with open(cache_path, "rb") as handle:
            data = pickle.load(handle)
        return data
    return None


def save_grain_cache(cache_path: Optional[Path], grains: Sequence[SegmentedGrain]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"[lunar_rve] Saving grain library cache to {cache_path}")
    with open(cache_path, "wb") as handle:
        pickle.dump(list(grains), handle)


def extract_grains(binary_stack: np.ndarray, voxel_size_um: float, min_volume_px: int = 64) -> List[SegmentedGrain]:
    from skimage.measure import label, regionprops

    labeled = label(binary_stack, connectivity=1)
    grains: List[SegmentedGrain] = []
    for region in regionprops(labeled):
        if region.area < min_volume_px:
            continue
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        sub = labeled[min_z:max_z, min_y:max_y, min_x:max_x] == region.label
        grains.append(
            SegmentedGrain(
                voxels=sub.astype(np.uint8),
                volume_px=int(region.area),
                bbox=(max_z - min_z, max_y - min_y, max_x - min_x),
                voxel_size_um=voxel_size_um,
            )
        )
    return grains


def build_grain_library(stack_infos: Sequence[ImageStackInfo], root_dir: Path, margin_voxels: int = 0) -> List[SegmentedGrain]:
    grains: List[SegmentedGrain] = []
    for info in tqdm(stack_infos, desc="Segmenting stacks"):
        mic_path = info.mic_path(root_dir)
        if not mic_path.exists():
            tqdm.write(f"[lunar_rve] Skipping stack '{info.name}' – file not found: {mic_path}")
            continue
        raw = load_raw_stack(info, root_dir)
        segmented = segment_stack(raw, info.phase_count)
        trimmed = trim_border(segmented, margin=margin_voxels if not info.captures_full_tube else 0)
        grains.extend(extract_grains(trimmed, info.voxel_size_um))
    return grains


def select_grain(grains: Sequence[SegmentedGrain], target_um: float, tolerance: float, rng: np.random.Generator) -> Optional[SegmentedGrain]:
    candidates: List[SegmentedGrain] = []
    threshold = tolerance * target_um
    for grain in grains:
        if abs(grain.equiv_diameter_um - target_um) <= threshold:
            candidates.append(grain)
    if not candidates:
        return None
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def place_grain_binary(
    canvas: np.ndarray,
    grain: SegmentedGrain,
    position: Tuple[int, int, int],
) -> bool:
    z0, y0, x0 = position
    gz, gy, gx = grain.voxels.shape
    z1, y1, x1 = z0 + gz, y0 + gy, x0 + gx
    if z1 > canvas.shape[0] or y1 > canvas.shape[1] or x1 > canvas.shape[2]:
        return False
    sub = canvas[z0:z1, y0:y1, x0:x1]
    if np.any(sub & grain.voxels):
        return False
    sub[:] |= grain.voxels
    return True


def generate_lunar_volume(profile: RVEProfile, grains: Sequence[SegmentedGrain], rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    grid_shape = profile.grid_shape
    volume = np.zeros(grid_shape, dtype=np.uint8)
    target_solid_voxels = int(np.prod(grid_shape) * profile.target_packing_fraction)

    families = profile.families
    family_weights = np.array([family.weight for family in families], dtype=np.float32)
    family_probs = family_weights / family_weights.sum()

    placed = 0
    attempts = 0

    while placed < target_solid_voxels and attempts < 5000:
        attempts += 1
        family_index = int(rng.choice(len(families), p=family_probs))
        family = families[family_index]
        target_diam = float(rng.lognormal(mean=math.log(family.median_diameter_um), sigma=family.sigma))
        target_diam = np.clip(target_diam, family.min_diameter_um, family.max_diameter_um)

        grain = select_grain(grains, target_diam, tolerance=0.35, rng=rng)
        if grain is None:
            continue

        gz, gy, gx = grain.voxels.shape
        margin_z = grid_shape[0] - gz
        margin_y = grid_shape[1] - gy
        margin_x = grid_shape[2] - gx
        if margin_z <= 0 or margin_y <= 0 or margin_x <= 0:
            continue

        position = (
            int(rng.integers(0, margin_z + 1)),
            int(rng.integers(0, margin_y + 1)),
            int(rng.integers(0, margin_x + 1)),
        )
        if place_grain_binary(volume, grain, position):
            placed += grain.volume_px

    metadata = {
        "target_solid_voxels": target_solid_voxels,
        "placed_voxels": int(volume.sum()),
        "attempts": attempts,
        "grains_used": int(volume.sum()),
        "profile": profile.name,
    }
    return volume, metadata


def init_worker(profile: RVEProfile, grains: Sequence[SegmentedGrain]) -> None:
    global WORKER_PROFILE, WORKER_GRAINS
    WORKER_PROFILE = profile
    WORKER_GRAINS = grains


def worker_generate(args: Tuple[int, int]) -> Tuple[int, np.ndarray, Dict]:
    idx, seed = args
    rng = np.random.default_rng(seed)
    volume, meta = generate_lunar_volume(WORKER_PROFILE, WORKER_GRAINS, rng)
    return idx, volume, meta


def save_volumes(chunks: List[torch.Tensor], output_dir: Path, index: int) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"lunar_rve_{index:04d}.pt.gz"
    with gzip.open(file_path, "wb") as handle:
        torch.save(chunks, handle)
    return str(file_path)


def generate_dataset(
    profile_name: str,
    output_dir: Path,
    sysconfig_path: Path,
    stack_dir: Path,
    num_volumes: int,
    volumes_per_file: int,
    seed: int,
    margin_voxels: int = 8,
    num_workers: int = 1,
    grain_cache: Optional[Path] = None,
) -> Dict:
    if profile_name not in DEFAULT_LUNAR_PROFILES:
        raise KeyError(f"Unknown lunar profile '{profile_name}'. Choices: {list(DEFAULT_LUNAR_PROFILES)}")

    profile = DEFAULT_LUNAR_PROFILES[profile_name]
    stack_infos = parse_sysconfig(sysconfig_path)

    grains = load_grain_cache(grain_cache)
    if grains is None:
        grains = build_grain_library(stack_infos, stack_dir, margin_voxels=margin_voxels)
        save_grain_cache(grain_cache, grains)
    else:
        tqdm.write(f"[lunar_rve] Using cached grain library with {len(grains)} grains")

    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=num_volumes, dtype=np.uint32)

    saved_paths: List[str] = []
    chunk: List[torch.Tensor] = []
    metadata: List[Dict] = []

    def handle(idx: int, volume: np.ndarray, meta: Dict) -> None:
        nonlocal chunk
        chunk.append(torch.from_numpy(volume.astype(np.uint8)))
        metadata.append(meta)
        if len(chunk) == volumes_per_file or idx == num_volumes - 1:
            out_path = save_volumes(chunk, output_dir, len(saved_paths))
            saved_paths.append(out_path)
            chunk = []

    if num_workers <= 1:
        for idx in tqdm(range(num_volumes), desc="Generating lunar RVEs"):
            rng = np.random.default_rng(int(seeds[idx]))
            volume, meta = generate_lunar_volume(profile, grains, rng)
            handle(idx, volume, meta)
    else:
        with Pool(processes=num_workers, initializer=init_worker, initargs=(profile, grains)) as pool:
            tasks = ((idx, int(seeds[idx])) for idx in range(num_volumes))
            for idx, volume, meta in tqdm(pool.imap(worker_generate, tasks), total=num_volumes, desc="Generating lunar RVEs"):
                handle(idx, volume, meta)

    summary = {
        "profile": profile_name,
        "sysconfig_path": str(sysconfig_path),
        "stack_dir": str(stack_dir),
        "num_grains_library": len(grains),
        "saved_files": saved_paths,
        "num_volumes": num_volumes,
        "volumes_per_file": volumes_per_file,
        "seed": seed,
        "metadata_per_volume": metadata,
        "profile_config": asdict(profile),
        "grain_cache": str(grain_cache) if grain_cache else None,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lunar simulant RVEs from real XCT stacks.")
    parser.add_argument("--profile", default="lunar_oprl2n_75_300", help=f"Profile name. Choices: {list(DEFAULT_LUNAR_PROFILES)}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sysconfig", type=Path, required=True)
    parser.add_argument("--stack-dir", type=Path, required=True)
    parser.add_argument("--num-volumes", type=int, default=128)
    parser.add_argument("--volumes-per-file", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin-voxels", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--grain-cache", type=Path, help="Optional path to cache segmented grains for reuse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    metadata_path = args.metadata_path or (args.output_dir / "lunar_generation_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Generated {summary['num_volumes']} lunar RVEs -> {len(summary['saved_files'])} files")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
