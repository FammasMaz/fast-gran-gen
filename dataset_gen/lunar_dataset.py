"""Utilities for parsing lunar simulant microstructure stacks.

This module provides helpers for working with the NIST lunar simulant database
subset that contains voxelized micro-CT image stacks (.mic.gz) and associated
configuration descriptors (.dat). The goal is to expose lightweight primitives
that the dataset generation scripts can use to construct representative volume
elements (RVEs) compatible with the existing diffusion training pipeline.

The parsing code intentionally avoids heavy dependencies so it can be reused in
both preprocessing notebooks and command-line utilities.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter
from skimage.filters import threshold_multiotsu, threshold_otsu


@dataclass(frozen=True)
class ImageStackInfo:
    """Metadata describing a single XCT image stack."""

    name: str
    nx: int
    ny: int
    nz: int
    voxel_size_um: float
    captures_full_tube: bool
    phase_count: int

    def mic_path(self, root_dir: Path) -> Path:
        return root_dir / f"{self.name}.mic.gz"


def parse_sysconfig(sysconfig_path: Path) -> List[ImageStackInfo]:
    """Parse a ``*-sysconfig.dat`` file into structured records."""

    records: List[ImageStackInfo] = []
    with open(sysconfig_path, "r", encoding="ascii", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 7:
                continue
            name = tokens[0]
            try:
                nx, ny, nz = (int(tokens[1]), int(tokens[2]), int(tokens[3]))
                voxel_size_um = float(tokens[4])
                capture_flag = int(tokens[5])
                phase_count = int(tokens[6])
            except ValueError:
                continue
            info = ImageStackInfo(
                name=name,
                nx=nx,
                ny=ny,
                nz=nz,
                voxel_size_um=voxel_size_um,
                captures_full_tube=(capture_flag == 0),
                phase_count=max(2, phase_count),
            )
            records.append(info)
    return records


def _iter_voxel_values(mic_path: Path, total_voxels: int) -> Iterator[int]:
    with gzip.open(mic_path, "rt", encoding="ascii", errors="ignore") as handle:
        for idx, raw in enumerate(handle):
            if idx >= total_voxels:
                break
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield int(stripped)
            except ValueError:
                continue


def load_raw_stack(info: ImageStackInfo, root_dir: Path, cache_dir: Optional[Path] = None) -> np.ndarray:
    """Load the raw grayscale XCT stack for ``info``."""

    mic_path = info.mic_path(root_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{info.name}_raw.npy"
        if cache_path.exists():
            array = np.load(cache_path, mmap_mode="r")
            return array

    total_voxels = info.nx * info.ny * info.nz
    flat = np.fromiter(_iter_voxel_values(mic_path, total_voxels), dtype=np.uint8, count=total_voxels)
    if flat.size != total_voxels:
        raise ValueError(f"Stack {info.name} expected {total_voxels} voxels, found {flat.size}")

    volume = flat.reshape((info.nz, info.ny, info.nx))
    volume = volume[:, ::-1, :]

    if cache_dir is not None:
        np.save(cache_dir / f"{info.name}_raw.npy", volume)
        volume = np.load(cache_dir / f"{info.name}_raw.npy", mmap_mode="r")
    return volume


def segment_stack(
    volume: np.ndarray,
    phase_count: int,
    solid_classes: Optional[Sequence[int]] = None,
    gaussian_sigma: float = 0.5,
    closing_iterations: int = 1,
    opening_iterations: int = 0,
) -> np.ndarray:
    """Segment a grayscale stack into a binary solid mask."""

    data = volume.astype(np.float32)
    if gaussian_sigma > 0.0:
        data = gaussian_filter(data, sigma=gaussian_sigma)

    if phase_count <= 2:
        threshold = threshold_otsu(data)
        labels = (data >= threshold).astype(np.uint8)
        unique_labels = [0, 1]
    else:
        thresholds = threshold_multiotsu(data, classes=phase_count)
        labels = np.digitize(data, thresholds)
        unique_labels = list(range(phase_count))

    if solid_classes is None:
        solid_classes = [unique_labels[-1]]
    solid_mask = np.isin(labels, solid_classes)

    if closing_iterations > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for _ in range(closing_iterations):
            solid_mask = binary_closing(solid_mask, structure=structure)

    if opening_iterations > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for _ in range(opening_iterations):
            solid_mask = binary_opening(solid_mask, structure=structure)

    return solid_mask.astype(np.uint8)


def trim_border(binary_stack: np.ndarray, margin: int) -> np.ndarray:
    """Zero out a safety margin near the cylindrical tube boundary."""

    if margin <= 0:
        return binary_stack
    trimmed = binary_stack.copy()
    trimmed[:margin, :, :] = 0
    trimmed[-margin:, :, :] = 0
    trimmed[:, :margin, :] = 0
    trimmed[:, -margin:, :] = 0
    trimmed[:, :, :margin] = 0
    trimmed[:, :, -margin:] = 0
    return trimmed


def compute_grain_volume(binary_stack: np.ndarray, voxel_size_um: float) -> float:
    """Return the physical volume in cubic millimeters of solid voxels."""

    voxel_volume_um3 = voxel_size_um ** 3
    return float(binary_stack.sum() * voxel_volume_um3 * 1e-9)
