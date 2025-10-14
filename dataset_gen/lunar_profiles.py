"""Profile definitions for lunar simulant RVEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from dataset_gen.lunar_lwt import load_multiple_lwt


@dataclass(frozen=True)
class GrainFamily:
    name: str
    weight: float
    median_diameter_um: float
    sigma: float
    min_diameter_um: float
    max_diameter_um: float
    aspect_ratio_range: Tuple[float, float]
    flatness_range: Tuple[float, float]
    exponent_range: Tuple[float, float]


@dataclass(frozen=True)
class RVEProfile:
    name: str
    grid_shape: Tuple[int, int, int]
    target_packing_fraction: float
    voxel_size_um: float
    families: List[GrainFamily]
    fill_fraction: float


DEFAULT_LUNAR_PROFILES: Dict[str, RVEProfile] = {
    "lunar_oprl2n_75_300": RVEProfile(
        name="lunar_oprl2n_75_300",
        grid_shape=(32, 96, 96),
        target_packing_fraction=0.54,
        voxel_size_um=3.4065,
        families=[
            GrainFamily(
                name="coarse_blocky",
                weight=0.35,
                median_diameter_um=210.0,
                sigma=0.25,
                min_diameter_um=120.0,
                max_diameter_um=320.0,
                aspect_ratio_range=(0.55, 1.25),
                flatness_range=(0.50, 1.1),
                exponent_range=(0.6, 0.9),
            ),
            GrainFamily(
                name="medium_irregular",
                weight=0.45,
                median_diameter_um=180.0,
                sigma=0.2,
                min_diameter_um=90.0,
                max_diameter_um=260.0,
                aspect_ratio_range=(0.6, 1.35),
                flatness_range=(0.55, 1.25),
                exponent_range=(0.65, 0.95),
            ),
            GrainFamily(
                name="fines",
                weight=0.20,
                median_diameter_um=120.0,
                sigma=0.18,
                min_diameter_um=75.0,
                max_diameter_um=200.0,
                aspect_ratio_range=(0.75, 1.4),
                flatness_range=(0.70, 1.35),
                exponent_range=(0.7, 1.0),
            ),
        ],
        fill_fraction=0.08,
    ),
}

