#!/usr/bin/env python3
"""Convert cached lunar grains into LMGC90-compatible polyhedron library."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pyvista as pv
import trimesh
from trimesh import repair
from scipy.ndimage import binary_closing, binary_fill_holes
from trimesh.voxel import ops as voxel_ops

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))

from dataset_gen.lunar_rve_generator import SegmentedGrain


class MeshRepairError(ValueError):
    """Raised when a grain mesh cannot be repaired into a watertight manifold."""


@dataclass
class ExportStats:
    total_grains: int = 0
    exported: int = 0
    skipped_too_small: int = 0
    skipped_failed_mc: int = 0
    skipped_zero_volume: int = 0
    skipped_too_complex: int = 0
    skipped_open_edges: int = 0
    skipped_non_manifold: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def load_grain_cache(path: Path) -> Sequence[SegmentedGrain]:
    import pickle

    with open(path, "rb") as handle:
        grains = pickle.load(handle)
    if not isinstance(grains, Sequence):
        raise TypeError(f"Unexpected grain cache type: {type(grains)!r}")
    return grains


def pick_indices(total: int, limit: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if limit <= 0 or limit >= total:
        perm = np.arange(total)
        rng.shuffle(perm)
        return perm
    return rng.choice(total, size=limit, replace=False)


def to_pyvista_mesh(grain: SegmentedGrain, *, smoothing_sigma: float = 0.0) -> pv.PolyData:
    mask = grain.voxels.astype(bool)
    if mask.sum() == 0:
        raise ValueError("Grain has zero voxels")

    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    padded = np.pad(mask, 1, mode="constant")
    pitch = grain.voxel_size_um * 1e-6

    tm = voxel_ops.matrix_to_marching_cubes(padded, pitch=pitch)
    if tm.vertices.size == 0 or tm.faces.size == 0:
        raise ValueError("Empty mesh from marching cubes")

    best_component = None
    for piece in tm.split(only_watertight=False):
        candidate = piece.copy()
        candidate.apply_translation(-candidate.centroid)
        candidate.merge_vertices()
        candidate.update_faces(candidate.unique_faces())
        candidate.remove_unreferenced_vertices()
        candidate.update_faces(candidate.nondegenerate_faces())
        candidate.remove_unreferenced_vertices()
        repair.fill_holes(candidate)
        repair.fix_normals(candidate)
        candidate.process(validate=True)
        if not candidate.is_watertight:
            continue
        if best_component is None or candidate.volume > best_component.volume:
            best_component = candidate

    if best_component is None:
        raise MeshRepairError("Could not obtain watertight component")

    tm = best_component
    target_faces = min(len(tm.faces), 1500)
    if target_faces > 0 and target_faces < len(tm.faces):
        try:
            simplified = tm.simplify_quadric_decimation(target_faces)
        except ModuleNotFoundError:
            simplified = None
        if simplified is not None and simplified.is_watertight:
            tm = simplified

    mesh = pv.PolyData(
        tm.vertices.astype(np.float64),
        np.hstack([np.full((tm.faces.shape[0], 1), 3, dtype=np.int32), tm.faces.astype(np.int32)]),
    )
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=8, relaxation_factor=0.05)
    mesh = mesh.compute_normals(auto_orient_normals=True, flip_normals=False)
    mesh = mesh.clean()

    if mesh.n_points < 4 or mesh.volume <= 0.0:
        raise ValueError("Mesh is degenerate after cleaning")

    return mesh


def simplify_mesh(
    mesh: pv.PolyData,
    *,
    target_triangles: int,
    max_faces: int,
    preserve_topology: bool = False,
) -> pv.PolyData:
    work = mesh

    def _decimate(current: pv.PolyData, target_count: int, slack: float = 0.0) -> pv.PolyData:
        faces_now = max(1, current.n_faces)
        if faces_now <= target_count:
            return current
        reduction = 1.0 - target_count / float(faces_now)
        reduction = min(max(reduction + slack, 0.0), 0.99)
        try:
            reduced = current.decimate_pro(
                target_reduction=reduction,
                preserve_topology=preserve_topology,
            )
            reduced.clean(inplace=True)
            if reduced.n_faces >= 4:
                return reduced
        except Exception:
            return current
        return current

    work = _decimate(work, target_triangles)

    iterations = 0
    while work.n_faces > max_faces and iterations < 6:
        work = _decimate(work, max_faces, slack=0.05)
        iterations += 1
    return work


def mesh_to_record(
    mesh: pv.PolyData,
    *,
    grain_id: int,
    scale_to_mm: bool = True,
) -> Tuple[int, dict]:
    clean_mesh = mesh.triangulate().clean()
    verts = clean_mesh.points.copy()
    faces = clean_mesh.faces.reshape((-1, 4))[:, 1:].astype(int)

    centroid = verts.mean(axis=0)
    verts_centered = verts.copy()

    if scale_to_mm:
        # verts are currently in meters; convert to millimeters to match LMGC90 datasets
        verts_centered = verts_centered * 1e3
        centroid = centroid * 1e3
        volume = clean_mesh.volume * 1e9  # m^3 -> mm^3
    else:
        volume = clean_mesh.volume

    record = {
        "id": grain_id,
        "vertices": verts_centered.tolist(),
        "faces": faces.tolist(),
        "centroid": centroid.tolist(),
        "n_vertices": int(clean_mesh.n_points),
        "n_faces": int(clean_mesh.n_faces),
        "volume": float(volume),
        "voxel_count": int(clean_mesh.n_cells),
    }
    return grain_id, record


def export_library(
    grains: Sequence[SegmentedGrain],
    *,
    output_path: Path,
    max_grains: int,
    seed: int | None,
    min_volume_px: int,
    target_triangles: int,
    max_faces: int,
) -> dict:
    stats = ExportStats(total_grains=len(grains))
    chosen_indices = pick_indices(len(grains), max_grains, seed)

    records: List[Tuple[int, dict]] = []
    for out_idx, grain_idx in enumerate(chosen_indices, start=1):
        grain = grains[grain_idx]
        if grain.volume_px < min_volume_px:
            stats.skipped_too_small += 1
            continue
        try:
            mesh = to_pyvista_mesh(grain)
        except MeshRepairError:
            stats.skipped_non_manifold += 1
            continue
        except ValueError:
            stats.skipped_failed_mc += 1
            continue
        if mesh.volume <= 0:
            stats.skipped_zero_volume += 1
            continue
        mesh = simplify_mesh(mesh, target_triangles=target_triangles, max_faces=max_faces)
        if mesh.n_faces > max_faces:
            stats.skipped_too_complex += 1
            continue
        grain_id, record = mesh_to_record(mesh, grain_id=out_idx)
        record["source_index"] = int(grain_idx)
        record["voxel_volume_um"] = float(grain.voxel_size_um ** 3)
        records.append((grain_id, record))
        stats.exported += 1

    metadata = {
        "source": str(output_path),
        "grain_cache_size": len(grains),
        "selection_limit": max_grains,
        "seed": seed,
        "min_volume_px": min_volume_px,
        "target_triangles": target_triangles,
        "max_faces": max_faces,
        "stats": stats.to_dict(),
    }

    payload = {
        "metadata": metadata,
        "polyhedrons": {str(idx): rec for idx, rec in records},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grain_cache", type=Path, help="Path to grain_cache.pkl")
    parser.add_argument("output", type=Path, help="Destination JSON file")
    parser.add_argument("--max-grains", type=int, default=2000, help="Maximum grains to export")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument(
        "--min-volume-px",
        type=int,
        default=256,
        help="Skip grains with fewer solid voxels than this threshold",
    )
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=1500,
        help="Simplify meshes to approximately this many triangles",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=6000,
        help="Discard grains whose meshes still exceed this many triangular faces after simplification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grains = load_grain_cache(args.grain_cache)
    metadata = export_library(
        grains,
        output_path=args.output,
        max_grains=args.max_grains,
        seed=args.seed,
        min_volume_px=args.min_volume_px,
        target_triangles=args.target_triangles,
        max_faces=args.max_faces,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
