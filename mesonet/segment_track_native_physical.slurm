#!/bin/bash
#SBATCH --job-name=track-native-seg
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/native-seg-%j.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/native-seg-%j.err
set -euo pipefail

source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1

CODE=/home/muhassan/workdir/fast-gran-gen
SOURCE=${SOURCE:?SOURCE generation directory is required}
OUT_ROOT=/home/projects/m26127/fast-gran-gen/outputs/track_native_physical
OUT="$OUT_ROOT/segmentation_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cd "$CODE"

python polyhedron_segmentation.py \
  --input "$SOURCE/profiled_track.npy" \
  --output "$OUT/physical20.json" \
  --voxel-spacing 0.003125 0.003125 0.003125 \
  --origin 0.0 -2.25 -2.10 \
  --method watershed_sdf \
  --marker-threshold-percentile 90 \
  --min-distance 2 \
  --erosion-iterations 0 \
  --sdf-scale 3.0 \
  --gaussian-sigma 0.3 \
  --decimation-ratio 0.50 \
  --smoothing-iterations 2 \
  --target-vertices 20 \
  --min-polyhedron-size 40 \
  --force-cpu \
  --fast-mesh-extraction \
  --stream-batch-size 0 \
  --batch-mesh-size 0 \
  --num-workers 16 \
  --num-export-workers 16 \
  --export-batch-size 400 \
  --max-chunk-workers 16 \
  --fast-paraview-export \
  --ultra-fast-mode \
  --no-paraview-multiblock \
  --fix-mesh-for-lmgc90 \
  --skip-paraview-export \
  --max-voxel-aspect-ratio 0 \
  --no-remove-boundary-polyhedrons \
  --no-enable-mesh-range-outlier-filter

python - "$SOURCE" "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
out = Path(sys.argv[2])
profile = np.load(source / "profiled_track.npy", mmap_mode="r")
payload = json.loads((out / "physical20.json").read_text())
grains = payload["polyhedrons"]
spacing = np.array([0.003125, 0.003125, 0.003125])
occupied_volume = float(np.count_nonzero(profile) * np.prod(spacing))
mesh_volume = float(sum(abs(float(g.get("volume", 0.0))) for g in grains.values()))
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
scale_to_72mm = 0.072 / float(diameters.mean())
physical_diameters = diameters * scale_to_72mm
retained_ratio = mesh_volume / occupied_volume
if not 0.90 <= retained_ratio <= 1.10:
    raise RuntimeError(f"volume ratio out of bounds: {retained_ratio}")
if not 0.070 <= physical_diameters.mean() <= 0.074:
    raise RuntimeError(f"mean diameter invalid: {physical_diameters.mean()}")
if len(grains) < 50_000:
    raise RuntimeError(f"particle count too small: {len(grains)}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

record = {
    "status": "complete",
    "job_id": out.name.removeprefix("segmentation_"),
    "scope": "native_physical_five_sleeper_ballast",
    "source_generation": str(source),
    "particle_count": len(grains),
    "occupied_voxel_volume_m3": occupied_volume,
    "mesh_volume_m3": mesh_volume,
    "mesh_to_voxel_volume_ratio": retained_ratio,
    "raw_mean_diameter_m": float(diameters.mean()),
    "lmgc90_scale_to_mean_diameter_0p072": scale_to_72mm,
    "diameter_m_quantiles_after_scale": np.quantile(physical_diameters, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
    "median_vertices": float(np.median(vertices_per_grain)),
    "source_axis_bounds_m": {
        "minimum": np.min(np.asarray(all_min), axis=0).tolist(),
        "maximum": np.max(np.asarray(all_max), axis=0).tolist(),
    },
    "polyhedra_sha256": sha256(out / "physical20.json"),
    "profile_sha256": sha256(source / "profiled_track.npy"),
}
(out / "SEGMENTED_OK.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

ln -sfn "$OUT" "$OUT_ROOT/latest_segmentation"
