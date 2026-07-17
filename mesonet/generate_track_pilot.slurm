#!/bin/bash
#SBATCH --job-name=track-diffusion
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/track-%j.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/track-%j.err
set -euo pipefail

source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/home/muhassan/workdir/fast-gran-gen
MODELS=/home/projects/m26127/fast-gran-gen/models
export OUT=/home/projects/m26127/fast-gran-gen/outputs/track_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$CODE"

# Generate one-third-scale learned geometry, then uniformly enlarge 3x after
# segmentation. This is a full-geometry coarse LMGC90 qualification pilot.
python railway_track.py \
  --model_path "$MODELS/base" \
  --inpainting_model_path "$MODELS/inpainting" \
  --output_dir "$OUT" \
  --target_depth 0.1 \
  --target_width 1.5 \
  --target_length 1.5 \
  --inference_steps 25 \
  --scheduler_type ddim \
  --batch_size 5 \
  --strip_batch_size 5 \
  --inpaint_batch_size 5 \
  --seed 25100 \
  --binary \
  --save_format npy \
  --output_name generated_prism

python prepare_track_profile.py \
  --input "$OUT/generated_prism.npy" \
  --output "$OUT/profiled_track.npy" \
  --metadata "$OUT/PROFILE.json" \
  --scale-factor 3.0 \
  --depth 0.30 \
  --base-width 4.50 \
  --crest-width 3.60 \
  --length 4.20

python polyhedron_segmentation.py \
  --input "$OUT/profiled_track.npy" \
  --output "$OUT/profiled_track_polyhedra.json" \
  --voxel-spacing 0.009375 0.0140625 0.0140625 \
  --origin 0.0 -2.25 -2.10 \
  --method watershed_sdf \
  --marker-threshold-percentile 90 \
  --min-distance 2 \
  --erosion-iterations 0 \
  --sdf-scale 3.0 \
  --gaussian-sigma 0.3 \
  --decimation-ratio 0.85 \
  --smoothing-iterations 8 \
  --target-vertices 12 \
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
  --max-voxel-aspect-ratio 0

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import numpy as np

out = Path(os.environ["OUT"])
profile = np.load(out / "profiled_track.npy", mmap_mode="r")
payload = json.loads((out / "profiled_track_polyhedra.json").read_text())
polyhedra = payload.get("polyhedrons", {})
assert profile.shape == (32, 320, 299), profile.shape
assert len(polyhedra) >= 500, len(polyhedra)
for key, grain in polyhedra.items():
    vertices = np.asarray(grain["vertices"], dtype=float)
    assert vertices.ndim == 2 and vertices.shape[1] == 3, (key, vertices.shape)
    assert np.isfinite(vertices).all(), key

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

record = {
    "status": "complete",
    "job_id": os.environ["SLURM_JOB_ID"],
    "scope": "full_geometry_coarse_diffusion_pilot",
    "not_final_grain_resolution": True,
    "profile_shape": list(profile.shape),
    "polyhedron_count": len(polyhedra),
    "profile_sha256": sha256(out / "profiled_track.npy"),
    "polyhedra_sha256": sha256(out / "profiled_track_polyhedra.json"),
}
(out / "GENERATED_OK.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
