#!/bin/bash
#SBATCH --job-name=track-native-gen
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/native-gen-%j.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/native-gen-%j.err
set -euo pipefail

source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE=/home/muhassan/workdir/fast-gran-gen
MODELS=/home/projects/m26127/fast-gran-gen/models
OUT_ROOT=/home/projects/m26127/fast-gran-gen/outputs/track_native_physical
OUT="$OUT_ROOT/generation_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cd "$CODE"

python railway_track.py \
  --model_path "$MODELS/base" \
  --inpainting_model_path "$MODELS/inpainting" \
  --output_dir "$OUT" \
  --target_depth 0.30 \
  --target_width 4.50 \
  --target_length 4.20 \
  --inference_steps 50 \
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
  --scale-factor 1.0 \
  --depth 0.30 \
  --base-width 4.50 \
  --crest-width 3.60 \
  --length 4.20

python - "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
profile = np.load(out / "profiled_track.npy", mmap_mode="r")
expected_shape = (96, 960, 896)
if profile.shape != expected_shape:
    raise RuntimeError(f"profile shape {profile.shape} != {expected_shape}")
occupied = int(np.count_nonzero(profile))
if occupied < 5_000_000:
    raise RuntimeError(f"occupied voxel count too small: {occupied}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

record = {
    "status": "complete",
    "job_id": out.name.removeprefix("generation_"),
    "scope": "native_physical_five_sleeper_ballast",
    "uniform_scale_factor": 1.0,
    "profile_shape_depth_width_length": list(profile.shape),
    "occupied_voxels": occupied,
    "profile_sha256": sha256(out / "profiled_track.npy"),
    "profile_metadata_sha256": sha256(out / "PROFILE.json"),
}
(out / "GENERATED_OK.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

ln -sfn "$OUT" "$OUT_ROOT/latest_generation"
