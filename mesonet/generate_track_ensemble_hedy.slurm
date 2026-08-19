#!/bin/bash
#SBATCH --job-name=track-ens-gen
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/ensemble-gen-%A_%a.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/ensemble-gen-%A_%a.err
set -euo pipefail

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT required}"
: "${CODE_ARCHIVE_SHA256:?CODE_ARCHIVE_SHA256 required}"
: "${SEED_MANIFEST_SHA256:?SEED_MANIFEST_SHA256 required}"
source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "${CAMPAIGN_ROOT}/gpu-generation-${SLURM_ARRAY_TASK_ID}.txt"

CODE=${CAMPAIGN_ROOT}/project
MODELS=/home/projects/m26127/fast-gran-gen/models
MANIFEST=${CODE}/mesonet/track_ensemble_seeds_v1.json
test "$(sha256sum "${CAMPAIGN_ROOT}/project.tar.gz" | awk '{print $1}')" = "$CODE_ARCHIVE_SHA256"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$SEED_MANIFEST_SHA256"

readarray -t VALUES < <(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" "$MODELS" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.load(open(sys.argv[1])); index=int(sys.argv[2]); models=Path(sys.argv[3])
seeds=manifest['seeds']
if len(seeds)!=len(set(seeds)) or manifest['historical_reference']['seed'] in seeds:
    raise SystemExit('seed manifest is not independent')
if index<0 or index>=len(seeds): raise SystemExit('array index out of range')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
for relative,expected in manifest['model_files'].items():
    path=models/relative
    if sha(path)!=expected: raise SystemExit(f'model mismatch: {relative}')
print(seeds[index]); print(manifest['campaign'])
PY
)
SEED=${VALUES[0]}; CAMPAIGN=${VALUES[1]}
OUT=${CAMPAIGN_ROOT}/realizations/seed_${SEED}/generation
mkdir -p "$(dirname "$OUT")"
test ! -e "$OUT"
mkdir "$OUT"
cd "$CODE"
python railway_track.py \
  --model_path "$MODELS/base" --inpainting_model_path "$MODELS/inpainting" \
  --output_dir "$OUT" --target_depth 0.30 --target_width 4.50 --target_length 4.20 \
  --inference_steps 50 --scheduler_type ddim --batch_size 5 --strip_batch_size 5 \
  --inpaint_batch_size 5 --seed "$SEED" --binary --save_format npy --output_name generated_prism
python prepare_track_profile.py --input "$OUT/generated_prism.npy" --output "$OUT/profiled_track.npy" \
  --metadata "$OUT/PROFILE.json" --scale-factor 1.0 --depth 0.30 --base-width 4.50 \
  --crest-width 3.60 --length 4.20
python - "$OUT" "$SEED" "$CAMPAIGN" "$CODE_ARCHIVE_SHA256" "$SEED_MANIFEST_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]); seed=int(sys.argv[2]); campaign=sys.argv[3]
source=np.load(out/'generated_prism.npy',mmap_mode='r'); profile=np.load(out/'profiled_track.npy',mmap_mode='r')
if profile.shape!=(96,960,896): raise SystemExit(f'profile shape {profile.shape}')
occupied=int(np.count_nonzero(profile))
if occupied<5_000_000: raise SystemExit(f'occupied voxels {occupied}')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
record={'schema':'track_realization_generation_v1','status':'generated_not_mechanically_qualified',
'campaign':campaign,'seed':seed,'slurm_job_id':str(__import__('os').environ['SLURM_JOB_ID']),
'array_task_id':int(__import__('os').environ['SLURM_ARRAY_TASK_ID']),
'code_archive_sha256':sys.argv[4],'seed_manifest_sha256':sys.argv[5],
'generated_shape_depth_width_length':list(source.shape),'profile_shape_depth_width_length':list(profile.shape),
'occupied_voxels':occupied,'generated_prism_sha256':sha(out/'generated_prism.npy'),
'profile_sha256':sha(out/'profiled_track.npy'),'profile_metadata_sha256':sha(out/'PROFILE.json')}
(out/'GENERATED_OK.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
chmod -R a-w "$OUT"
