#!/bin/bash
#SBATCH --job-name=track-ens-seg
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/ensemble-seg-%A_%a.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/ensemble-seg-%A_%a.err
set -euo pipefail

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT required}"
: "${CODE_ARCHIVE_SHA256:?CODE_ARCHIVE_SHA256 required}"
: "${SEED_MANIFEST_SHA256:?SEED_MANIFEST_SHA256 required}"
source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "${CAMPAIGN_ROOT}/gpu-segmentation-${SLURM_ARRAY_TASK_ID}.txt"
CODE=${CAMPAIGN_ROOT}/project
MANIFEST=${CODE}/mesonet/track_ensemble_seeds_v1.json
test "$(sha256sum "${CAMPAIGN_ROOT}/project.tar.gz"|awk '{print $1}')" = "$CODE_ARCHIVE_SHA256"
test "$(sha256sum "$MANIFEST"|awk '{print $1}')" = "$SEED_MANIFEST_SHA256"
SEED=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['seeds'][int(sys.argv[2])])
PY
)
SOURCE=${CAMPAIGN_ROOT}/realizations/seed_${SEED}/generation
OUT=${CAMPAIGN_ROOT}/realizations/seed_${SEED}/segmentation
GEN=${SOURCE}/GENERATED_OK.json
test -s "$GEN"; test ! -e "$OUT"; mkdir "$OUT"
python - "$GEN" "$SEED" "$CODE_ARCHIVE_SHA256" "$SEED_MANIFEST_SHA256" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='generated_not_mechanically_qualified' and x['seed']==int(sys.argv[2])
assert x['code_archive_sha256']==sys.argv[3] and x['seed_manifest_sha256']==sys.argv[4]
PY
cd "$CODE"
python polyhedron_segmentation.py --input "$SOURCE/profiled_track.npy" --output "$OUT/physical20.json" \
  --voxel-spacing 0.003125 0.0046875 0.0046875 --origin 0.0 -2.25 -2.10 \
  --method watershed_sdf --marker-threshold-percentile 90 --min-distance 2 \
  --erosion-iterations 0 --sdf-scale 3.0 --gaussian-sigma 0.3 --decimation-ratio 0.50 \
  --smoothing-iterations 2 --target-vertices 20 --min-polyhedron-size 40 --force-cpu \
  --fast-mesh-extraction --stream-batch-size 0 --batch-mesh-size 0 --num-workers 16 \
  --num-export-workers 16 --export-batch-size 400 --max-chunk-workers 16 \
  --fast-paraview-export --ultra-fast-mode --no-paraview-multiblock --fix-mesh-for-lmgc90 \
  --skip-paraview-export --max-voxel-aspect-ratio 0 --no-remove-boundary-polyhedrons \
  --no-enable-mesh-range-outlier-filter
python - "$SOURCE" "$OUT" "$SEED" "$CODE_ARCHIVE_SHA256" "$SEED_MANIFEST_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
source=Path(sys.argv[1]); out=Path(sys.argv[2]); seed=int(sys.argv[3])
profile=np.load(source/'profiled_track.npy',mmap_mode='r'); payload=json.loads((out/'physical20.json').read_text()); grains=payload['polyhedrons']
spacing=np.array([0.003125,0.0046875,0.0046875]); occupied_volume=float(np.count_nonzero(profile)*np.prod(spacing))
mesh_volume=float(sum(abs(float(g.get('volume',0.0))) for g in grains.values())); diameters=[]; vertices=[]
for key,g in grains.items():
 v=np.asarray(g['vertices'],dtype=float)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all(): raise SystemExit(f'invalid grain {key}')
 c=v.mean(0); diameters.append(2*np.linalg.norm(v-c,axis=1).max()); vertices.append(len(v))
retained=mesh_volume/occupied_volume; correction=retained**(-1/3); corrected=np.asarray(diameters)*correction
if not 0.82<=retained<=1.20: raise SystemExit(f'volume ratio {retained}')
if not 0.060<=corrected.mean()<=0.085: raise SystemExit(f'mean diameter {corrected.mean()}')
if len(grains)<50_000: raise SystemExit(f'particle count {len(grains)}')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
record={'schema':'track_realization_segmentation_v1','status':'segmented_not_mechanically_qualified','seed':seed,
'slurm_job_id':str(__import__('os').environ['SLURM_JOB_ID']),'array_task_id':int(__import__('os').environ['SLURM_ARRAY_TASK_ID']),
'code_archive_sha256':sys.argv[4],'seed_manifest_sha256':sys.argv[5],'particle_count':len(grains),
'occupied_voxel_volume_m3':occupied_volume,'mesh_volume_m3':mesh_volume,'mesh_to_voxel_volume_ratio':retained,
'lmgc90_volume_correction_scale':correction,'corrected_mean_diameter_m':float(corrected.mean()),
'diameter_m_quantiles_after_volume_correction':np.quantile(corrected,[0,.05,.5,.95,1]).tolist(),
'median_vertices':float(np.median(vertices)),'profile_sha256':sha(source/'profiled_track.npy'),
'polyhedra_sha256':sha(out/'physical20.json')}
(out/'SEGMENTED_OK.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
chmod -R a-w "$OUT"
