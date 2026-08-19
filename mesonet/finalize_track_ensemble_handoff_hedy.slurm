#!/bin/bash
#SBATCH --job-name=track-ens-final
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/ensemble-final-%A_%a.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/ensemble-final-%A_%a.err
set -euo pipefail

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT required}"
: "${SOURCE_CODE_ARCHIVE_SHA256:?SOURCE_CODE_ARCHIVE_SHA256 required}"
: "${FINALIZER_CODE_ARCHIVE_SHA256:?FINALIZER_CODE_ARCHIVE_SHA256 required}"
: "${SEED_MANIFEST_SHA256:?SEED_MANIFEST_SHA256 required}"
source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "${CAMPAIGN_ROOT}/gpu-finalization-${SLURM_ARRAY_TASK_ID}.txt"
CODE=${CAMPAIGN_ROOT}/finalizer-project
MANIFEST=${CODE}/mesonet/track_ensemble_seeds_v1.json
test "$(sha256sum "${CAMPAIGN_ROOT}/project.tar.gz"|awk '{print $1}')" = "$SOURCE_CODE_ARCHIVE_SHA256"
test "$(sha256sum "${CAMPAIGN_ROOT}/finalizer-project.tar.gz"|awk '{print $1}')" = "$FINALIZER_CODE_ARCHIVE_SHA256"
test "$(sha256sum "$MANIFEST"|awk '{print $1}')" = "$SEED_MANIFEST_SHA256"
SEED=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['seeds'][int(sys.argv[2])])
PY
)
ROOT=${CAMPAIGN_ROOT}/realizations/seed_${SEED}
SEG=${ROOT}/segmentation
OUT=${ROOT}/handoff
MARKER=${SEG}/SEGMENTED_OK.json
test -s "$MARKER"; test ! -e "$OUT"; mkdir "$OUT"
python - "$SEG" "$OUT" "$SEED" "$SOURCE_CODE_ARCHIVE_SHA256" "$FINALIZER_CODE_ARCHIVE_SHA256" "$SEED_MANIFEST_SHA256" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
seg,out=Path(sys.argv[1]),Path(sys.argv[2]); seed=int(sys.argv[3])
marker=json.loads((seg/'SEGMENTED_OK.json').read_text())
assert marker['status']=='segmented_not_mechanically_qualified' and marker['seed']==seed
assert marker['code_archive_sha256']==sys.argv[4]
assert marker['seed_manifest_sha256']==sys.argv[6]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
source=seg/'physical20.json'
assert sha(source)==marker['polyhedra_sha256']
payload=json.loads(source.read_text()); grains=payload['polyhedrons']; scale=float(marker['lmgc90_volume_correction_scale'])
diameters=[]
for key in sorted(grains,key=lambda x:int(x)):
 grain=grains[key]; vertices=np.asarray(grain['vertices'],dtype=float); center=vertices.mean(axis=0)
 corrected=center+scale*(vertices-center); grain['vertices']=corrected.tolist()
 if 'volume' in grain: grain['volume']=float(grain['volume'])*scale**3
 diameters.append(2*np.linalg.norm(corrected-corrected.mean(axis=0),axis=1).max())
target=out/'physical_corrected.json'; target.write_text(json.dumps(payload,separators=(',',':'))+'\n')
mesh_volume=float(sum(abs(float(g.get('volume',0.0))) for g in grains.values()))
ratio=mesh_volume/float(marker['occupied_voxel_volume_m3']); mean=float(np.mean(diameters))
if not 0.999<=ratio<=1.001: raise SystemExit(f'corrected volume ratio {ratio}')
if not 0.060<=mean<=0.085: raise SystemExit(f'corrected mean diameter {mean}')
record={'schema':'track_realization_handoff_v1','status':'mechanical_preflight_input_ready_not_mechanically_qualified',
'seed':seed,'slurm_job_id':str(os.environ['SLURM_JOB_ID']),'array_task_id':int(os.environ['SLURM_ARRAY_TASK_ID']),
'source_code_archive_sha256':sys.argv[4],'finalizer_code_archive_sha256':sys.argv[5],
'seed_manifest_sha256':sys.argv[6],
'segmentation_marker_sha256':sha(seg/'SEGMENTED_OK.json'),'source_polyhedra_sha256':marker['polyhedra_sha256'],
'profile_sha256':marker['profile_sha256'],'particle_count':len(grains),'applied_centroid_scale':scale,
'corrected_mesh_to_voxel_volume_ratio':ratio,'corrected_mean_diameter_m':mean,
'corrected_diameter_m_quantiles':np.quantile(diameters,[0,.05,.5,.95,1]).tolist(),
'corrected_polyhedra_sha256':sha(target)}
(out/'HANDOFF_READY.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
chmod -R a-w "$OUT"
