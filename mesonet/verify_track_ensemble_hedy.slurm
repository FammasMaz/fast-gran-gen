#!/bin/bash
#SBATCH --job-name=track-ens-verify
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/ensemble-verify-%j.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/ensemble-verify-%j.err
set -euo pipefail

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT required}"
: "${CODE_ARCHIVE_SHA256:?CODE_ARCHIVE_SHA256 required}"
: "${SEED_MANIFEST_SHA256:?SEED_MANIFEST_SHA256 required}"
source /home/muhassan/.venvs/granulai-redo/bin/activate
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "${CAMPAIGN_ROOT}/gpu-verification.txt"
python - "$CAMPAIGN_ROOT" "$CODE_ARCHIVE_SHA256" "$SEED_MANIFEST_SHA256" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); archive_sha=sys.argv[2]; manifest_sha=sys.argv[3]
manifest_path=root/'project/mesonet/track_ensemble_seeds_v1.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
if sha(root/'project.tar.gz')!=archive_sha or sha(manifest_path)!=manifest_sha: raise SystemExit('campaign identity mismatch')
manifest=json.load(open(manifest_path)); rows=[]
for index,seed in enumerate(manifest['seeds']):
 base=root/f'realizations/seed_{seed}'; gen_path=base/'generation/GENERATED_OK.json'; seg_path=base/'segmentation/SEGMENTED_OK.json'
 gen=json.load(open(gen_path)); seg=json.load(open(seg_path))
 if gen['seed']!=seed or seg['seed']!=seed: raise SystemExit(f'seed mismatch {seed}')
 if gen['array_task_id']!=index or seg['array_task_id']!=index: raise SystemExit(f'array mismatch {seed}')
 if gen['code_archive_sha256']!=archive_sha or seg['code_archive_sha256']!=archive_sha: raise SystemExit(f'code mismatch {seed}')
 if gen['seed_manifest_sha256']!=manifest_sha or seg['seed_manifest_sha256']!=manifest_sha: raise SystemExit(f'manifest mismatch {seed}')
 if gen['profile_sha256']!=seg['profile_sha256']: raise SystemExit(f'profile mismatch {seed}')
 if sha(base/'generation/profiled_track.npy')!=gen['profile_sha256']: raise SystemExit(f'profile bytes mismatch {seed}')
 if sha(base/'segmentation/physical20.json')!=seg['polyhedra_sha256']: raise SystemExit(f'polyhedra bytes mismatch {seed}')
 rows.append({'seed':seed,'generation_job_id':gen['slurm_job_id'],'segmentation_job_id':seg['slurm_job_id'],
 'profile_sha256':gen['profile_sha256'],'polyhedra_sha256':seg['polyhedra_sha256'],'particle_count':seg['particle_count'],
 'corrected_mean_diameter_m':seg['corrected_mean_diameter_m'],'mesh_to_voxel_volume_ratio':seg['mesh_to_voxel_volume_ratio']})
reference=manifest['historical_reference']
for field in ('profile_sha256','polyhedra_sha256'):
 values=[row[field] for row in rows]
 if len(values)!=len(set(values)): raise SystemExit(f'duplicate {field}')
 if reference[field] in values: raise SystemExit(f'{field} duplicates historical realization')
record={'schema':'track_realization_ensemble_v1','status':'independent_segmented_realizations_ready_for_mechanical_preflight',
'acceptance_authority':False,'mechanically_qualified':False,'test_or_model_evaluated':False,
'campaign':manifest['campaign'],'historical_reference':reference,
'code_archive_sha256':archive_sha,'seed_manifest_sha256':manifest_sha,
'verifier_job_id':os.environ['SLURM_JOB_ID'],'realizations':rows}
tmp=root/'ENSEMBLE_READY.json.tmp'; tmp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); os.replace(tmp,root/'ENSEMBLE_READY.json')
print(json.dumps(record,indent=2,sort_keys=True))
PY
chmod a-w "$CAMPAIGN_ROOT/ENSEMBLE_READY.json"
