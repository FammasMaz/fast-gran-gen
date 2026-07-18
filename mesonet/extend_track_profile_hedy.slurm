#!/bin/bash
#SBATCH --job-name=track-extend
#SBATCH --account=m26127
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/projects/m26127/fast-gran-gen/logs/extend-%j.out
#SBATCH --error=/home/projects/m26127/fast-gran-gen/logs/extend-%j.err
set -euo pipefail
source /home/muhassan/.venvs/granulai-redo/bin/activate
SRC=/home/projects/m26127/fast-gran-gen/outputs/track_native_physical/generation_7849/generated_prism.npy
OUT=/home/projects/m26127/fast-gran-gen/outputs/track_native_physical/extended_${SLURM_JOB_ID}
mkdir -p "$OUT"
python - "$SRC" "$OUT/extended.npy" <<'PY'
import sys,numpy as np
src=np.load(sys.argv[1],mmap_mode='r')
target=(112,1440,1344)
out=np.zeros(target,dtype=np.uint8)
# Center source; fill only missing horizontal bands with deterministic rolled copies.
z=min(src.shape[0],target[0]); y0=(target[1]-src.shape[1])//2; x0=(target[2]-src.shape[2])//2
core=(np.asarray(src[:z])>0.5).astype(np.uint8); out[:z,y0:y0+src.shape[1],x0:x0+src.shape[2]]=core
if y0:
 left=np.roll(core[:,-y0:,:],shift=137,axis=2); right=np.roll(core[:,:y0,:],shift=311,axis=2)
 out[:z,:y0,x0:x0+src.shape[2]]=left; out[:z,y0+src.shape[1]:,x0:x0+src.shape[2]]=right
if x0:
 center=out[:z,:,x0:x0+src.shape[2]]
 out[:z,:,:x0]=np.roll(center[:,:,-x0:],shift=173,axis=1)
 out[:z,:,x0+src.shape[2]:]=np.roll(center[:,:,:x0],shift=337,axis=1)
# Save exact full generated envelope; profile helper performs target crop and shoulder mask.
np.save(sys.argv[2],out)
print('source',src.shape,'extended',out.shape,'occupancy',out.mean())
PY
cd /home/muhassan/workdir/fast-gran-gen
python prepare_track_profile.py --input "$OUT/extended.npy" --output "$OUT/profiled_track.npy" --metadata "$OUT/PROFILE.json" --scale-factor 1.0 --isotropic-voxel-size 0.003125 --depth 0.30 --base-width 4.50 --crest-width 3.60 --length 4.20
python - "$OUT" <<'PY'
import hashlib,json,sys,numpy as np
from pathlib import Path
out=Path(sys.argv[1]); p=np.load(out/'profiled_track.npy',mmap_mode='r'); rec=json.loads((out/'PROFILE.json').read_text()); rec['status']='complete'; rec['source_method']='diffusion_generated_core_with_periodic_seam_extension'; rec['occupied_voxels']=int(np.count_nonzero(p)); (out/'EXTENDED_OK.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2))
PY
