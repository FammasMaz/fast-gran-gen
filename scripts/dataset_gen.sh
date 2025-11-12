#!/bin/bash 
cd /Users/fammasmaz/Downloads/fast-gran-gen

VTU_DIR="/Users/fammasmaz/Downloads/samples/vtus"
OUT_DIR="/Users/fammasmaz/Downloads/samples/voxels"

mkdir -p "$OUT_DIR"

for vtu in "$VTU_DIR"/*.vtu; do
    stem=$(basename "${vtu%.*}")
    python dataset_gen/voxelizer.py \
      --vtu_file "$vtu" \
      --num_slices 32 \
      --img_size 64 64 \
      --out_dir "$OUT_DIR/$stem" \
      --num_workers 8 \
      --stem "column_" \
      --save_voxel
done
