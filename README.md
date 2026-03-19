# Fast 3D Diffusion for Scalable Granular Media Synthesis

This repository contains the official implementation of "Fast Granular Media Generation" as described in the associated NeurIPS paper.

## Overview

Our method enables efficient generation of 3D granular media using a diffusion-based approach. The implementation relies on both unconditional generation and inpainting modes for creating realistic granular structures.

## Installation

```bash
# Clone the repository
[GIT LINK WILL BE PROVIDED]
cd fast-gran-gen

# Install dependencies (if using pip)
pip install -r requirements.txt
```

## Anonymized Dataset

We provide the granular media dataset used in our experiments.

- **Anonymized Download Link**: [Dataset Folder](https://drive.google.com/drive/folders/1MjvkpJ_X7L3eeRcRxGLsb9iq2RUEPRUN?usp=share_link)
- **Anonymized Main Dataset File**: `voxels_int.h5` in the main (voxels_shrink) folder

For quick evaluation, we provide cached data:
- **Anonymized Cache Folders**: [Quick Evaluation Cache](https://drive.google.com/drive/folders/1heBGT45_fZQ7C1qEGn_WJUv6VPHJL_yM?usp=share_link)

## Anonymized Pretrained Models

We provide pretrained models for both base (unconditional) generation and inpainting:

- **Anonymized Base & Inpainting Models**: [Pretrained Models](https://drive.google.com/drive/folders/1LNhs1EvwutPrZRt5LnMEXI_2qkmkAx_s?usp=share_link)

## Configuration

Set these environment variables according to your setup:

```bash
# Path to pretrained models
UNCOND_MODEL_PATH="/path/to/pretrained/base/model/"
INPAINT_MODEL_PATH="/path/to/pretrained/inpainting/model/"

# Path to dataset and cache
DATASET_PATH="/path/to/dataset/voxels_shrink/voxels_int.h5"
CACHE_DIR="/path/to/cache/"
NUM_GPU=2  # Number of GPUs to use
```

## Apple Silicon / MPS Support

This project fully supports **Apple Silicon Macs** (M1/M2/M3/M4) via PyTorch's Metal Performance Shaders (MPS) backend.

### Device Selection

All scripts accept a `--device` flag:

| Value | Behaviour |
|-------|----------|
| `auto` | **(default)** CUDA → MPS → CPU, picks best available |
| `cuda` | Force CUDA (falls back to CPU if unavailable) |
| `mps` | Force MPS (falls back to CPU if unavailable) |
| `cpu` | Force CPU |

### Low-Memory Inference

On Macs with limited unified memory (8–16 GB), combine these options to reduce peak memory:

```bash
# 1. Limit how much RAM PyTorch's MPS allocator can use (e.g. 50%)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

# 2. Use half precision + attention slicing
python eval.py --model_path "$BASE" \
  --device mps --dtype float16 --low-memory \
  ... # other args
```

| Flag | Effect |
|------|--------|
| `--dtype float16` | Loads model weights in half precision (halves memory) |
| `--low-memory` | Enables attention slicing (computes attention in chunks) |

> **Note:** MPS does not support `fp16` *training*. The `--dtype float16` flag is for **inference only**. During training, the code automatically disables mixed precision on MPS.

### Known MPS Limitations

- **Multi-GPU** is not applicable (Apple Silicon has a single unified GPU)
- **`fp16` mixed-precision training** is not supported; full precision is used automatically
- **`torch.Generator(device="mps")`** is not supported; the code transparently uses CPU generators

## Training

### Training the Base Unconditional Model

```bash
accelerate launch --num_processes=$NUM_GPU --mixed_precision='fp16' --num_machines=1 main.py \
--root_dir "$DATASET_PATH" \
--batch_size 16 \
--output_dir "out/models/base/" \
--epochs 1000 \
--scheduler "squaredcos_cap_v2" \
--diffusion_lr 1e-4 \
--small 0 \
--bw_ratio 0.2 \
--percentage 0.99 \
--num_workers 4 \
--use_sdf 0 \
--cache_dir "$CACHE_DIR" \
--disable_telegram \
--patience 1000 \
--timesteps 1000 \
--wandb_mode "offline" \
--project_name "3D_inpainting_diffusion" \
--inpainting_mode 0 \
--mask_type "gap_filling_compatible" \
--repaint_guidance \
--use_weighted_loss \
--sampler_type "ddpm" \
--check_for_edges
```

### Training the Inpainting Model

```bash
accelerate launch --num_processes=$NUM_GPU --mixed_precision='fp16' --num_machines=1 main.py \
--root_dir "$DATASET_PATH" \
--batch_size 16 \
--output_dir "out/models/inpainting/" \
--epochs 1000 \
--scheduler "squaredcos_cap_v2" \
--diffusion_lr 1e-4 \
--small 0 \
--bw_ratio 0.2 \
--percentage 0.99 \
--num_workers 4 \
--use_sdf 0 \
--cache_dir "$CACHE_DIR" \
--disable_telegram \
--patience 1000 \
--timesteps 1000 \
--wandb_mode "offline" \
--project_name "3D_inpainting_diffusion" \
--inpainting_mode 1 \
--mask_type "gap_filling_compatible" \
--repaint_guidance \
--use_weighted_loss \
--sampler_type "ddpm" \
--check_for_edges
```

## Evaluation

To reproduce the results from our paper:

```bash
BASE="/path/to/pretrained/base/model/"
INPAINTING="/path/to/pretrained/inpainting/model/"

python eval.py --model_path "$BASE" \
--min_bw_ratio 0.2 \
--max_retries 5 \
--stitching_mode separate_inpainting \
--inpainting_model_path "$INPAINTING" \
--output_dir "out/eval/" \
--scheduler_type "ddim" \
--n_blocks 100 \
--inpaint_region_size_ratio 0.3 \
--inference_steps 25 \
--device auto           # auto / cuda / mps / cpu
```

For **low-memory** setups (e.g. Apple Silicon with 8 GB RAM):

```bash
python eval.py --model_path "$BASE" \
--inpainting_model_path "$INPAINTING" \
--output_dir "out/eval/" \
--device mps --dtype float16 --low-memory \
--n_blocks 10 --batch_size 1 --inference_steps 25
```

### Enhanced Evaluation Pipeline

For comprehensive evaluation with advanced parameters:

```bash
python eval.py --model_path "/path/to/base/model" \
--inpainting_model_path "/path/to/inpainting/model" \
--output_dir "/path/to/output/" \
--n_blocks 100 \
--inference_steps 60 \
--scheduler_type "ddim" \
--seed 123 \
--stitching_mode "separate_inpainting" \
--mask_type "gap_filling_compatible" \
--overlap 8 \
--batch_size 8 \
--binary
```

### Polyhedron Segmentation

For fast polyhedron segmentation of generated granular media:

```bash
python polyhedron_segmentation.py \
--input /path/to/generated/file.vti \
--output /path/to/output/segmentation.json \
--no-paraview-multiblock
--decimation-ratio 0.99
--smoothing-iterations 40
--erosion-iterations 0 --min-polyhedron-size 100
--remove-boundary-polyhedrons
--max-voxel-aspect-ratio 0
--fast-mesh-extraction
--stream-batch-size 0
--num-workers 20
--batch-mesh-size 0
--force-cpu
--num-export-workers 20
--export-batch-size 400
--fast-paraview-export
--ultra-fast-mode
--max-chunk-workers 20
--target-vertices 10
--fix-mesh-for-lmgc90
```

For enabling chunking, add:
```bash
--use-chunking \
--fast-chunk-merge
```

### Railway Track Generation

For specialized railway track granular media generation:

```bash
python railway_track.py --model_path "/path/to/base/model" \
--inpainting_model_path "/path/to/inpainting/model" \
--output_dir "/path/to/output/" \
--target_length 16.0 \
--target_depth 0.3 \
--target_width 1.2 \
--inference_steps 50 \
--scheduler_type "ddim" \
--batch_size 16 \
--seed 484 \
--strip_batch_size 32 \
--inpaint_batch_size 5 \
--device auto           # auto / cuda / mps / cpu
```

For **low-memory** setups:

```bash
python railway_track.py --model_path "/path/to/base/model" \
--inpainting_model_path "/path/to/inpainting/model" \
--output_dir "/path/to/output/" \
--target_length 4.0 --target_depth 0.3 --target_width 0.6 \
--device mps --dtype float16 --low-memory \
--batch_size 1 --strip_batch_size 4 --inpaint_batch_size 2
```

Best Segmentation Results were achieved with these parameters:
```bash
python polyhedron_segmentation.py \
          --input "path/to/vti/or/json/file" \
          --output "path/to/vtp/and/json/on/output" \
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
          --fast-paraview-export \
          --ultra-fast-mode \
          --num-export-workers 6 \
          --export-batch-size 400 \
          --max-chunk-workers 6 \
--no-paraview-multiblock
```

## Heuristic Dataset Generation for Civil Sands

You can synthesize diffusion-ready RVEs for dense civil sands and recycled aggregates using a purely procedural generator. The script packs superquadric grains via gravity-driven placement, adds fines to close voids, and performs a light morphological closing to reach realistic solid fractions (~0.55).

```bash
python dataset_gen/heuristic_rve_generator.py \
  --output-dir out/sand_mixed \
  --num-volumes 256 \
  --volumes-per-file 64 \
  --num-workers 0 \
  --seed 13
```

- Produces `32×64×64` binary voxels stored as `.pt.gz` shards; compatible with `dataset_gen/hdf5_maker.py` for HDF5 packing.
- Configurable grain families (dense sand, fine sand, recycled chunks) plus fines-only filling and post-processing via `dataset_gen/heuristic_rve_generator.py`.
- Emits `generation_metadata.json` summarising solid fractions before/after post-processing, placement attempt statistics, and all profile parameters.
- Control CPU usage via `--num-workers` (set it to 0 to automatically use all available cores).

Re-run `dataset_gen/hdf5_maker.py` on the generated shard directory to assemble a single HDF5 volume for training.

## Additional Tools

- **Dataset Creation**: Create datasets from raw voxels using `dataset_gen/hdf5_maker.py`
- **Voxelizer**: Convert new data to voxel format for model input
- **Segmentation**: Output JSON files compatible with DEM software (Work in Progress but limited implementation is present)
- **Telegram Notifications**: Optionally enable by providing an `.env` file

## Citation

[Citation will be provided here]

## License

[License info will be provided]
