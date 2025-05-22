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
--project_name "3D_base" \
--inpainting_mode 0 \
--mask_type "central_large_block" \
--repaint_guidance \
--use_weighted_loss \
--sampler_type "ddpm" \
--central_block_max_ratio 0.7 \
--central_block_min_ratio 0.3 \
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
--project_name "3D_inpainting" \
--inpainting_mode 1 \
--mask_type "central_large_block" \
--repaint_guidance \
--use_weighted_loss \
--sampler_type "ddpm" \
--central_block_max_ratio 0.7 \
--central_block_min_ratio 0.3 \
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
--inference_steps 25
```

## Additional Tools

- **Dataset Creation**: Create datasets from raw voxels using `dataset_gen/hdf5_maker.py`
- **Voxelizer**: Convert new data to voxel format for model input
- **Segmentation**: Output JSON files compatible with DEM software (Work in Progress but limited implementation is present)
- **Telegram Notifications**: Optionally enable by providing an `.env` file

## Citation

[Citation will be provided here]

## License

[License info will be provided]

