#!/usr/bin/env python3
"""
Single block inpainting evaluation script.

This script:
1. Generates a single voxel block using an unconditional model.
2. Randomly removes a portion of this block (creates a mask).
3. Uses an inpainting model to fill the removed/masked region.
4. Outputs the original block, the masked block, and the inpainted block as VTI files.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import time

# Add tqdm for progress bars
from tqdm.auto import tqdm

# Add PyVista import for VTU saving
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    # print("Warning: PyVista not available. VTI saving will be skipped.") # Logging will handle this

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to Python path if necessary (though this script aims to be self-contained)
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Adjust if utils are in parent

from diffusers import DDIMScheduler, DDPMPipeline  # Assuming DDPMPipeline can load inpainting models too
from diffusers import DDPMScheduler  # Explicit import for type checking


def save_volume_as_vti(volume_data_np, vti_save_path, args_for_saving, descriptive_name="Volume"):
    """Saves a 3D numpy array as a VTI file using PyVista."""
    if not PYVISTA_AVAILABLE:
        logger.warning(f"PyVista not available. Skipping VTI save for {descriptive_name} to {vti_save_path}.")
        return

    try:
        D, H, W = volume_data_np.shape
        grid = pv.ImageData()
        grid.dimensions = np.array([W, H, D]) + 1  # PyVista dimensions (nx, ny, nz)
        grid.origin = (0, 0, 0)  # Default origin
        grid.spacing = (1, 1, 1)  # Default spacing

        # Process volume data (e.g., thresholding)
        processed_volume_data = volume_data_np.copy()
        if hasattr(args_for_saving, "binary") and args_for_saving.binary:
            logger.info(f"Applying binary threshold (>0.5) for {descriptive_name} VTI: {vti_save_path}")
            processed_volume_data = (processed_volume_data > 0.5).astype(np.float32)
        elif hasattr(args_for_saving, "threshold_value") and args_for_saving.threshold_value is not None:
            logger.info(
                f"Applying custom threshold (>{args_for_saving.threshold_value}) for {descriptive_name} VTI: {vti_save_path}"
            )
            processed_volume_data = (processed_volume_data > args_for_saving.threshold_value).astype(np.float32)

        # Ensure data is C-contiguous
        grid.cell_data["values"] = np.ascontiguousarray(processed_volume_data).flatten(order="C")

        grid.save(str(vti_save_path), binary=True)
        logger.info(f"{descriptive_name} saved as VTI to: {vti_save_path}")

    except Exception as e:
        logger.error(f"Error saving {descriptive_name} VTI file {vti_save_path}: {e}")
        import traceback

        traceback.print_exc()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Single block inpainting evaluation")

    # Model paths
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to unconditional (base) model for initial generation"
    )
    parser.add_argument("--inpainting_model_path", type=str, required=True, help="Path to inpainting model")

    # --- Masking Strategy ---
    parser.add_argument(
        "--masking_type",
        type=str,
        default="random_voxel",
        choices=["random_voxel", "internal_block"],
        help="Type of mask to apply: 'random_voxel' or 'internal_block'.",
    )

    # Parameters for 'random_voxel' masking
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.3,
        help="Ratio of voxels to randomly mask (used if masking_type='random_voxel')",
    )

    # Parameters for 'internal_block' masking
    parser.add_argument(
        "--num_masked_blocks",
        type=int,
        default=1,
        help="Number of internal blocks to mask (used if masking_type='internal_block')",
    )
    parser.add_argument(
        "--masked_block_min_dim_ratio",
        type=float,
        default=0.2,
        help="Min D,H,W ratio for internal masked blocks (used if masking_type='internal_block')",
    )
    parser.add_argument(
        "--masked_block_max_dim_ratio",
        type=float,
        default=0.4,
        help="Max D,H,W ratio for internal masked blocks (used if masking_type='internal_block')",
    )

    # --- Generation and Inpainting parameters ---
    parser.add_argument(
        "--inference_steps", type=int, default=50, help="Number of inference steps for generation and inpainting"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all operations")

    # Model configuration
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Scheduler type for both pipelines. Ensure it matches the loaded scheduler config if not re-initialized.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="out/eval_single/", help="Output directory for VTI and NPY files"
    )

    # Thresholding for VTI output
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Threshold value for VTI output (optional, overrides binary if set)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Threshold VTI output to binary mask (>0.5), ignored if threshold_value is set",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Ensure CUDA seeds are also set if using GPU, after device selection

    logger.info("=== Single Block Inpainting Evaluation ===")
    logger.info(f"Unconditional model: {args.model_path}")
    logger.info(f"Inpainting model: {args.inpainting_model_path}")
    logger.info(f"Masking type: {args.masking_type}")
    if args.masking_type == "random_voxel":
        logger.info(f"  Mask ratio: {args.mask_ratio}")
    elif args.masking_type == "internal_block":
        logger.info(f"  Num masked blocks: {args.num_masked_blocks}")
        logger.info(f"  Masked block min dim ratio: {args.masked_block_min_dim_ratio}")
        logger.info(f"  Masked block max dim ratio: {args.masked_block_max_dim_ratio}")
    logger.info(f"Output directory: {output_dir}")
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista not available. VTI saving will be skipped for all outputs.")

    # Check if models exist
    if not os.path.exists(args.model_path):
        logger.error(f"Base model not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.inpainting_model_path):
        logger.error(f"Inpainting model not found: {args.inpainting_model_path}")
        sys.exit(1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # Create the generator on the target device.
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # Load base unconditional model
        logger.info(f"Loading BASE diffusion pipeline from: {args.model_path}")
        base_pipeline = DDPMPipeline.from_pretrained(args.model_path)
        # Ensure correct scheduler is loaded/re-initialized based on args.scheduler_type
        # The from_pretrained might load a scheduler, this ensures we use the one specified or re-init it.
        if args.scheduler_type == "ddim":
            scheduler_config_path = Path(args.model_path) / "scheduler" / "scheduler_config.json"
            if scheduler_config_path.exists():
                base_pipeline.scheduler = DDIMScheduler.from_pretrained(Path(args.model_path) / "scheduler")
            else:
                logger.warning(
                    f"DDIM scheduler config not found at {scheduler_config_path}. Initializing DDIMScheduler from base model's scheduler config."
                )
                base_pipeline.scheduler = DDIMScheduler.from_config(base_pipeline.scheduler.config)
        elif (
            args.scheduler_type == "ddpm"
        ):  # Assuming DDPMPipeline might load DDIMScheduler by default from some checkpoints
            scheduler_config_path = Path(args.model_path) / "scheduler" / "scheduler_config.json"
            if (
                scheduler_config_path.exists()
                and "DDPMScheduler" in Path(args.model_path).joinpath("scheduler/scheduler_config.json").read_text()
            ):
                base_pipeline.scheduler = DDPMScheduler.from_pretrained(Path(args.model_path) / "scheduler")
            else:
                logger.warning(
                    f"DDPMScheduler config not found or not specified at {scheduler_config_path}. Initializing DDPMScheduler from base model's scheduler config."
                )
                base_pipeline.scheduler = DDPMScheduler.from_config(base_pipeline.scheduler.config)

        base_pipeline = base_pipeline.to(device)  # Move entire pipeline to device
        base_unet = base_pipeline.unet  # Keep unet on device
        base_scheduler = base_pipeline.scheduler  # Scheduler is not a nn.Module, but its methods use tensors

        logger.info(f"Base pipeline's UNet and Scheduler ({type(base_scheduler).__name__}) ready on {device}.")

        # Load inpainting model
        logger.info(f"Loading INPAINTING diffusion pipeline from: {args.inpainting_model_path}")
        inpainting_pipeline_obj = DDPMPipeline.from_pretrained(args.inpainting_model_path)
        if args.scheduler_type == "ddim":
            scheduler_config_path = Path(args.inpainting_model_path) / "scheduler" / "scheduler_config.json"
            if scheduler_config_path.exists():
                inpainting_pipeline_obj.scheduler = DDIMScheduler.from_pretrained(
                    Path(args.inpainting_model_path) / "scheduler"
                )
            else:
                logger.warning(
                    f"DDIM scheduler config not found for inpainting model at {scheduler_config_path}. Initializing DDIMScheduler from its scheduler config."
                )
                inpainting_pipeline_obj.scheduler = DDIMScheduler.from_config(inpainting_pipeline_obj.scheduler.config)
        elif args.scheduler_type == "ddpm":
            scheduler_config_path = Path(args.inpainting_model_path) / "scheduler" / "scheduler_config.json"
            if (
                scheduler_config_path.exists()
                and "DDPMScheduler"
                in Path(args.inpainting_model_path).joinpath("scheduler/scheduler_config.json").read_text()
            ):
                inpainting_pipeline_obj.scheduler = DDPMScheduler.from_pretrained(
                    Path(args.inpainting_model_path) / "scheduler"
                )
            else:
                logger.warning(
                    f"DDPMScheduler config not found or not specified for inpainting model at {scheduler_config_path}. Initializing DDPMScheduler from its scheduler config."
                )
                inpainting_pipeline_obj.scheduler = DDPMScheduler.from_config(inpainting_pipeline_obj.scheduler.config)

        inpainting_pipeline_obj = inpainting_pipeline_obj.to(device)
        inpainting_unet = inpainting_pipeline_obj.unet
        inpainting_scheduler = inpainting_pipeline_obj.scheduler
        logger.info(
            f"Inpainting pipeline's UNet and Scheduler ({type(inpainting_scheduler).__name__}) ready on {device}."
        )

        # Delete full pipeline objects to free memory
        del base_pipeline
        del inpainting_pipeline_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure inpainting UNet is configured for inpainting mode
        if not (hasattr(inpainting_unet, "inpainting_mode") and inpainting_unet.inpainting_mode) and not (
            hasattr(inpainting_unet, "config")
            and hasattr(inpainting_unet.config, "inpainting_mode")
            and inpainting_unet.config.inpainting_mode
        ):
            logger.warning(
                "INPAINTING UNET MAY NOT BE IN INPAINTING MODE. The UNet should have an 'inpainting_mode=True' attribute/config and expect concatenated input [latent, mask, masked_original]."
            )
        else:
            logger.info("Inpainting UNet appears to be configured for inpainting_mode.")

        logger.info("Models loaded successfully.")

        # 1. Generate initial block using manual diffusion loop
        logger.info("Generating initial block (manual loop)...")

        # Determine shape from UNet config
        if (
            not hasattr(base_unet, "config")
            or not hasattr(base_unet.config, "sample_size")
            or not hasattr(base_unet.config, "in_channels")
        ):
            logger.error(
                "Base UNet does not have 'config.sample_size' or 'config.in_channels'. Cannot determine generation shape."
            )
            sys.exit(1)

        D, H, W = base_unet.config.sample_size  # Should be (Depth, Height, Width)
        C = base_unet.config.in_channels  # Should be 1 for unconditional base model usually

        latents = torch.randn(
            (1, C, D, H, W),  # Batch, Channels, Depth, Height, Width
            generator=generator,  # Use the CPU generator
            device=device,  # Create on target device
            dtype=base_unet.dtype,
        )

        if isinstance(base_scheduler, DDPMScheduler):  # DDPMPipeline scales init noise
            latents = latents * base_scheduler.init_noise_sigma

        base_scheduler.set_timesteps(args.inference_steps, device=device)
        timesteps = base_scheduler.timesteps

        with torch.no_grad():
            for t in tqdm(timesteps, desc="Base Generation"):
                # model_input is latents for unconditional
                model_output = base_unet(latents, t).sample  # sample is the 5D output
                latents = base_scheduler.step(model_output, t, latents, generator=generator).prev_sample

        initial_block_tensor = latents  # This is the 5D tensor: (B, C, D, H, W)
        # Denormalize if necessary (common practice for diffusion models)
        initial_block_tensor = (initial_block_tensor / 2 + 0.5).clamp(0, 1)

        initial_block_np = initial_block_tensor.cpu().numpy().squeeze()  # Remove batch and channel if C=1
        # If channel is not 1, this squeeze might be problematic. Assuming C=1 for voxel data.
        if initial_block_np.ndim == 4 and initial_block_np.shape[0] == 1:  # (1,D,H,W)
            initial_block_np = initial_block_np.squeeze(0)
        elif initial_block_np.ndim != 3:  # Expecting (D,H,W)
            logger.error(
                f"Generated block has unexpected shape: {initial_block_np.shape}. Expected 3D (D,H,W) or 4D (1,D,H,W)."
            )
            sys.exit(1)

        logger.info(f"Initial block generated. Shape: {initial_block_np.shape}")

        # 2. Create mask based on strategy
        logger.info(f"Creating mask using '{args.masking_type}' strategy...")
        D, H, W = initial_block_np.shape
        mask_np = np.zeros_like(initial_block_np, dtype=np.float32)

        if args.masking_type == "random_voxel":
            num_total_voxels = D * H * W
            num_voxels_to_mask = int(args.mask_ratio * num_total_voxels)
            if num_voxels_to_mask > 0:
                flat_indices = np.random.choice(num_total_voxels, size=num_voxels_to_mask, replace=False)
                coords_to_mask = np.unravel_index(flat_indices, (D, H, W))
                mask_np[coords_to_mask] = 1.0  # 1.0 where we want to inpaint
            logger.info(f"Random voxel mask created. Num masked voxels: {np.sum(mask_np):.0f}")

        elif args.masking_type == "internal_block":
            for i in range(args.num_masked_blocks):
                # Determine random dimensions for the block to mask
                block_d = np.random.randint(
                    int(D * args.masked_block_min_dim_ratio), int(D * args.masked_block_max_dim_ratio) + 1
                )
                block_h = np.random.randint(
                    int(H * args.masked_block_min_dim_ratio), int(H * args.masked_block_max_dim_ratio) + 1
                )
                block_w = np.random.randint(
                    int(W * args.masked_block_min_dim_ratio), int(W * args.masked_block_max_dim_ratio) + 1
                )

                # Ensure block dimensions are at least 1
                block_d = max(1, block_d)
                block_h = max(1, block_h)
                block_w = max(1, block_w)

                # Determine random starting position for the block
                # Ensure the block fits within the main volume dimensions
                if D - block_d <= 0:
                    start_d = 0
                else:
                    start_d = np.random.randint(0, D - block_d)

                if H - block_h <= 0:
                    start_h = 0
                else:
                    start_h = np.random.randint(0, H - block_h)

                if W - block_w <= 0:
                    start_w = 0
                else:
                    start_w = np.random.randint(0, W - block_w)

                mask_np[start_d : start_d + block_d, start_h : start_h + block_h, start_w : start_w + block_w] = 1.0
                logger.info(f"Masked internal block {i + 1}/{args.num_masked_blocks}: ")
                logger.info(
                    f"  at D[{start_d}:{start_d + block_d}], H[{start_h}:{start_h + block_h}], W[{start_w}:{start_w + block_w}]"
                )
            logger.info(f"Internal block mask created. Total masked voxels: {np.sum(mask_np):.0f}")
        else:
            logger.error(f"Unknown masking type: {args.masking_type}")
            sys.exit(1)

        logger.info(f"Mask created. Shape: {mask_np.shape}, Num masked voxels: {np.sum(mask_np):.0f}")

        # 3. Prepare inputs for inpainting model
        # Diffusers inpainting pipelines usually expect:
        # - image: The original image (content for unmasked areas).
        # - mask_image: The mask (1s for areas to be inpainted, 0s for context).

        # `initial_block_tensor` is the generated block, (B,C,D,H,W)
        image_for_inpainting = initial_block_tensor.clone().to(device)  # Use the tensor directly
        mask_for_inpainting = (
            torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device, dtype=image_for_inpainting.dtype)
        )  # (B, C, D, H, W)

        masked_input_for_saving_np = initial_block_np * (1.0 - mask_np)  # For saving the "holey" input

        # 4. Perform inpainting using manual diffusion loop
        logger.info("Performing inpainting (manual loop)...")

        masked_original_tensor = image_for_inpainting * (1.0 - mask_for_inpainting)  # B,C,D,H,W

        # Initial latents for inpainting (noisy image to be denoised in masked areas)
        current_latents_to_denoise = torch.randn_like(
            image_for_inpainting,  # B, C, D, H, W
        )

        if isinstance(inpainting_scheduler, DDPMScheduler):  # DDPMPipeline scales init noise
            current_latents_to_denoise = current_latents_to_denoise * inpainting_scheduler.init_noise_sigma

        inpainting_scheduler.set_timesteps(args.inference_steps, device=device)
        inpainting_timesteps = inpainting_scheduler.timesteps

        # Ensure mask_for_inpainting and masked_original_tensor have same dtype as current_latents_to_denoise
        mask_for_inpainting = mask_for_inpainting.to(dtype=current_latents_to_denoise.dtype)
        masked_original_tensor = masked_original_tensor.to(dtype=current_latents_to_denoise.dtype)

        with torch.no_grad():
            for t in tqdm(inpainting_timesteps, desc="Inpainting"):
                # At each step, the UNet input is: [current_noisy_latent, mask, masked_original_image]
                # Assuming C=1 for each part, so total input channels for UNet = 3
                # This matches UNet3DModel's inpainting_mode logic: conv_in_channels = (in_channels * 2 + 1)
                # where in_channels is for the 'latent' part.

                model_input = torch.cat(
                    [current_latents_to_denoise, mask_for_inpainting, masked_original_tensor], dim=1
                )

                unet_output = inpainting_unet(model_input, t).sample  # sample is the 5D output (noise prediction)

                # The scheduler step denoises current_latents_to_denoise
                current_latents_to_denoise = inpainting_scheduler.step(
                    unet_output, t, current_latents_to_denoise, generator=generator
                ).prev_sample

        inpainted_result_tensor = current_latents_to_denoise
        # Denormalize if necessary
        inpainted_result_tensor = (inpainted_result_tensor / 2 + 0.5).clamp(0, 1)

        inpainted_result_np = inpainted_result_tensor.cpu().numpy().squeeze()
        if inpainted_result_np.ndim == 4 and inpainted_result_np.shape[0] == 1:
            inpainted_result_np = inpainted_result_np.squeeze(0)
        elif inpainted_result_np.ndim != 3:
            logger.error(f"Inpainted block has unexpected shape: {inpainted_result_np.shape}")
            sys.exit(1)
        logger.info(f"Inpainting completed. Inpainted block shape: {inpainted_result_np.shape}")

        # 5. Save outputs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        seed_str = f"seed{args.seed}"
        ratio_str = f"ratio{args.mask_ratio:.2f}"

        # Original block
        orig_npy_path = output_dir / f"original_block_{seed_str}_{timestamp}.npy"
        orig_vti_path = output_dir / f"original_block_{seed_str}_{timestamp}.vti"
        logger.info(f"Saving original block to: {orig_npy_path}")
        np.save(orig_npy_path, initial_block_np)
        save_volume_as_vti(initial_block_np, orig_vti_path, args, "Original Block")

        # Masked input block
        masked_npy_path = output_dir / f"masked_input_block_{seed_str}_{ratio_str}_{timestamp}.npy"
        masked_vti_path = output_dir / f"masked_input_block_{seed_str}_{ratio_str}_{timestamp}.vti"
        logger.info(f"Saving masked input block to: {masked_npy_path}")
        np.save(masked_npy_path, masked_input_for_saving_np)
        save_volume_as_vti(masked_input_for_saving_np, masked_vti_path, args, "Masked Input Block")

        # Mask itself (optional, but can be useful)
        mask_npy_path = output_dir / f"mask_{seed_str}_{ratio_str}_{timestamp}.npy"
        mask_vti_path = output_dir / f"mask_{seed_str}_{ratio_str}_{timestamp}.vti"
        logger.info(f"Saving mask to: {mask_npy_path}")
        np.save(mask_npy_path, mask_np)

        # Create a dummy args for mask saving if needed (no thresholding on mask itself usually)
        class MaskArgs:
            pass

        mask_save_args = MaskArgs()
        mask_save_args.binary = False  # Save mask as is
        mask_save_args.threshold_value = None
        save_volume_as_vti(mask_np, mask_vti_path, mask_save_args, "Mask")

        # Inpainted block
        inpainted_npy_path = output_dir / f"inpainted_block_{seed_str}_{ratio_str}_{timestamp}.npy"
        inpainted_vti_path = output_dir / f"inpainted_block_{seed_str}_{ratio_str}_{timestamp}.vti"
        logger.info(f"Saving inpainted block to: {inpainted_npy_path}")
        np.save(inpainted_npy_path, inpainted_result_np)
        save_volume_as_vti(inpainted_result_np, inpainted_vti_path, args, "Inpainted Block")

        # Summary Statistics
        logger.info("=== SUMMARY ===")
        logger.info(
            f"Original block - Shape: {initial_block_np.shape}, Min: {initial_block_np.min():.3f}, Max: {initial_block_np.max():.3f}, Mean: {initial_block_np.mean():.3f}"
        )
        logger.info(
            f"Masked input block - Shape: {masked_input_for_saving_np.shape}, Min: {masked_input_for_saving_np.min():.3f}, Max: {masked_input_for_saving_np.max():.3f}, Mean: {masked_input_for_saving_np.mean():.3f}"
        )
        logger.info(
            f"Inpainted block - Shape: {inpainted_result_np.shape}, Min: {inpainted_result_np.min():.3f}, Max: {inpainted_result_np.max():.3f}, Mean: {inpainted_result_np.mean():.3f}"
        )

        logger.info("Single block inpainting evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during single block evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
