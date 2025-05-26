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

    # Generation and Masking parameters
    parser.add_argument(
        "--mask_ratio", type=float, default=0.3, help="Ratio of voxels to randomly mask for inpainting (0.0 to 1.0)"
    )
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
        help="Scheduler type for both pipelines",
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
    logger.info(f"Mask ratio: {args.mask_ratio}")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # Load base unconditional model
        logger.info(f"Loading BASE diffusion pipeline from: {args.model_path}")
        base_pipeline = DDPMPipeline.from_pretrained(args.model_path)
        if args.scheduler_type == "ddim":
            base_pipeline.scheduler = DDIMScheduler.from_config(
                base_pipeline.scheduler.config
            )  # Re-init from config for DDIM
        base_pipeline = base_pipeline.to(device)
        logger.info(f"Base pipeline loaded with {args.scheduler_type.upper()} scheduler.")

        # Load inpainting model
        logger.info(f"Loading INPAINTING diffusion pipeline from: {args.inpainting_model_path}")
        inpainting_pipeline = DDPMPipeline.from_pretrained(args.inpainting_model_path)
        if args.scheduler_type == "ddim":
            inpainting_pipeline.scheduler = DDIMScheduler.from_config(
                inpainting_pipeline.scheduler.config
            )  # Re-init for DDIM
        inpainting_pipeline = inpainting_pipeline.to(device)

        # Compatibility for schedulers if loaded from different paths potentially
        if args.scheduler_type == "ddim":
            if not os.path.exists(Path(args.model_path) / "scheduler"):
                logger.warning(
                    f"Scheduler config for base model not found at {Path(args.model_path) / 'scheduler'}. Using default {args.scheduler_type} init."
                )
            else:
                base_pipeline.scheduler = DDIMScheduler.from_pretrained(Path(args.model_path) / "scheduler")
            if not os.path.exists(Path(args.inpainting_model_path) / "scheduler"):
                logger.warning(
                    f"Scheduler config for inpainting model not found at {Path(args.inpainting_model_path) / 'scheduler'}. Using default {args.scheduler_type} init."
                )
            else:
                inpainting_pipeline.scheduler = DDIMScheduler.from_pretrained(
                    Path(args.inpainting_model_path) / "scheduler"
                )

        base_pipeline.to(device)
        inpainting_pipeline.to(device)
        logger.info(f"Inpainting pipeline loaded with {args.scheduler_type.upper()} scheduler.")
        logger.info("Models loaded successfully.")

        # 1. Generate initial block
        logger.info("Generating initial block...")
        generator = torch.Generator(device=device).manual_seed(args.seed)  # Use same device as model for consistency

        # Assuming pipeline output is (batch, channels, D, H, W)
        initial_block_tensor = base_pipeline(
            batch_size=1,
            generator=generator,
            num_inference_steps=args.inference_steps,
            output_type="pt",  # Ensure PyTorch tensor output
        ).images

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

        # 2. Create random mask
        logger.info(f"Creating random mask with ratio: {args.mask_ratio}...")
        D, H, W = initial_block_np.shape

        mask_np = np.zeros_like(initial_block_np, dtype=np.float32)
        num_total_voxels = D * H * W
        num_voxels_to_mask = int(args.mask_ratio * num_total_voxels)

        if num_voxels_to_mask > 0:
            flat_indices = np.random.choice(num_total_voxels, size=num_voxels_to_mask, replace=False)
            coords_to_mask = np.unravel_index(flat_indices, (D, H, W))
            mask_np[coords_to_mask] = 1.0  # 1.0 where we want to inpaint

        logger.info(f"Mask created. Shape: {mask_np.shape}, Num masked voxels: {np.sum(mask_np):.0f}")

        # 3. Prepare inputs for inpainting model
        # Diffusers inpainting pipelines usually expect:
        # - image: The original image (content for unmasked areas).
        # - mask_image: The mask (1s for areas to be inpainted, 0s for context).
        image_for_inpainting = (
            torch.from_numpy(initial_block_np).unsqueeze(0).unsqueeze(0).to(device)
        )  # (B, C, D, H, W)
        mask_for_inpainting = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)  # (B, C, D, H, W)

        masked_input_for_saving_np = initial_block_np * (1.0 - mask_np)  # For saving the "holey" input

        # 4. Perform inpainting
        logger.info("Performing inpainting...")
        # Ensure generator is on the same device
        if generator.device.type != device.type:
            generator = torch.Generator(device=device).manual_seed(args.seed)

        inpainted_result_tensor = inpainting_pipeline(
            image=image_for_inpainting,
            mask_image=mask_for_inpainting,
            num_inference_steps=args.inference_steps,
            generator=generator,  # Re-use generator with correct device
            output_type="pt",
        ).images

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
