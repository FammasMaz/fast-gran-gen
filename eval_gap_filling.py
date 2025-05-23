#!/usr/bin/env python3
"""
Gap-filling evaluation script.

This script implements what the user actually wants:
1. Generate separate, non-overlapping voxel blocks using the unconditional model
2. Place them with gaps between them
3. Use the inpainting model to fill the gaps, treating adjacent blocks as context (unmasked)
   and gaps as the region to inpaint (masked)

This is different from the current approach which generates overlapping blocks and blends them.
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
    print("Warning: PyVista not available. VTI saving will be skipped.")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from diffusers import DDIMScheduler, DDPMPipeline, UNet3DConditionModel
from utils.eval_utils_gap_filling import generate_blocks_with_gaps_and_inpaint
import argparse as args_module


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gap-filling inpainting evaluation")

    # Model paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to unconditional (base) model")
    parser.add_argument("--inpainting_model_path", type=str, required=True, help="Path to inpainting model")

    # Generation parameters
    parser.add_argument("--n_blocks", type=int, default=5, help="Number of separate blocks to generate")
    parser.add_argument("--gap_size", type=int, default=16, help="Size of gap between blocks (in voxels)")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model configuration
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddim", "ddpm"], help="Scheduler type")
    parser.add_argument(
        "--generation_batch_size", type=int, default=1, help="Batch size for generating individual blocks"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="out/gap_filling/", help="Output directory")
    parser.add_argument("--save_debug", action="store_true", help="Save debug outputs for each gap")

    # Thresholding
    parser.add_argument(
        "--threshold_value", type=float, default=None, help="Threshold value for binary output (optional)"
    )

    # Binary output flag (like in eval.py)
    parser.add_argument("--binary", action="store_true", help="Threshold output to binary mask (>0.5)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Gap-Filling Inpainting Evaluation ===")
    logger.info(f"Unconditional model: {args.model_path}")
    logger.info(f"Inpainting model: {args.inpainting_model_path}")
    logger.info(f"Will generate {args.n_blocks} blocks with {args.gap_size} voxel gaps")
    logger.info(f"Output directory: {output_dir}")

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

    try:
        # Load models (adapted from eval.py)
        logger.info("Loading models...")

        # Load base unconditional model
        logger.info(f"Loading BASE diffusion pipeline from: {args.model_path}")
        base_pipeline = DDPMPipeline.from_pretrained(args.model_path).to(device)
        if args.scheduler_type == "ddim":
            base_pipeline.scheduler = DDIMScheduler.from_pretrained(Path(args.model_path) / "scheduler")
        base_unet = base_pipeline.unet
        base_scheduler = base_pipeline.scheduler
        logger.info(f"Base pipeline loaded with {args.scheduler_type.upper()} scheduler.")

        # Load inpainting model
        logger.info(f"Loading INPAINTING diffusion pipeline from: {args.inpainting_model_path}")
        inpainting_pipeline = DDPMPipeline.from_pretrained(args.inpainting_model_path).to(device)
        if args.scheduler_type == "ddim":
            inpainting_pipeline.scheduler = DDIMScheduler.from_pretrained(
                Path(args.inpainting_model_path) / "scheduler"
            )
        logger.info(f"Inpainting pipeline loaded with {args.scheduler_type.upper()} scheduler.")

        logger.info("Models loaded successfully")

        # Create a minimal args object for compatibility
        minimal_args = args_module.Namespace()
        minimal_args.mask_type = "none"  # For unconditional generation
        minimal_args.central_block_min_ratio = 0.3
        minimal_args.central_block_max_ratio = 0.7
        minimal_args.central_block_jitter_factor = 0.1

        # Generate volume with gap-filling approach
        logger.info("Starting gap-filling generation...")

        final_volume = generate_blocks_with_gaps_and_inpaint(
            base_unet=base_unet,
            base_scheduler=base_scheduler,
            inpainting_pipeline=inpainting_pipeline,
            num_inference_steps=args.inference_steps,
            seed=args.seed,
            n_blocks=args.n_blocks,
            gap_size=args.gap_size,
            device=device,
            args=minimal_args,
            output_dir=output_dir if args.save_debug else None,
            generation_batch_size=args.generation_batch_size,
        )

        if final_volume is None:
            logger.error("Gap-filling generation failed")
            sys.exit(1)

        logger.info(f"Gap-filling generation completed. Final volume shape: {final_volume.shape}")

        # Apply thresholding if specified
        if args.threshold_value is not None:
            logger.info(f"Applying threshold: {args.threshold_value}")
            final_volume = (final_volume > args.threshold_value).astype(np.float32)

        # Save as VTI file using PyVista (like in eval.py)
        if PYVISTA_AVAILABLE:
            try:
                logger.info("Saving gap-filled volume as VTI...")
                D_total, H, W = final_volume.shape

                grid = pv.ImageData()
                grid.dimensions = np.array([W, H, D_total]) + 1
                grid.origin = (0, 0, 0)
                grid.spacing = (1, 1, 1)

                # Apply binary threshold if requested
                volume_data = final_volume.copy()
                if args.binary:
                    volume_data = (volume_data > 0.5).astype(np.float32)

                grid.cell_data["values"] = np.ascontiguousarray(volume_data).flatten(order="C")

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                vti_filename = (
                    f"gap_filled_volume_{timestamp}_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.vti"
                )
                vti_save_path = output_dir / vti_filename
                grid.save(str(vti_save_path), binary=True)
                logger.info(f"Gap-filled volume saved as VTI to: {vti_save_path}")

            except Exception as e:
                logger.error(f"Error saving VTI file: {e}")
        else:
            logger.warning("VTI saving skipped (pyvista not available).")

        # Save the final result
        output_file = output_dir / f"gap_filled_volume_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.npy"
        logger.info(f"Saving final volume to: {output_file}")
        np.save(output_file, final_volume)

        # Also save as h5 if save_volume function supports it
        try:
            h5_output = output_dir / f"gap_filled_volume_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.h5"
            # Remove this save_volume call since the function doesn't exist
            # save_volume(final_volume, str(h5_output))
            # logger.info(f"Also saved as: {h5_output}")
        except Exception as e:
            logger.warning(f"Could not save as H5: {e}")

        # Print summary statistics
        logger.info("=== SUMMARY ===")
        logger.info(f"Generated volume shape: {final_volume.shape}")
        logger.info(f"Value range: [{final_volume.min():.3f}, {final_volume.max():.3f}]")
        logger.info(f"Mean value: {final_volume.mean():.3f}")

        # Calculate fill ratio
        if args.threshold_value is not None or args.binary:
            if args.binary:
                fill_ratio = np.mean(final_volume > 0.5)
                logger.info(f"Fill ratio (binary threshold >0.5): {fill_ratio:.3f}")
            else:
                fill_ratio = np.mean(final_volume > args.threshold_value)
                logger.info(f"Fill ratio (threshold >{args.threshold_value}): {fill_ratio:.3f}")
        else:
            # For non-thresholded volumes, show density
            fill_ratio = np.mean(final_volume > 0)
            logger.info(f"Non-zero ratio: {fill_ratio:.3f}")

        logger.info("Gap-filling evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
