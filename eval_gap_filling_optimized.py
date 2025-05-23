#!/usr/bin/env python3
"""
Optimized gap-filling evaluation script for existing mixed_edge_central models.

This script modifies the gap-filling approach to work better with models
trained on central blocks and edge masks rather than thin strips.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import time

# Add PyVista import for VTI saving
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized gap-filling inpainting evaluation")

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
    parser.add_argument("--output_dir", type=str, default="out/gap_filling_optimized/", help="Output directory")
    parser.add_argument("--save_debug", action="store_true", help="Save debug outputs for each gap")

    # Optimization parameters
    parser.add_argument(
        "--use_larger_regions", action="store_true", help="Use larger processing regions for inpainting"
    )
    parser.add_argument("--iterative_inpaint", action="store_true", help="Use iterative inpainting approach")
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of iterative inpainting steps")

    # Thresholding
    parser.add_argument(
        "--threshold_value", type=float, default=None, help="Threshold value for binary output (optional)"
    )
    parser.add_argument("--binary", action="store_true", help="Threshold output to binary mask (>0.5)")

    return parser.parse_args()


def generate_blocks_with_optimized_gap_filling(
    base_unet,
    base_scheduler,
    inpainting_pipeline,
    num_inference_steps,
    seed,
    n_blocks,
    gap_size,
    device,
    output_dir=None,
    generation_batch_size=1,
    use_larger_regions=False,
    iterative_inpaint=False,
    num_iterations=3,
):
    """
    Optimized gap-filling that works better with mixed_edge_central trained models.

    Key optimizations:
    1. Uses larger processing regions around gaps
    2. Creates masks more similar to training (central block style)
    3. Iterative refinement option
    """
    from utils.eval_utils import generate_single_volume
    import argparse as args_module

    logger.info(f"Starting optimized gap-filling: {n_blocks} blocks with {gap_size} voxel gaps")
    torch.cuda.empty_cache()

    # Get block dimensions from base model
    sample_size = base_unet.config.sample_size
    if isinstance(sample_size, int):
        D, H, W = sample_size, sample_size, sample_size
    else:
        D, H, W = sample_size
    logger.info(f"Block dimensions: D={D}, H={H}, W={W}")

    # Step 1: Generate separate blocks using unconditional model
    logger.info("Step 1: Generating separate blocks using unconditional model")
    all_blocks_np = []

    # Create minimal args for generation
    minimal_args = args_module.Namespace()
    minimal_args.mask_type = "none"
    minimal_args.central_block_min_ratio = 0.3
    minimal_args.central_block_max_ratio = 0.7
    minimal_args.central_block_jitter_factor = 0.1

    for i in range(n_blocks):
        current_seed = seed + i
        logger.info(f"Generating block {i + 1}/{n_blocks} with seed {current_seed}")

        try:
            block_batch_np = generate_single_volume(
                base_unet,
                base_scheduler,
                num_inference_steps,
                current_seed,
                1,  # batch_size=1
                device,
                args=minimal_args,
                min_bw_ratio=0.0,
                max_retries=0,
            )

            if block_batch_np is None or len(block_batch_np) != 1:
                raise RuntimeError(f"generate_single_volume failed for block {i + 1}")
            all_blocks_np.extend(block_batch_np)

        except Exception as e:
            logger.error(f"Error generating block {i + 1}: {e}")
            return None

    logger.info(f"Successfully generated {n_blocks} independent blocks")

    # Step 2: Calculate total volume size and place blocks with gaps
    logger.info("Step 2: Placing blocks with gaps")
    total_depth = n_blocks * D + (n_blocks - 1) * gap_size
    logger.info(f"Total volume depth: {total_depth}")

    C = base_unet.config.in_channels
    full_volume_pt = torch.zeros((1, C, total_depth, H, W), dtype=torch.float32, device="cpu")

    # Place blocks with gaps
    gap_positions = []
    current_pos = 0

    for i, block_np in enumerate(all_blocks_np):
        if block_np.ndim == 3:
            block_np = np.expand_dims(block_np, axis=0)

        # Place block
        block_pt = torch.from_numpy(block_np).float()
        end_pos = current_pos + D
        full_volume_pt[0, :, current_pos:end_pos, :, :] = block_pt

        logger.info(f"Placed block {i + 1} at position {current_pos}:{end_pos}")

        # Add gap (except after last block)
        if i < n_blocks - 1:
            gap_start = end_pos
            gap_end = gap_start + gap_size
            gap_positions.append((gap_start, gap_end))
            logger.info(f"Gap {i + 1} at position {gap_start}:{gap_end}")
            current_pos = gap_end
        else:
            current_pos = end_pos

    full_volume_pt = full_volume_pt.to(device)

    # Step 3: Optimized gap inpainting
    logger.info("Step 3: Optimized gap inpainting")
    inpainting_unet = inpainting_pipeline.unet
    inpainting_scheduler = inpainting_pipeline.scheduler
    inpainting_scheduler.set_timesteps(num_inference_steps)
    inpainting_unet.eval()

    from tqdm.auto import tqdm

    pbar_gap = tqdm(total=len(gap_positions), desc="Inpainting Gaps")

    for gap_idx, (gap_start, gap_end) in enumerate(gap_positions):
        logger.info(f"Inpainting gap {gap_idx + 1}/{len(gap_positions)} from {gap_start} to {gap_end}")

        # OPTIMIZATION 1: Use larger processing regions
        if use_larger_regions:
            # Use much larger context around the gap
            context_size = max(D, gap_size * 3)  # Large context
        else:
            context_size = D // 2  # Moderate context

        # Calculate processing region
        center = (gap_start + gap_end) // 2
        process_start = max(0, center - context_size // 2)
        process_end = min(total_depth, process_start + context_size)

        # Ensure processing region is compatible with UNet (divisible by 8)
        process_depth = process_end - process_start
        target_depth = ((process_depth + 7) // 8) * 8

        if target_depth != process_depth:
            # Adjust to get proper size
            center = (process_start + process_end) // 2
            process_start = max(0, center - target_depth // 2)
            process_end = min(total_depth, process_start + target_depth)
            if process_end - process_start != target_depth:
                process_start = max(0, process_end - target_depth)

        process_depth = process_end - process_start
        logger.info(f"Processing region: {process_start}:{process_end} (depth={process_depth})")

        # Extract processing region
        process_region = full_volume_pt[0, :, process_start:process_end, :, :].clone()

        if iterative_inpaint:
            # OPTIMIZATION 2: Iterative inpainting with progressively larger masks
            logger.info(f"Using iterative inpainting with {num_iterations} iterations")

            for iter_idx in range(num_iterations):
                # Create progressively larger masks
                base_gap_size = gap_end - gap_start
                iter_gap_size = base_gap_size * (0.5 + 0.5 * (iter_idx + 1) / num_iterations)
                iter_gap_size = int(iter_gap_size)

                # Center the mask around the gap
                local_gap_center = (gap_start + gap_end) // 2 - process_start
                local_gap_start = max(0, local_gap_center - iter_gap_size // 2)
                local_gap_end = min(process_depth, local_gap_start + iter_gap_size)

                # Create mask
                mask = torch.zeros(1, 1, process_depth, H, W, device=device)
                mask[0, 0, local_gap_start:local_gap_end, :, :] = 1.0

                logger.info(
                    f"  Iteration {iter_idx + 1}: mask size {iter_gap_size}, local region {local_gap_start}:{local_gap_end}"
                )

                # Run inpainting for this iteration
                process_region = run_single_inpainting(
                    process_region,
                    mask,
                    inpainting_unet,
                    inpainting_scheduler,
                    num_inference_steps,
                    seed + gap_idx * 100 + iter_idx,
                    device,
                )
        else:
            # OPTIMIZATION 3: Single inpainting with central-block style mask
            # Create a mask that's more similar to training (larger, more block-like)
            local_gap_start = max(0, gap_start - process_start)
            local_gap_end = min(process_depth, gap_end - process_start)

            # Expand the mask to be more block-like (similar to central_large_block training)
            expanded_gap_size = max(gap_size, process_depth // 4)  # At least 25% of region
            expanded_start = max(0, (local_gap_start + local_gap_end) // 2 - expanded_gap_size // 2)
            expanded_end = min(process_depth, expanded_start + expanded_gap_size)

            mask = torch.zeros(1, 1, process_depth, H, W, device=device)
            mask[0, 0, expanded_start:expanded_end, :, :] = 1.0

            logger.info(f"Using expanded mask: {expanded_start}:{expanded_end} (size={expanded_gap_size})")

            # Run inpainting
            process_region = run_single_inpainting(
                process_region,
                mask,
                inpainting_unet,
                inpainting_scheduler,
                num_inference_steps,
                seed + gap_idx * 100,
                device,
            )

        # Place result back
        actual_size = min(process_region.shape[1], process_end - process_start)
        full_volume_pt[0, :, process_start : process_start + actual_size, :, :] = process_region[:, :actual_size, :, :]

        logger.info(f"Gap {gap_idx + 1} inpainting complete")

        # Save debug output
        if output_dir:
            debug_dir = Path(output_dir) / f"optimized_gap_debug_seed{seed}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            gap_result_np = process_region.detach().cpu().numpy()
            np.save(debug_dir / f"gap_{gap_idx:02d}_optimized.npy", gap_result_np)

        pbar_gap.update(1)

    pbar_gap.close()
    logger.info("All gaps inpainted successfully")

    # Convert final result to numpy
    final_volume_np = full_volume_pt.squeeze(0).detach().cpu().numpy()
    if final_volume_np.shape[0] == 1:
        final_volume_np = final_volume_np.squeeze(0)

    logger.info(f"Final volume shape: {final_volume_np.shape}")
    return final_volume_np


def run_single_inpainting(
    process_region, mask, inpainting_unet, inpainting_scheduler, num_inference_steps, seed, device
):
    """Run a single inpainting pass on a region."""
    from tqdm.auto import tqdm

    # Add batch dimension
    process_region_b = process_region.unsqueeze(0).to(device)
    mask_b = mask.to(device)

    # Prepare generator and latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(process_region_b.shape, generator=generator, device=device)

    # Base noise for original content preservation
    base_noise = torch.randn(process_region_b.shape, generator=generator, device=device)

    # Inpainting loop
    with torch.no_grad():
        for t in tqdm(inpainting_scheduler.timesteps, desc="Inpainting", leave=False):
            # Create masked image
            masked_image = process_region_b * (1.0 - mask_b)

            # Scale latents
            scaled_latents = inpainting_scheduler.scale_model_input(latents, t)

            # Concatenate inputs
            unet_input = torch.cat([scaled_latents, mask_b, masked_image], dim=1)

            # Predict noise
            t_input = t.repeat(process_region_b.shape[0]).to(device)
            noise_pred = inpainting_unet(unet_input, t_input, return_dict=False)[0]

            # Scheduler step
            step_output = inpainting_scheduler.step(noise_pred, t, latents)
            prev_sample = step_output.prev_sample

            # Preserve original content in non-masked regions
            if t != inpainting_scheduler.timesteps[-1]:
                # Add noise to original content to match current timestep
                timestep_idx = (t == inpainting_scheduler.timesteps).nonzero().item()
                prev_timestep_idx = min(timestep_idx + 1, len(inpainting_scheduler.timesteps) - 1)
                prev_timestep = inpainting_scheduler.timesteps[prev_timestep_idx]
                prev_timestep = prev_timestep.to(dtype=torch.long, device=device)

                original_noised = inpainting_scheduler.add_noise(process_region_b, base_noise, prev_timestep)

                latents = prev_sample * mask_b + original_noised * (1.0 - mask_b)
            else:
                # Final step
                latents = prev_sample * mask_b + process_region_b * (1.0 - mask_b)

    return latents.squeeze(0)


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Optimized Gap-Filling Inpainting Evaluation ===")
    logger.info(f"Unconditional model: {args.model_path}")
    logger.info(f"Inpainting model: {args.inpainting_model_path}")
    logger.info(f"Will generate {args.n_blocks} blocks with {args.gap_size} voxel gaps")
    logger.info(f"Optimizations: larger_regions={args.use_larger_regions}, iterative={args.iterative_inpaint}")
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
        # Load models
        logger.info("Loading models...")

        # Load base model
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

        # Generate volume with optimized gap-filling
        logger.info("Starting optimized gap-filling generation...")

        final_volume = generate_blocks_with_optimized_gap_filling(
            base_unet=base_unet,
            base_scheduler=base_scheduler,
            inpainting_pipeline=inpainting_pipeline,
            num_inference_steps=args.inference_steps,
            seed=args.seed,
            n_blocks=args.n_blocks,
            gap_size=args.gap_size,
            device=device,
            output_dir=output_dir if args.save_debug else None,
            generation_batch_size=args.generation_batch_size,
            use_larger_regions=args.use_larger_regions,
            iterative_inpaint=args.iterative_inpaint,
            num_iterations=args.num_iterations,
        )

        if final_volume is None:
            logger.error("Optimized gap-filling generation failed")
            sys.exit(1)

        logger.info(f"Optimized gap-filling generation completed. Final volume shape: {final_volume.shape}")

        # Apply thresholding if specified
        if args.threshold_value is not None:
            logger.info(f"Applying threshold: {args.threshold_value}")
            final_volume = (final_volume > args.threshold_value).astype(np.float32)

        # Save as VTI file
        if PYVISTA_AVAILABLE:
            try:
                logger.info("Saving optimized gap-filled volume as VTI...")
                D_total, H, W = final_volume.shape

                grid = pv.ImageData()
                grid.dimensions = np.array([W, H, D_total]) + 1
                grid.origin = (0, 0, 0)
                grid.spacing = (1, 1, 1)

                volume_data = final_volume.copy()
                if args.binary:
                    volume_data = (volume_data > 0.5).astype(np.float32)

                grid.cell_data["values"] = np.ascontiguousarray(volume_data).flatten(order="C")

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                vti_filename = (
                    f"optimized_gap_filled_{timestamp}_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.vti"
                )
                vti_save_path = output_dir / vti_filename
                grid.save(str(vti_save_path), binary=True)
                logger.info(f"Optimized volume saved as VTI to: {vti_save_path}")

            except Exception as e:
                logger.error(f"Error saving VTI file: {e}")

        # Save numpy array
        output_file = output_dir / f"optimized_gap_filled_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.npy"
        logger.info(f"Saving final volume to: {output_file}")
        np.save(output_file, final_volume)

        # Print summary
        logger.info("=== SUMMARY ===")
        logger.info(f"Generated volume shape: {final_volume.shape}")
        logger.info(f"Value range: [{final_volume.min():.3f}, {final_volume.max():.3f}]")
        logger.info(f"Mean value: {final_volume.mean():.3f}")

        if args.binary or args.threshold_value is not None:
            threshold = args.threshold_value if args.threshold_value is not None else 0.5
            fill_ratio = np.mean(final_volume > threshold)
            logger.info(f"Fill ratio (threshold >{threshold}): {fill_ratio:.3f}")

        logger.info("Optimized gap-filling evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
