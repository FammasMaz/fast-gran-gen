#!/usr/bin/env python3
"""
Fixed gap-filling evaluation script that properly handles inpainting.

This version fixes the noise issues by:
1. Using proper RePaint guidance for inpainting
2. Correctly handling the noise scheduling
3. Simpler, more robust implementation
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import time
from tqdm.auto import tqdm

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
import argparse as args_module


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fixed gap-filling inpainting evaluation")

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
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for repaint")

    # Output
    parser.add_argument("--output_dir", type=str, default="out/gap_filling_fixed/", help="Output directory")
    parser.add_argument("--save_debug", action="store_true", help="Save debug outputs for each gap")

    # Thresholding
    parser.add_argument("--binary", action="store_true", help="Threshold output to binary mask (>0.5)")

    return parser.parse_args()


def inpaint_gap_repaint(
    inpainting_pipeline, volume_with_gap, mask, num_inference_steps, seed, device, guidance_scale=1.0
):
    """
    Inpaint a gap using RePaint guidance approach.

    Args:
        inpainting_pipeline: The inpainting diffusion pipeline
        volume_with_gap: [1, C, D, H, W] volume with gap (zeros in gap region)
        mask: [1, 1, D, H, W] mask (1=gap to fill, 0=keep original)
        num_inference_steps: Number of diffusion steps
        seed: Random seed
        device: PyTorch device
        guidance_scale: Guidance scale

    Returns:
        Inpainted volume [1, C, D, H, W]
    """
    from copy import deepcopy

    generator = torch.Generator(device=device).manual_seed(seed)

    # Get scheduler and unet
    scheduler = inpainting_pipeline.scheduler
    unet = inpainting_pipeline.unet

    # Create a local copy of scheduler to avoid device conflicts
    local_scheduler = deepcopy(scheduler)
    local_scheduler.set_timesteps(num_inference_steps)

    # Move scheduler tensors to device if they exist
    if hasattr(local_scheduler, "timesteps"):
        local_scheduler.timesteps = local_scheduler.timesteps.to(device)
    if hasattr(local_scheduler, "alphas_cumprod"):
        local_scheduler.alphas_cumprod = local_scheduler.alphas_cumprod.to(device)
    if hasattr(local_scheduler, "betas"):
        local_scheduler.betas = local_scheduler.betas.to(device)

    timesteps = local_scheduler.timesteps

    # Initialize noise
    latents = torch.randn(volume_with_gap.shape, generator=generator, device=device)

    # Noise for original content
    noise_for_known = torch.randn(volume_with_gap.shape, generator=generator, device=device)

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Inpainting", leave=False)):
            # Ensure t is on correct device
            t = t.to(device)

            # Prepare masked image (known regions)
            masked_image = volume_with_gap * (1.0 - mask)

            # Scale model input
            latent_model_input = local_scheduler.scale_model_input(latents, t)

            # Concatenate for inpainting: [latents, mask, masked_image]
            unet_input = torch.cat([latent_model_input, mask, masked_image], dim=1)

            # Predict noise - ensure t is on correct device
            t_tensor = t.repeat(volume_with_gap.shape[0]).to(device)
            noise_pred = unet(unet_input, t_tensor, return_dict=False)[0]

            # Scheduler step
            latents = local_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

            # RePaint: replace known regions with properly noised original content
            if i < len(timesteps) - 1:  # Not the last step
                next_t = timesteps[i + 1].to(device)
                # Add noise to original content - ensure next_t is on correct device
                noised_original = local_scheduler.add_noise(volume_with_gap, noise_for_known, next_t.unsqueeze(0))
                # Keep predicted for unknown (mask=1), replace with noised original for known (mask=0)
                latents = latents * mask + noised_original * (1.0 - mask)
            else:
                # Last step: use clean original for known regions
                latents = latents * mask + volume_with_gap * (1.0 - mask)

    return latents


def generate_single_volume_fixed(
    unet, scheduler, num_steps, seed, batch_size, device, min_bw_ratio=0.0, max_retries=5
):
    """
    Fixed version of generate_single_volume that properly handles device placement for DDPM.
    """
    volumes_np_0_1 = []

    # Get model dimensions
    sample_size = unet.config.sample_size
    if isinstance(sample_size, int):
        D, H, W = sample_size, sample_size, sample_size
    else:
        D, H, W = sample_size
    C = unet.config.in_channels

    # Generate initial latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((batch_size, C, D, H, W), generator=generator, device=device)

    from copy import deepcopy

    local_scheduler = deepcopy(scheduler)
    local_scheduler.set_timesteps(num_steps)

    # Move scheduler tensors to device if they exist
    if hasattr(local_scheduler, "timesteps"):
        local_scheduler.timesteps = local_scheduler.timesteps.to(device)
    if hasattr(local_scheduler, "alphas_cumprod"):
        local_scheduler.alphas_cumprod = local_scheduler.alphas_cumprod.to(device)
    if hasattr(local_scheduler, "betas"):
        local_scheduler.betas = local_scheduler.betas.to(device)

    timesteps = local_scheduler.timesteps

    unet.eval()
    with torch.no_grad():
        for t in timesteps:
            # Ensure t is on the correct device
            t = t.to(device)
            scaled_latents = local_scheduler.scale_model_input(latents, t)

            # For unconditional generation, just use scaled latents
            model_input = scaled_latents

            # Ensure timestep tensor is on correct device
            t_input = t.repeat(batch_size).to(device)
            noise_pred = unet(model_input, t_input, return_dict=False)[0]

            # Use the scheduler step with proper device handling
            latents = local_scheduler.step(noise_pred, t, latents).prev_sample

    # Process each volume in the batch
    for i in range(batch_size):
        volume_meets_criteria = False
        retries = 0
        last_clipped_vol = None

        while not volume_meets_criteria and retries <= max_retries:
            if retries > 0:
                logger.info(f"  Retrying volume {i + 1}/{batch_size} (attempt {retries}/{max_retries})...")
                volume_seed = torch.randint(0, 2**32, (1,)).item()

                volume_generator = torch.Generator(device=device).manual_seed(volume_seed)
                single_latent = torch.randn((1, C, D, H, W), generator=volume_generator, device=device)

                # Create another scheduler instance for retry
                retry_scheduler = deepcopy(scheduler)
                retry_scheduler.set_timesteps(num_steps)

                # Move scheduler tensors to device
                if hasattr(retry_scheduler, "timesteps"):
                    retry_scheduler.timesteps = retry_scheduler.timesteps.to(device)
                if hasattr(retry_scheduler, "alphas_cumprod"):
                    retry_scheduler.alphas_cumprod = retry_scheduler.alphas_cumprod.to(device)
                if hasattr(retry_scheduler, "betas"):
                    retry_scheduler.betas = retry_scheduler.betas.to(device)

                timesteps_retry = retry_scheduler.timesteps

                with torch.no_grad():
                    for t in timesteps_retry:
                        t = t.to(device)
                        latent_model_input = retry_scheduler.scale_model_input(single_latent, t)
                        t_input = t.repeat(1).to(device)
                        noise_pred = unet(latent_model_input, t_input, return_dict=False)[0]
                        single_latent = retry_scheduler.step(noise_pred, t, single_latent).prev_sample

                block_cpu = single_latent[0].cpu().squeeze(0)  # (D, H, W)
                del single_latent
            else:
                block_cpu = latents[i].cpu().squeeze(0)  # (D, H, W)

            vol_np = block_cpu.detach().numpy()
            scaled_vol = (vol_np + 1.0) / 2.0
            clipped_vol = np.clip(scaled_vol, 0.0, 1.0)

            last_clipped_vol = clipped_vol

            bw_ratio = np.mean(clipped_vol <= 0.5)
            logger.info(f"  Volume {i + 1}/{batch_size} BW ratio: {bw_ratio:.4f}")

            if bw_ratio >= min_bw_ratio or min_bw_ratio <= 0:
                volume_meets_criteria = True
                volumes_np_0_1.append(clipped_vol)
            else:
                retries += 1
                del vol_np, scaled_vol, clipped_vol, block_cpu
                torch.cuda.empty_cache()

        if not volume_meets_criteria:
            logger.warning(
                f"Minimum BW ratio ({min_bw_ratio}) not met for volume {i} after {max_retries} retries. Using last generated volume."
            )
            if last_clipped_vol is not None:
                volumes_np_0_1.append(last_clipped_vol)
            else:
                logger.error(f"Could not generate volume for index {i}")
                return None

    del latents
    torch.cuda.empty_cache()
    return volumes_np_0_1


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Fixed Gap-Filling Inpainting Evaluation ===")
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
        # Load models
        logger.info("Loading models...")

        # Load base unconditional model
        logger.info(f"Loading BASE diffusion pipeline from: {args.model_path}")
        base_pipeline = DDPMPipeline.from_pretrained(args.model_path).to(device)
        if args.scheduler_type == "ddim":
            base_pipeline.scheduler = DDIMScheduler.from_pretrained(Path(args.model_path) / "scheduler")

        # Load inpainting model
        logger.info(f"Loading INPAINTING diffusion pipeline from: {args.inpainting_model_path}")
        inpainting_pipeline = DDPMPipeline.from_pretrained(args.inpainting_model_path).to(device)
        if args.scheduler_type == "ddim":
            inpainting_pipeline.scheduler = DDIMScheduler.from_pretrained(
                Path(args.inpainting_model_path) / "scheduler"
            )

        logger.info("Models loaded successfully")

        # Get dimensions
        sample_size = base_pipeline.unet.config.sample_size
        if isinstance(sample_size, int):
            D, H, W = sample_size, sample_size, sample_size
        else:
            D, H, W = sample_size
        C = base_pipeline.unet.config.in_channels

        logger.info(f"Block dimensions: D={D}, H={H}, W={W}, C={C}")

        # Step 1: Generate individual blocks
        logger.info("Step 1: Generating individual blocks...")

        # Create minimal args for compatibility
        minimal_args = args_module.Namespace()
        minimal_args.mask_type = "none"
        minimal_args.central_block_min_ratio = 0.3
        minimal_args.central_block_max_ratio = 0.7
        minimal_args.central_block_jitter_factor = 0.1

        blocks = []
        for i in range(args.n_blocks):
            block_seed = args.seed + i
            logger.info(f"Generating block {i + 1}/{args.n_blocks} with seed {block_seed}")

            block_list = generate_single_volume_fixed(
                base_pipeline.unet,
                base_pipeline.scheduler,
                args.inference_steps,
                block_seed,
                1,  # batch_size=1
                device,
                min_bw_ratio=0.0,
                max_retries=0,
            )

            if block_list is None or len(block_list) == 0:
                logger.error(f"Failed to generate block {i + 1}")
                sys.exit(1)

            block = block_list[0]  # Get first (and only) block from batch
            if block.ndim == 3:
                block = np.expand_dims(block, axis=0)  # Add channel dim

            # Ensure correct number of channels
            if block.shape[0] != C:
                if block.shape[0] == 1 and C > 1:
                    block = np.repeat(block, C, axis=0)
                else:
                    logger.error(f"Channel mismatch: block has {block.shape[0]}, expected {C}")
                    sys.exit(1)

            blocks.append(block)

        logger.info(f"Generated {len(blocks)} blocks successfully")

        # Step 2: Assemble volume with gaps
        logger.info("Step 2: Assembling volume with gaps...")

        total_depth = args.n_blocks * D + (args.n_blocks - 1) * args.gap_size
        full_volume = np.zeros((C, total_depth, H, W), dtype=np.float32)

        gap_positions = []
        current_pos = 0

        for i, block in enumerate(blocks):
            # Place block
            end_pos = current_pos + D
            full_volume[:, current_pos:end_pos, :, :] = block
            logger.info(f"Placed block {i + 1} at position {current_pos}:{end_pos}")

            # Record gap position (except after last block)
            if i < len(blocks) - 1:
                gap_start = end_pos
                gap_end = gap_start + args.gap_size
                gap_positions.append((gap_start, gap_end))
                logger.info(f"Gap {i + 1} will be at position {gap_start}:{gap_end}")
                current_pos = gap_end
            else:
                current_pos = end_pos

        # Convert to torch and move to device
        full_volume_pt = torch.from_numpy(full_volume).unsqueeze(0).to(device)  # Add batch dim

        # Step 3: Inpaint each gap
        logger.info(f"Step 3: Inpainting {len(gap_positions)} gaps...")

        for gap_idx, (gap_start, gap_end) in enumerate(gap_positions):
            logger.info(f"Inpainting gap {gap_idx + 1}/{len(gap_positions)}: {gap_start} to {gap_end}")

            # Create processing window around the gap
            context_size = D // 4  # Some context from adjacent blocks
            process_start = max(0, gap_start - context_size)
            process_end = min(total_depth, gap_end + context_size)

            # Ensure size is compatible with UNet (multiple of 8)
            process_depth = process_end - process_start
            if process_depth % 8 != 0:
                target_depth = ((process_depth + 7) // 8) * 8
                # Try to center the window
                center = (process_start + process_end) // 2
                new_start = max(0, center - target_depth // 2)
                new_end = min(total_depth, new_start + target_depth)
                if new_end - new_start != target_depth:
                    new_start = max(0, new_end - target_depth)
                process_start, process_end = new_start, new_end
                process_depth = target_depth

            logger.info(f"Processing window: {process_start}:{process_end} (depth={process_depth})")

            # Extract processing region
            process_region = full_volume_pt[:, :, process_start:process_end, :, :]

            # Create mask for the gap within this region
            mask = torch.zeros(1, 1, process_depth, H, W, device=device)
            local_gap_start = max(0, gap_start - process_start)
            local_gap_end = min(process_depth, gap_end - process_start)
            mask[0, 0, local_gap_start:local_gap_end, :, :] = 1.0

            logger.info(f"Mask covers local range {local_gap_start}:{local_gap_end}")

            # Inpaint the gap
            gap_seed = args.seed + 1000 + gap_idx
            inpainted_region = inpaint_gap_repaint(
                inpainting_pipeline, process_region, mask, args.inference_steps, gap_seed, device, args.guidance_scale
            )

            # Place back into full volume
            full_volume_pt[:, :, process_start:process_end, :, :] = inpainted_region

            logger.info(f"Gap {gap_idx + 1} inpainting complete")

            # Save debug output
            if args.save_debug:
                debug_dir = output_dir / f"debug_seed{args.seed}"
                debug_dir.mkdir(exist_ok=True)
                gap_result = inpainted_region[0].cpu().numpy()  # Remove batch dim
                np.save(debug_dir / f"gap_{gap_idx:02d}_inpainted.npy", gap_result)
                mask_np = mask[0, 0].cpu().numpy()
                np.save(debug_dir / f"gap_{gap_idx:02d}_mask.npy", mask_np)

        # Convert final result
        final_volume = full_volume_pt[0].cpu().numpy()  # Remove batch dim
        if final_volume.shape[0] == 1:
            final_volume = final_volume[0]  # Remove channel dim if single channel

        logger.info(f"Final volume shape: {final_volume.shape}")

        # Apply binary threshold if requested
        if args.binary:
            final_volume = (final_volume > 0.5).astype(np.float32)
            logger.info("Applied binary threshold (>0.5)")

        # Save as VTI file using PyVista
        if PYVISTA_AVAILABLE:
            try:
                logger.info("Saving gap-filled volume as VTI...")
                if final_volume.ndim == 4:
                    # Multi-channel, use first channel
                    volume_for_vti = final_volume[0]
                else:
                    volume_for_vti = final_volume

                D_total, H_vti, W_vti = volume_for_vti.shape

                grid = pv.ImageData()
                grid.dimensions = np.array([W_vti, H_vti, D_total]) + 1
                grid.origin = (0, 0, 0)
                grid.spacing = (1, 1, 1)
                grid.cell_data["values"] = np.ascontiguousarray(volume_for_vti).flatten(order="C")

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                vti_filename = (
                    f"gap_filled_FIXED_{timestamp}_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.vti"
                )
                vti_save_path = output_dir / vti_filename
                grid.save(str(vti_save_path), binary=True)
                logger.info(f"VTI saved to: {vti_save_path}")

            except Exception as e:
                logger.error(f"Error saving VTI file: {e}")

        # Save the final result
        output_file = output_dir / f"gap_filled_FIXED_seed{args.seed}_blocks{args.n_blocks}_gap{args.gap_size}.npy"
        logger.info(f"Saving final volume to: {output_file}")
        np.save(output_file, final_volume)

        # Print summary statistics
        logger.info("=== SUMMARY ===")
        logger.info(f"Generated volume shape: {final_volume.shape}")
        logger.info(f"Value range: [{final_volume.min():.3f}, {final_volume.max():.3f}]")
        logger.info(f"Mean value: {final_volume.mean():.3f}")

        if args.binary:
            fill_ratio = np.mean(final_volume > 0.5)
            logger.info(f"Fill ratio (binary): {fill_ratio:.3f}")
        else:
            fill_ratio = np.mean(final_volume > 0.0)
            logger.info(f"Non-zero ratio: {fill_ratio:.3f}")

        logger.info("Fixed gap-filling evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
