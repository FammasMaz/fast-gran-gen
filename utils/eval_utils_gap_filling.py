import torch
import numpy as np
import argparse
from tqdm.auto import tqdm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def numpy_to_pt(np_array):
    """Convert numpy array to PyTorch tensor."""
    return torch.from_numpy(np_array).float()


def pt_to_numpy(pt_tensor):
    """Convert PyTorch tensor to numpy array."""
    return pt_tensor.detach().cpu().numpy()


def generate_blocks_with_gaps_and_inpaint(
    base_unet,
    base_scheduler,
    inpainting_pipeline,
    num_inference_steps,
    seed,
    n_blocks,
    gap_size,  # Size of gap between blocks
    device,
    args,
    output_dir=None,
    generation_batch_size=1,
):
    """
    Generate separate voxel blocks with gaps, then use inpainting to fill the gaps between them.

    This implements the user's intended workflow:
    1. Generate n_blocks separate, non-overlapping blocks using unconditional model
    2. Place them with gaps between them
    3. Use inpainting model to fill gaps, treating blocks as context (unmasked) and gaps as target (masked)

    Args:
        base_unet: Unconditional model for generating individual blocks
        base_scheduler: Scheduler for unconditional generation
        inpainting_pipeline: Inpainting model pipeline
        num_inference_steps: Number of inference steps
        seed: Random seed
        n_blocks: Number of blocks to generate
        gap_size: Size of gap between blocks (in voxels)
        device: PyTorch device
        args: Additional arguments
        output_dir: Directory for saving debug outputs
        generation_batch_size: Batch size for block generation

    Returns:
        Final stitched volume as numpy array [0, 1]
    """
    logger.info(f"Starting gap-filling approach: {n_blocks} blocks with {gap_size} voxel gaps")
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
    num_gen_batches = (n_blocks + generation_batch_size - 1) // generation_batch_size
    current_seed = seed

    pbar_gen = tqdm(total=n_blocks, desc="Generating Independent Blocks")

    for i in range(num_gen_batches):
        batch_seed = current_seed + i * generation_batch_size
        bs = min(generation_batch_size, n_blocks - len(all_blocks_np))
        if bs <= 0:
            break

        logger.debug(f"Generating block batch {i + 1}/{num_gen_batches} with size {bs} and seed {batch_seed}")

        try:
            # Create temporary args for unconditional generation
            temp_args_for_block_gen = argparse.Namespace(**vars(args))
            temp_args_for_block_gen.mask_type = "none"  # Ensure unconditional generation

            # Generate blocks using unconditional model
            block_batch_np = generate_single_volume(
                base_unet,
                base_scheduler,
                num_inference_steps,
                batch_seed,
                bs,
                device,
                args=temp_args_for_block_gen,
                min_bw_ratio=0.0,
                max_retries=0,
                progress_callback=lambda: pbar_gen.update(1),
            )

            if block_batch_np is None or len(block_batch_np) != bs:
                raise RuntimeError(f"generate_single_volume failed for batch {i + 1}")
            all_blocks_np.extend(block_batch_np)

        except Exception as e:
            logger.error(f"Error generating block batch {i + 1}: {e}")
            pbar_gen.close()
            return None

    pbar_gen.close()
    if len(all_blocks_np) != n_blocks:
        logger.error(f"Failed to generate required number of blocks ({len(all_blocks_np)}/{n_blocks})")
        return None
    logger.info(f"Successfully generated {n_blocks} independent blocks")

    # Step 2: Calculate total volume size and place blocks with gaps
    logger.info("Step 2: Placing blocks with gaps")
    total_depth = n_blocks * D + (n_blocks - 1) * gap_size
    logger.info(f"Total volume depth: {total_depth} ({n_blocks} blocks × {D} + {n_blocks - 1} gaps × {gap_size})")

    C = base_unet.config.in_channels
    full_volume_pt = torch.zeros((1, C, total_depth, H, W), dtype=torch.float32, device="cpu")

    # Place blocks in the volume with gaps
    block_positions = []  # Store (start, end) positions of each block
    gap_positions = []  # Store (start, end) positions of each gap

    current_pos = 0
    for i, block_np in enumerate(all_blocks_np):
        if block_np.ndim == 3:
            block_np = np.expand_dims(block_np, axis=0)  # Add channel dim

        # Ensure block has correct number of channels
        if block_np.shape[0] == 1 and C > 1:
            block_np = np.repeat(block_np, C, axis=0)
        elif block_np.shape[0] != C:
            logger.warning(f"Channel mismatch for block {i}: got {block_np.shape[0]}, expected {C}")

        # Place block
        block_pt = numpy_to_pt(block_np)
        end_pos = current_pos + D
        full_volume_pt[0, :, current_pos:end_pos, :, :] = block_pt
        block_positions.append((current_pos, end_pos))

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

    # Move to device
    full_volume_pt = full_volume_pt.to(device)

    # Step 3: Inpaint the gaps
    logger.info("Step 3: Inpainting gaps between blocks")
    inpainting_unet = inpainting_pipeline.unet
    inpainting_scheduler = inpainting_pipeline.scheduler
    inpainting_scheduler.set_timesteps(num_inference_steps)
    inpainting_unet.eval()

    pbar_gap = tqdm(total=len(gap_positions), desc="Inpainting Gaps")
    inpainting_base_seed = seed + n_blocks

    for gap_idx, (gap_start, gap_end) in enumerate(gap_positions):
        logger.info(f"Inpainting gap {gap_idx + 1}/{len(gap_positions)} from {gap_start} to {gap_end}")

        # Create processing region around the gap (include some context from adjacent blocks)
        # Ensure dimensions are compatible with UNet architecture (divisible by 8)
        context_size = min(D // 4, gap_size)  # Use some adjacent block context
        raw_process_start = max(0, gap_start - context_size)
        raw_process_end = min(total_depth, gap_end + context_size)
        raw_process_depth = raw_process_end - raw_process_start

        # Round to make compatible with UNet (ensure divisible by 8)
        target_process_depth = ((raw_process_depth + 7) // 8) * 8

        # Adjust start and end to achieve target depth while staying within bounds
        extra_needed = target_process_depth - raw_process_depth
        extra_start = extra_needed // 2
        extra_end = extra_needed - extra_start

        process_start = max(0, raw_process_start - extra_start)
        process_end = min(total_depth, raw_process_end + extra_end)
        process_depth = process_end - process_start

        # If we still don't have the right size, pad to the nearest multiple of 8
        if process_depth % 8 != 0:
            target_depth = ((process_depth + 7) // 8) * 8
            padding_needed = target_depth - process_depth

            # Try to expand both directions equally
            pad_start = padding_needed // 2
            pad_end = padding_needed - pad_start

            new_start = max(0, process_start - pad_start)
            new_end = min(total_depth, process_end + pad_end)

            # If we can't expand enough, just use what we can get and pad later
            if new_end - new_start != target_depth:
                # Fall back to the original D size if possible
                if D % 8 == 0:
                    target_depth = D
                    center = (gap_start + gap_end) // 2
                    process_start = max(0, center - target_depth // 2)
                    process_end = min(total_depth, process_start + target_depth)
                    if process_end - process_start != target_depth:
                        process_start = max(0, process_end - target_depth)
                    process_depth = process_end - process_start
                else:
                    # Use 32 as a safe fallback (most common UNet compatible size)
                    target_depth = 32
                    center = (gap_start + gap_end) // 2
                    process_start = max(0, center - target_depth // 2)
                    process_end = min(total_depth, process_start + target_depth)
                    if process_end - process_start != target_depth:
                        process_start = max(0, process_end - target_depth)
                    process_depth = process_end - process_start
            else:
                process_start = new_start
                process_end = new_end
                process_depth = process_end - process_start

        logger.info(f"Processing region: {process_start}:{process_end} (depth={process_depth}, includes context)")

        # Extract processing region
        process_region = full_volume_pt[0, :, process_start:process_end, :, :].clone()

        # If the extracted region doesn't match expected size, pad with zeros
        if process_region.shape[1] != process_depth:
            logger.warning(f"Processing region size mismatch. Expected {process_depth}, got {process_region.shape[1]}")
            # Create properly sized region and copy data
            padded_region = torch.zeros((C, process_depth, H, W), device=device)
            actual_size = min(process_region.shape[1], process_depth)
            padded_region[:, :actual_size, :, :] = process_region[:, :actual_size, :, :]
            process_region = padded_region

        # Create mask: 1 for gap (unknown), 0 for blocks (known)
        mask = torch.zeros(1, 1, process_depth, H, W, device=device)
        local_gap_start = max(0, gap_start - process_start)
        local_gap_end = min(process_depth, gap_end - process_start)
        if local_gap_end > local_gap_start:
            mask[0, 0, local_gap_start:local_gap_end, :, :] = 1.0

        logger.info(f"Mask covers local region {local_gap_start}:{local_gap_end} within processing region")

        # Add batch dimension
        process_region_b = process_region.unsqueeze(0).to(device)  # (1, C, D, H, W)
        mask_b = mask.to(device)  # (1, 1, D, H, W)

        # Prepare for inpainting
        gap_seed = inpainting_base_seed + gap_idx
        generator = torch.Generator(device=device).manual_seed(gap_seed)

        # Initial latents (noise) for the entire processing region
        latents = torch.randn(process_region_b.shape, generator=generator, device=device)

        # Run inpainting diffusion loop
        try:
            inpainting_scheduler.set_timesteps(num_inference_steps)
            timesteps = inpainting_scheduler.timesteps

            # Sample the base noise for original content once before the loop
            base_noise_for_original_content = torch.randn(process_region_b.shape, generator=generator, device=device)

            with torch.no_grad():
                for t in tqdm(timesteps, desc=f"Inpainting Gap {gap_idx + 1}", leave=False):
                    # Create masked image: known regions keep original values, unknown regions are zeros
                    masked_image = process_region_b * (1.0 - mask_b)

                    # Scale latents for model input
                    scaled_latents = inpainting_scheduler.scale_model_input(latents, t)

                    # Concatenate inputs for inpainting model: [latents, mask, masked_image]
                    unet_input = torch.cat([scaled_latents, mask_b, masked_image], dim=1)

                    # Predict noise
                    t_input = t.repeat(process_region_b.shape[0]).to(device)
                    noise_pred = inpainting_unet(unet_input, t_input, return_dict=False)[0]

                    # Scheduler step
                    step_output = inpainting_scheduler.step(noise_pred, t, latents)
                    prev_sample = step_output.prev_sample

                    # For known regions (mask=0), blend with original content
                    # This ensures we don't modify the existing blocks, only fill the gap
                    if t != timesteps[-1]:  # Not the final step
                        # Add appropriate noise to original content to match current timestep
                        timestep_idx = (t == inpainting_scheduler.timesteps).nonzero().item()
                        prev_timestep_idx = min(timestep_idx + 1, len(inpainting_scheduler.timesteps) - 1)
                        prev_timestep = inpainting_scheduler.timesteps[prev_timestep_idx]
                        prev_timestep = prev_timestep.to(dtype=torch.long, device=device)

                        # Add noise to original image to match previous timestep
                        # Use the fixed base_noise_for_original_content
                        original_noised = inpainting_scheduler.add_noise(
                            process_region_b, base_noise_for_original_content, prev_timestep
                        )

                        # Combine: use predicted sample in gap, original (noised) content in known regions
                        latents = prev_sample * mask_b + original_noised * (1.0 - mask_b)
                    else:
                        # Final step: use original content for known regions, predicted for gap
                        latents = prev_sample * mask_b + process_region_b * (1.0 - mask_b)

            # Place the inpainted result back into the full volume
            inpainted_result = latents.squeeze(0)  # Remove batch dim

            # Handle potential size mismatch when placing back
            original_region_size = process_end - process_start
            if inpainted_result.shape[1] == original_region_size:
                # Direct placement
                full_volume_pt[0, :, process_start:process_end, :, :] = inpainted_result
            else:
                # Size mismatch - extract the relevant portion
                actual_size = min(inpainted_result.shape[1], original_region_size)
                full_volume_pt[0, :, process_start : process_start + actual_size, :, :] = inpainted_result[
                    :, :actual_size, :, :
                ]
                logger.info(
                    f"Placed {actual_size} slices back into volume (processed {inpainted_result.shape[1]} slices)"
                )

            logger.info(f"Gap {gap_idx + 1} inpainting complete")

            # Save debug output
            if output_dir:
                debug_dir = Path(output_dir) / f"gap_inpainting_debug_seed{seed}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                gap_result_np = pt_to_numpy(inpainted_result)
                np.save(debug_dir / f"gap_{gap_idx:02d}_inpainted.npy", gap_result_np)
                mask_np = pt_to_numpy(mask[0, 0])
                np.save(debug_dir / f"gap_{gap_idx:02d}_mask.npy", mask_np)

        except Exception as e:
            logger.error(f"Error inpainting gap {gap_idx + 1}: {e}")
            import traceback

            traceback.print_exc()
            pbar_gap.close()
            return None

        pbar_gap.update(1)

    pbar_gap.close()
    logger.info("All gaps inpainted successfully")

    # Convert final result to numpy
    final_volume_np = pt_to_numpy(full_volume_pt.squeeze(0))  # Remove batch dim
    if final_volume_np.shape[0] == 1:
        final_volume_np = final_volume_np.squeeze(0)  # Remove channel dim if single channel

    logger.info(f"Final volume shape: {final_volume_np.shape}")
    return final_volume_np


# Import the generate_single_volume function from the existing eval_utils
from .eval_utils import generate_single_volume
