import torch
import numpy as np
import argparse
import os
from diffusers import DDPMPipeline, DDIMScheduler, DiffusionPipeline
from pathlib import Path
import time
from tqdm.auto import tqdm
from utils.eval_utils import (
    generate_single_volume,
    generate_stitched_volume_with_inpainting,
    numpy_to_pt,
    pt_to_numpy,
)
from modules.trainer import MaskGenerator3D
import pyvista as pv


def load_models(model_path, inpainting_model_path, scheduler_type="ddim"):
    """
    Load models once and return them. This avoids repeated loading.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = Path(model_path)
    print(f"Loading BASE diffusion pipeline from: {model_path}")

    try:
        pipeline = DDPMPipeline.from_pretrained(model_path).to(device)
        if scheduler_type == "ddim":
            pipeline.scheduler = DDIMScheduler.from_pretrained(model_path / "scheduler")
        unet = pipeline.unet
        scheduler = pipeline.scheduler
        print(f"Base pipeline loaded with {scheduler_type.upper()} scheduler.")

        # Apply workaround if needed
        if hasattr(unet, "_original_in_channels") and "_original_in_channels" not in unet.config:
            try:
                unet.config["_original_in_channels"] = unet._original_in_channels
                print("Applied workaround: Added '_original_in_channels' to unet.config")
            except TypeError:
                print("Warning: Could not directly add attribute to unet.config (FrozenDict is immutable).")

    except Exception as e:
        print(f"Error loading BASE pipeline: {e}")
        return None, None, None, None

    # Load inpainting model
    inpainting_pipeline = None
    if inpainting_model_path:
        inpainting_model_path = Path(inpainting_model_path)
        print(f"Loading INPAINTING diffusion pipeline from: {inpainting_model_path}")
        try:
            inpainting_pipeline = DiffusionPipeline.from_pretrained(inpainting_model_path).to(device)
            print("Inpainting pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading INPAINTING pipeline: {e}")
            return None, None, None, None

    return unet, scheduler, inpainting_pipeline, device


def create_railway_track_1d(
    unet,
    scheduler,
    inpainting_pipeline,
    device,
    output_dir,
    n_blocks_length,
    overlap=8,
    inference_steps=60,
    seed=123,
    stitching_mode="separate_inpainting",
    mask_type="gap_filling_compatible",
    batch_size=1,
    binary=False,
    **kwargs,
):
    """
    Create a 1D railway track using pre-loaded models.
    Uses the existing generate_stitched_volume_with_inpainting function.
    """
    print(f"    → Generating 1D track: {n_blocks_length} blocks, overlap={overlap}")

    # Create args object
    args = argparse.Namespace(mask_type=mask_type, binary=binary, **kwargs)

    # Use the existing stitched volume function (same as eval.py)
    # Note: With mask_type="gap_filling_compatible", this creates:
    # [Block1] [GAP] [Block2] [GAP] [Block3] ... then inpaints the gaps
    final_stitched_volume_np = generate_stitched_volume_with_inpainting(
        unet,
        scheduler,
        inpainting_pipeline,
        inference_steps,
        seed,
        n_blocks_length,
        overlap,  # In gap mode, this becomes gap_size
        device,
        args=args,
        output_dir=Path(output_dir) / "stitched_debug",
        generation_batch_size=batch_size,
        strength=1.0,
        inpaint_region_size_ratio=kwargs.get("inpaint_region_size_ratio", 0.3),
        inpaint_iteratively=kwargs.get("inpaint_iteratively", False),
        inpaint_iterations=kwargs.get("inpaint_iterations", 3),
        threshold_value=kwargs.get("threshold_value", None),
    )

    return final_stitched_volume_np


def stitch_volumes_along_axis_with_inpainting(
    volumes,
    axis,
    overlap,
    device,
    inpainting_pipeline,
    inference_steps,
    seed,
    mask_type="gap_filling_compatible",
    **kwargs,
):
    """
    Stitch multiple volumes along a specified axis using gap-filling or overlapping logic.
    """
    if len(volumes) == 1:
        return volumes[0]

    print(f"    Stitching {len(volumes)} volumes along axis {axis} with overlap/gap={overlap}")

    # Calculate dimensions
    first_vol = volumes[0]
    vol_shape = list(first_vol.shape)
    axis_size = vol_shape[axis]

    # Calculate total size with gap-filling logic (same as eval_utils for gap_filling_compatible)
    # In gap-filling mode: step = D + gap_size, blocks are separated by gaps
    if mask_type == "gap_filling_compatible":
        gap_size = overlap  # In gap mode, "overlap" parameter is actually gap size
        step = axis_size + gap_size  # Blocks are separated by gap_size
        total_axis_size = len(volumes) * axis_size + (len(volumes) - 1) * gap_size
        print(f"    Gap-filling mode: gap_size={gap_size}, step={step}, total={total_axis_size}")
    else:
        # Overlapping mode: blocks overlap and seams are inpainted
        step = axis_size - overlap
        total_axis_size = axis_size + (len(volumes) - 1) * step
        print(f"    Overlapping mode: overlap={overlap}, step={step}, total={total_axis_size}")

    print(f"    Axis {axis}: size={axis_size}, step={step}, total={total_axis_size}")

    # Create output volume and place volumes with gaps or overlaps
    final_shape = vol_shape.copy()
    final_shape[axis] = total_axis_size

    # Convert to PyTorch tensors for inpainting
    C = 1  # Assume single channel for now
    if len(vol_shape) == 3:
        full_volume_pt = torch.zeros((1, C, *final_shape), dtype=torch.float32, device="cpu")
    else:
        full_volume_pt = torch.zeros((1, *final_shape), dtype=torch.float32, device="cpu")

    # Place first volume
    first_vol_pt = numpy_to_pt(first_vol)
    if first_vol_pt.dim() == 3:
        first_vol_pt = first_vol_pt.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif first_vol_pt.dim() == 4:
        first_vol_pt = first_vol_pt.unsqueeze(0)  # Add batch dim

    if axis == 0:
        full_volume_pt[0, :, :axis_size, :, :] = first_vol_pt[0, :, :axis_size, :, :]
    elif axis == 1:
        full_volume_pt[0, :, :, :axis_size, :] = first_vol_pt[0, :, :, :axis_size, :]
    elif axis == 2:
        full_volume_pt[0, :, :, :, :axis_size] = first_vol_pt[0, :, :, :, :axis_size]

    # Place subsequent volumes with gaps or overlaps
    current_pos = step
    for i, volume in enumerate(volumes[1:], 1):
        end_pos = current_pos + axis_size

        vol_pt = numpy_to_pt(volume)
        if vol_pt.dim() == 3:
            vol_pt = vol_pt.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif vol_pt.dim() == 4:
            vol_pt = vol_pt.unsqueeze(0)  # Add batch dim

        if axis == 0:
            full_volume_pt[0, :, current_pos:end_pos, :, :] = vol_pt[0, :, :, :, :]
        elif axis == 1:
            full_volume_pt[0, :, :, current_pos:end_pos, :] = vol_pt[0, :, :, :, :]
        elif axis == 2:
            full_volume_pt[0, :, :, :, current_pos:end_pos] = vol_pt[0, :, :, :, :]

        current_pos += step

    full_volume_pt = full_volume_pt.to(device)

    # Now perform inpainting at junction regions
    if inpainting_pipeline is not None:
        print(f"    Performing inpainting at {len(volumes) - 1} junctions...")

        inpainting_unet = inpainting_pipeline.unet
        inpainting_scheduler = inpainting_pipeline.scheduler
        inpainting_scheduler.set_timesteps(inference_steps)
        inpainting_unet.eval()

        if mask_type == "gap_filling_compatible":
            # Gap-filling mode: inpaint the gaps between blocks
            gap_size = overlap  # In gap mode, overlap parameter is gap size
            for i in range(len(volumes) - 1):
                # Gap center is in the middle of the gap between blocks
                gap_start = (i + 1) * axis_size + i * gap_size  # End of previous block + previous gaps
                gap_center = gap_start + gap_size // 2

                # Define processing region around gap
                process_region_size = max(gap_size * 3, 16)  # Ensure sufficient context
                region_start = max(0, gap_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                print(
                    f"      Inpainting gap {i + 1}: gap_start={gap_start}, gap_center={gap_center}, region={region_start}:{region_end}"
                )

                # Extract region for inpainting
                if axis == 0:
                    image_slice = full_volume_pt[0, :, region_start:region_end, :, :].clone()
                elif axis == 1:
                    image_slice = full_volume_pt[0, :, :, region_start:region_end, :].clone()
                elif axis == 2:
                    image_slice = full_volume_pt[0, :, :, :, region_start:region_end].clone()

                # Create mask for gap region
                region_depth = region_end - region_start
                gap_local_start = gap_start - region_start
                gap_local_end = gap_local_start + gap_size
                gap_local_start = max(0, gap_local_start)
                gap_local_end = min(region_depth, gap_local_end)

                mask = torch.zeros_like(image_slice)
                if axis == 0:
                    mask[:, gap_local_start:gap_local_end, :, :] = 1.0
                elif axis == 1:
                    mask[:, :, gap_local_start:gap_local_end, :] = 1.0
                elif axis == 2:
                    mask[:, :, :, gap_local_start:gap_local_end] = 1.0

                # Perform proper diffusion inpainting
                mask_b = mask.unsqueeze(0).to(device)
                image_slice_b = image_slice.unsqueeze(0).to(device)

                # Create mask for UNet input (single channel)
                mask_unet_input_b = mask[0:1, ...].repeat(1, 1, 1, 1, 1).to(device)  # (B=1, C=1, D, H, W)

                # Generate seed for this gap
                gap_seed = seed + i * 1000 + 42
                generator = torch.Generator(device=device).manual_seed(gap_seed)

                try:
                    # Set up the diffusion process
                    inpainting_scheduler.set_timesteps(inference_steps)
                    timesteps = inpainting_scheduler.timesteps

                    # Create initial noise for the entire region
                    latents = torch.randn(image_slice_b.shape, generator=generator, device=device)

                    # Create masked images (original * (1 - mask) - mask)
                    masked_images = image_slice_b * (1.0 - mask_b) - mask_b

                    # Scale initial noise if needed
                    if hasattr(inpainting_scheduler, "init_noise_sigma"):
                        latents = latents * inpainting_scheduler.init_noise_sigma

                    with torch.no_grad():
                        for i_step, t in enumerate(timesteps):
                            t_input = t.repeat(image_slice_b.shape[0])  # Batch size

                            # For inpainting UNet: concatenate [noisy_latents, mask, masked_images]
                            unet_input = torch.cat([latents, mask_unet_input_b, masked_images], dim=1)

                            # Predict noise
                            noise_pred = inpainting_unet(unet_input, t_input, return_dict=False)[0]

                            # Scheduler step
                            step_output = inpainting_scheduler.step(noise_pred, t, latents, generator=generator)
                            x_prev_denoised_candidate = step_output.prev_sample

                            # RePaint-like guidance: ensure known regions stay consistent
                            if i_step < len(timesteps) - 1:
                                # Add noise to the original image for the next timestep
                                prev_t = timesteps[i_step + 1]
                                noise_for_gt_conditioning = torch.randn(
                                    image_slice_b.shape, device=device, dtype=image_slice_b.dtype
                                )

                                # Add noise to original image
                                x_prev_noised_known_gt = inpainting_scheduler.add_noise(
                                    image_slice_b, noise_for_gt_conditioning, prev_t.unsqueeze(0)
                                )

                                # Composite: use inpainted prediction for masked areas, noisy GT for known areas
                                latents = x_prev_denoised_candidate * mask_b + x_prev_noised_known_gt * (1.0 - mask_b)
                            else:
                                # Last step: composite final x0 prediction
                                latents = x_prev_denoised_candidate * mask_b + image_slice_b * (1.0 - mask_b)

                    # Update with inpainted result
                    image_slice_b = latents

                except Exception as e:
                    print(f"      Warning: Inpainting failed for gap {i + 1}: {e}")
                    # Fallback to simple blending
                    masked_region = mask_b > 0.5
                    if masked_region.any():
                        # Simple blending as fallback
                        image_slice_b[masked_region] = image_slice_b[masked_region] * 0.5

                # Put inpainted region back
                if axis == 0:
                    full_volume_pt[0, :, region_start:region_end, :, :] = image_slice_b[0]
                elif axis == 1:
                    full_volume_pt[0, :, :, region_start:region_end, :] = image_slice_b[0]
                elif axis == 2:
                    full_volume_pt[0, :, :, :, region_start:region_end] = image_slice_b[0]
        else:
            # Overlapping mode: inpaint the seam regions
            current_pos = step
            for i in range(len(volumes) - 1):
                junction_center = current_pos

                # Define processing region around junction
                process_region_size = max(axis_size, overlap * 3)
                region_start = max(0, junction_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                # Extract region for inpainting
                if axis == 0:
                    image_slice = full_volume_pt[0, :, region_start:region_end, :, :].clone()
                elif axis == 1:
                    image_slice = full_volume_pt[0, :, :, region_start:region_end, :].clone()
                elif axis == 2:
                    image_slice = full_volume_pt[0, :, :, :, region_start:region_end].clone()

                # Create mask for junction region
                region_depth = region_end - region_start
                junction_local_center = junction_center - region_start
                mask_width = overlap
                mask_start = max(0, junction_local_center - mask_width // 2)
                mask_end = min(region_depth, mask_start + mask_width)

                mask = torch.zeros_like(image_slice)
                if axis == 0:
                    mask[:, mask_start:mask_end, :, :] = 1.0
                elif axis == 1:
                    mask[:, :, mask_start:mask_end, :] = 1.0
                elif axis == 2:
                    mask[:, :, :, mask_start:mask_end] = 1.0

                # Perform proper diffusion inpainting for overlap regions
                mask_b = mask.unsqueeze(0).to(device)
                image_slice_b = image_slice.unsqueeze(0).to(device)

                # Create mask for UNet input (single channel)
                mask_unet_input_b = mask[0:1, ...].repeat(1, 1, 1, 1, 1).to(device)  # (B=1, C=1, D, H, W)

                # Generate seed for this junction
                junction_seed = seed + i * 1000 + current_pos
                generator = torch.Generator(device=device).manual_seed(junction_seed)

                try:
                    # Set up the diffusion process
                    inpainting_scheduler.set_timesteps(inference_steps)
                    timesteps = inpainting_scheduler.timesteps

                    # Create initial noise for the entire region
                    latents = torch.randn(image_slice_b.shape, generator=generator, device=device)

                    # Create masked images (original * (1 - mask) - mask)
                    masked_images = image_slice_b * (1.0 - mask_b) - mask_b

                    # Scale initial noise if needed
                    if hasattr(inpainting_scheduler, "init_noise_sigma"):
                        latents = latents * inpainting_scheduler.init_noise_sigma

                    with torch.no_grad():
                        for i_step, t in enumerate(timesteps):
                            t_input = t.repeat(image_slice_b.shape[0])  # Batch size

                            # For inpainting UNet: concatenate [noisy_latents, mask, masked_images]
                            unet_input = torch.cat([latents, mask_unet_input_b, masked_images], dim=1)

                            # Predict noise
                            noise_pred = inpainting_unet(unet_input, t_input, return_dict=False)[0]

                            # Scheduler step
                            step_output = inpainting_scheduler.step(noise_pred, t, latents, generator=generator)
                            x_prev_denoised_candidate = step_output.prev_sample

                            # RePaint-like guidance: ensure known regions stay consistent
                            if i_step < len(timesteps) - 1:
                                # Add noise to the original image for the next timestep
                                prev_t = timesteps[i_step + 1]
                                noise_for_gt_conditioning = torch.randn(
                                    image_slice_b.shape, device=device, dtype=image_slice_b.dtype
                                )

                                # Add noise to original image
                                x_prev_noised_known_gt = inpainting_scheduler.add_noise(
                                    image_slice_b, noise_for_gt_conditioning, prev_t.unsqueeze(0)
                                )

                                # Composite: use inpainted prediction for masked areas, noisy GT for known areas
                                latents = x_prev_denoised_candidate * mask_b + x_prev_noised_known_gt * (1.0 - mask_b)
                            else:
                                # Last step: composite final x0 prediction
                                latents = x_prev_denoised_candidate * mask_b + image_slice_b * (1.0 - mask_b)

                    # Update with inpainted result
                    image_slice_b = latents

                except Exception as e:
                    print(f"      Warning: Inpainting failed for junction {i + 1}: {e}")
                    # Fallback to simple blending
                    masked_region = mask_b > 0.5
                    if masked_region.any():
                        image_slice_b[masked_region] *= 0.7  # Reduce intensity in junction

                # Put inpainted region back
                if axis == 0:
                    full_volume_pt[0, :, region_start:region_end, :, :] = image_slice_b[0]
                elif axis == 1:
                    full_volume_pt[0, :, :, region_start:region_end, :] = image_slice_b[0]
                elif axis == 2:
                    full_volume_pt[0, :, :, :, region_start:region_end] = image_slice_b[0]

                current_pos += step

    # Convert back to numpy
    result_np = pt_to_numpy(full_volume_pt[0])
    if result_np.shape[0] == 1:  # Remove channel dimension if single channel
        result_np = result_np.squeeze(0)

    return result_np


def create_railway_track_3d(
    unet,
    scheduler,
    inpainting_pipeline,
    device,
    output_dir,
    grids_length,
    grids_width,
    grids_height,
    overlap_d=8,
    overlap_h=8,
    overlap_w=8,
    inference_steps=60,
    seed=123,
    stitching_mode="separate_inpainting",
    mask_type="gap_filling_compatible",
    batch_size=1,
    binary=False,
    **kwargs,
):
    """
    Create a 3D railway track by extending in multiple dimensions using pre-loaded models.

    Strategy:
    1. First create strips along length dimension (using 1D approach)
    2. Then stitch strips along width dimension
    3. Finally stitch layers along height dimension
    """
    print(f"Creating 3D railway track: {grids_length}x{grids_width}x{grids_height} grids")
    print(f"Overlaps: D={overlap_d}, H={overlap_h}, W={overlap_w}")

    # Step 1: Create strips along length dimension (D axis)
    print("Step 1: Creating length strips...")
    strips = []

    for j in range(grids_width):
        for k in range(grids_height):
            strip_seed = seed + j * 1000 + k * 100
            print(
                f"  Creating strip {len(strips) + 1}/{grids_width * grids_height} for position (width={j}, height={k}) with seed={strip_seed}"
            )

            # Create 1D track along length using pre-loaded models
            strip = create_railway_track_1d(
                unet=unet,
                scheduler=scheduler,
                inpainting_pipeline=inpainting_pipeline,
                device=device,
                output_dir=output_dir / f"strip_{j}_{k}",
                n_blocks_length=grids_length,
                overlap=overlap_d,
                inference_steps=inference_steps,
                seed=strip_seed,
                stitching_mode=stitching_mode,
                mask_type=mask_type,
                batch_size=batch_size,
                binary=binary,
                **kwargs,
            )

            if strip is not None:
                strips.append(
                    {
                        "volume": strip,
                        "position": (j, k),  # width, height position
                    }
                )
                print(f"    ✓ Strip completed. Shape: {strip.shape}")
            else:
                print(f"    ✗ Warning: Failed to create strip at position ({j}, {k})")

    if not strips:
        print("Error: No strips were created successfully.")
        return None

    print(f"Successfully created {len(strips)} strips")

    # Step 2: Stitch strips along width dimension (H axis)
    print("Step 2: Stitching strips along width dimension...")

    # Group strips by height
    height_layers = {}
    for strip_data in strips:
        j, k = strip_data["position"]  # width, height
        if k not in height_layers:
            height_layers[k] = []
        height_layers[k].append((j, strip_data["volume"]))

    stitched_layers = []
    for k in sorted(height_layers.keys()):
        print(f"  Stitching width strips for height layer {k}")

        # Sort strips by width position
        width_strips = sorted(height_layers[k], key=lambda x: x[0])

        if len(width_strips) == 1:
            # Only one strip in this layer
            stitched_layers.append(width_strips[0][1])
        else:
            # Stitch multiple strips along H axis
            layer = stitch_volumes_along_axis_with_inpainting(
                volumes=[strip[1] for strip in width_strips],
                axis=1,  # H axis
                overlap=overlap_h,
                device=device,
                inpainting_pipeline=inpainting_pipeline,
                inference_steps=inference_steps,
                seed=seed + k * 10000,
                mask_type=mask_type,
                **kwargs,
            )
            stitched_layers.append(layer)

    # Step 3: Stitch layers along height dimension (W axis)
    print("Step 3: Stitching layers along height dimension...")

    if len(stitched_layers) == 1:
        final_track = stitched_layers[0]
    else:
        final_track = stitch_volumes_along_axis_with_inpainting(
            volumes=stitched_layers,
            axis=2,  # W axis
            overlap=overlap_w,
            device=device,
            inpainting_pipeline=inpainting_pipeline,
            inference_steps=inference_steps,
            seed=seed + 100000,
            mask_type=mask_type,
            **kwargs,
        )

    # Apply binary thresholding if requested
    if binary:
        final_track = (final_track > 0.5).astype(np.float32)

    print(f"Final 3D railway track shape: {final_track.shape}")
    return final_track


def create_railway_track(
    model_path,
    inpainting_model_path,
    output_dir,
    target_volume,  # (depth, width, length) in real units
    base_volume=(0.1, 0.3, 0.3),  # Volume represented by a single (32, 64, 64) voxel grid
    overlap_d=8,
    overlap_w=8,
    overlap_l=8,
    scheduler_type="ddim",
    **kwargs,
):
    """
    Main function to create railway track of specified dimensions.
    Loads models once and reuses them for efficiency.
    """
    # Load models once
    print("Loading models...")
    unet, scheduler, inpainting_pipeline, device = load_models(model_path, inpainting_model_path, scheduler_type)

    if unet is None:
        print("Failed to load models")
        return None

    # Calculate how many grids needed in each dimension
    target_depth, target_width, target_length = target_volume
    base_depth, base_width, base_length = base_volume

    grids_depth = int(np.ceil(target_depth / base_depth))
    grids_width = int(np.ceil(target_width / base_width))
    grids_length = int(np.ceil(target_length / base_length))

    print(f"Target volume: {target_volume}")
    print(f"Base volume per grid: {base_volume}")
    print(f"Grids needed: {grids_depth} x {grids_width} x {grids_length}")

    if grids_depth == 1 and grids_width == 1 and grids_length == 1:
        # Single block case
        print("Single block case - using simple generation")
        args = argparse.Namespace(**kwargs)
        volumes = generate_single_volume(
            unet, scheduler, kwargs.get("inference_steps", 60), kwargs.get("seed", 123), 1, device, args
        )
        return volumes[0] if volumes else None

    elif grids_width == 1 and grids_length == 1:
        # 1D case (depth only)
        print("1D case - extending along depth")
        return create_railway_track_1d(
            unet=unet,
            scheduler=scheduler,
            inpainting_pipeline=inpainting_pipeline,
            device=device,
            output_dir=output_dir,
            n_blocks_length=grids_depth,
            overlap=overlap_d,
            **kwargs,
        )
    else:
        # 3D case
        print("3D case - extending in multiple dimensions")
        return create_railway_track_3d(
            unet=unet,
            scheduler=scheduler,
            inpainting_pipeline=inpainting_pipeline,
            device=device,
            output_dir=output_dir,
            grids_length=grids_depth,
            grids_width=grids_width,
            grids_height=grids_length,
            overlap_d=overlap_d,
            overlap_h=overlap_w,
            overlap_w=overlap_l,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Railway Track Generator - Create voxel tracks of specified dimensions using proper gap-filling"
    )

    # Required model paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model directory")
    parser.add_argument(
        "--inpainting_model_path", type=str, default=None, help="Path to the dedicated inpainting model directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="railway_tracks", help="Output directory for saving railway track files"
    )

    # Target dimensions
    parser.add_argument("--target_depth", type=float, required=True, help="Target depth of railway track (real units)")
    parser.add_argument("--target_width", type=float, required=True, help="Target width of railway track (real units)")
    parser.add_argument(
        "--target_length", type=float, required=True, help="Target length of railway track (real units)"
    )

    # Base unit dimensions (defaults match your description)
    parser.add_argument(
        "--base_depth", type=float, default=0.1, help="Depth represented by single voxel grid (default: 0.1)"
    )
    parser.add_argument(
        "--base_width", type=float, default=0.3, help="Width represented by single voxel grid (default: 0.3)"
    )
    parser.add_argument(
        "--base_length", type=float, default=0.3, help="Length represented by single voxel grid (default: 0.3)"
    )

    # Overlap/gap parameters
    parser.add_argument("--overlap_d", type=int, default=8, help="Gap size along depth dimension (default: 8)")
    parser.add_argument("--overlap_w", type=int, default=8, help="Gap size along width dimension (default: 8)")
    parser.add_argument("--overlap_l", type=int, default=8, help="Gap size along length dimension (default: 8)")

    # Generation parameters
    parser.add_argument("--scheduler_type", choices=["ddpm", "ddim"], default="ddim", help="Choose sampling scheduler")
    parser.add_argument("--inference_steps", type=int, default=60, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for generation")
    parser.add_argument(
        "--stitching_mode",
        type=str,
        default="separate_inpainting",
        choices=["sequential_latent", "cpu_simple", "separate_inpainting"],
        help="Method to use for joining blocks",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="gap_filling_compatible",
        choices=[
            "random_block",
            "multi_block",
            "random_noise",
            "slice_mask",
            "mixed",
            "edge_mask",
            "middle_mask",
            "central_large_block",
            "mixed_edge_central",
            "gap_filling_compatible",
        ],
        help="Type of mask to use for inpainting mode",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--binary", action="store_true", help="Threshold output to binary mask (>0.5)")

    # Inpainting parameters
    parser.add_argument(
        "--inpaint_region_size_ratio",
        type=float,
        default=0.3,
        help="Size of inpainting region as ratio of process region",
    )
    parser.add_argument("--inpaint_iteratively", action="store_true", help="Whether to inpaint in smaller iterations")
    parser.add_argument(
        "--inpaint_iterations", type=int, default=3, help="Number of iterations for iterative inpainting"
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Value to threshold the final volume at (None = no thresholding)",
    )

    # Output format
    parser.add_argument(
        "--save_format", choices=["vti", "npy", "both"], default="both", help="Format to save the railway track"
    )
    parser.add_argument(
        "--output_name", type=str, default=None, help="Custom name for output file (default: auto-generated)"
    )

    args = parser.parse_args()

    # Validate inpainting model requirement
    if args.stitching_mode == "separate_inpainting" and args.inpainting_model_path is None:
        raise ValueError("--inpainting_model_path is required when using --stitching_mode separate_inpainting")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Define target and base volumes
    target_volume = (args.target_depth, args.target_width, args.target_length)
    base_volume = (args.base_depth, args.base_width, args.base_length)

    # Generate railway track
    start_time = time.time()

    railway_track = create_railway_track(
        model_path=args.model_path,
        inpainting_model_path=args.inpainting_model_path,
        output_dir=output_dir,
        target_volume=target_volume,
        base_volume=base_volume,
        overlap_d=args.overlap_d,
        overlap_w=args.overlap_w,
        overlap_l=args.overlap_l,
        scheduler_type=args.scheduler_type,
        inference_steps=args.inference_steps,
        seed=args.seed,
        stitching_mode=args.stitching_mode,
        mask_type=args.mask_type,
        batch_size=args.batch_size,
        binary=args.binary,
        inpaint_region_size_ratio=args.inpaint_region_size_ratio,
        inpaint_iteratively=args.inpaint_iteratively,
        inpaint_iterations=args.inpaint_iterations,
        threshold_value=args.threshold_value,
    )

    end_time = time.time()

    if railway_track is None:
        print("Failed to generate railway track")
        return

    print(f"Railway track generation completed in {end_time - start_time:.2f} seconds")

    # Generate output filename
    if args.output_name:
        base_name = args.output_name
    else:
        base_name = f"railway_track_{args.target_depth}x{args.target_width}x{args.target_length}_{args.seed}"

    # Save in requested formats
    if args.save_format in ["npy", "both"]:
        npy_path = output_dir / f"{base_name}.npy"
        np.save(npy_path, railway_track)
        print(f"Saved numpy array to: {npy_path}")

    if args.save_format in ["vti", "both"]:
        try:
            vti_path = output_dir / f"{base_name}.vti"

            # Create VTK image data
            vtk_data = pv.ImageData(dimensions=railway_track.shape)
            vtk_data["voxel_data"] = railway_track.flatten(order="F")
            vtk_data.save(vti_path)
            print(f"Saved VTI file to: {vti_path}")
        except Exception as e:
            print(f"Warning: Could not save VTI file: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("RAILWAY TRACK GENERATION SUMMARY")
    print("=" * 50)
    print(f"Target volume: {target_volume}")
    print(f"Generated shape: {railway_track.shape}")
    print(f"Voxel count: {np.prod(railway_track.shape):,}")
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
