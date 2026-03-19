#!/usr/bin/env python3
"""
Single Test Script for Gap Filling using Base Model + Inpainting Model

This script:
1. Generates a voxel grid using the base model (trained without inpainting_mode)
2. Removes a depth strip in the middle (like in training)
3. Uses the inpainting model to fill the gap
4. Uses the same inference method as generate_sample_images() from trainer
5. Saves original, masked, and inpainted volumes as VTI files
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, DDPMPipeline
import os
import sys
from utils.device_utils import get_device, get_generator, manual_seed_all, get_torch_dtype, apply_low_memory_defaults

# Add PyVista import for VTI saving
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. VTI saving will be skipped.")

# Import the MaskGenerator3D class from trainer
from modules.trainer import MaskGenerator3D


class MockArgs:
    """Mock args class to work with MaskGenerator3D"""

    def __init__(self):
        self.mask_type = "middle_mask"
        self.middle_axis = "depth"
        self.edge_width = 0.2
        self.middle_mask_width_min = 0.08
        self.middle_mask_width_max = 0.15
        self.middle_mask_position_jitter = 0.05
        self.seed = 42


def save_volume_as_vti(volume_data_np, vti_save_path, descriptive_name="Volume"):
    """Saves a 3D numpy array as a VTI file using PyVista."""
    if not PYVISTA_AVAILABLE:
        print(f"PyVista not available. Skipping VTI save for {descriptive_name} to {vti_save_path}.")
        return

    try:
        D, H, W = volume_data_np.shape
        grid = pv.ImageData()
        grid.dimensions = np.array([W, H, D]) + 1  # PyVista dimensions (nx, ny, nz)
        grid.origin = (0, 0, 0)  # Default origin
        grid.spacing = (1, 1, 1)  # Default spacing

        # Process volume data (apply binary threshold >0.5)
        processed_volume_data = (volume_data_np > 0.5).astype(np.float32)

        # Ensure data is C-contiguous
        grid.cell_data["values"] = np.ascontiguousarray(processed_volume_data).flatten(order="C")

        grid.save(str(vti_save_path), binary=True)
        print(f"{descriptive_name} saved as VTI to: {vti_save_path}")

    except Exception as e:
        print(f"Error saving {descriptive_name} VTI file {vti_save_path}: {e}")
        import traceback

        traceback.print_exc()


def generate_with_base_model(base_pipeline, device, inference_steps=1000, seed=42):
    """Generate a voxel grid using the base model (non-inpainting)"""
    print("Generating voxel grid with base model...")

    base_unet = base_pipeline.unet
    base_scheduler = base_pipeline.scheduler

    # Determine shape from UNet config
    D, H, W = base_unet.config.sample_size  # Should be (Depth, Height, Width)
    C = base_unet.config.in_channels  # Should be 1 for base model

    # Initialize noise
    generator = get_generator(device, seed)
    latents = torch.randn((1, C, D, H, W), generator=generator, device=device, dtype=base_unet.dtype)

    # Scale initial noise if needed
    if isinstance(base_scheduler, DDPMScheduler):
        latents = latents * base_scheduler.init_noise_sigma

    # Set timesteps
    base_scheduler.set_timesteps(inference_steps, device=device)
    timesteps = base_scheduler.timesteps

    # Sampling loop
    with torch.no_grad():
        for t in tqdm(timesteps, desc="Base Generation"):
            # Model input is latents for unconditional generation
            model_output = base_unet(latents, t).sample
            latents = base_scheduler.step(model_output, t, latents, generator=generator).prev_sample

    print(f"Generated voxel grid with shape: {latents.shape}")
    return latents


def create_depth_strip_mask(shape, args):
    """Create a depth strip mask using MaskGenerator3D"""
    print("Creating depth strip mask...")

    mask_generator = MaskGenerator3D(args=args)
    mask = mask_generator(shape)  # Returns (B, 1, D, H, W)

    print(f"Created mask with shape: {mask.shape}, masked voxels: {mask.sum().item()}")
    return mask


def inpaint_with_inpainting_model(inpainting_pipeline, original_latents, mask, device, inference_steps=1000, seed=42):
    """Inpaint the masked region using the inpainting model"""
    print("Inpainting masked region...")

    inpainting_unet = inpainting_pipeline.unet
    inpainting_scheduler = inpainting_pipeline.scheduler

    # Create masked images (set masked areas to -1)
    masked_images = original_latents * (1.0 - mask) - mask

    # Initialize noise for inpainting
    generator = get_generator(device, seed + 1)  # Different seed for inpainting
    latents = torch.randn(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)

    # Scale initial noise if needed
    if isinstance(inpainting_scheduler, DDPMScheduler):
        latents = latents * inpainting_scheduler.init_noise_sigma

    # Set timesteps
    inpainting_scheduler.set_timesteps(inference_steps, device=device)
    timesteps = inpainting_scheduler.timesteps

    # Main sampling loop
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Inpainting (DDPM)")):
            t_input = t.repeat(1)  # Batch size is 1

            # For inpainting UNet: concatenate [noisy_latents, mask, masked_images]
            model_input = torch.cat([latents, mask, masked_images], dim=1)

            # Predict noise
            noise_pred = inpainting_unet(model_input, t_input).sample

            # Scheduler step
            step_output = inpainting_scheduler.step(noise_pred, t, latents, generator=generator)
            x_prev_denoised_candidate = step_output.prev_sample

            # RePaint-like guidance: ensure known regions stay consistent
            if i < len(timesteps) - 1:
                # Add noise to the original image for the next timestep
                prev_t = timesteps[i + 1]
                noise_for_gt_conditioning = torch.randn(
                    original_latents.shape, device=device, dtype=original_latents.dtype
                )

                # Always use the original scheduler for adding noise (matches training)
                x_prev_noised_known_gt = inpainting_scheduler.add_noise(
                    original_latents, noise_for_gt_conditioning, prev_t.unsqueeze(0)
                )

                # Composite: use inpainted prediction for masked areas, noisy GT for known areas
                latents = x_prev_denoised_candidate * mask + x_prev_noised_known_gt * (1.0 - mask)
            else:
                # Last step: composite final x0 prediction
                latents = x_prev_denoised_candidate * mask + original_latents * (1.0 - mask)

    print(f"Inpainting completed. Final shape: {latents.shape}")
    return latents


def main():
    parser = argparse.ArgumentParser(description="Single test script for gap filling")

    # Model paths
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="Path to base model (trained without inpainting_mode)"
    )
    parser.add_argument(
        "--inpainting_model_path",
        type=str,
        required=True,
        help="Path to inpainting model (trained with inpainting_mode)",
    )

    # Generation parameters
    parser.add_argument("--inference_steps", type=int, default=1000, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument("--output_dir", type=str, default="out/single_test/", help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use for computation.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16"],
                        help="Model precision. 'float16' halves memory usage.")
    parser.add_argument("--low-memory", action="store_true", default=False,
                        help="Enable low-memory optimisations (attention slicing).")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manual_seed_all(device, args.seed)

    print("=== Single Test: Gap Filling ===")
    print(f"Base model: {args.base_model_path}")
    print(f"Inpainting model: {args.inpainting_model_path}")
    print(f"Inference steps: {args.inference_steps}")
    print(f"Output directory: {output_dir}")

    try:
        torch_dtype = get_torch_dtype(args.dtype, device)
        dtype_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

        # Load base model
        print("Loading base model...")
        base_pipeline = DDPMPipeline.from_pretrained(args.base_model_path, **dtype_kwargs)
        base_pipeline = base_pipeline.to(device)
        if args.low_memory:
            apply_low_memory_defaults(base_pipeline, device)

        # Load inpainting model
        print("Loading inpainting model...")
        inpainting_pipeline = DDPMPipeline.from_pretrained(args.inpainting_model_path, **dtype_kwargs)
        inpainting_pipeline = inpainting_pipeline.to(device)
        if args.low_memory:
            apply_low_memory_defaults(inpainting_pipeline, device)

        # Verify inpainting model is configured correctly
        inpainting_unet = inpainting_pipeline.unet
        if not (
            hasattr(inpainting_unet, "config")
            and hasattr(inpainting_unet.config, "inpainting_mode")
            and inpainting_unet.config.inpainting_mode
        ):
            print("Warning: Inpainting model may not be in inpainting mode!")
        else:
            print("Inpainting model verified as inpainting-capable.")

        # 1. Generate initial voxel grid with base model
        original_latents = generate_with_base_model(base_pipeline, device, args.inference_steps, args.seed)

        # 2. Create depth strip mask
        mock_args = MockArgs()
        mock_args.seed = args.seed

        mask = create_depth_strip_mask(original_latents.shape, mock_args)
        mask = mask.to(device)

        # 3. Inpaint the masked region
        inpainted_latents = inpaint_with_inpainting_model(
            inpainting_pipeline, original_latents, mask, device, args.inference_steps, args.seed
        )

        # 4. Convert to numpy and save results
        print("Converting to numpy and saving results...")

        # Convert latents to [0,1] range for saving
        original_np = ((original_latents / 2 + 0.5).clamp(0, 1).cpu().numpy()).squeeze()
        inpainted_np = ((inpainted_latents / 2 + 0.5).clamp(0, 1).cpu().numpy()).squeeze()
        mask_np = mask.cpu().numpy().squeeze()

        # Create masked version for visualization
        masked_np = original_np * (1.0 - mask_np)

        # Save as numpy files
        np.save(output_dir / "original.npy", original_np)
        np.save(output_dir / "inpainted.npy", inpainted_np)
        np.save(output_dir / "mask.npy", mask_np)
        np.save(output_dir / "masked.npy", masked_np)

        print(f"Numpy files saved to {output_dir}")
        print(f"Original shape: {original_np.shape}")
        print(f"Mask shape: {mask_np.shape}")
        print(f"Masked voxels: {mask_np.sum():.0f}")

        # Save as VTI files if PyVista is available
        if PYVISTA_AVAILABLE:
            save_volume_as_vti(original_np, output_dir / "original.vti", "Original")
            save_volume_as_vti(inpainted_np, output_dir / "inpainted.vti", "Inpainted")
            save_volume_as_vti(masked_np, output_dir / "masked.vti", "Masked")
            save_volume_as_vti(mask_np, output_dir / "mask.vti", "Mask")

        print("✅ Single test completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during single test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
