import torch
import numpy as np
import argparse
import os
from diffusers import DDPMPipeline, DDIMScheduler, DiffusionPipeline
from pathlib import Path
import time
from tqdm.auto import tqdm
from utils.eval_utils import (
    generate_overlapping_volume,
    generate_single_volume,
    generate_cpu_overlapping_volume,
    generate_stitched_volume_with_inpainting,
)
import pyvista as pv

PYVISTA_AVAILABLE = True


def main():
    parser = argparse.ArgumentParser(description="CLI 3D Overlapping Volume Generator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model directory")
    parser.add_argument("--output_dir", type=str, default="cli_output", help="Output directory for saving VTI files")
    parser.add_argument("--scheduler_type", choices=["ddpm", "ddim"], default="ddpm", help="Choose sampling scheduler")
    parser.add_argument("--inference_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (default: random)")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=3,
        help="Number of overlapping blocks to generate (ignored if --generate_single)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        help="Number of slices to overlap between blocks (ignored if --generate_single)",
    )
    parser.add_argument("--generate_single", action="store_true", help="Generate only a single block (no overlap)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for --generate_single mode")
    parser.add_argument(
        "--overlap_batch_size", type=int, default=2, help="Number of blocks to generate per overlapping batch"
    )
    parser.add_argument(
        "--cpu_overlapping",
        action="store_true",
        help="Use CPU-based overlapping fallback (generate each block independently)",
    )
    parser.add_argument("--binary", action="store_true", help="Threshold output to binary mask (>0.5)")
    parser.add_argument(
        "--min_bw_ratio",
        type=float,
        default=0.0,
        help="Minimum required ratio of dark voxels (<=0.5) in the output volume. Generation retries if below this. Set to 0 to disable.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=5, help="Maximum number of retries if min_bw_ratio is not met."
    )

    parser.add_argument(
        "--stitching_mode",
        type=str,
        default="sequential_latent",
        choices=["sequential_latent", "cpu_simple", "separate_inpainting"],
        help="Method to use for joining blocks: 'sequential_latent' (original inpainting within one model), 'cpu_simple' (independent generation + basic stitch), 'separate_inpainting' (independent generation + dedicated inpainting model)",
    )
    parser.add_argument(
        "--inpainting_model_path",
        type=str,
        default=None,
        help="Path to the dedicated inpainting model directory (required for 'separate_inpainting' mode)",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="central_large_block",
        choices=[
            "random_block",
            "multi_block",
            "random_noise",
            "slice_mask",
            "mixed",
            "edge_mask",
            "middle_mask",
            "central_large_block",
        ],
        help="Type of mask to use for inpainting mode (only used with inpainting models)",
    )
    parser.add_argument(
        "--edge_type",
        type=str,
        default="right",
        choices=["right", "left", "top", "bottom", "front", "back"],
        help="Edge to mask when using edge_mask type",
    )
    parser.add_argument(
        "--edge_width", type=float, default=0.2, help="Width of edge mask as a proportion of dimension (0.0-1.0)"
    )
    parser.add_argument(
        "--middle_axis",
        type=str,
        default="depth",
        choices=["depth", "height", "width"],
        help="Axis to mask through the middle when using middle_mask type",
    )
    parser.add_argument(
        "--central_block_min_ratio",
        type=float,
        default=0.3,
        help="Minimum ratio of the central block size to the total dimension (for central_large_block).",
    )
    parser.add_argument(
        "--central_block_max_ratio",
        type=float,
        default=0.7,
        help="Maximum ratio of the central block size to the total dimension (for central_large_block).",
    )
    parser.add_argument(
        "--central_block_jitter_factor",
        type=float,
        default=0.1,
        help="Jitter factor for the central block position (for central_large_block).",
    )
    parser.add_argument(
        "--inpaint_region_size_ratio",
        type=float,
        default=0.3,
        help="Size of inpainting region as ratio of process region (used with 'separate_inpainting' mode)",
    )
    parser.add_argument(
        "--inpaint_iteratively",
        action="store_true",
        help="Whether to inpaint in smaller iterations (used with 'separate_inpainting' mode)",
    )
    parser.add_argument(
        "--inpaint_iterations",
        type=int,
        default=3,
        help="Number of iterations for iterative inpainting (used with 'separate_inpainting' mode)",
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Value to threshold the final volume at (None = no thresholding) (used with 'separate_inpainting' mode)",
    )

    args = parser.parse_args()

    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(8), "big")
        print(f"Using random seed: {args.seed}")

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    if args.stitching_mode == "separate_inpainting" and args.inpainting_model_path is None:
        raise ValueError(
            "Argument '--inpainting_model_path' is required when using '--stitching_mode separate_inpainting'"
        )
    if args.stitching_mode != "separate_inpainting" and args.inpainting_model_path is not None:
        print(
            "Warning: '--inpainting_model_path' is provided but '--stitching_mode' is not 'separate_inpainting'. The inpainting model will not be used."
        )

    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading BASE diffusion pipeline from: {model_path}")
    try:
        pipeline = DDPMPipeline.from_pretrained(model_path).to(device)
        if args.scheduler_type == "ddim":
            pipeline.scheduler = DDIMScheduler.from_pretrained(model_path / "scheduler")
        unet = pipeline.unet
        scheduler = pipeline.scheduler
        print(f"Base pipeline loaded with {args.scheduler_type.upper()} scheduler.")

        if hasattr(unet, "_original_in_channels") and "_original_in_channels" not in unet.config:
            config_dict = dict(unet.config)
            config_dict["_original_in_channels"] = unet._original_in_channels
            try:
                unet.config["_original_in_channels"] = unet._original_in_channels
                print("Applied workaround: Added '_original_in_channels' to unet.config")
            except TypeError:
                print("Warning: Could not directly add attribute to unet.config (FrozenDict is immutable).")

    except Exception as e:
        print(f"Error loading BASE pipeline or applying workaround: {e}")
        exit()

    inpainting_pipeline = None
    inpainting_unet = None
    inpainting_scheduler = None

    if args.stitching_mode == "separate_inpainting":
        inpainting_model_path = Path(args.inpainting_model_path)
        print(f"Loading INPAINTING diffusion pipeline from: {inpainting_model_path}")
        try:
            inpainting_pipeline = DiffusionPipeline.from_pretrained(inpainting_model_path).to(device)

            if hasattr(inpainting_pipeline, "unet"):
                inpainting_unet = inpainting_pipeline.unet
            else:
                raise ValueError("Loaded inpainting pipeline does not have a 'unet' attribute.")

            if hasattr(inpainting_pipeline, "scheduler"):
                inpainting_scheduler = inpainting_pipeline.scheduler
            else:
                raise ValueError("Loaded inpainting pipeline does not have a 'scheduler' attribute.")

            if not (inpainting_model_path / "scheduler" / "scheduler_config.json").exists():
                print(
                    "Warning: Could not find scheduler_config.json for inpainting pipeline. Attempting to use/clone base config."
                )
                if args.scheduler_type == "ddim":
                    inpainting_scheduler = DDIMScheduler.from_config(scheduler.config)
                else:
                    inpainting_scheduler = DDPMPipeline.from_config(scheduler.config)
                inpainting_pipeline.scheduler = inpainting_scheduler
                print(
                    f"Assigned {type(inpainting_scheduler).__name__} to inpainting pipeline based on base scheduler config."
                )
            elif inpainting_pipeline.scheduler is None:
                print("Error: Inpainting pipeline scheduler is None even with config. Trying to load explicitly.")
                inpainting_pipeline.scheduler = DDIMScheduler.from_pretrained(inpainting_model_path / "scheduler")

            if (
                hasattr(inpainting_unet, "_original_in_channels")
                and "_original_in_channels" not in inpainting_unet.config
            ):
                config_dict = dict(inpainting_unet.config)
                config_dict["_original_in_channels"] = inpainting_unet._original_in_channels
                try:
                    inpainting_unet.config["_original_in_channels"] = inpainting_unet._original_in_channels
                    print("Applied workaround: Added '_original_in_channels' to inpainting unet.config")
                except TypeError:
                    print(
                        "Warning: Could not directly add attribute to inpainting unet.config (FrozenDict is immutable)."
                    )

            print("Inpainting pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading INPAINTING pipeline: {e}")
            exit()
    start_time = time.time()
    try:
        if args.generate_single:
            print(f"Generating single 3D block with steps={args.inference_steps}, seed={args.seed}")
            volumes_batch = []
            with tqdm(total=args.batch_size, desc="Generating Single Volumes") as pbar:

                def update_progress():
                    pbar.update(1)

                volumes_batch = generate_single_volume(
                    unet,
                    scheduler,
                    args.inference_steps,
                    args.seed,
                    args.batch_size,
                    device,
                    args=args,
                    min_bw_ratio=args.min_bw_ratio,
                    max_retries=args.max_retries,
                    progress_callback=update_progress,
                )
            final_volumes_np_0_1 = volumes_batch

        elif args.stitching_mode == "cpu_simple":
            print(f"CPU overlapping fallback: generating {args.n_blocks} blocks independently and stitching on CPU...")
            current_seed = args.seed
            retries = 0
            volume_meets_criteria = False
            full_vol_np = None

            while not volume_meets_criteria and retries <= args.max_retries:
                if retries > 0:
                    print(f"  Retry {retries}/{args.max_retries} with seed {current_seed}...")

                try:
                    with tqdm(total=args.n_blocks, desc=f"CPU Overlap (Try {retries + 1})", leave=False) as pbar:

                        def update_progress():
                            pbar.update(1)

                        full_vol_np = generate_cpu_overlapping_volume(
                            unet,
                            scheduler,
                            args.inference_steps,
                            current_seed,
                            args.n_blocks,
                            args.overlap,
                            device,
                            args=args,
                            progress_callback=update_progress,
                            min_bw_ratio=args.min_bw_ratio,
                            max_retries=args.max_retries,
                        )

                    if full_vol_np is None:
                        raise RuntimeError("CPU overlapping function returned None.")

                    bw_ratio = np.mean(full_vol_np <= 0.5)
                    print(f"  Generated volume BW ratio: {bw_ratio:.4f}")
                    if bw_ratio >= args.min_bw_ratio or args.min_bw_ratio <= 0:
                        volume_meets_criteria = True
                        final_volumes_np_0_1 = [full_vol_np]
                        break
                    else:
                        raise ValueError(f"BW ratio {bw_ratio:.4f} below threshold {args.min_bw_ratio}")

                except Exception as e:
                    print(f"Error during CPU overlapping generation (Retry {retries}): {e}")
                    retries += 1
                    current_seed += 1
                    if "full_vol_np" in locals() and full_vol_np is not None:
                        del full_vol_np
                    torch.cuda.empty_cache()
                    if retries > args.max_retries:
                        print("Max retries reached for CPU simple mode.")
                        exit()
            if not volume_meets_criteria:
                print(
                    f"Error: Minimum BW ratio ({args.min_bw_ratio}) not met after {args.max_retries} retries in CPU simple mode."
                )
                exit()

        elif args.stitching_mode == "separate_inpainting":
            print(f"Separate Inpainting Mode: generating {args.n_blocks} blocks and inpainting seams...")
            final_stitched_volume_np = generate_stitched_volume_with_inpainting(
                unet,
                scheduler,
                inpainting_pipeline,
                args.inference_steps,
                args.seed,
                args.n_blocks,
                args.overlap,
                device,
                args=args,
                output_dir=output_dir / "stitched_debug",
                generation_batch_size=args.batch_size,
                strength=1.0,
                inpaint_region_size_ratio=args.inpaint_region_size_ratio,
                inpaint_iteratively=args.inpaint_iteratively,
                inpaint_iterations=args.inpaint_iterations,
                threshold_value=args.threshold_value,
            )
            if final_stitched_volume_np is not None:
                final_volumes_np_0_1 = [final_stitched_volume_np]

        elif args.stitching_mode == "sequential_latent":
            print(
                f"Generating overlapping 3D volume using sequential latent context with steps={args.inference_steps}, seed={args.seed}, blocks={args.n_blocks}, overlap={args.overlap}"
            )

            if not (hasattr(unet.config, "inpainting_mode") and unet.config.inpainting_mode):
                print(
                    f"Error: Stitching mode 'sequential_latent' requires the base model ({args.model_path}) to be configured for inpainting."
                )
                exit()

            current_seed = args.seed
            retries = 0
            volume_meets_criteria = False
            all_cpu_parts = []
            full_vol_np = None
            full_vol_tensor = None

            while not volume_meets_criteria and retries <= args.max_retries:
                if retries > 0:
                    print(
                        f"  Retry {retries}/{args.max_retries} for entire overlapping generation with starting seed {current_seed}..."
                    )

                all_cpu_parts = []
                current_context_gpu = None
                batch_base_seed = current_seed
                num_overlap_batches = (args.n_blocks + args.overlap_batch_size - 1) // args.overlap_batch_size
                print(f"Processing in {num_overlap_batches} GPU batches of size {args.overlap_batch_size}...")

                generation_successful = True
                try:
                    for batch_idx in range(num_overlap_batches):
                        blocks_in_batch = min(
                            args.overlap_batch_size, args.n_blocks - batch_idx * args.overlap_batch_size
                        )
                        print(f"  GPU Batch {batch_idx + 1}/{num_overlap_batches} ({blocks_in_batch} blocks)...")
                        batch_seed = batch_base_seed + batch_idx
                        batch_cpu_parts, current_context_gpu = generate_overlapping_volume(
                            unet,
                            scheduler,
                            args.inference_steps,
                            batch_seed,
                            blocks_in_batch,
                            args.overlap,
                            device,
                            args=args,
                            initial_context_gpu=current_context_gpu,
                        )
                        all_cpu_parts.extend(batch_cpu_parts)
                        print(f"  GPU Batch {batch_idx + 1} complete. Clearing cache.")
                        if current_context_gpu is not None:
                            current_context_gpu_mem = (
                                current_context_gpu.element_size() * current_context_gpu.nelement()
                            )
                            if current_context_gpu_mem > 10 * 1024**2:
                                print(f"  Clearing context GPU tensor ({(current_context_gpu_mem / 1024**2):.2f} MB)")
                                del current_context_gpu
                                current_context_gpu = None
                        torch.cuda.empty_cache()

                    print("Stitching generated parts for sequential_latent mode...")
                    if not all_cpu_parts:
                        raise RuntimeError("No parts generated successfully for sequential_latent mode.")

                    C_part, _, H_part, W_part = all_cpu_parts[0].shape
                    total_depth_calculated = sum(part.shape[1] for part in all_cpu_parts)
                    print(f"Total calculated depth from parts: {total_depth_calculated}")
                    full_vol_tensor = torch.cat(all_cpu_parts, dim=1)
                    full_vol_np = (full_vol_tensor / 2 + 0.5).clamp(0, 1)
                    full_vol_np = full_vol_np.cpu().numpy()
                    if C_part == 1:
                        full_vol_np = full_vol_np.squeeze(0)
                    else:
                        print(f"Warning: Output volume has {C_part} channels. Saving first channel to VTI.")
                        full_vol_np = full_vol_np[0]

                    bw_ratio = np.mean(full_vol_np <= 0.5)
                    print(f"  Generated volume BW ratio: {bw_ratio:.4f}")
                    if bw_ratio >= args.min_bw_ratio or args.min_bw_ratio <= 0:
                        volume_meets_criteria = True
                        final_volumes_np_0_1 = [full_vol_np]
                    else:
                        raise ValueError(f"BW ratio {bw_ratio:.4f} below threshold {args.min_bw_ratio}")

                except Exception as e:
                    print(f"Error during sequential latent generation/stitching (Retry {retries}): {e}")
                    generation_successful = False
                if not volume_meets_criteria:
                    retries += 1
                    current_seed += 1
                    if "all_cpu_parts" in locals():
                        del all_cpu_parts
                    if "full_vol_np" in locals():
                        del full_vol_np
                    if "full_vol_tensor" in locals():
                        del full_vol_tensor
                    if current_context_gpu is not None:
                        del current_context_gpu
                    all_cpu_parts = []
                    full_vol_np = None
                    full_vol_tensor = None
                    current_context_gpu = None
                    torch.cuda.empty_cache()
                    if retries > args.max_retries:
                        print(f"Error: Max retries ({args.max_retries}) reached for sequential latent generation.")
                        exit()
                else:
                    break
        else:
            if not final_volumes_np_0_1:
                print(f"Error: Unhandled generation case. Mode: {args.stitching_mode}, Single: {args.generate_single}")
                exit()

    except Exception as e:
        print(f"An unexpected error occurred during volume generation: {e}")
        import traceback

        traceback.print_exc()
        exit()

    end_time = time.time()
    print(f"Volume generation completed in {end_time - start_time:.2f} seconds.")

    if not final_volumes_np_0_1:
        print("Error: No volumes were generated successfully after retries.")
        exit()

    for idx, vol_np_0_1 in enumerate(final_volumes_np_0_1):
        if args.generate_single:
            example_vol_shape = vol_np_0_1.shape
            if len(example_vol_shape) == 3:
                D, H, W = example_vol_shape
            else:
                D, H, W = (0, 0, 0)
                print("Warning: Could not determine volume shape.")

            print(
                f"Generated {len(final_volumes_np_0_1)} volumes, shape: ({D}, {H}, {W}) each, in {end_time - start_time:.2f} seconds"
            )
        else:
            D_total, H, W = vol_np_0_1.shape
            print(f"Generated volume shape: ({D_total}, {H}, {W}) in {end_time - start_time:.2f} seconds")
            print(f"Total generation time: {end_time - start_time:.2f} seconds")

    vti_save_path = None
    if PYVISTA_AVAILABLE:
        try:
            if args.generate_single:
                print(f"Saving {len(final_volumes_np_0_1)} volumes as VTI...")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                for i, volume_data in enumerate(final_volumes_np_0_1):
                    grid = pv.ImageData()
                    D_vol, H_vol, W_vol = volume_data.shape
                    grid.dimensions = np.array([W_vol, H_vol, D_vol]) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    if args.binary:
                        volume_data = (volume_data > 0.5).astype(np.float32)
                    grid.cell_data["values"] = np.ascontiguousarray(volume_data).flatten(order="C")

                    vti_filename = f"single_volume_{timestamp}_s{args.seed}_t{args.inference_steps}_idx{i}.vti"
                    vti_save_path = output_dir / vti_filename
                    grid.save(str(vti_save_path), binary=True)
                    print(f"  Saved volume {i + 1}/{len(final_volumes_np_0_1)} to: {vti_save_path}")
            elif args.cpu_overlapping:
                print("Saving CPU-overlapped volume as VTI...")
                volume_np_0_1 = final_volumes_np_0_1[0]
                D_total, H, W = volume_np_0_1.shape
                grid = pv.ImageData()
                grid.dimensions = np.array([W, H, D_total]) + 1
                grid.origin = (0, 0, 0)
                grid.spacing = (1, 1, 1)
                if args.binary:
                    volume_np_0_1 = (volume_np_0_1 > 0.5).astype(np.float32)
                grid.cell_data["values"] = np.ascontiguousarray(volume_np_0_1).flatten(order="C")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                vti_filename = f"cpu_overlap_volume_{timestamp}_s{args.seed}_t{args.inference_steps}_b{args.n_blocks}_o{args.overlap}.vti"
                vti_save_path = output_dir / vti_filename
                grid.save(str(vti_save_path), binary=True)
                print(f"CPU-overlapped volume saved to: {vti_save_path}")
            else:
                print("Saving overlapping volume as VTI...")
                volume_np_0_1 = final_volumes_np_0_1[0]
                D_total, H, W = volume_np_0_1.shape

                grid = pv.ImageData()
                grid.dimensions = np.array([W, H, D_total]) + 1
                grid.origin = (0, 0, 0)
                grid.spacing = (1, 1, 1)
                if args.binary:
                    volume_np_0_1 = (volume_np_0_1 > 0.5).astype(np.float32)
                grid.cell_data["values"] = np.ascontiguousarray(volume_np_0_1).flatten(order="C")

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                vti_filename = f"overlap_volume_{timestamp}_s{args.seed}_t{args.inference_steps}_b{args.n_blocks}_o{args.overlap}.vti"
                vti_save_path = output_dir / vti_filename
                grid.save(str(vti_save_path), binary=True)
                print(f"Overlapping volume saved successfully to: {vti_save_path}")

        except Exception as e:
            print(f"Error saving VTI file(s): {e}")
    else:
        print("VTI saving skipped (pyvista not available).")


if __name__ == "__main__":
    main()
