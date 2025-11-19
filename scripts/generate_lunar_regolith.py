import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add project root to path
sys.path.append(os.getcwd())

from railway_track import load_models, create_railway_track_3d
from utils.eval_utils import pt_to_numpy


def main():
    parser = argparse.ArgumentParser(description="Lunar Regolith Testbed Generator")

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="path/to/regolith_base_model",
        help="Path to the base diffusion model for regolith",
    )
    parser.add_argument(
        "--inpainting_model_path",
        type=str,
        default="path/to/regolith_inpainting_model",
        help="Path to the inpainting diffusion model for regolith",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="out/lunar_regolith", help="Directory to save the generated volumes"
    )
    parser.add_argument("--filename", type=str, default="regolith_bed", help="Base filename for the output")

    # Dimensions (in number of 64^3 grids)
    parser.add_argument("--grids_length", type=int, default=4, help="Number of grids along the length (D axis)")
    parser.add_argument("--grids_width", type=int, default=2, help="Number of grids along the width (H axis)")
    parser.add_argument("--grids_height", type=int, default=1, help="Number of grids along the height (W axis)")

    # Overlap parameters
    parser.add_argument("--overlap_d", type=int, default=8, help="Overlap in voxels along D axis")
    parser.add_argument("--overlap_h", type=int, default=8, help="Overlap in voxels along H axis")
    parser.add_argument("--overlap_w", type=int, default=8, help="Overlap in voxels along W axis")

    # Generation parameters
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--binary", action="store_true", help="Binarize the output (0/1)")

    args = parser.parse_args()

    print("=" * 60)
    print("       LUNAR REGOLITH TESTBED GENERATOR")
    print("=" * 60)
    print(f"Generating regolith bed of size: {args.grids_length}x{args.grids_width}x{args.grids_height} grids")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Inpainting model path: {args.inpainting_model_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    # Note: This will fail if paths are invalid, so we catch it to allow dry runs with placeholders
    try:
        unet, scheduler, inpainting_pipeline, device = load_models(
            args.model_path, args.inpainting_model_path, scheduler_type="ddim"
        )
    except Exception as e:
        print(f"\n[WARNING] Could not load models: {e}")
        print("If you are running a test without actual models, this is expected.")
        print("Please provide valid model paths to generate actual data.")
        return

    if unet is None:
        print("Failed to load models. Exiting.")
        return

    start_time = time.time()

    # Generate the volume using the 3D track generation logic
    # This logic is generic enough to create any large 3D volume from blocks
    regolith_volume = create_railway_track_3d(
        unet=unet,
        scheduler=scheduler,
        inpainting_pipeline=inpainting_pipeline,
        device=device,
        output_dir=Path(args.output_dir),
        grids_length=args.grids_length,
        grids_width=args.grids_width,
        grids_height=args.grids_height,
        overlap_d=args.overlap_d,
        overlap_h=args.overlap_h,
        overlap_w=args.overlap_w,
        inference_steps=args.inference_steps,
        seed=args.seed,
        stitching_mode="separate_inpainting",  # Best mode for quality
        mask_type="gap_filling_compatible",  # Best mask type
        batch_size=args.batch_size,
        binary=args.binary,
        debug=False,
    )

    if regolith_volume is not None:
        # Save the final volume
        output_path = os.path.join(args.output_dir, f"{args.filename}.npy")
        np.save(output_path, regolith_volume)

        # Also save as VTK for visualization if pyvista is available
        try:
            import pyvista as pv

            vtk_path = os.path.join(args.output_dir, f"{args.filename}.vti")

            # Create a PyVista grid
            grid = pv.UniformGrid()
            grid.dimensions = np.array(regolith_volume.shape) + 1
            grid.spacing = (1, 1, 1)  # Assuming unit spacing
            grid.origin = (0, 0, 0)

            # Add the data
            # PyVista expects data in column-major order (F-contiguous) when flattened
            # or we can just transpose to match
            grid.cell_data["Density"] = regolith_volume.flatten(order="F")

            grid.save(vtk_path)
            print(f"Saved VTK visualization to: {vtk_path}")
        except ImportError:
            print("PyVista not installed, skipping VTK save.")
        except Exception as e:
            print(f"Error saving VTK: {e}")

        print(f"\nSUCCESS! Regolith bed generated in {time.time() - start_time:.2f} seconds.")
        print(f"Shape: {regolith_volume.shape}")
        print(f"Saved to: {output_path}")
    else:
        print("\nGeneration failed.")


if __name__ == "__main__":
    main()
