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

# Try to import pyvista with xvfb for headless rendering
# NOTE: On headless servers without X, PyVista causes a SEGFAULT (not catchable)
# We check for DISPLAY environment variable first to avoid crashes
PYVISTA_AVAILABLE = False
pv = None


def _check_pyvista_available():
    """Lazy check if PyVista can actually render (called on first use)."""
    global PYVISTA_AVAILABLE, pv
    if pv is not None:
        return PYVISTA_AVAILABLE

    # Check if we're on a headless system (no DISPLAY)
    display = os.environ.get("DISPLAY", "")
    if not display:
        print(
            "Note: No DISPLAY environment variable set - skipping PyVista (headless server)"
        )
        PYVISTA_AVAILABLE = False
        return False

    try:
        import pyvista as pv_module

        pv_module.OFF_SCREEN = True

        # Try to start virtual framebuffer
        try:
            pv_module.start_xvfb()
            print("Started Xvfb virtual framebuffer for PyVista rendering")
        except Exception:
            pass  # May still work without xvfb on some systems

        # Test if rendering actually works by creating a tiny test render
        try:
            test_plotter = pv_module.Plotter(off_screen=True, window_size=(100, 100))
            test_plotter.add_mesh(pv_module.Sphere(radius=0.1))
            test_plotter.close()
            pv = pv_module
            PYVISTA_AVAILABLE = True
            print("PyVista 3D rendering available")
        except Exception as render_err:
            print(
                f"Note: PyVista rendering test failed ({render_err}), 3D renders will be skipped"
            )
            PYVISTA_AVAILABLE = False

    except ImportError:
        print("Note: PyVista not installed, 3D renders will be skipped")
        PYVISTA_AVAILABLE = False

    return PYVISTA_AVAILABLE


# Try to import matplotlib for visualization
MPL_AVAILABLE = False
plt = None
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server
    import matplotlib.pyplot as plt_module
    from mpl_toolkits.mplot3d import Axes3D

    plt = plt_module
    MPL_AVAILABLE = True
except ImportError:
    pass


def save_vti_without_pyvista(data, output_path, spacing=(1.0, 1.0, 1.0)):
    """
    Save a 3D numpy array as VTI (VTK ImageData) file without PyVista.
    Uses raw VTK XML format that can be opened in ParaView.

    Args:
        data: 3D numpy array
        output_path: Path to save .vti file
        spacing: Voxel spacing (dx, dy, dz)
    """
    import base64
    import zlib

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Remove batch/channel dimensions
    while data.ndim > 3:
        data = data[0]

    # Ensure float32 for compatibility
    data = data.astype(np.float32)

    nx, ny, nz = data.shape
    dx, dy, dz = spacing

    # Check if data is too large for uint32 header (>4GB raw data)
    raw_size = nx * ny * nz * 4  # 4 bytes per float32
    if raw_size > 4294967295:  # uint32 max
        print(f"  WARNING: Data too large for VTI format ({raw_size / 1e9:.2f} GB)")
        print(f"  Saving as chunked VTI with UInt64 header...")
        # Use uint64 header for large files
        header_type = "UInt64"
        header_dtype = np.uint64
    else:
        header_type = "UInt32"
        header_dtype = np.uint32

    # Compress data with zlib
    raw_data = data.flatten(order="F").tobytes()
    compressed = zlib.compress(raw_data, level=6)
    encoded = base64.b64encode(compressed).decode("ascii")

    # Header for compressed data (4 or 8 bytes each: num_blocks, block_size, last_block_size, compressed_size)
    header = np.array(
        [1, len(raw_data), len(raw_data), len(compressed)], dtype=header_dtype
    )
    header_encoded = base64.b64encode(header.tobytes()).decode("ascii")

    vti_content = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="{header_type}" compressor="vtkZLibDataCompressor">
  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" Spacing="{dx} {dy} {dz}">
    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">
      <PointData>
      </PointData>
      <CellData Scalars="voxel_data">
        <DataArray type="Float32" Name="voxel_data" format="binary">
{header_encoded}{encoded}
        </DataArray>
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
'''

    with open(output_path, "w") as f:
        f.write(vti_content)

    print(f"  Saved VTI file: {output_path} (shape: {data.shape})")
    return True


def save_blocks_as_vti(blocks, output_dir, prefix="block"):
    """Save multiple voxel blocks as individual VTI files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, block in enumerate(blocks):
        vti_path = output_dir / f"{prefix}_{i:02d}.vti"
        try:
            save_vti_without_pyvista(block, vti_path)
            saved.append(vti_path)
        except Exception as e:
            print(f"  Warning: Could not save block {i}: {e}")

    print(f"  Saved {len(saved)} block VTI files to {output_dir}")
    return saved


def save_intermediate_visualization(data, output_path, title="", vmin=None, vmax=None):
    """Save a 3D volume visualization as PNG using orthogonal slices."""
    if not MPL_AVAILABLE:
        print(
            f"  Warning: matplotlib not available, skipping visualization: {output_path}"
        )
        return False

    try:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Remove batch/channel dimensions if present
        while data.ndim > 3:
            data = data[0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        mid_x, mid_y, mid_z = data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2

        im0 = axes[0].imshow(
            data[mid_x, :, :], cmap="YlOrBr", aspect="auto", vmin=vmin, vmax=vmax
        )
        axes[0].set_title(f"YZ slice (x={mid_x})")
        axes[0].set_xlabel("Z")
        axes[0].set_ylabel("Y")

        im1 = axes[1].imshow(
            data[:, mid_y, :], cmap="YlOrBr", aspect="auto", vmin=vmin, vmax=vmax
        )
        axes[1].set_title(f"XZ slice (y={mid_y})")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")

        im2 = axes[2].imshow(
            data[:, :, mid_z], cmap="YlOrBr", aspect="auto", vmin=vmin, vmax=vmax
        )
        axes[2].set_title(f"XY slice (z={mid_z})")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("X")

        fig.colorbar(im2, ax=axes, shrink=0.6, label="Voxel Value")

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved visualization: {output_path}")
        return True
    except Exception as e:
        print(f"  Warning: Could not save visualization {output_path}: {e}")
        return False


def save_blocks_visualization(blocks, output_path, max_blocks=6):
    """Save visualization of multiple voxel blocks side-by-side."""
    if not MPL_AVAILABLE:
        print(f"  Warning: matplotlib not available, skipping blocks visualization")
        return False

    try:
        n_blocks = min(len(blocks), max_blocks)
        fig, axes = plt.subplots(2, n_blocks, figsize=(4 * n_blocks, 8))

        colors = ["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

        for i in range(n_blocks):
            block = blocks[i]
            if isinstance(block, torch.Tensor):
                block = block.cpu().numpy()
            while block.ndim > 3:
                block = block[0]

            # Top row: XY slice (top-down view)
            mid_z = block.shape[2] // 2
            axes[0, i].imshow(block[:, :, mid_z], cmap="YlOrBr", aspect="auto")
            axes[0, i].set_title(f"Block {i + 1}\nXY slice", fontsize=10)
            axes[0, i].axis("off")

            # Bottom row: XZ slice (side view)
            mid_y = block.shape[1] // 2
            axes[1, i].imshow(block[:, mid_y, :], cmap="YlOrBr", aspect="auto")
            axes[1, i].set_title(f"XZ slice", fontsize=10)
            axes[1, i].axis("off")

        # Hide unused axes
        for i in range(n_blocks, len(blocks)):
            if i < axes.shape[1]:
                axes[0, i].axis("off")
                axes[1, i].axis("off")

        fig.suptitle(
            f"Sampled Voxel Blocks (showing {n_blocks}/{len(blocks)})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved blocks visualization: {output_path}")
        return True
    except Exception as e:
        print(f"  Warning: Could not save blocks visualization: {e}")
        return False


def save_3d_volume_render(volume, output_path, title=""):
    """Save a 3D volume render using PyVista."""
    global pv, PYVISTA_AVAILABLE

    # Lazy check - only test PyVista on first actual use
    if not _check_pyvista_available():
        print(f"  Skipping 3D render (PyVista not available): {output_path}")
        return False

    try:
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()
        while volume.ndim > 3:
            volume = volume[0]

        # Create VTK image data
        grid = pv.ImageData(dimensions=np.array(volume.shape) + 1)
        grid.cell_data["values"] = volume.flatten(order="F")

        # Threshold to show only occupied voxels
        threshed = grid.threshold(0.5, scalars="values")

        if threshed.n_cells == 0:
            print(f"  Warning: No voxels above threshold for {output_path}")
            return False

        plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
        plotter.set_background("white")

        plotter.add_mesh(
            threshed,
            cmap="YlOrBr",
            opacity=0.7,
            show_edges=False,
            show_scalar_bar=False,
        )
        plotter.add_mesh(grid.outline(), color="gray", line_width=2)

        if title:
            plotter.add_text(title, position="upper_left", font_size=12, color="black")

        plotter.camera_position = "iso"
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 20
        plotter.camera.zoom(0.9)

        plotter.screenshot(str(output_path))
        plotter.close()
        print(f"  Saved 3D render: {output_path}")
        return True
    except Exception as e:
        print(f"  Warning: Could not save 3D render {output_path}: {e}")
        # Mark PyVista as unavailable to avoid future crashes
        PYVISTA_AVAILABLE = False
        return False
        return False


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
        if (
            hasattr(unet, "_original_in_channels")
            and "_original_in_channels" not in unet.config
        ):
            try:
                unet.config["_original_in_channels"] = unet._original_in_channels
                print(
                    "Applied workaround: Added '_original_in_channels' to unet.config"
                )
            except TypeError:
                print(
                    "Warning: Could not directly add attribute to unet.config (FrozenDict is immutable)."
                )

    except Exception as e:
        print(f"Error loading BASE pipeline: {e}")
        return None, None, None, None

    # Load inpainting model
    inpainting_pipeline = None
    if inpainting_model_path:
        inpainting_model_path = Path(inpainting_model_path)
        print(f"Loading INPAINTING diffusion pipeline from: {inpainting_model_path}")
        try:
            inpainting_pipeline = DiffusionPipeline.from_pretrained(
                inpainting_model_path
            ).to(device)
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
    debug=False,
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
        output_dir=Path(output_dir) / "stitched_debug" if debug else None,
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
    inpaint_batch_size=20,
    save_intermediates=False,
    intermediates_dir=None,
    stitch_stage_name="",
    **kwargs,
):
    """
    Stitch multiple volumes along a specified axis using gap-filling or overlapping logic.
    Now uses batch inpainting optimization for improved performance.

    Args:
        save_intermediates: If True, saves the pre-inpainting volume
        intermediates_dir: Directory to save intermediate files
        stitch_stage_name: Name suffix for intermediate files (e.g., "length", "width", "height")
    """
    if len(volumes) == 1:
        return volumes[0]

    print(
        f"    Stitching {len(volumes)} volumes along axis {axis} with overlap/gap={overlap}"
    )

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
        print(
            f"    Gap-filling mode: gap_size={gap_size}, step={step}, total={total_axis_size}"
        )
    else:
        # Overlapping mode: blocks overlap and seams are inpainted
        step = axis_size - overlap
        total_axis_size = axis_size + (len(volumes) - 1) * step
        print(
            f"    Overlapping mode: overlap={overlap}, step={step}, total={total_axis_size}"
        )

    print(f"    Axis {axis}: size={axis_size}, step={step}, total={total_axis_size}")

    # Create output volume and place volumes with gaps or overlaps
    final_shape = vol_shape.copy()
    final_shape[axis] = total_axis_size

    # Convert to PyTorch tensors for inpainting
    C = 1  # Assume single channel for now
    if len(vol_shape) == 3:
        full_volume_pt = torch.zeros(
            (1, C, *final_shape), dtype=torch.float32, device="cpu"
        )
    else:
        full_volume_pt = torch.zeros(
            (1, *final_shape), dtype=torch.float32, device="cpu"
        )

    # Place first volume
    first_vol_pt = numpy_to_pt(first_vol)
    if first_vol_pt.dim() == 3:
        first_vol_pt = first_vol_pt.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and channel dims
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

    # Note: Pre-inpainting save is now done at the full stack level in create_railway_track_3d
    # (saves unified pre_inpainting_full_stack.vti with all strips placed with gaps)

    # Now perform inpainting at junction regions using batch optimization
    if inpainting_pipeline is not None:
        print(f"    Performing batch inpainting at {len(volumes) - 1} junctions...")

        # Use the optimized batch inpainting function
        from utils.eval_utils import batch_inpaint_junctions
        import argparse

        # Create junction information for batch inpainting
        junction_infos = []
        for i in range(len(volumes) - 1):
            if mask_type == "gap_filling_compatible":
                # Gap-filling mode: inpaint the gaps between blocks
                gap_size = overlap  # In gap mode, overlap parameter is gap size
                gap_start = (
                    i + 1
                ) * axis_size + i * gap_size  # End of previous block + previous gaps
                gap_center = gap_start + gap_size // 2

                # Define processing region around gap
                process_region_size = max(gap_size * 3, 16)  # Ensure sufficient context
                region_start = max(0, gap_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                junction_infos.append(
                    {
                        "junction_idx": i,
                        "region_start_d": region_start,
                        "region_end_d": region_end,
                        "junction_center_d": gap_center,
                        "region_depth": region_end - region_start,
                        "gap_size": gap_size,
                    }
                )
            else:
                # Overlapping mode: inpaint the overlapping regions
                junction_center = (i + 1) * axis_size - i * overlap - overlap // 2
                process_region_size = max(overlap * 3, 16)
                region_start = max(0, junction_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                junction_infos.append(
                    {
                        "junction_idx": i,
                        "region_start_d": region_start,
                        "region_end_d": region_end,
                        "junction_center_d": junction_center,
                        "region_depth": region_end - region_start,
                        "gap_size": 0,  # No gap in overlapping mode
                    }
                )

        # Create dummy args for batch inpainting
        dummy_args = argparse.Namespace(
            mask_type=mask_type,
        )

        # Apply batch inpainting
        batch_inpaint_junctions(
            full_volume_pt=full_volume_pt,
            junction_infos=junction_infos,
            inpainting_pipeline=inpainting_pipeline,
            num_inference_steps=inference_steps,
            device=device,
            args=dummy_args,
            seed=seed,
            use_gap_filling=(mask_type == "gap_filling_compatible"),
            output_dir=None,
            inpaint_iteratively=False,
            inpaint_iterations=3,
            inpaint_region_size_ratio=0.3,
            axis=axis,  # Pass the axis parameter
            max_batch_size=inpaint_batch_size,  # Use configurable batch size
        )

    # Convert back to numpy
    result_np = pt_to_numpy(full_volume_pt[0])
    if result_np.shape[0] == 1:  # Remove channel dimension if single channel
        result_np = result_np.squeeze(0)

    return result_np


def stitch_multiple_layers_with_cross_layer_batching(
    height_layers,
    axis,
    overlap,
    device,
    inpainting_pipeline,
    inference_steps,
    seed,
    mask_type="gap_filling_compatible",
    max_batch_size=20,  # Maximum number of junctions to process in a single batch
    save_intermediates=False,
    intermediates_dir=None,
    **kwargs,
):
    """
    Stitch multiple layers along a specified axis using true cross-layer batch inpainting.

    This function processes multiple height layers simultaneously, collecting all junctions
    from all layers and processing them in larger batches across layers for maximum efficiency.

    Args:
        height_layers: Dict mapping layer_id -> [(width_idx, volume), ...]
        axis: Axis along which to stitch (1 for width dimension)
        overlap: Overlap/gap size between volumes
        device: Device to use for computation
        inpainting_pipeline: Pipeline for inpainting
        inference_steps: Number of inference steps
        seed: Random seed base
        mask_type: Type of masking to use
        max_batch_size: Maximum junctions to process in a single batch
        **kwargs: Additional arguments

    Returns:
        List of stitched layers
    """
    print(
        f"    Cross-layer batching: Processing {len(height_layers)} layers with shared junction batching"
    )

    # Step 1: Set up all layer structures simultaneously
    layer_data = {}
    all_junction_tasks = []  # List of (layer_id, junction_info) tuples

    print(f"      Setting up all layer structures...")
    for layer_id, width_strips in height_layers.items():
        # Sort strips by width position
        width_strips = sorted(width_strips, key=lambda x: x[0])

        if len(width_strips) == 1:
            # Only one strip in this layer - no junctions needed
            layer_data[layer_id] = {"volume": width_strips[0][1], "junction_infos": []}
            continue

        # Create the volume structure (same as stitch_volumes_along_axis_with_inpainting)
        volumes = [strip[1] for strip in width_strips]

        # Calculate dimensions
        first_vol = volumes[0]
        vol_shape = list(first_vol.shape)
        axis_size = vol_shape[axis]

        # Calculate total size with gap-filling logic
        if mask_type == "gap_filling_compatible":
            gap_size = overlap
            step = axis_size + gap_size
            total_axis_size = len(volumes) * axis_size + (len(volumes) - 1) * gap_size
        else:
            step = axis_size - overlap
            total_axis_size = axis_size + (len(volumes) - 1) * step

        # Create output volume and place volumes with gaps or overlaps
        final_shape = vol_shape.copy()
        final_shape[axis] = total_axis_size

        # Convert to PyTorch tensors
        C = 1
        if len(vol_shape) == 3:
            full_volume_pt = torch.zeros(
                (1, C, *final_shape), dtype=torch.float32, device="cpu"
            )
        else:
            full_volume_pt = torch.zeros(
                (1, *final_shape), dtype=torch.float32, device="cpu"
            )

        # Place first volume
        first_vol_pt = numpy_to_pt(first_vol)
        if first_vol_pt.dim() == 3:
            first_vol_pt = first_vol_pt.unsqueeze(0).unsqueeze(0)
        elif first_vol_pt.dim() == 4:
            first_vol_pt = first_vol_pt.unsqueeze(0)

        if axis == 0:
            full_volume_pt[0, :, :axis_size, :, :] = first_vol_pt[
                0, :, :axis_size, :, :
            ]
        elif axis == 1:
            full_volume_pt[0, :, :, :axis_size, :] = first_vol_pt[
                0, :, :, :axis_size, :
            ]
        elif axis == 2:
            full_volume_pt[0, :, :, :, :axis_size] = first_vol_pt[
                0, :, :, :, :axis_size
            ]

        # Place subsequent volumes
        current_pos = step
        for i, volume in enumerate(volumes[1:], 1):
            end_pos = current_pos + axis_size

            vol_pt = numpy_to_pt(volume)
            if vol_pt.dim() == 3:
                vol_pt = vol_pt.unsqueeze(0).unsqueeze(0)
            elif vol_pt.dim() == 4:
                vol_pt = vol_pt.unsqueeze(0)

            if axis == 0:
                full_volume_pt[0, :, current_pos:end_pos, :, :] = vol_pt[0, :, :, :, :]
            elif axis == 1:
                full_volume_pt[0, :, :, current_pos:end_pos, :] = vol_pt[0, :, :, :, :]
            elif axis == 2:
                full_volume_pt[0, :, :, :, current_pos:end_pos] = vol_pt[0, :, :, :, :]

            current_pos += step

        # NOTE: Keep volume on CPU here - will be moved to GPU only when processing
        # This is critical for memory efficiency with many layers

        # Note: Pre-inpainting save is now done at the full stack level in create_railway_track_3d

        # Create junction information for this layer
        layer_junction_infos = []
        for i in range(len(volumes) - 1):
            if mask_type == "gap_filling_compatible":
                gap_size = overlap
                gap_start = (i + 1) * axis_size + i * gap_size
                gap_center = gap_start + gap_size // 2
                process_region_size = max(gap_size * 3, 16)
                region_start = max(0, gap_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                junction_info = {
                    "junction_idx": i,
                    "region_start_d": region_start,
                    "region_end_d": region_end,
                    "junction_center_d": gap_center,
                    "region_depth": region_end - region_start,
                    "gap_size": gap_size,
                }
            else:
                junction_center = (i + 1) * axis_size - i * overlap - overlap // 2
                process_region_size = max(overlap * 3, 16)
                region_start = max(0, junction_center - process_region_size // 2)
                region_end = min(total_axis_size, region_start + process_region_size)

                junction_info = {
                    "junction_idx": i,
                    "region_start_d": region_start,
                    "region_end_d": region_end,
                    "junction_center_d": junction_center,
                    "region_depth": region_end - region_start,
                    "gap_size": 0,
                }

            layer_junction_infos.append(junction_info)
            # Add to global task list with layer reference
            all_junction_tasks.append((layer_id, junction_info))

        layer_data[layer_id] = {
            "volume": full_volume_pt,  # Keep on CPU until processing
            "junction_infos": layer_junction_infos,
        }

    # Step 2: Process layers ONE AT A TIME to avoid GPU OOM
    # This is more memory efficient than loading all layers to GPU simultaneously
    if inpainting_pipeline is not None and all_junction_tasks:
        total_junctions = len(all_junction_tasks)
        print(
            f"    Processing {total_junctions} junctions across {len(height_layers)} layers (memory-efficient mode)..."
        )

        from utils.eval_utils import batch_inpaint_junctions
        import argparse

        # Create dummy args for batch inpainting
        dummy_args = argparse.Namespace(
            mask_type=mask_type,
        )

        # Group tasks by layer
        layer_tasks = {}
        for layer_id, junction_info in all_junction_tasks:
            if layer_id not in layer_tasks:
                layer_tasks[layer_id] = []
            layer_tasks[layer_id].append(junction_info)

        # Process layers ONE AT A TIME to minimize GPU memory usage
        layer_ids = list(layer_tasks.keys())
        processed_layers = 0

        for layer_id in layer_ids:
            junction_infos = layer_tasks[layer_id]

            if junction_infos:
                print(
                    f"      Processing {len(junction_infos)} junctions for layer {layer_id}"
                )

                # Move this layer's volume to GPU for processing
                layer_volume = layer_data[layer_id]["volume"]
                if not layer_volume.is_cuda:
                    layer_volume = layer_volume.to(device)
                    layer_data[layer_id]["volume"] = layer_volume

                batch_inpaint_junctions(
                    full_volume_pt=layer_data[layer_id]["volume"],
                    junction_infos=junction_infos,
                    inpainting_pipeline=inpainting_pipeline,
                    num_inference_steps=inference_steps,
                    device=device,
                    args=dummy_args,
                    seed=seed + layer_id * 10000,
                    use_gap_filling=(mask_type == "gap_filling_compatible"),
                    output_dir=None,
                    inpaint_iteratively=False,
                    inpaint_iterations=3,
                    inpaint_region_size_ratio=0.3,
                    axis=axis,
                    max_batch_size=max_batch_size,
                )

                # Move back to CPU immediately after processing to free GPU memory
                layer_data[layer_id]["volume"] = layer_data[layer_id]["volume"].cpu()
                torch.cuda.empty_cache()

                processed_layers += 1

        print(
            f"    Processed {processed_layers} layers with {total_junctions} total junctions"
        )

    # Step 3: Convert results back to numpy and return
    stitched_layers = []
    for layer_id in sorted(layer_data.keys()):
        layer_info = layer_data[layer_id]

        if isinstance(layer_info["volume"], torch.Tensor):
            # Convert tensor back to numpy
            result_np = pt_to_numpy(layer_info["volume"][0])
            if result_np.shape[0] == 1:
                result_np = result_np.squeeze(0)
            stitched_layers.append(result_np)
        else:
            # Already numpy array
            stitched_layers.append(layer_info["volume"])

    print(f"    Cross-layer batching completed for {len(stitched_layers)} layers")
    return stitched_layers


def create_strips_in_batches(
    unet,
    scheduler,
    inpainting_pipeline,
    device,
    output_dir,
    strip_positions,
    grids_length,
    overlap_d,
    inference_steps,
    seed,
    stitching_mode,
    mask_type,
    batch_size,
    binary,
    debug,
    strip_batch_size=4,
    save_intermediates=False,
    intermediates_dir=None,
    **kwargs,
):
    """
    Create multiple strips in batches to leverage parallelization.

    Args:
        strip_positions: List of (j, k) positions for strips to create
        strip_batch_size: Number of strips to process in parallel
        save_intermediates: If True, saves sampled blocks for visualization
        intermediates_dir: Directory to save intermediate outputs
        Other args: Same as create_railway_track_1d

    Returns:
        List of strip dictionaries with volume and position
    """
    strips = []
    total_strips = len(strip_positions)

    # Track all sampled blocks for intermediate saving
    all_sampled_blocks = [] if save_intermediates else None

    print(f"Creating {total_strips} strips in batches of {strip_batch_size}")

    # Process strips in batches
    for batch_start in range(0, total_strips, strip_batch_size):
        batch_end = min(batch_start + strip_batch_size, total_strips)
        batch_positions = strip_positions[batch_start:batch_end]
        current_batch_size = len(batch_positions)

        print(
            f"  Processing strip batch {batch_start // strip_batch_size + 1}/{(total_strips + strip_batch_size - 1) // strip_batch_size} "
            f"({current_batch_size} strips)"
        )

        # Generate all blocks for this batch of strips simultaneously
        batch_strips = []

        # Calculate total blocks needed for this batch
        total_blocks_needed = current_batch_size * grids_length

        # Generate all blocks for all strips in this batch at once
        print(
            f"    Generating {total_blocks_needed} blocks for {current_batch_size} strips in parallel..."
        )

        # Create a large batch with all blocks for all strips
        all_blocks = []
        all_block_seeds = []

        for idx, (j, k) in enumerate(batch_positions):
            strip_seed = seed + j * 1000 + k * 100

            # Generate seeds for each block in this strip
            for block_idx in range(grids_length):
                block_seed = strip_seed + block_idx
                all_block_seeds.append(block_seed)

        # Generate all blocks in large batches
        from utils.eval_utils import generate_single_volume
        import argparse

        # Create temporary args for block generation
        temp_args = argparse.Namespace(
            mask_type="none",  # Ensure unconditional generation for initial blocks
            **{k: v for k, v in kwargs.items() if k != "mask_type"},
        )

        # Generate blocks in batches
        effective_batch_size = min(
            batch_size * 2, total_blocks_needed
        )  # Use larger batch size

        for block_batch_start in range(0, total_blocks_needed, effective_batch_size):
            block_batch_end = min(
                block_batch_start + effective_batch_size, total_blocks_needed
            )
            current_block_batch_size = block_batch_end - block_batch_start

            # Use the first seed for this batch (we'll handle randomness differently)
            batch_seed = all_block_seeds[block_batch_start]

            blocks_batch = generate_single_volume(
                unet=unet,
                scheduler=scheduler,
                num_steps=inference_steps,
                seed=batch_seed,
                batch_size=current_block_batch_size,
                device=device,
                args=temp_args,
                min_bw_ratio=0.0,
                max_retries=0,
                progress_callback=None,
            )

            if blocks_batch:
                all_blocks.extend(blocks_batch)

            # Clear GPU cache after each block batch to prevent accumulation
            torch.cuda.empty_cache()

        print(f"    Generated {len(all_blocks)} blocks total")

        # Save sampled blocks for intermediate visualization
        if save_intermediates and intermediates_dir and len(all_blocks) > 0:
            # Save first batch of blocks as NPZ
            if batch_start == 0:  # Only save once for the first batch
                print(f"    Saving sampled blocks for visualization...")
                blocks_to_save = [
                    b.cpu().numpy() if isinstance(b, torch.Tensor) else b
                    for b in all_blocks[: min(12, len(all_blocks))]
                ]
                np.savez(
                    intermediates_dir / "sampled_blocks.npz",
                    blocks=np.array(blocks_to_save),
                )
                print(f"    Saved {len(blocks_to_save)} blocks to sampled_blocks.npz")

                # Save individual blocks as VTI files for ParaView
                save_blocks_as_vti(
                    blocks_to_save[:6],  # Save first 6 blocks
                    intermediates_dir / "blocks_vti",
                    prefix="block",
                )

                # Generate visualization
                save_blocks_visualization(
                    all_blocks[: min(6, len(all_blocks))],
                    intermediates_dir / "voxel_blocks.png",
                    max_blocks=6,
                )

        # Now organize blocks into strips and perform inpainting for each strip
        block_idx = 0
        for idx, (j, k) in enumerate(batch_positions):
            strip_seed = seed + j * 1000 + k * 100

            print(
                f"    Creating strip {batch_start + idx + 1}/{total_strips} for position (width={j}, height={k}) with seed={strip_seed}"
            )

            # Get blocks for this strip
            strip_blocks = all_blocks[block_idx : block_idx + grids_length]
            block_idx += grids_length

            if len(strip_blocks) == grids_length:
                # Create a single volume from pre-generated blocks
                # Use the optimized stitching function that supports batch inpainting
                from utils.eval_utils import stitch_blocks_with_batch_inpainting

                strip = stitch_blocks_with_batch_inpainting(
                    volumes=strip_blocks,
                    axis=0,  # Length axis
                    overlap=overlap_d,
                    inpainting_pipeline=inpainting_pipeline,
                    device=device,
                    output_dir=output_dir / f"strip_{j}_{k}" if debug else output_dir,
                    inference_steps=inference_steps,
                    seed=strip_seed,
                    binary=binary,
                    debug=debug,
                )

                if strip is not None:
                    batch_strips.append(
                        {
                            "volume": strip,
                            "position": (j, k),  # width, height position
                        }
                    )
                    print(f"      ✓ Strip completed. Shape: {strip.shape}")
                else:
                    print(
                        f"      ✗ Warning: Failed to create strip at position ({j}, {k})"
                    )
            else:
                print(
                    f"      ✗ Warning: Not enough blocks generated for strip at position ({j}, {k})"
                )

        strips.extend(batch_strips)
        print(f"    Batch completed. {len(batch_strips)} strips created successfully.")

        # Clear memory after each strip batch to prevent accumulation
        del all_blocks
        del batch_strips
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return strips


def create_railway_track_3d_chunked(
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
    debug=False,
    strip_batch_size=4,
    inpaint_batch_size=20,
    chunk_length=10,  # Number of length blocks per chunk
    num_gpus=1,  # Number of GPUs to use for parallel processing
    **kwargs,
):
    """
    Memory-efficient chunked 3D railway track generation.

    Processes the track in length chunks to avoid CUDA OOM for very long tracks.
    Each chunk is processed independently and saved to disk, then optionally
    stitched together at the end.

    Args:
        chunk_length: Number of length blocks to process per chunk (default: 10)
                     For base_length=0.3m, chunk_length=10 means 3m chunks
        num_gpus: Number of GPUs to use for parallel chunk processing (default: 1)
    """
    print(f"Creating 3D railway track (CHUNKED MODE): {grids_length}x{grids_width}x{grids_height} grids")
    print(f"Chunk size: {chunk_length} length blocks per chunk")
    print(f"Overlaps: D={overlap_d}, H={overlap_h}, W={overlap_w}")

    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Calculate number of chunks needed
    num_chunks = (grids_length + chunk_length - 1) // chunk_length
    print(f"Will process {num_chunks} chunks")

    # Detect available GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_num_gpus = min(num_gpus, available_gpus, num_chunks)

    if effective_num_gpus > 1:
        print(f"Multi-GPU mode: Using {effective_num_gpus} GPUs for parallel chunk processing")
        return _create_chunks_multi_gpu(
            unet=unet,
            scheduler=scheduler,
            inpainting_pipeline=inpainting_pipeline,
            output_dir=output_dir,
            chunks_dir=chunks_dir,
            grids_length=grids_length,
            grids_width=grids_width,
            grids_height=grids_height,
            overlap_d=overlap_d,
            overlap_h=overlap_h,
            overlap_w=overlap_w,
            inference_steps=inference_steps,
            seed=seed,
            stitching_mode=stitching_mode,
            mask_type=mask_type,
            batch_size=batch_size,
            binary=binary,
            debug=debug,
            strip_batch_size=strip_batch_size,
            inpaint_batch_size=inpaint_batch_size,
            chunk_length=chunk_length,
            num_chunks=num_chunks,
            num_gpus=effective_num_gpus,
            **kwargs,
        )

    # Single GPU mode (original implementation)
    print(f"Single GPU mode: Processing chunks sequentially on {device}")
    chunk_files = []

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_length
        chunk_end = min(chunk_start + chunk_length, grids_length)
        actual_chunk_length = chunk_end - chunk_start

        print(f"\n{'='*60}")
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: blocks {chunk_start}-{chunk_end-1}")
        print(f"{'='*60}")

        # Adjust seed for this chunk to ensure reproducibility
        chunk_seed = seed + chunk_idx * 100000

        # Generate this chunk using the regular 3D function but with reduced length
        chunk_track = create_railway_track_3d(
            unet=unet,
            scheduler=scheduler,
            inpainting_pipeline=inpainting_pipeline,
            device=device,
            output_dir=output_dir / f"chunk_{chunk_idx:03d}",
            grids_length=actual_chunk_length,
            grids_width=grids_width,
            grids_height=grids_height,
            overlap_d=overlap_d,
            overlap_h=overlap_h,
            overlap_w=overlap_w,
            inference_steps=inference_steps,
            seed=chunk_seed,
            stitching_mode=stitching_mode,
            mask_type=mask_type,
            batch_size=batch_size,
            binary=binary,
            debug=debug,
            strip_batch_size=strip_batch_size,
            inpaint_batch_size=inpaint_batch_size,
            save_intermediates=False,  # Don't save intermediates for chunks
            intermediates_dir=None,
            **kwargs,
        )

        if chunk_track is None:
            print(f"ERROR: Failed to generate chunk {chunk_idx}")
            continue

        # Save chunk to disk immediately to free memory
        chunk_file = chunks_dir / f"chunk_{chunk_idx:03d}.npy"
        np.save(chunk_file, chunk_track)
        chunk_files.append(chunk_file)
        print(f"Saved chunk {chunk_idx + 1}/{num_chunks} to {chunk_file} (shape: {chunk_track.shape})")

        # Free memory
        del chunk_track
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f"Memory cleared after chunk {chunk_idx + 1}")

    return _stitch_chunk_files(chunk_files, chunks_dir, binary)


def _process_chunk_on_gpu(args):
    """Worker function to process a single chunk on a specific GPU."""
    (chunk_idx, gpu_id, chunk_length, grids_length, grids_width, grids_height,
     overlap_d, overlap_h, overlap_w, inference_steps, seed, stitching_mode,
     mask_type, batch_size, binary, debug, strip_batch_size, inpaint_batch_size,
     output_dir, chunks_dir, model_path, inpainting_model_path, scheduler_type,
     kwargs) = args

    import torch
    import numpy as np
    from pathlib import Path

    # Set this process to use the specified GPU
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Processing chunk {chunk_idx}")

    # Load models on this GPU
    from railway_track import load_models, create_railway_track_3d
    unet, scheduler, inpainting_pipeline, _ = load_models(
        model_path, inpainting_model_path, scheduler_type
    )

    # Move models to this GPU
    unet = unet.to(device)
    if inpainting_pipeline is not None:
        inpainting_pipeline = inpainting_pipeline.to(device)

    chunk_start = chunk_idx * chunk_length
    chunk_end = min(chunk_start + chunk_length, grids_length)
    actual_chunk_length = chunk_end - chunk_start
    chunk_seed = seed + chunk_idx * 100000

    output_dir = Path(output_dir)
    chunks_dir = Path(chunks_dir)

    try:
        chunk_track = create_railway_track_3d(
            unet=unet,
            scheduler=scheduler,
            inpainting_pipeline=inpainting_pipeline,
            device=device,
            output_dir=output_dir / f"chunk_{chunk_idx:03d}",
            grids_length=actual_chunk_length,
            grids_width=grids_width,
            grids_height=grids_height,
            overlap_d=overlap_d,
            overlap_h=overlap_h,
            overlap_w=overlap_w,
            inference_steps=inference_steps,
            seed=chunk_seed,
            stitching_mode=stitching_mode,
            mask_type=mask_type,
            batch_size=batch_size,
            binary=binary,
            debug=debug,
            strip_batch_size=strip_batch_size,
            inpaint_batch_size=inpaint_batch_size,
            save_intermediates=False,
            intermediates_dir=None,
            **kwargs,
        )

        if chunk_track is not None:
            chunk_file = chunks_dir / f"chunk_{chunk_idx:03d}.npy"
            np.save(chunk_file, chunk_track)
            print(f"[GPU {gpu_id}] Saved chunk {chunk_idx} to {chunk_file}")
            return str(chunk_file)
        else:
            print(f"[GPU {gpu_id}] ERROR: Failed to generate chunk {chunk_idx}")
            return None
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR processing chunk {chunk_idx}: {e}")
        return None
    finally:
        # Clean up
        del unet, scheduler, inpainting_pipeline
        torch.cuda.empty_cache()


def _create_chunks_multi_gpu(
    unet, scheduler, inpainting_pipeline, output_dir, chunks_dir,
    grids_length, grids_width, grids_height,
    overlap_d, overlap_h, overlap_w, inference_steps, seed,
    stitching_mode, mask_type, batch_size, binary, debug,
    strip_batch_size, inpaint_batch_size, chunk_length, num_chunks, num_gpus,
    **kwargs
):
    """Process chunks in parallel across multiple GPUs."""
    import torch.multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Get model paths from kwargs or use defaults
    model_path = kwargs.get('model_path', None)
    inpainting_model_path = kwargs.get('inpainting_model_path', None)
    scheduler_type = kwargs.get('scheduler_type', 'ddim')

    if model_path is None:
        print("WARNING: model_path not provided for multi-GPU mode, falling back to single GPU")
        # Fall back to single GPU - can't reload models without path
        return None

    # Prepare arguments for each chunk
    chunk_args = []
    for chunk_idx in range(num_chunks):
        gpu_id = chunk_idx % num_gpus  # Round-robin GPU assignment
        args = (
            chunk_idx, gpu_id, chunk_length, grids_length, grids_width, grids_height,
            overlap_d, overlap_h, overlap_w, inference_steps, seed, stitching_mode,
            mask_type, batch_size, binary, debug, strip_batch_size, inpaint_batch_size,
            str(output_dir), str(chunks_dir), model_path, inpainting_model_path,
            scheduler_type, kwargs
        )
        chunk_args.append(args)

    print(f"Starting multi-GPU processing: {num_chunks} chunks on {num_gpus} GPUs")

    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    chunk_files = []
    try:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(_process_chunk_on_gpu, args): args[0]
                      for args in chunk_args}

            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        chunk_files.append(Path(result))
                        print(f"Chunk {chunk_idx} completed successfully")
                except Exception as e:
                    print(f"Chunk {chunk_idx} failed with error: {e}")

    except Exception as e:
        print(f"Multi-GPU processing failed: {e}")
        print("Falling back to single GPU sequential processing...")
        return None

    # Sort chunk files by index to maintain order
    chunk_files = sorted(chunk_files, key=lambda p: int(p.stem.split('_')[1]))

    return _stitch_chunk_files(chunk_files, chunks_dir, binary)


def _stitch_chunk_files(chunk_files, chunks_dir, binary):
    """Stitch chunk files together into a single track."""
    print(f"\n{'='*60}")
    print(f"All {len(chunk_files)} chunks generated successfully")
    print(f"Chunk files saved to: {chunks_dir}")
    print(f"{'='*60}")

    if len(chunk_files) == 0:
        print("ERROR: No chunks were generated")
        return None

    if len(chunk_files) == 1:
        return np.load(chunk_files[0])

    print("\nAttempting to stitch chunks together...")
    try:
        stitched_track = None

        for i, chunk_file in enumerate(chunk_files):
            chunk = np.load(chunk_file)

            if stitched_track is None:
                stitched_track = chunk
            else:
                stitched_track = np.concatenate([stitched_track, chunk], axis=0)

            del chunk

            if i % 5 == 0:
                import gc
                gc.collect()

        print(f"Successfully stitched all chunks. Final shape: {stitched_track.shape}")

        if binary:
            stitched_track = (stitched_track > 0.5).astype(np.float32)

        return stitched_track

    except MemoryError:
        print("\nWARNING: Not enough memory to stitch all chunks together")
        print(f"Chunks are saved individually in: {chunks_dir}")
        print("You can process them separately or stitch them on a machine with more RAM")
        return np.load(chunk_files[0])


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
    debug=False,
    strip_batch_size=4,
    inpaint_batch_size=20,
    layer_batch_size=4,  # New parameter for height layer batching
    save_intermediates=False,
    intermediates_dir=None,
    **kwargs,
):
    """
    Create a 3D railway track by extending in multiple dimensions using pre-loaded models.

    Strategy:
    1. First create strips along length dimension (using batched approach)
    2. Then stitch strips along width dimension
    3. Finally stitch layers along height dimension

    Args:
        save_intermediates: If True, saves intermediate pipeline stages for paper figures.
        intermediates_dir: Directory to save intermediate outputs.
    """
    print(
        f"Creating 3D railway track: {grids_length}x{grids_width}x{grids_height} grids"
    )
    print(f"Overlaps: D={overlap_d}, H={overlap_h}, W={overlap_w}")
    print(f"Strip batch size: {strip_batch_size}")
    if save_intermediates:
        print(f"Saving intermediate outputs to: {intermediates_dir}")

    # Step 1: Create strips along length dimension (D axis) using batched approach
    print("Step 1: Creating length strips in batches...")

    # Generate all strip positions
    strip_positions = []
    for j in range(grids_width):
        for k in range(grids_height):
            strip_positions.append((j, k))

    # Create strips in batches
    strips = create_strips_in_batches(
        unet=unet,
        scheduler=scheduler,
        inpainting_pipeline=inpainting_pipeline,
        device=device,
        output_dir=output_dir,
        strip_positions=strip_positions,
        grids_length=grids_length,
        overlap_d=overlap_d,
        inference_steps=inference_steps,
        seed=seed,
        stitching_mode=stitching_mode,
        mask_type=mask_type,
        batch_size=batch_size,
        binary=binary,
        debug=debug,
        save_intermediates=save_intermediates,
        intermediates_dir=intermediates_dir,
        strip_batch_size=strip_batch_size,
        **kwargs,
    )

    if not strips:
        print("Error: No strips were created successfully.")
        return None

    print(f"Successfully created {len(strips)} strips")

    # Save full stack with gaps BEFORE any width/height inpainting
    if save_intermediates and intermediates_dir and len(strips) > 0:
        print("Assembling full stack with gaps (before width/height inpainting)...")

        # Get dimensions from first strip
        first_strip = strips[0]["volume"]
        strip_shape = first_strip.shape  # (D, H_single, W_single) for a strip

        # Calculate full volume dimensions with gaps
        # Strips are arranged in a grid of (grids_width x grids_height)
        gap_h = overlap_h if mask_type == "gap_filling_compatible" else 0
        gap_w = overlap_w if mask_type == "gap_filling_compatible" else 0

        # For gap-filling mode: total = n * size + (n-1) * gap
        if mask_type == "gap_filling_compatible":
            total_h = grids_width * strip_shape[1] + (grids_width - 1) * gap_h
            total_w = grids_height * strip_shape[2] + (grids_height - 1) * gap_w
        else:
            # Overlapping mode
            total_h = strip_shape[1] + (grids_width - 1) * (strip_shape[1] - overlap_h)
            total_w = strip_shape[2] + (grids_height - 1) * (strip_shape[2] - overlap_w)

        full_stack = np.zeros((strip_shape[0], total_h, total_w), dtype=np.float32)

        # Place each strip in the full volume with gaps
        for strip_data in strips:
            j, k = strip_data["position"]  # width_idx, height_idx
            volume = strip_data["volume"]

            # Calculate position with gaps
            if mask_type == "gap_filling_compatible":
                h_start = j * (strip_shape[1] + gap_h)
                w_start = k * (strip_shape[2] + gap_w)
            else:
                h_start = j * (strip_shape[1] - overlap_h)
                w_start = k * (strip_shape[2] - overlap_w)

            h_end = h_start + strip_shape[1]
            w_end = w_start + strip_shape[2]

            # Place the strip (use max to handle overlaps)
            full_stack[:, h_start:h_end, w_start:w_end] = np.maximum(
                full_stack[:, h_start:h_end, w_start:w_end], volume
            )

        # Save as VTI for ParaView
        pre_inpaint_path = Path(intermediates_dir) / "pre_inpainting_full_stack.vti"
        try:
            save_vti_without_pyvista(full_stack, pre_inpaint_path)
            print(f"  Saved full stack with gaps to: {pre_inpaint_path}")
            print(f"  Shape: {full_stack.shape}")
        except Exception as e:
            print(f"  Warning: Could not save full stack VTI: {e}")

    # Step 2: Stitch strips along width dimension (H axis)
    print("Step 2: Stitching strips along width dimension...")

    # Group strips by height
    height_layers = {}
    for strip_data in strips:
        j, k = strip_data["position"]  # width, height
        if k not in height_layers:
            height_layers[k] = []
        height_layers[k].append((j, strip_data["volume"]))

    # Process multiple height layers in parallel batches
    height_layer_keys = sorted(height_layers.keys())

    stitched_layers = []
    total_layers = len(height_layer_keys)

    print(f"  Processing {total_layers} height layers using cross-layer batching")

    # Use the cross-layer batching function for better performance
    stitched_layers = stitch_multiple_layers_with_cross_layer_batching(
        height_layers=height_layers,
        axis=1,  # H axis
        overlap=overlap_h,
        device=device,
        inpainting_pipeline=inpainting_pipeline,
        inference_steps=inference_steps,
        seed=seed,
        mask_type=mask_type,
        max_batch_size=inpaint_batch_size,  # Use configurable batch size
        save_intermediates=save_intermediates,
        intermediates_dir=intermediates_dir,
        **kwargs,
    )

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
            inpaint_batch_size=inpaint_batch_size,
            save_intermediates=save_intermediates,
            intermediates_dir=intermediates_dir,
            stitch_stage_name="height",
            **kwargs,
        )

    # Apply binary thresholding if requested
    if binary:
        final_track = (final_track > 0.5).astype(np.float32)

    print(f"Final 3D railway track shape: {final_track.shape}")

    # Save stitched volume visualization if intermediates requested
    if save_intermediates and intermediates_dir:
        print("Saving stitched volume visualizations...")

        # Save the stitched volume as NPZ
        np.savez(intermediates_dir / "stitched_volume.npz", volume=final_track)
        print(f"  Saved stitched volume to stitched_volume.npz")

        # Save the stitched volume as VTI for ParaView
        try:
            save_vti_without_pyvista(
                final_track, intermediates_dir / "stitched_volume.vti"
            )
        except Exception as e:
            print(f"  Warning: Could not save stitched VTI: {e}")

        # Save 2D slice visualization
        save_intermediate_visualization(
            final_track,
            intermediates_dir / "stitched_volume_slices.png",
            title=f"Stitched Volume ({final_track.shape[0]}x{final_track.shape[1]}x{final_track.shape[2]})",
        )

        # Save 3D render if possible (will be skipped on headless servers)
        save_3d_volume_render(
            final_track,
            intermediates_dir / "full_voxel_volume.png",
            title=f"Full Stitched Volume",
        )

        print(f"  Pipeline visualizations saved to: {intermediates_dir}")

    return final_track


def create_railway_track(
    model_path,
    inpainting_model_path,
    output_dir,
    target_volume,  # (depth, width, length) in real units
    base_volume=(
        0.1,
        0.3,
        0.3,
    ),  # Volume represented by a single (32, 64, 64) voxel grid
    overlap_d=8,
    overlap_w=8,
    overlap_l=8,
    scheduler_type="ddim",
    debug=False,
    strip_batch_size=4,
    inpaint_batch_size=20,
    save_intermediates=False,
    chunk_length=None,  # Number of length blocks per chunk (None = auto, 0 = disabled)
    chunk_threshold=50,  # Auto-enable chunking when grids_depth exceeds this
    num_gpus=1,  # Number of GPUs for parallel chunk processing
    **kwargs,
):
    """
    Main function to create railway track of specified dimensions.
    Loads models once and reuses them for efficiency.

    Args:
        save_intermediates: If True, saves intermediate pipeline stages and visualizations
                           to output_dir/intermediates/ for paper figures.
        chunk_length: Number of length blocks per chunk for memory-efficient generation.
                     None = auto-detect based on track length
                     0 = disabled (process entire track at once)
                     >0 = use specified chunk size
        chunk_threshold: When grids_depth exceeds this, automatically enable chunking.
        num_gpus: Number of GPUs for parallel chunk processing. 0 = auto-detect all available.
    """
    # Create intermediates directory if saving
    intermediates_dir = None
    if save_intermediates:
        intermediates_dir = Path(output_dir) / "intermediates"
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        print(f"Intermediate outputs will be saved to: {intermediates_dir}")

    # Load models once
    print("Loading models...")
    unet, scheduler, inpainting_pipeline, device = load_models(
        model_path, inpainting_model_path, scheduler_type
    )

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

    # Determine if chunking should be used
    # Use the longest dimension for chunking decision
    longest_dimension = max(grids_depth, grids_width, grids_length)
    use_chunking = False
    effective_chunk_length = 10  # Default chunk size

    if chunk_length is None:
        # Auto-detect: enable chunking for long tracks
        if longest_dimension > chunk_threshold:
            use_chunking = True
            effective_chunk_length = min(20, max(10, longest_dimension // 10))
            print(f"Auto-enabling chunked mode for long track ({longest_dimension} blocks > {chunk_threshold} threshold)")
            print(f"Using chunk size: {effective_chunk_length} blocks")
    elif chunk_length > 0:
        use_chunking = True
        effective_chunk_length = chunk_length
        print(f"Chunked mode enabled with chunk size: {effective_chunk_length} blocks")
    else:
        print("Chunked mode disabled (chunk_length=0)")

    if grids_depth == 1 and grids_width == 1 and grids_length == 1:
        # Single block case
        print("Single block case - using simple generation")
        args = argparse.Namespace(**kwargs)
        volumes = generate_single_volume(
            unet,
            scheduler,
            kwargs.get("inference_steps", 60),
            kwargs.get("seed", 123),
            1,
            device,
            args,
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
            debug=debug,
            **kwargs,
        )
    else:
        # 3D case - use chunked mode if enabled
        if use_chunking:
            print("3D case - using CHUNKED mode for memory efficiency")
            # Auto-detect GPUs if num_gpus is 0
            effective_num_gpus = num_gpus
            if num_gpus == 0:
                effective_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                print(f"Auto-detected {effective_num_gpus} GPU(s)")

            return create_railway_track_3d_chunked(
                unet=unet,
                scheduler=scheduler,
                inpainting_pipeline=inpainting_pipeline,
                device=device,
                output_dir=output_dir,
                grids_length=grids_length,  # Chunk along the LONG dimension (e.g., 834 for 250m)
                grids_width=grids_width,
                grids_height=grids_depth,   # Short dimension (e.g., 3 for 0.3m depth)
                overlap_d=overlap_d,
                overlap_h=overlap_w,
                overlap_w=overlap_l,
                debug=debug,
                strip_batch_size=strip_batch_size,
                inpaint_batch_size=inpaint_batch_size,
                chunk_length=effective_chunk_length,
                num_gpus=effective_num_gpus,
                # Pass model paths for multi-GPU mode to reload models
                model_path=model_path,
                inpainting_model_path=inpainting_model_path,
                scheduler_type=scheduler_type,
                **kwargs,
            )
        else:
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
                debug=debug,
                strip_batch_size=strip_batch_size,
                inpaint_batch_size=inpaint_batch_size,
                layer_batch_size=4,  # Default layer batch size
                save_intermediates=save_intermediates,
                intermediates_dir=intermediates_dir,
                **kwargs,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Railway Track Generator - Create voxel tracks of specified dimensions using proper gap-filling"
    )

    # Required model paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained diffusion model directory",
    )
    parser.add_argument(
        "--inpainting_model_path",
        type=str,
        default=None,
        help="Path to the dedicated inpainting model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="railway_tracks",
        help="Output directory for saving railway track files",
    )

    # Target dimensions
    parser.add_argument(
        "--target_depth",
        type=float,
        required=True,
        help="Target depth of railway track (real units)",
    )
    parser.add_argument(
        "--target_width",
        type=float,
        required=True,
        help="Target width of railway track (real units)",
    )
    parser.add_argument(
        "--target_length",
        type=float,
        required=True,
        help="Target length of railway track (real units)",
    )

    # Base unit dimensions (defaults match your description)
    parser.add_argument(
        "--base_depth",
        type=float,
        default=0.1,
        help="Depth represented by single voxel grid (default: 0.1)",
    )
    parser.add_argument(
        "--base_width",
        type=float,
        default=0.3,
        help="Width represented by single voxel grid (default: 0.3)",
    )
    parser.add_argument(
        "--base_length",
        type=float,
        default=0.3,
        help="Length represented by single voxel grid (default: 0.3)",
    )

    # Overlap/gap parameters
    parser.add_argument(
        "--overlap_d",
        type=int,
        default=8,
        help="Gap size along depth dimension (default: 8)",
    )
    parser.add_argument(
        "--overlap_w",
        type=int,
        default=8,
        help="Gap size along width dimension (default: 8)",
    )
    parser.add_argument(
        "--overlap_l",
        type=int,
        default=8,
        help="Gap size along length dimension (default: 8)",
    )

    # Generation parameters
    parser.add_argument(
        "--scheduler_type",
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Choose sampling scheduler",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=60, help="Number of inference steps"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for generation"
    )
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
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--strip_batch_size",
        type=int,
        default=4,
        help="Number of strips to process in parallel (default: 4)",
    )
    parser.add_argument(
        "--inpaint_batch_size",
        type=int,
        default=20,
        help="Maximum batch size for inpainting junctions to avoid OOM (default: 20)",
    )
    parser.add_argument(
        "--binary", action="store_true", help="Threshold output to binary mask (>0.5)"
    )

    # Memory management / Chunking parameters
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=None,
        help="Number of length blocks per chunk for memory-efficient generation. "
             "None (default) = auto-detect based on track length, "
             "0 = disabled (process entire track at once), "
             ">0 = use specified chunk size. For 100m tracks, try 10-20.",
    )
    parser.add_argument(
        "--chunk_threshold",
        type=int,
        default=50,
        help="Auto-enable chunking when grids_depth exceeds this value (default: 50). "
             "For base_depth=0.3m, 50 blocks = 15m track length.",
    )
    parser.add_argument(
        "--no_chunking",
        action="store_true",
        help="Disable automatic chunking even for long tracks (may cause OOM)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel chunk processing (default: 1). "
             "Set to 0 for auto-detect all available GPUs.",
    )

    # Inpainting parameters
    parser.add_argument(
        "--inpaint_region_size_ratio",
        type=float,
        default=0.3,
        help="Size of inpainting region as ratio of process region",
    )
    parser.add_argument(
        "--inpaint_iteratively",
        action="store_true",
        help="Whether to inpaint in smaller iterations",
    )
    parser.add_argument(
        "--inpaint_iterations",
        type=int,
        default=3,
        help="Number of iterations for iterative inpainting",
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Value to threshold the final volume at (None = no thresholding)",
    )

    # Output format
    parser.add_argument(
        "--save_format",
        choices=["vti", "npy", "both"],
        default="both",
        help="Format to save the railway track",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom name for output file (default: auto-generated)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (saves intermediate strips)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        dest="save_intermediates",
        help="Save intermediate pipeline stages (blocks, stitched volume) and visualizations for paper figures",
    )

    args = parser.parse_args()

    # Validate inpainting model requirement
    if (
        args.stitching_mode == "separate_inpainting"
        and args.inpainting_model_path is None
    ):
        raise ValueError(
            "--inpainting_model_path is required when using --stitching_mode separate_inpainting"
        )

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
        debug=args.debug,
        inference_steps=args.inference_steps,
        seed=args.seed,
        stitching_mode=args.stitching_mode,
        mask_type=args.mask_type,
        batch_size=args.batch_size,
        strip_batch_size=args.strip_batch_size,
        inpaint_batch_size=args.inpaint_batch_size,
        binary=args.binary,
        save_intermediates=args.save_intermediates,
        inpaint_region_size_ratio=args.inpaint_region_size_ratio,
        inpaint_iteratively=args.inpaint_iteratively,
        inpaint_iterations=args.inpaint_iterations,
        threshold_value=args.threshold_value,
        chunk_length=0 if args.no_chunking else args.chunk_length,
        chunk_threshold=args.chunk_threshold,
        num_gpus=args.num_gpus,
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
        vti_path = output_dir / f"{base_name}.vti"

        # Try PyVista first (if available), otherwise use pure Python VTI writer
        if _check_pyvista_available():
            try:
                vtk_data = pv.ImageData(dimensions=railway_track.shape)
                vtk_data["voxel_data"] = railway_track.flatten(order="F")
                vtk_data.save(vti_path)
                print(f"Saved VTI file to: {vti_path}")
            except Exception as e:
                print(
                    f"Warning: PyVista VTI save failed ({e}), trying pure Python fallback..."
                )
                save_vti_without_pyvista(railway_track, vti_path)
        else:
            # Use pure Python VTI writer (no PyVista needed)
            print("Using pure Python VTI writer (PyVista not available)")
            save_vti_without_pyvista(railway_track, vti_path)

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
