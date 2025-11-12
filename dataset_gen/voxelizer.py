import argparse
import gzip
import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import vtk
from skimage.draw import polygon, line
from tqdm import tqdm
from vtk.util import numpy_support
from scipy.ndimage import binary_fill_holes


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate sparse slices from a .vtu file"
    )
    parser.add_argument(
        "--vtu_file", type=str, default=None, help="Path to the .vtu file"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory containing the .vtu files",
    )
    parser.add_argument(
        "--num_slices",
        "--ns",
        type=int,
        default=10,
        help="Number of slices to generate",
    )
    parser.add_argument(
        "--img_size",
        "--ii",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Size of the image (height, width)",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--num_workers", "--nw", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--min_z",
        type=float,
        default=0.05,
        help="Minimum z height for slicing (original dataset had a plate at the bottom)",
    )
    parser.add_argument(
        "--save_voxel", action="store_true", help="Save voxelized slices"
    )
    parser.add_argument(
        "--stem", type=str, default="gan_data_", help="Directory name stem to filter"
    )
    parser.add_argument(
        "--outlier_margin",
        type=float,
        default=1.0,
        help=(
            "Maximum distance (world units) a slice point may deviate from the slice mean "
            "before being discarded. Set to 0 or a negative value to disable outlier filtering."
        ),
    )
    parser.add_argument(
        "--voxel_mode",
        type=str,
        default="implicit",
        choices=["implicit", "slice"],
        help="Choose 'implicit' for distance-field voxelization (recommended) or 'slice' for legacy planar slicing.",
    )
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=0.02,
        help="Fractional padding to extend mesh bounds before voxelization (implicit mode only).",
    )
    return parser.parse_args()


def voxelize_vtu_implicit(vtu_file, depth, height, width, padding_frac=0.02):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()
    ugrid = reader.GetOutput()

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(ugrid)
    geom.Update()
    poly = geom.GetOutput()

    if poly.GetNumberOfCells() == 0:
        raise ValueError(f"Mesh '{vtu_file}' produced empty surface geometry.")

    bounds = list(poly.GetBounds())
    x_span = bounds[1] - bounds[0]
    y_span = bounds[3] - bounds[2]
    z_span = bounds[5] - bounds[4]

    pad_x = x_span * padding_frac
    pad_y = y_span * padding_frac
    pad_z = z_span * padding_frac
    bounds[0] -= pad_x
    bounds[1] += pad_x
    bounds[2] -= pad_y
    bounds[3] += pad_y
    bounds[4] -= pad_z
    bounds[5] += pad_z

    x_coords = np.linspace(bounds[0], bounds[1], width)
    y_coords = np.linspace(bounds[2], bounds[3], height)
    z_coords = np.linspace(bounds[4], bounds[5], depth)

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(poly)

    distances = np.array([implicit.EvaluateFunction(float(p[0]), float(p[1]), float(p[2])) for p in pts])
    volume = (distances <= 0.0).reshape((depth, height, width)).astype(np.uint8)

    global_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    return volume, z_coords, global_bounds


def volume_to_slice_indices(volume):
    slices = []
    for plane in volume:
        rr, cc = np.nonzero(plane)
        if rr.size == 0:
            slices.append(np.empty((2, 0), dtype=np.int32))
        else:
            slices.append(np.vstack((rr, cc)).astype(np.int32))
    return slices


def read_vtu(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def extract_geometry(mesh):
    return numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())


def serialize_vtk_object(vtk_object):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(vtk_object)
    writer.WriteToOutputStringOn()
    writer.Write()
    return writer.GetOutputString()


def deserialize_vtk_object(data_string):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(data_string)
    reader.Update()
    return reader.GetOutput()


def process_slice(
    serialized_mesh,
    z_slice,
    img_size,
    global_bounds,
    outlier_margin=1.0,
    debug=False,
):
    mesh = deserialize_vtk_object(serialized_mesh)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)

    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, z_slice)
    plane.SetNormal(0, 0, 1)
    cutter.SetCutFunction(plane)
    cutter.Update()

    slice_poly = cutter.GetOutput()

    if debug:
        print(f"\nProcessing slice at z={z_slice}")
        print(f"Number of points in slice: {slice_poly.GetNumberOfPoints()}")
        print(f"Number of cells in slice: {slice_poly.GetNumberOfCells()}")

    if slice_poly.GetNumberOfPoints() == 0:
        return np.empty((2, 0), dtype=np.int32)

    x_min, x_max, y_min, y_max = global_bounds

    x_min, x_max, y_min, y_max = x_min / 1.8, x_max / 1.8, y_min / 2.2, y_max / 2.2

    if debug:
        print(
            f"Global bounds used for normalization: x[{x_min}, {x_max}], y[{y_min}, {y_max}]"
        )

    points = numpy_support.vtk_to_numpy(slice_poly.GetPoints().GetData())
    cells = slice_poly.GetPolys()
    cell_conn = numpy_support.vtk_to_numpy(cells.GetConnectivityArray())
    cell_offsets = numpy_support.vtk_to_numpy(cells.GetOffsetsArray())
    cell_mode = "polygon"

    if cell_conn.size == 0 or cell_offsets.size <= 1:
        cells = slice_poly.GetLines()
        cell_conn = numpy_support.vtk_to_numpy(cells.GetConnectivityArray())
        cell_offsets = numpy_support.vtk_to_numpy(cells.GetOffsetsArray())
        cell_mode = "line"
        if debug:
            print(f"Falling back to line mode: lines={cells.GetNumberOfCells()}")

    if outlier_margin is not None and outlier_margin > 0:
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        keep_mask = (
            (points[:, 0] >= mean_x - outlier_margin)
            & (points[:, 0] <= mean_x + outlier_margin)
            & (points[:, 1] >= mean_y - outlier_margin)
            & (points[:, 1] <= mean_y + outlier_margin)
        )
    else:
        keep_mask = np.ones(points.shape[0], dtype=bool)

    valid_indices = np.nonzero(keep_mask)[0]
    if valid_indices.size == 0:
        return np.empty((2, 0), dtype=np.int32)

    valid_points = points[valid_indices]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

    valid_cell_conn = [index_map[idx] for idx in cell_conn if idx in index_map]

    if len(valid_cell_conn) == 0:
        return np.empty((2, 0), dtype=np.int32)

    valid_cell_conn = np.array(valid_cell_conn, dtype=np.int32)
    binary_image = np.zeros(img_size, dtype=np.uint8)

    for i in range(len(cell_offsets) - 1):
        start, end = cell_offsets[i], cell_offsets[i + 1]

        if end <= len(valid_cell_conn):
            poly_indices = valid_cell_conn[start:end]
            polygon_points = valid_points[poly_indices, :2]

            polygon_points[:, 0] = (
                (polygon_points[:, 0] - x_min) / (x_max - x_min) * (img_size[1] - 1)
            )
            polygon_points[:, 1] = (
                (polygon_points[:, 1] - y_min) / (y_max - y_min) * (img_size[0] - 1)
            )
            polygon_points = np.clip(
                polygon_points, 0, [img_size[1] - 1, img_size[0] - 1]
            )

            if cell_mode == "polygon" and polygon_points.shape[0] >= 3:
                rr, cc = polygon(
                    polygon_points[:, 1], polygon_points[:, 0], shape=img_size
                )
                binary_image[rr, cc] = 1
            elif cell_mode == "line" and polygon_points.shape[0] >= 2:
                p0, p1 = polygon_points[0], polygon_points[-1]
                rr, cc = line(
                    int(round(p0[1])),
                    int(round(p0[0])),
                    int(round(p1[1])),
                    int(round(p1[0])),
                )
                binary_image[rr, cc] = 1

    if cell_mode == "line":
        binary_image = binary_fill_holes(binary_image).astype(np.uint8)

    if debug:
        print(f"Slice at z={z_slice}: Non-zero elements = {binary_image.sum()}")

    indices = np.nonzero(binary_image)
    if indices[0].size == 0:
        return np.empty((2, 0), dtype=np.int32)
    return np.vstack(indices).astype(np.int32)


def slice_stack_vtu(mesh, num_slices, img_size, num_workers, debug, min_z=None, outlier_margin=1.0):
    points = extract_geometry(mesh)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    if min_z is not None:
        z_min = max(z_min, min_z)

    slice_positions = np.linspace(z_min, z_max, num_slices)

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    global_bounds = (x_min, x_max, y_min, y_max)

    if debug:
        print(
            f"Global bounds: x[{x_min}, {x_max}], y[{y_min}, {y_max}], z[{z_min}, {z_max}]"
        )

    serialized_mesh = serialize_vtk_object(mesh)
    process_args = [
        (serialized_mesh, z_slice, img_size, global_bounds, outlier_margin, debug)
        for z_slice in slice_positions
    ]

    if num_workers is None or num_workers <= 1:
        slices = [process_slice(*args) for args in process_args]
    else:
        with mp.Pool(processes=num_workers) as pool:
            slices = pool.starmap(process_slice, process_args)

    return slices, slice_positions, global_bounds


def plot_slice(indices, img_size, title):
    binary_image = np.zeros(img_size, dtype=np.uint8)
    if indices.shape[1] > 0:
        binary_image[indices[0], indices[1]] = 1
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image, cmap="binary")
    plt.title(title)
    plt.axis("off")
    plt.show()


def viz_stack(stack, inp_shape):
    # print(f"Num of slices in the stack: {len(stack)}")
    if not stack:
        print("Warning: Stack is empty.")
        return torch.tensor([])

    # Determine the maximum size for each slice to ensure consistent dimensions
    max_rows = inp_shape[0]
    max_cols = inp_shape[1]

    num_slices = len(stack)
    dense_stack = torch.zeros((num_slices, max_rows, max_cols), dtype=torch.uint8)

    for i, indices in enumerate(stack):
        if indices.shape[1] > 0:
            dense_stack[i, indices[0], indices[1]] = 1

    return dense_stack


def extract_and_save_blocks(voxel_grid, block_shape=(32, 128, 128), save_dir=None):
    grid_shape = voxel_grid.shape
    # print(f"Voxel grid shape: {grid_shape}")
    num_blocks = [grid_shape[i] // block_shape[i] for i in range(3)]

    # ensure the grid can be evenly divided into blocks
    for i in range(3):
        if grid_shape[i] % block_shape[i] != 0:
            raise ValueError(
                f"Dimension {i} of the grid ({grid_shape[i]}) is not divisible by the block size ({block_shape[i]})."
            )

    block_count = 0
    blocks = []
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            for k in range(num_blocks[2]):
                start_x = i * block_shape[0]
                end_x = start_x + block_shape[0]
                start_y = j * block_shape[1]
                end_y = start_y + block_shape[1]
                start_z = k * block_shape[2]
                end_z = start_z + block_shape[2]

                block = voxel_grid[start_x:end_x, start_y:end_y, start_z:end_z]

                blocks.append(block)
                block_count += 1

    with gzip.open(save_dir, "wb") as f:
        torch.save(blocks, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def _process_vtu_file(vtu_file, args, out_dir=None):
    if not os.path.exists(vtu_file):
        if args and args.debug:
            print(f"VTU file not found: {vtu_file}")
        return

    try:
        mesh = read_vtu(vtu_file)
    except Exception as e:
        if args and args.debug:
            print(f"Failed to read VTU file {vtu_file}: {e}")
        return

    if mesh is None:
        if args and args.debug:
            print(f"VTU file returned None: {vtu_file}")
        return

    if args and args.debug:
        print(f"\nProcessing VTU file: {vtu_file}")
        print(f"Mesh bounds: {mesh.GetBounds()}")
        print(f"Number of points in mesh: {mesh.GetNumberOfPoints()}")
        print(f"Number of cells in mesh: {mesh.GetNumberOfCells()}")

    target_dir = out_dir or (args.out_dir if args and args.out_dir else os.path.dirname(vtu_file))
    os.makedirs(target_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(vtu_file))[0]

    if args and args.voxel_mode == "implicit":
        depth = args.num_slices if args else 128
        height, width = tuple(args.img_size) if args else (512, 512)
        volume, slice_positions, global_bounds = voxelize_vtu_implicit(
            vtu_file,
            depth=depth,
            height=height,
            width=width,
            padding_frac=args.bbox_padding if args else 0.02,
        )
        slices = volume_to_slice_indices(volume)
        voxel_grid = torch.from_numpy(volume).to(torch.uint8)
    else:
        img_size = tuple(args.img_size) if args else (512, 512)
        slices, slice_positions, global_bounds = slice_stack_vtu(
            mesh,
            num_slices=args.num_slices if args else 128,
            img_size=img_size,
            num_workers=args.num_workers if args else 4,
            debug=args.debug if args else False,
            min_z=args.min_z if args else None,
            outlier_margin=args.outlier_margin if args else 1.0,
        )

        if args and args.debug:
            for i, indices in enumerate(slices):
                if indices.shape[1] > 0:
                    plot_slice(indices, img_size, f"Slice {i} at z={slice_positions[i]:.2f}")

        voxel_grid = viz_stack(slices, img_size)

    fname = f"slices_{stem}.pt.gz"
    with gzip.open(os.path.join(target_dir, fname), "wb") as f:
        torch.save(
            {
                "slices": [slice_indices for slice_indices in slices],
                "slice_positions": slice_positions,
                "global_bounds": global_bounds,
            },
            f,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

    if args and args.debug:
        total_non_zero = sum(slice_indices.shape[1] for slice_indices in slices)
        print(f"Saved {len(slices)} slices to {os.path.join(target_dir, fname)}")
        print(f"Total non-zero elements across all slices: {total_non_zero}")

    if args and args.save_voxel:
        voxels_dir = os.path.join(target_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)
        voxel_file_dir = os.path.join(voxels_dir, f"voxel_grid_{stem}.pt.gz")

        if voxel_grid.numel() == 0:
            print(f"Warning: Voxel grid for file '{vtu_file}' is empty. Skipping voxel saving.")
            return

        extract_and_save_blocks(voxel_grid, block_shape=(32, 64, 64), save_dir=voxel_file_dir)


def processor(root_dir, stem="gan_data_", args=None):
    dirs = [
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith(stem)
    ]
    for d in tqdm(dirs, desc="Processing directories"):
        vtu_file = os.path.join(root_dir, d, "DISPLAY", "tacts_2.vtu")
        out_dir = args.out_dir or os.path.join(root_dir, d)
        _process_vtu_file(vtu_file, args, out_dir=out_dir)


def main():
    """
    The voxelizer was run with the following arguments:
    python voxelizer.py --root_dir dataset/gan_data --num_slices 128  --img_size 512 512 --out_dir dataset/voxels_shrink --num_workers 40 --save_voxel

    The directory dataset/gan_data contains the .vtu files from the last time steps for the data samples.
    """

    args = parse_arguments()
    mp.freeze_support()  # needed on macos

    if args.vtu_file:
        default_out_dir = args.out_dir
        if default_out_dir is None:
            default_out_dir = os.path.join(os.path.dirname(args.vtu_file), "voxels")
        _process_vtu_file(args.vtu_file, args, out_dir=default_out_dir)
        return

    if not args.root_dir:
        raise ValueError("Either --root_dir or --vtu_file must be provided.")

    # validate output directory for batch mode
    if args.save_voxel and not args.out_dir:
        os.makedirs(os.path.join(args.root_dir, "voxels"), exist_ok=True)
        print("Error: --save_voxel requires --out_dir to specify where to save voxel blocks.")
        exit(1)
    print(f"Checking for the root directory: {args.root_dir}")
    processor(args.root_dir, stem=args.stem, args=args)


if __name__ == "__main__":
    main()
