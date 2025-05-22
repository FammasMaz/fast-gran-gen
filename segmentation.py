# WORK IN PROGRESS
# The Segmentation script uses many different methods right now. none of them are decisively better than the others.

import os
import time
import numpy as np
import pyvista as pv
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Set
import argparse


try:
    from skimage import measure as skimage_measure_global
    from skimage import segmentation as skimage_segmentation_global
    from skimage import feature as skimage_feature_global
except ImportError:
    skimage_measure_global = None
    skimage_segmentation_global = None
    skimage_feature_global = None
    print(
        "Warning: scikit-image (or some of its components) not found. Some features like scikit-image based marching cubes or watershed segmentation might be unavailable or fall back to alternatives."
    )


class VoxelToPolyhedron:
    def __init__(self, use_scikit_image_for_marching_cubes: bool = False):
        """
        Initialize the converter.

        Args:
            use_scikit_image_for_marching_cubes: If True, use scikit-image for marching cubes.
                                                 If False, use PyVista/VTK (default).
        """
        self.use_scikit_image_for_marching_cubes = use_scikit_image_for_marching_cubes

        # Assign globally imported skimage modules to instance variables
        # This allows methods to check for their availability.
        self.skimage_measure = skimage_measure_global
        self.skimage_segmentation = skimage_segmentation_global
        self.skimage_feature = skimage_feature_global

        if self.use_scikit_image_for_marching_cubes and not self.skimage_measure:
            print(
                "Warning: scikit-image for marching cubes was requested, but skimage.measure could not be imported. Falling back to PyVista for marching cubes."
            )
            self.use_scikit_image_for_marching_cubes = False

    def read_vti(self, filepath: str) -> Tuple[pv.ImageData, np.ndarray]:
        """
        Read a VTI file and return the grid and volume data.

        Args:
            filepath: Path to the VTI file

        Returns:
            grid: PyVista ImageData object
            volume: NumPy array containing the scalar field
        """
        print(f"Reading VTI file: {filepath}")

        # Read the VTI file using PyVista
        grid = pv.read(filepath)
        print(f"  Grid dimensions: {grid.dimensions}")
        print(f"  Grid spacing: {grid.spacing}")
        print(f"  Grid origin: {grid.origin}")
        print(f"  Grid n_points: {grid.n_points}, Grid n_cells: {grid.n_cells}")
        print(f"  Grid array names: {grid.array_names}")
        if len(grid.array_names) > 0:
            print(f"  Active scalars info: {grid.active_scalars_info}")

        # Get the scalar field (assumes the first scalar field is the one we want)
        initial_volume_data = None
        if "values" in grid.cell_data and grid.cell_data["values"] is not None:
            initial_volume_data = grid.cell_data["values"]
            print("  Using 'values' from cell_data.")
        elif len(grid.array_names) > 0:
            array_name = grid.array_names[0]
            print(f"  Attempting to use array: {array_name}")
            if array_name in grid.cell_data and grid.cell_data[array_name] is not None:
                initial_volume_data = grid.cell_data[array_name]
                print(f"  Using '{array_name}' from cell_data.")
            elif array_name in grid.point_data and grid.point_data[array_name] is not None:
                initial_volume_data = grid.point_data[array_name]
                print(f"  Using '{array_name}' from point_data.")

        if initial_volume_data is None and grid.active_scalars is not None:
            initial_volume_data = grid.active_scalars
            print(f"  Using active_scalars. Association: {grid.active_scalars_info.association}")
        elif initial_volume_data is None:
            raise ValueError("Could not retrieve scalar data from VTI file.")

        print(
            f"  Initial volume data shape: {initial_volume_data.shape}, min: {initial_volume_data.min()}, max: {initial_volume_data.max()}"
        )

        # Determine target shape based on data association
        # For ImageData, grid.dimensions gives the number of points along each axis.
        # Cell dimensions are typically (dim_x-1, dim_y-1, dim_z-1).
        target_shape_is_cell_dims = False
        if grid.active_scalars_info.association == pv.FieldAssociation.CELL:
            target_shape_is_cell_dims = True
            print("  Interpreting volume as cell-associated.")
        elif grid.active_scalars_info.association == pv.FieldAssociation.POINT:
            print(
                "  Interpreting volume as point-associated. This might require care in reshaping if cell-like processing is intended."
            )

        # Reshape the volume.
        expected_cell_elements = np.prod(np.array(grid.dimensions) - 1)
        if initial_volume_data.size == expected_cell_elements:
            volume = np.reshape(initial_volume_data, np.array(grid.dimensions[::-1]) - 1)
            print(f"  Reshaped to cell dimensions: {volume.shape}")
        elif initial_volume_data.size == grid.n_points:  # It's point data matching n_points
            print(
                f"  WARNING: Initial volume appears to be point data (size {initial_volume_data.size}). Current reshaping logic expects cell data size ({expected_cell_elements})."
            )
            volume = np.reshape(initial_volume_data, np.array(grid.dimensions[::-1]) - 1)
            print(f"  Attempted reshape to cell dimensions: {volume.shape}")
        else:
            raise ValueError(
                f"Volume data size {initial_volume_data.size} does not match expected cell elements ({expected_cell_elements}) or point elements ({grid.n_points}). Cannot reliably reshape."
            )

        return grid, volume

    def compute_sdf(self, binary: np.ndarray, scale: float = 5.0) -> np.ndarray:
        """
        Compute a Signed Distance Field (SDF) from a binary volume,
        with values clipped and scaled to [-1, 1].
        Positive values are outside, negative values are inside.

        Args:
            binary: Binary volume (0 = outside, 1 = inside)
            scale: Value for clipping and scaling. SDF will be in [-1, 1].

        Returns:
            sdf: Signed Distance Field (negative inside, positive outside, zero on boundary, scaled to [-1, 1])
        """
        print(f"Computing SDF (custom method with scale={scale})...")
        start_time = time.time()

        # Ensure binary is boolean for correct distance transform behavior
        binary_bool = binary.astype(bool)

        # Distance to the nearest True (inside) voxel from all voxels
        distance_inside = ndimage.distance_transform_edt(binary_bool)

        # Distance to the nearest False (outside) voxel from all voxels
        distance_outside = ndimage.distance_transform_edt(~binary_bool)

        # SDF: positive outside, negative inside
        sdf = distance_outside - distance_inside

        # Clip and scale to [-1, 1]
        sdf = np.clip(sdf, -scale, scale) / scale

        print(
            f"SDF computation (custom) completed in {time.time() - start_time:.2f} seconds. Min: {sdf.min()}, Max: {sdf.max()}"
        )
        return sdf.astype(np.float32)

    def label_connected_components(
        self, binary: np.ndarray, watershed_min_marker_distance: int = 5
    ) -> Tuple[np.ndarray, int]:
        """
        Segment a binary volume, primarily using watershed segmentation.
        Falls back to standard connected components labeling if scikit-image is unavailable.

        Args:
            binary: Binary volume (0 = outside, 1 = inside)
            watershed_min_marker_distance: Minimum distance between markers for watershed (voxels).

        Returns:
            labels: Volume with labeled regions (0 = background, 1...n = grains)
            num_labels: Number of distinct regions/grains found
        """

        can_do_watershed = self.skimage_segmentation and self.skimage_feature and ndimage

        if can_do_watershed:
            print(f"Attempting watershed segmentation (min_marker_distance={watershed_min_marker_distance})...")
            try:
                # Compute the distance from non-zero (i.e., True) points
                distance = ndimage.distance_transform_edt(binary)

                # Find local maxima of the distance map to use as markers
                footprint = np.ones((3, 3, 3), dtype=bool)
                local_maxi_mask = self.skimage_feature.peak_local_max(
                    distance,
                    footprint=footprint,
                    min_distance=watershed_min_marker_distance,
                    indices=False,  # Get a boolean array
                    exclude_border=False,  # Consider objects at the border
                )

                markers, num_markers = ndimage.label(local_maxi_mask, structure=footprint)

                if num_markers == 0:
                    print(
                        "  Watershed: No markers found. This might result in a single label or an empty label map. Check binary input and parameters (especially --watershed-min-marker-distance and --label-threshold, --pre-smoothing-sigma)."
                    )
                    if np.any(binary):
                        print(
                            "    Watershed: Input binary volume is not empty, but no markers generated. The entire foreground might be treated as one segment or segmentation may fail to separate regions."
                        )
                        # Fallback to labeling if no markers were found but there is binary data to label
                        print(
                            "    Watershed: Falling back to standard connected components labeling due to no markers."
                        )
                        can_do_watershed = False  # Force fallback
                    else:
                        # If binary is all False, num_markers will be 0, and labels should be all 0.
                        print("  Watershed: Input binary volume is empty. Result will be an empty label map.")
                        return np.zeros_like(binary, dtype=np.int32), 0

                if can_do_watershed:  # Re-check in case fallback was triggered above
                    print(f"  Watershed: Found {num_markers} markers.")
                    labels = self.skimage_segmentation.watershed(
                        -distance, markers, mask=binary, connectivity=footprint
                    )
                    num_labels = np.max(labels)
                    print(f"  Found {num_labels} distinct grain(s) using watershed.")
                    return labels.astype(np.int32), num_labels

            except Exception as e:
                print(f"Error during watershed segmentation: {e}. Falling back to standard labeling.")
                can_do_watershed = False  # Fallback on error

        # Fallback to standard labeling if watershed couldn't be done or failed
        print("Using standard connected components labeling (scipy.ndimage.label)...")
        structure = np.ones((3, 3, 3), dtype=np.bool_)  # 26-connectivity
        labels, num_labels = ndimage.label(binary, structure=structure)
        print(f"Found {num_labels} distinct grain(s) using standard labeling.")
        return labels.astype(np.int32), num_labels

    def extract_isosurface_pyvista(
        self, grid: pv.ImageData, scalar_field: np.ndarray, level: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface using PyVista/VTK.

        Args:
            grid: PyVista ImageData object
            scalar_field: NumPy array containing the scalar field.
                          This script prepares scalar_field to be cell-associated.
            level: Isosurface level (default: 0.0)

        Returns:
            verts: Array of vertex coordinates (N×3)
            faces: Array of triangular faces (M×3), each containing 3 vertex indices
        """
        temp_grid = grid.copy()

        print(
            f"    extract_isosurface_pyvista: Input scalar_field - shape: {scalar_field.shape}, min: {scalar_field.min()}, max: {scalar_field.max()}, level: {level}"
        )
        temp_grid.cell_data["scalar_field"] = np.ravel(scalar_field, order="F")

        print("    extract_isosurface_pyvista: Converting cell_data to point_data...")
        grid_for_contour = temp_grid.cell_data_to_point_data()
        print(
            f"    extract_isosurface_pyvista: Grid for contour - point data arrays: {grid_for_contour.point_data.keys()}"
        )
        if "scalar_field" in grid_for_contour.point_data:
            point_scalars = grid_for_contour.point_data["scalar_field"]
            print(
                f"    extract_isosurface_pyvista: Point scalars for contour - shape: {point_scalars.shape}, min: {point_scalars.min()}, max: {point_scalars.max()}"
            )
        else:
            print("    extract_isosurface_pyvista: 'scalar_field' not found in point_data after conversion!")

        print("    extract_isosurface_pyvista: Performing contour operation...")
        surface = grid_for_contour.contour([level], scalars="scalar_field")

        if surface.n_points == 0:
            print("Warning: Isosurface extraction resulted in an empty mesh")
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

        verts = surface.points

        vtk_faces = surface.faces
        if len(vtk_faces) == 0:
            print("Warning: No faces in extracted isosurface")
            return verts, np.empty((0, 3), dtype=np.int32)

        faces = []
        i = 0
        while i < len(vtk_faces):
            count = vtk_faces[i]
            assert count == 3, f"Expected triangles (count=3), got count={count}"
            faces.append(vtk_faces[i + 1 : i + 4])
            i += count + 1

        faces = np.array(faces, dtype=np.int32)

        return verts, faces

    def extract_isosurface_skimage(
        self,
        scalar_field: np.ndarray,
        level: float = 0.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface using scikit-image marching cubes.

        Args:
            scalar_field: NumPy array containing the scalar field
            level: Isosurface level (default: 0.0)
            spacing: Voxel spacing (dz, dy, dx)
            origin: Volume origin (oz, oy, ox)

        Returns:
            verts: Array of vertex coordinates (N×3)
            faces: Array of triangular faces (M×3), each containing 3 vertex indices
        """
        if not self.use_scikit_image_for_marching_cubes or not self.skimage_measure:
            raise ImportError(
                "scikit-image marching cubes requested or is a dependency, but skimage.measure is not available."
            )

        try:
            verts, faces, normals, values = self.skimage_measure.marching_cubes(
                scalar_field, level=level, spacing=spacing
            )

            verts += np.array(origin)

            return verts, faces
        except Exception as e:
            print(f"Error in marching cubes: {e}")
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    def build_edge_list(self, faces: np.ndarray) -> np.ndarray:
        """
        Build a list of edges from triangular faces.

        Args:
            faces: Array of triangular faces (M×3), each containing 3 vertex indices

        Returns:
            edges: Array of edges (E×2), each containing 2 vertex indices
        """
        edges = set()

        for tri in faces:
            i, j, k = tri

            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))

        edge_list = np.array(list(edges), dtype=np.int32)

        return edge_list

    def deduplicate_vertices(
        self, verts: np.ndarray, edges: np.ndarray, faces: np.ndarray, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deduplicate vertices and update edges and faces accordingly.

        Args:
            verts: Array of vertex coordinates (N×3)
            edges: Array of edges (E×2)
            faces: Array of faces (F×3)
            tol: Tolerance for considering vertices as duplicates

        Returns:
            new_verts: Deduplicated vertices
            new_edges: Updated edges
            new_faces: Updated faces
        """
        if len(verts) == 0:
            return verts, edges, faces

        print("Deduplicating vertices...")
        start_time = time.time()

        decimal_places = int(-np.log10(tol))
        rounded = np.round(verts, decimal_places)

        unique_verts, inverse_indices = np.unique(rounded, axis=0, return_inverse=True)

        new_edges = inverse_indices[edges]
        valid_edges = new_edges[:, 0] != new_edges[:, 1]
        new_edges = new_edges[valid_edges]

        new_faces = inverse_indices[faces]
        valid_faces_mask = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[valid_faces_mask]

        print(f"Reduced from {len(verts)} to {len(unique_verts)} vertices")
        print(f"Edges reduced from {len(edges)} to {len(new_edges)}")
        print(f"Faces reduced from {len(faces)} to {len(new_faces)}")
        print(f"Deduplication completed in {time.time() - start_time:.2f} seconds")

        return unique_verts, new_edges, new_faces

    def extract_polyhedron(
        self,
        grid: pv.ImageData,
        volume: np.ndarray,
        compute_sdf_flag: bool = True,
        level: float = 0.0,  # Renamed to avoid clash
        isosurface_level_non_sdf: float = 0.5,
        sdf_scale_param: float = 5.0,
    ) -> Dict:  # New param for sdf scale
        """
        Extract a polyhedron from a volume.

        Args:
            grid: PyVista ImageData object
            volume: NumPy array containing the scalar field
            compute_sdf_flag: If True, compute SDF. If volume is not binary, it's thresholded at 0.5 first.
            level: Isosurface level, primarily used for SDFs (typically 0.0).
            isosurface_level_non_sdf: Isosurface level to use if compute_sdf_flag is False.
            sdf_scale_param: Scale parameter for SDF normalization if compute_sdf_flag is True.

        Returns:
            Dictionary containing vertices and edges
        """
        is_binary = np.array_equal(volume, volume.astype(bool))
        print(f"  extract_polyhedron: Initial volume for this grain/object is_binary? {is_binary}")

        processed_scalar_field: np.ndarray
        contour_level: float

        if compute_sdf_flag:
            print("  extract_polyhedron: --compute-sdf flag is active for this grain/object.")
            contour_level = level  # Use the SDF-specific level (typically 0.0, from sdf_contour_level arg)
            binary_volume_for_sdf: np.ndarray
            if not is_binary:
                threshold = 0.5
                print(
                    f"  extract_polyhedron: Volume is not binary for SDF. Thresholding at {threshold} to create binary mask for SDF computation."
                )
                binary_volume_for_sdf = volume > threshold
                if not binary_volume_for_sdf.any():
                    print(
                        "  WARNING: Thresholding at 0.5 for SDF resulted in an all-false mask. SDF might be meaningless or all one sign."
                    )
                elif binary_volume_for_sdf.all():
                    print(
                        "  WARNING: Thresholding at 0.5 for SDF resulted in an all-true mask. SDF might be meaningless or all one sign."
                    )
            else:  # Volume is already binary
                print("  extract_polyhedron: Volume is binary. Using it directly for SDF computation.")
                binary_volume_for_sdf = volume.astype(bool)

            print(
                f"  extract_polyhedron: Computing SDF from binary mask (shape: {binary_volume_for_sdf.shape}, type: {binary_volume_for_sdf.dtype})."
            )
            processed_scalar_field = self.compute_sdf(binary_volume_for_sdf, scale=sdf_scale_param)
            print(
                f"  extract_polyhedron: SDF computed. Min: {processed_scalar_field.min()}, Max: {processed_scalar_field.max()}"
            )
            print(f"  extract_polyhedron: Using contour level {contour_level} for SDF.")

        else:  # compute_sdf_flag is False
            print("  extract_polyhedron: --compute-sdf flag is NOT active. Using raw volume as scalar field.")
            processed_scalar_field = volume
            contour_level = isosurface_level_non_sdf  # Use the non-SDF specific level
            # Warning if using level 0.0 for non-SDF data was here, now relying on appropriate default for isosurface_level_non_sdf
            print(f"  extract_polyhedron: Using contour level {contour_level} for raw volume.")

        print(
            f"  extract_polyhedron: Final scalar field for isosurface extraction - shape: {processed_scalar_field.shape}, min: {processed_scalar_field.min()}, max: {processed_scalar_field.max()}, mean: {processed_scalar_field.mean()}"
        )

        # Extract isosurface
        spacing = grid.spacing
        origin = grid.origin

        if self.use_scikit_image_for_marching_cubes and self.skimage_measure:
            # Use scikit-image for marching cubes
            verts, faces = self.extract_isosurface_skimage(
                processed_scalar_field, contour_level, spacing[::-1], origin[::-1]
            )
        else:
            # Use PyVista/VTK for marching cubes
            verts, faces = self.extract_isosurface_pyvista(grid, processed_scalar_field, contour_level)

            # Convert to world coordinates (if necessary)
            # This is not needed for PyVista as it already works in world coordinates

        # Build edge list
        edges = self.build_edge_list(faces)

        # Deduplicate vertices and update edges and faces
        verts, edges, faces = self.deduplicate_vertices(verts, edges, faces)

        return {
            "vertices": verts,
            "edges": edges,
            "faces": faces,  # Optional, in case you want triangular faces too
        }

    def process_vti_file(
        self,
        filepath: str,
        compute_sdf_flag: bool = True,  # Renamed to avoid clash with method
        separate_grains: bool = True,
        label_threshold: float = 0.5,
        isosurface_level_non_sdf: float = 0.5,
        filter_boundary_grains: bool = False,
        sdf_contour_level: float = 0.0,
        sdf_scale_param: float = 5.0,
        boundary_filter_voxel_margin: float = 1.5,
        watershed_min_marker_distance: int = 5,  # New param
        pre_smoothing_sigma: float = 0.0,  # New param for Gaussian smoothing
    ) -> Dict:  # New parameter for sdf scale
        """
        Process a VTI file and extract polyhedron(s).

        Args:
            filepath: Path to the VTI file
            compute_sdf_flag: If True, compute SDF from binary volume (or binarized version).
            separate_grains: If True, extract each grain separately after binarization if needed.
            label_threshold: Threshold used to binarize the input volume for grain labeling if separate_grains is True and volume is not binary.
            isosurface_level_non_sdf: Isosurface level to use when --compute-sdf is NOT active (default: 0.5).
            filter_boundary_grains: If True, remove grains that touch the boundary of the VTI volume.
            sdf_contour_level: Isosurface level to use when --compute-sdf IS active (default: 0.0). Try small positive values to shrink/separate.
            sdf_scale_param: Scale parameter for SDF normalization (default: 5.0). Smaller values (e.g., 1-2) make SDF gradient steeper near boundary.
            boundary_filter_voxel_margin: Number of mean voxel spacings to define the margin for boundary grain filtering (default: 1.5). Grain is filtered if any vertex is within this margin of VTI edge.
            watershed_min_marker_distance: Min distance between markers for watershed (voxels).
            pre_smoothing_sigma: Sigma for Gaussian pre-smoothing (0 for no smoothing).

        Returns:
            Dictionary of grains, each containing vertices and edges
        """
        grid, initial_volume = self.read_vti(filepath)  # Renamed to initial_volume to avoid conflict

        volume = initial_volume.copy()  # Work on a copy
        if pre_smoothing_sigma > 0:
            print(f"  Applying Gaussian pre-smoothing with sigma={pre_smoothing_sigma}...")
            volume = ndimage.gaussian_filter(volume, sigma=pre_smoothing_sigma)
            print(
                f"    Volume after smoothing: Min: {volume.min():.4f}, Max: {volume.max():.4f}, Mean: {volume.mean():.4f}"
            )

        original_is_binary = np.array_equal(
            volume, volume.astype(bool)
        )  # Check binary status *after* potential smoothing
        print(
            f"  process_vti_file: Volume for processing is_binary? {original_is_binary}. Min: {volume.min():.4f}, Max: {volume.max():.4f}, Mean: {volume.mean():.4f}"
        )

        grid_bounds = np.array(grid.bounds)  # [xmin, xmax, ymin, ymax, zmin, zmax]
        tolerance = 1e-5  # General numerical tolerance

        mean_voxel_spacing = np.mean(grid.spacing)
        boundary_check_margin = (
            mean_voxel_spacing * boundary_filter_voxel_margin
        )  # Using boundary_filter_voxel_margin for this, effectively boundary_filter_margin_voxels
        print(
            f"  process_vti_file: Boundary filter margin set to {boundary_check_margin:.4f} (based on {boundary_filter_voxel_margin} * mean voxel spacing {mean_voxel_spacing:.4f})"
        )

        result = {}
        num_labels = 0  # Initialize num_labels

        if separate_grains:
            print("  process_vti_file: --separate-grains flag is active.")
            binary_volume_for_labels: np.ndarray
            if not original_is_binary:
                # Use the provided label_threshold for binarization
                print(
                    f"  process_vti_file: Original volume not binary. Thresholding at {label_threshold} for grain labeling."
                )
                binary_volume_for_labels = volume > label_threshold
                print(
                    f"    process_vti_file: After thresholding for labeling (threshold={label_threshold}) - True voxels: {np.sum(binary_volume_for_labels)}, False voxels: {np.sum(~binary_volume_for_labels)}, Shape: {binary_volume_for_labels.shape}"
                )
                if not binary_volume_for_labels.any():
                    print(
                        "    WARNING: Thresholding for grain labeling resulted in an all-false mask. No grains will be found."
                    )
                elif binary_volume_for_labels.all():
                    print(
                        "    WARNING: Thresholding for grain labeling resulted in an all-true mask. May be treated as one large grain if not further processed."
                    )
            else:
                print("  process_vti_file: Original volume is binary. Using it directly for grain labeling.")
                binary_volume_for_labels = volume.astype(bool)

            # Label connected components
            labels, num_labels = self.label_connected_components(
                binary_volume_for_labels, watershed_min_marker_distance=watershed_min_marker_distance
            )
            print(f"  process_vti_file: Found {num_labels} distinct grain(s) after segmentation.")

            if num_labels == 0:
                print(
                    "  process_vti_file: No grains identified by segmentation. No individual polyhedra will be extracted."
                )
                # If no labels, result remains empty.

            # Process each grain separately
            for grain_id in range(1, num_labels + 1):
                print(f"  process_vti_file: Processing labeled grain {grain_id}/{num_labels}...")

                # Create a binary mask for this grain
                grain_mask = labels == grain_id

                volume_for_polyhedron_extraction = grain_mask.astype(np.int8)  # Convert boolean to int (0 or 1)

                # Extract polyhedron for this grain.
                grain_result = self.extract_polyhedron(
                    grid,
                    volume_for_polyhedron_extraction,
                    compute_sdf_flag,
                    level=sdf_contour_level,
                    isosurface_level_non_sdf=isosurface_level_non_sdf,
                    sdf_scale_param=sdf_scale_param,
                )  # Pass sdf_scale_param

                # Store result only if polyhderon was successfully extracted (e.g., non-empty vertices/edges)
                if grain_result and grain_result.get("vertices", np.array([])).size > 0:
                    # Boundary grain filtering
                    if filter_boundary_grains:
                        is_boundary_grain = False
                        grain_vertices = grain_result["vertices"]
                        for vertex in grain_vertices:
                            # Check if vertex is within the 'boundary_check_margin' of any VTI boundary plane
                            if (
                                vertex[0] < grid_bounds[0] + boundary_check_margin - tolerance
                                or vertex[0] > grid_bounds[1] - boundary_check_margin + tolerance
                                or vertex[1] < grid_bounds[2] + boundary_check_margin - tolerance
                                or vertex[1] > grid_bounds[3] - boundary_check_margin + tolerance
                                or vertex[2] < grid_bounds[4] + boundary_check_margin - tolerance
                                or vertex[2] > grid_bounds[5] - boundary_check_margin + tolerance
                            ):
                                is_boundary_grain = True
                                break
                        if is_boundary_grain:
                            print(
                                f"    INFO: Grain {grain_id} identified as a boundary grain and will be filtered out."
                            )
                        else:
                            result[grain_id] = grain_result  # Store if not a boundary grain
                    else:
                        result[grain_id] = grain_result  # Store if filtering is off
                else:
                    print(
                        f"    WARNING: No polyhedron extracted for labeled grain {grain_id}. It might be too small or result in an empty mesh."
                    )
        else:
            # Process the entire volume as a single grain
            print(
                "  process_vti_file: --separate-grains flag is NOT active. Processing entire volume as a single object..."
            )

            volume_for_polyhedron_extraction = volume  # Use original volume if not separating
            if not compute_sdf_flag and np.issubdtype(volume.dtype, np.bool_):
                volume_for_polyhedron_extraction = volume.astype(np.int8)

            single_result = self.extract_polyhedron(
                grid,
                volume_for_polyhedron_extraction,
                compute_sdf_flag,
                level=sdf_contour_level,
                isosurface_level_non_sdf=isosurface_level_non_sdf,
                sdf_scale_param=sdf_scale_param,
            )  # Pass sdf_scale_param
            if single_result and single_result.get("vertices", np.array([])).size > 0:
                # Boundary check for single grain (less common to filter, but consistent)
                if filter_boundary_grains:
                    is_boundary_grain = False
                    grain_vertices = single_result["vertices"]
                    for vertex in grain_vertices:
                        # Check if vertex is within the 'boundary_check_margin' of any VTI boundary plane
                        if (
                            vertex[0] < grid_bounds[0] + boundary_check_margin - tolerance
                            or vertex[0] > grid_bounds[1] - boundary_check_margin + tolerance
                            or vertex[1] < grid_bounds[2] + boundary_check_margin - tolerance
                            or vertex[1] > grid_bounds[3] - boundary_check_margin + tolerance
                            or vertex[2] < grid_bounds[4] + boundary_check_margin - tolerance
                            or vertex[2] > grid_bounds[5] - boundary_check_margin + tolerance
                        ):
                            is_boundary_grain = True
                            break
                        if is_boundary_grain:
                            print(
                                "    INFO: The single processed object identified as a boundary grain and will be filtered out."
                            )
                    else:
                        result[1] = single_result
                else:
                    result[1] = single_result
            else:
                print("    WARNING: No polyhedron extracted for the single object. It might result in an empty mesh.")

        num_polyhedra_made = len(result)
        print(f"INFO: Total number of polyhedra successfully extracted and stored: {num_polyhedra_made}")
        if separate_grains and num_labels > 0 and num_polyhedra_made < num_labels:
            print(
                f"  NOTE: Labeling found {num_labels} grains, but only {num_polyhedra_made} resulted in extractable polyhedra."
            )

        return result

    def save_results(self, results: Dict, output_dir: str, base_filename: str, format: str = "npz") -> None:
        """
        Save results to disk.

        Args:
            results: Dictionary of grains, each containing vertices and edges
            output_dir: Directory to save results
            base_filename: Base filename for saved files
            format: Format to save in ('npz', 'json', 'obj', 'vtm')
        """
        os.makedirs(output_dir, exist_ok=True)

        if format == "npz":
            # Save as NumPy arrays (.npz)
            for grain_id, grain in results.items():
                filename = os.path.join(output_dir, f"{base_filename}_grain_{grain_id}.npz")
                np.savez(filename, vertices=grain["vertices"], edges=grain["edges"])
                print(f"Saved grain {grain_id} to {filename}")

        elif format == "json":
            # Save as JSON
            import json

            for grain_id, grain in results.items():
                filename = os.path.join(output_dir, f"{base_filename}_grain_{grain_id}.json")
                with open(filename, "w") as f:
                    json.dump({"vertices": grain["vertices"].tolist(), "edges": grain["edges"].tolist()}, f)
                print(f"Saved grain {grain_id} to {filename}")

        elif format == "obj":
            # Save as OBJ files (vertices and edges only, no faces)
            for grain_id, grain in results.items():
                filename = os.path.join(output_dir, f"{base_filename}_grain_{grain_id}.obj")

                vertices = grain["vertices"]
                edges = grain["edges"]

                with open(filename, "w") as f:
                    # Write vertices
                    for v in vertices:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                    # Write edges (1-indexed in OBJ format)
                    for e in edges:
                        f.write(f"l {e[0] + 1} {e[1] + 1}\n")

                print(f"Saved grain {grain_id} to {filename}")

        elif format == "vtm":
            # Save all grains as a single VTK MultiBlock file (.vtm) for ParaView
            if not results:
                print("No results to save in VTM format.")
                return

            output_filepath = os.path.join(output_dir, f"{base_filename}.vtm")
            blocks = pv.MultiBlock()

            for grain_id, grain_data in results.items():
                vertices = grain_data.get("vertices")
                edges = grain_data.get("edges")
                # Optional: include faces if you want to save surfaces too
                faces_data = grain_data.get("faces")

                if vertices is None or vertices.size == 0:
                    print(f"Skipping grain {grain_id} for VTM (no vertices).")
                    continue

                grain_polydata = pv.PolyData()
                grain_polydata.points = vertices

                if edges is not None and edges.size > 0:
                    # Convert edges to VTK line cells
                    line_cells = []
                    for edge in edges:
                        line_cells.extend([2, edge[0], edge[1]])  # [num_points_in_cell, p1_idx, p2_idx]
                    # Storing lines as well, in case user wants wireframe in ParaView
                    # grain_polydata.lines = line_cells

                # If you have triangular faces and want to save them as surfaces:
                if faces_data is not None and faces_data.size > 0:
                    pv_faces = []
                    for face_tri in faces_data:
                        pv_faces.extend([3, face_tri[0], face_tri[1], face_tri[2]])
                    grain_polydata.faces = pv_faces
                elif edges is not None and edges.size > 0:  # Fallback to saving lines if no faces
                    line_cells = []
                    for edge in edges:
                        line_cells.extend([2, edge[0], edge[1]])
                    grain_polydata.lines = line_cells
                else:
                    print(f"Skipping grain {grain_id} for VTM: no faces or edges.")
                    continue  # Don't add empty polydata

                blocks.append(grain_polydata, name=f"Grain_{grain_id}")

            if blocks.n_blocks > 0:
                blocks.save(output_filepath)
                print(f"Saved all {blocks.n_blocks} grains to MultiBlock VTM file: {output_filepath}")
            else:
                print("No valid grains were added to the VTM file.")

        else:
            print(f"Unsupported format: {format}")

    def visualize_result(self, results: Dict, grain_id: int = 1, ax=None, show=True, hide_nodes: bool = False) -> None:
        """
        Visualize a polyhedron.

        Args:
            results: Dictionary of grains, each containing vertices and edges
            grain_id: ID of the grain to visualize
            ax: Matplotlib axes to plot on (optional)
            show: Whether to show the plot
            hide_nodes: If True, do not plot the vertices (nodes).
        """
        if grain_id not in results:
            print(f"Grain {grain_id} not found in results")
            return

        grain = results[grain_id]
        vertices = grain["vertices"]
        edges = grain["edges"]

        if len(vertices) == 0 or len(edges) == 0:
            print(f"No vertices or edges for grain {grain_id}")
            return

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        if not hide_nodes:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="b", marker="o", alpha=0.5)

        for edge in edges:
            i, j = edge
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                "k-",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Grain {grain_id} Polyhedron")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        max_range = (
            np.max([np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]), np.ptp(vertices[:, 2])])
            if vertices.size > 0
            else 1.0
        )  # Handle empty vertices

        mid_x = np.mean(vertices[:, 0]) if vertices.size > 0 else 0.0
        mid_y = np.mean(vertices[:, 1]) if vertices.size > 0 else 0.0
        mid_z = np.mean(vertices[:, 2]) if vertices.size > 0 else 0.0

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        if show:
            plt.tight_layout()
            plt.show()

    def visualize_all_grains_matplotlib(self, all_results: Dict, hide_nodes: bool = False) -> None:
        """
        Visualize all extracted polyhedra (grains) in a single Matplotlib plot.

        Args:
            all_results: Dictionary of all grains, each containing vertices and edges.
            hide_nodes: If True, do not plot the vertices (nodes).
        """
        if not all_results:
            print("No results to visualize.")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("All Extracted Grain Polyhedra")

        all_verts_list = []
        for grain_id, grain_data in all_results.items():
            vertices = grain_data.get("vertices")
            edges = grain_data.get("edges")

            if vertices is None or edges is None or vertices.size == 0:
                print(f"Skipping grain {grain_id} in combined plot (no vertices/edges).")
                continue

            all_verts_list.append(vertices)

            if not hide_nodes:
                ax.scatter(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2], marker="o", alpha=0.5, s=10
                )  # Smaller points for combined plot

            # If faces are available, plot them as a surface
            faces = grain_data.get("faces")
            if faces is not None and faces.size > 0:
                ax.plot_trisurf(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    triangles=faces,
                    color="lightblue",
                    alpha=0.7,
                    edgecolor="k",
                    linewidth=0.2,
                )
            else:  # Fallback to plotting edges if no faces
                for edge in edges:
                    i, j = edge
                    ax.plot(
                        [vertices[i, 0], vertices[j, 0]],
                        [vertices[i, 1], vertices[j, 1]],
                        [vertices[i, 2], vertices[j, 2]],
                        "k-",
                        linewidth=0.5,
                    )

        if not all_verts_list:
            print("No valid grains with vertices found for combined plot.")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()
            return

        # Determine overall bounds for all grains
        global_vertices = np.concatenate(all_verts_list, axis=0)
        min_coords = np.min(global_vertices, axis=0)
        max_coords = np.max(global_vertices, axis=0)
        mid_coords = (min_coords + max_coords) / 2
        max_range = np.max(max_coords - min_coords) * 1.1  # Add some padding
        if max_range == 0:
            max_range = 1.0  # Avoid zero range

        ax.set_xlim(mid_coords[0] - max_range / 2, mid_coords[0] + max_range / 2)
        ax.set_ylim(mid_coords[1] - max_range / 2, mid_coords[1] + max_range / 2)
        ax.set_zlim(mid_coords[2] - max_range / 2, mid_coords[2] + max_range / 2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.tight_layout()
        plt.show()

    def visualize_pyvista(self, results: Dict, grain_id: int = 1, hide_nodes: bool = False) -> None:
        """
        Visualize a polyhedron using PyVista.

        Args:
            results: Dictionary of grains, each containing vertices and edges
            grain_id: ID of the grain to visualize
            hide_nodes: If True, do not plot the vertices (nodes).
        """
        if grain_id not in results:
            print(f"Grain {grain_id} not found in results")
            return

        grain = results[grain_id]
        vertices = grain["vertices"]
        edges = grain["edges"]

        if len(vertices) == 0 or len(edges) == 0:
            print(f"No vertices or edges for grain {grain_id}")
            return

        point_cloud = pv.PolyData(vertices)

        lines = pv.PolyData()
        lines.points = vertices

        cells = []
        for edge in edges:
            cells.extend([2, edge[0], edge[1]])

        lines.lines = cells

        plotter = pv.Plotter()

        if not hide_nodes:
            point_cloud = pv.PolyData(vertices)  # Create only if needed
        plotter.add_mesh(point_cloud, color="blue", point_size=10, render_points_as_spheres=True)
        plotter.add_mesh(lines, color="black", line_width=2)

        plotter.show()

    def visualize_all_grains_pyvista(self, all_results: Dict, hide_nodes: bool = False) -> None:
        """
        Visualize all extracted polyhedra (grains) in a single PyVista scene.

        Args:
            all_results: Dictionary of all grains, each containing vertices and edges.
            hide_nodes: If True, do not plot the vertices (nodes).
        """
        if not all_results:
            print("No results to visualize.")
            return

        plotter = pv.Plotter(window_size=[800, 600])
        plotter.background_color = "white"  # Example: set background color

        any_grain_plotted = False
        for grain_id, grain_data in all_results.items():
            vertices = grain_data.get("vertices")
            edges = grain_data.get("edges")

            if vertices is None or edges is None or vertices.size == 0 or edges.size == 0:
                print(f"Skipping grain {grain_id} in combined PyVista plot (no vertices/edges).")
                continue

            any_grain_plotted = True

            # Create a PolyData object for the edges of this grain
            grain_polydata = pv.PolyData()
            grain_polydata.points = vertices

            faces = grain_data.get("faces")
            if faces is not None and faces.size > 0:
                # For PyVista, faces need to be in a specific format: [3, idx0, idx1, idx2, 3, idx0, idx1, idx2, ...]
                pv_faces = []
                for face in faces:
                    pv_faces.extend([3, face[0], face[1], face[2]])
                grain_polydata.faces = pv_faces
                plotter.add_mesh(
                    grain_polydata, color="lightblue", show_edges=True, edge_color="black", line_width=0.5, opacity=0.7
                )
            elif edges is not None and edges.size > 0:  # Fallback to lines if no faces
                cells = []
                for edge in edges:
                    cells.extend([2, edge[0], edge[1]])
                grain_polydata.lines = cells
                plotter.add_mesh(grain_polydata, color="black", line_width=1)
            else:
                print(f"Skipping grain {grain_id} in PyVista plot: no faces or edges.")
                continue  # Skip to next grain if no geometry

            if not hide_nodes:
                points_polydata = pv.PolyData(vertices)
                plotter.add_mesh(
                    points_polydata, color="blue", point_size=5, render_points_as_spheres=True
                )  # Smaller points

        if not any_grain_plotted:
            print("No valid grains found to plot in combined PyVista scene.")
            return  # Don't show an empty plotter

        plotter.enable_zoom_style()  # useful for multi-object scenes
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Convert VTI voxel grids to polyhedra")
    parser.add_argument("input_file", help="Input VTI file")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--format",
        "-f",
        choices=["npz", "json", "obj", "vtm"],
        default="npz",
        help="Output format. 'vtm' for ParaView MultiBlock file.",
    )
    parser.add_argument(
        "--compute-sdf",
        "-s",
        dest="compute_sdf_flag",
        action="store_true",
        help="Compute SDF. If volume is not binary, it's thresholded at 0.5 for SDF calculation. SDFs are contoured at level 0.0.",
    )
    parser.add_argument(
        "--no-compute-sdf",
        dest="compute_sdf_flag",
        action="store_false",
        help="Explicitly disable SDF computation (uses raw data with --isosurface-level).",
    )
    parser.set_defaults(
        compute_sdf_flag=False
    )  # Default to False if neither is specified, or choose a sensible default like True
    parser.add_argument(
        "--isosurface-level",
        type=float,
        default=0.5,
        help="Isosurface level to use when --compute-sdf is NOT active (default: 0.5).",
    )
    parser.add_argument(
        "--sdf-contour-level",
        type=float,
        default=0.0,
        help="Isosurface level to use for SDFs when --compute-sdf IS active (default: 0.0). Try small positive values to shrink/separate.",
    )
    parser.add_argument(
        "--sdf-scale",
        type=float,
        default=5.0,
        help="Scale parameter for SDF normalization (default: 5.0). Smaller values (e.g., 1-2) make SDF gradient steeper near boundary.",
    )
    parser.add_argument(
        "--boundary-filter-voxel-margin",
        type=float,
        default=1.5,
        help="Number of mean voxel spacings to define the margin for boundary grain filtering (default: 1.5). Grain is filtered if any vertex is within this margin of VTI edge.",
    )
    parser.add_argument(
        "--separate-grains",
        "-g",
        action="store_true",
        help="Extract each grain separately using watershed segmentation (falls back to simple labeling if scikit-image is unavailable). Binarizes input for segmentation using --label-threshold if not already binary.",
    )
    parser.add_argument(
        "--watershed-min-marker-distance",
        type=int,
        default=5,
        help="Minimum distance (in voxels) between markers for watershed segmentation (default: 5).",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing the input volume for grain labeling if --separate-grains is used and input is not binary (default: 0.5).",
    )
    parser.add_argument(
        "--pre-smoothing-sigma", type=float, default=0.0, help="Sigma for Gaussian pre-smoothing (0 for no smoothing)."
    )
    parser.add_argument(
        "--use-scikit-image",
        "-i",
        action="store_true",
        help="Use scikit-image for marching cubes instead of PyVista/VTK.",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize results using Matplotlib (one plot per grain)."
    )
    parser.add_argument(
        "--pyvista-viz",
        "-p",
        action="store_true",
        help="Use PyVista for visualization (more interactive, one plot per grain).",
    )
    parser.add_argument(
        "--hide-visualization-nodes",
        action="store_true",
        help="If visualizing, hide the nodes (vertices) and only show edges.",
    )
    parser.add_argument(
        "--filter-boundary-grains",
        action="store_true",
        help="Filter out grains that touch the boundary of the VTI volume.",
    )
    parser.add_argument(
        "--visualize-all", action="store_true", help="Visualize all extracted grains together using Matplotlib."
    )
    parser.add_argument(
        "--pyvista-viz-all", action="store_true", help="Visualize all extracted grains together using PyVista."
    )

    args = parser.parse_args()

    converter = VoxelToPolyhedron(use_scikit_image_for_marching_cubes=args.use_scikit_image)

    input_file = args.input_file
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    results = converter.process_vti_file(
        input_file,
        compute_sdf_flag=args.compute_sdf_flag,
        separate_grains=args.separate_grains,
        label_threshold=args.label_threshold,
        isosurface_level_non_sdf=args.isosurface_level,
        filter_boundary_grains=args.filter_boundary_grains,
        sdf_contour_level=args.sdf_contour_level,
        sdf_scale_param=args.sdf_scale,
        boundary_filter_voxel_margin=args.boundary_filter_voxel_margin,
        watershed_min_marker_distance=args.watershed_min_marker_distance,
        pre_smoothing_sigma=args.pre_smoothing_sigma,
    )

    converter.save_results(results, args.output_dir, base_filename, format=args.format)

    if args.visualize:
        for grain_id in results.keys():
            converter.visualize_result(results, grain_id, hide_nodes=args.hide_visualization_nodes)

    if args.pyvista_viz:
        for grain_id in results.keys():
            converter.visualize_pyvista(results, grain_id, hide_nodes=args.hide_visualization_nodes)

    if args.visualize_all:
        converter.visualize_all_grains_matplotlib(results, hide_nodes=args.hide_visualization_nodes)

    if args.pyvista_viz_all:
        converter.visualize_all_grains_pyvista(results, hide_nodes=args.hide_visualization_nodes)


if __name__ == "__main__":
    main()
