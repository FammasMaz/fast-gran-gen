import json
import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from tqdm import tqdm
import concurrent.futures
import warnings


# Helper function for multiprocessing
def _global_mesh_task(segmentation_instance, labeled_grid, label_id, smoothing_iterations, decimation_ratio, use_sdf):
    polyhedron_size = np.sum(labeled_grid == label_id)
    mesh_data = segmentation_instance.extract_polyhedron_mesh(
        labeled_grid, label_id, smoothing_iterations, decimation_ratio, use_sdf
    )
    return label_id, polyhedron_size, mesh_data


class PolyhedronSegmentation:
    """
    Advanced polyhedron segmentation for long voxel grids using watershed.
    Extracts individual polyhedrons and outputs them with global coordinates.
    """

    def __init__(
        self,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Initialize the segmentation module.

        Args:
            voxel_spacing: Physical spacing between voxels (dx, dy, dz)
            origin: Origin point of the voxel grid in world coordinates
        """
        self.voxel_spacing = np.array(voxel_spacing)
        self.origin = np.array(origin)

    def compute_sdf(self, binary_voxel: np.ndarray, scale: float = 5.0) -> np.ndarray:
        """
        Compute the signed distance field (SDF) of a binary voxel grid.

        The SDF provides smooth distance information that's crucial for proper
        watershed segmentation. Positive values are inside objects, negative outside.

        Args:
            binary_voxel: Binary voxel grid (True for object, False for background)
            scale: Scaling factor for the SDF values (larger = smoother gradients)

        Returns:
            SDF array where positive values are inside objects, negative outside
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Distance from background (outside)
            distance_outside = ndimage.distance_transform_edt(binary_voxel == 0)
            # Distance from foreground (inside)
            distance_inside = ndimage.distance_transform_edt(binary_voxel == 1)

        # Create signed distance field: positive inside, negative outside
        sdf = distance_inside - distance_outside

        # Apply scaling and clipping for numerical stability
        sdf = np.clip(sdf, -scale, scale) / scale
        return sdf.astype(np.float32)

    def compute_sdf_gradient_magnitude(self, sdf: np.ndarray) -> np.ndarray:
        """
        Compute the gradient magnitude of the SDF.
        High gradient regions indicate boundaries and potential watershed markers.

        Args:
            sdf: Signed distance field

        Returns:
            Gradient magnitude array
        """
        # Compute gradients in each direction
        grad_z, grad_y, grad_x = np.gradient(sdf)

        # Compute magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return gradient_magnitude

    def load_voxel_grid(self, filepath: str) -> np.ndarray:
        """
        Load voxel grid from various formats (.npy, .npz, .vti).

        Args:
            filepath: Path to the voxel grid file

        Returns:
            3D numpy array representing the voxel grid
        """
        print(f"Loading voxel grid from: {filepath}")

        if filepath.endswith(".npy"):
            voxel_grid = np.load(filepath)
        elif filepath.endswith(".npz"):
            data = np.load(filepath)
            # Try common key names
            if "voxel_grid" in data:
                voxel_grid = data["voxel_grid"]
            elif "data" in data:
                voxel_grid = data["data"]
            elif "arr_0" in data:
                voxel_grid = data["arr_0"]
            else:
                # Take the first array
                voxel_grid = data[list(data.keys())[0]]
        elif filepath.endswith(".vti"):
            # PyVista VTI format
            grid = pv.read(filepath)
            print(f"VTI grid dimensions: {grid.dimensions}")
            print(f"Active scalars shape: {grid.active_scalars.shape}")
            print(f"Active scalars size: {grid.active_scalars.size}")

            # Handle different VTI formats and orientations
            try:
                # First try the direct reshape
                voxel_grid = grid.active_scalars.reshape(grid.dimensions[::-1])
            except ValueError as e:
                print(f"Direct reshape failed: {e}")
                # Try alternative approaches
                try:
                    # Try with original dimensions order
                    voxel_grid = grid.active_scalars.reshape(grid.dimensions)
                except ValueError:
                    # Try to infer correct dimensions
                    total_points = grid.n_points
                    dims = grid.dimensions
                    expected_size = dims[0] * dims[1] * dims[2]
                    actual_size = grid.active_scalars.size

                    print(f"Expected size: {expected_size}, Actual size: {actual_size}")

                    if actual_size == expected_size:
                        # Size matches, try different reshape orders
                        try:
                            voxel_grid = grid.active_scalars.reshape(dims)
                        except ValueError:
                            voxel_grid = grid.active_scalars.reshape(dims[::-1])
                    else:
                        # Size mismatch - use grid points directly
                        print("Using point_data extraction method...")
                        # Get the actual 3D structure from the grid
                        voxel_grid = grid.point_data_to_cell_data().active_scalars.reshape(
                            [d - 1 for d in grid.dimensions][::-1]
                        )
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"Loaded voxel grid with shape: {voxel_grid.shape}")
        print(f"Value range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")

        return voxel_grid

    def preprocess_voxel_grid(
        self,
        voxel_grid: np.ndarray,
        binary_threshold: float = 0.5,
        gaussian_sigma: float = 0.5,
        remove_small_objects_min_size: int = 50,
    ) -> np.ndarray:
        """
        Preprocess voxel grid to clean up noise and prepare for segmentation.

        Args:
            voxel_grid: Input voxel grid
            binary_threshold: Threshold for binarization
            gaussian_sigma: Gaussian smoothing sigma (0 to disable)
            remove_small_objects_min_size: Remove connected components smaller than this

        Returns:
            Preprocessed binary voxel grid
        """
        print("Preprocessing voxel grid...")

        # Convert to binary if needed
        if not np.array_equal(voxel_grid, voxel_grid.astype(bool)):
            print(f"Converting to binary with threshold: {binary_threshold}")
            binary_grid = voxel_grid > binary_threshold
        else:
            binary_grid = voxel_grid.astype(bool)

        # Apply Gaussian smoothing if requested
        if gaussian_sigma > 0:
            print(f"Applying Gaussian smoothing with sigma: {gaussian_sigma}")
            # Smooth the original values, then re-threshold
            smoothed = filters.gaussian(voxel_grid.astype(float), sigma=gaussian_sigma)
            binary_grid = smoothed > binary_threshold

        # Remove small objects
        if remove_small_objects_min_size > 0:
            print(f"Removing small objects (< {remove_small_objects_min_size} voxels)")
            binary_grid = morphology.remove_small_objects(binary_grid, min_size=remove_small_objects_min_size)

        print(f"After preprocessing: {np.sum(binary_grid)} voxels are True")
        return binary_grid

    def segment_polyhedrons(
        self,
        binary_grid: np.ndarray,
        method: str = "watershed",  # Default changed from watershed_sdf
        min_distance: int = 5,
        erosion_iterations: int = 1,
        sdf_scale: float = 5.0,  # Kept for SDF method if chosen
        marker_threshold_percentile: float = 95.0,  # Kept for SDF method
        # New parameters for enhanced watershed
        gaussian_smooth_dt_sigma: float = 1.0,
        peak_local_max_footprint_size: int = 3,
    ) -> Tuple[np.ndarray, int]:
        """
        Segment binary voxel grid into individual polyhedrons.

        Args:
            binary_grid: Binary voxel grid
            method: Segmentation method ("watershed", "watershed_sdf", "connected_components", "dbscan")
            min_distance: Minimum distance between watershed markers (for all watershed types)
            erosion_iterations: Number of erosion iterations before watershed (for non-SDF watershed)
            sdf_scale: Scale parameter for SDF computation (for "watershed_sdf")
            marker_threshold_percentile: Percentile threshold for marker detection (for "watershed_sdf")
            gaussian_smooth_dt_sigma: Sigma for Gaussian smoothing of distance transform (for "watershed")
            peak_local_max_footprint_size: Footprint size for peak_local_max (for "watershed")


        Returns:
            Labeled array and number of labels
        """
        print(f"Segmenting polyhedrons using method: {method}")

        if method == "watershed_sdf":
            return self._watershed_sdf_segmentation(
                binary_grid, min_distance, erosion_iterations, sdf_scale, marker_threshold_percentile
            )
        elif method == "watershed":
            return self._watershed_segmentation(
                binary_grid,
                min_distance,
                erosion_iterations,
                gaussian_smooth_dt_sigma,
                peak_local_max_footprint_size,
            )
        elif method == "connected_components":
            return self._connected_components_segmentation(binary_grid)
        elif method == "dbscan":
            return self._dbscan_segmentation(binary_grid)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def _watershed_sdf_segmentation(
        self,
        binary_grid: np.ndarray,
        min_distance: int,
        erosion_iterations: int,  # Note: erosion_iterations is not directly used here, SDF on eroded grid is.
        sdf_scale: float,
        marker_threshold_percentile: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Advanced watershed segmentation using Signed Distance Field (SDF).
        This method uses SDF to:
        1. Generate smooth distance maps for watershed
        2. Detect markers based on SDF local maxima
        3. Use SDF gradients for better boundary detection
        """
        print("Performing SDF-based watershed segmentation...")

        # Step 1: Compute SDF for the binary grid
        print("Computing Signed Distance Field...")
        sdf = self.compute_sdf(binary_grid, scale=sdf_scale)

        # Step 2: Optionally apply erosion to separate touching objects for marker detection
        # The original code implied erosion was for the main binary_grid input to watershed,
        # but here it's used to refine sdf_for_markers.
        working_grid_for_markers = binary_grid.copy()
        if erosion_iterations > 0:  # This erosion is for marker generation
            print(f"Applying binary erosion with {erosion_iterations} iterations for marker refinement...")
            eroded_for_markers = working_grid_for_markers
            for _ in range(erosion_iterations):
                eroded_for_markers = morphology.binary_erosion(eroded_for_markers)

            if np.sum(eroded_for_markers) == 0:
                print(
                    "Warning: Erosion for marker refinement resulted in an empty grid. Using original grid for markers."
                )
                sdf_for_markers = sdf  # Use original SDF if erosion fails
            else:
                sdf_for_markers = self.compute_sdf(eroded_for_markers, scale=sdf_scale)
        else:
            sdf_for_markers = sdf

        # Step 3: Find watershed markers using SDF local maxima
        print("Detecting watershed markers from SDF...")

        # Use SDF values directly for marker detection (positive values = inside objects)
        # Ensure markers are sought within the *original* binary_grid or appropriately adjusted one
        # if erosion was meant to also modify the mask for watershed.
        # Here, sdf_inside uses binary_grid to mask sdf_for_markers.
        sdf_inside = np.where(binary_grid, sdf_for_markers, -np.inf)

        # Find local maxima in the SDF
        peak_coords = peak_local_max(
            sdf_inside,
            min_distance=min_distance,
            threshold_abs=np.percentile(sdf_inside[sdf_inside > -np.inf], marker_threshold_percentile)
            if np.any(sdf_inside > -np.inf)
            else None,
            exclude_border=False,
        )

        # Create marker array
        local_maxi_mask = np.zeros(sdf.shape, dtype=bool)
        if len(peak_coords) > 0:
            local_maxi_mask[tuple(peak_coords.T)] = True

        markers, num_markers_found = measure.label(local_maxi_mask, return_num=True)
        print(f"Found {num_markers_found} markers for SDF watershed.")

        if num_markers_found == 0:
            print("Warning: No markers found for SDF watershed. Falling back to connected components.")
            return self._connected_components_segmentation(binary_grid)

        # Step 4: Perform watershed using negative SDF as the distance map
        # We use negative SDF because watershed finds minima, but we want to segment from maxima
        print("Performing watershed on SDF...")
        # The mask for watershed should be the original binary_grid, not an eroded one unless intended
        labels = watershed(-sdf, markers, mask=binary_grid)

        num_labels = len(np.unique(labels)) - 1  # Exclude background (0)
        print(f"SDF-based watershed segmentation found {num_labels} objects")

        return labels, num_labels

    def _watershed_segmentation(
        self,
        binary_grid: np.ndarray,
        min_distance_pmL: int,  # Renamed for clarity
        erosion_iterations: int,
        gaussian_smooth_dt_sigma: float,
        peak_local_max_footprint_size: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Watershed segmentation using distance transform, adapted from 2D methodology.
        """
        print("Performing enhanced watershed segmentation (distance transform based)...")

        current_grid = binary_grid.copy()
        # Apply erosion to separate touching objects
        if erosion_iterations > 0:
            print(f"Applying binary erosion with {erosion_iterations} iterations...")
            eroded_grid = current_grid
            for _ in range(erosion_iterations):
                eroded_grid = morphology.binary_erosion(eroded_grid)
            if np.sum(eroded_grid) == 0:
                print("Warning: Erosion resulted in an empty grid. Using original grid for segmentation.")
            else:
                current_grid = eroded_grid

        # Compute distance transform on the (potentially eroded) grid
        distance = ndimage.distance_transform_edt(current_grid)

        # Optionally smooth the distance transform
        if gaussian_smooth_dt_sigma > 0:
            print(f"Smoothing distance transform with sigma: {gaussian_smooth_dt_sigma}")
            distance = filters.gaussian(distance, sigma=gaussian_smooth_dt_sigma)

        # Define footprint for peak_local_max
        footprint = None
        if peak_local_max_footprint_size > 0:
            footprint = np.ones(
                (peak_local_max_footprint_size, peak_local_max_footprint_size, peak_local_max_footprint_size),
                dtype=bool,
            )
            print(f"Using footprint of size {peak_local_max_footprint_size} for peak_local_max.")

        # Find local maxima as markers, ensuring peaks are within the current_grid
        # Removed threshold_abs=0.5, using labels=current_grid instead
        peak_coords = peak_local_max(
            distance,
            min_distance=min_distance_pmL,
            labels=current_grid,  # Ensures peaks are in foreground
            footprint=footprint,
            exclude_border=False,
        )

        local_maxi_mask = np.zeros(distance.shape, dtype=bool)
        if len(peak_coords) > 0:
            local_maxi_mask[tuple(peak_coords.T)] = True

        markers, num_markers_found = measure.label(local_maxi_mask, return_num=True)
        print(f"Found {num_markers_found} markers for watershed.")

        if num_markers_found == 0:
            print("Warning: No markers found for watershed. Consider connected components or check parameters.")
            # Fallback to connected components if no markers are found
            return self._connected_components_segmentation(binary_grid)

        # Perform watershed using the negative distance transform
        # Mask with current_grid (which could be the eroded version or original if erosion failed)
        labels = watershed(-distance, markers, mask=current_grid)

        num_labels = len(np.unique(labels)) - 1  # Exclude background (0)
        print(f"Enhanced watershed segmentation found {num_labels} objects")

        return labels, num_labels

    def _connected_components_segmentation(self, binary_grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Simple connected components labeling."""
        print("Performing connected components segmentation...")

        labels = measure.label(binary_grid, connectivity=3)  # 26-connectivity in 3D
        num_labels = len(np.unique(labels)) - 1  # Exclude background (0)

        print(f"Connected components found {num_labels} objects")
        return labels, num_labels

    def _dbscan_segmentation(
        self, binary_grid: np.ndarray, eps: float = 3.0, min_samples: int = 50
    ) -> Tuple[np.ndarray, int]:
        """DBSCAN clustering for complex shapes."""
        print("Performing DBSCAN segmentation...")

        # Get coordinates of all True voxels
        coords = np.column_stack(np.where(binary_grid))

        if len(coords) == 0:
            return np.zeros_like(binary_grid, dtype=int), 0

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

        # Create labeled array
        labels = np.zeros_like(binary_grid, dtype=int)
        for i, (z, y, x) in enumerate(coords):
            # DBSCAN labels start from 0, but we want background as 0
            labels[z, y, x] = clustering.labels_[i] + 1

        # Remove noise points (labeled as -1 in DBSCAN)
        labels[labels == 0] = 0  # Noise becomes background

        num_labels = len(np.unique(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        print(f"DBSCAN found {num_labels} clusters")

        return labels, num_labels

    def extract_polyhedron_mesh(
        self,
        labeled_grid: np.ndarray,
        label_id: int,
        smoothing_iterations: int = 10,
        decimation_ratio: float = 0.9,
        use_sdf: bool = True,
    ) -> Dict:
        """
        Extract mesh (vertices and faces) for a specific labeled polyhedron using SDF.

        Args:
            labeled_grid: Labeled segmentation array
            label_id: ID of the specific polyhedron to extract
            smoothing_iterations: Number of smoothing iterations to apply
            decimation_ratio: Ratio for mesh decimation (0-1, higher = more decimation)
            use_sdf: Whether to use SDF for better boundary detection

        Returns:
            Dictionary with vertices and faces in global coordinates
        """
        # Extract single polyhedron
        single_polyhedron = (labeled_grid == label_id).astype(bool)

        if not np.any(single_polyhedron):
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0}

        if use_sdf:
            # Use SDF for better boundary detection
            sdf = self.compute_sdf(single_polyhedron, scale=3.0)

            # Convert to PyVista for mesh operations
            grid = pv.ImageData(dimensions=sdf.shape, spacing=self.voxel_spacing, origin=self.origin)
            grid.point_data["values"] = sdf.flatten(order="F")  # Ensure Fortran order for PyVista

            # Extract isosurface at zero level (exact boundary)
            mesh = grid.contour(isosurfaces=[0.0])
        else:
            # Traditional binary approach
            single_polyhedron_uint8 = single_polyhedron.astype(np.uint8)
            grid = pv.ImageData(
                dimensions=single_polyhedron_uint8.shape, spacing=self.voxel_spacing, origin=self.origin
            )
            grid.point_data["values"] = single_polyhedron_uint8.flatten(order="F")  # Ensure Fortran order

            # Extract isosurface using marching cubes
            mesh = grid.contour(isosurfaces=[0.5])

        if mesh.n_points == 0 or mesh.n_cells == 0:
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0}

        # Apply smoothing to reduce voxel artifacts
        if smoothing_iterations > 0:
            mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)

        # Apply decimation to reduce polygon count
        if 0 < decimation_ratio < 1:
            target_reduction = decimation_ratio
            mesh = mesh.decimate(target_reduction)

        if mesh.n_points == 0 or mesh.n_cells == 0:  # Check again after processing
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0}

        # Extract vertices and faces
        vertices = mesh.points
        # PyVista's faces array is like [n_points, p0, p1, ..., n_points, p0, p1, ...]
        # We need to convert it to a list of lists
        faces_list = []
        raw_faces = mesh.faces
        i = 0
        while i < len(raw_faces):
            num_points_in_face = raw_faces[i]
            faces_list.append(raw_faces[i + 1 : i + 1 + num_points_in_face].tolist())
            i += num_points_in_face + 1

        # Calculate volume
        volume = mesh.volume if hasattr(mesh, "volume") else 0

        return {
            "vertices": vertices.tolist(),
            "faces": faces_list,
            "volume": volume,
            "n_vertices": mesh.n_points,  # Use mesh.n_points
            "n_faces": mesh.n_cells,  # Changed from mesh.n_faces to mesh.n_cells
        }

    def process_voxel_grid(
        self,
        voxel_grid: np.ndarray,
        segmentation_method: str = "watershed",  # Default changed to "watershed"
        min_polyhedron_size: int = 100,
        smoothing_iterations: int = 10,
        decimation_ratio: float = 0.8,
        num_workers: int = 1,
        # Preprocessing specific
        binary_threshold: float = 0.5,
        gaussian_sigma: float = 0.5,  # For initial grid smoothing
        remove_small_objects_min_size: int = 50,
        # Segmentation specific (common and new)
        min_distance: int = 5,  # For peak_local_max min_distance
        erosion_iterations: int = 1,  # For watershed pre-separation
        sdf_scale: float = 5.0,  # For "watershed_sdf"
        marker_threshold_percentile: float = 95.0,  # For "watershed_sdf"
        gaussian_smooth_dt_sigma: float = 1.0,  # For "watershed" distance transform smoothing
        peak_local_max_footprint_size: int = 3,  # For "watershed" peak_local_max footprint
        # Mesh extraction specific
        use_sdf: bool = True,
        remove_boundary_polyhedrons: bool = True,
        max_voxel_aspect_ratio: Optional[float] = 20.0,
        **other_kwargs,  # Catch any other unexpected kwargs
    ) -> Dict:
        """
        Complete processing pipeline for a voxel grid.

        Args:
            voxel_grid: Input voxel grid
            segmentation_method: Method for segmentation ("watershed", "watershed_sdf", etc.)
            min_polyhedron_size: Minimum size of polyhedrons to keep
            smoothing_iterations: Mesh smoothing iterations
            decimation_ratio: Mesh decimation ratio
            num_workers: Number of parallel workers for mesh extraction
            binary_threshold: Threshold for binarization
            gaussian_sigma: Gaussian smoothing sigma for input grid (0 to disable)
            remove_small_objects_min_size: Remove objects smaller than this
            min_distance: Minimum distance between watershed markers
            erosion_iterations: Number of erosion iterations before watershed
            sdf_scale: Scale parameter for SDF computation (for "watershed_sdf")
            marker_threshold_percentile: Percentile for SDF marker detection (for "watershed_sdf")
            gaussian_smooth_dt_sigma: Sigma for smoothing distance transform (for "watershed")
            peak_local_max_footprint_size: Footprint size for peak_local_max (for "watershed")
            use_sdf: Whether to use SDF for mesh extraction
            remove_boundary_polyhedrons: Whether to remove polyhedrons touching the grid boundary
            max_voxel_aspect_ratio: Maximum aspect ratio of voxel bounding box to keep a polyhedron (e.g., 20). Longest_side / shortest_side. Set to 0 or None to disable.
            other_kwargs: For any other potential future arguments

        Returns:
            Dictionary containing all extracted polyhedrons
        """
        print("=" * 60)
        print("POLYHEDRON SEGMENTATION PIPELINE")
        if segmentation_method == "watershed":
            print("Using: Enhanced Distance Transform Watershed")
        elif segmentation_method == "watershed_sdf":
            print("Using: SDF-based Watershed")
        print("=" * 60)

        preprocess_params = {
            "binary_threshold": binary_threshold,
            "gaussian_sigma": gaussian_sigma,
            "remove_small_objects_min_size": remove_small_objects_min_size,
        }

        segment_params = {
            "min_distance": min_distance,
            "erosion_iterations": erosion_iterations,
            # Specific to watershed_sdf
            "sdf_scale": sdf_scale,
            "marker_threshold_percentile": marker_threshold_percentile,
            # Specific to enhanced watershed
            "gaussian_smooth_dt_sigma": gaussian_smooth_dt_sigma,
            "peak_local_max_footprint_size": peak_local_max_footprint_size,
        }

        # Step 1: Preprocess
        binary_grid = self.preprocess_voxel_grid(voxel_grid, **preprocess_params)

        # Step 2: Segment
        labeled_grid, num_labels = self.segment_polyhedrons(binary_grid, method=segmentation_method, **segment_params)

        if num_labels == 0:
            print("No polyhedrons found after segmentation!")
            return {"polyhedrons": {}, "metadata": {"total_count": 0, "boundary_removed_count": 0}}

        # Step 3: Filter polyhedrons by size and optionally remove boundary polyhedrons
        print(f"\nInitial labels found: {num_labels}")
        polyhedrons = {}

        # First, filter by minimum size
        candidate_label_ids = [
            label_id
            for label_id in range(1, num_labels + 1)
            if np.sum(labeled_grid == label_id) >= min_polyhedron_size
        ]
        print(
            f"Labels remaining after min_polyhedron_size ({min_polyhedron_size} voxels) filter: {len(candidate_label_ids)}"
        )

        label_ids_to_process = candidate_label_ids
        boundary_removed_count = 0
        aspect_ratio_removed_count = 0

        print(
            f"DEBUG: Checking boundary removal. remove_boundary_polyhedrons={remove_boundary_polyhedrons}, len(label_ids_to_process)={len(label_ids_to_process)}"
        )
        if remove_boundary_polyhedrons and len(label_ids_to_process) > 0:
            print("Identifying and removing polyhedrons touching the grid boundary...")
            boundary_labels = set()
            dims = labeled_grid.shape
            # Z boundaries
            boundary_labels.update(np.unique(labeled_grid[0, :, :]))
            boundary_labels.update(np.unique(labeled_grid[dims[0] - 1, :, :]))
            # Y boundaries
            boundary_labels.update(np.unique(labeled_grid[:, 0, :]))
            boundary_labels.update(np.unique(labeled_grid[:, dims[1] - 1, :]))
            # X boundaries
            boundary_labels.update(np.unique(labeled_grid[:, :, 0]))
            boundary_labels.update(np.unique(labeled_grid[:, :, dims[2] - 1]))

            if 0 in boundary_labels:  # Remove background label if present
                boundary_labels.remove(0)

            initial_count_before_boundary_filter = len(label_ids_to_process)
            temp_label_ids_after_boundary = []
            for label_id in label_ids_to_process:
                if label_id not in boundary_labels:
                    temp_label_ids_after_boundary.append(label_id)
                else:
                    boundary_removed_count += 1

            label_ids_to_process = temp_label_ids_after_boundary

            print(
                f"Removed {boundary_removed_count} polyhedrons touching the boundary (out of {initial_count_before_boundary_filter} candidates)."
            )
            print(f"Labels remaining after boundary filter: {len(label_ids_to_process)}")

        # Aspect Ratio Filtering (new step)
        if max_voxel_aspect_ratio is not None and max_voxel_aspect_ratio > 0 and len(label_ids_to_process) > 0:
            print(f"Filtering polyhedrons by max voxel aspect ratio ({max_voxel_aspect_ratio})...")
            initial_count_before_aspect_filter = len(label_ids_to_process)
            label_ids_after_aspect_ratio_filter = []
            for label_id in label_ids_to_process:
                coords = np.argwhere(labeled_grid == label_id)
                if coords.shape[0] < 2:
                    label_ids_after_aspect_ratio_filter.append(label_id)
                    continue

                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                dims = max_coords - min_coords + 1
                dims = np.maximum(dims, 1)
                current_aspect_ratio = np.max(dims) / np.min(dims)

                if current_aspect_ratio <= max_voxel_aspect_ratio:
                    label_ids_after_aspect_ratio_filter.append(label_id)
                else:
                    aspect_ratio_removed_count += 1

            label_ids_to_process = label_ids_after_aspect_ratio_filter
            print(
                f"Removed {aspect_ratio_removed_count} polyhedrons due to aspect ratio filter (out of {initial_count_before_aspect_filter} candidates)."
            )
            print(f"Labels remaining after aspect ratio filter: {len(label_ids_to_process)}")

        if not label_ids_to_process:
            print("No polyhedrons left to process after filtering!")
            return {
                "polyhedrons": {},
                "metadata": {
                    "total_count": 0,
                    "original_labels": num_labels,
                    "boundary_removed_count": boundary_removed_count,
                    "aspect_ratio_removed_count": aspect_ratio_removed_count,
                },
            }

        # Step 4: Extract individual polyhedrons
        print(f"Extracting {len(label_ids_to_process)} polyhedrons...")

        if num_workers > 1 and len(label_ids_to_process) > 1:  # Check if there are labels to process in parallel
            print(f"Using {num_workers} parallel workers for mesh extraction...")
            # Ensure self can be pickled for multiprocessing if methods are directly passed
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _global_mesh_task,
                        self,  # Pass the instance
                        labeled_grid,
                        label_id,
                        smoothing_iterations,
                        decimation_ratio,
                        use_sdf,  # Pass mesh extraction use_sdf
                    )
                    for label_id in label_ids_to_process  # Use filtered list
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(label_ids_to_process),  # Use filtered list
                    desc="Extracting polyhedrons (parallel)",
                ):
                    try:
                        label_id, polyhedron_size, mesh_data = future.result()
                        if mesh_data and len(mesh_data.get("vertices", [])) > 0:
                            polyhedrons[str(label_id)] = {
                                "id": label_id,
                                "vertices": mesh_data["vertices"],
                                "faces": mesh_data["faces"],
                                "volume": mesh_data.get("volume", 0),
                                "n_vertices": mesh_data.get("n_vertices", 0),
                                "n_faces": mesh_data.get("n_faces", 0),
                                "voxel_count": int(polyhedron_size),
                                "centroid": self._calculate_centroid(mesh_data["vertices"]),
                                "bounding_box": self._calculate_bounding_box(mesh_data["vertices"]),
                            }
                    except Exception as e:
                        print(
                            f"Error processing future for label {label_id if 'label_id' in locals() else 'unknown'}: {e}"
                        )

        else:
            if num_workers > 1:
                print("Not enough labels or workers for parallel processing, using sequential.")
            for label_id in tqdm(
                label_ids_to_process, desc="Extracting polyhedrons (sequential)"
            ):  # Use filtered list
                polyhedron_size = np.sum(labeled_grid == label_id)
                mesh_data = self.extract_polyhedron_mesh(
                    labeled_grid,
                    label_id,
                    smoothing_iterations,
                    decimation_ratio,
                    use_sdf,  # Pass mesh extraction use_sdf
                )
                if mesh_data and len(mesh_data.get("vertices", [])) > 0:
                    polyhedrons[str(label_id)] = {
                        "id": label_id,
                        "vertices": mesh_data["vertices"],
                        "faces": mesh_data["faces"],
                        "volume": mesh_data.get("volume", 0),
                        "n_vertices": mesh_data.get("n_vertices", 0),
                        "n_faces": mesh_data.get("n_faces", 0),
                        "voxel_count": int(polyhedron_size),
                        "centroid": self._calculate_centroid(mesh_data["vertices"]),
                        "bounding_box": self._calculate_bounding_box(mesh_data["vertices"]),
                    }

        # Metadata
        metadata = {
            "total_count": len(polyhedrons),
            "original_labels": num_labels,
            "min_size_filtered_count": len(candidate_label_ids),
            "boundary_removed_count": boundary_removed_count,
            "aspect_ratio_removed_count": aspect_ratio_removed_count,
            "final_extracted_count": len(polyhedrons),
            "voxel_spacing": self.voxel_spacing.tolist(),
            "origin": self.origin.tolist(),
            "grid_shape": voxel_grid.shape,
            "segmentation_method": segmentation_method,
            "processing_parameters": {
                "min_polyhedron_size": min_polyhedron_size,
                "smoothing_iterations": smoothing_iterations,
                "decimation_ratio": decimation_ratio,
                "binary_threshold": binary_threshold,
                "gaussian_sigma_preprocess": gaussian_sigma,
                "remove_small_objects_min_size": remove_small_objects_min_size,
                "min_distance_markers": min_distance,
                "erosion_iterations_segment": erosion_iterations,
                "use_sdf_mesh_extraction": use_sdf,
                "remove_boundary_polyhedrons": remove_boundary_polyhedrons,
                "max_voxel_aspect_ratio": max_voxel_aspect_ratio,
            },
        }
        if segmentation_method == "watershed_sdf":
            metadata["processing_parameters"]["sdf_scale"] = sdf_scale
            metadata["processing_parameters"]["marker_threshold_percentile"] = marker_threshold_percentile
        elif segmentation_method == "watershed":
            metadata["processing_parameters"]["gaussian_smooth_dt_sigma"] = gaussian_smooth_dt_sigma
            metadata["processing_parameters"]["peak_local_max_footprint_size"] = peak_local_max_footprint_size

        print(f"Successfully extracted {len(polyhedrons)} polyhedrons using {segmentation_method}")
        return {"polyhedrons": polyhedrons, "metadata": metadata}

    def _calculate_centroid(self, vertices: List[List[float]]) -> List[float]:
        """Calculate centroid of vertices."""
        if not vertices:
            return [0.0, 0.0, 0.0]
        vertices_array = np.array(vertices)
        return np.mean(vertices_array, axis=0).tolist()

    def _calculate_bounding_box(self, vertices: List[List[float]]) -> Dict:
        """Calculate bounding box of vertices."""
        if not vertices:
            return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}

        vertices_array = np.array(vertices)
        return {"min": np.min(vertices_array, axis=0).tolist(), "max": np.max(vertices_array, axis=0).tolist()}

    def save_to_json(self, polyhedrons_data: Dict, output_path: str, indent: int = 2, compress: bool = False):
        """
        Save polyhedrons data to JSON file.

        Args:
            polyhedrons_data: Data dictionary from process_voxel_grid
            output_path: Output file path
            indent: JSON indentation (None for compact)
            compress: Whether to compress the JSON file
        """
        print(f"Saving polyhedrons to: {output_path}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if compress:
            import gzip

            with gzip.open(output_path + ".gz", "wt", encoding="utf-8") as f:
                json.dump(polyhedrons_data, f, indent=indent)
            print(f"Compressed JSON saved to: {output_path}.gz")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(polyhedrons_data, f, indent=indent)
            print(f"JSON saved to: {output_path}")

    def load_from_json(self, json_path: str) -> Dict:
        """Load polyhedrons data from JSON file."""
        if json_path.endswith(".gz"):
            import gzip

            with gzip.open(json_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def visualize_polyhedrons(
        self, polyhedrons_data: Dict, max_polyhedrons: int = 10, output_path: Optional[str] = None
    ):
        """
        Visualize extracted polyhedrons using PyVista.

        Args:
            polyhedrons_data: Data from process_voxel_grid
            max_polyhedrons: Maximum number of polyhedrons to visualize
            output_path: Path to save screenshot (optional)
        """
        import random

        polyhedrons = polyhedrons_data.get("polyhedrons", {})

        if not polyhedrons:
            print("No polyhedrons to visualize!")
            return

        # Limit number of polyhedrons for visualization
        poly_ids = list(polyhedrons.keys())
        if len(poly_ids) > max_polyhedrons:
            poly_ids_to_show = random.sample(poly_ids, max_polyhedrons)
            print(f"Visualizing {max_polyhedrons} random polyhedrons out of {len(polyhedrons)}")
        else:
            poly_ids_to_show = poly_ids
            print(f"Visualizing all {len(poly_ids_to_show)} polyhedrons")

        # Set PyVista to work properly in different environments
        pv.set_plot_theme("document")
        # For off_screen, ensure it's True if output_path is provided and not an interactive window format.
        # .html or .vtkjs might be interactive, .png/.jpg are not.
        is_static_screenshot = output_path and output_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".svg", ".eps", ".pdf")
        )
        plotter = pv.Plotter(off_screen=is_static_screenshot, window_size=[1024, 768])

        meshes_added = 0
        for i, poly_id in enumerate(poly_ids_to_show):
            poly_data = polyhedrons[poly_id]
            vertices = np.array(poly_data.get("vertices", []))
            faces_pv_list = poly_data.get("faces", [])  # This is list of lists

            if vertices.size == 0 or not faces_pv_list:
                print(f"Skipping polyhedron {poly_id}: empty vertices or faces")
                continue

            # Create PyVista faces array (e.g., [3, p0, p1, ..., n_points, p0, p1, ...])
            pv_faces = []
            valid_face_found = False
            for face_indices in faces_pv_list:
                if len(face_indices) >= 3:  # A face must have at least 3 points
                    pv_faces.extend([len(face_indices)] + face_indices)
                    valid_face_found = True

            if not valid_face_found:
                print(f"Skipping polyhedron {poly_id}: no valid faces after processing")
                continue

            try:
                mesh = pv.PolyData(vertices, np.asarray(pv_faces))
                if mesh.n_points == 0 or mesh.n_cells == 0:
                    print(f"Skipping polyhedron {poly_id}: mesh created with no points/cells.")
                    continue

                # Add with random color
                color = np.random.rand(3).tolist()  # pv.Color accepts list or string
                plotter.add_mesh(mesh, color=color, opacity=0.8, label=f"Polyhedron {poly_id}")
                meshes_added += 1
                # print(f"Added polyhedron {poly_id} with {mesh.n_points} vertices and {mesh.n_faces} faces")
            except Exception as e:
                print(f"Error creating or adding mesh for polyhedron {poly_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        if meshes_added == 0:
            print("No valid meshes were added to the plotter!")
            if is_static_screenshot:
                plotter.close()  # Close if off_screen and nothing to show
            return

        print(f"Total meshes added to plotter: {meshes_added}")

        plotter.add_legend()
        plotter.show_bounds(grid="front", location="outer", all_edges=True)
        plotter.add_axes()
        plotter.enable_zoom_scaling()

        if output_path:
            print(f"Attempting to save screenshot to: {output_path}")
            try:
                if is_static_screenshot:  # For truly off-screen like PNG
                    plotter.screenshot(output_path, transparent_background=False)
                    print(f"Screenshot saved to: {output_path}")
                else:  # For potentially interactive like HTML
                    plotter.show(auto_close=False)  # Show first
                    # For some formats like html, this might be it. For others, might need interaction.
                    # If a static format was intended but off_screen was False, this might hang.
                    # This path is a bit tricky. Safest is to ensure off_screen=True for static.
                    print(f"Interactive visualization displayed. Manual save or close required if not HTML/VTKJS.")
                    # plotter.screenshot(output_path) # This might or might not work depending on backend and window state
            except Exception as e:
                print(f"Error during visualization or saving screenshot: {e}")
                import traceback

                traceback.print_exc()
            finally:
                if is_static_screenshot:
                    plotter.close()

        else:
            try:
                print("Displaying interactive visualization window...")
                plotter.show()
            except Exception as e:
                print(f"Error showing plotter: {e}")
                print(
                    "This might be due to display/GUI limitations in your environment (e.g., running in a headless server). Try using --screenshot option."
                )
                import traceback

                traceback.print_exc()

    def save_to_paraview(self, polyhedrons_data: Dict, output_path: str, multiblock: bool = True):
        """
        Export all polyhedrons to a Paraview-compatible file (.vtp or .vtm).

        Args:
            polyhedrons_data: Data dictionary from process_voxel_grid
            output_path: Output file path (.vtp for single mesh, .vtm for multiblock)
            multiblock: If True, save as MultiBlock (.vtm), else as single PolyData (.vtp)
        """
        import pyvista as pv  # Ensure pv is available
        from pathlib import Path

        print(f"Exporting polyhedrons to Paraview file: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        polyhedrons = polyhedrons_data.get("polyhedrons", {})
        if not polyhedrons:
            print("No polyhedrons to export.")
            return

        if multiblock:
            mb = pv.MultiBlock()
            for poly_id_str, poly_data in polyhedrons.items():
                poly_id = int(poly_id_str)  # Ensure poly_id is int for field data
                vertices = np.array(poly_data.get("vertices", []))
                faces_list = poly_data.get("faces", [])  # List of lists

                if vertices.size == 0 or not faces_list:
                    continue

                pv_faces = []
                valid_face_found = False
                for face_indices in faces_list:
                    if len(face_indices) >= 3:
                        pv_faces.extend([len(face_indices)] + face_indices)
                        valid_face_found = True

                if not valid_face_found:
                    continue

                try:
                    mesh = pv.PolyData(vertices, np.asarray(pv_faces))
                    if mesh.n_points > 0:  # Add field data only if mesh is valid
                        mesh.point_data["poly_id"] = np.full(mesh.n_points, poly_id, dtype=int)
                        mesh.cell_data["poly_id"] = np.full(mesh.n_cells, poly_id, dtype=int)
                        mesh.field_data["volume"] = np.array([poly_data.get("volume", 0)])
                        mesh.field_data["voxel_count"] = np.array([poly_data.get("voxel_count", 0)])
                    mb[str(poly_id)] = mesh
                except Exception as e:
                    print(f"Could not create mesh for polyhedron {poly_id} for Paraview: {e}")

            if mb.n_blocks == 0:
                print("No valid meshes to save in MultiBlock.")
                return

            out_path = Path(output_path)
            if out_path.suffix.lower() != ".vtm":
                out_path = out_path.with_suffix(".vtm")
            mb.save(str(out_path), binary=True)
            print(f"MultiBlock file saved to: {out_path}")
        else:
            all_meshes = []
            for poly_id_str, poly_data in polyhedrons.items():
                poly_id = int(poly_id_str)
                vertices = np.array(poly_data.get("vertices", []))
                faces_list = poly_data.get("faces", [])

                if vertices.size == 0 or not faces_list:
                    continue

                pv_faces = []
                valid_face_found = False
                for face_indices in faces_list:
                    if len(face_indices) >= 3:
                        pv_faces.extend([len(face_indices)] + face_indices)
                        valid_face_found = True
                if not valid_face_found:
                    continue

                try:
                    mesh = pv.PolyData(vertices, np.asarray(pv_faces))
                    if mesh.n_points > 0:
                        mesh.point_data["poly_id"] = np.full(mesh.n_points, poly_id, dtype=int)
                        mesh.cell_data["poly_id"] = np.full(mesh.n_cells, poly_id, dtype=int)
                        # Storing volume and voxel_count as cell data might be more appropriate if it's per-polyhedron
                        # Or as field data if that's preferred for a combined mesh.
                        # For simplicity, let's add poly_id to distinguish them.
                    all_meshes.append(mesh)
                except Exception as e:
                    print(f"Could not create mesh for polyhedron {poly_id} for combined Paraview: {e}")

            if not all_meshes:
                print("No polyhedrons to combine and export.")
                return

            try:
                # Need to handle merging carefully if they have different point/cell data structures.
                # A simple merge might lose some data or require uniform fields.
                # For now, let's assume poly_id is the main distinguishing feature.
                combined_mesh = pv.MultiBlock(all_meshes).combine(merge_points=True, tolerance=1e-05)
                # If combine() doesn't preserve poly_id well across merged entities,
                # an alternative is to save them as separate datasets in a .vtmb or .pvd file.
                # Or, ensure 'poly_id' is correctly handled during combination.
            except Exception as e:
                print(f"Error combining meshes: {e}. Saving as MultiBlock instead.")
                # Fallback to multiblock if combine fails
                mb_fallback = pv.MultiBlock()
                for i, m in enumerate(all_meshes):
                    mb_fallback[f"poly_{i}"] = m  # Use original poly_id if available and unique
                out_path_fallback = Path(output_path).with_suffix(".vtm")
                mb_fallback.save(str(out_path_fallback), binary=True)
                print(f"Combined PolyData save failed. Saved as MultiBlock to: {out_path_fallback}")
                return

            out_path = Path(output_path)
            # When saving a combined mesh (UnstructuredGrid), .vtu is the appropriate extension.
            if out_path.suffix.lower() != ".vtu":
                out_path = out_path.with_suffix(".vtu")
            combined_mesh.save(str(out_path), binary=True)
            print(f"Combined UnstructuredGrid file saved to: {out_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Polyhedron Segmentation for Voxel Grids")

    # Input/Output
    parser.add_argument("--input", "-i", type=str, required=True, help="Input voxel grid file (.npy, .npz, .vti)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file path for polyhedrons data")

    # Preprocessing
    parser.add_argument("--binary-threshold", type=float, default=0.5, help="Threshold for binarization")
    parser.add_argument(
        "--gaussian-sigma", type=float, default=0.5, help="Gaussian smoothing sigma for input grid (0 to disable)"
    )
    parser.add_argument(
        "--remove-small-objects",
        type=int,
        default=50,
        help="Remove objects smaller than this many voxels post-binarization",
    )

    # Segmentation
    parser.add_argument(
        "--method",
        type=str,
        default="watershed",  # Default changed to "watershed"
        choices=["watershed", "watershed_sdf", "connected_components", "dbscan"],
        help="Segmentation method ('watershed' is enhanced distance transform based)",
    )
    # Parameters for all watershed types / general segmentation
    parser.add_argument(
        "--min-distance", type=int, default=7, help="Minimum distance between watershed markers (peak_local_max)"
    )  # Default changed slightly
    parser.add_argument(
        "--erosion-iterations", type=int, default=1, help="Erosion iterations before watershed/marker finding"
    )

    # Parameters specific to "watershed_sdf"
    parser.add_argument(
        "--sdf-scale", type=float, default=5.0, help="Scale parameter for SDF computation (for 'watershed_sdf')"
    )
    parser.add_argument(
        "--marker-threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for SDF marker detection (for 'watershed_sdf', e.g., 95.0 = top 5%)",
    )

    # Parameters specific to "watershed" (enhanced distance transform based)
    parser.add_argument(
        "--gaussian-smooth-dt-sigma",
        type=float,
        default=1.0,  # Default to 1.0 for some smoothing
        help="Sigma for Gaussian smoothing of distance transform in 'watershed' method (0 to disable)",
    )
    parser.add_argument(
        "--peak-local-max-footprint-size",
        type=int,
        default=3,  # Default to 3x3x3 footprint
        help="Footprint size for peak_local_max in 'watershed' method (e.g., 3 for 3x3x3, 0 for no explicit footprint)",
    )

    # Mesh processing
    parser.add_argument("--smoothing-iterations", type=int, default=10, help="Number of mesh smoothing iterations")
    parser.add_argument(
        "--decimation-ratio", type=float, default=0.8, help="Mesh decimation ratio (0-1, e.g., 0.8 for 80%% reduction)"
    )
    parser.add_argument(
        "--min-polyhedron-size", type=int, default=100, help="Minimum polyhedron size in voxels to keep"
    )  # Default changed
    parser.add_argument(
        "--use-sdf-mesh",  # Renamed for clarity
        action=argparse.BooleanOptionalAction,  # Allows --use-sdf-mesh / --no-use-sdf-mesh
        default=True,
        help="Use Signed Distance Field for mesh extraction (default: True)",
    )

    # Physical parameters
    parser.add_argument(
        "--voxel-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel spacing (dx dy dz)"
    )
    parser.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Grid origin (x y z)")

    # Options
    parser.add_argument("--compress", action="store_true", help="Compress output JSON with gzip")
    parser.add_argument(
        "--visualize", action="store_true", help="Show interactive 3D visualization of some polyhedrons"
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Save screenshot of visualization to this file path (e.g., preview.png)",
    )
    parser.add_argument(
        "--paraview-export",  # Renamed for clarity
        type=str,
        default=None,  # Default to None, export if path is given
        help="Export segmentation to Paraview .vtm (multiblock) or .vtp (single) file. Filename determines type, or provide suffix.",
    )
    parser.add_argument(
        "--paraview-multiblock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If exporting to Paraview, save as MultiBlock (.vtm) if true, else combined PolyData (.vtp)",
    )

    parser.add_argument(
        "--remove-boundary-polyhedrons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove polyhedrons that touch the voxel grid boundary (default: True)",
    )

    parser.add_argument(
        "--max-voxel-aspect-ratio",
        type=float,
        default=20.0,  # Default value, e.g. 20. Set to 0 to disable.
        help="Maximum aspect ratio of a polyhedron's voxel bounding box (longest_side/shortest_side) to keep it. Set to 0 to disable. (default: 20.0)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() // 2 or 1,  # Sensible default
        help="Number of parallel workers for mesh extraction",
    )

    args = parser.parse_args()

    # Initialize segmentation
    segmentation = PolyhedronSegmentation(voxel_spacing=tuple(args.voxel_spacing), origin=tuple(args.origin))

    # Load voxel grid
    print(f"Loading input grid: {args.input}")
    voxel_grid = segmentation.load_voxel_grid(args.input)

    # Process
    start_time = time.time()
    print(f"Starting polyhedron processing with method: {args.method}")

    # DEBUG: Print the value of remove_boundary_polyhedrons from args
    print(f"DEBUG in main: args.remove_boundary_polyhedrons = {args.remove_boundary_polyhedrons}")

    polyhedrons_data = segmentation.process_voxel_grid(
        voxel_grid,
        segmentation_method=args.method,
        min_polyhedron_size=args.min_polyhedron_size,
        smoothing_iterations=args.smoothing_iterations,
        decimation_ratio=args.decimation_ratio,
        num_workers=args.num_workers,
        # Preprocessing specific
        binary_threshold=args.binary_threshold,
        gaussian_sigma=args.gaussian_sigma,
        remove_small_objects_min_size=args.remove_small_objects,
        # Segmentation specific
        min_distance=args.min_distance,
        erosion_iterations=args.erosion_iterations,
        sdf_scale=args.sdf_scale,  # For watershed_sdf
        marker_threshold_percentile=args.marker_threshold_percentile,  # For watershed_sdf
        gaussian_smooth_dt_sigma=args.gaussian_smooth_dt_sigma,  # For watershed
        peak_local_max_footprint_size=args.peak_local_max_footprint_size,  # For watershed
        # Mesh extraction specific
        use_sdf=args.use_sdf_mesh,  # Passed to process_voxel_grid
        remove_boundary_polyhedrons=args.remove_boundary_polyhedrons,
        max_voxel_aspect_ratio=args.max_voxel_aspect_ratio,
    )

    processing_time = time.time() - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")

    # Save results
    if polyhedrons_data and polyhedrons_data.get("polyhedrons"):
        segmentation.save_to_json(polyhedrons_data, args.output, compress=args.compress)

        # Paraview export
        paraview_output_path = args.paraview_export
        if not paraview_output_path and args.output:  # Default paraview export name if not specified
            paraview_output_path = Path(args.output).with_suffix(".vtm" if args.paraview_multiblock else ".vtu")

        if paraview_output_path:
            # Ensure correct suffix based on multiblock flag if not already set by user
            paraview_path_obj = Path(paraview_output_path)
            expected_suffix = ".vtm" if args.paraview_multiblock else ".vtu"
            if paraview_path_obj.suffix.lower() != expected_suffix.lower():
                paraview_output_path = str(paraview_path_obj.with_suffix(expected_suffix))

            segmentation.save_to_paraview(polyhedrons_data, paraview_output_path, multiblock=args.paraview_multiblock)

        # Visualization
        if args.visualize or args.screenshot:
            segmentation.visualize_polyhedrons(polyhedrons_data, output_path=args.screenshot)

        print(
            f"Polyhedron segmentation completed. Found {polyhedrons_data.get('metadata', {}).get('total_count', 0)} polyhedrons."
        )
    else:
        print("Polyhedron segmentation completed, but no polyhedrons were extracted or an error occurred.")


if __name__ == "__main__":
    # Make sure to handle potential issues with multiprocessing in spawned processes,
    # especially if using PyVista or other GUI/OpenGL libraries in ways not friendly to fork/spawn.
    # The _global_mesh_task is a top-level function which helps.
    # Consider `if __name__ == "__main__":` for PyVista imports if they cause issues on module load for spawned workers.
    # However, current imports are at the top level.

    # For PyVista and multiprocessing on some systems (like Windows or macOS with 'spawn'),
    # it might be necessary to ensure that PyVista is not initialized in the global scope
    # in a way that interferes with child processes. Usually, plotter instances are created
    # and used within functions, which is safer.
    main()
