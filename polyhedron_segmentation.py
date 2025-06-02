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
import sys

if sys.platform == "darwin":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)


def validate_mesh_coordinates(vertices: np.ndarray, label_id: int, max_reasonable_coord: float = 1e4) -> bool:
    """
    Aggressive coordinate validation using the same logic as the converter's outlier detection.

    Args:
        vertices: Vertex array to validate
        label_id: Polyhedron ID for logging
        max_reasonable_coord: Maximum reasonable coordinate value

    Returns:
        True if coordinates are valid, False if they should be filtered out
    """
    if vertices.size == 0:
        return False

    # Check for invalid coordinates (NaN, inf)
    if np.any(~np.isfinite(vertices)):
        print(f"WARNING: Polyhedron {label_id} has invalid coordinates (NaN/inf). Filtering out...")
        return False

    # Multiple cascading thresholds for aggressiveness
    extreme_thresholds = [1e10, 1e8, 1e6, max_reasonable_coord]
    max_coord = np.max(np.abs(vertices))

    for threshold in extreme_thresholds:
        if max_coord > threshold:
            print(
                f"WARNING: Polyhedron {label_id} has extreme coordinates (max: {max_coord:.2e} > {threshold:.2e}). Filtering out..."
            )
            return False

    # Check for extreme aspect ratios
    coord_ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    if np.any(coord_ranges > 0):
        non_zero_ranges = coord_ranges[coord_ranges > 0]
        if len(non_zero_ranges) > 1:
            max_range = np.max(non_zero_ranges)
            min_range = np.min(non_zero_ranges)
            aspect_ratio = max_range / min_range

            if aspect_ratio > 1000:  # Aggressive threshold
                print(
                    f"WARNING: Polyhedron {label_id} has extreme aspect ratio ({aspect_ratio:.1f}). Filtering out..."
                )
                return False

    return True


# Helper function for multiprocessing
def _global_mesh_task(
    segmentation_instance,
    labeled_grid,
    label_id,
    smoothing_iterations,
    decimation_ratio,
    use_sdf,
    coordinate_sanity_check=True,
    coordinate_validation_threshold=1e4,
):
    polyhedron_size = np.sum(labeled_grid == label_id)
    mesh_data = segmentation_instance.extract_polyhedron_mesh(
        labeled_grid,
        label_id,
        smoothing_iterations,
        decimation_ratio,
        use_sdf,
        coordinate_sanity_check,
        coordinate_validation_threshold,
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

    def _create_overlapping_chunks(
        self, grid_shape: Tuple[int, int, int], chunk_size: Tuple[int, int, int], overlap: int
    ) -> List[Tuple[slice, ...]]:
        """
        Create overlapping chunks for large grid processing.

        Args:
            grid_shape: Shape of the full grid (z, y, x)
            chunk_size: Size of each chunk (z, y, x)
            overlap: Overlap size in voxels

        Returns:
            List of slice tuples defining each chunk
        """
        chunks = []
        z_max, y_max, x_max = grid_shape
        chunk_z, chunk_y, chunk_x = chunk_size

        # Calculate step sizes (chunk size minus overlap)
        step_z = max(1, chunk_z - overlap)
        step_y = max(1, chunk_y - overlap)
        step_x = max(1, chunk_x - overlap)

        z_starts = list(range(0, z_max, step_z))
        y_starts = list(range(0, y_max, step_y))
        x_starts = list(range(0, x_max, step_x))

        # Ensure we don't miss the end of the grid
        if z_starts[-1] + chunk_z < z_max:
            z_starts.append(z_max - chunk_z)
        if y_starts[-1] + chunk_y < y_max:
            y_starts.append(y_max - chunk_y)
        if x_starts[-1] + chunk_x < x_max:
            x_starts.append(x_max - chunk_x)

        for z_start in z_starts:
            for y_start in y_starts:
                for x_start in x_starts:
                    z_end = min(z_start + chunk_z, z_max)
                    y_end = min(y_start + chunk_y, y_max)
                    x_end = min(x_start + chunk_x, x_max)

                    chunk_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
                    chunks.append(chunk_slice)

        return chunks

    def _process_chunk(
        self,
        binary_chunk: np.ndarray,
        chunk_slice: Tuple[slice, ...],
        segmentation_method: str,
        segment_params: Dict,
        chunk_id: int = 0,
    ) -> Dict:
        """
        Process a single chunk of the grid.

        Args:
            binary_chunk: Binary chunk data
            chunk_slice: Slice defining chunk position in full grid
            segmentation_method: Segmentation method to use
            segment_params: Parameters for segmentation
            chunk_id: Unique identifier for this chunk

        Returns:
            Dictionary containing labeled chunk and metadata
        """
        print(f"Processing chunk {chunk_id} with shape {binary_chunk.shape}")

        # Skip empty chunks
        if not np.any(binary_chunk):
            return {
                "labels": np.zeros_like(binary_chunk, dtype=np.int32),
                "num_labels": 0,
                "chunk_slice": chunk_slice,
                "chunk_id": chunk_id,
            }

        # Segment this chunk
        labels, num_labels = self.segment_polyhedrons(binary_chunk, method=segmentation_method, **segment_params)

        return {
            "labels": labels.astype(np.int32),
            "num_labels": num_labels,
            "chunk_slice": chunk_slice,
            "chunk_id": chunk_id,
        }

    def _merge_chunk_results(
        self, chunk_results: List[Dict], grid_shape: Tuple[int, int, int], overlap: int, fast_merge: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Merge results from overlapping chunks with optimized performance.

        Args:
            chunk_results: List of chunk processing results
            grid_shape: Shape of the full grid
            overlap: Overlap size used
            fast_merge: Use fast merging strategy (True) or detailed strategy (False)

        Returns:
            Merged labeled array and total number of labels
        """
        print(f"Merging {len(chunk_results)} chunk results with {'fast' if fast_merge else 'detailed'} algorithm...")

        # Initialize full grid
        full_labels = np.zeros(grid_shape, dtype=np.int32)

        # Global label counter
        global_label_id = 1

        # Sort chunks by number of labels (process chunks with more labels first)
        # This gives priority to chunks with more content
        chunk_results = sorted(chunk_results, key=lambda x: x["num_labels"], reverse=True)

        # Process each chunk with simplified overlap resolution
        for i, chunk_result in enumerate(chunk_results):
            if chunk_result["num_labels"] == 0:
                continue

            chunk_labels = chunk_result["labels"]
            chunk_slice = chunk_result["chunk_slice"]

            # Create mapping for this chunk
            local_labels = np.unique(chunk_labels)
            local_labels = local_labels[local_labels > 0]  # Exclude background

            if len(local_labels) == 0:
                continue

            # Create efficient mapping
            chunk_mapping = {}
            for local_label in local_labels:
                chunk_mapping[local_label] = global_label_id
                global_label_id += 1

            # Apply mapping efficiently using vectorized operations
            mapped_labels = np.zeros_like(chunk_labels, dtype=np.int32)
            for local_label, global_label in chunk_mapping.items():
                mapped_labels[chunk_labels == local_label] = global_label

            # Get the region in the full grid
            grid_region = full_labels[chunk_slice]

            # Optimized overlap resolution using vectorized operations
            if i == 0:
                # First chunk - just place it
                grid_region[mapped_labels > 0] = mapped_labels[mapped_labels > 0]
            else:
                # For subsequent chunks, use efficient overlap resolution
                self._resolve_overlap_vectorized(grid_region, mapped_labels, overlap, fast_merge)

        # Efficient label compaction using numpy operations
        final_labels, final_count = self._compact_labels_efficient(full_labels)

        print(f"Merged chunks: {final_count} total labels after efficient compaction")
        return final_labels, final_count

    def _resolve_overlap_vectorized(
        self, grid_region: np.ndarray, mapped_labels: np.ndarray, overlap: int, fast_merge: bool
    ):
        """
        Efficiently resolve overlaps between grid region and new mapped labels using vectorized operations.

        Strategy: Use "largest connected component wins" principle with efficient computation.
        """
        # Find overlap regions (where both have non-zero values)
        existing_mask = grid_region > 0
        new_mask = mapped_labels > 0
        overlap_mask = existing_mask & new_mask

        if not np.any(overlap_mask):
            # No overlap - simple assignment
            grid_region[new_mask] = mapped_labels[new_mask]
            return

        # Count total overlap voxels
        overlap_count = np.sum(overlap_mask)

        # Use different strategies based on overlap size and user preference
        if fast_merge or overlap_count > 10000:  # For large overlaps or when fast mode is requested
            self._resolve_overlap_fast(grid_region, mapped_labels, existing_mask, new_mask, overlap_mask)
        else:
            self._resolve_overlap_detailed(grid_region, mapped_labels, existing_mask, new_mask, overlap_mask)

    def _resolve_overlap_fast(
        self,
        grid_region: np.ndarray,
        mapped_labels: np.ndarray,
        existing_mask: np.ndarray,
        new_mask: np.ndarray,
        overlap_mask: np.ndarray,
    ):
        """
        Fast overlap resolution using simple heuristics for large overlap regions.
        """
        # Simple strategy: for each unique label pair in overlap, keep the one with more total volume
        overlap_existing_labels = grid_region[overlap_mask]
        overlap_new_labels = mapped_labels[overlap_mask]

        # Get unique pairs
        unique_existing = np.unique(overlap_existing_labels)
        unique_new = np.unique(overlap_new_labels)

        # For each new label, decide whether to keep it or not based on global volume
        for new_label in unique_new:
            new_label_mask = mapped_labels == new_label
            new_total_volume = np.sum(new_label_mask)

            # Find which existing labels it conflicts with
            conflict_mask = overlap_mask & new_label_mask
            if not np.any(conflict_mask):
                continue

            conflicting_existing_labels = np.unique(grid_region[conflict_mask])

            # For each conflicting existing label, compare volumes
            keep_new = True
            for existing_label in conflicting_existing_labels:
                if existing_label == 0:
                    continue
                existing_total_volume = np.sum(grid_region == existing_label)

                # If existing label has significantly more volume, keep it
                if existing_total_volume > new_total_volume * 1.2:  # 20% threshold
                    keep_new = False
                    break

            if keep_new:
                # Keep the new label in overlap regions
                update_mask = new_label_mask
                grid_region[update_mask] = mapped_labels[update_mask]

        # Assign non-overlapping new regions
        non_overlap_new = new_mask & (~overlap_mask)
        grid_region[non_overlap_new] = mapped_labels[non_overlap_new]

    def _resolve_overlap_detailed(
        self,
        grid_region: np.ndarray,
        mapped_labels: np.ndarray,
        existing_mask: np.ndarray,
        new_mask: np.ndarray,
        overlap_mask: np.ndarray,
    ):
        """
        Detailed overlap resolution using local neighborhood support for smaller overlap regions.
        """
        # Create a decision mask: True = keep new label, False = keep existing
        decision_mask = np.zeros(overlap_mask.shape, dtype=bool)

        # Get overlap coordinates for efficient processing
        overlap_coords = np.where(overlap_mask)

        if len(overlap_coords[0]) > 0:
            # Batch process overlap coordinates
            existing_labels_at_overlap = grid_region[overlap_coords]
            new_labels_at_overlap = mapped_labels[overlap_coords]

            # For each overlap point, do a quick local support count
            for i in range(len(overlap_coords[0])):
                z, y, x = overlap_coords[0][i], overlap_coords[1][i], overlap_coords[2][i]

                existing_label = existing_labels_at_overlap[i]
                new_label = new_labels_at_overlap[i]

                # Quick local support check (3x3x3 neighborhood)
                z_start, z_end = max(0, z - 1), min(grid_region.shape[0], z + 2)
                y_start, y_end = max(0, y - 1), min(grid_region.shape[1], y + 2)
                x_start, x_end = max(0, x - 1), min(grid_region.shape[2], x + 2)

                existing_support = np.sum(grid_region[z_start:z_end, y_start:y_end, x_start:x_end] == existing_label)
                new_support = np.sum(mapped_labels[z_start:z_end, y_start:y_end, x_start:x_end] == new_label)

                # Keep new label if it has better or equal local support
                if new_support >= existing_support:
                    decision_mask[z, y, x] = True

        # Apply decisions efficiently
        update_mask = (~existing_mask) | (overlap_mask & decision_mask)
        grid_region[update_mask & new_mask] = mapped_labels[update_mask & new_mask]

    def _compact_labels_efficient(self, labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Efficiently compact labels to remove gaps using vectorized operations.
        """
        # Find unique non-zero labels
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return labels, 0

        # Check if labels are already compact
        if len(unique_labels) == unique_labels[-1] and unique_labels[0] == 1:
            # Already compact
            return labels, len(unique_labels)

        # Create compaction mapping efficiently
        compacted_labels = np.zeros_like(labels)

        # Use vectorized approach for compaction
        for new_id, old_id in enumerate(unique_labels, 1):
            compacted_labels[labels == old_id] = new_id

        return compacted_labels, len(unique_labels)

    def process_voxel_grid_chunked(
        self,
        voxel_grid: np.ndarray,
        chunk_size: Tuple[int, int, int] = (256, 256, 256),
        overlap: int = 32,
        max_chunk_workers: int = None,
        fast_merge: bool = True,  # New parameter for merge strategy
        **kwargs,
    ) -> Dict:
        """
        Process large voxel grids using chunking strategy for improved performance.

        Args:
            voxel_grid: Input voxel grid
            chunk_size: Size of each chunk (z, y, x)
            overlap: Overlap size in voxels between chunks
            max_chunk_workers: Maximum workers for chunk processing (None = auto)
            fast_merge: Use fast merging strategy for better performance (default: True)
            **kwargs: All other parameters from process_voxel_grid

        Returns:
            Same format as process_voxel_grid but processed in chunks
        """
        print("=" * 60)
        print("CHUNKED POLYHEDRON SEGMENTATION PIPELINE")
        print(f"Grid shape: {voxel_grid.shape}")
        print(f"Chunk size: {chunk_size}")
        print(f"Overlap: {overlap}")
        print("=" * 60)

        # Extract parameters
        segmentation_method = kwargs.get("segmentation_method", "watershed")

        # Step 1: Preprocess the entire grid first
        preprocess_params = {
            "binary_threshold": kwargs.get("binary_threshold", 0.5),
            "gaussian_sigma": kwargs.get("gaussian_sigma", 0.5),
            "remove_small_objects_min_size": kwargs.get("remove_small_objects_min_size", 50),
        }

        print("Preprocessing full grid...")
        binary_grid = self.preprocess_voxel_grid(voxel_grid, **preprocess_params)

        # Step 2: Create chunks
        chunks = self._create_overlapping_chunks(binary_grid.shape, chunk_size, overlap)
        print(f"Created {len(chunks)} overlapping chunks")

        # Step 3: Prepare segmentation parameters
        segment_params = {
            "min_distance": kwargs.get("min_distance", 5),
            "erosion_iterations": kwargs.get("erosion_iterations", 1),
            "sdf_scale": kwargs.get("sdf_scale", 5.0),
            "marker_threshold_percentile": kwargs.get("marker_threshold_percentile", 95.0),
            "gaussian_smooth_dt_sigma": kwargs.get("gaussian_smooth_dt_sigma", 1.0),
            "peak_local_max_footprint_size": kwargs.get("peak_local_max_footprint_size", 3),
        }

        # Step 4: Process chunks
        chunk_workers = max_chunk_workers or min(len(chunks), os.cpu_count())
        chunk_results = []

        if chunk_workers > 1 and len(chunks) > 1:
            print(f"Processing chunks in parallel with {chunk_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_workers) as executor:
                futures = []
                for i, chunk_slice in enumerate(chunks):
                    binary_chunk = binary_grid[chunk_slice]
                    future = executor.submit(
                        self._process_chunk, binary_chunk, chunk_slice, segmentation_method, segment_params, i
                    )
                    futures.append(future)

                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(chunks), desc="Processing chunks"
                ):
                    try:
                        result = future.result()
                        chunk_results.append(result)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
        else:
            print("Processing chunks sequentially...")
            for i, chunk_slice in enumerate(tqdm(chunks, desc="Processing chunks")):
                binary_chunk = binary_grid[chunk_slice]
                result = self._process_chunk(binary_chunk, chunk_slice, segmentation_method, segment_params, i)
                chunk_results.append(result)

        # Step 5: Merge chunk results
        labeled_grid, num_labels = self._merge_chunk_results(chunk_results, binary_grid.shape, overlap, fast_merge)

        if num_labels == 0:
            print("No polyhedrons found after chunked segmentation!")
            return {"polyhedrons": {}, "metadata": {"total_count": 0, "chunked": True}}

        # Step 6: Continue with standard pipeline for filtering and mesh extraction
        print(f"Continuing with standard pipeline for {num_labels} labels...")

        # Use the existing process_voxel_grid logic for the rest of the pipeline
        # but skip the preprocessing and segmentation steps
        kwargs_copy = kwargs.copy()
        kwargs_copy["_skip_preprocessing"] = True
        kwargs_copy["_skip_segmentation"] = True
        kwargs_copy["_provided_labels"] = labeled_grid
        kwargs_copy["_provided_num_labels"] = num_labels

        return self._process_labeled_grid(binary_grid, labeled_grid, num_labels, **kwargs_copy)

    def _process_labeled_grid(
        self, binary_grid: np.ndarray, labeled_grid: np.ndarray, num_labels: int, **kwargs
    ) -> Dict:
        """
        Process an already-labeled grid through the filtering and mesh extraction pipeline.
        This is used by the chunked processing to avoid duplicating the post-segmentation logic.
        """
        # Extract parameters (same as in process_voxel_grid)
        min_polyhedron_size = kwargs.get("min_polyhedron_size", 100)
        smoothing_iterations = kwargs.get("smoothing_iterations", 10)
        decimation_ratio = kwargs.get("decimation_ratio", 0.8)
        num_workers = kwargs.get("num_workers", 1)
        use_sdf = kwargs.get("use_sdf", True)
        remove_boundary_polyhedrons = kwargs.get("remove_boundary_polyhedrons", True)
        max_voxel_aspect_ratio = kwargs.get("max_voxel_aspect_ratio", 20.0)
        coordinate_validation_threshold = kwargs.get("coordinate_validation_threshold", 1e4)
        enable_mesh_range_outlier_filter = kwargs.get("enable_mesh_range_outlier_filter", True)
        mesh_range_outlier_iqr_factor = kwargs.get("mesh_range_outlier_iqr_factor", 1.5)
        mesh_range_outlier_median_factor = kwargs.get("mesh_range_outlier_median_factor", 100.0)

        # NEW: Fast mesh extraction options
        fast_mesh_extraction = kwargs.get("fast_mesh_extraction", True)
        batch_mesh_size = kwargs.get("batch_mesh_size", 10)
        skip_sdf_for_small = kwargs.get("skip_sdf_for_small", True)
        small_polyhedron_threshold = kwargs.get("small_polyhedron_threshold", 500)
        reduce_smoothing_for_small = kwargs.get("reduce_smoothing_for_small", True)

        # Step 1: Filter polyhedrons by size and optionally remove boundary polyhedrons
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

        # Aspect Ratio Filtering
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
                    "chunked": True,
                },
            }

        # Step 2: Extract individual polyhedrons with optimizations
        print(f"Extracting {len(label_ids_to_process)} polyhedrons...")
        if fast_mesh_extraction:
            print(
                f"Using fast mesh extraction (batch_size={batch_mesh_size}, skip_sdf_for_small={skip_sdf_for_small})"
            )

        candidate_polyhedron_mesh_data_list = []

        if fast_mesh_extraction:
            # Use optimized batch processing
            candidate_polyhedron_mesh_data_list = self._extract_polyhedrons_fast(
                labeled_grid,
                label_ids_to_process,
                smoothing_iterations,
                decimation_ratio,
                use_sdf,
                coordinate_validation_threshold,
                batch_mesh_size,
                skip_sdf_for_small,
                small_polyhedron_threshold,
                reduce_smoothing_for_small,
                num_workers,
            )
        else:
            # Use original method
            if num_workers > 1 and len(label_ids_to_process) > 1:
                print(f"Using {num_workers} parallel workers for mesh extraction...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            _global_mesh_task,
                            self,
                            labeled_grid,
                            label_id,
                            smoothing_iterations,
                            decimation_ratio,
                            use_sdf,
                            True,  # coordinate_sanity_check
                            coordinate_validation_threshold,
                        )
                        for label_id in label_ids_to_process
                    ]
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(label_ids_to_process),
                        desc="Extracting polyhedrons (parallel)",
                    ):
                        try:
                            label_id, polyhedron_size, mesh_data_item = future.result()
                            if mesh_data_item and len(mesh_data_item.get("vertices", [])) > 0:
                                candidate_polyhedron_mesh_data_list.append(
                                    {"polyhedron_size": polyhedron_size, **mesh_data_item}
                                )
                        except Exception as e:
                            print(f"Error processing future for a label: {e}")
            else:
                if num_workers > 1:
                    print("Not enough labels or workers for parallel processing, using sequential.")
                for label_id in tqdm(label_ids_to_process, desc="Extracting polyhedrons (sequential)"):
                    polyhedron_size = np.sum(labeled_grid == label_id)
                    mesh_data_item = self.extract_polyhedron_mesh(
                        labeled_grid,
                        label_id,
                        smoothing_iterations,
                        decimation_ratio,
                        use_sdf,
                        True,  # coordinate_sanity_check
                        coordinate_validation_threshold,
                    )
                    if mesh_data_item and len(mesh_data_item.get("vertices", [])) > 0:
                        candidate_polyhedron_mesh_data_list.append(
                            {"polyhedron_size": polyhedron_size, **mesh_data_item}
                        )

        print(f"  Collected {len(candidate_polyhedron_mesh_data_list)} candidate meshes after initial extraction.")

        # Step 3: Apply mesh range outlier filter (same as original)
        final_polyhedrons_list = []
        if enable_mesh_range_outlier_filter and candidate_polyhedron_mesh_data_list:
            print(
                f"Applying global mesh range outlier filter (IQR factor: {mesh_range_outlier_iqr_factor}, Median factor: {mesh_range_outlier_median_factor})..."
            )

            all_mesh_ranges_stats = {"x": [], "y": [], "z": []}
            for data in candidate_polyhedron_mesh_data_list:
                ranges = data.get("ranges")
                if ranges and len(ranges) == 3:
                    all_mesh_ranges_stats["x"].append(ranges[0])
                    all_mesh_ranges_stats["y"].append(ranges[1])
                    all_mesh_ranges_stats["z"].append(ranges[2])

            axis_final_thresholds = {}
            for i, axis_name in enumerate(["x", "y", "z"]):
                ranges_for_axis = all_mesh_ranges_stats[axis_name]
                if not ranges_for_axis:
                    axis_final_thresholds[axis_name] = float("inf")
                    print(f"    No range data for axis {axis_name}, effectively disabling filter for it.")
                    continue

                q75 = np.percentile(ranges_for_axis, 75)
                q25 = np.percentile(ranges_for_axis, 25)
                iqr = q75 - q25
                iqr = max(iqr, 0)

                upper_bound_iqr = q75 + mesh_range_outlier_iqr_factor * iqr

                median_val = np.median(ranges_for_axis)
                if median_val == 0 and mesh_range_outlier_median_factor > 0:
                    if np.all(np.array(ranges_for_axis) == 0):
                        abs_threshold_median = 0
                    else:
                        abs_threshold_median = float("inf")
                        print(
                            f"    Median for axis {axis_name} is 0, but non-zero ranges exist. Median factor threshold disabled for this axis."
                        )
                else:
                    abs_threshold_median = median_val * mesh_range_outlier_median_factor

                current_axis_threshold = min(upper_bound_iqr, abs_threshold_median)
                axis_final_thresholds[axis_name] = current_axis_threshold
                print(
                    f"    Axis {axis_name}: Q25={q25:.2e}, Q75={q75:.2e}, IQR={iqr:.2e}, IQR_thresh={upper_bound_iqr:.2e}, Med_thresh={abs_threshold_median:.2e} => Final Threshold={current_axis_threshold:.2e}"
                )

            range_outlier_removed_count = 0
            for poly_data in candidate_polyhedron_mesh_data_list:
                current_poly_ranges = poly_data.get("ranges")
                poly_label_id = poly_data.get("label_id", "Unknown")
                is_outlier = False
                if current_poly_ranges and len(current_poly_ranges) == 3:
                    for i, axis_name in enumerate(["x", "y", "z"]):
                        if current_poly_ranges[i] > axis_final_thresholds[axis_name]:
                            print(
                                f"    Filtering out polyhedron {poly_label_id} due to range outlier on axis {axis_name}: {current_poly_ranges[i]:.2e} > {axis_final_thresholds[axis_name]:.2e}"
                            )
                            is_outlier = True
                            range_outlier_removed_count += 1
                            break

                if not is_outlier:
                    final_polyhedrons_list.append(poly_data)
            print(f"  Removed {range_outlier_removed_count} polyhedrons due to global mesh range outlier filter.")
        else:
            final_polyhedrons_list = candidate_polyhedron_mesh_data_list
            if not enable_mesh_range_outlier_filter:
                print("  Global mesh range outlier filter is disabled.")

        # Step 4: Create final polyhedrons dictionary (same as original)
        polyhedrons = {}
        for poly_data_item in final_polyhedrons_list:
            label_id = poly_data_item["label_id"]
            polyhedrons[str(label_id)] = {
                "id": label_id,
                "vertices": poly_data_item["vertices"],
                "faces": poly_data_item["faces"],
                "volume": poly_data_item.get("volume", 0),
                "n_vertices": poly_data_item.get("n_vertices", 0),
                "n_faces": poly_data_item.get("n_faces", 0),
                "voxel_count": int(poly_data_item.get("polyhedron_size", 0)),
                "centroid": self._calculate_centroid(poly_data_item["vertices"]),
                "bounding_box": self._calculate_bounding_box(poly_data_item["vertices"]),
                "ranges": poly_data_item.get("ranges"),
            }

        total_final_polyhedrons = len(polyhedrons)
        print(f"Total polyhedrons extracted after all filters: {total_final_polyhedrons}")

        # Metadata
        segmentation_method = kwargs.get("segmentation_method", "watershed")
        metadata = {
            "total_count": len(polyhedrons),
            "original_labels": num_labels,
            "min_size_filtered_count": len(candidate_label_ids),
            "boundary_removed_count": boundary_removed_count,
            "aspect_ratio_removed_count": aspect_ratio_removed_count,
            "final_extracted_count": len(polyhedrons),
            "voxel_spacing": self.voxel_spacing.tolist(),
            "origin": self.origin.tolist(),
            "grid_shape": binary_grid.shape,
            "segmentation_method": segmentation_method,
            "chunked": True,
            "processing_parameters": {
                "min_polyhedron_size": min_polyhedron_size,
                "smoothing_iterations": smoothing_iterations,
                "decimation_ratio": decimation_ratio,
                "use_sdf_mesh_extraction": use_sdf,
                "remove_boundary_polyhedrons": remove_boundary_polyhedrons,
                "max_voxel_aspect_ratio": max_voxel_aspect_ratio,
                "coordinate_validation_threshold": coordinate_validation_threshold,
                "enable_mesh_range_outlier_filter": enable_mesh_range_outlier_filter,
                "mesh_range_outlier_iqr_factor": mesh_range_outlier_iqr_factor,
                "mesh_range_outlier_median_factor": mesh_range_outlier_median_factor,
                "fast_mesh_extraction": fast_mesh_extraction,
            },
        }

        print(f"Successfully extracted {len(polyhedrons)} polyhedrons using chunked {segmentation_method}")
        return {"polyhedrons": polyhedrons, "metadata": metadata}

    def _extract_polyhedrons_fast(
        self,
        labeled_grid: np.ndarray,
        label_ids: List[int],
        smoothing_iterations: int,
        decimation_ratio: float,
        use_sdf: bool,
        coordinate_validation_threshold: float,
        batch_size: int = 10,
        skip_sdf_for_small: bool = True,
        small_threshold: int = 500,
        reduce_smoothing_for_small: bool = True,
        num_workers: int = 1,
    ) -> List[Dict]:
        """
        Fast batch mesh extraction with multiple optimizations.

        Returns:
            List of mesh data dictionaries
        """
        print(f"Starting fast mesh extraction for {len(label_ids)} polyhedrons")

        # Group polyhedrons by size for differential processing
        small_polyhedrons = []
        large_polyhedrons = []

        for label_id in label_ids:
            polyhedron_size = np.sum(labeled_grid == label_id)
            if polyhedron_size < small_threshold:
                small_polyhedrons.append((label_id, polyhedron_size))
            else:
                large_polyhedrons.append((label_id, polyhedron_size))

        print(f"  Small polyhedrons (< {small_threshold} voxels): {len(small_polyhedrons)}")
        print(f"  Large polyhedrons (>= {small_threshold} voxels): {len(large_polyhedrons)}")

        results = []

        # Process small polyhedrons with optimizations
        if small_polyhedrons:
            print(f"Processing {len(small_polyhedrons)} small polyhedrons with optimizations...")
            small_results = self._extract_polyhedrons_batch(
                labeled_grid,
                small_polyhedrons,
                smoothing_iterations=max(1, smoothing_iterations // 3)
                if reduce_smoothing_for_small
                else smoothing_iterations,
                decimation_ratio=max(0.5, decimation_ratio),  # More aggressive decimation for small ones
                use_sdf=False if skip_sdf_for_small else use_sdf,  # Skip SDF for small polyhedrons
                coordinate_validation_threshold=coordinate_validation_threshold,
                batch_size=batch_size,
                num_workers=num_workers,
                fast_mode=True,
            )
            results.extend(small_results)

        # Process large polyhedrons normally but in batches
        if large_polyhedrons:
            print(f"Processing {len(large_polyhedrons)} large polyhedrons with normal quality...")
            large_results = self._extract_polyhedrons_batch(
                labeled_grid,
                large_polyhedrons,
                smoothing_iterations=smoothing_iterations,
                decimation_ratio=decimation_ratio,
                use_sdf=use_sdf,
                coordinate_validation_threshold=coordinate_validation_threshold,
                batch_size=max(1, batch_size // 2),  # Smaller batches for large polyhedrons
                num_workers=num_workers,
                fast_mode=False,
            )
            results.extend(large_results)

        print(f"Fast mesh extraction completed. Processed {len(results)} polyhedrons successfully.")
        return results

    def _extract_polyhedrons_batch(
        self,
        labeled_grid: np.ndarray,
        polyhedron_list: List[Tuple[int, int]],  # (label_id, size)
        smoothing_iterations: int,
        decimation_ratio: float,
        use_sdf: bool,
        coordinate_validation_threshold: float,
        batch_size: int,
        num_workers: int,
        fast_mode: bool = False,
    ) -> List[Dict]:
        """
        Process polyhedrons in batches for better efficiency.
        """
        all_results = []

        # Process in batches
        for i in tqdm(range(0, len(polyhedron_list), batch_size), desc="Processing batches"):
            batch = polyhedron_list[i : i + batch_size]

            if num_workers > 1 and len(batch) > 1:
                # Parallel processing within batch
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(batch))) as executor:
                    futures = []
                    for label_id, polyhedron_size in batch:
                        future = executor.submit(
                            self._extract_single_polyhedron_optimized,
                            labeled_grid,
                            label_id,
                            polyhedron_size,
                            smoothing_iterations,
                            decimation_ratio,
                            use_sdf,
                            coordinate_validation_threshold,
                            fast_mode,
                        )
                        futures.append(future)

                    # Collect results
                    for future in futures:
                        try:
                            result = future.result()
                            if result and len(result.get("vertices", [])) > 0:
                                all_results.append(result)
                        except Exception as e:
                            print(f"    Error in batch processing: {e}")
            else:
                # Sequential processing within batch
                for label_id, polyhedron_size in batch:
                    try:
                        result = self._extract_single_polyhedron_optimized(
                            labeled_grid,
                            label_id,
                            polyhedron_size,
                            smoothing_iterations,
                            decimation_ratio,
                            use_sdf,
                            coordinate_validation_threshold,
                            fast_mode,
                        )
                        if result and len(result.get("vertices", [])) > 0:
                            all_results.append(result)
                    except Exception as e:
                        print(f"    Error processing polyhedron {label_id}: {e}")

        return all_results

    def _extract_single_polyhedron_optimized(
        self,
        labeled_grid: np.ndarray,
        label_id: int,
        polyhedron_size: int,
        smoothing_iterations: int,
        decimation_ratio: float,
        use_sdf: bool,
        coordinate_validation_threshold: float,
        fast_mode: bool = False,
    ) -> Dict:
        """
        Optimized single polyhedron extraction with reduced overhead.
        """
        # Extract single polyhedron with minimal copies
        single_polyhedron = labeled_grid == label_id

        if not np.any(single_polyhedron):
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0, "label_id": label_id}

        try:
            if use_sdf:
                # Compute SDF only for the bounding box region to save computation
                coords = np.argwhere(single_polyhedron)
                min_coords = np.max([np.min(coords, axis=0) - 2, [0, 0, 0]], axis=0)
                max_coords = np.min([np.max(coords, axis=0) + 3, single_polyhedron.shape], axis=0)

                # Extract minimal region
                region_slice = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
                region_polyhedron = single_polyhedron[region_slice]

                if np.any(region_polyhedron):
                    # Compute SDF on smaller region
                    sdf = self.compute_sdf(region_polyhedron, scale=3.0)

                    # Create PyVista grid for smaller region
                    region_origin = self.origin + min_coords * self.voxel_spacing
                    grid = pv.ImageData(dimensions=sdf.shape, spacing=self.voxel_spacing, origin=region_origin)
                    grid.point_data["values"] = sdf.flatten(order="F")

                    # Extract isosurface
                    mesh = grid.contour(isosurfaces=[0.0])
                else:
                    return {
                        "vertices": [],
                        "faces": [],
                        "volume": 0,
                        "n_vertices": 0,
                        "n_faces": 0,
                        "label_id": label_id,
                    }
            else:
                # Binary approach - also use bounding box optimization
                coords = np.argwhere(single_polyhedron)
                min_coords = np.max([np.min(coords, axis=0) - 1, [0, 0, 0]], axis=0)
                max_coords = np.min([np.max(coords, axis=0) + 2, single_polyhedron.shape], axis=0)

                region_slice = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
                region_polyhedron = single_polyhedron[region_slice].astype(np.uint8)

                region_origin = self.origin + min_coords * self.voxel_spacing
                grid = pv.ImageData(
                    dimensions=region_polyhedron.shape, spacing=self.voxel_spacing, origin=region_origin
                )
                grid.point_data["values"] = region_polyhedron.flatten(order="F")

                mesh = grid.contour(isosurfaces=[0.5])

            if mesh.n_points == 0 or mesh.n_cells == 0:
                return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0, "label_id": label_id}

            # Fast coordinate validation (single check instead of multiple)
            if not fast_mode:
                if not validate_mesh_coordinates(mesh.points, label_id, coordinate_validation_threshold):
                    return {
                        "vertices": [],
                        "faces": [],
                        "volume": 0,
                        "n_vertices": 0,
                        "n_faces": 0,
                        "skipped_reason": "failed_coordinate_validation",
                        "label_id": label_id,
                    }

            # Apply processing with reduced iterations for fast mode
            actual_smoothing = max(1, smoothing_iterations // 2) if fast_mode else smoothing_iterations
            if actual_smoothing > 0:
                mesh = mesh.smooth(n_iter=actual_smoothing, relaxation_factor=0.1)

            # Decimation
            if 0 < decimation_ratio < 1:
                mesh = mesh.decimate(decimation_ratio)

            if mesh.n_points == 0 or mesh.n_cells == 0:
                return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0, "label_id": label_id}

            # Extract vertices and faces efficiently
            vertices = mesh.points
            faces_list = []
            raw_faces = mesh.faces
            i = 0
            while i < len(raw_faces):
                num_points_in_face = raw_faces[i]
                faces_list.append(raw_faces[i + 1 : i + 1 + num_points_in_face].tolist())
                i += num_points_in_face + 1

            # Calculate volume and ranges
            volume = mesh.volume if hasattr(mesh, "volume") else 0
            coord_ranges = np.zeros(3)
            if vertices.shape[0] > 0:
                coord_ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)

            return {
                "vertices": vertices.tolist(),
                "faces": faces_list,
                "volume": volume,
                "n_vertices": mesh.n_points,
                "n_faces": mesh.n_cells,
                "ranges": coord_ranges.tolist(),
                "label_id": label_id,
                "polyhedron_size": polyhedron_size,
            }

        except Exception as e:
            print(f"Error extracting polyhedron {label_id}: {e}")
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0, "label_id": label_id}

    def extract_polyhedron_mesh(
        self,
        labeled_grid: np.ndarray,
        label_id: int,
        smoothing_iterations: int = 10,
        decimation_ratio: float = 0.9,
        use_sdf: bool = True,
        coordinate_sanity_check: bool = True,
        max_reasonable_coord: float = 1e4,  # Keep this as the parameter name for consistency
    ) -> Dict:
        """
        Extract mesh (vertices and faces) for a specific labeled polyhedron using SDF.

        Args:
            labeled_grid: Labeled segmentation array
            label_id: ID of the specific polyhedron to extract
            smoothing_iterations: Number of smoothing iterations to apply
            decimation_ratio: Ratio for mesh decimation (0-1, higher = more decimation)
            use_sdf: Whether to use SDF for better boundary detection
            coordinate_sanity_check: Whether to validate coordinates for sanity
            max_reasonable_coord: Maximum reasonable coordinate value for aggressive validation

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

        # AGGRESSIVE COORDINATE VALIDATION - Check immediately after mesh creation
        if coordinate_sanity_check:
            if not validate_mesh_coordinates(mesh.points, label_id, max_reasonable_coord):
                return {
                    "vertices": [],
                    "faces": [],
                    "volume": 0,
                    "n_vertices": 0,
                    "n_faces": 0,
                    "skipped_reason": "failed_initial_coordinate_validation",
                }

        # Apply smoothing to reduce voxel artifacts
        if smoothing_iterations > 0:
            mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)

            # Re-check coordinates after smoothing
            if coordinate_sanity_check:
                if not validate_mesh_coordinates(mesh.points, label_id, max_reasonable_coord):
                    return {
                        "vertices": [],
                        "faces": [],
                        "volume": 0,
                        "n_vertices": 0,
                        "n_faces": 0,
                        "skipped_reason": "failed_post_smoothing_validation",
                    }

        # Apply decimation to reduce polygon count
        if 0 < decimation_ratio < 1:
            target_reduction = decimation_ratio
            mesh = mesh.decimate(target_reduction)

            # Re-check coordinates after decimation
            if coordinate_sanity_check:
                if not validate_mesh_coordinates(mesh.points, label_id, max_reasonable_coord):
                    return {
                        "vertices": [],
                        "faces": [],
                        "volume": 0,
                        "n_vertices": 0,
                        "n_faces": 0,
                        "skipped_reason": "failed_post_decimation_validation",
                    }

        if mesh.n_points == 0 or mesh.n_cells == 0:  # Check again after processing
            return {"vertices": [], "faces": [], "volume": 0, "n_vertices": 0, "n_faces": 0}

        # Final coordinate validation before returning
        if coordinate_sanity_check:
            if not validate_mesh_coordinates(mesh.points, label_id, max_reasonable_coord):
                return {
                    "vertices": [],
                    "faces": [],
                    "volume": 0,
                    "n_vertices": 0,
                    "n_faces": 0,
                    "skipped_reason": "failed_final_validation",
                }

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

        # Calculate coordinate ranges
        coord_ranges = np.zeros(3)
        if vertices.shape[0] > 0:
            coord_ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)

        return {
            "vertices": vertices.tolist(),
            "faces": faces_list,
            "volume": volume,
            "n_vertices": mesh.n_points,
            "n_faces": mesh.n_cells,
            "ranges": coord_ranges.tolist(),
            "label_id": label_id,
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
        coordinate_validation_threshold: float = 1e4,  # New parameter
        # New parameters for mesh range outlier filter
        enable_mesh_range_outlier_filter: bool = True,
        mesh_range_outlier_iqr_factor: float = 1.5,
        mesh_range_outlier_median_factor: float = 100.0,
        # Fast mesh extraction options
        fast_mesh_extraction: bool = True,
        batch_mesh_size: int = 10,
        skip_sdf_for_small: bool = True,
        small_polyhedron_threshold: int = 500,
        reduce_smoothing_for_small: bool = True,
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
            marker_threshold_percentile: Percentile threshold for SDF marker detection (for "watershed_sdf")
            gaussian_smooth_dt_sigma: Sigma for smoothing distance transform (for "watershed")
            peak_local_max_footprint_size: Footprint size for peak_local_max (for "watershed")
            use_sdf: Whether to use SDF for mesh extraction
            remove_boundary_polyhedrons: Whether to remove polyhedrons touching the grid boundary
            max_voxel_aspect_ratio: Maximum aspect ratio of voxel bounding box to keep a polyhedron (e.g., 20). Longest_side / shortest_side. Set to 0 or None to disable.
            coordinate_validation_threshold: Maximum reasonable coordinate value for aggressive validation
            other_kwargs: For any other potential future arguments
            enable_mesh_range_outlier_filter: Whether to enable the mesh range outlier filter.
            mesh_range_outlier_iqr_factor: IQR factor for mesh range outlier detection.
            mesh_range_outlier_median_factor: Median factor for mesh range outlier detection.

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
        if fast_mesh_extraction:
            print(
                f"Using fast mesh extraction (batch_size={batch_mesh_size}, skip_sdf_for_small={skip_sdf_for_small})"
            )

        candidate_polyhedron_mesh_data_list = []  # Store (label_id, polyhedron_size, mesh_data)

        if fast_mesh_extraction:
            # Use optimized batch processing
            candidate_polyhedron_mesh_data_list = self._extract_polyhedrons_fast(
                labeled_grid,
                label_ids_to_process,
                smoothing_iterations,
                decimation_ratio,
                use_sdf,
                coordinate_validation_threshold,
                batch_mesh_size,
                skip_sdf_for_small,
                small_polyhedron_threshold,
                reduce_smoothing_for_small,
                num_workers,
            )
        elif num_workers > 1 and len(label_ids_to_process) > 1:
            print(f"Using {num_workers} parallel workers for mesh extraction...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _global_mesh_task,
                        self,
                        labeled_grid,
                        label_id,
                        smoothing_iterations,
                        decimation_ratio,
                        use_sdf,
                        True,  # coordinate_sanity_check
                        coordinate_validation_threshold,
                    )
                    for label_id in label_ids_to_process
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(label_ids_to_process),
                    desc="Extracting polyhedrons (parallel)",
                ):
                    try:
                        label_id, polyhedron_size, mesh_data_item = future.result()
                        if mesh_data_item and len(mesh_data_item.get("vertices", [])) > 0:
                            # mesh_data_item already contains label_id from the modified extract_polyhedron_mesh
                            candidate_polyhedron_mesh_data_list.append(
                                {"polyhedron_size": polyhedron_size, **mesh_data_item}
                            )
                    except Exception as e:
                        processed_label_id = "unknown"
                        # Attempt to get label_id from future if possible, though it's tricky
                        # This part is heuristical as direct access to submitted args isn't straightforward post-submission
                        print(f"Error processing future for a label: {e}")

        else:
            if num_workers > 1:
                print("Not enough labels or workers for parallel processing, using sequential.")
            for label_id in tqdm(label_ids_to_process, desc="Extracting polyhedrons (sequential)"):
                polyhedron_size = np.sum(labeled_grid == label_id)  # Calculate size here for consistency
                mesh_data_item = self.extract_polyhedron_mesh(
                    labeled_grid,
                    label_id,
                    smoothing_iterations,
                    decimation_ratio,
                    use_sdf,
                    True,  # coordinate_sanity_check
                    coordinate_validation_threshold,  # use the renamed parameter here
                )
                if mesh_data_item and len(mesh_data_item.get("vertices", [])) > 0:
                    candidate_polyhedron_mesh_data_list.append({"polyhedron_size": polyhedron_size, **mesh_data_item})

        print(f"  Collected {len(candidate_polyhedron_mesh_data_list)} candidate meshes after initial extraction.")

        # Step 5: Global Mesh Range Outlier Filter
        final_polyhedrons_list = []
        if enable_mesh_range_outlier_filter and candidate_polyhedron_mesh_data_list:
            print(
                f"Applying global mesh range outlier filter (IQR factor: {mesh_range_outlier_iqr_factor}, Median factor: {mesh_range_outlier_median_factor})..."
            )
            all_mesh_ranges_stats = {"x": [], "y": [], "z": []}
            for data in candidate_polyhedron_mesh_data_list:
                ranges = data.get("ranges")
                if ranges and len(ranges) == 3:
                    all_mesh_ranges_stats["x"].append(ranges[0])
                    all_mesh_ranges_stats["y"].append(ranges[1])
                    all_mesh_ranges_stats["z"].append(ranges[2])

            axis_final_thresholds = {}
            for i, axis_name in enumerate(["x", "y", "z"]):
                ranges_for_axis = all_mesh_ranges_stats[axis_name]
                if not ranges_for_axis:
                    axis_final_thresholds[axis_name] = float("inf")  # No filter if no data for this axis
                    print(f"    No range data for axis {axis_name}, effectively disabling filter for it.")
                    continue

                q75 = np.percentile(ranges_for_axis, 75)
                q25 = np.percentile(ranges_for_axis, 25)
                iqr = q75 - q25

                # Handle cases where iqr is zero (e.g. all ranges are the same)
                # If iqr is 0, upper_bound might become q75, which could be too restrictive if all values are identical.
                # A small epsilon or alternative handling might be needed if iqr is very small or zero.
                # For now, if iqr is 0, the iqr_component of the threshold will be 0, relying on median part.
                # Or, if all values are same, iqr is 0, median is that value. abs_threshold = val * factor. upper_bound = val.
                # So final_threshold = val. This would filter all if factor > 1.
                # Let's ensure iqr is not negative (can happen with percentiles in weird data, though unlikely here)
                iqr = max(iqr, 0)

                upper_bound_iqr = q75 + mesh_range_outlier_iqr_factor * iqr

                median_val = np.median(ranges_for_axis)
                # Handle cases where median_val is zero to avoid threshold of 0
                if (
                    median_val == 0 and mesh_range_outlier_median_factor > 0
                ):  # if median is 0, this threshold is 0, which is too aggressive
                    # If median is 0, but other values exist, this factor might be too aggressive.
                    # Fallback for zero median: use mean or a small fraction of max range?
                    # Or simply rely on IQR part if median is 0.
                    # For now, if median is 0, this threshold part becomes 0.
                    # Let's set abs_threshold_median to infinity if median is 0 to effectively disable it
                    # unless ranges_for_axis contains only zeros.
                    if np.all(np.array(ranges_for_axis) == 0):
                        abs_threshold_median = 0  # All ranges are 0, so threshold is 0.
                    else:  # Median is 0, but other non-zero ranges exist
                        abs_threshold_median = float(
                            "inf"
                        )  # Effectively disable median part if median is 0 but not all ranges are 0
                        print(
                            f"    Median for axis {axis_name} is 0, but non-zero ranges exist. Median factor threshold disabled for this axis."
                        )
                else:
                    abs_threshold_median = median_val * mesh_range_outlier_median_factor

                current_axis_threshold = min(upper_bound_iqr, abs_threshold_median)
                axis_final_thresholds[axis_name] = current_axis_threshold
                print(
                    f"    Axis {axis_name}: Q25={q25:.2e}, Q75={q75:.2e}, IQR={iqr:.2e}, IQR_thresh={upper_bound_iqr:.2e}, Med_thresh={abs_threshold_median:.2e} => Final Threshold={current_axis_threshold:.2e}"
                )

            range_outlier_removed_count = 0
            for poly_data in candidate_polyhedron_mesh_data_list:
                current_poly_ranges = poly_data.get("ranges")
                poly_label_id = poly_data.get("label_id", "Unknown")  # Get label_id for logging
                is_outlier = False
                if current_poly_ranges and len(current_poly_ranges) == 3:
                    for i, axis_name in enumerate(["x", "y", "z"]):
                        if current_poly_ranges[i] > axis_final_thresholds[axis_name]:
                            print(
                                f"    Filtering out polyhedron {poly_label_id} due to range outlier on axis {axis_name}: {current_poly_ranges[i]:.2e} > {axis_final_thresholds[axis_name]:.2e}"
                            )
                            is_outlier = True
                            range_outlier_removed_count += 1
                            break

                if not is_outlier:
                    final_polyhedrons_list.append(poly_data)
            print(f"  Removed {range_outlier_removed_count} polyhedrons due to global mesh range outlier filter.")
        else:
            final_polyhedrons_list = candidate_polyhedron_mesh_data_list
            if not enable_mesh_range_outlier_filter:
                print("  Global mesh range outlier filter is disabled.")
            elif not candidate_polyhedron_mesh_data_list:
                print("  No candidate meshes to apply global mesh range outlier filter.")

        # Populate the final polyhedrons dictionary
        polyhedrons = {}
        for poly_data_item in final_polyhedrons_list:
            label_id = poly_data_item["label_id"]
            polyhedrons[str(label_id)] = {
                "id": label_id,
                "vertices": poly_data_item["vertices"],
                "faces": poly_data_item["faces"],
                "volume": poly_data_item.get("volume", 0),
                "n_vertices": poly_data_item.get("n_vertices", 0),
                "n_faces": poly_data_item.get("n_faces", 0),
                "voxel_count": int(poly_data_item.get("polyhedron_size", 0)),  # Make sure polyhedron_size is available
                "centroid": self._calculate_centroid(poly_data_item["vertices"]),
                "bounding_box": self._calculate_bounding_box(poly_data_item["vertices"]),
                "ranges": poly_data_item.get("ranges"),  # Keep ranges in the final output
            }

        total_final_polyhedrons = len(polyhedrons)
        print(f"Total polyhedrons extracted after all filters: {total_final_polyhedrons}")

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
                "coordinate_validation_threshold": coordinate_validation_threshold,
                "enable_mesh_range_outlier_filter": enable_mesh_range_outlier_filter,
                "mesh_range_outlier_iqr_factor": mesh_range_outlier_iqr_factor,
                "mesh_range_outlier_median_factor": mesh_range_outlier_median_factor,
                "fast_mesh_extraction": fast_mesh_extraction,
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

    def save_to_paraview(
        self, polyhedrons_data: Dict, output_path: str, multiblock: bool = True, include_z_depth: bool = True
    ):
        """
        Export all polyhedrons to a Paraview-compatible file (.vtp or .vtm).

        Args:
            polyhedrons_data: Data dictionary from process_voxel_grid
            output_path: Output file path (.vtp for single mesh, .vtm for multiblock)
            multiblock: If True, save as MultiBlock (.vtm), else as single PolyData (.vtp)
            include_z_depth: If True, add z-depth related data for coloring
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

                        # Add z-depth related data for coloring
                        if include_z_depth and vertices.size > 0:
                            # Point data: z-coordinate of each vertex
                            mesh.point_data["z_coordinate"] = vertices[:, 2]

                            # Calculate z statistics for this polyhedron
                            z_coords = vertices[:, 2]
                            z_min = float(np.min(z_coords))
                            z_max = float(np.max(z_coords))
                            z_mean = float(np.mean(z_coords))
                            z_range = z_max - z_min

                            # Cell data: uniform z statistics for all faces of this polyhedron
                            mesh.cell_data["z_min"] = np.full(mesh.n_cells, z_min, dtype=float)
                            mesh.cell_data["z_max"] = np.full(mesh.n_cells, z_max, dtype=float)
                            mesh.cell_data["z_mean"] = np.full(mesh.n_cells, z_mean, dtype=float)
                            mesh.cell_data["z_range"] = np.full(mesh.n_cells, z_range, dtype=float)

                            # Point data: z statistics repeated for each vertex
                            mesh.point_data["z_min"] = np.full(mesh.n_points, z_min, dtype=float)
                            mesh.point_data["z_max"] = np.full(mesh.n_points, z_max, dtype=float)
                            mesh.point_data["z_mean"] = np.full(mesh.n_points, z_mean, dtype=float)
                            mesh.point_data["z_range"] = np.full(mesh.n_points, z_range, dtype=float)

                            # Field data: polyhedron-level z statistics
                            mesh.field_data["z_min"] = np.array([z_min])
                            mesh.field_data["z_max"] = np.array([z_max])
                            mesh.field_data["z_mean"] = np.array([z_mean])
                            mesh.field_data["z_range"] = np.array([z_range])

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

                        # Add polyhedron-specific data
                        mesh.cell_data["volume"] = np.full(mesh.n_cells, poly_data.get("volume", 0), dtype=float)
                        mesh.cell_data["voxel_count"] = np.full(
                            mesh.n_cells, poly_data.get("voxel_count", 0), dtype=int
                        )
                        mesh.point_data["volume"] = np.full(mesh.n_points, poly_data.get("volume", 0), dtype=float)
                        mesh.point_data["voxel_count"] = np.full(
                            mesh.n_points, poly_data.get("voxel_count", 0), dtype=int
                        )

                        # Add z-depth related data for coloring
                        if include_z_depth and vertices.size > 0:
                            # Point data: z-coordinate of each vertex
                            mesh.point_data["z_coordinate"] = vertices[:, 2]

                            # Calculate z statistics for this polyhedron
                            z_coords = vertices[:, 2]
                            z_min = float(np.min(z_coords))
                            z_max = float(np.max(z_coords))
                            z_mean = float(np.mean(z_coords))
                            z_range = z_max - z_min

                            # Cell data: uniform z statistics for all faces of this polyhedron
                            mesh.cell_data["z_min"] = np.full(mesh.n_cells, z_min, dtype=float)
                            mesh.cell_data["z_max"] = np.full(mesh.n_cells, z_max, dtype=float)
                            mesh.cell_data["z_mean"] = np.full(mesh.n_cells, z_mean, dtype=float)
                            mesh.cell_data["z_range"] = np.full(mesh.n_cells, z_range, dtype=float)

                            # Point data: z statistics repeated for each vertex
                            mesh.point_data["z_min"] = np.full(mesh.n_points, z_min, dtype=float)
                            mesh.point_data["z_max"] = np.full(mesh.n_points, z_max, dtype=float)
                            mesh.point_data["z_mean"] = np.full(mesh.n_points, z_mean, dtype=float)
                            mesh.point_data["z_range"] = np.full(mesh.n_points, z_range, dtype=float)

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
        default=10,
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
        "--include-z-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include z-depth data for coloring in Paraview exports (default: True). Adds z_coordinate, z_min, z_max, z_mean, z_range fields.",
    )

    parser.add_argument(
        "--remove-boundary-polyhedrons",
        action=argparse.BooleanOptionalAction,
        default=True,  # Changed to True for consistency
        help="Remove polyhedrons that touch the voxel grid boundary (default: True). Use --no-remove-boundary-polyhedrons to keep them.",
    )

    parser.add_argument(
        "--max-voxel-aspect-ratio",
        type=float,
        default=20.0,  # Default from process_voxel_grid
        help="Filter polyhedrons by max voxel aspect ratio (0 or negative to disable). E.g., 20 means longest_side/shortest_side <= 20.",
    )
    parser.add_argument(
        "--coordinate-validation-threshold",
        type=float,
        default=1e4,  # Default from process_voxel_grid
        help="Max reasonable coordinate value for aggressive validation during mesh extraction (e.g., 1e4).",
    )

    # New CLI args for mesh range outlier filter
    parser.add_argument(
        "--enable-mesh-range-outlier-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the global mesh range outlier filter (default: True).",
    )
    parser.add_argument(
        "--mesh-range-outlier-iqr-factor",
        type=float,
        default=1.5,
        help="IQR factor for mesh range outlier detection (default: 1.5).",
    )
    parser.add_argument(
        "--mesh-range-outlier-median-factor",
        type=float,
        default=100.0,
        help="Median factor for mesh range outlier detection (default: 100.0).",
    )

    # Fast mesh extraction options
    parser.add_argument(
        "--fast-mesh-extraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast mesh extraction optimizations (default: True).",
    )
    parser.add_argument(
        "--batch-mesh-size",
        type=int,
        default=10,
        help="Batch size for fast mesh extraction (default: 10).",
    )
    parser.add_argument(
        "--skip-sdf-for-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip SDF computation for small polyhedrons in fast mode (default: True).",
    )
    parser.add_argument(
        "--small-polyhedron-threshold",
        type=int,
        default=500,
        help="Threshold for considering polyhedrons as 'small' for fast processing (default: 500 voxels).",
    )
    parser.add_argument(
        "--reduce-smoothing-for-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce smoothing iterations for small polyhedrons in fast mode (default: True).",
    )

    parser.add_argument(
        "--num-workers", type=int, default=os.cpu_count(), help="Number of parallel workers for mesh extraction"
    )

    # Chunking options for performance optimization
    parser.add_argument(
        "--use-chunking",
        action="store_true",
        help="Use chunking strategy for large grids to improve performance and reduce memory usage",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help="Size of each chunk (z y x) when using chunking strategy (default: 256 256 256)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=32, help="Overlap size in voxels between chunks (default: 32)"
    )
    parser.add_argument(
        "--max-chunk-workers",
        type=int,
        default=None,
        help="Maximum workers for chunk processing (default: auto-detect based on CPU count and chunk count)",
    )
    parser.add_argument(
        "--fast-chunk-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast chunk merging strategy for better performance (default: True). Use --no-fast-chunk-merge for detailed merging.",
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

    # Choose processing method based on chunking flag
    if args.use_chunking:
        print(f"Using chunked processing with chunk size: {tuple(args.chunk_size)}")
        polyhedrons_data = segmentation.process_voxel_grid_chunked(
            voxel_grid,
            chunk_size=tuple(args.chunk_size),
            overlap=args.chunk_overlap,
            max_chunk_workers=args.max_chunk_workers,
            fast_merge=args.fast_chunk_merge,
            # Pass all other parameters
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
            sdf_scale=args.sdf_scale,
            marker_threshold_percentile=args.marker_threshold_percentile,
            gaussian_smooth_dt_sigma=args.gaussian_smooth_dt_sigma,
            peak_local_max_footprint_size=args.peak_local_max_footprint_size,
            # Mesh extraction specific
            use_sdf=args.use_sdf_mesh,
            remove_boundary_polyhedrons=args.remove_boundary_polyhedrons,
            max_voxel_aspect_ratio=args.max_voxel_aspect_ratio,
            coordinate_validation_threshold=args.coordinate_validation_threshold,
            # Mesh range outlier filter parameters
            enable_mesh_range_outlier_filter=args.enable_mesh_range_outlier_filter,
            mesh_range_outlier_iqr_factor=args.mesh_range_outlier_iqr_factor,
            mesh_range_outlier_median_factor=args.mesh_range_outlier_median_factor,
            # Fast mesh extraction parameters
            fast_mesh_extraction=args.fast_mesh_extraction,
            batch_mesh_size=args.batch_mesh_size,
            skip_sdf_for_small=args.skip_sdf_for_small,
            small_polyhedron_threshold=args.small_polyhedron_threshold,
            reduce_smoothing_for_small=args.reduce_smoothing_for_small,
        )
    else:
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
            coordinate_validation_threshold=args.coordinate_validation_threshold,
            # Pass new mesh range outlier filter parameters
            enable_mesh_range_outlier_filter=args.enable_mesh_range_outlier_filter,
            mesh_range_outlier_iqr_factor=args.mesh_range_outlier_iqr_factor,
            mesh_range_outlier_median_factor=args.mesh_range_outlier_median_factor,
            # Fast mesh extraction parameters
            fast_mesh_extraction=args.fast_mesh_extraction,
            batch_mesh_size=args.batch_mesh_size,
            skip_sdf_for_small=args.skip_sdf_for_small,
            small_polyhedron_threshold=args.small_polyhedron_threshold,
            reduce_smoothing_for_small=args.reduce_smoothing_for_small,
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

            segmentation.save_to_paraview(
                polyhedrons_data,
                paraview_output_path,
                multiblock=args.paraview_multiblock,
                include_z_depth=args.include_z_depth,
            )

        # Visualization
        if args.visualize or args.screenshot:
            segmentation.visualize_polyhedrons(polyhedrons_data, output_path=args.screenshot)

        print(
            f"Polyhedron segmentation completed. Found {polyhedrons_data.get('metadata', {}).get('total_count', 0)} polyhedrons."
        )
    else:
        print("Polyhedron segmentation completed, but no polyhedrons were extracted or an error occurred.")


if __name__ == "__main__":
    main()
