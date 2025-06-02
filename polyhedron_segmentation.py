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
import platform
import subprocess

if sys.platform == "darwin":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)


# GPU Backend Support
class GPUBackend:
    """GPU backend management for CUDA and MPS acceleration."""

    def __init__(self):
        self.backend = None
        self.device = None
        self.available_backends = []
        self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check availability of GPU backends."""
        # Check CUDA
        try:
            import cupy as cp

            if cp.cuda.is_available():
                self.available_backends.append("cuda")
                print(f"CUDA detected: {cp.cuda.runtime.getDeviceCount()} GPU(s) available")
        except ImportError:
            pass

        # Check MPS (Metal Performance Shaders for Apple Silicon)
        try:
            import torch

            if torch.backends.mps.is_available():
                self.available_backends.append("mps")
                print("MPS (Metal Performance Shaders) detected and available")
        except ImportError:
            pass

        if not self.available_backends:
            print("No GPU backends available, will use CPU-only processing")

    def set_backend(self, backend: str, memory_fraction: float = 0.8):
        """Set the GPU backend to use."""
        if backend == "auto":
            if "cuda" in self.available_backends:
                backend = "cuda"
            elif "mps" in self.available_backends:
                backend = "mps"
            else:
                backend = "cpu"

        if backend == "cpu":
            self.backend = "cpu"
            self.device = None
            print("Using CPU backend")
            return True

        if backend not in self.available_backends:
            print(f"Backend '{backend}' not available. Available: {self.available_backends}")
            return False

        self.backend = backend

        if backend == "cuda":
            import cupy as cp

            # Set memory pool limit
            try:
                memory_limit = int(cp.cuda.runtime.memGetInfo()[1] * memory_fraction)
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=memory_limit)
                self.device = cp.cuda.Device()
                print(f"CUDA backend initialized with {memory_fraction * 100:.1f}% memory limit")
            except Exception as e:
                print(f"Warning: Could not set CUDA memory limit: {e}")

        elif backend == "mps":
            import torch

            self.device = torch.device("mps")
            print("MPS backend initialized")

        return True

    def to_gpu(self, array: np.ndarray):
        """Move array to GPU."""
        if self.backend == "cuda":
            import cupy as cp

            return cp.asarray(array)
        elif self.backend == "mps":
            import torch

            return torch.from_numpy(array).to(self.device)
        else:
            return array

    def to_cpu(self, array):
        """Move array to CPU."""
        if self.backend == "cuda":
            import cupy as cp

            if isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
        elif self.backend == "mps":
            import torch

            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
        return array

    def get_module(self, module_name: str):
        """Get the appropriate module for the backend."""
        if self.backend == "cuda":
            import cupy as cp

            if module_name == "numpy":
                return cp
            elif module_name == "ndimage":
                from cupyx.scipy import ndimage as cupy_ndimage

                return cupy_ndimage
            elif module_name == "filters":
                try:
                    from cucim.skimage import filters as cucim_filters

                    return cucim_filters
                except ImportError:
                    # Fallback to manual implementation
                    return None
        elif self.backend == "mps":
            import torch
            import torch.nn.functional as F

            if module_name == "torch":
                return torch
            elif module_name == "F":
                return F

        # CPU fallback
        if module_name == "numpy":
            return np
        elif module_name == "ndimage":
            return ndimage
        elif module_name == "filters":
            return filters

        return None


class GPUAcceleratedOperations:
    """GPU-accelerated implementations of key operations."""

    def __init__(self, gpu_backend: GPUBackend):
        self.gpu = gpu_backend

    def distance_transform_edt(self, binary_array: np.ndarray) -> np.ndarray:
        """GPU-accelerated distance transform."""
        if self.gpu.backend == "cuda":
            try:
                import cupy as cp
                from cupyx.scipy import ndimage as cupy_ndimage

                gpu_array = cp.asarray(binary_array)
                distance = cupy_ndimage.distance_transform_edt(gpu_array)
                return cp.asnumpy(distance)
            except Exception as e:
                print(f"CUDA distance transform failed, falling back to CPU: {e}")

        elif self.gpu.backend == "mps":
            try:
                import torch
                import torch.nn.functional as F

                # Convert to torch tensor
                tensor = torch.from_numpy(binary_array.astype(np.float32)).to(self.gpu.device)

                # Custom MPS distance transform implementation
                distance = self._mps_distance_transform(tensor)
                return distance.cpu().numpy()
            except Exception as e:
                print(f"MPS distance transform failed, falling back to CPU: {e}")

        # CPU fallback
        return ndimage.distance_transform_edt(binary_array)

    def _mps_distance_transform(self, binary_tensor):
        """Custom distance transform implementation for MPS."""
        import torch
        import torch.nn.functional as F

        # This is a simplified distance transform using morphological operations
        # For better accuracy, consider implementing chamfer distance transform
        binary_float = binary_tensor.float()

        # Create structuring element for 3D
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=binary_tensor.device)
        kernel = kernel / kernel.sum()

        # Iterative distance computation
        distance = torch.zeros_like(binary_float)
        current = binary_float.clone()

        for i in range(1, 50):  # Maximum distance
            if current.sum() == 0:
                break

            # Add current distance level
            distance += current * i

            # Erode current mask
            padded = F.pad(current.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="constant", value=0)
            eroded = F.conv3d(padded, kernel, padding=0)
            current = (eroded.squeeze() > 0.99).float()

            # Remove already processed pixels
            current = current * (distance == 0).float()

        return distance

    def gaussian_filter(self, array: np.ndarray, sigma: float) -> np.ndarray:
        """GPU-accelerated Gaussian filtering."""
        if self.gpu.backend == "cuda":
            try:
                import cupy as cp
                from cupyx.scipy import ndimage as cupy_ndimage

                gpu_array = cp.asarray(array)
                filtered = cupy_ndimage.gaussian_filter(gpu_array, sigma=sigma)
                return cp.asnumpy(filtered)
            except Exception as e:
                print(f"CUDA Gaussian filter failed, falling back to CPU: {e}")

        elif self.gpu.backend == "mps":
            try:
                import torch
                import torch.nn.functional as F

                tensor = torch.from_numpy(array.astype(np.float32)).to(self.gpu.device)
                filtered = self._mps_gaussian_filter(tensor, sigma)
                return filtered.cpu().numpy()
            except Exception as e:
                print(f"MPS Gaussian filter failed, falling back to CPU: {e}")

        # CPU fallback
        return filters.gaussian(array, sigma=sigma)

    def _mps_gaussian_filter(self, tensor, sigma):
        """Custom 3D Gaussian filter for MPS."""
        import torch
        import torch.nn.functional as F

        # Create 3D Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Generate 3D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 3D kernel
        kernel_3d = kernel_1d.view(-1, 1, 1) * kernel_1d.view(1, -1, 1) * kernel_1d.view(1, 1, -1)
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)

        # Apply convolution
        tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        filtered = F.conv3d(tensor_5d, kernel_3d, padding=padding)

        return filtered.squeeze()

    def binary_erosion(self, binary_array: np.ndarray, iterations: int = 1) -> np.ndarray:
        """GPU-accelerated binary erosion."""
        if self.gpu.backend == "cuda":
            try:
                import cupy as cp
                from cupyx.scipy import ndimage as cupy_ndimage

                gpu_array = cp.asarray(binary_array)
                result = gpu_array.copy()
                for _ in range(iterations):
                    result = cupy_ndimage.binary_erosion(result)
                return cp.asnumpy(result)
            except Exception as e:
                print(f"CUDA erosion failed, falling back to CPU: {e}")

        elif self.gpu.backend == "mps":
            try:
                import torch
                import torch.nn.functional as F

                tensor = torch.from_numpy(binary_array.astype(np.float32)).to(self.gpu.device)
                result = self._mps_binary_erosion(tensor, iterations)
                return (result > 0.5).cpu().numpy()
            except Exception as e:
                print(f"MPS erosion failed, falling back to CPU: {e}")

        # CPU fallback
        result = binary_array.copy()
        for _ in range(iterations):
            result = morphology.binary_erosion(result)
        return result

    def _mps_binary_erosion(self, binary_tensor, iterations):
        """Custom binary erosion for MPS."""
        import torch
        import torch.nn.functional as F

        # 3D erosion kernel (6-connectivity)
        kernel = torch.zeros(1, 1, 3, 3, 3, device=binary_tensor.device)
        kernel[0, 0, 1, 1, 1] = 1  # center
        kernel[0, 0, 0, 1, 1] = 1  # front
        kernel[0, 0, 2, 1, 1] = 1  # back
        kernel[0, 0, 1, 0, 1] = 1  # left
        kernel[0, 0, 1, 2, 1] = 1  # right
        kernel[0, 0, 1, 1, 0] = 1  # bottom
        kernel[0, 0, 1, 1, 2] = 1  # top

        result = binary_tensor.float()
        for _ in range(iterations):
            padded = F.pad(result.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="constant", value=0)
            convolved = F.conv3d(padded, kernel, padding=0)
            result = (convolved.squeeze() >= 7).float()  # All 7 pixels must be True

        return result

    def watershed_segmentation(self, distance: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """GPU-accelerated watershed segmentation."""
        if self.gpu.backend == "cuda":
            try:
                # Try cuCIM if available
                from cucim.skimage.segmentation import watershed as cuda_watershed

                gpu_distance = self.gpu.to_gpu(distance)
                gpu_markers = self.gpu.to_gpu(markers)
                gpu_mask = self.gpu.to_gpu(mask)

                result = cuda_watershed(gpu_distance, gpu_markers, mask=gpu_mask)
                return self.gpu.to_cpu(result)
            except ImportError:
                print("cuCIM not available, falling back to CPU watershed")
            except Exception as e:
                print(f"CUDA watershed failed, falling back to CPU: {e}")

        # CPU fallback (MPS doesn't have good watershed implementation)
        return watershed(distance, markers, mask=mask)


# Initialize global GPU backend
_gpu_backend = GPUBackend()
_gpu_ops = GPUAcceleratedOperations(_gpu_backend)


def get_optimal_worker_count(task_type: str = "cpu_intensive") -> int:
    """
    Get optimal number of workers based on system architecture.

    For Apple Silicon Macs, this tries to detect P-core count for CPU-intensive tasks.
    For other systems, falls back to CPU count with reasonable limits.

    Args:
        task_type: "cpu_intensive" for mesh processing, "mixed" for chunk processing

    Returns:
        Optimal number of workers
    """
    total_cores = os.cpu_count()

    # Try to detect Apple Silicon and get P-core count
    if sys.platform == "darwin":
        try:
            # Check if this is Apple Silicon
            machine = platform.machine()
            if machine in ["arm64", "aarch64"]:
                # Try to get performance core count using system_profiler
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5
                    )

                    if result.returncode == 0:
                        # Look for performance cores info
                        for line in result.stdout.split("\n"):
                            if "Performance Cores" in line:
                                # Extract number of P-cores
                                parts = line.split(":")
                                if len(parts) > 1:
                                    p_cores = int(parts[1].strip())
                                    print(f"Detected Apple Silicon with {p_cores} P-cores, {total_cores} total cores")

                                    if task_type == "cpu_intensive":
                                        # For CPU-intensive tasks, use P-cores + 1-2 extra for efficiency
                                        optimal = min(p_cores + 2, total_cores)
                                        print(f"Using {optimal} workers for CPU-intensive tasks (P-cores + 2)")
                                        return optimal
                                    else:
                                        # For mixed tasks, can use more cores
                                        optimal = min(p_cores * 2, total_cores)
                                        return optimal

                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                    pass

                # Fallback for Apple Silicon if system_profiler fails
                # apple silicon m4 has 6 efficiency cores and 4 power cores
                p_cores = 4

                # for cpu intensive tasks use 1 more core than the power cores
                if task_type == "cpu_intensive":
                    optimal = min(p_cores + 1, total_cores)
                    print(f"Using {optimal} workers for CPU-intensive tasks")
                    return optimal
                else:
                    optimal = min(p_cores + 2, total_cores)
                    return optimal

        except Exception as e:
            print(f"Failed to detect Apple Silicon configuration: {e}")

    # Fallback for non-Apple Silicon or detection failure
    if task_type == "cpu_intensive":
        # For CPU-intensive tasks, use fewer workers to avoid thread thrashing
        optimal = min(total_cores, max(1, total_cores - 2))
    else:
        # For mixed/I/O tasks, can use more workers
        optimal = total_cores

    print(f"Using {optimal} workers (fallback strategy)")
    return optimal


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
    Supports GPU acceleration via CUDA and MPS backends.
    """

    def __init__(
        self,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        gpu_backend: str = "auto",
        gpu_memory_fraction: float = 0.8,
        min_size_for_gpu: int = 1000000,  # Use GPU for grids larger than 1M voxels
    ):
        """
        Initialize the segmentation module.

        Args:
            voxel_spacing: Physical spacing between voxels (dx, dy, dz)
            origin: Origin point of the voxel grid in world coordinates
            gpu_backend: GPU backend to use ('auto', 'cuda', 'mps', 'cpu')
            gpu_memory_fraction: Fraction of GPU memory to use (0.1-0.9)
            min_size_for_gpu: Minimum grid size to trigger GPU acceleration
        """
        self.voxel_spacing = np.array(voxel_spacing)
        self.origin = np.array(origin)
        self.min_size_for_gpu = min_size_for_gpu

        # Initialize GPU backend
        self.gpu_enabled = _gpu_backend.set_backend(gpu_backend, gpu_memory_fraction)
        self.gpu_backend_name = _gpu_backend.backend

        if self.gpu_enabled and self.gpu_backend_name != "cpu":
            print(f"GPU acceleration enabled with {self.gpu_backend_name.upper()} backend")
        else:
            print("Using CPU-only processing")

    def _should_use_gpu(self, array_size: int) -> bool:
        """Determine if GPU should be used based on array size and availability."""
        return self.gpu_enabled and self.gpu_backend_name != "cpu" and array_size >= self.min_size_for_gpu

    def _get_gpu_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information."""
        if self.gpu_backend_name == "cuda":
            try:
                import cupy as cp

                free, total = cp.cuda.runtime.memGetInfo()
                return {"free": free, "total": total, "used": total - free}
            except Exception:
                return {"free": 0, "total": 0, "used": 0}
        elif self.gpu_backend_name == "mps":
            try:
                import torch

                allocated = torch.mps.current_allocated_memory()
                # MPS doesn't provide total memory info easily
                return {"free": -1, "total": -1, "used": allocated}
            except Exception:
                return {"free": 0, "total": 0, "used": 0}
        return {"free": 0, "total": 0, "used": 0}

    def _estimate_gpu_memory_needed(self, grid_shape: Tuple[int, int, int], dtype=np.float32) -> int:
        """Estimate GPU memory needed for processing a grid of given shape."""
        bytes_per_element = np.dtype(dtype).itemsize
        # Estimate: input grid + distance transform + SDF + working arrays
        multiplier = 5  # Conservative estimate for temporary arrays
        return int(np.prod(grid_shape) * bytes_per_element * multiplier)

    def _optimize_chunk_size_for_gpu(
        self, grid_shape: Tuple[int, int, int], default_chunk_size: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Optimize chunk size based on available GPU memory."""
        if not self._should_use_gpu(np.prod(grid_shape)):
            return default_chunk_size

        memory_info = self._get_gpu_memory_info()
        if memory_info["free"] <= 0:  # Unknown memory or MPS
            return default_chunk_size

        # Use 70% of available memory
        available_memory = int(memory_info["free"] * 0.7)

        # Calculate optimal chunk size
        default_memory_needed = self._estimate_gpu_memory_needed(default_chunk_size)

        if default_memory_needed <= available_memory:
            return default_chunk_size

        # Scale down chunk size
        scale_factor = (available_memory / default_memory_needed) ** (1 / 3)
        optimized_size = tuple(max(64, int(s * scale_factor)) for s in default_chunk_size)

        print(f"GPU memory optimization: chunk size adjusted from {default_chunk_size} to {optimized_size}")
        return optimized_size

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
        use_gpu = self._should_use_gpu(binary_voxel.size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if use_gpu:
                print(f"Computing SDF using {self.gpu_backend_name.upper()} acceleration...")
                # Distance from background (outside)
                distance_outside = _gpu_ops.distance_transform_edt(binary_voxel == 0)
                # Distance from foreground (inside)
                distance_inside = _gpu_ops.distance_transform_edt(binary_voxel == 1)
            else:
                # CPU fallback
                distance_outside = ndimage.distance_transform_edt(binary_voxel == 0)
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
            use_gpu = self._should_use_gpu(voxel_grid.size)
            if use_gpu:
                print(f"Using {self.gpu_backend_name.upper()} acceleration for Gaussian smoothing")
                smoothed = _gpu_ops.gaussian_filter(voxel_grid.astype(float), sigma=gaussian_sigma)
            else:
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
        use_gpu = self._should_use_gpu(sdf.size)
        if use_gpu:
            print(f"Using {self.gpu_backend_name.upper()} acceleration for SDF watershed")
            labels = _gpu_ops.watershed_segmentation(-sdf, markers, binary_grid)
        else:
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
        use_gpu = self._should_use_gpu(current_grid.size)

        # Apply erosion to separate touching objects
        if erosion_iterations > 0:
            print(f"Applying binary erosion with {erosion_iterations} iterations...")
            if use_gpu:
                print(f"Using {self.gpu_backend_name.upper()} acceleration for erosion")
                eroded_grid = _gpu_ops.binary_erosion(current_grid, erosion_iterations)
            else:
                eroded_grid = current_grid
                for _ in range(erosion_iterations):
                    eroded_grid = morphology.binary_erosion(eroded_grid)

            if np.sum(eroded_grid) == 0:
                print("Warning: Erosion resulted in an empty grid. Using original grid for segmentation.")
            else:
                current_grid = eroded_grid

        # Compute distance transform on the (potentially eroded) grid
        if use_gpu:
            print(f"Computing distance transform using {self.gpu_backend_name.upper()} acceleration")
            distance = _gpu_ops.distance_transform_edt(current_grid)
        else:
            distance = ndimage.distance_transform_edt(current_grid)

        # Optionally smooth the distance transform
        if gaussian_smooth_dt_sigma > 0:
            print(f"Smoothing distance transform with sigma: {gaussian_smooth_dt_sigma}")
            if use_gpu:
                distance = _gpu_ops.gaussian_filter(distance, sigma=gaussian_smooth_dt_sigma)
            else:
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
        if use_gpu:
            print(f"Performing watershed using {self.gpu_backend_name.upper()} acceleration")
            labels = _gpu_ops.watershed_segmentation(-distance, markers, current_grid)
        else:
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
        ultra_fast_mode: bool = False,  # NEW: Ultra-fast mode with aggressive optimizations
        max_labels_threshold: int = 5000,  # NEW: Threshold for enabling aggressive optimizations
        stream_batch_size: int = 50,  # NEW: Batch size for streaming large label sets
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
        print(f"GPU backend: {self.gpu_backend_name.upper()}")

        # Optimize chunk size for GPU if enabled
        original_chunk_size = chunk_size
        if self._should_use_gpu(voxel_grid.size):
            chunk_size = self._optimize_chunk_size_for_gpu(voxel_grid.shape, chunk_size)
            if chunk_size != original_chunk_size:
                print(f"GPU-optimized chunk size: {chunk_size} (was {original_chunk_size})")

        print(f"Chunk size: {chunk_size}")
        print(f"Overlap: {overlap}")
        if ultra_fast_mode:
            print("ULTRA-FAST MODE ENABLED - Prioritizing speed over accuracy")
        print("=" * 60)

        # Optimize overlap for ultra-fast mode
        if ultra_fast_mode and overlap > 16:
            original_overlap = overlap
            overlap = max(8, overlap // 2)  # Reduce overlap for speed
            print(f"Ultra-fast mode: Reducing overlap from {original_overlap} to {overlap} for speed")

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
        chunk_workers = max_chunk_workers or min(len(chunks), get_optimal_worker_count("mixed"))
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

        # Step 5: Merge chunk results with optimization selection
        if ultra_fast_mode:
            print("Using ultra-fast chunk merging...")
            labeled_grid, num_labels = self._merge_chunk_results_ultra_fast(chunk_results, binary_grid.shape, overlap)
        else:
            labeled_grid, num_labels = self._merge_chunk_results(chunk_results, binary_grid.shape, overlap, fast_merge)

        if num_labels == 0:
            print("No polyhedrons found after chunked segmentation!")
            return {"polyhedrons": {}, "metadata": {"total_count": 0, "chunked": True}}

        # Step 6: Continue with optimized pipeline
        print(f"Continuing with optimized pipeline for {num_labels} labels...")

        # Check if we need aggressive optimizations for large label counts
        use_aggressive_opts = ultra_fast_mode or num_labels > max_labels_threshold
        if use_aggressive_opts:
            print(f"Large label count ({num_labels}) detected - enabling aggressive optimizations...")
            kwargs["_use_early_filtering"] = True
            kwargs["_use_streaming"] = True
            kwargs["_stream_batch_size"] = stream_batch_size

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

        # Step 1: Filter polyhedrons - use optimized early filtering if enabled
        print(f"\nInitial labels found: {num_labels}")
        polyhedrons = {}

        # Check if we should use fast early filtering
        use_early_filtering = kwargs.get("_use_early_filtering", False)

        if use_early_filtering:
            print("Using optimized early filtering...")
            label_ids_to_process, filter_stats = self._early_filter_labels_in_chunks(
                labeled_grid, num_labels, min_polyhedron_size, remove_boundary_polyhedrons, max_voxel_aspect_ratio
            )
            boundary_removed_count = filter_stats["boundary_removed_count"]
            aspect_ratio_removed_count = filter_stats["aspect_ratio_removed_count"]
            # For metadata consistency, get the initial count of labels that passed min size filter
            min_size_filtered_count = filter_stats.get("initial_size_filtered_count", 0)
        else:
            # Original filtering logic
            print("Using standard filtering...")

            # First, filter by minimum size
            candidate_label_ids = [
                label_id
                for label_id in range(1, num_labels + 1)
                if np.sum(labeled_grid == label_id) >= min_polyhedron_size
            ]
            print(
                f"Labels remaining after min_polyhedron_size ({min_polyhedron_size} voxels) filter: {len(candidate_label_ids)}"
            )

            min_size_filtered_count = len(candidate_label_ids)  # Store for metadata
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

        # Check if we should use streaming processing for large label sets
        use_streaming = kwargs.get("_use_streaming", False)
        stream_batch_size = kwargs.get("_stream_batch_size", 50)

        if use_streaming and stream_batch_size > 0 and len(label_ids_to_process) > stream_batch_size:
            print(f"Using streaming processing for {len(label_ids_to_process)} labels...")

            # Prepare processing parameters
            processing_params = {
                "smoothing_iterations": smoothing_iterations,
                "decimation_ratio": decimation_ratio,
                "use_sdf": use_sdf,
                "coordinate_validation_threshold": coordinate_validation_threshold,
                "batch_mesh_size": batch_mesh_size,
                "skip_sdf_for_small": skip_sdf_for_small,
                "small_polyhedron_threshold": small_polyhedron_threshold,
                "reduce_smoothing_for_small": reduce_smoothing_for_small,
                "num_workers": num_workers,
                "fast_mesh_extraction": fast_mesh_extraction,
            }

            candidate_polyhedron_mesh_data_list = self._stream_process_large_label_set(
                labeled_grid, label_ids_to_process, processing_params, stream_batch_size
            )
        elif fast_mesh_extraction:
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
                # Optimize worker count for CPU-intensive mesh extraction
                optimal_workers = min(num_workers, get_optimal_worker_count("cpu_intensive"))
                print(f"Using {optimal_workers} parallel workers for mesh extraction...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
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
            "min_size_filtered_count": min_size_filtered_count,
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

        # If batch_size is 0 or negative, process all polyhedrons in one batch
        if batch_size <= 0:
            batch_size = len(polyhedron_list)
            print(f"Batch size is 0, processing all {len(polyhedron_list)} polyhedrons in one batch (no batching)")

        # Process in batches
        for i in tqdm(range(0, len(polyhedron_list), batch_size), desc="Processing batches"):
            batch = polyhedron_list[i : i + batch_size]

            if num_workers > 1 and len(batch) > 1:
                # Parallel processing within batch - use ThreadPoolExecutor for I/O bound mesh operations
                optimal_batch_workers = min(num_workers, len(batch), get_optimal_worker_count("mixed"))
                with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_batch_workers) as executor:
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
            # Use optimized batch processing with optimal worker count
            optimal_workers = min(num_workers, get_optimal_worker_count("cpu_intensive"))
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
                optimal_workers,
            )
        elif num_workers > 1 and len(label_ids_to_process) > 1:
            # Optimize worker count for CPU-intensive mesh extraction
            optimal_workers = min(num_workers, get_optimal_worker_count("cpu_intensive"))
            print(f"Using {optimal_workers} parallel workers for mesh extraction...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
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

    def _convert_numpy_types(self, obj):
        """
        Recursively convert NumPy types to native Python types for JSON serialization.

        Args:
            obj: Object that may contain NumPy types

        Returns:
            Object with NumPy types converted to Python types
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

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

        # Convert NumPy types to native Python types for JSON serialization
        json_compatible_data = self._convert_numpy_types(polyhedrons_data)

        if compress:
            import gzip

            with gzip.open(output_path + ".gz", "wt", encoding="utf-8") as f:
                json.dump(json_compatible_data, f, indent=indent)
            print(f"Compressed JSON saved to: {output_path}.gz")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_compatible_data, f, indent=indent)
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
        self,
        polyhedrons_data: Dict,
        output_path: str,
        multiblock: bool = True,
        include_z_depth: bool = True,
        fast_export: bool = True,  # New parameter for speed optimization
        export_batch_size: int = 100,  # Process polyhedrons in batches
        num_export_workers: int = None,  # Parallel processing workers
    ):
        """
        Export all polyhedrons to a Paraview-compatible file (.vtp or .vtm).
        Optimized for speed with parallel processing and batch operations.

        Args:
            polyhedrons_data: Data dictionary from process_voxel_grid
            output_path: Output file path (.vtp for single mesh, .vtm for multiblock)
            multiblock: If True, save as MultiBlock (.vtm), else as single PolyData (.vtp)
            include_z_depth: If True, add z-depth related data for coloring
            fast_export: Use optimized export with parallel processing
            export_batch_size: Number of polyhedrons to process in each batch
            num_export_workers: Number of parallel workers (None = auto-detect)
        """
        import pyvista as pv
        from pathlib import Path
        import concurrent.futures
        from tqdm import tqdm

        print(f"Exporting polyhedrons to Paraview file: {output_path}")
        if fast_export:
            print("Using optimized fast export with parallel processing...")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        polyhedrons = polyhedrons_data.get("polyhedrons", {})
        if not polyhedrons:
            print("No polyhedrons to export.")
            return

        # Determine optimal number of workers
        if num_export_workers is None:
            num_export_workers = min(len(polyhedrons), get_optimal_worker_count("mixed"))

        if fast_export and len(polyhedrons) > 10:
            # Use optimized parallel export
            if multiblock:
                self._save_to_paraview_multiblock_fast(
                    polyhedrons, output_path, include_z_depth, export_batch_size, num_export_workers
                )
            else:
                self._save_to_paraview_combined_fast(
                    polyhedrons, output_path, include_z_depth, export_batch_size, num_export_workers
                )
        else:
            # Use original method for small datasets or when fast_export is disabled
            self._save_to_paraview_original(polyhedrons, output_path, multiblock, include_z_depth)

    def _create_mesh_worker(self, poly_data_tuple):
        """
        Worker function for parallel mesh creation.

        Args:
            poly_data_tuple: (poly_id_str, poly_data, include_z_depth)

        Returns:
            (poly_id, mesh) or None if mesh creation failed
        """
        import pyvista as pv

        poly_id_str, poly_data, include_z_depth = poly_data_tuple
        poly_id = int(poly_id_str)

        try:
            vertices = np.array(poly_data.get("vertices", []))
            faces_list = poly_data.get("faces", [])

            if vertices.size == 0 or not faces_list:
                return None

            # Optimized face conversion using list comprehension
            pv_faces = []
            for face_indices in faces_list:
                if len(face_indices) >= 3:
                    pv_faces.extend([len(face_indices)] + face_indices)

            if not pv_faces:
                return None

            mesh = pv.PolyData(vertices, np.asarray(pv_faces, dtype=np.int32))
            if mesh.n_points == 0 or mesh.n_cells == 0:
                return None

            # Add data efficiently
            mesh.point_data["poly_id"] = np.full(mesh.n_points, poly_id, dtype=np.int32)
            mesh.cell_data["poly_id"] = np.full(mesh.n_cells, poly_id, dtype=np.int32)

            # Add volume and voxel count data
            volume = poly_data.get("volume", 0)
            voxel_count = poly_data.get("voxel_count", 0)

            mesh.field_data["volume"] = np.array([volume], dtype=np.float32)
            mesh.field_data["voxel_count"] = np.array([voxel_count], dtype=np.int32)

            # Add z-depth data if requested
            if include_z_depth and vertices.size > 0:
                # Vectorized z-coordinate operations
                z_coords = vertices[:, 2].astype(np.float32)
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                z_mean = np.mean(z_coords)
                z_range = z_max - z_min

                # Point data
                mesh.point_data["z_coordinate"] = z_coords
                mesh.point_data["z_min"] = np.full(mesh.n_points, z_min, dtype=np.float32)
                mesh.point_data["z_max"] = np.full(mesh.n_points, z_max, dtype=np.float32)
                mesh.point_data["z_mean"] = np.full(mesh.n_points, z_mean, dtype=np.float32)
                mesh.point_data["z_range"] = np.full(mesh.n_points, z_range, dtype=np.float32)

                # Cell data
                mesh.cell_data["z_min"] = np.full(mesh.n_cells, z_min, dtype=np.float32)
                mesh.cell_data["z_max"] = np.full(mesh.n_cells, z_max, dtype=np.float32)
                mesh.cell_data["z_mean"] = np.full(mesh.n_cells, z_mean, dtype=np.float32)
                mesh.cell_data["z_range"] = np.full(mesh.n_cells, z_range, dtype=np.float32)

                # Field data
                mesh.field_data["z_min"] = np.array([z_min], dtype=np.float32)
                mesh.field_data["z_max"] = np.array([z_max], dtype=np.float32)
                mesh.field_data["z_mean"] = np.array([z_mean], dtype=np.float32)
                mesh.field_data["z_range"] = np.array([z_range], dtype=np.float32)

            return (poly_id, mesh)

        except Exception as e:
            print(f"Error creating mesh for polyhedron {poly_id}: {e}")
            return None

    def _save_to_paraview_multiblock_fast(
        self, polyhedrons: Dict, output_path: str, include_z_depth: bool, batch_size: int, num_workers: int
    ):
        """Fast parallel export for MultiBlock format."""
        import pyvista as pv
        from pathlib import Path
        import concurrent.futures
        from tqdm import tqdm

        print(f"Fast MultiBlock export with {num_workers} workers...")

        # Prepare data for parallel processing
        poly_items = list(polyhedrons.items())

        # Process in parallel batches
        mb = pv.MultiBlock()
        successful_exports = 0

        if num_workers > 1 and len(poly_items) > 10:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Create tasks
                tasks = [(poly_id_str, poly_data, include_z_depth) for poly_id_str, poly_data in poly_items]

                # Submit all tasks
                futures = [executor.submit(self._create_mesh_worker, task) for task in tasks]

                # Collect results with progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Creating meshes"
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            poly_id, mesh = result
                            mb[str(poly_id)] = mesh
                            successful_exports += 1
                    except Exception as e:
                        print(f"Error in parallel mesh creation: {e}")
        else:
            # Sequential processing with progress bar
            for poly_id_str, poly_data in tqdm(poly_items, desc="Creating meshes"):
                result = self._create_mesh_worker((poly_id_str, poly_data, include_z_depth))
                if result is not None:
                    poly_id, mesh = result
                    mb[str(poly_id)] = mesh
                    successful_exports += 1

        if mb.n_blocks == 0:
            print("No valid meshes to save in MultiBlock.")
            return

        # Save with optimized settings
        out_path = Path(output_path)
        if out_path.suffix.lower() != ".vtm":
            out_path = out_path.with_suffix(".vtm")

        print(f"Saving {successful_exports} meshes to MultiBlock file...")
        mb.save(str(out_path), binary=True)
        print(f"MultiBlock file saved to: {out_path}")

    def _save_to_paraview_combined_fast(
        self, polyhedrons: Dict, output_path: str, include_z_depth: bool, batch_size: int, num_workers: int
    ):
        """Fast parallel export for combined format."""
        import pyvista as pv
        from pathlib import Path
        import concurrent.futures
        from tqdm import tqdm

        print(f"Fast combined export with {num_workers} workers...")

        # Prepare data for parallel processing
        poly_items = list(polyhedrons.items())
        all_meshes = []
        successful_exports = 0

        if num_workers > 1 and len(poly_items) > 10:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Create tasks
                tasks = [(poly_id_str, poly_data, include_z_depth) for poly_id_str, poly_data in poly_items]

                # Submit all tasks
                futures = [executor.submit(self._create_mesh_worker, task) for task in tasks]

                # Collect results with progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Creating meshes"
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            poly_id, mesh = result
                            # Add additional data for combined mesh
                            volume = polyhedrons[str(poly_id)].get("volume", 0)
                            voxel_count = polyhedrons[str(poly_id)].get("voxel_count", 0)

                            mesh.cell_data["volume"] = np.full(mesh.n_cells, volume, dtype=np.float32)
                            mesh.cell_data["voxel_count"] = np.full(mesh.n_cells, voxel_count, dtype=np.int32)
                            mesh.point_data["volume"] = np.full(mesh.n_points, volume, dtype=np.float32)
                            mesh.point_data["voxel_count"] = np.full(mesh.n_points, voxel_count, dtype=np.int32)

                            all_meshes.append(mesh)
                            successful_exports += 1
                    except Exception as e:
                        print(f"Error in parallel mesh creation: {e}")
        else:
            # Sequential processing with progress bar
            for poly_id_str, poly_data in tqdm(poly_items, desc="Creating meshes"):
                result = self._create_mesh_worker((poly_id_str, poly_data, include_z_depth))
                if result is not None:
                    poly_id, mesh = result
                    # Add additional data for combined mesh
                    volume = poly_data.get("volume", 0)
                    voxel_count = poly_data.get("voxel_count", 0)

                    mesh.cell_data["volume"] = np.full(mesh.n_cells, volume, dtype=np.float32)
                    mesh.cell_data["voxel_count"] = np.full(mesh.n_cells, voxel_count, dtype=np.int32)
                    mesh.point_data["volume"] = np.full(mesh.n_points, volume, dtype=np.float32)
                    mesh.point_data["voxel_count"] = np.full(mesh.n_points, voxel_count, dtype=np.int32)

                    all_meshes.append(mesh)
                    successful_exports += 1

        if not all_meshes:
            print("No polyhedrons to combine and export.")
            return

        print(f"Combining {len(all_meshes)} meshes...")
        try:
            # More efficient combining
            combined_mesh = pv.MultiBlock(all_meshes).combine(merge_points=True, tolerance=1e-6)
        except Exception as e:
            print(f"Error combining meshes: {e}. Saving as MultiBlock instead.")
            # Fallback to multiblock
            mb_fallback = pv.MultiBlock()
            for i, mesh in enumerate(all_meshes):
                mb_fallback[f"poly_{i}"] = mesh
            out_path_fallback = Path(output_path).with_suffix(".vtm")
            mb_fallback.save(str(out_path_fallback), binary=True)
            print(f"Saved as MultiBlock fallback to: {out_path_fallback}")
            return

        # Save combined mesh
        out_path = Path(output_path)
        if out_path.suffix.lower() != ".vtu":
            out_path = out_path.with_suffix(".vtu")

        print(f"Saving combined mesh with {successful_exports} polyhedrons...")
        combined_mesh.save(str(out_path), binary=True)
        print(f"Combined mesh saved to: {out_path}")

    def _save_to_paraview_original(self, polyhedrons: Dict, output_path: str, multiblock: bool, include_z_depth: bool):
        """Original export method (kept for compatibility)."""
        import pyvista as pv
        from pathlib import Path

        if multiblock:
            mb = pv.MultiBlock()
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
                        mesh.field_data["volume"] = np.array([poly_data.get("volume", 0)])
                        mesh.field_data["voxel_count"] = np.array([poly_data.get("voxel_count", 0)])

                        if include_z_depth and vertices.size > 0:
                            z_coords = vertices[:, 2]
                            z_min = float(np.min(z_coords))
                            z_max = float(np.max(z_coords))
                            z_mean = float(np.mean(z_coords))
                            z_range = z_max - z_min

                            mesh.point_data["z_coordinate"] = z_coords
                            mesh.cell_data["z_min"] = np.full(mesh.n_cells, z_min, dtype=float)
                            mesh.cell_data["z_max"] = np.full(mesh.n_cells, z_max, dtype=float)
                            mesh.cell_data["z_mean"] = np.full(mesh.n_cells, z_mean, dtype=float)
                            mesh.cell_data["z_range"] = np.full(mesh.n_cells, z_range, dtype=float)

                            mesh.point_data["z_min"] = np.full(mesh.n_points, z_min, dtype=float)
                            mesh.point_data["z_max"] = np.full(mesh.n_points, z_max, dtype=float)
                            mesh.point_data["z_mean"] = np.full(mesh.n_points, z_mean, dtype=float)
                            mesh.point_data["z_range"] = np.full(mesh.n_points, z_range, dtype=float)

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
            # Combined export logic (similar to above but for single file)
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
                        mesh.cell_data["volume"] = np.full(mesh.n_cells, poly_data.get("volume", 0), dtype=float)
                        mesh.cell_data["voxel_count"] = np.full(
                            mesh.n_cells, poly_data.get("voxel_count", 0), dtype=int
                        )
                        mesh.point_data["volume"] = np.full(mesh.n_points, poly_data.get("volume", 0), dtype=float)
                        mesh.point_data["voxel_count"] = np.full(
                            mesh.n_points, poly_data.get("voxel_count", 0), dtype=int
                        )

                        if include_z_depth and vertices.size > 0:
                            z_coords = vertices[:, 2]
                            z_min = float(np.min(z_coords))
                            z_max = float(np.max(z_coords))
                            z_mean = float(np.mean(z_coords))
                            z_range = z_max - z_min

                            mesh.point_data["z_coordinate"] = z_coords
                            mesh.cell_data["z_min"] = np.full(mesh.n_cells, z_min, dtype=float)
                            mesh.cell_data["z_max"] = np.full(mesh.n_cells, z_max, dtype=float)
                            mesh.cell_data["z_mean"] = np.full(mesh.n_cells, z_mean, dtype=float)
                            mesh.cell_data["z_range"] = np.full(mesh.n_cells, z_range, dtype=float)

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
                combined_mesh = pv.MultiBlock(all_meshes).combine(merge_points=True, tolerance=1e-05)
            except Exception as e:
                print(f"Error combining meshes: {e}. Saving as MultiBlock instead.")
                mb_fallback = pv.MultiBlock()
                for i, m in enumerate(all_meshes):
                    mb_fallback[f"poly_{i}"] = m
                out_path_fallback = Path(output_path).with_suffix(".vtm")
                mb_fallback.save(str(out_path_fallback), binary=True)
                print(f"Combined PolyData save failed. Saved as MultiBlock to: {out_path_fallback}")
                return

            out_path = Path(output_path)
            if out_path.suffix.lower() != ".vtu":
                out_path = out_path.with_suffix(".vtu")
            combined_mesh.save(str(out_path), binary=True)
            print(f"Combined UnstructuredGrid file saved to: {out_path}")

    def _merge_chunk_results_ultra_fast(
        self, chunk_results: List[Dict], grid_shape: Tuple[int, int, int], overlap: int
    ) -> Tuple[np.ndarray, int]:
        """
        Ultra-fast chunk merging with minimal overlap resolution for maximum performance.

        Strategy: Prioritize speed over perfect overlap resolution.
        Uses simple "first wins" strategy with minimal computation.
        """
        print(f"Ultra-fast merging {len(chunk_results)} chunk results...")

        # Initialize full grid
        full_labels = np.zeros(grid_shape, dtype=np.int32)
        global_label_id = 1

        # Sort chunks by number of labels (largest first for priority)
        chunk_results = sorted(chunk_results, key=lambda x: x["num_labels"], reverse=True)

        for i, chunk_result in enumerate(chunk_results):
            if chunk_result["num_labels"] == 0:
                continue

            chunk_labels = chunk_result["labels"]
            chunk_slice = chunk_result["chunk_slice"]

            # Get unique labels in this chunk
            unique_labels = np.unique(chunk_labels)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background

            if len(unique_labels) == 0:
                continue

            # Create simple 1:1 mapping - much faster than volume comparison
            label_mapping = {old_label: global_label_id + j for j, old_label in enumerate(unique_labels)}
            global_label_id += len(unique_labels)

            # Apply mapping using numpy's advanced indexing - very fast
            mapped_chunk = np.zeros_like(chunk_labels, dtype=np.int32)
            for old_label, new_label in label_mapping.items():
                mapped_chunk[chunk_labels == old_label] = new_label

            # Simple assignment strategy: only fill empty regions
            grid_region = full_labels[chunk_slice]
            empty_mask = (grid_region == 0) & (mapped_chunk > 0)
            grid_region[empty_mask] = mapped_chunk[empty_mask]

        # Count final labels without compaction (much faster)
        unique_final = np.unique(full_labels)
        final_count = len(unique_final) - 1 if 0 in unique_final else len(unique_final)

        print(f"Ultra-fast merge completed: {final_count} total labels")
        return full_labels, final_count

    def _early_filter_labels_in_chunks(
        self,
        labeled_grid: np.ndarray,
        num_labels: int,
        min_polyhedron_size: int,
        remove_boundary_polyhedrons: bool,
        max_voxel_aspect_ratio: Optional[float] = None,
    ) -> Tuple[List[int], Dict]:
        """
        Perform early filtering to reduce the number of labels before expensive mesh extraction.
        This can dramatically reduce processing time.
        """
        print(f"Early filtering {num_labels} labels...")

        # Pre-compute label sizes for all labels at once (vectorized)
        label_sizes = np.bincount(labeled_grid.ravel())

        # Filter by minimum size (vectorized)
        size_filtered_labels = np.where(label_sizes >= min_polyhedron_size)[0]
        size_filtered_labels = size_filtered_labels[size_filtered_labels > 0]  # Remove background

        print(f"Size filter: {len(size_filtered_labels)} labels remain after min size {min_polyhedron_size}")

        # Filter boundary labels if requested
        boundary_removed_count = 0
        if remove_boundary_polyhedrons and len(size_filtered_labels) > 0:
            print("Fast boundary detection...")
            dims = labeled_grid.shape

            # Use sets for fast boundary detection
            boundary_labels = set()
            boundary_labels.update(np.unique(labeled_grid[0, :, :]))  # Front Z
            boundary_labels.update(np.unique(labeled_grid[-1, :, :]))  # Back Z
            boundary_labels.update(np.unique(labeled_grid[:, 0, :]))  # Front Y
            boundary_labels.update(np.unique(labeled_grid[:, -1, :]))  # Back Y
            boundary_labels.update(np.unique(labeled_grid[:, :, 0]))  # Front X
            boundary_labels.update(np.unique(labeled_grid[:, :, -1]))  # Back X
            boundary_labels.discard(0)  # Remove background

            # Filter out boundary labels using set operations (very fast)
            boundary_filtered_labels = [label for label in size_filtered_labels if label not in boundary_labels]
            boundary_removed_count = len(size_filtered_labels) - len(boundary_filtered_labels)
            size_filtered_labels = boundary_filtered_labels

            print(
                f"Boundary filter: {len(size_filtered_labels)} labels remain after removing {boundary_removed_count} boundary labels"
            )

        # Fast aspect ratio filtering if requested
        aspect_ratio_removed_count = 0
        if max_voxel_aspect_ratio is not None and max_voxel_aspect_ratio > 0 and len(size_filtered_labels) > 0:
            print(f"Fast aspect ratio filtering with threshold {max_voxel_aspect_ratio}...")

            # Batch process aspect ratios
            aspect_filtered_labels = []
            for label_id in size_filtered_labels:
                # Get bounding box efficiently
                coords = np.argwhere(labeled_grid == label_id)
                if len(coords) < 2:
                    aspect_filtered_labels.append(label_id)
                    continue

                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                dims = max_coords - min_coords + 1
                dims = np.maximum(dims, 1)
                aspect_ratio = np.max(dims) / np.min(dims)

                if aspect_ratio <= max_voxel_aspect_ratio:
                    aspect_filtered_labels.append(label_id)
                else:
                    aspect_ratio_removed_count += 1

            size_filtered_labels = aspect_filtered_labels
            print(
                f"Aspect ratio filter: {len(size_filtered_labels)} labels remain after removing {aspect_ratio_removed_count} high aspect ratio labels"
            )

        filter_stats = {
            "boundary_removed_count": boundary_removed_count,
            "aspect_ratio_removed_count": aspect_ratio_removed_count,
            "final_filtered_count": len(size_filtered_labels),
            "initial_size_filtered_count": len(np.where(label_sizes >= min_polyhedron_size)[0])
            - 1,  # Count excludes background
        }

        return size_filtered_labels, filter_stats

    def _stream_process_large_label_set(
        self, labeled_grid: np.ndarray, label_ids: List[int], processing_params: Dict, max_batch_size: int = 50
    ) -> List[Dict]:
        """
        Stream process large sets of labels in small batches to avoid memory issues.
        """
        # If max_batch_size is 0, process all labels in one batch (no streaming)
        if max_batch_size <= 0:
            max_batch_size = len(label_ids)
            print(f"Stream batch size is 0, processing all {len(label_ids)} labels in one batch...")
        else:
            print(f"Stream processing {len(label_ids)} labels in batches of {max_batch_size}...")

        all_results = []
        num_batches = (len(label_ids) + max_batch_size - 1) // max_batch_size

        for i in range(0, len(label_ids), max_batch_size):
            batch_labels = label_ids[i : i + max_batch_size]
            batch_num = i // max_batch_size + 1

            print(f"Processing batch {batch_num}/{num_batches} ({len(batch_labels)} labels)...")

            # Process this batch
            if processing_params.get("fast_mesh_extraction", True):
                batch_results = self._extract_polyhedrons_fast(
                    labeled_grid,
                    batch_labels,
                    processing_params["smoothing_iterations"],
                    processing_params["decimation_ratio"],
                    processing_params["use_sdf"],
                    processing_params["coordinate_validation_threshold"],
                    min(processing_params["batch_mesh_size"], len(batch_labels)),
                    processing_params["skip_sdf_for_small"],
                    processing_params["small_polyhedron_threshold"],
                    processing_params["reduce_smoothing_for_small"],
                    processing_params["num_workers"],
                )
            else:
                # Fallback to original method for this batch
                batch_results = []
                for label_id in batch_labels:
                    polyhedron_size = np.sum(labeled_grid == label_id)
                    mesh_data = self.extract_polyhedron_mesh(
                        labeled_grid,
                        label_id,
                        processing_params["smoothing_iterations"],
                        processing_params["decimation_ratio"],
                        processing_params["use_sdf"],
                        True,  # coordinate_sanity_check
                        processing_params["coordinate_validation_threshold"],
                    )
                    if mesh_data and len(mesh_data.get("vertices", [])) > 0:
                        batch_results.append({"polyhedron_size": polyhedron_size, **mesh_data})

            all_results.extend(batch_results)

            # Memory cleanup
            import gc

            gc.collect()

        return all_results


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
        "--fast-paraview-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast parallel export for Paraview files (default: True). Significantly speeds up export.",
    )
    parser.add_argument(
        "--export-batch-size",
        type=int,
        default=100,
        help="Batch size for Paraview export processing (default: 100). Larger values use more memory but may be faster.",
    )
    parser.add_argument(
        "--num-export-workers",
        type=int,
        default=None,
        help="Number of parallel workers for Paraview export (default: auto-detect based on CPU cores).",
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
        help="Batch size for fast mesh extraction (default: 10). Set to 0 to process all polyhedrons in one batch.",
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
        default=1,
        help="Threshold for considering polyhedrons as 'small' for fast processing (default: 500 voxels).",
    )
    parser.add_argument(
        "--reduce-smoothing-for-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce smoothing iterations for small polyhedrons in fast mode (default: True).",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=get_optimal_worker_count("cpu_intensive"),
        help="Number of parallel workers for mesh extraction (for darwin apple silicon the power cores are different than efficeiency ones)",
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
    parser.add_argument(
        "--ultra-fast-mode",
        action="store_true",
        help="Enable ultra-fast mode with aggressive optimizations for maximum speed (trades some accuracy for performance)",
    )
    parser.add_argument(
        "--max-labels-threshold",
        type=int,
        default=5000,
        help="Threshold for enabling aggressive optimizations when label count is high (default: 5000)",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=50,
        help="Batch size for streaming processing of large label sets (default: 50). Set to 0 to disable streaming/batching entirely.",
    )

    # GPU acceleration options
    parser.add_argument(
        "--gpu-backend",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="GPU backend to use for acceleration (default: auto). 'auto' selects the best available backend.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use (0.1-0.9, default: 0.8). Only applies to CUDA backend.",
    )
    parser.add_argument(
        "--min-size-for-gpu",
        type=int,
        default=500000000000000,
        help="Minimum grid size (number of voxels) to trigger GPU acceleration (default: 1000000).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only processing, disabling all GPU acceleration.",
    )

    args = parser.parse_args()

    # Initialize segmentation with GPU options
    gpu_backend = "cpu" if args.force_cpu else args.gpu_backend
    segmentation = PolyhedronSegmentation(
        voxel_spacing=tuple(args.voxel_spacing),
        origin=tuple(args.origin),
        gpu_backend=gpu_backend,
        gpu_memory_fraction=args.gpu_memory_fraction,
        min_size_for_gpu=args.min_size_for_gpu,
    )

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
            ultra_fast_mode=args.ultra_fast_mode,
            max_labels_threshold=args.max_labels_threshold,
            stream_batch_size=args.stream_batch_size,
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
                fast_export=args.fast_paraview_export,
                export_batch_size=args.export_batch_size,
                num_export_workers=args.num_export_workers,
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
