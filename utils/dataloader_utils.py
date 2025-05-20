import numpy as np
from scipy import interpolate
from scipy import ndimage


def passes_edge_checks(voxel, edge_thickness):
    """
    Checks if the voxel passes the edge thickness criteria.
    Requires at least one grain voxel (value > 0) to be present
    within the specified thickness on at least one of the 6 faces.
    """
    n = edge_thickness
    has_crossing = (
        np.any(voxel[:n, :, :] > 0)
        or np.any(voxel[-n:, :, :] > 0)
        or np.any(voxel[:, :n, :] > 0)
        or np.any(voxel[:, -n:, :] > 0)
        or np.any(voxel[:, :, :n] > 0)
        or np.any(voxel[:, :, -n:] > 0)
    )
    return has_crossing


def passes_bw_ratio(voxel, bw_ratio):
    """
    Checks if the voxel passes the black-white ratio criteria.
    """
    total_elements = voxel.size

    return (voxel == 1).sum() / total_elements > bw_ratio


def upscale_voxels(voxels, target_size=(64, 64, 64)):
    """
    Upscale voxels to target size using linear interpolation and thresholding.
    Maintains binary nature of voxels while providing smooth upscaling.
    """
    if voxels.shape == target_size:
        return voxels

    # Create coordinates for interpolation
    z, y, x = np.mgrid[
        0 : 1 : target_size[0] * 1j,
        0 : 1 : target_size[1] * 1j,
        0 : 1 : target_size[2] * 1j,
    ]

    # Create original coordinates
    z_orig, y_orig, x_orig = np.mgrid[
        0 : 1 : voxels.shape[0] * 1j,
        0 : 1 : voxels.shape[1] * 1j,
        0 : 1 : voxels.shape[2] * 1j,
    ]

    # Perform interpolation
    interpolator = interpolate.RegularGridInterpolator(
        (
            np.linspace(0, 1, voxels.shape[0]),
            np.linspace(0, 1, voxels.shape[1]),
            np.linspace(0, 1, voxels.shape[2]),
        ),
        voxels,
        method="linear",
    )

    pts = np.array([z.flatten(), y.flatten(), x.flatten()]).T
    interpolated = interpolator(pts).reshape(target_size)

    # Threshold to maintain binary nature
    return (interpolated > 0.5).astype(np.float32)


def compute_sdf(binary_voxel, scale=5.0):
    """
    Compute the signed distance field (SDF) of a binary voxel. It was noted in the tests that SDF is better
    to understand the shape of the object and the interaction between the candidate and antagonist grains.
    However, for unconditional diffusion models, SDF is not used, since binary masks produced better results.
    """
    distance_outside = ndimage.distance_transform_edt(binary_voxel == 0)
    distance_inside = ndimage.distance_transform_edt(binary_voxel == 1)
    sdf = distance_outside - distance_inside
    sdf = np.clip(sdf, -scale, scale) / scale
    return sdf.astype(np.float32)
