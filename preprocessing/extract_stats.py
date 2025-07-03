"""
Statistical Feature Extraction for 3D Voxel Grids

This script extracts statistical features from voxel grids stored in HDF5 files
and saves them as conditioning information for the diffusion model.
"""

import h5py
import numpy as np
import scipy.ndimage
from scipy import ndimage
from skimage import measure
import argparse
import os
import pickle
from tqdm import tqdm
from pathlib import Path
import json


def extract_voxel_statistics(voxel_grid):
    """
    Extract statistical features from a single voxel grid.
    
    Args:
        voxel_grid (np.ndarray): 3D binary voxel grid
        
    Returns:
        dict: Dictionary of statistical features
    """
    # Ensure binary voxel grid
    binary_grid = (voxel_grid > 0.5).astype(np.uint8)
    
    stats = {}
    
    # Basic density/porosity statistics
    stats['density'] = np.mean(binary_grid)
    stats['volume'] = np.sum(binary_grid)
    stats['total_voxels'] = binary_grid.size
    
    # Shape characteristics
    D, H, W = binary_grid.shape
    stats['aspect_ratio_dh'] = D / H if H > 0 else 0.0
    stats['aspect_ratio_dw'] = D / W if W > 0 else 0.0
    stats['aspect_ratio_hw'] = H / W if W > 0 else 0.0
    
    # Connected components analysis
    labeled_grid, num_components = ndimage.label(binary_grid)
    stats['num_components'] = num_components
    
    if num_components > 0:
        # Component sizes
        component_sizes = np.bincount(labeled_grid.ravel())[1:]  # Skip background (0)
        stats['largest_component_size'] = np.max(component_sizes)
        stats['largest_component_ratio'] = stats['largest_component_size'] / stats['volume'] if stats['volume'] > 0 else 0.0
        stats['mean_component_size'] = np.mean(component_sizes)
        stats['std_component_size'] = np.std(component_sizes)
    else:
        stats['largest_component_size'] = 0
        stats['largest_component_ratio'] = 0.0
        stats['mean_component_size'] = 0.0
        stats['std_component_size'] = 0.0
    
    # Surface area approximation (count exposed faces)
    if stats['volume'] > 0:
        # Simple approximation: count voxels with at least one empty neighbor
        padded = np.pad(binary_grid, 1, mode='constant', constant_values=0)
        
        # Check 6-connectivity (faces)
        neighbors_sum = (
            padded[:-2, 1:-1, 1:-1] +  # front
            padded[2:, 1:-1, 1:-1] +   # back
            padded[1:-1, :-2, 1:-1] +  # top
            padded[1:-1, 2:, 1:-1] +   # bottom
            padded[1:-1, 1:-1, :-2] +  # left
            padded[1:-1, 1:-1, 2:]     # right
        )
        
        # Surface voxels have fewer than 6 solid neighbors
        surface_voxels = binary_grid & (neighbors_sum < 6)
        stats['surface_area_approx'] = np.sum(surface_voxels)
        stats['surface_to_volume_ratio'] = stats['surface_area_approx'] / stats['volume'] if stats['volume'] > 0 else 0.0
    else:
        stats['surface_area_approx'] = 0
        stats['surface_to_volume_ratio'] = 0.0
    
    # Bounding box statistics
    if stats['volume'] > 0:
        coords = np.where(binary_grid)
        min_coords = [np.min(c) for c in coords]
        max_coords = [np.max(c) for c in coords]
        bbox_size = [max_coords[i] - min_coords[i] + 1 for i in range(3)]
        bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
        
        stats['bbox_volume'] = bbox_volume
        stats['bbox_fill_ratio'] = stats['volume'] / bbox_volume if bbox_volume > 0 else 0.0
        stats['bbox_aspect_dh'] = bbox_size[0] / bbox_size[1] if bbox_size[1] > 0 else 0.0
        stats['bbox_aspect_dw'] = bbox_size[0] / bbox_size[2] if bbox_size[2] > 0 else 0.0
        stats['bbox_aspect_hw'] = bbox_size[1] / bbox_size[2] if bbox_size[2] > 0 else 0.0
    else:
        stats['bbox_volume'] = 0
        stats['bbox_fill_ratio'] = 0.0
        stats['bbox_aspect_dh'] = 0.0
        stats['bbox_aspect_dw'] = 0.0
        stats['bbox_aspect_hw'] = 0.0
    
    return stats


def get_feature_vector(stats_dict, feature_names):
    """
    Convert statistics dictionary to feature vector.
    
    Args:
        stats_dict (dict): Statistics dictionary
        feature_names (list): Ordered list of feature names
        
    Returns:
        np.ndarray: Feature vector
    """
    return np.array([stats_dict.get(name, 0.0) for name in feature_names])


def normalize_features(feature_vectors):
    """
    Normalize feature vectors using z-score normalization.
    
    Args:
        feature_vectors (np.ndarray): Array of shape (n_samples, n_features)
        
    Returns:
        tuple: (normalized_vectors, means, stds)
    """
    means = np.mean(feature_vectors, axis=0)
    stds = np.std(feature_vectors, axis=0)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1.0, stds)
    
    normalized = (feature_vectors - means) / stds
    
    return normalized, means, stds


def process_hdf5_file(h5_file_path, feature_names, save_stats=True):
    """
    Process a single HDF5 file to extract features.
    
    Args:
        h5_file_path (str): Path to HDF5 file
        feature_names (list): List of feature names to extract
        save_stats (bool): Whether to save raw statistics
        
    Returns:
        tuple: (feature_vectors, raw_stats_list)
    """
    print(f"Processing {h5_file_path}...")
    
    with h5py.File(h5_file_path, 'r') as h5f:
        voxel_dataset = h5f['voxels']
        n_samples = voxel_dataset.shape[0]
        
        feature_vectors = []
        raw_stats_list = []
        
        # Process in batches to manage memory
        batch_size = 100
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting features"):
            end_idx = min(i + batch_size, n_samples)
            batch_voxels = voxel_dataset[i:end_idx]
            
            for voxel_grid in batch_voxels:
                stats = extract_voxel_statistics(voxel_grid)
                feature_vector = get_feature_vector(stats, feature_names)
                
                feature_vectors.append(feature_vector)
                if save_stats:
                    raw_stats_list.append(stats)
    
    return np.array(feature_vectors), raw_stats_list


def save_features_to_hdf5(h5_file_path, normalized_features):
    """
    Save normalized features to HDF5 file.
    
    Args:
        h5_file_path (str): Path to HDF5 file
        normalized_features (np.ndarray): Normalized feature vectors
    """
    with h5py.File(h5_file_path, 'a') as h5f:
        # Remove existing conditioning_stats if it exists
        if 'conditioning_stats' in h5f:
            del h5f['conditioning_stats']
        
        # Save normalized features
        h5f.create_dataset('conditioning_stats', data=normalized_features)
        print(f"Saved {len(normalized_features)} conditioning vectors to {h5_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract statistical features from voxel grids")
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to HDF5 file or directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='preprocessing/output',
                       help='Directory to save preprocessing metadata')
    parser.add_argument('--features', type=str, nargs='+', 
                       default=['density', 'volume', 'num_components', 'largest_component_ratio',
                               'surface_to_volume_ratio', 'bbox_fill_ratio', 
                               'aspect_ratio_dh', 'aspect_ratio_hw'],
                       help='List of features to extract')
    parser.add_argument('--save_raw_stats', action='store_true',
                       help='Save raw statistics for analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find HDF5 files
    data_path = Path(args.data_path)
    if data_path.is_file() and data_path.suffix == '.h5':
        h5_files = [str(data_path)]
    elif data_path.is_dir():
        h5_files = list(data_path.glob('*.h5'))
        h5_files = [str(f) for f in h5_files]
    else:
        raise ValueError(f"Invalid data path: {args.data_path}")
    
    if not h5_files:
        raise ValueError(f"No HDF5 files found in {args.data_path}")
    
    print(f"Found {len(h5_files)} HDF5 files")
    print(f"Extracting features: {args.features}")
    
    # Extract features from all files
    all_features = []
    all_raw_stats = []
    file_info = []
    
    for h5_file in h5_files:
        features, raw_stats = process_hdf5_file(h5_file, args.features, args.save_raw_stats)
        all_features.append(features)
        all_raw_stats.extend(raw_stats)
        
        file_info.append({
            'file_path': h5_file,
            'n_samples': len(features),
            'start_idx': sum(len(f) for f in all_features[:-1]),
            'end_idx': sum(len(f) for f in all_features)
        })
    
    # Combine all features
    all_features = np.vstack(all_features)
    print(f"Extracted {len(all_features)} total samples with {len(args.features)} features each")
    
    # Normalize features
    print("Normalizing features...")
    normalized_features, means, stds = normalize_features(all_features)
    
    # Save normalization parameters
    normalization_params = {
        'feature_names': args.features,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'n_features': len(args.features),
        'n_samples': len(normalized_features)
    }
    
    with open(os.path.join(args.output_dir, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f, indent=2)
    
    # Save file information
    with open(os.path.join(args.output_dir, 'file_info.json'), 'w') as f:
        json.dump(file_info, f, indent=2)
    
    # Save raw statistics if requested
    if args.save_raw_stats and all_raw_stats:
        with open(os.path.join(args.output_dir, 'raw_statistics.pkl'), 'wb') as f:
            pickle.dump(all_raw_stats, f)
        print(f"Saved raw statistics for {len(all_raw_stats)} samples")
    
    # Save normalized features back to HDF5 files
    print("Saving normalized features to HDF5 files...")
    for file_info_item in file_info:
        start_idx = file_info_item['start_idx']
        end_idx = file_info_item['end_idx']
        file_features = normalized_features[start_idx:end_idx]
        save_features_to_hdf5(file_info_item['file_path'], file_features)
    
    # Print summary statistics
    print("\nFeature Summary:")
    print("-" * 50)
    for i, feature_name in enumerate(args.features):
        print(f"{feature_name:25s}: mean={means[i]:8.4f}, std={stds[i]:8.4f}")
    
    print(f"\nPreprocessing complete!")
    print(f"Normalization parameters saved to: {os.path.join(args.output_dir, 'normalization_params.json')}")
    print(f"Conditioning stats added to HDF5 files as 'conditioning_stats' dataset")


if __name__ == "__main__":
    main()