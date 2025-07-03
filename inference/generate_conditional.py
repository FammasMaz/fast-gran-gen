#!/usr/bin/env python3
"""
Script for generating conditional voxel grids using trained diffusion model.

This script provides a command-line interface for generating voxel grids
with specific statistical characteristics using classifier-free guidance.
"""

import argparse
import torch
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from conditional_sampler import create_inference_pipeline
import h5py
from typing import Dict, List


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_voxel_grid(voxel_grid: np.ndarray, output_path: str, metadata: Dict = None):
    """
    Save voxel grid to HDF5 file with metadata.
    
    Args:
        voxel_grid: Voxel grid to save
        output_path: Output file path
        metadata: Optional metadata dictionary
    """
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('voxels', data=voxel_grid)
        
        if metadata:
            # Save metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    f.attrs[key] = np.array(value)
                else:
                    f.attrs[key] = str(value)


def visualize_voxel_slice(voxel_grid: np.ndarray, output_path: str, slice_axis: int = 0):
    """
    Create visualization of voxel grid slices.
    
    Args:
        voxel_grid: Voxel grid to visualize
        output_path: Output image path
        slice_axis: Axis to slice along (0, 1, or 2)
    """
    if slice_axis == 0:
        mid_slice = voxel_grid[voxel_grid.shape[0] // 2, :, :]
        title = f"XY slice (Z={voxel_grid.shape[0] // 2})"
    elif slice_axis == 1:
        mid_slice = voxel_grid[:, voxel_grid.shape[1] // 2, :]
        title = f"XZ slice (Y={voxel_grid.shape[1] // 2})"
    else:
        mid_slice = voxel_grid[:, :, voxel_grid.shape[2] // 2]
        title = f"XY slice (Z={voxel_grid.shape[2] // 2})"
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mid_slice, cmap='binary', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_actual_stats(voxel_grid: np.ndarray) -> Dict[str, float]:
    """
    Compute actual statistics of generated voxel grid.
    
    Args:
        voxel_grid: Binary voxel grid
        
    Returns:
        Dictionary of computed statistics
    """
    from scipy import ndimage
    
    binary_grid = (voxel_grid > 0.5).astype(np.uint8)
    
    stats = {}
    stats['density'] = np.mean(binary_grid)
    stats['volume'] = np.sum(binary_grid)
    
    D, H, W = binary_grid.shape
    stats['aspect_ratio_dh'] = D / H if H > 0 else 0.0
    stats['aspect_ratio_dw'] = D / W if W > 0 else 0.0
    stats['aspect_ratio_hw'] = H / W if W > 0 else 0.0
    
    # Connected components
    labeled_grid, num_components = ndimage.label(binary_grid)
    stats['num_components'] = num_components
    
    if num_components > 0:
        component_sizes = np.bincount(labeled_grid.ravel())[1:]
        stats['largest_component_size'] = np.max(component_sizes)
        stats['largest_component_ratio'] = stats['largest_component_size'] / stats['volume'] if stats['volume'] > 0 else 0.0
    else:
        stats['largest_component_size'] = 0
        stats['largest_component_ratio'] = 0.0
    
    # Surface area approximation
    if stats['volume'] > 0:
        padded = np.pad(binary_grid, 1, mode='constant', constant_values=0)
        neighbors_sum = (
            padded[:-2, 1:-1, 1:-1] +  # front
            padded[2:, 1:-1, 1:-1] +   # back
            padded[1:-1, :-2, 1:-1] +  # top
            padded[1:-1, 2:, 1:-1] +   # bottom
            padded[1:-1, 1:-1, :-2] +  # left
            padded[1:-1, 1:-1, 2:]     # right
        )
        surface_voxels = binary_grid & (neighbors_sum < 6)
        stats['surface_area_approx'] = np.sum(surface_voxels)
        stats['surface_to_volume_ratio'] = stats['surface_area_approx'] / stats['volume']
    else:
        stats['surface_area_approx'] = 0
        stats['surface_to_volume_ratio'] = 0.0
    
    # Bounding box
    if stats['volume'] > 0:
        coords = np.where(binary_grid)
        min_coords = [np.min(c) for c in coords]
        max_coords = [np.max(c) for c in coords]
        bbox_size = [max_coords[i] - min_coords[i] + 1 for i in range(3)]
        bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
        stats['bbox_volume'] = bbox_volume
        stats['bbox_fill_ratio'] = stats['volume'] / bbox_volume if bbox_volume > 0 else 0.0
    else:
        stats['bbox_volume'] = 0
        stats['bbox_fill_ratio'] = 0.0
    
    return stats


def load_conditioning_config(config_path: str) -> Dict:
    """Load conditioning configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate conditional voxel grids")
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--normalization_params', type=str, required=True,
                       help='Path to conditioning normalization parameters')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save generated samples')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--scheduler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                       help='Scheduler type for sampling')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible generation')
    
    # Conditioning specification
    parser.add_argument('--conditioning_config', type=str, default=None,
                       help='Path to JSON file with conditioning specifications')
    parser.add_argument('--target_density', type=float, default=None,
                       help='Target density value')
    parser.add_argument('--target_volume', type=float, default=None,
                       help='Target volume value')
    parser.add_argument('--target_components', type=int, default=None,
                       help='Target number of connected components')
    
    # Model configuration
    parser.add_argument('--conditioning_dim', type=int, default=8,
                       help='Dimension of conditioning features')
    parser.add_argument('--sample_size', type=int, nargs=3, default=[32, 64, 64],
                       help='Size of generated voxel grids (D H W)')
    
    # Output options
    parser.add_argument('--save_hdf5', type=str2bool, default=True,
                       help='Save voxel grids as HDF5 files')
    parser.add_argument('--save_visualizations', type=str2bool, default=True,
                       help='Save slice visualizations')
    parser.add_argument('--compute_stats', type=str2bool, default=True,
                       help='Compute and save actual statistics')
    
    # Advanced options
    parser.add_argument('--interpolate', type=str2bool, default=False,
                       help='Generate interpolation between two conditioning points')
    parser.add_argument('--interpolation_steps', type=int, default=10,
                       help='Number of interpolation steps')
    parser.add_argument('--use_dynamic_guidance', type=str2bool, default=False,
                       help='Use time-varying guidance scale')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Threshold for binary conversion')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    
    # Create inference pipeline
    print("Loading model and creating inference pipeline...")
    sampler = create_inference_pipeline(
        model_path=args.model_path,
        normalization_params_path=args.normalization_params,
        conditioning_dim=args.conditioning_dim,
        scheduler_type=args.scheduler,
        num_inference_steps=args.num_steps
    )
    
    # Prepare conditioning
    target_conditioning = {}
    
    if args.conditioning_config:
        # Load from configuration file
        config = load_conditioning_config(args.conditioning_config)
        if 'target_conditioning' in config:
            target_conditioning = config['target_conditioning']
        elif 'examples' in config:
            # Use first example
            example_name = list(config['examples'].keys())[0]
            target_conditioning = config['examples'][example_name]
            print(f"Using example conditioning: {example_name}")
    
    # Override with command line arguments
    if args.target_density is not None:
        target_conditioning['density'] = args.target_density
    if args.target_volume is not None:
        target_conditioning['volume'] = args.target_volume
    if args.target_components is not None:
        target_conditioning['num_components'] = args.target_components
    
    # Generate samples
    if not target_conditioning:
        print("No conditioning specified, generating unconditional samples...")
        
        voxels = sampler.generate_unconditional(
            batch_size=args.num_samples,
            sample_size=tuple(args.sample_size),
            generator=generator
        )
        
        conditioning_used = None
        
    elif args.interpolate and args.num_samples >= 2:
        print("Generating interpolation between conditioning points...")
        
        # Create second conditioning point (modify first one)
        start_conditioning = target_conditioning.copy()
        end_conditioning = target_conditioning.copy()
        
        # Modify some values for end point
        if 'density' in end_conditioning:
            end_conditioning['density'] = max(0.1, min(0.9, end_conditioning['density'] + 0.3))
        if 'num_components' in end_conditioning:
            end_conditioning['num_components'] = max(1, end_conditioning['num_components'] + 3)
        
        voxels = sampler.interpolate_conditioning(
            start_conditioning=start_conditioning,
            end_conditioning=end_conditioning,
            num_steps=args.interpolation_steps,
            guidance_scale=args.guidance_scale,
            sample_size=tuple(args.sample_size),
            generator=generator
        )
        
        conditioning_used = {
            'start': start_conditioning,
            'end': end_conditioning,
            'interpolation_steps': args.interpolation_steps
        }
        
    else:
        print(f"Generating {args.num_samples} conditional samples...")
        print(f"Target conditioning: {target_conditioning}")
        
        voxels = sampler.generate_conditional(
            target_conditioning=target_conditioning,
            batch_size=args.num_samples,
            guidance_scale=args.guidance_scale,
            sample_size=tuple(args.sample_size),
            generator=generator,
            use_dynamic_guidance=args.use_dynamic_guidance
        )
        
        conditioning_used = target_conditioning
    
    # Postprocess voxels
    voxel_grids = sampler.postprocess_voxels(voxels, threshold=args.threshold)
    
    print(f"Generated {len(voxel_grids)} voxel grids")
    
    # Save results
    for i, voxel_grid in enumerate(voxel_grids):
        sample_name = f"sample_{i:04d}"
        
        # Compute actual statistics
        actual_stats = None
        if args.compute_stats:
            actual_stats = compute_actual_stats(voxel_grid)
            print(f"\nSample {i} actual statistics:")
            for key, value in actual_stats.items():
                print(f"  {key}: {value:.4f}")
        
        # Prepare metadata
        metadata = {
            'sample_index': i,
            'guidance_scale': args.guidance_scale,
            'num_denoising_steps': args.num_steps,
            'scheduler_type': args.scheduler,
            'threshold': args.threshold,
            'seed': args.seed
        }
        
        if conditioning_used:
            metadata['target_conditioning'] = conditioning_used
        
        if actual_stats:
            metadata['actual_stats'] = actual_stats
        
        # Save HDF5 file
        if args.save_hdf5:
            h5_path = os.path.join(args.output_dir, f"{sample_name}.h5")
            save_voxel_grid(voxel_grid, h5_path, metadata)
            print(f"Saved voxel grid: {h5_path}")
        
        # Save visualizations
        if args.save_visualizations:
            for axis in range(3):
                vis_path = os.path.join(args.output_dir, f"{sample_name}_slice_axis{axis}.png")
                visualize_voxel_slice(voxel_grid, vis_path, axis)
            print(f"Saved visualizations for sample {i}")
    
    # Save generation metadata
    generation_metadata = {
        'num_samples': len(voxel_grids),
        'model_path': args.model_path,
        'normalization_params': args.normalization_params,
        'generation_parameters': {
            'guidance_scale': args.guidance_scale,
            'num_denoising_steps': args.num_steps,
            'scheduler_type': args.scheduler,
            'sample_size': args.sample_size,
            'seed': args.seed,
            'threshold': args.threshold,
            'use_dynamic_guidance': args.use_dynamic_guidance
        },
        'conditioning_used': conditioning_used
    }
    
    metadata_path = os.path.join(args.output_dir, 'generation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(generation_metadata, f, indent=2)
    
    print(f"\nGeneration complete! Results saved to: {args.output_dir}")
    print(f"Generation metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()