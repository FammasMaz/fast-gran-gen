"""
Conditioning utilities for 3D voxel diffusion model.

This module provides utilities for handling conditioning information,
including feature normalization, null conditioning, and CFG utilities.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Union


class ConditioningManager:
    """
    Manages conditioning information for the diffusion model.
    Handles feature normalization, null conditioning, and validation.
    """
    
    def __init__(self, normalization_params_path: Optional[str] = None):
        """
        Initialize the conditioning manager.
        
        Args:
            normalization_params_path: Path to normalization parameters JSON file
        """
        self.feature_names = None
        self.means = None
        self.stds = None
        self.n_features = None
        
        if normalization_params_path and os.path.exists(normalization_params_path):
            self.load_normalization_params(normalization_params_path)
    
    def load_normalization_params(self, params_path: str):
        """
        Load normalization parameters from JSON file.
        
        Args:
            params_path: Path to normalization parameters file
        """
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        self.feature_names = params['feature_names']
        self.means = np.array(params['means'])
        self.stds = np.array(params['stds'])
        self.n_features = params['n_features']
        
        print(f"Loaded normalization params for {self.n_features} features: {self.feature_names}")
    
    def normalize_features(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Normalize features using stored parameters.
        
        Args:
            features: Raw feature values
            
        Returns:
            Normalized features as torch tensor
        """
        if self.means is None or self.stds is None:
            raise ValueError("Normalization parameters not loaded. Call load_normalization_params first.")
        
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        features = features.float()
        
        means_tensor = torch.from_numpy(self.means).float()
        stds_tensor = torch.from_numpy(self.stds).float()
        
        # Move to same device as features
        means_tensor = means_tensor.to(features.device)
        stds_tensor = stds_tensor.to(features.device)
        
        normalized = (features - means_tensor) / stds_tensor
        return normalized
    
    def denormalize_features(self, normalized_features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Denormalize features back to original scale.
        
        Args:
            normalized_features: Normalized feature values
            
        Returns:
            Denormalized features as torch tensor
        """
        if self.means is None or self.stds is None:
            raise ValueError("Normalization parameters not loaded. Call load_normalization_params first.")
        
        if isinstance(normalized_features, np.ndarray):
            normalized_features = torch.from_numpy(normalized_features)
        
        normalized_features = normalized_features.float()
        
        means_tensor = torch.from_numpy(self.means).float()
        stds_tensor = torch.from_numpy(self.stds).float()
        
        # Move to same device
        means_tensor = means_tensor.to(normalized_features.device)
        stds_tensor = stds_tensor.to(normalized_features.device)
        
        denormalized = normalized_features * stds_tensor + means_tensor
        return denormalized
    
    def create_null_conditioning(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create null conditioning vector for CFG.
        
        Args:
            batch_size: Number of samples in batch
            device: Target device
            
        Returns:
            Null conditioning tensor of shape (batch_size, n_features)
        """
        if self.n_features is None:
            raise ValueError("Number of features not set. Load normalization parameters first.")
        
        return torch.zeros(batch_size, self.n_features, device=device, dtype=torch.float32)
    
    def validate_conditioning_vector(self, conditioning: torch.Tensor) -> bool:
        """
        Validate that conditioning vector has correct shape.
        
        Args:
            conditioning: Conditioning tensor
            
        Returns:
            True if valid, False otherwise
        """
        if self.n_features is None:
            return False
        
        if conditioning.dim() != 2:
            return False
        
        if conditioning.shape[-1] != self.n_features:
            return False
        
        return True
    
    def create_target_conditioning(self, target_stats: Dict[str, float], 
                                 batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        Create conditioning vector from target statistics.
        
        Args:
            target_stats: Dictionary of target feature values
            batch_size: Number of conditioning vectors to create
            device: Target device
            
        Returns:
            Normalized conditioning tensor
        """
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Call load_normalization_params first.")
        
        if device is None:
            device = torch.device('cpu')
        
        # Create feature vector from target stats
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in target_stats:
                feature_vector.append(target_stats[feature_name])
            else:
                # Use mean value for missing features
                feature_idx = self.feature_names.index(feature_name)
                feature_vector.append(self.means[feature_idx])
        
        feature_vector = np.array(feature_vector)
        
        # Normalize
        normalized = self.normalize_features(feature_vector)
        
        # Expand to batch size
        conditioning = normalized.unsqueeze(0).repeat(batch_size, 1)
        conditioning = conditioning.to(device)
        
        return conditioning
    
    def get_feature_description(self) -> str:
        """
        Get a human-readable description of the features.
        
        Returns:
            Feature description string
        """
        if self.feature_names is None:
            return "No features loaded"
        
        descriptions = {
            'density': 'Voxel density (0-1)',
            'volume': 'Total volume in voxels',
            'num_components': 'Number of connected components',
            'largest_component_ratio': 'Ratio of largest component to total volume',
            'surface_to_volume_ratio': 'Surface area to volume ratio',
            'bbox_fill_ratio': 'Bounding box fill ratio',
            'aspect_ratio_dh': 'Depth to height aspect ratio',
            'aspect_ratio_dw': 'Depth to width aspect ratio', 
            'aspect_ratio_hw': 'Height to width aspect ratio',
            'mean_component_size': 'Average component size',
            'std_component_size': 'Standard deviation of component sizes'
        }
        
        result = "Available conditioning features:\n"
        for i, feature in enumerate(self.feature_names):
            desc = descriptions.get(feature, 'No description available')
            mean_val = self.means[i] if self.means is not None else 'N/A'
            std_val = self.stds[i] if self.stds is not None else 'N/A'
            result += f"  {feature}: {desc} (μ={mean_val:.3f}, σ={std_val:.3f})\n"
        
        return result


def apply_conditioning_dropout(conditioning: torch.Tensor, null_conditioning: torch.Tensor, 
                             dropout_prob: float) -> torch.Tensor:
    """
    Apply conditioning dropout for CFG training.
    
    Args:
        conditioning: Original conditioning tensor
        null_conditioning: Null conditioning tensor
        dropout_prob: Probability of replacing with null conditioning
        
    Returns:
        Conditioning tensor with dropout applied
    """
    if dropout_prob <= 0:
        return conditioning
    
    batch_size = conditioning.shape[0]
    mask = torch.rand(batch_size, device=conditioning.device) < dropout_prob
    
    # Apply mask
    result = conditioning.clone()
    result[mask] = null_conditioning[mask]
    
    return result


def prepare_cfg_conditioning(conditioning: torch.Tensor, null_conditioning: torch.Tensor) -> torch.Tensor:
    """
    Prepare conditioning for CFG inference by concatenating conditional and unconditional.
    
    Args:
        conditioning: Conditional features
        null_conditioning: Null conditioning for unconditional generation
        
    Returns:
        Concatenated conditioning tensor [unconditional, conditional]
    """
    return torch.cat([null_conditioning, conditioning], dim=0)


def apply_cfg_guidance(noise_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """
    Apply classifier-free guidance to noise predictions.
    
    Args:
        noise_pred: Concatenated noise predictions [unconditional, conditional]
        guidance_scale: Guidance strength
        
    Returns:
        Guided noise prediction
    """
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred_guided


def validate_guidance_scale(guidance_scale: float) -> float:
    """
    Validate and clamp guidance scale to reasonable range.
    
    Args:
        guidance_scale: Input guidance scale
        
    Returns:
        Validated guidance scale
    """
    if guidance_scale < 1.0:
        print(f"Warning: guidance_scale {guidance_scale} < 1.0, setting to 1.0")
        return 1.0
    elif guidance_scale > 20.0:
        print(f"Warning: guidance_scale {guidance_scale} > 20.0, consider using lower values")
    
    return guidance_scale


class ConditioningScheduler:
    """
    Manages conditioning strength throughout the diffusion process.
    Allows for time-dependent conditioning strength.
    """
    
    def __init__(self, schedule_type: str = "constant", min_scale: float = 1.0, max_scale: float = 7.5):
        """
        Initialize conditioning scheduler.
        
        Args:
            schedule_type: Type of schedule ('constant', 'linear', 'cosine')
            min_scale: Minimum guidance scale
            max_scale: Maximum guidance scale
        """
        self.schedule_type = schedule_type
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def get_guidance_scale(self, timestep: int, total_timesteps: int) -> float:
        """
        Get guidance scale for current timestep.
        
        Args:
            timestep: Current timestep
            total_timesteps: Total number of timesteps
            
        Returns:
            Guidance scale for current timestep
        """
        if self.schedule_type == "constant":
            return self.max_scale
        
        # Normalize timestep to [0, 1]
        t = timestep / total_timesteps
        
        if self.schedule_type == "linear":
            # Linear interpolation from max to min scale
            return self.max_scale - t * (self.max_scale - self.min_scale)
        elif self.schedule_type == "cosine":
            # Cosine schedule
            scale_range = self.max_scale - self.min_scale
            return self.min_scale + scale_range * (1 + np.cos(np.pi * t)) / 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")