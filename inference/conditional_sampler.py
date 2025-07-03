"""
Conditional sampling pipeline for 3D voxel diffusion with CFG.

This module provides utilities for generating voxel grids with
classifier-free guidance conditioning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from diffusers import DDPMScheduler, DDIMScheduler
from utils.conditioning_utils import (
    ConditioningManager, 
    prepare_cfg_conditioning, 
    apply_cfg_guidance,
    validate_guidance_scale,
    ConditioningScheduler
)
import json
from pathlib import Path


class ConditionalVoxelSampler:
    """
    Conditional sampler for 3D voxel generation using classifier-free guidance.
    """
    
    def __init__(
        self,
        model,
        scheduler_type: str = "ddpm",
        num_inference_steps: int = 50,
        device: str = "cuda",
        normalization_params_path: Optional[str] = None
    ):
        """
        Initialize the conditional sampler.
        
        Args:
            model: Trained UNet3DModel with conditioning support
            scheduler_type: Type of scheduler ("ddpm" or "ddim")
            num_inference_steps: Number of denoising steps
            device: Device to run inference on
            normalization_params_path: Path to conditioning normalization parameters
        """
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        # Setup scheduler
        if scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2"
            )
        elif scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2"
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Set inference timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Setup conditioning manager
        self.conditioning_manager = None
        if normalization_params_path:
            self.conditioning_manager = ConditioningManager(normalization_params_path)
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
    
    def generate_unconditional(
        self,
        batch_size: int = 1,
        sample_size: Tuple[int, int, int] = (32, 64, 64),
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate unconditional voxel grids.
        
        Args:
            batch_size: Number of samples to generate
            sample_size: Size of voxel grids (D, H, W)
            generator: Random generator for reproducibility
            
        Returns:
            Generated voxel grids of shape (batch_size, 1, D, H, W)
        """
        # Create random noise
        voxels = torch.randn(
            (batch_size, 1) + sample_size,
            device=self.device,
            generator=generator,
            dtype=torch.float32
        )
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # No conditioning for unconditional generation
                noise_pred = self.model(voxels, t)
                
                if hasattr(noise_pred, 'sample'):
                    noise_pred = noise_pred.sample
                
                # Scheduler step
                voxels = self.scheduler.step(noise_pred, t, voxels).prev_sample
        
        return voxels
    
    def generate_conditional(
        self,
        target_conditioning: Union[Dict[str, float], torch.Tensor],
        batch_size: int = 1,
        guidance_scale: float = 7.5,
        sample_size: Tuple[int, int, int] = (32, 64, 64),
        generator: Optional[torch.Generator] = None,
        use_dynamic_guidance: bool = False,
        guidance_scheduler: Optional[ConditioningScheduler] = None
    ) -> torch.Tensor:
        """
        Generate conditional voxel grids using CFG.
        
        Args:
            target_conditioning: Target conditioning (dict of features or tensor)
            batch_size: Number of samples to generate
            guidance_scale: CFG guidance strength
            sample_size: Size of voxel grids (D, H, W)
            generator: Random generator for reproducibility
            use_dynamic_guidance: Whether to use time-varying guidance scale
            guidance_scheduler: Scheduler for dynamic guidance (if enabled)
            
        Returns:
            Generated voxel grids of shape (batch_size, 1, D, H, W)
        """
        if self.conditioning_manager is None:
            raise ValueError("Conditioning manager not initialized. Provide normalization_params_path.")
        
        # Validate guidance scale
        guidance_scale = validate_guidance_scale(guidance_scale)
        
        # Prepare conditioning tensor
        if isinstance(target_conditioning, dict):
            conditioning_tensor = self.conditioning_manager.create_target_conditioning(
                target_conditioning, batch_size, self.device
            )
        else:
            conditioning_tensor = target_conditioning.to(self.device)
            if conditioning_tensor.shape[0] != batch_size:
                conditioning_tensor = conditioning_tensor.repeat(batch_size, 1)
        
        # Create null conditioning for CFG
        null_conditioning = self.conditioning_manager.create_null_conditioning(
            batch_size, self.device
        )
        
        # Create random noise
        voxels = torch.randn(
            (batch_size, 1) + sample_size,
            device=self.device,
            generator=generator,
            dtype=torch.float32
        )
        
        # Setup dynamic guidance if requested
        if use_dynamic_guidance and guidance_scheduler is None:
            guidance_scheduler = ConditioningScheduler(
                schedule_type="linear",
                min_scale=1.0,
                max_scale=guidance_scale
            )
        
        # Denoising loop with CFG
        for i, t in enumerate(self.scheduler.timesteps):
            # Get current guidance scale
            if use_dynamic_guidance and guidance_scheduler is not None:
                current_guidance_scale = guidance_scheduler.get_guidance_scale(
                    i, len(self.scheduler.timesteps)
                )
            else:
                current_guidance_scale = guidance_scale
            
            with torch.no_grad():
                # Prepare input for CFG (concatenate conditional and unconditional)
                voxel_input = torch.cat([voxels] * 2)
                timestep_input = torch.cat([t.unsqueeze(0)] * 2)
                conditioning_input = torch.cat([null_conditioning, conditioning_tensor])
                
                # Model forward pass
                noise_pred = self.model(
                    voxel_input, 
                    timestep_input, 
                    conditioning_stats=conditioning_input
                )
                
                if hasattr(noise_pred, 'sample'):
                    noise_pred = noise_pred.sample
                
                # Apply CFG
                noise_pred = apply_cfg_guidance(noise_pred, current_guidance_scale)
                
                # Scheduler step
                voxels = self.scheduler.step(noise_pred, t, voxels).prev_sample
        
        return voxels
    
    def generate_batch_with_different_conditioning(
        self,
        conditioning_list: List[Union[Dict[str, float], torch.Tensor]],
        guidance_scale: float = 7.5,
        sample_size: Tuple[int, int, int] = (32, 64, 64),
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate multiple voxel grids with different conditioning.
        
        Args:
            conditioning_list: List of conditioning specifications
            guidance_scale: CFG guidance strength
            sample_size: Size of voxel grids
            generator: Random generator
            
        Returns:
            Generated voxel grids of shape (len(conditioning_list), 1, D, H, W)
        """
        if self.conditioning_manager is None:
            raise ValueError("Conditioning manager not initialized.")
        
        batch_size = len(conditioning_list)
        
        # Prepare conditioning tensors
        conditioning_tensors = []
        for cond in conditioning_list:
            if isinstance(cond, dict):
                cond_tensor = self.conditioning_manager.create_target_conditioning(
                    cond, 1, self.device
                )
            else:
                cond_tensor = cond.to(self.device).unsqueeze(0)
            conditioning_tensors.append(cond_tensor)
        
        conditioning_batch = torch.cat(conditioning_tensors, dim=0)
        
        # Generate using batch conditioning
        return self.generate_conditional(
            conditioning_batch,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            sample_size=sample_size,
            generator=generator
        )
    
    def interpolate_conditioning(
        self,
        start_conditioning: Union[Dict[str, float], torch.Tensor],
        end_conditioning: Union[Dict[str, float], torch.Tensor],
        num_steps: int = 10,
        guidance_scale: float = 7.5,
        sample_size: Tuple[int, int, int] = (32, 64, 64),
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate voxel grids with interpolated conditioning.
        
        Args:
            start_conditioning: Starting conditioning
            end_conditioning: Ending conditioning
            num_steps: Number of interpolation steps
            guidance_scale: CFG guidance strength
            sample_size: Size of voxel grids
            generator: Random generator
            
        Returns:
            Generated voxel grids for each interpolation step
        """
        if self.conditioning_manager is None:
            raise ValueError("Conditioning manager not initialized.")
        
        # Convert to tensors
        if isinstance(start_conditioning, dict):
            start_tensor = self.conditioning_manager.create_target_conditioning(
                start_conditioning, 1, self.device
            )
        else:
            start_tensor = start_conditioning.to(self.device).unsqueeze(0)
        
        if isinstance(end_conditioning, dict):
            end_tensor = self.conditioning_manager.create_target_conditioning(
                end_conditioning, 1, self.device
            )
        else:
            end_tensor = end_conditioning.to(self.device).unsqueeze(0)
        
        # Create interpolation
        alphas = torch.linspace(0, 1, num_steps, device=self.device).unsqueeze(1)
        interpolated_conditioning = (1 - alphas) * start_tensor + alphas * end_tensor
        
        # Generate with interpolated conditioning
        return self.generate_conditional(
            interpolated_conditioning,
            batch_size=num_steps,
            guidance_scale=guidance_scale,
            sample_size=sample_size,
            generator=generator
        )
    
    def postprocess_voxels(
        self,
        voxels: torch.Tensor,
        threshold: float = 0.0,
        apply_sigmoid: bool = False
    ) -> np.ndarray:
        """
        Postprocess generated voxel grids.
        
        Args:
            voxels: Generated voxel grids
            threshold: Threshold for binary conversion
            apply_sigmoid: Whether to apply sigmoid before thresholding
            
        Returns:
            Postprocessed voxel grids as numpy arrays
        """
        # Move to CPU and convert to numpy
        voxels_np = voxels.cpu().numpy()
        
        # Remove channel dimension
        if voxels_np.shape[1] == 1:
            voxels_np = voxels_np.squeeze(1)
        
        # Apply sigmoid if requested
        if apply_sigmoid:
            voxels_np = 1 / (1 + np.exp(-voxels_np))
        
        # Apply threshold for binary conversion
        binary_voxels = (voxels_np > threshold).astype(np.float32)
        
        return binary_voxels
    
    def save_conditioning_examples(self, output_path: str):
        """
        Save example conditioning configurations to file.
        
        Args:
            output_path: Path to save examples
        """
        if self.conditioning_manager is None:
            print("Warning: No conditioning manager available")
            return
        
        examples = {
            "low_density": {
                "density": 0.1,
                "volume": 500,
                "num_components": 1,
                "largest_component_ratio": 0.9,
                "surface_to_volume_ratio": 0.8,
                "bbox_fill_ratio": 0.2,
                "aspect_ratio_dh": 1.0,
                "aspect_ratio_hw": 1.0
            },
            "high_density": {
                "density": 0.8,
                "volume": 5000,
                "num_components": 1,
                "largest_component_ratio": 0.95,
                "surface_to_volume_ratio": 0.3,
                "bbox_fill_ratio": 0.8,
                "aspect_ratio_dh": 1.0,
                "aspect_ratio_hw": 1.0
            },
            "fragmented": {
                "density": 0.5,
                "volume": 2000,
                "num_components": 10,
                "largest_component_ratio": 0.3,
                "surface_to_volume_ratio": 0.9,
                "bbox_fill_ratio": 0.5,
                "aspect_ratio_dh": 1.0,
                "aspect_ratio_hw": 1.0
            },
            "elongated": {
                "density": 0.4,
                "volume": 1500,
                "num_components": 2,
                "largest_component_ratio": 0.7,
                "surface_to_volume_ratio": 0.6,
                "bbox_fill_ratio": 0.4,
                "aspect_ratio_dh": 2.0,
                "aspect_ratio_hw": 0.5
            }
        }
        
        # Add feature description
        output_data = {
            "feature_description": self.conditioning_manager.get_feature_description(),
            "examples": examples,
            "usage_instructions": {
                "basic_generation": "Use examples directly as target_conditioning dict",
                "custom_values": "Modify individual feature values as needed", 
                "interpolation": "Use two examples as start/end for interpolation",
                "guidance_scale": "Typical range: 1.0 (weak) to 15.0 (strong), default 7.5"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Conditioning examples saved to: {output_path}")


def load_conditional_model(
    model_path: str,
    conditioning_dim: int,
    device: str = "cuda"
):
    """
    Load a trained conditional model for inference.
    
    Args:
        model_path: Path to saved model checkpoint
        conditioning_dim: Dimension of conditioning features
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    from modules.unet import UNet3DModel
    
    # Create model with conditioning
    model = UNet3DModel(
        sample_size=(32, 64, 64),
        in_channels=1,
        out_channels=1,
        conditioning_dim=conditioning_dim
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def create_inference_pipeline(
    model_path: str,
    normalization_params_path: str,
    conditioning_dim: int = 8,
    scheduler_type: str = "ddim",
    num_inference_steps: int = 50,
    device: str = "cuda"
) -> ConditionalVoxelSampler:
    """
    Create a complete inference pipeline.
    
    Args:
        model_path: Path to trained model
        normalization_params_path: Path to conditioning normalization parameters
        conditioning_dim: Dimension of conditioning features
        scheduler_type: Type of scheduler
        num_inference_steps: Number of denoising steps
        device: Device for inference
        
    Returns:
        Ready-to-use conditional sampler
    """
    # Load model
    model = load_conditional_model(model_path, conditioning_dim, device)
    
    # Create sampler
    sampler = ConditionalVoxelSampler(
        model=model,
        scheduler_type=scheduler_type,
        num_inference_steps=num_inference_steps,
        device=device,
        normalization_params_path=normalization_params_path
    )
    
    return sampler