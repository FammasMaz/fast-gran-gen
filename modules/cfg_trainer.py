"""
Classifier-Free Guidance (CFG) Training utilities for conditional 3D voxel diffusion.

This module provides utilities and classes for training diffusion models with
classifier-free guidance for conditional generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from utils.conditioning_utils import apply_conditioning_dropout, ConditioningManager


class CFGTrainingMixin:
    """
    Mixin class that adds CFG training capabilities to any trainer.
    Can be mixed into existing trainer classes to add conditioning support.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize conditioning-related attributes
        self.conditioning_mode = getattr(self.args, 'conditioning_mode', False)
        self.cfg_dropout_prob = getattr(self.args, 'cfg_dropout_prob', 0.1)
        self.conditioning_manager = None
        
        if self.conditioning_mode:
            self._setup_conditioning()
    
    def _setup_conditioning(self):
        """Setup conditioning manager and validate configuration."""
        normalization_params_path = getattr(self.args, 'normalization_params_path', None)
        
        if normalization_params_path:
            self.conditioning_manager = ConditioningManager(normalization_params_path)
            print(f"Conditioning manager initialized with normalization params from: {normalization_params_path}")
        else:
            print("Warning: Conditioning mode enabled but no normalization params path provided")
            self.conditioning_manager = None
    
    def prepare_batch_with_conditioning(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare batch data for training, handling both dict and tensor formats.
        
        Args:
            batch: Batch data (either tensor or dict with 'voxels' and 'conditioning' keys)
            
        Returns:
            Tuple of (voxel_data, conditioning_data)
        """
        if isinstance(batch, dict):
            # New format with conditioning
            voxel_data = batch['voxels']
            conditioning_data = batch.get('conditioning', None)
        else:
            # Legacy format - just voxel data
            voxel_data = batch
            conditioning_data = None
        
        # Move to device
        voxel_data = voxel_data.to(self.device)
        if conditioning_data is not None:
            conditioning_data = conditioning_data.to(self.device)
        
        return voxel_data, conditioning_data
    
    def apply_cfg_dropout(self, conditioning_data: torch.Tensor) -> torch.Tensor:
        """
        Apply conditioning dropout for CFG training.
        
        Args:
            conditioning_data: Conditioning tensor
            
        Returns:
            Conditioning tensor with dropout applied
        """
        if not self.conditioning_mode or conditioning_data is None:
            return conditioning_data
        
        if self.conditioning_manager is None:
            # Create null conditioning manually
            null_conditioning = torch.zeros_like(conditioning_data)
        else:
            null_conditioning = self.conditioning_manager.create_null_conditioning(
                conditioning_data.shape[0], conditioning_data.device
            )
        
        return apply_conditioning_dropout(
            conditioning_data, null_conditioning, self.cfg_dropout_prob
        )
    
    def compute_conditional_loss(
        self, 
        model_output: torch.Tensor, 
        target: torch.Tensor, 
        conditioning_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for conditional training.
        Can be overridden to implement custom loss functions.
        
        Args:
            model_output: Model predictions
            target: Ground truth targets
            conditioning_data: Conditioning information (for potential loss weighting)
            
        Returns:
            Computed loss
        """
        # Standard MSE loss - can be extended for more sophisticated conditioning losses
        loss = F.mse_loss(model_output, target)
        
        # Optional: Add conditioning-specific loss terms
        if self.conditioning_mode and conditioning_data is not None:
            # Could add reconstruction loss, perceptual loss, etc.
            pass
        
        return loss
    
    def get_model_kwargs_for_conditioning(self, conditioning_data: Optional[torch.Tensor]) -> Dict:
        """
        Prepare model kwargs with conditioning information.
        
        Args:
            conditioning_data: Conditioning tensor
            
        Returns:
            Dictionary of kwargs for model forward pass
        """
        kwargs = {}
        
        if self.conditioning_mode and conditioning_data is not None:
            kwargs['conditioning_stats'] = conditioning_data
        
        return kwargs


class CFGTrainer(CFGTrainingMixin):
    """
    Standalone CFG trainer class.
    Can be used as a base class for new trainers or integrated with existing ones.
    """
    
    def __init__(self, model, train_loader, val_loader, args, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = args.device
        
        # Initialize mixin
        super().__init__()
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=getattr(args, 'diffusion_lr', 1e-4)
        )
        
        # Setup noise scheduler
        from diffusers import DDPMScheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=getattr(args, 'timesteps', 1000),
            beta_schedule="squaredcos_cap_v2"
        )
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Perform a single training step with CFG.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Prepare batch data
        voxel_data, conditioning_data = self.prepare_batch_with_conditioning(batch)
        batch_size = voxel_data.shape[0]
        
        # Apply CFG dropout to conditioning
        if conditioning_data is not None:
            conditioning_data = self.apply_cfg_dropout(conditioning_data)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to voxels
        noise = torch.randn_like(voxel_data)
        noisy_voxels = self.noise_scheduler.add_noise(voxel_data, noise, timesteps)
        
        # Prepare model kwargs
        model_kwargs = self.get_model_kwargs_for_conditioning(conditioning_data)
        
        # Forward pass
        noise_pred = self.model(noisy_voxels, timesteps, **model_kwargs)
        
        # Compute loss
        if hasattr(noise_pred, 'sample'):
            noise_pred = noise_pred.sample
        
        loss = self.compute_conditional_loss(noise_pred, noise, conditioning_data)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def validate(self) -> Dict[str, float]:
        """
        Perform validation with conditional model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                voxel_data, conditioning_data = self.prepare_batch_with_conditioning(batch)
                batch_size = voxel_data.shape[0]
                
                # Don't apply CFG dropout during validation
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise
                noise = torch.randn_like(voxel_data)
                noisy_voxels = self.noise_scheduler.add_noise(voxel_data, noise, timesteps)
                
                # Forward pass
                model_kwargs = self.get_model_kwargs_for_conditioning(conditioning_data)
                noise_pred = self.model(noisy_voxels, timesteps, **model_kwargs)
                
                if hasattr(noise_pred, 'sample'):
                    noise_pred = noise_pred.sample
                
                # Compute loss
                loss = self.compute_conditional_loss(noise_pred, noise, conditioning_data)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}


def integrate_cfg_with_existing_trainer(trainer_instance, args):
    """
    Integrate CFG capabilities into an existing trainer instance.
    
    Args:
        trainer_instance: Existing trainer instance
        args: Arguments containing conditioning configuration
        
    Returns:
        Modified trainer instance with CFG capabilities
    """
    # Add CFG attributes
    trainer_instance.conditioning_mode = getattr(args, 'conditioning_mode', False)
    trainer_instance.cfg_dropout_prob = getattr(args, 'cfg_dropout_prob', 0.1)
    trainer_instance.conditioning_manager = None
    
    if trainer_instance.conditioning_mode:
        normalization_params_path = getattr(args, 'normalization_params_path', None)
        if normalization_params_path:
            trainer_instance.conditioning_manager = ConditioningManager(normalization_params_path)
    
    # Add CFG methods
    trainer_instance.prepare_batch_with_conditioning = lambda batch: CFGTrainingMixin.prepare_batch_with_conditioning(trainer_instance, batch)
    trainer_instance.apply_cfg_dropout = lambda conditioning_data: CFGTrainingMixin.apply_cfg_dropout(trainer_instance, conditioning_data)
    trainer_instance.get_model_kwargs_for_conditioning = lambda conditioning_data: CFGTrainingMixin.get_model_kwargs_for_conditioning(trainer_instance, conditioning_data)
    trainer_instance.compute_conditional_loss = lambda model_output, target, conditioning_data=None: CFGTrainingMixin.compute_conditional_loss(trainer_instance, model_output, target, conditioning_data)
    
    return trainer_instance


def create_conditional_model(args):
    """
    Create a conditional UNet3DModel based on arguments.
    
    Args:
        args: Arguments containing model configuration
        
    Returns:
        Configured UNet3DModel with conditioning support
    """
    from .unet import UNet3DModel
    
    # Determine conditioning parameters
    conditioning_dim = getattr(args, 'conditioning_dim', None) if getattr(args, 'conditioning_mode', False) else None
    conditioning_hidden_dim = getattr(args, 'conditioning_hidden_dim', None)
    conditioning_dropout = getattr(args, 'conditioning_dropout', 0.1)
    
    # Create model
    model = UNet3DModel(
        sample_size=(32, 64, 64),  # Default size, should match dataset
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        mid_block_type="UNetMidBlock3D",
        act_fn="silu",
        norm_num_groups=32,
        inpainting_mode=getattr(args, 'inpainting_mode', False),
        conditioning_dim=conditioning_dim,
        conditioning_hidden_dim=conditioning_hidden_dim,
        conditioning_dropout=conditioning_dropout
    )
    
    return model