"""
Conditioning modules for 3D voxel diffusion model.

This module provides conditioning encoders and related components for
adding statistical conditioning to the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConditioningEncoder(nn.Module):
    """
    MLP-based encoder for statistical conditioning features.
    
    Encodes statistical features into embeddings that can be combined
    with timestep embeddings for conditional generation.
    """
    
    def __init__(
        self,
        stats_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "silu",
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize the conditioning encoder.
        
        Args:
            stats_dim: Dimension of input statistical features
            embed_dim: Output embedding dimension (should match time embedding dim)
            hidden_dim: Hidden layer dimension (defaults to embed_dim)
            num_layers: Number of MLP layers (minimum 2)
            activation: Activation function ('silu', 'relu', 'gelu')
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim
        self.num_layers = max(num_layers, 2)  # Ensure minimum 2 layers
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        # Choose activation function
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(stats_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """
        Encode statistical features to embeddings.
        
        Args:
            stats: Statistical features tensor of shape (batch_size, stats_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embed_dim)
        """
        if stats.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {stats.dim()}D")
        
        if stats.shape[-1] != self.stats_dim:
            raise ValueError(f"Expected stats_dim={self.stats_dim}, got {stats.shape[-1]}")
        
        return self.encoder(stats)


class AdaptiveConditioningEncoder(nn.Module):
    """
    Adaptive conditioning encoder that can handle variable-length feature vectors.
    Useful when different samples might have different numbers of features.
    """
    
    def __init__(
        self,
        max_stats_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        use_attention: bool = True
    ):
        """
        Initialize adaptive conditioning encoder.
        
        Args:
            max_stats_dim: Maximum dimension of statistical features
            embed_dim: Output embedding dimension
            hidden_dim: Hidden dimension for processing
            use_attention: Whether to use attention for adaptive pooling
        """
        super().__init__()
        
        self.max_stats_dim = max_stats_dim
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        # Feature embedding layer
        self.feature_embed = nn.Linear(1, hidden_dim)
        
        if use_attention:
            # Self-attention for adaptive pooling
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, stats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive feature processing.
        
        Args:
            stats: Statistical features (batch_size, max_features)
            mask: Optional mask for valid features (batch_size, max_features)
            
        Returns:
            Embedding tensor (batch_size, embed_dim)
        """
        batch_size, num_features = stats.shape
        
        # Embed each feature individually
        stats_reshaped = stats.unsqueeze(-1)  # (batch_size, num_features, 1)
        feature_embeds = self.feature_embed(stats_reshaped)  # (batch_size, num_features, hidden_dim)
        
        if self.use_attention:
            # Apply self-attention
            if mask is not None:
                # Convert mask to attention format (True = masked)
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            
            attended_features, _ = self.attention(
                feature_embeds, feature_embeds, feature_embeds,
                key_padding_mask=attn_mask
            )
            
            # Global average pooling (considering mask)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                attended_features = attended_features * mask_expanded
                pooled = attended_features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = attended_features.mean(dim=1)
        else:
            # Simple average pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                feature_embeds = feature_embeds * mask_expanded
                pooled = feature_embeds.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = feature_embeds.mean(dim=1)
        
        # Final projection
        output = self.output_proj(pooled)
        return output


class MultiScaleConditioningEncoder(nn.Module):
    """
    Multi-scale conditioning encoder that processes features at different scales.
    Useful for hierarchical conditioning or when features have different importance.
    """
    
    def __init__(
        self,
        stats_dim: int,
        embed_dim: int,
        num_scales: int = 3,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize multi-scale conditioning encoder.
        
        Args:
            stats_dim: Input statistical features dimension
            embed_dim: Output embedding dimension
            num_scales: Number of different scales to process
            hidden_dim: Hidden dimension for each scale
        """
        super().__init__()
        
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        if hidden_dim is None:
            hidden_dim = embed_dim // num_scales
        
        # Create encoders for different scales
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_scales)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Scale-specific normalization
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_scales)
        ])
    
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            stats: Statistical features (batch_size, stats_dim)
            
        Returns:
            Multi-scale embedding (batch_size, embed_dim)
        """
        scale_features = []
        
        for i, (encoder, norm) in enumerate(zip(self.scale_encoders, self.scale_norms)):
            # Apply different transformations for each scale
            # Scale 0: raw features
            # Scale 1: squared features (emphasize larger values)
            # Scale 2: log features (emphasize smaller values)
            if i == 0:
                scale_input = stats
            elif i == 1:
                scale_input = torch.square(stats)
            elif i == 2:
                scale_input = torch.log(torch.abs(stats) + 1e-8)
            else:
                # Additional scales can use other transformations
                scale_input = torch.tanh(stats * (i - 1))
            
            encoded = encoder(scale_input)
            normalized = norm(encoded)
            scale_features.append(normalized)
        
        # Concatenate and fuse
        concatenated = torch.cat(scale_features, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused


class ConditioningCrossAttention(nn.Module):
    """
    Cross-attention module for conditioning the UNet features directly.
    Alternative to embedding addition approach.
    """
    
    def __init__(
        self,
        feature_dim: int,
        conditioning_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize conditioning cross-attention.
        
        Args:
            feature_dim: Dimension of UNet features
            conditioning_dim: Dimension of conditioning embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Conditioning projection
        self.conditioning_proj = nn.Linear(conditioning_dim, feature_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning.
        
        Args:
            features: UNet features (batch_size, seq_len, feature_dim)
            conditioning: Conditioning embeddings (batch_size, conditioning_dim)
            mask: Optional attention mask
            
        Returns:
            Conditioned features (batch_size, seq_len, feature_dim)
        """
        # Project conditioning to feature dimension
        cond_proj = self.conditioning_proj(conditioning)  # (batch_size, feature_dim)
        cond_proj = cond_proj.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # Apply cross-attention
        attended_features, _ = self.cross_attention(
            query=features,
            key=cond_proj,
            value=cond_proj,
            key_padding_mask=mask
        )
        
        # Residual connection and normalization
        output = self.norm(features + attended_features)
        
        return output