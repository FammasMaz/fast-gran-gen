"""
Ternary (1.58-bit) Quantization-Aware Training for 3D Diffusion UNet.

Implements BitNet b1.58-style ternary weight quantization {-1, 0, +1} with
absmean scaling and Straight-Through Estimator (STE) for gradient computation.

References:
    - BitNet b1.58 (arXiv:2504.12285)
    - BitNet b1.58 Reloaded (arXiv:2407.09527)
    - TernaryLM (arXiv:2602.07374)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class TernaryQuantize(torch.autograd.Function):
    """Quantize weights to {-1, 0, +1} with absmean scaling.

    Forward: w_q = clamp(round(w / alpha), -1, 1) * alpha
        where alpha = mean(|w|)
    Backward: Straight-Through Estimator (gradient passes through unchanged).
    """

    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean()
        w_scaled = w / (alpha + 1e-8)
        w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)
        ctx.save_for_backward(w)
        return w_ternary * alpha

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ARG004 - ctx unused but required by autograd
        return grad_output  # STE: pass gradient through unchanged


class BitConv3d(nn.Module):
    """Drop-in replacement for nn.Conv3d with ternary weight quantization.

    Stores full-precision latent weights for gradient updates. During forward,
    weights are quantized to {-1, 0, +1} scaled by absmean. Gradients flow
    through the quantization via STE.

    state_dict keys are identical to nn.Conv3d ('weight', 'bias').
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size, **factory_kwargs
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_quantized: torch.Tensor = TernaryQuantize.apply(self.weight)
        return F.conv3d(x, w_quantized, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_conv3d(cls, conv):
        """Create BitConv3d from an existing nn.Conv3d, preserving device and dtype."""
        device = conv.weight.device
        dtype = conv.weight.dtype
        new = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=device,
            dtype=dtype,
        )
        new.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)
        return new


class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weight quantization.

    Same pattern as BitConv3d but for linear layers.
    state_dict keys are identical to nn.Linear ('weight', 'bias').
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_quantized: torch.Tensor = TernaryQuantize.apply(self.weight)
        return F.linear(x, w_quantized, self.bias)

    @classmethod
    def from_linear(cls, linear):
        """Create BitLinear from an existing nn.Linear, preserving device and dtype."""
        device = linear.weight.device
        dtype = linear.weight.dtype
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        new.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        return new


# Fixed skip list — these modules stay full precision
_DEFAULT_SKIP_PREFIXES = ('conv_in', 'conv_out', 'time_embedding')


def quantize_model(model, skip_prefixes=None):
    """In-place swap of Conv3d -> BitConv3d and Linear -> BitLinear.

    Walks the module tree and replaces layers, preserving device and dtype.
    Skips modules whose full path starts with any prefix in skip_prefixes.

    Args:
        model: nn.Module to quantize in-place.
        skip_prefixes: Tuple of module name prefixes to skip.
            Defaults to ('conv_in', 'conv_out', 'time_embedding').
    """
    if skip_prefixes is None:
        skip_prefixes = _DEFAULT_SKIP_PREFIXES

    replacements = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix) or name == prefix for prefix in skip_prefixes):
            continue
        if isinstance(module, nn.Conv3d) and not isinstance(module, BitConv3d):
            replacements.append((name, BitConv3d.from_conv3d(module)))
        elif isinstance(module, nn.Linear) and not isinstance(module, BitLinear):
            replacements.append((name, BitLinear.from_linear(module)))

    for name, new_module in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model


def log_quantization_stats(model):
    """Compute and return quantization statistics for all BitConv3d/BitLinear layers.

    Returns:
        dict: {
            'layers': {name: {'sparsity': float, 'pos_frac': float, 'neg_frac': float, 'alpha': float}},
            'avg_sparsity': float,
            'avg_alpha': float,
            'num_quantized_layers': int,
        }
    """
    stats: dict = {'layers': {}}
    all_sparsity: list[float] = []
    all_alpha: list[float] = []

    for name, module in model.named_modules():
        if isinstance(module, (BitConv3d, BitLinear)):
            w = module.weight.data
            alpha = w.abs().mean().item()
            w_scaled = w / (alpha + 1e-8)
            w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)

            total = w_ternary.numel()
            zeros = (w_ternary == 0).sum().item()
            pos = (w_ternary == 1).sum().item()
            neg = (w_ternary == -1).sum().item()

            sparsity = zeros / total
            stats['layers'][name] = {
                'sparsity': sparsity,
                'pos_frac': pos / total,
                'neg_frac': neg / total,
                'alpha': alpha,
            }
            all_sparsity.append(sparsity)
            all_alpha.append(alpha)

    stats['num_quantized_layers'] = len(all_sparsity)
    stats['avg_sparsity'] = sum(all_sparsity) / len(all_sparsity) if all_sparsity else 0.0
    stats['avg_alpha'] = sum(all_alpha) / len(all_alpha) if all_alpha else 0.0

    return stats


def detect_quantization_from_metadata(model_path):
    """Check training_args.json at model_path for quantization metadata.

    Args:
        model_path: Path to the model directory (str or Path).

    Returns:
        bool: True if the model was trained with --quantize.
    """
    training_args_path = Path(model_path) / "training_args.json"
    if training_args_path.exists():
        with open(training_args_path) as f:
            training_args = json.load(f)
        return training_args.get("quantize", False)
    return False
