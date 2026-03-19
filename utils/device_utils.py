"""
Centralized device detection and helpers for MPS / CUDA / CPU backends.

Usage:
    from utils.device_utils import get_device, empty_cache, get_generator, get_system_info

    device = get_device("auto")          # resolves to best available backend
    gen    = get_generator(device, 42)   # CPU generator on MPS, device-matched otherwise
    empty_cache(device)                  # dispatches to the right cache-clear call
"""

import torch
import platform
import os


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _enable_mps_fallback():
    """Enable CPU fallback for MPS ops not yet implemented in Metal.

    This must be set *before* any MPS tensor is created.  Affected ops
    include ``aten::upsample_trilinear3d`` used by the 3-D U-Net.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("MPS: enabled CPU fallback for unsupported ops (trilinear 3D upsample, etc.)")

def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference string to a concrete ``torch.device``.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.
            * ``"auto"`` picks CUDA if available, then MPS, then CPU.
            * Any explicit name is honoured if the backend is available,
              otherwise falls back to CPU with a warning.

    Returns:
        A ``torch.device`` instance.
    """
    preference = preference.lower().strip()

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _enable_mps_fallback()
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _enable_mps_fallback()
            return torch.device("mps")
        print("Warning: MPS requested but not available — falling back to CPU.")
        return torch.device("cpu")

    # "cpu" or anything unrecognised
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_mps(device) -> bool:
    """Return ``True`` when *device* is MPS (Metal Performance Shaders)."""
    return torch.device(device).type == "mps"


def is_cuda(device) -> bool:
    """Return ``True`` when *device* is CUDA."""
    return torch.device(device).type == "cuda"


def get_device_name(device) -> str:
    """Human-readable device name for logging/notification messages."""
    device = torch.device(device)
    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(device.index or 0)
        except Exception:
            return "CUDA (unknown GPU)"
    if device.type == "mps":
        machine = platform.machine()
        return f"Apple Silicon MPS ({machine})"
    return "CPU"


def get_device_count(device) -> int:
    """Number of accelerators available for the given backend."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.device_count()
    if device.type == "mps":
        return 1  # MPS always exposes a single device
    return 0


def empty_cache(device) -> None:
    """Clear the accelerator memory cache (no-op on CPU)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # torch.mps.empty_cache() is available from PyTorch ≥ 2.1
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def get_generator(device, seed: int) -> torch.Generator:
    """Return a seeded ``torch.Generator`` compatible with *device*.

    MPS does **not** support on-device generators, so we always return a CPU
    generator in that case.  ``torch.randn(..., generator=cpu_gen, device="mps")``
    works correctly — the random numbers are generated on CPU and then moved.
    """
    device = torch.device(device)
    if device.type == "mps":
        return torch.Generator(device="cpu").manual_seed(seed)
    return torch.Generator(device=device).manual_seed(seed)


def randn_compatible(shape, device, generator=None, dtype=None):
    """Generate random normal tensor compatible with any device.

    On MPS the generator lives on CPU, but ``torch.randn(..., device='mps',
    generator=cpu_gen)`` raises in PyTorch ≥ 2.1.  This helper generates on
    CPU and moves the result to *device* transparently.
    """
    device = torch.device(device)
    if device.type == "mps":
        # Generate on CPU, then move
        kwargs = {}
        if generator is not None:
            kwargs["generator"] = generator
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.randn(shape, **kwargs).to(device)
    else:
        kwargs = {"device": device}
        if generator is not None:
            kwargs["generator"] = generator
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.randn(shape, **kwargs)


def manual_seed_all(device, seed: int) -> None:
    """Set global seeds for the given backend (plus the standard ``torch.manual_seed``)."""
    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # MPS and CPU don't have a separate manual_seed_all


def get_system_info(device) -> dict:
    """Gather system information for logging, safe on any backend."""
    device = torch.device(device)
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": device.type,
        "device_name": get_device_name(device),
        "device_count": get_device_count(device),
    }

    if device.type == "cuda":
        info["cuda_available"] = True
        info["current_device"] = torch.cuda.current_device()
    elif device.type == "mps":
        info["mps_available"] = True
    else:
        info["cpu_only"] = True

    return info


def get_mixed_precision(device, requested: bool = True) -> str | None:
    """Return the mixed-precision string suitable for ``Accelerator``.

    MPS does not support ``fp16`` mixed-precision training, so this
    returns ``None`` (full precision) when the device is MPS.

    Args:
        device: Target device.
        requested: Whether the user asked for mixed precision.

    Returns:
        ``"fp16"`` or ``None``.
    """
    if not requested:
        return None
    device = torch.device(device)
    if device.type == "mps":
        print("Note: fp16 mixed precision is not supported on MPS — using full precision.")
        return None
    return "fp16"


def get_torch_dtype(dtype_str: str, device=None):
    """Convert a dtype string (``'float32'``, ``'float16'``) to a ``torch.dtype``.

    Returns ``None`` when the string is ``'auto'`` — callers should omit
    ``torch_dtype`` from ``from_pretrained()`` in that case so the model
    keeps its saved precision.

    If *device* is MPS and ``float16`` is requested, falls back to ``float32``
    because the MPS backend does not reliably support fp16 model weights
    (mixed f16/f32 operations crash in Metal).
    """
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "auto": None,
    }
    dtype = mapping.get(dtype_str.lower().strip(), torch.float32)

    # MPS does not reliably support fp16 for 3D convolution models.
    # Multiple internal ops (GroupNorm, timestep embeddings, residual adds)
    # mix fp32/fp16 and crash with "mps.add requires same element type".
    # Apple Silicon also lacks Tensor Cores, so fp16 provides no speedup.
    if dtype == torch.float16 and device is not None and torch.device(device).type == "mps":
        print("Warning: float16 is not supported on MPS for 3D models — using float32 instead.")
        print("  Tip: use --low-memory for memory savings on MPS.")
        return torch.float32

    return dtype


def apply_low_memory_defaults(pipeline, device=None) -> None:
    """Apply memory-saving settings to a diffusers pipeline.

    Currently enables **attention slicing** which computes attention in
    chunks instead of all-at-once.  This is the safest low-memory
    optimisation and works on every backend (CUDA, MPS, CPU).

    Args:
        pipeline: A diffusers ``DiffusionPipeline`` (or subclass).
        device: Optional device — reserved for future per-backend tuning.
    """
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("auto")
        print("Low-memory mode: enabled attention slicing")

