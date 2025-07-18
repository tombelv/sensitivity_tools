"""
Sensitivity Tools - A package for computing closed-loop sensitivity of dynamical systems

This package provides tools for computing sensitivities of dynamical systems
with implementations in JAX, CasADi, and PyTorch.
"""

__version__ = "0.1.0"

from .settings import INTEGRATOR_TYPES
from .utils import ellipsoid_radius, ellipsoid_radii, integrate_rk4_function

# JAX implementation
try:
    from . import jax_models

    _JAX_AVAILABLE = True
except ImportError:
    jax_models = None
    _JAX_AVAILABLE = False

# CasADi implementation
try:
    from . import casadi_models
    from .casadi_models import ellipsoid_radii_casadi

    _CASADI_AVAILABLE = True
except ImportError:
    casadi_models = None
    _CASADI_AVAILABLE = False

# PyTorch implementation
try:
    from . import torch_models
    from .torch_models import ellipsoid_radius_torch, ellipsoid_radii_torch

    _PYTORCH_AVAILABLE = True
except ImportError:
    torch_models = None
    _PYTORCH_AVAILABLE = False

__all__ = [
    "INTEGRATOR_TYPES",
    "ellipsoid_radius",
    "ellipsoid_radii",
    "integrate_rk4_function",
    "jax_models",
    "casadi_models",
    "torch_models",
]

# Add backend-specific functions if available
if _CASADI_AVAILABLE:
    __all__.append("ellipsoid_radii_casadi")

if _PYTORCH_AVAILABLE:
    __all__.extend(["ellipsoid_radius_torch", "ellipsoid_radii_torch"])
