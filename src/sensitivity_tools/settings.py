"""
Settings and constants for sensitivity tools package
"""

from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, field

# Integrator types supported by the package
INTEGRATOR_TYPES = [
    "si_euler",  # Semi-implicit Euler
    "euler",  # Explicit Euler
    "rk4",  # Runge-Kutta 4th order
    "custom_discrete",  # Custom discrete integrator
]


@dataclass
class ModelConfig:
    """Common configuration for all model backends"""
    
    # Required fields (no defaults)
    nq: int  # number of generalized coordinates
    nv: int  # number of degrees of freedom
    nu: int  # number of control inputs
    dt: float  # time step
    
    # Optional fields (with defaults)
    ny: int  # number of outputs (for controllers)
    next: int = 0  # number of external variables
    
    # Parameters
    p_nom: Union[list, Any] = None
    
    # Integration
    integrator_params: Dict = field(default_factory=lambda: {"method": "si_euler", # Default to semi-implicit Euler
                                                              "step_size": 0.0})  

    
    # Bounds (for optimization/control)
    input_bounds: Union[tuple, Any] = (-float('inf'), float('inf'))
    
    # Backend-specific
    device: Optional[Any] = None  # For PyTorch
    
    def __post_init__(self):
        """Validate configuration"""
        integrator_type = self.integrator_params.get("method", "si_euler")
        if integrator_type not in INTEGRATOR_TYPES:
            raise ValueError(
                f"Integrator type '{integrator_type}' not supported. "
                f"Available types: {INTEGRATOR_TYPES}"
            )
        
        if self.p_nom is None:
            self.p_nom = []

        if self.dt <= 0.0:
            raise ValueError("Time step 'dt' must be greater than 0.0")
        else:
            self.integrator_params["step_size"] = self.dt


def create_model_config(
    nq: int,
    nv: int,
    nu: int,
    p_nom: Union[list, Any] = None,
    dt: float = 0.0,
    integrator_type: str = "si_euler",
    ny: int = None,
    next: int = 0,
    input_bounds: Union[tuple, Any] = (-float('inf'), float('inf')),
    device: Optional[Any] = None,
    **kwargs
) -> ModelConfig:
    """
    Create a standardized model configuration
    
    Parameters
    ----------
    nq : int
        Number of generalized coordinates
    nv : int
        Number of degrees of freedom
    nu : int
        Number of control inputs
    p_nom : list or array-like, optional
        Nominal parameter values
    dt : float, default=0.0
        Integration time step
    integrator_type : str, default="si_euler"
        Integration method
    ny : int, default=0
        Number of outputs
    next : int, default=0
        Number of external variables
    input_bounds : tuple, default=(-inf, inf)
        Input bounds for optimization
    device : optional
        Device for PyTorch backend
    **kwargs
        Additional backend-specific parameters
    
    Returns
    -------
    ModelConfig
        Standardized configuration object
    """
    integrator_params = kwargs.get('integrator_params', {"method": integrator_type, "step_size": dt})
    if "method" not in integrator_params:
        integrator_params["method"] = integrator_type
    if "step_size" not in integrator_params:
        integrator_params["step_size"] = dt
        
    if ny is None:
        ny = nu

    return ModelConfig(
        nq=nq,
        nv=nv,
        nu=nu,
        dt=dt,
        ny=ny,
        next=next,
        p_nom=p_nom,
        integrator_params=integrator_params,
        input_bounds=input_bounds,
        device=device
    )