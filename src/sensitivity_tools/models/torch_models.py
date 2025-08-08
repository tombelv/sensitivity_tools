"""
PyTorch implementation of sensitivity computation for dynamical systems
"""

from typing import Optional, Dict, Union, Any
from abc import ABC, abstractmethod

try:
    import torch
    from torch.func import jacfwd

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    jacfwd = None
    _TORCH_AVAILABLE = False

from ..settings import INTEGRATOR_TYPES


class BaseModel(ABC):
    """Base class for PyTorch-based sensitivity models"""
    
    def __init__(self, nq: int, nv: int, nu: int, ny: int, np: int, next: int, input_bounds, device: Optional[Any] = None):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this module")
            
        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.ny = ny
        self.np = np  # number of parameters
        self.next = next  # number of external variables (default)
        self.device = device or torch.device('cpu')
        
        if isinstance(input_bounds, tuple) and input_bounds == (-float('inf'), float('inf')):
            self.input_min = torch.full((self.nu,), input_bounds[0], device=self.device)
            self.input_max = torch.full((self.nu,), input_bounds[1], device=self.device)
        else:
            self.input_min = torch.tensor(input_bounds[0], device=self.device)
            self.input_max = torch.tensor(input_bounds[1], device=self.device)

        self.p_nom = []


class ModelParametric(BaseModel):
    """Parametric model for sensitivity computation using PyTorch"""

    def __init__(
        self,
        model_dynamics_parametric,
        nq: int,
        nv: int,
        nu: int,
        ny: int,
        np: int,
        next: int,
        input_bounds,
        integrator_params,
        controller=None,
        device: Optional[Any] = None,
    ):

        super().__init__(nq, nv, nu, ny, np, next, input_bounds, device)

        self.dynamics_parametric = model_dynamics_parametric
        self.controller = controller  # Initialize controller as None
        if controller is not None:
            self._setup_controller_sensitivity(controller)
        else:
            self.controller_sens = None

        integrator_type = integrator_params["method"]
        self.dt = integrator_params["step_size"]

        if integrator_type == INTEGRATOR_TYPES[0]:  # si_euler
            self.integrate_parametric = self.integrate_si_euler
        elif integrator_type == INTEGRATOR_TYPES[1]:  # euler
            self.integrate_parametric = self.integrate_euler
        elif integrator_type == INTEGRATOR_TYPES[2]:  # rk4
            self.integrate_parametric = self.integrate_rk4
        elif integrator_type == INTEGRATOR_TYPES[3]:  # custom_discrete
            self.integrate_parametric = model_dynamics_parametric
        else:
            raise ValueError(
                f"Integrator type '{integrator_type}' not supported. "
                f"Available types: {INTEGRATOR_TYPES}"
            )

        # Setup forward mode automatic differentiation functions
        self._setup_forward_mode_functions()

    def _setup_controller_sensitivity(self, controller):
        """Setup controller sensitivity computation"""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for controller sensitivity")
        
        def controller_wrapper(state_ctrl, reference, params, ext):
            return controller(state_ctrl, reference, params, ext)
        
        # Create the jacobian function
        self.controller_sens = jacfwd(controller_wrapper, argnums=0)
        


    @property
    def controller(self):
        return self._controller
    
    @controller.setter
    def controller(self, value):
        self._controller = value
        if value is not None:
            self._setup_controller_sensitivity(value)
        else:
            self.controller_sens = None


    def _setup_forward_mode_functions(self):
        """Setup forward mode automatic differentiation functions"""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for forward mode automatic differentiation")
        
        # Create wrapper functions for jacfwd
        def integrate_wrapper_state(state, inputs, params, ext, dt):
            return self.integrate_parametric(state, inputs, params, ext, dt)
        
        def integrate_wrapper_inputs(inputs, state, params, ext, dt):
            return self.integrate_parametric(state, inputs, params, ext, dt)
        
        def integrate_wrapper_params(params, state, inputs, ext, dt):
            return self.integrate_parametric(state, inputs, params, ext, dt)
        
        # Create forward mode Jacobian functions
        self.jacfwd_state = jacfwd(integrate_wrapper_state, argnums=0)
        self.jacfwd_inputs = jacfwd(integrate_wrapper_inputs, argnums=0)
        self.jacfwd_params = jacfwd(integrate_wrapper_params, argnums=0)
        

    def integrate_rk4(self, state: Any, inputs: Any, params: Any, ext, dt: float) -> Any:
        """
        One-step integration of the dynamics using RK4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params, ext)
        k2 = self.dynamics_parametric(state + k1 * dt / 2.0, inputs, params, ext)
        k3 = self.dynamics_parametric(state + k2 * dt / 2.0, inputs, params, ext)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params, ext)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_euler(self, state: Any, inputs: Any, params: Any, ext, dt: float) -> Any:
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics_parametric(state, inputs, params, ext)

    def integrate_si_euler(self, state: Any, inputs: Any, params: Any, ext, dt: float) -> Any:
        """
        Semi-implicit Euler integration.
        """
        v_kp1 = state[self.nq :] + dt * self.dynamics_parametric(state, inputs, params, ext)[self.nq :]
        q_kp1 = (
            state[: self.nq]
            + dt
            * self.dynamics_parametric(torch.cat([state[: self.nq], v_kp1]), inputs, params, ext)[
                : self.nq
            ]
        )
        return torch.cat([q_kp1, v_kp1])

    def sensitivity_step(
        self,
        state: Any,
        inputs: Any,
        params: Any,
        state_sensitivity: Any,
        input_gains: Any,
        dt: float,
        ext=0,
    ) -> Any:
        """
        Compute sensitivity step using forward mode automatic differentiation
        
        Args:
            state: Current state (nx,)
            inputs: Current inputs (nu,)
            params: Parameters (np,)
            state_sensitivity: State sensitivity matrix (nx, np)
            input_gains: Input gain matrix (nu, nx) - how inputs depend on states
            dt: Time step
            ext: External variables
            
        Returns:
            Next state sensitivity matrix (nx, np)
        """
        # Compute Jacobians using forward mode AD
        p_sens_state = self.jacfwd_state(state, inputs, params, ext, dt)  # (nx, nx)
        p_sens_inputs = self.jacfwd_inputs(inputs, state, params, ext, dt)  # (nx, nu)
        p_sens_params = self.jacfwd_params(params, state, inputs, ext, dt)  # (nx, np)

        return (p_sens_state + p_sens_inputs @ input_gains) @ state_sensitivity + p_sens_params


    def sensitivity_step_closed_loop(
        self,
        state: Any,
        reference: Any,
        params: Any,
        state_sensitivity: Any,
        dt: float,
        ext=0,
    ) -> Any:
        """
        Compute closed-loop sensitivity step using forward mode automatic differentiation
        """
        if self.controller is None:
            raise ValueError("Controller must be defined for closed-loop sensitivity computation")
            
        inputs = self.controller(state, reference, params, ext)

        # Compute Jacobians using forward mode AD
        p_sens_state = self.jacfwd_state(state, inputs, params, ext, dt)
        p_sens_inputs = self.jacfwd_inputs(inputs, state, params, ext, dt)
        p_sens_params = self.jacfwd_params(params, state, inputs, ext, dt)

        input_sens_state = self.controller_sens(state, reference, params, ext)

        return (
            p_sens_state + p_sens_inputs @ input_sens_state
        ) @ state_sensitivity + p_sens_params, inputs


class Model(ModelParametric):
    """Non-parametric model with fixed nominal parameters"""

    def __init__(self, config, model_dynamics, controller=None, device: Optional[Any] = None):
        """
        PyTorch model with ModelConfig interface.
        
        Args:
            config: ModelConfig object containing all model parameters
            model_dynamics: Model dynamics function with signature (state, inputs, params, ext)
            controller: Optional controller function with signature (state, reference, params, ext)
            device: PyTorch device for computations
        """
        
        # Import here to avoid circular imports
        from ..settings import ModelConfig
        
        if not isinstance(config, ModelConfig):
            raise ValueError("First argument must be a ModelConfig object")
        
        # Extract parameters from config
        p_nom = config.p_nom if config.p_nom is not None else []
        
        super().__init__(
            model_dynamics,
            config.nq,
            config.nv,
            config.nu,
            config.ny,
            len(p_nom),
            config.next,
            input_bounds=(-float('inf'), float('inf')),
            integrator_params=config.integrator_params,
            controller=controller,
            device=device,
        )

        self.p_nom = torch.tensor(p_nom, device=self.device)
        self.controller = controller
        
        # Add torch.compile support for better performance
        self._compile()

    def _compile(self):
        """Setup compiled functions for better performance"""
        # Compile integration function
        self.integrate = torch.compile(self.integrate)
        
        # Compile sensitivity functions
        self.sensitivity_step = torch.compile(self.sensitivity_step)
        
        self.sensitivity_step_closed_loop = torch.compile(self.sensitivity_step_closed_loop)

    def integrate(self, state, inputs, ext=0):
        """Integrate the system one step forward"""
        return self.integrate_parametric(state, inputs, self.p_nom, ext, self.dt)

    def sensitivity_step(self, state, inputs, state_sensitivity, input_gains, ext=0):
        """Compute open-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step(state, inputs, self.p_nom, state_sensitivity, input_gains, self.dt, ext)

    def sensitivity_step_closed_loop(self, state, reference, state_sensitivity, ext=0):
        """Compute closed-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step_closed_loop(state, reference, self.p_nom, state_sensitivity, self.dt, ext)



def ellipsoid_radius_torch(
    sensitivity: Any, scaling_matrix: Any, direction: Optional[Any] = None
) -> Any:
    """
    Compute the ellipsoid radius using PyTorch tensors

    Parameters
    ----------
    sensitivity : torch.Tensor
        Sensitivity matrix (Pi)
    scaling_matrix : torch.Tensor
        Scaling/weighting matrix (W)
    direction : torch.Tensor, optional
        Direction vector (n). If None, returns radii along Euclidean basis

    Returns
    -------
    torch.Tensor
        Ellipsoid radius(i) in the specified direction(s)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this function")

    if direction is None:
        return torch.sqrt(torch.diag(sensitivity @ scaling_matrix @ sensitivity.T))
    else:
        return torch.sqrt(direction.T @ sensitivity @ scaling_matrix @ sensitivity.T @ direction)


def ellipsoid_radii_torch(sensitivity: Any, scaling_matrix: Any) -> Any:
    """
    Compute ellipsoid radii along all coordinate axes using PyTorch

    Parameters
    ----------
    sensitivity : torch.Tensor
        Sensitivity matrix (Pi)
    scaling_matrix : torch.Tensor
        Scaling/weighting matrix (W)

    Returns
    -------
    torch.Tensor
        Array of ellipsoid radii along each coordinate axis
    """
    return ellipsoid_radius_torch(sensitivity, scaling_matrix, direction=None)
