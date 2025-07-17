"""
PyTorch implementation of sensitivity computation for dynamical systems
"""

from typing import Optional, Dict, Union, Any
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from .settings import INTEGRATOR_TYPES


class BaseModel(ABC):
    """Base class for PyTorch-based sensitivity models"""

    def __init__(
        self,
        nq: int,
        nv: int,
        nu: int,
        np: int = 0,
        input_bounds: Union[tuple, Any] = (-float('inf'), float('inf')),
        device: Optional[Any] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this module")

        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.np = np
        self.device = device or torch.device('cpu')

        if isinstance(input_bounds, tuple) and input_bounds == (-float('inf'), float('inf')):
            self.input_min = torch.full((self.nu,), input_bounds[0], device=self.device)
            self.input_max = torch.full((self.nu,), input_bounds[1], device=self.device)
        else:
            self.input_min = torch.tensor(input_bounds[0], device=self.device)
            self.input_max = torch.tensor(input_bounds[1], device=self.device)

        self.nominal_parameters = torch.tensor([], device=self.device)

    def get_nq(self):
        return self.nq

    @abstractmethod
    def integrate(self, state: Any, inputs: Any, dt: float) -> Any:
        pass

    def integrate_sim(self, state: Any, inputs: Any, dt: float) -> Any:
        return self.integrate(state, inputs, dt)

    @abstractmethod
    def integrate_rollout(self, state: Any, inputs: Any, dt: float) -> Any:
        pass

    @abstractmethod
    def integrate_rollout_single(self, state: Any, inputs: Any, dt: float) -> Any:
        pass

    @abstractmethod
    def sensitivity_step(
        self,
        state: Any,
        inputs: Any,
        params: Any,
        state_sensitivity: Any,
        input_sensitivity: Any,
        dt: float,
    ) -> Any:
        pass


class ModelParametric(BaseModel):
    """Parametric model for sensitivity computation using PyTorch"""

    def __init__(
        self,
        model_dynamics_parametric,
        nq: int,
        nv: int,
        nu: int,
        np: int = 0,
        input_bounds: Union[tuple, Any] = (-float('inf'), float('inf')),
        integrator_type: str = "si_euler",
        integrator_params: Optional[Dict] = None,
        device: Optional[Any] = None,
    ):

        super().__init__(nq, nv, nu, np, input_bounds, device)

        self.dynamics_parametric = model_dynamics_parametric

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
                f"""
            Integrator type '{integrator_type}' not supported.
            Available types: {INTEGRATOR_TYPES}
            """
            )

    def integrate_rk4(self, state: Any, inputs: Any, params: Any, dt: float) -> Any:
        """
        One-step integration of the dynamics using RK4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params)
        k2 = self.dynamics_parametric(state + k1 * dt / 2.0, inputs, params)
        k3 = self.dynamics_parametric(state + k2 * dt / 2.0, inputs, params)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_euler(self, state: Any, inputs: Any, params: Any, dt: float) -> Any:
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics_parametric(state, inputs, params)

    def integrate_si_euler(self, state: Any, inputs: Any, params: Any, dt: float) -> Any:
        """
        Semi-implicit Euler integration.
        """
        v_kp1 = state[self.nq :] + dt * self.dynamics_parametric(state, inputs, params)[self.nq :]
        q_kp1 = (
            state[: self.nq]
            + dt
            * self.dynamics_parametric(torch.cat([state[: self.nq], v_kp1]), inputs, params)[
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
        input_sensitivity: Any,
        dt: float,
    ) -> Any:
        """
        Compute sensitivity step using automatic differentiation
        """
        # Enable gradient computation
        state_req_grad = state.requires_grad_(True)
        inputs_req_grad = inputs.requires_grad_(True)
        params_req_grad = params.requires_grad_(True)

        # Forward pass
        next_state = self.integrate_parametric(state_req_grad, inputs_req_grad, params_req_grad, dt)

        # Compute Jacobians
        batch_size = state.shape[0] if state.dim() > 1 else 1
        state_dim = state.shape[-1]
        input_dim = inputs.shape[-1]
        param_dim = params.shape[-1]

        # Initialize Jacobian matrices
        p_sens_state = torch.zeros(state_dim, state_dim, device=self.device)
        p_sens_inputs = torch.zeros(state_dim, input_dim, device=self.device)
        p_sens_params = torch.zeros(state_dim, param_dim, device=self.device)

        # Compute gradients for each output dimension
        for i in range(state_dim):
            grad_outputs = torch.zeros_like(next_state)
            grad_outputs[i] = 1.0

            grads = torch.autograd.grad(
                outputs=next_state,
                inputs=[state_req_grad, inputs_req_grad, params_req_grad],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )

            if grads[0] is not None:
                p_sens_state[i] = grads[0]
            if grads[1] is not None:
                p_sens_inputs[i] = grads[1]
            if grads[2] is not None:
                p_sens_params[i] = grads[2]

        return p_sens_state @ state_sensitivity + p_sens_inputs @ input_sensitivity + p_sens_params


class Model(ModelParametric):
    """Non-parametric model with fixed nominal parameters"""

    def __init__(
        self,
        model_dynamics,
        nq: int,
        nv: int,
        nu: int,
        nominal_parameters: list = [],
        input_bounds: Union[tuple, Any] = (-float('inf'), float('inf')),
        integrator_type: str = "si_euler",
        device: Optional[Any] = None,
    ):
        """
        :param model_dynamics: Should have signature (state, inputs, params)
        :param nq: Number of configuration variables
        :param nv: Number of pseudo velocities
        :param nu: Number of inputs
        :param nominal_parameters: Fixed parameters for the model
        :param input_bounds: Lower and upper bound of the inputs
        :param integrator_type: Available integrators are [si_euler, euler, rk4, custom_discrete]
        :param device: PyTorch device for computations
        """
        super().__init__(
            model_dynamics,
            nq,
            nv,
            nu,
            len(nominal_parameters),
            input_bounds,
            integrator_type,
            device=device,
        )

        self.nominal_parameters = torch.tensor(nominal_parameters, device=self.device)

        self.integrate_rollout_single = self.integrate

    def integrate(self, state: Any, inputs: Any, dt: float) -> Any:
        return self.integrate_parametric(state, inputs, self.nominal_parameters, dt)

    def integrate_rollout(self, state: Any, inputs: Any, dt: float) -> Any:
        return self.integrate(state, inputs, dt)

    def integrate_rollout_single(self, state: Any, inputs: Any, dt: float) -> Any:
        return self.integrate(state, inputs, dt)


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
