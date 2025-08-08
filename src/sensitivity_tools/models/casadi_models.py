from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Union
from ..settings import INTEGRATOR_TYPES, ModelConfig

try:
    import casadi as cs
except ImportError:
    cs = None


class BaseModel(ABC):
    """Base class for CasADi-based sensitivity models"""
    
    def __init__(self, nq: int, nv: int, nu: int, ny: int, np: int, next: int, input_bounds):
        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.ny = ny
        self.np = np  # number of parameters
        self.next = next  # number of external variables (default)
        self.input_bounds = input_bounds
        self.p_nom = []


class ModelParametric(BaseModel):
    """Parametric model for sensitivity computation using CasADi"""
    
    def __init__(
        self,
        model_dynamics_parametric,
        nq: int,
        nv: int,
        nu: int,
        np: int,
        ny:int,
        next: int,
        input_bounds,
        integrator_params,
        controller=None
    ):
        
        super().__init__(nq, nv, nu, np, ny, next, input_bounds)
        
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
        
        # Create symbolic variables for sensitivity computation
        self._setup_sensitivity_functions()
        
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
    
    def _setup_sensitivity_functions(self):
        """Setup CasADi functions for sensitivity computation"""
        if cs is None:
            raise ImportError("CasADi is required for this class")
            
        x = cs.MX.sym('x', self.nx, 1)
        u = cs.MX.sym('u', self.nu, 1)
        p = cs.MX.sym('p', self.np, 1)
        ext = cs.MX.sym('ext', self.next, 1)
        dt = cs.MX.sym('dt', 1, 1)
        
        # Integrate one step
        x_kp1 = self.integrate_parametric(x, u, p, ext, dt)
        
        # Compute Jacobians
        dfdx = cs.jacobian(x_kp1, x)
        dfdu = cs.jacobian(x_kp1, u)
        dfdp = cs.jacobian(x_kp1, p)
        
        self.partial_sens_all = cs.Function(
            'partial_sens_all',
            [x, u, p, ext, dt],
            [dfdx, dfdu, dfdp],
            ['x', 'u', 'p', 'ext', 'dt'],
            ['dfdx', 'dfdu', 'dfdp'],
            {"cse": True}
        )
    
    def _setup_controller_sensitivity(self, controller):
            
        x = cs.MX.sym('x', self.nx, 1)
        ref = cs.MX.sym('ref', self.ny, 1)
        p = cs.MX.sym('p', self.np, 1)
        ext_sym = cs.MX.sym('ext', self.next, 1)
        
        ctrl_action = controller(x, ref, p, ext_sym)
        dudx = cs.jacobian(ctrl_action, x)
        
        self.controller_sens = cs.Function(
            'controller_sens',
            [x, ref, p, ext_sym],
            [dudx],
            ['x', 'ref', 'p', 'ext'],
            ['dudx'],
            {"cse": True},
        )

    def integrate_rk4(self, state, inputs, params, ext, dt):
        """One-step integration using RK4 method"""
        k1 = self.dynamics_parametric(state, inputs, params, ext)
        k2 = self.dynamics_parametric(state + k1 * dt / 2.0, inputs, params, ext)
        k3 = self.dynamics_parametric(state + k2 * dt / 2.0, inputs, params, ext)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params, ext)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_euler(self, state, inputs, params, ext, dt):
        """One-step integration using Euler method"""
        return state + dt * self.dynamics_parametric(state, inputs, params, ext)

    def integrate_si_euler(self, state, inputs, params, ext, dt):
        """Semi-implicit Euler integration"""
        v_kp1 = (
            state[self.nq :] + dt * self.dynamics_parametric(state, inputs, params, ext)[self.nq :]
        )
        return cs.vertcat(
            state[: self.nq]
            + dt * self.dynamics_parametric(cs.vertcat(state[: self.nq], v_kp1), inputs, params, ext)[: self.nq],
            v_kp1,
        )

    def sensitivity_step(self, state, inputs, state_sensitivity, input_gains, ext, dt):
        """Compute open-loop sensitivity propagation"""
        p_sens_state, p_sens_inputs, p_sens_params = self.partial_sens_all(
            state, inputs, self.p_nom, ext, dt
        )
        return (p_sens_state + p_sens_inputs @ input_gains) @ state_sensitivity + p_sens_params

    def sensitivity_step_closed_loop(self, state, reference, params, state_sensitivity, ext, dt):
        """Compute closed-loop sensitivity propagation"""
        # In this case, inputs are some exogenous variables that enter the controller
        if self.controller is None:
            raise ValueError("Controller must be defined for closed-loop sensitivity computation")
            
        if self.controller_sens is None:
            raise ValueError("Controller sensitivity function must be set up before closed-loop computation")
            
        inputs = self.controller(state, reference, params, ext)

        state_sens_all = self.partial_sens_all(state, inputs, params, ext, dt)

        p_sens_state = state_sens_all[0]
        p_sens_inputs = state_sens_all[1]
        p_sens_params = state_sens_all[2]

        # Use pre-computed controller sensitivity function
        input_sens_state = self.controller_sens(state, reference, params, ext)

        return (
            p_sens_state + p_sens_inputs @ input_sens_state
        ) @ state_sensitivity + p_sens_params, inputs




class Model(ModelParametric):

    
    def __init__(self, config: ModelConfig, model_dynamics, controller=None):
        """
        CasADi model with ModelConfig interface.
        
        Args:
            config: ModelConfig object containing all model parameters
            model_dynamics: Model dynamics function with signature (state, inputs, params, ext)
            controller: Optional controller function with signature (state, reference, params, ext)
        """
        
        # Extract nominal parameters and determine np
        p_nom = config.p_nom
        if p_nom is not None:
            try:
                np_val = len(p_nom)
            except (TypeError, AttributeError):
                # Handle CasADi objects or other types that don't have len()
                np_val = p_nom.numel() if hasattr(p_nom, 'numel') else 0
        else:
            np_val = 0
        
        # Create ModelParametric with config parameters
        super().__init__(
            model_dynamics, 
            config.nq, 
            config.nv, 
            config.nu,
            config.ny, 
            np_val, 
            config.next,
            input_bounds=(-float('inf'), float('inf')),
            integrator_params=config.integrator_params,
            controller=controller,
        )
        
        # Store additional config parameters
        self.p_nom = p_nom
        self.controller = controller
        self.ny = config.ny
        
    
    def integrate(self, state, inputs, ext=0):
        """Integrate the system one step forward"""
        return self.integrate_parametric(state, inputs, self.p_nom, ext, self.dt)

    def sensitivity_step(self, state, inputs, state_sensitivity, input_gains, ext=0):
        """Compute open-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step(state, inputs, state_sensitivity, input_gains, ext, self.dt)

    def sensitivity_step_closed_loop(self, state, reference, state_sensitivity, ext=0):
        """Compute closed-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step_closed_loop(state, reference, self.p_nom, state_sensitivity, ext, self.dt)



def ellipsoid_radius_casadi(sensitivity, scaling_matrix, direction=None):
    """
    Compute the ellipsoid radius using CasADi operations

    Parameters
    ----------
    sensitivity : casadi.MX or casadi.SX
        Sensitivity matrix (Pi)
    scaling_matrix : casadi.MX or casadi.SX
        Scaling/weighting matrix (W)
    direction : casadi.MX or casadi.SX, optional
        Direction vector (n). If None, returns radii along Euclidean basis

    Returns
    -------
    casadi.MX or casadi.SX
        Ellipsoid radius(i) in the specified direction(s)
    """
    if cs is None:
        raise ImportError("CasADi is required for this function")
    
    if direction is None:
        return cs.sqrt(cs.diag(sensitivity @ scaling_matrix @ sensitivity.T))
    else:
        return cs.sqrt(direction.T @ sensitivity @ scaling_matrix @ sensitivity.T @ direction)


def ellipsoid_radii_casadi(sensitivity, scaling_matrix):
    """
    Compute ellipsoid radii using CasADi operations

    Parameters
    ----------
    sensitivity : casadi.MX or casadi.SX
        Sensitivity matrix (Pi)
    scaling_matrix : casadi.MX or casadi.SX
        Scaling/weighting matrix (W)

    Returns
    -------
    casadi.MX or casadi.SX
        Array of ellipsoid radii along each coordinate axis
    """
    return ellipsoid_radius_casadi(sensitivity, scaling_matrix, direction=None)
