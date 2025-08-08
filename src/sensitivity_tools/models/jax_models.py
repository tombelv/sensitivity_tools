from typing import Optional, Dict
from functools import partial

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from ..settings import INTEGRATOR_TYPES


class BaseModel(ABC):
    def __init__(self, nq: int, nv: int, nu: int, np: int, next: int, input_bounds):
        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.np = np
        self.next = next  # number of external variables
        if jnp.array_equal(input_bounds, (-jnp.inf, jnp.inf)):
            self.input_min = input_bounds[0] * jnp.ones(self.nu, dtype=jnp.float32)
            self.input_max = input_bounds[1] * jnp.ones(self.nu, dtype=jnp.float32)
        else:
            self.input_min = input_bounds[0]
            self.input_max = input_bounds[1]

        self.p_nom = []


class ModelParametric(BaseModel):
    def __init__(
        self,
        model_dynamics_parametric,
        nq: int,
        nv: int,
        nu: int,
        np: int,
        next: int,
        input_bounds,
        integrator_params,
        controller=None
    ):

        super().__init__(nq, nv, nu, np, next, input_bounds)

        self.dynamics_parametric = model_dynamics_parametric
        self.controller = controller  # Initialize controller as None
        if controller is not None:
            self.controller_sens = jax.jacfwd(controller, argnums=0)

        integrator_type = integrator_params["method"]
        self.dt = integrator_params["step_size"]

        if integrator_type == INTEGRATOR_TYPES[0]:
            self.integrate_parametric = self.integrate_si_euler
        elif integrator_type == INTEGRATOR_TYPES[1]:
            self.integrate_parametric = self.integrate_euler
        elif integrator_type == INTEGRATOR_TYPES[2]:
            self.integrate_parametric = self.integrate_rk4
        elif integrator_type == INTEGRATOR_TYPES[3]:
            self.integrate_parametric = model_dynamics_parametric
        else:
            raise ValueError(
                """
            Integrator type not supported.
            Available types: si_euler, euler, rk4, custom_discrete
            """
            )

        self.partial_sens_all = jax.jacfwd(self.integrate_parametric, argnums=(0, 1, 2))

    @partial(jax.jit, static_argnums=(0,))
    def integrate_rk4(self, state, inputs, params, ext, dt: float):
        """
        One-step integration of the dynamics using Rk4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params, ext)
        k2 = self.dynamics_parametric(state + k1 * dt / 2.0, inputs, params, ext)
        k3 = self.dynamics_parametric(state + k2 * dt / 2.0, inputs, params, ext)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params, ext)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @partial(jax.jit, static_argnums=(0,))
    def integrate_euler(self, state, inputs, params, ext, dt: float):
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics_parametric(state, inputs, params, ext)

    @partial(jax.jit, static_argnums=(0,))
    def integrate_si_euler(self, state, inputs, params, ext, dt: float):
        """
        Semi-implicit Euler integration.
        As of now this is probably implemented inefficiently because the whole dynamics is evaluated two times.
        """
        v_kp1 = state[self.nq :] + dt * self.dynamics_parametric(state, inputs, params, ext)[self.nq :]
        return jnp.concatenate(
            [
                state[: self.nq]
                + dt
                * self.dynamics_parametric(
                    jnp.concatenate([state[: self.nq], v_kp1]), inputs, params, ext
                )[: self.nq],
                v_kp1,
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def sensitivity_step(self, state, inputs, params, state_sensitivity, input_gains, ext, dt):
        """Compute open-loop sensitivity propagation using JIT compiled functions"""
        p_sens_all = self.partial_sens_all(state, inputs, params, ext, dt)
        p_sens_state = p_sens_all[0]
        p_sens_inputs = p_sens_all[1]
        p_sens_params = p_sens_all[2]

        return (p_sens_state + p_sens_inputs @ input_gains) @ state_sensitivity + p_sens_params

    @partial(jax.jit, static_argnums=(0,))
    def sensitivity_step_closed_loop(self, state, reference, params, state_sensitivity, ext, dt):
        """Compute closed-loop sensitivity propagation using JIT compiled functions"""
        if self.controller is None:
            raise ValueError("Controller must be defined for closed-loop sensitivity computation")
            
        inputs = self.controller(state, reference, params, ext)

        p_sens_all = self.partial_sens_all(state, inputs, params, ext, dt)
        p_sens_state = p_sens_all[0]
        p_sens_inputs = p_sens_all[1]
        p_sens_params = p_sens_all[2]

        input_sens_state = self.controller_sens(state, reference, params, ext)

        return (
            p_sens_state + p_sens_inputs @ input_sens_state
        ) @ state_sensitivity + p_sens_params, inputs


class Model(ModelParametric):
    def __init__(self, config, model_dynamics, controller=None):
        """
        JAX model with ModelConfig interface.
        
        Args:
            config: ModelConfig object containing all model parameters
            model_dynamics: Model dynamics function with signature (state, inputs, params, ext)
            controller: Optional controller function with signature (state, reference, params, ext)
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
            len(p_nom), 
            config.next,
            input_bounds=(-jnp.inf, jnp.inf),
            integrator_params=config.integrator_params,
            controller=controller,
        )

        self.p_nom = p_nom
        self.controller = controller
        self.ny = config.ny

    @partial(jax.jit, static_argnums=(0,))
    def integrate(self, state, inputs, ext=None):
        """Integrate the system one step forward"""
        return self.integrate_parametric(state, inputs, self.p_nom, ext, self.dt)

    def sensitivity_step(self, state, inputs, state_sensitivity, input_gains, ext=None):
        """Compute open-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step(state, inputs, self.p_nom, state_sensitivity, input_gains, ext, self.dt)

    def sensitivity_step_closed_loop(self, state, reference, state_sensitivity, ext=None):
        """Compute closed-loop sensitivity propagation with fixed parameters"""
        return super().sensitivity_step_closed_loop(state, reference, self.p_nom, state_sensitivity, ext, self.dt)


def ellipsoid_radius_jax(sensitivity, scaling_matrix, direction=None):
    """
    Compute the ellipsoid radius using JAX arrays

    Parameters
    ----------
    sensitivity : jax.Array
        Sensitivity matrix (Pi)
    scaling_matrix : jax.Array
        Scaling/weighting matrix (W)
    direction : jax.Array, optional
        Direction vector (n). If None, returns radii along Euclidean basis

    Returns
    -------
    jax.Array
        Ellipsoid radius(i) in the specified direction(s)
    """
    if direction is None:
        return jnp.sqrt(jnp.diag(sensitivity @ scaling_matrix @ sensitivity.T))
    else:
        return jnp.sqrt(direction.T @ sensitivity @ scaling_matrix @ sensitivity.T @ direction)


def ellipsoid_radii_jax(sensitivity, scaling_matrix):
    """
    Compute ellipsoid radii along all coordinate axes using JAX

    Parameters
    ----------
    sensitivity : jax.Array
        Sensitivity matrix (Pi)
    scaling_matrix : jax.Array
        Scaling/weighting matrix (W)

    Returns
    -------
    jax.Array
        Array of ellipsoid radii along each coordinate axis
    """
    return ellipsoid_radius_jax(sensitivity, scaling_matrix, direction=None)
