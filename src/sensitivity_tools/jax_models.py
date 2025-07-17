from typing import Optional, Dict

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from .settings import MODEL_PARAMETRIC_INTEGRATOR_TYPES


class BaseModel(ABC):
    def __init__(self, nq: int, nv: int, nu: int, np=0, input_bounds=(-jnp.inf, jnp.inf)):
        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.np = np
        if jnp.array_equal(input_bounds, (-jnp.inf, jnp.inf)):
            self.input_min = input_bounds[0]*jnp.ones(self.nu, dtype=jnp.float32)
            self.input_max = input_bounds[1]*jnp.ones(self.nu, dtype=jnp.float32)
        else:
            self.input_min = input_bounds[0]
            self.input_max = input_bounds[1]

        self.nominal_parameters = []

    def get_nq(self):
        return self.nq

    def integrate(self, state, inputs, dt):
        pass

    def integrate_sim(self, state, inputs, dt):
        return self.integrate(state, inputs, dt)

    def integrate_rollout(self, state, inputs, dt):
        pass

    def integrate_rollout_single(self, state, inputs, dt):
        pass

    def sensitivity_step(self, state, inputs, params, state_sensitivity, input_sensitivity, dt):
        pass



class ModelParametric(BaseModel):
    def __init__(self, model_dynamics_parametric,
                 nq: int,
                 nv: int,
                 nu: int,
                 np=0,
                 input_bounds=(-jnp.inf, jnp.inf),
                 integrator_type="si_euler",
                 integrator_params: Optional[Dict] = None):

        super().__init__(nq, nv, nu, np, input_bounds)

        self.dynamics_parametric = model_dynamics_parametric

        if integrator_type == MODEL_PARAMETRIC_INTEGRATOR_TYPES[0]:
            self.integrate_parametric = self.integrate_si_euler
        elif integrator_type == MODEL_PARAMETRIC_INTEGRATOR_TYPES[1]:
            self.integrate_parametric = self.integrate_euler
        elif integrator_type == MODEL_PARAMETRIC_INTEGRATOR_TYPES[2]:
            self.integrate_parametric = self.integrate_rk4
        elif integrator_type == MODEL_PARAMETRIC_INTEGRATOR_TYPES[3]:
            self.integrate_parametric = model_dynamics_parametric
        else:
            raise ValueError("""
            Integrator type not supported.
            Available types: si_euler, euler, rk4, custom_discrete
            """)

        self.partial_sens_all = jax.jacfwd(self.integrate_parametric, argnums=(0, 1, 2))

    def integrate_rk4(self, state, inputs, params, dt: float):
        """
        One-step integration of the dynamics using Rk4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params)
        k2 = self.dynamics_parametric(state + k1*dt/2., inputs, params)
        k3 = self.dynamics_parametric(state + k2 * dt / 2., inputs, params)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params)
        return state + (dt/6.) * (k1 + 2. * k2 + 2. * k3 + k4)

    def integrate_euler(self, state, inputs, params, dt: float):
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics_parametric(state, inputs, params)

    def integrate_si_euler(self, state, inputs, params, dt: float):
        """
        Semi-implicit Euler integration.
        As of now this is probably implemented inefficiently because the whole dynamics is evaluated two times.
        """
        v_kp1 = state[self.nq:] + dt * self.dynamics_parametric(state, inputs, params)[self.nq:]
        return jnp.concatenate([
                    state[:self.nq] + dt * self.dynamics_parametric(jnp.concatenate([state[:self.nq], v_kp1]), inputs, params)[:self.nq],
                    v_kp1])

    def sensitivity_step(self, state, inputs, params, state_sensitivity, input_sensitivity, dt):

        p_sens_all = self.partial_sens_all(state, inputs, params, dt)
        p_sens_state = p_sens_all[0]
        p_sens_inputs = p_sens_all[1]
        p_sens_params = p_sens_all[2]

        return p_sens_state @ state_sensitivity + p_sens_inputs @ input_sensitivity + p_sens_params


class Model(ModelParametric):
    def __init__(self, model_dynamics,
                 nq: int,
                 nv: int,
                 nu: int,
                 nominal_parameters=[],
                 input_bounds=(-jnp.inf, jnp.inf),
                 integrator_type="si_euler"):
        """
        :param model_dynamics: Should have signature (state, inputs, params)
        :param nq: Number of configuration variables
        :param nv: Number of pseudo velocities
        :param nu: Number of inputs
        :param nominal_parameters: Note that nominal parameters are baked in from the start due to jitting
        :param input_bounds: Lower and upper bound of the inputs
        :param integrator_type: Available integrators are [si_euler, euler, rk4, custom_discrete]
        """

        super().__init__(model_dynamics, nq, nv, nu, len(nominal_parameters), input_bounds, integrator_type)

        self.nominal_parameters = nominal_parameters

        self.integrate_rollout_single = self.integrate

    def integrate(self, state, inputs, dt):
        return self.integrate_parametric(state, inputs, self.nominal_parameters, dt)


