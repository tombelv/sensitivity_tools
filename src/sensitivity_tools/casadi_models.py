from abc import ABC
import numpy as np
from .utils import ellipsoid_radius, integrate_rk4_function

try:
    import casadi as cs
except ImportError:
    cs = None

class Model:
    def __init__(self, model_dynamics_parametric,
                 nq: int,
                 nv: int,
                 nu: int,
                 nominal_parameters,
                 controller = None,
                 ny = 0,
                 dt=0.01,
                 integrator_type="si_euler",
                 next = 0):

        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.next = next # number of robots (external variables)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        self.ny = ny  # number of outputs (reference for the controller)
        self.np = len(nominal_parameters)
        # self.np = nominal_parameters.numel() 
        self.dt = dt
        self.nominal_parameters = nominal_parameters

        self.dynamics_parametric = model_dynamics_parametric

        if integrator_type == "si_euler":
            self.integrate_parametric = self.integrate_si_euler
        elif integrator_type == "euler":
            self.integrate_parametric = self.integrate_euler
        elif integrator_type == "rk4":
            self.integrate_parametric = self.integrate_rk4
        elif integrator_type == "custom_discrete":
            self.integrate_parametric = model_dynamics_parametric
        else:
            raise ValueError("""
            Integrator type not supported.
            Available types: si_euler, euler, rk4, custom_discrete
            """)

        self.control_function = controller
        self.s_sens_input = None

        x = cs.MX.sym('x', self.nx, 1)
        u = cs.MX.sym('u', self.nu, 1)
        ext = cs.MX.sym('ext', self.next, 1)
        p = cs.MX.sym('p', self.np, 1)

        x_kp1 = self.integrate_parametric(x, u, p, ext)

        dfdx = cs.jacobian(x_kp1, x)
        dfdu = cs.jacobian(x_kp1, u)
        dfdp = cs.jacobian(x_kp1, p)


        self.partial_sens_all = cs.Function('partial_sens_all',
                                 [x, u, p, ext],  # Input arguments
                                 [dfdx, dfdu, dfdp],  # Output
                                 ['x', 'u', 'p', 'ext'],  # Input names
                                 ['dfdx', 'dfdu', 'dfdp'], {"cse": True}) # Output names
        
        ref = cs.MX.sym('ref', self.ny, 1)
        
        if controller is not None:
            ctrl_action = controller(x, ref, p, ext)
            dudx = cs.jacobian(ctrl_action, x)
        
            self.controller_sens = cs.Function('controller_sens', [x, ref, p, ext], [dudx], ['x', 'ref', 'p', 'ext'], ['dudx'], {"cse": True})
        else:
            self.controller_sens = 0



    def integrate_rk4(self, state, inputs, params, ext=0):
        """
        One-step integration of the dynamics using Rk4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params, ext)
        k2 = self.dynamics_parametric(state + k1 * self.dt / 2., inputs, params, ext)
        k3 = self.dynamics_parametric(state + k2 * self.dt / 2., inputs, params, ext)
        k4 = self.dynamics_parametric(state + k3 * self.dt, inputs, params, ext)
        return state + (self.dt / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)

    def integrate_euler(self, state, inputs, params, ext=0):
        """
        One-step integration of the dynamics using Euler method
        """
        return state + self.dt * self.dynamics_parametric(state, inputs, params, ext)

    def integrate_si_euler(self, state, inputs, params, ext=0): 
        """
        Semi-implicit Euler integration.
        As of now this is probably implemented inefficiently because the whole dynamics is evaluated two times.
        """
        v_kp1 = state[self.nq:] + self.dt * self.dynamics_parametric(state, inputs, params, ext)[self.nq:]
        return cs.vertcat(state[:self.nq] + self.dt * self.dynamics_parametric(
                              cs.vertcat(state[:self.nq], v_kp1), inputs, params, ext)[:self.nq],
                          v_kp1)

    def sensitivity_step(self, state, inputs, state_sensitivity, input_gains=0, ext=0):
        p_sens_state, p_sens_inputs, p_sens_params = self.partial_sens_all(state, inputs, self.nominal_parameters, ext)
        s_sens_input = input_gains
        return (p_sens_state + p_sens_inputs @ s_sens_input) @ state_sensitivity + p_sens_params
    
    def sensitivity_step_cl(self, state, reference, params, state_sensitivity, ext=0):
        # In this case, inputs are some exogenous variables the enter the controller
        
        inputs = self.control_function(state, reference, params, ext)

        state_sens_all = self.partial_sens_all(state, inputs, params, ext)
        
        p_sens_state = state_sens_all[0]
        p_sens_inputs = state_sens_all[1]
        p_sens_params = state_sens_all[2]
        
        input_sens_state = self.controller_sens(state, reference, params, ext)
        
        return (p_sens_state +  p_sens_inputs @ input_sens_state) @ state_sensitivity + p_sens_params, inputs


    def integrate(self, state, inputs, ext=0):
        return self.integrate_parametric(state, inputs, self.nominal_parameters, ext)


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
    if cs is None:
        raise ImportError("CasADi is required for this function")
    return cs.sqrt(cs.diag(sensitivity @ scaling_matrix @ sensitivity.T))
