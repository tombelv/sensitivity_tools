#!/usr/bin/env python3
"""Test CasADi Model with ext=None parameter handling"""

from sensitivity_tools.models.casadi_models import Model
from sensitivity_tools.settings import create_model_config
import casadi as cs
import numpy as np

def simple_dynamics(state, inputs, params, ext=0):
    A = cs.DM([[0.9, 0.1], [0.0, 0.8]])
    B = cs.DM([[1.0], [0.5]])
    return A @ state + B @ inputs[0]

def controller(state, ref, K_feedback, ext):
    return -K_feedback @ (state - ref)

config = create_model_config(
    nq=2, nv=0, nu=1,
    p_nom=np.ones(1),
    dt=0.1,
    integrator='custom_discrete',
    ny=2  # Reference dimension should match state dimension
)

model = Model(config, simple_dynamics, controller)

# Test with ext=None (should convert to ext=0)
state = np.array([1.0, 2.0])
inputs = np.array([0.5])
state_sens = np.eye(2)
input_gains = np.array([[0.1, 0.2]])  # nu x nx

result = model.sensitivity_step(state, inputs, state_sens, input_gains, ext=None)
print('CasADi Model with ext=None works correctly!')
print(f'Result shape: {result.shape}')
print('Test passed! ✅')
