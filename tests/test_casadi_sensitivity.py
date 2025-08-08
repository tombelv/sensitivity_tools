#!/usr/bin/env python3
"""
Test CasADi sensitivity computation to understand the expected behavior
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from sensitivity_tools.settings import create_model_config
from sensitivity_tools.models.casadi_models import Model as CasADiModel

def simple_dynamics(state, inputs, params, dt, ext=0):
    """Simple linear dynamics: x_dot = -params[0]*x + params[1]*u"""
    return state + dt * (-params[0] * state + params[1] * inputs)

def simple_controller(state, reference, params, ext=0):
    """Simple proportional controller"""
    return -params[2] * state + reference

def test_casadi_sensitivity():
    """Test CasADi sensitivity computation"""
    print("Testing CasADi sensitivity computation...")
    
    # Create model config
    config = create_model_config(nq=2, nv=2, nu=2, p_nom=[0.1, 0.2, 0.3])
    
    # Create CasADi model
    casadi_model = CasADiModel(config, simple_dynamics, simple_controller)
    
    # Test data
    state = np.array([1.0, 2.0])
    inputs = np.array([0.5, -0.5])
    params = np.array([0.1, 0.2, 0.3])
    dt = 0.1
    
    print(f"State shape: {state.shape}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Params shape: {params.shape}")
    
    # Test integration
    new_state = casadi_model.integrate_parametric(state, inputs, params, dt)
    print(f"New state shape: {new_state.shape}")
    print(f"New state: {new_state}")
    
    # Test partial sensitivities
    print("\nTesting partial sensitivities...")
    p_sens_state, p_sens_inputs, p_sens_params = casadi_model.partial_sens_all(state, inputs, params, 0, dt)
    
    print(f"p_sens_state shape: {p_sens_state.shape}")
    print(f"p_sens_inputs shape: {p_sens_inputs.shape}")
    print(f"p_sens_params shape: {p_sens_params.shape}")
    
    print(f"p_sens_state:\n{p_sens_state}")
    print(f"p_sens_inputs:\n{p_sens_inputs}")
    print(f"p_sens_params:\n{p_sens_params}")
    
    # Test sensitivity step with different sensitivity matrices
    print("\nTesting sensitivity step...")
    
    # Test 1: Identity state and input sensitivity
    state_sensitivity = np.eye(2)
    input_sensitivity = np.eye(2)
    
    print(f"State sensitivity shape: {state_sensitivity.shape}")
    print(f"Input sensitivity shape: {input_sensitivity.shape}")
    
    try:
        sens_result = casadi_model.sensitivity_step(state, inputs, params, state_sensitivity, input_sensitivity, dt)
        print(f"Sensitivity result shape: {sens_result.shape}")
        print(f"Sensitivity result:\n{sens_result}")
    except Exception as e:
        print(f"Error in sensitivity step: {e}")
    
    # Test 2: Try different sensitivity matrix shapes
    print("\nTesting with parameter-sized sensitivity matrix...")
    try:
        # Try with 3x3 sensitivity matrices (matching parameter dimension)
        state_sensitivity_3x3 = np.eye(3)
        input_sensitivity_3x3 = np.eye(3)
        
        # This should probably fail since state is 2D
        sens_result_3x3 = casadi_model.sensitivity_step(state, inputs, params, state_sensitivity_3x3, input_sensitivity_3x3, dt)
        print(f"3x3 sensitivity result shape: {sens_result_3x3.shape}")
        print(f"3x3 sensitivity result:\n{sens_result_3x3}")
    except Exception as e:
        print(f"Error with 3x3 sensitivity: {e}")

if __name__ == "__main__":
    test_casadi_sensitivity()
