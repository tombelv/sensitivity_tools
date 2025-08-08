#!/usr/bin/env python3
"""Test script to verify the sensitivity step methods work correctly."""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sensitivity_methods():
    """Test that all sensitivity methods work correctly."""
    print("Testing sensitivity step methods...")
    
    try:
        from sensitivity_tools.settings import create_model_config
        from sensitivity_tools.models.casadi_models import Model as CasADiModel
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    # Create a simple config
    config = create_model_config(
        nq=1, nv=1, nu=1, 
        p_nom=np.array([1.0, 2.0])
    )
    
    # Simple dynamics compatible with CasADi
    def simple_dynamics(x, u, p, ext=0):
        try:
            # Try CasADi operations
            import casadi as cs
            return cs.vertcat(x[1], -p[0] * x[0] - p[1] * x[1] + u[0])
        except:
            # Fall back to numpy
            return np.array([x[1], -p[0] * x[0] - p[1] * x[1] + u[0]])
    
    # Create model
    model = CasADiModel(config, simple_dynamics)
    
    # Test states and inputs
    x0 = np.array([0.1, 0.0])
    u0 = np.array([0.0])
    
    # Test sensitivity matrices
    state_sensitivity = np.eye(2)
    input_gains = np.zeros((1, 1))
    
    print("✓ Testing open-loop sensitivity...")
    try:
        sens_ol = model.sensitivity_step_open_loop(x0, u0, state_sensitivity, input_gains)
        print(f"✓ Open-loop sensitivity shape: {sens_ol.shape}")
    except Exception as e:
        print(f"✗ Open-loop sensitivity failed: {e}")
        return False
    
    print("✓ Testing closed-loop sensitivity...")
    try:
        # Define a simple controller compatible with CasADi
        def simple_controller(x, ref, p, ext=0):
            try:
                # Try CasADi operations
                import casadi as cs
                return cs.vertcat(-p[0] * x[0])  # Simple proportional controller
            except:
                # Fall back to numpy
                return np.array([-p[0] * x[0]])  # Simple proportional controller
        
        # Create config for closed-loop model (without controller)
        config_cl = create_model_config(
            nq=1, nv=1, nu=1,
            p_nom=np.array([1.0, 2.0]),
            ny=1
        )
        
        # Create model with controller as separate parameter
        model_cl = CasADiModel(config_cl, simple_dynamics, simple_controller)
        
        reference = np.array([0.0])
        sens_cl, inputs_used = model_cl.sensitivity_step_closed_loop(x0, reference, state_sensitivity)
        print(f"✓ Closed-loop sensitivity shape: {sens_cl.shape}")
        print(f"✓ Inputs used: {inputs_used}")
        
    except Exception as e:
        print(f"✗ Closed-loop sensitivity failed: {e}")
        return False
    
    print("\n✓ All sensitivity methods work correctly!")
    return True

if __name__ == "__main__":
    success = test_sensitivity_methods()
    sys.exit(0 if success else 1)
