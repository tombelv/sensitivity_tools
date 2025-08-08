#!/usr/bin/env python3
"""Test the updated CasADi controller sensitivity setup."""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_casadi_controller_sensitivity():
    """Test that the CasADi controller sensitivity setup works correctly."""
    print("Testing CasADi controller sensitivity setup...")
    
    try:
        from sensitivity_tools.settings import create_model_config
        from sensitivity_tools.models.casadi_models import Model as CasADiModel
        import casadi as cs
        print("✓ CasADi models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    # Simple dynamics
    def simple_dynamics(x, u, p, ext=0):
        return cs.vertcat(x[1], -p[0] * x[0] - p[1] * x[1] + u[0])
    
    # Simple controller
    def simple_controller(x, ref, p, ext=0):
        return cs.vertcat(-p[0] * x[0])
    
    # Create config
    config = create_model_config(
        nq=1, nv=1, nu=1, 
        p_nom=np.array([1.0, 2.0]),
        ny=1
    )
    
    print("✓ Testing model with controller...")
    try:
        model = CasADiModel(config, simple_dynamics, simple_controller)
        print(f"✓ Model created: controller={model.controller is not None}")
        print(f"✓ Controller sensitivity function: {model.controller_sens is not None}")
        
        # Test closed-loop sensitivity
        x0 = np.array([0.1, 0.0])
        reference = np.array([0.0])
        state_sensitivity = np.eye(2)
        
        sens_cl, inputs_used = model.sensitivity_step_closed_loop(x0, reference, state_sensitivity)
        print(f"✓ Closed-loop sensitivity shape: {sens_cl.shape}")
        print(f"✓ Inputs used: {inputs_used}")
        
    except Exception as e:
        print(f"✗ Model with controller failed: {e}")
        return False
    
    print("✓ Testing model without controller...")
    try:
        model_no_ctrl = CasADiModel(config, simple_dynamics)
        print(f"✓ Model created: controller={model_no_ctrl.controller is None}")
        print(f"✓ Controller sensitivity function: {model_no_ctrl.controller_sens is None}")
        
        # This should fail
        try:
            sens_cl, inputs_used = model_no_ctrl.sensitivity_step_closed_loop(x0, reference, state_sensitivity)
            print("✗ Should have failed without controller")
            return False
        except ValueError as e:
            print(f"✓ Correctly failed without controller: {e}")
        
    except Exception as e:
        print(f"✗ Model without controller failed: {e}")
        return False
    
    print("\n✓ All CasADi controller sensitivity tests passed!")
    return True

if __name__ == "__main__":
    success = test_casadi_controller_sensitivity()
    sys.exit(0 if success else 1)
