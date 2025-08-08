#!/usr/bin/env python3
"""
Simple pendulum example demonstrating sensitivity computation with different backends
"""

import numpy as np


def main():
    print("Sensitivity Tools - Pendulum Example")
    print("=" * 50)
    
    # Test basic functionality
    try:
        from sensitivity_tools import INTEGRATOR_TYPES, ellipsoid_radius
        print(f"✓ Package imported successfully")
        print(f"✓ Available integrators: {INTEGRATOR_TYPES}")
        
        # Test ellipsoid computation
        sensitivity = np.array([[1.0, 0.1], [0.1, 1.0]])
        scaling_matrix = np.eye(2)
        radii = ellipsoid_radius(sensitivity, scaling_matrix)
        print(f"✓ Ellipsoid radii: {radii}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    
    # Test JAX backend
    print("\n" + "JAX Backend".center(30, "-"))
    try:
        from sensitivity_tools.models.jax_models import Model
        from sensitivity_tools.settings import create_model_config
        import jax.numpy as jnp
        
        def pendulum_dynamics(state, inputs, params, ext=0):
            q, qdot = state[0], state[1]
            u = inputs[0]
            m, l, g, b = params
            qddot = (u - m * g * l * jnp.sin(q) - b * qdot) / (m * l**2)
            return jnp.array([qdot, qddot])
        
        config = create_model_config(
            nq=1, nv=1, nu=1,
            p_nom=[1.0, 1.0, 9.81, 0.1],
            dt=0.01
        )
        
        model = Model(config, pendulum_dynamics)
        
        state = jnp.array([0.1, 0.0])
        inputs = jnp.array([0.0])
        
        next_state = model.integrate(state, inputs)
        print(f"✓ JAX integration: {state} -> {next_state}")
        
    except ImportError:
        print("✗ JAX backend not available")
    except Exception as e:
        print(f"✗ JAX backend error: {e}")
    
    # Test CasADi backend
    print("\n" + "CasADi Backend".center(30, "-"))
    try:
        from sensitivity_tools.models.casadi_models import Model
        from sensitivity_tools.settings import create_model_config
        import casadi as cs
        
        def pendulum_dynamics(state, inputs, params, ext=0):
            q, qdot = state[0], state[1]
            u = inputs[0]
            m, l, g, b = params[0], params[1], params[2], params[3]
            qddot = (u - m * g * l * cs.sin(q) - b * qdot) / (m * l**2)
            return cs.vertcat(qdot, qddot)
        
        config = create_model_config(
            nq=1, nv=1, nu=1,
            p_nom=np.array([1.0, 1.0, 9.81, 0.1]),
            dt=0.01
        )
        
        model = Model(config, pendulum_dynamics)
        
        state = np.array([0.1, 0.0])
        inputs = np.array([0.0])
        
        next_state = model.integrate(state, inputs)
        print(f"✓ CasADi integration: {state} -> {next_state}")
        
    except ImportError:
        print("✗ CasADi backend not available")
    except Exception as e:
        print(f"✗ CasADi backend error: {e}")
    
    # Test PyTorch backend
    print("\n" + "PyTorch Backend".center(30, "-"))
    try:
        from sensitivity_tools.models.torch_models import Model
        from sensitivity_tools.settings import create_model_config
        import torch
        
        def pendulum_dynamics(state, inputs, params, ext=0):
            q, qdot = state[0], state[1]
            u = inputs[0]
            m, l, g, b = params[0], params[1], params[2], params[3]
            qddot = (u - m * g * l * torch.sin(q) - b * qdot) / (m * l**2)
            return torch.stack([qdot, qddot])
        
        config = create_model_config(
            nq=1, nv=1, nu=1,
            p_nom=[1.0, 1.0, 9.81, 0.1],
            dt=0.01
        )
        
        model = Model(config, pendulum_dynamics)
        
        state = torch.tensor([0.1, 0.0])
        inputs = torch.tensor([0.0])
        
        next_state = model.integrate(state, inputs)
        print(f"✓ PyTorch integration: {state} -> {next_state}")
        
    except ImportError:
        print("✗ PyTorch backend not available")
    except Exception as e:
        print(f"✗ PyTorch backend error: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
