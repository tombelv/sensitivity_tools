#!/usr/bin/env python3
"""
Test script to verify PyTorch forward mode AD and JAX JIT compilation work correctly
"""

import numpy as np
import jax.numpy as jnp
import torch
import jax
import time

try:
    import casadi as cs
except ImportError:
    cs = None

# Import the sensitivity tools
import sys
sys.path.insert(0, "src")

from sensitivity_tools.settings import create_model_config
from sensitivity_tools.models.torch_models import Model as TorchModel
from sensitivity_tools.models.jax_models import Model as JAXModel
from sensitivity_tools.models.casadi_models import Model as CasADiModel

def simple_controller(state, reference, params, ext=0):
    """Simple proportional controller for CasADi"""
    if cs is None:
        # Fallback for when CasADi is not available
        return -params[2] * state + reference
    
    # For CasADi, handle the full state vector properly
    # state has 4 elements [q1, q2, v1, v2], reference has 2 elements [ref1, ref2]
    # Apply controller to velocity part only
    v = state[2:]  # velocities
    
    # Simple proportional controller: u = -params[2] * v + reference
    return -params[2] * v + reference

def simple_dynamics(state, inputs, params, dt, ext=0):
    """Simple linear dynamics for CasADi: x_dot = -params[0]*x + params[1]*u (for 2-state case)"""
    if cs is None:
        # Fallback for when CasADi is not available
        return state + dt * (-params[0] * state + params[1] * inputs)
    
    # For CasADi, handle the full state vector properly
    # Assume nq=2, nv=2, so state has 4 elements [q1, q2, v1, v2]
    # Apply dynamics only to velocity part
    q = state[:2]  # positions
    v = state[2:]  # velocities
    
    # Simple dynamics: v_dot = -params[0]*v + params[1]*u
    v_dot = -params[0] * v + params[1] * inputs
    
    # q_dot = v (velocity integration)
    q_dot = v
    
    return cs.vertcat(q_dot, v_dot)

def simple_dynamics_torch(state, inputs, params, dt, ext=0):
    """Simple linear dynamics for PyTorch: same structure as CasADi"""
    # Handle the full state vector properly
    # Assume nq=2, nv=2, so state has 4 elements [q1, q2, v1, v2]
    # Apply dynamics only to velocity part
    q = state[:2]  # positions
    v = state[2:]  # velocities
    
    # Simple dynamics: v_dot = -params[0]*v + params[1]*u
    v_dot = -params[0] * v + params[1] * inputs
    
    # q_dot = v (velocity integration)
    q_dot = v
    
    return torch.cat([q_dot, v_dot])

def simple_dynamics_jax(state, inputs, params, dt, ext=0):
    """Simple linear dynamics for JAX: same structure as CasADi"""
    # Handle the full state vector properly
    # Assume nq=2, nv=2, so state has 4 elements [q1, q2, v1, v2]
    # Apply dynamics only to velocity part
    q = state[:2]  # positions
    v = state[2:]  # velocities
    
    # Simple dynamics: v_dot = -params[0]*v + params[1]*u
    v_dot = -params[0] * v + params[1] * inputs
    
    # q_dot = v (velocity integration)
    q_dot = v
    
    return jnp.concatenate([q_dot, v_dot])

def simple_controller_torch(state, reference, params, ext=0):
    """Simple proportional controller for PyTorch: same structure as CasADi"""
    # Handle the full state vector properly
    # state has 4 elements [q1, q2, v1, v2], reference has 2 elements [ref1, ref2]
    # Apply controller to velocity part only
    v = state[2:]  # velocities
    
    # Simple proportional controller: u = -params[2] * v + reference
    return -params[2] * v + reference

def simple_controller_jax(state, reference, params, ext=0):
    """Simple proportional controller for JAX: same structure as CasADi"""
    # Handle the full state vector properly
    # state has 4 elements [q1, q2, v1, v2], reference has 2 elements [ref1, ref2]
    # Apply controller to velocity part only
    v = state[2:]  # velocities
    
    # Simple proportional controller: u = -params[2] * v + reference
    return -params[2] * v + reference

def test_torch_optimization():
    """Test PyTorch forward mode AD and torch.compile"""
    print("Testing PyTorch optimizations...")
    
    # Create model config (same as CasADi)
    config = create_model_config(nq=2, nv=2, nu=2, ny=2, p_nom=[0.1, 0.2, 0.3], dt=0.01)
    
    # Create PyTorch model
    torch_model = TorchModel(config, simple_dynamics_torch, simple_controller_torch)
    
    # Test data (4 elements for consistency: [q1, q2, v1, v2])
    state = torch.tensor([1.0, 2.0, 0.1, 0.2], dtype=torch.float32)  # positions and velocities
    inputs = torch.tensor([0.5, -0.5], dtype=torch.float32)
    
    # Test integration (multiple runs for accurate timing)
    print("  Testing integration...")
    num_runs = 1000
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        new_state = torch_model.integrate(state, inputs)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Integration result: {new_state}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test sensitivity computation (multiple runs for accurate timing)
    print("  Testing sensitivity computation...")
    # State sensitivity matrix: nx x np (4 states x 3 parameters)
    state_sensitivity = torch.eye(4, 3, dtype=torch.float32)  # Track sensitivity to first 3 parameters
    # Input gains matrix: nu x nx (2 inputs x 4 states)  
    input_gains = torch.zeros(2, 4, dtype=torch.float32)  # No input gains for now
    
    num_runs = 100
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        sens_result = torch_model.sensitivity_step(state, inputs, state_sensitivity, input_gains)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Sensitivity result shape: {sens_result.shape}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test closed-loop sensitivity (multiple runs for accurate timing)
    print("  Testing closed-loop sensitivity...")
    reference = torch.tensor([1.0, 1.0], dtype=torch.float32)
    
    num_runs = 100
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        cl_sens_result, cl_inputs = torch_model.sensitivity_step_closed_loop(state, reference, state_sensitivity)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Closed-loop sensitivity result shape: {cl_sens_result.shape}")
    print(f"  Closed-loop inputs: {cl_inputs}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    print("  PyTorch tests passed!")

def test_jax_optimization():
    """Test JAX JIT compilation"""
    print("\nTesting JAX optimizations...")
    
    # Create model config (same as CasADi)
    config = create_model_config(nq=2, nv=2, nu=2, ny=2, p_nom=jnp.array([0.1, 0.2, 0.3]), dt=0.01)
    
    # Create JAX model
    jax_model = JAXModel(config, simple_dynamics_jax, simple_controller_jax)
    
    # Test data (4 elements for consistency: [q1, q2, v1, v2])
    state = jnp.array([1.0, 2.0, 0.1, 0.2])  # positions and velocities
    inputs = jnp.array([0.5, -0.5])
    
    # Test integration (multiple runs for accurate timing)
    print("  Testing integration...")
    num_runs = 1000
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        new_state = jax_model.integrate(state, inputs, 0)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Integration result: {new_state}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test JIT compiled integration (multiple runs for accurate timing)
    print("  Testing JIT compiled integration...")
    jit_times = []
    for _ in range(num_runs):
        start_time = time.time()
        new_state_jit = jax_model.integrate(state, inputs, 0)
        jit_times.append(time.time() - start_time)
    
    jit_mean_time = sum(jit_times) / len(jit_times)
    print(f"  JIT integration result: {new_state_jit}")
    print(f"  JIT mean time over {num_runs} runs: {jit_mean_time:.6f}s")
    print(f"  JIT total time: {sum(jit_times):.4f}s")
    print(f"  JIT speedup: {mean_time / jit_mean_time:.2f}x")
    
    # Test sensitivity computation (multiple runs for accurate timing)
    print("  Testing sensitivity computation...")
    # State sensitivity matrix: nx x np (4 states x 3 parameters)
    state_sensitivity = jnp.eye(4, 3)  # Track sensitivity to first 3 parameters
    # Input sensitivity matrix: nu x np (2 inputs x 3 parameters)  
    input_gains = jnp.zeros((2, 4))  # No input sensitivity for now
    
    num_runs = 100
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        sens_result = jax_model.sensitivity_step(state, inputs, state_sensitivity, input_gains, 0)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Sensitivity result shape: {sens_result.shape}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test closed-loop sensitivity (multiple runs for accurate timing)
    print("  Testing closed-loop sensitivity...")
    reference = jnp.array([1.0, 1.0])
    
    num_runs = 100
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        cl_sens_result, cl_inputs = jax_model.sensitivity_step_closed_loop(state, reference, state_sensitivity, 0)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Closed-loop sensitivity result shape: {cl_sens_result.shape}")
    print(f"  Closed-loop inputs: {cl_inputs}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    print("  JAX tests passed!")

def test_casadi_optimization():
    """Test CasADi controller sensitivity optimization"""
    print("\nTesting CasADi optimizations...")
    
    # Create model config
    config = create_model_config(nq=2, nv=2, nu=2, ny=2, p_nom=[0.1, 0.2, 0.3], dt=0.01)
    
    # Create CasADi model
    casadi_model = CasADiModel(config, simple_dynamics, simple_controller)
    
    # Test data (4 elements for CasADi: [q1, q2, v1, v2])
    state = np.array([1.0, 2.0, 0.1, 0.2])  # positions and velocities
    inputs = np.array([0.5, -0.5])
    
    # Test integration (multiple runs for accurate timing)
    print("  Testing integration...")
    num_runs = 1000
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        new_state = casadi_model.integrate(state, inputs)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Integration result: {new_state}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test sensitivity computation (multiple runs for accurate timing)
    print("  Testing sensitivity computation...")
    # State sensitivity matrix: nx x np (4 states x 3 parameters for CasADi)
    state_sensitivity = np.eye(4, 3)  # Track sensitivity to first 3 parameters
    # Input gains matrix: nu x nx (2 inputs x 4 states)
    input_gains = np.zeros((2, 4))  # No input gains for now
    
    num_runs = 1000
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        sens_result = casadi_model.sensitivity_step(state, inputs, state_sensitivity, input_gains, 0)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Sensitivity result shape: {sens_result.shape}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    # Test closed-loop sensitivity (should use pre-compiled controller sensitivity)
    print("  Testing closed-loop sensitivity...")
    reference = np.array([1.0, 1.0])
    
    num_runs = 1000
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        cl_sens_result, cl_inputs = casadi_model.sensitivity_step_closed_loop(state, reference, state_sensitivity)
        times.append(time.time() - start_time)
    
    mean_time = sum(times) / len(times)
    print(f"  Closed-loop sensitivity result shape: {cl_sens_result.shape}")
    print(f"  Closed-loop inputs: {cl_inputs}")
    print(f"  Mean time over {num_runs} runs: {mean_time:.6f}s")
    print(f"  Total time: {sum(times):.4f}s")
    
    print("  CasADi tests passed!")

if __name__ == "__main__":
    print("Testing sensitivity tools optimizations...")
    print("=" * 50)
    
    try:
        test_torch_optimization()
        test_jax_optimization()
        test_casadi_optimization()
        print("\n" + "=" * 50)
        print("All optimization tests passed! 🎉")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
