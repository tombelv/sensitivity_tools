# Sensitivity Tools

A Python package for computing closed-loop sensitivity of dynamical systems with implementations in JAX, CasADi, and PyTorch.

## Features

- **Multi-backend support**: JAX, CasADi, and PyTorch implementations
- **High-performance computing**: JIT compilation (JAX), symbolic computation (CasADi), and torch.compile optimization (PyTorch)
- **Multiple integrators**: Semi-implicit Euler, Euler, RK4, and custom discrete integrators
- **Sensitivity computation**: Automatic differentiation for computing sensitivities with respect to states, inputs, and parameters
- **Ellipsoid analysis**: Tools for computing ellipsoid radii for uncertainty propagation
- **Flexible architecture**: Abstract base classes for easy extension

## Installation

### Using pixi (recommended)

```bash
# Add to your project
pixi add sensitivity-tools

# Or install with specific backend
pixi add sensitivity-tools[jax]     # JAX support
pixi add sensitivity-tools[casadi]  # CasADi support  
pixi add sensitivity-tools[torch]   # PyTorch support
pixi add sensitivity-tools[all]     # All backends
```

### Using pip

```bash
pip install sensitivity-tools
```

### From source

```bash
git clone https://github.com/tombelv/sensitivity-tools.git
cd sensitivity-tools
pixi install
pixi run install-dev
```

## Optional Dependencies

The package supports multiple backends. Install the ones you need:

### With pixi (recommended)
```bash
# Switch to different environments
pixi shell jax      # JAX environment
pixi shell casadi   # CasADi environment
pixi shell torch    # PyTorch environment
pixi shell all      # All backends environment
```

### With pip
```bash
# For JAX support
pip install sensitivity-tools[jax]

# For CasADi support  
pip install sensitivity-tools[casadi]

# For PyTorch support
pip install sensitivity-tools[torch]

# Install all backends
pip install sensitivity-tools[all]
```

## Quick Start

### JAX Example

```python
import jax.numpy as jnp
from sensitivity_tools.models.jax_models import Model
from sensitivity_tools.settings import create_model_config

# Define your dynamics function
def pendulum_dynamics(state, inputs, params, ext):
    q, qdot = state[0], state[1]
    u = inputs[0]
    m, l, g, b = params
    
    qddot = (u - m * g * l * jnp.sin(q) - b * qdot) / (m * l**2)
    return jnp.array([qdot, qddot])

# Create model configuration
config = create_model_config(
    nq=1,  # position dimension
    nv=1,  # velocity dimension  
    nu=1,  # input dimension
    p_nom=[1.0, 1.0, 9.81, 0.1],  # m, l, g, b
    dt=0.01
)

# Create model
model = Model(config, pendulum_dynamics)

# Integrate one step
state = jnp.array([0.1, 0.0])  # [q, qdot]
inputs = jnp.array([0.0])      # [u]

next_state = model.integrate(state, inputs)
```

### CasADi Example

```python
import casadi as cs
import numpy as np
from sensitivity_tools.models.casadi_models import Model
from sensitivity_tools.settings import create_model_config

# Define dynamics
def pendulum_dynamics(state, inputs, params, ext=0):
    q, qdot = state[0], state[1]
    u = inputs[0]
    m, l, g, b = params[0], params[1], params[2], params[3]
    
    qddot = (u - m * g * l * cs.sin(q) - b * qdot) / (m * l**2)
    return cs.vertcat(qdot, qddot)

# Create model configuration
config = create_model_config(
    nq=1, nv=1, nu=1,
    p_nom=np.array([1.0, 1.0, 9.81, 0.1]),
    dt=0.01
)

# Create model
model = Model(config, pendulum_dynamics)

# Integration
state = np.array([0.1, 0.0])
inputs = np.array([0.0])

next_state = model.integrate(state, inputs)

# Sensitivity computation
state_sensitivity = np.eye(2)  # 2x2 identity
input_sensitivity = np.zeros((1, 2))  # 1x2 matrix

sensitivity = model.sensitivity_step(state, inputs, state_sensitivity, input_sensitivity)
```

### PyTorch Example

```python
import torch
from sensitivity_tools.models.torch_models import Model
from sensitivity_tools.settings import create_model_config

# Define dynamics
def pendulum_dynamics(state, inputs, params, ext):
    q, qdot = state[0], state[1]
    u = inputs[0]
    m, l, g, b = params[0], params[1], params[2], params[3]
    
    qddot = (u - m * g * l * torch.sin(q) - b * qdot) / (m * l**2)
    return torch.stack([qdot, qddot])

# Create model configuration
config = create_model_config(
    nq=1, nv=1, nu=1,
    p_nom=[1.0, 1.0, 9.81, 0.1],
    dt=0.01
)

# Create model
model = Model(config, pendulum_dynamics)

# Integrate
state = torch.tensor([0.1, 0.0])
inputs = torch.tensor([0.0])

next_state = model.integrate(state, inputs)
```

## API Reference

### Core Classes

- `BaseModel`: Abstract base class for all models
- `ModelParametric`: Parametric model supporting different integrators
- `Model`: Non-parametric model with fixed parameters

### Integrators

- `si_euler`: Semi-implicit Euler method
- `euler`: Explicit Euler method  
- `rk4`: 4th-order Runge-Kutta method
- `custom_discrete`: User-defined discrete integrator

### Utility Functions

- `ellipsoid_radius()`: Compute ellipsoid radii for uncertainty analysis
- `ellipsoid_radii()`: Compute all coordinate axis radii
- `integrate_rk4_function()`: Standalone RK4 integrator


## Development

### Using pixi (recommended)

```bash
# Clone and set up development environment
git clone https://github.com/tombelv/sensitivity-tools.git
cd sensitivity-tools

# Install development dependencies
pixi install

# Install package in development mode
pixi run install-dev

# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Format code
pixi run format

# Type checking
pixi run type-check

# Run example
pixi run example

# Verify installation
pixi run verify

# Build package
pixi run build

# Publish to PyPI (optional - after configuring credentials)
# pixi run publish
```

### Using hatch

```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Format code
hatch run lint:fmt

# Type checking
hatch run lint:typing
```

### Development workflow with pixi

```bash
# Start development in JAX environment
pixi shell jax

# Start development in PyTorch environment  
pixi shell torch

# Start development in CasADi environment
pixi shell casadi

# Start development with all backends
pixi shell all
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sensitivity_tools,
  author = {Belvedere, Tommaso},
  title = {Sensitivity Tools: A Python package for computing closed-loop sensitivity of dynamical systems},
  url = {https://github.com/tombelv/sensitivity-tools},
  year = {2025}
}
```