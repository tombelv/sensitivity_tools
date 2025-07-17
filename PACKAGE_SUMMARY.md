# Sensitivity Tools Package - Pixi-Based Setup Summary

## Package Overview

You now have a complete Python package `sensitivity-tools` for computing closed-loop sensitivity of dynamical systems with support for multiple backends:

- **JAX**: For high-performance computing with automatic differentiation
- **CasADi**: For symbolic computation and optimization
- **PyTorch**: For deep learning integration and GPU acceleration

## Package Structure

```
sensitivity_tools/
├── src/sensitivity_tools/
│   ├── __init__.py          # Main package initialization
│   ├── settings.py          # Configuration and constants
│   ├── utils.py             # Utility functions
│   ├── jax_models.py        # JAX implementation
│   ├── casadi_models.py     # CasADi implementation
│   └── torch_models.py      # PyTorch implementation
├── tests/                   # Test suite
├── examples/                # Example usage
├── pyproject.toml           # Hatch configuration
├── pixi.toml               # Pixi configuration (primary)
└── verify_install.py       # Installation verification
```

## Installation Options

### Using Pixi (Recommended)

```bash
# Add to your project
pixi add sensitivity-tools

# With specific backend
pixi add sensitivity-tools[jax]     # JAX support
pixi add sensitivity-tools[casadi]  # CasADi support
pixi add sensitivity-tools[torch]   # PyTorch support
pixi add sensitivity-tools[all]     # All backends
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/tombelv/sensitivity-tools.git
cd sensitivity-tools

# Install with pixi
pixi install
pixi run install-dev
```

## Package Features

### Core Components

1. **Base Classes**: Abstract base classes for all model types
2. **Integrators**: Multiple numerical integration methods
   - Semi-implicit Euler (`si_euler`)
   - Explicit Euler (`euler`)
   - 4th-order Runge-Kutta (`rk4`)
   - Custom discrete integrators
3. **Sensitivity Computation**: Automatic differentiation for all backends
4. **Utility Functions**: Ellipsoid radius computation for uncertainty analysis

### Backend-Specific Features

- **JAX**: JIT compilation, GPU/TPU support, forward-mode AD
- **CasADi**: Symbolic computation, nonlinear optimization
- **PyTorch**: Deep learning integration, GPU acceleration

## Usage Examples

### Basic Usage

```python
import numpy as np
from sensitivity_tools import INTEGRATOR_TYPES, ellipsoid_radius

# Check available integrators
print(f"Available integrators: {INTEGRATOR_TYPES}")

# Compute ellipsoid radius
sensitivity = np.array([[1.0, 0.1], [0.1, 1.0]])
scaling_matrix = np.eye(2)
radii = ellipsoid_radius(sensitivity, scaling_matrix)
```

### JAX Backend

```python
import jax.numpy as jnp
from sensitivity_tools.jax_models import Model

def pendulum_dynamics(state, inputs, params):
    q, qdot = state[0], state[1]
    u = inputs[0]
    m, l, g, b = params
    qddot = (u - m * g * l * jnp.sin(q) - b * qdot) / (m * l**2)
    return jnp.array([qdot, qddot])

model = Model(
    model_dynamics=pendulum_dynamics,
    nq=1, nv=1, nu=1,
    nominal_parameters=[1.0, 1.0, 9.81, 0.1]
)

state = jnp.array([0.1, 0.0])
inputs = jnp.array([0.0])
dt = 0.01

next_state = model.integrate(state, inputs, dt)
```

## Development Workflow

### Using Pixi (Primary Method)

```bash
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

# Publish to PyPI
pixi run publish
```

### Backend-Specific Development

```bash
# Work with JAX environment
pixi shell jax

# Work with CasADi environment
pixi shell casadi

# Work with PyTorch environment
pixi shell torch

# Work with all backends
pixi shell all
```

### Using Hatch (Alternative)

```bash
# Run tests
hatch run test

# Format code
hatch run lint:fmt

# Type checking
hatch run lint:typing
```

## Testing

The package includes comprehensive tests:

```bash
# Run all tests
pixi run test

# Run with coverage
pixi run test-cov

# Run verification script
pixi run verify

# Run example
pixi run example
```

## Building Distribution Packages

### Python Wheels

```bash
# Build wheel with pixi
pixi run build

# Or with hatch
hatch build
```

## Current Status

✅ **Package Structure**: Complete with proper organization  
✅ **Multi-Backend Support**: JAX, CasADi, PyTorch implementations  
✅ **Build System**: Hatch configuration with pyproject.toml  
✅ **Package Manager**: Pixi as primary package manager  
✅ **Testing**: Comprehensive test suite  
✅ **Documentation**: README with usage examples  
✅ **Examples**: Working pendulum example  
✅ **CI/CD**: GitHub Actions workflow with pixi  
✅ **Development Tools**: Formatting, linting, type checking  

## Next Steps

1. **Add More Examples**: Create examples for different dynamical systems
2. **Extended Documentation**: Add API documentation with Sphinx
3. **Performance Benchmarks**: Compare backend performance
4. **Additional Integrators**: Add more numerical integration methods
5. **Publish to PyPI**: Upload to Python Package Index
6. **Community Features**: Add contributing guidelines and issue templates

The package is now streamlined for pixi-based development and distribution!
