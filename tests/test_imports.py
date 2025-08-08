#!/usr/bin/env python3

try:
    from sensitivity_tools.models import jax_models
    print('JAX import successful')
except ImportError as e:
    print(f'JAX import failed: {e}')

try:
    from sensitivity_tools.models import casadi_models
    print('CasADi import successful')
except ImportError as e:
    print(f'CasADi import failed: {e}')

try:
    from sensitivity_tools.models import torch_models
    print('PyTorch import successful')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
