#!/usr/bin/env python3
"""
Installation verification script for sensitivity-tools
"""

import sys
import importlib

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description} - {e}")
        return False

def main():
    print("Sensitivity Tools Installation Verification")
    print("=" * 50)
    
    # Test core package
    success = test_import("sensitivity_tools", "Core package")
    success &= test_import("sensitivity_tools.utils", "Utility functions")
    success &= test_import("sensitivity_tools.settings", "Settings module")
    
    # Test backends
    jax_available = test_import("sensitivity_tools.jax_models", "JAX backend")
    casadi_available = test_import("sensitivity_tools.casadi_models", "CasADi backend")
    torch_available = test_import("sensitivity_tools.torch_models", "PyTorch backend")
    
    print("\nBackend Summary:")
    print(f"JAX: {'Available' if jax_available else 'Not installed'}")
    print(f"CasADi: {'Available' if casadi_available else 'Not installed'}")
    print(f"PyTorch: {'Available' if torch_available else 'Not installed'}")
    
    if success:
        print("\n✓ Core package installed successfully!")
        return 0
    else:
        print("\n✗ Installation verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
