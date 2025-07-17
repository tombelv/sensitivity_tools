"""
Test basic functionality of sensitivity_tools package
"""

import pytest
import numpy as np
from sensitivity_tools import INTEGRATOR_TYPES, ellipsoid_radius, ellipsoid_radii


def test_integrator_types():
    """Test that integrator types are defined correctly"""
    expected_types = ["si_euler", "euler", "rk4", "custom_discrete"]
    assert INTEGRATOR_TYPES == expected_types


def test_ellipsoid_radius():
    """Test ellipsoid radius computation"""
    # Simple test case
    sensitivity = np.array([[1.0, 0.0], [0.0, 1.0]])
    scaling_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Test without direction (should return diagonal elements)
    radii = ellipsoid_radius(sensitivity, scaling_matrix)
    np.testing.assert_array_almost_equal(radii, [1.0, 1.0])
    
    # Test with direction
    direction = np.array([1.0, 0.0])
    radius = ellipsoid_radius(sensitivity, scaling_matrix, direction)
    np.testing.assert_almost_equal(radius, 1.0)


def test_ellipsoid_radii():
    """Test ellipsoid radii computation"""
    sensitivity = np.array([[2.0, 0.0], [0.0, 3.0]])
    scaling_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    radii = ellipsoid_radii(sensitivity, scaling_matrix)
    expected = np.sqrt([4.0, 9.0])  # sqrt of diagonal elements
    np.testing.assert_array_almost_equal(radii, expected)


def test_package_imports():
    """Test that package imports work correctly"""
    import sensitivity_tools
    
    # Test that version is defined
    assert hasattr(sensitivity_tools, '__version__')
    assert isinstance(sensitivity_tools.__version__, str)
    
    # Test that main components are available
    assert hasattr(sensitivity_tools, 'INTEGRATOR_TYPES')
    assert hasattr(sensitivity_tools, 'ellipsoid_radius')
    assert hasattr(sensitivity_tools, 'ellipsoid_radii')


def test_optional_imports():
    """Test that optional backend imports work"""
    import sensitivity_tools
    
    # These should be None if backends are not installed, or modules if they are
    jax_models = getattr(sensitivity_tools, 'jax_models', None)
    casadi_models = getattr(sensitivity_tools, 'casadi_models', None)
    torch_models = getattr(sensitivity_tools, 'torch_models', None)
    
    # We don't test the actual functionality here since backends might not be installed
    # Just check that the attributes exist
    assert hasattr(sensitivity_tools, 'jax_models')
    assert hasattr(sensitivity_tools, 'casadi_models')
    assert hasattr(sensitivity_tools, 'torch_models')


if __name__ == "__main__":
    pytest.main([__file__])
