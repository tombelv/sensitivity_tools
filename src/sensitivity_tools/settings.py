"""
Settings and constants for sensitivity tools package
"""

# Integrator types supported by the package
INTEGRATOR_TYPES = [
    "si_euler",  # Semi-implicit Euler
    "euler",  # Explicit Euler
    "rk4",  # Runge-Kutta 4th order
    "custom_discrete",  # Custom discrete integrator
]

# Alias for backward compatibility
MODEL_PARAMETRIC_INTEGRATOR_TYPES = INTEGRATOR_TYPES
