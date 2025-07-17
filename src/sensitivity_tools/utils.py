"""
Utility functions for sensitivity analysis
"""

import numpy as np


def ellipsoid_radius(sensitivity, scaling_matrix, direction=None):
    """
    Compute the ellipsoid radius with the formula
    r = sqrt( n^T * Pi * W * Pi^T * n )
    where n is the direction in which we want to compute the radius.

    Parameters
    ----------
    sensitivity : numpy.ndarray
        Sensitivity matrix (Pi)
    scaling_matrix : numpy.ndarray
        Scaling/weighting matrix (W)
    direction : numpy.ndarray, optional
        Direction vector (n). If None, returns radii along Euclidean basis

    Returns
    -------
    float or numpy.ndarray
        Ellipsoid radius(i) in the specified direction(s)
    """
    if direction is None:
        return np.sqrt((sensitivity @ scaling_matrix @ sensitivity.T).diagonal())
    else:
        return np.sqrt(direction.T @ sensitivity @ scaling_matrix @ sensitivity.T @ direction)


def ellipsoid_radii(sensitivity, scaling_matrix):
    """
    Compute ellipsoid radii along all coordinate axes.
    
    Parameters
    ----------
    sensitivity : numpy.ndarray
        Sensitivity matrix (Pi)
    scaling_matrix : numpy.ndarray
        Scaling/weighting matrix (W)
        
    Returns
    -------
    numpy.ndarray
        Array of ellipsoid radii along each coordinate axis
    """
    return ellipsoid_radius(sensitivity, scaling_matrix, direction=None)


def integrate_rk4_function(funct, state, inputs, params, time, dt):
    """
    One-step integration of the dynamics using RK4 method
    
    Parameters
    ----------
    funct : callable
        Function to integrate with signature (state, inputs, params, time)
    state : array-like
        Current state
    inputs : array-like
        Control inputs
    params : array-like
        Parameters
    time : float
        Current time
    dt : float
        Time step
        
    Returns
    -------
    array-like
        Next state after integration
    """
    k1 = funct(state, inputs, params, time)
    k2 = funct(state + k1 * dt / 2., inputs, params, time + dt / 2)
    k3 = funct(state + k2 * dt / 2., inputs, params, time + dt / 2)
    k4 = funct(state + k3 * dt, inputs, params, time + dt)
    return state + (dt / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
