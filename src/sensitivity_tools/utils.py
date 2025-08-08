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

