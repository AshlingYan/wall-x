"""
Data utilities for robot state/action processing.
This module provides rotation-related functions for robot manipulation.
"""

import numpy as np


def euler_to_matrix_zyx_6d_nb(euler_angles):
    """
    Convert Euler angles (ZYX) to 6D rotation representation.
    This is a stub implementation for compatibility.

    Args:
        euler_angles: Euler angles in ZYX convention

    Returns:
        6D rotation representation
    """
    # Stub: return zeros for now
    if isinstance(euler_angles, np.ndarray):
        return np.zeros_like(euler_angles)
    return np.zeros(3)


def so3_to_euler_zyx_batch_nb(rotation_6d):
    """
    Convert 6D rotation representation to Euler angles (ZYX).
    This is a stub implementation for compatibility.

    Args:
        rotation_6d: 6D rotation representation

    Returns:
        Euler angles in ZYX convention
    """
    # Stub: return zeros for now
    if isinstance(rotation_6d, np.ndarray):
        shape = rotation_6d.shape[:-1] if rotation_6d.ndim > 1 else (3,)
        return np.zeros(shape)
    return np.zeros(3)


def compose_state_and_delta_to_abs_rpy(state_rpy, delta_rpy):
    """
    Compose state rotation and delta rotation to absolute rotation.
    This is a stub implementation for compatibility.

    Args:
        state_rpy: State rotation in roll-pitch-yaw
        delta_rpy: Delta rotation in roll-pitch-yaw

    Returns:
        Absolute rotation in roll-pitch-yaw
    """
    # Stub: simple addition for now
    if isinstance(state_rpy, np.ndarray) and isinstance(delta_rpy, np.ndarray):
        return state_rpy + delta_rpy
    return state_rpy + delta_rpy
