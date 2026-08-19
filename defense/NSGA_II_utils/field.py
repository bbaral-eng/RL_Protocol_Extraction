import numpy as np
import defense.NSGA_II_utils.config as config
from helper_functions import voltage_grid

# field.py converts the defense schedule into actual EMF values based on which electrodes are actuated. 

def real_field(electrodes):
    """
    EMF produced by the droplet at a given electrode. 
    """
    return voltage_grid(tuple(electrodes), config.REAL_SIGMA, config.OBSTACLES, config.GRID_SIZE)


def dummy_field(electrodes):
    """
    EMF produced by one dummy actuation: lower peak, tighter spread.
    """
    grid = voltage_grid(tuple(electrodes), config.DUMMY_SIGMA, config.OBSTACLES, config.GRID_SIZE)
    return config.DUMMY_AMPLITUDE * grid


def dummy_sum(electrodes):
    """
    Sum of EMF from the dummies on their own, with the droplet left out.
    """
    total = np.zeros((config.GRID_SIZE, config.GRID_SIZE))
    for electrode in electrodes:
        total = total + dummy_field(electrode)
    return total


def combined_field(true_electrode, dummy_electrode):
    """
    EMF from dummies + droplet EMF
    """
    total = real_field(true_electrode) + dummy_sum(dummy_electrode)
    return np.clip(total, 0.0, 1.0)


def field_sequence(protocol, schedule):
    """
    used for plotting purposes, shows combined field per timesep 
    """
    return [combined_field(electrode, dummies) for electrode, dummies in zip(protocol, schedule)]
