"""
Utilities for SGD dynamics simulation.
Originally implemented in Mathematica, converted to Python.
"""

from .sgd_simulator import SGDSimulator, sgd_step, sgd_trajectory
from .loss_functions import (
    smooth_nonlinear_loss,
    piecewise_linear_loss,
    smooth_approximation,
    compute_gradient
)
from .visualization import (
    plot_trajectories,
    plot_loss_landscape,
    plot_escape_times,
    plot_stationary_distribution
)
from .sde_tools import (
    langevin_dynamics,
    ito_process_2d,
    compute_escape_time,
    kramers_rate
)

__all__ = [
    'SGDSimulator',
    'sgd_step',
    'sgd_trajectory',
    'smooth_nonlinear_loss',
    'piecewise_linear_loss',
    'smooth_approximation',
    'compute_gradient',
    'plot_trajectories',
    'plot_loss_landscape',
    'plot_escape_times',
    'plot_stationary_distribution',
    'langevin_dynamics',
    'ito_process_2d',
    'compute_escape_time',
    'kramers_rate',
]
