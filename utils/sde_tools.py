"""
Tools for stochastic differential equations (SDE) and Langevin dynamics.

Implements Ito processes, escape time calculations, and theoretical predictions
based on Kramers formula and Fokker-Planck equations.
"""

import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
from typing import Callable, Tuple, Optional
import warnings


def langevin_dynamics(
    initial_state: np.ndarray,
    drift_fn: Callable,
    diffusion: float,
    dt: float,
    n_steps: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Simulate Langevin dynamics using Euler-Maruyama method.
    
    The Langevin equation is:
    dx = drift(x) dt + sqrt(2 * diffusion) dW
    
    Parameters
    ----------
    initial_state : np.ndarray
        Initial state vector
    drift_fn : callable
        Drift function: drift_fn(x) -> drift vector
    diffusion : float
        Diffusion coefficient (related to temperature)
    dt : float
        Time step
    n_steps : int
        Number of steps
    random_state : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Trajectory, shape (n_steps+1, n_dims)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = len(initial_state)
    trajectory = np.zeros((n_steps + 1, n_dims))
    trajectory[0] = initial_state
    
    sqrt_dt = np.sqrt(dt)
    sqrt_2D = np.sqrt(2 * diffusion)
    
    for i in range(n_steps):
        x = trajectory[i]
        drift = drift_fn(x)
        noise = np.random.randn(n_dims)
        
        trajectory[i + 1] = x + drift * dt + sqrt_2D * sqrt_dt * noise
    
    return trajectory


def ito_process_2d(
    initial_state: np.ndarray,
    drift_fn: Callable,
    diffusion_matrix: np.ndarray,
    dt: float,
    n_steps: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Simulate 2D Ito process.
    
    dx = drift(x) dt + diffusion @ dW
    
    Parameters
    ----------
    initial_state : np.ndarray
        Initial state [x1, x2]
    drift_fn : callable
        Drift function: drift_fn([x1, x2]) -> [drift1, drift2]
    diffusion_matrix : np.ndarray
        Diffusion matrix (2x2)
    dt : float
        Time step
    n_steps : int
        Number of steps
    random_state : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Trajectory, shape (n_steps+1, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    trajectory = np.zeros((n_steps + 1, 2))
    trajectory[0] = initial_state
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(n_steps):
        x = trajectory[i]
        drift = drift_fn(x)
        noise = np.random.randn(2)
        
        trajectory[i + 1] = x + drift * dt + diffusion_matrix @ noise * sqrt_dt
    
    return trajectory


def compute_escape_time(
    initial_state: np.ndarray,
    target_region: Callable,
    drift_fn: Callable,
    diffusion: float,
    dt: float,
    max_steps: int = 100000,
    random_state: Optional[int] = None
) -> Tuple[int, bool]:
    """
    Compute escape time from a region.
    
    Parameters
    ----------
    initial_state : np.ndarray
        Starting point
    target_region : callable
        Function that returns True if point is in target region
    drift_fn : callable
        Drift function for Langevin dynamics
    diffusion : float
        Diffusion coefficient
    dt : float
        Time step
    max_steps : int
        Maximum number of steps before giving up
    random_state : int, optional
        Random seed
        
    Returns
    -------
    escape_time : int
        Number of steps to escape (or max_steps if didn't escape)
    escaped : bool
        Whether escape occurred
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = len(initial_state)
    x = initial_state.copy()
    
    sqrt_dt = np.sqrt(dt)
    sqrt_2D = np.sqrt(2 * diffusion)
    
    for step in range(max_steps):
        # Check if in target region
        if target_region(x):
            return step, True
        
        # Evolve dynamics
        drift = drift_fn(x)
        noise = np.random.randn(n_dims)
        x = x + drift * dt + sqrt_2D * sqrt_dt * noise
    
    return max_steps, False


def kramers_rate(
    barrier_height: float,
    diffusion: float,
    omega_min: float,
    omega_barrier: float
) -> float:
    """
    Calculate escape rate using Kramers formula.
    
    For a potential U(x), the escape rate from a minimum is:
    k = (omega_min / (2*pi)) * exp(-barrier_height / diffusion)
    
    where omega_min is the curvature at the minimum and omega_barrier
    is the curvature at the barrier top.
    
    Parameters
    ----------
    barrier_height : float
        Height of the energy barrier
    diffusion : float
        Diffusion coefficient (proportional to temperature)
    omega_min : float
        Frequency at the minimum
    omega_barrier : float
        Frequency at the barrier
    
    Returns
    -------
    float
        Escape rate (inverse of mean escape time)
    """
    if diffusion <= 0:
        return 0.0
    
    prefactor = (omega_min * omega_barrier) / (2 * np.pi)
    exponential = np.exp(-barrier_height / diffusion)
    
    return prefactor * exponential


def mean_escape_time_kramers(
    barrier_height: float,
    diffusion: float,
    omega_min: float = 1.0,
    omega_barrier: float = 1.0
) -> float:
    """
    Calculate mean escape time using Kramers formula.
    
    Parameters
    ----------
    barrier_height : float
        Height of the energy barrier
    diffusion : float
        Diffusion coefficient (related to learning rate in SGD)
    omega_min : float
        Frequency at the minimum (default: 1.0)
    omega_barrier : float
        Frequency at the barrier (default: 1.0)
        
    Returns
    -------
    float
        Mean escape time
    """
    rate = kramers_rate(barrier_height, diffusion, omega_min, omega_barrier)
    
    if rate <= 0:
        return np.inf
    
    return 1.0 / rate


def fokker_planck_stationary_1d(
    x_grid: np.ndarray,
    potential_fn: Callable,
    diffusion: float
) -> np.ndarray:
    """
    Compute stationary distribution for 1D Fokker-Planck equation.
    
    The stationary distribution is:
    p_stat(x) ∝ exp(-U(x) / D)
    
    where U(x) is the potential and D is the diffusion coefficient.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Grid of x values
    potential_fn : callable
        Potential function U(x)
    diffusion : float
        Diffusion coefficient
        
    Returns
    -------
    np.ndarray
        Normalized stationary distribution
    """
    # Compute unnormalized distribution
    potential = np.array([potential_fn(x) for x in x_grid])
    p_unnorm = np.exp(-potential / diffusion)
    
    # Normalize (trapezoidal rule)
    dx = x_grid[1] - x_grid[0]
    norm = np.trapz(p_unnorm, dx=dx)
    
    return p_unnorm / norm


def drift_from_loss(
    loss_fn: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    epsilon: float = 1e-5
) -> Callable:
    """
    Create drift function for SGD from loss function.
    
    The drift is the negative gradient: drift = -grad(loss)
    
    Parameters
    ----------
    loss_fn : callable
        Loss function: loss_fn(params, x, y) -> scalar
    x_data, y_data : np.ndarray
        Training data
    epsilon : float
        Step size for finite differences
        
    Returns
    -------
    callable
        Drift function: drift_fn(params) -> drift vector
    """
    def drift_fn(params):
        n_params = len(params)
        grad = np.zeros(n_params)
        
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            loss_plus = loss_fn(params_plus, x_data, y_data)
            loss_minus = loss_fn(params_minus, x_data, y_data)
            
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Drift is negative gradient (gradient descent)
        return -grad
    
    return drift_fn


def sgd_as_langevin(
    learning_rate: float,
    batch_size: int,
    n_data: int,
    gradient_variance: float
) -> float:
    """
    Map SGD to Langevin dynamics to get effective diffusion coefficient.
    
    For SGD with learning rate η and minibatch size b:
    D = (η^2 / 2) * (n - b) / (n - 1) * Var[gradient]
    
    Parameters
    ----------
    learning_rate : float
        SGD learning rate
    batch_size : int
        Minibatch size
    n_data : int
        Total number of data points
    gradient_variance : float
        Variance of stochastic gradient
        
    Returns
    -------
    float
        Effective diffusion coefficient
    """
    if batch_size >= n_data:
        # Full batch: no noise
        return 0.0
    
    sampling_factor = (n_data - batch_size) / (n_data - 1)
    diffusion = 0.5 * learning_rate**2 * sampling_factor * gradient_variance
    
    return diffusion


def estimate_gradient_variance(
    gradient_fn: Callable,
    params: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_samples: int = 100
) -> float:
    """
    Estimate variance of stochastic gradient at a point.
    
    Parameters
    ----------
    gradient_fn : callable
        Gradient function
    params : np.ndarray
        Parameters at which to estimate variance
    x_data, y_data : np.ndarray
        Training data
    n_samples : int
        Number of samples for estimation
        
    Returns
    -------
    float
        Estimated gradient variance (mean over dimensions)
    """
    n_data = len(x_data)
    n_params = len(params)
    
    # Compute gradients for individual data points
    gradients = []
    for _ in range(n_samples):
        idx = np.random.randint(n_data)
        grad = gradient_fn(params, x_data[idx:idx+1], y_data[idx:idx+1])
        gradients.append(grad)
    
    gradients = np.array(gradients)
    
    # Compute variance for each parameter dimension
    variances = np.var(gradients, axis=0)
    
    # Return mean variance
    return np.mean(variances)
