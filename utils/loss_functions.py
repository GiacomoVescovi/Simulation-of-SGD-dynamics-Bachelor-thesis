"""
Loss functions and gradient computations for SGD simulations.

Implements various loss landscapes including smooth nonlinear functions
and piecewise linear approximations.
"""

import numpy as np
from typing import Callable, Tuple


def smooth_nonlinear_loss(params: np.ndarray, x: np.ndarray, y: np.ndarray, p: float = 1.0) -> float:
    """
    Smooth nonlinear loss function.
    
    This uses a sigmoid-based smooth function:
    f(x; a, b) = x * (1 + b * (1 + exp(-a*x))^(-1))
    
    Parameters
    ----------
    params : np.ndarray
        Parameters [a, b]
    x : np.ndarray
        Input data
    y : np.ndarray
        Target data
    p : float
        Shape parameter (default: 1.0)
        
    Returns
    -------
    float
        Mean squared error
    """
    a, b = params[0], params[1]
    
    # Compute predictions
    predictions = x * (1 + b / (1 + np.exp(-a * x)))
    
    # Compute MSE
    return np.mean((y - predictions) ** 2)


def smooth_nonlinear_gradient(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of smooth nonlinear loss.
    
    Parameters
    ----------
    params : np.ndarray
        Parameters [a, b]
    x : np.ndarray
        Input data
    y : np.ndarray
        Target data
        
    Returns
    -------
    np.ndarray
        Gradient [grad_a, grad_b]
    """
    a, b = params[0], params[1]
    
    # Compute intermediate values
    exp_term = np.exp(-a * x)
    sigmoid = 1.0 / (1 + exp_term)
    predictions = x * (1 + b * sigmoid)
    residuals = predictions - y
    
    # Gradient w.r.t. a
    # d/da of (1 + b * sigmoid) = b * sigmoid * (1 - sigmoid) * x
    grad_a = 2 * np.mean(residuals * x * b * sigmoid * (1 - sigmoid) * x)
    
    # Gradient w.r.t. b
    # d/db of (1 + b * sigmoid) = sigmoid
    grad_b = 2 * np.mean(residuals * x * sigmoid)
    
    return np.array([grad_a, grad_b])


def piecewise_linear_loss(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """
    Piecewise linear loss function.
    
    Parameters
    ----------
    params : np.ndarray
        Linear parameters [intercept, slope]
    x : np.ndarray
        Input data
    y : np.ndarray
        Target data
        
    Returns
    -------
    float
        Mean squared error
    """
    predictions = params[0] + params[1] * x
    return np.mean((y - predictions) ** 2)


def piecewise_linear_gradient(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of piecewise linear loss.
    
    Parameters
    ----------
    params : np.ndarray
        Parameters [intercept, slope]
    x : np.ndarray
        Input data
    y : np.ndarray
        Target data
        
    Returns
    -------
    np.ndarray
        Gradient [grad_intercept, grad_slope]
    """
    predictions = params[0] + params[1] * x
    residuals = predictions - y
    
    grad_intercept = 2 * np.mean(residuals)
    grad_slope = 2 * np.mean(residuals * x)
    
    return np.array([grad_intercept, grad_slope])


def smooth_approximation(x: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """
    Smooth approximation of a piecewise function using sigmoid.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    a : float
        Steepness parameter (larger = sharper transition)
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Smoothed output
    """
    return x * (1 + b / (1 + np.exp(-a * x)))


def true_function(x: np.ndarray, p: float = 1.0) -> np.ndarray:
    """
    True underlying function for data generation.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    p : float
        Shape parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return x * (1 + p / (1 + np.exp(-x)))


def generate_noisy_data(
    x_range: Tuple[float, float],
    n_points: int,
    n_samples_per_point: int = 1,
    noise_std: float = 1.0,
    p: float = 1.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy data from the true function.
    
    Parameters
    ----------
    x_range : tuple
        (min, max) for x values
    n_points : int
        Number of x points
    n_samples_per_point : int
        Number of noisy samples per x point
    noise_std : float
        Standard deviation of Gaussian noise
    p : float
        Shape parameter for true function
    random_state : int, optional
        Random seed
        
    Returns
    -------
    x_data : np.ndarray
        Input values (repeated for multiple samples)
    y_data : np.ndarray
        Noisy output values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate x values
    x_points = np.linspace(x_range[0], x_range[1], n_points)
    
    # Compute true function values
    y_true = true_function(x_points, p)
    
    # Generate noisy samples
    x_data = []
    y_data = []
    
    for i, (x, y) in enumerate(zip(x_points, y_true)):
        for _ in range(n_samples_per_point):
            noise = np.random.normal(0, noise_std)
            x_data.append(x)
            y_data.append(y + noise)
    
    return np.array(x_data), np.array(y_data)


def compute_gradient(
    loss_fn: Callable,
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute gradient using finite differences (numerical approximation).
    
    Parameters
    ----------
    loss_fn : callable
        Loss function: loss_fn(params, x, y) -> scalar
    params : np.ndarray
        Parameters at which to compute gradient
    x : np.ndarray
        Input data
    y : np.ndarray
        Target data
    epsilon : float
        Step size for finite differences
        
    Returns
    -------
    np.ndarray
        Approximate gradient
    """
    n_params = len(params)
    grad = np.zeros(n_params)
    
    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        loss_plus = loss_fn(params_plus, x, y)
        loss_minus = loss_fn(params_minus, x, y)
        
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad


class LossFunction:
    """Base class for loss functions with automatic gradient."""
    
    def __call__(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """Compute loss."""
        raise NotImplementedError
    
    def gradient(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        raise NotImplementedError


class SmoothNonlinearLoss(LossFunction):
    """Smooth nonlinear loss with analytic gradient."""
    
    def __init__(self, p: float = 1.0):
        self.p = p
    
    def __call__(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return smooth_nonlinear_loss(params, x, y, self.p)
    
    def gradient(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return smooth_nonlinear_gradient(params, x, y)


class PiecewiseLinearLoss(LossFunction):
    """Piecewise linear loss with analytic gradient."""
    
    def __call__(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return piecewise_linear_loss(params, x, y)
    
    def gradient(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return piecewise_linear_gradient(params, x, y)
