"""
SGD Simulator for studying stochastic gradient descent dynamics.

This module implements the core SGD algorithm and trajectory generation,
supporting both standard SGD and its analysis as a stochastic differential equation.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SGDConfig:
    """Configuration for SGD simulation."""
    learning_rate: float = 0.01
    batch_size: int = 1
    n_iterations: int = 1000
    random_state: Optional[int] = None
    decay_rate: float = 0.0
    

class SGDSimulator:
    """
    Simulates SGD dynamics on arbitrary loss functions.
    
    This class treats SGD as a discrete-time stochastic process and can
    generate multiple trajectories for statistical analysis.
    """
    
    def __init__(self, config: SGDConfig):
        """
        Initialize the SGD simulator.
        
        Parameters
        ----------
        config : SGDConfig
            Configuration parameters for SGD
        """
        self.config = config
        if config.random_state is not None:
            np.random.seed(config.random_state)
    
    def step(
        self,
        params: np.ndarray,
        gradient_fn: Callable,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        iteration: int = 0
    ) -> np.ndarray:
        """
        Perform a single SGD step.
        
        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        gradient_fn : callable
            Function that computes gradient: gradient_fn(params, x, y) -> gradient
        x_batch : np.ndarray
            Input data for this batch
        y_batch : np.ndarray
            Target data for this batch
        iteration : int
            Current iteration number (for learning rate decay)
            
        Returns
        -------
        np.ndarray
            Updated parameters
        """
        # Compute gradient
        grad = gradient_fn(params, x_batch, y_batch)
        
        # Apply learning rate decay if specified
        lr = self.config.learning_rate
        if self.config.decay_rate > 0:
            lr = lr / (1 + self.config.decay_rate * iteration)
        
        # Update parameters
        new_params = params - lr * grad
        
        return new_params
    
    def run_trajectory(
        self,
        initial_params: np.ndarray,
        gradient_fn: Callable,
        x_data: np.ndarray,
        y_data: np.ndarray,
        save_every: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a complete SGD trajectory.
        
        Parameters
        ----------
        initial_params : np.ndarray
            Starting parameter values
        gradient_fn : callable
            Gradient function
        x_data : np.ndarray
            Full input dataset
        y_data : np.ndarray
            Full target dataset
        save_every : int
            Save trajectory every N iterations
            
        Returns
        -------
        trajectory : np.ndarray
            Shape (n_saved, n_params) array of parameter values
        iterations : np.ndarray
            Iteration numbers for saved points
        """
        n_data = len(x_data)
        n_params = len(initial_params)
        n_iter = self.config.n_iterations
        batch_size = self.config.batch_size
        
        # Allocate trajectory array
        n_saved = n_iter // save_every + 1
        trajectory = np.zeros((n_saved, n_params))
        iterations = np.zeros(n_saved, dtype=int)
        
        # Initialize
        params = initial_params.copy()
        trajectory[0] = params
        iterations[0] = 0
        save_idx = 1
        
        # Run SGD
        for i in range(1, n_iter + 1):
            # Sample minibatch
            batch_indices = np.random.choice(n_data, batch_size, replace=False)
            x_batch = x_data[batch_indices]
            y_batch = y_data[batch_indices]
            
            # Update parameters
            params = self.step(params, gradient_fn, x_batch, y_batch, i)
            
            # Save if needed
            if i % save_every == 0:
                trajectory[save_idx] = params
                iterations[save_idx] = i
                save_idx += 1
        
        return trajectory[:save_idx], iterations[:save_idx]
    
    def run_ensemble(
        self,
        initial_params: np.ndarray,
        gradient_fn: Callable,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_trajectories: int = 100,
        save_every: int = 1
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run multiple SGD trajectories to study statistical properties.
        
        Parameters
        ----------
        initial_params : np.ndarray
            Starting parameter values (same for all trajectories)
        gradient_fn : callable
            Gradient function
        x_data : np.ndarray
            Full input dataset
        y_data : np.ndarray
            Full target dataset
        n_trajectories : int
            Number of independent trajectories to generate
        save_every : int
            Save trajectory every N iterations
            
        Returns
        -------
        list of tuples
            Each tuple contains (trajectory, iterations) for one run
        """
        trajectories = []
        
        for _ in range(n_trajectories):
            traj, iters = self.run_trajectory(
                initial_params, gradient_fn, x_data, y_data, save_every
            )
            trajectories.append((traj, iters))
        
        return trajectories


def sgd_step(
    params: np.ndarray,
    gradient_fn: Callable,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    learning_rate: float = 0.01
) -> np.ndarray:
    """
    Convenience function for a single SGD step.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameters
    gradient_fn : callable
        Gradient function: gradient_fn(params, x, y) -> gradient
    x_batch : np.ndarray
        Input batch
    y_batch : np.ndarray
        Target batch
    learning_rate : float
        Step size
        
    Returns
    -------
    np.ndarray
        Updated parameters
    """
    grad = gradient_fn(params, x_batch, y_batch)
    return params - learning_rate * grad


def sgd_trajectory(
    initial_params: np.ndarray,
    gradient_fn: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    batch_size: int = 1,
    save_every: int = 1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to run a single SGD trajectory.
    
    Parameters
    ----------
    initial_params : np.ndarray
        Starting parameters
    gradient_fn : callable
        Gradient function
    x_data, y_data : np.ndarray
        Training data
    n_iterations : int
        Number of SGD iterations
    learning_rate : float
        Step size
    batch_size : int
        Minibatch size
    save_every : int
        Save frequency
    random_state : int, optional
        Random seed
        
    Returns
    -------
    trajectory : np.ndarray
        Parameter trajectory
    iterations : np.ndarray
        Iteration numbers
    """
    config = SGDConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_iterations=n_iterations,
        random_state=random_state
    )
    simulator = SGDSimulator(config)
    return simulator.run_trajectory(initial_params, gradient_fn, x_data, y_data, save_every)
