"""
Visualization utilities for SGD dynamics.

Provides plotting functions for trajectories, loss landscapes,
escape times, and statistical distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List, Tuple, Optional
import seaborn as sns


def plot_trajectories(
    trajectories: List[np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "SGD Trajectories",
    labels: Optional[List[str]] = None,
    show_start: bool = True,
    show_end: bool = True,
    alpha: float = 0.7
) -> plt.Axes:
    """
    Plot 2D parameter trajectories.
    
    Parameters
    ----------
    trajectories : list of np.ndarray
        List of trajectory arrays, each shape (n_steps, 2)
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str
        Plot title
    labels : list of str, optional
        Labels for each trajectory
    show_start : bool
        Mark starting points
    show_end : bool
        Mark ending points
    alpha : float
        Transparency of trajectory lines
        
    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, traj in enumerate(trajectories):
        label = labels[i] if labels is not None and i < len(labels) else f"Trajectory {i+1}"
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], '-', alpha=alpha, label=label, linewidth=1.5)
        
        # Mark start and end
        if show_start:
            ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=8, 
                   markeredgecolor='black', markerfacecolor='green')
        if show_end:
            ax.plot(traj[-1, 0], traj[-1, 1], 's', markersize=8,
                   markeredgecolor='black', markerfacecolor='red')
    
    ax.set_xlabel('Parameter a', fontsize=12)
    ax.set_ylabel('Parameter b', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


def plot_loss_landscape(
    loss_fn: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    param_range: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 100,
    trajectory: Optional[np.ndarray] = None,
    contour_levels: int = 30,
    ax: Optional[plt.Axes] = None,
    title: str = "Loss Landscape"
) -> plt.Axes:
    """
    Plot 2D loss landscape with optional trajectory overlay.
    
    Parameters
    ----------
    loss_fn : callable
        Loss function: loss_fn(params, x, y) -> scalar
    x_data, y_data : np.ndarray
        Training data
    param_range : tuple of tuples
        ((a_min, a_max), (b_min, b_max))
    n_points : int
        Grid resolution
    trajectory : np.ndarray, optional
        Trajectory to overlay, shape (n_steps, 2)
    contour_levels : int
        Number of contour levels
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
        
    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid
    (a_min, a_max), (b_min, b_max) = param_range
    a_vals = np.linspace(a_min, a_max, n_points)
    b_vals = np.linspace(b_min, b_max, n_points)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # Compute loss on grid
    Z = np.zeros_like(A)
    for i in range(n_points):
        for j in range(n_points):
            params = np.array([A[i, j], B[i, j]])
            Z[i, j] = loss_fn(params, x_data, y_data)
    
    # Plot contours
    contour = ax.contour(A, B, Z, levels=contour_levels, cmap='viridis', alpha=0.6)
    contourf = ax.contourf(A, B, Z, levels=contour_levels, cmap='viridis', alpha=0.4)
    plt.colorbar(contourf, ax=ax, label='Loss')
    
    # Overlay trajectory if provided
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, 
               label='SGD Trajectory', alpha=0.8)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, 
               label='Start', markeredgecolor='black')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=10,
               label='End', markeredgecolor='black')
        ax.legend()
    
    ax.set_xlabel('Parameter a', fontsize=12)
    ax.set_ylabel('Parameter b', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_loss_landscape_3d(
    loss_fn: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    param_range: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 50,
    trajectory: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None,
    title: str = "3D Loss Landscape"
) -> Tuple[plt.Figure, Axes3D]:
    """
    Plot 3D loss landscape.
    
    Parameters
    ----------
    loss_fn : callable
        Loss function
    x_data, y_data : np.ndarray
        Training data
    param_range : tuple of tuples
        Parameter ranges
    n_points : int
        Grid resolution
    trajectory : np.ndarray, optional
        Trajectory to overlay
    fig : plt.Figure, optional
        Figure to plot on
    title : str
        Plot title
        
    Returns
    -------
    fig : plt.Figure
        Figure object
    ax : Axes3D
        3D axes object
    """
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    (a_min, a_max), (b_min, b_max) = param_range
    a_vals = np.linspace(a_min, a_max, n_points)
    b_vals = np.linspace(b_min, b_max, n_points)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # Compute loss
    Z = np.zeros_like(A)
    for i in range(n_points):
        for j in range(n_points):
            params = np.array([A[i, j], B[i, j]])
            Z[i, j] = loss_fn(params, x_data, y_data)
    
    # Plot surface
    surf = ax.plot_surface(A, B, Z, cmap=cm.viridis, alpha=0.7, 
                          linewidth=0, antialiased=True)
    
    # Overlay trajectory if provided
    if trajectory is not None:
        # Compute loss along trajectory
        traj_loss = np.array([loss_fn(p, x_data, y_data) for p in trajectory])
        ax.plot(trajectory[:, 0], trajectory[:, 1], traj_loss, 'r-', 
               linewidth=3, label='SGD Trajectory')
        ax.plot([trajectory[0, 0]], [trajectory[0, 1]], [traj_loss[0]], 
               'go', markersize=10, label='Start')
        ax.plot([trajectory[-1, 0]], [trajectory[-1, 1]], [traj_loss[-1]], 
               'rs', markersize=10, label='End')
        ax.legend()
    
    ax.set_xlabel('Parameter a', fontsize=11)
    ax.set_ylabel('Parameter b', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def plot_escape_times(
    learning_rates: np.ndarray,
    escape_times: np.ndarray,
    theoretical_curve: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Escape Time vs Learning Rate",
    log_scale: bool = True
) -> plt.Axes:
    """
    Plot escape times as a function of learning rate.
    
    Parameters
    ----------
    learning_rates : np.ndarray
        Learning rate values
    escape_times : np.ndarray
        Empirical escape times
    theoretical_curve : np.ndarray, optional
        Theoretical prediction (e.g., from Kramers formula)
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
    log_scale : bool
        Use log scale for both axes
        
    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot empirical data
    ax.plot(learning_rates, escape_times, 'o-', markersize=8, 
           linewidth=2, label='Empirical', color='blue')
    
    # Plot theoretical curve if provided
    if theoretical_curve is not None:
        ax.plot(learning_rates, theoretical_curve, '--', linewidth=2,
               label='Theoretical (Kramers)', color='red')
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Escape Time (iterations)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    return ax


def plot_stationary_distribution(
    samples: np.ndarray,
    param_range: Tuple[Tuple[float, float], Tuple[float, float]],
    theoretical_dist: Optional[Callable] = None,
    n_bins: int = 50,
    fig: Optional[plt.Figure] = None,
    title: str = "Stationary Distribution"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot 2D stationary distribution from samples.
    
    Parameters
    ----------
    samples : np.ndarray
        Sample points, shape (n_samples, 2)
    param_range : tuple of tuples
        ((a_min, a_max), (b_min, b_max))
    theoretical_dist : callable, optional
        Theoretical distribution function
    n_bins : int
        Number of histogram bins
    fig : plt.Figure, optional
        Figure to plot on
    title : str
        Plot title
        
    Returns
    -------
    fig : plt.Figure
        Figure object
    axes : list of plt.Axes
        List of axes objects
    """
    if fig is None:
        fig = plt.figure(figsize=(16, 5))
    
    # 2D histogram
    ax1 = fig.add_subplot(131)
    (a_min, a_max), (b_min, b_max) = param_range
    
    h, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1], 
        bins=n_bins, 
        range=[[a_min, a_max], [b_min, b_max]]
    )
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax1.imshow(h.T, origin='lower', extent=extent, aspect='auto', cmap='hot')
    ax1.set_xlabel('Parameter a', fontsize=11)
    ax1.set_ylabel('Parameter b', fontsize=11)
    ax1.set_title('2D Histogram', fontsize=12)
    plt.colorbar(im, ax=ax1)
    
    # Marginal distribution for parameter a
    ax2 = fig.add_subplot(132)
    ax2.hist(samples[:, 0], bins=n_bins, density=True, alpha=0.7, color='blue', label='Empirical')
    ax2.set_xlabel('Parameter a', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Marginal Distribution (a)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Marginal distribution for parameter b
    ax3 = fig.add_subplot(133)
    ax3.hist(samples[:, 1], bins=n_bins, density=True, alpha=0.7, color='green', label='Empirical')
    ax3.set_xlabel('Parameter b', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Marginal Distribution (b)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, [ax1, ax2, ax3]


def plot_parameter_evolution(
    iterations: np.ndarray,
    trajectory: np.ndarray,
    param_names: List[str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Parameter Evolution"
) -> plt.Axes:
    """
    Plot parameter values over time.
    
    Parameters
    ----------
    iterations : np.ndarray
        Iteration numbers
    trajectory : np.ndarray
        Parameter trajectory, shape (n_steps, n_params)
    param_names : list of str, optional
        Names for each parameter
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
        
    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    n_params = trajectory.shape[1]
    if param_names is None:
        param_names = [f'Parameter {i+1}' for i in range(n_params)]
    
    for i in range(n_params):
        ax.plot(iterations, trajectory[:, i], '-', linewidth=2, 
               label=param_names[i], alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return ax
