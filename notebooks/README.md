# SGD Dynamics Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks that convert the original Mathematica notebooks to Python, exploring Stochastic Gradient Descent (SGD) from a statistical physics perspective.

## Overview

These notebooks study SGD as a stochastic differential equation (SDE), connecting optimization dynamics to Langevin dynamics, fluctuation-dissipation relations, and escape time theory from statistical physics.

## Notebooks

### [01_sgd_basics.ipynb](01_sgd_basics.ipynb)
**Basic SGD Implementation** (Converted from `sgd_example.nb`)

- Data generation with Gaussian noise
- Loss landscape visualization (2D and 3D)
- SGD trajectory tracking on loss landscapes
- Parameter evolution over time
- Multiple trajectories from different initializations

**Key Concepts:** SGD fundamentals, loss landscapes, optimization dynamics, convergence

---

### [02_fluctuation_dissipation.ipynb](02_fluctuation_dissipation.ipynb)
**Fluctuation-Dissipation Relation** (Converted from `sgd_example(2).nb`)

- Connection between SGD and Langevin dynamics
- Effective diffusion coefficient calculation
- Detailed balance analysis
- Stationary distribution (Boltzmann distribution)
- Temperature effects (learning rate as temperature)

**Key Concepts:** Langevin equation, thermal equilibrium, Boltzmann distribution, fluctuation-dissipation theorem

**Mathematical Foundation:**
```
dθ = -∇L(θ)dt + √(2D)dW
D ≈ (η²/2) · (N-b)/(N-1) · Var[∇L]
p_stat(θ) ∝ exp(-L(θ)/D)
```

---

### [03_sgd_sampling_escape_times.ipynb](03_sgd_sampling_escape_times.ipynb)
**Ensemble Analysis and Escape Times** (Converted from `sgd_example(2)_sampling.nb` and `SDE_escapetime.nb`)

- Multiple trajectory generation (ensemble statistics)
- Escape time calculations from local minima
- Kramers formula verification
- Learning rate dependence of escape times
- Barrier crossing visualization

**Key Concepts:** Kramers theory, activated escape, barrier crossing, ensemble averaging

**Kramers Formula:**
```
τ_escape ∝ (1/ω) · exp(ΔE/D)
```
where ΔE is barrier height, D is diffusion coefficient, ω is attempt frequency.

---

### [04_smooth_approximations.ipynb](04_smooth_approximations.ipynb)
**Smooth Approximations** (Converted from `sgd_example_2.nb` and `Danilo_piecewiselinlosses.nb`)

- Piecewise linear loss functions
- Smooth sigmoid approximations
- Comparison of optimization dynamics
- Effect of smoothness on convergence
- Gradient continuity analysis

**Key Concepts:** Piecewise linear functions, smooth approximations, activation functions, loss landscape geometry

**Smooth Approximation:**
```
f(x; a, b) = x(1 + b/(1 + e^(-ax)))
```
As a → ∞, approaches piecewise linear behavior.

---

### [05_sde_escape_times.ipynb](05_sde_escape_times.ipynb)
**SDE Escape Time Analysis** (Converted from `SDE_escapetime.nb` and `2d_ito_diff_graph.nb`)

- 2D Ito process simulation
- Double-well potential dynamics
- Systematic escape time measurements
- Kramers formula verification in 2D
- Barrier height dependence
- Connection to SGD learning rate

**Key Concepts:** Ito processes, 2D SDEs, double-well potentials, systematic escape time analysis

**2D SDE:**
```
dx = f(x)dt + G·dW
```

## Getting Started

### Prerequisites

All required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to the notebooks directory**

3. **Run notebooks in order (recommended):**
   - Start with `01_sgd_basics.ipynb` for foundations
   - Progress through `02` → `03` → `04` → `05` for complete understanding

Each notebook is self-contained but builds conceptually on previous ones.

## Utilities

The notebooks import from the `utils/` directory:

- **`sgd_simulator.py`**: SGD implementation and trajectory generation
- **`loss_functions.py`**: Various loss functions with gradients
- **`visualization.py`**: Plotting utilities for trajectories and landscapes
- **`sde_tools.py`**: SDE simulation and Kramers theory calculations

## Mathematical Background

### Core Equation

SGD can be viewed as a discretization of the Langevin equation:

```
θ_{t+1} = θ_t - η∇L(θ_t; B_t)
        ≈ θ_t - η∇L(θ_t) + √(2D)ξ_t
```

where:
- `η` = learning rate
- `B_t` = minibatch at time t
- `D` = effective diffusion coefficient
- `ξ_t` = Gaussian noise

### Key Results

1. **Stationary Distribution:**
   ```
   p_stat(θ) ∝ exp(-L(θ)/D)
   ```

2. **Escape Time (Kramers):**
   ```
   τ ∝ exp(ΔE/D) where D ∝ η²
   ```

3. **Practical Implication:**
   ```
   Small η → trapped in local minima
   Large η → fast exploration but poor convergence
   Optimal η → balance escape and convergence
   ```

## Key Insights

1. **SGD is a Sampling Algorithm**: Not just an optimizer, SGD samples from a distribution related to the loss landscape

2. **Learning Rate as Temperature**: Higher learning rates → more exploration (like higher temperature in physics)

3. **Noise Enables Escape**: Stochastic gradient noise allows escape from bad local minima

4. **Kramers Formula Applies**: Escape times follow exponential dependence on barrier height divided by "temperature"

5. **Flat Minima Generalize Better**: SGD naturally finds flatter minima due to noise-driven dynamics

## References

These notebooks implement concepts from:

- **Statistical Physics**: Langevin dynamics, Kramers theory, Fokker-Planck equations
- **Stochastic Processes**: Ito calculus, stochastic differential equations
- **Machine Learning**: SGD optimization, loss landscapes, generalization

## Original Sources

Converted from Mathematica notebooks:
- `sgd_example.nb`
- `sgd_example(2).nb`
- `sgd_example(2)_sampling.nb`
- `sgd_example_2.nb`
- `SDE_escapetime.nb`
- `2d_ito_diff_graph.nb`
- `Danilo_piecewiselinlosses.nb`

## Contributing

When extending these notebooks:
1. Maintain the connection to statistical physics concepts
2. Include mathematical derivations where relevant
3. Use visualization utilities from `utils/`
4. Add markdown explanations for new concepts

## License

Part of the "Simulation of SGD Dynamics" Bachelor thesis project.

## Contact

For questions about the implementation or theory, please refer to the main repository README.

---

**Happy Exploring! 🚀**

*Understanding SGD through the lens of statistical physics reveals deep connections between optimization, thermodynamics, and stochastic processes.*
