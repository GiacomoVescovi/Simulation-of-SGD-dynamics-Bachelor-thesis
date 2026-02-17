# Simulation of SGD Dynamics - Bachelor Thesis

A comprehensive Python implementation for simulating and analyzing **Stochastic Gradient Descent (SGD) dynamics** as stochastic differential equations. This repository provides tools and Jupyter notebooks for studying SGD convergence, escape times, and statistical properties.

## 🎓 About This Project

This project is a **Python/Jupyter conversion** of a Bachelor thesis originally implemented in Mathematica. The original work focused on:
- Modeling SGD as Langevin dynamics and stochastic differential equations (SDEs)
- Analyzing escape times and transition rates between local minima
- Verifying fluctuation-dissipation relations and detailed balance conditions
- Comparing theoretical predictions (Kramers formula, Fokker-Planck) with empirical results

**Original Implementation:** This project was originally developed in Mathematica as part of a Bachelor thesis on SGD dynamics. The code has been completely converted to Python with Jupyter notebooks for improved accessibility and reproducibility.

**Acknowledgments:** Code related to piecewise linear losses was provided by Dr. Danilo Forastiere as a foundational step in the analysis.

## 📚 Repository Structure

```
.
├── notebooks/              # Jupyter notebooks (main analysis)
│   ├── 01_sgd_basics.ipynb
│   ├── 02_fluctuation_dissipation.ipynb
│   ├── 03_sgd_sampling_escape_times.ipynb
│   ├── 04_smooth_approximations.ipynb
│   ├── 05_sde_escape_times.ipynb
│   ├── 06_stationary_distributions.ipynb
│   ├── 07_2d_ito_visualization.ipynb
│   ├── 08_trajectory_simulations.ipynb
│   ├── 09_trajectory_simulations_v2.ipynb
│   ├── 10_multidimensional_examples.ipynb
│   └── 11_piecewise_linear_losses.ipynb
├── utils/                  # Python utility modules
│   ├── sgd_simulator.py   # Core SGD implementation
│   ├── loss_functions.py  # Loss functions and gradients
│   ├── visualization.py   # Plotting utilities
│   └── sde_tools.py       # SDE and Langevin dynamics
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GiacomoVescovi/Simulation-of-SGD-dynamics.git
   cd Simulation-of-SGD-dynamics-Bachelor-thesis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Then navigate to the `notebooks/` directory and open any notebook. We recommend starting with:
1. **01_sgd_basics.ipynb** - Introduction to SGD dynamics
2. **02_fluctuation_dissipation.ipynb** - Connection to statistical physics
3. **03_sgd_sampling_escape_times.ipynb** - Ensemble analysis

## 📖 Notebook Overview

### Core SGD Analysis
- **01_sgd_basics.ipynb**: Basic SGD implementation, loss landscapes, trajectory visualization
- **02_fluctuation_dissipation.ipynb**: Fluctuation-dissipation theorem, Langevin dynamics connection
- **03_sgd_sampling_escape_times.ipynb**: Ensemble trajectories, escape time analysis, Kramers formula
- **04_smooth_approximations.ipynb**: Smooth approximations of piecewise functions

### SDE and Dynamics
- **05_sde_escape_times.ipynb**: SDE escape time analysis, learning rate dependence
- **06_stationary_distributions.ipynb**: Fokker-Planck equation, equilibrium distributions
- **07_2d_ito_visualization.ipynb**: 2D Ito processes, phase space visualization

### Trajectory Analysis
- **08_trajectory_simulations.ipynb**: Multiple trajectory generation and analysis
- **09_trajectory_simulations_v2.ipynb**: Enhanced trajectory simulations
- **10_multidimensional_examples.ipynb**: Multi-dimensional SGD dynamics (2D/3D)

### Specialized Methods
- **11_piecewise_linear_losses.ipynb**: Piecewise linear losses and smooth approximations

## 🔬 Key Concepts

### SGD as Langevin Dynamics
SGD can be modeled as a discrete-time approximation to the Langevin equation:
```
dx/dt = -∇L(x) + √(2D) η(t)
```
where:
- `L(x)` is the loss function
- `D` is the effective diffusion coefficient (related to learning rate and batch size)
- `η(t)` is Gaussian white noise

### Kramers Escape Rate
The mean escape time from a local minimum is given by:
```
τ = (2π / ω_min) exp(ΔE / D)
```
where:
- `ΔE` is the barrier height
- `ω_min` is the curvature at the minimum
- `D` is the diffusion coefficient

### Stationary Distribution
At equilibrium, SGD samples from:
```
p(x) ∝ exp(-L(x) / D)
```
This connects SGD to simulated annealing and statistical physics.

## 🛠️ Utility Modules

### `sgd_simulator.py`
- `SGDSimulator`: Main class for running SGD simulations
- `sgd_trajectory()`: Generate single trajectories
- `run_ensemble()`: Generate multiple trajectories for statistical analysis

### `loss_functions.py`
- Smooth nonlinear loss functions
- Piecewise linear losses
- Gradient computations (analytic and numerical)
- Data generation utilities

### `visualization.py`
- Loss landscape plotting (2D and 3D)
- Trajectory visualization
- Escape time plots
- Stationary distribution visualization

### `sde_tools.py`
- Langevin dynamics simulation
- Ito process integration
- Kramers rate calculations
- Fokker-Planck stationary distributions

## 📊 Example Results

The notebooks demonstrate:
- **SGD convergence** from multiple initializations
- **Escape time scaling** with learning rate
- **Fluctuation-dissipation** relation verification
- **Stationary distribution** sampling
- **Multi-modal optimization** landscape navigation

## 🔧 Technical Details

### Dependencies
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing, optimization
- **Matplotlib**: Visualization
- **Seaborn**: Statistical plots
- **Jupyter**: Interactive notebooks
- **sdeint**: SDE integration (optional)

### Numerical Methods
- **Euler-Maruyama**: SDE integration
- **Finite differences**: Gradient approximation
- **Monte Carlo**: Ensemble averaging
- **Histogram estimation**: Distribution approximation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{vescovi2024sgd,
  author = {Vescovi, Giacomo},
  title = {Simulation of SGD Dynamics},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GiacomoVescovi/Simulation-of-SGD-dynamics-Bachelor-thesis}},
  note = {Python conversion of Mathematica implementation. Original analysis contributed by Dr. Danilo Forastiere.}
}
```

## 📚 References

1. **Langevin Dynamics**: Gardiner, C. W. (1985). Handbook of stochastic methods.
2. **Kramers Theory**: Hänggi, P., Talkner, P., & Borkovec, M. (1990). Reaction-rate theory: fifty years after Kramers.
3. **SGD Theory**: Mandt, S., Hoffman, M. D., & Blei, D. M. (2017). Stochastic gradient descent as approximate Bayesian inference.
4. **Fokker-Planck**: Risken, H. (1996). The Fokker-Planck equation.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Add new notebooks or analyses
- Improve documentation

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

## 👤 Author

**Giacomo Vescovi**
- GitHub: [@GiacomoVescovi](https://github.com/GiacomoVescovi)

**Original Contributions:**
- Dr. Danilo Forastiere (piecewise linear loss analysis)

## 🙏 Acknowledgments

- Original Mathematica implementation as part of Bachelor thesis research
- Dr. Danilo Forastiere for foundational code and guidance
- Statistical physics and stochastic processes community

---

**Note**: This is a research project for educational purposes. The code is provided as-is for studying SGD dynamics and stochastic optimization.
