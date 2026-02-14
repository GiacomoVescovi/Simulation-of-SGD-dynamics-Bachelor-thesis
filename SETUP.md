# Setup and Installation Guide

This guide will help you set up the environment to run the SGD dynamics simulation notebooks.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GiacomoVescovi/Simulation-of-SGD-dynamics-Bachelor-thesis.git
cd Simulation-of-SGD-dynamics-Bachelor-thesis
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to avoid conflicts with other Python packages.

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- NumPy (numerical computing)
- SciPy (scientific computing)
- Matplotlib (plotting)
- Seaborn (statistical visualization)
- Jupyter (interactive notebooks)
- sdeint (SDE integration)
- pandas (data manipulation)
- tqdm (progress bars)

### 4. Verify Installation

Test that the installation was successful:

```bash
python -c "import numpy, scipy, matplotlib, jupyter; print('All packages installed successfully!')"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to the `notebooks/` directory and open any notebook.

## Recommended Learning Path

If you're new to this project, follow this sequence:

1. **01_sgd_basics.ipynb** - Start here to understand basic SGD
2. **02_fluctuation_dissipation.ipynb** - Learn the physics connection
3. **03_sgd_sampling_escape_times.ipynb** - Understand statistical properties
4. **05_sde_escape_times.ipynb** - Dive into SDE theory
5. Continue with other notebooks based on your interests

## Troubleshooting

### Import Errors

If you get import errors when running notebooks, make sure:
1. You're running Jupyter from the repository root directory
2. The virtual environment is activated
3. All dependencies are installed

### Kernel Issues

If Jupyter can't find the Python kernel:
```bash
python -m ipykernel install --user --name=venv
```

Then select the "venv" kernel in Jupyter.

### Memory Issues

Some notebooks generate many trajectories. If you run out of memory:
- Reduce `n_trajectories` in ensemble simulations
- Reduce `n_points` in grid-based visualizations
- Close other applications

### Slow Performance

To speed up computations:
- Reduce `n_iterations` for SGD simulations
- Use smaller grid sizes for loss landscapes
- Run fewer parallel trajectories

## Advanced Setup

### Using conda

If you prefer conda:

```bash
conda create -n sgd-dynamics python=3.10
conda activate sgd-dynamics
pip install -r requirements.txt
```

### GPU Acceleration

For faster computations (optional):
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

Modify notebook code to use CuPy instead of NumPy for array operations.

### Development Setup

If you want to modify the utility modules:

```bash
pip install -e .  # Install in editable mode
```

This allows you to edit `utils/` modules without reinstalling.

## Running Tests

To verify your setup works correctly:

```bash
# Test basic imports
python -c "from utils.sgd_simulator import SGDSimulator; print('Import successful!')"

# Test notebook execution (requires nbconvert)
pip install nbconvert
jupyter nbconvert --to notebook --execute notebooks/01_sgd_basics.ipynb
```

## Additional Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Jupyter Shortcuts**: Press `H` in Jupyter for keyboard shortcuts
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the notebook README: `notebooks/README.md`
3. Open an issue on GitHub
4. Check Python/package documentation

## Next Steps

Once setup is complete:
1. Open `notebooks/01_sgd_basics.ipynb`
2. Run all cells (`Cell -> Run All`)
3. Explore the visualizations
4. Modify parameters to see how results change
5. Move on to more advanced notebooks

Happy exploring SGD dynamics! 🚀
