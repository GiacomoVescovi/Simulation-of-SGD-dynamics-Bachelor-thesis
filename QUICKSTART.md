# Quick Start Guide

Welcome! This guide will get you running the SGD dynamics notebooks in 5 minutes.

## 🚀 Fast Track (3 Steps)

### 1. Install Dependencies
```bash
pip install numpy scipy matplotlib seaborn jupyter
```

### 2. Start Jupyter
```bash
jupyter notebook
```

### 3. Open First Notebook
Navigate to `notebooks/01_sgd_basics.ipynb` and run it!

## 📖 What You'll Learn

The notebooks cover:
- **SGD Basics**: How stochastic gradient descent works
- **Statistical Physics**: Connection to Langevin dynamics
- **Escape Times**: Kramers theory and barrier crossing
- **Visualization**: Beautiful 2D/3D loss landscapes

## 🎯 Recommended Path

**For Beginners:**
1. `01_sgd_basics.ipynb` - Start here!
2. `04_smooth_approximations.ipynb` - Simple extensions
3. `08_trajectory_simulations.ipynb` - Multiple runs

**For ML Practitioners:**
1. `01_sgd_basics.ipynb` - Basics
2. `02_fluctuation_dissipation.ipynb` - Theory
3. `03_sgd_sampling_escape_times.ipynb` - Advanced

**For Physics Students:**
1. `02_fluctuation_dissipation.ipynb` - Langevin connection
2. `05_sde_escape_times.ipynb` - SDEs
3. `06_stationary_distributions.ipynb` - Fokker-Planck

## 💡 Key Concepts

**SGD = Langevin Dynamics**
```
dx/dt = -∇L(x) + √(2D) η(t)
```

**Kramers Formula**
```
escape_time ~ exp(barrier_height / temperature)
```

**Stationary Distribution**
```
p(x) ∝ exp(-L(x) / D)
```

## 🛠️ Troubleshooting

**"Module not found"?**
```bash
pip install -r requirements.txt
```

**Plots not showing?**
Add this to notebook cells:
```python
%matplotlib inline
```

**Kernel issues?**
```bash
python -m ipykernel install --user
```

## 📚 More Information

- Full setup guide: `SETUP.md`
- Repository README: `README.md`
- Notebook details: `notebooks/README.md`

## 🤝 Need Help?

1. Check `SETUP.md` for detailed installation
2. Read individual notebook introductions
3. Open an issue on GitHub

## 🎉 Ready to Explore!

Start with `notebooks/01_sgd_basics.ipynb` and enjoy exploring SGD dynamics!

---

**From Mathematica to Python**: This is a complete Python conversion of the original Mathematica thesis project, making cutting-edge SGD research accessible to everyone.
