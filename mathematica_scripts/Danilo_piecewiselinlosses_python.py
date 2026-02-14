# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:49:25 2024

@author: jackv
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the true and modified functions
p = 1
def f_true(x):
    return x * (1 + p * (1 + np.exp(-x))**(-1))

def f_mod(x, a, b):
    return x * (1 + b * (1 + np.exp(-a * x))**(-1))

# Generate sample from f_true
nn = 20
ll = 7
xs = np.arange(1, nn + 1) - 11
fi = [f_true(x) for x in xs]
eps = 4
epss = np.random.uniform(-eps, eps, ll * len(fi))
ys = [(xs[i], [fi[i] + epss[i + j] for j in range(ll)]) for i in range(len(fi))]

# Plot the samples
plt.figure(figsize=(8, 6))
for j in range(ll):
    plt.plot([0.5 * i - 5.5 for i in range(1, nn + 1)], [ys[i][1][j] for i in range(nn)], 'o')
plt.xlabel('x')
plt.ylabel('f(x) + noise')
plt.title('Samples')
plt.grid(True)
plt.show()

# randomly divide the samples into minibatches#########################

import random

def divide_indices(indices, num_subsets):
    # Calculate the size of each subset
    subset_size = len(indices) // num_subsets
    remaining_indices = indices.copy()
    subsets = []

    # Distribute indices into subsets
    for _ in range(num_subsets - 1):
        subset = random.sample(remaining_indices, subset_size)
        subsets.append(subset)
        remaining_indices = [idx for idx in remaining_indices if idx not in subset]

    # Add remaining indices to the last subset
    subsets.append(remaining_indices)

    return subsets





# # Example usage:
# indices_list = [i for i in range(20)]  # Example indices list
# num_subsets = 4  # Number of subsets
# subsets = divide_indices(indices_list, num_subsets)
# print("Subsets:", subsets)

##################################










# Define the loss function
def loss(a, b):
    return sum(sum((ys[i][1][j] - f_mod(ys[i][0], a, b))**2 for j in range(ll)) for i in range(nn))


# Define the loss function on a minibatch
def loss(a, b):
    return sum(sum((ys[i][1][j] - f_mod(ys[i][0], a, b))**2 for j in range(ll)) for i in range(nn))

# Generate points for plotting
a_values = np.linspace(-1.5, 2, 100)
b_values = np.linspace(-5, 5, 100)
A, B = np.meshgrid(a_values, b_values)
Z = np.array([[loss(a, b) for a, b in zip(a_row, b_row)] for a_row, b_row in zip(A, B)])

# Plot the loss function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, Z, cmap='viridis')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.set_title('Loss Function')
plt.show()



a1_values = np.linspace(-1, 2, 100)
b1_values = np.linspace(-1, 2, 100)
A1, B1 = np.meshgrid(a_values, b_values)

# Define the absolute gradient function
def absg(a, b):
    loss_gradient_a = np.gradient(Z, a1_values, axis=0)
    loss_gradient_b = np.gradient(Z, b1_values, axis=1)
    gradient_abs = np.sqrt(loss_gradient_a**2 + loss_gradient_b**2)
    return gradient_abs





# Plot the norm of the gradient in 3D
Z1 = absg(A1,B1)


z=np.linspace(-0, 180000,100)
X_0, Z_0 = np.meshgrid(a_values, z)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A1, B1, Z1, cmap='viridis')
# piano che prende una sezione; ax.plot_surface(X_0, 1, Z_0,cmap='plasma' )
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Grad')
ax.set_title('Gradient norm')
plt.show()

################ Verifica del decoupling 












#### sotto compila a parte, ma non serve e compila male con il resto del codice, uso per eventuale futuro


# b_section=np.array([0,1.03])

# gradient_abs_b_0 = absg(a1_values,b_section[0] )

# #Plot the one-dimensional projections

# plt.plot(a1_values, gradient_abs_b_0[33], label='b=0')
# plt.plot(a1_values, gradient_abs_b_0[66], label='b=1.03')
# plt.xlabel('a')
# plt.ylabel('abs loss gradient')
# plt.legend()
# plt.title('One-Dimensional Projections of Absolute Loss Gradient')
# plt.show()





