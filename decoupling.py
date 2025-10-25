# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:03:36 2024

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


# Define the loss function
def loss(a, b):
    return sum(sum((ys[i][1][j] - f_mod(ys[i][0], a, b))**2 for j in range(ll)) for i in range(nn))

# Generate points for plotting
a_values = np.linspace(-1.5, 2, 100)
b_values = np.linspace(-5, 5, 100)
A, B = np.meshgrid(a_values, b_values)
Z = np.array([[loss(a, b) for a, b in zip(a_row, b_row)] for a_row, b_row in zip(A, B)])



a1_values = np.linspace(-1, 2, 100)
b1_values = np.linspace(-1, 2, 100)
A1, B1 = np.meshgrid(a_values, b_values)

# Define the absolute gradient function
def absg(a, b):
    loss_gradient_a = np.gradient(Z, a1_values, axis=0)
    loss_gradient_b = np.gradient(Z, b1_values, axis=1)
    gradient_abs = np.sqrt(loss_gradient_a**2 + loss_gradient_b**2)
    return gradient_abs



################ Verifica del decoupling 
a2_values = np.linspace(-1.5, 2, 100)
b2_values = np.linspace(-5, 5, 100)
x2_values =  np.linspace(-6, 6, 100)
X2, A2, B2 = np.meshgrid(x2_values, a2_values, b2_values)



Z2= np.array([[f_mod(x,a, b) for x, a, b in zip(x_row, a_row, b_row)] for x_row, a_row, b_row in zip(X2, A2, B2)])


v= [np.gradient(Z2, a2_values, axis=1),np.gradient(Z2, b2_values, axis=2)]

def coupled(a,b):
    return sum(sum(((ys[i][1][j] - f_mod(ys[i][0], a, b))**2)*np.matmul(v,np.transpose(v)) for j in range(ll)) for i in range(nn))




def l_mu(x,a,b,labels):
    return 0.5*(f_mod(x, a, b)-labels)**2


def C_mu(x,a,b):
    return np.matmul([],[])

