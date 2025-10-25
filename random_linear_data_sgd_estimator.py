# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:29:30 2023

@author: jackv
"""

import numpy as np


import matplotlib.pyplot as plt


# Create independent variable 
x = np.arange(0,200,0.5) # Produces [0, 100) with steps of 2.

# Use a linear function to obtain the dependent variable 
y = 3.0000*x + 0.60000 # Parameters are arbitrary. 

# Noise generation

# Genearte noise with same size as that of the data.
noise = np.random.normal(0,2, len(x)) #  μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.

# Add the noise to the data. 
y_noised = y + noise 

#plt.plot (x, y_noised, 'ro')

############

def ssr_gradient(x, y, b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()


def sgd(
    gradient, x, y, n_vars=None, start=None, learn_rate=0.1,
    decay_rate=0.0, batch_size=1, n_iter=50, tolerance=1e-06,
    dtype="float64", random_state=None
):
    # Checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    # Setting up the data type for NumPy arrays
    dtype_ = np.dtype(dtype)

    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # Initializing the random number generator
    seed = None if random_state is None else int(random_state)
    
    rng = np.random.default_rng(seed)

    # Initializing the values of the variables
    vector = (
        rng.normal(size=int(n_vars)).astype(dtype_)
        if start is None else
        np.array(start, dtype=dtype_)
    )

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the decay rate
    decay_rate = np.array(decay_rate, dtype=dtype_)
    if np.any(decay_rate < 0) or np.any(decay_rate > 1):
        raise ValueError("'decay_rate' must be between zero and one")

    # Setting up and checking the size of minibatches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError(
            "'batch_size' must be greater than zero and less than "
            "or equal to the number of observations"
        )

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Setting the difference to zero for the first iteration
    diff = 0
    # Initialize empty lists for x and y coordinates
    x_coordinates = []
    y_coordinates = []
    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Shuffle x and y
        rng.shuffle(xy)
        #append values for the graph
        x_coordinates.append(_)
        y_coordinates.append(vector[1])

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            #print(x_batch)
            #print(y_batch)
            #print('xxx')
            break
            # Recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
            print(grad[1])
            diff = decay_rate * diff - learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff
    
    plt.scatter(x_coordinates, y_coordinates, label='Points')
    return vector if vector.shape else vector.item()




results=sgd(ssr_gradient, x, y_noised, n_vars=2, learn_rate=0.0001,decay_rate=0.8, batch_size=3, n_iter=1000, random_state=0 )
#print(results)