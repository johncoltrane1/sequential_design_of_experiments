import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc
import imse_mixed_variables
import sys
import itertools
import json


assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

# Options
n_grid = 10000
n_particles = 1000

n_runs = 10

num_new = 1

# Fetch variables
with open("variables.json", "r") as f:
    _variables = json.load(f)

variables = []

for _variable in _variables:
    if _variable[0] == "l":
        variables.append(_variable[1])
    elif _variable[0] == "t":
        variables.append(tuple(_variable[1]))
    else:
        raise ValueError(_variable)

# Build model
model = gpc.Model_ConstantMeanMaternpML(
    "GP_seq_dim",
    output_dim=1,
    covariance_params={"p": 0},
    rng=np.random.default_rng(),
    box=None
)

# design
xi = np.load("xi.npy")
zi = np.load("zi.npy")

for j in range(xi.shape[1]):
    for i in range(xi.shape[0]):
        if isinstance(variables[i], tuple):
            assert xi[i, j] in variables[i]
        elif isinstance(variables[i], list):
            assert variables[i][0] <= xi[i, j] and xi[i, j] <= variables[i][1]
        else:
            raise ValueError("{}, {}, {}".format(i, j, xi[i, j]))

# grid
grid_list = []

for variable in variables:
    if isinstance(variable, list):
        _grid = (variable[1] - variable[0]) * np.random.uniform(size=n_grid) + variable[0]
    elif isinstance(variable, tuple):
        _grid = np.random.choice(variable, size=n_grid)
    else:
        raise ValueError(variable)

    grid_list.append(_grid.reshape(-1, 1))

grid = np.hstack(grid_list)

algo = imse_mixed_variables.IMSE_MIXED_VARIABLES(variables, n_particles, grid, xi, zi, model)

print("Size: ", algo.xi.shape[0])

for _ in range(n_runs):

    ###
    algo.step()

    print("Size: ", algo.xi.shape[0])
