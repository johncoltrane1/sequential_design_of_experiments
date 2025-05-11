import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc
import gpmp.num as gnp
import imse_mixed_variables
import sys, os
import json


assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)


path = sys.argv[1]

# Fetch variables
with open(os.path.join(path, "variables.json"), "r") as f:
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
xi = np.load(os.path.join(path, "xi.npy"))
zi = np.load(os.path.join(path, "zi.npy"))

for j in range(xi.shape[1]):
    for i in range(xi.shape[0]):
        if isinstance(variables[j], list):
            assert xi[i, j] in variables[j], (xi[i, j], variables[j])
        elif isinstance(variables[j], tuple):
            assert variables[j][0] <= xi[i, j] <= variables[j][1], (xi[i, j], variables[j])
        else:
            raise ValueError("{}, {}, {}".format(i, j, xi[i, j]))

# grid
grid_list = []

for variable in variables:
    if isinstance(variable, list):
        _grid = (variable[1] - variable[0]) * np.random.uniform(size=1000) + variable[0]
    elif isinstance(variable, tuple):
        _grid = np.random.choice(variable, size=1000)
    else:
        raise ValueError(variable)

    grid_list.append(_grid.reshape(-1, 1))

grid = np.hstack(grid_list)

algo = imse_mixed_variables.IMSE_MIXED_VARIABLES(variables, 1000, grid, xi, zi, model)

zloo, sigma2loo, eloo = algo.models[0]["model"].loo(algo.xi, algo.zi)

zloo = zloo.ravel().numpy()
sigma2loo = sigma2loo.ravel().numpy()

Y = algo.zi.ravel().numpy()

LOO_MSE = np.square(Y - zloo.ravel()).mean()

print("R2: ", 1 - LOO_MSE / Y.var())

min_value = min(zloo.min(), Y.min())
max_value = min(zloo.max(), Y.max())

plt.subplots(1, 2)

plt.subplot(1, 2, 1)

plt.errorbar(Y.ravel(), zloo.ravel(), yerr=2 * np.sqrt(sigma2loo).ravel(), fmt='bo')
plt.plot([min_value, max_value], [min_value, max_value], 'k-')

plt.subplot(1, 2, 2)

plt.plot(
    (Y.ravel() - zloo.ravel()) / np.sqrt(sigma2loo).ravel(),
    "o",
    label="points"
)

plt.axhline(-2, color="r")
plt.axhline(2, color="r", label="-/+ 2 sigma")

plt.axhline(-3, color="k")
plt.axhline(3, color="k", label="-/+ 3 sigma")

plt.show()
