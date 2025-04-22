import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import gpmp as gp
import gpmpcontrib as gpc
import imse


assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

input_box = [[0.0], [10.0]]

n_grid = 1000
n_particles = 1000

model = gpc.Model_ConstantMeanMaternpML(
    "GP_seq_dim",
    output_dim=1,
    covariance_params={"p": 0},
    rng=np.random.default_rng(),
    box=input_box
)

xi = 10 * np.random.uniform(size=20).reshape(-1, 1)
zi = (xi ** 2).ravel()

grid = np.random.uniform(size=n_grid).reshape(-1, 1)

algo = imse.IMSE(10, grid, input_box, n_particles, xi, zi, model)
