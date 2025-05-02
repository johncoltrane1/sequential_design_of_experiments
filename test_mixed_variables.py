import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc
import imse_mixed_variables
import sys
import itertools


assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

discrete_variables = [[0, 2], [1, 3]]
continuous_variables = [(0.0, 10.0), (3.0, 5.0)]

n_grid = 500
n_particles = 1000

n_runs = 10

num_new = 1

model = gpc.Model_ConstantMeanMaternpML(
    "GP_seq_dim",
    output_dim=1,
    covariance_params={"p": 0},
    rng=np.random.default_rng(),
    box=None
)

# design
base_continuous = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
base_continuous_x = (continuous_variables[0][1] - continuous_variables[0][0]) * base_continuous + continuous_variables[0][0]
base_continuous_y = (continuous_variables[1][1] - continuous_variables[1][0]) * base_continuous + continuous_variables[1][0]

xx1, xx2, xx3, xx4 = np.meshgrid(base_continuous_x, discrete_variables[0], base_continuous_y, discrete_variables[1])

xi = np.hstack((xx1.reshape(-1, 1), xx2.reshape(-1, 1), xx3.reshape(-1, 1), xx4.reshape(-1, 1)))

zi = xi[:, 0] ** 2 + 50 * np.cos(2 * xi[:, 1]) + 10 * xi[:, 2] + 0.1 * np.exp(xi[:, 3])

# grid
continuous_grid = np.random.uniform(size=(n_grid, 2))
continuous_grid[:, 0] = (continuous_variables[0][1] - continuous_variables[0][0]) * continuous_grid[:, 0] + continuous_variables[0][0]
continuous_grid[:, 1] = (continuous_variables[1][1] - continuous_variables[1][0]) * continuous_grid[:, 1] + continuous_variables[1][0]

discete_grid = np.hstack((
    np.random.choice(discrete_variables[0], size=n_grid).reshape(-1, 1),
    np.random.choice(discrete_variables[1], size=n_grid).reshape(-1, 1)
))

grid = np.hstack((continuous_grid[:, [0]], discete_grid[:, [0]], continuous_grid[:, [1]], discete_grid[:, [1]]))

variables = [continuous_variables[0], discrete_variables[0], continuous_variables[1], discrete_variables[1]]
algo = imse_mixed_variables.IMSE_MIXED_VARIABLES(variables, n_particles, grid, xi, zi, model)

print("Size: ", algo.xi.shape[0])

for _ in range(n_runs):
    ###
    plt.subplots(2, 4)

    ###
    k_list = list(itertools.product(*[val[1] for val in algo.discrete_variables]))
    cpt = 0
    for k in k_list:

        ##
        plt.subplot(2, 4, 2 * cpt + 1)

        #
        algo.criterion = algo.build_criterion(k)

        #
        size_grid_contour_plot = 100

        grid_contour_plot_base = np.linspace(0, 1, size_grid_contour_plot)
        grid_contour_plot_base_x = (continuous_variables[0][1] - continuous_variables[0][0]) * grid_contour_plot_base + continuous_variables[0][0]
        grid_contour_plot_base_y = (continuous_variables[1][1] - continuous_variables[1][0]) * grid_contour_plot_base + continuous_variables[1][0]

        grid_contour_plot_x, grid_contour_plot_y = np.meshgrid(grid_contour_plot_base_x, grid_contour_plot_base_y)

        output = np.zeros([size_grid_contour_plot, size_grid_contour_plot])

        for i in range(size_grid_contour_plot):
            for j in range(size_grid_contour_plot):
                output[i, j] = algo.criterion(np.array([grid_contour_plot_x[i, j], grid_contour_plot_y[i, j]]))

        #
        plt.contour(grid_contour_plot_x, grid_contour_plot_y, output)

        ##
        plt.subplot(2, 4, 2 * cpt + 2)

        #
        _xi = algo.xi.clone()
        for p, v in enumerate(algo.discrete_variables):
            _xi = _xi[_xi[:, v[0]] == k[p], :]

        #
        plt.plot(_xi[:, 0], _xi[:, 1], 'bo')

        ##
        cpt += 1

    ###
    algo.step()

    ###
    k_list = list(itertools.product(*[val[1] for val in algo.discrete_variables]))
    cpt = 0
    for k in k_list:
        print("Size: ", algo.xi.shape[0])

        print(algo.models)

        plt.plot(algo.xi[(-num_new):, 0], algo.xi[(-num_new):, 1], 'go')

        cpt += 1

    ###
    plt.show()
