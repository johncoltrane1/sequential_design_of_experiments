import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc
import imse
import sys


num_new = int(sys.argv[1])

assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

input_box = [[0.0, 3.0], [10.0, 5.0]]

n_grid = 500
n_particles = 1000

n_runs = 10

model = gpc.Model_ConstantMeanMaternpML(
    "GP_seq_dim",
    output_dim=1,
    covariance_params={"p": 0},
    rng=np.random.default_rng(),
    box=input_box
)

base = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
base_x = (input_box[1][0] - input_box[0][0]) * base + input_box[0][0]
base_y = (input_box[1][1] - input_box[0][1]) * base + input_box[0][1]

xx, yy = np.meshgrid(base_x, base_y)
xi = np.hstack((xx.reshape(-1, 1),  yy.reshape(-1, 1)))

zi = xi[:, 0] ** 2 + 50 * np.cos(2 * xi[:, 1])

x_test = np.random.uniform(size=(10001, 2))
x_test[:, 0] = (input_box[1][0] - input_box[0][0]) * x_test[:, 0] + input_box[0][0]
x_test[:, 1] = (input_box[1][1] - input_box[0][1]) * x_test[:, 1] + input_box[0][1]

grid = np.random.uniform(size=(n_grid, 2))
grid[:, 0] = (input_box[1][0] - input_box[0][0]) * grid[:, 0] + input_box[0][0]
grid[:, 1] = (input_box[1][1] - input_box[0][1]) * grid[:, 1] + input_box[0][1]

algo = imse.IMSE(grid, num_new, input_box, n_particles, xi, zi, model)

print("Size: ", algo.xi.shape[0])

for _ in range(n_runs):
    #
    size_grid_contour_plot = 100

    grid_contour_plot_base = np.linspace(0, 1, size_grid_contour_plot)
    grid_contour_plot_base_x = (input_box[1][0] - input_box[0][0]) * grid_contour_plot_base + input_box[0][0]
    grid_contour_plot_base_y = (input_box[1][1] - input_box[0][1]) * grid_contour_plot_base + input_box[0][1]

    grid_contour_plot_x, grid_contour_plot_y = np.meshgrid(grid_contour_plot_base_x, grid_contour_plot_base_y)

    output = np.zeros([size_grid_contour_plot, size_grid_contour_plot])

    for i in range(size_grid_contour_plot):
        for j in range(size_grid_contour_plot):
            output[i, j] = algo.criterion(np.array([grid_contour_plot_x[i, j], grid_contour_plot_y[i, j]]))

    #
    plt.subplots(2, 1)

    plt.subplot(2, 1, 1)
    plt.plot(algo.xi[:, 0], algo.xi[:, 1], 'bo')

    algo.step()

    print("Size: ", algo.xi.shape[0])

    print(algo.models)

    plt.plot(algo.xi[(-num_new):, 0], algo.xi[(-num_new):, 1], 'go')

    plt.subplot(2, 1, 2)

    plt.contour(grid_contour_plot_x, grid_contour_plot_y, output)

    plt.show()
