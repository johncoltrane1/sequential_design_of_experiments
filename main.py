import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc
import imse


assert gp.num._gpmp_backend_ == "torch", "{} is used, please install Torch.".format(gp.num._gpmp_backend_)

input_box = [[0.0], [10.0]]

num_new = 5

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

xi = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]).reshape(-1, 1)
zi = (xi ** 2).ravel()

x_test = np.linspace(0, 10, 10001).reshape(-1, 1)

grid = np.linspace(0, 10, n_grid).reshape(-1, 1)

algo = imse.IMSE(grid, num_new, input_box, n_particles, xi, zi, model)

print("Size: ", algo.xi.shape[0])

for _ in range(n_runs):
    criterion_values = - algo.criterion(x_test)
    current_criterion = algo.model.predict(algo.xi, algo.zi, grid)[1].mean()

    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.plot(algo.xi.ravel(), algo.zi.ravel(), 'bo')

    algo.step()

    print("Size: ", algo.xi.shape[0])

    plt.plot(algo.xi[[-num_new], 0], algo.zi[[-num_new], 0], 'go')

    plt.subplot(1, 2, 2)

    plt.axhline(current_criterion, color="k")
    plt.plot(x_test, criterion_values)

    plt.show()
