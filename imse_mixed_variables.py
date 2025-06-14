# --------------------------------------------------------------
# Authors: Sébastien Petit <sebastien.petit@lne.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import numpy as np
import gpmp.num as gnp
from imse import IMSE
import itertools
from integratedcriterion import IntegratedCriterion


class IMSE_MIXED_VARIABLES(IMSE):

    def __init__(self, variables, n_particles, grid, xi, zi, model):

        # grid
        self.grid = grid

        # number of new points at each step
        self.num_new = 1

        self.discrete_variables = [[i, v] for i, v in enumerate(variables) if isinstance(v, list)]
        self.continuous_variables = [[i, v] for i, v in enumerate(variables) if isinstance(v, tuple)]

        input_box = [[_v[1][0] for _v in self.continuous_variables], [_v[1][1] for _v in self.continuous_variables]]

        # box utils
        self.input_box = input_box
        self.multiplied_input_box = [self.num_new * self.input_box[0], self.num_new * self.input_box[1]]

        # number of SMC particles
        self.n_particles = n_particles

        # model initialization
        super(IntegratedCriterion, self).__init__(model=model)

        # initial design
        self.xi = gnp.asarray(xi)
        self.zi = gnp.asarray(zi).reshape(-1, 1)

        # estimate model parameters
        self.update_params()

        # search space
        self.smc = self.init_smc()

        # criterion values
        self.criterion_values = None

        # criterion
        self.criterion = None

    def complete(self, _x, k):
        assert 1 <= _x.ndim <= 2

        if _x.ndim == 1:
            x = gnp.zeros([self.xi.shape[1]])
        else:
            x = gnp.zeros([_x.shape[0], self.xi.shape[1]])

        for i in range(self.xi.shape[1]):
            # FIXME:() Optimize

            discrete_variables_indexes = [val[0] for val in self.discrete_variables]
            continuous_variables_indexes = [val[0] for val in self.continuous_variables]

            if i in discrete_variables_indexes:
                idx = discrete_variables_indexes.index(i)

                if x.ndim == 1:
                    x[i] = k[idx]
                else:
                    x[:, i] = k[idx]

            elif i in continuous_variables_indexes:
                idx = continuous_variables_indexes.index(i)

                if x.ndim == 1:
                    x[i] = _x[idx]
                else:
                    x[:, i] = _x[:, idx]

            else:
                raise RuntimeError

        return x

    def build_criterion(self, k):
        k = gnp.asarray(k)

        def criterion(_x):
            assert 1 <= _x.ndim <= 2

            if _x.ndim == 2:
                res = []
                for i in range(_x.shape[0]):
                    res.append(criterion(_x[i, :]))
                return gnp.vstack(res)

            _x = gnp.asarray(_x)

            x = self.complete(_x, k)

            assert x.shape[0] % self.xi.shape[1] == 0

            x_array = x
            z_array = gnp.zeros([1, 1])

            xi_augmented = gnp.vstack((self.xi, x_array))
            zi_augmented = gnp.vstack((self.zi, z_array))

            _, zpv = self.model.predict(xi_augmented, zi_augmented, self.grid, convert_out=False)
            value = - zpv.mean()
            return value

        return criterion

    def get_target(self):
        return np.inf

    def update_search_space(self):
        target = self.get_target()

        self.smc.subset(
            func=self.boxify_criterion,
            target=target,
            p0=0.2,
            debug=False,
            max_iter=2
        )

    def untile(self, x):
        raise NotImplemented

    def step(self):

        k_list = list(itertools.product(*[val[1] for val in self.discrete_variables]))
        x_new_list = []
        scores_list = []
        for idx_k, k in enumerate(k_list):
            print("Running {}-th optim.".format(idx_k))

            # criterion
            self.criterion = self.build_criterion(k)

            # run smc
            self.init_smc()
            self.update_search_space()

            # evaluate the criterion on the search space
            self.criterion_values = self.criterion(self.smc.particles.x)

            assert not gnp.isnan(self.criterion_values).any()

            # make new evaluation
            x_new = self.smc.particles.x[gnp.argmax(gnp.asarray(self.criterion_values))].reshape(1, -1)
            print("Best value after SMC: {}".format(float(self.criterion(x_new))))

            # improve with local optimizer
            x_new = self.local_criterion_opt(gnp.to_np(x_new).ravel())

            # criterion vaue
            criterion_x_new = self.criterion(x_new)

            print("Value found after optim: {}".format(float(criterion_x_new)))

            # store
            x_new_list.append(x_new)
            scores_list.append(criterion_x_new)

        # get best
        print("Best score: ", float(max(scores_list)))
        idx_max = np.array(scores_list).argmax()
        x_new = x_new_list[idx_max]
        k_new = k_list[idx_max]

        x_new = self.complete(x_new, k_new)
        x_new = x_new.reshape(1, -1)

        # store
        self.make_new_eval(x_new)

