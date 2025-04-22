# --------------------------------------------------------------
# Authors: Sébastien Petit <sebastien.petit@lne.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import SequentialPrediction
from gpmpcontrib import SMC
from gpmpcontrib.smc import ParamSError
import collections


class IntegratedCriterion(SequentialPrediction):
    def __init__(self, input_box, n_particles, xi, zi, model):

        self.input_box = input_box
        self.n_particles = n_particles

        # model initialization
        super().__init__(model=model)

        # initial design
        self.xi = xi
        self.zi = zi

        # estimate model parameters
        self.update_params()

        # search space
        self.smc = self.init_smc()

        # criterion values
        self.criterion_values = None

        # criterion
        self.criterion = self.build_criterion()

    def init_smc(self):
        return SMC(box=self.input_box, n=self.n_particles)

    def get_target(self):
        criterion_xi = self.criterion(self.xi)

        criterion_particles = self.criterion(self.smc.particles.x)

        target = max(criterion_xi.max(), criterion_particles.max())

        return target

    def boxify_criterion(self, x):
        input_box = gnp.asarray(self.input_box)
        b = sampcrit.isinbox(input_box, x)

        res = self.criterion(x).flatten()

        res = gnp.where(gnp.asarray(b), res, - gnp.inf)

        return res

    def update_search_space(self):
        method = self.options["smc_method"]

        target = self.get_target()

        if method == "subset":
            self.smc.subset(
                func=self.boxify_criterion,
                target=target,
                p0=0.2,
                xi=self.xi,
                debug=False
            )
        else:
            raise ValueError(method)

    def set_initial_design(self, xi, update_model=True, update_search_space=True):
        raise NotImplemented

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        znew = self.model.compute_conditional_simulations(xi=self.xi, zi=self.zi, xt=xnew)

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        if update_search_space:
            self.update_search_space()

    def local_criterion_opt(self, init):
        """
            init : ndarray
        Initial guess of the criterion maximizer.
        """

        def crit_(x):
            x_row = x.reshape(1, -1)
            criterion_value = self.criterion(x_row)
            print(criterion_value.shape)
            return - criterion_value

        crit_jit = gnp.jax.jit(crit_)

        dcrit = gnp.jax.jit(gnp.grad(crit_jit))

        box = self.computer_experiments_problem.input_box
        assert all([len(_v) == len(box[0]) for _v in box])

        bounds = [tuple(box[i][k] for i in range(len(box))) for k in range(len(box[0]))]
        criterion_argmin = gp.kernel.autoselect_parameters(
            init, crit_jit, dcrit, bounds=bounds
        )

        if gnp.numpy.isnan(criterion_argmin).any():
            return init

        for i in range(criterion_argmin.shape[0]):
            if criterion_argmin[i] < bounds[i][0]:
                criterion_argmin[i] = bounds[i][0]
            if bounds[i][1] < criterion_argmin[i]:
                criterion_argmin[i] = bounds[i][1]

        if crit_(criterion_argmin) < crit_(init):
            output = criterion_argmin
        else:
            output = init

        return gnp.asarray(output.reshape(1, -1))

    def step(self):
        # evaluate the criterion on the search space
        self.criterion_values = self.criterion(self.smc.particles.x)

        assert not gnp.isnan(self.criterion_values).any()

        # make new evaluation
        x_new = self.smc.particles.x[gnp.argmax(gnp.asarray(self.criterion_values))].reshape(1, -1)

        x_new = self.local_criterion_opt(gnp.to_np(x_new).ravel())

        self.make_new_eval(x_new)
