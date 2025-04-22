# --------------------------------------------------------------
# Authors: Sébastien Petit <sebastien.petit@lne.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
from integratedcriterion import IntegratedCriterion


class IMSE(IntegratedCriterion):

    def __init__(self, num_new, grid, *args, **kwargs):
        self.grid = grid
        self.num_new = num_new
        super().__init__(*args, **kwargs)

    def build_criterion(self):
        def criterion(x):
            assert 1 <= x.ndim <= 2
            if x.ndim == 2:
                res = []
                for i in range(x.shape[0]):
                    res.append(criterion(x[i, :]))
                return gnp.vstack(res)

            assert x.shape[0] % self.xi.shape[1] == 0

            x_array = x.reshape(-1, self.xi.shape[1])
            z_array = gnp.zeros([x_array.shape[0]])

            xi_augmented = gnp.vstack((gnp.asarray(self.xi), x_array))
            zi_augmented = gnp.concatenate((gnp.asarray(self.zi), z_array)).reshape(-1, 1)

            _, zpv = self.model.predict(xi_augmented, zi_augmented, self.grid, convert_out=False)
            value = - zpv.mean()
            return value

        return criterion

