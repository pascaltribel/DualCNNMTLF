from typing import Any

from hyperopt import hp
from sklearn.cross_decomposition import PLSRegression

from Models.Model import Model


class MPLS(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("PLSRegression", params)

        # get hyperparameters
        n_components = int(params["n_components"]) if "n_components" in params.keys() else 2
        max_iter = int(params["max_iter"]) if "max_iter" in params.keys() else 500

        # init the model
        self.model = PLSRegression(n_components=n_components, max_iter=max_iter)

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'n_components': hp.quniform('n_components', 1, kwargs["n_features"], 1),
                'max_iter': hp.quniform('max_iter', 500, 2000, 1)
            },
        ])
        return space
