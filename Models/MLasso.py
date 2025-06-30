from typing import Any

from hyperopt import hp
from sklearn.linear_model import Lasso, MultiTaskLasso

from Models.Model import Model


class MLasso(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("Lasso", params)

        self.selection_choices = ['cyclic', 'random']

        # get hyperparameters
        alpha = params["alpha"] if "alpha" in params.keys() else 0.01
        max_iter = int(params["max_iter"]) if "max_iter" in params.keys() else 1000
        random_state = params["random_state"] if "random_state" in params.keys() else 42
        n_targets = int(params["n_targets"]) if "n_targets" in params.keys() else 1

        selection = params["selection"] if "selection" in params.keys() else 'cyclic'

        # init the model
        if n_targets == 1:
            self.model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state, selection=selection)
        else:
            self.model = MultiTaskLasso(alpha=alpha, max_iter=max_iter, selection=selection, random_state=random_state)

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'alpha': hp.uniform('alpha', 0.1, 2),
                'max_iter': hp.quniform('max_iter', 500, 1500, 1),
                'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                'selection': hp.choice('selection', self.selection_choices),
            },
        ])
        return space