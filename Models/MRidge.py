from typing import Any

from hyperopt import hp
from sklearn.linear_model import Ridge

from Models.Model import Model


class MRidge(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("Ridge", params)

        # get hyperparameters
        alpha = params["alpha"] if "alpha" in params.keys() else 0.01
        random_state = params["random_state"] if "random_state" in params.keys() else 42

        # init the model
        self.model = Ridge(alpha=alpha, max_iter=1000, solver='auto', random_state=random_state)

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'alpha': hp.uniform('alpha', 0.1, 2)
            },
        ])
        return space
