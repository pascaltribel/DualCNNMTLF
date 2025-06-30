from typing import Any

from hyperopt import hp
from sklearn.neighbors import KNeighborsRegressor

from Models.Model import Model


class MKNN(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("KNN", params)

        self.n_neighbors_choices = [1, 3, 5, 7, 9]
        self.weights_choices = ["uniform", "distance"]
        self.algorithm_choices = ["brute", "ball_tree", "kd_tree"]

        # get hyperparameters
        n_neighbors = int(params["n_neighbors"]) if "n_neighbors" in params.keys() else 5
        weights = params["weights"] if "weights" in params.keys() else 'uniform'
        algorithm = params["algorithm"] if "algorithm" in params.keys() else "auto"

        # init the model
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'n_neighbors': hp.choice('n_neighbors', self.n_neighbors_choices),
                'weights': hp.choice('weights', self.weights_choices),
                'algorithm': hp.choice('algorithm', self.algorithm_choices),
            },
        ])
        return space
