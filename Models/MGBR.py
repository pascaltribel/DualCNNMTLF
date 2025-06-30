from typing import Any

from hyperopt import hp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import RegressorChain

from Models.Model import Model


class MGBR(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("HGBR", params)
        self.loss = ["squared_error", "poisson", "absolute_error"]

        # get hyperparameters
        n_targets = int(params["n_targets"]) if "n_targets" in params.keys() else 1
        random_state = params["random_state"] if "random_state" in params.keys() else 42
        loss = params["loss"] if "loss" in params.keys() else "squared_error"
        learning_rate = params["learning_rate"] if "learning_rate" in params.keys() else 1
        max_iter = int(params["max_iter"]) if "max_iter" in params.keys() else 100
        max_leaf_nodes = int(params["max_leaf_nodes"]) if "max_leaf_nodes" in params.keys() else 31  # must > 1
        max_depth = int(params["max_depth"]) if "max_depth" in params.keys() else None
        min_samples_leaf = int(params["min_samples_leaf"]) if "min_samples_leaf" in params.keys() else 20
        l2_regularization = params["l2_regularization"] if "l2_regularization" in params.keys() else 0

        # init the model
        # HistGradientBoostingRegressor is much faster than GradientBoostingRegressor for big datasets (n_samples >= 10 000).
        self.model = HistGradientBoostingRegressor(random_state=random_state,
                                                   loss=loss, learning_rate=10 ** -learning_rate, max_iter=max_iter,
                                                   max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                                                   min_samples_leaf=min_samples_leaf,
                                                   l2_regularization=l2_regularization, verbose=0)

        if n_targets != 1:
            self.model = RegressorChain(base_estimator=self.model, verbose=False,
                                        random_state=random_state)

    def get_complexity(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        n_params = sum(
            sum(len(p[0].nodes) for p in dimension._predictors) for dimension in self.model.estimators_)
        return n_params

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'max_depth': hp.quniform('max_depth', 1, 30, 1),
                'max_iter': hp.quniform('max_iter', 50, 200, 1),
                'max_leaf_nodes': hp.quniform('max_leaf_nodes', 2, 50, 1),
                'l2_regularization': hp.uniform('l2_regularization', 0, 2),
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
                'learning_rate': hp.uniform('learning_rate', 1, 5),
                'loss': hp.choice('loss', self.loss),
            },
        ])
        return space
