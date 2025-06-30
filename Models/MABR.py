from hyperopt import hp
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any
from Models.Model import Model


class MABR(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("AdaBoostRegressor", params)

        self.criterion = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        self.loss = ["linear", "square", "exponential"]

        # get hyperparameters
        n_targets = int(params["n_targets"]) if "n_targets" in params.keys() else 1
        max_depth = int(params["max_depth"]) if "max_depth" in params.keys() else None
        max_depth = None if max_depth is None or max_depth == 0 else max_depth  # useful for HPO

        n_estimators = int(params["n_estimators"]) if "n_estimators" in params.keys() else 50
        criterion = params["criterion"] if "criterion" in params.keys() else "squared_error"
        random_state = params["random_state"] if "random_state" in params.keys() else 42
        min_samples_split = params["min_samples_split"] if "min_samples_split" in params.keys() else 2
        min_samples_leaf = params["min_samples_leaf"] if "min_samples_leaf" in params.keys() else 1
        learning_rate = params["learning_rate"] if "learning_rate" in params.keys() else 1
        loss = params["loss"] if "loss" in params.keys() else "linear"

        # init the model
        self.model = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, random_state=random_state,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf),
            n_estimators=n_estimators, learning_rate=10 ** -learning_rate, random_state=random_state, loss=loss)

        if n_targets != 1:
            self.model = MultiOutputRegressor(self.model)

    def get_complexity(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        n_params = sum(
            sum(dtr.tree_.node_count for dtr in dimension.estimators_) for dimension in self.model.estimators_)
        return n_params

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'max_depth': hp.quniform('max_depth', 1, 10, 1),
                'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
                'criterion': hp.choice('criterion', self.criterion),
                'min_samples_split': hp.uniform('min_samples_split', 0.001, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', .001, .999),
                'learning_rate': hp.uniform('learning_rate', 1, 5),
                'loss': hp.choice('loss', self.loss),
            },
        ])
        return space
