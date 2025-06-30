from typing import Any

from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from Models.Model import Model


class MPipeRF(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("PipeRF", params)

        self.criterion_choices = ["squared_error", "friedman_mse", "absolute_error", "poisson"]

        # get hyperparameters
        n_estimators = int(params["n_estimators"]) if "n_estimators" in params.keys() else 100
        criterion = params["criterion"] if "criterion" in params.keys() else 'squared_error'
        max_depth = int(params["max_depth"]) if "max_depth" in params.keys() else None
        random_state = params["random_state"] if "random_state" in params.keys() else 42
        min_samples_split = params["min_samples_split"] if "min_samples_split" in params.keys() else 1
        min_samples_leaf = params["min_samples_leaf"] if "min_samples_leaf" in params.keys() else .9

        # init the model
        self.model = Pipeline([
            ('feature_selection',
             SelectFromModel(RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                   random_state=random_state, min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf, verbose=0, n_jobs=5))),
            ('regression', RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                 random_state=random_state, min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf, verbose=0, n_jobs=5))
        ])

    def get_complexity(self):
        # See https://stackoverflow.com/questions/51139875/sklearn-randomforestregressor-number-of-trainable-parameters
        rf = self.model.named_steps["regression"]
        n_params = sum(tree.tree_.node_count for tree in rf.estimators_) * 5
        return n_params

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'max_depth': hp.quniform('max_depth', 5, 50, 1),
                'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
                'criterion': hp.choice('criterion', self.criterion_choices),
                'min_samples_split': hp.uniform('min_samples_split', 0.001, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', .001, .999)
            },
        ])
        return space
