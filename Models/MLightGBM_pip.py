from typing import Any
from sklearn.multioutput import MultiOutputRegressor
from hyperopt import hp
from lightgbm import LGBMRegressor

from Models.Model import Model


class MLightGBMPIP(Model):

    def __init__(self, params: [str, Any | None] = {}):
        super().__init__("LightGBM_pip", params)


        self.boosting_type_choices = ["rf", "gbdt", "dart"]

        # get hyperparameters
        self.lags = int(params["lags"]) if "lags" in params.keys() else 1
        output_chunk_length = int(params["H"]) if "H" in params.keys() else 5
        random_state = params["random_state"] if "random_state" in params.keys() else 42
        boosting_type = params["boosting_type"] if "boosting_type" in params.keys() else "gbdt"
        max_depth = int(params["max_depth"]) if "max_depth" in params.keys() else -1
        num_leaves = int(params["num_leaves"]) if "num_leaves" in params.keys() else 31
        learning_rate = params["learning_rate"] if "learning_rate" in params.keys() else 1
        n_estimators = int(params["n_estimators"]) if "n_estimators" in params.keys() else 100
        min_child_samples = int(params["min_child_samples"]) if "min_child_samples" in params.keys() else 20
        reg_alpha = params["reg_alpha"] if "reg_alpha" in params.keys() else 0.0
        reg_lambda = params["reg_lambda"] if "reg_lambda" in params.keys() else 0.0

        # init the model

        self.model = LGBMRegressor(random_state=random_state,
                                   max_depth=max_depth, num_leaves=num_leaves, learning_rate=10 ** -learning_rate,
                                   n_estimators=n_estimators, reg_alpha=reg_alpha, subsample_freq=5, subsample=0.9,
                                   min_child_samples=min_child_samples, reg_lambda=reg_lambda,
                                   boosting_type=boosting_type, verbose=-1)

    def get_space(self, **kwargs: object):

        space = hp.choice('model', [
            {
                'name': self.name,
                'boosting_type': hp.choice('boosting_type', self.boosting_type_choices),
                'num_leaves': hp.quniform('num_leaves', 2, 100, 1),
                'max_depth': hp.quniform('max_depth', 0, 20, 1),  # the minimum is zero to use no limit
                'learning_rate': hp.uniform('learning_rate', 1, 5),
                'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
                'min_child_samples': hp.quniform('min_child_samples', 1, 50, 1),
                'reg_alpha': hp.uniform('reg_alpha', 0, 2),
                'reg_lambda': hp.uniform('reg_lambda', 0, 2)
            },
        ])
        return space
