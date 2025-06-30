import os
import sys
from typing import Any

import numpy
from hyperopt import hp


class Model:
    def __init__(self, name: str, params: [str, Any | None] = {}):
        self.name = name
        self.model = None
        self.params = params

    def _fit(self, X: numpy.ndarray, **kwargs: object) -> None:
        try:
            Y = kwargs["Y"]
            self.model.fit(X, Y)
        except KeyError as error:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(
                f'The algorithm {self.name} use de base fit method but no targets were provided.')
            print(type(error).__name__, "–", error, "not found. File:", fname, ", Line:", exc_tb.tb_lineno)
            sys.exit(1)

    def _predict(self, X: numpy.ndarray, **kwargs) -> numpy.ndarray:
        yhat = self.model.predict(X)
        return yhat

    def get_complexity(self) -> int:
        return 0

    def get_space(self, **kwargs: object):
        space = hp.choice('model', [
            {
                'name': self.name
            },
        ])

        return space
