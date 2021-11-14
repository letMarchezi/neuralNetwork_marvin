#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        # Uma rede neural Perceptron Multicamadas
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70, ), random_state=1, verbose=True)
        clf.fit(self.marvin_dataset["X_train"], self.marvin_dataset["y_train"])  # Treino do classificador

        self.marvin_model = {
            "clf": clf,
            "vect": self.marvin_dataset["vect"]
        }

