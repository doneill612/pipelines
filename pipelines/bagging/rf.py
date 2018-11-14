# -*- coding: utf-8 -*-
from core import model_logging
from core.defaults import required_rf_params
from core.abstract_model import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score

class RandomForest(AbstractModel):

    def __init__(self, data_loc, params):
        super(RandomForest, self).__init__(data_loc, params)

    def verify_parameters(self, params):
        missing_keys = []
        for key in required_rf_params:
            if key not in params:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            model_logging.fatal('assertion',
                                'Could not create RandomForest model. '
                                'Missing pararamters : {}'.format(missing_keys))
        return params

    def to_string(self):
        return 'Random Forest Classifier'

    def set_model_def(self):
        self._model = RandomForestClassifier(n_jobs=-1)

    def get_parameter_grid(self):
        return self._params['rf_parameter_grid']
