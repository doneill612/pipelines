# -*- coding: utf-8 -*-
from core import model_logging
from core.defaults import required_gbm_params
from core.abstract_model import AbstractModel
from sklearn.ensemble import GradientBoostingClassifier

class GradientBooster(AbstractModel):

    def __init__(self, data_loc, params=default_xgboost_params):
        super(AbstractModel, self).__init__(data_loc, params)

    def verify_parameters(self, params):
        missing_keys = []
        for key in required_gbm_params:
            if key not in params:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            model_logging.fatal('assertion',
                                'Could not create GradientBooster model. '
                                'Missing pararamters : {}'.format(missing_keys))
        return params

    def to_string(self):
        return 'Gradient Boosted Machine'

    def set_model_def(self):
        self._model = RandomForestClassifier(n_jobs=-1)

    def get_parameter_grid(self):
        return self._params['rf_parameter_grid']

if __name__ == '__main__':
    xgboost = XGBoostModel(dsetloc)
    print(xgboost.to_string())
