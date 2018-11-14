# -*- coding: utf-8 -*-
import copy
import xgboost as xgb
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from pnd import AbstractModel, logging

dsetloc = '../data/PLACEHOLDER'
default_xgboost_params = {
    'boost_params': {
        'max_depth': 2,
        'nthread': 4,
        'eval_metric': ['auc'],
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic'
    },
    'test_size': 0.3,
    'target': 'pnd',
    'features': ['type_id_rao', 'pdis', 'nbr_plis',
                 'population', 'pct_not_vertical_40',
                 'delta_index']
}

class XGBoostModel(AbstractModel):

    def __init__(self, data_loc, params=default_xgboost_params):
        super(AbstractModel, self).__init__(data_loc, params)

    def load_data(self):
        if not self._df_loaded:
            self._df = pd.read_csv(self._data_loc)
        else: logging.info('Pandas DataFrame already loaded.')

    def eval_metrics(self):
        '''
        Returns a deep copy of this xgboost model's evaluation metrics.
        '''
        boost_params = self._params['boost_params']
        return copy.deepcopy(boost_params['eval_metric'])

if __name__ == '__main__':
    xgboost = XGBoostModel(dsetloc)
    print(xgboost.to_string())
