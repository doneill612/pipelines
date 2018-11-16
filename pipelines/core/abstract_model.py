# -*- coding: utf-8 -*-
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split

class AbstractModel(metaclass=ABCMeta):

    def __init__(self, data_loc, params):
        self._data_loc = data_loc
        self._params = self.verify_parameters(params)
        self._df_loaded = False
        self._df = None
        self.set_model_def()
        self.load_data()

    def clear(self):
        del self._df
        self._df = None
        self._df_loaded = False

    def _extract_features(self, scale=False):
        if not self._df_loaded:
            model_logging.fatal('assertion', 'Pandas DataFrame not yet loaded. '
                          'Could not extract features.')
        features = self._df.drop(self._params['target'], axis=1)
        if scale:
            features = ((features - features.mean()) /
                        (features.max() - features.min()))
        return features

    def _extract_labels(self):
        if not self._df_loaded:
            model_logging.fatal('assertion', 'Pandas DataFrame not yet loaded. '
                          'Could not extract features.')
        return self._df[self._params['target']]

    def build_train_test_set(self):
        features = self._extract_features()
        labels = self._extract_labels()
        return train_test_split(features, labels,
                                test_size=self._params['test_size'],
                                stratify=labels,
                                random_state=1337)

    def load_data(self):
        if not self._df_loaded:
            self._df = pd.read_csv(self._data_loc, usecols=self._params['features'],
                                                   sep=';')
            self._df.dropna(inplace=True)
            self._df_loaded = True
        else: model_logging.info('Pandas DataFrame already loaded.')

    def get_scorers(self):
        return self._params['scorers']

    def get_model_def(self):
        return self._model

    @abstractmethod
    def verify_parameters(self, params):
        model_logging.fatal('ni', 'Abstract method verify_parameters() must be implemented.')

    @abstractmethod
    def get_parameter_grid(self):
        model_logging.fatal('ni', 'Abstract method get_parameter_grid() must be implemented.')

    @abstractmethod
    def set_model_def(self):
        model_logging.fatal('ni', 'Abstract method set_model_def() must be implemented.')

    @abstractmethod
    def to_string(self):
        model_logging.fatal('ni', 'Abstract method '
                            'to_string() must be implemented.')
