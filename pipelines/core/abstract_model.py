import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split

class AbstractModel(metaclass=ABCMeta):

    def __init__(self, data_loc, params):
        self._data_loc = data_loc
        self._params = params
        self._df_loaded = False
        self._df = None
        self.load_data()

    def clear(self):
        del self._df
        self._df = None
        self._df_loaded = False

    def extract_features(self, scale=True):
        if not self._df_loaded:
            model_logging.fatal('assertion', 'Pandas DataFrame not yet loaded. '
                          'Could not extract features.')
        features = self._df.drop(self._params['target'], axis=1)
        if scale:
            features = ((features - features.mean()) /
                        (features.max() - features.min()))
        return features

    def extract_labels(self):
        if not self._df_loaded:
            model_logging.fatal('assertion', 'Pandas DataFrame not yet loaded. '
                          'Could not extract features.')
        return self._df[self._params['target']]

    def build_train_test_set(self, features, labels):
        return train_test_split(features, labels,
                                test_size=self._params['test_size'],
                                random_state=1337)

    def load_data(self):
        if not self._df_loaded:
            self._df = pd.read_csv(self._data_loc, usecols=self._params['features'],
                                                   sep=';')
            self._df.dropna(inplace=True)
            #indicies = ~self._df.isin([np.nan, np.inf, -np.inf]).any(1)
            #self._df = self._df[indicies].astype(np.float64)
            self._df_loaded = True
        else: model_logging.info('Pandas DataFrame already loaded.')

    @abstractmethod
    def train(self):
        logging.fatal('ni', 'Abstract method train() must be implemented.')

    @abstractmethod
    def test(self):
        logging.fatal('ni', 'Abstract method test() must be implemented.')

    @abstractmethod
    def to_string(self):
        logging.fatal('ni', 'Abstract method '
                      'to_string() must be implemented.')

    @abstractmethod
    def eval_metrics(self):
        logging.fatal('ni', 'Abstract method '
                      'eval_metrics() must be implemented.')
