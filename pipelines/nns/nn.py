from abc import ABCMeta, abstractmethod
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Dropout

class NeuralNetwork(AbstractModel, metaclass=ABCMeta):

    def __init__(self, data_loc, params):
        super(CNN, self).__init__(data_loc, params)

    def _add_dense_layer(self, units, shape=None, activation: str='relu',
                        time_distributed: bool=False):
        dense = None
        if shape is None:
            dense = Dense(units, activation=activation)
        else:
            dense = Dense(units, input_shape=shape, activation=activation)
        if time_distributed:
            dense = TimeDistributed(dense)
        self._add_layer(dense)

    def _add_dropout_layer(self, dropout: float=0.2):
        dropout = Dropout(dropout)
        self._add_layer(dropout)

    def _add_layer(self, layer):
        self._model.add(layer)

    def compile(self):
        self._model.compile(loss=self._params['loss_function'],
                            optimizer=self._params['optimizer'])

    def fit(self, X_train, y_train, X_validate, y_validate):
        self._model.fit(X_train, X_test,
                        batch_size=model._params['batch_size'],
                        nb_epoch=model._params['epochs'],
                        show_accuracy=True, verbose=1,
                        validation_data=(X_validate, y_validate))

    def evaluate(self, X_test, y_test):
        model_logging.info('Evaluating performance...')
        score = self._model.evaluate(X_test, y_test)
        model_logging.info(score[0])
        model_logging.info(score[1])
