from nns.nn import NeuralNetwork
from keras.models import Sequential
from keras.layers import LSTM

class RNN(NeuralNetwork):

    def __init__(self, data_loc, params):
        super(CNN, self).__init__(data_loc, params)

    def verify_parameters(self, params):
        return params

    def to_string(self):
        return 'Recurrent Neural Network (RNN)'

    def set_model_def(self):
        self._model = Sequential()

    def _add_lstm(cells, in_shape, return_sequences: bool=True):
        lstm = LSTM(cells, input_shape=in_shape,
                    return_sequences=return_sequences)
        self._add_layer(lstm)
