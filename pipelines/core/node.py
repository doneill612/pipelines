from core import model_logging
from core.abstract_model import AbstractModel
from core.nns.nn import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

class NodeType(object):
    TRANSORM = 'transform'
    FIT = 'fit'
    FIT_TRANSFORM = 'fit_transform'

class Node(object):

    def __init__(self, type, name, op):
        self._type = type
        self._name = name
        self._op = op

    def is_tranform_node(self):
        return self._type == NodeType.TRANSORM

    def is_fit_node(self):
        return self._type == NodeType.FIT

    def is_fit_transform_node(self):
        return self._type == NodeType.FIT_TRANSFORM

    def fit(self, data):
        if self._type != NodeType.FIT and self._type != Node.FIT_TRANSFORM:
            model_logging.fatal('assertion',
                                'Cannot call fit() on non fit-transform node.')
        return self._op.fit(data)

    def transform(self, data):
        if self._type != NodeType.TRANSFORM and self._type != Node.FIT_TRANSFORM:
            model_logging.fatal('assertion',
                                'Cannot call transform() on non fit-transform node.')
        return self._op.transform(data)

    def get_op(self):
        return self._op

    def get_name(self):
        return self._name

    @staticmethod
    def pca_node(name, n_components):
        pca = PCA(n_components=n_components)
        pca_node = Node(type=NodeType.FIT_TRANSFORM,
                        name=name, op=pca)
        return pca_node

    @staticmethod
    def std_scaler_node(name):
        scaler = StandardScaler()
        scaler_node = Node(type=NodeType.TRANSORM,
                           name=name, op=scaler)
        return scaler_node

    @staticmethod
    def smote_node(name, seed: int=1337, ratio: float=1.0):
        smote = SMOTE(random_state=seed, ratio=ratio)
        smote_node = Node(type=NodeType.FIT_TRANSFORM,
                          name=name, op=smote)
        return smote_node

    @staticmethod
    def ml_node(name, model):
        if not isinstance(model, AbstractModel):
            model_logging.fatal('assertion', 'Attempted to use unsupported '
                                'model type. Machine learning '
                                'models must be subclasses of '
                                'AbstractModel.')
        model_node = Node(type=NodeType.FIT_TRANSFORM,
                          name=name, op=model)
        return model_node

    @staticmethod
    def dl_node(name, model):
        if not isinstance(model, NeuralNetwork):
            model_logging.fatal('assertion', 'Attempted to use unsupported '
                                'model type. Deep learning '
                                'models must be subclasses of '
                                'NeuralNetwork.')
        model_node = Node(type=NodeType.FIT,
                          name=name, op=model)
        return model_node
